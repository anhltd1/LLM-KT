#!/usr/bin/env python
"""
Prediction CLI for Knowledge Tracing LLM

This script allows manual testing of the trained model using sample inputs.
It loads a JSON file with student history and target questions, then predicts performance.

Usage:
    python predict.py --checkpoint checkpoints/llm/small/best_model --input examples/sample.json
    python predict.py --checkpoint checkpoints/llm/small/best_model --interactive
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import pickle
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models import KTEmbeddingModel, KnowledgeTracingLLM

def parse_args():
    parser = argparse.ArgumentParser(description="Run predictions with KT-LLM")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--input", default="examples/sample.json", help="Path to input JSON file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--mappings-dir", default="mappings", help="Path to ID mappings directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", default="small", help="Config preset to use")
    parser.add_argument("--processed-dir", default=None, help="Path to processed data (to find mappings)")
    return parser.parse_args()

def load_mappings_dict(mappings_dir: str) -> Dict[str, Dict[str, int]]:
    """Load question and concept mappings."""
    q_map_path = os.path.join(mappings_dir, "question_mapping.pkl")
    c_map_path = os.path.join(mappings_dir, "concept_mapping.pkl")
    
    if not os.path.exists(q_map_path) or not os.path.exists(c_map_path):
        print(f"Warning: Mappings not found in {mappings_dir}")
        print("Using empty mappings (all IDs will be unknown)")
        return {'questions': {}, 'concepts': {}}
        
    with open(q_map_path, 'rb') as f:
        q_map = pickle.load(f)
    with open(c_map_path, 'rb') as f:
        c_map = pickle.load(f)
        
    return {'questions': q_map, 'concepts': c_map}

def get_id(mapping: Dict[str, int], key: str, default: int = 0) -> int:
    """Get ID from mapping or return default."""
    return mapping.get(key, default)

def run_prediction(model, sample, mappings, device):
    """Run prediction for a single sample."""
    q_map = mappings['questions']
    c_map = mappings['concepts']
    
    # Process history
    history = []
    h_qids = []
    h_cids = []
    h_resps = []
    
    for item in sample['history']:
        qid_str = item['problem_id']
        cid_strs = item.get('concepts', [])
        correct = item['correct']
        
        # Map IDs
        qid = get_id(q_map, qid_str)
        # Use first concept ID for simplicity in sequence, or map all
        # SequenceEncoder expects single concept ID sequence?
        # Actually our data loader often picks the first concept. 
        cid = get_id(c_map, cid_strs[0]) if cid_strs else 0
        
        history.append((qid, cid, correct == 1))
        h_qids.append(qid)
        h_cids.append(cid)
        h_resps.append(1 if correct else 0)
        
    # Process target
    target = sample['target']
    t_qid_str = target['problem_id']
    t_cid_strs = target.get('concepts', [])
    t_q_text = target.get('question_text', "Unknown Question")
    t_c_text = target.get('concept_text', "")
    if not t_c_text and t_cid_strs:
        t_c_text = t_cid_strs[0] # Fallback name
        
    t_qid = get_id(q_map, t_qid_str)
    t_cid = get_id(c_map, t_cid_strs[0]) if t_cid_strs else 0
    
    # Convert to tensors
    h_q_tensor = torch.tensor(h_qids, dtype=torch.long).to(device)
    h_c_tensor = torch.tensor(h_cids, dtype=torch.long).to(device)
    h_r_tensor = torch.tensor(h_resps, dtype=torch.long).to(device)
    h_mask = torch.ones(len(h_qids), dtype=torch.long).to(device) # No padding
    
    # Run prediction
    prediction, prob = model.predict(
        history=history,
        target_question_text=t_q_text,
        target_question_id=t_qid,
        target_concept_text=t_c_text,
        target_concept_id=t_cid,
        history_question_ids=h_q_tensor,
        history_concept_ids=h_c_tensor,
        history_responses=h_r_tensor,
        history_mask=h_mask
    )
    
    ground_truth = target.get('correct')
    
    print("\n------------------------------------------------")
    print(f"User: {sample.get('user_id', 'Unknown')}")
    print(f"History Length: {len(history)}")
    print(f"Target Question: {t_q_text[:50]}...")
    print(f"Concepts: {t_cid_strs}")
    print(f"Prediction: {'Correct (Yes)' if prediction else 'Incorrect (No)'} ({prob:.2%})")
    if ground_truth is not None:
         print(f"Ground Truth: {'Correct' if ground_truth else 'Incorrect'}")
         is_correct = (prediction == ground_truth) or (prediction == 1 and ground_truth == 1) # simple match
         print(f"Result: {'Match' if is_correct else 'Mismatch'}")
    print("------------------------------------------------")

def main():
    args = parse_args()
    Config.use_preset(args.preset)
    
    # Determine mappings directory
    mappings_dir = args.mappings_dir
    if args.processed_dir:
        potential_path = os.path.join(args.processed_dir, "mappings")
        if os.path.exists(potential_path):
            mappings_dir = potential_path
            
    print(f"Loading mappings from {mappings_dir}...")
    mappings = load_mappings_dict(mappings_dir)
    num_q = len(mappings['questions'])
    num_c = len(mappings['concepts'])
    print(f"Loaded {num_q} questions, {num_c} concepts.")
    
    # Initialize Model (requires dimensions from mappings/config)
    # If mappings are empty/small, we might need config defaults or valid counts
    # BUT, we need to match the trained checkpoint dimensions!
    # We should load dimensions from checkpoint or config if possible.
    # We'll use Config values as defaults if mappings are missing, but ideally mappings count should match.
    # HOWEVER, KTEmbeddingModel init needs num_q/num_c.
    # If we use arbitrary numbers, loading state_dict might fail if shapes mismatch.
    # Safest is to rely on Config (which might be updated by preset) or try to infer.
    
    # Actually, Config doesn't store num_questions. 'test.py' loads metadata.
    # We should load metadata.json if available.
    
    meta_path = os.path.join(args.processed_dir, "metadata.json") if args.processed_dir else None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            num_q = metadata.get('num_questions', num_q)
            num_c = metadata.get('num_concepts', num_c)
            # Ensure unique + padding
    else:
        # Fallback: add buffer or use mapping length
        num_q = max(num_q, 10000) # Safe buffer? No, shape mismatch error.
        # We MUST know the exact dimensions the model was trained with.
        # Checkpoint usually doesn't store this metadata easily unless we saved it.
        # But we saved `metadata.json` in processed dir.
        print("Warning: metadata.json not found. Using mapping lengths (might mismatch checkpoint).")
        # Add 1 for padding if not included (usually mapping is 1-based or 0-based?)
        # Our ID mapping starts at 1, 0 is padding. So len(map) + 1.
        num_q = num_q + 1
        num_c = num_c + 1

    print(f"Initializing model with Q={num_q}, C={num_c}")
    
    embedding_model = KTEmbeddingModel(
        num_questions=num_q,
        num_concepts=num_c,
        embed_dim=Config.EMBED_DIM,
        num_seq_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        max_seq_len=Config.MAX_SEQ_LEN,
        device=args.device
    )
    
    model = KnowledgeTracingLLM(
        llm_model_name=Config.LLM_MODEL_NAME,
        embedding_model=embedding_model,
        lora_r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        use_4bit=True,
        device=args.device
    )
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        # Determine if it's a directory or file
        if os.path.isdir(args.checkpoint):
             model.load(args.checkpoint)
        else:
             print("Error: Checkpoint must be a directory containing components")
             sys.exit(1)
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
        
    model.to(args.device)
    model.eval()
    
    # Load input
    if args.interactive:
        print("Interactive mode not yet implemented. Use file input.")
        return
        
    if os.path.exists(args.input):
        with open(args.input, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        print(f"Loaded {len(samples)} samples.")
        for sample in samples:
            run_prediction(model, sample, mappings, args.device)
            
    else:
        print(f"Error: Input file {args.input} not found.")

if __name__ == "__main__":
    main()
