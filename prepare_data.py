#!/usr/bin/env python
"""
Data Preparation CLI for Knowledge Tracing LLM

Usage:
    python prepare_data.py --problems-path dataset/MOOCRadar/problem.json \
                           --interactions-path dataset/MOOCRadar/student-problem-coarse-flattened.json \
                           --output-dir processed/

This script processes raw MOOCRadar data and saves it in an optimized format
for faster training. Run this once before training.
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import (
    ProblemDatabase,
    StudentInteractions,
    ConceptMapper,
    QuestionMapper,
    build_mappings_from_dataset
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare MOOCRadar data for Knowledge Tracing training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input paths
    parser.add_argument(
        "--problems-path",
        type=str,
        default=None,
        help="Path to problem.json (default: from config)"
    )
    parser.add_argument(
        "--interactions-path",
        type=str,
        default=None,
        help="Path to student-problem-coarse-flattened.json (default: from config)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed",
        help="Directory to save processed data"
    )
    
    # Split ratios
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    
    # Sequence parameters
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=200,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=5,
        help="Minimum sequence length for a sample"
    )
    
    # Limit data for testing
    parser.add_argument(
        "--max-students",
        type=int,
        default=None,
        help="Maximum number of students to process (for testing)"
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=None,
        help="Maximum number of interactions per student"
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["small", "standard", "phi3", "qwen", "llama2"],
        help="Use preset configuration for data limits"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_samples_for_student(
    user_id: str,
    interactions: List[Dict],
    problem_db: ProblemDatabase,
    question_mapper: QuestionMapper,
    concept_mapper: ConceptMapper,
    max_seq_len: int,
    min_seq_len: int
) -> List[Dict]:
    """
    Build training samples from a single student's interactions.
    
    Each sample contains:
    - History of question_ids, concept_ids, responses
    - Target question, concepts, and response
    - Text content for context encoding
    """
    samples = []
    
    if len(interactions) < min_seq_len:
        return samples
    
    for i in range(min_seq_len, len(interactions)):
        history = interactions[:i]
        target = interactions[i]
        
        # Limit history length
        if len(history) > max_seq_len:
            history = history[-max_seq_len:]
        
        # Process history
        question_ids = []
        concept_ids = []
        responses = []
        
        for inter in history:
            pid = inter.get('problem_id', '')
            
            # Question ID
            qid = question_mapper.get_id(pid, add_if_missing=False)
            question_ids.append(qid if qid != -1 else 0)
            
            # Concept ID (first concept)
            concepts = problem_db.get_concepts(pid)
            if concepts:
                cid = concept_mapper.get_id(concepts[0], add_if_missing=False)
                concept_ids.append(cid if cid != -1 else 0)
            else:
                concept_ids.append(0)
            
            # Response
            response = inter.get('is_correct', inter.get('score', 0))
            if isinstance(response, float):
                response = 1 if response > 0.5 else 0
            responses.append(response)
        
        # Process target
        target_pid = target.get('problem_id', '')
        target_qid = question_mapper.get_id(target_pid, add_if_missing=False)
        target_qid = target_qid if target_qid != -1 else 0
        
        target_concepts = problem_db.get_concepts(target_pid)
        target_concept_ids = []
        target_concept_texts = []
        for concept in target_concepts[:3]:
            cid = concept_mapper.get_id(concept, add_if_missing=False)
            target_concept_ids.append(cid if cid != -1 else 0)
            target_concept_texts.append(problem_db.get_concept_description(concept))
        
        # Pad concept IDs to 3
        while len(target_concept_ids) < 3:
            target_concept_ids.append(0)
            target_concept_texts.append("")
        
        target_response = target.get('is_correct', target.get('score', 0))
        if isinstance(target_response, float):
            target_response = 1 if target_response > 0.5 else 0
        
        target_question_text = problem_db.get_text(target_pid)
        
        sample = {
            'user_id': user_id,
            'question_ids': question_ids,
            'concept_ids': concept_ids,
            'responses': responses,
            'seq_len': len(question_ids),
            'target_question_id': target_qid,
            'target_question_id_str': target_pid,
            'target_concept_ids': target_concept_ids,
            'target_response': target_response,
            'target_question_text': target_question_text,
            'target_concept_texts': target_concept_texts
        }
        samples.append(sample)
    
    return samples


def process_and_split_data(
    problem_db: ProblemDatabase,
    interactions: StudentInteractions,
    question_mapper: QuestionMapper,
    concept_mapper: ConceptMapper,
    train_ratio: float,
    val_ratio: float,
    max_seq_len: int,
    min_seq_len: int,
    max_students: int = None,
    max_interactions: int = None,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Process interactions and split into train/val/test.
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    set_seed(seed)
    
    # Get all students
    all_students = interactions.get_all_students()
    random.shuffle(all_students)
    
    # Limit students if specified
    if max_students:
        all_students = all_students[:max_students]
    
    # Split students
    n_train = int(len(all_students) * train_ratio)
    n_val = int(len(all_students) * val_ratio)
    
    train_students = all_students[:n_train]
    val_students = all_students[n_train:n_train + n_val]
    test_students = all_students[n_train + n_val:]
    
    print(f"  → Student split: Train={len(train_students)}, Val={len(val_students)}, Test={len(test_students)}")
    
    # Build samples for each split
    def build_split_samples(student_list: List[str], split_name: str) -> List[Dict]:
        samples = []
        for user_id in tqdm(student_list, desc=f"Building {split_name} samples"):
            user_interactions = interactions.get_student_sequence(
                user_id,
                max_length=max_interactions
            )
            student_samples = build_samples_for_student(
                user_id, user_interactions,
                problem_db, question_mapper, concept_mapper,
                max_seq_len, min_seq_len
            )
            samples.extend(student_samples)
        return samples
    
    train_samples = build_split_samples(train_students, "train")
    val_samples = build_split_samples(val_students, "val")
    test_samples = build_split_samples(test_students, "test")
    
    return train_samples, val_samples, test_samples


def save_processed_data(
    output_dir: str,
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    question_mapper: QuestionMapper,
    concept_mapper: ConceptMapper,
    max_seq_len: int,
    config: Dict[str, Any]
):
    """Save processed data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    mappings_dir = os.path.join(output_dir, "mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
    # Save samples
    print(f"\n  → Saving train samples ({len(train_samples)})...")
    torch.save(train_samples, os.path.join(output_dir, "train.pt"))
    
    print(f"  → Saving val samples ({len(val_samples)})...")
    torch.save(val_samples, os.path.join(output_dir, "val.pt"))
    
    print(f"  → Saving test samples ({len(test_samples)})...")
    torch.save(test_samples, os.path.join(output_dir, "test.pt"))
    
    # Save mappings
    print("  → Saving mappings...")
    question_mapper.save(os.path.join(mappings_dir, "question_mapping.pkl"))
    concept_mapper.save(os.path.join(mappings_dir, "concept_mapping.pkl"))
    
    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "num_questions": question_mapper.vocab_size,
        "num_concepts": concept_mapper.vocab_size,
        "max_seq_len": max_seq_len,
        "config": config
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  → Saved metadata to {os.path.join(output_dir, 'metadata.json')}")


def main():
    """Main data preparation function."""
    args = parse_args()
    
    # Apply preset if specified
    if args.preset:
        Config.use_preset(args.preset)
        if args.max_students is None:
            args.max_students = getattr(Config, 'MAX_USERS', None)
        if args.max_seq_len == 200:  # Only override if default
            args.max_seq_len = getattr(Config, 'MAX_SEQ_LEN', 200)
    
    # Set seed
    set_seed(args.seed)
    
    # Set paths
    problems_path = args.problems_path or Config.PROBLEM_JSON
    interactions_path = args.interactions_path or Config.STUDENT_JSON
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE TRACING - DATA PREPARATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Problems:      {problems_path}")
    print(f"  Interactions:  {interactions_path}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Train/Val:     {args.train_ratio}/{args.val_ratio}")
    print(f"  Max Seq Len:   {args.max_seq_len}")
    print(f"  Min Seq Len:   {args.min_seq_len}")
    print(f"  Max Students:  {args.max_students or 'All'}")
    print(f"  Seed:          {args.seed}")
    
    # Check files exist
    if not os.path.exists(problems_path):
        print(f"\n✗ Problem file not found: {problems_path}")
        sys.exit(1)
    
    if not os.path.exists(interactions_path):
        print(f"\n✗ Interactions file not found: {interactions_path}")
        sys.exit(1)
    
    # Load data
    print("\n[Loading Data]")
    problem_db = ProblemDatabase().load_from_file(problems_path)
    interactions = StudentInteractions().load_from_file(interactions_path)
    interactions.set_problem_database(problem_db)
    
    # Build mappings
    print("\n[Building Mappings]")
    question_mapper, concept_mapper = build_mappings_from_dataset(
        problems_path,
        args.output_dir  # Save to output dir temporarily
    )
    
    # Process and split data
    print("\n[Processing Data]")
    train_samples, val_samples, test_samples = process_and_split_data(
        problem_db=problem_db,
        interactions=interactions,
        question_mapper=question_mapper,
        concept_mapper=concept_mapper,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        max_students=args.max_students,
        max_interactions=args.max_interactions,
        seed=args.seed
    )
    
    print(f"\n  → Total samples: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Save processed data
    print("\n[Saving Processed Data]")
    config = {
        "problems_path": problems_path,
        "interactions_path": interactions_path,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "max_seq_len": args.max_seq_len,
        "min_seq_len": args.min_seq_len,
        "max_students": args.max_students,
        "seed": args.seed,
        "preset": args.preset
    }
    
    save_processed_data(
        output_dir=args.output_dir,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        question_mapper=question_mapper,
        concept_mapper=concept_mapper,
        max_seq_len=args.max_seq_len,
        config=config
    )
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved to: {args.output_dir}/")
    print(f"  - train.pt ({len(train_samples)} samples)")
    print(f"  - val.pt ({len(val_samples)} samples)")
    print(f"  - test.pt ({len(test_samples)} samples)")
    print(f"  - metadata.json")
    print(f"  - mappings/question_mapping.pkl")
    print(f"  - mappings/concept_mapping.pkl")
    print(f"\nNext step: python train.py --processed-dir {args.output_dir}")


if __name__ == "__main__":
    main()
