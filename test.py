#!/usr/bin/env python
"""
CLI Testing Script for Knowledge Tracing LLM

Usage:
    python test.py --checkpoint checkpoints/llm/small/best_model
    python test.py --checkpoint checkpoints/llm/phi3/best_model --output-file results.json
    python test.py --help

This script evaluates a trained Knowledge Tracing model.
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, List

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, MODEL_PRESETS
from utils import (
    create_data_loaders,
    load_mappings,
    ProblemDatabase
)
from models import (
    KTEmbeddingModel,
    KnowledgeTracingLLM
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Knowledge Tracing LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    
    # Optional preset (for config)
    parser.add_argument(
        "--preset", "-p",
        type=str,
        default="small",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset (used for configuration)"
    )
    
    # Data arguments
    parser.add_argument(
        "--problems-path",
        type=str,
        default=None,
        help="Path to problem.json"
    )
    parser.add_argument(
        "--interactions-path",
        type=str,
        default=None,
        help="Path to interactions file"
    )
    parser.add_argument(
        "--mappings-dir",
        type=str,
        default=None,
        help="Path to mappings directory"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions"
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=None,
        help="Path to save predictions"
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions (0 or 1)
        labels: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = (predictions == labels).mean()
    
    # True/False positives/negatives
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.0
    
    # Matthews Correlation Coefficient
    try:
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(labels, predictions)
    except:
        mcc = 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'mcc': float(mcc),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_samples': len(labels),
        'positive_rate': float(labels.mean()),
        'predicted_positive_rate': float(predictions.mean())
    }


def evaluate_model(
    model: KnowledgeTracingLLM,
    data_loader,
    device: str,
    max_samples: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        max_samples: Maximum samples to evaluate
        verbose: Print detailed output
        
    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_user_ids = []
    all_question_ids = []
    
    total_loss = 0.0
    num_batches = 0
    samples_evaluated = 0
    
    with torch.no_grad():
        progress = tqdm(data_loader, desc="Evaluating", disable=not verbose)
        
        for batch in progress:
            # Forward pass
            outputs = model(batch, compute_embeddings=True)
            
            # Collect predictions
            predictions = outputs['predictions'].cpu().numpy()
            labels = batch['target_response'].numpy()
            
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            all_user_ids.extend(batch['user_id'])
            all_question_ids.extend(batch['target_question_id_str'])
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            samples_evaluated += len(predictions)
            
            # Check max samples
            if max_samples and samples_evaluated >= max_samples:
                break
    
    # Convert to numpy
    all_predictions = np.array(all_predictions[:max_samples] if max_samples else all_predictions)
    all_labels = np.array(all_labels[:max_samples] if max_samples else all_labels)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['avg_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Store predictions
    predictions_data = {
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'user_ids': all_user_ids[:len(all_predictions)],
        'question_ids': all_question_ids[:len(all_predictions)]
    }
    
    return {
        'metrics': metrics,
        'predictions': predictions_data
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Apply preset
    Config.use_preset(args.preset)
    
    # Override batch size if specified
    if args.batch_size:
        Config.LLM_BATCH_SIZE = args.batch_size
    
    # Set paths
    problems_path = args.problems_path or Config.PROBLEM_JSON
    interactions_path = args.interactions_path or Config.STUDENT_JSON
    mappings_dir = args.mappings_dir or os.path.join(
        os.path.dirname(Config.LLM_CHECKPOINT_DIR), "mappings"
    )
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE TRACING LLM - EVALUATION")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Preset: {args.preset}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    
    # Check data files
    if not os.path.exists(problems_path):
        print(f"\n✗ Problem file not found: {problems_path}")
        sys.exit(1)
    
    if not os.path.exists(interactions_path):
        print(f"\n✗ Interactions file not found: {interactions_path}")
        sys.exit(1)
    
    # Create data loaders
    print("\n[Loading Data]")
    train_loader, val_loader, test_loader, metadata = create_data_loaders(
        problems_path=problems_path,
        interactions_path=interactions_path,
        mappings_dir=mappings_dir,
        batch_size=Config.LLM_BATCH_SIZE,
        max_seq_len=Config.MAX_SEQ_LEN,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    # Select data loader
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    print(f"  → Evaluating on {len(data_loader)} batches")
    
    # Create embedding model
    print("\n[Loading Model]")
    embedding_model = KTEmbeddingModel(
        num_questions=metadata['num_questions'],
        num_concepts=metadata['num_concepts'],
        embed_dim=Config.EMBED_DIM,
        context_model="intfloat/multilingual-e5-base",
        num_seq_layers=2,
        num_heads=8,
        max_seq_len=Config.MAX_SEQ_LEN,
        device=args.device
    )
    
    # Create LLM model
    model = KnowledgeTracingLLM(
        llm_model_name=Config.LLM_MODEL_NAME,
        embedding_model=embedding_model,
        lora_r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        use_4bit=True,
        max_prompt_length=Config.MAX_PROMPT_LENGTH,
        device=args.device
    )
    
    # Load checkpoint
    model.load(args.checkpoint)
    print(f"  → Loaded checkpoint from {args.checkpoint}")
    
    # Evaluate
    print(f"\n[Evaluating on {args.split} set]")
    results = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=args.device,
        max_samples=args.max_samples,
        verbose=args.verbose
    )
    
    # Print results
    metrics = results['metrics']
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.2%}")
    print(f"   Recall:    {metrics['recall']:.2%}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   AUC:       {metrics['auc']:.4f}")
    print(f"   MCC:       {metrics['mcc']:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Positives:  {metrics['true_positives']:,}")
    print(f"   True Negatives:  {metrics['true_negatives']:,}")
    print(f"   False Positives: {metrics['false_positives']:,}")
    print(f"   False Negatives: {metrics['false_negatives']:,}")
    
    print(f"\n📋 Dataset Statistics:")
    print(f"   Total Samples:      {metrics['total_samples']:,}")
    print(f"   Positive Rate:      {metrics['positive_rate']:.2%}")
    print(f"   Predicted Positive: {metrics['predicted_positive_rate']:.2%}")
    
    # Save results
    if args.output_file:
        output_data = {
            'checkpoint': args.checkpoint,
            'preset': args.preset,
            'split': args.split,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to {args.output_file}")
    
    # Save predictions
    if args.save_predictions:
        predictions_file = args.predictions_file or os.path.join(
            os.path.dirname(args.checkpoint),
            f"{args.split}_predictions.json"
        )
        
        with open(predictions_file, 'w') as f:
            json.dump(results['predictions'], f)
        
        print(f"✓ Predictions saved to {predictions_file}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return metrics


if __name__ == "__main__":
    main()
