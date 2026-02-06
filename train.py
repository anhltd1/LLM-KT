#!/usr/bin/env python
"""
CLI Training Script for Knowledge Tracing LLM

Usage:
    python train.py --preset small --epochs 5
    python train.py --preset phi3 --batch-size 2 --lr 5e-5
    python train.py --help

This script provides command-line training for the Knowledge Tracing model.
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, MODEL_PRESETS
from utils import (
    create_data_loaders,
    create_data_loaders_from_processed,
    build_mappings_from_dataset,
    load_mappings
)
from models import (
    KTEmbeddingModel,
    KnowledgeTracingLLM
)

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Knowledge Tracing LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Preset selection
    parser.add_argument(
        "--preset", "-p",
        type=str,
        default="small",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset to use"
    )
    
    # Data arguments
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to pre-processed data directory (from prepare_data.py)"
    )
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
        help="Path to interactions file (default: from config)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=5,
        help="Minimum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps"
    )
    
    # Model arguments
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--num-seq-layers",
        type=int,
        default=2,
        help="Number of sequence encoder layers"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--context-model",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="Pre-trained model for context encoding (supports Chinese)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log metrics every N steps"
    )
    
    # Experiment tracking
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run"
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
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: KnowledgeTracingLLM,
    train_loader,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    grad_accum: int = 1,
    log_every: int = 10,
    use_wandb: bool = False
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        leave=True
    )
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Forward pass
        outputs = model(batch)
        
        loss = outputs['loss'] / grad_accum
        loss.backward()
        
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += outputs['loss'].item()
        total_ce_loss += outputs['ce_loss'].item()
        total_accuracy += outputs['accuracy'].item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        avg_acc = total_accuracy / num_batches
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{avg_acc:.2%}'
        })
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE and (step + 1) % log_every == 0:
            wandb.log({
                'train/loss': avg_loss,
                'train/ce_loss': total_ce_loss / num_batches,
                'train/accuracy': avg_acc,
                'train/lr': scheduler.get_last_lr()[0]
            })
    
    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }


def validate(
    model: KnowledgeTracingLLM,
    val_loader,
    device: str,
    use_wandb: bool = False
):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            outputs = model(batch, compute_embeddings=True)
            
            total_loss += outputs['loss'].item()
            total_accuracy += outputs['accuracy'].item()
            
            all_preds.extend(outputs['predictions'].cpu().tolist())
            all_labels.extend(batch['target_response'].tolist())
    
    num_batches = len(val_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    
    # Calculate AUC if possible
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_accuracy': accuracy,
        'val_auc': auc
    }
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({f'val/{k}': v for k, v in metrics.items()})
    
    return metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Apply preset
    Config.use_preset(args.preset)
    
    # Override config with command line arguments
    if args.epochs is not None:
        Config.LLM_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.LLM_BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LLM_LR = args.lr
    if args.embed_dim is not None:
        Config.EMBED_DIM = args.embed_dim
    if args.max_seq_len is not None:
        Config.MAX_SEQ_LEN = args.max_seq_len
    if args.lora_r is not None:
        Config.LORA_R = args.lora_r
    if args.lora_alpha is not None:
        Config.LORA_ALPHA = args.lora_alpha
    if args.lora_dropout is not None:
        Config.LORA_DROPOUT = args.lora_dropout
    if args.grad_accum is not None:
        Config.GRADIENT_ACCUMULATION = args.grad_accum
    
    # Set paths
    output_dir = args.output_dir or Config.LLM_CHECKPOINT_DIR
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE TRACING LLM - TRAINING")
    print("=" * 70)
    Config.print_config()
    
    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE and Config.WANDB_API_KEY
    if use_wandb:
        run_name = args.run_name or f"{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project or Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=run_name,
            config={
                'preset': args.preset,
                'epochs': Config.LLM_EPOCHS,
                'batch_size': Config.LLM_BATCH_SIZE,
                'lr': Config.LLM_LR,
                'embed_dim': Config.EMBED_DIM,
                'llm_model': Config.LLM_MODEL_NAME,
                'lora_r': Config.LORA_R,
                'lora_alpha': Config.LORA_ALPHA
            }
        )
        print(f"\n✓ Wandb initialized: {run_name}")
    
    # Create data loaders
    print("\n[Loading Data]")
    
    if args.processed_dir:
        # Load from pre-processed data (faster)
        if not os.path.exists(args.processed_dir):
            print(f"\n✗ Processed data directory not found: {args.processed_dir}")
            print("  Please run: python prepare_data.py --output-dir <dir>")
            sys.exit(1)
        
        train_loader, val_loader, test_loader, metadata = create_data_loaders_from_processed(
            processed_dir=args.processed_dir,
            batch_size=Config.LLM_BATCH_SIZE,
            max_seq_len=Config.MAX_SEQ_LEN if args.max_seq_len is None else args.max_seq_len,
            num_workers=args.num_workers
        )
    else:
        # Load from raw JSON files
        problems_path = args.problems_path or Config.PROBLEM_JSON
        interactions_path = args.interactions_path or Config.STUDENT_JSON
        mappings_dir = os.path.join(os.path.dirname(output_dir), "mappings")
        os.makedirs(mappings_dir, exist_ok=True)
        
        # Check files exist, download from Google Drive if needed
        from utils.gdrive_downloader import ensure_dataset_files
        
        files_ok = True
        if not os.path.exists(problems_path) or not os.path.exists(interactions_path):
            print("\n[Checking/Downloading Dataset Files]")
            files_ok = ensure_dataset_files(Config.DATA_DIR, Config.GDRIVE_FILES)
            
            if not os.path.exists(problems_path):
                print(f"\n[FAIL] Problem file not found: {problems_path}")
                files_ok = False
            if not os.path.exists(interactions_path):
                print(f"\n[FAIL] Interactions file not found: {interactions_path}")
                files_ok = False
        
        if not files_ok:
            print("\nCould not obtain dataset files.")
            print("  Use --processed-dir to load pre-processed data, or check network connection.")
            sys.exit(1)
        
        train_loader, val_loader, test_loader, metadata = create_data_loaders(
            problems_path=problems_path,
            interactions_path=interactions_path,
            mappings_dir=mappings_dir,
            batch_size=Config.LLM_BATCH_SIZE,
            max_seq_len=Config.MAX_SEQ_LEN,
            min_seq_len=args.min_seq_len,
            seed=args.seed,
            num_workers=args.num_workers
        )
    
    num_questions = metadata['num_questions']
    num_concepts = metadata['num_concepts']
    
    print(f"  → Train batches: {len(train_loader)}")
    print(f"  → Val batches: {len(val_loader)}")
    print(f"  → Test batches: {len(test_loader)}")
    
    # Create embedding model
    print("\n[Creating Embedding Model]")
    embedding_model = KTEmbeddingModel(
        num_questions=num_questions,
        num_concepts=num_concepts,
        embed_dim=Config.EMBED_DIM,
        context_model=args.context_model,
        num_seq_layers=args.num_seq_layers,
        num_heads=args.num_heads,
        max_seq_len=Config.MAX_SEQ_LEN,
        dropout=Config.LORA_DROPOUT,
        freeze_context=True,
        combine_method="add",
        use_cache=True,
        device=args.device
    )
    
    # Create LLM model
    print("\n[Creating Knowledge Tracing LLM]")
    model = KnowledgeTracingLLM(
        llm_model_name=Config.LLM_MODEL_NAME,
        embedding_model=embedding_model,
        lora_r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        lora_target_modules=Config.LORA_TARGET_MODULES,
        use_4bit=not args.no_4bit,
        max_prompt_length=Config.MAX_PROMPT_LENGTH,
        num_soft_tokens=Config.NUM_SOFT_TOKENS,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\n[Resuming from {args.resume}]")
        model.load(args.resume)
        # Try to extract epoch number from checkpoint name
        try:
            start_epoch = int(args.resume.split('epoch_')[-1].split('.')[0]) + 1
        except:
            pass
    
    # Set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LLM_LR,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * Config.LLM_EPOCHS // Config.GRADIENT_ACCUMULATION
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // 3,
        T_mult=2
    )
    
    # Training loop
    print("\n[Starting Training]")
    best_val_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, Config.LLM_EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{Config.LLM_EPOCHS}")
        print(f"{'=' * 50}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            epoch=epoch + 1,
            grad_accum=Config.GRADIENT_ACCUMULATION,
            log_every=args.log_every,
            use_wandb=use_wandb
        )
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=args.device,
            use_wandb=use_wandb
        )
        
        print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.2%}, AUC: {val_metrics['val_auc']:.4f}")
        
        # Save best model
        if val_metrics['val_accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['val_accuracy']
            best_epoch = epoch + 1
            
            best_path = os.path.join(output_dir, "best_model")
            model.save(best_path)
            print(f"\n✓ New best model saved (acc: {best_val_accuracy:.2%})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch + 1}")
            model.save(ckpt_path)
            print(f"✓ Checkpoint saved: {ckpt_path}")
    
    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    
    # Load best model
    model.load(os.path.join(output_dir, "best_model"))
    
    test_metrics = validate(
        model=model,
        val_loader=test_loader,
        device=args.device,
        use_wandb=use_wandb
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['val_loss']:.4f}")
    print(f"  Accuracy: {test_metrics['val_accuracy']:.2%}")
    print(f"  AUC: {test_metrics['val_auc']:.4f}")
    
    # Save final results
    results = {
        'preset': args.preset,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_metrics['val_accuracy'],
        'test_auc': test_metrics['val_auc'],
        'train_epochs': Config.LLM_EPOCHS,
        'config': {
            'llm_model': Config.LLM_MODEL_NAME,
            'embed_dim': Config.EMBED_DIM,
            'batch_size': Config.LLM_BATCH_SIZE,
            'lr': Config.LLM_LR,
            'lora_r': Config.LORA_R,
            'lora_alpha': Config.LORA_ALPHA
        }
    }
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    if use_wandb:
        wandb.log({
            'test/accuracy': test_metrics['val_accuracy'],
            'test/auc': test_metrics['val_auc']
        })
        wandb.finish()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
