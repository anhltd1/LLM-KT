#!/usr/bin/env python
"""
LLM Fine-tuning CLI for Knowledge Tracing

This script fine-tunes the LLM using the pre-trained AKT encoder.
It loads the processed data and the trained encoder checkpoint,
then trains the LLM to predict student responses using LoRA.
"""

import argparse
import os
import sys
from datetime import datetime
import json
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.embedding_model import KTEmbeddingModel
from models.kt_llm import KnowledgeTracingLLM
from utils import create_data_loaders_from_processed

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Knowledge Tracing")
    
    # Data arguments
    parser.add_argument("--processed-dir", type=str, default=None, help="Path to processed data")
    parser.add_argument("--encoder-path", type=str, default=None, help="Path to pre-trained encoder checkpoint")
    parser.add_argument("--preset", type=str, default="small", choices=["small", "standard", "phi3", "qwen", "llama2"])
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    
    # Model arguments
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    
    return parser.parse_args()

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, grad_accum=1, log_every=10, use_wandb=False):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for i, batch in enumerate(pbar):
        # Move batch to device (handled inside model forward usually, but let's be safe)
        # Actually in train.py it was handled in collate_fn or manually? 
        # KTDataset collate_fn returns tensors.
        
        # Forward pass
        outputs = model(batch)
        loss = outputs['loss']
        
        # Backward pass
        loss = loss / grad_accum
        loss.backward()
        
        if (i + 1) % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Metrics
        current_loss = loss.item() * grad_accum
        total_loss += current_loss
        
        # Calculate accuracy from logits
        # LLM output is next token prediction. 
        # For simple accuracy, we check if the predicted token matches " Correct" or " Incorrect"
        # However, model.forward returns generic causal LM loss. 
        # To get accuracy we usually need generation or checking logits at the answer position.
        
        pbar.set_postfix({'loss': current_loss})
        
        if use_wandb and i % log_every == 0:
            wandb.log({
                "train/loss": current_loss,
                "train/lr": scheduler.get_last_lr()[0]
            })
            
    return {"loss": total_loss / len(train_loader)}

def validate(model, val_loader, device, use_wandb=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # We can perform a few generations to verify
    # But full generation is slow. 
    # For now, let's just report loss on validation set.
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            outputs = model(batch)
            loss = outputs['loss']
            total_loss += loss.item()
            
            # TODO: Implement proper accuracy metric for LLM next-token prediction
            # This requires inspecting the logits at the position corresponding to the answer.
            
    return {"val_loss": total_loss / len(val_loader)}

def main():
    args = parse_args()
    Config.use_preset(args.preset)
    
    # Initialize WandB
    if args.wandb and Config.WANDB_API_KEY:
        run_name = args.run_name or f"finetune_{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=run_name,
            config={
                "preset": args.preset,
                "epochs": args.epochs or Config.LLM_EPOCHS,
                "batch_size": args.batch_size or Config.LLM_BATCH_SIZE,
                "lr": args.lr or Config.LLM_LR
            }
        )
    
    # Check data
    if not args.processed_dir:
        print("Error: --processed-dir is required")
        sys.exit(1)
        
    print(f"\n[Loading Data from {args.processed_dir}]")
    train_loader, val_loader, test_loader, metadata = create_data_loaders_from_processed(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size or Config.LLM_BATCH_SIZE,
        num_workers=0
    )
    
    # Initialize Embedding Model
    print("\n[Initializing Embedding Model]")
    embedding_model = KTEmbeddingModel(
        num_questions=metadata['num_questions'],
        num_concepts=metadata['num_concepts'],
        embed_dim=Config.EMBED_DIM,
        num_seq_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        max_seq_len=metadata.get('max_seq_len', Config.MAX_SEQ_LEN),
        device=args.device
    )
    
    # Load Pre-trained Encoder
    if args.encoder_path:
        print(f"  → Loading encoder weights from {args.encoder_path}")
        if os.path.exists(args.encoder_path):
            state_dict = torch.load(args.encoder_path, map_location=args.device)
            # Handle potential key mismatch if saved from wrapper model
            if all(k.startswith('embedding_model.') for k in state_dict.keys()):
                 state_dict = {k.replace('embedding_model.', ''): v for k, v in state_dict.items()}
            
            embedding_model.load_state_dict(state_dict, strict=False)
            print("  ✓ Encoder weights loaded")
        else:
            print(f"  ✗ Encoder checkpoint not found: {args.encoder_path}")
            sys.exit(1)
    else:
        print("  ⚠️ No encoder checkpoint provided. Training from scratch (random initialization).")
    
    # Initialize LLM
    print("\n[Initializing Knowledge Tracing LLM]")
    model = KnowledgeTracingLLM(
        llm_model_name=Config.LLM_MODEL_NAME,
        embedding_model=embedding_model,
        lora_r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        use_4bit=not args.no_4bit,
        device=args.device
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr or Config.LLM_LR)
    
    total_steps = len(train_loader) * (args.epochs or Config.LLM_EPOCHS)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps//3)
    
    # Training Loop
    epochs = args.epochs or Config.LLM_EPOCHS
    save_dir = Config.LLM_CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n[Starting Fine-tuning for {epochs} epochs]")
    
    for epoch in range(epochs):
        metrics = train_epoch(
            model, train_loader, optimizer, scheduler, 
            args.device, epoch+1, 
            grad_accum=Config.GRADIENT_ACCUMULATION,
            use_wandb=args.wandb
        )
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {metrics['loss']:.4f}")
        
        # Validation
        val_metrics = validate(model, val_loader, args.device)
        print(f"  Validation Loss: {val_metrics['val_loss']:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            # We save the adapter weights and embedding model
            model.save(os.path.join(save_dir, f"epoch_{epoch+1}"))
            print(f"  → Saved checkpoint to {path}")
            
    print("\n[Fine-tuning Complete]")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
