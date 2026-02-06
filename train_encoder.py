#!/usr/bin/env python
"""
Encoder Training CLI for Knowledge Tracing

This script trains the AKT (Augmented Knowledge Tracing) encoder.
It learns to predict student performance using the dual-encoder architecture
(Context + Sequence), but without the LLM decoder.

The trained encoder serves as the embedding foundation for the subsequent LLM fine-tuning.
"""

import argparse
import os
import sys
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.embedding_model import KTEmbeddingModel
from utils import create_data_loaders_from_processed, create_data_loaders

class AKTModel(nn.Module):
    """
    Wrapper model for training the encoder (AKT).
    Adds a prediction head on top of KTEmbeddingModel.
    """
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        # Prediction head: Embedding -> Logit (Probability of correct)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_model.embed_dim, embedding_model.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_model.embed_dim // 2, 1)
        )
    
    def forward(self, batch):
        # Extract batch data
        question_ids = batch['question_ids']
        concept_ids = batch['concept_ids']
        responses = batch['responses']
        
        # Target info
        target_qid = batch['target_question_id']
        target_cid = batch.get('target_concept_ids')
        if target_cid is not None and target_cid.dim() > 1:
            # Take first concept if multiple
            target_cid = target_cid[:, 0]
            
        target_q_text = batch['target_question_text']
        
        # Get embeddings for the target step
        # The embedding model handles combining context (text) and sequence (history)
        # We use 'get_question_embedding' which returns the state before the target response
        embeddings = self.embedding_model.get_question_embedding(
            question_text=target_q_text,
            question_ids=question_ids,
            concept_ids=concept_ids,
            responses=responses,
            project_to_llm=False # We want the raw encoder embedding
        )
        
        # Predict logits
        logits = self.output_layer(embeddings)
        return logits.squeeze(-1)

def parse_args():
    parser = argparse.ArgumentParser(description="Train AKT Encoder")
    
    # Data arguments
    parser.add_argument("--processed-dir", type=str, default=None, help="Path to pre-processed data")
    parser.add_argument("--preset", type=str, default="small", choices=["small", "standard", "phi3", "qwen", "llama2"])
    
    # Model arguments
    parser.add_argument("--embed-dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        optimizer.zero_grad()
        logits = model(batch)
        targets = batch['target_response'].float().to(device)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            logits = model(batch)
            targets = batch['target_response'].float().to(device)
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    if len(all_targets) == 0:
        return 0.0, 0.5, 0.0
    
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Avoid crash if only one class is present in validation batch
        auc = 0.5
        
    acc = accuracy_score(all_targets, [1 if p > 0.5 else 0 for p in all_preds])
    
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return avg_loss, auc, acc

def main():
    args = parse_args()
    Config.use_preset(args.preset)
    
    # Check for processed data
    if not args.processed_dir:
        print("Error: --processed-dir is required")
        sys.exit(1)
        
    # Load Data with auto-download if needed
    from utils.gdrive_downloader import ensure_dataset_files
    # Check if we need to verify file existence (processed dir implies data exists, 
    # but processed dataset creation needs raw files). If processed exists, we assume it's good.
    if not os.path.exists(args.processed_dir):
         print(f"Error: Processed directory {args.processed_dir} not found.")
         print("Please run prepare_data.py first.")
         sys.exit(1)

    print(f"\n[Loading Data from {args.processed_dir}]")
    train_loader, val_loader, test_loader, metadata = create_data_loaders_from_processed(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size or Config.AKT_BATCH_SIZE,
        num_workers=0 # Avoid Windows multiprocessing issues
    )
    
    # Initialize Model
    print("\n[Initializing Model]")
    kt_model = KTEmbeddingModel(
        num_questions=metadata['num_questions'],
        num_concepts=metadata['num_concepts'],
        embed_dim=args.embed_dim or Config.EMBED_DIM,
        num_seq_layers=args.num_layers or Config.NUM_LAYERS,
        num_heads=args.num_heads or Config.NUM_HEADS,
        max_seq_len=metadata.get('max_seq_len', Config.MAX_SEQ_LEN),
        device=args.device
    )
    
    model = AKTModel(kt_model).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr or Config.AKT_LR)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    best_auc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    epochs = args.epochs or Config.AKT_EPOCHS
    
    print(f"\n[Starting Training for {epochs} epochs]")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, args.device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint_dir = Config.AKT_CHECKPOINT_DIR
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, "best_akt_model.pt")
            
            # Save the inner KTEmbeddingModel state dict (for LLM reuse)
            torch.save(model.embedding_model.state_dict(), path)
            print(f"  → Saved new best model to {path}")
            
    # Visualize Training
    print("\n[Saving Training Arguments]")
    os.makedirs("assets", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('AKT Training Loss')
    plt.legend()
    plt.savefig(f"assets/training_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    print("\n[Training Complete]")

if __name__ == "__main__":
    main()
