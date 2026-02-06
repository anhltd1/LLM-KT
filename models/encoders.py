"""
Encoders for Knowledge Tracing with LLM

This module contains two types of encoders:
1. ContextEncoder: Encodes text content (question text, concept descriptions)
2. SequenceEncoder: Encodes learning history sequences

The final embeddings are computed as:
- Question_i = ContextEncoder(question_text) + SequenceEncoder(question_sequence)
- Concept_j = ContextEncoder(concept_text) + SequenceEncoder(concept_sequence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from transformers import AutoModel, AutoTokenizer
import numpy as np


class ContextEncoder(nn.Module):
    """
    Context Encoder for encoding text content.
    
    Uses a pre-trained language model (e.g., multilingual-e5-base for Chinese support)
    to encode question text and concept descriptions into dense embeddings.
    
    For Chinese text, we use multilingual models that support Chinese well.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        output_dim: int = 768,
        freeze_base: bool = True,
        use_projection: bool = True,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Context Encoder.
        
        Args:
            model_name: Pre-trained model for text encoding (supports Chinese)
            output_dim: Output embedding dimension
            freeze_base: Whether to freeze the base model weights
            use_projection: Whether to add a projection layer
            dropout: Dropout rate for projection layer
            device: Device to use
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = device
        
        # Load pre-trained encoder (multilingual for Chinese support)
        print(f"  → Loading ContextEncoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get the encoder's hidden size
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"  → Base model frozen ({self.hidden_size}-dim)")
        
        # Optional projection layer to match output dimension
        self.use_projection = use_projection
        if use_projection and self.hidden_size != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            print(f"  → Projection: {self.hidden_size} → {output_dim}")
        else:
            self.projection = nn.Identity()
            
    def forward(
        self,
        texts: List[str],
        max_length: int = 256,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Encode text content into embeddings.
        
        Args:
            texts: List of text strings to encode
            max_length: Maximum token length
            return_attention: Whether to return attention weights
            
        Returns:
            Text embeddings of shape [batch_size, output_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.encoder(**inputs)
        
        # Use [CLS] token embedding (or mean pooling)
        # For e5 models, use mean pooling over non-padding tokens
        attention_mask = inputs['attention_mask']
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Apply projection
        embeddings = self.projection(embeddings)
        
        if return_attention:
            return embeddings, outputs.attentions
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode a large batch of texts in smaller batches.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Embeddings tensor
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.forward(batch_texts)
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)


class SequenceEncoder(nn.Module):
    """
    Sequence Encoder for encoding learning history.
    
    Uses a Transformer encoder to capture the temporal patterns in:
    - Question sequence: The sequence of questions a student has answered
    - Concept sequence: The concepts the student has learned
    
    This is similar to the AKT model but focused on producing embeddings
    that can be combined with context embeddings.
    """
    
    def __init__(
        self,
        num_questions: int,
        num_concepts: int,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        include_response: bool = True
    ):
        """
        Initialize the Sequence Encoder.
        
        Args:
            num_questions: Total number of unique questions
            num_concepts: Total number of unique concepts
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            include_response: Whether to include response (correct/incorrect) in encoding
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.max_seq_len = max_seq_len
        self.include_response = include_response
        
        # Embeddings for questions and concepts
        # vocab_size already includes padding (0), so no need to add 1
        self.question_embed = nn.Embedding(num_questions, embed_dim, padding_idx=0)
        self.concept_embed = nn.Embedding(num_concepts, embed_dim, padding_idx=0)
        
        # Response embedding (correct/incorrect)
        if include_response:
            self.response_embed = nn.Embedding(3, embed_dim)  # 0: pad, 1: incorrect, 2: correct
        
        # Positional embedding
        self.position_embed = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Question output projection (for question-level embeddings)
        self.question_projection = nn.Linear(embed_dim, embed_dim)
        
        # Concept output projection (for concept-level embeddings)
        self.concept_projection = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        self._init_weights()
        
        print(f"  → SequenceEncoder initialized:")
        print(f"     Questions: {num_questions}, Concepts: {num_concepts}")
        print(f"     Embed dim: {embed_dim}, Layers: {num_layers}, Heads: {num_heads}")
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def forward(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_all_positions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode learning history sequences.
        
        Args:
            question_ids: Question IDs [batch_size, seq_len]
            concept_ids: Concept IDs [batch_size, seq_len]
            responses: Response labels (0/1) [batch_size, seq_len]
            mask: Padding mask [batch_size, seq_len]
            return_all_positions: Return embeddings for all positions
            
        Returns:
            Tuple of (question_embeddings, concept_embeddings)
            Each of shape [batch_size, embed_dim] or [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = question_ids.shape
        device = question_ids.device
        
        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        q_emb = self.question_embed(question_ids)
        c_emb = self.concept_embed(concept_ids)
        p_emb = self.position_embed(positions)
        
        # Combine question and concept embeddings with position
        combined = q_emb + c_emb + p_emb
        
        # Add response embedding if available
        if self.include_response and responses is not None:
            # Clamp responses to 0-1 range, then shift: 0->1 (incorrect), 1->2 (correct)
            response_binary = torch.clamp(responses, 0, 1).long()
            response_idx = response_binary + 1
            r_emb = self.response_embed(response_idx)
            combined = combined + r_emb
        
        # Create attention masks
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        
        # Causal mask for autoregressive encoding
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), 
            diagonal=1
        ).bool()
        
        # Transformer encoding
        hidden = self.transformer(
            combined, 
            mask=causal_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Apply layer norm and dropout
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        
        if return_all_positions:
            # Return embeddings for all positions
            question_seq_emb = self.question_projection(hidden)
            concept_seq_emb = self.concept_projection(hidden)
            return question_seq_emb, concept_seq_emb
        else:
            # Get the last valid position for each sequence
            if mask is not None:
                # Get indices of last valid positions
                seq_lengths = mask.sum(dim=1).long() - 1
                seq_lengths = torch.clamp(seq_lengths, min=0)
                batch_indices = torch.arange(batch_size, device=device)
                last_hidden = hidden[batch_indices, seq_lengths]
            else:
                last_hidden = hidden[:, -1, :]
            
            # Project for question and concept embeddings
            question_emb = self.question_projection(last_hidden)
            concept_emb = self.concept_projection(last_hidden)
            
            return question_emb, concept_emb
    
    def get_question_embedding(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        target_position: int = -1
    ) -> torch.Tensor:
        """
        Get question embedding for a specific position.
        
        The embedding represents the student's state before answering 
        the question at target_position.
        
        Args:
            question_ids: Question IDs [batch_size, seq_len]
            concept_ids: Concept IDs [batch_size, seq_len]
            responses: Response labels [batch_size, seq_len]
            mask: Padding mask [batch_size, seq_len]
            target_position: Position to get embedding for (-1 for last)
            
        Returns:
            Question embeddings [batch_size, embed_dim]
        """
        question_emb, _ = self.forward(
            question_ids, concept_ids, responses, mask, 
            return_all_positions=True
        )
        
        if target_position == -1:
            if mask is not None:
                seq_lengths = mask.sum(dim=1).long() - 1
                seq_lengths = torch.clamp(seq_lengths, min=0)
                batch_indices = torch.arange(question_ids.size(0), device=question_ids.device)
                return question_emb[batch_indices, seq_lengths]
            return question_emb[:, -1, :]
        
        return question_emb[:, target_position, :]
    
    def get_concept_embedding(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        target_position: int = -1
    ) -> torch.Tensor:
        """
        Get concept embedding for a specific position.
        
        Args:
            question_ids: Question IDs
            concept_ids: Concept IDs
            responses: Response labels
            mask: Padding mask
            target_position: Position to get embedding for
            
        Returns:
            Concept embeddings [batch_size, embed_dim]
        """
        _, concept_emb = self.forward(
            question_ids, concept_ids, responses, mask,
            return_all_positions=True
        )
        
        if target_position == -1:
            if mask is not None:
                seq_lengths = mask.sum(dim=1).long() - 1
                seq_lengths = torch.clamp(seq_lengths, min=0)
                batch_indices = torch.arange(question_ids.size(0), device=question_ids.device)
                return concept_emb[batch_indices, seq_lengths]
            return concept_emb[:, -1, :]
        
        return concept_emb[:, target_position, :]


class HybridEncoder(nn.Module):
    """
    Hybrid Encoder that combines Context and Sequence encoders.
    
    Computes final embeddings as:
    - Question_i = ContextEncoder(question_text) + SequenceEncoder(question_seq)
    - Concept_j = ContextEncoder(concept_text) + SequenceEncoder(concept_seq)
    """
    
    def __init__(
        self,
        context_encoder: ContextEncoder,
        sequence_encoder: SequenceEncoder,
        combine_method: str = "add",  # "add", "concat", "gate"
        output_dim: Optional[int] = None
    ):
        """
        Initialize the Hybrid Encoder.
        
        Args:
            context_encoder: Pre-initialized ContextEncoder
            sequence_encoder: Pre-initialized SequenceEncoder
            combine_method: How to combine context and sequence embeddings
            output_dim: Output dimension (for concat method)
        """
        super().__init__()
        
        self.context_encoder = context_encoder
        self.sequence_encoder = sequence_encoder
        self.combine_method = combine_method
        
        context_dim = context_encoder.output_dim
        seq_dim = sequence_encoder.embed_dim
        
        if combine_method == "add":
            assert context_dim == seq_dim, \
                f"Dimensions must match for add: {context_dim} vs {seq_dim}"
            self.output_dim = context_dim
            
        elif combine_method == "concat":
            self.output_dim = output_dim or (context_dim + seq_dim)
            self.fusion = nn.Linear(context_dim + seq_dim, self.output_dim)
            
        elif combine_method == "gate":
            assert context_dim == seq_dim, \
                f"Dimensions must match for gate: {context_dim} vs {seq_dim}"
            self.output_dim = context_dim
            self.gate = nn.Sequential(
                nn.Linear(context_dim * 2, context_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")
        
        print(f"  → HybridEncoder: {combine_method} combination, output_dim={self.output_dim}")
    
    def forward(
        self,
        text: str,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        embedding_type: str = "question"  # "question" or "concept"
    ) -> torch.Tensor:
        """
        Compute hybrid embedding by combining context and sequence.
        
        Args:
            text: Text to encode (question text or concept description)
            question_ids: Sequence of question IDs
            concept_ids: Sequence of concept IDs
            responses: Sequence of responses
            mask: Padding mask
            embedding_type: "question" or "concept"
            
        Returns:
            Combined embedding
        """
        # Get context embedding
        context_emb = self.context_encoder([text])  # [1, dim]
        
        # Get sequence embedding
        if embedding_type == "question":
            seq_emb, _ = self.sequence_encoder(question_ids, concept_ids, responses, mask)
        else:
            _, seq_emb = self.sequence_encoder(question_ids, concept_ids, responses, mask)
        
        # Combine embeddings
        if self.combine_method == "add":
            combined = context_emb + seq_emb
        elif self.combine_method == "concat":
            combined = self.fusion(torch.cat([context_emb, seq_emb], dim=-1))
        elif self.combine_method == "gate":
            gate = self.gate(torch.cat([context_emb, seq_emb], dim=-1))
            combined = gate * context_emb + (1 - gate) * seq_emb
        
        return combined
