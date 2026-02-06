"""
Embedding Model for Knowledge Tracing

This module combines the ContextEncoder and SequenceEncoder to produce
final question and concept embeddings for the LLM prompt.

The embeddings are structured as:
- QuestionEmbedding: context(question_text) + sequence(question_history)
- ConceptEmbedding: context(concept_text) + sequence(concept_history)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import json
import os

from .encoders import ContextEncoder, SequenceEncoder, HybridEncoder


class EmbeddingCache:
    """
    Cache for pre-computed embeddings to avoid redundant computation.
    
    Caches both context embeddings (text-based) and sequence embeddings.
    """
    
    def __init__(self, cache_dir: str = "./cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.context_cache: Dict[str, torch.Tensor] = {}
        self.sequence_cache: Dict[str, torch.Tensor] = {}
    
    def get_context_embedding(self, key: str) -> Optional[torch.Tensor]:
        """Get cached context embedding."""
        return self.context_cache.get(key)
    
    def set_context_embedding(self, key: str, embedding: torch.Tensor):
        """Cache context embedding."""
        self.context_cache[key] = embedding.detach().cpu()
    
    def save_context_cache(self, filename: str = "context_embeddings.pt"):
        """Save context embeddings to disk."""
        path = os.path.join(self.cache_dir, filename)
        torch.save(self.context_cache, path)
        print(f"  → Saved {len(self.context_cache)} context embeddings to {path}")
    
    def load_context_cache(self, filename: str = "context_embeddings.pt"):
        """Load context embeddings from disk."""
        path = os.path.join(self.cache_dir, filename)
        if os.path.exists(path):
            self.context_cache = torch.load(path)
            print(f"  → Loaded {len(self.context_cache)} context embeddings from {path}")
            return True
        return False
    
    def clear(self):
        """Clear all caches."""
        self.context_cache.clear()
        self.sequence_cache.clear()


class QuestionEmbedding(nn.Module):
    """
    Question Embedding Module.
    
    Combines:
    - Context embedding of question text (content, options)
    - Sequence embedding of the student's question history
    
    Final embedding = context_embed + sequence_embed
    """
    
    def __init__(
        self,
        context_encoder: ContextEncoder,
        sequence_encoder: SequenceEncoder,
        combine_method: str = "add",
        use_cache: bool = True
    ):
        super().__init__()
        
        self.context_encoder = context_encoder
        self.sequence_encoder = sequence_encoder
        self.combine_method = combine_method
        
        self.embed_dim = context_encoder.output_dim
        assert sequence_encoder.embed_dim == self.embed_dim, \
            f"Dimension mismatch: context={self.embed_dim}, sequence={sequence_encoder.embed_dim}"
        
        self.use_cache = use_cache
        if use_cache:
            self.cache = EmbeddingCache()
        
        # Fusion layer for more complex combination
        if combine_method == "gate":
            self.gate = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.Sigmoid()
            )
        elif combine_method == "attention":
            self.attention = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
    
    def forward(
        self,
        question_text: str,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        question_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute question embedding.
        
        Args:
            question_text: The question text content
            question_ids: History of question IDs [batch_size, seq_len]
            concept_ids: History of concept IDs [batch_size, seq_len]
            responses: History of responses [batch_size, seq_len]
            mask: Padding mask [batch_size, seq_len]
            question_id: Optional question ID for caching
            
        Returns:
            Question embedding [batch_size, embed_dim]
        """
        device = question_ids.device
        
        # Get context embedding (can be cached)
        if self.use_cache and question_id:
            context_emb = self.cache.get_context_embedding(f"q_{question_id}")
            if context_emb is not None:
                context_emb = context_emb.to(device)
            else:
                context_emb = self.context_encoder([question_text])
                self.cache.set_context_embedding(f"q_{question_id}", context_emb)
        else:
            # Handle batch input (list/tuple) vs single input
            if isinstance(question_text, (list, tuple)):
                context_emb = self.context_encoder(list(question_text))
            else:
                context_emb = self.context_encoder([question_text])
        
        # Expand context_emb for batch
        batch_size = question_ids.size(0)
        if context_emb.size(0) == 1 and batch_size > 1:
            context_emb = context_emb.expand(batch_size, -1)
        
        # Get sequence embedding (question-focused)
        seq_emb = self.sequence_encoder.get_question_embedding(
            question_ids, concept_ids, responses, mask
        )
        
        # Combine embeddings
        if self.combine_method == "add":
            return context_emb + seq_emb
        elif self.combine_method == "gate":
            gate = self.gate(torch.cat([context_emb, seq_emb], dim=-1))
            return gate * context_emb + (1 - gate) * seq_emb
        else:
            return context_emb + seq_emb
    
    def batch_compute(
        self,
        question_texts: List[str],
        question_ids_list: List[str],
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute embeddings for a batch of questions.
        
        Args:
            question_texts: List of question texts
            question_ids_list: List of question ID strings (for caching)
            question_ids: Question ID sequences [batch_size, seq_len]
            concept_ids: Concept ID sequences [batch_size, seq_len]
            responses: Response sequences [batch_size, seq_len]
            mask: Padding mask [batch_size, seq_len]
            
        Returns:
            Question embeddings [batch_size, embed_dim]
        """
        # Get context embeddings (batch)
        context_emb = self.context_encoder(question_texts)
        
        # Get sequence embeddings
        seq_emb = self.sequence_encoder.get_question_embedding(
            question_ids, concept_ids, responses, mask
        )
        
        if self.combine_method == "add":
            return context_emb + seq_emb
        elif self.combine_method == "gate":
            gate = self.gate(torch.cat([context_emb, seq_emb], dim=-1))
            return gate * context_emb + (1 - gate) * seq_emb
        else:
            return context_emb + seq_emb


class ConceptEmbedding(nn.Module):
    """
    Concept Embedding Module.
    
    Combines:
    - Context embedding of concept description (concept name, knowledge type)
    - Sequence embedding of the student's concept learning history
    
    Final embedding = context_embed + sequence_embed
    """
    
    def __init__(
        self,
        context_encoder: ContextEncoder,
        sequence_encoder: SequenceEncoder,
        combine_method: str = "add",
        use_cache: bool = True
    ):
        super().__init__()
        
        self.context_encoder = context_encoder
        self.sequence_encoder = sequence_encoder
        self.combine_method = combine_method
        
        self.embed_dim = context_encoder.output_dim
        assert sequence_encoder.embed_dim == self.embed_dim, \
            f"Dimension mismatch: context={self.embed_dim}, sequence={sequence_encoder.embed_dim}"
        
        self.use_cache = use_cache
        if use_cache:
            self.cache = EmbeddingCache()
        
        # Fusion layer
        if combine_method == "gate":
            self.gate = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        concept_text: str,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        concept_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute concept embedding.
        
        Args:
            concept_text: The concept description text
            question_ids: History of question IDs [batch_size, seq_len]
            concept_ids: History of concept IDs [batch_size, seq_len]
            responses: History of responses [batch_size, seq_len]
            mask: Padding mask [batch_size, seq_len]
            concept_id: Optional concept ID for caching
            
        Returns:
            Concept embedding [batch_size, embed_dim]
        """
        device = question_ids.device
        
        # Get context embedding (can be cached)
        if self.use_cache and concept_id:
            context_emb = self.cache.get_context_embedding(f"c_{concept_id}")
            if context_emb is not None:
                context_emb = context_emb.to(device)
            else:
                context_emb = self.context_encoder([concept_text])
                self.cache.set_context_embedding(f"c_{concept_id}", context_emb)
        else:
            # Handle batch input (list/tuple) vs single input
            if isinstance(concept_text, (list, tuple)):
                context_emb = self.context_encoder(list(concept_text))
            else:
                context_emb = self.context_encoder([concept_text])
        
        # Expand for batch
        batch_size = question_ids.size(0)
        if context_emb.size(0) == 1 and batch_size > 1:
            context_emb = context_emb.expand(batch_size, -1)
        
        # Get sequence embedding (concept-focused)
        seq_emb = self.sequence_encoder.get_concept_embedding(
            question_ids, concept_ids, responses, mask
        )
        
        # Combine embeddings
        if self.combine_method == "add":
            return context_emb + seq_emb
        elif self.combine_method == "gate":
            gate = self.gate(torch.cat([context_emb, seq_emb], dim=-1))
            return gate * context_emb + (1 - gate) * seq_emb
        else:
            return context_emb + seq_emb


class KTEmbeddingModel(nn.Module):
    """
    Knowledge Tracing Embedding Model.
    
    Main class that manages both question and concept embeddings
    for the Knowledge Tracing LLM system.
    
    This model produces embeddings that will be injected into the LLM prompt
    using special tokens like [QuesEmbed38] and [ConcEmbed219].
    """
    
    def __init__(
        self,
        num_questions: int,
        num_concepts: int,
        embed_dim: int = 768,
        context_model: str = "intfloat/multilingual-e5-base",
        num_seq_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        freeze_context: bool = True,
        combine_method: str = "add",
        use_cache: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the KT Embedding Model.
        
        Args:
            num_questions: Total number of unique questions
            num_concepts: Total number of unique concepts
            embed_dim: Embedding dimension (should match LLM hidden size or have projection)
            context_model: Pre-trained model for context encoding
            num_seq_layers: Number of transformer layers in sequence encoder
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            freeze_context: Whether to freeze context encoder weights
            combine_method: How to combine context and sequence ("add", "gate")
            use_cache: Whether to cache context embeddings
            device: Device to use
        """
        super().__init__()
        
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        self.device = device
        
        print(f"\n[KTEmbeddingModel] Initializing...")
        print(f"  → Questions: {num_questions}, Concepts: {num_concepts}")
        print(f"  → Embed dim: {embed_dim}, Combine: {combine_method}")
        
        # Initialize Context Encoder
        self.context_encoder = ContextEncoder(
            model_name=context_model,
            output_dim=embed_dim,
            freeze_base=freeze_context,
            use_projection=True,
            dropout=dropout,
            device=device
        )
        
        # Initialize Sequence Encoder
        self.sequence_encoder = SequenceEncoder(
            num_questions=num_questions,
            num_concepts=num_concepts,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_seq_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            include_response=True
        )
        
        # Initialize Question and Concept Embedding modules
        self.question_embedding = QuestionEmbedding(
            context_encoder=self.context_encoder,
            sequence_encoder=self.sequence_encoder,
            combine_method=combine_method,
            use_cache=use_cache
        )
        
        self.concept_embedding = ConceptEmbedding(
            context_encoder=self.context_encoder,
            sequence_encoder=self.sequence_encoder,
            combine_method=combine_method,
            use_cache=use_cache
        )
        
        # Projection to LLM hidden size (if needed)
        self.llm_projection = None
        
        print(f"  → KTEmbeddingModel ready!")
    
    def set_llm_projection(self, llm_hidden_size: int):
        """
        Add projection layer to match LLM hidden size.
        
        Args:
            llm_hidden_size: The hidden size of the LLM
        """
        if llm_hidden_size != self.embed_dim:
            self.llm_projection = nn.Linear(self.embed_dim, llm_hidden_size)
            print(f"  → Added LLM projection: {self.embed_dim} → {llm_hidden_size}")
        else:
            self.llm_projection = nn.Identity()
    
    def get_question_embedding(
        self,
        question_text: str,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        question_id: Optional[str] = None,
        project_to_llm: bool = True
    ) -> torch.Tensor:
        """
        Get question embedding for LLM prompt.
        
        Args:
            question_text: Question content text
            question_ids: History of question IDs
            concept_ids: History of concept IDs
            responses: History of responses
            mask: Padding mask
            question_id: Question ID for caching
            project_to_llm: Whether to project to LLM hidden size
            
        Returns:
            Question embedding
        """
        emb = self.question_embedding(
            question_text, question_ids, concept_ids, responses, mask, question_id
        )
        
        if project_to_llm and self.llm_projection is not None:
            emb = self.llm_projection(emb)
        
        return emb
    
    def get_concept_embedding(
        self,
        concept_text: str,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        concept_id: Optional[str] = None,
        project_to_llm: bool = True
    ) -> torch.Tensor:
        """
        Get concept embedding for LLM prompt.
        
        Args:
            concept_text: Concept description text
            question_ids: History of question IDs
            concept_ids: History of concept IDs
            responses: History of responses
            mask: Padding mask
            concept_id: Concept ID for caching
            project_to_llm: Whether to project to LLM hidden size
            
        Returns:
            Concept embedding
        """
        emb = self.concept_embedding(
            concept_text, question_ids, concept_ids, responses, mask, concept_id
        )
        
        if project_to_llm and self.llm_projection is not None:
            emb = self.llm_projection(emb)
        
        return emb
    
    def get_embeddings_for_prompt(
        self,
        question_text: str,
        question_id: str,
        concept_text: str,
        concept_id: str,
        history_question_ids: torch.Tensor,
        history_concept_ids: torch.Tensor,
        history_responses: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both question and concept embeddings for a prompt.
        
        Args:
            question_text: Target question text
            question_id: Target question ID
            concept_text: Target concept description
            concept_id: Target concept ID
            history_question_ids: Historical question IDs
            history_concept_ids: Historical concept IDs
            history_responses: Historical responses
            mask: Padding mask
            
        Returns:
            Tuple of (question_embedding, concept_embedding)
        """
        q_emb = self.get_question_embedding(
            question_text, history_question_ids, history_concept_ids,
            history_responses, mask, question_id
        )
        
        c_emb = self.get_concept_embedding(
            concept_text, history_question_ids, history_concept_ids,
            history_responses, mask, concept_id
        )
        
        return q_emb, c_emb
    
    def save_embeddings(self, save_dir: str):
        """Save cached embeddings to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save question embedding cache
        q_cache_path = os.path.join(save_dir, "question_context_cache.pt")
        if self.question_embedding.cache:
            self.question_embedding.cache.save_context_cache("question_context_cache.pt")
        
        # Save concept embedding cache
        c_cache_path = os.path.join(save_dir, "concept_context_cache.pt")
        if self.concept_embedding.cache:
            self.concept_embedding.cache.save_context_cache("concept_context_cache.pt")
        
        print(f"  → Embedding caches saved to {save_dir}")
    
    def load_embeddings(self, save_dir: str):
        """Load cached embeddings from disk."""
        if self.question_embedding.cache:
            self.question_embedding.cache.load_context_cache("question_context_cache.pt")
        
        if self.concept_embedding.cache:
            self.concept_embedding.cache.load_context_cache("concept_context_cache.pt")
