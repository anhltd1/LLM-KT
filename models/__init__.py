"""
Models package for LLM-KT (Knowledge Tracing with LLM)

This package contains:
- encoders: Context Encoder and Sequence Encoder for embeddings
- embedding_model: Combined Question/Concept embedding model
- kt_llm: Knowledge Tracing LLM with embedding injection
"""

from .encoders import ContextEncoder, SequenceEncoder, HybridEncoder
from .embedding_model import (
    QuestionEmbedding, 
    ConceptEmbedding, 
    KTEmbeddingModel,
    EmbeddingCache
)
from .kt_llm import KnowledgeTracingLLM, PromptFormatter

__all__ = [
    # Encoders
    'ContextEncoder',
    'SequenceEncoder',
    'HybridEncoder',
    
    # Embedding modules
    'QuestionEmbedding',
    'ConceptEmbedding',
    'KTEmbeddingModel',
    'EmbeddingCache',
    
    # LLM
    'KnowledgeTracingLLM',
    'PromptFormatter'
]
