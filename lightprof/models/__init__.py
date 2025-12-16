"""Model components for LightPROF.

This package contains the neural network components for the LightPROF
Knowledge Graph Question Answering system.

Modules:
    hop_predictor: BERT-based classifier for predicting reasoning hop count.
    knowledge_encoder: Transformer encoder for knowledge embedding fusion.
    projector: MLP projector to LLM embedding space.
"""

from __future__ import annotations

from lightprof.models.hop_predictor import HopPredictor
from lightprof.models.knowledge_encoder import KnowledgeEncoder, StructuralEmbedding
from lightprof.models.projector import Projector

__all__ = [
    "HopPredictor",
    "KnowledgeEncoder",
    "Projector",
    "StructuralEmbedding",
]
