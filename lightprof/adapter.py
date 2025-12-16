"""Knowledge Adapter for LightPROF.

This module implements the Knowledge Adapter that converts reasoning subgraphs
into compact soft prompt embeddings for the LLM.

Reference: LightPROF paper Section 4.2 (Knowledge Embedding)

Architecture:
    1. Text Encoder: Frozen BERT for entity/relation text embedding
    2. Structural Encoder: H+R-T positional encoding (MLP-based)
    3. Knowledge Encoder: Transformer layers for fusion
    4. Projector: 2-layer MLP to LLM embedding space (4096)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from lightprof.models.knowledge_encoder import KnowledgeEncoder, StructuralEmbedding
from lightprof.models.projector import Projector

if TYPE_CHECKING:
    from lightprof.config import AdapterConfig, ModelConfig

# Type alias for reasoning subgraphs
Subgraph = list[list[tuple[str, str, str]]]


class KnowledgeAdapter(nn.Module):
    """Knowledge Adapter per LightPROF paper Section 4.2.

    Converts reasoning subgraphs (lists of KG paths) into soft prompt
    embeddings that can be prepended to LLM input via inputs_embeds.

    Architecture:
        - BERT (frozen): Text encoding for entities and relations
        - StructuralEmbedding: H+R-T positional encoding
        - KnowledgeEncoder: Transformer-based fusion
        - Projector: 2-layer MLP to LLM embedding space

    Args:
        model_config: Model configuration with LLM and BERT settings.
        adapter_config: Adapter-specific configuration.

    Attributes:
        text_encoder: Frozen BERT model for text encoding.
        tokenizer: BERT tokenizer.
        structural_embedding: H+R-T structural encoder.
        knowledge_encoder: Transformer encoder for fusion.
        projector: MLP projector to LLM space.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        adapter_config: AdapterConfig,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.adapter_config = adapter_config

        # 1. Frozen BERT for text encoding
        self.text_encoder = AutoModel.from_pretrained(model_config.bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.bert_model_name)

        # Freeze BERT parameters
        if model_config.freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.bert_embedding_dim = self.text_encoder.config.hidden_size

        # 2. Structural embedding (H+R-T encoding)
        self.structural_embedding = StructuralEmbedding(
            hidden_dim=adapter_config.structural_mlp_hidden_dim
        )

        # Fusion dimension: BERT dim + structural dim
        self.fusion_dim = (
            self.bert_embedding_dim + adapter_config.structural_mlp_hidden_dim
        )

        # 3. Knowledge Encoder (Transformer layers)
        self.knowledge_encoder = KnowledgeEncoder(
            input_dim=self.fusion_dim,
            num_layers=adapter_config.knowledge_encoder_layers,
            num_heads=adapter_config.knowledge_encoder_heads,
            dropout=adapter_config.dropout,
        )

        # 4. Projector to LLM embedding space
        self.projector = Projector(
            input_dim=self.fusion_dim,
            hidden_dim=adapter_config.projector_hidden_dim,
            output_dim=model_config.llm_embedding_dim,
            dropout=adapter_config.dropout,
        )

    @classmethod
    def from_configs(
        cls,
        model_config: ModelConfig,
        adapter_config: AdapterConfig,
    ) -> KnowledgeAdapter:
        """Create KnowledgeAdapter from configuration objects.

        Args:
            model_config: Model configuration.
            adapter_config: Adapter configuration.

        Returns:
            Initialized KnowledgeAdapter instance.
        """
        return cls(model_config, adapter_config)

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using frozen BERT.

        Args:
            text: Text string to encode.

        Returns:
            [CLS] token embedding [1, bert_embedding_dim].
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )

        # Move to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)

        # Return [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :]

    def encode_triple(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> torch.Tensor:
        """Encode a single KG triple.

        Per paper: Fuses text and structural embeddings for each triple.

        Args:
            head: Head entity text.
            relation: Relation text.
            tail: Tail entity text.

        Returns:
            Fused triple embedding [1, fusion_dim].
        """
        # Get text embeddings
        head_emb = self.encode_text(head)
        rel_emb = self.encode_text(relation)
        tail_emb = self.encode_text(tail)

        # Mean pooling over triple elements
        text_emb = (head_emb + rel_emb + tail_emb) / 3

        # Get structural embedding
        struct_emb = self.structural_embedding(1, self.device)

        # Fuse text + structural
        fused = torch.cat([text_emb, struct_emb], dim=-1)

        return fused

    def encode_path(
        self,
        path: list[tuple[str, str, str]],
    ) -> torch.Tensor:
        """Encode a reasoning path (sequence of triples).

        Args:
            path: List of (head, relation, tail) triples.

        Returns:
            Path embedding [1, fusion_dim].
        """
        if not path:
            return torch.zeros(1, self.fusion_dim, device=self.device)

        triple_embeddings = []
        for head, relation, tail in path:
            triple_emb = self.encode_triple(head, relation, tail)
            triple_embeddings.append(triple_emb)

        # Stack triples: [num_triples, fusion_dim]
        triple_stack = torch.cat(triple_embeddings, dim=0)

        # Add batch dimension for Transformer: [1, num_triples, fusion_dim]
        triple_stack = triple_stack.unsqueeze(0)

        # Encode through Transformer
        encoded = self.knowledge_encoder(triple_stack)

        # Mean pool over triples
        path_emb = encoded.mean(dim=1)

        return path_emb

    def forward(
        self,
        reasoning_subgraph: list[list[tuple[str, str, str]]],
    ) -> torch.Tensor:
        """Process reasoning subgraph into soft prompt embeddings.

        Args:
            reasoning_subgraph: List of reasoning paths, where each path
                is a list of (head, relation, tail) triples.

        Returns:
            Soft prompt embeddings [num_paths, llm_embedding_dim].
        """
        if not reasoning_subgraph:
            return torch.zeros(
                1, self.model_config.llm_embedding_dim, device=self.device
            )

        path_embeddings = []

        for path in reasoning_subgraph:
            path_emb = self.encode_path(path)
            path_embeddings.append(path_emb)

        # Stack all path embeddings: [num_paths, fusion_dim]
        all_paths = torch.cat(path_embeddings, dim=0)

        # Project to LLM embedding space: [num_paths, llm_embedding_dim]
        soft_prompts = self.projector(all_paths)

        return soft_prompts

    def forward_batched(
        self,
        reasoning_subgraphs: list[list[list[tuple[str, str, str]]]],
    ) -> list[torch.Tensor]:
        """Process multiple reasoning subgraphs (batch processing).

        Args:
            reasoning_subgraphs: List of reasoning subgraphs.

        Returns:
            List of soft prompt tensors, one per subgraph.
        """
        return [self.forward(subgraph) for subgraph in reasoning_subgraphs]

    def get_num_soft_prompt_tokens(
        self,
        reasoning_subgraph: list[list[tuple[str, str, str]]],
    ) -> int:
        """Get the number of soft prompt tokens for a subgraph.

        This is equal to the number of reasoning paths.

        Args:
            reasoning_subgraph: List of reasoning paths.

        Returns:
            Number of soft prompt tokens.
        """
        return max(len(reasoning_subgraph), 1)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Get list of trainable parameters.

        BERT is frozen, so only structural embedding, knowledge encoder,
        and projector parameters are trainable.

        Returns:
            List of trainable parameters.
        """
        trainable = []

        # Structural embedding (trainable)
        trainable.extend(self.structural_embedding.parameters())

        # Knowledge encoder (trainable)
        trainable.extend(self.knowledge_encoder.parameters())

        # Projector (trainable)
        trainable.extend(self.projector.parameters())

        return list(trainable)

    @property
    def num_trainable_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters)

    @property
    def num_total_parameters(self) -> int:
        """Get total number of parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"KnowledgeAdapter(\n"
            f"  bert_model={self.model_config.bert_model_name},\n"
            f"  fusion_dim={self.fusion_dim},\n"
            f"  llm_embedding_dim={self.model_config.llm_embedding_dim},\n"
            f"  trainable_params={self.num_trainable_parameters:,},\n"
            f"  total_params={self.num_total_parameters:,}\n"
            f")"
        )


# Standalone usage for testing
if __name__ == "__main__":
    from lightprof.config import AdapterConfig, ModelConfig

    # Create configurations
    model_config = ModelConfig()
    adapter_config = AdapterConfig()

    # Initialize adapter
    adapter = KnowledgeAdapter(model_config, adapter_config)

    print(adapter)

    # Test with dummy subgraph
    dummy_subgraph = [
        [("Barack Obama", "born in", "Honolulu")],
        [("Barack Obama", "president of", "United States")],
    ]

    # Forward pass
    soft_prompts = adapter(dummy_subgraph)

    print(f"\nInput: {len(dummy_subgraph)} paths")
    print(f"Output shape: {soft_prompts.shape}")
    print(f"Expected: [{len(dummy_subgraph)}, {model_config.llm_embedding_dim}]")
