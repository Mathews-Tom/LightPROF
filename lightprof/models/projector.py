"""Projector module for LightPROF.

This module implements the 2-layer MLP projector that maps knowledge
embeddings to the LLM embedding space.

Reference: LightPROF paper Section 4.2 (Knowledge Embedding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from lightprof.config import AdapterConfig, ModelConfig


class Projector(nn.Module):
    """2-layer MLP projector to LLM embedding space.

    Per paper Section 4.2: The projector maps the fused knowledge
    embeddings to the LLM's embedding space (4096 dimensions for LLaMA-3).

    Architecture:
        Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, output_dim)

    Args:
        input_dim: Input dimension (from KnowledgeEncoder).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (LLM embedding dim, e.g., 4096).
        dropout: Dropout probability.

    Attributes:
        mlp: Sequential MLP layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @classmethod
    def from_config(
        cls,
        bert_embedding_dim: int,
        model_config: ModelConfig,
        adapter_config: AdapterConfig,
    ) -> Projector:
        """Create Projector from configuration.

        Args:
            bert_embedding_dim: BERT embedding dimension.
            model_config: Model configuration.
            adapter_config: Adapter configuration.

        Returns:
            Initialized Projector instance.
        """
        input_dim = bert_embedding_dim + adapter_config.structural_mlp_hidden_dim
        return cls(
            input_dim=input_dim,
            hidden_dim=adapter_config.projector_hidden_dim,
            output_dim=model_config.llm_embedding_dim,
            dropout=adapter_config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings to LLM space.

        Args:
            x: Knowledge embeddings [batch_size, seq_len, input_dim]
               or [batch_size, input_dim].

        Returns:
            Projected embeddings with same batch dimensions but output_dim.
        """
        return self.mlp(x)

    def __repr__(self) -> str:
        return (
            f"Projector(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  output_dim={self.output_dim}\n"
            f")"
        )


class MultiHeadProjector(nn.Module):
    """Multi-head projector for different knowledge types.

    Optional variant that uses separate projection heads for different
    components (head entities, relations, tail entities).

    Args:
        input_dim: Input dimension per head.
        hidden_dim: Hidden layer dimension per head.
        output_dim: Output dimension (LLM embedding dim).
        num_heads: Number of projection heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Separate projector for each head
        self.projectors = nn.ModuleList([
            Projector(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_heads)
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * num_heads, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Project and fuse multiple input heads.

        Args:
            inputs: List of tensors, one per head [batch_size, input_dim].

        Returns:
            Fused projection [batch_size, output_dim].
        """
        if len(inputs) != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} inputs, got {len(inputs)}"
            )

        # Project each head
        projected = [
            proj(x) for proj, x in zip(self.projectors, inputs, strict=True)
        ]

        # Concatenate and fuse
        concatenated = torch.cat(projected, dim=-1)
        return self.fusion(concatenated)

    def forward_single(
        self,
        x: torch.Tensor,
        head_idx: int,
    ) -> torch.Tensor:
        """Project through a single head.

        Args:
            x: Input tensor [batch_size, input_dim].
            head_idx: Index of the projection head to use.

        Returns:
            Projected tensor [batch_size, output_dim].
        """
        return self.projectors[head_idx](x)
