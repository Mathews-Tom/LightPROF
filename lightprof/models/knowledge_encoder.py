"""Knowledge Encoder module for LightPROF.

This module implements the Knowledge Encoder components for embedding
knowledge graph triples, including:
- StructuralEmbedding: H+R-T positional encoding (Equation 4 in paper)
- KnowledgeEncoder: Transformer-based encoder for fusing embeddings

Reference: LightPROF paper Section 4.2 (Knowledge Embedding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from lightprof.config import AdapterConfig


class StructuralEmbedding(nn.Module):
    """Structural embedding using H+R-T encoding.

    Per paper Equation 4: "We use the MLP layers to convert the head,
    relation, and tail embedding vector E_h, E_r, E_t into structural
    embeddings using the formula E_h + E_r - E_t."

    This captures the structural position of entities and relations
    within knowledge graph triples.

    Args:
        hidden_dim: Dimension of structural embeddings.

    Attributes:
        position_embedding: Embedding layer for head/relation/tail positions.
        mlp: MLP to transform structural information.
    """

    # Position indices
    HEAD_POS = 0
    REL_POS = 1
    TAIL_POS = 2

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Position embeddings for head, relation, tail
        self.position_embedding = nn.Embedding(3, hidden_dim)

        # MLP to transform structural information
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute structural embedding for a batch.

        The structural embedding encodes the H+R-T relationship pattern.

        Args:
            batch_size: Number of items in batch.
            device: Device to place tensors on.

        Returns:
            Structural embeddings [batch_size, hidden_dim].
        """
        # Get position embeddings
        head_pos = self.position_embedding(
            torch.tensor([self.HEAD_POS], device=device)
        )
        rel_pos = self.position_embedding(
            torch.tensor([self.REL_POS], device=device)
        )
        tail_pos = self.position_embedding(
            torch.tensor([self.TAIL_POS], device=device)
        )

        # H + R - T encoding (per paper Eq. 4)
        structural_info = head_pos + rel_pos - tail_pos

        # Transform through MLP
        structural_emb = self.mlp(structural_info)

        # Expand to batch size
        return structural_emb.expand(batch_size, -1)

    def forward_per_position(
        self,
        position: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get structural embedding for a specific position.

        Useful when processing head, relation, tail separately.

        Args:
            position: Position index (0=head, 1=rel, 2=tail).
            batch_size: Number of items in batch.
            device: Device to place tensors on.

        Returns:
            Position-specific embedding [batch_size, hidden_dim].
        """
        pos_tensor = torch.tensor([position], device=device)
        pos_emb = self.position_embedding(pos_tensor)
        pos_emb = self.mlp(pos_emb)
        return pos_emb.expand(batch_size, -1)


class KnowledgeEncoder(nn.Module):
    """Transformer-based encoder for knowledge embedding fusion.

    Per paper Section 4.2: The Knowledge Encoder uses Transformer layers
    to encode and fuse the concatenated text and structural embeddings.

    Args:
        input_dim: Input dimension (text_dim + structural_dim).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        dim_feedforward: Dimension of feedforward network.

    Attributes:
        encoder: Transformer encoder module.
        layer_norm: Final layer normalization.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Default feedforward dim is 4x input (standard Transformer)
        if dim_feedforward is None:
            dim_feedforward = input_dim * 4

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )

        # Stack multiple layers
        # Note: enable_nested_tensor=False required when norm_first=True
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

    @classmethod
    def from_config(
        cls,
        bert_embedding_dim: int,
        adapter_config: AdapterConfig,
    ) -> KnowledgeEncoder:
        """Create KnowledgeEncoder from configuration.

        Args:
            bert_embedding_dim: BERT embedding dimension.
            adapter_config: Adapter configuration.

        Returns:
            Initialized KnowledgeEncoder instance.
        """
        input_dim = bert_embedding_dim + adapter_config.structural_mlp_hidden_dim
        return cls(
            input_dim=input_dim,
            num_layers=adapter_config.knowledge_encoder_layers,
            num_heads=adapter_config.knowledge_encoder_heads,
            dropout=adapter_config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Transformer encoder.

        Args:
            x: Input embeddings [batch_size, seq_len, input_dim].
            src_key_padding_mask: Mask for padded positions [batch_size, seq_len].

        Returns:
            Encoded embeddings [batch_size, seq_len, input_dim].
        """
        # Transformer encoding
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Final layer norm
        return self.layer_norm(encoded)

    def forward_pooled(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Forward pass with pooling over sequence dimension.

        Args:
            x: Input embeddings [batch_size, seq_len, input_dim].
            src_key_padding_mask: Mask for padded positions.
            pooling: Pooling strategy ('mean', 'max', 'first').

        Returns:
            Pooled embeddings [batch_size, input_dim].
        """
        # Get encoded sequence
        encoded = self.forward(x, src_key_padding_mask)

        # Apply pooling
        if pooling == "mean":
            if src_key_padding_mask is not None:
                # Masked mean pooling
                mask = ~src_key_padding_mask.unsqueeze(-1)
                encoded = encoded * mask
                return encoded.sum(dim=1) / mask.sum(dim=1)
            return encoded.mean(dim=1)

        if pooling == "max":
            if src_key_padding_mask is not None:
                # Masked max pooling
                mask = src_key_padding_mask.unsqueeze(-1)
                encoded = encoded.masked_fill(mask, float("-inf"))
            return encoded.max(dim=1).values

        if pooling == "first":
            return encoded[:, 0, :]

        raise ValueError(f"Unknown pooling strategy: {pooling}")

    def __repr__(self) -> str:
        return (
            f"KnowledgeEncoder(\n"
            f"  input_dim={self.input_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads}\n"
            f")"
        )
