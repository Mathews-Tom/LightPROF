"""Tests for model components."""

from __future__ import annotations

import pytest
import torch

from lightprof.models.hop_predictor import HopPredictor
from lightprof.models.knowledge_encoder import KnowledgeEncoder, StructuralEmbedding
from lightprof.models.projector import MultiHeadProjector, Projector


class TestHopPredictor:
    """Tests for HopPredictor."""

    @pytest.fixture
    def hop_predictor(self) -> HopPredictor:
        """Create HopPredictor with tiny BERT."""
        return HopPredictor(
            bert_model_name="prajjwal1/bert-tiny",
            num_classes=4,
            freeze_bert=True,
        )

    def test_instantiation(self) -> None:
        """Test HopPredictor can be instantiated."""
        predictor = HopPredictor(
            bert_model_name="prajjwal1/bert-tiny",
            num_classes=4,
        )
        assert isinstance(predictor, HopPredictor)

    def test_forward_shape(self, hop_predictor: HopPredictor) -> None:
        """Test forward pass produces correct shape."""
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        logits = hop_predictor(input_ids, attention_mask)

        assert logits.shape == (2, 4)

    def test_predict(self, hop_predictor: HopPredictor) -> None:
        """Test predict method returns valid hop count."""
        question = "Where was Barack Obama born?"
        hops = hop_predictor.predict(question)

        assert 1 <= hops <= 4

    def test_compute_loss(self, hop_predictor: HopPredictor) -> None:
        """Test compute_loss returns scalar."""
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        labels = torch.tensor([1, 3])  # Hop counts (1-4, 1-indexed)

        loss = hop_predictor.compute_loss(input_ids, attention_mask, labels)

        assert loss.dim() == 0
        assert loss.item() > 0


class TestStructuralEmbedding:
    """Tests for StructuralEmbedding."""

    @pytest.fixture
    def struct_emb(self) -> StructuralEmbedding:
        """Create StructuralEmbedding instance."""
        return StructuralEmbedding(hidden_dim=64)

    def test_instantiation(self) -> None:
        """Test StructuralEmbedding can be instantiated."""
        emb = StructuralEmbedding(hidden_dim=128)
        assert isinstance(emb, StructuralEmbedding)

    def test_forward_shape(self, struct_emb: StructuralEmbedding) -> None:
        """Test forward produces correct shape."""
        device = torch.device("cpu")
        batch_size = 4

        output = struct_emb(batch_size, device)

        assert output.shape == (batch_size, 64)

    def test_forward_different_batch_sizes(
        self, struct_emb: StructuralEmbedding
    ) -> None:
        """Test forward works with different batch sizes."""
        device = torch.device("cpu")

        for batch_size in [1, 2, 8, 16]:
            output = struct_emb(batch_size, device)
            assert output.shape[0] == batch_size


class TestKnowledgeEncoder:
    """Tests for KnowledgeEncoder."""

    @pytest.fixture
    def encoder(self) -> KnowledgeEncoder:
        """Create KnowledgeEncoder instance."""
        return KnowledgeEncoder(
            input_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )

    def test_instantiation(self) -> None:
        """Test KnowledgeEncoder can be instantiated."""
        enc = KnowledgeEncoder(input_dim=256, num_layers=3, num_heads=8)
        assert isinstance(enc, KnowledgeEncoder)

    def test_forward_shape(self, encoder: KnowledgeEncoder) -> None:
        """Test forward produces correct shape."""
        x = torch.randn(2, 5, 128)  # batch=2, seq=5, hidden=128

        output = encoder(x)

        assert output.shape == x.shape

    def test_forward_with_mask(self, encoder: KnowledgeEncoder) -> None:
        """Test forward with padding mask."""
        x = torch.randn(2, 5, 128)
        mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, True, True, True],
        ])

        output = encoder(x, src_key_padding_mask=mask)

        assert output.shape == x.shape


class TestProjector:
    """Tests for Projector."""

    @pytest.fixture
    def projector(self) -> Projector:
        """Create Projector instance."""
        return Projector(
            input_dim=128,
            hidden_dim=256,
            output_dim=512,
            dropout=0.0,
        )

    def test_instantiation(self) -> None:
        """Test Projector can be instantiated."""
        proj = Projector(input_dim=64, hidden_dim=128, output_dim=256)
        assert isinstance(proj, Projector)

    def test_forward_2d(self, projector: Projector) -> None:
        """Test forward with 2D input."""
        x = torch.randn(4, 128)

        output = projector(x)

        assert output.shape == (4, 512)

    def test_forward_3d(self, projector: Projector) -> None:
        """Test forward with 3D input."""
        x = torch.randn(2, 5, 128)

        output = projector(x)

        assert output.shape == (2, 5, 512)

    def test_repr(self, projector: Projector) -> None:
        """Test __repr__ method."""
        repr_str = repr(projector)
        assert "Projector" in repr_str
        assert "input_dim" in repr_str
        assert "output_dim" in repr_str


class TestMultiHeadProjector:
    """Tests for MultiHeadProjector."""

    @pytest.fixture
    def multi_projector(self) -> MultiHeadProjector:
        """Create MultiHeadProjector instance."""
        return MultiHeadProjector(
            input_dim=64,
            hidden_dim=128,
            output_dim=256,
            num_heads=3,
            dropout=0.0,
        )

    def test_instantiation(self) -> None:
        """Test MultiHeadProjector can be instantiated."""
        proj = MultiHeadProjector(
            input_dim=32,
            hidden_dim=64,
            output_dim=128,
            num_heads=2,
        )
        assert isinstance(proj, MultiHeadProjector)

    def test_forward(self, multi_projector: MultiHeadProjector) -> None:
        """Test forward with multiple heads."""
        inputs = [
            torch.randn(4, 64),
            torch.randn(4, 64),
            torch.randn(4, 64),
        ]

        output = multi_projector(inputs)

        assert output.shape == (4, 256)

    def test_forward_wrong_num_heads(self, multi_projector: MultiHeadProjector) -> None:
        """Test forward raises error with wrong number of inputs."""
        inputs = [
            torch.randn(4, 64),
            torch.randn(4, 64),
        ]  # Only 2 inputs for 3-head projector

        with pytest.raises(ValueError, match="Expected 3 inputs"):
            multi_projector(inputs)

    def test_forward_single(self, multi_projector: MultiHeadProjector) -> None:
        """Test forward_single method."""
        x = torch.randn(4, 64)

        output = multi_projector.forward_single(x, head_idx=0)

        assert output.shape == (4, 256)
