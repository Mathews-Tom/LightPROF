"""Tests for KnowledgeAdapter module."""

from __future__ import annotations

import pytest
import torch

from lightprof.adapter import KnowledgeAdapter
from lightprof.config import AdapterConfig, ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    """Create test model configuration."""
    return ModelConfig(
        llm_name="meta-llama/Llama-3.2-1B",  # Smaller model for testing
        llm_embedding_dim=2048,
        bert_model_name="prajjwal1/bert-tiny",  # Tiny BERT for fast tests
        freeze_llm=True,
        freeze_bert=True,
    )


@pytest.fixture
def adapter_config() -> AdapterConfig:
    """Create test adapter configuration."""
    return AdapterConfig(
        structural_mlp_hidden_dim=64,
        knowledge_encoder_layers=1,
        knowledge_encoder_heads=2,
        projector_hidden_dim=128,
        dropout=0.0,
    )


@pytest.fixture
def adapter(model_config: ModelConfig, adapter_config: AdapterConfig) -> KnowledgeAdapter:
    """Create KnowledgeAdapter instance."""
    return KnowledgeAdapter(model_config, adapter_config)


class TestKnowledgeAdapterInstantiation:
    """Test KnowledgeAdapter initialization."""

    def test_knowledge_adapter_instantiation(
        self, model_config: ModelConfig, adapter_config: AdapterConfig
    ) -> None:
        """Test adapter can be instantiated with configs."""
        adapter = KnowledgeAdapter(model_config, adapter_config)
        assert isinstance(adapter, KnowledgeAdapter)

    def test_from_configs_classmethod(
        self, model_config: ModelConfig, adapter_config: AdapterConfig
    ) -> None:
        """Test from_configs classmethod."""
        adapter = KnowledgeAdapter.from_configs(model_config, adapter_config)
        assert isinstance(adapter, KnowledgeAdapter)

    def test_fusion_dim_calculation(self, adapter: KnowledgeAdapter) -> None:
        """Test fusion dimension is correctly calculated."""
        expected_fusion_dim = (
            adapter.bert_embedding_dim + adapter.adapter_config.structural_mlp_hidden_dim
        )
        assert adapter.fusion_dim == expected_fusion_dim


class TestKnowledgeAdapterProperties:
    """Test KnowledgeAdapter properties."""

    def test_trainable_parameters(self, adapter: KnowledgeAdapter) -> None:
        """Test trainable_parameters returns correct params."""
        trainable = adapter.trainable_parameters
        assert isinstance(trainable, list)
        assert len(trainable) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in trainable)

    def test_num_trainable_parameters(self, adapter: KnowledgeAdapter) -> None:
        """Test num_trainable_parameters returns correct count."""
        num_trainable = adapter.num_trainable_parameters
        assert num_trainable > 0
        assert num_trainable < adapter.num_total_parameters

    def test_num_total_parameters(self, adapter: KnowledgeAdapter) -> None:
        """Test num_total_parameters returns correct count."""
        num_total = adapter.num_total_parameters
        assert num_total > 0

    def test_device_property(self, adapter: KnowledgeAdapter) -> None:
        """Test device property returns valid device."""
        device = adapter.device
        assert isinstance(device, torch.device)


class TestKnowledgeAdapterEncoding:
    """Test KnowledgeAdapter encoding methods."""

    def test_encode_text(self, adapter: KnowledgeAdapter) -> None:
        """Test text encoding produces correct shape."""
        text = "Barack Obama"
        embedding = adapter.encode_text(text)

        assert embedding.shape == (1, adapter.bert_embedding_dim)

    def test_encode_triple(self, adapter: KnowledgeAdapter) -> None:
        """Test triple encoding produces correct shape."""
        head = "Barack Obama"
        relation = "born in"
        tail = "Hawaii"

        triple_emb = adapter.encode_triple(head, relation, tail)

        assert triple_emb.shape == (1, adapter.fusion_dim)

    def test_encode_path(self, adapter: KnowledgeAdapter) -> None:
        """Test path encoding produces correct shape."""
        path = [
            ("Barack Obama", "born in", "Hawaii"),
            ("Hawaii", "located in", "USA"),
        ]

        path_emb = adapter.encode_path(path)

        assert path_emb.shape == (1, adapter.fusion_dim)

    def test_encode_empty_path(self, adapter: KnowledgeAdapter) -> None:
        """Test empty path returns zeros."""
        path: list[tuple[str, str, str]] = []

        path_emb = adapter.encode_path(path)

        assert path_emb.shape == (1, adapter.fusion_dim)
        assert torch.allclose(path_emb, torch.zeros_like(path_emb))


class TestKnowledgeAdapterForward:
    """Test KnowledgeAdapter forward pass."""

    def test_forward_single_path(self, adapter: KnowledgeAdapter) -> None:
        """Test forward pass with single path."""
        subgraph = [
            [("Barack Obama", "president of", "United States")],
        ]

        soft_prompts = adapter(subgraph)

        assert soft_prompts.shape == (1, adapter.model_config.llm_embedding_dim)

    def test_forward_multiple_paths(self, adapter: KnowledgeAdapter) -> None:
        """Test forward pass with multiple paths."""
        subgraph = [
            [("Barack Obama", "born in", "Hawaii")],
            [("Barack Obama", "president of", "United States")],
            [("Barack Obama", "married to", "Michelle Obama")],
        ]

        soft_prompts = adapter(subgraph)

        assert soft_prompts.shape == (3, adapter.model_config.llm_embedding_dim)

    def test_forward_empty_subgraph(self, adapter: KnowledgeAdapter) -> None:
        """Test forward pass with empty subgraph."""
        subgraph: list[list[tuple[str, str, str]]] = []

        soft_prompts = adapter(subgraph)

        assert soft_prompts.shape == (1, adapter.model_config.llm_embedding_dim)

    def test_forward_batched(self, adapter: KnowledgeAdapter) -> None:
        """Test batched forward pass."""
        subgraphs = [
            [[("Entity1", "rel1", "Entity2")]],
            [[("Entity3", "rel2", "Entity4")], [("Entity5", "rel3", "Entity6")]],
        ]

        soft_prompts_list = adapter.forward_batched(subgraphs)

        assert len(soft_prompts_list) == 2
        assert soft_prompts_list[0].shape == (1, adapter.model_config.llm_embedding_dim)
        assert soft_prompts_list[1].shape == (2, adapter.model_config.llm_embedding_dim)


class TestKnowledgeAdapterGradients:
    """Test gradient flow through adapter."""

    def test_gradients_flow_through_trainable(self, adapter: KnowledgeAdapter) -> None:
        """Test gradients flow through trainable parameters."""
        subgraph = [[("Head", "relation", "Tail")]]

        soft_prompts = adapter(subgraph)
        loss = soft_prompts.sum()
        loss.backward()

        # Check trainable params have gradients
        for param in adapter.trainable_parameters:
            assert param.grad is not None

    def test_frozen_bert_no_gradients(self, adapter: KnowledgeAdapter) -> None:
        """Test BERT parameters don't get gradients when frozen."""
        subgraph = [[("Head", "relation", "Tail")]]

        soft_prompts = adapter(subgraph)
        loss = soft_prompts.sum()
        loss.backward()

        # Check BERT params have no gradients (frozen)
        for param in adapter.text_encoder.parameters():
            assert param.grad is None or not param.requires_grad
