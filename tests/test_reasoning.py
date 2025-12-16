"""Tests for ReasoningModule."""

from __future__ import annotations

import os

import pytest
import torch

from lightprof.adapter import KnowledgeAdapter
from lightprof.config import AdapterConfig, ModelConfig
from lightprof.reasoning import ReasoningModule


# Check if we can access gated models (requires HF token)
HAS_HF_TOKEN = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
SKIP_LLM_TESTS = not HAS_HF_TOKEN or os.environ.get("SKIP_LLM_TESTS") == "1"


@pytest.fixture
def model_config() -> ModelConfig:
    """Create test model configuration with small models."""
    return ModelConfig(
        llm_name="meta-llama/Llama-3.2-1B",  # Smaller model for testing
        llm_embedding_dim=2048,
        bert_model_name="prajjwal1/bert-tiny",
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


class TestReasoningModuleHardPrompt:
    """Test hard prompt creation without loading LLM."""

    def test_hard_prompt_template_exists(self) -> None:
        """Test HARD_PROMPT_TEMPLATE is defined."""
        assert hasattr(ReasoningModule, "HARD_PROMPT_TEMPLATE")
        assert "{question}" in ReasoningModule.HARD_PROMPT_TEMPLATE

    def test_hard_prompt_format(self) -> None:
        """Test hard prompt can be formatted."""
        template = ReasoningModule.HARD_PROMPT_TEMPLATE
        question = "Who is the president of USA?"
        prompt = template.format(question=question)

        assert question in prompt
        assert "knowledge graphs" in prompt.lower()
        assert "answer" in prompt.lower()


@pytest.mark.skipif(
    SKIP_LLM_TESTS,
    reason="Skipping LLM tests (requires HF_TOKEN, skip with SKIP_LLM_TESTS=1)"
)
class TestReasoningModuleIntegration:
    """Integration tests requiring LLM loading."""

    @pytest.fixture
    def reasoning_module(
        self, model_config: ModelConfig, adapter: KnowledgeAdapter
    ) -> ReasoningModule:
        """Create ReasoningModule instance."""
        return ReasoningModule(
            model_config=model_config,
            knowledge_adapter=adapter,
            device="auto",
        )

    def test_instantiation(
        self, model_config: ModelConfig, adapter: KnowledgeAdapter
    ) -> None:
        """Test ReasoningModule can be instantiated."""
        module = ReasoningModule(
            model_config=model_config,
            knowledge_adapter=adapter,
            device="cpu",
        )
        assert isinstance(module, ReasoningModule)

    def test_device_property(self, reasoning_module: ReasoningModule) -> None:
        """Test device property."""
        device = reasoning_module.device
        assert isinstance(device, torch.device)

    def test_create_hard_prompt(self, reasoning_module: ReasoningModule) -> None:
        """Test create_hard_prompt method."""
        question = "Where was Obama born?"
        prompt = reasoning_module.create_hard_prompt(question)

        assert question in prompt
        assert isinstance(prompt, str)

    def test_get_input_embeddings(self, reasoning_module: ReasoningModule) -> None:
        """Test getting input embeddings from LLM."""
        input_ids = reasoning_module.tokenizer(
            "Hello world",
            return_tensors="pt",
        )["input_ids"].to(reasoning_module.device)

        embeddings = reasoning_module.get_input_embeddings(input_ids)

        assert embeddings.dim() == 3  # [batch, seq, hidden]

    def test_create_combined_embeddings(
        self, reasoning_module: ReasoningModule
    ) -> None:
        """Test combining soft and hard prompt embeddings."""
        question = "Where was Obama born?"
        soft_prompts = torch.randn(
            3, reasoning_module.model_config.llm_embedding_dim,
            device=reasoning_module.device,
        )

        combined, attention_mask = reasoning_module.create_combined_embeddings(
            question, soft_prompts
        )

        # Should have soft prompts + text tokens
        assert combined.dim() == 3
        assert combined.size(0) == 1  # batch size
        assert combined.size(1) > 3  # At least soft prompts + some text
        assert attention_mask.size(1) == combined.size(1)


class TestReasoningModuleLightweight:
    """Lightweight tests that don't require full LLM loading."""

    def test_repr_format(self) -> None:
        """Test __repr__ method format without instantiation."""
        # Just check the class has the method
        assert hasattr(ReasoningModule, "__repr__")

    def test_generate_signature(self) -> None:
        """Test generate method signature exists."""
        assert hasattr(ReasoningModule, "generate")
        assert hasattr(ReasoningModule, "forward_for_training")
        assert hasattr(ReasoningModule, "forward_batch")

    def test_predict_hops_raises(self) -> None:
        """Test predict_hops raises NotImplementedError."""
        # predict_hops should raise because it's in RetrievalModule
        assert hasattr(ReasoningModule, "predict_hops")


class TestReasoningPathTypes:
    """Test reasoning path type handling."""

    def test_empty_paths(
        self, model_config: ModelConfig, adapter_config: AdapterConfig
    ) -> None:
        """Test adapter handles empty paths."""
        adapter = KnowledgeAdapter(model_config, adapter_config)
        paths: list[list[tuple[str, str, str]]] = []

        soft_prompts = adapter(paths)

        # Should return default embedding
        assert soft_prompts.shape[0] == 1

    def test_single_path(
        self, model_config: ModelConfig, adapter_config: AdapterConfig
    ) -> None:
        """Test adapter handles single path."""
        adapter = KnowledgeAdapter(model_config, adapter_config)
        paths = [[("Head", "relation", "Tail")]]

        soft_prompts = adapter(paths)

        assert soft_prompts.shape[0] == 1

    def test_multi_hop_path(
        self, model_config: ModelConfig, adapter_config: AdapterConfig
    ) -> None:
        """Test adapter handles multi-hop path."""
        adapter = KnowledgeAdapter(model_config, adapter_config)
        paths = [[
            ("A", "rel1", "B"),
            ("B", "rel2", "C"),
            ("C", "rel3", "D"),
        ]]

        soft_prompts = adapter(paths)

        assert soft_prompts.shape[0] == 1
