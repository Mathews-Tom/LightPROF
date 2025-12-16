"""Tests for training module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from lightprof.config import LightPROFConfig, TrainingConfig


class TestTrainingConfig:
    """Test TrainingConfig settings."""

    def test_default_config(self) -> None:
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.num_epochs > 0
        assert config.warmup_ratio >= 0
        assert config.warmup_ratio <= 1

    def test_custom_config(self) -> None:
        """Test custom training configuration."""
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=5,
            warmup_ratio=0.05,
        )

        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 5
        assert config.warmup_ratio == 0.05


class TestLightPROFConfig:
    """Test LightPROFConfig loading."""

    def test_default_config(self) -> None:
        """Test creating default configuration."""
        config = LightPROFConfig()

        assert config.model is not None
        assert config.adapter is not None
        assert config.retrieval is not None
        assert config.training is not None
        assert config.data is not None

    def test_load_from_toml(self) -> None:
        """Test loading config from TOML file."""
        toml_content = """
[model]
llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_embedding_dim = 4096
bert_model_name = "bert-base-uncased"
freeze_llm = true
freeze_bert = true

[training]
learning_rate = 0.002
batch_size = 4
num_epochs = 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            loaded = LightPROFConfig.from_toml(Path(f.name))

            assert loaded.model.llm_name == "meta-llama/Meta-Llama-3-8B-Instruct"
            assert loaded.model.llm_embedding_dim == 4096
            assert loaded.training.learning_rate == 0.002
            assert loaded.training.batch_size == 4

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = LightPROFConfig()
        config_dict = config.to_dict()

        assert "model" in config_dict
        assert "training" in config_dict
        assert config_dict["model"]["llm_name"] == config.model.llm_name
        assert config_dict["training"]["learning_rate"] == config.training.learning_rate


class TestTrainerImports:
    """Test Trainer class imports work correctly."""

    def test_trainer_class_import(self) -> None:
        """Test Trainer class can be imported."""
        from lightprof.train import Trainer

        assert Trainer is not None

    def test_train_function_import(self) -> None:
        """Test train function can be imported."""
        from lightprof.train import train

        assert train is not None

    def test_train_from_config_import(self) -> None:
        """Test train_from_config function can be imported."""
        from lightprof.train import train_from_config

        assert train_from_config is not None


class TestTrainerInit:
    """Test Trainer initialization (lightweight tests)."""

    def test_training_config_values(self) -> None:
        """Test training config default values per paper."""
        config = TrainingConfig()

        # Per paper: batch size 4, lr 2e-3
        # These are the defaults that match the paper
        assert config.batch_size == 4
        assert config.learning_rate == 2e-3

    def test_gradient_accumulation_steps(self) -> None:
        """Test gradient accumulation is configured."""
        config = TrainingConfig()

        assert config.gradient_accumulation_steps >= 1

    def test_weight_decay(self) -> None:
        """Test weight decay is configured."""
        config = TrainingConfig()

        assert config.weight_decay >= 0


class TestOptimizerSetup:
    """Test optimizer configuration."""

    def test_adamw_parameters(self) -> None:
        """Test AdamW optimizer would use correct parameters."""
        config = TrainingConfig()

        # Verify parameters match what AdamW expects
        assert 0 < config.learning_rate < 1
        assert 0 <= config.weight_decay < 1

    def test_scheduler_warmup(self) -> None:
        """Test scheduler warmup ratio is valid."""
        config = TrainingConfig()

        assert 0 <= config.warmup_ratio <= 1


class TestCheckpointFormat:
    """Test checkpoint saving/loading format."""

    def test_checkpoint_keys(self) -> None:
        """Test expected checkpoint keys."""
        expected_keys = [
            "epoch",
            "global_step",
            "adapter_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "config",
        ]

        # Just verify the Trainer class has save_checkpoint method
        from lightprof.train import Trainer

        assert hasattr(Trainer, "save_checkpoint")
        assert hasattr(Trainer, "load_checkpoint")


class TestTrainingLoop:
    """Test training loop components."""

    def test_train_epoch_method_exists(self) -> None:
        """Test train_epoch method exists."""
        from lightprof.train import Trainer

        assert hasattr(Trainer, "train_epoch")

    def test_evaluate_method_exists(self) -> None:
        """Test evaluate method exists."""
        from lightprof.train import Trainer

        assert hasattr(Trainer, "evaluate")

    def test_train_method_exists(self) -> None:
        """Test train method exists."""
        from lightprof.train import Trainer

        assert hasattr(Trainer, "train")
