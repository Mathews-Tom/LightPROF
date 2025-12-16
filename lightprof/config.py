"""Configuration management for LightPROF.

This module provides dataclass-based configuration for all LightPROF components,
aligned with the paper arXiv:2504.03137.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration for LLM and text encoder.

    Attributes:
        llm_name: HuggingFace model name for the LLM (frozen during training).
        llm_embedding_dim: Hidden dimension of the LLM (4096 for LLaMA-3-8B).
        bert_model_name: BERT model for text encoding in Knowledge Adapter.
        bert_embedding_dim: Hidden dimension of BERT (768 for bert-base-uncased).
        freeze_llm: Whether to freeze LLM parameters (always True per paper).
        freeze_bert: Whether to freeze BERT parameters (always True per paper).
    """

    llm_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_embedding_dim: int = 4096
    bert_model_name: str = "bert-base-uncased"
    bert_embedding_dim: int = 768
    freeze_llm: bool = True
    freeze_bert: bool = True


@dataclass(frozen=True)
class AdapterConfig:
    """Knowledge Adapter configuration per paper Section 4.2.

    Attributes:
        structural_mlp_hidden_dim: Hidden dimension for structural embedding MLP.
        knowledge_encoder_layers: Number of Transformer layers in KnowledgeEncoder.
        knowledge_encoder_heads: Number of attention heads in KnowledgeEncoder.
        projector_hidden_dim: Hidden dimension in 2-layer MLP projector.
        dropout: Dropout rate for regularization.
    """

    structural_mlp_hidden_dim: int = 64
    knowledge_encoder_layers: int = 2
    knowledge_encoder_heads: int = 8
    projector_hidden_dim: int = 2048
    dropout: float = 0.1


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval module configuration per paper Section 4.1.

    Attributes:
        max_hops: Maximum reasoning hops (4 for CWQ, 2 for WebQSP).
        top_k_relations: Number of top relation links to keep after ranking.
        max_paths_per_question: Maximum reasoning paths to sample per question.
        use_hop_predictor: Whether to use BERT hop predictor or fixed max_hops.
    """

    max_hops: int = 4
    top_k_relations: int = 10
    max_paths_per_question: int = 20
    use_hop_predictor: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration per paper Section 5.3.

    Attributes:
        batch_size: Training batch size (4 per paper).
        learning_rate: Initial learning rate (2e-3 per paper).
        num_epochs: Number of training epochs (1 per paper).
        warmup_ratio: Ratio of warmup steps to total steps.
        weight_decay: Weight decay for AdamW optimizer.
        gradient_accumulation_steps: Steps for gradient accumulation.
        max_grad_norm: Maximum gradient norm for clipping.
        fp16: Whether to use mixed precision training.
        seed: Random seed for reproducibility.
    """

    batch_size: int = 4
    learning_rate: float = 2e-3
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    """Data paths and preprocessing configuration.

    Attributes:
        kg_path: Path to knowledge graph triples file.
        train_path: Path to training data.
        val_path: Path to validation data.
        test_path: Path to test data.
        dataset_name: Dataset name ('webqsp' or 'cwq').
    """

    kg_path: Path = field(default_factory=lambda: Path("data/freebase_triples.tsv"))
    train_path: Path = field(default_factory=lambda: Path("data/train.json"))
    val_path: Path = field(default_factory=lambda: Path("data/val.json"))
    test_path: Path = field(default_factory=lambda: Path("data/test.json"))
    dataset_name: Literal["webqsp", "cwq"] = "webqsp"


@dataclass(frozen=True)
class LightPROFConfig:
    """Main configuration container for LightPROF.

    Combines all sub-configurations into a single object.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_toml(cls, path: Path) -> LightPROFConfig:
        """Load configuration from TOML file.

        Args:
            path: Path to TOML configuration file.

        Returns:
            LightPROFConfig instance with loaded values.
        """
        with open(path, "rb") as f:
            config_dict = tomllib.load(f)

        model_config = ModelConfig(**config_dict.get("model", {}))
        adapter_config = AdapterConfig(**config_dict.get("adapter", {}))
        retrieval_config = RetrievalConfig(**config_dict.get("retrieval", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        data_dict = config_dict.get("data", {})
        if "kg_path" in data_dict:
            data_dict["kg_path"] = Path(data_dict["kg_path"])
        if "train_path" in data_dict:
            data_dict["train_path"] = Path(data_dict["train_path"])
        if "val_path" in data_dict:
            data_dict["val_path"] = Path(data_dict["val_path"])
        if "test_path" in data_dict:
            data_dict["test_path"] = Path(data_dict["test_path"])
        data_config = DataConfig(**data_dict)

        return cls(
            model=model_config,
            adapter=adapter_config,
            retrieval=retrieval_config,
            training=training_config,
            data=data_config,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        from dataclasses import asdict

        result = asdict(self)
        # Convert Path objects to strings
        for key in ["kg_path", "train_path", "val_path", "test_path"]:
            if key in result.get("data", {}):
                result["data"][key] = str(result["data"][key])
        return result
