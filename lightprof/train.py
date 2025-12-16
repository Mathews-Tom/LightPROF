"""Training pipeline for LightPROF.

This module implements the training loop for the Knowledge Adapter
with a frozen LLM (LLaMA-3-8B-Instruct).

Reference: LightPROF paper Section 5 (Experiments)

Training:
    - Only Knowledge Adapter parameters are trained
    - LLM and BERT are frozen
    - Loss is computed on answer tokens only (causal LM loss)
    - Optimizer: AdamW with weight decay
    - Scheduler: Cosine annealing with warmup
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lightprof.adapter import KnowledgeAdapter
from lightprof.config import LightPROFConfig
from lightprof.data.freebase import FreebaseKG
from lightprof.reasoning import ReasoningModule
from lightprof.retrieval import RetrievalModule

if TYPE_CHECKING:
    from lightprof.data.cwq import CWQDataset
    from lightprof.data.webqsp import WebQSPDataset

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for LightPROF Knowledge Adapter.

    Handles the training loop, optimizer setup, and logging.

    Args:
        config: LightPROF configuration.
        adapter: Knowledge Adapter model.
        reasoning: Reasoning module with frozen LLM.
        train_dataset: Training dataset.
        val_dataset: Validation dataset (optional).

    Attributes:
        config: Configuration settings.
        adapter: Knowledge Adapter being trained.
        reasoning: Reasoning module for loss computation.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
    """

    def __init__(
        self,
        config: LightPROFConfig,
        adapter: KnowledgeAdapter,
        reasoning: ReasoningModule,
        train_dataset: WebQSPDataset | CWQDataset,
        val_dataset: WebQSPDataset | CWQDataset | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.reasoning = reasoning
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # Process one sample at a time due to variable paths
            shuffle=True,
            num_workers=0,  # Avoid pickling issues with complex objects
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )

        # Optimizer (only train adapter parameters)
        self.optimizer = AdamW(
            self.adapter.trainable_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        num_training_steps = (
            len(self.train_loader)
            * config.training.num_epochs
            // config.training.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

    @property
    def device(self) -> torch.device:
        """Get the training device."""
        return self.reasoning.device

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.adapter.train()
        total_loss = 0.0
        num_samples = 0

        accumulation_steps = self.config.training.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Extract batch data
            question = batch["question"]
            if isinstance(question, list):
                question = question[0]

            reasoning_paths = batch["reasoning_paths"]
            if isinstance(reasoning_paths, list) and len(reasoning_paths) == 1:
                reasoning_paths = reasoning_paths[0]

            answers = batch["answers"]
            if isinstance(answers, list) and len(answers) > 0:
                if isinstance(answers[0], list):
                    answers = answers[0]
                answer_text = ", ".join(answers) if answers else ""
            else:
                answer_text = ""

            if not answer_text:
                continue

            # Forward pass
            loss = self.reasoning.forward_for_training(
                question=question,
                reasoning_paths=reasoning_paths,
                answer=answer_text,
            )

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps
            num_samples += 1

            # Optimizer step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.adapter.trainable_parameters,
                    self.config.training.max_grad_norm,
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step}, Loss: {loss.item() * accumulation_steps:.4f}, LR: {lr:.2e}"
                    )

        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            nn.utils.clip_grad_norm_(
                self.adapter.trainable_parameters,
                self.config.training.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        return total_loss / max(num_samples, 1)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation set.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.val_loader is None:
            return {}

        self.adapter.eval()
        total_loss = 0.0
        num_samples = 0

        for batch in self.val_loader:
            question = batch["question"]
            if isinstance(question, list):
                question = question[0]

            reasoning_paths = batch["reasoning_paths"]
            if isinstance(reasoning_paths, list) and len(reasoning_paths) == 1:
                reasoning_paths = reasoning_paths[0]

            answers = batch["answers"]
            if isinstance(answers, list) and len(answers) > 0:
                if isinstance(answers[0], list):
                    answers = answers[0]
                answer_text = ", ".join(answers) if answers else ""
            else:
                answer_text = ""

            if not answer_text:
                continue

            loss = self.reasoning.forward_for_training(
                question=question,
                reasoning_paths=reasoning_paths,
                answer=answer_text,
            )

            total_loss += loss.item()
            num_samples += 1

        return {
            "val_loss": total_loss / max(num_samples, 1),
        }

    def train(self) -> KnowledgeAdapter:
        """Run full training loop.

        Returns:
            Trained Knowledge Adapter.
        """
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.training.num_epochs}")
        logger.info(f"  Train samples: {len(self.train_dataset)}")
        logger.info(f"  Trainable params: {self.adapter.num_trainable_parameters:,}")

        best_val_loss = float("inf")

        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")

            # Training
            train_loss = self.train_epoch()
            logger.info(f"  Train Loss: {train_loss:.4f}")

            # Validation
            if self.val_loader is not None:
                metrics = self.evaluate()
                val_loss = metrics.get("val_loss", float("inf"))
                logger.info(f"  Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info("  New best validation loss!")

        logger.info("\nTraining complete!")
        return self.adapter

    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "adapter_state_dict": self.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")


def train(
    config: LightPROFConfig,
    train_dataset: WebQSPDataset | CWQDataset,
    val_dataset: WebQSPDataset | CWQDataset | None = None,
    checkpoint_dir: str | Path | None = None,
) -> KnowledgeAdapter:
    """Train Knowledge Adapter with frozen LLM.

    Convenience function for training.

    Args:
        config: LightPROF configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset (optional).
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Trained Knowledge Adapter.
    """
    # Set random seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Initialize adapter
    adapter = KnowledgeAdapter(config.model, config.adapter)

    # Initialize reasoning module (includes frozen LLM)
    reasoning = ReasoningModule(
        model_config=config.model,
        knowledge_adapter=adapter,
        device="auto",
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        adapter=adapter,
        reasoning=reasoning,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Train
    trained_adapter = trainer.train()

    # Save final checkpoint
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / "final_adapter.pt"
        trainer.save_checkpoint(checkpoint_path)

    return trained_adapter


def train_from_config(
    config_path: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> KnowledgeAdapter:
    """Train from configuration file.

    Args:
        config_path: Path to TOML configuration file.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Trained Knowledge Adapter.
    """
    # Load configuration
    config = LightPROFConfig.from_toml(config_path)

    # Load knowledge graph
    kg = FreebaseKG(config.data.kg_path)

    # Create retrieval module for datasets
    retrieval = RetrievalModule(kg=kg, config=config.retrieval)

    # Import dataset classes
    from lightprof.data.webqsp import WebQSPDataset
    from lightprof.data.cwq import CWQDataset

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets based on config
    if config.data.dataset_name == "webqsp":
        train_dataset = WebQSPDataset(
            config.data.train_path,
            kg=kg,
            tokenizer=tokenizer,
            max_hops=config.retrieval.max_hops,
        )
        val_dataset = None
        if Path(config.data.val_path).exists():
            val_dataset = WebQSPDataset(
                config.data.val_path,
                kg=kg,
                tokenizer=tokenizer,
                max_hops=config.retrieval.max_hops,
            )
    elif config.data.dataset_name == "cwq":
        train_dataset = CWQDataset(
            config.data.train_path,
            kg=kg,
            tokenizer=tokenizer,
            max_hops=config.retrieval.max_hops,
        )
        val_dataset = None
        if Path(config.data.val_path).exists():
            val_dataset = CWQDataset(
                config.data.val_path,
                kg=kg,
                tokenizer=tokenizer,
                max_hops=config.retrieval.max_hops,
            )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")

    return train(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
    )


# CLI entry point
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train LightPROF Knowledge Adapter")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    trained_adapter = train_from_config(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\nTraining complete!")
    print(f"Adapter: {trained_adapter}")
