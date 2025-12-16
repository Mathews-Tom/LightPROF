"""Training script for HopPredictor.

This script trains the BERT-based hop predictor classifier on labeled
question-hop pairs extracted from WebQSP and CWQ datasets.

Reference: LightPROF paper Section 4.1 (Semantic Extraction)

Usage:
    uv run python scripts/train_hop_predictor.py --data data/hop_labels.json --output models/hop_predictor.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from lightprof.models.hop_predictor import HopPredictor

logger = logging.getLogger(__name__)


class HopDataset(Dataset[dict[str, object]]):
    """Dataset for hop prediction training.

    Args:
        data: List of (question, hop_count) pairs.
        tokenizer: BERT tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        data: list[dict[str, object]],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        question = item["question"]
        # Hop count is 1-indexed in dataset, convert to 0-indexed for classification
        hop_count = int(item["hops"]) - 1  # 0-3 for 1-4 hops

        encoding = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(hop_count, dtype=torch.long),
        }


def create_hop_labels_from_datasets(
    webqsp_path: str | Path | None = None,
    cwq_path: str | Path | None = None,
    output_path: str | Path = "data/hop_labels.json",
) -> list[dict[str, object]]:
    """Create hop labels from WebQSP and CWQ datasets.

    WebQSP questions are typically 1-2 hops.
    CWQ questions are typically 2-4 hops (based on complexity).

    Args:
        webqsp_path: Path to WebQSP data.
        cwq_path: Path to CWQ data.
        output_path: Path to save hop labels.

    Returns:
        List of (question, hops) pairs.
    """
    data: list[dict[str, object]] = []

    if webqsp_path and Path(webqsp_path).exists():
        with open(webqsp_path) as f:
            webqsp_data = json.load(f)

        for item in webqsp_data:
            question = item.get("ProcessedQuestion") or item.get("RawQuestion", "")
            # WebQSP is typically 1-2 hops
            # Heuristic: short questions = 1 hop, longer = 2 hops
            word_count = len(question.split())
            hops = 1 if word_count <= 8 else 2
            data.append({"question": question, "hops": hops})

    if cwq_path and Path(cwq_path).exists():
        with open(cwq_path) as f:
            cwq_data = json.load(f)

        for item in cwq_data:
            question = item.get("question", "")
            # CWQ has complexity types that correlate with hops
            complexity = item.get("compositionality_type", "").lower()

            if complexity == "conjunction":
                hops = 2
            elif complexity == "composition":
                hops = 3
            elif complexity == "comparative":
                hops = 3
            elif complexity == "superlative":
                hops = 4
            else:
                hops = 2  # Default

            data.append({"question": question, "hops": hops})

    # Save labels
    if data:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} hop labels to {output_path}")

    return data


def train_hop_predictor(
    data_path: str | Path,
    output_path: str | Path,
    bert_model: str = "bert-base-uncased",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    val_split: float = 0.1,
    device: str = "auto",
) -> HopPredictor:
    """Train the hop predictor model.

    Args:
        data_path: Path to hop labels JSON.
        output_path: Path to save trained model.
        bert_model: BERT model name.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        warmup_ratio: Warmup ratio for scheduler.
        val_split: Validation split ratio.
        device: Device to train on.

    Returns:
        Trained HopPredictor model.
    """
    # Load data
    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples from {data_path}")

    # Split data
    val_size = int(len(data) * val_split)
    train_data = data[val_size:]
    val_data = data[:val_size]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    # Create datasets
    train_dataset = HopDataset(train_data, tokenizer)
    val_dataset = HopDataset(val_data, tokenizer)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    model = HopPredictor(bert_model_name=bert_model, freeze_bert=False)

    # Move to device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    logger.info(f"Training on {device}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            loss = model.compute_loss(input_ids, attention_mask, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                predictions = logits.argmax(dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(total, 1)

        logger.info(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "bert_model": bert_model,
                },
                output_path,
            )
            logger.info(f"Saved best model (acc={val_acc:.4f})")

    logger.info(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")
    return model


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train HopPredictor")
    parser.add_argument(
        "--data",
        type=str,
        default="data/hop_labels.json",
        help="Path to hop labels JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/hop_predictor.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--create-labels",
        action="store_true",
        help="Create hop labels from WebQSP/CWQ",
    )
    parser.add_argument(
        "--webqsp-path",
        type=str,
        default="data/webqsp/WebQSP.train.json",
        help="Path to WebQSP training data",
    )
    parser.add_argument(
        "--cwq-path",
        type=str,
        default="data/cwq/ComplexWebQuestions_train.json",
        help="Path to CWQ training data",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="BERT model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )

    args = parser.parse_args()

    # Create labels if requested
    if args.create_labels:
        create_hop_labels_from_datasets(
            webqsp_path=args.webqsp_path,
            cwq_path=args.cwq_path,
            output_path=args.data,
        )

    # Check if data exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.error("Use --create-labels to create hop labels from datasets")
        return

    # Train
    train_hop_predictor(
        data_path=args.data,
        output_path=args.output,
        bert_model=args.bert_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
