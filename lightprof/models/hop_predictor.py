"""Hop Predictor module for LightPROF.

This module implements a BERT-based classifier for predicting the number
of reasoning hops required to answer a knowledge graph question.

Reference: LightPROF paper Section 4.1 (Semantic Extraction)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from lightprof.config import ModelConfig


class HopPredictor(nn.Module):
    """BERT-based classifier for predicting reasoning hop count.

    Per paper Section 4.1: "We initially employ a trained encoder to capture
    the semantic information of the questions...the encoder classifies
    questions into different hop categories."

    The hop count determines the maximum depth of BFS traversal in the
    knowledge graph during relation retrieval.

    Args:
        bert_model_name: Name of the BERT model to use.
        num_classes: Number of hop classes (default 4 for hops 1-4).
        dropout: Dropout probability for classification head.
        freeze_bert: Whether to freeze BERT parameters during training.

    Attributes:
        bert: Pretrained BERT encoder.
        tokenizer: Associated tokenizer.
        classifier: Linear classification head.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.bert_model_name = bert_model_name

        # Load BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # Optionally freeze BERT
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    @classmethod
    def from_config(cls, config: ModelConfig) -> HopPredictor:
        """Create HopPredictor from configuration.

        Args:
            config: Model configuration.

        Returns:
            Initialized HopPredictor instance.
        """
        return cls(
            bert_model_name=config.bert_model_name,
            num_classes=4,  # Fixed for 1-4 hops
            dropout=0.1,
            freeze_bert=False,  # Train the hop predictor
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for hop prediction.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            Logits for hop classes [batch_size, num_classes].
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def predict(self, question: str) -> int:
        """Predict the number of hops for a question.

        Args:
            question: Natural language question.

        Returns:
            Predicted hop count (1-4).
        """
        # Tokenize
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )

        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            logits = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
            )

        # Get predicted class (0-indexed) and convert to hop count (1-indexed)
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class + 1

    def predict_batch(self, questions: list[str]) -> list[int]:
        """Predict hop counts for a batch of questions.

        Args:
            questions: List of natural language questions.

        Returns:
            List of predicted hop counts (1-4).
        """
        # Tokenize batch
        inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )

        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            logits = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
            )

        # Convert to hop counts
        predicted_classes = logits.argmax(dim=-1).tolist()
        return [c + 1 for c in predicted_classes]

    def predict_with_confidence(
        self, question: str
    ) -> tuple[int, torch.Tensor]:
        """Predict hop count with class probabilities.

        Args:
            question: Natural language question.

        Returns:
            Tuple of (predicted_hop, class_probabilities).
        """
        # Tokenize
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )

        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            logits = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            probs = F.softmax(logits, dim=-1)

        # Get predicted class
        predicted_class = logits.argmax(dim=-1).item()

        return predicted_class + 1, probs.squeeze(0)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for training.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Ground truth hop counts (1-indexed) [batch_size].

        Returns:
            Cross-entropy loss.
        """
        logits = self.forward(input_ids, attention_mask)

        # Convert 1-indexed labels to 0-indexed for cross-entropy
        labels_0indexed = labels - 1

        return F.cross_entropy(logits, labels_0indexed)

    def __repr__(self) -> str:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"HopPredictor(\n"
            f"  bert_model={self.bert_model_name},\n"
            f"  num_classes={self.num_classes},\n"
            f"  trainable_params={trainable_params:,},\n"
            f"  total_params={total_params:,}\n"
            f")"
        )
