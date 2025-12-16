"""Evaluation script for LightPROF.

This script evaluates trained Knowledge Adapter checkpoints on WebQSP and CWQ
datasets using the Hits@1 metric per the paper.

Reference: LightPROF paper Section 5 (Experiments)

Usage:
    uv run python scripts/evaluate.py --config configs/default.toml --checkpoint checkpoints/final_adapter.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from lightprof.adapter import KnowledgeAdapter
from lightprof.config import LightPROFConfig
from lightprof.data.freebase import FreebaseKG
from lightprof.reasoning import ReasoningModule
from lightprof.retrieval import RetrievalModule

if TYPE_CHECKING:
    from lightprof.data.cwq import CWQDataset
    from lightprof.data.webqsp import WebQSPDataset

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison.

    Args:
        answer: Raw answer string.

    Returns:
        Normalized lowercase answer with punctuation removed.
    """
    # Lowercase
    answer = answer.lower()
    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)
    # Remove extra whitespace
    answer = " ".join(answer.split())
    return answer


def compute_hits_at_1(
    predicted: str,
    ground_truth: list[str],
) -> float:
    """Compute Hits@1 for a single prediction.

    Args:
        predicted: Predicted answer string.
        ground_truth: List of acceptable answers.

    Returns:
        1.0 if predicted matches any ground truth, 0.0 otherwise.
    """
    pred_norm = normalize_answer(predicted)

    for gt in ground_truth:
        gt_norm = normalize_answer(gt)
        if pred_norm == gt_norm or gt_norm in pred_norm or pred_norm in gt_norm:
            return 1.0

    return 0.0


def evaluate_dataset(
    reasoning: ReasoningModule,
    retrieval: RetrievalModule,
    dataset: WebQSPDataset | CWQDataset,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Evaluate on a dataset.

    Args:
        reasoning: Reasoning module with loaded adapter.
        retrieval: Retrieval module for path extraction.
        dataset: Dataset to evaluate on.
        max_samples: Maximum samples to evaluate (for debugging).

    Returns:
        Dictionary with evaluation metrics.
    """
    total_hits = 0.0
    total_samples = 0
    results: list[dict[str, object]] = []

    num_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[idx]

        question = sample["question"]
        if isinstance(question, list):
            question = question[0]

        answers = sample["answers"]
        if isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], list):
                answers = answers[0]
        else:
            continue

        if not answers:
            continue

        # Get reasoning paths
        reasoning_paths = sample.get("reasoning_paths", [])
        if isinstance(reasoning_paths, list) and len(reasoning_paths) == 1:
            reasoning_paths = reasoning_paths[0]

        # If no paths from dataset, use retrieval module
        if not reasoning_paths:
            anchor_entities = sample.get("anchor_entities", [])
            reasoning_paths = retrieval.retrieve(question, anchor_entities)

        # Generate answer
        try:
            predicted = reasoning.generate(
                question=question,
                reasoning_paths=reasoning_paths,
                max_new_tokens=64,
            )
        except Exception as e:
            logger.warning(f"Error generating answer for sample {idx}: {e}")
            predicted = ""

        # Compute Hits@1
        hit = compute_hits_at_1(predicted, answers)
        total_hits += hit
        total_samples += 1

        results.append({
            "idx": idx,
            "question": question,
            "predicted": predicted,
            "ground_truth": answers,
            "hit": hit,
        })

        # Log progress every 100 samples
        if (idx + 1) % 100 == 0:
            current_hits = total_hits / total_samples
            logger.info(f"  Samples {idx + 1}/{num_samples}, Hits@1: {current_hits:.4f}")

    hits_at_1 = total_hits / max(total_samples, 1)

    return {
        "hits@1": hits_at_1,
        "total_samples": total_samples,
        "total_hits": total_hits,
        "results": results,
    }


def evaluate(
    config: LightPROFConfig,
    checkpoint_path: str | Path,
    dataset_name: str = "webqsp",
    split: str = "test",
    max_samples: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Run evaluation pipeline.

    Args:
        config: LightPROF configuration.
        checkpoint_path: Path to adapter checkpoint.
        dataset_name: Dataset to evaluate ('webqsp' or 'cwq').
        split: Data split to evaluate ('test' or 'val').
        max_samples: Maximum samples to evaluate.
        output_path: Path to save results JSON.

    Returns:
        Dictionary with evaluation metrics.
    """
    logger.info(f"Evaluating on {dataset_name} ({split} split)")
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load knowledge graph
    kg = FreebaseKG(config.data.kg_path)

    # Initialize adapter
    adapter = KnowledgeAdapter(config.model, config.adapter)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Initialize reasoning module
    reasoning = ReasoningModule(
        model_config=config.model,
        knowledge_adapter=adapter,
        device="auto",
    )

    # Initialize retrieval module
    retrieval = RetrievalModule(kg=kg, config=config.retrieval)

    # Load dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "webqsp":
        from lightprof.data.webqsp import WebQSPDataset

        data_path = config.data.test_path if split == "test" else config.data.val_path
        dataset = WebQSPDataset(
            data_path,
            kg=kg,
            tokenizer=tokenizer,
            max_hops=config.retrieval.max_hops,
        )
    elif dataset_name == "cwq":
        from lightprof.data.cwq import CWQDataset

        data_path = config.data.test_path if split == "test" else config.data.val_path
        dataset = CWQDataset(
            data_path,
            kg=kg,
            tokenizer=tokenizer,
            max_hops=config.retrieval.max_hops,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Loaded {len(dataset)} samples from {data_path}")

    # Evaluate
    metrics = evaluate_dataset(
        reasoning=reasoning,
        retrieval=retrieval,
        dataset=dataset,
        max_samples=max_samples,
    )

    logger.info(f"\nResults:")
    logger.info(f"  Hits@1: {metrics['hits@1']:.4f} ({metrics['total_hits']:.0f}/{metrics['total_samples']})")

    # Save results
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "dataset": dataset_name,
            "split": split,
            "checkpoint": str(checkpoint_path),
            "hits@1": metrics["hits@1"],
            "total_samples": metrics["total_samples"],
            "total_hits": metrics["total_hits"],
            "results": metrics["results"],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved results to {output_path}")

    return {
        "hits@1": metrics["hits@1"],
        "total_samples": metrics["total_samples"],
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate LightPROF")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to adapter checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="webqsp",
        choices=["webqsp", "cwq"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Data split to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for debugging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    # Load config
    config = LightPROFConfig.from_toml(args.config)

    # Run evaluation
    metrics = evaluate(
        config=config,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        output_path=args.output,
    )

    print(f"\n{'='*50}")
    print(f"Final Hits@1: {metrics['hits@1']:.4f}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
