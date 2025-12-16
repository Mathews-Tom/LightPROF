"""ComplexWebQuestions (CWQ) dataset loader and utilities.

This module provides dataset loading for the ComplexWebQuestions dataset,
which contains 34,689 questions requiring up to 4-hop reasoning.

Reference:
- Paper: "The Web as a Knowledge-base for Answering Complex Questions"
- URL: https://www.tau-nlp.org/compwebq
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from lightprof.data.freebase import FreebaseKG


@dataclass
class CWQSample:
    """A single ComplexWebQuestions sample.

    Attributes:
        question_id: Unique identifier for the question.
        question: Original question text.
        machine_question: Machine-generated compositional question.
        composition_answer: Answer to the compositional question.
        webqsp_question: Original WebQSP question (if derived from one).
        webqsp_id: Original WebQSP ID (if applicable).
        answers: List of answer entities.
        anchor_entities: List of (entity_id, entity_name) tuples.
        sparql: SPARQL query (if available).
        complexity: Question complexity type (e.g., "composition", "conjunction").
    """

    question_id: str
    question: str
    machine_question: str
    composition_answer: str
    webqsp_question: str | None
    webqsp_id: str | None
    answers: list[dict[str, str]]
    anchor_entities: list[tuple[str, str]]
    sparql: str | None = None
    complexity: str | None = None


class CWQDataset(Dataset[dict[str, object]]):
    """ComplexWebQuestions dataset for KGQA.

    Loads CWQ data and provides samples with reasoning paths retrieved
    from the knowledge graph.

    Args:
        data_path: Path to the JSON data file.
        kg: FreebaseKG instance for path retrieval.
        tokenizer: Tokenizer for encoding answers.
        max_hops: Maximum reasoning hops (default 4 for CWQ).
        max_paths: Maximum number of reasoning paths per question.
    """

    def __init__(
        self,
        data_path: str | Path,
        kg: FreebaseKG,
        tokenizer: PreTrainedTokenizer,
        max_hops: int = 4,
        max_paths: int = 20,
    ) -> None:
        self.data_path = Path(data_path)
        self.kg = kg
        self.tokenizer = tokenizer
        self.max_hops = max_hops
        self.max_paths = max_paths

        if not self.data_path.exists():
            raise FileNotFoundError(f"CWQ data file not found: {self.data_path}")

        self.samples = self._load_data()

    def _load_data(self) -> list[CWQSample]:
        """Load and parse CWQ data file.

        Returns:
            List of CWQSample objects.

        Raises:
            ValueError: If data format is invalid.
        """
        if self.data_path.suffix == ".jsonl":
            return self._load_jsonl()
        return self._load_json()

    def _load_jsonl(self) -> list[CWQSample]:
        """Load JSONL format data."""
        samples = []

        with open(self.data_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    sample = self._parse_item(item, line_num)
                    if sample is not None:
                        samples.append(sample)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_num}: {e}"
                    ) from e

        if not samples:
            raise ValueError(f"No valid samples found in {self.data_path}")

        return samples

    def _load_json(self) -> list[CWQSample]:
        """Load JSON format data."""
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle list or dict format
        if isinstance(data, dict):
            data = data.get("questions", data.get("Questions", []))

        if not isinstance(data, list):
            raise ValueError(
                f"Expected list of questions, got {type(data).__name__}"
            )

        samples = []
        for idx, item in enumerate(data):
            sample = self._parse_item(item, idx)
            if sample is not None:
                samples.append(sample)

        if not samples:
            raise ValueError(f"No valid samples found in {self.data_path}")

        return samples

    def _parse_item(self, item: dict, idx: int) -> CWQSample | None:
        """Parse a single CWQ item.

        CWQ has a specific format with compositional questions.

        Args:
            item: Raw dictionary from JSON.
            idx: Item index for error messages.

        Returns:
            CWQSample or None if item is invalid.
        """
        # Extract question ID
        question_id = item.get("ID", item.get("id", f"cwq_{idx}"))

        # Extract question text (CWQ uses "question" field)
        question = item.get("question", "")
        machine_question = item.get("machine_question", question)

        if not question and not machine_question:
            return None

        # Extract compositional answer
        composition_answer = item.get("composition_answer", "")

        # Extract WebQSP origin (if applicable)
        webqsp_question = item.get("webqsp_question")
        webqsp_id = item.get("webqsp_ID", item.get("webqsp_id"))

        # Extract answers
        answers = self._extract_answers(item)

        # Extract anchor entities
        anchor_entities = self._extract_anchor_entities(item)

        # Extract SPARQL
        sparql = item.get("sparql", item.get("SPARQL"))

        # Extract complexity type
        complexity = item.get("compositionality_type", item.get("complexity"))

        return CWQSample(
            question_id=str(question_id),
            question=question,
            machine_question=machine_question,
            composition_answer=composition_answer,
            webqsp_question=webqsp_question,
            webqsp_id=webqsp_id,
            answers=answers,
            anchor_entities=anchor_entities,
            sparql=sparql,
            complexity=complexity,
        )

    def _extract_answers(self, item: dict) -> list[dict[str, str]]:
        """Extract answer entities from item.

        CWQ answers can be in various formats.

        Args:
            item: Raw dictionary from JSON.

        Returns:
            List of answer dictionaries with 'answer' and 'answer_id' keys.
        """
        answers = []

        raw_answers = item.get("answers", item.get("Answers", []))

        for ans in raw_answers:
            if isinstance(ans, dict):
                answer_entry = {
                    "answer": ans.get("answer", ans.get("EntityName", "")),
                    "answer_id": ans.get("answer_id", ans.get("AnswerArgument", "")),
                    "aliases": ans.get("aliases", []),
                }
                if answer_entry["answer"] or answer_entry["answer_id"]:
                    answers.append(answer_entry)
            elif isinstance(ans, str):
                answers.append({"answer": ans, "answer_id": "", "aliases": []})

        return answers

    def _extract_anchor_entities(
        self, item: dict
    ) -> list[tuple[str, str]]:
        """Extract anchor entities from item.

        CWQ may have multiple seed entities due to compositional questions.

        Args:
            item: Raw dictionary from JSON.

        Returns:
            List of (entity_id, entity_name) tuples.
        """
        entities = []

        # Format 1: seed_entities
        for ent in item.get("seed_entities", []):
            if isinstance(ent, dict):
                mid = ent.get("mid", ent.get("id", ""))
                name = ent.get("name", "")
            elif isinstance(ent, str):
                mid = ent
                name = ""
            else:
                continue
            if mid and (mid, name) not in entities:
                entities.append((mid, name))

        # Format 2: compositional_entities
        for ent in item.get("compositional_entities", []):
            if isinstance(ent, dict):
                mid = ent.get("mid", ent.get("id", ""))
                name = ent.get("name", "")
            elif isinstance(ent, str):
                mid = ent
                name = ""
            else:
                continue
            if mid and (mid, name) not in entities:
                entities.append((mid, name))

        # Format 3: entities field
        for ent in item.get("entities", []):
            if isinstance(ent, dict):
                mid = ent.get("mid", ent.get("id", ""))
                name = ent.get("name", "")
            elif isinstance(ent, str):
                mid = ent
                name = ""
            else:
                continue
            if mid and (mid, name) not in entities:
                entities.append((mid, name))

        # Format 4: topic entities from SPARQL analysis
        for ent in item.get("topic_entities", []):
            if isinstance(ent, dict):
                mid = ent.get("mid", "")
                name = ent.get("name", "")
            elif isinstance(ent, str):
                mid = ent
                name = ""
            else:
                continue
            if mid and (mid, name) not in entities:
                entities.append((mid, name))

        return entities

    def get_reasoning_paths(
        self, anchor_entities: list[tuple[str, str]]
    ) -> list[list[tuple[str, str, str]]]:
        """Retrieve reasoning paths from anchor entities.

        For CWQ, we may need to explore more hops (up to 4).

        Args:
            anchor_entities: List of (entity_id, entity_name) tuples.

        Returns:
            List of reasoning paths, each path is a list of (head, rel, tail) triples.
        """
        all_paths: list[list[tuple[str, str, str]]] = []

        for entity_id, entity_name in anchor_entities:
            # Try both entity_id and entity_name for matching
            anchor = entity_id if self.kg.entity_exists(entity_id) else entity_name

            if not self.kg.entity_exists(anchor):
                # Try to find entity by name matching
                matches = self.kg.find_entities_by_name(entity_name, max_results=1)
                if matches:
                    anchor = matches[0]
                else:
                    continue

            # BFS to collect relation links (up to max_hops)
            relation_links = self.kg.bfs_relation_links(anchor, self.max_hops)

            # Sample concrete paths from relation links
            paths_per_link = max(1, self.max_paths // max(len(relation_links), 1))
            paths = self.kg.sample_reasoning_paths(
                anchor,
                relation_links,
                max_paths_per_link=paths_per_link,
            )
            all_paths.extend(paths)

            if len(all_paths) >= self.max_paths:
                break

        return all_paths[: self.max_paths]

    def _format_answer_text(self, answers: list[dict[str, str]]) -> str:
        """Format answers as comma-separated string.

        Args:
            answers: List of answer dictionaries.

        Returns:
            Formatted answer string.
        """
        answer_texts = []
        for ans in answers:
            text = ans.get("answer", ans.get("answer_id", ""))
            if text:
                answer_texts.append(text)
        return ", ".join(answer_texts)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Get a single sample with reasoning paths.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - question: Question text
                - question_id: Question identifier
                - reasoning_paths: List of reasoning paths
                - answer_ids: Tokenized answer tensor
                - answers: List of answer dictionaries
                - complexity: Question complexity type
        """
        sample = self.samples[idx]

        # Get reasoning paths from KG
        reasoning_paths = self.get_reasoning_paths(sample.anchor_entities)

        # Tokenize answers
        answer_text = self._format_answer_text(sample.answers)
        answer_encoding = self.tokenizer(
            answer_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )

        return {
            "question": sample.question,
            "question_id": sample.question_id,
            "reasoning_paths": reasoning_paths,
            "answer_ids": answer_encoding["input_ids"],
            "answers": sample.answers,
            "anchor_entities": sample.anchor_entities,
            "complexity": sample.complexity,
        }

    def iter_samples(self) -> Iterator[CWQSample]:
        """Iterate over raw samples without processing.

        Yields:
            CWQSample objects.
        """
        yield from self.samples

    def get_samples_by_complexity(
        self, complexity: str
    ) -> list[CWQSample]:
        """Get all samples of a specific complexity type.

        CWQ complexity types include:
        - composition: Multi-hop compositional
        - conjunction: Questions with AND
        - superlative: Comparative questions
        - comparative: Comparison between entities

        Args:
            complexity: Complexity type string.

        Returns:
            List of matching samples.
        """
        return [s for s in self.samples if s.complexity == complexity]

    @property
    def num_samples(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    @property
    def complexity_distribution(self) -> dict[str, int]:
        """Get distribution of complexity types.

        Returns:
            Dictionary mapping complexity type to count.
        """
        distribution: dict[str, int] = {}
        for sample in self.samples:
            key = sample.complexity or "unknown"
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def get_sample_by_id(self, question_id: str) -> CWQSample | None:
        """Get sample by question ID.

        Args:
            question_id: Question identifier.

        Returns:
            CWQSample or None if not found.
        """
        for sample in self.samples:
            if sample.question_id == question_id:
                return sample
        return None

    def __repr__(self) -> str:
        return (
            f"CWQDataset(path={self.data_path}, "
            f"samples={len(self.samples)}, max_hops={self.max_hops})"
        )
