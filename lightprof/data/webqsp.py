"""WebQSP dataset loader and utilities.

This module provides dataset loading for the WebQuestions Semantic Parses
dataset, which contains 4,737 questions requiring up to 2-hop reasoning.

Reference:
- Paper: "The Value of Semantic Parse Labeling for Knowledge Base Question Answering"
- URL: https://www.microsoft.com/en-us/download/details.aspx?id=52763
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from lightprof.data.freebase import FreebaseKG


@dataclass
class WebQSPSample:
    """A single WebQSP sample.

    Attributes:
        question_id: Unique identifier for the question.
        question: Original question text.
        processed_question: Cleaned question text.
        anchor_entities: List of (entity_id, entity_name) tuples.
        answers: List of answer entity names.
        sparql: SPARQL query (if available).
    """

    question_id: str
    question: str
    processed_question: str
    anchor_entities: list[tuple[str, str]]
    answers: list[str]
    sparql: str | None = None


class WebQSPDataset(Dataset[dict[str, object]]):
    """WebQSP dataset for KGQA.

    Loads WebQSP data and provides samples with reasoning paths retrieved
    from the knowledge graph.

    Args:
        data_path: Path to the JSON/JSONL data file.
        kg: FreebaseKG instance for path retrieval.
        tokenizer: Tokenizer for encoding answers.
        max_hops: Maximum reasoning hops (default 2 for WebQSP).
        max_paths: Maximum number of reasoning paths per question.
    """

    def __init__(
        self,
        data_path: str | Path,
        kg: FreebaseKG,
        tokenizer: PreTrainedTokenizer,
        max_hops: int = 2,
        max_paths: int = 20,
    ) -> None:
        self.data_path = Path(data_path)
        self.kg = kg
        self.tokenizer = tokenizer
        self.max_hops = max_hops
        self.max_paths = max_paths

        if not self.data_path.exists():
            raise FileNotFoundError(f"WebQSP data file not found: {self.data_path}")

        self.samples = self._load_data()

    def _load_data(self) -> list[WebQSPSample]:
        """Load and parse WebQSP data file.

        Supports both JSON and JSONL formats.

        Returns:
            List of WebQSPSample objects.

        Raises:
            ValueError: If data format is invalid.
        """
        samples = []

        if self.data_path.suffix == ".jsonl":
            samples = self._load_jsonl()
        else:
            samples = self._load_json()

        if not samples:
            raise ValueError(f"No valid samples found in {self.data_path}")

        return samples

    def _load_jsonl(self) -> list[WebQSPSample]:
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

        return samples

    def _load_json(self) -> list[WebQSPSample]:
        """Load JSON format data."""
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list and dict with "Questions" key
        if isinstance(data, dict):
            data = data.get("Questions", data.get("questions", []))

        if not isinstance(data, list):
            raise ValueError(
                f"Expected list of questions, got {type(data).__name__}"
            )

        samples = []
        for idx, item in enumerate(data):
            sample = self._parse_item(item, idx)
            if sample is not None:
                samples.append(sample)

        return samples

    def _parse_item(self, item: dict, idx: int) -> WebQSPSample | None:
        """Parse a single WebQSP item.

        Args:
            item: Raw dictionary from JSON.
            idx: Item index for error messages.

        Returns:
            WebQSPSample or None if item is invalid.
        """
        # Extract question ID
        question_id = item.get("QuestionId", item.get("id", f"webqsp_{idx}"))

        # Extract question text
        question = item.get("RawQuestion", item.get("question", ""))
        processed_question = item.get(
            "ProcessedQuestion",
            item.get("processed_question", question),
        )

        if not question and not processed_question:
            return None

        # Extract anchor entities (topic entities in WebQSP)
        anchor_entities = self._extract_anchor_entities(item)

        # Extract answers
        answers = self._extract_answers(item)

        # Extract SPARQL (optional)
        parses = item.get("Parses", [])
        sparql = None
        if parses and isinstance(parses, list) and len(parses) > 0:
            sparql = parses[0].get("Sparql", parses[0].get("sparql"))

        return WebQSPSample(
            question_id=str(question_id),
            question=question,
            processed_question=processed_question,
            anchor_entities=anchor_entities,
            answers=answers,
            sparql=sparql,
        )

    def _extract_anchor_entities(
        self, item: dict
    ) -> list[tuple[str, str]]:
        """Extract anchor entities from item.

        WebQSP uses different formats for entity annotations.

        Args:
            item: Raw dictionary from JSON.

        Returns:
            List of (entity_id, entity_name) tuples.
        """
        entities = []

        # Format 1: TopicEntityMid + TopicEntityName
        topic_mid = item.get("TopicEntityMid", item.get("topic_entity_mid"))
        topic_name = item.get("TopicEntityName", item.get("topic_entity_name", ""))
        if topic_mid:
            entities.append((topic_mid, topic_name))

        # Format 2: Parses with TopicEntityMid
        for parse in item.get("Parses", []):
            mid = parse.get("TopicEntityMid")
            name = parse.get("TopicEntityName", "")
            if mid and (mid, name) not in entities:
                entities.append((mid, name))

        # Format 3: entities list
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

        return entities

    def _extract_answers(self, item: dict) -> list[str]:
        """Extract answer entities from item.

        Args:
            item: Raw dictionary from JSON.

        Returns:
            List of answer entity names.
        """
        answers = []

        # Format 1: Parses with Answers
        for parse in item.get("Parses", []):
            for ans in parse.get("Answers", []):
                if isinstance(ans, dict):
                    name = ans.get("EntityName", ans.get("AnswerArgument", ""))
                    if name:
                        answers.append(name)
                elif isinstance(ans, str):
                    answers.append(ans)

        # Format 2: Direct answers field
        if not answers:
            for ans in item.get("answers", item.get("Answers", [])):
                if isinstance(ans, dict):
                    name = ans.get("name", ans.get("EntityName", ""))
                    if name:
                        answers.append(name)
                elif isinstance(ans, str):
                    answers.append(ans)

        return answers

    def get_reasoning_paths(
        self, anchor_entities: list[tuple[str, str]]
    ) -> list[list[tuple[str, str, str]]]:
        """Retrieve reasoning paths from anchor entities.

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

            # BFS to collect relation links
            relation_links = self.kg.bfs_relation_links(anchor, self.max_hops)

            # Sample concrete paths from relation links
            paths = self.kg.sample_reasoning_paths(
                anchor,
                relation_links,
                max_paths_per_link=self.max_paths // max(len(relation_links), 1),
            )
            all_paths.extend(paths)

            if len(all_paths) >= self.max_paths:
                break

        return all_paths[: self.max_paths]

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
                - answers: List of answer strings
        """
        sample = self.samples[idx]

        # Get reasoning paths from KG
        reasoning_paths = self.get_reasoning_paths(sample.anchor_entities)

        # Tokenize answers
        answer_text = ", ".join(sample.answers) if sample.answers else ""
        answer_encoding = self.tokenizer(
            answer_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )

        return {
            "question": sample.processed_question or sample.question,
            "question_id": sample.question_id,
            "reasoning_paths": reasoning_paths,
            "answer_ids": answer_encoding["input_ids"],
            "answers": sample.answers,
            "anchor_entities": sample.anchor_entities,
        }

    def iter_samples(self) -> Iterator[WebQSPSample]:
        """Iterate over raw samples without processing.

        Yields:
            WebQSPSample objects.
        """
        yield from self.samples

    @property
    def num_samples(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def get_sample_by_id(self, question_id: str) -> WebQSPSample | None:
        """Get sample by question ID.

        Args:
            question_id: Question identifier.

        Returns:
            WebQSPSample or None if not found.
        """
        for sample in self.samples:
            if sample.question_id == question_id:
                return sample
        return None

    def __repr__(self) -> str:
        return (
            f"WebQSPDataset(path={self.data_path}, "
            f"samples={len(self.samples)}, max_hops={self.max_hops})"
        )
