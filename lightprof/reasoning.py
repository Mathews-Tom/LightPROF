"""Reasoning Module for LightPROF.

This module implements the knowledge-prompted reasoning pipeline that
combines soft prompts (from Knowledge Adapter) with hard prompts to
generate answers using a frozen LLM.

Reference: LightPROF paper Section 4.3 (Knowledge Prompts Mixed Reasoning)

Architecture:
    - Soft Prompts: Knowledge embeddings from Knowledge Adapter
    - Hard Prompts: Instruction templates for the LLM
    - LLM: Frozen LLaMA-3-8B-Instruct for answer generation
    - Injection: Via inputs_embeds parameter (not text tokens)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from lightprof.adapter import KnowledgeAdapter
    from lightprof.config import ModelConfig

# Type alias for reasoning paths
ReasoningPath = list[tuple[str, str, str]]


class ReasoningModule:
    """Reasoning module with soft prompt injection into LLaMA.

    Per paper Section 4.3: Knowledge prompts are created by combining
    soft prompts (knowledge embeddings) with hard prompts (text instructions).
    The soft prompts are injected via the inputs_embeds parameter.

    Args:
        model_config: Model configuration.
        knowledge_adapter: Knowledge Adapter for generating soft prompts.
        device: Device to use ('cuda', 'cpu', or 'auto').

    Attributes:
        adapter: Knowledge Adapter instance.
        tokenizer: LLaMA tokenizer.
        model: Frozen LLaMA model.
    """

    # Hard prompt template per paper
    HARD_PROMPT_TEMPLATE = (
        "Based on the knowledge graphs, please answer the given question. "
        "Please keep the answer as simple as possible and return all the "
        "possible answers as a list.\n\nQuestion: {question}\nAnswer:"
    )

    def __init__(
        self,
        model_config: ModelConfig,
        knowledge_adapter: KnowledgeAdapter,
        device: str = "auto",
    ) -> None:
        self.model_config = model_config
        self.adapter = knowledge_adapter

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load frozen LLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.llm_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        # Freeze LLM parameters (per paper requirement)
        if model_config.freeze_llm:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from LLM's embedding layer.

        Args:
            input_ids: Token IDs [batch_size, seq_len].

        Returns:
            Token embeddings [batch_size, seq_len, embed_dim].
        """
        return self.model.get_input_embeddings()(input_ids)

    def create_hard_prompt(self, question: str) -> str:
        """Create hard prompt from question.

        Args:
            question: Natural language question.

        Returns:
            Formatted hard prompt string.
        """
        return self.HARD_PROMPT_TEMPLATE.format(question=question)

    def create_combined_embeddings(
        self,
        question: str,
        soft_prompts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine soft prompts with hard prompt embeddings.

        Per paper: Knowledge embeddings are prepended to the text
        embeddings to create the full input.

        Args:
            question: Natural language question.
            soft_prompts: Knowledge embeddings from adapter [num_paths, embed_dim].

        Returns:
            Tuple of (combined_embeddings, attention_mask).
        """
        # Create hard prompt
        hard_prompt = self.create_hard_prompt(question)

        # Tokenize hard prompt
        inputs = self.tokenizer(
            hard_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.device)

        # Get text embeddings
        text_embeds = self.get_input_embeddings(input_ids)

        # Ensure soft prompts have batch dimension [1, num_paths, embed_dim]
        if soft_prompts.dim() == 2:
            soft_prompts = soft_prompts.unsqueeze(0)

        # Ensure same device
        soft_prompts = soft_prompts.to(self.device)

        # Concatenate: [soft_prompts] + [text_embeds]
        combined = torch.cat([soft_prompts, text_embeds], dim=1)

        # Create attention mask (all ones for both soft and hard prompts)
        soft_prompt_mask = torch.ones(
            1, soft_prompts.size(1), device=self.device, dtype=torch.long
        )
        text_mask = inputs["attention_mask"].to(self.device)
        attention_mask = torch.cat([soft_prompt_mask, text_mask], dim=1)

        return combined, attention_mask

    def generate(
        self,
        question: str,
        reasoning_paths: list[ReasoningPath],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Generate answer using soft prompt injection.

        Per paper Section 4.3: The frozen LLM generates answers based on
        the combined soft and hard prompts.

        Args:
            question: Natural language question.
            reasoning_paths: Reasoning paths from retrieval module.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 for greedy).
            do_sample: Whether to use sampling.

        Returns:
            Generated answer string.
        """
        # Get soft prompts from Knowledge Adapter
        soft_prompts = self.adapter(reasoning_paths)

        # Create combined input embeddings
        inputs_embeds, attention_mask = self.create_combined_embeddings(
            question, soft_prompts
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated tokens (skip soft prompt positions)
        num_soft_tokens = soft_prompts.size(0) if soft_prompts.dim() == 2 else soft_prompts.size(1)
        generated_ids = outputs[0, num_soft_tokens:]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def forward_for_training(
        self,
        question: str,
        reasoning_paths: list[ReasoningPath],
        answer: str,
    ) -> torch.Tensor:
        """Forward pass for training - returns loss.

        Computes causal language modeling loss only on answer tokens.

        Args:
            question: Natural language question.
            reasoning_paths: Reasoning paths from retrieval.
            answer: Ground truth answer string.

        Returns:
            Cross-entropy loss on answer tokens.
        """
        # Get soft prompts
        soft_prompts = self.adapter(reasoning_paths)

        # Create combined prompt embeddings
        inputs_embeds, attention_mask = self.create_combined_embeddings(
            question, soft_prompts
        )

        # Tokenize answer
        answer_encoding = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            add_special_tokens=False,
        )
        answer_ids = answer_encoding["input_ids"].to(self.device)

        # Get answer embeddings
        answer_embeds = self.get_input_embeddings(answer_ids)

        # Concatenate: [prompt_embeds] + [answer_embeds]
        full_embeds = torch.cat([inputs_embeds, answer_embeds], dim=1)

        # Update attention mask
        answer_mask = answer_encoding["attention_mask"].to(self.device)
        full_attention_mask = torch.cat([attention_mask, answer_mask], dim=1)

        # Create labels: -100 for prompt tokens (ignored in loss), answer ids for answer tokens
        prompt_length = inputs_embeds.size(1)
        labels = torch.full(
            (1, full_embeds.size(1)),
            -100,
            device=self.device,
            dtype=torch.long,
        )
        labels[0, prompt_length:] = answer_ids[0]

        # Forward pass with loss computation
        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )

        return outputs.loss

    def forward_batch(
        self,
        questions: list[str],
        reasoning_paths_batch: list[list[ReasoningPath]],
        answers: list[str],
    ) -> torch.Tensor:
        """Batch forward pass for training.

        Note: Due to variable-length soft prompts, processes samples
        sequentially and averages loss.

        Args:
            questions: List of questions.
            reasoning_paths_batch: List of reasoning paths per question.
            answers: List of ground truth answers.

        Returns:
            Average loss over batch.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        for question, paths, answer in zip(
            questions, reasoning_paths_batch, answers, strict=True
        ):
            loss = self.forward_for_training(question, paths, answer)
            total_loss = total_loss + loss

        return total_loss / len(questions)

    def predict_hops(self, question: str) -> int:
        """Predict reasoning hops using the hop predictor.

        Delegates to the adapter's hop predictor if available.

        Args:
            question: Natural language question.

        Returns:
            Predicted hop count.

        Raises:
            NotImplementedError: If no hop predictor is available.
        """
        raise NotImplementedError(
            "Use RetrievalModule.predict_hops() instead. "
            "The HopPredictor is part of the retrieval pipeline."
        )

    def __repr__(self) -> str:
        return (
            f"ReasoningModule(\n"
            f"  llm={self.model_config.llm_name},\n"
            f"  frozen={self.model_config.freeze_llm},\n"
            f"  adapter_params={self.adapter.num_trainable_parameters:,}\n"
            f")"
        )
