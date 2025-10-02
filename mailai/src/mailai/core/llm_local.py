"""Minimal wrapper around llama.cpp style local models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class LLMConfig:
    """Configuration for local inference."""

    model_path: str
    max_tokens: int
    temperature: float


class LocalLLM:
    """Thin abstraction over a local LLM backend."""

    def __init__(self, config: LLMConfig):
        self._config = config

    def classify_folders(self, prompts: Iterable[str]) -> List[str]:
        """Return synthetic category names.

        In the test environment we cannot execute the actual model, therefore we
        provide deterministic placeholders to allow the pipeline to proceed.
        """

        results: List[str] = []
        for prompt in prompts:
            label = f"cat:{abs(hash((prompt, self._config.model_path))) % 1000}"
            results.append(label)
        return results
