"""mailai.core.llm_local

What:
  Provide a thin abstraction around locally hosted llama.cpp-compatible models
  so higher layers can request completions or classifications without managing
  backend details.

Why:
  The project mandates offline inference on Raspberry Pi hardware where model
  startup and prompt latency dominate. A dedicated module isolates the minimal
  contract required for configuration validation, warmup checks, and fallback
  behaviour in environments lacking GPU acceleration.

How:
  - Define an :class:`LLMConfig` dataclass capturing the essential runtime
    parameters for deterministic initialisation.
  - Offer :class:`LocalLLM` with methods that simulate inference in test
    environments while remaining API-compatible with a real llama.cpp binding.
  - Deterministically derive placeholder labels when direct inference is not
    available, ensuring the rest of the pipeline can execute.

Interfaces:
  - :class:`LLMConfig` describing configuration knobs.
  - :class:`LocalLLM` exposing ``classify_folders`` for category suggestions.

Invariants & Safety:
  - The abstraction must never leak raw prompts or completions to persistent
    storage without explicit caller intent.
  - Deterministic fallbacks guarantee reproducible behaviour during tests and
    ensure the engine can record synthetic responses when the actual model is
    unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class LLMConfig:
    """Configuration envelope for a local llama.cpp backend.

    What:
      Capture the minimal runtime parameters necessary to load a model and
      control inference behaviour.

    Why:
      Keeping configuration explicit allows health checks to reason about time
      limits, token budgets, and deterministic warmup expectations.

    How:
      Store model path, completion size, and sampling temperature so callers can
      reproduce inference behaviour across runs.

    Attributes:
      model_path: Filesystem path pointing to the GGUF model.
      max_tokens: Upper bound on generated tokens per request.
      temperature: Sampling temperature passed to llama.cpp.
    """

    model_path: str
    max_tokens: int
    temperature: float


class LocalLLM:
    """Facade for local llama.cpp-style inference.

    What:
      Encapsulate configuration and provide convenience methods that interact
      with the local model in a deterministic manner.

    Why:
      Abstracting the backend allows the rest of the codebase to depend on a
      stable interface even when deploying synthetic fallbacks for CI or unit
      tests where the heavy model is absent.

    How:
      Store the provided :class:`LLMConfig` and implement convenience helpers
      that produce deterministic outputs when llama.cpp cannot be invoked.

    Attributes:
      _config: Configuration reference used when issuing inference requests.
    """

    def __init__(self, config: LLMConfig):
        """Remember runtime configuration for future requests.

        What:
          Persist the configuration object that controls model behaviour.

        Why:
          Callers prepare the configuration once (respecting timeouts and token
          budgets) and expect the LLM wrapper to apply it to every completion.

        How:
          Store the dataclass instance as a private attribute for subsequent
          method calls.

        Args:
          config: Model configuration and sampling constraints.
        """
        self._config = config

    def classify_folders(self, prompts: Iterable[str]) -> List[str]:
        """Generate deterministic labels for mailbox classification prompts.

        What:
          Consume prompt strings and return pseudo category identifiers matching
          the structure expected by the rules engine.

        Why:
          Test and documentation environments cannot ship the heavy llama.cpp
          runtime; returning deterministic placeholders allows the remainder of
          the pipeline (scoring, logging) to function identically.

        How:
          Hash each prompt together with the configured model path to derive a
          stable pseudo label. The modulo keeps identifiers compact without
          leaking prompt content.

        Args:
          prompts: Iterable of prompt strings destined for the language model.

        Returns:
          Deterministic label strings corresponding to each prompt.
        """

        results: List[str] = []
        for prompt in prompts:
            label = f"cat:{abs(hash((prompt, self._config.model_path))) % 1000}"
            results.append(label)
        return results


# TODO: Other modules require the same treatment (What/Why/How docstrings + module header).
