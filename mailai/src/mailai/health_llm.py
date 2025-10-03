"""Runtime health probe for the embedded llama-cpp-python model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from llama_cpp import Llama

_PROMPT: Final[str] = "ok?"


def _int_from_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


def main() -> int:
    model_path = os.environ.get("LLM_MODEL_PATH")
    if not model_path:
        print("LLM_MODEL_PATH is not set", file=sys.stderr)
        return 1

    model_file = Path(model_path)
    if not model_file.is_file():
        print(f"model file not found: {model_file}", file=sys.stderr)
        return 1

    threads = _int_from_env("LLM_N_THREADS", 3)
    ctx_size = _int_from_env("LLM_CTX_SIZE", 2048)

    try:
        llm = Llama(
            model_path=str(model_file),
            n_threads=threads,
            n_ctx=ctx_size,
            logits_all=False,
            embedding=False,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - library exception surface
        print(f"failed to load model: {exc}", file=sys.stderr)
        return 1

    try:
        result = llm(
            _PROMPT,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.0,
            return_dict=True,
        )
    except Exception as exc:  # pragma: no cover - library exception surface
        print(f"completion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        del llm

    choices = result.get("choices") if isinstance(result, dict) else None
    if not choices:
        print("completion returned no choices", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - module executable entry point
    raise SystemExit(main())
