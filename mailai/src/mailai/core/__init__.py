"""Aggregated exports for MailAI's rule-processing core.

What:
  Provide a light-weight package facade exposing frequently used engine,
  feature, and learning primitives while deferring heavy imports until they are
  needed.

Why:
  Core modules pull in large dependencies (NumPy, llama bindings) that slow
  down interpreter start-up on constrained hardware. Lazy access keeps CLI
  tools responsive and reduces memory pressure when only a subset of features
  is required.

How:
  Defines ``__all__`` explicitly for static analyzers and implements
  ``__getattr__`` to import submodules on demand. Each attribute request is
  routed to the module that owns the symbol and the attribute is returned
  directly.

Interfaces:
  ``Engine``, ``Message``, ``FeatureSketch``, ``extract_features``,
  ``hash_text_window``, ``Example``, ``LearningCfg``, ``Metrics``,
  ``ModelBundle``, ``Prediction``.

Invariants & Safety:
  - ``__getattr__`` only exposes names from ``__all__``; unexpected attributes
    raise :class:`AttributeError` to prevent leaking implementation details.
  - Imports occur lazily but still rely on canonical module paths to avoid
    circular dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Engine",
    "Message",
    "FeatureSketch",
    "extract_features",
    "hash_text_window",
    "build_mail_feature_record",
    "IntentFeatureSettings",
    "IntentLLMSettings",
    "IntentThresholds",
    "IntentFeatures",
    "MailFeatureRecord",
    "ParsedMailMeta",
    "TextStats",
    "UrlInfo",
    "Example",
    "LearningCfg",
    "Metrics",
    "ModelBundle",
    "Prediction",
]


def __getattr__(name: str) -> Any:
    """Resolve attributes lazily to keep import costs minimal.

    What:
      Dynamically imports the relevant core submodule and returns the requested
      attribute.

    Why:
      Avoids loading heavy dependencies when they are not needed (for example
      during lightweight status checks or configuration validation).

    How:
      Checks the requested ``name`` against known groups, performs the import
      locally to leverage Python's module cache, and returns the attribute via
      :func:`getattr`. Raises :class:`AttributeError` for unknown names so static
      tooling and callers receive immediate feedback.

    Args:
      name: Attribute requested by import-time attribute access.

    Returns:
      The resolved attribute from the appropriate submodule.

    Raises:
      AttributeError: If ``name`` is not part of the public surface.
    """

    if name in {"Engine", "Message"}:
        from . import engine

        return getattr(engine, name)
    if name in {
        "FeatureSketch",
        "extract_features",
        "hash_text_window",
        "build_mail_feature_record",
        "IntentFeatureSettings",
        "IntentLLMSettings",
        "IntentThresholds",
        "IntentFeatures",
        "MailFeatureRecord",
        "ParsedMailMeta",
        "TextStats",
        "UrlInfo",
    }:
        from . import features

        return getattr(features, name)
    if name in {"Example", "LearningCfg", "Metrics", "ModelBundle", "Prediction"}:
        from . import learner

        return getattr(learner, name)
    raise AttributeError(name)


# TODO: Document remaining modules with the same What/Why/How structure.
