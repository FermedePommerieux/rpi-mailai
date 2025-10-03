"""Core subsystems for MailAI."""

__all__ = [
    "Engine",
    "Message",
    "FeatureSketch",
    "extract_features",
    "hash_text_window",
    "Example",
    "LearningCfg",
    "Metrics",
    "ModelBundle",
    "Prediction",
]


def __getattr__(name: str):
    if name in {"Engine", "Message"}:
        from . import engine

        return getattr(engine, name)
    if name in {"FeatureSketch", "extract_features", "hash_text_window"}:
        from . import features

        return getattr(features, name)
    if name in {"Example", "LearningCfg", "Metrics", "ModelBundle", "Prediction"}:
        from . import learner

        return getattr(learner, name)
    raise AttributeError(name)
