"""Core subsystems for MailAI."""

from .engine import Engine, Message
from .features import FeatureSketch, extract_features, hash_text_window
from .learner import Example, LearningCfg, Metrics, ModelBundle, Prediction

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
