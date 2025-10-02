"""Learning pipeline using a simple multinomial Naive Bayes classifier."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class Example:
    features: Dict[str, float]
    label: str


@dataclass
class LearningCfg:
    min_samples_per_class: int


@dataclass
class Prediction:
    label: str
    prob: float
    top_features: List[str]


@dataclass
class Metrics:
    macro_f1: float


@dataclass
class ModelBundle:
    priors: Dict[str, float]
    feature_totals: Dict[str, float]
    feature_counts: Dict[str, Dict[str, float]]
    vocabulary: List[str]


ALPHA = 1.0


def train(dataset: Iterable[Example], cfg: LearningCfg) -> ModelBundle:
    examples = list(dataset)
    if not examples:
        raise ValueError("Dataset is empty")
    class_counts: Dict[str, int] = {}
    feature_counts: Dict[str, Dict[str, float]] = {}
    feature_totals: Dict[str, float] = {}
    vocabulary: Dict[str, None] = {}
    for example in examples:
        class_counts[example.label] = class_counts.get(example.label, 0) + 1
        feature_counts.setdefault(example.label, {})
        feature_totals.setdefault(example.label, 0.0)
        for feature, value in example.features.items():
            if value <= 0:
                continue
            vocabulary[feature] = None
            feature_counts[example.label][feature] = feature_counts[example.label].get(feature, 0.0) + value
            feature_totals[example.label] += value
    for label, count in class_counts.items():
        if count < cfg.min_samples_per_class:
            raise ValueError("Not enough samples per class")
    total_examples = sum(class_counts.values())
    priors = {label: count / total_examples for label, count in class_counts.items()}
    return ModelBundle(
        priors=priors,
        feature_totals=feature_totals,
        feature_counts=feature_counts,
        vocabulary=list(vocabulary.keys()),
    )


def predict(model: ModelBundle, features: Dict[str, float]) -> List[Prediction]:
    scores: Dict[str, float] = {}
    vocab_size = max(len(model.vocabulary), 1)
    for label, prior in model.priors.items():
        log_prob = math.log(prior)
        for feature, value in features.items():
            if value <= 0:
                continue
            count = model.feature_counts.get(label, {}).get(feature, 0.0)
            total = model.feature_totals.get(label, 0.0)
            likelihood = (count + ALPHA) / (total + ALPHA * vocab_size)
            log_prob += value * math.log(likelihood)
        scores[label] = log_prob
    max_score = max(scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    norm = sum(exp_scores.values())
    predictions = [
        Prediction(
            label=label,
            prob=exp_scores[label] / norm if norm else 0.0,
            top_features=sorted(features, key=features.get, reverse=True)[:5],
        )
        for label in sorted(scores, key=scores.get, reverse=True)
    ]
    return predictions


def evaluate(model: ModelBundle, dataset: Iterable[Example]) -> Metrics:
    examples = list(dataset)
    if not examples:
        return Metrics(macro_f1=0.0)
    labels = sorted(model.priors)
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}
    for example in examples:
        prediction = predict(model, example.features)[0].label
        if prediction == example.label:
            tp[example.label] += 1
        else:
            fp[prediction] += 1
            fn[example.label] += 1
    f1_scores = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return Metrics(macro_f1=macro)
