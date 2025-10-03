"""mailai.core.learner

What:
  Implement a lightweight multinomial Naive Bayes learner used to classify
  messages into anonymised categories based on hashed feature vectors.

Why:
  The Raspberry Pi deployment constraints make heavyweight ML libraries
  impractical. A pure-Python learner offers deterministic behaviour, modest
  resource usage, and straightforward auditing to verify how categories are
  produced.

How:
  - Represent examples, configuration, predictions, and metrics via dataclasses
    for clarity and serialization friendliness.
  - Train class priors and per-feature counts with Laplace smoothing, rejecting
    under-sampled classes early to avoid unreliable predictions.
  - Compute predictions by scoring features in log-space and normalising with a
    stable softmax step to avoid floating-point underflow.
  - Provide evaluation helpers that report macro-averaged F1 for sanity checks.

Interfaces:
  - :class:`Example`, :class:`LearningCfg`, :class:`ModelBundle`,
    :class:`Prediction`, :class:`Metrics` describing learner inputs/outputs.
  - :func:`train`, :func:`predict`, :func:`evaluate` operating on hashed feature
    maps.

Invariants & Safety:
  - Input features are assumed to be peppered hashes; the learner never stores
    plaintext terms.
  - Training enforces ``min_samples_per_class`` to avoid degenerate priors.
  - Log-space probability computations mitigate floating-point precision loss
    for sparse vectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class Example:
    """Single labelled sample supplied to the learner.

    What:
      Bundle sparse feature weights with the anonymised target label.

    Why:
      Expressing examples as dataclasses makes datasets explicit and aids unit
      testing when constructing synthetic samples.

    How:
      Expect positive feature weights keyed by hashed identifiers so the learner
      can accumulate counts efficiently.

    Attributes:
      features: Mapping of hashed feature identifiers to weights.
      label: Canonicalised anonymised label associated with the sample.
    """

    features: Dict[str, float]
    label: str


@dataclass
class LearningCfg:
    """Configuration switches controlling training.

    What:
      Capture hyper-parameters that gate training data quality.

    Why:
      MailAI needs a minimal sample size per class to avoid overfitting sparse
      data; this configuration makes the constraint explicit.

    How:
      Store the minimum number of samples as a dataclass field for clarity.

    Attributes:
      min_samples_per_class: Required count before accepting a class during
        training.
    """

    min_samples_per_class: int


@dataclass
class Prediction:
    """Learner output describing the most likely label.

    What:
      Provide the predicted label, associated probability, and the most
      influential features for explainability.

    Why:
      Downstream components (status emails, audits) need to know which hashed
      features contributed to a decision even when absolute probabilities are
      low.

    How:
      Store the probability resulting from the softmax normalisation and slice
      the input feature keys to capture the top contributors.

    Attributes:
      label: Predicted anonymised label.
      prob: Confidence estimate bounded between 0 and 1.
      top_features: Ordered list of feature identifiers contributing most weight.
    """

    label: str
    prob: float
    top_features: List[str]


@dataclass
class Metrics:
    """Evaluation results summarising prediction quality.

    What:
      Expose macro-averaged F1 score to characterise class-balanced accuracy.

    Why:
      Macro F1 is resilient to class imbalance and suits anonymised labels where
      each class should contribute equally to evaluation.

    How:
      Provide a dataclass wrapper to enable direct JSON serialisation when
      reporting evaluation runs.

    Attributes:
      macro_f1: Macro-averaged F1 value between 0 and 1.
    """

    macro_f1: float


@dataclass
class ModelBundle:
    """Parameters required for inference and evaluation.

    What:
      Aggregate priors, feature totals, per-class feature counts, and the known
      vocabulary after training.

    Why:
      Bundling the trained artefacts simplifies persistence and makes the
      contract of :func:`train` explicit to callers performing inference later.

    How:
      Collect dictionaries keyed by class labels plus a vocabulary list to avoid
      recomputing them at inference time.

    Attributes:
      priors: Prior probabilities for each class.
      feature_totals: Sum of feature weights per class (normalisation factor).
      feature_counts: Nested mapping storing counts per feature and class.
      vocabulary: List of unique feature identifiers observed during training.
    """

    priors: Dict[str, float]
    feature_totals: Dict[str, float]
    feature_counts: Dict[str, Dict[str, float]]
    vocabulary: List[str]


ALPHA = 1.0


def train(dataset: Iterable[Example], cfg: LearningCfg) -> ModelBundle:
    """Fit a multinomial Naive Bayes model on hashed features.

    What:
      Consume labelled examples and estimate class priors alongside per-feature
      likelihood counts ready for inference.

    Why:
      Training encapsulates the minimal statistics necessary to reuse the model
      without storing the full dataset, honouring privacy and resource limits.

    How:
      - Materialise the dataset to enable validation of sample counts.
      - Accumulate class counts and feature totals, skipping non-positive
        feature values because they carry no information for multinomial NB.
      - Enforce ``min_samples_per_class`` and compute priors via relative
        frequencies.

    Args:
      dataset: Iterable of :class:`Example` instances providing features and
        labels.
      cfg: Training configuration containing minimum sample threshold.

    Returns:
      :class:`ModelBundle` capturing the learned parameters.

    Raises:
      ValueError: If the dataset is empty or a class lacks enough samples.
    """

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
                # NOTE: Multinomial NB ignores non-positive counts, so we skip to
                # maintain numeric stability and avoid polluting totals.
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
    """Score feature vectors against the trained model.

    What:
      Evaluate a sparse feature dictionary and produce predictions sorted by
      posterior probability.

    Why:
      The rules engine and diagnostics require ordered label suggestions to
      drive automation or display helpful hints to operators.

    How:
      - Iterate through each class, accumulate log-probabilities using Laplace
        smoothing and the learned priors.
      - Apply a numerically stable softmax by shifting scores by the maximum
        value before exponentiation.
      - Package the resulting probabilities and top contributing features into
        :class:`Prediction` objects sorted by confidence.

    Args:
      model: Trained :class:`ModelBundle` with priors and counts.
      features: Sparse feature weights for the message being classified.

    Returns:
      Predictions sorted by descending probability.
    """

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
    """Measure macro F1 of the learner on a labelled dataset.

    What:
      Assess how well the trained model predicts labels across all classes using
      macro-averaged F1.

    Why:
      Macro F1 treats each class equally, preventing dominant labels from
      masking poor minority performance—a critical property when categories are
      anonymised and expected to balance actions.

    How:
      - Materialise the dataset for deterministic iteration and handle empty
        datasets gracefully.
      - Compute per-class true/false positives and negatives using the model's
        top prediction.
      - Derive precision, recall, and F1 for each class before averaging.

    Args:
      model: Trained :class:`ModelBundle`.
      dataset: Iterable of labelled :class:`Example` records to evaluate.

    Returns:
      :class:`Metrics` containing the macro F1 score.
    """

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


# TODO: Other modules require the same treatment (What/Why/How docstrings + module header).
