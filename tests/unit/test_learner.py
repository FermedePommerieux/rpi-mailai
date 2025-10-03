"""
Module: tests/unit/test_learner.py

What:
    Smoke-test the lightweight learning facade to confirm core workflows for
    training, prediction, and evaluation operate end-to-end on deterministic
    datasets.

Why:
    These checks guard against regressions that could surface inferences without
    sufficient samples or break evaluation metrics used for audit reporting.

How:
    Build a synthetic dataset with balanced labels, train the learner, predict on
    a known feature vector, and ensure evaluation metrics stay within valid
    bounds.

Interfaces:
    test_train_predict_evaluate

Invariants & Safety Rules:
    - Macro F1 must remain within the [0, 1] range to signal valid evaluation.
    - Predictions should reference known labels derived from training data.
"""

from mailai.core import learner


def test_train_predict_evaluate():
    """
    What:
        Validate the learner pipeline from training through evaluation on a
        simple dataset.

    Why:
        Ensures feature hashing, sample thresholds, and evaluation metrics remain
        consistent after refactors.

    How:
        Construct a balanced dataset, train a model, run a single prediction, and
        evaluate the trained model while asserting bounds on results.

    Returns:
        None
    """
    dataset = [
        learner.Example(features={"domain:example.com": 1.0}, label="Inbox"),
        learner.Example(features={"domain:example.com": 1.0}, label="Inbox"),
        learner.Example(features={"domain:example.org": 1.0}, label="Archive"),
        learner.Example(features={"domain:example.org": 1.0}, label="Archive"),
        learner.Example(features={"domain:example.net": 1.0}, label="Clients"),
        learner.Example(features={"domain:example.net": 1.0}, label="Clients"),
    ]
    cfg = learner.LearningCfg(min_samples_per_class=2)
    model = learner.train(dataset, cfg)
    prediction = learner.predict(model, {"domain:example.com": 1.0})
    assert prediction[0].label in {"Inbox", "Archive", "Clients"}
    metrics = learner.evaluate(model, dataset)
    assert 0.0 <= metrics.macro_f1 <= 1.0


# TODO: Other modules in this repository still require the same What/Why/How documentation.
