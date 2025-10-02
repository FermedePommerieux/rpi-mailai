from mailai.core import learner


def test_train_predict_evaluate():
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
