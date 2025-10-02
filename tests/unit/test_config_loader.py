import pathlib

import pytest

from mailai.config import loader
from mailai.config.schema import RulesV2


def test_load_rules(tmp_path: pathlib.Path):
    data = pathlib.Path("examples/rules.yaml").read_bytes()
    document = loader.load_rules(data)
    assert isinstance(document.model, RulesV2)
    assert document.model.version == 2
    dumped = loader.dump_rules(document.model)
    assert dumped.startswith(b"version: 2")


def test_invalid_rules_raises():
    with pytest.raises(loader.YamlValidationError):
        loader.load_rules(b"version: 1")
