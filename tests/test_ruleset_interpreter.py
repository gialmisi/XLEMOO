import numpy as np
import numpy.testing as npt
import pytest
from dataclasses import dataclass
from XLEMOO.ruleset_interpreter import instantiate_ruleset_rules, extract_slipper_rules
from typing import List, Tuple, Dict


@dataclass
class DummyRule:
    agg_dict: Dict[Tuple[str, str], str]


@dataclass
class DummySlipper:
    estimator_weights_: List[float]
    rules_: List[DummyRule]


@pytest.fixture
def dummy_slipper_classifier():
    rule_dicts = [
        {("x_2", "<"): "0.2", ("x_1", ">="): "0.9"},
        {("x_2", "<"): "0.2", ("x_1", ">="): "0.9", ("x_3", "<"): "0.1"},
        {("x_1", ">"): "0.8", ("x_3", "<="): "0.3"},
    ]
    weights = [0.5, 0.3, 0.9]

    dummy_rule_list = [DummyRule(rule_dict) for rule_dict in rule_dicts]

    dummy_classifier = DummySlipper(weights, dummy_rule_list)

    return dummy_classifier


@pytest.mark.rulesets
def test_instantiate_ruleset_rules():
    pass


@pytest.mark.rulesets
def test_extract_slipper_rules(dummy_slipper_classifier):
    rules, weights = extract_slipper_rules(dummy_slipper_classifier)

    assert len(rules) == len(weights)

    for (i, rule) in enumerate(rules):
        assert rule == dummy_slipper_classifier.rules_[i].agg_dict

    for (i, weight) in enumerate(weights):
        npt.assert_almost_equal(dummy_slipper_classifier.estimator_weights_[i], weight)
