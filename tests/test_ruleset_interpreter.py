import numpy as np
import numpy.testing as npt
import pytest
from dataclasses import dataclass
from XLEMOO.ruleset_interpreter import (
    instantiate_ruleset_rules,
    _instantiate_ruleset_rules,
    extract_slipper_rules,
    instantiate_rules,
    Rules,
)
from typing import List, Tuple, Dict


@dataclass
class DummyRule:
    agg_dict: Rules


@dataclass
class DummySlipper:
    estimator_weights_: List[float]
    rules_: List[DummyRule]


@pytest.fixture
def dummy_slipper_classifier():
    rule_dicts = [
        {("x_2", "<"): "12.0", ("x_1", ">="): "9.5", ("x_0", ">"): "2.5"},
        {("x_2", "<"): "0.2", ("x_1", ">="): "0.9", ("x_3", "<"): "0.1"},
        {("x_1", ">"): "7.0", ("x_3", "<="): "16.0", ("x_1", ">"): "8.0"},
    ]
    weights = [0.5, 0.3, 0.9]

    dummy_rule_list = [DummyRule(rule_dict) for rule_dict in rule_dicts]

    dummy_classifier = DummySlipper(weights, dummy_rule_list)

    return dummy_classifier


@pytest.mark.rulesets
def test_instantiate_ruleset_rules(dummy_slipper_classifier):
    rules_list, weights = extract_slipper_rules(dummy_slipper_classifier)

    feature_limits = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    n_features = 4
    n_samples = 1000

    new_samples = instantiate_ruleset_rules(rules_list, weights, n_features, feature_limits, n_samples)

    assert new_samples.shape[1] == n_features
    # might not return exactly n_samples in total, but that is fine
    npt.assert_allclose(new_samples.shape[0], n_samples, atol=5)


@pytest.mark.rulesets
def test__instantiate_ruleset_rules(dummy_slipper_classifier):
    rules_list, weights = extract_slipper_rules(dummy_slipper_classifier)

    feature_limits = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    n_features = 4
    n_samples = 1000

    new_samples = _instantiate_ruleset_rules(rules_list, weights, n_features, feature_limits, n_samples)

    assert len(new_samples) == 3

    # check that samples were generated according to weights
    ## first weight
    assert new_samples[0].shape[0] > new_samples[1].shape[0]
    assert new_samples[0].shape[0] < new_samples[2].shape[0]

    ## second weight
    assert new_samples[1].shape[0] < new_samples[0].shape[0]
    assert new_samples[1].shape[0] < new_samples[2].shape[0]

    ## third weight
    assert new_samples[2].shape[0] > new_samples[0].shape[0]
    assert new_samples[2].shape[0] > new_samples[1].shape[0]


@pytest.mark.rulesets
def test__instantiate_ruleset_rules_equalop(dummy_slipper_classifier):
    rules_list = [
        {("x_1", ">"): "8.0", ("x_3", "=="): "16.0"},
        {("x_1", "="): "8.0", ("x_3", ">="): "16.0"},
    ]
    weights = [1, 1]

    feature_limits = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    n_features = 4
    n_samples = 1000

    new_samples = _instantiate_ruleset_rules(rules_list, weights, n_features, feature_limits, n_samples)

    assert len(new_samples) == 2

    npt.assert_almost_equal(new_samples[0][:, 3], 16.0)
    npt.assert_almost_equal(new_samples[1][:, 1], 8.0)


@pytest.mark.rulesets
def test__instantiate_ruleset_rules_negative_ws(dummy_slipper_classifier):
    rules_list, _ = extract_slipper_rules(dummy_slipper_classifier)
    weights = [0.5, -0.3, 0.9]

    feature_limits = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    n_features = 4
    n_samples = 1000

    new_samples = _instantiate_ruleset_rules(rules_list, weights, n_features, feature_limits, n_samples)

    # one rule should be dropped with the negative weights
    assert len(new_samples) == 2

    # check that samples were generated according to weights
    assert new_samples[0].shape[0] < new_samples[1].shape[0]
    assert new_samples[1].shape[0] > new_samples[0].shape[0]


@pytest.mark.rulesets
def test_instaniate_rules(dummy_slipper_classifier):
    rules_list, _ = extract_slipper_rules(dummy_slipper_classifier)

    first_rule = rules_list[0]

    feature_limits = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    n_features = 4
    n_samples = 1000

    new_samples = instantiate_rules(first_rule, n_features, feature_limits, n_samples)

    # check dimensions
    assert new_samples.shape[0] == n_samples
    assert new_samples.shape[1] == n_features

    # check correct limits
    ## x_0 > 0, x_0 <5
    assert np.all(new_samples[:, 0] > 0)
    assert np.all(new_samples[:, 0] < 5)
    ## x_1 > 5, x_1 < 10
    assert np.all(new_samples[:, 1] > 5)
    assert np.all(new_samples[:, 1] < 10)
    ## x_2 > 10, x_2 < 15
    assert np.all(new_samples[:, 2] > 10)
    assert np.all(new_samples[:, 2] < 15)
    ## x_3 > 15, x_3 < 20
    assert np.all(new_samples[:, 3] > 15)
    assert np.all(new_samples[:, 3] < 20)

    # check rules
    ## x_0 > 2.5
    assert np.all(new_samples[:, 0] > 2.5)
    ## x_1 >= 9.5
    assert np.all(new_samples[:, 1] > 9.5)
    ## x_2 < 12.00
    assert np.all(new_samples[:, 2] < 12.0)

    # If redundant rules are given, the stricter is adheret to
    last_rule = rules_list[2]

    new_samples_last = instantiate_rules(last_rule, n_features, feature_limits, n_samples)

    # check correct limits
    ## x_0 > 0, x_0 <5
    assert np.all(new_samples_last[:, 0] > 0)
    assert np.all(new_samples_last[:, 0] < 5)
    ## x_1 > 5, x_1 < 10
    assert np.all(new_samples_last[:, 1] > 5)
    assert np.all(new_samples_last[:, 1] < 10)
    ## x_2 > 10, x_2 < 15
    assert np.all(new_samples_last[:, 2] > 10)
    assert np.all(new_samples_last[:, 2] < 15)
    ## x_3 > 15, x_3 < 20
    assert np.all(new_samples_last[:, 3] > 15)
    assert np.all(new_samples_last[:, 3] < 20)

    # check rules
    ## x_1 > 0.8 (override > 0.7)
    assert np.all(new_samples_last[:, 1] > 0.8)
    ## x_3 < 16.0
    assert np.all(new_samples_last[:, 3] < 16.0)


@pytest.mark.rulesets
def test_extract_slipper_rules(dummy_slipper_classifier):
    rules, weights = extract_slipper_rules(dummy_slipper_classifier)

    assert len(rules) == len(weights)

    for (i, rule) in enumerate(rules):
        assert rule == dummy_slipper_classifier.rules_[i].agg_dict

    for (i, weight) in enumerate(weights):
        npt.assert_almost_equal(dummy_slipper_classifier.estimator_weights_[i], weight)
