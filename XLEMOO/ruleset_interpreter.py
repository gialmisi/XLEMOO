from typing import List, Dict, Tuple
from imodels import SlipperClassifier
import numpy as np


def extract_slipper_rules(
    classifier: SlipperClassifier,
) -> Tuple[List[Dict[Tuple[str, str], str]], List[float]]:
    """Given a trained SlipperClassifier, extracts the trained rules alongside the weight for each rule.
    The rules are returned in a list of dictionaries. Each rule is represented by one
    dictionary. Each dictionary is of the format:

    {("feature_name", "comparison_op"): "value"} where comparison_op can be "<", "<=", ">", or ">=".

    The weigth represents the importance of each rule. The feature names are expected to be of the format
    "x_i" where 'i' is zero-indexed (first feature is 'x_0' etc.).
    """
    weights = classifier.estimator_weights_
    raw_rules = classifier.rules_

    rules = [rule.agg_dict for rule in raw_rules]

    return rules, weights


def instantiate_ruleset_rules(
    rules: List[Dict[Tuple[str, str], str]],
    weights: List[float],
    n_features: int,
    feature_limits: List[Tuple[float, float]],
    n_samples: int,
) -> np.ndarray:

    return
