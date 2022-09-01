from typing import List, Dict, Tuple
from imodels import SlipperClassifier
import numpy as np

Rules = Dict[Tuple[str, str], str]


def extract_slipper_rules(
    classifier: SlipperClassifier,
) -> Tuple[List[Rules], List[float]]:
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
    rules: List[Rules],
    weights: List[float],
    n_features: int,
    feature_limits: List[Tuple[float, float]],
    n_samples: int,
) -> np.ndarray:

    return


def instantiate_rules(
    rules: Rules,
    n_features: int,
    feature_limits: List[Tuple[float, float]],
    n_samples: int,
) -> np.ndarray:
    """Takes Rules and instantiates them producing n_samples of new decison variable vectors corresponding
    to the rules. If there are no rules for a variable, a random value is generated for that variable
    between its limits. Notice that when rules define a range for a variable, then that variable's
    value will be generated between those ranges randomly.

    Args:
        rules (Rules): Should be a dict with the following structure:
            {("feature_name", "comparison_op"): "value"} where comparison_op can be "<", "<=", ">", or ">=".
            The feature names are expected to be formatted as "x_i" where 'i' is zero indexed (i.e., x_0,
            x_1, x_2, etc.).
        n_features (int): Number of features to instantiate based on the rules provided.
        feature_limits (List[Tuple[float, float]]): 2D array, each row corresponds to a decision variable.
            The first column has the lower limits for each variable and the second the upper limit.
        n_samples (int): How many samples to generate based on the rules provided.

    Returns:
        np.ndarray: The new samples generated based on the provided rules in a 2D array.
    """
    # collect generated samples
    samples = []

    # collect each rule in tupe of threes and put them in a list
    # cast the indices to int and limits to float
    index_op_value = list(
        map(
            lambda key: (int(key[0].split("_")[-1]), key[1], float(rules[key])),
            rules.keys(),
        )
    )

    # group each rule according to the feature index
    op_value_per_index = {}
    for i, op, val in index_op_value:
        if i in op_value_per_index:
            op_value_per_index[i].append((op, val))
        else:
            op_value_per_index[i] = [(op, val)]

    # check and eliminate redundancies
    for i in op_value_per_index:
        current_max = feature_limits[i][0]
        current_min = feature_limits[i][1]

        for (j, op, value) in enumerate(op_value_per_index[i]):
            if value < feature_limits[i][0] or value > feature_limits[i][1]:
                # value is out of bounds, drop the rule
                del op_value_per_index[i][j]

            elif op in ["<", "<="]:
                # less than rule
                pass
            elif op in [">", ">="]:
                # greater than rule
                pass

    pass
