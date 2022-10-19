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

    if weights == []:
        # The error of the classifier is so small that it does not even begin to fit. Just set the weights of all rules
        # to one
        weights = [1] * len(raw_rules)

    rules = [rule.agg_dict for rule in raw_rules]

    return rules, weights


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

    new_samples = np.zeros((n_samples, n_features))
    # go through each feature index and instantiate the rules
    # for feature_i in op_value_per_index:
    for feature_i in range(n_features):
        # keep track of the lower and upper bounds for each feature, by default,
        # the bounds should be the given feature limits
        current_min = feature_limits[feature_i][0]
        current_max = feature_limits[feature_i][1]

        if not feature_i in op_value_per_index:
            # no rules for feature, instantaite between min and max
            new_samples[:, feature_i] = np.random.uniform(
                current_min, current_max, n_samples
            )

            continue

        for (rule_i, (op, value)) in enumerate(op_value_per_index[feature_i]):
            if op in ["<", "<="]:
                # less than
                if value < current_max and value > current_min:
                    current_max = value
            elif op in [">", ">="]:
                # greater than
                if value > current_min and value < current_max:
                    current_min = value
            elif op in ["=", "=="]:
                # equality
                current_min = value
                current_max = value
            else:
                # unkown operator
                print(
                    f"When instantiating rule {rules} got unkown operator {op}. Skipping.."
                )
                pass

        # instantitate features in the samples according to rules
        new_samples[:, feature_i] = np.random.uniform(
            current_min, current_max, n_samples
        )

    return new_samples


def _instantiate_ruleset_rules(
    rules: List[Rules],
    weights: List[float],
    n_features: int,
    feature_limits: List[Tuple[float, float]],
    n_samples: int,
) -> List[np.ndarray]:
    """Helper to 'instantiate_ruleset_rules'. See its description.

    List[np.ndarray]: A list of samples per rule.
    """
    # based on the weights, figure out how many of the samples should be generated based on
    # each rule in the rule set.
    # ignore rules with negative weight

    w_arr = np.array(weights)

    # ignore negative weights
    fractions = w_arr[w_arr >= 0] / np.sum(w_arr[w_arr >= 0])

    n_per_rule = np.round(fractions * n_samples)

    instantiated = []

    rules_pos_w = np.array(rules)[w_arr >= 0]

    for (rule_i, rule) in enumerate(rules_pos_w):
        instantiated.append(
            instantiate_rules(rule, n_features, feature_limits, int(n_per_rule[rule_i]))
        )

    return instantiated


def instantiate_ruleset_rules(
    rules: List[Rules],
    weights: List[float],
    n_features: int,
    feature_limits: List[Tuple[float, float]],
    n_samples: int,
) -> np.ndarray:
    """Instantiate samples according to a rule set. Instantiates in total approximately n_samples
    of new samples according to the rules and features limits. If for some feature there are no rules,
    then only the feature limits are used. The feature limits will override rules if there is a
    conflict. The given weights will dictate how large of a fraction of n_samples will be generated
    for each rule. It is assumed that the rules supplied (in a list) have a weight at the same index
    in the argument weights.

    Args:
        rules (List[Rules]): The rules contained in the rule set. See 'instantiate_rules'.
        weights (List[float]): The weights for each rule in the rule set. It is assumed tht the weight
            at index i corresponds to the weight of rules at index i in 'rules'.
        n_features (int): How many new samples to generate according to the rules. This
            is approximate, but the total of new samples should be relatively close to this number.
        feature_limits (List[Tuple[float, float]]): Pairs representing the lower and upper bounds of
            each feature.
        n_samples (int): Approximately how many new samples to generate in total.

    Returns:
        np.ndarray: A 2D array with all the new generated samples. If a list of samples per rule is desired,
            see '_instantiate_ruleset_rules'.
    """
    instantiated = _instantiate_ruleset_rules(
        rules, weights, n_features, feature_limits, n_samples
    )

    return np.vstack(instantiated)
