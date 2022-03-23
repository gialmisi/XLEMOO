from tkinter.tix import Tree
import numpy as np
import sklearn

from typing import TypedDict, List, Tuple


class TreePath(TypedDict):
    rules: List
    samples: float
    impurity: float
    classification: int


def find_all_paths(tree: "sklearn.tree") -> List[TreePath]:
    paths: List[TreePath] = []

    def traverse_tree(tree, rules: list, node_id: int) -> None:
        # check if current node is a leaf, if true, do not recurse
        if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
            # find classificaiton
            classification = np.argmax(tree.tree_.value[node_id])
            entry = {}
            entry["rules"] = rules
            entry["samples"] = tree.tree_.weighted_n_node_samples[node_id]
            entry["impurity"] = tree.tree_.impurity[node_id]
            entry["classification"] = classification

            entry: TreePath = entry
            paths.append(entry)

        # is root? if root, then new path begins
        elif node_id == 0:
            threshold = tree.tree_.threshold[node_id]
            feature = tree.tree_.feature[node_id]
            rule_left = [feature, "lte", threshold]
            rule_right = [feature, "gt", threshold]

            left_id = tree.tree_.children_left[node_id]
            right_id = tree.tree_.children_right[node_id]

            # go left and right
            left_rules = [rule_left]
            right_rules = [rule_right]

            traverse_tree(tree, left_rules, left_id)
            traverse_tree(tree, right_rules, right_id)

        # we have a node
        else:
            threshold = tree.tree_.threshold[node_id]
            feature = tree.tree_.feature[node_id]
            rule_left = [feature, "lte", threshold]
            rule_right = [feature, "gt", threshold]

            left_id = tree.tree_.children_left[node_id]
            right_id = tree.tree_.children_right[node_id]

            left_rules = rules + [rule_left]
            right_rules = rules + [rule_right]

            traverse_tree(tree, left_rules, left_id)
            traverse_tree(tree, right_rules, right_id)

    traverse_tree(tree, [], 0)

    return paths


def instantiate_tree_rules(
    paths: List[TreePath],
    n_features: int,
    feature_limits: List[Tuple[float, float]],
    n_samples: int,
    desired_classification: int,
) -> np.ndarray:
    # find out how many of the paths have the desired classification
    n_matching_paths = sum(
        [1 if p["classification"] == desired_classification else 0 for p in paths]
    )

    if n_matching_paths == 0:
        # no paths found with desired classification, return empty array
        return np.atleast_3d([])

    # for each path, create an array with n_features and set each element according to the bounds
    limits = np.array(feature_limits)
    instantiated = np.atleast_3d(
        np.random.uniform(
            limits[:, 0], limits[:, 1], (n_matching_paths, n_samples, n_features)
        )
    )

    # for each path with the desired classification, start populating an array of NaN with attributes according to the rules
    path_i = 0
    for p in paths:
        if p["classification"] == desired_classification:
            for rule in p["rules"]:
                # features are 1-indexed, henche, -1
                feature = rule[0] - 1
                comp = rule[1]
                threshold = rule[2]

                instantiated[path_i][:][feature] = (
                    np.random.uniform(threshold, feature_limits[feature][1], n_samples)
                    if comp == "gte"
                    else np.random.uniform(
                        feature_limits[feature][0], threshold, n_samples
                    )
                )

            path_i += 1

    # set the attributes according to the rule and according to the feature limits
    # return the as many arrays required by n_samples (split the distribution according to the weighted samples of each path)
    return instantiated
