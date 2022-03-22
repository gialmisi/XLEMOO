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
    rules: List[TreePath],
    n_features: int,
    feature_limits: List[Tuple[float, float]],
    n_samples: int,
) -> np.ndarray:
    return np.zeros(3)
