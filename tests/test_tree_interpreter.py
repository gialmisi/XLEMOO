from multiprocessing import dummy
from XLEMOO.tree_interpreter import TreePath, find_all_paths, instantiate_tree_rules
import pytest

from sklearn.datasets import load_iris
from sklearn import tree
import sklearn
import numpy as np
from typing import List


@pytest.fixture
def dummy_rules() -> List[TreePath]:
    path_1 = TreePath(
        rules=[[1, "gte", 0.6]], impurity=0.5, samples=33.0, classification=0
    )

    return [path_1]


@pytest.fixture
def dummy_tree() -> "sklearn.tree":
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    clf = clf.fit(X, y)

    return clf


@pytest.mark.tree
def test_find_all_paths(dummy_tree):
    paths = find_all_paths(dummy_tree)

    assert len(paths) == 3

    for path in paths:
        assert "rules" in path
        assert "impurity" in path
        assert "samples" in path
        assert "classification" in path


@pytest.mark.tree
def test_instantiate_rules(dummy_rules):
    assert True
