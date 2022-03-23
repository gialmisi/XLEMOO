from multiprocessing import dummy
from XLEMOO.tree_interpreter import TreePath, find_all_paths, instantiate_tree_rules
import pytest

from sklearn.datasets import load_iris
from sklearn import tree
import sklearn
import numpy as np
import numpy.testing as npt
from typing import List, Tuple


@pytest.fixture
def dummy_paths_and_limits() -> Tuple[List[TreePath], List[Tuple[float, float]]]:
    path_1 = TreePath(
        rules=[[1, "gte", 0.6]], impurity=0.5, samples=33.0, classification=0
    )
    path_2 = TreePath(
        rules=[[1, "lt", 0.6], [2, "gte", 1.5]],
        impurity=0.33,
        samples=15.0,
        classification=1,
    )
    path_3 = TreePath(
        rules=[[1, "lt", 0.6], [2, "lt", 1.5], [3, "gte", 3.3]],
        impurity=0.5,
        samples=22.0,
        classification=2,
    )

    limits = [(0, 2), (1, 3), (2, 4)]

    return [path_1, path_2, path_2, path_3], limits


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
def test_instantiate_rules(dummy_paths_and_limits):
    dummy_paths, limits = dummy_paths_and_limits
    n_samples = 100

    # check that empty array returned when no matching classification is found
    res = instantiate_tree_rules(dummy_paths, len(limits), limits, n_samples, 10)
    assert res.shape == (1, 0, 1)

    # matching path is found
    res = instantiate_tree_rules(dummy_paths, len(limits), limits, n_samples, 1)

    assert res.shape[0] == 2
    assert res.shape[1] == n_samples
    assert res.shape[2] == len(limits)

    assert all(res[0, :, 0] < 0.6)
    assert all(res[0, :, 1] >= 1.5)

    # check with just one rule per classification
    res = instantiate_tree_rules(dummy_paths, len(limits), limits, n_samples, 2)
    assert res.shape[0] == 1
    assert res.shape[1] == n_samples
    assert res.shape[2] == len(limits)

    assert all(res[0, :, 0] < 0.6)
    assert all(res[0, :, 1] < 1.5)
    assert all(res[0, :, 2] >= 3.3)
