from desdeo_emo.population import Population
from desdeo_tools.scalarization.ASF import ASFBase
import numpy as np


def naive_sum(objectives_fitnesses: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.sum(objectives_fitnesses, axis=1)).T


def dist_to_ideal(objectives_fitnesses: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    # TODO: fix me, ideal should have a default value if not given
    return np.atleast_2d(np.linalg.norm(objectives_fitnesses - ideal, axis=1)).T


def must_sum_to_one(objectives_fitnesses: np.ndarray) -> np.ndarray:
    sums = np.sum(objectives_fitnesses**2, axis=1)
    return np.atleast_2d(abs(sums - 1)).T


def single_objective(objective_fitnesses: np.ndarray, obj_index: int = 0) -> np.ndarray:
    return np.atleast_2d(objectives_fitnesses[:, obj_index]).T


def asf_wrapper(asf: ASFBase, asf_kwargs: dict) -> np.ndarray:
    def fun(objectives_fitnesses: np.ndarray):
        asf_values = asf(objectives_fitnesses, **asf_kwargs)
        return np.atleast_2d(asf_values).T

    return fun
