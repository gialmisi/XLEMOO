from desdeo_emo.population import Population
from desdeo_tools.scalarization.ASF import ASFBase
from desdeo_tools.utilities.quality_indicator import hypervolume_indicator
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


def hypervolume_contribution(ref_point: np.ndarray) -> np.ndarray:
    # Compute the contribution of each invididual in a population to the hypervolume
    def fun(front: np.ndarray, ref_point: np.ndarray = ref_point):
        # compute the base-line hypervolume
        hv_baseline = hypervolume_indicator(front, ref_point)

        # one at a time, take each objective vector out of the front, compute the
        # hypervolume of the front witout the vector, and subtract the hypervolume
        # from the baseline. The difference will be the contribution of the excluded
        # vector to the hypervolume.
        contributions = np.zeros(front.shape[0])
        mask = np.ones(front.shape[0], dtype=bool)

        for i in range(front.shape[0]):
            # i is the index of the excluded objective vector
            mask[i] = False
            hv_without_i = hypervolume_indicator(front[mask], ref_point)
            contributions[i] = hv_baseline - hv_without_i
            mask[i] = True

        # return the NEGATIVE contributions because internally in XLEMOO, a smaller fitness
        # is better
        return -np.atleast_2d(contributions).T

    return fun
