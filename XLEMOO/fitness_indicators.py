from desdeo_emo.population import Population
from desdeo_tools.scalarization.ASF import ASFBase
from desdeo_tools.utilities.quality_indicator import hypervolume_indicator
import numpy as np
from typing import Callable, Optional


def naive_sum(
    objectives_fitnesses: np.ndarray, variables: Optional[np.ndarray] = None
) -> np.ndarray:
    """An indicator that computes the sum of the objective function values in each objective vector provided.

    Args:
        objectives_fitnesses (np.ndarray): A 2D vector with each row, and objective vector, representing the objective values of
            an objective function. Minimization assumed for each objective.
        variables (Optional[np.ndarray], optional): A 2D array of the decision variable value
            corresponding to the objective vectors in ``objective_fitness``. Not used. Defaults to None.

    Returns:
        np.ndarray: A 2D array with singleton entries, where each entry is the sum of the
        objective function value in each objective vector. The lower the value of the sum, the
        better the fitness.
    """
    return np.atleast_2d(np.sum(objectives_fitnesses, axis=1)).T


def asf_wrapper(asf: ASFBase, asf_kwargs: dict) -> Callable[[np.ndarray, [Optional[np.ndarray]]], np.ndarray]:
    """Wraps a scalarization function into a callable function that accepts a 2D array of objective
    function values and a 2D array of decision variable values. This callable can then be used as an
    indicator. The lower the scalarized value is, the better.

    Args:
        asf (ASFBase): A scalarization function.
        asf_kwargs (dict): The keyword arguments passed to the scalarization function when it is evaluated.

    Returns:
        Callable[[np.ndarray, [Optional[np.ndarray]]], np.ndarray]: A function that accepts a 2D array of objective
        vectors, and optionally a 2D array of the corresponding decision variable vectors, and computes the scalarized 
        values of the objective vectors returning them in a 2D array of singleton values where each value is the
        result of the scalarization.
    """
    def fun(objectives_fitnesses: np.ndarray, variables: Optional[np.ndarray] = None) -> np.ndarray:
        asf_values = asf(objectives_fitnesses, **asf_kwargs)
        return np.atleast_2d(asf_values).T

    return fun


def hypervolume_contribution(
    ref_point: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Creates a function based on the hypervolume. This function accepts a 2D array
    of objective vectors, and optionally a 2D array of decision variable vectors as well.
    This function when computes the individual contribution of each objective vector
    to the cumulative hypervolume of all the vectors. This function may then be used
    as an indicator. The decision variable values are not currently used.

    Args:
        ref_point (np.ndarray): The reference point used to compute the hypervolume.

    Returns:
        Callable[[np.ndarray, np.ndarray], np.ndarray]: A hypervolume indicator
        function that computes the contribution of each provided objective vector
        to the cumulative hypervolume of all the vectors. The lower the value,
        the better.
    """
    # Compute the contribution of each individual in a population to the hypervolume
    def fun(
        front: np.ndarray,
        variables: Optional[np.ndarray] = None,
        ref_point: np.ndarray = ref_point,
    ) -> np.ndarray:
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


def inside_ranges(
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    sim_cost: float,
    asf_fun: Callable[[np.ndarray], np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Creates a function that computes a fitness value for objective vectors based on whether the values
    conform to given bounds. As an optional second argument, this function accepts a 2D array
    of decision variable vectors. Additionally, a similarity penalty can also be included where an objective vector
    is penalized if it is similar to another, and a scalarization contribution may also be added.
    The decision variable vectors are not used currently.

    Args:
        lower_limits (np.ndarray): An array with the lower bounds of the objective function values.
        upper_limits (np.ndarray): An array with the upper bounds of the objective function values.
        sim_cost (float): The similarity penalty.
        asf_fun (Callable[[np.ndarray], np.ndarray], optional): A wrapped scalarization function that
            is used to compute the scalarization contribution for each objective vector.
            See :func:`asf_wrapper`. Defaults to None.

    Raises:
        ValueError: The shape of the ``lower_limits`` and ``upper_limits`` mismatch.

    Returns:
        Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]: A function that
        computes the fitness values of objective vectors based on whether the values
        are withing defined lower and upper bounds penalized by a similarity cost.
        Additional contributions may be calculated with a scalarization function.
    """
    if lower_limits.shape != upper_limits.shape:
        raise ValueError(f"The shapes of lower_limits and upper_limits must match!")

    def fun(
        objective_vectors: np.ndarray,
        variables: Optional[np.ndarray] = None,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        sim_cost=sim_cost,
    ) -> np.ndarray:
        # lower = np.where(a - lower_limits > 0, 0, np.abs(a - lower_limits))
        lower_breach = np.where(
            objective_vectors - lower_limits > 0,
            0,
            np.abs(objective_vectors - lower_limits),
        )
        upper_breach = np.where(
            upper_limits - objective_vectors > 0,
            0,
            np.abs(upper_limits - objective_vectors),
        )

        if objective_vectors.shape[0] == 1:
            print(upper_breach)
            print(lower_breach)

        sum_of_breaches = np.sum(lower_breach + upper_breach, axis=1)

        if asf_fun is not None:
            # add asf contribution
            sum_of_breaches = np.where(
                np.isclose(sum_of_breaches, 0),
                sum_of_breaches + asf_fun(objective_vectors).T,
                sum_of_breaches,
            )

        if sim_cost > 0:
            # add similarity cost
            minus_each = objective_vectors - objective_vectors[:, None]
            sums_of_differences = np.sum(minus_each, axis=2)
            count_of_similars = np.count_nonzero(
                np.isclose(sums_of_differences, 0, atol=1e-6), axis=0
            )

            similarity_mask = count_of_similars > 1

            similarity_penalties = np.where(
                similarity_mask, sim_cost * (count_of_similars - 1), 0
            )

            sum_of_breaches += similarity_penalties

        return np.atleast_2d(sum_of_breaches).T

    return fun
