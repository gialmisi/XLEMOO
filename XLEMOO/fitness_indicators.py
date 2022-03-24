from desdeo_emo.population import Population
from desdeo_tools.scalarization.ASF import ASFBase
import numpy as np


def naive_sum(population: Population) -> np.ndarray:
    objective_values = population.problem.evaluate(population.individuals).objectives
    return np.atleast_2d(np.sum(objective_values, axis=1)).T


def dist_to_ideal(population: Population, ideal: np.ndarray) -> np.ndarray:
    # TODO: fix me, ideal should have a default value if not given
    objective_values = population.problem.evaluate(population.individuals).objectives
    return np.atleast_2d(np.linalg.norm(objective_values - ideal, axis=1)).T


def must_sum_to_one(population: Population) -> np.ndarray:
    objective_values = population.problem.evaluate(population.individuals).objectives
    sums = np.sum(objective_values**2, axis=1)
    return np.atleast_2d(abs(sums - 1)).T


def single_objective(population: Population, obj_index: int = 0) -> np.ndarray:
    objective_values = population.problem.evaluate(population.individuals).objectives
    return np.atleast_2d(objective_values[:, obj_index]).T


def asf_wrapper(asf: ASFBase, asf_kwargs: dict) -> np.ndarray:
    def fun(population):
        objective_values = population.problem.evaluate(
            population.individuals
        ).objectives
        asf_values = asf(objective_values, **asf_kwargs)
        return np.atleast_2d(asf_values).T

    return fun
