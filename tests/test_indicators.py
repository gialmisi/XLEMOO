from XLEMOO.fitness_indicators import (
    asf_wrapper,
    naive_sum,
    hypervolume_contribution,
    inside_ranges,
)
from desdeo_tools.scalarization import SimpleASF, GuessASF
from desdeo_tools.utilities import hypervolume_indicator
from desdeo_problem.testproblems import test_problem_builder as problem_builder
from desdeo_emo.population import Population
import pytest
import numpy as np
import numpy.testing as npt


@pytest.fixture
def dummy_population():
    n_problem_variables = 5
    n_problem_objectives = 3
    problem = problem_builder("DTLZ2", n_problem_variables, n_problem_objectives)
    pop = Population(problem, 100)
    return pop


@pytest.mark.fitness
def test_asf_wrapper(dummy_population):
    n_objectives = dummy_population.problem.n_of_objectives
    asf = GuessASF(np.ones(n_objectives))
    fitness_f = asf_wrapper(
        asf,
        {"reference_point": np.zeros(dummy_population.problem.n_of_objectives)},
    )

    fitness = fitness_f(dummy_population.fitness)
    naive = naive_sum(dummy_population.fitness)

    # check shape
    assert fitness.shape == naive.shape


@pytest.mark.fitness
def test_hypervolume_contribution():

    front = np.array(
        [
            # non-dominated points
            [2.0, 5.0, 4.0],
            [3.0, -1.5, 3.5],
            [2.5, -0.5, 0.75],
            # dominated point
            [3.4, 5.5, 4.2],
        ]
    )

    nadir = np.array([8.0, 8.0, 8.0])

    hv_indicator = hypervolume_contribution(nadir)

    contributions = hv_indicator(front)

    baseline_hv = hypervolume_indicator(front, nadir)

    npt.assert_almost_equal(contributions[3], 0)

    # minus sign because hypervolume_contribution returns negative values
    assert np.all(contributions > -baseline_hv)


@pytest.mark.fitness
def test_inside_ranges():
    lower_limits = np.array([0.5, -1.5, 0.2])
    upper_limits = np.array([2.5, 1.5, 0.8])

    front = np.array(
        [
            # inside
            [1.2, 0.9, 0.6],  # -> 0 good!
            # first outside
            [3.0, 1.2, 0.7],  # -> 0.5
            # second outside
            [1.3, -2.0, 0.3],  # -> 0.5
            # third outside
            [2.1, -1.3, 1.0],  # -> 0.2
            # all breach
            [-1.0, 3.0, 1.2],  # -> 1.5 + 1.5 + 0.4 = 3.4
        ]
    )

    indicator_f = inside_ranges(lower_limits, upper_limits, sim_cost=0)

    breaches = indicator_f(front)

    npt.assert_almost_equal(breaches, np.atleast_2d([0, 0.5, 0.5, 0.2, 3.4]).T)


@pytest.mark.fitness
def test_inside_ranges_with_sim():
    lower_limits = np.array([0.5, -1.5, 0.2])
    upper_limits = np.array([2.5, 1.5, 0.8])

    front = np.array(
        [
            # same as last, should be penalized
            [1.2, 0.9, 0.6],  # -> 0.1, same as last!
            # first outside
            [3.0, 1.2, 0.7],  # -> 0.5
            # second outside
            [1.3, -2.0, 0.3],  # -> 0.5
            # third outside
            [2.1, -1.3, 1.0],  # -> 0.2
            # all breach
            [-1.0, 3.0, 1.2],  # -> 1.5 + 1.5 + 0.4 = 3.4
            # same as first, should be penalized
            [1.2, 0.9, 0.6],  # -> 0.1, same as last!
            # three times the same guy, very close
            [1.2, 1.0000001, 0.7],  # -> 0.2
            [1.2, 1.0, 0.7],  # -> 0.2
            [1.1999999999, 1.0, 0.70000001],  # -> 0.2
            [1.2001, 0.9998, 0.6999],  # -> no penalty!
        ]
    )

    indicator_f = inside_ranges(lower_limits, upper_limits, sim_cost=0.1)

    breaches = indicator_f(front)

    npt.assert_almost_equal(
        breaches, np.atleast_2d([0.1, 0.5, 0.5, 0.2, 3.4, 0.1, 0.2, 0.2, 0.2, 0]).T
    )
