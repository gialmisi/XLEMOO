from XLEMOO.fitness_indicators import asf_wrapper, naive_sum, hypervolume_contribution
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
