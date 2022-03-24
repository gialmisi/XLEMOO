from XLEMOO.fitness_indicators import asf_wrapper, naive_sum
from desdeo_tools.scalarization import SimpleASF, GuessASF
from desdeo_problem.testproblems import test_problem_builder as problem_builder
from desdeo_emo.population import Population
import pytest
import numpy as np


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

    fitness = fitness_f(dummy_population)
    naive = naive_sum(dummy_population)

    # check shape
    assert fitness.shape == naive.shape
