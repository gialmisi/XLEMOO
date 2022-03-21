import pytest
import numpy as np
import numpy.testing as npt
from XLEMOO.LEMOO import (
    LEMOO,
    CrossOverOP,
    MutationOP,
    SelectionOP,
    EAParams,
    MLParams,
    LEMParams,
)
from XLEMOO.fitness_indicators import naive_sum, single_objective
from desdeo_emo.recombination import BP_mutation, SBX_xover
from desdeo_emo.selection import TournamentSelection

# needs to be renamed, otherwise pytest thinks it is a test to be run
from desdeo_problem.testproblems import test_problem_builder as problem_builder
from imodels import C45TreeClassifier


def test_subclasses():
    assert issubclass(SBX_xover, CrossOverOP)
    assert issubclass(BP_mutation, MutationOP)
    assert issubclass(TournamentSelection, SelectionOP)


def test_init():
    problem = problem_builder("DTLZ2", 5, 3)
    xover_op = SBX_xover()
    mutation_op = BP_mutation(
        problem.get_variable_lower_bounds(), problem.get_variable_upper_bounds()
    )
    selection_op = TournamentSelection(None, tournament_size=2)

    lem_params = LEMParams(10, 20, 1, True, True, naive_sum)
    ea_params = EAParams(50, xover_op, mutation_op, selection_op, "RandomDesign")
    ml_params = MLParams(0.3, 0.3, C45TreeClassifier(), single_objective)

    lemoo = LEMOO(problem, lem_params, ea_params, ml_params)

    return lemoo


@pytest.fixture
def toy_model():
    return test_init()


def test_update_population(toy_model):
    # test the updating of the population
    lower_bounds = toy_model._population.problem.get_variable_lower_bounds()
    pop_size = toy_model._population.pop_size

    new_individuals = np.random.rand(pop_size, len(lower_bounds))

    # current individuals in population should differ from new ones
    npt.assert_raises(
        AssertionError,
        npt.assert_allclose,
        toy_model._population.individuals,
        new_individuals,
    )

    # after updating the population, the individuals in the population should be the same as in new_individuals
    toy_model.update_population(new_individuals)
    npt.assert_allclose(toy_model._population.individuals, new_individuals)


def test_darwin_mode(toy_model):
    # test that the darwin step returns a new population that differs from the previous one
    old_individuals = toy_model._population.individuals

    new_individuals = toy_model.darwinian_mode()

    npt.assert_raises(
        AssertionError, npt.assert_allclose, old_individuals, new_individuals
    )


def test_learning_mode(toy_model):
    # test that the learning step return a new population that differs from the previous one
    old_individuals = toy_model._population.individuals

    new_individuals = toy_model.learning_mode()

    npt.assert_raises(
        AssertionError, npt.assert_allclose, old_individuals, new_individuals
    )
