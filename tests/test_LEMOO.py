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
from sklearn.tree import DecisionTreeClassifier

# needs to be renamed, otherwise pytest thinks it is a test to be run
from desdeo_problem.testproblems import test_problem_builder as problem_builder


class SpoofML:
    """Does absolutely nothing, do not trust!"""

    def fit(self, X: np.ndarray, Y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0])


def test_subclasses():
    assert issubclass(SBX_xover, CrossOverOP)
    assert issubclass(BP_mutation, MutationOP)
    assert issubclass(TournamentSelection, SelectionOP)


def test_init():
    problem = problem_builder("DTLZ2", 3, 2)
    xover_op = SBX_xover()
    mutation_op = BP_mutation(
        problem.get_variable_lower_bounds(), problem.get_variable_upper_bounds()
    )
    selection_op = TournamentSelection(None, tournament_size=2)

    lem_params = LEMParams(2, 50, 1, True, True, naive_sum, past_gens_to_consider=1)
    ea_params = EAParams(50, xover_op, mutation_op, selection_op, "RandomDesign")
    ml_params = MLParams(0.3, 0.3, DecisionTreeClassifier(), naive_sum)

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

    toy_model.darwinian_mode()

    new_individuals = toy_model._population.individuals

    # Raises AssertionError if: old and new individuals differ in size, or when
    # they are of the same size, but have different individual values.
    # If nothing is raised (the test fails), it means that the two
    # poulations have exactly the same size and individuals, thus, the
    # test fails as expected.
    npt.assert_raises(
        AssertionError, npt.assert_allclose, old_individuals, new_individuals
    )

    assert len(new_individuals.shape) == 2


def test_learning_mode(toy_model):
    # test that the learning step return a new population that differs from the previous one
    toy_model._generation_history = [toy_model._population]
    old_individuals = toy_model._population.individuals

    new_individuals = toy_model.learning_mode()

    npt.assert_raises(
        AssertionError, npt.assert_allclose, old_individuals, new_individuals
    )

    assert len(new_individuals.shape) == 2
    assert new_individuals.shape[0] == toy_model._ea_params.population_size


def test_run(toy_model):
    # history should be empty
    assert len(toy_model._population_history) == 0

    # run
    history = toy_model.run()

    # 1 + is initial iteration
    should_be = 1 + (
        toy_model._lem_params.n_total_iterations
        * (
            toy_model._lem_params.n_ea_gen_per_iter
            + toy_model._lem_params.n_ml_gen_per_iter
        )
        + toy_model._lem_params.n_ea_gen_per_iter
    )

    assert should_be == len(history)


def test_add_populatin_to_history(toy_model):
    # check that the initial population was added to history
    assert len(toy_model._generation_history) == 1

    # do evolution once and update the history
    toy_model.darwinian_mode()
    toy_model.add_population_to_history()

    # check history
    assert len(toy_model._generation_history) == 2

    ## check that the added generation differs from the previous one
    # individuals
    npt.assert_raises(
        AssertionError,
        npt.assert_allclose,
        toy_model._generation_history[0].individulas,
        toy_model._generation_history[1].individulas,
    )
    # objective fitness
    npt.assert_raises(
        AssertionError,
        npt.assert_allclose,
        toy_model._generation_history[0].objectives_fitnesses,
        toy_model._generation_history[1].objectives_fitnesses,
    )

    # fitness fun values
    npt.assert_raises(
        AssertionError,
        npt.assert_allclose,
        toy_model._generation_history[0].fitness_fun_values,
        toy_model._generation_history[1].fitness_fun_values,
    )


def test_reset_population_history(toy_model):
    # check that the initial population was added to history
    assert len(toy_model._generation_history) == 1

    # do evolution a few times and update the history
    n = 4
    for i in range(4):
        toy_model.darwinian_mode()
        toy_model.add_population_to_history()

    # check history is populated
    assert len(toy_model._generation_history) == 5

    # reset history
    toy_model.reset_generation_history()

    # check history is empty
    assert len(toy_model._generation_history) == 0
