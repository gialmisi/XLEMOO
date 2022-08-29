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


@pytest.mark.lemoo
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

    lem_params = LEMParams(2, 50, 1, True, True, naive_sum, 10, 10, 1.05, 1.05)
    ea_params = EAParams(50, xover_op, mutation_op, selection_op, "RandomDesign")
    ml_params = MLParams(0.3, 0.3, DecisionTreeClassifier(), naive_sum)

    lemoo = LEMOO(problem, lem_params, ea_params, ml_params)

    return lemoo


@pytest.fixture
def toy_model():
    return test_init()


@pytest.mark.lemoo
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


@pytest.mark.lemoo
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
    assert False


def test_run(toy_model):
    assert False


@pytest.mark.lemoo
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
        toy_model._generation_history[0].individuals,
        toy_model._generation_history[1].individuals,
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


@pytest.mark.lemoo
def test_add_population_to_history_manually(toy_model):
    toy_model.reset_generation_history()
    assert len(toy_model._generation_history) == 0

    # create dummy data and add it to the history
    individuals = np.array([np.ones(3), 2 * np.ones(3), 3 * np.ones(3)])
    objective_fitnesses = np.array([np.ones(2), 2 * np.ones(2), 3 * np.ones(2)])

    # add these to the history a bunch of times
    n_times = 10
    for _ in range(n_times):
        toy_model.add_population_to_history(
            individuals=individuals, objectives_fitnesses=objective_fitnesses
        )

    # check correct length of history
    assert len(toy_model._generation_history) == n_times


@pytest.mark.lemoo
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


@pytest.mark.lemoo
def test_collect_population(toy_model):
    toy_model.reset_generation_history()
    assert len(toy_model._generation_history) == 0

    # create dummy data and add it to the history
    individuals = np.array([np.ones(3), 2 * np.ones(3), 3 * np.ones(3)])
    objectives_fitnesses = np.array([np.ones(2), 2 * np.ones(2), 3 * np.ones(2)])

    # add these to the history a bunch of times
    n_times = 10
    for _ in range(n_times):
        toy_model.add_population_to_history(
            individuals=individuals, objectives_fitnesses=objectives_fitnesses
        )

    # check correct length of history
    assert len(toy_model._generation_history) == n_times

    # test getting past generations
    past_n = 5
    (
        individuals_5,
        objectives_fitnessess_5,
        fitness_fun_values_5,
    ) = toy_model.collect_n_past_generations(past_n)

    # check lengths
    assert individuals_5.shape[0] == individuals.shape[0] * past_n
    assert objectives_fitnessess_5.shape[0] == objectives_fitnesses.shape[0] * past_n
    assert (
        fitness_fun_values_5.shape[0] == objectives_fitnesses.shape[0] * past_n
    )  # one fitness value for each objective function

    # check dims
    assert individuals_5.shape[1] == 3
    assert objectives_fitnessess_5.shape[1] == 2
    assert fitness_fun_values_5.shape[1] == 1

    ## test getting past generations when n is larger than history
    past_n = 15
    (
        individuals_15,
        objectives_fitnessess_15,
        fitness_fun_values_15,
    ) = toy_model.collect_n_past_generations(past_n)

    # check lengths
    assert individuals_15.shape[0] == individuals.shape[0] * n_times
    assert objectives_fitnessess_15.shape[0] == objectives_fitnesses.shape[0] * n_times
    assert (
        fitness_fun_values_15.shape[0] == objectives_fitnesses.shape[0] * n_times
    )  # one fitness value for each objective function

    # check dims
    assert individuals_15.shape[1] == 3
    assert objectives_fitnessess_15.shape[1] == 2
    assert fitness_fun_values_15.shape[1] == 1

    ## test getting past generations when n is one
    past_n = 1
    (
        individuals_1,
        objectives_fitnessess_1,
        fitness_fun_values_1,
    ) = toy_model.collect_n_past_generations(past_n)

    # check lengths
    assert individuals_1.shape[0] == individuals.shape[0] * past_n
    assert objectives_fitnessess_1.shape[0] == objectives_fitnesses.shape[0] * past_n
    assert (
        fitness_fun_values_1.shape[0] == objectives_fitnesses.shape[0] * past_n
    )  # one fitness value for each objective function

    # check dims
    assert individuals_1.shape[1] == 3
    assert objectives_fitnessess_1.shape[1] == 2
    assert fitness_fun_values_1.shape[1] == 1


def test_check_darwing_condiiton_best(toy_model):
    toy_model.reset_generation_history()
    assert len(toy_model._generation_history) == 0

    toy_model._lem_params.darwin_probe = 2
    toy_model._lem_params.darwin_threshold = (
        0.95  # expect new best fitness to be less than 0.95*old_best_fitness
    )

    # add toy data to generations, only fitness value matters
    # fitness is the sum of the objectives
    n_times = 5

    for i in range(n_times):
        toy_model.add_population_to_history(
            individuals=[np.zeros(3), np.zeros(3), np.zeros(3)],
            objectives_fitnesses=[i * np.ones(2), i * np.ones(2)],
        )

    assert toy_model.check_darwin_condition_best()
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 6.0)

    # consider one more generation
    toy_model._lem_params.darwin_probe = 3

    assert toy_model.check_darwin_condition_best()
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 4.0)

    # consider all generations
    toy_model._lem_params.darwin_probe = 99

    assert toy_model.check_darwin_condition_best()
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 0.0)

    # rest best, consider two past generations, test threshold
    toy_model._best_fitness_fun_value = 8.0  # from last generation
    toy_model._lem_params.darwin_probe = 2
    toy_model._lem_params.darwin_threshold = (
        0.3  # expect new best fitness to be 0.3*old_best_fitness
    )

    # condition should not be true
    assert not toy_model.check_darwin_condition_best()

    # best fitness should not change
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 8.0)

    # increase past generations to include the second to last generation, condition should be met
    toy_model._lem_params.darwin_probe = 4

    # condition should be true
    assert toy_model.check_darwin_condition_best()

    # check correct best fitness
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 2.0)
