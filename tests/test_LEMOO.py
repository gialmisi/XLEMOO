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


def test_check_condition_best(toy_model):
    toy_model.reset_generation_history()
    assert len(toy_model._generation_history) == 0

    probe = 2
    threshold = 0.95  # expect new best fitness to be less than 0.95*old_best_fitness
    # for testing, otherwise the best is updated upon initialization
    toy_model._best_fitness_fun_value = np.inf

    # add toy data to generations, only fitness value matters
    # fitness is the sum of the objectives
    n_times = 5

    for i in range(n_times):
        toy_model.add_population_to_history(
            individuals=[np.zeros(3), np.zeros(3), np.zeros(3)],
            objectives_fitnesses=[i * np.ones(2), i * np.ones(2)],
        )

    assert toy_model.check_condition_best(probe, threshold)
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 6.0)

    # consider one more generation
    probe = 3

    assert toy_model.check_condition_best(probe, threshold)
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 4.0)

    # consider all generations
    probe = 99

    assert toy_model.check_condition_best(probe, threshold)
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 0.0)

    # rest best, consider two past generations, test threshold
    toy_model._best_fitness_fun_value = 8.0  # from last generation
    probe = 2
    threshold = 0.3  # expect new best fitness to be 0.3*old_best_fitness

    # condition should not be true
    assert not toy_model.check_condition_best(probe, threshold)

    # best fitness should not change
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 8.0)

    # increase past generations to include the second to last generation, condition should be met
    probe = 4

    # condition should be true
    assert toy_model.check_condition_best(probe, threshold)

    # check correct best fitness
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, 2.0)


def test_run_darwin_only(toy_model):
    # do not use ML mode at all!
    toy_model._lem_params.use_ml = False

    # test early termination
    initial_best = 1.07

    toy_model._lem_params.darwin_probe = 10
    toy_model._lem_params.darwin_threshold = 0.97
    toy_model._best_fitness_fun_value = initial_best

    counters = toy_model.run()

    # the new best solution should be better than the given threshold times initial_best
    assert (
        toy_model._best_fitness_fun_value
        < toy_model._lem_params.darwin_threshold * initial_best
    )

    # reset history
    toy_model.reset_generation_history()
    toy_model.initialize_population()

    assert len(toy_model._generation_history) == 1

    # test forced termination when darwin_probe is reached

    toy_model._lem_params.darwin_probe = 10
    toy_model._lem_params.darwin_threshold = 0.001  # impossible!
    toy_model._best_fitness_fun_value = initial_best

    counters_forced = toy_model.run()

    # should have iterated darwin_probe times
    assert counters_forced["darwin_mode"] == toy_model._lem_params.darwin_probe

    # the current best solution should be worse than given threshold time initial_best
    assert (
        toy_model._best_fitness_fun_value
        >= toy_model._lem_params.darwin_threshold * initial_best
    )


def test_update_best_fitness(toy_model):
    new_individuals = np.array(
        [[0.3, 0.3, 0.3], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]]
    )
    toy_model.update_population(new_individuals)

    assert len(toy_model._population.individuals) == 4

    dummy_best = 9999
    toy_model._best_fitness_fun_value = dummy_best

    # find best, should return True
    assert toy_model.update_best_fitness()

    # check the new best
    # should be less than initial best
    assert toy_model._best_fitness_fun_value < dummy_best

    # should be the fitness value of evaluationg the third objective vector in new_individuals
    actual_objectives = toy_model._problem.evaluate(new_individuals[2]).objectives
    actual_best = toy_model._lem_params.fitness_indicator(actual_objectives)

    npt.assert_almost_equal(toy_model._best_fitness_fun_value, actual_best)


def test_update_best_fitness_no_update(toy_model):
    # the best fitness should be updated only if a better fitness is found in the current population.
    new_individuals = np.array(
        [[0.3, 0.3, 0.3], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7]]
    )
    toy_model.update_population(new_individuals)

    assert len(toy_model._population.individuals) == 4

    dummy_best = 0.001
    toy_model._best_fitness_fun_value = dummy_best

    # find best, should not be able to, return False
    assert not toy_model.update_best_fitness()

    # check that best fitness has not changed
    npt.assert_almost_equal(toy_model._best_fitness_fun_value, dummy_best)


def test_learning_mode(toy_model):
    # do darwin a few times to have a population
    assert len(toy_model._generation_history) == 1

    n_darwin = 3
    for _ in range(n_darwin):
        toy_model.darwinian_mode()
        toy_model.add_population_to_history()

    assert len(toy_model._generation_history) == 1 + n_darwin

    old_individuals = toy_model._population.individuals

    toy_model.learning_mode()

    new_individuals = toy_model._population.individuals

    # population should have changed
    difference = old_individuals - new_individuals
    npt.assert_raises(AssertionError, npt.assert_allclose, difference, 0.0)


def test_run_learning_mode_only(toy_model):
    # do not use ML mode at all!
    toy_model._lem_params.use_darwin = False

    # test early termination
    initial_best = 1.05

    toy_model._lem_params.ml_probe = 10
    toy_model._lem_params.ml_threshold = 0.96
    toy_model._best_fitness_fun_value = initial_best

    counters = toy_model.run()

    # the new best solution should be better than the given threshold times initial_best
    assert (
        toy_model._best_fitness_fun_value
        < toy_model._lem_params.ml_threshold * initial_best
    )

    # reset history
    toy_model.reset_generation_history()
    toy_model.initialize_population()

    assert len(toy_model._generation_history) == 1

    # test forced termination when ml_probe is reached

    toy_model._lem_params.ml_probe = 10
    toy_model._lem_params.ml_threshold = 0.001  # impossible!
    toy_model._best_fitness_fun_value = initial_best

    counters_forced = toy_model.run()

    # should have iterated ml_probe times
    assert counters_forced["learning_mode"] == toy_model._lem_params.ml_probe

    # the current best solution should be worse than given threshold time initial_best
    assert (
        toy_model._best_fitness_fun_value
        >= toy_model._lem_params.ml_threshold * initial_best
    )


def test_run(toy_model):
    toy_model._lem_params.ml_probe = 5
    toy_model._lem_params.darwin_probe = 5
    toy_model._lem_params.ml_threshold = 0.999
    toy_model._lem_params.darwin_threshold = 0.999
    toy_model._lem_params.use_ml = True
    toy_model._lem_params.use_darwin = True

    counters = toy_model.run()

    # both modes should have run for at least their probe amounts once, but actually more
    assert counters["learning_mode"] > toy_model._lem_params.ml_probe
    assert counters["darwin_mode"] > toy_model._lem_params.darwin_probe

    # the best fitness value should have changed
    assert toy_model._best_fitness_fun_value < np.inf
