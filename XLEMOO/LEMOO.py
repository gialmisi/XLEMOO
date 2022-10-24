import copy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Union, Dict

from desdeo_emo.population import (
    SurrogatePopulation,
    create_new_individuals,
    Population,
)
from desdeo_emo.recombination import SBX_xover, BP_mutation
from desdeo_emo.selection import TournamentSelection
from desdeo_problem.problem import MOProblem

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from imodels import SlipperClassifier, BoostedRulesClassifier

from XLEMOO.tree_interpreter import find_all_paths, instantiate_tree_rules
from XLEMOO.ruleset_interpreter import extract_slipper_rules, instantiate_ruleset_rules
from XLEMOO.selection import SelectNBest


class CrossOverOP(ABC):
    @abstractmethod
    def do(self, population: Population, mating_pop_ids: List[int]) -> np.ndarray:
        pass


CrossOverOP.register(SBX_xover)


class MutationOP(ABC):
    @abstractmethod
    def do(self, offsprings: np.ndarray) -> np.ndarray:
        pass


MutationOP.register(BP_mutation)


class SelectionOP(ABC):
    @abstractmethod
    def __init__(self, pop, tournament_size):
        pass

    @abstractmethod
    def do(self, pop: Population, fitness: np.ndarray) -> List[int]:
        pass


SelectionOP.register(TournamentSelection)
SelectionOP.register(SelectNBest)


class ElitismOP(ABC):
    @abstractmethod
    def do(
        self,
        pop1: Population,
        pop1_fitness: np.ndarray,
        pop2: Population,
        pop2_fitness: np.ndarray,
    ) -> np.ndarray:
        pass


class MLModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


# Register supported ML models here
MLModel.register(DecisionTreeClassifier)
MLModel.register(SlipperClassifier)
MLModel.register(BoostedRulesClassifier)


class DummyPopulation:
    def __init__(self, individuals: np.ndarray, problem: MOProblem):
        self.individuals = individuals
        self.problem = problem


@dataclass
class EAParams:
    population_size: int
    cross_over_op: CrossOverOP
    mutation_op: MutationOP
    selection_op: SelectionOP
    population_init_design: str
    iterations_per_cycle: int


@dataclass
class MLParams:
    H_split: float
    L_split: float
    ml_model: MLModel
    instantation_factor: float
    iterations_per_cycle: int


@dataclass
class LEMParams:
    use_ml: bool
    use_darwin: bool
    fitness_indicator: Callable[[np.ndarray], np.ndarray]
    ml_probe: int
    darwin_probe: int
    ml_threshold: float
    darwin_threshold: float
    total_iterations: int


@dataclass
class PastGeneration:
    """A past generation with the individuals (decison space) and their
    corresponsing objective fitness values and fitness function values.

    """

    individuals: np.ndarray
    objectives_fitnesses: np.ndarray
    fitness_fun_values: np.ndarray


class LEMOO:
    def __init__(
        self,
        problem: MOProblem,
        lem_params: LEMParams,
        ea_params: EAParams,
        ml_params: MLParams,
    ):
        self._problem: MOProblem = problem
        self._lem_params: LEMParams = lem_params
        self._ea_params: EAParams = ea_params
        self._ml_params: MLParams = ml_params
        self.current_ml_model: MLModel = ml_params.ml_model
        self._population: Union[None, SurrogatePopulation] = None
        self._best_fitness_fun_value: float = np.inf  # lower is better!

        # initialize the population and the evolutionary operators
        self._generation_history: List[PastGeneration] = []
        self.initialize_population()

    def initialize_population(self) -> None:
        """Use the defined initialization design to init a new population, add the initial population to the history of populations."""
        initial_population = create_new_individuals(
            self._ea_params.population_init_design,
            self._problem,
            pop_size=self._ea_params.population_size,
        )

        self._population = SurrogatePopulation(
            self._problem,
            self._ea_params.population_size,
            initial_population,
            self._ea_params.cross_over_op,
            self._ea_params.mutation_op,
            None,
        )

        self.add_population_to_history()
        self.update_best_fitness()

        return

    def reset_population(self) -> None:
        """Reset the current population. Do not add the new population to history."""
        initial_population = create_new_individuals(
            self._ea_params.population_init_design,
            self._problem,
            pop_size=self._ea_params.population_size,
        )

        self._population = SurrogatePopulation(
            self._problem,
            self._ea_params.population_size,
            initial_population,
            self._ea_params.cross_over_op,
            self._ea_params.mutation_op,
            None,
        )

        return

    def reset_generation_history(self) -> None:
        """Reset the population history."""
        self._generation_history = []

    def update_best_fitness(self) -> bool:
        """Find the best fitness value in the current population and update the value stored in
        self._best_fitness_fun_value if the found fitness is better.

        Returns:
            bool: True if the best fitness was updated, otherwise False.
        """
        fitness_fun_values = self._lem_params.fitness_indicator(self._population.fitness)
        min_value = np.min(fitness_fun_values)

        if min_value < self._best_fitness_fun_value:
            self._best_fitness_fun_value = min_value

            return True

        else:
            return False

    def add_population_to_history(
        self,
        individuals: Optional[np.ndarray] = None,
        objectives_fitnesses: Optional[np.ndarray] = None,
    ) -> None:
        if individuals is not None and objectives_fitnesses is not None:
            # add current population to history
            fitness_fun_values = self._lem_params.fitness_indicator(objectives_fitnesses)
            gen = PastGeneration(individuals, objectives_fitnesses, fitness_fun_values)
            self._generation_history.append(gen)

            return

        else:
            # add supplied individuals and fitnesses to history
            fitness_fun_values = self._lem_params.fitness_indicator(self._population.fitness)
            gen = PastGeneration(
                self._population.individuals,
                self._population.fitness,
                fitness_fun_values,
            )
            self._generation_history.append(gen)

            return

    def update_population(self, new_individuals: np.ndarray) -> None:
        self._population.delete(np.arange(len(self._population.individuals)))
        self._population.add(new_individuals)

        return

    def collect_n_past_generations(self, n: int):
        """Collect the n past generations into a single numpy array for easier handling.
        Returns the collected individuals, objective fitness values, and fitness function values.

        Args:
            n (int): number of past generations to collect.

        Returns:
            Tuple[nd.array, np.ndarray, np.ndarray]: A tuple with the collected individuals, objective fitness values, and
            fitness function values. Each array is 2-dimensional.
        """

        if n > len(self._generation_history):
            # n bigger than available history, truncate to length of history
            n = len(self._generation_history)

        past_slice = self._generation_history[-n:]

        individuals = np.concatenate([gen.individuals for gen in past_slice])
        objectives_fitnesses = np.concatenate([gen.objectives_fitnesses for gen in past_slice])
        fitness_fun_values = np.concatenate([gen.fitness_fun_values for gen in past_slice])

        return (
            np.atleast_2d(individuals),
            np.atleast_2d(objectives_fitnesses),
            np.atleast_2d(fitness_fun_values),
        )

    def darwinian_mode(self) -> None:
        """Do evolution. The size of the population can vary."""
        # mate
        offspring = self._population.mate()
        self._population.add(offspring, False)

        fitness_fun_values = self._lem_params.fitness_indicator(self._population.fitness)
        selected = self._ea_params.selection_op.do(self._population, fitness_fun_values)

        self._population.keep(selected)

        return

    def learning_mode(self) -> None:
        instantiation_factor = self._ml_params.instantation_factor

        # collect all past generations and sort them in ascending order accoring to the fitness function values.
        (
            all_individuals,
            all_objectives_fitnesses,
            all_fitness_fun_values,
        ) = self.collect_n_past_generations(len(self._generation_history))

        if not isinstance(self._ml_params.ml_model, MLModel):
            raise TypeError(f"MLModel of type {type(self._ml_params.ml_model)} is not supported in learning mode.")

        sorted_indices = np.argsort(np.squeeze(all_fitness_fun_values))

        # formulate the H and L groups
        # calculate the cut-off indices for both groups, the split might not be exactly
        # accurate due to the cast to int
        h_split_id = int(self._ml_params.H_split * len(sorted_indices))
        l_split_id = int(self._ml_params.L_split * len(sorted_indices))

        # because the indices are now sorted, we can just pick the top best and bottom worst
        # and set them as the H and L groups
        h_indices = sorted_indices[0:h_split_id]
        l_indices = sorted_indices[-l_split_id:][::-1]  # reversing might not really be needed here

        # pick the individuals according to the calculated indices
        h_group = all_individuals[h_indices]
        l_group = all_individuals[l_indices]

        # create training data to train an ML model:
        x_train = np.vstack((h_group, l_group))

        if isinstance(self._ml_params.ml_model, DecisionTreeClassifier):
            # for tree, 1 good, -1 bad
            y_train = np.hstack(
                (
                    np.ones(len(h_group), dtype=int),
                    -1 * np.ones(len(l_group), dtype=int),
                )
            )
        elif isinstance(self._ml_params.ml_model, SlipperClassifier) or isinstance(
            self._ml_params.ml_model, BoostedRulesClassifier
        ):
            # for slipper, 1 good, 0 bad
            y_train = np.hstack((np.ones(len(h_group), dtype=int), np.zeros(len(l_group), dtype=int)))
        else:
            raise TypeError(f"MLModel of type {type(self._ml_params.ml_model)} is not supported in learning mode.")

        n_to_instantiate = int(all_individuals.shape[0] * instantiation_factor)

        # TODO: this is tree specific, check if tree or ruleset!
        #
        if isinstance(self._ml_params.ml_model, DecisionTreeClassifier):
            # do tree stuff
            classifier = self._ml_params.ml_model.fit(x_train, y_train)
            self.current_ml_model = classifier

            # after training the model, it must be used to instantiate new individuals according to
            # the H-group description. I.e., find new individuals that have a classification of 1.
            # by default, instantiate twice as many new individuals as peresent in the whole history.
            # this factor is controlled by the instantiation_factor argument (default 2)

            paths = find_all_paths(classifier)
            instantiated = instantiate_tree_rules(
                paths,
                self._problem.n_of_variables,
                self._problem.get_variable_bounds(),
                n_to_instantiate,
                1,
            )

            # reshape to have just a list of new individuals
            instantiated = instantiated.reshape((-1, instantiated.shape[2]))

        elif isinstance(self._ml_params.ml_model, SlipperClassifier) or isinstance(
            self._ml_params.ml_model, BoostedRulesClassifier
        ):
            # do Slipper stuff
            classifier = self._ml_params.ml_model.fit(x_train, y_train)
            self.current_ml_model = classifier

            ruleset, weights = extract_slipper_rules(classifier)
            instantiated = instantiate_ruleset_rules(
                ruleset,
                weights,
                self._problem.n_of_variables,
                self._problem.get_variable_bounds(),
                n_to_instantiate,
            )

        else:
            raise TypeError(f"MLModel of type {type(self._ml_params.ml_model)} is not supported in learning mode.")

        # mix with the existing H-group
        instantiated_and_h = np.vstack((h_group, instantiated))

        # compute objectives fitnesses
        objective_fitnesses_new = self._problem.evaluate(instantiated_and_h).fitness

        # compute fitness fun values
        fitness_fun_values_new = self._lem_params.fitness_indicator(objective_fitnesses_new)

        # sort the individuals according to their fitness value in ascending order
        sorted_indices_new = np.argsort(np.squeeze(fitness_fun_values_new))

        # pic the best fraction that is the same size as the last population
        selected_indices = sorted_indices_new[0 : self._population.individuals.shape[0]]
        selected_individuals = instantiated_and_h[selected_indices]

        # replace the current population with the new selected individuals
        self.update_population(selected_individuals)

        return

    def check_condition_best(self, n_lookback: int, threshold: float) -> bool:
        """
        Check whether the darwin termination criterion is met. In the past n_lookback iterations.

        Return True and update current best value if condition is met, just return False otherwise.

        """
        # get past generations
        (
            past_individuals,
            past_objectives_fitnesses,
            past_fitness_fun_values,
        ) = self.collect_n_past_generations(n_lookback)

        # find index of best (lowest fitness value) individual
        best_idx = np.argmin(past_fitness_fun_values)

        # check condition
        if ((past_fitness_fun_values[best_idx] / self._best_fitness_fun_value)) < threshold:
            self._best_fitness_fun_value = past_fitness_fun_values[best_idx][0]

            return True

        else:

            return False

    def run(self) -> Dict:
        # counters:
        counters = {"darwin_mode": 0, "learning_mode": 0}

        keep_running = True

        while keep_running:
            improved_in_learning = False
            improved_in_darwin = False

            # start in ML mode
            if self._lem_params.use_ml:
                learning_iters = 0
                while learning_iters < self._lem_params.ml_probe:
                    # always assume previous population has been saved to history before
                    self.learning_mode()
                    self.add_population_to_history()
                    counters["learning_mode"] += 1
                    learning_iters += 1

                    # check generations saved so far in learning more if they meet the termination criterion
                    if self.check_condition_best(learning_iters, self._lem_params.ml_threshold):
                        improved_in_learning = True
                        break
            else:
                improved_in_learning = False

            # do Darwinian mode
            if self._lem_params.use_darwin:
                darwin_iters = 0
                while darwin_iters < self._lem_params.darwin_probe:
                    # we assume the population has been saved in a previous iteration
                    self.darwinian_mode()
                    self.add_population_to_history()
                    counters["darwin_mode"] += 1
                    darwin_iters += 1

                    # iterate until condition is True
                    # check the generations saved so far in Darwin mode, that is
                    # why we keep the darwin_iters counter.
                    if self.check_condition_best(darwin_iters, self._lem_params.darwin_threshold):
                        improved_in_darwin = True
                        break
            else:
                improved_in_darwin = False

            if not improved_in_learning and not improved_in_darwin:
                keep_running = False

        return counters

    def run_iterations(self) -> Dict:
        # counters:
        counters = {"darwin_mode": 0, "learning_mode": 0}

        total_iterations: int = 0

        while total_iterations < self._lem_params.total_iterations:

            # Darwinian (ea) mode
            if self._lem_params.use_darwin:
                iterations_in_ea: int = 0

                while iterations_in_ea < self._ea_params.iterations_per_cycle:
                    # we assume the population has been saved in a previous iteration
                    self.darwinian_mode()
                    self.add_population_to_history()
                    counters["darwin_mode"] += 1
                    iterations_in_ea += 1

            # start in ML mode
            if self._lem_params.use_ml:
                iterations_in_ml: int = 0

                while iterations_in_ml < self._ml_params.iterations_per_cycle:
                    # always assume previous population has been saved to history before
                    self.learning_mode()
                    self.add_population_to_history()
                    counters["learning_mode"] += 1
                    iterations_in_ml += 1

            total_iterations += 1

            print(".")

        counters.update({"total_iterations": total_iterations})

        return counters


if __name__ == "__main__":
    print(issubclass(SBX_xover, CrossOverOP))
