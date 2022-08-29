import copy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Union

from desdeo_emo.population import (
    SurrogatePopulation,
    create_new_individuals,
    Population,
)
from desdeo_emo.recombination import SBX_xover, BP_mutation
from desdeo_emo.selection import TournamentSelection
from desdeo_problem.problem import MOProblem

import numpy as np

from XLEMOO.tree_interpreter import find_all_paths, instantiate_tree_rules


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


@dataclass
class MLParams:
    H_split: float
    L_split: float
    ml_model: MLModel
    ml_fitness: Callable[[Population], np.ndarray]


@dataclass
class LEMParams:
    n_total_iterations: int
    n_ea_gen_per_iter: int
    n_ml_gen_per_iter: int
    use_ml: bool
    use_darwin: bool
    fitness_indicator: Callable[[np.ndarray], np.ndarray]
    ml_probe: int
    darwin_probe: int
    ml_threshold: float
    darwin_threshold: float


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

        # initialize the population and the evolutionary operators
        self._generation_history: List[PastGeneration] = []
        self.initialize_population()
        self.add_population_to_history()

    def initialize_population(self) -> None:
        """Use the defined initialization design to init a new population"""
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

    def reset_population(self):
        """Reset the current population."""
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

    def reset_generation_history(self):
        """Reset the population history."""
        self._generation_history = []

    def add_population_to_history(
        self,
        individuals: Optional[np.ndarray] = None,
        objectives_fitnesses: Optional[np.ndarray] = None,
    ) -> None:
        if individuals is not None and objectives_fitnesses is not None:
            # add current population to history
            fitness_fun_values = self._lem_params.fitness_indicator(individuals)
            gen = PastGeneration(individuals, objectives_fitnesses, fitness_fun_values)
            self._generation_history.append(gen)

            return

        else:
            # add supplied individuals and fitnesses to history
            fitness_fun_values = self._lem_params.fitness_indicator(
                self._population.fitness
            )
            gen = PastGeneration(
                self._population.individuals,
                self._population.fitness,
                fitness_fun_values,
            )
            self._generation_history.append(gen)

            return

    def darwinian_mode(self) -> None:
        """Do evolution. The size of the population can vary."""
        # mate
        offspring = self._population.mate()
        self._population.add(offspring, False)

        fitness_fun_values = self._lem_params.fitness_indicator(
            self._population.fitness
        )
        selected = self._ea_params.selection_op.do(self._population, fitness_fun_values)

        self._population.keep(selected)

        return

    def learning_mode(self) -> np.ndarray:
        # sort individuals in the current population according to their fitness value
        if self._lem_params.past_gens_to_consider > len(self._generation_history):
            print(
                f"The number of past generations to consider ({self._lem_params.past_gens_to_consider}) is greater than the number of past generations ({len(self._generation_history)}. Using {len(self._generation_history)} past populations instead."
            )
            n_to_consider = len(self._generation_history)
        else:
            n_to_consider = self._lem_params.past_gens_to_consider

        fitness = (
            np.array(
                [
                    self._ml_params.ml_fitness(p)
                    for p in self._generation_history[-n_to_consider:]
                ]
            )
            .squeeze()
            .reshape(-1)
        )
        sorted_ids = np.squeeze(np.argsort(fitness, axis=0))

        # formulate the H and L groups
        h_split_id = int(self._ml_params.H_split * len(sorted_ids))
        l_split_id = int(self._ml_params.L_split * len(sorted_ids))

        h_group_ids = sorted_ids[0:h_split_id]
        l_group_ids = sorted_ids[-l_split_id:]

        individuals = np.vstack(
            [p.individuals for p in self._generation_history[-n_to_consider:]]
        )

        h_sample = individuals[h_group_ids]
        l_sample = individuals[l_group_ids]

        Y = np.hstack(
            (np.ones(len(h_sample), dtype=int), -np.ones(len(l_sample), dtype=int))
        )
        X = np.vstack((h_sample, l_sample))

        # create and train a classifier on the H and L samples
        classifier = self._ml_params.ml_model.fit(X, Y)
        self.current_ml_model = classifier

        # based on the trained model, generate new individuals and combine them with the existing H sample
        # n_individuals_needed = len(sorted_ids) - len(h_group_ids)
        n_individuals_needed = self._ea_params.population_size

        paths = find_all_paths(classifier)

        # do this until enough fit individuals are found
        instantiated = instantiate_tree_rules(
            paths,
            self._problem.n_of_variables,
            self._problem.get_variable_bounds(),
            n_individuals_needed,
            1,
        )

        instantiated = instantiated.reshape((-1, instantiated.shape[2]))

        selected_individuals = instantiated[
            np.random.choice(
                instantiated.shape[0], n_individuals_needed, replace=False
            ),
            :,
        ]

        # final_individuals = np.vstack((h_sample, selected_individuals))
        final_individuals = selected_individuals

        return final_individuals

    def update_population(self, new_individuals: np.ndarray):
        self._population.delete(np.arange(len(self._population.individuals)))
        self._population.add(new_individuals)
        return

    """
    def cherry_pick(self, ml_individuals: np.ndarray) -> np.ndarray:
        ea_individuals = self._population.individuals

        individuals = np.vstack((ea_individuals, ml_individuals))

        ea_fitness = self._lem_params.fitness_indicator(self._population)
        ml_fitness = self._lem_params.fitness_indicator(
            DummyPopulation(ml_individuals, self._problem)
        )
        fitness = np.vstack((ea_fitness, ml_fitness))

        sorted_ids = np.argsort(fitness, axis=0)

        # how many from ml population and how many from ea population were chosen
        from_ea = (
            sorted_ids[: self._ea_params.population_size]
            < self._ea_params.population_size
        ).sum()
        from_ml = self._ea_params.population_size - from_ea

        print(f"Cherry pick results: EA: {from_ea}; ML: {from_ml}.")

        cherries = np.squeeze(
            individuals[sorted_ids[: self._ea_params.population_size]]
        )

        return cherries
    """

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
        objectives_fitnesses = np.concatenate(
            [gen.objectives_fitnesses for gen in past_slice]
        )
        fitness_fun_values = np.concatenate(
            [gen.fitness_fun_values for gen in past_slice]
        )

        return (
            np.atleast_2d(individuals),
            np.atleast_2d(objectives_fitnesses),
            np.atleast_2d(fitness_fun_values),
        )

    def check_darwin_condition(self):
        """
        Check whether the darwin termination criterion is met.
        Current criterion, the best fitness value in the past 'darwin_probe' generations must have increased by darwin_threshold.

        """
        pass

    def run(self) -> None:
        # start in ML mode
        if self._lem_params.use_ml:
            # TODO: reimplement me!
            pass

        # do Darwinian mode
        if self._lem_params.use_darwin:
            pass

        return


if __name__ == "__main__":
    print(issubclass(SBX_xover, CrossOverOP))
