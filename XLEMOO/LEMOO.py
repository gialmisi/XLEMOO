import copy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Callable

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
    use_ea: bool
    fitness_indicator: Callable[[Population], np.ndarray]
    past_gens_to_consider: int


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

        # initialize the population and the evolutionary operators
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

        self._population_history = []

    def reset_population(self):
        self._population_history = []
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

    def darwinian_mode(self) -> Population:
        """Do evolution."""
        # mate
        offspring = self._population.mate()
        self._population.add(offspring, False)

        fitness = self._lem_params.fitness_indicator(self._population)
        selected = self._ea_params.selection_op.do(self._population, fitness)

        self._population.keep(selected)

        """
        fitness = self._lem_params.fitness_indicator(self._population)

        # Selection: select individuals to mate and mate
        keep_alive = self._ea_params.selection_op.do(self._population, fitness)
        offsprings = self._population.mate()

        self._population.add(offsprings)
        self._population.keep(keep_alive)
        """
        return

    def learning_mode(self) -> np.ndarray:
        # sort individuals in the current population according to their fitness value
        if self._lem_params.past_gens_to_consider > len(self._population_history):
            print(
                f"The number of past generations to consider ({self._lem_params.past_gens_to_consider}) is greater than the number of past generations ({len(self._population_history)}. Using {len(self._population_history)} past populations instead."
            )
            n_to_consider = len(self._population_history)
        else:
            n_to_consider = self._lem_params.past_gens_to_consider

        fitness = (
            np.array(
                [
                    self._ml_params.ml_fitness(p)
                    for p in self._population_history[-n_to_consider:]
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
            [p.individuals for p in self._population_history[-n_to_consider:]]
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

    def run(self) -> List[Population]:
        # save initial population
        self._population_history.append(copy.copy(self._population))

        for _ in range(self._lem_params.n_total_iterations):
            # Darwinian mode
            for _ in range(self._lem_params.n_ea_gen_per_iter):
                if not self._lem_params.use_ea:
                    break
                self.darwinian_mode()
                self._population_history.append(copy.copy(self._population))

            # Learning mode
            for _ in range(self._lem_params.n_ml_gen_per_iter):
                if not self._lem_params.use_ml:
                    break
                print("starting learning mode")
                new_ml_individuals = self.learning_mode()
                cherries = self.cherry_pick(new_ml_individuals)
                self.update_population(cherries)
                self._population_history.append(copy.copy(self._population))

        # Finish with Darwinian mode``
        for _ in range(self._lem_params.n_ea_gen_per_iter):
            if not self._lem_params.use_ea:
                break
            self.darwinian_mode()
            self._population_history.append(copy.copy(self._population))

        return self._population_history


if __name__ == "__main__":
    print(issubclass(SBX_xover, CrossOverOP))
