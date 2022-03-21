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


class MLModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


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
    n_total_itarations: int
    n_ea_gen_per_iter: int
    n_ml_gen_per_iter: int
    use_ml: bool
    use_ea: bool
    fitness_indicator: Callable[[Population], np.ndarray]


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

        # initialize the population and the evolutionary operators
        initial_population = create_new_individuals(
            self._ea_params.population_init_design,
            self._problem,
            pop_size=self._ea_params.population_size,
        )
        self._population = SurrogatePopulation(
            problem,
            ea_params.population_size,
            initial_population,
            self._ea_params.cross_over_op,
            self._ea_params.mutation_op,
            None,
        )

        self._population_history = []

        return

    def darwinian_mode(self) -> np.ndarray:
        # compute the fitnesses of the current population
        fitness = self._lem_params.fitness_indicator(self._population)
        print(f"fitness: {fitness}")

        # select individuals to mate
        to_mate = self._ea_params.selection_op.do(self._population, fitness)

        # mate
        new_individuals = self._population.mate(to_mate)

        # update the population
        # self._population.delete(np.arange(len(self._population.individuals)))
        # self._population.add(new_population)

        return new_individuals

    def learning_mode(self) -> np.ndarray:
        # sort individuals in the current population according to their fitness value
        fitness = self._ml_params.ml_fitness(self._population)
        sorted_ids = np.squeeze(np.argsort(fitness, axis=0))

        # formulate the H and L groups
        h_split_id = int(self._ml_params.H_split * len(sorted_ids))
        l_split_id = int(self._ml_params.L_split * len(sorted_ids))

        h_group_ids = sorted_ids[0:h_split_id]
        l_group_ids = sorted_ids[-l_split_id:]

        h_sample = self._population.individuals[h_group_ids]
        l_sample = self._population.individuals[l_group_ids]

        Y = np.hstack((np.ones(len(h_sample)), -np.ones(len(h_sample))))
        X = np.vstack((h_sample, l_sample))

        # create and train a classifier on the H and L samples
        classifier = self._ml_params.ml_model.fit(X, Y)

        # based on the trained model, generate new individuals and combine them with the existing H sample
        # TODO: do this in a smarter way utilizing the model
        n_individuals_needed = len(sorted_ids) - len(h_group_ids)
        n_found = 0

        lower_bounds = self._problem.get_variable_lower_bounds()
        upper_bounds = self._problem.get_variable_upper_bounds()

        ranges = upper_bounds - lower_bounds

        new_individuals = []

        while n_found < n_individuals_needed:
            candidate = np.random.rand(len(ranges)) * ranges + lower_bounds

            if classifier.predict(np.atleast_2d(candidate))[0] == 1:
                new_individuals.append(candidate)
                n_found += 1

        final_individuals = np.vstack((h_sample, np.array(new_individuals)))

        return final_individuals

    def update_population(self, new_individuals: np.ndarray):
        self._population.delete(np.arange(len(self._population.individuals)))
        self._population.add(new_individuals)
        return

    def run(self):
        return


if __name__ == "__main__":
    print(issubclass(SBX_xover, CrossOverOP))
