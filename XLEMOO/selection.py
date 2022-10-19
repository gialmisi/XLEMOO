"""
class SelectionOP(ABC):
    @abstractmethod
    def __init__(self, pop, tournament_size):
        pass

    @abstractmethod
    def do(self, pop: Population, fitness: np.ndarray) -> List[int]:
        pass
"""

from typing import List
import numpy as np
from desdeo_emo.population import SurrogatePopulation


class SelectNBest:
    """A class implementing a selection operator that simply returns the indices of n population members with the
    best (lowest) fitnesses. If the population is smaller than n_best, then return the indices of the whole population.
    """

    def __init__(self, pop: SurrogatePopulation, n_best: int):
        self._n_best = n_best

    def do(self, pop: SurrogatePopulation, fitness: np.ndarray) -> List[int]:
        sorted_indices = np.argsort(fitness, axis=0)
        if len(sorted_indices) < self._n_best:
            # population size less than desired best
            return sorted_indices
        else:
            selected_indices = sorted_indices[: self._n_best]

            return selected_indices
