import numpy as np
from desdeo_emo.population import Population


def simple_elitism(
    pop1: Population,
    pop1_fitness: np.ndarray,
    pop2: Population,
    pop2_fitness: np.ndarray,
):
    # check population shapes
    if pop1.individuals.shape[1] != pop2.individuals.shape[1]:
        raise ValueError(
            "The shapes of pop1.individuals and pop2.individuals must match."
        )
