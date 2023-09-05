import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .LEMOO import PastGeneration
from desdeo_tools.utilities.quality_indicator import hypervolume_indicator
from imodels.util.rule import Rule


def plot_best_fitnesses(generations: List[PastGeneration], ylog=False) -> None:
    """Plots the best fitness value for each generation.

    Args:
        generations (List[PastGeneration]): A list of past generations.
        ylog (bool, optional): Should the y-scale be logarithmic? Defaults to False.
    """
    best_values = np.array([np.min(gen.fitness_fun_values) for gen in generations])
    if ylog:
        plt.yscale("symlog")
    plt.plot(np.arange(best_values.shape[0]), best_values)

    return


def plot_mean_fitnesses(generations: List[PastGeneration], ylog=False) -> None:
    """Plots the mean fitness value of each generation.

    Args:
        generations (List[PastGeneration]): A list of past generations.
        ylog (bool, optional): Should the y-scale be logarithmic? Defaults to False.
    """
    mean_values = np.array([np.mean(gen.fitness_fun_values) for gen in generations])
    if ylog:
        plt.yscale("symlog")
    plt.plot(np.arange(mean_values.shape[0]), mean_values)

    return


def plot_std_fitnesses(generations: List[PastGeneration]) -> None:
    """Plots the standard deviation of the fitness value of each generation.

    Args:
        generations (List[PastGeneration]): A list of past generations.
    """
    std_values = np.array([np.std(gen.fitness_fun_values) for gen in generations])
    plt.plot(np.arange(std_values.shape[0]), std_values)

    return


def plot_hypervolume(generations: List[PastGeneration], ref_point: np.ndarray) -> None:
    """Plots the hypervolume of each generation.

    Args:
        generations (List[PastGeneration]): A list of past generations.
        ref_point (np.ndarray): A reference point utilized in computing the hypervolume.
    """
    hwas = np.array(
        [
            hypervolume_indicator(gen.objectives_fitnesses, ref_point)
            for gen in generations
        ]
    )
    plt.plot(np.arange(hwas.shape[0]), hwas)

    return