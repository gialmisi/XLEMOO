import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .LEMOO import PastGeneration


def plot_best_fitnesses(generations: List[PastGeneration]) -> None:
    best_values = np.array([np.min(gen.fitness_fun_values) for gen in generations])
    plt.plot(np.arange(best_values.shape[0]), best_values)

    return


def plot_mean_fitnesses(generations: List[PastGeneration]) -> None:
    mean_values = np.array([np.mean(gen.fitness_fun_values) for gen in generations])
    plt.plot(np.arange(mean_values.shape[0]), mean_values)

    return


def plot_std_fitnesses(generations: List[PastGeneration]) -> None:
    std_values = np.array([np.std(gen.fitness_fun_values) for gen in generations])
    plt.plot(np.arange(std_values.shape[0]), std_values)

    return
