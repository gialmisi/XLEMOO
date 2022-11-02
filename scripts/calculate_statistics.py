import numpy as np
import json
import matplotlib.pyplot as plt
from desdeo_tools.utilities.quality_indicator import hypervolume_indicator

run_n = snakemake.config["runs_per_experiment"]
filenames = snakemake.input

all_individuals = []
all_fitness_fun_values = []
all_objective_values = []


for filename in filenames[:run_n]:
    with open(filename, "r") as f:
        data = json.load(f)
        all_individuals.append(np.squeeze(data["individuals"]))
        all_fitness_fun_values.append(np.squeeze(data["fitness_fun_values"]))
        all_objective_values.append(np.squeeze(data["objective_values"]))

all_individuals = np.array(all_individuals)
all_fitness_fun_values = np.array(all_fitness_fun_values)
all_objective_values = np.array(all_objective_values)

data = {}

# Mean best fitness
best_fitness_per_gen = np.min(all_fitness_fun_values, axis=2)
mean_best_fitness_per_iter = np.mean(best_fitness_per_gen, axis=0)
std_best_fitness_per_iter = np.std(best_fitness_per_gen, axis=0)

data.update(
    {
        "mean_best_fitness_per_iter": mean_best_fitness_per_iter.tolist(),
        "std_best_fitness_per_iter": std_best_fitness_per_iter.tolist(),
    }
)

# Mean mean fitness
mean_fitness_per_gen = np.mean(all_fitness_fun_values, axis=2)
mean_mean_fitness_per_iter = np.mean(mean_fitness_per_gen, axis=0)
std_mean_fitness_per_iter = np.std(mean_fitness_per_gen, axis=0)

data.update(
    {
        "mean_mean_fitness_per_iter": mean_mean_fitness_per_iter.tolist(),
        "std_mean_fitness_per_iter": std_mean_fitness_per_iter.tolist(),
    }
)

# Mean hyper
nadir = np.array(snakemake.config["nadir"])  # REMEMBER TO CHANGE ME!
hyper_per_gen = np.array(
    [
        [
            hypervolume_indicator(all_objective_values[run_i, gen_i, :], nadir)
            for gen_i in range(2001)
        ]
        for run_i in range(run_n)
    ]
)
mean_hyper_per_iter = np.mean(hyper_per_gen, axis=0)
std_hyper_per_iter = np.std(hyper_per_gen, axis=0)

data.update(
    {
        "mean_hyper_per_iter": mean_hyper_per_iter.tolist(),
        "std_hyper_per_iter": std_hyper_per_iter.tolist(),
    }
)

# Cumulative number of unique solutions
def count_cumulative_uniques(individuals):
    unique_counts = []
    for i_gen in range(individuals.shape[0]):
        stacked = np.vstack(individuals[0 : i_gen + 1])
        uniques = np.unique(stacked, axis=0)
        count = uniques.shape[0]
        unique_counts.append(count)

        if i_gen % 100 == 0:
            print(f"Gen {i_gen}")

    return unique_counts


cumsum_per_run = []
for run_i in range(run_n):
    print(f"Run: {run_i}")
    cumsum = count_cumulative_uniques(all_individuals[run_i])
    cumsum_per_run.append(cumsum)

cumsum_per_run = np.array(cumsum_per_run)
mean_cumsum_per_iter = np.mean(cumsum_per_run, axis=0)
std_cumsum_per_iter = np.std(cumsum_per_run, axis=0)

data.update(
    {
        "mean_cumsum_per_iter": mean_cumsum_per_iter.tolist(),
        "std_cumsum_per_iter": std_cumsum_per_iter.tolist(),
    }
)

output_filename = snakemake.output[0]
with open(output_filename, "w") as f:
    json.dump(data, f)
