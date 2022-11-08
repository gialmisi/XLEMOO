import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import itertools
from sklearn.preprocessing import MinMaxScaler

colors = [
    "#D81B60",
    "#1E88E5",
    "#FFC107",
    "#004D40",
    "#99A078",
    "#3DB89B",
    "#B334D2",
    "#8107B5",
    "#F5AFDA",
    "#55001F",
    "#D9CEA3",
]

"""
data_dir = "/home/kilo/workspace/XLEMOO/data/"
hl_splits = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
frequencies = [2, 4, 5, 8, 10, 20, 25, 50, 100, 200, 500]
for_iter = 5
"""

data_dir = snakemake.config["data_dir"]
hl_splits = [int(100 * split) for split in snakemake.config["hl_split"]]
frequencies = snakemake.config["ml_every_n"]
for_iter = int(snakemake.wildcards["for_iter"])

data = {}

# load files
for freq, split in itertools.product(frequencies, hl_splits):
    with open(
        data_dir
        + f"/stats_mlevery_{freq}_hlsplit_{split}_{snakemake.config['problem']}.json"
    ) as fh:
        data[(freq, split)] = {}
        json_dict = json.load(fh)

        for key in json_dict:
            data[(freq, split)][key] = np.squeeze(json_dict[key])

# collect data in numpy arrays for plotting
mean_best_2d = np.zeros((len(frequencies), len(hl_splits)))
std_best_2d = np.zeros((len(frequencies), len(hl_splits)))
mean_mean_2d = np.zeros((len(frequencies), len(hl_splits)))
std_mean_2d = np.zeros((len(frequencies), len(hl_splits)))
mean_hyper_2d = np.zeros((len(frequencies), len(hl_splits)))
std_hyper_2d = np.zeros((len(frequencies), len(hl_splits)))
mean_sum_2d = np.zeros((len(frequencies), len(hl_splits)))
std_sum_2d = np.zeros((len(frequencies), len(hl_splits)))

for freq, split in itertools.product(frequencies, hl_splits):
    # mean best fitness
    freq_i = frequencies.index(freq)
    split_i = hl_splits.index(split)

    mean_best_2d[freq_i, split_i] = data[(freq, split)]["mean_best_fitness_per_iter"][
        for_iter
    ]
    std_best_2d[freq_i, split_i] = data[(freq, split)]["std_best_fitness_per_iter"][
        for_iter
    ]
    mean_mean_2d[freq_i, split_i] = data[(freq, split)]["mean_mean_fitness_per_iter"][
        for_iter
    ]
    std_mean_2d[freq_i, split_i] = data[(freq, split)]["std_mean_fitness_per_iter"][
        for_iter
    ]
    mean_hyper_2d[freq_i, split_i] = data[(freq, split)]["mean_hyper_per_iter"][
        for_iter
    ]
    std_hyper_2d[freq_i, split_i] = data[(freq, split)]["std_hyper_per_iter"][for_iter]
    mean_sum_2d[freq_i, split_i] = data[(freq, split)]["mean_cumsum_per_iter"][for_iter]
    std_sum_2d[freq_i, split_i] = data[(freq, split)]["std_cumsum_per_iter"][for_iter]

# transform the std to percentages of mean value
std_best_2d_per = np.round(np.array((std_best_2d / mean_best_2d) * 100, dtype=float), 1)
std_mean_2d_per = np.round(np.array((std_mean_2d / mean_mean_2d) * 100, dtype=float), 1)
std_hyper_2d_per = np.round(
    np.array((std_hyper_2d / mean_hyper_2d) * 100, dtype=float), 1
)
std_sum_2d_per = np.round(np.array((std_sum_2d / mean_sum_2d) * 100, dtype=float), 1)

### mean matrix
fig, axs = plt.subplots(2, 2)
# fig.suptitle(f"After {for_iter} iterations")
fig.set_size_inches(1.414 * 10, 10)
fig.tight_layout(pad=4.0)

lw = 0.01
xlabel = "Frequency (Nth iteration)"
ylabel = "H/L split (%)"

# mean best
axs[0, 0].set_title("Mean of best fitnesses")
sns.heatmap(ax=axs[0, 0], data=mean_best_2d.T, linewidths=lw, annot=std_best_2d_per.T)

# mean mean
axs[0, 1].set_title("Average of mean fitnesses")
sns.heatmap(ax=axs[0, 1], data=mean_mean_2d.T, linewidths=lw, annot=std_mean_2d_per.T)

# mean hyper
axs[1, 0].set_title("Mean of hypervolumes")
sns.heatmap(ax=axs[1, 0], data=mean_hyper_2d.T, linewidths=lw, annot=std_hyper_2d_per.T)

# mean sum
axs[1, 1].set_title("Mean of cumulative sums")
sns.heatmap(ax=axs[1, 1], data=mean_sum_2d.T, linewidths=lw, annot=std_sum_2d_per.T)

for ax in axs.flat:
    # invert y
    ax.invert_yaxis()

    # set xtick texts
    ax.set_xticks(np.arange(0.5, len(frequencies) + 0.5, 1))
    ax.set_xticklabels(frequencies)

    # set y tick texts
    ax.set_yticks(np.arange(0.5, len(hl_splits) + 0.5, 1))
    ax.set_yticklabels(hl_splits)

    # set xy labels
    ax.set(xlabel=xlabel, ylabel=ylabel)

plt.savefig(snakemake.output[0])
