import matplotlib.pyplot as plt
import numpy as np
import json

n_iters = 200
data_dir = "./data"
problem_name = "vehiclecrashworthiness"
output_dir = "./figures"

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
    "#3FD17E",
]

frequencies = [2, 4, 5, 8, 10, 20, 25, 50, 100, 200, 500]
data_files = [
    f"{data_dir}/stats_mlevery_{freq}_hlsplit_20_{problem_name}.json"
    for freq in frequencies
]

data = {}

# load files
for df, freq in zip(data_files, frequencies):
    with open(df) as fhandle:
        data[f"{freq}"] = {}
        json_dict = json.load(fhandle)

        for key in json_dict:
            data[f"{freq}"][key] = np.squeeze(json_dict[key])


fig, axs = plt.subplots(2, 2)
fig.set_size_inches(1.414 * 10, 10)

# plot avg fitness
for i, freq in enumerate(frequencies):
    axs[0, 0].set_title("Mean of best fitnesses")
    axs[0, 0].set(ylabel="Fitness")
    axs[0, 0].plot(
        range(n_iters),
        data[f"{freq}"]["mean_best_fitness_per_iter"][:n_iters],
        c=colors[i],
        label=f"{freq}",
    )
    axs[0, 1].set_title("Average of mean fitnesses")
    axs[0, 1].set(ylabel="Fitness")
    axs[0, 1].plot(
        range(n_iters),
        data[f"{freq}"]["mean_mean_fitness_per_iter"][:n_iters],
        c=colors[i],
        label=f"{freq}",
    )
    axs[1, 0].set_title("Mean of hypervolumes")
    axs[1, 0].set(ylabel="Hypervolume")
    axs[1, 0].plot(
        range(n_iters),
        data[f"{freq}"]["mean_hyper_per_iter"][:n_iters],
        c=colors[i],
        label=f"{freq}",
    )
    axs[1, 1].set_title("Mean of cumulative sums")
    axs[1, 1].set(ylabel="Sum of unique solutions")
    axs[1, 1].plot(
        range(n_iters),
        data[f"{freq}"]["mean_cumsum_per_iter"][:n_iters],
        c=colors[i],
        label=f"{freq}",
    )

for ax in axs.flat:
    ax.set(xlabel="Iteration")

plt.legend(
    bbox_to_anchor=(1.48, 2.22),
    fancybox=True,
    shadow=True,
    ncol=2,
    title="Learning mode frequency",
)
fig.subplots_adjust(right=0.8)
# plt.show()
plt.savefig(
    f"{output_dir}/many_per_frequency_n_{n_iters}_{problem_name}.pdf"
)
