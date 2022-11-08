import matplotlib.pyplot as plt
import numpy as np
import json

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
n_iters = 1001
data_dir = "/home/kilo/workspace/XLEMOO/data/"
frequencies = [2, 4, 5, 8, 10, 20, 25, 50, 100, 200, 500]
data_files = [
    data_dir + f"stats_mlevery_{freq}_hlsplit_50_carsideimpact.json"
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


# plot avg fitness
for i, freq in enumerate(frequencies):
    plt.plot(
        range(n_iters),
        data[f"{freq}"]["mean_best_fitness_per_iter"][:n_iters],
        c=colors[i],
        label=f"{freq}",
    )

plt.legend()
plt.show()
