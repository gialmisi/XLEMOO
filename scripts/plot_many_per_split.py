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
]
n_iters = 1001
data_dir = "/home/kilo/workspace/XLEMOO/data/"
hl_splits = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
data_files = [
    data_dir + f"stats_mlevery_100_hlsplit_{split}_carsideimpact.json"
    for split in hl_splits
]

data = {}

# load files
for df, split in zip(data_files, hl_splits):
    with open(df) as fhandle:
        data[f"{split}"] = {}
        json_dict = json.load(fhandle)

        for key in json_dict:
            data[f"{split}"][key] = np.squeeze(json_dict[key])


# plot avg fitness
for i, split in enumerate(hl_splits):
    plt.plot(
        range(n_iters),
        data[f"{split}"]["mean_mean_fitness_per_iter"][:n_iters],
        c=colors[i],
        label=f"{split}",
    )

plt.legend()
plt.show()
