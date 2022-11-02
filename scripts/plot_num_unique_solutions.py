import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

data_dir = "/home/kilo/workspace/XLEMOO/data"
filename = "run_1_mlevery_500_hlsplit_10_carsideimpact.json"

with open(f"{data_dir}/{filename}", "r") as f:
    data = json.load(f)

individuals = np.squeeze(np.array([gen for gen in data["individuals"]]))

unique_counts = []
for i_gen in range(individuals.shape[0]):
    stacked = np.vstack(individuals[0 : i_gen + 1])
    uniques = np.unique(stacked, axis=0)
    count = uniques.shape[0]
    unique_counts.append(count)

plt.title(f"Unique count")
plt.xlabel("Generation")
plt.ylabel("N uniques")
plt.plot(range(0, len(unique_counts)), unique_counts)
plt.show()
