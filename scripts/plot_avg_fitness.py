import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

data_dir = "/home/kilo/workspace/XLEMOO/data"
filename = "run_1_mlevery_2_hlsplit_40_carsideimpact.json"

with open(f"{data_dir}/{filename}", "r") as f:
    data = json.load(f)

fitness_fun_values = np.squeeze(np.array([gen for gen in data["fitness_fun_values"]]))

avg_fitnesses = np.mean(fitness_fun_values, axis=1)

plt.yscale("symlog")

plt.title(f"Mean fitness")
plt.xlabel("Generation")
plt.ylabel("Mean fitness value")
plt.plot(range(0, avg_fitnesses.shape[0]), avg_fitnesses)
plt.show()
