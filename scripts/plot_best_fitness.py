import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

data_dir = "/home/kilo/workspace/XLEMOO/data"
filename = "run_1_mlevery_2_hlsplit_40_carsideimpact.json"

with open(f"{data_dir}/{filename}", "r") as f:
    data = json.load(f)

fitness_fun_values = np.squeeze(np.array([gen for gen in data["fitness_fun_values"]]))

best_fitnesses = np.min(fitness_fun_values, axis=1)

plt.yscale("symlog")

plt.title(f"Best fitness (best {best_fitnesses[-1]})")
plt.xlabel("Generation")
plt.ylabel("Fitnes value")
plt.plot(range(0, best_fitnesses.shape[0]), best_fitnesses)
plt.show()
