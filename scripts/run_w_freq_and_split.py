import sys

sys.path.append("../XLEMOO")

import numpy as np
from desdeo_problem.testproblems import vehicle_crashworthiness
from XLEMOO.fitness_indicators import asf_wrapper
from XLEMOO.LEMOO import EAParams, LEMParams, MLParams, LEMOO
from XLEMOO.selection import SelectNBest
from desdeo_tools.scalarization.ASF import PointMethodASF
from desdeo_emo.recombination import SBX_xover, BP_mutation
from imodels import SkopeRulesClassifier
import json

n_objectives = 3
n_of_variables = 5

nadir = np.array([1700.0, 12.0, 0.2])
ideal = np.array([1600.0, 6.0, 0.038])

ref_point = np.array([1650.0, 7.0, 0.05])

problem = vehicle_crashworthiness()

fitness_fun = asf_wrapper(
    PointMethodASF(nadir=nadir, ideal=ideal), {"reference_point": ref_point}
)

lem_params = LEMParams(
    use_darwin=True,
    use_ml=True,
    fitness_indicator=fitness_fun,
    ml_probe=1,
    ml_threshold=None,
    darwin_probe=5,
    darwin_threshold=None,
    total_iterations=5,  # TODO: must be calculated
)

pop_size = 200

ea_params = EAParams(
    population_size=pop_size,
    cross_over_op=SBX_xover(),
    mutation_op=BP_mutation(
        problem.get_variable_lower_bounds(), problem.get_variable_upper_bounds()
    ),
    selection_op=SelectNBest(None, pop_size),
    population_init_design="LHSDesign",
    iterations_per_cycle=1,  # TODO: must be calculated,
)

# TODO: document me
ml = SkopeRulesClassifier(
    precision_min=0.1,
    n_estimators=30,
    max_features=None,
    max_depth=None,
    bootstrap=True,
    bootstrap_features=True,
)

ml_params = MLParams(
    H_split=0.20,
    L_split=0.10,
    ml_model=ml,
    instantation_factor=10,  # TODO: document me
    generation_lookback=0,
    ancestral_recall=0,
    unique_only=True,
    iterations_per_cycle=1,  # TODO: must be calculated
)

lemoo = LEMOO(problem, lem_params, ea_params, ml_params)

lemoo.run_iterations()

data = {"individuals": [], "fitness_fun_values": [], "objective_values": []}

for gen in lemoo._generation_history:
    data["individuals"].append(gen.individuals.tolist())
    data["fitness_fun_values"].append(gen.fitness_fun_values.tolist())
    data["objective_values"].append(gen.objectives_fitnesses.tolist())

print(f"Done. Len of generation history: {len(lemoo._generation_history)}")

with open("./data.json", "w") as f:
    json.dump(data, f)
