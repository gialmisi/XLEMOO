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
from datetime import datetime

n_objectives = 3
n_variables = 5

nadir = np.array([1700.0, 12.0, 0.2])
ideal = np.array([1600.0, 6.0, 0.038])

ref_point = np.array([1650.0, 7.0, 0.05])

problem = vehicle_crashworthiness()
problem_name = "vehicle crash worthiness"

fitness_fun = asf_wrapper(PointMethodASF(nadir=nadir, ideal=ideal), {"reference_point": ref_point})
fitness_fun_name = "PointMethodASF"

use_darwin = True
use_ml = True
ml_probe = None
ml_threshold = None
darwin_probe = None
darwin_threshold = None
total_iterations = 5  # TODO: must be calculated

lem_params = LEMParams(
    use_darwin=use_darwin,
    use_ml=use_ml,
    fitness_indicator=fitness_fun,
    ml_probe=ml_probe,
    ml_threshold=ml_threshold,
    darwin_probe=darwin_probe,
    darwin_threshold=darwin_threshold,
    total_iterations=total_iterations,
)

pop_size = 200
cross_over_op = SBX_xover()
mutation_op = BP_mutation(problem.get_variable_lower_bounds(), problem.get_variable_upper_bounds())
selection_op = SelectNBest(None, pop_size)
populatin_init_design = "LHSDesign"
ea_iterations_per_cycle = 1

ea_params = EAParams(
    population_size=pop_size,
    cross_over_op=cross_over_op,
    mutation_op=mutation_op,
    selection_op=selection_op,
    population_init_design=populatin_init_design,
    iterations_per_cycle=ea_iterations_per_cycle,
)

ml_model_name = "SkopedRuleClassifier"
ml_precision_min = 0.1
ml_n_estimators = 30
ml_max_features = None
ml_max_depth = None
ml_bootstrap = True
ml_bootstrap_features = True

ml = SkopeRulesClassifier(
    precision_min=ml_precision_min,
    n_estimators=ml_n_estimators,
    max_features=ml_max_features,
    max_depth=ml_max_depth,
    bootstrap=ml_bootstrap,
    bootstrap_features=ml_bootstrap_features,
)

h_split = 0.2
l_split = 0.2
instantation_factor = 10
generation_lookback = 0
ancestral_recall = 0
unique_only = True
ml_iterations_per_cycle = 1

ml_params = MLParams(
    H_split=h_split,
    L_split=l_split,
    ml_model=ml,
    instantation_factor=instantation_factor,
    generation_lookback=generation_lookback,
    ancestral_recall=ancestral_recall,
    unique_only=unique_only,
    iterations_per_cycle=ml_iterations_per_cycle,
)

lemoo = LEMOO(problem, lem_params, ea_params, ml_params)

lemoo.run_iterations()

data = {"individuals": [], "fitness_fun_values": [], "objective_values": []}

info = {
    "date": datetime.today().strftime("%Y-%m-%d"),
    "problem": problem_name,
    "n_variables": n_variables,
    "n_objectives": n_objectives,
    "ideal": ideal.tolist(),
    "nadir": nadir.tolist(),
    "ref_point": ref_point.tolist(),
    "fitness_fun": fitness_fun_name,
}

data["info"] = info

parameters = {
    "lemoo": {
        "use_darwin": use_darwin,
        "use_ml": use_ml,
        "ml_probe": ml_probe,
        "ml_threshold": ml_threshold,
        "darwin_probe": darwin_probe,
        "darwin_threshold": darwin_threshold,
        "total_iterations": total_iterations,
    },
    "ea_mode": {
        "pop_size": pop_size,
        "cross_over_op": str(cross_over_op),
        "mutation_op": str(mutation_op),
        "selection_op": str(selection_op),
        "populatin_init_design": populatin_init_design,
        "ea_iterations_per_cycle": ea_iterations_per_cycle,
    },
    "ml_model": {
        "ml_model_name": ml_model_name,
        "ml_precision_min": ml_precision_min,
        "ml_n_estimators": ml_n_estimators,
        "ml_max_features": ml_max_features,
        "ml_max_depth": ml_max_depth,
        "ml_bootstrap": ml_bootstrap,
        "ml_bootstrap_features": ml_bootstrap_features,
    },
    "ml_mode": {
        "h_split": h_split,
        "l_split": l_split,
        "instantation_factor": instantation_factor,
        "generation_lookback": generation_lookback,
        "ancestral_recall": ancestral_recall,
        "unique_only": unique_only,
        "ml_iterations_per_cycle": ml_iterations_per_cycle,
    },
}

data["parameters"] = parameters

for gen in lemoo._generation_history:
    data["individuals"].append(gen.individuals.tolist())
    data["fitness_fun_values"].append(gen.fitness_fun_values.tolist())
    data["objective_values"].append(gen.objectives_fitnesses.tolist())

print(f"Done. Len of generation history: {len(lemoo._generation_history)}")

with open("./data.json", "w") as f:
    json.dump(data, f)
