import sys

sys.path.append(snakemake.config["path_to_xlemoo"])

import numpy as np
from desdeo_problem.testproblems import car_side_impact
from XLEMOO.fitness_indicators import asf_wrapper
from XLEMOO.LEMOO import EAParams, LEMParams, MLParams, LEMOO
from XLEMOO.selection import SelectNBest
from desdeo_tools.scalarization.ASF import PointMethodASF
from desdeo_emo.recombination import SBX_xover, BP_mutation
from imodels import SkopeRulesClassifier
import json
from datetime import datetime

n_objectives = snakemake.config["n_objectives"]
n_variables = snakemake.config["n_variables"]

nadir = np.array(snakemake.config["nadir"])
ideal = np.array(snakemake.config["ideal"])

ref_point = np.array(snakemake.config["ref_point"])

problem = car_side_impact(three_obj=False)
problem_name = "car_side_impact"

fitness_fun = asf_wrapper(
    PointMethodASF(nadir=nadir, ideal=ideal), {"reference_point": ref_point}
)
fitness_fun_name = "PointMethodASF"

n_total_iterations = snakemake.config["total_iterations"]
ml_every_n = int(snakemake.wildcards["ml_every"])
n_ea_per_cycle = ml_every_n - 1
lemoo_total_iterations = int(n_total_iterations / (n_ea_per_cycle + 1))

use_darwin = snakemake.config["use_darwin"]
use_ml = snakemake.config["use_ml"]
ml_probe = snakemake.config["ml_probe"]
ml_threshold = snakemake.config["ml_threshold"]
darwin_probe = snakemake.config["darwin_probe"]
darwin_threshold = snakemake.config["darwin_threshold"]
total_iterations = lemoo_total_iterations

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

pop_size = snakemake.config["pop_size"]
cross_over_op = SBX_xover()
mutation_op = BP_mutation(
    problem.get_variable_lower_bounds(), problem.get_variable_upper_bounds()
)
selection_op = SelectNBest(None, pop_size)
population_init_design = snakemake.config["population_init_design"]
ea_iterations_per_cycle = n_ea_per_cycle

ea_params = EAParams(
    population_size=pop_size,
    cross_over_op=cross_over_op,
    mutation_op=mutation_op,
    selection_op=selection_op,
    population_init_design=population_init_design,
    iterations_per_cycle=ea_iterations_per_cycle,
)

ml_model_name = snakemake.config["ml_model_name"]
ml_precision_min = snakemake.config["ml_precision_min"]
ml_n_estimators = snakemake.config["ml_n_estimators"]
ml_max_features = snakemake.config["ml_max_features"]
ml_max_depth = snakemake.config["ml_max_depth"]
ml_bootstrap = snakemake.config["ml_bootstrap"]
ml_bootstrap_features = snakemake.config["ml_bootstrap_features"]

ml = SkopeRulesClassifier(
    precision_min=ml_precision_min,
    n_estimators=ml_n_estimators,
    max_features=ml_max_features,
    max_depth=ml_max_depth,
    bootstrap=ml_bootstrap,
    bootstrap_features=ml_bootstrap_features,
)

h_split = int(snakemake.wildcards["hlsplit"]) / 100
l_split = int(snakemake.wildcards["hlsplit"]) / 100
instantiation_factor = snakemake.config["instantiation_factor"]
generation_lookback = snakemake.config["generation_lookback"]
ancestral_recall = snakemake.config["ancestral_recall"]
unique_only = snakemake.config["unique_only"]
ml_iterations_per_cycle = snakemake.config["ml_iterations_per_cycle"]

ml_params = MLParams(
    H_split=h_split,
    L_split=l_split,
    ml_model=ml,
    instantiation_factor=instantiation_factor,
    generation_lookback=generation_lookback,
    ancestral_recall=ancestral_recall,
    unique_only=unique_only,
    iterations_per_cycle=ml_iterations_per_cycle,
)

lemoo = LEMOO(problem, lem_params, ea_params, ml_params)

print(f"Generating {snakemake.output[0]}...")
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
    "n_total_iterations": n_total_iterations,
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
        "population_init_design": population_init_design,
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
        "instantiation_factor": instantiation_factor,
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

with open(snakemake.output[0], "w") as f:
    json.dump(data, f)
