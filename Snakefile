configfile: "experiments.yaml"
script_name = config["script_name"]
problem_name = config["problem"]

""""
# EXPERIMENT 1
rule all_parameters_experiment_1:
    input:
        expand("data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_vehiclecrashworthiness.json",
            run_i=range(1, config["runs_per_experiment"]+1),
            ml_every=config["ml_every_n"],
            hlsplit=[int(100*split) for split in config["hl_split"]],
            problem_name=problem_name
            )

rule parameters_experiment_1:
    params:
        run_i={"run_i"},
        ml_every={"ml_every"},
        hlsplit={"hlsplit"},
        problem_name={"problem_name"}
    output:
        "data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_vehiclecrashworthiness.json" 
    script:
        "scripts/vehicle_crash_worthiness.py"

"""

# EXPERIMENT, collect data
rule all_parameters_experiment:
    input:
        expand("data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_{problem_name}.json",
            run_i=range(1, config["runs_per_experiment"]+1),
            ml_every=config["ml_every_n"],
            hlsplit=[int(100*split) for split in config["hl_split"]],
            problem_name=problem_name
            )

rule parameters_experiment:
    params:
        run_i={"run_i"},
        ml_every={"ml_every"},
        hlsplit={"hlsplit"},
    output:
        "data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_{problem_name}.json" 
    script:
        "scripts/multiple_clutch_brakes.py"

# EXPERIMENT, calcualte statistics from data
rule all_statistics:
    input:
        expand("data/stats_mlevery_{ml_every}_hlsplit_{hlsplit}_{problem_name}.json",
            ml_every=config["ml_every_n"],
            hlsplit=[int(100*split) for split in config["hl_split"]],
            problem_name=problem_name
            )

rule statistics:
    params:
        run_i={"run_i"},
        ml_every={"ml_every"},
        hlsplit={"hlsplit"},
    input:
        expand("data/run_{run_i}_mlevery_{{ml_every}}_hlsplit_{{hlsplit}}_{{problem_name}}.json",
            run_i=range(1, config["runs_per_experiment"]+1),
            )
    output:
        "data/stats_mlevery_{ml_every}_hlsplit_{hlsplit}_{problem_name}.json"
    script:
        "scripts/calculate_statistics.py"