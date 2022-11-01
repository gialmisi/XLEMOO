configfile: "experiments.yaml"
problem_name = config["problem"]

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


# EXPERIMENT 2
rule all_parameters_experiment_2:
    input:
        expand("data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_carsideimpact.json",
            run_i=range(1, config["runs_per_experiment"]+1),
            ml_every=config["ml_every_n"],
            hlsplit=[int(100*split) for split in config["hl_split"]],
            problem_name=problem_name
            )

rule parameters_experiment_2:
    params:
        run_i={"run_i"},
        ml_every={"ml_every"},
        hlsplit={"hlsplit"},
        problem_name={"problem_name"}
    output:
        "data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_carsideimpact.json" 
    script:
        "scripts/carside_impact.py"