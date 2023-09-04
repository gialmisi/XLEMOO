configfile: "experiments.yaml"
script_name = config["script_name"]
problem_name = config["problem"]
plot_out_dir = config["plot_out_dir"]

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
        f"scripts/{script_name}"

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


# PLOTTING, of heatmaps
rule all_heatmaps:
    input:
        expand("{plot_out_dir}/heatmap_{for_iter}_{problem_name}.pdf",
            plot_out_dir=plot_out_dir,
            for_iter=config["plot_for_each_iter"],
            problem_name=problem_name)

rule heatmaps:
    params:
        plot_out_dir={"plot_out_dir"},
        for_iter={"for_iter"},
        problem_name={"problem_name"}
    output:
        "{plot_out_dir}/heatmap_{for_iter}_{problem_name}.pdf"
    script:
        "scripts/plot_freq_split_heat.py" 

rule all_heatmaps_per_iter:
    input:
        expand("{plot_out_dir}/heatmaps_{measure}_{problem_name}.pdf",
            plot_out_dir=plot_out_dir,
            measure=["best", "mean", "hyper", "sum"],
            problem_name=problem_name)

rule heatmaps_per_iter:
    output:
        "{plot_out_dir}/heatmaps_best_{problem_name}.pdf",
        "{plot_out_dir}/heatmaps_mean_{problem_name}.pdf",
        "{plot_out_dir}/heatmaps_hyper_{problem_name}.pdf",
        "{plot_out_dir}/heatmaps_sum_{problem_name}.pdf"

    script:
        "scripts/plot_freq_split_heat_iters.py" 

