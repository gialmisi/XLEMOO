configfile: "experiments.yaml"
problem_name = config["problem"]

rule all:
    input:
        expand("data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_{problem_name}.json",
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
        "data/run_{run_i}_mlevery_{ml_every}_hlsplit_{hlsplit}_{problem_name}.json" 
    script:
        "scripts/run_w_freq_and_split.py"

rule run_test_1:
    output:
        "data/output.dat"
    shell:
        "echo 'hello' > data/output.dat"