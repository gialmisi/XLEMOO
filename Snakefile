rule run_test_1:
    output:
        "data/output.dat"
    shell:
        "echo 'hello' > data/output.dat"