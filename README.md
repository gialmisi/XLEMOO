# The XLEMOO framework

![Darwinian and Learning Modes](docs/figures/modes.svg)

## Introduction

XLEMOO (Explainable Learnable Multiobjective Optimization) is a Python framework for evolutionary multiobjective optimization integrated with machine learning. The key concept is to blend Darwinian-inspired evolutionary algorithms with interpretable machine learning models to discover a population of near-Pareto optimal solutions for multiobjective optimization problems. The combination of evolutionary algorithms and machine learning leads to two modes: the Darwinian mode and the learning mode. The framework enables explainability by building an understanding of what characterizes good solutions in a population.

## Getting Started

### Requirements

- Python version 3.9 or 3.10
- git
- [Poetry](https://python-poetry.org/)

### Installation

1. Clone the XLEMOO repository:

   ```shell
   git clone https://github.com/gialmisi/XLEMOO
   cd XLEMOO
   ```

2. Create and activate a new virtual environment with Poetry:

   ```shell
   poetry shell
   ```

3. Install the framework:

   ```shell
   poetry install
   ```

   To include development dependencies, use:

   ```shell
   poetry install --with dev
   ```

### Running Tests

1. XLEMOO utilizes pytest for unit testing. Make sure development dependencies are installed:

   ```shell
   poetry install --with dev
   ```

2. Run the unit tests:

   ```shell
   pytest --reruns 5
   ```

## Documentation

The main main documentation of the XLEMOO framework is hosted on readthedocs and can be found >HERE<.

Alternatively, you can build the documentation manually. First, make sure the development dependencies are installed with poetry.
Then, run the following command from the root directory of the project:

```shell
cd docs
make html
```

This should build the documentation in a html format in the `docs/_build` directory. You can open the documentation with your favorite web browser by issuing the command (example with Firefox):

```shell
firefox _build/html/index.html
```

### Next Steps

- For a usage example, refer to the Notebooks section in the main documentation.
- To use and start modifying the framework, refer to the Basic Usage section in the main documentation.
- To reproduce the numerical experiments, refer to the Reproducibility section in the main documentation.
- The API documentation provides more information on the specific parts of the code in the framework.

## Citation

If you utilize the XLEMOO framework in your research, please cite the following publication:

Misitano, G. (2023). Exploring the Explainable Aspects and Performance of a Learnable Evolutionary Multiobjective Optimization Method. ACM Transactions on Evolutionary Learning and Optimization. To be published.
