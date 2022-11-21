# The XLEMOO framework

This framework is for experimenting and implementing explainable and learnable multiobjective optimization (**XLEMOO**) methods.

## Introduction

Learnable evolutionary models are a special kind of model that combine both Darwinian inspired evolutionary
algorithms and machine learning in what are called a _Darwinian mode_ and a _learning mode_, respectively.
This idea can be utilized to implement evolutionary multiobejctive optimization methods as well, which is the
main purpose of this framework. If interpretable machine learning is used, then explanations emerging
from a learning mode may be leveraged to gian further insights about the multiobjective optimization problem
and more.

We recommend reading the artcile TBA for further information on expainable and learnable evolutioanry
multiobjective optimization.

## Installation

We recommend installing the framework utilizing `poetry`. For information on how to install and use `poetry`, see
[link](https://python-poetry.org/).

We have also included a `requirements.txt` file for use with `pip`. However, the `requirements.txt` file might
not always be up to date, as it is only updated when the documentation build system (Read the docs) requires it.

XLEMOO has been run successfully on Python versoin 3.8.12. Other versions have not been tested, but newer versions at
least should work.

## Getting started and documentation

We have provided notebooks for the user to get a quick glimpse on how the framework can be utilized. We have also provided an API documentaton with further details about the framework (link to be adde).

## Testing

Tests have been impelemented for most of the code present in the framework. We utilized `pytest`. To run all the tests at
once, one can run `$ poetry`.

## Citation

To be added...
