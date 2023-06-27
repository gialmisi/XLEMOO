Introduction: what is XLEMOO?
===============

.. note::

    The reader is assumed to be familiar with the basic concepts of multiobjective optimization.

XLEMOO (*Explainable learnable multiobjective optimization*) is a Python framework
for evolutionary multiobjective optimization enhanced with machine learning. The core idea
is to utilize both Darwinian inspired evolutionary algorithms and interpretable machine learning models together to
find a population of near-Pareto optimal solutions to a multiobjective optimization problem.
This combination of evolutionary algorithms and machine learning gives raise to two modes:
a *Darwininan mode* and a *learning mode*. These modes have been illustrated in :ref:`Figure 1<modes>`.

.. _modes:

.. figure:: figures/darwinlearning.svg
    :alt: A figure depicting the Darwinian and learning modes.
    :align: center

    Figure 1: A Darwinian and learning mode depicted. From [Misitano2023]_.

In utilizing interpretable machine learning models, we have the ability to build a rudimentary understanding
on what kind of solutions in a population are *good* and what kind are less good. In practice, this means
finding what kind of decision variables are needed to produce near-Pareto optimal solutions.
This understanding gives raise to the explainable nature of the XLEMOO approach.

For futher details and a more in-depth description of the XLEMOO approach, please see [Misitano2023]_.

Getting started
===============

In this section, instuctions are given to install the XLEMOO frameowrk and verify it is working.

.. note::

    These instructions have been verified to work on a Linux-based operating system. They should
    apply to other \*nix systems as well and Windows.

    Tools required to install the frameowrk according to the current documentation are 
    `git`_, `Poetry`_, and Python version 3.9 or 3.10.
    It is highly advised that users utilize Poetry to install the framework.

Installation
------------

Begin by cloning the XLEMOO repository and chaning changing the workind directory to the root of the repository:

.. code-block:: shell

    $ git clone https://github.com/gialmisi/XLEMOO
    $ cd XLEMOO

Next, create a new virtual environment with poetry and switch to it:

.. code-block:: shell

    $ poetry shell

Then, install the frameowork with the command:

.. code-block:: shell

    $ poetry install

or with development dependencies included:

.. code-block:: shell

    $ poetry install --with dev


The XLEMOO framework should now be installed locally on your machine. 

Tests
-----

XLEMOO utilized `pytest`_ for unit testing, which is included in the development dependencies. Before continuing,
make sure development dependencies are installed:

.. code-block:: shell

    $ poetry install --with dev

To run the unit tests, run:

.. code-block:: shell

    $ pytest --reruns 5

.. note::

    The ``--reruns 5`` options is used to ensure that some tests are run multiple times in case of failure. Because
    of the heuristic nature of some computations, all tests may not always pass due to some numerical checks.
    This is expected.

If everything is working as expected, the tests should all pass with no errors (some warnings are expected).


Hacking the framework
=====================

Citation
========

If you utilize the XLEMOO frameowork in your own work, it would be greatly appreciated if you cited
the publication [Misitano2023]_.

References
==========

.. note::

    References will be updated when published.

.. [Misitano2023]
    Misitano, G. (2023). Exploring the Explainable Aspects and Performance of a Learnable Evolutionary Multiobjective Optimization Method. ACM Transactions on Evolutionary Learning and Optimization. To be published.

.. _git: https://git-scm.com/
.. _Poetry: https://python-poetry.org/
.. _pytest: https://docs.pytest.org/en/7.3.x/