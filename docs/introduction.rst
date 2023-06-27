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

Installation
------------

Tests
-----

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
