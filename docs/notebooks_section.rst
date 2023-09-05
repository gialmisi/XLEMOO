.. _notebooks:

Examples
========

The examples below showcase how the XLEMOO framework can be utilized. For most of its multiobjective
optimization needs, XLEMOO utilizes the DESDEO framework [Misitano2021]_. 
Particularly, in the below examples, the test problems from the `desdeo-problem`_ package are used. The scalarization
functions are in turn utilized from the `desdeo-tools`_ package, while the
evolutionary operators utilized are from the `desdeo-emo`_ package. The DESDEO framework provides many more
additional multiobjective optimization problems, scalarization functions, and
evolutionary operators that may be
utilized with the XLEMOO framework. The problems, functions, and operators considered in these examples
are explained in more detail in the XLMEOO article [Misitano2023a]_.

.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   notebooks/Showcase
   notebooks/How_to_extract_rules_example

.. _desdeo-problem: https://desdeo-problem.readthedocs.io/en/latest/problems.html 
.. _desdeo-tools: https://desdeo-tools.readthedocs.io/en/latest/autoapi/scalarization/index.html
.. _desdeo-emo: https://desdeo-emo.readthedocs.io/en/latest/autoapi/desdeo_emo/recombination/index.html 