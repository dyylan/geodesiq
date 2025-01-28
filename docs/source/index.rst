Geodesiq documentation
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
========

Geodesiq is a Python package for compiling hardware-optimised quantum circuits. It provides tools for:

- Module `compiler`: Compiler class for quantum circuits, designed to handle compilation from a 
   Qiskit QuantumCircuit or QASM input, build circuit graphs, optimize gate parameters, and apply 
   fusion algorithms to reduce circuit complexity.
- Module `lie`: Lie algebra basis classes.
- Module `optimizer`: Uses the geodesic algorithm to find the Hamiltonian with restriction that 
   implements a desired unitary transformation.
- Module `utils`: Utility functions.

Installation
============

You can install Geodesiq using pip:

.. code-block:: bash 
   
   pip install geodesiq


Modules
=======

The following modules are available in Geodesiq:

.. automodule:: geodesiq.compiler
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: geodesiq.lie
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: geodesiq.optimizer
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: geodesiq.utils
    :members:
    :undoc-members:
    :show-inheritance:
