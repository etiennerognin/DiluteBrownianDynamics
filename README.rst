DiluteBrownianDynamics
======================

**Dilute Brownian Dynamics** simulation package written in Python, with
readability and extensibility as the main goals. It is primarily intended for
testing toy models in polymer science.

*Dilute* means that molecules are simulated individually, allowing simple data
structure (there is no *simulation box*) and parallel support. A variety of
molecule models are already implemented.

Beta version.

Installation
------------

In the target directory, clone this repository:

.. code-block:: bash

    git clone https://github.com/dynamicslab/pysindy.git

Then run the install script:

.. code-block:: bash

    pip install .

Examples
--------
See the Jupyter notebooks in the ``/examples`` folder.

Developing new models
---------------------
A straightforward approach is to copy and modify an existing model in the
``/dilutebrowniandynamics/molecules`` folder. A model is a Python class which
in addition to constructors should provide at least three methods:

``solve``: solve tensions and other forces given flow field and constraints.

``measure``: compute and return in a Python dictionary what needs to be
  recorded at each time step (for example the moment of forces for the stress
  tensor estimator).
    
``evolve``: evolve the system to the next step given the forces and flow field,
  and draw new random forces.

The main simulation loop will concatenate each measured observable into a
dictionary of time series. Then if an ensemble of molecules is simulated, average and
standard deviation of these series with respect to the ensemble are
computed. In this way there is full flexibility towards what each model should
compute and record. 

Related packages
----------------

- **BDpack** <http://amir-saadat.github.io/BDpack> Brownain dynamics in Fortran.
- **QPolymer** <https://sourceforge.net/projects/qpolymer/> Qt-based GUI for
  polymer dynamics.
