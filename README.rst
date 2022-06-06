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

    git clone https://github.com/dynamicslab/pysindy.git

Then run the install script:

.. code-block:: bash
    pip install .




Related packages
----------------

- **BDpack** <http://amir-saadat.github.io/BDpack> Brownain dynamics in Fortran.
- **QPolymer** <https://sourceforge.net/projects/qpolymer/> Qt-based GUI for
  polymer dynamics.
