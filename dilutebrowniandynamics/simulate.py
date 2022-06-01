# Base file to simulate a bunch of molecules
import numpy as np
from multiprocessing import Pool
import tqdm
from functools import partial

SUBIT_MAX_LEVEL = 20


def simulate_batch(molecules, gradU, n_rec, dt, n_proc=4):
    """Simulate forward in time a batch of molecules.

    Parameters
    ----------
    molecules : list of Molecule objects
        Molecules to simulate.
    gradU : {callable, ndarray (3, 3)}
        Velocity gradient (3, 3) for the simulation. If `gradU` is
        callable, then it will be evaluated at each time step.
    n_rec : int
        Number of points to record.
    dt : float
        Dimensionless time step.
    n_proc : int, default 4
        Number of processor cores to use.

    Returns
    -------
    A : (Nrec, 3, 3) ndarray
        Time series of the covariance estimator of the end-to-end vectors
        (aka Conformation tensor).
    S: (Nrec, 3, 3) ndarray
        Time series of the estimator of the stress.
    molecules_out: list of Molecule objects
        List of molecules after the last time step.
    """
    n_ensemble = len(molecules)

    print("Physical time to compute:", n_rec*dt)
    A = np.zeros((n_rec, 3, 3))
    S = np.zeros((n_rec, 3, 3))
    molecules_out = []

    simulate_para = partial(simulate,
                            gradU=gradU,
                            n_rec=n_rec,
                            dt=dt,
                            full_trajectory=False)

    with Pool(n_proc) as p:
        print("Calculation started on {} cores.".format(n_proc))
        results = list(tqdm.tqdm(p.imap(simulate_para, molecules),
                                 total=n_ensemble))

    # Get elementary stress and add to compute estimators
    for elemA, elemS, molecule in results:
        A += elemA/n_ensemble
        S += elemS/n_ensemble
        molecules_out.append(molecule)

    return A, S, molecules_out


def simulate(molecule, gradU, n_rec, dt, full_trajectory):
    """Compute trajectory (in the modelcular dynamics sense) of a dumbbell.

    Parameters
    ----------
    molecule : Molecule object
        Molecule to simulate
    gradU : {callable, ndarray shape (3, 3)}
        Velocity gradient (3, 3) for the simulation. If `gradU` is callable,
        then it will be evaluated at each time step.
    n_rec : int
        Number of points to record.
    dt : float
        Dimensionless time step.
    full_trajectory : bool
        If True returns trajectory as a list of Molecule at each time step.

    Returns
    -------
    elemA : ndarray (n_rec, 3, 3)
        Time series of elementary conformation QQ.
    elemS: ndarray (n_rec, 3, 3)
        Elementary stress.
    molecule_out: Molecule object
        Molecule after the last time step, or full list at each time step.
    """

    # For parallel compatibility:
    np.random.seed()

    # Trajectories.
    elemA = np.zeros((n_rec, 3, 3))
    elemS = np.zeros((n_rec, 3, 3))
    if full_trajectory:
        import copy
        trajectory = []

    # Time step subdivision level
    level = 0
    dt_local = dt

    for i in range(n_rec):
        subit = 0   # Part of the time step job done
        while subit < dt:

            # Evaluate velocity gradient
            gradUt = gradU(i*dt+subit) if callable(gradU) else gradU

            try:
                elemA_loc, elemS_loc = molecule.evolve(gradUt, dt_local)

                # If this is a success, it means we can increment time:
                subit += dt_local

                # Averaging over sub-iterations (see variance in Ito calculus)
                elemA[i] += elemA_loc/(2**level)
                elemS[i] += elemS_loc/(2**level)

            except ValueError:
                # Fail to evolve molecule. We subdivide the time step to
                # increase stability and evolve again.
                if level < SUBIT_MAX_LEVEL:
                    level += 1
                    dt_local = dt/2**level
                else:
                    raise RuntimeError("Convergence failed and maximum level "
                                       "of time step subdivision reached.")

        if full_trajectory:
            trajectory.append(copy.deepcopy(molecule))
        if level > 0:
            level += -1
            dt_local = dt/2**level
    if full_trajectory:
        return elemA, elemS, trajectory
    else:
        return elemA, elemS, molecule
