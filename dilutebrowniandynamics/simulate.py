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
    observables: dict
        Dictionary of time series.
    molecules_out: list of Molecule objects
        List of molecules after the last time step.
    """
    n_ensemble = len(molecules)

    print("Physical time to compute:", n_rec*dt)

    simulate_para = partial(simulate,
                            gradU=gradU,
                            n_rec=n_rec,
                            dt=dt,
                            full_trajectory=False)

    with Pool(n_proc) as p:
        print("Calculation started on {} cores.".format(n_proc))
        results = list(tqdm.tqdm(p.imap(simulate_para, molecules),
                                 total=n_ensemble))

    # Compute average and standard deviation of observables
    observables_list = [obs for obs, molecule in results]
    molecules_out = [molecule for obs, molecule in results]
    observables = _statistics(observables_list)

    return observables, molecules_out


def simulate(molecule, gradU, n_rec, dt, full_trajectory, progress=False):
    """Simulate a molecule and collect data.

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
    progress : bool, default False
        If True, display tqdm progress bar

    Returns
    -------
    observables : dict
        Dictionary of measured quantities (depends on model)
    molecule_out: Molecule object
        Molecule after the last time step, or full list at each time step.
    """

    # Data output
    observables = []

    if full_trajectory:
        import copy
        trajectory = []

    # Time step subdivision level
    level = 0
    dt_local = dt

    if progress:
        iterations = tqdm.tqdm(range(n_rec))
    else:
        iterations = range(n_rec)

    for i in iterations:
        subit = 0.    # Part of the time step job done
        first_subit = True  # Flag to indicate first sub-iteration
        subobs = []   # Collection of observables which will be averaged
        weights = []
        while subit < dt:

            # Evaluate velocity gradient
            gradUt = gradU(i*dt+subit) if callable(gradU) else gradU

            try:
                # Solve internal tensions with constraints.
                molecule.solve(gradUt, dt_local)
                # Measure whatever the model is set to output.
                subobs.append(molecule.measure())
                weights.append(0.5**level)
                if full_trajectory and subit == 0.:
                    trajectory.append(copy.deepcopy(molecule))
                # Evolve the model by one time step.
                molecule.evolve(first_subit=first_subit)
                first_subit = False

                # If this is a success, it means we can increment time:
                subit += dt_local

            except ValueError:
                # Fail to evolve molecule. We subdivide the time step to
                # increase stability and evolve again.
                if level < SUBIT_MAX_LEVEL:
                    level += 1
                    dt_local = dt/2**level
                else:
                    raise RuntimeError("Convergence failed and maximum level "
                                       "of time step subdivision reached.")

        observables.append(_sum_dict(subobs, weights))

        if level > 0:
            level += -1
            dt_local = dt/2**level

    observables = _dict_series(observables)
    if full_trajectory:
        return observables, trajectory
    else:
        return observables, molecule


def _sum_dict(dicts, weights):
    """Sum a collection of obsevables according to weights.

    Parameters
    ----------
    dicts : list of dict
        List of dictionaries which should all have the same data structure.
    weights : list of float
        List of weights for averaging

    Returns
    -------
    dict
        Average dictionary"""
    average = {}
    for key in dicts[0].keys():
        weighted_values = [w*dict_[key] for w, dict_ in zip(weights, dicts)]
        average[key] = np.sum(weighted_values, axis=0)
    return average


def _dict_series(dicts):
    """Concatenate dictionaries of ndarrays along a new axis.

    Returns
    -------
    dict
        Dictionary of time series.
    """
    series = {}
    for key in dicts[0].keys():
        series[key] = np.array([dict_[key] for dict_ in dicts])
    return series


def _statistics(dicts):
    """Computes statistics from a list of dicts.

    Returns
    -------
    dict
        Dictionary containing the statistics. `key_average` for the averaged
        values, `key_std` for standard deviation.
    """
    statistics = {}
    for key in dicts[0].keys():
        frame = np.array([dict_[key] for dict_ in dicts])
        statistics[f"{key}_average"] = np.average(frame, axis=0)
        statistics[f"{key}_std"] = np.std(frame, axis=0)
    return statistics
