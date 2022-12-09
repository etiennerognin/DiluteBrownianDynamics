# Base file to simulate a bunch of molecules
import numpy as np
from multiprocessing import Pool
import tqdm
from functools import partial

SUBIT_MAX_LEVEL = 20


class ConvergenceError(Exception):
    """Exception to handle convergence error in simulation."""
    pass


def simulate_batch(molecules, gradU, dt, n_steps,
                   write_interval=1, no_average=None, n_proc=4):
    """Simulate forward in time a batch of molecules.

    Parameters
    ----------
    molecules : list of Molecule objects
        Molecules to simulate.
    gradU : {callable, ndarray (3, 3)}
        Velocity gradient (3, 3) for the simulation. If `gradU` is
        callable, then it will be evaluated at each time step.
    dt : float
        Dimensionless time step.
    n_steps: int
        Number of time steps to simulate.
    write_interval: int, default 1
        Record data every ``write_interval`` steps.
    no_average : list of str
        Molecular observable to exclude from batch averaging.
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

    print("Physical time to compute:", n_steps*dt)

    simulate_para = partial(simulate,
                            gradU=gradU,
                            dt=dt,
                            n_steps=n_steps,
                            write_interval=write_interval
                            )

    with Pool(n_proc) as p:
        print("Calculation started on {} cores.".format(n_proc))
        results = list(tqdm.tqdm(p.imap(simulate_para, molecules),
                                 total=n_ensemble))

    # Compute average and standard deviation of observables
    observables_list = [obs for obs, molecule in results]
    molecules_out = [molecule for obs, molecule in results]
    observables = _statistics(observables_list, no_average)

    return observables, molecules_out


def simulate(molecule, gradU, dt, n_steps,
             write_interval=1, full_trajectory=False, progress=False):
    """Simulate a molecule and collect data.

    Parameters
    ----------
    molecule : Molecule object
        Molecule to simulate
    gradU : {callable, ndarray shape (3, 3)}
        Velocity gradient (3, 3) for the simulation. If `gradU` is callable,
        then it will be evaluated at each time step.
    dt : float
        Dimensionless time step.
    n_steps: int
        Number of time steps to simulate.
    write_interval: int, default 1
        Record data every ``write_interval`` steps.
    full_trajectory : bool, default False
        If True returns trajectory as a list of Molecule at each time step.
    progress : bool, default False
        If True, display tqdm progress bar.

    Returns
    -------
    observables : dict
        Dictionary of measured quantities (depends on model).
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
        iterations = tqdm.tqdm(range(n_steps))
    else:
        iterations = range(n_steps)

    for i in iterations:
        # print(i)
        subit = 0.    # Part of the time step job done
        first_subit = True  # Flag to indicate first sub-iteration
        if i % write_interval == 0:
            subobs = []   # Collection of observables which will be averaged
            weights = []

        while subit < dt - dt/2**(SUBIT_MAX_LEVEL+1):

            # Evaluate velocity gradient
            if callable(gradU):
                gradUt = gradU(i*dt+subit)
            else:
                gradUt = gradU

            try:
                # Solve internal tensions with constraints.
                molecule.solve(gradUt, dt_local)
                # Measure whatever the model is set to output.
                subobs.append(molecule.measure())
                weights.append(0.5**level/write_interval)
                if full_trajectory and (first_subit and
                                        (i+1) % write_interval == 0):
                    trajectory.append(copy.deepcopy(molecule))
                # Evolve the model by one time step.
                molecule.evolve(first_subit=first_subit)
                first_subit = False

                # If this is a success, it means we can increment time:
                subit += dt_local

            except ConvergenceError:
                # Fail to evolve molecule. We subdivide the time step to
                # increase stability and evolve again.
                if level < SUBIT_MAX_LEVEL:
                    level += 1
                    dt_local = dt/2**level
                else:
                    raise RuntimeError("Convergence failed and maximum level "
                                       "of time step subdivision reached.")
        if (i+1) % write_interval == 0:
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


def _statistics(dicts, no_average):
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
        if no_average is not None and key in no_average:
            statistics[key] = frame
    return statistics
