import numpy as np
import dilutebrowniandynamics.simulate as dbds


def test_sum_dict():
    """Sum observables from two dictionaries."""
    dicts = [{'A': np.ones((3, 3)), 'B': 1.},
             {'A': 2*np.ones((3, 3)), 'B': 1.}]
    weights = [1., 0.5]
    result = dbds._sum_dict(dicts, weights)
    expected = {'A': 2*np.ones((3, 3)), 'B': 1.5}
    for key, value in result.items():
        assert np.allclose(value, expected[key])


def test_dict_series():
    dicts = [{'A': np.ones((3, 3)), 'B': 1.},
             {'A': 2*np.ones((3, 3)), 'B': 1.}]
    result = dbds._dict_series(dicts)
    expected = {'A': np.array([np.ones((3, 3)), 2*np.ones((3, 3))]),
                'B': [1., 1.]}
    for key, value in result.items():
        assert np.allclose(value, expected[key])
