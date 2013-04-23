""" Testing outlier detection
"""

import numpy as np

import outliers

# Only needed if working interactively
reload(outliers)

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_almost_equal


def test_compute_mu_var():
    # Test computation of maybe multivariable mean and variance
    assert_equal(outliers.compute_mu_var([[1, 1, 1, 1]]), (1, 0))
    assert_equal(outliers.compute_mu_var([[-1, 0, 1]]), (0, 1))
    # Make a random number generator, seed it to make numbers predictable
    rng = np.random.RandomState(42)
    vector = rng.normal(3, 7, size=(100,))
    mu, var = outliers.compute_mu_var(vector)
    assert_almost_equal(mu, vector.mean())
    # We used 1 df for the variance estimation
    assert_almost_equal(var, vector.var(ddof=1))
    # Does it also work for a 2D (row) vector?
    mu, var = outliers.compute_mu_var(vector.reshape((1, 100)))
    assert_almost_equal(mu, vector.mean())
    assert_almost_equal(var, vector.var(ddof=1))
    # A list ?
    mu, var = outliers.compute_mu_var(vector.tolist())
    assert_almost_equal(mu, vector.mean())
    assert_almost_equal(var, vector.var(ddof=1))
    # 2D matrix
    arr2d = rng.normal(3, 7, size=(2, 100))
    mu, var = outliers.compute_mu_var(arr2d)
    assert_almost_equal(mu, arr2d.mean(axis=1))
    demeaned = arr2d - mu[:, None]
    est_var = np.dot(demeaned, demeaned.T) / 99
    assert_almost_equal(var, est_var)


def test_mahal():
    # Test mahalanobis distance
    # Basic 1D check - lists as input, 2D row vector list
    assert_almost_equal(
        outliers.compute_mahal([[-1, 0, 1]], 1, 1), [ 4.,  1.,  0.])
    # Arrays as input, 1D vector
    assert_almost_equal(
        outliers.compute_mahal(np.array([-1, 0, 1]), np.array(1), np.array(1)),
        [ 4.,  1.,  0.])
    # For some random numbers
    rng = np.random.RandomState(42)
    vector = rng.normal(3, 7, size=(100,))
    distances = outliers.compute_mahal(vector, 3, 7)
    z = (vector - 3) / 7.
    assert_almost_equal(distances, z ** 2)


def test_estimate_mu_var():
    pass
