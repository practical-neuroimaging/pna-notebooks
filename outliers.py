""" Outlier detection
"""

import numpy as np
import scipy.stats as sst

from nose.tools import assert_equal, assert_true

"""
Algorithm

1. select h observation
2. compute sig and mu
3. compute Malhanobis distance
4. take the h observ that have smallest Md
5. Stop criteria: compare with previous sigma and mu

Loop - or stop
"""

def compute_mu_var(y):
    """ Compute mean and (co)variance for one or more measures

    Parameters
    ----------
    y : (N,) or (P, N) ndarray
        One row per measure, one column per observation. If a vector, treat as a
        (1, N) array

    Returns
    -------
    mu : (P,) ndarray
        Mean of each measure across columns.
    var : (P, P) ndarray
        Variances (diagonal) and covariances of measures
    """
    # Make sure a vector becumes a 1, N 2D array
    y = np.atleast_2d(y)
    # Degrees of freedom
    P, N = y.shape
    df = N - 1
    # Mean for each row
    mu = y.mean(axis=1)
    # Remove mean
    yc = y - mu
    # Variance(s) and covariances
    var = yc.dot(yc.T) / df
    return mu, var


def test_compute_mu_var():
    mu, var = compute_mu_var([[1, 1, 1, 1]])
    assert_equal(mu, 1)
    assert_equal(var, 0)
    mu, var = compute_mu_var([[-1, 0, 1]])
    assert_equal(mu, 0)
    assert_equal(var, 1)


def compute_mahal(y, mu, var):
    """ Compute Mahalanobis distance for `y` given `mu`, `var`

    Parameters
    ----------
    y : (N,) or (P, N) ndarray
        One row per measure, one column per observation. If a vector, treat as a
        (1, N) array
    mu : (P,) ndarray
        Mean of each measure across columns
    var : (P, P) ndarray
        Variances (diagonal) and covariances of measures

    Returns
    -------
    mahals : (N,) ndarray
        Mahalanobis distances of each observation from `mu`, given `var`
    """
    y = np.atleast_2d(y)
    # Mean correct
    yc = y - mu
    # Correct for (co)variances. For single row, this is the same as dividing by
    # the variance
    y_white = np.linalg.inv(var).dot(yc)
    # Z score for one row is (y - mu) / sqrt(var).
    # Z**2 is (y - mu) (y-nu) / var, which is:
    z2 = yc * y_white
    # Mahalanobis distance is mean z2 over measures
    return z2.mean(axis=0)


def test_mahal():
    tmparr1 = np.asarray([[-1, 0, 1]])
    npmu = np.asarray([[1]])
    npvar = np.asarray([[1]])
    assert_true(np.allclose(
        compute_mahal(tmparr1, npmu, npvar ), np.array([ 4.,  1.,  0.])))


# Initialize
n = 100
sigma = 1.
np.random.seed(42)
Y = np.random.normal(0.,sigma,size=(1,n))

pc = .70
h = int(pc * n)

outliers = np.random.normal(0.,sigma*10,size=(1,n-h))
Y[:,0:n-h] = outliers


def estimate_mu_var(y, n_inliers=0.7, maxiter=10, tol=1e-6):
    """ Estimate corrected `mu` and `var` for `y` given number of inliers

    Algorithm from:

    Fritsch, V., Varoquaux, G., Thyreau, B., Poline, J. B., & Thirion, B. (2012).
    Detecting outliers in high-dimensional neuroimaging datasets with robust
    covariance estimators. Medical Image Analysis.

    Parameters
    ----------
    y : (N,) or (P, N) ndarray
        One row per measure, one column per observation. If a vector, treat as a
        (1, N) array
    n_inliers : int or float, optional
        If int, the number H (H < N) of observations in the center of the
        distributions that can be assumed to be non-outliers. If float, should
        be between 0 and 1, and give proportion of inliers
    maxiter : int, optional
        Maximum number of iterations to refine estimate of outliers
    tol : float, optional
        Smallest change in variance estimate for which we contine to iterate.
        Changes smaller than `tol` indicate that iteration of outlier detection
        has converged

    Returns
    -------
    mu : (P,) ndarray
        Mean per measure in `y`, correcting for possible outliers
    var : (P, P) ndarray
        Variances (diagonal) and covariances (off-diagonal) for measures `y`,
        correcting for possible outliers
    """
    y = np.atleast_2d(y)
    P, N = y.shape
    if n_inliers > 0 and n_inliers < 1: # Proportion of inliers
        n_inliers = int(np.round(N * n_inliers))
    if n_inliers <= 0:
        raise ValueError('n_inliers should be > 0')
    # Compute first estimate of mu and varances
    mu, var = compute_mu_var(y)
    if n_inliers >= N:
        return mu, var
    # Initialize estimate of which are the inlier values.
    prev_inlier_indices = np.arange(n_inliers)
    # Keep pushing outliers to the end until we are done
    for i in range(maxiter):
        distances = compute_mahal(y, mu, var)
        # Pick n_inliers observatons with lowest distances
        inlier_indices = np.argsort(distances)[:n_inliers]
        # If we found the same inliers as last time, we'll get the same mu, var,
        # so stop iterating
        if np.all(inlier_indices == prev_inlier_indices):
            break
        # Re-estimate mu and var with new inliers
        mu_new, var_new = compute_mu_var(y[:, inlier_indices])
        # Check if var has changed - if not - stop iterating
        if np.max(np.abs(var - var_new)) < tol:
            break
        # Set mu, var, indices for next loop iteration
        var = var_new
        mu = mu_new
        prev_inlier_indices = inlier_indices
    return mu, var


mu, var = estimate_mu_var(Y)

print mu, var

# choose a false positive rate - Bonferroni corrected for number of samples
alpha = 0.1/n

# Standard deviation - for our 1D case
sig = np.sqrt(var)
# z - in 1D case
z = (Y - mu) / sig

# _out = plt.hist(z[0,:])

# Normal distribution object
normdist = sst.norm(mu, sig)
# Probability of z value less than or equal to each observation
p_values = normdist.cdf(z)
# Where probability too high therefore z value too large.  We're only looking
# for z's that are too positive, disregarding zs that are too negative
some_bad_ones = p_values > (1 - alpha)

# Show what we found
print "volumes to remove :", some_bad_ones
print z[some_bad_ones]
print Y[some_bad_ones]

# <codecell>

# exercice : 
# print histogram of the good ones:

# <codecell>

# scrap cell
#========================================#

# here - just one variable
#z0 = z[0,:]
#mu0 = mu[0]
#sig0 = sig[0,0]
#print good_ones
#good_ones = np.where(some_bad_ones == False)
#print good_ones.shape

# <codecell>


