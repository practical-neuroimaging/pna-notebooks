""" Some more tests for the awesome module

These tests don't work yet.

Put this file in the same directory as ``awesome.py``

Run the tests with::

    nosetests test_awesome_more.py

"""

# Numpy is the array processing package
import numpy as np

# We will need some useful testing routines from numpy
from numpy.testing import assert_array_equal

# Nibabel is a package for loading neuroimaging files
import nibabel as nib

# Load the routine from awesome.py
import awesome

# This loads the module again in case you have edited it since you started
# Python.  You don't need this command unless you are working interactively
reload(awesome)

# Declare global variables (variables in the module)
# Global variables should be ALL CAPS so we can see they are global and we don't
# get confused (like I did in class)
FNAME = 'bold.nii.gz'
IMG = nib.load(FNAME)
DATA = IMG.get_data()


def test_beginning_end_vols():
    # Test that we can replace the first and last volumes
    # This will not work with the current version of awesome - go fix!
    # Check we can fix the first volume in the series
    start_fixed = awesome.replace_vol(DATA, 0)
    assert_array_equal(start_fixed[:, :, :, 0], DATA[:, :, :, 1])
    # Check we can fix the last volume in the series
    n_scans = DATA.shape[-1]
    last_scan = n_scans - 1
    end_fixed = awesome.replace_vol(DATA, last_scan)
    assert_array_equal(end_fixed[:, :, :, last_scan],
                       DATA[:, :, :, last_scan-1])
    # Check we can use -1 or -2 for the last or second to last scan
    end_fixed = awesome.replace_vol(DATA, -1)
    assert_array_equal(end_fixed[:, :, :, last_scan],
                       DATA[:, :, :, last_scan-1])
    # If we're using -2, we want the mean of the 3rd to last and the last.
    mean_31 = (DATA[:, :, :, -3] + DATA[:, :, :, -1]) / 2.
    end_fixed = awesome.replace_vol(DATA, -2)
    assert_array_equal(end_fixed[:, :, :, last_scan-1],
                       mean_31)
