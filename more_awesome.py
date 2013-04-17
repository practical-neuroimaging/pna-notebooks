#!/usr/bin/env python
""" The line above tells Unix systems (Linux, OSX) this is a Python script

http://en.wikipedia.org/wiki/Shebang_(Unix)

You can run the tests on this file with:

    nosetests awesome.py
"""
# sys is a module that comes with Python in the standard library
import sys

# Numpy is the array processing package
import numpy as np

# Nibabel is a package for loading neuroimaging files
import nibabel as nib

# A global variable (a dictionary) to store stuff we'll use for the tests
_TEST_STUFF = {}

def set_test_stuff():
    # We need to declare the global variable
    global _TEST_STUFF
    fname = 'bold.nii.gz' # From two_example_images.zip
    img = nib.load(fname)
    data = img.get_data()
    _TEST_STUFF['data'] = data


def difference_rms(img_arr):
    """ Calculate the root mean square differences for an array

    This is a "docstring".  It gives some help about what a function does.  See:

    http://www.pythonforbeginners.com/basics/python-docstrings/
    """
    # Make the array a floating point type to avoid overflowing, if this is an
    # integer image (data type is integer)
    data = img_arr.astype(np.float32)
    difference_volumes = np.diff(data, axis=3)
    sq_diff = difference_volumes ** 2
    n_voxels = np.prod(img_arr.shape[:-1])
    n_scans = img_arr.shape[-1]
    vox_by_scans = np.reshape(sq_diff, (n_voxels, n_scans - 1))
    scan_means = np.mean(vox_by_scans, axis=0)
    return np.sqrt(scan_means)


def test_diff_rms():
    # Test the RMS
    global _TEST_STUFF # Get the variable storing the test variables
    if not 'data' in _TEST_STUFF: # Check if we've set the state
        set_test_stuff()
    data = _TEST_STUFF['data']
    result = difference_rms(data)
    assert np.allclose(result[:4],
                       np.array([21.37829399,  21.55786514,  22.90981674,
                                 21.0092392]))

def replace_vol(img_arr, vol_no):
    """ Replace volume number `vol_no` with mean of vols either side

    The arguments we pass in are ``img_arr`` (a 4D array) and ``vol_no``, in
    integer, giving the volume index (starting at 0)
    """
    n_vols = img_arr.shape[-1]
    if vol_no < 0: # Deal with negative indices
        n_vols = img_arr.shape[-1]
        vol_no = n_vols + vol_no
    # We need to copy the original data, ``img_arr``, otherwise we would
    # overwrite it.  We also need the data to be floating point type.  The
    # following command will copy the data, and make it into floating point
    data = np.array(img_arr, dtype=np.float32)
    # Deal with first and last
    if vol_no == 0:
        new_vol = data[..., 1]
    elif vol_no == n_vols - 1:
        new_vol = data[..., -2]
    else:
        # Take the mean of volumes either side
        left = data[..., vol_no - 1]
        right = data[..., vol_no + 1]
        new_vol = (left + right) / 2.0
    # Replace volume 65 with the mean
    data[..., vol_no] = new_vol
    return data


def test_replace_vol():
    # Test routine to replace volumes
    global _TEST_STUFF # Get the variable storing the test variables
    if not 'data' in _TEST_STUFF: # Check if we've set the state
        set_test_stuff()
    data = _TEST_STUFF['data']
    fixed = replace_vol(data, 65) # Call our new function
    assert not np.all(fixed == data) # We changed the array
    mean_either_side = np.mean(data[:, :, :, [64, 66]], axis=3)
    assert np.all(fixed[:, :, :, 65] == mean_either_side) # This should work


"""
This stuff below only gets run, if you run this script from the command line
with any of::

    python awesome.py
    ./awesome.py

    ipython (then)
    [1] run awesome.py

See : http://stackoverflow.com/questions/419163/what-does-if-name-main-do
"""
if __name__ == '__main__':
    # We get the first thing entered on the command line calling us.  If there
    # is nothing on this command line, this will raise an IndexError, because
    # there is not a ``sys.argv[1]`` to find.
    # See http://docs.python.org/2/library/sys.html and
    # http://stackoverflow.com/questions/4117530/sys-argv1-meaning-in-script
    # for some explanation
    system_fname = sys.argv[1]
    system_image = nib.load(system_fname)
    system_data = system_image.get_data()
    print("Volume name is " + system_fname)
    print("RMS for volume:")
    print(difference_rms(system_data))
