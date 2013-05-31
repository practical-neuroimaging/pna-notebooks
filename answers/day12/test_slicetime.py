""" Testing slicetime module
"""
import os

# For temporary files and directories
import tempfile

# For deleting the temporary directories
import shutil

import numpy as np

import scipy.interpolate as spi

import nibabel as nib

from slicetime import pad_ends, interp_slice, slice_time_image, slice_time_file

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_pad_ends():
    # Test pad_ends routine
    # 1D case
    assert_array_equal(pad_ends(0, [2, 3], 5), [0, 2, 3, 5])
    # 2D case
    a = np.zeros((5, ))
    b = np.ones((5, 4)) * 10
    c = np.ones((5, )) * 99
    assert_array_equal(pad_ends(a, b, c),
                       np.concatenate((a[..., None], b, c[..., None]),
                                      axis=-1))
    # 3D case
    a = np.zeros((2, 3))
    b = np.ones((2, 3, 4)) * 10
    c = np.ones((2, 3)) * 99
    assert_array_equal(pad_ends(a, b, c),
                       np.concatenate((a[..., None], b, c[..., None]),
                                      axis=-1))


def test_interp_slice():
    # Test interpolation over 3D slices
    data = np.random.normal(size=(4, 5, 6))
    old_times = np.arange(6) * 2.5
    # Add some random jitter
    new_times = old_times + np.random.normal(0, 0.1, size=(6,))
    # Do manual thing
    pad_front = new_times[0] < old_times[0]
    pad_back = new_times[-1] > old_times[-1]
    if pad_front and pad_back:
        pad_times = [new_times[0]] + list(old_times) + [new_times[-1]]
        pad_data = np.concatenate((data[..., 0:1], data, data[..., 5:6]),
                                  axis=-1)
    elif pad_front:
        pad_times = [new_times[0]] + list(old_times)
        pad_data = np.concatenate((data[..., 0:1], data), axis=-1)
    elif pad_back:
        pad_times = list(old_times) + [new_times[-1]]
        pad_data = np.concatenate((data, data[..., 5:6]), axis=-1)
    else:
        pad_times = old_times
        pad_data = data
    for order in ('linear', 'cubic', 1, 4):
        interper = spi.interp1d(pad_times, pad_data, order, axis=-1)
        interped = interper(new_times)
        assert_almost_equal(interped,
                            interp_slice(old_times, data, new_times, order))


def test_slice_time_image():
    # Test slice timing on image in memory
    n_slices = 4
    n_vols = 5
    TR = 3.5
    vol_times = np.arange(n_vols) * TR
    # Add some random jitter per slice
    slice_times = np.arange(n_slices) * TR / n_slices
    slice_times = slice_times + np.random.normal(0, 0.05, size=(n_slices,))

    # TEST AXIAL
    data = np.random.normal(size=(2, 3, n_slices, n_vols))
    img = nib.Nifti1Image(data, np.eye(4))
    # Test against interpolation with interp_slice
    for order in ('linear', 'cubic', 1, 4):
        interped = np.zeros_like(data)
        for slice_no in range(n_slices):
            orig_times = vol_times + slice_times[slice_no]
            interped[:, :, slice_no, :] = interp_slice(
                orig_times, data[:, :, slice_no, :], vol_times, order)
        interped_img = slice_time_image(img, slice_times, TR, order)
        assert_almost_equal(interped_img.get_data(), interped)

    # TEST SAGITTAL
    data = np.random.normal(size=(n_slices, 2, 3, n_vols))
    img = nib.Nifti1Image(data, np.eye(4))
    for order in ('linear', 'cubic', 1, 4):
        interped = np.zeros_like(data)
        for slice_no in range(n_slices):
            orig_times = vol_times + slice_times[slice_no]
            interped[slice_no, ...] = interp_slice(
                orig_times, data[slice_no, ...], vol_times, order)
        interped_img = slice_time_image(img, slice_times, TR, order,
                                        slice_axis=0)
        assert_almost_equal(interped_img.get_data(), interped)


def test_slice_time_file():
    # Test slice_time_file from slice_time_image
    n_slices = 5
    n_vols = 4
    TR = 2.0
    data = np.random.normal(size=(2, 3, n_slices, n_vols))
    img = nib.Nifti1Image(data, np.eye(4))
    slice_times = np.arange(n_slices) * TR / n_slices + 0.1
    tmpdir = tempfile.mkdtemp()
    try:
        fname = os.path.join(tmpdir, 'myfile.nii')
        a_fname = os.path.join(tmpdir, 'amyfile.nii')
        nib.save(img, fname)
        for order in ('linear', 'cubic', 1, 4):
            interped_img = slice_time_image(img, slice_times, TR, order)
            slice_time_file(fname, slice_times, TR, order)
            a_img = nib.load(a_fname)
            assert_almost_equal(interped_img.get_data(), a_img.get_data())
            del a_img
    finally:
        shutil.rmtree(tmpdir)
