""" Do slice timing on scans """

import os

import numpy as np

import nibabel as nib

import scipy.interpolate as spi


def pad_ends(first, middle, last):
    """ Pad array `middle` along last axis with `first` value and `last` value
    """
    middle = np.array(middle) # Make sure middle is an array
    pad_ax_len = middle.shape[-1] # Length of the axis we are padding
    pad_shape = middle.shape[:-1] + (pad_ax_len + 2,) # Shape of the padded array
    padded = np.empty(pad_shape, dtype=middle.dtype) # Padded array ready to fill
    padded[..., 0] = first
    padded[..., 1:-1] = middle
    padded[..., -1] = last
    return padded


def interp_slice(old_times, slice_nd, new_times, kind='linear'):
    """ Interpolate a 3D slice `slice_nd` with times changing from `old_times` to `new_times`
    """
    n_time = slice_nd.shape[-1]
    assert n_time == len(old_times)
    padded_times = pad_ends(old_times[0] - (old_times[1] - old_times[0]), 
                        old_times,
                        old_times[-1] + (old_times[-1] - old_times[-2]))
    to_interpolate = pad_ends(slice_nd[..., 0], slice_nd, slice_nd[..., -1])
    interpolator = spi.interp1d(padded_times, to_interpolate, kind, axis=-1)
    return interpolator(new_times)


def slice_time_image(img, slice_times, TR, kind='cubic'):
    """ Take nibabel image `img` and run slice timing correction using `slice_times`
    """
    data = img.get_data()
    assert len(slice_times) == img.shape[-2]
    n_scans = img.shape[-1]
    scan_starts = np.arange(n_scans) * TR
    interp_data = np.empty(data.shape)
    desired_times = scan_starts
    for slice_no in range(data.shape[-2]):
        these_times = slice_times[slice_no] + scan_starts
        data_slice = data[:, :, slice_no, :]
        interped = interp_slice(these_times, data_slice, desired_times, kind)
        interp_data[:, :, slice_no, :] = interped
    new_img = nib.Nifti1Image(interp_data, img.get_affine(), img.get_header())
    return new_img


def slice_time_file(fname, slice_times, TR, kind='cubic'):
    pth, fname = os.path.split(fname)
    new_fname = os.path.join(pth, 'a' + fname)
    raw_img = nib.load(fname)
    interp_img = slice_time_image(raw_img, slice_times, TR, kind)
    nib.save(interp_img, new_fname)
