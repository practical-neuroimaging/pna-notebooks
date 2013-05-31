import os

import numpy as np
import nibabel as nib
import scipy.interpolate as spi


def pad_ends(first, middle, last):
    """Pad last axis of array 'middle' with 'first' and 'last'.

    Add 'first' to the front and 'last' to the back.  If one of 'first'
    and 'last' is None, only the other is padded.  They can't both be
    None.

    Parameters
    ----------
    first: integer, double, array_like, or None
        If not None, element to be padded to the front of the last axis
        of 'middle'.  If 'middle' is three-dimensional, 'first' would be
        two-dimensional.  If 'middle' is two-dimensional, 'first' would
        be a single value.

    middle: array_like
        Array to which values should be padded.

    last: integer, double, array_like, or None
        If not None, element to be padded to the back of the last axis
        of 'middle'.  If 'middle' is three-dimensional, 'first' would be
        two-dimensional.  If 'middle' is two-dimensional, 'first' would
        be a single value.

    """
    middle = np.array(middle) 
    middle_shape = middle.shape
    last_axis_len = middle_shape[-1]
    padded_shape = list(middle_shape)
    if first is not None and last is not None:
        padded_shape[-1] = last_axis_len + 2
        padded = np.zeros(padded_shape)
        padded[..., 0] = first
        padded[...,1:-1] = middle
        padded[..., -1] = last
    elif first is not None:
        padded_shape[-1] = last_axis_len + 1
        padded = np.zeros(padded_shape)
        padded[..., 0] = first
        padded[..., 1:] = middle
    elif last is not None:
        padded_shape[-1] = last_axis_len + 1
        padded = np.zeros(padded_shape)
        padded[..., :-1] = middle
        padded[..., -1] = last
    return padded    


def interp_slice(old_times, slice_nd, new_times, kind='cubic'):
    """Interpolate a slice 'slice_nd' from 'old_times' to 'new_times'.

    Parameters
    ----------
    old_times: array_like
        Acquisition times for each point in the slice timeseries
        'slice_nd'.

    slice_nd: array_like
        Three-dimensional array representing values for a two-
        dimensional slice at each timepoint.

    new_times: array_like
        Times for which slice values are desired.

    kind: string or integer, optional
        Specifies the kind of interpolation as a string
        ('linear','nearest', 'zero', 'slinear', 'quadratic, 'cubic')
        or as an integer specifying the order of the spline interpolator
        to use. Default is 'cubic'.

    """
    n_time = slice_nd.shape[-1]
    assert n_time == len(old_times)
    pad_front = new_times[0] < old_times[0]
    pad_back = new_times[-1] > old_times[-1]
    if pad_front and pad_back:
        padded_times = pad_ends(new_times[0], old_times, new_times[-1])
        padded_slice_nd = pad_ends(slice_nd[..., 0], slice_nd,
                                   slice_nd[..., -1])
    elif pad_front:
        padded_times = pad_ends(new_times[0], old_times, None)
        padded_slice_nd = pad_ends(slice_nd[..., 0], slice_nd, None)
    elif pad_back:
        padded_times = pad_ends(None, old_times, new_times[-1])
        padded_slice_nd = pad_ends(None, slice_nd, slice_nd[..., -1])
    else:
        padded_times = old_times
        padded_slice_nd = slice_nd
    interpolator = spi.interp1d(padded_times, padded_slice_nd, kind, axis=-1,
                                copy=False) # Don't copy arrays to save memory
    return interpolator(new_times)


def slice_time_image(img, slice_times, TR, kind='cubic', slice_axis=2):
    """Run slice-timing correction on nibabel image 'img'.

    Parameters
    ----------
    img: nibabel image
        Image to be slice-time corrected.

    slice_times: array_like
        The time at which each slice was acquired relative to the start
        of the TR, listed in spatial order.

    TR: double
        Repetition time in seconds.

    kind: string or integer, optional
        Specifies the kind of interpolation as a string
        ('linear','nearest', 'zero', 'slinear', 'quadratic, 'cubic')
        or as an integer specifying the order of the spline interpolator
        to use. Default is 'cubic'.

    slice_axis: integer, optional
        Axis across which to perform slice-timing correction (with
        numbering starting at 0).

    Returns
    -------
    st_img : Nifti1Image
        A new copy of the input image with slice-time interpolation applied
    """
    data = img.get_data()
    n_dimensions = len(img.shape)
    assert len(slice_times) == img.shape[slice_axis] == data.shape[slice_axis]
    interp_data = np.empty(data.shape)
    scan_starts = np.arange(img.shape[-1]) * TR
    desired_times = scan_starts
    for slice_no in range(data.shape[slice_axis]):
        these_times = slice_times[slice_no] + scan_starts
        # We make the slice objects that correspond to indexing [:, :, :, :]
        slicer = [slice(None)] * n_dimensions
        # Change the : at the slice axis position to the slice number we want
        slicer[slice_axis] = slice_no
        # Get the data with the slicing list
        data_slice = data[slicer]
        # Do the interpolation
        interped = interp_slice(these_times, data_slice, desired_times, kind)
        # Put the data into the new array
        interp_data[slicer] = interped
    new_img = nib.Nifti1Image(interp_data, img.get_affine(), img.get_header())
    return new_img


def slice_time_file(fname, slice_times, TR, kind='cubic', slice_axis=2):
    """Perform slice-time correction on an image file.

    Write the corrected file as 'a' appended to the original file name.

    Parameters
    ----------
    fname: string
        Name of the image file (in any format nibabel.load accepts).

    slice_times: array_like
        The time at which each slice was acquired relative to the start
        of the TR, listed in spatial order.

    TR: double
        Repetition time in seconds.

    kind: string or integer, optional
        Specifies the kind of interpolation as a string
        ('linear','nearest', 'zero', 'slinear', 'quadratic, 'cubic')
        or as an integer specifying the order of the spline interpolator
        to use. Default is 'cubic'.

    slice_axis: integer, optional
        Axis across which to perform slice-timing correction (with
        numbering starting at 0).

    """
    path, f = os.path.split(fname)
    new_fname = os.path.join(path, 'a' + f)
    raw_img = nib.load(fname)
    interp_img = slice_time_image(raw_img, slice_times, TR, kind, slice_axis)
    nib.save(interp_img, new_fname)
