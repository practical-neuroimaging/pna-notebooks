import numpy as np

import nibabel as nib

fname = 'bold.nii.gz' # From two_example_images.zip
img = nib.load(fname)
data = img.get_data()
data = data.astype(np.float32) # We need higher precision for our calculations


def difference_rms(img_arr):
    data = img_arr.astype(np.float32)
    difference_volumes = np.diff(data, axis=3)
    sq_diff = difference_volumes ** 2
    n_voxels = np.prod(img.shape[:-1])
    n_scans = img.shape[-1]
    vox_by_scans = np.reshape(sq_diff, (n_voxels, n_scans - 1))
    scan_means = np.mean(vox_by_scans, axis=0)
    return np.sqrt(scan_means)


def test_diff_rms():
    result = difference_rms(data)
    assert np.allclose(result[:4],
                       np.array([21.37829399,  21.55786514,  22.90981674,
                                 21.0092392]))

def replace_vol(img_arr, vol_no):
    """ Replace volume number `vol_no` with mean of vols either side """
    # Copy the original data, ``img_arr``
    data = img_arr.copy()
    # Take the mean of volumes either side
    left = data[..., vol_no - 1]
    right = data[..., vol_no + 1]
    mean_either_side = (left + right) / 2.0
    # Replace volume 65 with the mean
    data[..., vol_no] = mean_either_side
    return data


def test_replace_vol():
    fixed = replace_vol(data, 65) # Call our new function
    assert not np.all(fixed == data) # We changed the array
    mean_either_side = np.mean(data[:, :, :, [64, 66]], axis=3)
    assert np.all(fixed[:, :, :, 65] == mean_either_side) # This should work
