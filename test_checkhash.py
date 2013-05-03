""" Test Utilities for checking hashes of 3D volumes in 4D images
"""

from os.path import join as pjoin

import tempfile
import shutil

import numpy as np

import nibabel as nib

from checkhash import check_store_hash

from nose.tools import assert_equal

def test_check_store_hash():
    data1 = np.random.normal(size=(2, 3, 4, 5))
    data2 = np.random.normal(size=(2, 3, 4, 5))
    img1 = nib.Nifti1Image(data1, None)
    fname1 = 'fname1.nii'
    img2 = nib.Nifti1Image(data2, None)
    fname2 = 'fname2.nii'
    hash_dict = {}
    tmpdir = tempfile.mkdtemp()
    try:
        # Save the new data to a temporary directory
        fname1 = pjoin(tmpdir, 'fname1.nii')
        nib.save(img1, fname1)
        fname2 = pjoin(tmpdir, 'fname2.nii')
        nib.save(img2, fname2)
        # Check there's no duplicates in first image
        duplicates = check_store_hash(fname1, hash_dict)
        assert_equal(len(duplicates), 0)
        assert_equal(len(hash_dict), 5) # Number of volumes checked
        # Check none in first *or* second
        duplicates = check_store_hash(fname2, hash_dict)
        assert_equal(len(duplicates), 0)
        assert_equal(len(hash_dict), 10) # Number of checked
        # But rechecking second - duplicates
        duplicates = check_store_hash(fname2, hash_dict)
        assert_equal(len(duplicates), 5)
        assert_equal(len(hash_dict), 10) # Number of checked
        del img1, img2
    finally: # Clean up temporary directory
        shutil.rmtree(tmpdir)
