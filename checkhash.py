#!/usr/bin/env python
""" Script to compare raw data for volumes across runs
"""

from hashlib import sha1

import numpy as np

import nibabel as nib


def check_store_hash(fname, hash_dict):
    """ Get hash for contents of `fname`, check for duplicates in `hash_dict`

    Parameters
    ----------
    fname : str
        filename of 4D image file. We will get the hash of every 3D volume in
        the 4D series.
    hash_dict : dict
        dict with (key, value) pairs (sha1 hash, fname-volno) giving the
        filename and volume index corresponding to the sha1 hash of the contents

    Returns
    -------
    duplicate_pairs : list
        list of duplicates, which may be empty. Duplicate are of form
        (fname-volno for duplicate, fname-volno for original), where the
        original is the fname-volno previously in `hash_dict`
    """
    img = nib.load(fname)
    data = img.get_data().copy()
    duplicates = []
    for i, epi in enumerate(np.rollaxis(data, -1)):
        fname_volno = "%s-%04d" % (fname, i)
        hash = sha1(epi.tostring()).hexdigest()
        if hash in hash_dict:
            duplicates.append((fname_volno, hash_dict[hash]))
        else:
            hash_dict[hash] = fname_volno
    return duplicates


def print_check_store(fname, hash_dict):
    """ Run check_store_hash and print duplicates
    """
    duplicates = check_store_hash(fname, hash_dict)
    for duplicate, original in duplicates:
        print('Oops - hash of {0} matches {1}'.format(
            duplicate,
            original))
