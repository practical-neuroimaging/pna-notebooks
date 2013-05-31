#!/usr/bin/env python
""" Script to run slice timing on Haxby Open FMRI dataset """
import sys

import numpy as np

# Library to fetch filenames from Open FMRI data layout
from openfmri import get_subjects

# Library for running slice timing on 4D images
from slicetime import slice_time_file


def main():
    try:
        DATA_PATH = sys.argv[1]
    except IndexError:
        raise RuntimeError("Pass data path on command line")
    N_SLICES = 40
    TR = 2.5
    
    # Figure out slice times.
    slice_order = np.array(range(0, N_SLICES, 2) + range(1, N_SLICES, 2))
    space_to_order = np.argsort(slice_order)
    time_one_slice = TR / N_SLICES
    slice_times = space_to_order * time_one_slice + time_one_slice / 2.0
    
    subjects = get_subjects(DATA_PATH)
    for name, subject in subjects.items():
        for run in subject['functionals']:
            fname = run['filename']
            print("Running slice timing on " + fname)
            slice_time_file(fname, slice_times, TR, slice_axis=0)


if __name__ == '__main__':
    main()
