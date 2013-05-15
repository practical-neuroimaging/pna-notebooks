#!/usr/bin/env python
""" Script to run slice timing on Haxby Open FMRI dataset """
import sys

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
    # You need to work out slice times here
    subjects = get_subjects(DATA_PATH)
    for name, subject in subjects.items():
        for run in subject['functionals']:
            fname = run['filename']
            print("Running slice timing on " + fname)
            slice_time_file(fname, slice_times, TR)


if __name__ == '__main__':
    main()
