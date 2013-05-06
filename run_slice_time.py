import numpy as np
import slicetime

fname = 'bold.nii.gz'

slice_order = np.array(range(0, 35, 2) + range(1, 35, 2))
TR = 3.0
n_slices = 35
time_one_slice = TR / n_slices
space_to_order = np.argsort(slice_order)
slice_times = space_to_order * time_one_slice + (time_one_slice / 2)

slicetime.slice_time_file(fname, slice_times, TR)
