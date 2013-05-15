import os

import nipy
from nipy.algorithms.diagnostics import screens
from openfmri import get_visits

for name, visit in get_visits('ds105').items():
    for functional in visit['functionals']:
        filename = functional['filename']
        img = nipy.load_image(filename)
        res = screens.screen(img, slice_axis=0)
        pth, fname = os.path.split(filename)
        froot, ext = os.path.splitext(fname)
        screens.write_screen_res(res, pth, froot)
