""" Test Utilities for gathering Open FMRI dataset
"""

from os.path import join as pjoin, abspath, expanduser

import nibabel as nib

from openfmri import get_subjects

FILEROOT = abspath(expanduser(pjoin('~', 'data', 'ds105')))
SUBJ_NAMES = ['sub%03d' % i for i in range(1, 7)]

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_almost_equal


def test_big_picture():
    subjects = get_subjects(FILEROOT)
    assert_equal(len(subjects), 6)
    assert_equal(sorted(subjects.keys()), SUBJ_NAMES)
    for name, subject in subjects.items():
        assert_equal(len(subject['anatomicals']), 1)
        assert_equal(subject['tasks'], [1])
        if name == 'sub005':
            assert_equal(len(subject['functionals']), 11)
        else:
            assert_equal(len(subject['functionals']), 12)
        for i, rundef in enumerate(subject['functionals']):
            assert_true(rundef['filename'].endswith('bold.nii.gz'))
            assert_equal(rundef['task_no'], 1)
            assert_equal(rundef['run_no'], i+1)
            assert_true('task001_run%03d' % (i + 1,) in rundef['filename'])
            img = nib.load(rundef['filename'])
            assert_equal(img.shape, (40, 64, 64, 121))
            hdr = img.get_header()
            assert_almost_equal(hdr['pixdim'][1:4], (3.5, 3.75, 3.75))
