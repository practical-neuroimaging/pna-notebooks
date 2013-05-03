""" Utilities for gathering Open FMRI datasets

The structure of an Open FMRI dataset defined here:

https://openfmri.org/content/data-organization
"""

import re
from glob import glob

from os import listdir
from os.path import (join as pjoin, abspath, split as psplit)

SUBJ_RE = re.compile(r'sub\d\d\d')
ANAT_RE = re.compile(r'highres\d\d\d\.nii\.gz')
TASK_RE = re.compile(r'task(\d\d\d)_run(\d\d\d)')

def get_subjects(fileroot):
    """ Create dictionary of subject dictionaries starting at `fileroot`

    Parameters
    ----------
    fileroot : str
        path containing subject subdirectories

    Returns
    -------
    subjects : dict
        dict with (key, value) pairs of (subject name, subject_dict), where
        ``subject_dict`` is a dictionary containing the information for the
        named subject
    """
    subjects = {}
    for fname in listdir(fileroot):
        if SUBJ_RE.match(fname):
            subjects[fname] = get_subject(pjoin(fileroot, fname))
    return subjects


def get_subject(subj_path):
    """ Create `subject` dictionary for subject at path `subj_path`

    Parameters
    ----------
    subj_path : str
        path containing subject data

    Returns
    -------
    subj_dict : dict
        dictionary containing information for this subject
    """
    subj_path = abspath(subj_path)
    subject = dict(anatomicals = [],
                 functionals = [])
    # Get anatomicals
    anat_search = pjoin(subj_path, 'anatomy', 'highres*.nii.gz')
    for anat_path in sorted(glob(anat_search)):
        pth, fname = psplit(anat_path)
        if ANAT_RE.match(fname):
            subject['anatomicals'].append(anat_path)
    # Get functionals (as dicts)
    func_search = pjoin(subj_path, 'BOLD', '*', 'bold.nii.gz')
    for func_path in sorted(glob(func_search)):
        rundef = get_rundef(func_path)
        subject['functionals'].append(rundef)
    # Sort functionals by task_no, run_no
    subject['functionals'].sort(key=_run_key)
    # Compile list of tasks for convenience
    all_tasks = [f['task_no'] for f in subject['functionals']]
    unique_tasks = set(all_tasks) # Retain only unique values
    subject['tasks'] = sorted(unique_tasks)
    return subject


def _run_key(rundef):
    """ Creates a key useful for sorting functional run dictionaries """
    return (rundef['task_no'], rundef['run_no'])


def get_rundef(func_path):
    """ Create dictionary defining single functional run at path `func_path`

    Parameters
    ----------
    func_path : str
        path containing functional data
    """
    pth, fname = psplit(func_path)
    pth, task_run = psplit(pth)
    task_match = TASK_RE.match(task_run)
    if not task_match:
        raise ValueError('Did not expect this task_run value: ' + task_run)
    task, run = task_match.groups()
    return dict(filename=func_path,
                task_no = int(task),
                run_no = int(run))
