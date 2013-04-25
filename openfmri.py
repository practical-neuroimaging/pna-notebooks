""" Utilities for gathering Open FMRI datasets
"""

import re
from glob import glob

from os import listdir
from os.path import (join as pjoin, abspath, split as psplit)

SUBJ_RE = re.compile(r'sub\d\d\d')
ANAT_RE = re.compile(r'highres\d\d\d\.nii\.gz')
TASK_RE = re.compile(r'task(\d\d\d)_run(\d\d\d)')

def get_visits(fileroot):
    visits = {}
    for fname in listdir(fileroot):
        if SUBJ_RE.match(fname):
            visits[fname] = get_visit(pjoin(fileroot, fname))
    return visits


def get_visit(fileroot):
    fileroot = abspath(fileroot)
    visit = dict(anatomicals = [],
                 functionals = [])
    # Get anatomicals
    anat_search = pjoin(fileroot, 'anatomy', 'highres*.nii.gz')
    for anat_path in sorted(glob(anat_search)):
        pth, fname = psplit(anat_path)
        if ANAT_RE.match(fname):
            visit['anatomicals'].append(anat_path)
    # Get functionals (as dicts)
    func_search = pjoin(fileroot, 'BOLD', '*', 'bold.nii.gz')
    for func_path in sorted(glob(func_search)):
        rundef = get_rundef(func_path)
        visit['functionals'].append(rundef)
    # Sort functionals by task_no, run_no
    visit['functionals'].sort(key=_run_key)
    # Compile list of tasks for convenience
    all_tasks = [f['task_no'] for f in visit['functionals']]
    unique_tasks = set(all_tasks) # Retain only unique values
    visit['tasks'] = sorted(unique_tasks)
    return visit


def _run_key(rundef):
    return (rundef['task_no'], rundef['run_no'])


def get_rundef(func_path):
    pth, fname = psplit(func_path)
    pth, task_run = psplit(pth)
    task_match = TASK_RE.match(task_run)
    if not task_match:
        raise ValueError('Did not expect this task_run value: ' + task_run)
    task, run = task_match.groups()
    return dict(filename=func_path,
                task_no = int(task),
                run_no = int(run))
