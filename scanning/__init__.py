"""
Scanning :: Behavioral analysis of place fields on closed-loop tracks
Author :: Joseph Monaco (github.com/jdmonaco)
"""

import os
import subprocess


HOME = os.getenv("HOME")
SRCDIR = os.path.split(__file__)[0]

PROJECT_PREFIX = "scanning"
PROJECT_HOME = os.path.join(HOME, 'projects', PROJECT_PREFIX)
PROJECT_ROOT = os.path.join(PROJECT_HOME, "root")

DATA_FILE = os.path.join(PROJECT_ROOT, 'scanning.h5')
BACKUP_PATH = os.path.join(PROJECT_ROOT, 'backup')


def git_revision(short=False):
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    if not short:
        cmd.remove('--short')
    try:
        rev = subprocess.check_output(cmd, cwd=SRCDIR)
    except subprocess.CalledProcessError:
        return None
    return str(rev).strip()
