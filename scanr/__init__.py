"""
scanr :: Attentive Head Scanning
Author :: Joseph Monaco (jmonaco@jhu.edu)
Created :: 06-08-2016
"""

__version__ = '0.1.0'

import os

SRCDIR = os.path.split(__file__)[0]
HOME = os.getenv("HOME")

PROJECT_PREFIX = "scanr"
PROJECT_HOME = os.path.join(HOME, 'projects', PROJECT_PREFIX)
PROJECT_ROOT = os.path.join(PROJECT_HOME, 'data')

DATA_FILE = os.path.join(PROJECT_ROOT, '{}.h5'.format(PROJECT_PREFIX))
BACKUP_PATH = os.path.join(PROJECT_ROOT, 'backup')

#DATA_ROOT = os.path.join(HOME, 'data', 'xlab', 'DataSet')

