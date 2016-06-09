#!/usr/bin/env python
# encoding: utf-8
"""
clear_lfp.py -- Remove all EEG data from an H5 data file

Created by Joe Monaco on 2012-02-08.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

USAGE = """Usage: clear_lfp.py data_file
"""

# Library imports
import sys
import os
from os.path import isfile as there
import tables as tb

# Package imports
from ..tools.bash import CPrint

# SCANR Constants
DTS_STUB = 'DTS%02d'
DATA_STUB = 'EEG%02d'
ROOT_PATH = '/kdata'


def main(out, fn):

    data_file = tb.openFile(fn, mode='r+')

    dts_pre = DTS_STUB.split('%')[0]
    eeg_pre = DATA_STUB.split('%')[0]

    for node in data_file.walkNodes(ROOT_PATH, classname='Array'):

        name = node.name
        if not (name.startswith(dts_pre) or name.startswith(eeg_pre)):
            continue

        parent = node._v_parent._v_pathname
        data_file.removeNode(node)

        out('Removed %s from %s.'%(name, parent))
        data_file.flush()

    # Create clean copy of data file
    out('Now creating clean backup copy of file...')
    new_fn = fn + '.%d'%hash(data_file)
    data_file.copyFile(new_fn)
    data_file.close()

    out('Created clean copy:\n\tORIG:\t%s\n\tCOPY:\t%s'%(fn,new_fn))


if __name__ == '__main__':

    if len(sys.argv) != 2 or not there(sys.argv[1]):
        sys.stderr.write(USAGE)
        sys.exit(1)

    fn = os.path.abspath(sys.argv[1])
    sys.stdout.write('To begin removing EEG data:\n\tFROM:\t%s\n... Enter Y: '%fn)
    if not raw_input().strip().lower().startswith('y'):
        sys.stdout('\nExiting.\n')
        sys.exit(0)

    log_fd = file('clear_lfp.log', 'w')
    out = CPrint(prefix='ClearLFP', color='lightgreen', outfd=log_fd, timestamp=False)

    main(out, fn)

    log_fd.close()
