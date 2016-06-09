#!/usr/bin/env python
# encoding: utf-8
"""
copy_lfp.py -- Isomorphic copy of EEG data from one H5 File to another

Created by Joe Monaco on 2012-02-08.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

USAGE = """Usage: copy_lfp.py src_file dst_file
"""

# Library imports
import sys
import os
from os.path import isfile as there
import tables as tb
import gc

# Package imports
from ..tools.bash import CPrint

# SCANR Constants
DTS_STUB = 'DTS%02d'
DATA_STUB = 'EEG%02d'
ROOT_PATH = '/kdata'


def main(out, src, dst):

    src_file = tb.openFile(src, mode='r')
    dst_file = tb.openFile(dst, mode='r+')

    dts_pre = DTS_STUB.split('%')[0]
    eeg_pre = DATA_STUB.split('%')[0]

    for node in src_file.walkNodes(ROOT_PATH, classname='Array'):

        name = node.name
        if not (name.startswith(dts_pre) or name.startswith(eeg_pre)):
            continue

        parent = node._v_parent._v_pathname

        new_array = dst_file.createArray(parent, name, node.read(), title=node.title)
        dst_file.copyNodeAttrs(node, new_array)

        out('Copied %s from %s.'%(name, parent))
        dst_file.flush()
        gc.collect()

    dst_file.close()
    src_file.close()

    out('Copied EEG data:\n\tFROM:\t%s\n\tTO:\t%s'%(src,dst))


if __name__ == '__main__':

    if len(sys.argv) != 3 or not (there(sys.argv[1]) and there(sys.argv[2])):
        sys.stderr.write(USAGE)
        sys.exit(1)

    src, dst = map(os.path.abspath, tuple(sys.argv[1:]))
    sys.stdout.write('To begin copying EEG data:\n\tFROM:\t%s\n\tTO:\t%s\n... Enter Y: '%(src,dst))
    if not raw_input().strip().lower().startswith('y'):
        sys.stdout('\nExiting.\n')
        sys.exit(0)

    log_fd = file('copy_lfp.log', 'w')
    out = CPrint(prefix='CopyLFP', color='lightgreen', outfd=log_fd, timestamp=False)

    main(out, src, dst)

    log_fd.close()
