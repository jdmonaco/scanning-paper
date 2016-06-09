#!/usr/bin/env python
#encoding: utf-8

"""
count_clusters.py -- Scan the kdata tree and count cluster leaf nodes

Written by Joe Monaco, 09/06/2013
"""

import os
import sys
import re
import tables

from scanr.data import get_data_file_path, get_group_path
from scanr.meta import walk_mazes

from ..tools.bash import lightgreen, lightred

NAME = "count-clusters.py"
DESC = "Scan the kdata tree and count cluster leaf nodes"
USAGE = "Usage: %s [non_standard_kdata.h5]\n\n%s" % (NAME, DESC)

def main(kfile):
    kdata = kfile.root.kdata
    cluster_name = re.compile('t\d+c\d+')
    session_counts = []
    for rds in walk_mazes():
        rat, day, session = rds
        grp_path = get_group_path(rat=rat, day=day, session=session)
        try:
            grp = kfile.getNode(grp_path)
        except tables.NoSuchNodeError:
            session_counts.append((grp_path, '---'))
            sys.stdout.write(lightred('.'))
        else:
            leaves = grp._v_children.keys()
            count = len(filter(lambda k: re.match(cluster_name, k) is not None, leaves))
            session_counts.append((grp_path, count))
            sys.stdout.write(lightgreen('.'))
        sys.stdout.flush()
    sys.stdout.write('\n')

    output_fn = NAME[:-2] + 'log'
    with file(output_fn, 'w') as fd:
        for path, count in session_counts:
            print >>fd, path.ljust(28), str(count).rjust(3)
    sys.stdout.write('wrote: %s\n' % output_fn)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        h5path = os.path.realpath(sys.argv[1])
    elif len(sys.argv) == 1:
        h5path = get_data_file_path('data')
    else:
        print USAGE
        sys.exit(1)

    if os.path.exists(h5path):
        kfile = tables.openFile(h5path, mode='r')
        sys.stdout.write('opened: %s\n' % h5path)
    else:
        sys.stderr.write('file does not exist: %s\n' % h5path)
        sys.exit(1)

    main(kfile)

    kfile.close()
    sys.stdout.write('closed: %s\n' % h5path)
