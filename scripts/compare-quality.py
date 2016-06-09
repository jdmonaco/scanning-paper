#!/usr/bin/env python
# encoding: utf-8
"""
compare_quality.py -- Compare and optionally synchronize cluster isolation
    quality between current and specified kdata file

Usage: compare_quality.py [-u] kdata-compare.h5

Created by Joe Monaco on September 9, 2013.
Copyright (c) 2013 Johns Hopkins University. All rights reserved.
"""

# Library imports
import os
import sys
import tables as tb

# Package imports
from ..tools.bash import CPrint

# SCANR package imports
from scanr.meta import walk_mazes
from scanr.data import get_data_file_path, close_file
from scanr.paths import get_group_path
from scanr.spike import get_session_clusters


def main(out, kfilepath1, kfilepath2=None, perform_sync=False):

    kfile1 = tb.openFile(kfilepath1, mode='r')
    if kfilepath2 is None:
        kfilepath2 = get_data_file_path('data')
        close_file()
    mode2 = perform_sync and 'a' or 'r'
    kfile2 = tb.openFile(kfilepath2, mode=mode2)

    out('File 1: %s' % kfilepath1)
    out('File 2: %s' % kfilepath2)

    for rds in walk_mazes():
        rat, day, session = rds
        grp_path = get_group_path(rat=rat, day=day, session=session)

        try:
            grp1 = kfile1.getNode(grp_path)
            grp2 = kfile2.getNode(grp_path)
        except tb.NoSuchNodeError:
            continue
        else:
            out.printf('.', color='lightgreen')

        clusters1 = get_session_clusters(grp1)
        clusters2 = get_session_clusters(grp2)

        common_clusters = set(clusters1).intersection(clusters2)

        for tt_cl in common_clusters:
            tc = 't%dc%d' % tt_cl

            cl1 = kfile1.getNode(grp1, tc)
            cl2 = kfile2.getNode(grp2, tc)

            q1 = cl1._v_attrs['quality']
            q2 = cl2._v_attrs['quality']

            if q1 != q2:
                out.printf('\n')
                out('%s %s: quality = %s -> %s' % (grp_path, tc, q1, q2))
                if perform_sync:
                    cl2._v_attrs['quality'] = q1

    out.printf('\n')

    if perform_sync:
        kfile2.flush()

    kfile1.close()
    kfile2.close()


if __name__ == '__main__':

    do_sync = False
    if "-u" in sys.argv:
        do_sync = True
        sys.stdout.write("Performing synchronization.\n")
        sys.argv.remove("-u")
    if len(sys.argv) == 2:
        kfilecompare = str(sys.argv[1])
    else:
        sys.stderr.write("Usage: compare_quality.py [-u] kdata-compare.h5\n")
        sys.exit(1)

    log_fd = file('compare_quality.log', 'w')
    out = CPrint(prefix='CompareQuality', color='lightcyan', outfd=log_fd, timestamp=False)

    main(out, kfilecompare, perform_sync=do_sync)

    log_fd.close()

