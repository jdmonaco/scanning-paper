#!/usr/bin/env python
# encoding: utf-8
"""
find_session_timing.py -- Walk the metadata tree and read the timestamps of
    recording sessions from cluster files to determine canonical timing

Usage: find_session_timing.py

Created by Joe Monaco on August 12, 2013.
Copyright (c) 2013 Johns Hopkins University. All rights reserved.
"""

import numpy as np

# Package imports
from ..tools.bash import CPrint
from ..tools.misc import AutoVivification

# SCANR package imports
from scanr.lib import (Config, walk_days, get_maze_list, get_tetrode_list,
    get_kdata_file)
from scanr.meta import write_yaml_file
from scanr.spike import find_spike_files, session_timing_from_cluster_file


def main(out):

    timing = AutoVivification()

    for rat, day in walk_days():
        tetrodes = get_tetrode_list(rat, day)

        for session in get_maze_list(rat, day):
            recording_start = []
            recording_end = []
            rds = (rat, day, session)
            rds_str = 'rat%03d-%02d-m%d'%rds

            for tt in tetrodes:
                files, clusters = find_spike_files(rat, day, session, tt)
                if not clusters:
                    continue

                for fn, cl in zip(files, clusters):
                    tc = tt, cl
                    clname = 't%dc%d'%tc
                    start, end = session_timing_from_cluster_file(fn)
                    recording_start.append(start)
                    recording_end.append(end)
                    out('%s: %11u to %11u  [%s]' % (rds_str, start, end, clname))

            if len(recording_start):
                meta = timing[rat]['days'][day]['sessions'][session]
                for t, which in [   (recording_start, 'start'),
                                    (recording_end, 'end')]:
                    meta[which] = int(np.median(t))
                    num = len(set(t))
                    if num > 1:
                        out('warning: %s: found %d different %s times'%(rds_str, num, which))

    write_yaml_file(timing, 'timing')
    out('All done!')


if __name__ == '__main__':

    log_fd = file('find-session-timing.log', 'w')
    out = CPrint(prefix='FindTiming', color='lightgreen', outfd=log_fd, timestamp=False)

    main(out)

    log_fd.close()

