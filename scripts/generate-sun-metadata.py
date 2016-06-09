#!/usr/bin/env python
# encoding: utf-8
"""
generate-sun-metadata.py -- Create sun-datasets.yaml metadata file

Usage: generate-sun-metadata.py

Created by Joe Monaco on April 9, 2014.
Copyright (c) 2014 Johns Hopkins University. All rights reserved.
"""

# Library imports
import os
import re
import glob
import yaml
import numpy as np

# Package imports
from ..tools.bash import CPrint
from ..tools.misc import AutoVivification

ROOT = "/kdata/Sun_data_rat18_to_44"

def session_timing_from_cluster_file(fn):
    """Read the start/end timestamps of the recording session listed in the
    given cluster file
    """
    skip = 11
    with file(fn, 'r') as fd:
        while skip:
            fd.readline()
            skip -= 1
        start = long(fd.readline())
        end = long(fd.readline())
    return start, end

def main(out):

    datasets = AutoVivification()
    maze_cluster_pattern = re.compile('cl-maze(\d+)\.(\d+)')

    for dpath, dirs, files in os.walk(ROOT):

        wayup, up, parent = dpath.split(os.path.sep)[-3:]
        if not parent:
            up, parent = wayup, up
        ratdaysearch = re.match('[Rr]at(\d+)-(\d+)$', parent)
        if not ratdaysearch:
            continue

        rat, day = [int(n) for n in ratdaysearch.groups()]
        out('Processing rat %d, day %d...' % (rat, day))

        # Get tetrodes from ScXX subdirectories
        tetrodes = []
        for Sc in glob.glob(os.path.join(dpath, 'Sc*')):
            Sc_tail = os.path.split(Sc)[1]
            if re.match('Sc\d+$', Sc_tail):
                tetrodes.append(int(Sc_tail[2:]))
        tetrodes.sort()
        for tt in tetrodes:
            datasets[rat]['days'][day]['tetrodes'][tt] = None

        cluster_timing = AutoVivification()

        for cluster_file in glob.glob(os.path.join(dpath, 'Sc*/cl-maze*')):
            mc_search = re.search(maze_cluster_pattern, cluster_file)
            if mc_search is None:
                continue

            maze, cluster = [int(n) for n in mc_search.groups()]
            start, end = session_timing_from_cluster_file(cluster_file)

            if maze not in cluster_timing:
                cluster_timing[maze]['start'] = []
                cluster_timing[maze]['end'] = []

            cluster_timing[maze]['start'].append(start)
            cluster_timing[maze]['end'].append(end)

        maze_list = sorted(cluster_timing.keys())
        counts = { 'STD':0, 'MIS':0 }
        for maze in maze_list:
            session_start = int(np.median(cluster_timing[maze]['start']))
            session_end = int(np.median(cluster_timing[maze]['end']))
            session_type = ['MIS', 'STD'][maze % 2]
            counts[session_type] += 1

            S = datasets[rat]['days'][day]['sessions'][maze]
            S['start'] = session_start
            S['end'] = session_end
            S['type'] = session_type
            S['parameter'] = 0
            S['start_comment'] = "maze %d start (autogen)" % maze
            S['end_comment'] = "maze %d end (autogen)" % maze
            S['number'] = counts[session_type]

        if 'exptr' not in datasets[rat]:
            datasets[rat]['exptr'] = 'Inah'

    yaml_fn = os.path.join(ROOT, 'sun-datasets.yaml')
    with file(yaml_fn, 'w') as fd:
        yaml.dump(datasets.dict_copy(), stream=fd, default_flow_style=False)
        out("Wrote: %s" % yaml_fn)

    out('All done!')


if __name__ == '__main__':

    log_fd = file('generate-sun-metadata.log', 'w')
    out = CPrint(prefix='SunData', color='lightgreen', outfd=log_fd, timestamp=False)

    main(out)

    log_fd.close()

