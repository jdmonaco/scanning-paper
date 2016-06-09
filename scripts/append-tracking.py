#!/usr/bin/env python
# encoding: utf-8
"""
append_tracking.py -- Walk the metadata tree and load tracking data into the H5
    data file

Usage: append_tracking.py [-n]

Arguments: -n specified only new data

Created by Joe Monaco on 2011-04-04.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.
"""

# Library imports
import sys
import numpy as np

# Package imports
from ..tools.bash import CPrint

# SCANR package imports
from scanr.lib import (Config, walk_mazes, get_data_file_path, get_kdata_file,
    load_tracking_data, get_start_end, time_slice_sample)
from scanr.tracking import tracking_sample_rate, read_position_file
from scanr.data import initialize_groups, get_group

# Constants
DEBUG = Config['debug_mode']


def main(out, new=False):

    # Reinitialize if tree group node not found
    kfile = get_kdata_file()
    tree = get_group()
    if tree is None:
        if raw_input('No groups found. Initialize? (y/n) ').strip().lower()[0] == 'y':
            initialize_groups()
        else:
            sys.exit(0)

    for rds in walk_mazes():

        # If only adding new data, skip sessions with pre-existing data
        tgroup = get_group(rds=rds)
        children = tgroup._v_children.keys()
        if new and np.all([k in children for k in ('t_pos', 'x', 'y', 'hd')]):
            out('Rat%03d-%02d-m%d: found previous data, skipping...'%rds)
            continue

        # Load the position data from the Pos.p.ascii file
        status = {}
        ts, x, y, hd = load_tracking_data(*rds, status=status)
        tgroup._v_attrs['tracking_sample_rate'] = status['fs']

        # Set HD_missing flag if too many invalid samples found
        invalid = status['hd_invalid']
        hd_success = status['hd_fix_success']
        total = status['N_samples']
        if invalid > 0.25*total or not hd_success:
            out('Rat%03d-%02d-m%d: HD missing: %s/%s invalid samples'%
                (rds+(str(invalid).rjust(5), str(total).rjust(5))))
            tgroup._v_attrs['HD_missing'] = True
        else:
            tgroup._v_attrs['HD_missing'] = False

        # Set timing_jumps flag if too many backward timestamps found
        N = (np.diff(ts)<0).sum()
        if N >= 10:
            out('Rat%03d-%02d-m%d: Timing jumps: %d backward jumps'%(rds+(N,)))
            tgroup._v_attrs['timing_jumps'] = True
        else:
            out('Rat%03d-%02d-m%d: Timing fine'%rds)
            tgroup._v_attrs['timing_jumps'] = False

        # Create the t/x/y/hd arrays, after removing previous data if necessary
        if 't_pos' in children:
            tgroup.t_pos._f_remove()
        kfile.createArray(tgroup, 't_pos', ts, title='Time, timestamps')

        if 'x' in children:
            tgroup.x._f_remove()
        kfile.createArray(tgroup, 'x', x, title='X position, cm')

        if 'y' in children:
            tgroup.y._f_remove()
        kfile.createArray(tgroup, 'y', y, title='Y position, cm')

        if 'hd' in children:
            tgroup.hd._f_remove()
        kfile.createArray(tgroup, 'hd', hd, title='Head direction, degrees')

        out('Saved position data to %s'%tgroup)
        kfile.flush()

    # Close the file
    out('Finished appending tracking data to %s'%get_data_file_path('data'))
    kfile.flush()
    kfile.close()


if __name__ == '__main__':

    log_fd = file('append_tracking.log', 'w')
    out = CPrint(prefix='AppendTracking', color='lightgreen', outfd=log_fd, timestamp=False)

    only_new = len(sys.argv) > 1 and sys.argv[1] == '-n'
    main(out, new=only_new)

    log_fd.close()

