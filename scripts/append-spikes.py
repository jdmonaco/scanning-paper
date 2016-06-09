#!/usr/bin/env python
# encoding: utf-8
"""
append_spikes.py -- Walk the metadata tree and load spike-train data into the H5
    data file. Run a second time to load spike width and quality metadata.

Usage: append_spikes.py

Created by Joe Monaco on 2011-05-05.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.
"""

# Library imports
import sys

# Package imports
from ..tools.bash import CPrint
from ..tools.misc import AutoVivification

# SCANR package imports
from scanr.lib import (Config, walk_days, get_maze_list, get_tetrode_list, get_data_file_path,
    get_kdata_file)
from scanr.data import initialize_groups, get_group, new_array, update_group_metadata
from scanr.spike import find_spike_files, read_spike_file, load_mean_spike_width, load_cluster_quality


def main(out):

    # Reinitialize if tree group node not found
    kfile = get_kdata_file()
    initialize_groups(overwrite=False)
    update_group_metadata()

    # Walk sessions and tetrodes to find spike files and load into the tree
    for rat, day in walk_days():

        tetrodes = get_tetrode_list(rat, day)
        for session in get_maze_list(rat, day):

            sgroup = get_group(rat=rat, day=day, session=session)
            for tt in tetrodes:

                files, clusters = find_spike_files(rat, day, session, tt)
                if not clusters:
                    continue

                for fn, cl in zip(files, clusters):
                    tc = tt, cl
                    clname = 't%dc%d'%tc
                    try:
                        st = getattr(sgroup, clname)
                    except AttributeError:
                        spikes = read_spike_file(fn)
                        if len(spikes):
                            st = new_array(sgroup, clname, spikes,
                                title='Spike Times: Tetrode %d, Cluster %d'%tc,
                                force=True)
                            out('Saved %s'%st._v_pathname)
                    else:
                        out('Found %s'%st._v_pathname)
                        quality = load_cluster_quality(rat, day, tt, cl)
                        st._v_attrs['quality'] = str(quality[0])
                        st._v_attrs['comment'] = quality[1]
                        st._v_attrs['maxwidth'] = load_mean_spike_width(
                            rat, day, session, tt, cl)
                        out('Quality = %s, Spike Width = %.3f'%
                            (quality[0], st._v_attrs['maxwidth']))

                if not kfile.isopen:
                    kfile = get_kdata_file()
                kfile.flush()

    # Close the file
    out('Finished appending spike-train data to %s'%get_data_file_path('data'))
    kfile.flush()
    kfile.close()


if __name__ == '__main__':

    log_fd = file('append_spikes.log', 'w')
    out = CPrint(prefix='AppendSpikes', color='lightgreen', outfd=log_fd, timestamp=False)

    main(out)

    log_fd.close()

