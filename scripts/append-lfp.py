#!/usr/bin/env python
# encoding: utf-8
"""
append_lfp.py -- Walk the metadata tree and load continuous local field data
    into the H5 data file.

Usage: append_lfp.py

Created by Joe Monaco on 2012-01-09.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

# Library imports
import os
import sys
import gc

# Package imports
from ..tools.bash import CPrint

# SCANR package imports
from scanr.paths import get_data_file_path
from scanr.data import get_group, new_array, initialize_groups, update_group_metadata
from scanr.meta import walk_tetrodes, get_maze_list
from scanr.time import compress_timestamps, time_slice_sample, find_sample_rate
from scanr.neuralynx import read_ncs_file
from scanr.eeg import (DTS_STUB, DATA_STUB, get_eeg_file, close_eeg_file,
    flush_eeg_file, Resample2K)

def main(out):

    eeg_file = get_eeg_file(False) # open in read-write mode
    initialize_groups(eeg=True, overwrite=False) # update group structure if necessary
    update_group_metadata(eeg=True)

    for rat, day, tt in walk_tetrodes():
        if (rat, day) == (64, 7):
            out('This dataset has issues. Pretending it doesn\'t exist for now...')
            continue

        nsc_filename = get_data_file_path('continuous', rat=rat, day=day,
            tetrode=tt, search=True)
        if nsc_filename is None:
            out('Missing continuous data file: rat%03d-%02d Sc%02d'%(
                rat, day, tt), error=True)
            continue

        time_name = DTS_STUB%tt
        array_name = DATA_STUB%tt

        tt_str = 'rat%03d-%02d-Sc%02d' % (rat, day, tt)

        first_group = get_group(eeg=True, rds=(rat, day, 1))
        if array_name in first_group:
            out('%s: Found pre-existing data. Skipping.'%tt_str)
            continue

        out('Loading %s...'%tt_str)
        ts, data = read_ncs_file(nsc_filename, verbose=False)
        if ts.size == 0:
            out('%s: No samples found in Ncs file. Skipping.'%tt_str, error=True)
            continue

        for maze in get_maze_list(rat, day):

            data_group = get_group(eeg=True, rds=(rat, day, maze))
            attrs = data_group._v_attrs

            session_ts, session_data = time_slice_sample(ts, data,
                start=attrs['start'], end=attrs['end'])

            if not len(session_ts):
                out('%s: No session data found. Timing issue? Skipping.'%tt_str, error=True)
                continue

            # For 2kHz data sets, resample down to 1kHz before storing
            fs = find_sample_rate(session_ts)
            if fs > 1900.0:
                out('%s: Found %.1Hz sample rate: resampling...'%(tt_str, fs))
                session_ts, session_data = Resample2K.timeseries(session_ts, session_data)
                fs = find_sample_rate(session_ts)

            delta_ts = compress_timestamps(session_ts)

            time_array = new_array(data_group, time_name, delta_ts,
                title='Delta-Compressed Timestamps: Tetrode %d'%tt,
                force=True, eeg=True)
            data_array = new_array(data_group, array_name, session_data,
                title='Continuous EEG Data: Tetrode %d'%tt,
                force=True, eeg=True)

            time_array._v_attrs['sample_rate'] = fs
            data_array._v_attrs['sample_rate'] = fs

            out('Saved %s.'%time_array._v_pathname)
            out('Saved %s.'%data_array._v_pathname)

            gc.collect()
        flush_eeg_file()

    out('Finished appending LFP data to %s'%get_data_file_path('eeg'))
    close_eeg_file()


if __name__ == '__main__':

    log_fd = file('append_lfp.log', 'w')
    out = CPrint(prefix='AppendLFP', color='lightgreen', outfd=log_fd, timestamp=False)

    main(out)

    log_fd.close()
