# coding: utf-8

"""
chewing.py -- Chewing event detection

Created by Joe Monaco on May 15, 2015.
Copyright (c) 2015 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

import os
import sys

import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian
from scipy.interpolate import interp1d
from traits.api import Float, Bool

from .session import SessionData
from .spike import TetrodeSelect
from .data import get_node, new_table, flush_file, unique_sessions
from .time import time_slice, elapsed
from .meta import get_maze_list
from .eeg import AbstractFilter, get_eeg_timeseries, Theta, CLIP, safe_hilbert
from .ripples import find_pyramidale_tetrodes

from .tools.misc import contiguous_groups, merge_adjacent_groups
from .tools.plot import quicktitle

ChewingDescr =  {   'id'        :   tb.UInt16Col(pos=1),
                    'rat'       :   tb.UInt16Col(pos=2),
                    'day'       :   tb.UInt16Col(pos=3),
                    'session'   :   tb.UInt16Col(pos=4),
                    'start'     :   tb.UInt64Col(pos=5),
                    'end'       :   tb.UInt64Col(pos=6),
                    'tlim'      :   tb.UInt64Col(shape=(2,), pos=7),
                    'number'    :   tb.UInt16Col(pos=8),
                    'duration'  :   tb.Float32Col(pos=9)   }


def create_chewing_table(detector=None):
    """Create /behavior/chewing table containing."""
    CA1_datasets = TetrodeSelect.datasets('(EEG==True)&(area=="CA1")')
    chewing_table = new_table('/behavior', 'chewing',
        ChewingDescr, title='Chewing Events')

    Chewing = (detector is None) and ChewingFilter() or detector

    for k in sorted(Chewing.traits(chewing_param=True).keys()):
        val = getattr(Chewing, k)
        sys.stdout.write(' * %s = %f\n'%(k, val))
        chewing_table._v_attrs[k] = val

    chew_id = 0
    row = chewing_table.row
    for dataset in CA1_datasets:
        rat, day = dataset

        for maze in get_maze_list(rat, day):
            rds = rat, day, maze
            chews = Chewing.detect(rds, debug=True)

            for i, chew in enumerate(chews):
                row['id'] = chew_id
                row['rat'], row['day'], row['session'] = rds
                row['start'] = chew[0]
                row['end'] = chew[1]
                row['tlim'] = (chew[0], chew[1])
                row['number'] = i + 1
                row['duration'] = elapsed(chew[0], chew[1])
                row.append()
                chew_id += 1

            sys.stdout.write(
                'chewing_table: rat%d-%02d-m%d: %d chews\n'%(rds + (i+1,)))
        chewing_table.flush()

    sys.stdout.write('Found %d chews.\n'%chew_id)
    flush_file()


class ChewingFilter(AbstractFilter):

    # Filter definition
    band = 'ripple'
    _decimate_factor = 1

    # Chewing detect parameters
    chew_Z0 = Float(0.9, chewing_param=True) # 1.0
    chew_min_time = Float(0.55, chewing_param=True) # 0.50
    chew_gap_tol = Float(0.18, chewing_param=True) # 0.25

    # Event filters
    chew_filter = Bool(True, chewing_param=True)
    clip_filter = Bool(True, chewing_param=True)

    def detect(self, rds, out=None, debug=True):
        """Detect chew events for the given session.

        Return list of (t_start, t_end) intervals tuples.
        """
        min_sz = int(self.chew_min_time * self.fs)
        gap_sz = int(self.chew_gap_tol * self.fs)

        pyr_tetrodes = find_pyramidale_tetrodes(rds[:2], verbose=False)
        if not pyr_tetrodes:
            return []

        def save_output(key, signal):
            if out is None:
                return
            if key in out:
                out[key] = np.vstack((out[key], signal))
            else:
                out[key] = signal

        # Load the EEG data from all the sessions in this dataset
        AMP = None
        t = None
        for tt in pyr_tetrodes:
            if debug:
                sys.stdout.write('loading rat%d-%02d-m%d Sc%02d\n'%(rds+(tt,)))

            data = get_eeg_timeseries(rds, tt)
            if data is None:
                continue

            if t is None:
                t = data[0]

            EEG = data[1]
            save_output('eeg', EEG)

            X = self.filter(EEG)
            save_output('bp', X)

            if AMP is None:
                AMP = self.amplitude(X)
            else:
                AMP += self.amplitude(X)

        if AMP is None:
            return []

        Z = (AMP - np.median(AMP)) / AMP.std()

        if out is not None:
            out['t'] = t
            out['zamp'] = Z
            out['pyr_tt'] = pyr_tetrodes

        t_theta, ripple_theta = Theta.timeseries(t, Z)
        P_rt = Theta.power(ripple_theta, filtered=True)
        z_P_rt = (lambda x: (x - x.mean()) / x.std())(P_rt)
        f_z_P_rt = interp1d(t_theta, z_P_rt, fill_value=0.0, bounds_error=False)
        z_P_rt_up = f_z_P_rt(t)

        if out is not None:
            out['t_theta'] = t_theta
            out['ripple_theta'] = ripple_theta
            out['z_P_rt'] = z_P_rt

        events = filter(lambda v: v[1] - v[0] >= min_sz,
            merge_adjacent_groups(
                contiguous_groups(z_P_rt_up >= self.chew_Z0), tol=gap_sz))
        t_events = map(lambda v: (t[v[0]], t[v[1] - 1]), events)

        if debug:
            sys.stdout.write('chewing_detect: found %d chews\n'%len(events))

        return t_events

    def amplitude(self, bp):
        """Compute a smoothed z-score amplitude of a ripple-band signal."""
        return convolve(
            np.sqrt(bp**2 + np.imag(safe_hilbert(bp))**2), # |X_a|
            (lambda x: x / np.trapz(x))(gaussian(75, 0.015 * self.fs)), # 15-ms gaussian
            mode='same')
