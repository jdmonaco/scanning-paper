# encoding: utf-8
"""
ripples.py -- Ripple oscillation event detection

Created by Joe Monaco on April 20, 2012.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian
from scipy.interpolate import interp1d
from traits.api import Float, Bool

# Package imports
from .session import SessionData
from .spike import TetrodeSelect
from .data import get_node, new_table, flush_file, unique_sessions
from .time import stamp_to_time, time_slice, select_from, elapsed
from .meta import get_maze_list
from .eeg import AbstractFilter, get_eeg_timeseries, Theta, CLIP, safe_hilbert
from .cluster import (AND, PrincipalCellCriteria,
    get_tetrode_restriction_criterion, get_min_quality_criterion)

# Local imports
from .tools.misc import contiguous_groups, merge_adjacent_groups
from .tools.plot import quicktitle

# Ripple table
RippleDescr =   {   'id'        :   tb.UInt16Col(pos=1),
                    'rat'       :   tb.UInt16Col(pos=2),
                    'day'       :   tb.UInt16Col(pos=3),
                    'session'   :   tb.UInt16Col(pos=4),
                    'start'     :   tb.UInt64Col(pos=5),
                    'peak'      :   tb.UInt64Col(pos=6),
                    'end'       :   tb.UInt64Col(pos=7),
                    'tlim'      :   tb.UInt64Col(shape=(2,), pos=8),
                    'number'    :   tb.UInt16Col(pos=9),
                    'duration'  :   tb.Float32Col(pos=10)   }


def find_pyramidale_tetrodes(dataset, condn='(EEG==True)&(area=="CA1")',
    min_count=1, min_quality='none', verbose=False, cell_counts=None):
    """Loop through tetrodes and sessions, counting the overall number of
    recorded principal cells as a measure of proximity to s. pyr.

    Returns a list of all tetrodes

    Arguments:
    dataset -- (rat, day) tuple specifying dataset
    condn -- /metadata/tetrodes conditional query, defaults to CA1 tetrodes
    min_count -- cell count threshold for tetrodes
    min_quality -- isolation threshold for counting cells on a tetrode
    verbose -- whether to display cell counts on the console
    cell_counts -- value by reference dictionary of cell counts
    """
    all_tetrodes = TetrodeSelect.tetrodes(dataset, condn, quiet=(not verbose))
    if not all_tetrodes:
        return []
    if cell_counts is None:
        cell_counts = {}

    Quality = get_min_quality_criterion(min_quality)
    pyr_tetrodes = []

    for tt in all_tetrodes:
        ThisTetrode = get_tetrode_restriction_criterion(tt)
        principal_cells = set()

        for maze in get_maze_list(*dataset):
            session = SessionData.get(dataset + (maze,))
            principal_cells = principal_cells.union(
                session.get_clusters(
                    AND(PrincipalCellCriteria, Quality, ThisTetrode)))

        N = len(principal_cells)
        if verbose:
            sys.stdout.write('Sc%d cell count = %d\n'%(tt, N))

        if N < min_count:
            continue

        cell_counts[tt] = N
        pyr_tetrodes.append(tt)

    return pyr_tetrodes

def create_ripple_table(detector=None):
    """Create /physiology/ripples table containing all detected ripple events

    Pass in a RippleFilter detector object or a default instance will be
    created for processing the data.
    """
    CA1_datasets = TetrodeSelect.datasets('(EEG==True)&(area=="CA1")')
    ripple_table = new_table('/physiology', 'ripples',
        RippleDescr, title='High-Frequency Ripple Oscillation Events')

    Ripple = (detector is None) and RippleFilter() or detector

    for k in sorted(Ripple.traits(ripple_param=True).keys()):
        val = getattr(Ripple, k)
        sys.stdout.write(' * %s = %f\n'%(k, val))
        ripple_table._v_attrs[k] = val

    ripple_id = 0
    row = ripple_table.row
    for dataset in CA1_datasets:
        rat, day = dataset

        for maze in get_maze_list(rat, day):
            rds = rat, day, maze
            ripples = Ripple.detect(rds, debug=True)

            for i, ripple in enumerate(ripples):
                row['id'] = ripple_id
                row['rat'], row['day'], row['session'] = rds
                row['start'] = ripple[0]
                row['peak'] = ripple[1]
                row['end'] = ripple[2]
                row['tlim'] = (ripple[0], ripple[2])
                row['number'] = i + 1
                row['duration'] = elapsed(ripple[0], ripple[2])
                row.append()
                ripple_id += 1

            sys.stdout.write(
                'ripple_table: rat%d-%02d-m%d: %d ripples\n'%(rds + (i+1,)))
        ripple_table.flush()

    sys.stdout.write('Found %d ripples.\n'%ripple_id)
    flush_file()


class RippleFilter(AbstractFilter):

    # Filter definition
    band = 'ripple'
    _decimate_factor = 1

    # Ripple detection parameters
    Z0 = Float(2.0, ripple_param=True) # 1.5
    Z1 = Float(4.24, ripple_param=True) # 3.0
    min_time = Float(0.03, ripple_param=True)
    gap_tol = Float(0.01, ripple_param=True)

    # Chewing detect parameters
    chew_Z0 = Float(0.9, ripple_param=True) # 1.0
    chew_min_time = Float(0.55, ripple_param=True) # 0.50
    chew_gap_tol = Float(0.18, ripple_param=True) # 0.25

    # Event filters
    chew_filter = Bool(True, ripple_param=True)
    clip_filter = Bool(True, ripple_param=True)

    def detect(self, rds, out=None, debug=False):
        """For the EEG timeseries data (t, EEG), perform off-line ripple
        detection and return a list of (start, peak, stop) tuples of the time
        points (based on *t*) that define each ripple event.

        Events with any EEG clipping are filtered out.
        """
        min_sz = int(self.min_time * self.fs)
        gap_sz = int(self.gap_tol * self.fs)

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

        # Events from event-threshold, filtered by peak-threshold
        N = []
        events = contiguous_groups(Z >= self.Z0)
        events = filter(lambda v: np.max(Z[slice(*v)]) >= self.Z1, events)
        N.append(('thresh', len(events)))

        # Merge adjacent events
        events = merge_adjacent_groups(events, tol=gap_sz)
        N.append(('merge', len(events)))

        # Filter out short events
        events = filter(lambda v: v[1] - v[0] >= min_sz, events)
        N.append(('size', len(events)))

        # Filter out events with any EEG clipping
        if self.clip_filter:
            events = filter(
                lambda v: np.max(EEG[...,slice(*v)]) < CLIP, events)
            events = filter(
                lambda v: np.min(EEG[...,slice(*v)]) >= -CLIP, events)
            N.append(('clip', len(events)))

        # Filter out events that overlap with chewing
        if self.chew_filter:
            chewing_events = self.detect_chewing(t, Z)
            if debug:
                sys.stdout.write('chew_detect: %d chewing events\n'%
                    len(chewing_events))
            chewing = select_from(t, chewing_events)
            events = filter(lambda v: not chewing[slice(*v)].any(), events)
            N.append(('chew', len(events)))

        if debug:
            sys.stdout.write('ripple_detect: %s\n'%(
                (', ').join('%s: %d'%(k.title(), num) for k, num in N)))

        # Event timing
        return map(lambda v: tuple(map(lambda i: t[i],
            (v[0], v[0] + np.argmax(Z[slice(*v)]), v[1]-1))), events)

    def amplitude(self, bp):
        """Compute a 50-ms smoothed z-score amplitude of the given ripple-band
        signal for ripple detection
        """
        return convolve(
            np.sqrt(bp**2 + np.imag(safe_hilbert(bp))**2), # |X_a|
            (lambda x: x / np.trapz(x))(gaussian(75, 0.015 * self.fs)), # 15-ms gaussian
            mode='same')

    def detect_chewing(self, t, z_amp, out=None):
        """For the given EEG timeseries data, detect chewing barrages for
        exclusion from detection as ripple events
        """
        t_theta, ripple_theta = Theta.timeseries(t, z_amp)
        P_rt = Theta.power(ripple_theta, filtered=True)
        z_P_rt = (lambda x: (x - x.mean()) / x.std())(P_rt)
        f_z_P_rt = interp1d(t_theta, z_P_rt, fill_value=0.0, bounds_error=False)
        z_P_rt_up = f_z_P_rt(t)

        if out is not None:
            out['t_theta'] = t_theta
            out['ripple_theta'] = ripple_theta
            out['z_P_rt'] = z_P_rt

        min_sz = int(self.chew_min_time * self.fs)
        gap_sz = int(self.chew_gap_tol * self.fs)

        # Find chewing events, and merge overlapping events
        events = filter(lambda v: v[1] - v[0] >= min_sz,
            merge_adjacent_groups(
                contiguous_groups(z_P_rt_up >= self.chew_Z0), tol=gap_sz))
        t_events = map(lambda v: (t[v[0]], t[v[1] - 1]), events)

        return t_events

    def plot_ripples(self, rds, ax=None, out=None):
        """Plot EEG signal and derivative components used to detect ripples
        """
        if out is None:
            signal = dict()
        else:
            signal = out

        data = SessionData.get(rds)
        ripples = self.detect(rds, out=signal, debug=True)
        if not ripples:
            sys.stdout.write('No ripples found.\n')
            return

        t = data.T_(signal['t'])
        bp = signal['bp']
        eeg = signal['eeg']
        zamp = signal['zamp']
        pyr_tt = signal['pyr_tt']

        plt.ioff()
        if ax is None:
            f = plt.figure(figsize=(18,11))
            f.suptitle('Ripple Detection Signals')
            ax = f.add_subplot(111)

        dy = -1.5
        y_trace = -4
        for i in xrange(bp.shape[0]):
            h_bp = ax.plot(t, y_trace + bp[i] / 400, 'b-', lw=1, zorder=1)
            y_trace += dy

            h_eeg = ax.plot(t, y_trace + eeg[i] / float(CLIP), 'k-', lw=0.5, zorder=0)
            y_trace += dy

            if i == 0:
                h_bp[0].set_label('ripple-band')
                h_eeg[0].set_label('eeg')

        ax.plot(t, zamp, 'r-', lw=1.5, label='avg z-ampl.', zorder=2)
        ax.axhline(self.Z0, c='r', ls='--', zorder=-1)
        ax.axhline(self.Z1, c='r', ls='--', zorder=-1)
        ax.axhline(0.0, c='k', ls='-', zorder=-2)

        y_ts_text = 6
        for ripple in ripples:
            t_ripple = data.T_(ripple)
            ax.text(t_ripple[0], y_ts_text, str(ripple[0]), ha='right',
                size='small', family='monospace', clip_box=ax.bbox)
            ax.text(t_ripple[2], y_ts_text, str(ripple[2]), ha='left',
                size='small', family='monospace', clip_box=ax.bbox)
            ax.axvline(t_ripple[1], c='r', ls='-', lw=1, zorder=5)
            ax.axvspan(t_ripple[0], t_ripple[2], facecolor='r', lw=0,
                zorder=-2, alpha=0.2)
        ax.set_title('Rat %d, Day %d, M%d: %s'%(rds +
            (', '.join(['Sc%02d'%tt for tt in pyr_tt]),)))
        ax.set_xlim(0, 10)

        ax.legend()
        plt.ion()
        plt.show()

    def plot_chewing(self, rds, ax=None, debug=True):
        """Plot EEG signal and components of the chewing detection analysis
        """
        signal = dict()
        _old_chew_filter = self.chew_filter
        self.chew_filter = False
        ripples = self.plot_ripples(rds, ax=ax, out=signal)
        chewing = self.detect_chewing(signal['t'], signal['zamp'], out=signal)
        self.chew_filter = _old_chew_filter

        ax = plt.gca()
        plt.ioff()

        data = SessionData.get(rds)
        t_theta = data.T_(signal['t_theta'])
        ripple_theta = signal['ripple_theta']
        theta_mod = signal['z_P_rt']

        ax.plot(t_theta, ripple_theta, 'm-', lw=1, label='ripple theta')
        ax.plot(t_theta, theta_mod, 'g-', lw=2, label='chewing')
        ax.axhline(self.chew_Z0, c='g', ls='--', zorder=-1)

        y_ts_text = 5
        for barrage in chewing:
            t_barrage = data.T_(barrage)
            ax.text(t_barrage[0], y_ts_text, str(barrage[0]), ha='right',
                size='small', family='monospace', clip_box=ax.bbox)
            ax.text(t_barrage[1], y_ts_text, str(barrage[1]), ha='left',
                size='small', family='monospace', clip_box=ax.bbox)
            ax.axvspan(t_barrage[0], t_barrage[1], facecolor='g',
                lw=0, zorder=-3, alpha=0.2)
            if debug:
                sys.stdout.write("chewing: %.2f -> %.2f\n"%tuple(t_barrage))

        ax.legend()
        plt.ion()
        plt.show()
