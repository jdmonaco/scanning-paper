# coding: utf-8

"""
chewing.py -- Chewing event report

Created by Joe Monaco on May 15, 2015.
Copyright (c) 2015 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

import numpy as np

from scanr.session import SessionData
from scanr.data import get_node, unique_sessions
from scanr.time import time_slice
from scanr.eeg import get_eeg_timeseries, CLIP
from scanr.ripples import find_pyramidale_tetrodes
from scanr.chewing import ChewingFilter

from .core.report import BaseReport
from .tools.plot import quicktitle


class ChewingReport(BaseReport):

    """
    Show individual detected chewing events for a given dataset
    """

    label = 'chewing report'

    nrows = 12
    ncols = 8

    def collect_data(self, dataset=(57,1), lag=0.25):
        """Detect all chews in dataset and plot events and signals."""
        chewing_table = get_node('/behavior', 'chewing')
        tetrode_query = '(area=="CA1")&(EEG==True)'
        dataset_query = '(rat==%d)&(day==%d)'%dataset
        pyr_tetrodes = find_pyramidale_tetrodes(dataset, verbose=False)
        rat, day = dataset

        # Initialize accumulators
        time_slices = []
        EEG_slices = []
        power_slices = []
        events = []
        timestamps = []

        Chewing = ChewingFilter()

        # Loop through sessions, detecting and storing ripple slices
        for rds in unique_sessions(chewing_table, condn=dataset_query):
            data = SessionData.get(rds)

            self.out('Loading data for rat%03d-%02d-m%d...'%rds)
            ts = None
            EEG = None
            P = None
            for tt in pyr_tetrodes:
                X = get_eeg_timeseries(rds, tt)
                if X is None:
                    continue
                if ts is None:
                    ts = X[0]
                if EEG is None:
                    EEG = X[1]
                else:
                    EEG = np.vstack((EEG, X[1]))
                if P is None:
                    P = Chewing.power(X[1])
                else:
                    P = np.vstack((P, Chewing.power(X[1])))

            if P.ndim == 2:
                P = np.mean(P, axis=0)

            ts_chewing = [(rec['start'], rec['end'])
                for rec in chewing_table.where(data.session_query)]
            t = data.T_(ts)

            for timing in ts_chewing:
                start, end = data.T_(timing)
                peak = (start + end) / 2
                chunk = time_slice(t, peak - lag, peak + lag)
                time_slices.append(t[chunk] - peak)
                EEG_slices.append(EEG[...,chunk])
                power_slices.append(P[chunk])
                events.append((start - peak, end - peak))
                timestamps.append(timing[1])

        self.out('Plotting EEG traces of chewing events...')
        LW = 0.4
        norm = lambda x: x.astype('d') / float(CLIP)
        for i, ax in self.get_plot(range(len(time_slices))):
            t_chunk = time_slices[i]
            traces = EEG_slices[i]
            if traces.ndim == 1:
                ax.plot(t_chunk, norm(traces), 'k-', lw=1.5*LW,
                    alpha=1, zorder=0)
            else:
                ax.plot(t_chunk, norm(traces).T, 'k-', lw=LW,
                    alpha=0.5, zorder=-1)
                ax.plot(t_chunk, norm(np.mean(traces, axis=0)), 'k-', lw=LW,
                    alpha=1, zorder=0)
            ax.plot(t_chunk, power_slices[i] / power_slices[i].max(), 'b-',
                lw=1.5*LW, alpha=1, zorder=1)
            ax.axhline(0, ls='-', c='k', lw=LW, zorder=0)
            ax.axvline(events[i][0], ls='-', c='k', lw=LW, zorder=2)
            ax.axvline(0, ls=':', c='k', lw=LW, zorder=2, alpha=0.5)
            ax.axvline(events[i][1], ls='-', c='k', lw=LW, zorder=2)
            ax.set_xlim(-lag, lag)
            ax.set_ylim(-1, 1)
            ax.set_axis_off()
            quicktitle(ax, '%d'%timestamps[i], size='xx-small')

            self.out.printf('.')
        self.out.printf('\n')
