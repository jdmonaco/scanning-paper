# encoding: utf-8
"""
scan_ripples.py -- Analysis of relationship between ripples and head scanning

Created by Joe Monaco on April 20, 2012.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from scipy.signal import gaussian, convolve, medfilt
from scipy.interpolate import interp1d

# Package imports
from scanr.session import SessionData
from scanr.spike import xcorr, find_theta_tetrode
from scanr.tracking import plot_track_underlay
from scanr.data import get_node, unique_datasets, unique_rats, get_unique_row
from scanr.time import (stamp_to_time, time_slice, time_slice_sample, select_from,
    exclude_from)
from scanr.meta import get_maze_list
from scanr.eeg import Theta, get_eeg_timeseries
from scanr.behavior import ScanPoints, ScanPhases

# Local imports
from .core.analysis import AbstractAnalysis
from scanr.tools.plot import quicktitle, shaded_error
from scanr.tools.filters import quick_boxcar

# Constants
THETA_POWER_SMOOTHING = 0.050 # 50 ms
THETA_FREQ_SMOOTHING = 0.300 # 300 ms

# Helper functions
F = lambda x, y: interp1d(x, y, fill_value=0.0, bounds_error=False)
Z = lambda x: (x - np.median(x)) / np.std(x)
norm = lambda x: x / np.trapz(x)

# Table descriptions
SessionDescr =  {   'id'        :   tb.UInt16Col(pos=1),
                    'rat'       :   tb.UInt16Col(pos=2),
                    'day'       :   tb.UInt16Col(pos=3),
                    'session'   :   tb.UInt16Col(pos=4),
                    'type'      :   tb.StringCol(itemsize=4, pos=5),
                    't_theta'   :   tb.StringCol(itemsize=16, pos=6),
                    'ZP_theta'  :   tb.StringCol(itemsize=16, pos=7),
                    'f_theta'   :   tb.StringCol(itemsize=16, pos=8),
                    'scans'     :   tb.StringCol(itemsize=16, pos=9),
                    'pauses'    :   tb.StringCol(itemsize=16, pos=10),
                    'ripples'   :   tb.StringCol(itemsize=16, pos=11)    }


class ScanRipples(AbstractAnalysis):

    """
    Analyze relationship between the LFP and head-scanning behavior
    """

    label = "scan ripples"

    def collect_data(self):
        """Collate ripple and head-scan events across CA3/CA1 datasets
        """
        scan_table = get_node('/behavior', 'scans')
        ripple_table = get_node('/physiology', 'ripples')

        # Get datasets, sessions, and rats with detected ripples
        self.results['datasets'] = dataset_list = unique_datasets(ripple_table)
        self.results['N_datasets'] = len(dataset_list)
        session_list = []
        for dataset in dataset_list:
            session_list.extend([dataset + (maze,)
                for maze in get_maze_list(*dataset)])
        self.results['sessions'] = session_list
        self.results['N_sessions'] = len(session_list)
        self.results['rats'] = rat_list = unique_rats(ripple_table)
        self.results['N_rats'] = len(rat_list)

        # Open a new data file
        data_file = self.open_data_file()
        array_group = data_file.createGroup('/', 'arrays', title='Array Data')
        session_table = data_file.createTable('/', 'sessions', SessionDescr,
            'Sessions for Scan-Ripple Analysis')

        # Loop through sessions, detecting ripples and getting head scans
        id_fmt = 'data_%06d'
        array_id = 0
        session_id = 0
        row = session_table.row
        for rds in session_list:
            rds_str = 'rat%d-%02d-m%d...'%rds
            self.out('Loading data for %s'%rds_str)
            data = SessionData.get(rds)
            theta_tt = find_theta_tetrode(rds[:2], condn='(EEG==True)&(area=="CA1")')
            if theta_tt is None:
                continue
            theta_tt = theta_tt[0]
            self.out('Using theta tetrode Sc%02d.'%theta_tt)

            row['id'] = session_id
            row['rat'], row['day'], row['session'] = rds
            if data.attrs['type'] in ('STD', 'MIS'):
                row['type'] = 'DR'
            else:
                row['type'] = 'NOV'

            # Compute smoothed theta power and frequency time-series
            ts, EEG = get_eeg_timeseries(rds, theta_tt)
            t = data.T_(ts) # time represented as elapsed time within session
            t_theta, x_theta = Theta.timeseries(t, EEG)
            ZP_theta = Z(Theta.power(x_theta, filtered=True))
            f_theta = quick_boxcar(Theta.frequency(x_theta, filtered=True),
                M=(lambda i: (i % 2 == 0) and (i+1) or i)(int(Theta.fs * THETA_FREQ_SMOOTHING)))

            # Get scans, pauses, and ripples
            scans = data.T_([tuple(map(lambda k: rec[k], ScanPoints))
                for rec in scan_table.where(data.session_query)])
            pauses = data.T_(data.pause_list)
            ripples = data.T_([(rec['start'], rec['peak'], rec['end'])
                for rec in ripple_table.where(data.session_query)])

            # Save the array data as resources for analysis
            data_file.createArray(array_group, id_fmt%array_id, t_theta,
                title='%s t_theta'%rds_str)
            row['t_theta'] = id_fmt%array_id
            array_id += 1

            data_file.createArray(array_group, id_fmt%array_id, ZP_theta,
                title='%s ZP_theta'%rds_str)
            row['ZP_theta'] = id_fmt%array_id
            array_id += 1

            data_file.createArray(array_group, id_fmt%array_id, f_theta,
                title='%s f_theta'%rds_str)
            row['f_theta'] = id_fmt%array_id
            array_id += 1

            data_file.createArray(array_group, id_fmt%array_id, scans,
                title='%s scans'%rds_str)
            row['scans'] = id_fmt%array_id
            array_id += 1

            data_file.createArray(array_group, id_fmt%array_id, pauses,
                title='%s pauses'%rds_str)
            row['pauses'] = id_fmt%array_id
            array_id += 1

            data_file.createArray(array_group, id_fmt%array_id, ripples,
                title='%s ripples'%rds_str)
            row['ripples'] = id_fmt%array_id
            array_id += 1

            row.append()
            if array_id % 10 == 0:
                session_table.flush()
            session_id += 1

        # Good-bye
        session_table.flush()
        self.out('All done!')

    def new_figure(self, label, title, size=(9,8)):
        """Helper method to create a new figure
        """
        if self.figure is None:
            self.figure = {}
        self.figure[label] = f = plt.figure(figsize=size)
        f.suptitle(title)
        return f

    def run_signal_xcorrs(self, event='scans', ripple_nonscan=False,
        signal='ZP_theta', lag=3, fc='b', ylim=None):
        """Cross-correlations of behaviors with continuous signals
        """
        data_file = self.get_data_file()
        session_table = data_file.root.sessions

        if event == 'scans':
            event_points = ScanPoints
        elif event == 'pauses':
            event_points = ('start', 'end')
        elif event == 'ripples':
            event_points = ('start', 'peak', 'end')
        else:
            raise ValueError, 'unknown event type "%s"'%event

        rat_averages = {}
        rat_slices = { k: {} for k in event_points }

        lag_samples = 2 * lag * Theta.fs

        for session in session_table.iterrows():
            self.out.printf('.')

            t_theta = data_file.getNode('/arrays', session['t_theta'])[:]
            x_theta = data_file.getNode('/arrays', session[signal])[:]
            events = data_file.getNode('/arrays', session[event])[:]
            rat = session['rat']

            # If analyzing ripples, need scans to restrict to scan-ripples
            if event == 'ripples':
                scans = data_file.getNode('/arrays', session['scans'])[:]

            for i, phase in enumerate(event_points):
                if events.size == 0:
                    continue
                t_event_list = events[:,i]

                # Restrict events list if analyzing ripples to just scan-ripples
                if event == 'ripples':
                    if ripple_nonscan:
                        t_event_list = t_event_list[
                            exclude_from(t_event_list, scans[:,ScanPhases['related']])]
                    else:
                        t_event_list = t_event_list[
                            select_from(t_event_list, scans[:,ScanPhases['related']])]

                for t_scan in t_event_list:
                    scan_ix = np.argmin(np.abs(t_theta - t_scan))
                    win_ix = scan_ix - lag_samples, scan_ix + lag_samples
                    if win_ix[0] < 0 or win_ix[1] > x_theta.size:
                        continue

                    signal_slice = x_theta[win_ix[0]:win_ix[1]]

                    if rat in rat_slices[phase]:
                        if signal_slice.size != rat_slices[phase][rat].shape[1]:
                            self.out('Bad signal size: rat%d @ t = %.2f'%(rat, t_scan))
                            continue
                        rat_slices[phase][rat] = np.vstack(
                            (rat_slices[phase][rat], signal_slice))
                    else:
                        rat_slices[phase][rat] = signal_slice[np.newaxis]

        self.out.printf('\n')

        self.out('Averaging rat signal slices...')
        for phase in event_points:
            N_rats = len(rat_slices[phase].keys())
            averages = []
            for rat in rat_slices[phase].keys():
                averages.append(rat_slices[phase][rat].mean(axis=0))
            rat_averages[phase] = np.array(averages)

        plt.ioff()
        f = self.new_figure('%s_%s_xcorrs'%(signal, event[:-1]),
            '%s x %s Phase Cross-Correlations'%(signal, event[:-1].title()),
            size=(16,10))

        nrows, ncols = 2, 3

        for i, phase in enumerate(event_points):
            self.out('Plotting xcorrs for "%s" phase'%phase)
            N_rats, N_t = rat_averages[phase].shape
            mu = rat_averages[phase].mean(axis=0) - 0.2
            sem = rat_averages[phase].std(axis=0) / np.sqrt(N_rats)
            t = np.linspace(-lag, lag, N_t)
            ax = f.add_subplot(nrows, ncols, i+1)
            ax.plot(t, mu, '-', c=fc, zorder=1)
            shaded_error(t, mu, sem, ax=ax, fc=fc, alpha=0.3, zorder=0)
            ax.axhline(0.0, c='k', ls='-', zorder=-2)
            ax.set_xlim(-lag, lag)
            if ylim is not None:
                ax.set_ylim(ylim)
            elif signal == 'ZP_theta':
                ax.set_ylim(-0.4, 0.4)
            elif signal == 'f_theta':
                ax.set_ylim(7.2, 7.7)
            ax.tick_params(top=False, right=False, left=False)
            quicktitle(ax, phase.title())

        plt.ion()
        plt.show()

    def plot_scan_example(self, rds=(95,4,1), scan_number=22, margin=1.0):
        """Plot example scan with relevant behavior variables and scan phases
        """
        data = SessionData.get(rds)
        scan_table = get_node('/behavior', 'scans')
        scan = get_unique_row(scan_table,
            data.session_query + '&(number==%d)'%scan_number)

        t_scan = { k: data.T_(scan[k]) for k in ScanPoints }
        window = t_scan['downshift'] - margin, t_scan['upshift'] + margin

        plt.ioff()
        f = self.new_figure('scan_example',
            'Scan Example: Rat %d, Day %d, M%d @ t=%.2f'%(rds + (window[0],)))

        traj = data.trajectory
        t = data.T_(traj.ts)
        s = time_slice(t, start=window[0], end=window[1])
        scan_slice = time_slice(t, start=t_scan['start'], end=t_scan['end'])

        y = 0
        dy = -2
        norm = lambda x: x / np.max(np.abs(x))
        ax = f.add_subplot(111)
        ax.axhline(y, c='k', ls='-', zorder=0)
        ax.plot(t[s], y + norm(traj.radius[s]), 'b-', lw=2, zorder=1); y += dy
        ax.axhline(y, c='k', ls='-', zorder=0)
        ax.plot(t[s], y + norm(traj.radial_velocity[s]), 'b-', lw=2, zorder=1); y += dy
        ax.axhline(y, c='k', ls='-', zorder=0)
        ax.plot(t[s], y + norm(traj.forward_velocity[s]), 'b-', lw=2, zorder=1); y += dy

        ax.axvspan(t_scan['downshift'], t_scan['start'], lw=0, fc='g', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['start'], t_scan['max'], lw=0, fc='m', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['max'], t_scan['return'], lw=0, fc='y', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['return'], t_scan['end'], lw=0, fc='m', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['end'], t_scan['upshift'], lw=0, fc='g', alpha=0.3, zorder=-2)

        for k in ScanPoints:
            ax.axvline(t_scan[k], c='k', ls='-', zorder=-1)

        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['r', 'rs', 'fwd'])
        ax.set_ylim(-4.75, 1.25)
        ax.set_xlim(*window)
        ax.tick_params(top=False, right=False)

        f = self.new_figure('scan_example_space', 'Scan Example', (4,4))
        ax = f.add_subplot(111)
        ax.plot(traj.x[s], traj.y[s], 'k-', alpha=0.6)
        ax.plot(traj.x[scan_slice], traj.y[scan_slice], 'r-', lw=2)
        ax.axis('equal')
        ax.set_axis_off()
        plot_track_underlay(ax=ax, ls='dotted')

        plt.ion()
        plt.show()

    def plot_LFP_example(self, rds=(95,4,1), scan_number=22, margin=2.0):
        """Plot example LFP traces showing theta power during a head scan
        """
        data = SessionData.get(rds)
        scan_table = get_node('/behavior', 'scans')
        scan = get_unique_row(scan_table,
            data.session_query + '&(number==%d)'%scan_number)

        t_scan = { k: data.T_(scan[k]) for k in ScanPoints }
        window = t_scan['downshift'] - margin, t_scan['upshift'] + margin

        plt.ioff()
        f = self.new_figure('scan_LFP_example',
            'Scan LFP Example: Rat %d, Day %d, M%d @ t=%.2f'%(rds + (window[0],)))

        theta_tt = find_theta_tetrode(rds[:2], condn='(EEG==True)&(area=="CA1")')[0]
        ts, EEG = get_eeg_timeseries(rds, theta_tt)
        t_EEG = data.T_(ts)

        data_file = self.get_data_file()
        session = get_unique_row(data_file.root.sessions, data.session_query)
        t_theta, x_theta = Theta.timeseries(t_EEG, EEG)
        ZP_theta = Z(Theta.power(x_theta, filtered=True)) #data_file.getNode('/arrays', session['ZP_theta'])
        f_theta = data_file.getNode('/arrays', session['f_theta'])

        traj = data.trajectory
        t_traj = data.T_(traj.ts)
        s_traj = time_slice(t_traj, start=window[0], end=window[1])
        s_EEG = time_slice(t_EEG, start=window[0], end=window[1])
        s_theta = time_slice(t_theta, start=window[0], end=window[1])

        y = 0
        dy = -2
        norm = lambda x: (x - np.mean(x)) / (1.1 * float(np.max(np.abs((x - np.mean(x))))))
        ax = f.add_subplot(111)
        ax.axhline(y, c='k', ls='-', zorder=0)
        ax.plot(t_traj[s_traj], y + norm(traj.radius[s_traj]), 'b-', lw=2, zorder=1); y += dy
        ax.plot(t_EEG[s_EEG], y + norm(EEG[s_EEG]), 'k-', lw=1, zorder=1); y += dy
        ax.plot(t_theta[s_theta], y + norm(x_theta[s_theta]), 'k-', lw=1, zorder=1); y += dy
        ax.plot(t_theta[s_theta], y + norm(ZP_theta[s_theta]), 'k-', lw=1, zorder=1); y += dy
        ax.plot(t_theta[s_theta], y + norm(f_theta[s_theta]), 'k-', lw=1, zorder=1); y += dy

        ax.axvspan(t_scan['downshift'], t_scan['start'], lw=0, fc='g', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['start'], t_scan['max'], lw=0, fc='m', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['max'], t_scan['return'], lw=0, fc='y', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['return'], t_scan['end'], lw=0, fc='m', alpha=0.3, zorder=-2)
        ax.axvspan(t_scan['end'], t_scan['upshift'], lw=0, fc='g', alpha=0.3, zorder=-2)

        ax.set_yticks([0, -2, -4, -6, -8])
        ax.set_yticklabels(['r', 'EEG', 'theta', 'ZP_theta', 'f_theta'])
        ax.set_ylim(-8.75, 1.25)
        ax.set_xlim(*window)
        ax.tick_params(top=False, right=False)

        plt.ion()
        plt.show()

    def run_ripple_xcorrs(self, lag=4, numbins=71, ripple_lock='peak'):
        """Compute scan-ripple cross-correlograms
        """
        # Load results data
        data_file = self.get_data_file()
        sessions = data_file.root.sessions
        scan_table = get_node('/behavior', 'scans')

        # Correlogram bins
        edges = np.linspace(-lag, lag, numbins+1)
        centers = (edges[:-1] + edges[1:]) / 2

        # Overall scanning xcorrs
        self.out("Computing ripple-scan cross-correlations...")
        r_ix = dict(start=0, peak=1, end=2)[ripple_lock]
        C = { k: np.zeros(numbins, 'd') for k in ScanPoints }
        C['pstart'] = np.zeros(numbins, 'd')
        C['pend'] = np.zeros(numbins, 'd')

        for session in sessions.iterrows():
            scans = data_file.getNode('/arrays', session['scans'])
            pauses = data_file.getNode('/arrays', session['pauses'])
            ripples = data_file.getNode('/arrays', session['ripples'])
            if (len(scans) and len(ripples)):
                for i, pt in enumerate(ScanPoints):
                    C[pt] += xcorr(scans[:,i], ripples[:,r_ix], maxlag=lag, bins=numbins)[0]
            if (len(pauses) and len(ripples)):
                C['pstart'] += xcorr(pauses[:,0], ripples[:,1], maxlag=lag, bins=numbins)[0]
                C['pend'] += xcorr(pauses[:,1], ripples[:,1], maxlag=lag, bins=numbins)[0]

        f = self.new_figure('xcorrs', 'Scan-Ripple Correlations', (11,12))

        ax = f.add_subplot(321)
        ax.plot(centers, C['downshift'], drawstyle='steps-mid', label='down')
        ax.plot(centers, C['upshift'], drawstyle='steps-mid', label='up')
        ax.legend(loc=0)
        ax.set(xlim=(-lag, lag), xticks=[-lag, -lag/2., 0, lag/2., lag],
            yticks=[])
        ax.set_ylim(bottom=0)
        ax.tick_params(top=False)
        quicktitle(ax, 'Gear Shifting x Ripples')

        ax = f.add_subplot(322)
        ax.plot(centers, C['start'], drawstyle='steps-mid', label='start')
        ax.plot(centers, C['end'], drawstyle='steps-mid', label='end')
        ax.legend(loc=0)
        ax.set(xlim=(-lag, lag), xticks=[-lag, -lag/2., 0, lag/2., lag],
            yticks=[])
        ax.set_ylim(bottom=0)
        ax.tick_params(top=False)
        quicktitle(ax, 'Scans x Ripples')

        ax = f.add_subplot(323)
        ax.plot(centers, C['max'], drawstyle='steps-mid', label='max')
        ax.plot(centers, C['return'], drawstyle='steps-mid', label='return')
        ax.legend(loc=0)
        ax.set(xlim=(-lag, lag), xticks=[-lag, -lag/2., 0, lag/2., lag],
            yticks=[])
        ax.set_ylim(bottom=0)
        ax.tick_params(top=False)
        quicktitle(ax, 'Dwells x Ripples')

        ax = f.add_subplot(324)
        ax.plot(centers, C['pstart'], drawstyle='steps-mid', label='pause start')
        ax.plot(centers, C['pend'], drawstyle='steps-mid', label='pause end')
        ax.legend(loc=0)
        ax.set(xlim=(-lag, lag), xticks=[-lag, -lag/2., 0, lag/2., lag],
            yticks=[])
        ax.set_ylim(bottom=0)
        ax.tick_params(top=False)
        quicktitle(ax, 'Pauses x Ripples')

        # Ripple-event fractions across event and experiment types
        self.out("Computing ripple-event fractions...")
        event_types = ('gearshifts', 'scans', 'pauses')
        expt_types = ('DR', 'NOV')
        frac = np.empty((len(event_types)*len(expt_types),), 'd')
        frac_rat_mu = np.empty_like(frac)
        frac_rat_sem = np.empty_like(frac)
        labels = []
        i = 0
        for expt in expt_types:
            for event in event_types:
                hits = N = 0
                hits_rat = {}
                N_rat = {}

                for session in sessions.where('type=="%s"'%expt):
                    if event in ('gearshifts', 'scans'):
                        events = data_file.getNode('/arrays', session['scans'])
                        if event == 'scans':
                            events = events[:,ScanPhases['scan']] # start -> end
                        elif event == 'gearshifts':
                            events = events[:,ScanPhases['related']] # down -> upshift
                    else:
                        events = data_file.getNode('/arrays', session[event])

                    ripples = data_file.getNode('/arrays', session['ripples'])
                    N_events = events.shape[0]
                    if N_events == 0:
                        continue

                    # Count hits based on ripple peaks and 1) pauses: start->end,
                    # and 2) ripples: down->upshift and start->end
                    H = 0
                    for v in events:
                        H += int(np.any(np.logical_and(
                            ripples[:,r_ix] >= v[0], ripples[:,r_ix] < v[-1])))

                    N += N_events
                    hits += H

                    if session['rat'] in N_rat:
                        hits_rat[session['rat']] += H
                        N_rat[session['rat']] += N_events
                    else:
                        hits_rat[session['rat']] = H
                        N_rat[session['rat']] = N_events

                labels.append('%s %s'%(expt, event))

                frac[i] = hits / float(N)
                frac_rat = [hits_rat[rat] / float(N_rat[rat])
                    for rat in N_rat]
                frac_rat_mu[i] = np.mean(frac_rat)
                frac_rat_sem[i] = np.std(frac_rat) / np.sqrt(len(N_rat))

                i += 1

        ax = f.add_subplot(325)
        x = np.array([0, 0.5, 1.0, 2, 2.5, 3.0])
        fmt = dict(mfc='w', mec='k', mew=1, ms=6)
        ax.plot(x + 0.075, frac, 'o', label='overall', **fmt)
        ax.errorbar(x - 0.075, frac_rat_mu, yerr=frac_rat_sem, fmt='s',
            ecolor='k', elinewidth=1.5, capsize=5, label='across rats', **fmt)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, size='small', rotation=45)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Events')
        ax.set_ylabel('Fraction')
        ax.tick_params(top=False, right=False)
        quicktitle(ax, 'Fraction of Events with Ripples', size='small')

        # Fraction of behavior events across session type containing ripples
        self.out("Computing ripple-phase distributions...")
        phase_partition = ('downshift', 'out', 'dwell', 'inb', 'upshift')
        counts = np.zeros(len(phase_partition))
        counts_rat = {}

        for i, phase in enumerate(phase_partition):
            for session in sessions.iterrows():
                ripples = data_file.getNode('/arrays', session['ripples'])
                scans = data_file.getNode('/arrays', session['scans'])

                if not (len(ripples) and len(scans)):
                    continue

                phase_events = scans[:,ScanPhases[phase]]
                hits = np.sum(select_from(ripples[:,r_ix], phase_events))

                counts[i] += hits

                rat = session['rat']
                if rat not in counts_rat:
                    counts_rat[rat] = np.zeros(len(phase_partition))
                counts_rat[rat][i] += hits

        N_rats = len(counts_rat)
        p_phase = np.empty((N_rats, len(phase_partition)), 'd')
        for i, rat in enumerate(counts_rat.keys()):
            p_phase[i] = counts_rat[rat] / counts_rat[rat].sum()

        p_phase_mu = p_phase.mean(axis=0)
        p_phase_sem = p_phase.std(axis=0) / np.sqrt(N_rats)

        ax = f.add_subplot(326)
        x = np.arange(len(phase_partition))
        ax.plot(x + 0.1, counts / counts.sum(), 'o', label='overall', **fmt)
        ax.errorbar(x - 0.1, p_phase_mu, yerr=p_phase_sem, fmt='s',
            ecolor='k', elinewidth=1.5, capsize=5, label='across rats', **fmt)
        ax.set_xticks(x)
        ax.set_xticklabels(('Downshift', 'Outbound', 'Dwell', 'Inbound', 'Upshift'),
            size='small', rotation=45)
        ax.set_xlim(-0.5, len(phase_partition) - 0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Head-Scan Phase')
        ax.set_ylabel('p[Phase]')
        ax.tick_params(top=False, right=False)
        ax.legend(loc=0)
        quicktitle(ax, 'P[phase|ripple]', size='small')

        plt.ion()
        plt.show()
