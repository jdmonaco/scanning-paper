# encoding: utf-8
"""
phase_coupling.py -- Phase-amplitude coupling during head scanning movements

Created by Joe Monaco on May 8, 2012.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Package imports
from scanr.lib import Config
from scanr.session import SessionData
from scanr.spike import TetrodeSelect, find_pyramidale_tetrode, find_theta_tetrode
from scanr.time import select_from, exclude_from
from scanr.meta import get_maze_list
from scanr.eeg import (get_eeg_timeseries, phase_modulation_timeseries,
    phase_amplitude_distribution as PAD, plottable_phase_distribution,
    modulation_index)

# Local imports
from .core.analysis import AbstractAnalysis
from .tools.misc import Reify
from .tools.plot import quicktitle, shaded_error

# Constants
CfgBand = Config['band']


class ScanPhaseCoupling(AbstractAnalysis):

    """
    Analyze relationship between head scanning event and EEG frequency
    """

    label = "phase coupling"

    def collect_data(self, area='CA1', phase_band='theta', amp_band='gamma',
        tetrode='theta', cycles=2, nbins=72):
        """Collate phase-amplitude modulation data about head-scan events
        """
        tetrode_query = '(area=="%s")&(EEG==True)'%area
        dataset_list = TetrodeSelect.datasets(tetrode_query,
            allow_ambiguous=True)

        self.results['phase_band'] = phase_band
        self.results['amp_band'] = amp_band

        # Dataset accumulators
        rat_number = self.results['rat_number'] = []
        P_running = {}
        P_scan = {}
        P_pause = {}

        for dataset in dataset_list:
            rat, day = dataset

            # Find the tetrode with the higheset overall relative theta power
            if tetrode == 'theta':
                roi_tt, _rtheta = find_theta_tetrode(dataset, condn=tetrode_query,
                    ambiguous=True)
            else:
                roi_tt = find_pyramidale_tetrode(dataset, condn=tetrode_query,
                    ambiguous=True)
            self.out('Rat%03d-%02d: using tetrode Sc%d'%(rat, day, roi_tt))

            # Session accumulators
            phase_t = np.array([], 'i8')
            phase = np.array([], 'd')
            running_amp_t = np.array([], 'i8')
            running_amp = np.array([], 'd')
            scan_amp_t = np.array([], 'i8')
            scan_amp = np.array([], 'd')
            pause_amp_t = np.array([], 'i8')
            pause_amp = np.array([], 'd')

            self.out('Collating phase-amplitude data for rat%03d-%02d...'%dataset)
            for session in get_maze_list(rat, day):
                rds = rat, day, session
                data = SessionData.get(rds)

                phase_data, amp_data = phase_modulation_timeseries(
                    *get_eeg_timeseries(rds, roi_tt), phase=phase_band, amp=amp_band)
                t_phase, phi_x = phase_data
                t_amp, A_x = amp_data

                phase_t = np.r_[phase_t, t_phase]
                phase = np.r_[phase, phi_x]

                ix = data.velocity_filter(t_amp)
                running_amp_t = np.r_[running_amp_t, t_amp[ix]]
                running_amp = np.r_[running_amp, A_x[ix]]

                ix = select_from(t_amp, data.scan_list)
                scan_amp_t = np.r_[scan_amp_t, t_amp[ix]]
                scan_amp = np.r_[scan_amp, A_x[ix]]

                ix = select_from(t_amp, data.pause_list)
                pause_amp_t = np.r_[pause_amp_t, t_amp[ix]]
                pause_amp = np.r_[pause_amp, A_x[ix]]

            # Initialize per-rat phase distributions
            if rat not in rat_number:
                rat_number.append(rat)
                P_running[rat] = []
                P_scan[rat] = []
                P_pause[rat] = []

            self.out('...computing phase distributions...')
            phase_series = (phase_t, phase)
            P_running[rat].append(PAD(phase_series, (running_amp_t, running_amp), nbins=nbins))
            P_scan[rat].append(PAD(phase_series, (scan_amp_t, scan_amp), nbins=nbins))
            P_pause[rat].append(PAD(phase_series, (pause_amp_t, pause_amp), nbins=nbins))

        self.out('Averaging dataset distributions to rat distributions...')
        norm = lambda P: P / P.sum()
        for rat in rat_number:
            P_running[rat] = norm(np.array(P_running[rat]).mean(axis=0))
            P_scan[rat] = norm(np.array(P_scan[rat]).mean(axis=0))
            P_pause[rat] = norm(np.array(P_pause[rat]).mean(axis=0))

        # Initialize data storage and accumulators
        running_distro = []
        running_index = []
        scan_distro = []
        scan_index = []
        pause_distro = []
        pause_index = []

        self.out('Computing display distributions and modulation indexes...')
        plottable = lambda P: plottable_phase_distribution(P, cycles=cycles)
        for rat in rat_number:
            phase_bins, P = plottable(P_running[rat])
            if 'phase_bins' not in self.results:
                self.results['phase_bins'] = phase_bins

            running_distro.append(P)
            running_index.append(modulation_index(P_running[rat]))

            scan_distro.append(plottable(P_scan[rat])[1])
            scan_index.append(modulation_index(P_scan[rat]))

            pause_distro.append(plottable(P_pause[rat])[1])
            pause_index.append(modulation_index(P_pause[rat]))

        # Store results data
        self.results['running_distro'] = np.array(running_distro)
        self.results['running_index'] = np.array(running_index)
        self.results['scan_distro'] = np.array(scan_distro)
        self.results['scan_index'] = np.array(scan_index)
        self.results['pause_distro'] = np.array(pause_distro)
        self.results['pause_index'] = np.array(pause_index)

        # Good-bye!
        self.out('All done!')

    def process_data(self):
        """Display frequency distributions and statistics for scan/pause events
        """
        plt.ioff()

        # Load results data
        res = Reify(self.results)
        N = len(res.rat_number)
        F = res.phase_bins
        plim = F[0], F[-1]+(F[-1]-F[-2])

        def CI(P):
            return P.mean(axis=0), 1.96 * P.std(axis=0) / np.sqrt(P.shape[0])

        def plot_rat_distros(ax, data, label, xlim=plim):
            ax.imshow(data, interpolation='nearest', origin='upper', aspect='auto',
                extent=[xlim[0], xlim[1], 0, N])
            ax.set(yticks=(np.arange(N)+0.5), yticklabels=map(str, res.rat_number[::-1]))
            ax.tick_params(axis='y', right=False, labelsize='xx-small', direction='out', length=3)
            quicktitle(ax, '%s'%label)

        # Create the summary comparison figure
        self.figure = {}
        band_label = '%s-%s'%(res.phase_band.title(), res.amp_band.title())
        self.figure['phase_modulations'] = f = plt.figure(figsize=(11,8.5))
        f.suptitle('%s Modulation During Running and Non-running Behaviors'%band_label)
        running_label = 'RUN'
        ax = f.add_subplot(221)
        ax.plot(F, res.running_distro.mean(axis=0), 'k-', lw=1, label=running_label)
        ax.plot(F, res.pause_distro.mean(axis=0), 'r-', lw=1, label='PAU')
        ax.plot(F, res.scan_distro.mean(axis=0), 'b-', lw=1, label='SCN')
        shaded_error(*((F,) + CI(res.running_distro)), ax=ax, fc='k', alpha=0.4)
        shaded_error(*((F,) + CI(res.pause_distro)), ax=ax, fc='r', alpha=0.4)
        shaded_error(*((F,) + CI(res.scan_distro)), ax=ax, fc='b', alpha=0.4)
        ax.legend(loc=4)
        ax.set_xlim(plim)
        quicktitle(ax, 'all sessions')

        plot_rat_distros(f.add_subplot(222), res.scan_distro, 'scan phase distro')
        plot_rat_distros(f.add_subplot(223), res.pause_distro, 'pause phase distro')
        plot_rat_distros(f.add_subplot(224), res.running_distro, 'running phase distro')

        # Create the modulation index bar-plot figure
        self.figure['modulation_index'] = f = plt.figure(figsize=(11,8.5))
        f.suptitle('%s Modulation Index across Behaviors'%band_label)

        index_data = np.c_[res.running_index, res.scan_index, res.pause_index]
        index_labels = ['RUN', 'SCN', 'PAU']

        width = 0.8
        lefts = np.array([0, 1, 2]) - width / 2
        bar_fmt = dict(ecolor='k', capsize=0, color=['k', 'b', 'r'], bottom=0,
            width=width)

        ax = f.add_subplot(221)
        ax.bar(lefts, index_data.mean(axis=0), yerr=CI(index_data)[1], **bar_fmt)
        ax.set(xticks=[0, 1, 2], xticklabels=index_labels, xlim=(-0.5, 2.5))
        ax.tick_params(labelsize='x-small', top=False)
        quicktitle(ax, 'rats')

        ax = f.add_subplot(222)
        ax.plot([0, 1, 2], index_data.T, 'k-', lw=0.75, alpha=0.6)
        ax.set(xticks=[0, 1, 2], xticklabels=index_labels, xlim=(-0.5, 2.5))
        ax.tick_params(labelsize='x-small', top=False)
        quicktitle(ax, 'rats [SCN>RUN: %d/%d]'%(
            (index_data[:,1]>index_data[:,0]).sum(), index_data.shape[0]))

        # Create the rat cross-correlations figure
        self.figure['correlations'] = f = plt.figure(figsize=(11,8.5))
        f.suptitle('Behavioral Cross-correlations of %s Modulation'%band_label)

        clim = (-plim[1] / 2, plim[1] / 2)
        running_acorr = np.empty_like(res.running_distro)
        scan_xcorrs = np.empty_like(res.running_distro)
        pause_xcorrs = np.empty_like(res.running_distro)
        scan_pause_xcorrs = np.empty_like(res.running_distro)
        for i in xrange(N):
            running_acorr[i] = np.correlate(res.running_distro[i], res.running_distro[i], mode='same')
            scan_xcorrs[i] = np.correlate(res.running_distro[i], res.scan_distro[i], mode='same')
            pause_xcorrs[i] = np.correlate(res.running_distro[i], res.pause_distro[i], mode='same')
            scan_pause_xcorrs[i] = np.correlate(res.pause_distro[i], res.scan_distro[i], mode='same')

        plot_rat_distros(f.add_subplot(221), scan_xcorrs, 'C[running]', xlim=clim)
        plot_rat_distros(f.add_subplot(222), scan_xcorrs, 'C[scan, running]', xlim=clim)
        plot_rat_distros(f.add_subplot(223), pause_xcorrs, 'C[pause, running]', xlim=clim)
        plot_rat_distros(f.add_subplot(224), scan_pause_xcorrs, 'C[scan, pause]', xlim=clim)

        plt.draw()
        plt.show()
