# encoding: utf-8
"""
event_lfp.py -- Event-level analysis of theta power and ripples to examine
    the conconcurrence of behavioral and LFP events

Created by Joe Monaco on May 20, 2012.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from scipy.interpolate import interp1d

# Package imports
from scanr.session import SessionData
from scanr.spike import TetrodeSelect, find_theta_tetrode
from scanr.data import get_node
from scanr.time import time_slice, select_from
from scanr.meta import get_maze_list
from scanr.eeg import get_eeg_timeseries, Ripple, Theta

# Local imports
from .core.analysis import AbstractAnalysis
from .core.report import BaseReport
from scanr.tools.misc import Reify
from scanr.tools.plot import quicktitle, grouped_bar_plot
from scanr.tools.stats import t_welch, t_one_tailed

# Table descriptions
BehDescr =     {    'id'        :   tb.UInt16Col(pos=1),
                    'rat'       :   tb.UInt16Col(pos=2),
                    'theta_avg' :   tb.Float32Col(pos=3),
                    'theta_max' :   tb.Float32Col(pos=4),
                    'ripples'   :   tb.UInt16Col(pos=5) }

RippleDescr =   {   'rat'       :   tb.UInt16Col(pos=1),
                    'theta'     :   tb.Float32Col(pos=2),
                    'running'   :   tb.BoolCol(pos=3),
                    'pause'     :   tb.BoolCol(pos=4),
                    'scan'      :   tb.BoolCol(pos=5)   }


class ScanThetaRipples(AbstractAnalysis):

    """
    Analyze relationship between head scanning event and EEG ripple events
    """

    label = "theta ripples"

    def collect_data(self, area_query='(area=="CA3")|(area=="CA1")'):
        """Collate ripple, theta power across head-scan events
        """
        scan_table = get_node('/behavior', 'scans')
        pause_table = get_node('/behavior', 'pauses')

        tetrode_query = '(%s)&(EEG==True)'%area_query
        dataset_list = TetrodeSelect.datasets(tetrode_query, allow_ambiguous=True)

        # Tables and iterators
        data_file = self.open_data_file()
        scan_data_table = data_file.createTable('/', 'scan_data',
            BehDescr, title='Scan Data')
        pause_data_table = data_file.createTable('/', 'pause_data',
            BehDescr, title='Pause Data')
        ripple_data_table = data_file.createTable('/', 'ripple_data',
            RippleDescr, title='Ripple Data')
        scan_row = scan_data_table.row
        pause_row = pause_data_table.row
        ripple_row = ripple_data_table.row

        for dataset in dataset_list:
            rat, day = dataset

            # Find the tetrode based on the chosen tetrode strategy
            roi_tt = find_theta_tetrode(dataset, condn=tetrode_query)
            if type(roi_tt) is tuple:
                roi_tt = roi_tt[0]
            self.out('Rat%03d-%02d: using tetrode Sc%d'%(rat, day, roi_tt))

            # Loop through sessions
            for session in get_maze_list(rat, day):
                rds = rat, day, session
                data = SessionData.get(rds)

                ts, EEG = get_eeg_timeseries(rds, roi_tt)
                ripple_list = Ripple.detect(ts, EEG)
                if len(ripple_list):
                    ripple_peaks = np.array(ripple_list)[:,1]
                else:
                    ripple_peaks = np.array([])

                ts_theta, x_theta = Theta.timeseries(ts, EEG)
                zpow = (lambda x: (x - x.mean()) / x.std())(
                    Theta.power(x_theta, filtered=True))

                # Loop through scans and pauses
                for row, table in [(scan_row, scan_table), (pause_row, pause_table)]:
                    for rec in table.where(data.session_query):
                        theta = zpow[select_from(ts_theta, [rec['tlim']])]
                        row['id'] = rec['id']
                        row['rat'] = rat
                        row['theta_avg'] = theta.mean()
                        row['theta_max'] = theta.max()
                        row['ripples'] = select_from(ripple_peaks, [rec['tlim']]).sum()
                        row.append()
                scan_data_table.flush()
                pause_data_table.flush()

                # Loop through ripples
                zpow_t = interp1d(ts_theta, zpow, fill_value=0.0, bounds_error=False)
                for t_ripple in ripple_peaks:
                    ripple_row['rat'] = rat
                    ripple_row['theta'] = zpow_t(t_ripple) # interpolate z-power at ripple peak
                    ripple_row['running'] = data.velocity_filter(t_ripple)
                    ripple_row['scan'] = np.any(select_from([t_ripple], data.scan_list))
                    ripple_row['pause'] = np.any(select_from([t_ripple], data.pause_list))
                    ripple_row.append()
                ripple_data_table.flush()

        self.out('All done!')

    def process_data(self, use_max=False, zlim=(-3,3), numbins=64, siglevel=0.05):
        """Compute scan-ripple event distributions

        use_max -- use the max z-power theta across event instead of average
        zlim -- z-power limits for distributions
        numbins -- number of histogram bins use to compute distributions
        """
        self.out.outfd = file(os.path.join(self.datadir, 'figure.log'), 'w')

        # Load results data
        data_file = self.get_data_file()
        root = data_file.root

        # Overall and rat-specific scan/pause distributions contingent on ripples
        theta = use_max and 'theta_max' or 'theta_avg'
        pause_noripple = np.array([rec[theta] for rec in root.pause_data.where('ripples==0')])
        pause_ripple = np.array([rec[theta] for rec in root.pause_data.where('ripples>0')])
        scan_noripple = np.array([rec[theta] for rec in root.scan_data.where('ripples==0')])
        scan_ripple = np.array([rec[theta] for rec in root.scan_data.where('ripples>0')])

        rat_list = np.unique(root.scan_data.cols.rat[:])
        pause_noripple_rat = {}
        pause_ripple_rat = {}
        scan_noripple_rat = {}
        scan_ripple_rat = {}
        for rat in rat_list:
            pause_noripple_rat[rat] = np.array([
                rec[theta] for rec in root.pause_data.where('(ripples==0)&(rat==%d)'%rat)])
            pause_ripple_rat[rat] = np.array([
                rec[theta] for rec in root.pause_data.where('(ripples>0)&(rat==%d)'%rat)])
            scan_noripple_rat[rat] = np.array([
                rec[theta] for rec in root.scan_data.where('(ripples==0)&(rat==%d)'%rat)])
            scan_ripple_rat[rat] = np.array([
                rec[theta] for rec in root.scan_data.where('(ripples>0)&(rat==%d)'%rat)])

        # Ripple distributions
        ripple_theta = root.ripple_data.cols.theta[:]
        ripple_running = np.array([rec['theta'] for rec in root.ripple_data.where('running==True')])
        ripple_pause = np.array([rec['theta'] for rec in root.ripple_data.where('pause==True')])
        ripple_scan = np.array([rec['theta'] for rec in root.ripple_data.where('scan==True')])

        ripple_theta_rat = {}
        ripple_running_rat = {}
        ripple_pause_rat = {}
        ripple_scan_rat = {}
        for rat in rat_list:
            ripple_theta_rat[rat] = np.array([
                rec['theta'] for rec in root.ripple_data.where('rat==%d'%rat)])
            ripple_running_rat[rat] = np.array([
                rec['theta'] for rec in root.ripple_data.where('(running==True)&(rat==%d)'%rat)])
            ripple_pause_rat[rat] = np.array([
                rec['theta'] for rec in root.ripple_data.where('(pause==True)&(rat==%d)'%rat)])
            ripple_scan_rat[rat] = np.array([
                rec['theta'] for rec in root.ripple_data.where('(scan==True)&(rat==%d)'%rat)])

        # Output statistics
        stat_msgs = []
        stat_msgs += ['Pauses with ripples: %d / %d'%(len(pause_ripple), root.pause_data.nrows)]
        stat_msgs += ['Scans with ripples: %d / %d'%(len(scan_ripple), root.scan_data.nrows)]
        stat_msgs += ['--']
        stat_msgs += ['Pause, ripple vs. no ripple: T = %.4f, p < %.5f'%t_welch(pause_ripple, pause_noripple)]
        stat_msgs += ['Scan, ripple vs. no ripple: T = %.4f, p < %.5f'%t_welch(scan_ripple, scan_noripple)]
        stat_msgs += ['--']
        stat_msgs += ['Ripples during running: %d / %d'%(len(ripple_running), root.ripple_data.nrows)]
        stat_msgs += ['Ripples during pause: %d / %d'%(len(ripple_pause), root.ripple_data.nrows)]
        stat_msgs += ['Ripples during scan: %d / %d'%(len(ripple_scan), root.ripple_data.nrows)]
        stat_msgs += ['--']
        stat_msgs += ['Ripple, overall: %0.3f +/- %0.3f; T = %.4f, p < %.5f'%(
            (ripple_theta.mean(), ripple_theta.std()) + t_one_tailed(ripple_theta, 0))]
        stat_msgs += ['Ripple, running: %0.3f +/- %0.3f; T = %.4f, p < %.5f'%(
            (np.mean(ripple_running), np.std(ripple_running)) + t_one_tailed(ripple_running, 0))]
        stat_msgs += ['Ripple, pause: %0.3f +/- %0.3f; T = %.4f, p < %.5f'%(
            (np.mean(ripple_pause), np.std(ripple_pause)) + t_one_tailed(ripple_pause, 0))]
        stat_msgs += ['Ripple, scan: %0.3f +/- %0.3f; T = %.4f, p < %.5f'%(
            (np.mean(ripple_scan), np.std(ripple_scan)) + t_one_tailed(ripple_scan, 0))]
        stat_msgs += ['--']
        stat_msgs += ['Ripple, scan vs. running: T = %.4f, p < %.5f'%t_welch(ripple_scan, ripple_running)]
        stat_msgs += ['Ripple, scan vs. pause: T = %.4f, p < %.5f'%t_welch(ripple_scan, ripple_pause)]
        self.out('\n'.join(stat_msgs))

        # Histogram / distro
        bins = np.linspace(zlim[0], zlim[1], numbins+1)
        centers = (bins[:-1] + bins[1:]) / 2

        def distro(values):
            H = np.histogram(values, bins=bins)[0]
            return H.astype('d') / H.sum()

        # Create the event|ripple figure
        self.figure = {}
        self.figure['ripple_effect'] = f = plt.figure(figsize=(6,8))
        f.suptitle('Event Distributions of Theta Power Conditioned on Ripples')

        # Pauses
        ax = f.add_subplot(211)
        ax.plot(centers, distro(pause_ripple), 'b-', centers, distro(pause_noripple), 'r-')
        ax.legend(['w/ ripples', 'no ripples'])
        ax.set_xlim(*zlim)
        quicktitle(ax, 'Pauses')

        # Scans
        ax = f.add_subplot(212)
        ax.plot(centers, distro(scan_ripple), 'b-', centers, distro(scan_noripple), 'r-')
        ax.legend(['w/ ripples', 'no ripples'])
        ax.set_xlim(*zlim)
        quicktitle(ax, 'Scans')

        # Create the ripple|behavior figure
        self.figure['behavior'] = f = plt.figure(figsize=(9,4))
        f.suptitle('Ripple Distributions of Theta Power at Peak Across Behaviors')

        ax = f.add_subplot(111)
        ax.plot(centers, distro(ripple_theta), 'k--', label='overall')
        ax.plot(centers, distro(ripple_running), 'k-', label='running')
        ax.plot(centers, distro(ripple_pause), 'r-', label='pauses')
        ax.plot(centers, distro(ripple_scan), 'b-', label='scans')
        ax.legend()
        ax.set_xlim(*zlim)
        quicktitle(ax, 'Ripples')

        # Create per-rat plots of event distribution mean differences
        self.figure['across_rats'] = f = plt.figure(figsize=(11,8))
        f.suptitle('Mean Theta Power Difference Across Rats')

        pause_diff = np.zeros(len(rat_list), 'd')
        scan_diff = np.zeros(len(rat_list), 'd')
        pause_pval = np.ones(len(rat_list), 'd')
        scan_pval = np.ones(len(rat_list), 'd')
        for i, rat in enumerate(rat_list):
            if len(pause_ripple_rat[rat]) > 1 and len(pause_noripple_rat[rat]) > 1:
                pause_diff[i] = pause_ripple_rat[rat].mean() - pause_noripple_rat[rat].mean()
                pause_pval[i] = t_welch(pause_ripple_rat[rat], pause_noripple_rat[rat])[1]
            if len(scan_ripple_rat[rat]) > 1 and len(scan_noripple_rat[rat]) > 1:
                scan_diff[i] = scan_ripple_rat[rat].mean() - scan_noripple_rat[rat].mean()
                scan_pval[i] = t_welch(scan_ripple_rat[rat], scan_noripple_rat[rat])[1]

        pause_sig = pause_pval <= siglevel
        scan_sig = scan_pval <= siglevel
        N = len(rat_list)
        x = np.arange(N)

        for subp, sig, diff, label in [(221,pause_sig,pause_diff,'pause'), (222,scan_sig,scan_diff,'scan')]:
            ax = f.add_subplot(subp)
            h1 = ax.stem(x[sig], diff[sig], linefmt='k-', basefmt='k-', markerfmt='bo')
            h2 = ax.stem(x[True - sig], diff[True - sig], linefmt='k-', basefmt='k-', markerfmt='ro')
            h1[0].set_zorder(100)
            h2[0].set_zorder(100)
            ax.set(xticks=[], xlim=(-1, N))
            ax.tick_params(top=False, right=False)
            ax.tick_params(axis='x', direction='out', labelsize='small')
            if label == "pause":
                ax.set_ylabel(r'$\Delta$[ripples, no ripples]')
            quicktitle(ax, '%ss'%label.title())

        # Create per-rat plots of ripple distribution scan mean differences
        running_diff = np.zeros(len(rat_list), 'd')
        scanpause_diff = np.zeros(len(rat_list), 'd')
        running_pval = np.ones(len(rat_list), 'd')
        scanpause_pval = np.ones(len(rat_list), 'd')
        for i, rat in enumerate(rat_list):
            if len(ripple_scan_rat[rat]) > 1 and len(ripple_running_rat[rat]) > 1:
                running_diff[i] = ripple_scan_rat[rat].mean() - ripple_running_rat[rat].mean()
                running_pval[i] = t_welch(ripple_scan_rat[rat], ripple_running_rat[rat])[1]
            if len(ripple_scan_rat[rat]) > 1 and len(ripple_pause_rat[rat]) > 1:
                scanpause_diff[i] = ripple_scan_rat[rat].mean() - ripple_pause_rat[rat].mean()
                scanpause_pval[i] = t_welch(ripple_scan_rat[rat], ripple_pause_rat[rat])[1]

        running_sig = running_pval <= siglevel
        scanpause_sig = scanpause_pval <= siglevel

        for subp, sig, diff, label in [(223,running_sig,running_diff,'running'), (224,scanpause_sig,scanpause_diff,'pause')]:
            ax = f.add_subplot(subp)
            h1 = ax.stem(x[sig], diff[sig], linefmt='k-', basefmt='k-', markerfmt='bo')
            h2 = ax.stem(x[True - sig], diff[True - sig], linefmt='k-', basefmt='k-', markerfmt='ro')
            h1[0].set_zorder(100)
            h2[0].set_zorder(100)
            ax.set(xticks=x[::3], xticklabels=map(str,rat_list)[::3], xlim=(-1, N))
            ax.tick_params(top=False, right=False)
            ax.tick_params(axis='x', direction='out', labelsize='small')
            if label == "running":
                ax.set_ylabel(r'$\Delta$[scan, non-scan]')
            quicktitle(ax, '%ss'%label.title())

        # Create pdf reports of individual rat distributions of ripple vs
        # no-ripple peak z-power theta for both scan and pause events
        class ThetaRatReport(BaseReport):
            label = 'rat report'
            def collect_data(self, which='scan'):
                if which == 'scan':
                    ripple_rat = scan_ripple_rat
                    noripple_rat = scan_noripple_rat
                elif which == 'pause':
                    ripple_rat = pause_ripple_rat
                    noripple_rat = pause_noripple_rat
                else:
                    raise ValueError, 'bad event type: %s'%which

                for rat, ax in self.get_plot(rat_list):
                    cdf = np.cumsum(distro(ripple_rat[rat]))
                    cdf_null = np.cumsum(distro(noripple_rat[rat]))
                    ax.plot(centers, cdf, 'b-', centers, cdf_null, 'r-')
                    ax.set(xlim=(centers[0],centers[-1]), ylim=(0,1))
                    ax.tick_params(top=False, right=False)
                    quicktitle(ax, 'rat%03d'%rat)
                    if self.firstonpage:
                        ax.set_ylabel('CDF')
                    else:
                        ax.set_yticklabels([])
                    if self.lastonpage:
                        ax.legend(['w/ ripples', 'no ripples'], loc=2,
                            bbox_to_anchor=(1.1,1))
                        ax.set_xlabel('Peak Z(P(Theta))')
                    else:
                        ax.set_xticklabels([])

        for which in 'scan', 'pause':
            reportdir = os.path.join(self.datadir, 'report')
            if not os.path.isdir(reportdir):
                os.makedirs(reportdir)
            report = ThetaRatReport(desc='%s events'%which, datadir=reportdir)
            report(which=which)
            report.open()
            os.rename(report.results['report'],
                os.path.join(self.datadir, 'rat_%s_cdfs.pdf'%which))
            os.system('rm -rf %s'%reportdir)

        data_file.close()

