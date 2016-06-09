# encoding: utf-8
"""
scan_stats.py -- Firing rates during scan vs. non-scan

Created by Joe Monaco on 2011-05-06.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division
from collections import namedtuple

# Library imports
import os
import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt
import tables as tb

# Package imports
from scanr.lib import *
from .core.analysis import AbstractAnalysis
from scanr.tools.stats import (t_one_sample, t_welch, t_paired, smooth_pdf, IQR,
    oneway, oneway_str, friedman, friedman_str)
from scanr.tools.plot import quicktitle, grouped_bar_plot, AxesList
from scanr.tools.misc import AutoVivification

# Constants
CfgData = Config['h5']
EXPTR_EXCLUDE = ['Eric']
# EXPTR_EXCLUDE = []


# Table descriptions
BehDescr =     {    'rat'       :   tb.UInt16Col(pos=1),
                    'day'       :   tb.UInt16Col(pos=2),
                    'session'   :   tb.UInt16Col(pos=3),
                    'type'      :   tb.StringCol(itemsize=4, pos=4),
                    'expt_type' :   tb.StringCol(itemsize=4, pos=5),
                    'running_t' :   tb.Float32Col(pos=6),
                    'session_t' :   tb.Float32Col(pos=6),
                    'scan_t'    :   tb.Float32Col(pos=6),
                    'scans'     :   tb.UInt16Col(pos=7) }

CueControl =    {   'scan_id'   :   tb.UInt16Col(pos=1),
                    'rat'       :   tb.UInt16Col(pos=2),
                    'mismatch'  :   tb.UInt8Col(pos=3),
                    'magnitude' :   tb.FloatCol(pos=4),
                    'duration'  :   tb.FloatCol(pos=5),
                    'direction' :   tb.StringCol(itemsize=4, pos=6),
                    'novel'     :   tb.BoolCol(pos=7) }


class ScanIntervalComparison(AbstractAnalysis):

    """
    Compare variance of temporal- and distance-based intervals between scans
    """

    label = 'scan intervals'

    def collect_data(self, potentiation_rats=True, scan_point='start'):
        """Collect temporal and spatial intervals across rats
        """
        session_table = get_node('/metadata', 'sessions')
        scan_table = get_node('/behavior', 'scans')

        if potentiation_rats:
            rats = unique_rats('/physiology', 'potentiation')
        else:
            rats = unique_rats(session_table)
        self.out('Found %d rats' % len(rats))

        @self.logcall
        def _collect_interval_data():
            I = AutoVivification()
            for rat in rats:
                self.out.printf('Rat %03d: ' % rat, color='lightgray')
                for rds in unique_sessions(session_table, condn='rat==%d' % rat):
                    self.out.printf('.')
                    data = SessionData.get(rds=rds, quiet=True)
                    ts = np.array([rec[scan_point] for rec in scan_table.where(data.session_query)])

                    T_scan = data.T_(ts)
                    Phi_scan = data.F_('alpha_unwrapped')(T_scan)

                    if 'T' in I[rat]:
                        I[rat]['T'] = np.r_[I[rat]['T'], np.diff(T_scan)]
                        I[rat]['Phi'] = np.r_[I[rat]['Phi'], -1 * np.diff(Phi_scan)]
                    else:
                        I[rat]['T'] = np.diff(T_scan)
                        I[rat]['Phi'] = -1 * np.diff(Phi_scan)

                self.out.printf('\n')
            return dict(I)
        intervals = _collect_interval_data()

        @self.logcall
        def _compute_interval_stats(I):
            D = AutoVivification()
            D['T'] = np.empty(len(rats), 'd')
            D['Phi'] = np.empty(len(rats), 'd')
            D['mean_ISI'] = np.empty(len(rats), 'd')
            D['mean_ISAngle'] = np.empty(len(rats), 'd')
            for i, rat in enumerate(rats):
                self.out.printf('.')
                D['T'][i] = I[rat]['T'].std() / I[rat]['T'].mean()
                D['Phi'][i] = I[rat]['Phi'].std() / I[rat]['Phi'].mean()
                D['mean_ISI'][i] = I[rat]['T'].mean()
                D['mean_ISAngle'][i] = I[rat]['Phi'].mean()
            self.out.printf('\n')
            return dict(D)
        stats = _compute_interval_stats(intervals)

        @self.logcall
        def _save_results():
            self.results['dispersion_stats'] = stats
            self.results['scan_point'] = scan_point
            self.results['rats'] = rats
        _save_results()

        self.out('Good-bye!')

    def process_data(self, cvlim=(0, 1.05)):
        self.start_logfile('interval_stats')

        rats = self.results['rats']
        scan_point = self.results['scan_point']
        D = self.results['dispersion_stats']

        @self.logcall
        def _perform_t_tests():
            def _test(label, x, y):
                self.out('%s time - angle: T(%d) = %f, P < %g' % ((label,2*len(rats)-2) + t_paired(x, y)))
            _test('CV', D['T'], D['Phi'])
        _perform_t_tests()

        def _key_str(k):
            if k in ('T', 'Phi'):
                return 'CV(%s)' % k
            return str(k)

        @self.logcall
        def _descriptive_stats():
            self.out('Values are mean +/- s.d. across %d rats' % len(rats))
            for dom in ('T', 'Phi', 'mean_ISI', 'mean_ISAngle'):
                v = D[dom]
                self.out('%s = %g +/- %g' % (_key_str(dom), v.mean(), v.std()))
        _descriptive_stats()

        plt.ioff()

        line_fmt = dict(c='k', ls='--', lw=1, alpha=0.5, zorder=-5)
        scatter_fmt = dict(s=18, c='k', linewidths=1, edgecolor='k', facecolor='none', zorder=0)

        @self.logcall
        def _plot_scatter_data():
            title = 'Time and Distance Intervals Between Scans [%s]\n%d rats' % (scan_point.title(), len(rats))
            f = self.new_figure('intervals', title, figsize=(8.5, 11))

            ax = f.add_subplot(321)
            ax.plot([0,1.2], [0,1.2], **line_fmt)
            ax.scatter(D['Phi'], D['T'], **scatter_fmt)
            ax.set(xlabel='CV(Track-angle distance)', ylabel='CV(Interval)')
            ax.set_xlim(cvlim)
            ax.set_ylim(cvlim)
            ax.tick_params(top=False, right=False, direction='out')
            quicktitle(ax, 'CV')

            ax = f.add_subplot(323)
            ax.boxplot(np.c_[D['T'], D['Phi']])
            ax.set_xticks([1,2])
            ax.set_xlim(0.5, 2.5)
            ax.set_ylim(bottom=0)
            ax.tick_params(top=False, right=False, direction='out')
            ax.set_xticklabels(['CV(T)', r'CV($\Phi$)'])
        _plot_scatter_data()

        plt.ion()
        plt.show()
        self.close_logfile()


class ScanMismatchControl(AbstractAnalysis):

    """
    Quantify scan properties as a function of experimental cue manipulations
    and measured across rats
    """

    label = 'scan cue control'

    def collect_data(self, potentiation_rats_only=False):
        """Create a data table of scans across different session types
        """
        scan_table = get_node('/behavior', 'scans')
        session_table = get_node('/metadata', 'sessions')

        data_file = self.open_data_file()
        cue_table = data_file.createTable('/', 'scans', CueControl,
            title='Cue-control of Scanning Properties')

        self.out('Collecting scan and mismatch data...')
        row = cue_table.row
        mismatch_checklist = {}

        # Restrict to potentiation-event rats if specified
        if potentiation_rats_only:
            rat_list = unique_rats(get_node('/physiology', 'potentiation'))
        else:
            rat_list = unique_rats(session_table)

        self.out('Gathering session scan data...')
        for rat in rat_list:
            for session in session_table.where('(type=="MIS")&(rat==%d)'%rat):
                mismatch = int(session['parameter'])
                if mismatch == 0:
                    continue
                if rat not in mismatch_checklist:
                    mismatch_checklist[rat] = []
                if mismatch in mismatch_checklist[rat]:
                    novel_mismatch = False
                else:
                    mismatch_checklist[rat].append(mismatch)
                    novel_mismatch = True

                self.out('rat%(rat)03d-%(day)02d-m%(session)d %(parameter)f'%session)
                session_query = '(rat==%(rat)d)&(day==%(day)d)&(session==%(session)d)'%session
                for scan in scan_table.where(session_query):
                    row['scan_id'] = scan['id']
                    row['rat'] = scan['rat']
                    row['duration'] = scan['duration']
                    row['magnitude'] = scan['magnitude']
                    row['direction'] = scan['type']
                    row['mismatch'] = int(session['parameter'])
                    row['novel'] = novel_mismatch

                    if scan['type'] == "AMB":
                        self.out.printf('.', color='lightred')
                    else:
                        self.out.printf('.')

                    row.append()
                self.out.printf('\n')
                cue_table.flush()
            self.out.printf('\n')

        self.results['results_in_data_file'] = True

        self.close_data_file()
        self.out('All done!')

    def process_data(self, scan_property='magnitude', zscore=True, zmax=0.2):
        """Compute per-rat statistics for different cue manipulations
        """
        self.out('Analyzing scan %s %s z-score:'%(scan_property, zscore and 'with' or 'without'))
        data_file = self.get_data_file()
        cue_table = data_file.root.scans

        rats = unique_rats(cue_table)
        mismatch_angles = (45, 90, 135, 180)
        N_angles = len(mismatch_angles)
        N_rats = len(rats)

        self.out('Found %d rats with mismatch sessions.'%len(rats))
        Normal = namedtuple("Normal", "mu sigma")

        for restrict_to_novel in (False, True):
            if restrict_to_novel:
                self.out('No mismatch repeats.')
                novel_query = '&(novel==True)'
            else:
                self.out('All mismatch presentations.')
                novel_query = ''

            data = [('ALL', np.empty((N_rats, N_angles), 'd')),
                    ('INT', np.empty((N_rats, N_angles), 'd')),
                    ('EXT', np.empty((N_rats, N_angles), 'd'))]

            for scan_direction, X in data:
                if scan_direction in ("INT", "EXT"):
                    self.out('Restricting analysis to %s scans:'%scan_direction)
                    dir_query = '&(direction=="%s")'%scan_direction
                else:
                    self.out('Using all non-ambiguous scans:')
                    dir_query = '&(direction!="AMB")'

                for i, rat in enumerate(rats):
                    self.out.printf('.')
                    if zscore:
                        rat_ix = cue_table.getWhereList('(rat==%d)%s%s'%(rat, dir_query, novel_query))
                        rat_x = cue_table[rat_ix][scan_property]
                        Z = Normal(rat_x.mean(), rat_x.std())

                    for j, mismatch in enumerate(mismatch_angles):
                        query = '(mismatch==%d)&(rat==%d)%s%s'%(mismatch, rat, dir_query, novel_query)
                        ix = cue_table.getWhereList(query)
                        mu = np.mean(cue_table[ix][scan_property])
                        if zscore:
                            X[i,j] = (mu - Z.mu) / Z.sigma
                        else:
                            X[i,j] = mu
                self.out.printf('\n')

            plt.ioff()
            if type(self.figure) is not dict:
                self.figure = {}
            suffix = zscore and "zscore" or "raw"
            suffix += restrict_to_novel and "_norepeats" or "_all"
            self.figure['mismatch_%s_%s'%(scan_property, suffix)] = f = plt.figure(figsize=(8.5,11))
            plt.clf()
            postfix = zscore and 'Z-Scored' or 'Raw Data'
            postfix += restrict_to_novel and ', No Mismatch Repeats' or ', All Presentations'
            prop_str = scan_property.title()
            f.suptitle('Mismatch Cue Control: Scan %s\n%s'%(prop_str, postfix))

            real_unit = dict(magnitude="cm", duration="s")[scan_property]
            real_ymax = dict(magnitude=12.0, duration=3.5)[scan_property]
            x = np.arange(N_angles)

            for i, data_item in enumerate(data):
                label, X = data_item

                self.out('%s: %s'%(label, friedman_str(X)))

                ax1 = f.add_subplot(321 + 2*i)
                ax1.errorbar(x, X.mean(axis=0), yerr=X.std(axis=0)/np.sqrt(N_rats),
                    fmt='k-', capsize=5, lw=1.5)
                ax1.set_xlim(-0.5, N_angles - 0.5)
                if zscore:
                    ax1.set_ylim(-zmax, zmax)
                    unit = 'Z'
                else:
                    ax1.set_ylim(0.0, real_ymax)
                    unit = real_unit
                ax1.set_xticks(x)
                ax1.set_xticklabels(map(str, mismatch_angles))
                ax1.set_ylabel('Scan %s (%s) [%s]'%(prop_str, unit, label))

                ax2 = f.add_subplot(322 + 2*i)
                ax2.boxplot(X)
                if zscore:
                    ax2.set_ylim(-3.5*zmax, 3.5*zmax)
                else:
                    ax2.set_ylim(bottom=0.0)
                ax2.set_xticks(x + 1)
                ax2.set_xticklabels(map(str, mismatch_angles))

                if i == 2:
                    ax1.set_xlabel('Mismatch Angle (degrees)')
                    ax2.set_xlabel('Mismatch Angle (degrees)')

            self.out('---')

        plt.ion()
        plt.show()
        self.close_data_file()

    def run_script(self):
        """Run analysis for all scan and normalization types
        """
        self.out.outfd = file(os.path.join(self.datadir, 'script.log'), 'w')
        self.out.timestamp = False
        for zscore in (True, False):
            for scan_property in ('magnitude', 'duration'):
                self.process_data(scan_property=scan_property, zscore=zscore)
        self.save_plots_and_close()
        self.out.outfd.close()
        self.out("All done!")


class ScanFrequencyStats(AbstractAnalysis):

    """
    Normalized scanning frequencies/durations across rats for session types
    """

    label = 'scan stats'

    def collect_data(self, potentiation_rats_only=False, min_scan_duration=0.0):
        """Collect data on head scanning across session number and type

        Optionally restrict the rats to those which have place-field
        potentiation events stored in /physiology/potentiation.
        """
        sessions = get_node('/metadata', 'sessions')
        scans = get_node('/behavior', 'scans')

        # Tables and iterators
        data_file = self.open_data_file()
        scan_data_table = data_file.createTable('/', 'scan_data',
            BehDescr, title='Scan Data')
        row = scan_data_table.row

        # Restrict to potentiation-event rats if specified
        if potentiation_rats_only:
            rat_list = unique_rats('/physiology', 'potentiation')
        else:
            rat_list = unique_rats(sessions)

        self.out('Gathering session scan data...')
        for rat in rat_list:
            for session in sessions.where('rat==%d'%rat):
                if session['session'] == 1:
                    self.out.printf('|', color='cyan')
                    scan_data_table.flush()

                self.out.printf('.')
                for k in ('rat', 'day', 'session', 'type', 'expt_type'):
                    row[k] = session[k]
                traj = TrajectoryData(rds=(
                    session['rat'], session['day'], session['session']))
                running_frac = (traj.forward_velocity >= 10).sum() / traj.N
                row['running_t'] = running_frac * elapsed(session['start'],
                    session['end'])
                row['session_t'] = elapsed(session['start'], session['end'])
                row['scan_t'] = np.array([rec['duration'] for rec in
                    scans.where(
                        '(rat==%(rat)d)&(day==%(day)d)&(session==%(session)d)'%
                        session) if rec['duration'] >= min_scan_duration]).sum()
                row['scans'] = len(scans.getWhereList(
                    '(duration>=%f)&'%min_scan_duration +
                    '(rat==%(rat)d)&(day==%(day)d)&(session==%(session)d)'%session))
                row.append()
        self.out.printf('\n')

        self.results['there_are_no_results'] = True
        self.out('All done!')

    def compare_experiments(self, do_freq=True):
        """Get frequency/duration stats for the two experiment types and do an
        unpaired t-test
        """
        scans = do_freq and 'scans' or 'scan_t'
        duration = 'session_t'
        which = do_freq and 'frequency' or 'duration'

        self.out.outfd = file(os.path.join(self.datadir, '%s.log'%which), 'w')
        self.out.timestamp = False

        data_file = self.get_data_file()
        scan_data = data_file.root.scan_data

        # Double rotation rat-average data
        expt_type_list = np.unique(scan_data.col("expt_type"))
        rats = {}
        F = {}
        for expt in expt_type_list:
            rats[expt] = np.unique([rec['rat'] for rec in scan_data.where('(expt_type=="%s")'%expt)])
            F[expt] = np.empty(len(rats[expt]), 'd')

            for i, rat in enumerate(rats[expt]):
                T = sum([rec[duration] for rec in scan_data.where('(rat==%d)&(expt_type=="%s")'%(rat,expt))])
                N = sum([rec[scans] for rec in scan_data.where('(rat==%d)&(expt_type=="%s")'%(rat,expt))])
                F[expt][i] = N / T

            self.out('%s experiment: N rats = %d'%(expt, len(rats[expt])))
            self.out('%s: %f +/- %f s.d.'%(expt, F[expt].mean(), F[expt].std()))

        self.out('---')
        self.out('NOV - DR: T(%(df)d): T = %(t)f, p < %(p)f'%t_welch(F['NOV'], F['DR'], return_tpdf=True))

        self.out.outfd.close()
        self.close_data_file()

    def process_data(self, first_day=False, do_freq=False, norm_running=False,
        ylim=(0.0, 1.5), norm=True, paired_tests=False):
        figname = (do_freq and 'frequency' or 'duration') + '_stats' + (first_day and '_day1' or '_alldays')
        self.out.outfd = file(os.path.join(self.datadir, '%s.log'%figname), 'w')
        self.out.timestamp = False

        data_file = self.get_data_file()
        scan_data = data_file.root.scan_data

        if do_freq:
            scans = 'scans'
        else:
            scans = 'scan_t'

        if norm_running:
            duration = 'running_t'
        else:
            duration = 'session_t'

        if first_day:
            daystr = '(day==1)&'
        else:
            daystr = '(day!=0)&'

        # Overall scan frequencies
        rat_list = np.unique([rec['rat'] for rec in scan_data.where(daystr[:-1])])
        F = dict()
        F_exm1 = dict()
        scan_dur = []
        for rat in rat_list:
            # Base line scan frequency or duration
            T = sum([rec[duration] for rec in scan_data.where(daystr + '(rat==%d)'%rat)])
            N = sum([rec[scans] for rec in scan_data.where(daystr + '(rat==%d)'%rat)])
            F[rat] = N / T

            # Base line scan frequency or duration, excluding m1 sessions
            T = sum([rec[duration] for rec in scan_data.where(daystr + '(session!=1)&(rat==%d)'%rat)])
            N = sum([rec[scans] for rec in scan_data.where(daystr + '(session!=1)&(rat==%d)'%rat)])
            F_exm1[rat] = N / T

            # For computing average scan duration across rats
            scan_total_time = sum([rec['scan_t'] for rec in scan_data.where(daystr + '(rat==%d)'%rat)])
            scan_num = sum([rec['scans'] for rec in scan_data.where(daystr + '(rat==%d)'%rat)])
            scan_dur.append(scan_total_time / scan_num)

        self.out('N = %d rats'%len(rat_list))
        self.out('---')
        overall = np.array(F.values())
        self.out('Overall range: %.3f -> %.3f'%(overall.min(), overall.max()))
        self.out('Mean, SD, median: %.3f +/- %.4f, %.3f'%(overall.mean(), overall.std(), np.median(overall)))
        self.out('---')
        overall_exm1 = np.array(F_exm1.values())
        self.out('Overall [excl. m1] range: %.3f -> %.3f'%(overall_exm1.min(), overall_exm1.max()))
        self.out('Mean, SD, median: %.3f +/- %.4f, %.3f'%(overall_exm1.mean(), overall_exm1.std(), np.median(overall_exm1)))
        self.out('---')
        scan_dur = np.array(scan_dur)
        self.out('Average scan duration range: %.3f -> %.3f sec'%(scan_dur.min(), scan_dur.max()))
        self.out('Mean, SD, median: %.3f +/- %.4f, %.3f sec'%(scan_dur.mean(), scan_dur.std(), np.median(scan_dur)))

        def freq_cmp(N, T, f):
            if norm:
                if f and T:
                    return (N / float(T)) / f # np.median(overall) # / f
            elif T:
                u = N / float(T)
                return u
            return 0.0

        # Normalized, DR, across maze type
        rat_list = np.unique([rec['rat'] for rec in scan_data.where(daystr + '(expt_type=="DR")')])
        F_DR_type = np.empty((len(rat_list), 2), 'd')
        for j, mtype in enumerate(('STD', 'MIS')):
            for i, rat in enumerate(rat_list):
                T = sum([rec[duration] for rec in scan_data.where(daystr + '(session!=1)&(rat==%d)&(type=="%s")'%(rat, mtype))])
                N = sum([rec[scans] for rec in scan_data.where(daystr + '(session!=1)&(rat==%d)&(type=="%s")'%(rat, mtype))])
                F_DR_type[i,j] = freq_cmp(N, T, F_exm1[rat])

        # Normalized, DR, across maze number
        rat_list = np.unique([rec['rat'] for rec in scan_data.where(daystr + '(expt_type=="DR")')])
        F_DR_num = np.empty((len(rat_list), 5), 'd')
        for j, number in enumerate(range(1, 6)):
            for i, rat in enumerate(rat_list):
                T = sum([rec[duration] for rec in scan_data.where(daystr + '(rat==%d)&(expt_type=="DR")&(session==%d)'%(rat, number))])
                N = sum([rec[scans] for rec in scan_data.where(daystr + '(rat==%d)&(expt_type=="DR")&(session==%d)'%(rat, number))])
                F_DR_num[i,j] = freq_cmp(N, T, F[rat])

        # Normalized, NOV, across maze type
        rat_list = np.unique([rec['rat'] for rec in scan_data.where(daystr + '(expt_type=="NOV")')])
        F_nov_type = np.empty((len(rat_list), 2), 'd')
        for j, mtype in enumerate(('FAM', 'NOV')):
            for i, rat in enumerate(rat_list):
                T = sum([rec[duration] for rec in scan_data.where(daystr + '(session!=1)&(rat==%d)&(type=="%s")'%(rat, mtype))])
                N = sum([rec[scans] for rec in scan_data.where(daystr + '(session!=1)&(rat==%d)&(type=="%s")'%(rat, mtype))])
                F_nov_type[i,j] = freq_cmp(N, T, F_exm1[rat])

        # Normalized, NOV, across maze number
        rat_list = np.unique([rec['rat'] for rec in scan_data.where(daystr + '(expt_type=="NOV")')])
        F_nov_num = np.empty((len(rat_list), 3), 'd')
        for j, number in enumerate(range(1, 4)):
            for i, rat in enumerate(rat_list):
                T = sum([rec[duration] for rec in scan_data.where(daystr + '(rat==%d)&(expt_type=="NOV")&(session==%d)'%(rat, number))])
                N = sum([rec[scans] for rec in scan_data.where(daystr + '(rat==%d)&(expt_type=="NOV")&(session==%d)'%(rat, number))])
                F_nov_num[i,j] = freq_cmp(N, T, F[rat])

        self.close_data_file()

        # Bar charts, maze type
        if type(self.figure) != dict:
            self.figure = {}
        self.figure[figname] = f = plt.figure(figsize=(7,7))
        label = do_freq and 'Frequency' or 'Duration'
        if norm_running:
            prefix = 'Running-'
        else:
            prefix = ''
        f.suptitle('%sNormalized Scanning %s Across Sessions, For Rats\n%s'%(
            prefix, label, (first_day and 'Only Day 1' or 'All Days')), size='small')

        fmt = dict(width=0.4, linewidth=1, color='w', edgecolor='k', ecolor='k', capsize=3)
        sem = lambda x: x.std(axis=0) / np.sqrt(x.shape[0])

        ax = f.add_subplot(221)
        ax.bar(
            [0, 0.4, 1, 1.4],
            np.r_[F_DR_type.mean(axis=0), F_nov_type.mean(axis=0)],
            yerr=np.r_[sem(F_DR_type), sem(F_nov_type)], **fmt)
        ax.set_xlim(-0.5, 3.9)
        ax.tick_params(right=False)
        ax.set_ylim(ylim)
        ax.axhline(1.0, c='k', ls='--', lw=1)
        ax.set(xticks=[], xticklabels=[])

        ax = f.add_subplot(222)
        ax.boxplot([F_DR_type[:,0], F_DR_type[:,1], F_nov_type[:,0], F_nov_type[:,1]])

        ax = f.add_subplot(223)
        results = ( np.r_[F_DR_num.mean(axis=0), F_nov_num.mean(axis=0)],
                    np.r_[sem(F_DR_num), sem(F_nov_num)])
        ax.bar(
            [0, 0.4, 0.8, 1.2, 1.6, 2.4, 2.8, 3.2],
            results[0], yerr=results[1], **fmt)
        ax.set_xlim(-0.5, 4.1)
        ax.tick_params(right=False)
        ax.set_ylim(ylim)
        ax.axhline(1.0, c='k', ls='--', lw=1)
        ax.set(xticks=[], xticklabels=[])

        ax = f.add_subplot(224)
        ax.boxplot([F_DR_num[:,0], F_DR_num[:,1], F_DR_num[:,2], F_DR_num[:,3], F_DR_num[:,4],
            F_nov_num[:,0], F_nov_num[:,1], F_nov_num[:,2]])

        mu = int(norm)
        t_two_sample = paired_tests and t_paired or t_welch

        self.out('---')
        self.out('One-sample comparisons to mu = %d'%mu)
        self.out('---')
        self.out('T-tests are %s'%(paired_tests and 'paired' or 'unpaired'))
        self.out('---')
        self.out('DR STD <> mu: T=%.3f, p<%f'%t_one_sample(F_DR_type[:,0], mu))
        self.out('DR MIS <> mu: T=%.3f, p<%f'%t_one_sample(F_DR_type[:,1], mu))
        self.out('DR MIS <> STD: T=%.3f, p<%f'%t_two_sample(F_DR_type[:,1], F_DR_type[:,0]))
        self.out('STD: %.6f +/- %.6f; MIS: %.6f +/- %.6f'%(F_DR_type[:,0].mean(), sem(F_DR_type[:,0]),
            F_DR_type[:,1].mean(), sem(F_DR_type[:,1])))

        self.out('---')
        self.out('NOV FAM <> mu: T=%.3f, p<%f'%t_one_sample(F_nov_type[:,0], mu))
        self.out('NOV NOV <> mu: T=%.3f, p<%f'%t_one_sample(F_nov_type[:,1], mu))
        self.out('NOV NOV <> FAM: T=%.3f, p<%f'%t_two_sample(F_nov_type[:,1], F_nov_type[:,0]))
        self.out('FAM: %.6f +/- %.6f; NOV: %.6f +/- %.6f'%(F_nov_type[:,0].mean(), sem(F_nov_type[:,0]),
            F_nov_type[:,1].mean(), sem(F_nov_type[:,1])))

        self.out('---')
        self.out('DR M1 <> mu: T=%.3f, p<%f'%t_one_sample(F_DR_num[:,0], mu))
        self.out('DR M2 <> mu: T=%.3f, p<%f'%t_one_sample(F_DR_num[:,1], mu))
        self.out('DR M3 <> mu: T=%.3f, p<%f'%t_one_sample(F_DR_num[:,2], mu))
        self.out('DR M4 <> mu: T=%.3f, p<%f'%t_one_sample(F_DR_num[:,3], mu))
        self.out('DR M5 <> mu: T=%.3f, p<%f'%t_one_sample(F_DR_num[:,4], mu))
        self.out('DR M1 <> M2: T=%.3f, p<%f'%t_two_sample(F_DR_num[:,0], F_DR_num[:,1]))
        self.out('DR M1 <> M3: T=%.3f, p<%f'%t_two_sample(F_DR_num[:,0], F_DR_num[:,2]))
        self.out('DR M2 <> M3: T=%.3f, p<%f'%t_two_sample(F_DR_num[:,1], F_DR_num[:,2]))
        self.out('DR M2 <> M4: T=%.3f, p<%f'%t_two_sample(F_DR_num[:,1], F_DR_num[:,3]))
        self.out('DR M1 <> M5: T=%.3f, p<%f'%t_two_sample(F_DR_num[:,0], F_DR_num[:,4]))
        self.out('---')
        self.out('DR 1-5 one-way anova: F(%d,%d)=%.3f, p<%f'%oneway(F_DR_num))
        self.out('DR type one-way anova: F(%d,%d)=%.3f, p<%f'%oneway(F_DR_type))

        self.out('---')
        self.out('NOV M1 <> mu: T=%.3f, p<%f'%t_one_sample(F_nov_num[:,0], mu))
        self.out('NOV M2 <> mu: T=%.3f, p<%f'%t_one_sample(F_nov_num[:,1], mu))
        self.out('NOV M3 <> mu: T=%.3f, p<%f'%t_one_sample(F_nov_num[:,2], mu))
        self.out('NOV M1 <> M2: T=%.3f, p<%f'%t_two_sample(F_nov_num[:,0], F_nov_num[:,1]))
        self.out('NOV M1 <> M3: T=%.3f, p<%f'%t_two_sample(F_nov_num[:,0], F_nov_num[:,2]))
        self.out('NOV M2 <> M3: T=%.3f, p<%f'%t_two_sample(F_nov_num[:,1], F_nov_num[:,2]))
        self.out('---')
        nov, fam = F_nov_num[:,1].mean(), F_nov_num[:,2].mean()
        self.out('NOV <M2> = %f, <M3> = %f, <M2>/<M3> = %f'%(nov, fam, nov/fam))
        self.out('---')
        self.out('NOV 1-3 one-way anova: F(%d,%d)=%.3f, p<%f'%oneway(F_nov_num))
        self.out('NOV type one-way anova: F(%d,%d)=%.3f, p<%f'%oneway(F_nov_type))

        self.out.outfd.close()

        return results

    def processing_script(self, **kwds):
        """Run duration x frequency and all-days x day-one
        """
        for do_freq in (True, False):
            ymax = do_freq and 1.5 or 1.8
            for first_day in (True, False):
                process_kwds = dict(do_freq=do_freq, first_day=first_day, ylim=(0.0, ymax))
                process_kwds.update(**kwds)
                self.process_data(**process_kwds)
        # self.save_plots_and_close()

    def grouped_bar_plot(self, **kwds):
        """Merge frequency and scanning duration into a grouped bar plot
        """
        freq, freq_err = self.process_data(do_freq=True, first_day=False, **kwds)
        dur, dur_err = self.process_data(do_freq=False, first_day=False, **kwds)
        freq_day1, freq_day1_err = self.process_data(do_freq=True, first_day=True, **kwds)
        dur_day1, dur_day1_err = self.process_data(do_freq=False, first_day=True, **kwds)

        plt.ioff()
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure['grouped_bar_plot'] = f = plt.figure(figsize=(7,8))
        f.suptitle('Normalized Scan Duration and Frequency Across Sessions')

        ax1 = f.add_subplot(321)
        ax2 = f.add_subplot(322)
        ax3 = f.add_subplot(323)
        ax4 = f.add_subplot(324)
        axlist = AxesList()
        axlist.add_figure(f)

        grouped_bar_plot(
            np.c_[freq[:5], dur[:5]],
            map(str, range(1,6)),
            [('frequency', 'c'), ('duration', 'g')],
            errors=np.c_[freq_err[:5], dur_err[:5]],
            capsize=0, ecolor='k', linewidth=0, alpha=0.75, # bar keywords
            legend_loc='lower right',
            ax=ax1)

        grouped_bar_plot(
            np.c_[freq_day1[:5], dur_day1[:5]],
            map(str, range(1,6)),
            [('frequency', 'c'), ('duration', 'g')],
            errors=np.c_[freq_day1_err[:5], dur_day1_err[:5]],
            capsize=0, ecolor='k', linewidth=0, alpha=0.75, # bar keywords
            legend=False,
            ax=ax2)

        grouped_bar_plot(
            np.c_[freq[5:], dur[5:]],
            map(str, range(1,4)),
            [('frequency', 'c'), ('duration', 'g')],
            errors=np.c_[freq_err[5:], dur_err[5:]],
            capsize=0, ecolor='k', linewidth=0, alpha=0.75, # bar keywords
            legend=False,
            ax=ax3)

        grouped_bar_plot(
            np.c_[freq_day1[5:], dur_day1[5:]],
            map(str, range(1,4)),
            [('frequency', 'c'), ('duration', 'g')],
            errors=np.c_[freq_day1_err[5:], dur_day1_err[5:]],
            capsize=0, ecolor='k', linewidth=0, alpha=0.75, # bar keywords
            legend=False,
            ax=ax4)

        for ax in axlist:
            ax.set_xlim(-1, 5)
            ax.tick_params(top=False, right=False, direction='out', labelsize='x-small')
            ax.axhline(1.0, c='0.5', ls='-', lw=0.5, zorder=100)

        # ax1.set_ylim(0.0, 1.4)
        # ax2.set_ylim(0.0, 1.6)
        # ax3.set_ylim(0.0, 1.4)
        # ax4.set_ylim(0.0, 1.6)

        quicktitle(ax1, 'DR all days')
        quicktitle(ax2, 'DR day 1')
        quicktitle(ax3, 'NOV all days')
        quicktitle(ax4, 'NOV day 1')

        plt.ion()
        plt.show()

        return axlist

    def rat_histograms(self, dur_max=6.0, rate_max=0.25, interval_max=15.0,
        nbins=15, numdays=12):
        """Create histograms across rats of basic scan properties such as
        average duration and scanning frequency.
        """
        self.out.outfd = file(os.path.join(self.datadir, 'histograms.log'), 'w')
        self.out.timestamp = False

        data_file = self.get_data_file()
        scan_data = data_file.root.scan_data

        rat_list = unique_rats(scan_data)
        N_rats = len(rat_list)
        scan_rates = np.empty(N_rats, 'd')
        interscan_intervals = np.empty(N_rats, 'd')
        scan_durations = np.empty(N_rats, 'd')
        scan_rates_days = np.empty((N_rats, numdays), 'd')
        intervals_days = np.empty((N_rats, numdays), 'd')
        day_flag = np.zeros_like(scan_rates_days, '?')
        self.out('Found data for %d rats.'%N_rats)

        for i, rat in enumerate(rat_list):
            self.out.printf('[%d]'%rat)

            T = sum([rec['session_t'] for rec in scan_data.where('(rat==%d)'%rat)])
            N = sum([rec['scans'] for rec in scan_data.where('(rat==%d)'%rat)])
            scan_rates[i] = N / T
            interscan_intervals[i] = T / N

            scanning_time = sum([rec['scan_t'] for rec in scan_data.where('(rat==%d)'%rat)])
            scan_durations[i] = scanning_time / N

            for j, day in enumerate(range(1, numdays+1)):
                T_day = sum([rec['session_t'] for rec in scan_data.where('(rat==%d)&(day==%d)'%(rat,day))])
                N_day = sum([rec['scans'] for rec in scan_data.where('(rat==%d)&(day==%d)'%(rat,day))])
                if T_day:
                    scan_rates_days[i,j] = N_day / T_day
                    intervals_days[i,j] = T_day / N_day
                else:
                    day_flag[i,j] = True

        self.out.printf('\n')

        plt.ioff()
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure['histograms'] = f = plt.figure(figsize=(7, 8))
        f.suptitle('Scan Duration and Frequency Histograms Across Rats and Days')

        dur_bins = np.linspace(0, dur_max, nbins + 1)
        rate_bins = np.linspace(0, rate_max, nbins + 1)
        interval_bins = np.linspace(0, interval_max, nbins + 1)
        hist_fmt = dict(color='0.4', ec='none', rwidth=0.8, normed=True)

        # Average Scan Durations across rats
        ax = f.add_subplot(321)
        ax.hist(scan_durations, bins=dur_bins, **hist_fmt)
        ax.plot(*smooth_pdf(scan_durations), c='r', lw=1, zorder=1)
        ax.axvline(np.median(scan_durations), c='r', ls='--', zorder=2)
        self.out('Median average-duration across rats: %f s'%np.median(scan_durations))
        ax.tick_params(right=False, top=False, direction='out', labelsize='x-small')
        ax.set_xlim(0, dur_max)
        ax.set_ylabel('Pr[Duration]', size='x-small')
        ax.set_xlabel('Duration (s)', size='x-small')
        quicktitle(ax, 'Avg. Scan Duration', size='x-small')

        # Scan Event Rate averages across rats
        ax = f.add_subplot(323)
        ax.hist(scan_rates, bins=rate_bins, **hist_fmt)
        ax.plot(*smooth_pdf(scan_rates), c='r', lw=1, zorder=1)
        ax.axvline(np.median(scan_rates), c='r', ls='--', zorder=2)
        self.out('Median rate across rats: %f scans/s'%np.median(scan_rates))
        ax.tick_params(right=False, top=False, direction='out', labelsize='x-small')
        ax.set_ylabel('Pr[Rate]', size='x-small')
        ax.set_xlabel('Rate (scans/s)', size='x-small')
        ax.set_xlim(0, rate_max)
        quicktitle(ax, 'Overall Scan Event-Rate', size='x-small')

        # Interscan Interval averages across rats
        ax = f.add_subplot(324)
        ax.hist(interscan_intervals, bins=interval_bins, **hist_fmt)
        ax.plot(*smooth_pdf(interscan_intervals), c='r', lw=1, zorder=1)
        ax.axvline(np.median(interscan_intervals), c='r', ls='--', zorder=2)
        self.out('Median interval across rats: %f s/scan'%np.median(interscan_intervals))
        ax.tick_params(right=False, top=False, direction='out', labelsize='x-small')
        ax.set_ylabel('Pr[ISI]', size='x-small')
        ax.set_xlabel('Interval (s/scan)', size='x-small')
        ax.set_xlim(0, interval_max)
        quicktitle(ax, 'Inter-scan Interval', size='x-small')

        # Scan Event Rates and Inter-scan Interval across testing days
        scan_rates_plot_data = ('Scan Rate', 'scans/s', 325, scan_rates_days)
        intervals_plot_data = ('Interscan Intervals', 's/scan', 326, intervals_days)
        N_rats_days = N_rats - day_flag.sum(axis=0)
        width = 0.8
        self.out('# Rats across days 1 - %d: %s'%(numdays, N_rats_days))
        for label, units, subp, data_x_days in [scan_rates_plot_data, intervals_plot_data]:
            R = np.ma.masked_where(day_flag, data_x_days)

            IQR_days = np.empty((numdays, 2), 'd')
            median_days = np.empty(numdays, 'd')
            for i in xrange(numdays):
                valid_data = data_x_days[day_flag[:,i] == False, i]
                IQR_days[i] = IQR(valid_data, factor=0.0)
                median_days[i] = np.median(valid_data)

            ax = f.add_subplot(subp)
            x_days = np.arange(numdays) + width/2
            R_mean = R.mean(axis=0)
            ax.bar(x_days - width/2,
                R_mean,
                yerr=R.std(axis=0) / np.sqrt(N_rats_days),
                width=width, ec='none', color='0.4', capsize=0, ecolor='k')
            ax.plot(x_days, IQR_days, 'r:', lw=1, zorder=1)
            ax.plot(x_days, median_days, 'r-', lw=1, zorder=1)
            ax.set_xticks(x_days)
            ax.set_xticklabels(np.arange(1, numdays + 1), size='x-small')
            ax.set_xlim(-1, max(nbins, x_days[-1] + 1))
            ax.tick_params(right=False, top=False, direction='out', labelsize='x-small')
            ax.set_xlabel('Test Day', size='x-small')
            ax.set_ylabel('%s (%s)' % (label, units), size='x-small')
            ax.set_ylim(top=max(0.2, ax.axis()[-1]))

            slope, intercept, r_value, p_value, std_err = st.linregress(x_days, R_mean)
            self.out('%s: LSR slope=%f, r=%f, p<%f'%(label, slope, r_value, p_value))
            self.out('%s: R-squared=%f'%(label, r_value**2))

        plt.ion()
        plt.show()
        self.out.outfd.close()


class HeadScanSessionStats(AbstractAnalysis):

    """
    Analysis of per-session statistics of head scanning events
    """

    label = "session stats"

    def collect_data(self, expt_type='DR', behavior_table='scans'):
        """Collate and summarize statistics of head scanning episodes

        Arguments:
        expt_type -- experiment type for which to collate data (defaults to
            'DR', otherwise specify 'NOV' or 'ALL')
        """
        # Set up spreadsheet output
        col_init = dict(s='', d=0, f=0.0)
        cols = [('dataset', 's'), ('maze', 'd'), ('type', 's'), ('parameter', 'd'),
                ('duration', 'f'), ('total_scan_duration', 'f'), ('scan_proportion', 'f'),
                ('total_scans', 'd'), ('avg_scan_duration', 'f'),
                ('away_duration', 'f'), ('return_duration', 'f'), ('away_return_ratio', 'f')]
        cols += [('lap%d'%l, 'd') for l in range(1, 21)]

        def new_record():
            return { col: col_init[dtype] for col, dtype in cols }

        record_string = ','.join(['%%(%s)%s'%col for col in cols])+'\n'
        self.out('Record string: ' + record_string)

        spreadsheet = file(os.path.join(self.datadir, 'scan_stats.csv'), 'w')
        spreadsheet.write(','.join([col for col, dtype in cols])+'\n')

        # Get tables
        session_table = get_node('/metadata', 'sessions')
        scan_table = get_node('/behavior', behavior_table)

        # Scan through all sessions
        session_query = "(timing_issue==False)"
        if expt_type == 'ALL':
            session_query += "&(expt_type=='DR')|(expt_type=='NOV')"
        else:
            session_query += "&(expt_type=='%s')"%expt_type
        for session in session_table.where(session_query):
            rat, day, maze = session['rat'], session['day'], session['session']
            rds = rat, day, maze
            dataset = 'rat%d-%02d'%rds[:2]

            self.out('Generating scan stats for %s maze %d'%(dataset, rds[2]))

            record = new_record()
            record['dataset'] = dataset
            record['maze'] = maze
            record['type'] = session['type']
            record['parameter'] = session['parameter']
            record['duration'] = time.elapsed(session['start'], session['end'])

            # Tracking, behavior, and laps data
            traj = tracking.TrajectoryData(rds=rds)
            ts, x, y = traj.tsxyhd()[:3]
            fs = traj.fs
            M = Moment.get(traj)
            laps = tracking.find_laps(ts, x, y, start=session['start'])
            laps.append(session['end'])
            laps = np.asarray(laps)

            # Temporary variables
            away_duration = 0.0
            return_duration = 0.0
            scan_duration = 0.0
            away_return_ratio = 0.0

            # Scan through all head scans for this session
            scan_query = "(rat==%d)&(day==%d)&(session==%d)"%rds
            for scan in scan_table.where(scan_query):
                record['total_scans'] += 1
                away = time.elapsed(scan['start'], scan['max'])
                retrn = time.elapsed(scan['max'], scan['end'])
                away_duration += away
                return_duration += retrn
                if retrn:
                    away_return_ratio += away / retrn
                scan_duration += scan['duration']

                lap_ix = np.max((scan['start']>=laps).nonzero()[0])
                if lap_ix < 20:
                    record['lap%d'%(lap_ix+1)] += 1

            record['away_duration'] = away_duration
            record['return_duration'] = return_duration
            record['total_scan_duration'] = scan_duration
            record['scan_proportion'] = scan_duration / record['duration']
            if record['total_scans']:
                record['away_return_ratio'] = away_return_ratio / record['total_scans']
                record['avg_scan_duration'] = scan_duration / record['total_scans']

            spreadsheet.write(record_string%record)

        spreadsheet.close()
        self.out('All done!')


class HeadScanStats(AbstractAnalysis):

    """
    Analysis of occurrences of head scanning episodes
    """

    label = 'head scan stats'

    def collect_data(self, behavior_table='scans'):
        """Collate and summarize statistics of head scanning episodes
        """
        rat_data = {    'number_per_session': [],
                        'interscan_interval': [],
                        'average_duration': [],
                        'scan_fraction': []    }

        day_info = {    'day_number': []    }
        day_data = {    'N_scans':  [],
                        'N_scans_norm': [],
                        'duration': [],
                        'duration_norm': []    }

        maze_info = {   'maze_number': [],
                        'maze_type': [] }
        maze_data = {   'N_scans':  [],
                        'N_scans_norm': [],
                        'duration': [],
                        'duration_norm': []    }

        scan_info = {   'start_time': [],
                        'scan_number': []    }
        scan_data = {   'scan_duration': [],
                        'scan_duration_norm':   []  }

        stable = get_node('/behavior', behavior_table)

        for rat in get_rat_list():
            rat_group = get_group(rat=rat)
            if rat_group._v_attrs.exptr in EXPTR_EXCLUDE:
                continue

            r_ix = stable.getWhereList('rat==%d'%rat)
            total_scan_duration = np.sum([stable[i]['duration'] for i in r_ix])
            N_total_scans = len(r_ix)
            average_scan_duration = total_scan_duration / N_total_scans

            N_sessions = 0
            total_duration = 0.0

            day_list = get_day_list(rat)
            N_days = len(day_list)
            d0 = day_list[0]
            for day in day_list:
                day_group = get_group(rat=rat, day=day)
                rd = rat, day

                d_ix = stable.getWhereList('(rat==%d)&(day==%d)'%rd)

                day_info['day_number'].append(day - d0 + 1)
                day_data['N_scans'].append(len(d_ix))
                day_data['duration'].append(
                    np.mean([stable[i]['duration'] for i in d_ix]))

                day_data['N_scans_norm'].append(
                    len(d_ix) / (float(N_total_scans) / N_days)  )
                day_data['duration_norm'].append(
                    day_data['duration'][-1] / average_scan_duration  )

                maze_list = get_maze_list(rat, day)
                N_mazes = len(maze_list)
                for session in maze_list:
                    maze_group = get_group(rat=rat, day=day, session=session)
                    attrs = maze_group._v_attrs

                    rds = rat, day, session

                    N_sessions += 1
                    total_duration += get_duration(*rds)

                    m_ix = stable.getWhereList('(rat==%d)&(day==%d)&(session==%d)'%rds)

                    maze_info['maze_number'].append(session)
                    maze_info['maze_type'].append(attrs.type == 'MIS')
                    maze_data['N_scans'].append(len(m_ix))
                    maze_data['N_scans_norm'].append(
                        len(m_ix) / (float(len(d_ix)) / N_mazes)    )

                    if len(m_ix):
                        maze_data['duration'].append(
                            np.mean([stable[i]['duration'] for i in m_ix])   )
                        maze_data['duration_norm'].append(
                            maze_data['duration'][-1] / day_data['duration'][-1]    )
                    else:
                        maze_data['duration'].append(0.0)
                        maze_data['duration_norm'].append(0.0)


                    for scan_ix in m_ix:
                        scan = stable[scan_ix]

                        start = time.elapsed(attrs.start, scan['start'])
                        scan_info['start_time'].append(start)
                        scan_info['scan_number'].append(scan['number'])
                        scan_data['scan_duration'].append(scan['duration'])
                        scan_data['scan_duration_norm'].append(
                            scan['duration'] / maze_data['duration'][-1]    )

            rat_data['number_per_session'].append( N_total_scans / float(N_sessions) )
            rat_data['interscan_interval'].append( total_duration / N_total_scans )
            rat_data['average_duration'].append( average_scan_duration )
            rat_data['scan_fraction'].append( total_scan_duration / total_duration )

        # Save results data
        self.results['rat_data'] = rat_data
        self.results['day_data'] = day_data
        self.results['day_info'] = day_info
        self.results['maze_data'] = maze_data
        self.results['maze_info'] = maze_info
        self.results['scan_data'] = scan_data
        self.results['scan_info'] = scan_info

        # Convert lists to numpy arrays and store N's
        for key in self.results.keys():
            tokens = key.split('_')
            for dkey in self.results[key].keys():
                self.results[key][dkey] = np.asarray(self.results[key][dkey])
            if tokens[1] == 'data':
                probe = self.results[key].keys()[0]
                self.results['N_'+tokens[0]] = len(self.results[key][probe])

    def process_data(self, banner_width=60, exclude_first_maze=True):
        """Create figure plotting distributions of scan-firing measures and
        compute various relevant statistics
        """
        from scanr.tools.stats import smooth_pdf
        from scanr.tools.string import snake2title
        from scipy.stats import ttest_ind as ttest, ks_2samp as kstest, pearsonr

        os.chdir(self.datadir)
        self.out.outfd = file('head_scan_stats.txt', 'w')
        self.out.timestamp = False
        portrait = (8.5, 11)
        landscape = (11, 8.5)

        def mean_pm_sem(a):
            return '%0.4f +/- %0.4f'%(a.mean(), a.std()/np.sqrt(a.size))

        def print_stats(data, label, across):
            self.out((' ' + label + ' ').center(banner_width, '-'))
            var_label = '(%s | %s)'%(label, across)
            self.out('Median %s = %.4f'%(var_label, np.median(data)))
            self.out('Mean/SEM %s = %s'%(var_label, mean_pm_sem(data)))
            self.out('S.D. %s = %.4f'%(var_label, data.std()))
            self.out('Range %s = [ %0.4f , %0.4f ]'%
                (var_label, data.min(), data.max()))

        res = self.results
        rat_data = res['rat_data']
        day_info = res['day_info']
        day_data = res['day_data']
        maze_info = res['maze_info']
        maze_data = res['maze_data']
        scan_info = res['scan_info']
        scan_data = res['scan_data']

        # Create per-rat plots
        plt.ioff()
        self.figure = {}
        self.figure['rat'] = f = plt.figure(figsize=portrait)
        f.suptitle('Per-Rat Averages of Head Scan Statistics')
        self.out((' Per-Rat (N=%d) '%res['N_rat']).center(banner_width, '='))

        N = len(rat_data)
        kw = dict(lw=2, aa=True)
        for i, data_type in enumerate(sorted(rat_data.keys())):
            ax = plt.subplot(N, 1, i+1)
            label = snake2title(data_type)
            data = rat_data[data_type]

            ax.plot(*smooth_pdf(data), **kw)

            ax.axis('tight')
            v = list(ax.axis())
            v[3] *= 1.1
            ax.axis(v)

            med_data_type = np.median(data)
            ax.plot([med_data_type]*2, [v[2], v[3]], 'k--')
            print_stats(data, label, 'Rats')

            ax.set_ylabel('p[ %s ]'%label)

        # Create per-day plots
        self.figure['day'] = f = plt.figure(figsize=portrait)
        f.suptitle('Per-Day Head Scan Statistics')
        self.out('')
        self.out((' Per-Day (N=%d) '%res['N_day']).center(banner_width, '='))

        day_number = day_info['day_number']
        days = np.unique(day_number)
        days = days[days <= 10]
        N = len(day_data)
        for i, data_type in enumerate(sorted(day_data.keys())):
            ax = plt.subplot(N, 1, i+1)
            label = snake2title(data_type)
            data = day_data[data_type]
            data_by_days = [data[day_number==day] for day in days]

            ax.boxplot(data_by_days, positions=days)
            ax.set_xticklabels(days)

            print_stats(data, label, 'Days')
            self.out('----')
            t = ttest(data_by_days[0], data_by_days[1])
            k = kstest(data_by_days[0], data_by_days[1])
            self.out('T-test (%s | Day 1->2): t = %.4f, p = %.8f'%(label, t[0], t[1]))
            self.out('KS-test (%s | Day 1->2): D = %.4f, p = %.8f'%(label, k[0], k[1]))
            self.out('')
            if t[1] < 0.05:
                ax.text(0.5, 0.8, '*t', size='x-large', transform=ax.transAxes)
            if k[1] < 0.05:
                ax.text(0.5, 0.6, '*KS', size='x-large', transform=ax.transAxes)

            ax.set_ylabel(label)
            if i == N - 1:
                ax.set_xlabel('Days')

        # Create per-maze plots - maze number
        self.figure['maze_number'] = f = plt.figure(figsize=portrait)
        f.suptitle('Per-Maze Head Scan Statistics - By Maze Number')
        self.out('')
        self.out((' Per-Maze (N=%d) '%res['N_maze']).center(banner_width, '='))

        maze_number = maze_info['maze_number']
        mazes = np.unique(maze_number)
        N = len(maze_data)
        mazes = mazes[mazes <= 5]
        for i, data_type in enumerate(sorted(maze_data.keys())):
            ax = plt.subplot(N, 1, i+1)
            label = snake2title(data_type)
            data = maze_data[data_type]
            data_by_mazes = [data[maze_number==maze] for maze in mazes]

            ax.boxplot(data_by_mazes, positions=mazes)
            ax.set_xticklabels(mazes)

            print_stats(data, label, 'Mazes')
            self.out('----')
            t = ttest(data_by_mazes[0], data_by_mazes[1])
            k = kstest(data_by_mazes[0], data_by_mazes[1])
            self.out('T-test (%s | Maze 1->2): t = %.4f, p = %.8f'%(label, t[0], t[1]))
            self.out('KS-test (%s | Maze 1->2): D = %.4f, p = %.8f'%(label, k[0], k[1]))
            self.out('')
            if t[1] < 0.05:
                ax.text(0.4, 0.8, '*t', size='x-large', transform=ax.transAxes)
            if k[1] < 0.05:
                ax.text(0.4, 0.6, '*KS', size='x-large', transform=ax.transAxes)

            ax.set_ylabel(label)
            if i == N - 1:
                ax.set_xlabel('Maze Number')

        # Create per-maze plots - maze type
        self.figure['maze_type'] = f = plt.figure(figsize=landscape)
        if exclude_first_maze:
            f.suptitle('Per-Maze Head Scan Statistics - By Maze Type, Excluding M1')
        else:
            f.suptitle('Per-Maze Head Scan Statistics - By Maze Type')
        self.out('')
        self.out(' STD-MIS Mazes '.center(banner_width, '-'))

        maze_type = maze_info['maze_type']
        for i, data_type in enumerate(sorted(maze_data.keys())):
            ax = plt.subplot(1, N, i+1)
            label = snake2title(data_type)
            data = maze_data[data_type]
            if exclude_first_maze:
                data_by_type = [data[np.logical_and(maze_type==is_MIS, maze_number>1)]
                    for is_MIS in (False, True)]
            else:
                data_by_type = [data[maze_type==is_MIS] for is_MIS in (False, True)]

            ax.boxplot(data_by_type)
            ax.set_xticklabels(['STD', 'MIS'])

            print_stats(data_by_type[0], label, 'STD Mazes')
            print_stats(data_by_type[1], label, 'MIS Mazes')
            self.out('----')
            t = ttest(data_by_type[0], data_by_type[1])
            k = kstest(data_by_type[0], data_by_type[1])
            self.out('T-test (%s): t = %.4f, p = %.8f'%(label, t[0], t[1]))
            self.out('KS-test (%s): D = %.4f, p = %.8f'%(label, k[0], k[1]))
            self.out('')
            if t[1] < 0.05:
                ax.text(0.5, 0.9, '*t', size='x-large', transform=ax.transAxes)
            if k[1] < 0.05:
                ax.text(0.5, 0.8, '*KS', size='x-large', transform=ax.transAxes)

            ax.set_title(label)
            ax.set_xlabel('Maze Type')

        # Create per-scan plots
        self.figure['scan'] = f = plt.figure(figsize=landscape)
        f.suptitle('Within-Session Head Scan Measurements')
        self.out('')
        self.out((' Per-Scan (N=%d) '%res['N_scan']).center(banner_width, '='))

        scan_time = scan_info['start_time']
        scan_number = scan_info['scan_number']
        scans = np.unique(scan_number)
        N = len(scan_data)
        scans = scans[scans <= 20]

        for i, data_type in enumerate(sorted(scan_data.keys())):
            ax = plt.subplot(2, N, i+1)
            label = snake2title(data_type)
            data = scan_data[data_type]

            ymax = data[np.argsort(data)[-int(0.01*res['N_scan'])]]
            H, xe, ye = np.histogram2d(scan_time, data,
                range=[[0, 600], [0, ymax]], bins=64)
            ax.imshow(H.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                interpolation='nearest', cmap=plt.cm.gray_r)
            ax.axis('tight')

            print_stats(data, label, 'Scans')
            r, p = pearsonr(scan_time, data)
            self.out('----')
            self.out('Pearson r (Time x %s) = %0.5f, p = %0.8f'%(label, r, p))
            self.out('')

            ax.set_xlabel('Start Time (s)')
            ax.set_ylabel(label)

            ax = plt.subplot(2, N, i+3)
            data_by_sn = [data[scan_number==sn] for sn in scans]

            ax.boxplot(data_by_sn, positions=scans)

            ax.set_xticklabels(scans)
            ax.set_xlabel('Scan Number')
            ax.set_ylabel(label)

        plt.ion()
        plt.show()
        self.out.outfd.close()

