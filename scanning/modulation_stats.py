# encoding: utf-8
"""
modulation_stats.py -- Place-field modulation event statistics, with bootstrapping

Created by Joe Monaco on July 10, 2012.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division

# Library imports
import os
import cPickle
import numpy as np
import tables as tb
import matplotlib.pylab as plt

# Package imports
from scanr.lib import *
from scanr.cluster import get_min_quality_criterion
from .core.analysis import AbstractAnalysis
from scanr.field import mark_all_fields
from scanr.tools.stats import t_one_sample, t_welch, CI
from scanr.tools.string import snake2title
import tools.stats as stat

# Table descriptions
FieldDescr =     {  'rat'       :   tb.UInt16Col(pos=1),
                    'day'       :   tb.UInt16Col(pos=2),
                    'session'   :   tb.UInt16Col(pos=3),
                    'tc'        :   tb.StringCol(itemsize=8, pos=4),
                    'num'       :   tb.UInt16Col(pos=5),
                    'type'      :   tb.StringCol(itemsize=4, pos=6),
                    'area'      :   tb.StringCol(itemsize=4, pos=7),
                    'expt_type' :   tb.StringCol(itemsize=4, pos=8),
                    'events'    :   tb.UInt16Col(pos=9) }


class ModulationStats(AbstractAnalysis):

    """
    Bootstrapped prevalence statistics of place-field modulation events across
    session number and type
    """

    label = 'mod stats'

    def collect_data(self, min_quality='fair', table_name='potentiation',
        bins=Config['placefield']['default_bins'],
        min_rate=Config['placefield']['min_peak_rate']):
        """Merge event information from a /physiology/<event_table> into a new
        table with rows for every place field in the database, allowing for
        proper bootstrapping of the fractional prevalence of potentiation
        events.
        """
        area_query = '(area=="CA1")|(area=="CA3")'
        bin_width = 360.0 / bins

        # Check modulation event table
        self.results['table_name'] = table_name
        sessions_table = get_node('/metadata', 'sessions')
        mod_table = get_node('/physiology', table_name)
        tetrode_table = get_node('/metadata', 'tetrodes')
        self.out('Using %s with %d rows.'%(mod_table._v_pathname, mod_table.nrows))

        # Place-field tables and iterator
        data_file = self.open_data_file()
        field_table = data_file.createTable('/', 'place_fields',
            FieldDescr, title='Place Field Event Data')
        row = field_table.row

        # Quality criterion
        Quality = get_min_quality_criterion(min_quality)

        self.out('Gathering place field data...')
        for dataset in TetrodeSelect.datasets(area_query, allow_ambiguous=True):
            rat, day = dataset

            Tetrodes = TetrodeSelect.criterion(dataset, area_query,
                allow_ambiguous=True)
            Criteria = AND(Quality, Tetrodes, PlaceCellCriteria)

            for session in sessions_table.where('(rat==%d)&(day==%d)'%dataset):
                rds = rat, day, session['session']

                # Set cluster criteria and load session data
                session_data = SessionData.get(rds)
                session_data.cluster_criteria = Criteria

                self.out.printf('Scanning: ', color='lightgray')
                for tc in session_data.get_clusters():

                    tt, cl = parse_cell_name(tc)
                    area = get_unique_row(tetrode_table,
                        '(rat==%d)&(day==%d)&(tt==%d)'%(rat, day, tt))['area']

                    # Create firing-rate map to find fields for accurate field count
                    filter_kwds = dict( velocity_filter=True,
                                        exclude_off_track=True,
                                        exclude=session_data.extended_scan_and_pause_list )
                    ratemap_kwds = dict(bins=bins, blur_width=bin_width)
                    ratemap_kwds.update(filter_kwds)

                    R_full = session_data.get_cluster_ratemap(tc, **ratemap_kwds)
                    if R_full.max() < min_rate:
                        continue

                    for f, field in enumerate(mark_all_fields(R_full)):

                        # Check for any events for this field
                        field_num = f + 1
                        events = mod_table.getWhereList(session_data.session_query +
                            '&(tc=="%s")&(fieldnum==%d)' % (tc, field_num))

                        if len(events):
                            self.out.printf(u'\u25a1', color='green')
                        else:
                            self.out.printf(u'\u25a1', color='red')

                        # Add row to place-field table
                        row['rat'] = rat
                        row['day'] = day
                        row['session'] = session['session']
                        row['tc'] = tc
                        row['num'] = field_num
                        row['type'] = session['type']
                        row['area'] = area
                        row['expt_type'] = session['expt_type']
                        row['events'] = len(events)
                        row.append()

                self.out.printf('\n')
            field_table.flush()
        self.out('All done!')

    def process_data(self, first_day=False, ymax=0.5):
        """Compute prevalence fractions across session number but calculated within-rat
        and compared across-session
        """
        suffix = first_day and 'firstday' or 'alldays'
        self.start_logfile('figure-within-rat-%s' % suffix)

        if first_day:
            self.out('Restricting analysis to first day of testing.')
            daystr = '(day==1)&'
            daylabel = 'First Day '
        else:
            self.out('Prevalence analysis on all data.')
            daystr = '(day!=0)&'
            daylabel = ''

        data = self.get_data_file().root.place_fields
        self.out("Found %d events across %d place fields." % (
            np.sum(data.col('events')), data.nrows))

        f = {}
        for expt, num_sessions in [('DR', 5), ('NOV', 3)]:
            rat_list = unique_rats(data, condn='%s(expt_type=="%s")' % (daystr, expt))
            f[expt] = np.zeros((len(rat_list), num_sessions), 'd')
            mask = np.zeros_like(f[expt], '?')
            self.out('Computing %s: N = %d rats' % (expt, len(rat_list)))
            for j, number in enumerate(range(1, num_sessions + 1)):
                self.out('Session = %d'%number)
                for i,rat in enumerate(rat_list):
                    self.out.printf('.')
                    ix = data.getWhereList('%s(rat==%d)&(expt_type=="%s")&(session==%d)' % (
                        daystr, rat, expt, number))
                    fields = len(ix)
                    hits = np.sum(data[ix]['events'] > 0)
                    if fields:
                        f[expt][i,j] = hits / fields
                    else:
                        mask[i,j] = True
                self.out.printf('\n')
            f[expt] = np.ma.masked_where(mask, f[expt])
        self.close_data_file()

        # Between group differences
        f_DR = f['DR']
        f_NOV = f['NOV']
        self.out('DR test: %s' % stat.friedman_str(f_DR))
        self.out('NOV test: %s' % stat.friedman_str(f_NOV))

        # Bar charts, maze number and maze type
        f = self.new_figure('prevalence_%s' % suffix,
            'Within-Rat Prevalence Fraction of %sEvents' % daylabel, figsize=(10,7))

        fmt = dict(width=0.3, linewidth=0, color='0.4', edgecolor='none', ecolor='k', capsize=0)

        def error(F):
            return F.std(axis=0) / np.sqrt(F.shape[0])

        ax = f.add_subplot(221)
        ax.bar(
            [0, 0.4, 0.8, 1.2, 1.6, 2.4, 2.8, 3.2],
            np.r_[f_DR.mean(axis=0), f_NOV.mean(axis=0)],
            yerr=np.r_[error(f_DR), error(f_NOV)], **fmt)
        ax.set_xlim(-0.3, 5.1)
        ax.set_ylim(0, ymax)
        ax.set(xticks=[], xticklabels=[])
        ax.tick_params(right=False, direction='out')
        ax.axhline(1.0, c='k', ls='--', lw=1)

        ax = f.add_subplot(222)
        ax.boxplot([f_DR[:,0], f_DR[:,1], f_DR[:,2], f_DR[:,3], f_DR[:,4],
            f_NOV[:,0], f_NOV[:,1], f_NOV[:,2]])
        ax.set_ylim(0, ymax)

        self.close_logfile()
        self.close_data_file()

    def bootstrap_overall_fractions(self, first_day=False, shuffles=1000, ymax=0.6):
        """Note: this is old processing code, before the proper within-rat analysis
        was implemented
        """
        self.out.outfd = file(os.path.join(self.datadir, 'figure.log'), 'w')
        self.out.timestamp = False

        if first_day:
            daystr = '(day==1)&'
            suffix = '_day1'
            daylabel = 'First Day '
        else:
            daystr = '(day!=0)&'
            suffix = ''
            daylabel = ''

        data_file = self.get_data_file()
        field_data = data_file.root.place_fields
        N = field_data.nrows
        self.out("Found %d rows of place field data."%N)

        if os.path.exists(os.path.join(self.datadir, 'F_DR_type%s.npy'%suffix)):
            self.out('Loading: DR, across maze type')
            F_DR_type = np.load(os.path.join(self.datadir, 'F_DR_type%s.npy'%suffix))
        else:
            self.out('Computing: DR, across maze type')
            F_DR_type = np.zeros((shuffles + 1, 2), 'd')
            for j, mtype in enumerate(('STD', 'MIS')):
                self.out('Type = %s'%mtype)
                I = field_data.getWhereList(daystr + '(session!=1)&(type=="%s")'%mtype)
                N = len(I)
                if not N:
                    continue
                B = np.vstack((np.arange(N), np.random.random_integers(0, N - 1, size=(shuffles, N))))
                for i in xrange(shuffles + 1):
                    self.out.printf('.')
                    F_DR_type[i,j] = sum([field_data[I[ix]]['events'] > 0 for ix in B[i]]) / N
                self.out.printf('\n')
            np.save(os.path.join(self.datadir, 'F_DR_type%s'%suffix), F_DR_type)

        if os.path.exists(os.path.join(self.datadir, 'F_DR_num%s.npy'%suffix)):
            self.out('Loading: DR, across maze number')
            F_DR_num = np.load(os.path.join(self.datadir, 'F_DR_num%s.npy'%suffix))
        else:
            self.out('Computing: DR, across maze number')
            F_DR_num = np.zeros((shuffles + 1, 5), 'd')
            for j, number in enumerate(range(1, 6)):
                self.out('Session = %d'%number)
                I = field_data.getWhereList(daystr + '(expt_type=="DR")&(session==%d)'%number)
                N = len(I)
                if not N:
                    continue
                B = np.vstack((np.arange(N), np.random.random_integers(0, N - 1, size=(shuffles, N))))
                for i in xrange(shuffles + 1):
                    self.out.printf('.')
                    F_DR_num[i,j] = sum([field_data[I[ix]]['events'] > 0 for ix in B[i]]) / N
                self.out.printf('\n')
            np.save(os.path.join(self.datadir, 'F_DR_num%s'%suffix), F_DR_num)

        if os.path.exists(os.path.join(self.datadir, 'F_nov_type%s.npy'%suffix)):
            self.out('Loading: Nov, across maze type')
            F_nov_type = np.load(os.path.join(self.datadir, 'F_nov_type%s.npy'%suffix))
        else:
            self.out('Computing: Nov, across maze type')
            F_nov_type = np.zeros((shuffles + 1, 2), 'd')
            for j, mtype in enumerate(('FAM', 'NOV')):
                self.out('Type = %s'%mtype)
                I = field_data.getWhereList(daystr + '(session!=1)&(type=="%s")'%mtype)
                N = len(I)
                if not N:
                    continue
                B = np.vstack((np.arange(N), np.random.random_integers(0, N - 1, size=(shuffles, N))))
                for i in xrange(shuffles + 1):
                    self.out.printf('.')
                    F_nov_type[i,j] = sum([field_data[I[ix]]['events'] > 0 for ix in B[i]]) / N
                self.out.printf('\n')
            np.save(os.path.join(self.datadir, 'F_nov_type%s'%suffix), F_nov_type)

        if os.path.exists(os.path.join(self.datadir, 'F_nov_num%s.npy'%suffix)):
            self.out('Loading: Nov, across maze number')
            F_nov_num = np.load(os.path.join(self.datadir, 'F_nov_num%s.npy'%suffix))
        else:
            self.out('Computing: Nov, across maze number')
            F_nov_num = np.zeros((shuffles + 1, 3), 'd')
            for j, number in enumerate(range(1, 4)):
                self.out('Session = %d'%number)
                I = field_data.getWhereList(daystr + '(expt_type=="NOV")&(session==%d)'%number)
                N = len(I)
                if not N:
                    continue
                B = np.vstack((np.arange(N), np.random.random_integers(0, N - 1, size=(shuffles, N))))
                for i in xrange(shuffles + 1):
                    self.out.printf('.')
                    F_nov_num[i,j] = sum([field_data[I[ix]]['events'] > 0 for ix in B[i]]) / N
                self.out.printf('\n')
            np.save(os.path.join(self.datadir, 'F_nov_num%s'%suffix), F_nov_num)

        self.close_data_file()

        # Bar charts, maze number and maze type
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure['field_stats%s'%suffix] = f = plt.figure(figsize=(10,7))
        f.suptitle('Prevalence Fraction of %sEvents: %s'%(daylabel, snake2title(self.results['table_name'])))

        fmt = dict(width=0.3, linewidth=0, color='0.4', edgecolor='none', ecolor='k', capsize=0)

        def error(F, alpha=0.05):
            e = np.empty((2, F.shape[1]), 'd')
            for i, vals in enumerate(F[1:].T):
                e[:,i] = CI(vals, alpha=alpha) # empirical confidence interval
            e -= F[0] # errorbar yerr weirdness, mean - row1 to mean + row2
            e[0] *= -1 # must invert to get appropriate behavior!!!
            return e

        ax = f.add_subplot(221)
        ax.bar(
            [0, 0.4, 1, 1.4],
            np.r_[F_DR_type[0], F_nov_type[0]],
            yerr=error(np.c_[F_DR_type, F_nov_type]), **fmt)
        ax.set_xlim(-0.3, 3.7)
        ax.set_ylim(0, ymax)
        ax.set(xticks=[], xticklabels=[])
        ax.tick_params(right=False, direction='out')
        ax.axhline(1.0, c='k', ls='--', lw=1)

        ax = f.add_subplot(222)
        ax.boxplot([F_DR_type[:,0], F_DR_type[:,1], F_nov_type[:,0], F_nov_type[:,1]])
        ax.set_ylim(0, ymax)

        ax = f.add_subplot(223)
        ax.bar(
            [0, 0.4, 0.8, 1.2, 1.6, 2.4, 2.8, 3.2],
            np.r_[F_DR_num[0], F_nov_num[0]],
            yerr=error(np.c_[F_DR_num, F_nov_num]), **fmt)
        ax.set_xlim(-0.3, 5.1)
        ax.set_ylim(0, ymax)
        ax.set(xticks=[], xticklabels=[])
        ax.tick_params(right=False, direction='out')
        ax.axhline(1.0, c='k', ls='--', lw=1)

        ax = f.add_subplot(224)
        ax.boxplot([F_DR_num[:,0], F_DR_num[:,1], F_DR_num[:,2], F_DR_num[:,3], F_DR_num[:,4],
            F_nov_num[:,0], F_nov_num[:,1], F_nov_num[:,2]])
        ax.set_ylim(0, ymax)

        self.out.outfd.close()
        self.close_data_file()
