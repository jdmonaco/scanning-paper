# encoding: utf-8
"""
forward.py -- Forward analysis of scan-activatied potentiation

Created by Joe Monaco on August 27, 2013.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division
from collections import namedtuple

# Library imports
import os
import re
import subprocess
import operator as op
import numpy as np
import tables as tb
import scipy.stats as st
import matplotlib.pyplot as plt
import enthought.traits.api as traits

# Package imports
from scanr.config import Config
from scanr.session import SessionData
from scanr.spike import parse_cell_name
from scanr.behavior import Scan
from scanr.data import (get_unique_row, unique_rats, unique_values, unique_datasets,
    get_node, unique_sessions, get_maze_list)
from scanr.cluster import (AND, get_min_quality_criterion,
    get_tetrode_restriction_criterion, PlaceCellCriteria)

# Local imports
from .core.analysis import AbstractAnalysis
from .tools.plot import densitymap, quicktitle
from .tools.misc import memoize, memoize_limited, DataSpreadsheet
from .tools.string import snake2title
from .tools.stats import KT_estimate2, t_one_sample

# Constants/defaults
HALF_WINDOWS_DEFAULT = (5,10,15,20,25,35,45,90,180)


class ScanIndex(object):

    # Names = ('S', 'K', 'R', 'max_radius', 'median_radius')
    AllNames = ('K', 'R', 'S')
    ActiveNames = ('R', 'S')

    def __init__(self, scan_phase='scan'):
        self.scan_deviation_index = ScanFiringDeviationIndex(scan_phase=scan_phase)
        self.scan_start_point, self.scan_end_point = Scan.PhasePoints[scan_phase]

    def _scan_spikes(self, scan, cluster):
        return np.logical_and(
            cluster.spikes >= scan['start'], cluster.spikes <= scan['end'])

    def _scan_spike_count(self, scan, cluster):
        return np.sum(self._scan_spikes(scan, cluster))

    def _spike_behavior(self, which, session, scan, cluster):
        scan_spikes = cluster.spikes[self._scan_spikes(scan, cluster)]
        return session.F_(which)(session.T_(scan_spikes))

    def compute(self, which, session, scan, cluster):
        if which == 'S':
            index = self.scan_deviation_index.compute(session, scan, cluster)
        elif which == 'K':
            index = self._scan_spike_count(scan, cluster)
        elif which == 'R':
            K = self._scan_spike_count(scan, cluster)
            index = K / scan['duration']
        elif which == 'max_radius':
            rad = self._spike_behavior('radius', session, scan, cluster)
            index = rad.size and np.max(np.abs(rad)) or -1
        elif which == 'median_radius':
            rad = self._spike_behavior('radius', session, scan, cluster)
            index = rad.size and np.abs(np.median(rad)) or -1
        else:
            raise ValueError, 'unknown index %s' % which

        return index


class ScanFiringDeviationIndex(traits.HasTraits):

    """
    Algorithm for computing scan-firing index for deviation from expectation
    """

    # Parameters
    dt = traits.Float(0.1)
    x_bins = traits.Int(24)
    k_edges = traits.Tuple(tuple(xrange(20)))
    scan_phase = traits.Trait(Scan.PhaseNames)

    # Properties
    min_phase_dts = traits.Long(4e5)
    dts = traits.Long
    H_bins = traits.Array

    def compute(self, session, scan, cluster, plot=False):
        self.H_bins = self._get_H_bins()

        scan_start, scan_end = tuple(
            scan[k] for k in Scan.PhasePoints[self.scan_phase])
        if scan_end - scan_start < self.min_phase_dts:
            return 0.0

        H_xk = self._H_xk(session, scan['prefix'], cluster)
        H_xk_scan = self._H_xk_scan(session, scan_start, scan_end, cluster)
        if H_xk.ndim != 2 or H_xk_scan.ndim != 2:
            return 0.0

        P_xk = np.zeros_like(H_xk)
        P_xk_scan = np.zeros_like(H_xk)
        P_xk_diff = np.zeros_like(H_xk)

        N_x = H_xk.sum(axis=1)
        D_hist = (N_x != 0) # non-zero domain
        P_xk[D_hist] = H_xk[D_hist].cumsum(axis=1) / N_x[D_hist][:,np.newaxis]

        N_x_scan = H_xk_scan.sum(axis=1)
        W_x = (N_x_scan / N_x_scan.sum())[:,np.newaxis]
        D = (N_x != 0) * (N_x_scan != 0) # non-zero domain

        S = 0.0
        if np.any(D):
            P_xk_scan[D] = H_xk_scan[D].cumsum(axis=1) / N_x_scan[D][:,np.newaxis]
            P_xk_diff[D] = P_xk[D] - P_xk_scan[D]
            max_abs = np.argmax(np.abs(P_xk_diff[D]), axis=1)
            S = np.sum(W_x[D] * P_xk_diff[D, max_abs][:, np.newaxis])

        if plot:
            self._plot_results(session.rds, cluster.name, scan, S, P_xk,
                P_xk_scan, P_xk_diff, W_x)

        return S

    def _plot_results(self, rds, tc, scan, S, P_xk, P_xk_scan, P_xk_diff, W_x):
        f = plt.figure()
        density_kwds = dict(cmask='c', cmap='gray') #, norm=False, cmin=0, cmax=1)
        diff_kwds = dict(cmask='w', cmap='RdBu', norm=False, cmin=-0.5, cmax=0.5)
        f.suptitle('rat%03d-%02d-m%d %s scan%02d\nS = %.5f' % (
            rds + (tc, scan['number'], S)))
        ax = densitymap(P_xk, (0, 360), (0, self.k_edges[-1] + 1),
            ax=plt.subplot(221), **density_kwds)
        quicktitle(ax, 'P_xk')
        ax = densitymap(P_xk_scan, (0, 360), (0, self.k_edges[-1] + 1),
            ax=plt.subplot(222), **density_kwds)
        quicktitle(ax, 'P_xk_scan')
        ax = densitymap(P_xk_diff, (0, 360), (0, self.k_edges[-1] + 1),
            ax=plt.subplot(223), **diff_kwds)
        quicktitle(ax, 'P_xk_diff')
        ax = f.add_subplot(224)
        centers = (lambda e: (e[:-1] + e[1:]) / 2)(self.H_bins[0])
        ax.plot(centers, W_x.flatten(), 'b-o', mew=0, ms=6, mfc='b', lw=2)
        ax.set_xlim(0, 360)
        quicktitle(ax, 'W_x')

    def _get_H_bins(self):
        return [np.linspace(0, 360, self.x_bins+1),
                np.array(self.k_edges + (np.inf,))]

    @memoize_limited(100)
    def _first_scan_interval(self, session, scan_start):
        t, running = self._samples(session)
        i = 0
        while t[i+1] < scan_start:
            i += 1
        return i

    def _H_xk(self, session, scan_start, cluster):
        i_scan = self._first_scan_interval(session, scan_start)
        return np.histogram2d(
            self._track_angles(session)[:i_scan],
            self._spike_counts(session, cluster)[:i_scan],
            bins=self.H_bins)[0]

    def _H_xk_scan(self, session, scan_start, scan_end, cluster):
        return np.histogram2d(
            self._track_angles_scan(session, scan_start, scan_end),
            self._spike_counts_scan(scan_start, scan_end, cluster),
            bins=self.H_bins)[0]

    @memoize_limited(100)
    def _track_angles(self, session):
        T, F = session.T_, session.F_('alpha_unwrapped')
        t, running = self._samples(session)
        bins = ((t[1:] + t[:-1]) / 2).astype(long)[running]
        return F(T(bins)) % 360

    @memoize_limited(100)
    def _spike_counts(self, session, cluster):
        t, running = self._samples(session)
        return np.histogram(cluster.spikes, bins=t)[0][running]

    @memoize_limited(100)
    def _samples(self, session):
        ts = np.arange(session.start, session.end, self.dts)
        running = session.filter_tracking_data(ts, **self._filter(session))
        i = 0
        edge_filter = []
        while i < ts.size - 1:
            edge_filter.append(running[i] and running[i+1])
            i += 1
        return ts, np.nonzero(edge_filter)

    @memoize_limited(100)
    def _track_angles_scan(self, session, scan_start, scan_end):
        T, F = session.T_, session.F_('alpha_unwrapped')
        t = self._samples_scan(scan_start, scan_end)
        bins = ((t[1:] + t[:-1]) / 2).astype(long)
        return F(T(bins)) % 360

    def _spike_counts_scan(self, scan_start, scan_end, cluster):
        t = self._samples_scan(scan_start, scan_end)
        return np.histogram(cluster.spikes, bins=t)[0]

    @memoize_limited(100)
    def _samples_scan(self, scan_start, scan_end):
        return np.arange(scan_start, scan_end, self.dts)

    def _dts_default(self):
        return long(self.dt * Config['sample_rate']['time'])

    @memoize_limited(100)
    def _filter(self, session):
        return dict(
            boolean_index=True,
            velocity_filter=False,
            speed_filter=True,
            exclude=session.pause_list,
            exclude_off_track=False
        )

def create_table(the_file, where, name, description, **kwds):
    try:
        table = the_file.getNode(where, name=name)
    except tb.NoSuchNodeError:
        pass
    else:
        raw_input('Found previous %s. Erasing...' % table._v_pathname)
        the_file.removeNode(where, name=name)
    finally:
        table = the_file.createTable(where, name, description, **kwds)
    return table

def create_group(the_file, where, name, **kwds):
    try:
        grp = the_file.getNode(where, name=name)
    except tb.NoSuchNodeError:
        pass
    else:
        raw_input('Found previous %s. Erasing...' % grp._v_pathname)
        the_file.removeNode(where, name=name, recursive=True)
    finally:
        grp = the_file.createGroup(where, name, description, **kwds)
    return grp

def create_array(the_file, where, name, array_object, **kwds):
    try:
        arr = the_file.getNode(where, name=name)
    except tb.NoSuchNodeError:
        pass
    else:
        the_file.removeNode(where, name=name)
    finally:
        arr = the_file.createArray(where, name, array_object, **kwds)
    return arr


class ForwardAnalysis(AbstractAnalysis):

    """
    Forward analysis of scan-activation of place-field potentiation
    """

    label = 'forward analysis'

    min_quality = traits.Str('fair')

    def __init__(self, *args, **traits):
        AbstractAnalysis.__init__(self, *args, **traits)
        self.results['there_is_no_spoon'] = True
        self.finished = True

    def collect_data(self):
        """Iterate across all scan–cell pairs, find scans with unexpectedly high levels
        of cell firing, then compute the predictive power of high scan firing for the
        existence of a subsequent potentiation event
        """
        # self.create_scan_cell_table()
        # self.create_outcome_table()

    def create_scan_cell_table(self, scan_phase='scan'):
        """For every scan–cell pair, compute the relative index of cell firing that
        occurred during the scan and previous cell firing on the track
        """
        scan_table_description = {
            'id'              :  tb.UInt32Col(pos=1),
            'scan_id'         :  tb.UInt16Col(pos=2),
            'rat'             :  tb.UInt16Col(pos=3),
            'day'             :  tb.UInt16Col(pos=4),
            'session'         :  tb.UInt16Col(pos=5),
            'session_start_angle'     :  tb.FloatCol(pos=6),
            'session_end_angle'       :  tb.FloatCol(pos=7),
            'tc'              :  tb.StringCol(itemsize=8, pos=8),
            'type'            :  tb.StringCol(itemsize=4, pos=9),
            'expt_type'       :  tb.StringCol(itemsize=4, pos=10),
            'area'            :  tb.StringCol(itemsize=4, pos=11),
            'subdiv'          :  tb.StringCol(itemsize=4, pos=12),
            'duration'        :  tb.FloatCol(pos=13),
            'magnitude'       :  tb.FloatCol(pos=14),
            'angle'           :  tb.FloatCol(pos=15)
        }

        def add_scan_index_column_descriptors(descr):
            pos = 16
            for name in ScanIndex.AllNames:
                descr[name] = tb.FloatCol(pos=pos)
                pos += 1
        add_scan_index_column_descriptors(scan_table_description)

        data_file = self.get_data_file(mode='a')
        scan_cell_table = create_table(data_file, '/', 'scan_cell_info',
            scan_table_description, title='Metadata for Scan-Cell Pairs')
        scan_cell_table._v_attrs['scan_phase'] = scan_phase
        row = scan_cell_table.row
        row_id = 0

        scans_table = get_node('/behavior', 'scans')
        sessions_table = get_node('/metadata', 'sessions')
        tetrodes_table = get_node('/metadata', 'tetrodes')

        cornu_ammonis_query = '(area=="CA1")|(area=="CA3")'
        hippocampal_datasets = unique_datasets('/metadata', 'tetrodes',
            condn=cornu_ammonis_query)

        quality_place_cells = AND(get_min_quality_criterion(self.min_quality),
            PlaceCellCriteria)

        index = ScanIndex(scan_phase=scan_phase)

        for dataset in hippocampal_datasets:
            dataset_query = '(rat==%d)&(day==%d)' % dataset

            hippocampal_tetrodes = unique_values(tetrodes_table, column='tt',
                condn='(%s)&(%s)' % (dataset_query, cornu_ammonis_query))
            cluster_criteria = AND(quality_place_cells,
                get_tetrode_restriction_criterion(hippocampal_tetrodes))

            for maze in get_maze_list(*dataset):
                rds = dataset + (maze,)
                session = SessionData(rds=rds)
                place_cells = session.get_clusters(cluster_criteria)
                session_start_angle = np.median(session.trajectory.alpha_unwrapped[:5])
                session_end_angle = np.median(session.trajectory.alpha_unwrapped[-5:])

                self.out('Computing scan index for %s...' % session.data_group._v_pathname)

                for scan in scans_table.where(session.session_query):
                    self.out.printf('|', color='cyan')

                    for cell in place_cells:
                        cluster = session.cluster_data(cell)

                        tt, cl = parse_cell_name(cluster.name)
                        tetrode = get_unique_row(tetrodes_table,
                            '(rat==%d)&(day==%d)&(tt==%d)' % (rds[0], rds[1], tt))

                        row['id'] = row_id
                        row['scan_id'] = scan['id']
                        row['rat'], row['day'], row['session'] = rds
                        row['session_start_angle'] = session_start_angle
                        row['session_end_angle'] = session_end_angle
                        row['tc'] = cluster.name
                        row['type'] = session.attrs['type']
                        row['expt_type'] = get_unique_row(sessions_table,
                            session.session_query)['expt_type']
                        row['area'] = tetrode['area']
                        row['subdiv'] = tetrode['area'] + tetrode['subdiv'][:1]
                        row['angle'] = session.F_('alpha_unwrapped')(session.T_(scan['start']))
                        row['duration'] = scan['duration']
                        row['magnitude'] = scan['magnitude']

                        for index_name in ScanIndex.AllNames:
                            row[index_name] = index.compute(index_name, session, scan, cluster)

                        self.out.printf('.', color='green')
                        row_id += 1
                        row.append()

                        if row_id % 100 == 0:
                            scan_cell_table.flush()

                self.out.printf('\n')

        scan_cell_table.flush()
        self.out('Finished creating %s.' % scan_cell_table._v_pathname)

    def generate_index_validation_set(self, N=100):
        data_file = self.get_data_file(mode='r')
        scan_cell_table = data_file.root.scan_cell_info
        scan_table = get_node('/behavior', 'scans')
        index = ScanFiringDeviationIndex()

        def get_subdir(name):
            subdir = os.path.join(self.datadir, name)
            if not os.path.isdir(subdir):
                os.makedirs(subdir)
            return subdir

        output_dir = get_subdir('index_validation')
        figure_dir = get_subdir(os.path.join('index_validation', 'figures'))

        spreadsheet = DataSpreadsheet(
            os.path.join(output_dir, 'scan_index_sample.csv'),
            [('sample', 'd'), ('rat', 'd'), ('day', 'd'), ('session', 'd'),
             ('cell', 's'), ('scan_number', 'd'), ('scan_start', 'd'),
             ('surprise', 'f')]
        )
        rec = spreadsheet.get_record()

        sample_ix = np.random.permutation(scan_cell_table.nrows)[:N]
        scan_index = scan_cell_table.col('S')[sample_ix]
        sample_ix = sample_ix[np.argsort(scan_index)]

        plt.ioff()
        figure_files = []
        sample_id = 1

        self.out('Generating sample records and figures...')
        for ix in sample_ix:
            self.out.printf('.')

            row = scan_cell_table[ix]
            rds = row['rat'], row['day'], row['session']
            scan = scan_table[row['scan_id']]

            rec['sample'] = sample_id
            rec['rat'], rec['day'], rec['session'] = rds
            rec['cell'] = row['tc']
            rec['scan_number'] = scan['number']
            rec['scan_start'] = scan['start']
            rec['surprise'] = row['S']
            spreadsheet.write_record(rec)

            session = SessionData.get(rds)
            cluster = session.cluster_data(row['tc'])
            index.compute(session, scan, cluster, plot=True)
            figure_fn = os.path.join(figure_dir, 'distros_%03d.pdf' % sample_id)
            figure_files.append(figure_fn)
            plt.savefig(figure_fn)
            plt.close()

            sample_id += 1

        self.out.printf('\n')
        spreadsheet.close()
        plt.ion()

        def concatenate_distro_figures_into_report():
            pdftk = '/opt/local/bin/pdftk'
            if os.path.exists(pdftk):
                distro_report = os.path.join(output_dir, 'distro_figures.pdf')
                retcode = subprocess.call([pdftk] + figure_files +
                    ['cat', 'output', distro_report])
                if retcode == 0:
                    self.out('Saved distribution figures to:\n%s' % distro_report)
                else:
                    self.out('Error saving figure report.', error=True)

        concatenate_distro_figures_into_report()

    def create_outcome_table(self, event_table='potentiation', half_windows=HALF_WINDOWS_DEFAULT):
        outcome_table_description = dict(scan_cell_id=tb.UInt32Col(pos=1))
        window_cols = map(lambda h: 'window_%d' % h, half_windows)

        def add_halfwindow_column_descriptors(descr):
            pos = 2
            for col in window_cols:
                descr[col] = tb.BoolCol(pos=pos)
                pos += 1

        add_halfwindow_column_descriptors(outcome_table_description)
        data_file = self.get_data_file(mode='a')
        outcome_table = create_table(data_file, '/', 'hit_outcome',
            outcome_table_description, title='Place-Field Event (Hit) Outcomes')
        row = outcome_table.row

        scan_cell_table = data_file.root.scan_cell_info
        scans = get_node('/behavior', 'scans')
        potentiation = get_node('/physiology', event_table)
        outcome_table._v_attrs['event_table'] = event_table

        # Adjust hit angle for depotentiation hack, in which event is actually last active traversal
        hit_angle = -360.0 # forward one lap
        if event_table == 'depotentiation':
            hti_angle = 0.0 # same lap

        def precache_sessions():
            [SessionData.get(rds, load_clusters=False) for rds in unique_sessions(scan_cell_table)]
        precache_sessions()

        @memoize
        def cell_query(session, tc):
            return '(%s)&(tc=="%s")' % (session.session_query, tc)

        def print_test_indicator(result):
            color = result and 'green' or 'red'
            self.out.printf(u'\u25a1', color=color)

        def test_for_event_hit(scan_angle, h, bounds):
            window = tuple(scan_angle + hit_angle + np.array([h, -h]))
            start_before = bounds[0] >= window[1]
            end_after = bounds[1] <= window[0]
            return start_before and end_after

        for pair in scan_cell_table.iterrows():
            self.out.printf('|', color='cyan')

            rds = pair['rat'], pair['day'], pair['session']
            session = SessionData.get(rds, load_clusters=False)
            event_bounds = lambda t: tuple(session.F_('alpha_unwrapped')(session.T_(t)))

            scan_angle = pair['angle']
            row['scan_cell_id'] = pair['id']

            for event in potentiation.where(cell_query(session, pair['tc'])):
                self.out.printf('|', color='lightgray')
                bounds = event_bounds(event['tlim'])

                for h, col in zip(half_windows, window_cols):
                    row[col] = test_for_event_hit(scan_angle, h, bounds)
                    print_test_indicator(row[col])

            row.append()
            if pair['id'] % 100 == 0:
                outcome_table.flush()
                self.out.printf(' [flush]\n%d / %d scan-cell pairs\n' % (
                    outcome_table.nrows, scan_cell_table.nrows), color='lightgray')

        self.out.printf('\n')
        self.close_data_file()

    def _find_half_windows(self, outcome_table):
        window_match = re.compile('window_(\d+)')
        h_col_map = {}
        for col in outcome_table.colnames:
            match = re.match(window_match, col)
            if match:
                h_col_map[int(match.groups()[0])] = col
        return h_col_map

    def _roc_array_name(self, name, h):
        return '%s_roc_window_%d' % (name, h)

    def _auc_array_name(self, name):
        return '%s_auc' % name

    def _rat_roc_array_name(self, rat, index_name):
        return 'rat%03d_%s_roc' % (rat, index_name)

    def _column_data(self, dtable, colname, qtable, condn):
        """Retrieve scan-cell-pair data from a column (colname) of a data table
        (dtable) using an optional query filter (condn) based on another
        scan-cell-pair table (qtable).
        """
        data = dtable.col(colname)
        if condn is not None:
            data = data[qtable.getWhereList(condn)]
        return data

    def _get_roc_criteria(self, x, decimation):
        return np.unique(np.r_[x.min(), x[::decimation], x.max()])

    def _roc_curve(self, index, outcome, criteria, test='ge'):
        TPR = np.empty(criteria.size, 'd')
        FPR = np.empty_like(TPR)
        P = np.sum(outcome)
        N = outcome.size - P
        f = getattr(op, test)
        for i, thr in enumerate(criteria):
            call = f(index, thr)
            TP = np.sum(np.logical_and(call, outcome))
            FP = np.sum(np.logical_and(call, np.logical_not(outcome)))
            TPR[i] = TP / P
            FPR[i] = FP / N
        return FPR, TPR

    def _get_roc_data_array(self, index, outcome, criteria, test_op):
        if np.all(outcome) or np.all(np.logical_not(outcome)):
            return np.array([[],[]])
        return np.vstack(self._roc_curve(index, outcome, criteria, test=test_op))

    def _exclude_marginal_laps(self, do_exclusion, condn, events):
        if do_exclusion:
            if events == 'potentiation':
                query = '(angle<session_start_angle-360)&(angle>session_end_angle+1080)'
            elif events == 'depotentiation':
                query = '(angle<session_start_angle-720)&(angle>session_end_angle+1080)'
            else:
                self.out('Warning: marginal laps undefined for %s events' % events)
                query = ''
            if condn is None:
                condn = query
            elif query:
                condn = '(%s)&(%s)' % (query, condn)
        return condn

    def generate_roc_curves(self, randomize=False, decimation=20, exclude_margin_laps=True,
        test_operator='ge', condn=None):

        data_file = self.get_data_file(mode='a')
        index_table = data_file.root.scan_cell_info
        outcome_table = data_file.root.hit_outcome
        event_table = outcome_table._v_attrs['event_table']

        h_to_colname = self._find_half_windows(outcome_table)
        h_windows = sorted(h_to_colname.keys())
        self.out('Found half-windows: %s' % str(h_windows))

        condn = self._exclude_marginal_laps(exclude_margin_laps, condn, event_table)
        if condn is not None:
            self.out('Using scan-cell-pair filter: "%s"' % condn)

        if hasattr(data_file.root, 'roc_curves'):
            data_file.removeNode(data_file.root, 'roc_curves', recursive=True)

        for index_name in ScanIndex.ActiveNames:

            scan_index = self._column_data(index_table, index_name,
                index_table, condn)
            self.out('Found %d scan-cell pairs.' % scan_index.size)

            if randomize:
                scan_index = np.random.permutation(scan_index)

            criteria = self._get_roc_criteria(scan_index, decimation)

            for h in h_windows:

                outcome = self._column_data(outcome_table, h_to_colname[h],
                    index_table, condn)

                arr = create_array(
                    data_file, '/roc_curves', self._roc_array_name(index_name, h),
                    self._get_roc_data_array(scan_index, outcome, criteria, test_operator),
                    title='%s ROC for Half-Window %d' % (index_name.title(), h),
                    createparents=True)

                self.out('Saved %s.' % arr._v_pathname)
                data_file.flush()

        attrs = data_file.root.roc_curves._v_attrs
        attrs['randomize'] = randomize
        attrs['decimation'] = decimation
        attrs['exclude_margin_laps'] = exclude_margin_laps
        attrs['test_operator'] = test_operator
        attrs['condn'] = condn
        attrs['event_table'] = outcome_table._v_attrs['event_table']

        self.close_data_file()

    def generate_rat_roc_curves(self, h=15, clear=True, index_name='R', decimation=5, condn=None,
        randomize=False, exclude_margin_laps=True, test_operator='ge'):
        assert index_name in ScanIndex.ActiveNames, 'invalid scan index: %s' % index_name

        data_file = self.get_data_file(mode='a')
        index_table = data_file.root.scan_cell_info
        outcome_table = data_file.root.hit_outcome
        event_table = outcome_table._v_attrs['event_table']

        h_to_colname = self._find_half_windows(outcome_table)
        def validate_half_window(h):
            if h in h_to_colname:
                self.out('Using half-window %d data ("%s")' % (h, h_to_colname[h]))
            else:
                raise ValueError, 'missing outcome data for half-window %d.' % h
        validate_half_window(h)

        condn = self._exclude_marginal_laps(exclude_margin_laps, condn, event_table)

        def rat_query(rat):
            query = 'rat==%d' % rat
            if condn is not None:
                query = '(%s)&(%s)' % (query, condn)
            return query

        if hasattr(data_file.root, 'rat_roc_curves') and clear:
            data_file.removeNode(data_file.root, 'rat_roc_curves', recursive=True)

        rat_list = unique_rats(index_table)
        self.out('Generating ROC curves for %d rats...' % len(rat_list))

        for rat in rat_list:

            query = rat_query(rat)
            scan_index = self._column_data(index_table, index_name, index_table, query)

            if randomize:
                scan_index = np.random.permutation(scan_index)

            outcome = self._column_data(outcome_table, h_to_colname[h], index_table, query)
            criteria = self._get_roc_criteria(scan_index, decimation)

            if not np.any(outcome):
                self.out('Skipping rat %d, no events...' % rat)
                continue

            arr = create_array(
                data_file, '/rat_roc_curves', self._rat_roc_array_name(rat, index_name),
                self._get_roc_data_array(scan_index, outcome, criteria, test_operator),
                title='%s ROC for Rat %d, Half-Window %d' % (index_name.title(), rat, h),
                createparents=True)

            self.out('Saved %s.' % arr._v_pathname)
            data_file.flush()

        attrs = data_file.root.rat_roc_curves._v_attrs
        attrs['decimation'] = decimation
        attrs['h_window'] = h
        attrs['condn'] = condn
        attrs['randomize'] = randomize
        attrs['test_operator'] = test_operator
        attrs['event_table'] = outcome_table._v_attrs['event_table']

        self.close_data_file()

    def _make_figure_title(self, index, h, condn, phase, events, test, randomized=False, paren=''):
        title = 'ROC Curves %s: %s Events for %s %s Criterion' % (
            paren and paren+' ' or '', snake2title(events), snake2title(index),
            dict(ge=">=", gt=">", le="<=", lt="<")[test])
        if randomized:
            title = '[Randomized] ' + title
        if np.iterable(h) and len(h) > 10:
            h_str = '[%s, ..., %s]' % (', '.join(map(str, h[:3])), ', '.join(map(str, h[-3:])))
        else:
            h_str = str(h)
        title += '\nHalf-Window = %s' % h_str
        if condn:
            title += '\nScan-Cell Filter: "%s"' % condn
        if phase != 'scan':
            title += '\nScan Phase: %s' % phase.title()
        return title

    def _compute_AUC(self, TPR, FPR, test):
        assert test in ('ge', 'gt', 'le', 'lt'), "bad test operation: '%s'" % test
        if test in ('ge', 'gt'):
            return -1 * np.trapz(TPR, FPR)
        elif test in ('le', 'lt'):
            return np.trapz(TPR, FPR)

    def plot_roc_curves(self, ylim=(0.5, 1.0), skinny_sweep_ax=False):
        self.start_logfile('roc_auc')

        data_file = self.get_data_file(mode='a')
        roc_group = data_file.root.roc_curves
        outcome_table = data_file.root.hit_outcome

        scan_phase = data_file.root.scan_cell_info._v_attrs['scan_phase']
        roc_attrs = roc_group._v_attrs
        scan_cell_condn = roc_attrs['condn']
        is_randomized = roc_attrs['randomize']
        test = roc_attrs['test_operator']
        event_table = roc_attrs['event_table']

        h_to_colname = self._find_half_windows(outcome_table)
        h_windows = sorted(h_to_colname.keys())

        self.close_figures()
        plt.ioff()
        if type(self.figure) is not dict:
            self.figure = {}

        fmt = dict(lw=2, ls='-') #, drawstyle='steps-mid')
        ndfmt = dict(c='0.4', ls='--', lw=1.5, zorder=-1)
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(h_windows)))

        for index_name in ScanIndex.ActiveNames:

            f = self.new_figure(
                '%s_ROC_curves' % index_name,
                self._make_figure_title(index_name, h_windows, scan_cell_condn, scan_phase,
                    event_table, test, randomized=is_randomized),
                figsize=(11,8))

            ax_roc = f.add_subplot(121)
            if skinny_sweep_ax:
                ax_auc = f.add_subplot(2,5,10)
            else:
                ax_auc = f.add_subplot(224)

            AUC = np.empty(len(h_windows), 'd')
            ndline = ax_roc.plot([0, 1], [0, 1], **ndfmt)

            for i, h in enumerate(h_windows):
                FPR, TPR = data_file.getNode(roc_group, self._roc_array_name(index_name, h)).read()
                AUC[i] = self._compute_AUC(TPR, FPR, test)
                ax_roc.plot(FPR, TPR, label=str(h), c=colors[i], zorder=len(h_windows)-i, **fmt)
                self.out('%s index, window %d: AUC = %f' % (snake2title(index_name), h, AUC[i]))

            arr = create_array(data_file, '/roc_curves', self._auc_array_name(index_name),
                AUC, title='ROC-AUC Across Half-Windows for %s Index' % index_name)
            self.out('Saved %s.' % arr._v_pathname)

            if len(h_windows) < 11:
                ax_roc.legend(loc='lower right')

            ax_roc.set(xlim=(0, 1), ylim=(0, 1))
            ax_roc.set_ylabel('TPR')
            ax_roc.set_xlabel('FPR')
            ax_roc.axis('scaled')

            chance_line = ax_auc.axhline(0.5, **ndfmt)
            ax_auc.plot(h_windows, AUC, '-', c='steelblue', lw=3, solid_capstyle='round')
            ax_auc.set(
                xlabel='Half-Window, degrees', ylabel="AUC",
                xlim=(0, h_windows[-1] + h_windows[0]), ylim=ylim,
                xticks=h_windows,
                xticklabels=([h_windows[0]] + ['']*(len(h_windows)-2) + [h_windows[-1]]))

        plt.ion()
        plt.show()

        self.close_logfile()
        self.close_data_file()

    def plot_rat_roc_curves(self, index_name='R', ylim=(0.5,1.0)):

        data_file = self.get_data_file(mode='r')
        roc_group = data_file.root.rat_roc_curves

        scan_phase = data_file.root.scan_cell_info._v_attrs['scan_phase']
        roc_attrs = roc_group._v_attrs
        scan_cell_condn = roc_attrs['condn']
        half_window = roc_attrs['h_window']
        is_randomized = roc_attrs['randomize']
        event_table = roc_attrs['event_table']
        test = roc_attrs['test_operator']

        suffix = is_randomized and 'randomized' or 'stats'
        self.start_logfile('rat_%s_auc_%s' % (index_name, suffix))

        def get_rats_with_roc_curves():
            rat_list = unique_rats(data_file.root.scan_cell_info)
            remove = []
            for rat in rat_list:
                try:
                    data_file.getNode(roc_group, self._rat_roc_array_name(rat, index_name))
                except tb.NoSuchNodeError:
                    remove.append(rat)
            [rat_list.remove(rat) for rat in remove]
            return rat_list
        rat_list = get_rats_with_roc_curves()
        N_rats = len(rat_list)
        if N_rats == 0:
            raise ValueError, 'no matching ROC curves for "%s" index' % index_name
        self.out('Found %d per-rat ROC curves for "%s" index and H = %d.' % (
            N_rats, index_name, half_window))

        plt.ioff()
        f = self.new_figure(
                'rat_%s_ROC_curves' % index_name,
                self._make_figure_title(
                    index_name, half_window, scan_cell_condn,
                    scan_phase, event_table, test,
                    randomized=is_randomized, paren='(N=%d)' % N_rats),
                figsize=(11,8))
        ax_roc = f.add_subplot(121)
        ax_auc = f.add_subplot(224)

        colors = plt.cm.Dark2(np.linspace(0, 1, N_rats))
        ndfmt = dict(c='0.4', ls='--', lw=1.5)
        ndline = ax_roc.plot([0, 1], [0, 1], zorder=-1, **ndfmt)
        fmt = dict(lw=2, ls='-') #, drawstyle='steps-mid')

        AUC = np.empty(N_rats, 'd')

        for i, rat in enumerate(rat_list):

            FPR, TPR = data_file.getNode(roc_group, self._rat_roc_array_name(rat, index_name)).read()
            ax_roc.plot(FPR, TPR, label=str(rat), c=colors[i], **fmt)
            AUC[i] = self._compute_AUC(TPR, FPR, test)
            self.out('Rat %d AUC = %f' % (rat, AUC[i]))

        ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel='FPR', ylabel='TPR')
        ax_roc.axis('scaled')

        def plot_sorted_rat_aucs(ax):
            s = np.argsort(AUC)[::-1]
            x = np.arange(1, N_rats+1)
            ax.plot(x, AUC[s], ls='', marker='o', mec='steelblue', mfc='none',
                mew=1.5, ms=6, zorder=10)
            ax.axhline(0.5, **ndfmt)
            ax.tick_params(top=False, right=False)
            ax.set_xticks(x)
            ax.set_xticklabels(map(lambda i: str(rat_list[i]), s), rotation=90)
            ax.set(xlabel='Rats, sorted', ylabel="AUC", xlim=(0, N_rats+1), ylim=ylim)
        plot_sorted_rat_aucs(ax_auc)

        plt.ion()
        plt.show()

        self.out('---')
        self.out('AUC = %f +/- %f' % (AUC.mean(), AUC.std()))
        self.out('T(%d) = %f, p < %e' % ((N_rats-1,) +
            t_one_sample(AUC, 0.5, tails=1)))

        self.close_logfile()
        self.close_data_file()

    def compute_ranksum_statistics(self, randomize=False, condn=None):

        fn = randomize and 'ranksum-randomized' or 'ranksum'
        self.start_logfile(fn)

        data_file = self.get_data_file()
        scan_cell_table = data_file.root.scan_cell_info
        outcome_table = data_file.root.hit_outcome

        h_to_colname = self._find_half_windows(outcome_table)
        h_windows = sorted(h_to_colname.keys())

        for index_name in ScanIndex.ActiveNames:
            scan_index = self._column_data(scan_cell_table, index_name,
                scan_cell_table, condn)

            if randomize:
                scan_index = np.random.permutation(scan_index)

            for h in h_windows:
                outcomes = self._column_data(outcome_table, h_to_colname[h],
                    scan_cell_table, condn)

                negatives = np.logical_not(outcomes)
                positives = outcomes

                U, p = st.mannwhitneyu(
                    scan_index[negatives],
                    scan_index[positives])

                self.out('%s index, half-window %d: U = %d, p < %f' %
                    (snake2title(index_name), h, U, p))

        self.out('Note p-values are one-sided, multiply by 2 for two-sided.')
        self.close_logfile()
        self.close_data_file()


def run_script(decimation=20, rat_decimation=5, ymax=1.0, randomize=False):

    ana_depot = ForwardAnalysis.load_data('/Users/joe/projects/output/forward_analysis/depotentiation/00')
    ana_pot = ForwardAnalysis.load_data('/Users/joe/projects/output/forward_analysis/beta/00/')

    test_condns = ('rat!=0', 'S>0', 'S<0')
    test_ops = ('ge', 'le')

    for ana in (ana_depot, ana_pot):
        for condn in test_condns:
            for test_operator in test_ops:
                ana.generate_roc_curves(condn=condn, test_operator=test_operator,
                    randomize=randomize, decimation=decimation)
                ana.plot_roc_curves(ylim=(0.0, ymax))
                for index in ScanIndex.ActiveNames:
                    ana.generate_rat_roc_curves(index_name=index, condn=condn,
                        test_operator=test_operator, randomize=randomize, decimation=rat_decimation)
                    ana.plot_rat_roc_curves(index_name=index, ylim=(0.0, ymax))
                ana.save_plots_and_close()

