# encoding: utf-8
"""
predictive.py -- Analysis for assessing predictiveness of scan activity for
    subsequent place field modulation.

Created by Joe Monaco on April 17, 2012.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division
from collections import namedtuple

# Library imports
import os
import numpy as np
import cPickle
import tables as tb
import scipy.stats as st
import matplotlib.pyplot as plt
from traits.api import HasTraits, Int, Float

# Package imports
from scanr.behavior import Scan
from scanr.session import SessionData
from scanr.time import elapsed
from scanr.spike import parse_cell_name
from scanr.data import (get_unique_row, unique_values, unique_rats, unique_cells,
    unique_sessions, get_node, get_group, dump_table)

# Local imports
from .core.analysis import AbstractAnalysis
from .core.report import BaseReport
from scanr.tools.plot import AxesList, quicktitle, shaded_region, heatmap
from scanr.tools.misc import memoize
from scanr.tools.stats import (pvalue, CI, t_one_sample, bootstrap_array,
    FDR_control, Interval)
from scanr.tools.radians import xy_to_deg, circle_diff

# Hit window constants
TimeInterval = namedtuple("Interval", "start end")
HitWindow = namedtuple("HitWindow", "start end")
N_RATE_BOOTSTRAPS = 1000
N_SCAN_SHUFFLES = 5000
DEFAULT_HIT_LAP = -1
DEFAULT_MIN_BUFFER = 30
DEFAULT_MAX_BUFFER = 10

# Predictive value results types
DataSlice = namedtuple("DataSlice", "domain field comparator comparanda")
Results = namedtuple("Results", "data ppv expected lower upper std")
ShiftedResults = namedtuple("ShiftedResults", "expt dt ppv observed expected "
    "lower upper")
CrossCorrelation = namedtuple("CrossCorrelation", "dlim bins observed expected "
    "shuffled CI event_slice scan_slice")

# Place-field event table structure
EventDescr = {    'id'              :  tb.UInt16Col(pos=1),
                  'rat'             :  tb.UInt16Col(pos=3),
                  'day'             :  tb.UInt16Col(pos=4),
                  'session'         :  tb.UInt16Col(pos=5),
                  'tc'              :  tb.StringCol(itemsize=8, pos=6),
                  'type'            :  tb.StringCol(itemsize=4, pos=7),
                  'expt_type'       :  tb.StringCol(itemsize=4, pos=8),
                  'area'            :  tb.StringCol(itemsize=4, pos=9),
                  'subdiv'          :  tb.StringCol(itemsize=4, pos=10),
                  'laps'            :  tb.FloatCol(pos=11),
                  'angle_start'     :  tb.FloatCol(pos=12),
                  'angle_COM'       :  tb.FloatCol(pos=13),
                  'angle_end'       :  tb.FloatCol(pos=14)  }

# Scan event table structure; this is extended dynamically to contain fields
# for each defined scan phase
ScanDescr = {     'id'              :  tb.UInt32Col(pos=1),
                  'scan_id'         :  tb.UInt32Col(pos=2),
                  'event_id'        :  tb.UInt16Col(pos=3),
                  'angle'           :  tb.FloatCol(pos=4),
                  'duration'        :  tb.FloatCol(pos=5),
                  'magnitude'       :  tb.FloatCol(pos=6) }

# Hit window table structure; track angles and timestamp limits
HitWindowDescr = {'event_id'        :  tb.UInt16Col(pos=1),
                  'angle_start'     :  tb.FloatCol(pos=2),
                  'angle_end'       :  tb.FloatCol(pos=3),
                  'width'           :  tb.FloatCol(pos=4),
                  'tracklim'        :  tb.FloatCol(shape=(2,), pos=5),
                  'start'           :  tb.UInt64Col(pos=6),
                  'end'             :  tb.UInt64Col(pos=7),
                  'duration'        :  tb.FloatCol(pos=8),
                  'tlim'            :  tb.UInt64Col(shape=(2,), pos=9),
                  'field_spikes'    :  tb.UInt16Col(pos=10),
                  'field_duration'  :  tb.FloatCol(pos=11),
                  'rate_CI_lower'   :  tb.FloatCol(pos=12),
                  'rate_CI_upper'   :  tb.FloatCol(pos=13)  }

# Table structure for describing predictive hit-testing results across scans
# that is dynamically extended with phase_pct and phase_pass columns for
# all scan phases (*_pct are firing-rate percentiles, and *_pass are boolean
# flags indicating successful scan-activation prediction)
HitScanDescr = {  'event_id'        :  tb.UInt16Col(pos=1),
                  'scan_id'         :  tb.UInt32Col(pos=2),
                  'in_window'       :  tb.BoolCol(pos=3)    }

# Post-hoc predictive scan tables
PredictiveScanDescr = { 'id'        :  tb.UInt16Col(pos=1),
                        'event_id'  :  tb.UInt16Col(pos=2),
                        'scan_id'   :  tb.UInt16Col(pos=3)  }

# Post-hoc predictive event table
PredictedEventDescr = { 'event_id'  :  tb.UInt16Col(pos=1),
                        'rat'       :  tb.UInt16Col(pos=2),
                        'day'       :  tb.UInt16Col(pos=3),
                        'session'   :  tb.UInt16Col(pos=4),
                        'expt_type' :  tb.StringCol(itemsize=4, pos=5),
                        'tc'        :  tb.StringCol(itemsize=8, pos=6),
                        'event_lap' :  tb.UInt16Col(pos=7),
                        'hit_window':  tb.FloatCol(shape=(2,), pos=8),
                        'predicted' :  tb.StringCol(itemsize=1, pos=9),
                        'num_scans' :  tb.UInt16Col(pos=10) }


def session_start_angle(event_rec):
    """Get the track-angle position of the initial tracking location for the
    given session.
    """
    rds = tuple(map(lambda k: int(event_rec[k]), ['rat', 'day', 'session']))
    grp = get_group(rds=rds)
    return xy_to_deg(grp.x[0], grp.y[0])


class AdaptiveHitWindow(HasTraits):

    """
    Track-angle window for detecting field event-associated scans that adapts
    based on the size and COM of the field event
    """

    hit_lap = Int(DEFAULT_HIT_LAP)
    min_buffer = Float(DEFAULT_MIN_BUFFER)
    max_buffer = Float(DEFAULT_MAX_BUFFER)

    def for_event(self, event):
        """Dual-mode ranging for scan hits for the given place-field
        event record
        """
        start, end = event['angle_start'], event['angle_end']
        COM = event['angle_COM']
        min_window = (  COM - 360 * self.hit_lap + self.min_buffer,
                        COM - 360 * self.hit_lap - self.min_buffer   )
        max_window = (  start - 360 * self.hit_lap + self.max_buffer,
                        end   - 360 * self.hit_lap - self.max_buffer  )
        window = HitWindow( max(min_window[0], max_window[0]),
                            min(min_window[1], max_window[1])   )
        return window

    def plot(self, start=None, end=None, COM=None, **kwds):
        """Pass in arrays for event angle start, end, and COM, to plot
        resulting adaptive hit windows
        """
        if None in (start, end, COM):
            start, end, COM = (
                np.linspace(-720, -660, 100),
                np.linspace(-720, -780, 100),
                np.linspace(-720, -690, 100) )

        N = start.size
        W = np.empty((N, 2), 'd')
        for i in xrange(N):
            test_event = dict(angle_start=start[i], angle_end=end[i],
                angle_COM=COM[i])
            W[i] = self.for_event(test_event, **kwds)

        plt.ioff()
        f = plt.figure(150)
        plt.clf()
        f.suptitle('Testing the Adaptive Hit-Window (lap=%d, min=%d, max=%d)'%(
            kwds.get('lap', self.hit_lap),
            kwds.get('min_buffer', self.min_buffer),
            kwds.get('max_buffer', self.max_buffer)))

        ax = f.add_subplot(111)
        ax.plot(start, 'g-', label='start')
        ax.plot(end, 'r-', label='end')
        ax.plot(COM, 'b--', label='COM')
        ax.plot(W[:,0], 'k-', label='low')
        ax.plot(W[:,1], 'k--', label='high')
        ax.set_ylabel('Unwrapped Track Angle')
        ax.set_xlabel('Field Size Tests')
        plt.ion()
        plt.show()


class PredictiveValueAnalysis(AbstractAnalysis):

    """
    Compute unwrapped track-angle relationship between scanning activity and
    detected field modulation events

    Empirical null (expected) distributions are created by resampling pooled
    distributions of scan times and simulating the corresponding spike counts
    """

    label = "predictive analysis"

    @memoize
    def event_counts_for_rats(self):
        """Get a mapping of rats to number of potentiation events
        """
        ptable = get_node('/physiology', self.results['table'])
        rats = unique_rats(ptable)
        return { rat: len(ptable.getWhereList('rat==%d'%rat)) for rat in rats }

    def collect_data(self, table='potentiation', pool_factor=10,
        max_offset=10, truncate_scan_pool=False, zero_overlap=True):
        """Collect unwrapped track-angle data for scan trains and field
        potentiation events

        table -- name of place-field event table under /physiology containing
            the events for analysis
        pool_factor -- sampling factor for shuffled scans
        max_offset -- time half-window in seconds for randomly sampling new
            shuffled scan times
        truncate_scan_pool -- exclude scans past the end of the event session
            from the randomized scan pool
        zero_overlap -- whether time-shifting windows at dt=0 start by
            overlapping the observed scan timing
        """
        scan_table = get_node('/behavior', 'scans')
        session_table = get_node('/metadata', 'sessions')
        tetrode_table = get_node('/metadata', 'tetrodes')

        self.results['table'] = table
        self.results['pool_factor'] = pool_factor
        self.results['max_offset'] = max_offset
        self.results['t_shifts'] = \
            t_shifts = np.linspace(0, max_offset, pool_factor + 1)

        # Extend scan table with scan-phase spike-count fields
        scan_descr = ScanDescr.copy()
        for phase in Scan.PhaseNames:
            scan_descr[phase + '_count'] = tb.UInt8Col(pos=len(scan_descr))
            scan_descr[phase + '_dt'] = tb.FloatCol(pos=len(scan_descr))

        # Create tables and data groups for observed and shuffled data
        data_file = self.open_data_file()
        event_table = data_file.createTable('/', 'events', EventDescr,
            title='Place-Field Change Events')
        event_row = event_table.row

        observed_table = data_file.createTable('/', 'observed_scans', scan_descr,
            title='Observed Head Scan Events')
        scan_row = observed_table.row

        shuffled_table = data_file.createTable('/', 'shuffled_scans', scan_descr,
            title='Random-Shuffled Scans Across Events')
        shuffled_row = shuffled_table.row

        scan_descr['direction'] = tb.IntCol(pos=len(scan_descr))
        scan_descr['parameter'] = tb.FloatCol(pos=len(scan_descr))
        shifted_table = data_file.createTable('/', 'shifted_scans', scan_descr,
            title='Time-Shifted Scans Across Events')
        shifted_row = shifted_table.row

        # Pre-cache all the session data
        field_table = get_node('/physiology', table)
        for rds in unique_sessions(field_table):
            SessionData.get(rds, load_clusters=False)

        i_event = 0
        i_scans = 0
        for rds, tc in unique_cells(field_table):

            # Session and cluster data for this place-field event
            session_data = SessionData.get(rds, load_clusters=False)
            F_alpha = session_data.F_('alpha_unwrapped')
            attrs = get_unique_row(session_table, session_data.session_query)
            cluster = session_data.cluster_data(tc)
            tt, cl = parse_cell_name(tc)
            tetrode = get_unique_row(tetrode_table, '(rat==%d)&(day==%d)&(tt==%d)'%(
                rds[0], rds[1], tt))

            # Get initial activity and scan info from data table
            cell_query = session_data.session_query + '&(tc=="%s")'%tc
            for event in field_table.where(cell_query):
                self.out.printf('.')
                event_row['id'] = event['id']
                event_row['rat'], event_row['day'], event_row['session'] = rds
                event_row['tc'] = tc
                event_row['type'] = attrs['type']
                event_row['expt_type'] = attrs['expt_type']
                event_row['area'] = tetrode['area']
                event_row['subdiv'] = tetrode['area'] + tetrode['subdiv'][:1]
                session_start = session_start_angle(event)
                tracklim = F_alpha(session_data.T_(event['tlim']))
                event_row['laps'] = \
                    1.0 + (session_start - tracklim[0]) / 360
                event_row['angle_start'] = tracklim[0]
                event_row['angle_COM'] = (
                    circle_diff(event['COM'], event_row['angle_start'],
                        degrees=True)
                    + event_row['angle_start'])
                event_row['angle_end'] = tracklim[1]
                event_row.append()

                for scan in scan_table.where(session_data.session_query):
                    scan_row['id'] = i_scans
                    scan_row['event_id'] = event['id']
                    scan_row['scan_id'] = scan['id']
                    scan_row['duration'] = scan['duration']
                    scan_row['magnitude'] = scan['magnitude']
                    scan_row['angle'] = F_alpha(session_data.T_(scan['start']))

                    for phase in Scan.PhaseNames:
                        start, end = Scan.PhasePoints[phase]
                        scan_row[phase + '_count'] = np.sum(np.logical_and(
                            cluster.spikes >= scan[start],
                            cluster.spikes <= scan[end]))
                        scan_row[phase + '_dt'] = elapsed(scan[start], scan[end])

                    scan_row.append()
                    i_scans += 1

                i_event += 1
                if i_event % 10 == 0:
                    event_table.flush()
                    observed_table.flush()
                    self.out.printf('|', color='cyan')

        self.out.printf('\n')

        data_file.flush()
        self.results['N_events'] = N_events = i_event
        self.out('Collected %d field events from %s.'%(N_events,
            field_table._v_pathname))

        # Shuffled scan-timing data for each observed place-field event
        self.out('Shuffling scans for null distributions...')
        i_shuffled_scans = 0
        i_shifted_scans = 0

        def get_scan_timing(session_start, scan_record):
            t_scan = { k: elapsed(session_start, scan_record[k])
                        for k in Scan.PointNames }
            dt = { p: t_scan[Scan.PhasePoints[p][1]] -
                      t_scan[Scan.PhasePoints[p][0]]
                        for p in Scan.PhaseNames }
            return t_scan, dt

        def random_offsets():
            return 2 * max_offset * (np.random.rand(pool_factor) - 0.5)

        for event in event_table.iterrows():
            self.out.printf('.', color='green')

            rds = event['rat'], event['day'], event['session']
            dataset_query = '(rat==%d)&(day==%d)'%rds[:2]
            tc = event['tc']

            # Load session data and null-hypothesis-eligible spike times
            session_data = SessionData.get(rds, load_clusters=False)
            cluster = session_data.cluster_data(tc)
            t_spikes = session_data.T_(cluster.spikes) # use all spikes
            N_scans = len(scan_table.getWhereList(session_data.session_query))
            if not N_scans:
                continue

            # Accumulate pool of all scan information in all *other sessions*
            # of the *same dataset* for random resampling
            t_session = { rec['session']: rec['start']
                for rec in session_table.where(dataset_query) }
            F_alpha = session_data.F_('alpha_unwrapped')
            scan_query = '%s&(session!=%d)'%(dataset_query, rds[2])

            for scan in scan_table.where(scan_query):
                t_scan, dt = get_scan_timing(t_session[scan['session']], scan)
                t_pool = t_scan['start'] + random_offsets()

                for t_scan_start in t_pool:
                    # Exclude random scans past the end of the event session
                    if truncate_scan_pool and (
                        t_scan_start + dt['scan'] > session_data.duration):
                        continue

                    shuffled_row['id'] = i_shuffled_scans
                    shuffled_row['event_id'] = event['id']
                    shuffled_row['scan_id'] = scan['id']
                    shuffled_row['angle'] = F_alpha(t_scan_start)
                    shuffled_row['duration'] = dt['scan']
                    shuffled_row['magnitude'] = scan['magnitude']

                    # Sample spikes from scan-start-centered interval
                    for phase in Scan.PhaseNames:
                        shuffled_row[phase + '_count'] = np.sum(np.logical_and(
                            t_spikes <= t_scan_start + dt[phase] / 2,
                            t_spikes >= t_scan_start - dt[phase] / 2))
                        shuffled_row[phase + '_dt'] = dt[phase]

                    shuffled_row.append()
                    i_shuffled_scans += 1
            shuffled_table.flush()

            # Construct the time-shifted scans from the *observed session*
            for scan in scan_table.where(session_data.session_query):
                t_scan, dt = get_scan_timing(session_data.start, scan)

                if zero_overlap:
                    neg_start = { phase: t_scan[Scan.PhasePoints[phase][1]] for phase in Scan.PhaseNames }
                    pos_start = { phase: t_scan[Scan.PhasePoints[phase][0]] for phase in Scan.PhaseNames }
                else:
                    neg_start = { phase: t_scan[Scan.PhasePoints[phase][0]] for phase in Scan.PhaseNames }
                    pos_start = { phase: t_scan[Scan.PhasePoints[phase][1]] for phase in Scan.PhaseNames }

                for t_scan_shift in t_shifts:
                    # Negatively time-shifted scanning data
                    shifted_row['id'] = i_shifted_scans
                    shifted_row['event_id'] = event['id']
                    shifted_row['scan_id'] = scan['id']
                    shifted_row['angle'] = F_alpha(neg_start['postfix'] - t_scan_shift
                                            - (t_scan['postfix'] - t_scan['start']))
                    shifted_row['duration'] = dt['scan']
                    shifted_row['magnitude'] = scan['magnitude']
                    shifted_row['direction'] = -1
                    shifted_row['parameter'] = -t_scan_shift

                    for phase in Scan.PhaseNames:
                        shifted_row[phase + '_count'] = np.sum(np.logical_and(
                            t_spikes <= neg_start[phase] - t_scan_shift,
                            t_spikes >= neg_start[phase] - t_scan_shift - dt[phase]))
                        shifted_row[phase + '_dt'] = dt[phase]

                    shifted_row.append()
                    i_shifted_scans += 1

                    # Positively time-shifted scanning data
                    shifted_row['id'] = i_shifted_scans
                    shifted_row['event_id'] = event['id']
                    shifted_row['scan_id'] = scan['id']
                    shifted_row['angle'] = F_alpha(pos_start['prefix'] + t_scan_shift + dt['prefix'])
                    shifted_row['duration'] = dt['scan']
                    shifted_row['magnitude'] = scan['magnitude']
                    shifted_row['direction'] = +1
                    shifted_row['parameter'] = t_scan_shift

                    for phase in Scan.PhaseNames:
                        shifted_row[phase + '_count'] = np.sum(np.logical_and(
                            t_spikes <= pos_start[phase] + t_scan_shift + dt[phase],
                            t_spikes >= pos_start[phase] + t_scan_shift))
                        shifted_row[phase + '_dt'] = dt[phase]

                    shifted_row.append()
                    i_shifted_scans += 1
            shifted_table.flush()

        self.out.printf('\nDone!\n', color='white')
        data_file.flush()

    def process_data(self, time_bin_us=2e4, **window_kwds):
        """Create an intermediary data structure with the results of predictive
        hit-window and hit-interval testing across all scans, including
        observed, shuffled, and shifted.

        Keyword arguments are passed to AdaptiveHitWindow.
        """
        self.out.outfd = file(os.path.join(self.datadir, 'hit_processing.log'), 'w')
        hit_window = AdaptiveHitWindow(**window_kwds)

        # Load event and scan data
        data_file = self.get_data_file(mode='a')
        root = data_file.root

        # Create new groups for predictive test data
        try:
            predictive_group = data_file.getNode('/', 'predictive')
        except tb.NoSuchNodeError:
            pass
        else:
            data_file.removeNode(predictive_group, recursive=True)
        finally:
            predictive_group = data_file.createGroup('/', 'predictive',
                title='Predictive Hit-Testing Results Across Scans')
            predictive_group._v_attrs['lap_name'] = \
                self.predictive_group_name(hit_window.hit_lap)

        # Extend scan table with scan-phase fields
        scan_descr = HitScanDescr.copy()
        for phase in Scan.PhaseNames:
            scan_descr[phase + '_pct'] = tb.FloatCol(pos=len(scan_descr))
            scan_descr[phase + '_pass'] = tb.BoolCol(pos=len(scan_descr))

        # Create tables for different types of scans
        HitTests = namedtuple("HitTests", "windows observed shuffled shifted")
        tests = HitTests(
            data_file.createTable(predictive_group, 'windows', HitWindowDescr,
                title='Place-field Event Hit Windows'),
            data_file.createTable(predictive_group, 'observed', scan_descr,
                title='Hit-Test Data for Observed Scans'),
            data_file.createTable(predictive_group, 'shuffled', scan_descr,
                title='Hit-Test Data for Shuffled Scans'),
            data_file.createTable(predictive_group, 'shifted', scan_descr,
                title='Hit-Test Data for Shifted Scans'),
        )

        # Pre-cache all the session data
        for rds in unique_sessions(root.events):
            SessionData.get(rds, load_clusters=False)

        table_pairs = ( (root.observed_scans, tests.observed),
                        (root.shuffled_scans, tests.shuffled),
                        (root.shifted_scans, tests.shifted) )

        for event in root.events.iterrows():

            # Get the track-angle hit window and save in the event window table
            event_window = hit_window.for_event(event)
            tests.windows.row['event_id'] = event['id']
            tests.windows.row['angle_start'] = event_window.start
            tests.windows.row['angle_end'] = event_window.end
            tests.windows.row['width'] = event_window.start - event_window.end
            tests.windows.row['tracklim'] = event_window

            rds = (event['rat'], event['day'], event['session'])
            session_data = SessionData.get(rds, load_clusters=False)

            # Find the innermost temporal bounds of the event-window traversal
            # to use as the temporal "hit interval"
            traj = session_data.trajectory
            alpha = traj.alpha_unwrapped
            after_start = (alpha <= event_window.start).astype('i')
            before_end = (alpha >= event_window.end).astype('i')

            try:
                start_ix = ((after_start[1:] - after_start[:-1]) == 1).nonzero()[0][-1]
                end_ix = 1 + start_ix + \
                    ((before_end[start_ix+1:] - before_end[start_ix:-1]) == -1).nonzero()[0][0]

            except IndexError:
                tests.windows.row['start'] = 0
                tests.windows.row['end'] = 0
                tests.windows.row['duration'] = 0.0
                tests.windows.row['tlim'] = (0,0)
                tests.windows.row['field_spikes'] = 0
                tests.windows.row['field_duration'] = 0.0
                tests.windows.row['rate_CI_lower'] = 0.0
                tests.windows.row['rate_CI_upper'] = 0.0
                self.out('Event %04d: hit window out of session limits'%event['id'])

            else:
                hit_interval = HitWindow(traj.ts[start_ix], traj.ts[end_ix])
                hit_dt = elapsed(*hit_interval)

                tests.windows.row['start'] = hit_interval.start
                tests.windows.row['end'] = hit_interval.end
                tests.windows.row['duration'] = hit_dt
                tests.windows.row['tlim'] = hit_interval

                # Set up time bins for computing expected firing rate distributions
                time_bin_edges = np.arange(
                    hit_interval.start, hit_interval.end + time_bin_us, time_bin_us)
                time_bin_filter = session_data.filter_tracking_data(time_bin_edges,
                    velocity_filter=True, exclude_off_track=True,
                    exclude=session_data.extended_scan_and_pause_list,
                    boolean_index=True)

                # Count spikes across filtered time bins
                spike_bins = []
                N_time_bins = 0
                cluster = session_data.cluster_data(event['tc'])
                for i, ts_start in enumerate(time_bin_edges[:-1]):
                    if np.all(time_bin_filter[i:i+2]):
                        bin = TimeInterval(time_bin_edges[i], time_bin_edges[i+1])
                        N_time_bins += 1
                        spike_bins.append(np.sum(np.logical_and(
                            cluster.spikes >= bin.start,
                            cluster.spikes < bin.end)))
                spike_bins = np.array(spike_bins)

                # Bootstrap spike counts and get firing-rate confidence intervals
                field_duration = (time_bin_us * 1e-6) * N_time_bins
                if field_duration:
                    F = lambda x: np.sum(x) / field_duration
                    rate_bootstraps = bootstrap_array(F, spike_bins,
                        bootstraps=N_RATE_BOOTSTRAPS)
                else:
                    rate_bootstraps = np.zeros(N_RATE_BOOTSTRAPS)
                rate_CI = CI(rate_bootstraps, alpha=0.05)

                tests.windows.row['field_spikes'] = spike_bins.sum()
                tests.windows.row['field_duration'] = field_duration
                tests.windows.row['rate_CI_lower'] = rate_CI.lower
                tests.windows.row['rate_CI_upper'] = rate_CI.upper

                self.out('Event %04d: found %d spikes / %.3f seconds, CI = %s'%(
                    event['id'], spike_bins.sum(), field_duration, rate_CI))

            tests.windows.row.append()

            for data_table, test_table in table_pairs:
                self.out('Processing %s -> %s...'%(data_table._v_pathname,
                    test_table._v_pathname))

                for scan in data_table.where('event_id==%d'%event['id']):
                    test_table.row['event_id'] = scan['event_id']
                    test_table.row['scan_id'] = scan['id']

                    in_window = np.logical_and(
                        scan['angle'] <= event_window.start,
                        scan['angle'] >= event_window.end)
                    test_table.row['in_window'] = in_window
                    if not in_window:
                        for phase in Scan.PhaseNames:
                            test_table.row[phase + '_pct'] = 0.0
                            test_table.row[phase + '_pass'] = False
                        test_table.row.append()
                        self.out.printf('.', color='red')
                        continue

                    for phase in Scan.PhaseNames:
                        k = scan[phase + '_count']
                        dt = scan[phase + '_dt']

                        test_pct = 0.0
                        if dt:
                            test_pct = np.sum(rate_bootstraps < (k / dt)
                                ) / N_RATE_BOOTSTRAPS
                        test_table.row[phase + '_pct'] = test_pct

                        test_pass = k > (rate_CI.upper * dt)
                        test_table.row[phase + '_pass'] = test_pass

                    test_table.row.append()

                    self.out.printf('.', color='green')
                self.out.printf('\n')
                data_file.flush()

            data_file.flush()

        self.close_data_file()
        self.out.outfd.close()

    def predictive_values_all_phases(self, **kwds):
        """Run predictive-values analysis for every scan phase

        Keyword arguments are passed to predictive_values().
        """
        for phase in Scan.PhaseNames:
            kwds.update(phase=phase)
            self.predictive_values(**kwds)

    def predictive_values(self, phase='scan', activity_pass_pct=None,
        N_shuffles=N_SCAN_SHUFFLES, which_panels='main', expt=None):
        """Compute predictive values across 2D sweeps of spike-count threshold
        and track-angle window size for the experiment x event with marginals.

        Keyword arguments:
        phase -- scan phase upon which predictive values shall be computed
        activity_pass_pct -- specifiy activity-percentile criterion for a scan
            to be counted as a hit (default upper 95pct confidence interval
            bound ~ 97.5pct)
        N_shuffles -- number of random shuffles of null scan data to use to
            compute expected predictive value and confidence intervals
        which_panels -- keyword specifying the set of marginal panels to compute
        expt -- set to 'DR' or 'NOV' to restrict data to one experiment type
        """
        self.out.outfd = file(os.path.join(self.datadir, 'predictive.log'), 'w')
        root = self.get_data_file().root

        if which_panels == 'main':
            # Main figure panels
            data_slices = [
                DataSlice("event", "type", '==', ('"STD"', '"MIS"', '"FAM"', '"NOV"', '"FIRST"')),
                DataSlice("event", "rat", '==', map(str, unique_rats(root.events))),
                DataSlice("event", "area", '==', ('"CA1"', '"CA3"')),
                DataSlice("event", "day", '==', map(str, tuple(range(1,9)))),
            ]
        elif which_panels == 'suppl':
            # Supplementary figure panels
            data_slices = [
                DataSlice("event", "session", '==', map(str, (1,2,3,4,5))),
                DataSlice("event", "laps", '>=', map(str, np.arange(2.0, 12.0, 2.0))),
                DataSlice("scan", "magnitude", '>=', map(str, np.arange(2.5, 10.0, 2.5))),
                DataSlice("scan", "duration", '>=', map(str, np.arange(0.5, 6.0, 1.0)))
            ]
        elif which_panels == 'sessions':
            # Just the session numbers
            data_slices = [
                DataSlice("event", "session", '==', map(str, (1,2,3,4,5)))
            ]
        elif which_panels == 'days':
            # Just the testing days
            data_slices = [
                DataSlice("event", "day", '==', map(str, tuple(range(1,9))))
            ]
        elif which_panels == 'rats':
            # Just the rats
            data_slices = [
                DataSlice("event", "rat", '==', map(str, unique_rats(root.events)))
            ]
        elif which_panels == 'subdiv':
            # Analysis across recording area subdivisions
            data_slices = [
                DataSlice("event", "subdiv", '==', ('"CA3c"', '"CA3b"', '"CA3a"', '"CA1p"', '"CA1i"', '"CA1d"'))
            ]
        elif which_panels in ('DR', 'NOV', 'FIRST'):
            # Single-experiment slices to support across-lap processing
            data_slices = [
                DataSlice("event", "expt_type", '==', ('"%s"'%which_panels,))
            ]
        elif which_panels == 'ALL':
            # Single slice for all experiments to support across-lap processing
            data_slices = [
                DataSlice("event", "rat", "!=", (0,))
            ]

        self.out('Computing predictive values for %s scan phase'%phase)
        results = []

        # Load predictive scan data into dictionaries for fast access
        if activity_pass_pct is None:
            phase_pass = phase + '_pass'
            observed_scan_pass = {
                rec['scan_id']: rec[phase_pass]
                    for rec in root.predictive.observed.iterrows() }
            shuffled_scan_pass = {
                rec['scan_id']: rec[phase_pass]
                    for rec in root.predictive.shuffled.iterrows() }
        else:
            phase_pct = phase + '_pct'
            observed_scan_pass = {
                rec['scan_id']: rec[phase_pct] >= activity_pass_pct
                    for rec in root.predictive.observed.iterrows() }
            shuffled_scan_pass = {
                rec['scan_id']: rec[phase_pct] >= activity_pass_pct
                    for rec in root.predictive.shuffled.iterrows() }

        for D in data_slices:
            self.out('Processing %s/%s...'%(D.domain, D.field))

            N_comparanda = len(D.comparanda)
            observed_ppv, expected_ppv, lower_ppv, upper_ppv, std_ppv = \
                tuple(np.empty(N_comparanda, 'd') for i in xrange(5))

            for i_comparand, comparand in enumerate(D.comparanda):

                # A little hack to get the initial presentations of novel room
                # configuration in novelty experiments:
                if comparand == '"FIRST"':
                    field_query = '(type=="NOV")&(day==1)'
                else:
                    field_query = '%s%s%s'%(D.field, D.comparator, comparand)

                if D.domain == 'event':
                    if expt is not None:
                        if expt in ('DR', 'NOV'):
                            field_query = '(%s)&(expt_type=="%s")'%(field_query, expt)
                        else:
                            raise ValueError, 'expt must be \'DR\' or \'NOV\''
                    event_ids = np.array([rec['id'] for
                        rec in root.events.where(field_query)])
                else:
                    event_ids = root.events.col('id')

                N_events = event_ids.size
                self.out('Found %d events for %s %s %s'%(
                    N_events, D.field, D.comparator, comparand))

                event_shuffle_pass = np.empty((N_events, N_shuffles), '?')

                N_hits = 0
                for i_event, event_id in enumerate(event_ids):
                    self.out.printf('.')

                    event_id_query = '(event_id==%d)'%event_id
                    if D.domain == 'scan':
                        scan_field_query = '&(%s)'%field_query
                    else:
                        scan_field_query = ''
                    event_scan_query = event_id_query + scan_field_query
                    scan_ids = np.array([rec['id'] for rec in
                        root.observed_scans.where(event_scan_query)])

                    # Find observed hit
                    for scan_id in scan_ids:
                        if observed_scan_pass[scan_id]:
                            N_hits += 1
                            break

                    # Find shuffled distribution of hits
                    N_scans = scan_ids.size
                    shuffled_scan_ids = np.array([rec['id']
                        for rec in root.shuffled_scans.where(event_scan_query)])
                    shuffled = np.array([shuffled_scan_pass[scan_id]
                        for scan_id in shuffled_scan_ids])
                    shuffle_matrix_ix = np.array(
                        [np.random.permutation(shuffled.size)[:N_scans]
                            for i in xrange(N_shuffles)]).T
                    event_shuffle_pass[i_event] = np.any(
                        shuffled[shuffle_matrix_ix], axis=0)

                self.out.printf('\n')

                if N_events:
                    observed_ppv[i_comparand] = ppv = N_hits / N_events
                    shuffled_ppvs = event_shuffle_pass.sum(axis=0) / N_events
                    self.out('Found %d hits for PPV = %.3f'%(N_hits, ppv))
                else:
                    observed_ppv[i_comparand] = ppv = 0.0
                    shuffled_ppvs = np.zeros(event_shuffle_pass.shape[1], '?')
                    self.out('Warning: No events found for %s'%str(D), error=True)

                shuffled_CI = CI(shuffled_ppvs)
                lower_ppv[i_comparand] = shuffled_CI.lower
                upper_ppv[i_comparand] = shuffled_CI.upper
                expected_ppv[i_comparand] = E_ppv = shuffled_ppvs.mean() # or median?
                std_ppv[i_comparand] = shuffled_ppvs.std()
                self.out('Found expected = %f, CI = %s'%(E_ppv, shuffled_CI))

            results.append(
                Results(D, observed_ppv, expected_ppv, lower_ppv, upper_ppv, std_ppv))

        # Save predictive value data for reporting (see predictive_report())
        if expt is None:
            fn = 'ppv_%s_%s.pickle'%(phase, which_panels)
        else:
            fn = 'ppv_%s_%s_%s.pickle'%(phase, which_panels, expt)
        results_fd = file(os.path.join(self.datadir, fn), 'w')
        cPickle.dump(results, results_fd)
        results_fd.close()

        self.out.outfd.close()
        self.close_data_file()

        if which_panels not in ('main', 'suppl', 'subdiv'):
            return results

    def predictive_reports_all_phases(self, suffix=None, **kwds):
        """Run predictive-values analysis for every scan phase

        Keyword arguments are passed to predictive_values().
        """
        files = []
        for phase in Scan.PhaseNames:
            kwds.update(phase=phase)
            files.append(self.predictive_reports(**kwds))
        if suffix is not None and not suffix.startswith('_'):
            suffix = '_' + suffix
        else:
            suffix = ''
        os.system('pdftk %s cat output %s'%(' '.join(files),
            os.path.join(self.datadir, 'ppv_reports_all_phases%s.pdf'%suffix)))

    def predictive_reports(self, phase='scan', main_ymax=0.45, norm_ymax=3.0, rat_ymax=0.45,
        which_panels='main', hide_rats_less_than_num_events=10, expt=None):
        """Generate a report of previously saved predictive-value results for
        the given scan phase
        """
        if expt is None:
            stem = 'ppv_%s_%s'%(phase, which_panels)
        else:
            stem = 'ppv_%s_%s_%s'%(phase, which_panels, expt)
        fn = '%s.pickle'%stem
        results = np.load(os.path.join(self.datadir, fn))
        norm_title = ''
        panels_title = ''
        if which_panels != 'main':
            panels_title += '_%s'%which_panels
        if expt is not None:
            panels_title += '_%s'%expt

        rat_N = self.event_counts_for_rats()
        fixed_xmax = 12.5

        class PredictiveReport(BaseReport):
            xnorm = False
            ynorm = False
            figwidth = 5
            figheight = 10
            nrows = 4
            ncols = 1

            def collect_data(self, null_norm=False, do_effect_size=False):
                for res, ax in self.get_plot(results):
                    PPV = res.ppv
                    E_PPV = res.expected
                    E_INT = np.vstack((
                        res.expected - res.lower, res.upper - res.expected))

                    if res.data.field == 'rat':
                        s_all = np.argsort(PPV)[::-1]
                        s = np.array([ix for ix in s_all
                            if rat_N[int(res.data.comparanda[ix])] >= hide_rats_less_than_num_events])
                        N = len(s)
                    else:
                        N = len(res.data.comparanda)
                        s = slice(None)

                    x = np.arange(N)

                    if do_effect_size:
                        PPV_Delta = (PPV[s] - E_PPV[s]) / res.std[s]
                        plt.rcParams['lines.linewidth'] = 1.5
                        ax.stem(x, PPV_Delta, markerfmt='bo')[0].set(mec='b', mfc='w', mew=1.5,
                            marker='o', ms=6, zorder=10)
                        plt.rcParams['lines.linewidth'] = plt.rcParamsDefault['lines.linewidth']
                    else:
                        if null_norm:
                            PPV = PPV / res.expected
                            E_PPV = np.ones_like(res.expected)
                            E_INT = E_INT / res.expected
                        ax.plot(x, PPV[s], 'ro', mfc='none', mew=1.5, mec='r', ms=6,
                            zorder=10)
                        ax.errorbar(x, E_PPV[s], yerr=E_INT[:,s],
                            fmt='_', lw=1.0, ecolor='k', capsize=4,
                            mew=1.0, mec='k', ms=6)
                    ax.tick_params(direction='out')
                    ax.tick_params(top=False, right=False, labelsize='small')
                    ax.set_xticks(x)
                    if N >= 10:
                        ax.set_xticklabels(np.array(res.data.comparanda)[s],
                            rotation=90)
                    elif type(res.data.comparanda[0]) is str:
                        ax.set_xticklabels(
                            map(lambda s: s.replace('"', ''), np.array(res.data.comparanda)[s]),
                            rotation=90)
                    else:
                        ax.set_xticklabels(np.array(res.data.comparanda)[s])
                    if res.data.field == 'rat':
                        ax.set_xlim(-0.5, max(N-0.5, fixed_xmax))
                    else:
                        ax.set_xlim(-0.5, fixed_xmax)
                    if null_norm:
                        ymax = norm_ymax
                    elif res.data.field == 'rat':
                        ymax = rat_ymax
                    else:
                        ymax = main_ymax
                    ax.set_ylim(0.0, ymax)
                    title = '%s %s'%(res.data.domain, res.data.field)
                    if res.data.comparator != "==":
                        title += ' (%s)'%res.data.comparator
                    quicktitle(ax, title)
                    if self.firstonpage or res.data.field == 'rat':
                        ax.set_ylabel('PPV')
                    else:
                        ax.set_yticklabels([])

        # Generate absolute, normalized, and effect-size reports
        reportsdir = os.path.join(self.datadir, 'reports')
        if not os.path.isdir(reportsdir):
            os.makedirs(reportsdir)

        report = PredictiveReport(label=stem, datadir=reportsdir)
        report(do_effect_size=False, null_norm=False)

        report = PredictiveReport(label='%s_norm'%stem, datadir=reportsdir)
        report(do_effect_size=False, null_norm=True)

        report = PredictiveReport(label='%s_delta'%stem, datadir=reportsdir)
        report(do_effect_size=True, null_norm=False)

    def shifted_predictive_values(self, phase='scan', expt_type='ALL',
        N_shuffles=N_SCAN_SHUFFLES, test_event_num=None):
        """Compute observed and expected predictive values for the time-shifted
        scan data

        A data slice can be specified to optionally restrict the event or
        scan data going toward the predictive value calculation
        """
        self.out.outfd = file(os.path.join(self.datadir,
            'predictive_shifted.log'), 'w')
        root = self.get_data_file().root

        if expt_type != 'ALL':
            event_ids = np.array([rec['id'] for rec in
                root.events.where('expt_type=="%s"'%expt_type)])
        else:
            event_ids = root.events.col('id')

        if test_event_num is not None:
            event_ids = event_ids[:test_event_num]

        N_events = event_ids.size

        phase_pass = phase + '_pass'
        observed_scan_pass = {
            rec['scan_id']: rec[phase_pass]
                for rec in root.predictive.observed.iterrows() }
        shuffled_scan_pass = {
            rec['scan_id']: rec[phase_pass]
                for rec in root.predictive.shuffled.iterrows() }
        shifted_scan_pass = {
            rec['scan_id']: rec[phase_pass]
                for rec in root.predictive.shifted.iterrows() }

        # Compute observed and shuffled predictive values (scalar)
        self.out('Computing observed and expected predictive values...')
        N_hits = 0
        event_shuffle_pass = np.empty((N_events, N_shuffles), '?')
        for i_event, event_id in enumerate(event_ids):
            self.out.printf('.')
            scan_query = '(event_id==%d)'%event_id
            scan_ids = np.array([rec['id'] for rec in
                root.observed_scans.where(scan_query)])

            for scan_id in scan_ids:
                if observed_scan_pass[scan_id]:
                    N_hits += 1
                    break

            # Find shuffled distribution of hits
            N_scans = scan_ids.size
            shuffled_scan_ids = np.array([rec['id']
                for rec in root.shuffled_scans.where(scan_query)])
            shuffled = np.array([shuffled_scan_pass[scan_id]
                for scan_id in shuffled_scan_ids])
            shuffle_matrix_ix = np.array(
                [np.random.permutation(shuffled.size)[:N_scans]
                    for i in xrange(N_shuffles)]).T
            event_shuffle_pass[i_event] = np.any(
                shuffled[shuffle_matrix_ix], axis=0)

        self.out.printf('\n')
        observed_ppv = N_hits / N_events
        shuffled_ppvs = event_shuffle_pass.sum(axis=0) / N_events
        expected_ppv = shuffled_ppvs.mean()
        shuffled_CI = CI(shuffled_ppvs)

        # Compute time-shifted predictive values (time-shift vector)
        N_hits = 0
        t_shifts = np.array(
            unique_values(root.shifted_scans, column='parameter'), 'd')
        shifted_ppv = np.empty_like(t_shifts)
        for i_t, dt in enumerate(t_shifts):
            self.out('Computing predictive value for %d seconds...'%dt)

            N_hits = 0
            for event_id in event_ids:
                scan_query = '(event_id==%d)&(parameter==%f)'%(event_id, dt)
                if dt == 0.0:
                    scan_query += '&(direction==1)' # pick a direction for dt=0
                scan_ids = np.array([rec['id'] for rec in
                    root.shifted_scans.where(scan_query)])

                c = 'red'
                for scan_id in scan_ids:
                    if shifted_scan_pass[scan_id]:
                        c = 'green'
                        N_hits += 1
                        break
                self.out.printf(u'\u25a0', color=c)

            self.out.printf('\n')
            shifted_ppv[i_t] = N_hits / N_events

        # Save predictive value data
        t_shifts = np.array(t_shifts)

        results = ShiftedResults(expt_type, t_shifts, shifted_ppv,
            observed_ppv, expected_ppv, shuffled_CI.lower, shuffled_CI.upper)
        results_fd = file(
            os.path.join(self.datadir, 'shifted_%s_%s.pickle'%(phase, expt_type)), 'w')
        cPickle.dump(results, results_fd)
        results_fd.close()

        self.out.outfd.close()
        self.close_data_file()

    def shifted_predictive_values_plot(self, ax=None, phase='scan', expt_type='ALL'):
        """Plot the time-shifted predictive values
        """
        results = np.load(os.path.join(self.datadir, 'shifted_%s_%s.pickle'%(phase, expt_type)))
        if ax is None:
            if self.figure is None:
                self.figure = {}
            self.figure['shifted_%s_%s'%(phase, expt_type)] = f = plt.figure(100)
            plt.clf()
            f.suptitle('Time-Shifted Predictive Values\n%s Experiments, "%s" Scan Phase'%(
                expt_type, phase))
            ax = plt.gca()

        ax.plot(results.dt, results.ppv, 'r-', lw=2, zorder=1, solid_capstyle='round')
        ax.axhline(results.observed, c='r', ls='--', lw=1.0, zorder=-1, solid_capstyle='projecting')
        ax.axhline(results.expected, c='k', ls='--', lw=1.0, zorder=-1, solid_capstyle='projecting')

        ax.axhspan(results.lower, results.upper, facecolor='k', edgecolor='none', alpha=0.4, zorder=-2)

        # ax.axhline(results.lower, ls='--', c='k', lw=0.75, zorder=-1, solid_capstyle='projecting')
        # ax.axhline(results.upper, ls='--', c='k', lw=0.75, zorder=-1, solid_capstyle='projecting')

    def shifted_predictive_values_report(self, phase='scan', dtmax=10):
        """Create a figure with shifted PPV plots for all experiment types
        """
        plt.ioff()
        if self.figure is None:
            self.figure = {}
        self.figure['shifted_%s_report'%phase] = f = plt.figure(25, figsize=(6, 10))
        plt.clf()
        f.suptitle('Time-Shifted Predictive Values\n"%s" Scan Phase'%phase)

        nrows = 4
        ncols = 3

        ax1 = f.add_subplot(nrows, ncols, 1)
        ax2 = f.add_subplot(nrows, ncols, 4)
        ax3 = f.add_subplot(nrows, ncols, 7)

        for ax, expt in [(ax1, 'ALL'), (ax2, 'DR'), (ax3, 'NOV')]:
            self.shifted_predictive_values_plot(ax, expt_type=expt)

        axlist = AxesList()
        axlist.extend([ax1, ax2, ax3])

        axlist.apply('tick_params', right=False, top=False, direction='out')
        axlist.set(xlim=(-(dtmax+1), dtmax+1), ylim=(0.0, 0.35),
            yticks=[0,0.15,0.3], xticks=[-dtmax, 0, dtmax])

        ax1.set_ylabel('ALL', size='small')
        ax2.set_ylabel('DR', size='small')
        ax3.set_ylabel('NOV', size='small')
        ax3.set_xlabel('Time shift (s)', size='small')

        plt.ion()
        plt.show()

    def cross_correlations(self, phase='scan', N_shuffles=1000,
        event_slice=None, scan_slice=None, bins=12, window=(-540, -180),
        use_field_end=True, alpha=0.05, test_events=None):
        """Compute scan-field cross-correlograms for observations and shuffled
        scan-timing data, as well as baseline random expectations.

        Event- and scan-domain data slices can be optionally specified. They
        should contain only one comparandum, that is, the comparanda attribute
        should be a list with a single value.
        """
        root = self.get_data_file().root

        field_key = use_field_end and 'angle_end' or 'angle_start'
        event_query = scan_query = ''
        if event_slice:
            cmpd = event_slice.comparanda[0]
            if type(cmpd) is str:
                if cmpd == "FIRST":
                    event_query = '(type=="NOV")&(day==1)'
                else:
                    event_query = '%s%s"%s"'%(event_slice.field, event_slice.comparator, cmpd)
            else:
                event_query = '%s%s%s'%(event_slice.field, event_slice.comparator, cmpd)
            event_ids = np.array([rec['id'] for rec in
                root.events.where(event_query)])
            field_angles = np.array([rec[field_key] for rec in
                root.events.where(event_query)])
        else:
            event_ids = root.events.col('id')
            field_angles = root.events.col(field_key)

        if test_events is not None:
            event_ids = event_ids[:test_events]
            field_angles = field_angles[:test_events]
        N_events = event_ids.size

        phase_pass = phase + '_pass'
        observed_scan_pass = {
            rec['scan_id']: rec[phase_pass]
                for rec in root.predictive.observed.iterrows() }
        shuffled_scan_pass = {
            rec['scan_id']: rec[phase_pass]
                for rec in root.predictive.shuffled.iterrows() }
        shuffled_scan_angle = {
            rec['id']: rec['angle']
                for rec in root.shuffled_scans.iterrows() }

        # Histogram parameters
        bins = (bins % 2 == 0) and bins + 1 or bins # force odd bins
        dlim = window
        edges = np.linspace(dlim[0], dlim[1], bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        def xcorr(scan_angles):
            xcorr = np.zeros(bins, 'i')
            for i in xrange(N_events):
                xcorr += np.histogram(field_angles[i] - scan_angles[i], bins=edges)[0]
            return xcorr.astype('d')

        def intervals(M):
            ul = np.empty((2, M.shape[1]), 'd')
            for j in xrange(M.shape[1]):
                ul[:,j] = CI(M[:,j], alpha=alpha)
            return Interval(ul[0], ul[1])

        # Compute observed and bootstrap cross-correlations
        self.out('Computing observed xcorr...')
        scan_angles = np.empty(N_events, 'O')
        sampled_angles = np.empty((N_shuffles, N_events), 'O')
        for i_event, event_id in enumerate(event_ids):
            self.out.printf('.', color='green')

            scan_query = '(event_id==%d)'%event_id
            if scan_slice:
                scan_query += '&(%s%s%s)'%(scan_slice.field,
                    scan_slice.comparator, scan_slice.comparand[0])

            N_scans = 0
            hit_scan_angles = []
            for scan in root.observed_scans.where(scan_query):
                N_scans += 1
                if observed_scan_pass[scan['id']]:
                    hit_scan_angles.append(scan['angle'])

            scan_angles[i_event] = hit_scan_angles

            # Find shuffled distribution of hits
            shuffled_scans_id = [rec['id'] for rec in
                root.shuffled_scans.where(scan_query)]
            N_shuffled_scans = len(shuffled_scans_id)
            shuffled_angles = np.empty(N_shuffled_scans, 'd')
            for i_scan, scan_id in enumerate(shuffled_scans_id):
                if shuffled_scan_pass[scan_id]:
                    shuffled_angles[i_scan] = shuffled_scan_angle[scan_id]
                else:
                    # set bogus value to remove scan angle from histogram:
                    shuffled_angles[i_scan] = 1e4
            shuffle_matrix_ix = np.array(
                [np.random.permutation(N_shuffled_scans)[:N_scans]
                    for i in xrange(N_shuffles)])

            # Construct shuffles x events x scans matrix of resampled scan
            # angles of shuffled scans
            M_samples = shuffled_angles[shuffle_matrix_ix]
            for i_shuffle in xrange(N_shuffles):
                sampled_angles[i_shuffle, i_event] = M_samples[i_shuffle]

        self.out.printf('\n')
        obs_xcorr = xcorr(scan_angles)
        shuffled_xcorrs = np.empty((N_shuffles, obs_xcorr.size), 'd')
        for i_shuffle in xrange(N_shuffles):
            shuffled_xcorrs[i_shuffle] = xcorr(sampled_angles[i_shuffle])
        expected_xcorr = shuffled_xcorrs.mean(axis=0)
        shuffled_CI = intervals(shuffled_xcorrs)

        # Normalize cross-correlations by number of events
        obs_xcorr /= N_events
        expected_xcorr /= N_events
        shuffled_CI.lower[:] /= N_events
        shuffled_CI.upper[:] /= N_events
        shuffled_xcorrs /= N_events

        # Save xcorr data for plotting etc
        xcorr_data = CrossCorrelation(window, centers, obs_xcorr, expected_xcorr,
            shuffled_xcorrs, shuffled_CI, event_slice, scan_slice)
        xcorr_fd = file(os.path.join(self.datadir, 'xcorr_%s.pickle'%phase), 'w')
        cPickle.dump(xcorr_data, xcorr_fd)
        xcorr_fd.close()

        self.close_data_file()

    def cross_correlations_plot(self, phase='scan', alpha=0.05,
        use_FDR_control=False, ax_pair=None):
        """Create figure with observed and null cross-correlations with
        confidence intervals represented by shaded regions.
        """
        xcorr = np.load(os.path.join(self.datadir, 'xcorr_%s.pickle'%phase))
        self.out('Plotting cross-correlation data for phase %s...'%phase)

        plt.ioff()
        if ax_pair is None:
            # if type(self.figure) is not dict:
            self.figure = {}
            figname = 'xcorr_%s'%phase
            event_label = (xcorr.event_slice is None) and 'All Events' or str(xcorr.event_slice)
            scan_label = (xcorr.scan_slice is None) and 'All Scans' or str(xcorr.scan_slice)
            self.figure[figname] = f = plt.figure(num=101, figsize=(10, 8))
            plt.clf()
            f.suptitle(u'%s\n%s'%(event_label.title(), scan_label.title()))
            xcorr_ax = f.add_subplot(211)
            pval_ax = f.add_subplot(212)
            quicktitle(xcorr_ax, 'observed vs expected confidence intervals')
            quicktitle(pval_ax, 'p-value')
        else:
            xcorr_ax, pval_ax = ax_pair

        # Plot observed and null cross-correlations
        xcorr_ax.plot(xcorr.bins, xcorr.observed, 'b-', lw=1.5, zorder=10)
        xcorr_ax.plot(xcorr.bins, xcorr.expected, 'r-', lw=0.75, zorder=5)
        shaded_region(xcorr. bins, xcorr.CI.lower, xcorr.CI.upper,
            ax=xcorr_ax, ec='none', fc='r', alpha=0.4, zorder=0)
        xcorr_ax.set_xlim(*xcorr.dlim)
        xcorr_ax.set_ylim(0, 1.1 * np.r_[xcorr.observed, xcorr.CI.upper].max())

        # Plot p-values
        bins = xcorr.bins.size
        pvals = np.empty_like(xcorr.observed)
        for i in xrange(bins):
            pvals[i] = pvalue(xcorr.observed[i], xcorr.shuffled[:,i])
        if use_FDR_control:
            sig_ix = FDR_control(pvals, alpha=alpha)
            plabel = 'FDR(%.3f)'%alpha
        else:
            sig_ix = pvals <= alpha
            plabel = '<%.3f'%alpha
        pval_ax.axhline(0.05, c='k', ls='--')
        pval_ax.axhline(0.01, c='k', ls=':')
        pval_ax.semilogy(xcorr.bins, pvals, 'r:', label='p', lw=1)
        pval_ax.semilogy(np.ma.masked_where(True - sig_ix, xcorr.bins),
            np.ma.masked_where(True - sig_ix, pvals), 'r-o', label=plabel,
            lw=2, zorder=5)
        pval_ax.set_xlim(*xcorr.dlim)
        if ax_pair is None:
            pval_ax.legend()

        plt.ion()
        plt.show()

    def create_predictive_tables(self, phase='scan', alpha=None):
        """Create data tables listing predictive scans and predicted potentiation
        events. For observed/shuffled/shifted scans that will be a paired list
        of predictive scans and the corresponding place-field potentiation events.
        Both scans and events will be represented as id/index into /behavior/scans
        and /physiology/potentiation, respectively. Events will be recorded with
        relevant information such as lap number and hit window angles, and whether
        or not they were predicted by how many scans.

        All data tables will be stored under /posthoc in the analysis data file.
        """
        # Load data file to create the predictive scans tables
        data_file = self.get_data_file(mode='a')
        root = data_file.root

        # Create new groups for predictive test data
        try:
            posthoc_group = data_file.getNode('/', 'posthoc')
        except tb.NoSuchNodeError:
            pass
        else:
            self.out("Removing old post-hoc tables...")
            data_file.removeNode(posthoc_group, recursive=True)
        finally:
            posthoc_group = data_file.createGroup('/', 'posthoc',
                title='Post-Hoc Data For Predictive Scans')

        # Create tables for different types of scans
        PostHoc = namedtuple("PostHocData", "observed shuffled shifted")
        posthoc = PostHoc(
            data_file.createTable(posthoc_group, 'observed', PredictiveScanDescr,
                title='Post-Hoc Data for Predictive Observed Scans'),
            data_file.createTable(posthoc_group, 'shuffled', PredictiveScanDescr,
                title='Post-Hoc Data for Predictive Shuffled Scans'),
            data_file.createTable(posthoc_group, 'shifted', PredictiveScanDescr,
                title='Post-Hoc Data for Predictive Shifted Scans')
        )
        events_table = data_file.createTable(posthoc_group, 'events',
            PredictedEventDescr, title='Predictive Value Results Across Events')

        if alpha is None:
            pass_query = '%s_pass==True'%phase
        else:
            pass_query = '%s_pct>=%f'%(phase, 1 - float(alpha) / 2)

        # Collect predictive scan-associations potentiations across observed,
        # shuffled, and shifted scan tables
        for table_name in ('observed', 'shuffled', 'shifted'):
            self.out('Collating %s scans...'%table_name)

            predictive_table = getattr(root.predictive, table_name)
            posthoc_table = getattr(posthoc, table_name)

            pass_id = 0
            row = posthoc_table.row
            for scan in predictive_table.where(pass_query):
                self.out.printf('.', color='green')
                row['id'] = pass_id
                row['scan_id'] = scan['scan_id']
                row['event_id'] = scan['event_id']

                if pass_id % 10 == 0:
                    posthoc_table.flush()

                row.append()
                pass_id += 1

            self.out.printf('\n')
            posthoc_table.flush()

        # Collect scan-prediction information about all place-field events
        self.out('Collating event information...')
        potentiation_table = get_node('/physiology', self.results['table'])
        event_row = events_table.row
        i = 0
        for event in root.events.iterrows():
            self.out.printf('.')

            potentiation = potentiation_table[event['id']]
            assert potentiation['id'] == event['id'], 'event id mismatch'

            window = root.predictive.windows[event['id']]
            assert window['event_id'] == event['id'], 'window id mismatch'

            event_row['event_id'] = event['id']
            event_row['rat'] = event['rat']
            event_row['day'] = event['day']
            event_row['session'] = event['session']
            event_row['expt_type'] = event['expt_type']
            event_row['tc'] = event['tc']
            event_row['event_lap'] = potentiation['lap']

            tracklim = window['tracklim']
            hit_window = tracklim[0] % 360, tracklim[1] % 360
            event_row['hit_window'] = hit_window

            scan_hit_query = '(event_id==%d)&(%s)'%(event['id'], pass_query)
            num_hit_scans = len(root.predictive.observed.getWhereList(scan_hit_query))

            event_row['predicted'] = bool(num_hit_scans) and 'y' or 'n'
            event_row['num_scans'] = num_hit_scans
            event_row.append()

            if i % 10 == 0:
                events_table.flush()
                self.out.printf('|', color='cyan')
            i += 1

        self.out.printf("\n")

        # Save table of predictive scans and predicted events as csv files
        dump_table(root.posthoc.observed,
            filename=os.path.join(self.datadir, 'predictive_scans_observed.csv'))
        dump_table(root.posthoc.events,
            filename=os.path.join(self.datadir, 'predicted_events.csv'))

        self.close_data_file()
        self.out("All done!")

    def create_predicted_event_spike_report(self):
        """Create a compactified lap spikes report for all predicted events
        """
        from .field_modulation import CompactLapSpikesReport
        root = self.get_data_file().root
        predicted_ix = root.posthoc.events.getWhereList('predicted=="y"')
        predicted_ids = map(int, root.posthoc.events[predicted_ix]['event_id'])
        report = CompactLapSpikesReport(desc='predicted events',
            datadir=os.path.join(self.datadir, "compact_report"))
        report(condn_or_ids=predicted_ids)
        report.open()

    def posthoc_analysis(self, phase='scan', lap_sweep=range(3,14,2)):
        """Given the set of scan-activated (predictive) field events and its
        complement (non-predictive), assess the relative distributions of
        relevant characteristics of hit scans and hit events.
        """
        self.start_logfile('posthoc')
        root = self.get_data_file().root

        predictive = {'duration': [], 'magnitude': [], 'baseline': [], 'potentiation': [], 'event_diff': [],
            'fdiff': [], 'start_distance': [], 'end_distance': [], 'norm_distance': []}
        inwindow = {'duration': [], 'magnitude': [], 'fdiff': [],
            'start_distance': [], 'end_distance': [], 'norm_distance': []}
        nonpredictive = {'duration': [], 'magnitude': [], 'baseline': [], 'potentiation': [],
            'event_diff': [], 'fdiff': []}

        predictive_rat = {'fdiff': {}} #outbound': {}, 'inbound': {}}
        inwindow_rat = {'fdiff': {}} #outbound': {}, 'inbound': {}}
        nonpredictive_rat = {'fdiff': {}} #outbound': {}, 'inbound': {}}

        predictive_lap = {}
        inwindow_lap = {}
        for lap in lap_sweep:
            predictive_lap[lap] = {'min': [], 'max': []}
            inwindow_lap[lap] = {'min': [], 'max': []}

        phase_pass = phase + '_pass'

        def fzero(num, dur):
            if dur:
                return num / dur
            return 0.0

        def fzero_ufunc(u, v):
            m = np.empty_like(u)
            nz = np.nonzero(v)
            m[nz] = u[nz] / v[nz]
            m[v == 0.0] = 0.0
            return m

        self.out('Collecting post-hoc data for observed scans...')
        for test in root.predictive.observed.iterrows():

            scan = root.observed_scans[test['scan_id']]
            assert scan['id'] == test['scan_id'], 'scan id mismatch'

            event = root.events[test['event_id']]
            assert event['id'] == test['event_id'], 'event id mismatch'

            window = root.predictive.windows[test['event_id']]
            assert window['event_id'] == test['event_id'], 'window/event id mismatch'

            if test[phase_pass]:
                results = predictive
                results_rat = predictive_rat
                results_lap = predictive_lap
            elif test['in_window']:
                results = inwindow
                results_rat = inwindow_rat
                results_lap = inwindow_lap
            else:
                results = nonpredictive
                results_rat = nonpredictive_rat
                results_lap = None

            results['duration'].append(scan['duration'])
            results['magnitude'].append(scan['magnitude'])

            k_outbound = scan['outbound_count'] + scan['prefix_count']
            k_inbound = scan['inbound_count'] + scan['postfix_count']
            dt_outbound = scan['outbound_dt'] + scan['prefix_dt']
            dt_inbound = scan['inbound_dt'] + scan['postfix_dt']
            k = k_outbound + k_inbound
            dt = dt_outbound + dt_inbound
            F = k / dt

            if k > 0: #k_outbound > 0 and k_inbound > 0:
                F_outbound = fzero(k_outbound, dt_outbound)
                F_inbound = fzero(k_inbound, dt_inbound)
                F_diff = fzero((F_outbound - F_inbound), (F_outbound + F_inbound))

                results['fdiff'].append(F_diff)
                rat = event['rat']
                if rat not in results_rat['fdiff']:
                    results_rat['fdiff'][rat] = []
                results_rat['fdiff'][rat].append(F_diff)

            norm_distance = (window['angle_start'] - scan['angle']) / window['width']
            if 'start_distance' in results:
                results['start_distance'].append(window['angle_start'] - scan['angle'])
                results['end_distance'].append(window['angle_end'] - scan['angle'])
                results['norm_distance'].append(norm_distance)

            if results_lap is not None:
                for lap in lap_sweep:
                    if event['laps'] >= lap:
                        results_lap[lap]['min'].append(norm_distance)
                    elif event['laps'] <= lap:
                        results_lap[lap]['max'].append(norm_distance)

            if scan['id'] % 100 == 0:
                self.out.printf('.')
        self.out.printf('\n')

        self.out('Collecting post-hoc data for events...')
        potentiation_table = get_node('/physiology', self.results['table'])
        for posthoc in root.posthoc.events.iterrows():

            event = potentiation_table[posthoc['event_id']]
            assert event['id'] == posthoc['event_id'], 'event id mismatch'

            if posthoc['predicted'] == 'y':
                results = predictive
            else:
                results = nonpredictive

            results['baseline'].append(event['baseline'])
            results['potentiation'].append(event['strength'])
            results['event_diff'].append(event['strength'] - event['baseline'])

            if event['id'] % 10 == 0:
                self.out.printf('.')
        self.out.printf('\n')

        #
        # Scans figure
        #

        self.out('Creating figures...')
        plt.ioff()
        self.figure = {}
        self.figure['posthoc_scans'] = f = plt.figure(num=102, figsize=(8,11))
        plt.clf()
        f.suptitle('Post-Hoc Scan Distributions')

        ECDF = lambda H: np.cumsum(H) / float(np.sum(H))

        dbins = np.linspace(0.0, 12.0, 256)
        H_duration_predictive = np.histogram(predictive['duration'], bins=dbins)[0]
        H_duration_inwindow = np.histogram(inwindow['duration'], bins=dbins)[0]
        H_duration_nonpredictive = np.histogram(nonpredictive['duration'], bins=dbins)[0]
        dcenters = (dbins[1:] + dbins[:-1]) / 2

        ax = f.add_subplot(321)
        ax.plot(dcenters, ECDF(H_duration_predictive), 'r-', lw=1.5, zorder=1)
        ax.plot(dcenters, ECDF(H_duration_inwindow), 'r--', lw=1.5, zorder=0)
        ax.plot(dcenters, ECDF(H_duration_nonpredictive), 'k-', lw=1.5, zorder=-1)
        ax.set_xlim(dcenters[0], dcenters[-1])
        ax.set_xlabel('duration (s)')
        ax.set_ylabel('cumulative probability')

        mbins = np.linspace(0, 25.0, 256)
        H_magnitude_predictive = np.histogram(predictive['magnitude'], bins=mbins)[0]
        H_magnitude_inwindow = np.histogram(inwindow['magnitude'], bins=mbins)[0]
        H_magnitude_nonpredictive = np.histogram(nonpredictive['magnitude'], bins=mbins)[0]
        mcenters = (mbins[1:] + mbins[:-1]) / 2

        ax = f.add_subplot(322)
        ax.plot(mcenters, ECDF(H_magnitude_predictive), 'r-', lw=1.5, zorder=1)
        ax.plot(mcenters, ECDF(H_magnitude_inwindow), 'r--', lw=1.5, zorder=0)
        ax.plot(mcenters, ECDF(H_magnitude_nonpredictive), 'k-', lw=1.5, zorder=-1)
        ax.set_xlim(mcenters[0], mcenters[-1])
        ax.set_xlabel('magnitude (cm)')

        self.out('N scans pred = %d'%len(predictive['magnitude']))
        self.out('N scans win = %d'%len(inwindow['magnitude']))
        self.out('N scans nonpred = %d'%len(nonpredictive['magnitude']))

        self.out('Magnitude pred - nonpred: D = %.3f, p < %.8f'%st.ks_2samp(predictive['magnitude'],
            nonpredictive['magnitude']))

        self.out('Magnitude pred - win: D = %.3f, p < %.8f'%st.ks_2samp(predictive['magnitude'],
            inwindow['magnitude']))

        fbins = np.linspace(-1.0, 1.0, 256)
        H_fdiff_predictive = np.histogram(predictive['fdiff'], bins=fbins)[0]
        H_fdiff_inwindow = np.histogram(inwindow['fdiff'], bins=fbins)[0]
        H_fdiff_nonpredictive = np.histogram(nonpredictive['fdiff'], bins=fbins)[0]
        fcenters = (fbins[1:] + fbins[:-1]) / 2

        ax = f.add_subplot(312)
        ax.plot(fcenters, ECDF(H_fdiff_predictive), 'b-', lw=2, label='pred. diff')
        ax.plot(fcenters, ECDF(H_fdiff_inwindow), 'g-', lw=2, label='win. diff')
        ax.plot(fcenters, ECDF(H_fdiff_nonpredictive), 'r-', lw=2, label='non-pred. diff')
        ax.set_xlim(fcenters[0], fcenters[-1])
        ax.set_xlabel('normalized firing-rate difference')
        ax.set_ylabel('cumulative probability')
        ax.legend(loc=4)

        #
        # Inbound/outbound firing-rate bar plots
        #

        width = 0.8
        lefts = [0, 1, 2]
        data = [nonpredictive['fdiff'], inwindow['fdiff'], predictive['fdiff']]
        labels = ['non', 'win', 'pred']
        heights = [np.mean(x) for x in data]
        errs = [fzero(np.std(x), np.sqrt(len(x))) for x in data]
        cols = ['0.7'] * 3 #, '0.4'] * 3

        ax = f.add_subplot(325)
        h = ax.bar(lefts, heights, yerr=errs, width=width, linewidth=0, color=cols, ecolor='k')
        ax.set_xticks(np.array([0,1,2]) + width / 2)
        ax.set_xticklabels(labels)
        ax.tick_params(top=False, right=False)
        ax.tick_params(axis='x', direction='out')
        ax.set_xlim(-width / 2, 2 + 1.5 * width)
        ax.set_xlabel('scan predictiveness')
        ax.set_ylabel('normalized firing rate')
        quicktitle(ax, 'across scans')

        data_rat = [nonpredictive_rat['fdiff'], inwindow_rat['fdiff'], predictive_rat['fdiff']]
        heights_rat = [np.mean([np.mean(x) for x in data.values()]) for data in data_rat]
        errs_rat = [np.std([np.mean(x) for x in data.values()]) / np.sqrt(len(data)) for data in data_rat]

        ax = f.add_subplot(326)
        h = ax.bar(lefts, heights_rat, yerr=errs_rat, width=width, linewidth=0, color=cols, ecolor='k')
        ax.set_xticks(np.array([0,1,2]) + width / 2)
        ax.set_xticklabels(labels)
        ax.tick_params(top=False, right=False)
        ax.tick_params(axis='x', direction='out')
        ax.set_xlim(-width / 2, 2 + 1.5 * width)
        ax.set_xlabel('scan predictiveness')
        quicktitle(ax, 'across rats')

        #
        # Windows figure (scan position)
        #

        self.figure['posthoc_windows'] = f = plt.figure(num=103, figsize=(8, 7))
        plt.clf()
        f.suptitle('Post-Hoc Within-Window Scan Distributions')

        angle_bins = np.linspace(-120, 120, 256)
        H_startdist_predictive = np.histogram(predictive['start_distance'], bins=angle_bins)[0]
        H_startdist_inwindow = np.histogram(inwindow['start_distance'], bins=angle_bins)[0]
        angles = (angle_bins[1:] + angle_bins[:-1]) / 2

        self.out('Start distance: D = %.3f, p < %.8f'%st.ks_2samp(predictive['start_distance'],
            inwindow['start_distance']))

        ax = f.add_subplot(221)
        ax.plot(angles, ECDF(H_startdist_predictive), 'r-', lw=1.5, label='pred')
        ax.plot(angles, ECDF(H_startdist_inwindow), 'r--', lw=1.5, label='win')
        ax.set_xlim(angles[0], angles[-1])
        ax.set_xlabel('angle (degrees)')
        ax.set_ylabel('cumulative probability')
        ax.legend(loc=4)

        H_enddist_predictive = np.histogram(predictive['end_distance'], bins=angle_bins)[0]
        H_enddist_inwindow = np.histogram(inwindow['end_distance'], bins=angle_bins)[0]

        self.out('End distance: D = %.3f, p < %.8f'%st.ks_2samp(predictive['end_distance'],
            inwindow['end_distance']))

        ax.plot(angles, ECDF(H_enddist_predictive), 'r-', lw=1.5)
        ax.plot(angles, ECDF(H_enddist_inwindow), 'r--', lw=1.5)
        ax.set_xlim(angles[0], angles[-1])
        ax.set_xlabel('angle (degrees)')
        quicktitle(ax, 'end/left, start/right scan-lag')

        norm_bins = np.linspace(0, 1, 256)
        H_normdist_predictive = np.histogram(predictive['norm_distance'], bins=norm_bins)[0]
        H_normdist_inwindow = np.histogram(inwindow['norm_distance'], bins=norm_bins)[0]
        norm_centers = (norm_bins[1:] + norm_bins[:-1]) / 2

        self.out('Norm. distance: D = %.3f, p < %.8f'%st.ks_2samp(predictive['norm_distance'],
            inwindow['norm_distance']))

        chi_bins = 30
        df = chi_bins - 1
        F_normdist_predictive = np.histogram(predictive['norm_distance'], bins=np.linspace(0, 1, chi_bins+1))[0]
        F_normdist_inwindow = np.histogram(inwindow['norm_distance'], bins=np.linspace(0, 1, chi_bins+1))[0]
        self.out('Uniformity of win norm. dist.: Chi2(%d) = %f, p<%f'%((df,) + st.chisquare(F_normdist_inwindow)))
        self.out('Uniformity of pred norm. dist: Chi2(%d) = %f, p<%f'%((df,) + st.chisquare(F_normdist_predictive)))

        ax = f.add_subplot(222)
        ax.plot(norm_centers, ECDF(H_normdist_predictive), 'r-', lw=1.5)
        ax.plot(norm_centers, ECDF(H_normdist_inwindow), 'r--', lw=1.5)
        ax.set_xlim(norm_centers[0], norm_centers[-1])
        ax.set_xlabel('normalized scan position')
        ax.set_ylabel('cumulative probability')
        quicktitle(ax, 'normalized lag')

        M = np.empty((len(norm_centers), len(lap_sweep)), 'd')
        for j, lap in enumerate(lap_sweep):
            ECDF_p = ECDF(np.histogram(predictive_lap[lap]['min'], bins=norm_bins)[0])
            ECDF_w = ECDF(np.histogram(inwindow_lap[lap]['min'], bins=norm_bins)[0])
            M[:,j] = fzero_ufunc((ECDF_p - ECDF_w), (ECDF_w + ECDF_p))

        M[M==1] = 0.0

        ax = f.add_subplot(223)
        ax.imshow(M, origin='lower', interpolation='nearest', aspect='auto', #cmap='jet',
            extent=[-0.5, len(lap_sweep)-0.5, norm_bins[0], norm_bins[-1]])
        ax.axis('tight')
        ax.set_xticks(range(len(lap_sweep)))
        ax.set_xticklabels(map(str, lap_sweep))
        ax.set_xlabel('minimum lap')
        ax.set_ylabel('norm. scan position')
        ax.tick_params(direction='out')
        ax.tick_params(top=False, right=False)
        quicktitle(ax, 'dist.-distro. diff. x laps')
        self.out('Norm diff range for min sweep: %f to %f'%(M.min(), M.max()))

        M = np.empty((len(norm_centers), len(lap_sweep)), 'd')
        for j, lap in enumerate(lap_sweep):
            ECDF_p = ECDF(np.histogram(predictive_lap[lap]['max'], bins=norm_bins)[0])
            ECDF_w = ECDF(np.histogram(inwindow_lap[lap]['max'], bins=norm_bins)[0])
            M[:,j] = fzero_ufunc((ECDF_p - ECDF_w), (ECDF_w + ECDF_p))

        M[M==1] = 0.0

        ax = f.add_subplot(224)
        ax.imshow(M, origin='lower', interpolation='nearest', aspect='auto', #cmap='jet',
            extent=[-0.5, len(lap_sweep)-0.5, norm_bins[0], norm_bins[-1]])
        ax.axis('tight')
        ax.set_xticks(range(len(lap_sweep)))
        ax.set_xticklabels(map(str, lap_sweep))
        ax.set_xlabel('maximum lap')
        ax.tick_params(direction='out')
        ax.tick_params(top=False, right=False)
        quicktitle(ax, 'dist.-distro. diff. x laps')
        self.out('Norm diff range for max sweep: %f to %f'%(M.min(), M.max()))

        #
        # Events figure
        #

        self.figure['posthoc_events'] = f = plt.figure(num=104, figsize=(8,11))
        plt.clf()
        f.suptitle('Post-Hoc Event Distributions')

        bbins = np.linspace(0.0, 3.0, 256)
        H_baseline_predictive = np.histogram(predictive['baseline'], bins=bbins)[0]
        H_baseline_nonpredictive = np.histogram(nonpredictive['baseline'], bins=bbins)[0]
        H_potentiation_predictive = np.histogram(predictive['potentiation'], bins=bbins)[0]
        H_potentiation_nonpredictive = np.histogram(nonpredictive['potentiation'], bins=bbins)[0]
        H_eventdiff_predictive = np.histogram(predictive['event_diff'], bins=bbins)[0]
        H_eventdiff_nonpredictive = np.histogram(nonpredictive['event_diff'], bins=bbins)[0]
        bcenters = (bbins[1:] + bbins[:-1]) / 2

        ax = f.add_subplot(321)
        ax.plot(bcenters, ECDF(H_baseline_predictive), 'r-', label='pred. baseline')
        ax.plot(bcenters, ECDF(H_baseline_nonpredictive), 'k-', label='non-pred. baseline')
        ax.plot(bcenters, ECDF(H_potentiation_predictive), 'r-', lw=2, label='pred. potentiation')
        ax.plot(bcenters, ECDF(H_potentiation_nonpredictive), 'k-', lw=2, label='non-pred. potentiation')
        ax.set_xlim(bcenters[0], bcenters[-1])
        ax.set_xlabel('field strength (d.p.)')
        ax.set_ylabel('cumulative probability')
        ax.legend(loc=4)

        def print_mean_sd(x, label):
            self.out('--> %s: %f +/- %f' % (label.title(), np.mean(x), np.std(x)))

        self.out('Baseline: D = %.3f, p < %.8f'%st.ks_2samp(predictive['baseline'],
            nonpredictive['baseline']))
        print_mean_sd(predictive['baseline'], 'predictive')
        print_mean_sd(nonpredictive['baseline'], 'nonpredictive')

        self.out('Potentiation: D = %.3f, p < %.8f'%st.ks_2samp(predictive['potentiation'],
            nonpredictive['potentiation']))
        print_mean_sd(predictive['potentiation'], 'predictive')
        print_mean_sd(nonpredictive['potentiation'], 'nonpredictive')

        ax = f.add_subplot(322)
        ax.plot(bcenters, ECDF(H_eventdiff_predictive), 'r-', lw=1.5, label='pred. diff')
        ax.plot(bcenters, ECDF(H_eventdiff_nonpredictive), 'k-', lw=1.5, label='non-pred. diff')
        ax.set_xlim(bcenters[0], bcenters[-1])
        ax.set_xlabel('potentiation (d.p.)')
        ax.legend(loc=4)

        self.out('Field Change: D = %.3f, p < %.8f'%st.ks_2samp(predictive['event_diff'],
            nonpredictive['event_diff']))
        print_mean_sd(predictive['event_diff'], 'predictive')
        print_mean_sd(nonpredictive['event_diff'], 'nonpredictive')

        ax = f.add_subplot(312)
        ax.plot([0, 1], np.c_[nonpredictive['baseline'], nonpredictive['potentiation']].T,
            'k-', lw=0.75, alpha=0.2, zorder=0)
        ax.plot([0, 1], [np.median(nonpredictive['baseline']), np.median(nonpredictive['potentiation'])],
            'r-', lw=2, zorder=1)
        ax.axvline(0, c='k', ls=':', lw=0.5)
        ax.axvline(1, c='k', ls=':', lw=0.5)

        ax.plot([1.5, 2.5], np.c_[predictive['baseline'], predictive['potentiation']].T,
            'k-', lw=0.75, alpha=0.2, zorder=0)
        ax.plot([1.5, 2.5], [np.median(nonpredictive['baseline']), np.median(nonpredictive['potentiation'])],
            'r-', lw=2, zorder=1) # for comparison
        ax.plot([1.5, 2.5], [np.median(predictive['baseline']), np.median(predictive['potentiation'])],
            'c-', lw=2, zorder=2)
        ax.axvline(1.5, c='k', ls=':', lw=0.5)
        ax.axvline(2.5, c='k', ls=':', lw=0.5)
        ax.set_xlim(-0.5, 3.0)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0,1,1.5,2.5])
        ax.set_xticklabels(['Non/Base', 'Non/Pot', 'Pred/Base', 'Pred/Pot'], size='small')
        quicktitle(ax, 'predicted', size='x-small')

        heatbins = [np.linspace(0, 1.5, 10), np.linspace(0, 3.0, 12)]

        ax = f.add_subplot(325)
        heatmap(predictive['baseline'], predictive['potentiation'], ax=ax, bins=heatbins)
        quicktitle(ax, 'predicted events', size='x-small')
        ax.set_ylabel('potentiation (d.p.)')
        ax.set_xlabel('baseline (d.p.)')

        ax = f.add_subplot(326)
        heatmap(nonpredictive['baseline'], nonpredictive['potentiation'], ax=ax, bins=heatbins)
        quicktitle(ax, 'non-predicted events', size='x-small')
        ax.set_xlabel('baseline (d.p.)')

        plt.ion()
        plt.show()

        self.out('All done!')
        self.close_logfile()

    def predictive_group_name(self, lap):
        """Returns the name for the predictive group that holds data for adaptive
        window testing on a given hit lap
        """
        if lap < 0:
            new_group_name = 'predictive_%d_before'%abs(lap)
        elif lap == 0:
            new_group_name = 'predictive_eventlap'
        else:
            new_group_name = 'predictive_%d_after'%lap
        return new_group_name

    def process_cross_correlations_across_laps(self, test_hit_laps=(-3,-2,-1,0,1,2,3)):
        """Run the cross-correlation and predictive-value analyses for this
        scan-field analysis data for adaptive windows centered on a window
        of different laps
        """
        data_file = self.get_data_file()
        root = data_file.root
        assert hasattr(root, 'events'), "please run data collection first"
        self.close_data_file()

        cwd = os.getcwd()
        os.chdir(self.datadir)

        for hit_lap in test_hit_laps:
            self.out('---\nProcessing predictive testing for lap = %d:\n---'%hit_lap)

            # Compute the predictive hit tests for the current hit lap
            self.process_data(hit_lap=hit_lap)

            for expt in ('ALL', 'DR', 'NOV', 'FIRST'):

                if expt == 'ALL':
                    expt_slice = None
                else:
                    expt_slice = DataSlice('event', 'expt_type', '==', (expt,))

                for phase in Scan.PhaseNames:

                    filename_key = '%s_%s_lap%d'%(phase, expt, hit_lap)
                    if os.path.exists('xcorr_%s.pickle'%filename_key):
                        self.out('Found data: %s'%(', '.join(filename_key.split('_'))))
                        continue

                    # Run the cross correlations, and save out the plot of the individual
                    # hit-lap mode
                    self.cross_correlations(phase=phase, event_slice=expt_slice,
                        window=(360 * hit_lap - 180, 360 * hit_lap + 180))
                    self.cross_correlations_plot(phase=phase)
                    self.save_plots_and_close()

                    # Rename the saved data and plot files so the different hit laps being
                    # tested don't collide
                    os.rename('xcorr_%s_00.pdf'%phase, 'xcorr_%s.pdf'%filename_key)
                    os.rename('xcorr_%s.pickle'%phase, 'xcorr_%s.pickle'%filename_key)

            # Rename predictive data group to be unique for each hit lap
            predictive_group_name = self.predictive_group_name(hit_lap)
            data_file = self.get_data_file(mode='a')
            data_file.root.predictive._v_attrs['lap_name'] = predictive_group_name
            data_file.renameNode('/predictive', predictive_group_name, overwrite=True)
            self.out('Renamed /predictive to %s'%getattr(
                data_file.root, predictive_group_name)._v_pathname)
            self.close_data_file()

        # Save the test hit laps for plotting
        save_fd = file(os.path.join(self.datadir, 'xcorr_laps.pickle'), 'w')
        cPickle.dump(test_hit_laps, save_fd)
        save_fd.close()
        os.chdir(cwd)

    def cross_correlations_across_laps_plot(self):
        """Plot the aggregate cross correlation data across laps in a single plot
        """
        hit_laps = np.load(os.path.join(self.datadir, 'xcorr_laps.pickle'))

        plt.ioff()
        self.figure = {}
        fignum = 110
        rows, cols = 5, 2 #tiling_dims(len(Scan.PhaseNames))

        for expt in ('ALL', 'DR', 'NOV', 'FIRST'):

            self.out('Creating figures for "%s" experiments.'%expt)

            figname = 'xcorr_laps_%s'%expt
            self.figure[figname] = f_x = plt.figure(num=fignum, figsize=(8.5, 11))
            plt.clf()
            f_x.suptitle('Predictive Cross-Correlations Across Hit Laps\n%s Experiments'%expt)
            fignum += 1

            self.figure[figname + '_pvals'] = f_p = plt.figure(num=fignum, figsize=(8.5, 11))
            plt.clf()
            f_p.suptitle('P-Values Across Hit Laps\n%s Experiments'%expt)
            fignum += 1

            ymax = 0.0

            for i, phase in enumerate(Scan.PhaseNames):

                self.out('Plotting "%s" scan phase...'%phase)

                xcorr_ax = f_x.add_subplot(rows, cols, i+1)
                pval_ax = f_p.add_subplot(rows, cols, i+1)
                quicktitle(xcorr_ax, phase.title(), size='small')
                quicktitle(pval_ax, phase.title(), size='small')

                for lap in hit_laps:
                    self.cross_correlations_plot(phase='%s_%s_lap%d'%(phase, expt, lap),
                        ax_pair=(xcorr_ax, pval_ax))

                xcorr_ax.axis('tight')
                xlim = (360 * min(hit_laps) - 180, 360 * max(hit_laps) + 180)
                xcorr_ax.set_xlim(xlim)
                pval_ax.set_ylim(1e-4, 10)
                pval_ax.set_xlim(xlim)

                ylim_max = xcorr_ax.get_ylim()[1]
                if ylim_max > ymax:
                    ymax = ylim_max

                xcorr_ax.set_xticks(np.array(hit_laps) * 360)
                xcorr_ax.set_xticklabels(map(str, hit_laps))

                xcorr_ax.tick_params(top=False, right=False, labelsize='x-small')
                pval_ax.tick_params(top=False, right=False, labelsize='x-small')

                if i == 0:
                    xcorr_ax.set_ylabel('Correlation', size='small')
                    pval_ax.set_ylabel('P-Value', size='small')
                else:
                    pval_ax.set_yticklabels([])

                if i == len(Scan.PhaseNames) - 1:
                    xcorr_ax.set_xlabel('Hit Lap', size='small')
                    pval_ax.set_xlabel('Hit Lap', size='small')

            # Normalize y-axis scale across phase plots
            ymax *= 1.05
            xcorr_axlist = AxesList()
            xcorr_axlist.add_figure(f_x)
            for i,ax in enumerate(xcorr_axlist):
                ax.set_ylim(0.0, ymax)

        plt.ion()
        plt.show()

    def process_predictive_value_across_laps(self, test_hit_laps=(-3,-2,-1,0,1,2,3),
        start_fresh=False):
        """Run the predictive value analysis for this for adaptive windows
        centered on a window of different laps
        """
        data_file = self.get_data_file(mode='a')
        root = data_file.root
        assert hasattr(root, 'events'), "please run data collection first"

        # Results = namedtuple("Results", "data ppv expected lower upper")
        PPVLapsDesc = {
            'expt'      : tb.StringCol(itemsize=4, pos=1),
            'phase'     : tb.StringCol(itemsize=16, pos=2),
            'lap'       : tb.Int16Col(pos=3),
            'ppv'       : tb.FloatCol(pos=4),
            'expected'  : tb.FloatCol(pos=5),
            'lower'     : tb.FloatCol(pos=6),
            'upper'     : tb.FloatCol(pos=7)
            }
        if not hasattr(data_file.root, 'acrosslaps') or \
            not hasattr(data_file.root.acrosslaps, 'ppv') or start_fresh:
            if hasattr(data_file.root, 'acrosslaps') and hasattr(data_file.root.acrosslaps, 'ppv'):
                data_file.removeNode(data_file.root.acrosslaps, 'ppv')
            data_file.createTable('/acrosslaps', 'ppv', PPVLapsDesc, createparents=True,
                title='Predictive Value Results Across Hit-Laps')
            self.out('Created new results table: %s'%data_file.root.acrosslaps.ppv._v_pathname)

        # If the /predictive table exists, rename it to its hit-lap descriptive name
        try:
            predictive_group = root.predictive
        except tb.NoSuchNodeError:
            pass
        else:
            data_file.renameNode(predictive_group, predictive_group._v_attrs['lap_name'])

        for hit_lap in test_hit_laps:
            self.out('Processing predictive tests for lap = %d'%hit_lap)

            # Move the new hit-lap predictive data group into the default /predictive node
            # or run the predictive tests if a previous data group does not exist
            predictive_group_name = self.predictive_group_name(hit_lap)
            data_file = self.get_data_file(mode='a')
            try:
                data_file.renameNode('/%s'%predictive_group_name, 'predictive')
            except tb.NoSuchNodeError:
                self.process_data(hit_lap=hit_lap)

            for expt in ('DR', 'NOV'): # ('ALL', 'DR', 'NOV', 'FIRST'):
                for phase in ('scan',): # Scan.PhaseNames:

                    # Check to see if data already exists
                    data_file = self.get_data_file()
                    results_query = '(expt=="%s")&(phase=="%s")&(lap==%d)'%(expt[:4], phase, hit_lap)
                    already_done = bool(len(data_file.root.acrosslaps.ppv.getWhereList(results_query)))
                    if already_done:
                        self.out('Found data: %s, %s, lap %d'%(expt, phase, hit_lap))
                        continue

                    # Run the predictive value analysis for the given experiment type
                    # and scan phase
                    PPV = self.predictive_values(phase=phase, which_panels=expt)[0]

                    # Add a new row with the results to the hit-lap results data table
                    data_file = self.get_data_file(mode='a')
                    row = data_file.root.acrosslaps.ppv.row
                    row['expt'] = expt
                    row['phase'] = phase
                    row['lap'] = hit_lap
                    row['ppv'] = PPV.ppv[0]
                    row['expected'] = PPV.expected[0]
                    row['lower'] = PPV.lower[0]
                    row['upper'] = PPV.upper[0]
                    row.append()
                    data_file.flush()

            # Rename predictive data group to be unique for each hit lap
            data_file = self.get_data_file(mode='a')
            data_file.renameNode('/predictive', predictive_group_name, overwrite=True)
            self.out('Renamed /predictive to %s'%getattr(
                data_file.root, predictive_group_name)._v_pathname)

        # Save the test hit laps for plotting
        save_fd = file(os.path.join(self.datadir, 'ppv_hit_laps.pickle'), 'w')
        cPickle.dump(test_hit_laps, save_fd)
        save_fd.close()

        self.close_data_file()

    def predictive_values_across_laps_plot(self, ymax=0.33, ynorm=False):
        """Plot the aggregate predictive-value data across laps in a single plot
        """
        hit_laps = np.array(np.load(os.path.join(self.datadir, 'ppv_hit_laps.pickle')))
        N_laps = len(hit_laps)

        plt.ioff()
        self.figure = {}
        fignum = 120
        rows, cols = 5, 2 #tiling_dims(len(Scan.PhaseNames))

        data_file = self.get_data_file()
        ppv_table = data_file.root.acrosslaps.ppv

        for expt in ('DR', 'NOV'): # ('ALL', 'DR', 'NOV', 'FIRST'):

            self.out('Creating figures for "%s" experiments.'%expt)

            figname = 'ppv_laps_%s'%expt
            self.figure[figname] = f = plt.figure(num=fignum, figsize=(8.5, 11))
            plt.clf()
            f.suptitle('Positive Predictive Values Across Hit Laps\n%s Experiments'%expt)
            fignum += 1

            for i, phase in enumerate(['scan']): # Scan.PhaseNames):

                self.out('Plotting "%s" scan phase...'%phase)

                ax = f.add_subplot(rows, cols, i+1)
                quicktitle(ax, phase.title(), size='small')

                PPV = np.empty(N_laps, 'd')
                E_PPV = np.empty(N_laps, 'd')
                CI_PPV = np.empty((2, N_laps), 'd')

                for j, lap in enumerate(hit_laps):
                    query = '(expt=="%s")&(phase=="%s")&(lap==%d)'%(expt[:4], phase, lap)
                    row = get_unique_row(ppv_table, query)
                    PPV[j] = row['ppv']
                    E_PPV[j] = row['expected']
                    CI_PPV[:,j] = row['expected'] - row['lower'], row['upper'] - row['expected']

                ax.plot(hit_laps, PPV, 'ro', mfc='none', mew=1.5,
                    mec='r', ms=6, zorder=10)
                ax.errorbar(hit_laps, E_PPV, yerr=CI_PPV,
                    fmt='k_', lw=1.0, ecolor='k', capsize=4,
                    mew=1.0, ms=6)

                ax.axis('tight')
                xlim = (hit_laps[0] - 0.5, hit_laps[-1] + 0.5)
                ax.set_xlim(xlim)

                if ynorm:
                    ylim_max = 1.1 * ax.get_ylim()[1]
                    if ylim_max > ymax:
                        ymax = ylim_max

                ax.set_xticks(hit_laps)
                ax.tick_params(top=False, right=False, direction='out', labelsize='small')

                if i == 0:
                    ax.set_ylabel('PPV', size='small')
                else:
                    ax.set_yticklabels([])

                if i == len(Scan.PhaseNames) - 1:
                    ax.set_xlabel('Hit Lap', size='small')

            # Normalize y-axis scale across phase plots
            ppv_axlist = AxesList()
            ppv_axlist.add_figure(f)
            ppv_axlist.apply(ylim=(0.0, ymax))

        plt.ion()
        plt.show()
        self.close_data_file()
        self.out('All done!')

    def effect_size_across_laps_plot(self, phase='scan'):
        """Plot the aggregate predictive-value data across laps in a single plot
        """
        self.out.outfd = file(os.path.join(self.datadir, 'effect_size.log'), 'w')
        self.out.timestamp = False

        hit_laps = np.array(np.load(os.path.join(self.datadir, 'xcorr_laps.pickle')))
        N_laps = len(hit_laps)

        plt.ioff()
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure['effect_size_laps'] = f = plt.figure(num=125, figsize=(8.5, 11))
        f.suptitle('Predictive Effect Sizes Across Hit Laps (%s Phase)' % phase.title())
        rows, cols = 5, 2
        plt.clf()

        data_file = self.get_data_file()
        ppv_table = data_file.root.acrosslaps.ppv

        ymin = np.inf
        ymax = 0.0

        post_es = np.array([])
        post_laps = hit_laps >= 0
        self.out('Using post laps: %s' % str(hit_laps[post_laps]))

        for i, expt in enumerate(['ALL', 'DR', 'NOV']):
            self.out('Creating plot for "%s" experiments.' % expt)

            ax = f.add_subplot(rows, cols, 2 * i + 1)
            quicktitle(ax, expt, size='small')

            delta = np.empty(N_laps, 'd')

            for j, lap in enumerate(hit_laps):
                query = '(expt=="%s")&(phase=="%s")&(lap==%d)'%(expt[:4], phase, lap)
                row = get_unique_row(ppv_table, query)
                PPV = row['ppv']
                E_PPV = row['expected']
                CI = row['lower'], row['upper']
                sigma = np.diff(CI) / (2 * 1.96)
                delta[j] = (PPV - E_PPV) / sigma
                if lap == -1:
                    self.out('Lap delta -1 = %f' % delta[j])

            if expt != 'ALL':
                post_es = np.r_[post_es, delta[post_laps]]

            ax.plot(hit_laps, delta, 'ro', mfc='none', mew=1.5,
                mec='r', ms=6, zorder=10)

            ax.axis('tight')
            xlim = (hit_laps[0] - 0.5, hit_laps[-1] + 0.5)
            ax.set_xlim(xlim)

            ylim_min, ylim_max = ax.get_ylim()
            if ylim_max > ymax:
                ymax = ylim_max
            if ylim_min < ymin:
                ymin = ylim_min

            ax.set_xticks(hit_laps)
            ax.tick_params(top=False, right=False, direction='out', labelsize='small')

            ax.set_ylabel(r'$\Delta$', size='small')

            if i == 2:
                ax.set_xlabel('Hit Lap', size='small')

        # Normalize y-axis scale across phase plots
        ppv_axlist = AxesList()
        ppv_axlist.add_figure(f)
        ydelta = ymax - ymin
        ppv_axlist.apply(ylim=(ymin - 0.1 * ydelta, ymax + 0.1 * ydelta))
        ppv_axlist.apply(func='axhline', c='k', lw=1, ls='-', zorder=-1)

        # Post-potentiation laps statistics
        self.out('Found %d post-potentiation lap effect sizes.' % len(post_es))
        self.out('Median %f, range %f to %f sigma' % (np.median(post_es), post_es.min(), post_es.max()))
        self.out('Mean +/- s.d. = %f +/- %f sigma' % (post_es.mean(), post_es.std()))
        self.out('Diff from 0: T(%d) = %f, p < %f' % ((len(post_es)-1,) + t_one_sample(post_es, 0.0)))
        Npos_neg = np.sum(post_es<0), np.sum(post_es>0)
        self.out('Binomial: <0: %d, >0: %d; p = %f' % (Npos_neg + (st.binom_test(),)))
        self.out('')

        plt.ion()
        plt.show()

        self.close_data_file()
        self.out.outfd.close()
        self.out('All done!')

    def scan_potentiation_timing(self):
        """Measure within-rat averages of the elapsed time between predictive scans
        and the corresponding events
        """
        self.out.outfd = file(os.path.join(self.datadir, 'scan_potentiation_timing.log'), 'w')
        self.out.timestamp = False

        data_file = self.get_data_file()
        posthoc_table = data_file.root.posthoc.observed
        observed_scan_table = data_file.root.observed_scans
        potentiation_table = get_node('/physiology', self.results['table'])
        scan_table = get_node('/behavior', 'scans')

        dt = {}
        for row in posthoc_table.iterrows():
            event_id = row['event_id']
            scan_id = row['scan_id']

            event = potentiation_table[event_id]
            scan = scan_table[observed_scan_table[scan_id]['scan_id']]

            dt_s = 1e-6 * (event['tlim'][0] - scan['start'])

            rat = event['rat']
            if rat not in dt:
                dt[rat] = []
            dt[rat].append(dt_s)

        for rat in dt.keys():
            dt[rat] = np.array(dt[rat])

        self.out('Found scan-potentiation dt for %d rats.' % len(dt))

        dtbar = np.array([a.mean() for a in dt.values()])
        self.out('Range from %f to %f s' % (dtbar.min(), dtbar.max()))
        self.out('Median dt = %f' % np.median(dtbar))
        self.out('Mean +/- s.d. = %f +/- %f s' % (dtbar.mean(), dtbar.std()))

        self.close_data_file()
