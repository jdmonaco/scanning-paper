# encoding: utf-8
"""
field_modulation.py -- Compute dependence of place field changes on firing
    during previous head scan events

Created by Joe Monaco on 2011-10-26.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division

# Library imports
import os, sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D

# Package imports
from scanr.config import Config
from .reports import BaseSessionReport, BaseDatasetReport
from .behavior_reports import ForwardSpeedReport
from scanr.session import SessionData
from scanr.cluster import AND, PlaceCellCriteria, ClusterData, get_min_quality_criterion
from scanr.spike import SpikePartition, TetrodeSelect, parse_cell_name, PrimaryAreas
from scanr.tracking import plot_track_underlay
from scanr.data import get_unique_row, unique_cells, new_table, get_node, dump_table, flush_file
from scanr.meta import get_maze_list
from scanr.time import time_slice_sample, select_from
from scanr.field import (mark_max_field, mark_all_fields, field_extent,
    cut_laps_opposite_field, center_of_mass, track_bins)
from .data_reports import TRAJ_FMT, SPIKE_FMT

# Local imports
from .core.analysis import AbstractAnalysis
from .core.report import BaseReport
from .tools.radians import (xy_to_rad_vec, xy_to_deg_vec, get_angle_array,
    circle_diff_vec_deg, circle_diff_vec, unwrap_deg, rot2D_vec)
from .tools.plot import grouped_bar_plot, AxesList, quicktitle, textlabel
from .tools.misc import Reify, DataSpreadsheet, contiguous_groups
from .tools.images import masked_array_to_rgba, rgba_to_image, array_to_image
from .tools.stats import zscore, sem, pvalue
from .tools.string import snake2title
from .tools import circstat

# Traits imports
from traits.api import HasTraits, Int, Float, List, Bool, Instance, Range

# Constants
CornuAmmonis = ['CA3', 'CA1']
HippocampalAreas = CornuAmmonis + ['DG']
DEFAULT_BINS = Config['placefield']['default_bins']
MIN_FIELD_RATE = Config['placefield']['min_peak_rate']

# Novel field formation table
FieldModDescr = {    'id'            :   tb.UInt16Col(pos=1),
                     'rat'           :   tb.UInt16Col(pos=2),
                     'day'           :   tb.UInt16Col(pos=3),
                     'session'       :   tb.UInt16Col(pos=4),
                     'type'          :   tb.StringCol(itemsize=4, pos=5),
                     'tetrode'       :   tb.UInt16Col(pos=6),
                     'cluster'       :   tb.UInt16Col(pos=7),
                     'tc'            :   tb.StringCol(itemsize=8, pos=8),
                     'area'          :   tb.StringCol(itemsize=4, pos=9),
                     'fieldnum'      :   tb.UInt16Col(pos=10),
                     'eventnum'      :   tb.UInt16Col(pos=11),
                     'lap'           :   tb.UInt16Col(pos=12),
                     'tlim'          :   tb.UInt64Col(shape=(2,), pos=13),
                     'tracklim'      :   tb.FloatCol(shape=(2,), pos=14),
                     'COM'           :   tb.FloatCol(pos=15),
                     'COM_field'     :   tb.FloatCol(pos=16),
                     'baseline'      :   tb.FloatCol(pos=17),
                     'strength'      :   tb.FloatCol(pos=18)    }

# Convenience functions

def count_spikes(ts, start, end):
    return len(time_slice_sample(ts, start=start, end=end))

def shuffle_interval(start, end, cut, block_start, duration):
    delta = end - start
    start -= cut
    if start < block_start:
        start += duration
    return start, start + delta

def process_area_argument(area):
    """Get list of areas from an optional recording area argument

    Values:
    individual area -- restrict to single (primary) area
    'ALL' -- restrict to hippocampal areas (DG/CA1/CA3)
    'CAX' -- restrict to CA1/CA3 (default)
    """
    if area is None or area == 'CAX':
        area = CornuAmmonis
    elif area == 'ALL':
        area = HippocampalAreas
    elif type(area) is str and area in PrimaryAreas:
        area = [area]
    else:
        assert type(area) is list, "bad area specification"
    return area

def get_area_query(area_list):
    return '|'.join(["(area=='%s')"%area for area in area_list])


def update_table_with_ids(table_name):
    """Add an 'id' column to an existing field modulation event table
    """
    event_table = get_node('/physiology', table_name)
    event_table.rename('%s_old'%table_name)
    updated_table = new_table('/physiology', table_name, FieldModDescr,
        title=event_table._v_title)
    row = updated_table.row
    cols = FieldModDescr.keys()
    event_id = 0
    for rec in event_table.iterrows():
        for k in cols:
            if k == 'id':
                row[k] = event_id
            else:
                row[k] = rec[k]
        row.append()
        event_id += 1
        if event_id % 10 == 0:
            update_table.flush()
        sys.stdout.write('.')
    flush_file()
    sys.stdout.write('\nAll done!\n')


class GoldStandard(HasTraits):

    bins = Int(DEFAULT_BINS, param=True)
    min_field_rate = Float(MIN_FIELD_RATE, param=True)

    min_baseline = Int(2, param=True)
    max_baseline = Int(5, param=True) # set to >15 for full baseline check
    min_following = Int(2, param=True)
    min_following_hits = Int(3, param=True)
    max_following = Int(5, param=True)
    min_event_spikes = Int(2, param=True)

    min_dp = Float(0.1, param=True)
    max_dp = Float(0.1, param=True)
    relative_dp = Float(0.5, param=True)
    relative_tol = Float(0.3, param=True)
    allow_multiple_events = Bool(True, param=True)
    min_laps_between_events = Range(low=1, high=20, value=4, param=True)
    allow_multiple_fields = Bool(True, param=True)

    reverse_search = Bool(False, param=True)

    last_test_results = List
    debug = Bool(False)

    def parameters(self):
        """Dictionary of current gold-standard test parameters
        """
        return { k: getattr(self, k) for k in self.traits(param=True).keys() }

    def field_modulation(self, rds, tc, bins=None):
        """Test for mid-session appearance of a new place field, returning a
        (lap_index, track_angle) tuple of the field's appearance if so.

        Arguments:
        rds -- recording session to be searched for potentiation
        tc -- name of the cell (e.g., 't1c3') to be analyzed
        bins -- number of bins to use for dot-product computations

        NOTE: Session data laps partitions are recomputed here based on the
        COM location of the maximal place field.
        """
        if bins is None:
            bins = self.bins
        def fail(why, lap=None):
            if lap is None:
                pre = ''
            else:
                pre = 'lap %d: '%(lap + 1)
            if self.debug:
                sys.stdout.write('failed: %s%s\n'%(pre, why))
        def putative(lap):
            if self.debug:
                sys.stdout.write('found: lap %d: putative event\n'%(lap+1))

        # Construct the scan and pause list for spike filtering
        data = SessionData.get(rds)

        # Load the cluster data and compute full-session place field
        cluster = data.clusts[tc]
        bin_width = 360.0 / bins
        filter_kwds = dict( velocity_filter=True,
                            exclude_off_track=True,
                            exclude=data.extended_scan_and_pause_list   )
        ratemap_kwds = dict(bins=bins, blur_width=bin_width)
        ratemap_kwds.update(filter_kwds)
        R_full = data.get_cluster_ratemap(tc, **ratemap_kwds)

        # Cell-specific (field-independent) data
        cell_data = dict(
            R_full=R_full,
            traj_time=data.T_(data.trajectory.ts),
            traj_alpha=data.trajectory.alpha,
            all_spike_times=data.T_(cluster.spikes),
            all_spike_alpha=xy_to_deg_vec(cluster.x, cluster.y))
        cell_results = [cell_data]

        # Fail if place-cell response not strong enough
        if R_full.max() < self.min_field_rate:
            fail('no field: max rate %.3f < %.3f'%(R_full.max(), self.min_field_rate))
            return cell_results

        # Restrict test to ratemaps with singular place field
        marked_fields = mark_all_fields(R_full)
        N_fields = marked_fields.shape[0]
        if not self.allow_multiple_fields and N_fields > 1:
            fail('detected multiple (%d) fields'%N_fields)
            return cell_results

        fieldnum = 0
        for field in marked_fields:

            # Field-specific results dict with events list, dict added to cell-
            # specific results list
            fieldnum += 1
            results = dict(fieldnum=fieldnum, success=False)
            events = results['events'] = []
            cell_results.append(results)

            if self.debug:
                sys.stdout.write('---\n')

            # Compute COM, cut laps opposite field COM, generate ratemaps
            COM = center_of_mass(R_full, F=field, degrees=True)
            data._compute_laps(cut=(np.pi / 180) * (COM - 180))
            R_lap = data.get_population_lap_matrix(clusters=tc,
                **ratemap_kwds).squeeze().T

            # Compute dot products and truncate an empty final lap
            Rl, Rf = R_lap[:,field], R_full[field]
            DP = np.dot(Rl, Rf) / np.dot(Rf, Rf)

            if self.reverse_search: # depotentiation
                DP = DP[::-1]

            if DP[-2] >= self.min_dp and DP[-1] < self.max_dp:
                R_lap = R_lap[:-1]
                DP = DP[:-1]
            laps, bins = R_lap.shape

            # Save event-independent data about this field
            start, end = field_extent(field)
            results['fieldlim'] = start, end
            results['COM'] = COM
            results['R_lap'] = R_lap
            results['DP'] = DP
            if self.reverse_search:
                results['DP'] = results['DP'][::-1]

            # Loop through possible non-baseline/followup laps
            skip = 0
            for lap_ix in xrange(self.min_baseline, laps - self.min_following):

                # Skip laps since last detected event if specified
                if skip:
                    fail('skipping %d more lap%s'%(skip,
                        { True: 's', False: '' }[skip > 1]), lap_ix)
                    skip -= 1
                    continue

                # Set the double threshold, according to relative or absolute
                lap_adjust = 0.5 + (lap_ix + 1) / float(laps) # parity @ half-way
                if self.relative_dp:
                    max_dp = DP[lap_ix] - lap_adjust * self.relative_dp
                else:
                    max_dp = lap_adjust * self.max_dp

                # Check baseline criteria (relatively low field strength)
                baseline_start = max(0, lap_ix - self.max_baseline)
                baseline_max = DP[baseline_start:lap_ix].max()
                baseline_check = (baseline_max < max_dp)
                if not baseline_check:
                    fail('DP baseline %.3f [%.3f]'%(baseline_max, max_dp),
                        lap_ix)
                    continue

                # Check for appearance of the field
                if self.relative_dp: # adaptive relative follow-up threshold:
                    min_dp = (DP[lap_ix]
                        - self.relative_tol * (DP[lap_ix] - baseline_max))
                else:
                    min_dp = lap_adjust * self.min_dp
                appearance_check = (DP[lap_ix] != 0) and (DP[lap_ix] >= min_dp)
                if not appearance_check:
                    fail('appearance: DP %.3f [%.3f]'%(DP[lap_ix], min_dp),
                        lap_ix)
                    continue

                # Check followup criteria (field activity on following laps)
                following_laps = min(laps - lap_ix - 1, self.max_following)
                required_laps = min(following_laps, self.min_following_hits)
                follow_dp = DP[lap_ix+1:lap_ix+following_laps+1]
                followup_check = np.sum(follow_dp >= min_dp) >= required_laps

                if not followup_check:
                    fail('followup: %d/%d required laps [>%.3f]'%(
                        np.sum(follow_dp > min_dp),
                        required_laps, min_dp), lap_ix)
                    fail('follow_corr: %s'%str(follow_dp))
                    continue

                # After passing baseline, appearance, and followup tests, append
                # the resulting event to the events list and then continue
                # detection unless allow_multiple_events is False.
                events.append(dict(lap=lap_ix, min_dp=min_dp, max_dp=max_dp,
                    DP_baseline=baseline_max, DP_potentiation=DP[lap_ix]))
                putative(lap_ix)

                if self.allow_multiple_events:
                    skip = self.min_laps_between_events - 1
                else:
                    break

            found = bool(len(events))
            if self.debug:
                sys.stdout.write('---\n')
            if not found:
                fail('all laps failed')
                continue

            # Swing back around if reversed for depotentiation search
            if self.reverse_search:
                for event in events:
                    #NOTE: The "- 1" is provisional here as it would indicate the strong
                    # field on lap prior to depotentiation after re-reversal; this may
                    # be best (only?) way to get the rest of the event metadata as normal,
                    # otherwise, in the case of fields turning off, there will be no
                    # specific data points to use as markers for the event
                    event['lap'] = data.N_laps - event['lap'] - 1

            # Initial event-independent post-processing:
            alpha_test = (end < start) and np.logical_or or np.logical_and
            alpha = xy_to_deg_vec(cluster.x, cluster.y)
            filter_kwds.update(boolean_index=True)
            in_field = np.logical_and(
                        alpha_test(alpha >= start, alpha <= end),
                        data.filter_tracking_data(cluster.spikes, cluster.x,
                            cluster.y, **filter_kwds))

            # Event-dependent post-processing:
            empty = []
            eventnum = 0
            for i, event in enumerate(events):

                # Get the first-lap in-field spike data
                in_lap = select_from(cluster.spikes,
                    [(data.laps[event['lap']], data.laps[event['lap']+1])])
                spike_ix = np.logical_and(in_field, in_lap)
                if not spike_ix.any():
                    fail('event %d had no spikes'%(i+1))
                    empty.append(i)
                    continue

                ts_spikes = cluster.spikes[spike_ix]
                alpha_spikes = alpha[spike_ix]

                # Find earliest field traversal within the detected lap
                alpha_unwrap = data.F_('alpha_unwrapped')(data.T_(ts_spikes))
                next_lap_spikes = (alpha_unwrap[0] - alpha_unwrap > 180).nonzero()[0]
                if len(next_lap_spikes):
                    ts_spikes = ts_spikes[:next_lap_spikes[0]]
                    alpha_spikes = alpha_spikes[:next_lap_spikes[0]]

                # Exclude low-spike events
                if len(ts_spikes) < self.min_event_spikes:
                    fail('event %d had %d<%d spikes'%(i+1, len(ts_spikes),
                        self.min_event_spikes))
                    empty.append(i)
                    continue

                # Place-field event is now considered a hit, so save the data
                event['tlim'] = ts_spikes[0], ts_spikes[-1]
                event['tracklim'] = alpha_spikes[0], alpha_spikes[-1]
                event['spikes'] = ts_spikes
                event['alpha'] = alpha_spikes
                event['COM'] = (180/np.pi) * circstat.mean(
                    (np.pi/180) * alpha_spikes)
                event['spike_times'] = data.T_(ts_spikes)

                eventnum += 1
                event['eventnum'] = eventnum

                if self.debug:
                    sys.stdout.write('Field modulation on lap %d at %.3f degrees\n'%(
                        event['lap']+1, event['COM']))

            [events.pop(i) for i in reversed(empty)]
            results['success'] = bool(len(events))

        return cell_results

    def test_field_modulation(self, rds, tc, bins=None):
        """Debugging test for field_modulation algorithm
        """
        self.debug = True
        results = self.field_modulation(rds, tc, bins=bins)
        self.last_test_results = results
        self.debug = False

        # Split out the cell data from the field results dictionaries
        cell = results.pop(0)
        field_results = results
        del results

        if not len(field_results):
            sys.stdout.write('No place fields were found in the response.\n')
            return

        plt.ioff()
        for results in field_results:
            N_laps, bins = results['R_lap'].shape

            # Create a figure with test data
            f = plt.figure(100 + results['fieldnum'], figsize=(12,8))
            f.clf()
            f.suptitle('Field %d Test - Rat %d, Day %d, Maze %d, Cell %s'%(
                (results['fieldnum'],) + rds + (tc,)))
            angles = track_bins(bins, degrees=True)
            laps = np.arange(1, N_laps + 1)
            crosshairs = dict(color='w', lw=1.5, alpha=0.8)

            # The place field ratemap
            ax = plt.subplot(221)
            ax.plot(angles, cell['R_full'], 'k-', lw=2)
            ax.set_xlim(angles[0], angles[-1])
            ax.set_ylim(bottom=0)
            ax.set_ylabel('Firing Rate (spk/s)')
            ax.axvline(results['COM'], ls='--', c='k')
            if 'fieldlim' in results:
                start, end = results['fieldlim']
                ax.axvline(start, ls='-', c='g')
                ax.axvline(end, ls='-', c='r')
            quicktitle(ax, 'Full Session Ratemap', size='small')

            # The lap matrix with field modulation events highlighted
            ax = plt.subplot(222)
            ax.imshow(results['R_lap'].T, origin='lower', aspect='auto',
                extent=[0.5, N_laps+0.5, 0, 360],
                interpolation='nearest')
            ax.set_xticks([1, N_laps])
            ax.set_yticks([0, 360])

            quicktitle(ax, 'Lap-by-Lap Matrix', size='small')
            if results['success']:
                for event in results['events']:
                    ax.axhline(event['COM'], **crosshairs)
                    ax.axvline(event['lap']+1, **crosshairs)
                    ax.plot([event['lap']+1],  [event['COM']], marker='o', mew=0,
                        ms=8, mfc='w', mec='w', alpha=0.8)

            # The dot-product and field-strength measures
            ax = plt.subplot(224)
            ax.plot(laps, results['DP'], 'b-o')
            if results['success']:
                linekwds = dict(ls='-', color='b', lw=2)
                arrowkwds = dict(mfc='b', mec='b')
                for event in results['events']:
                    x = event['lap']+0.5, event['lap']+1.5
                    ax.plot(x, [event['min_dp']]*2, **linekwds)
                    ax.plot(x, [event['max_dp']]*2, **linekwds)
                    if self.reverse_search:
                        ax.plot([event['lap']+1.5], [event['max_dp']], '>', **arrowkwds)
                        ax.plot([event['lap']+0.5], [event['min_dp']], '<', **arrowkwds)
                    else:
                        ax.plot([event['lap']+1.5], [event['min_dp']], '>', **arrowkwds)
                        ax.plot([event['lap']+0.5], [event['max_dp']], '<', **arrowkwds)
            ax.set_ylabel('D.P.', color='b')
            ax.set_xlim(0.5, N_laps+0.5)
            quicktitle(ax, 'Lap-by-Lap Dot Products', size='small')

            # Track-angle x time spike plot
            fmt = SPIKE_FMT.copy()
            fmt['s'] = 20
            fmt['alpha'] = 1.0
            ax = plt.subplot(223)
            ax.plot(cell['traj_time'], cell['traj_alpha'], 'k-', lw=0.75)
            ax.scatter(cell['all_spike_times'], cell['all_spike_alpha'],
                **fmt)
            if results['success']:
                fmt['edgecolor'] = 'b'
                fmt['zorder'] = 10
                for event in results['events']:
                    ax.scatter(event['spike_times'], event['alpha'], **fmt)
            ax.set_xlim(cell['traj_time'][0], cell['traj_time'][-1])
            ax.set_ylim(0, 360)
            quicktitle(ax, 'Track-angle Spiking Time-series')

        plt.ion()
        plt.show()


class FindFieldModulation(AbstractAnalysis, TetrodeSelect):

    """
    Use gold standard field formation test to scan all place cell lap matrices
    for mid-session field formation.
    """

    label = 'field events'

    def collect_data(self, area='CAX', min_quality='fair', table='potentiation',
        **test_params):
        """Scan all hippocampal data sets for novel field formation and store
        the results in a spreadsheet and data table.

        Arguments:
        area -- string name or list of names of hippocampal recording areas,
            'ALL' for all hippocampal areas, 'CAX' for ammonis, default 'CAX'
        min_quality -- minimum cluster isolation quality for including cells
        table -- configured table key or table name to store successful tests

        Remaining keyword arguments set GoldStandard test parameters.
        """
        area_list = process_area_argument(area)
        area_query = get_area_query(area_list)
        self.out('Processing recording areas: %s'%str(area_list)[1:-1])
        self.results['area'] = area_list
        self.results['quality'] = min_quality
        tetrode_table = get_node('/metadata', 'tetrodes')

        Quality = get_min_quality_criterion(min_quality)
        self.out('Quality threshold: at least %s'%min_quality)

        self.results['table_name'] = table
        self.results['fields'] = fields = []

        # Set up new field formation table
        field_table = new_table('/physiology', table, FieldModDescr,
            title='Place Field Modulation [%s]'%snake2title(table))
        field_row = field_table.row

        # Set up the gold standard field formation test and store the test
        # parameters in the data table attributes
        param_str = 'Gold standard parameters:\n'
        gold_std = GoldStandard(**test_params)
        params = gold_std.parameters()
        for k in params:
            field_table._v_attrs[k] = params[k]
            param_str += '  *  %s = %s\n'%(k, str(params[k]))
        self.out(param_str)

        # Build the session list based on hippocampal datasets
        session_list = []
        for rat, day in TetrodeSelect.datasets(area_query):
            session_list.extend([(rat, day, maze) for maze in
                get_maze_list(rat, day)])

        # Set up session data accumulators
        fields_tested = []
        positives = []
        rat_number = []
        day_number = []
        session_type = []
        mismatch_angle = []
        maze_number = []
        type_number = []

        area_counts = {a: {'fields': 0, 'events': 0} for a in area_list}

        event_id = 0
        for rds in session_list:
            rat, day, maze = rds
            dataset = (rat, day)
            self.out('Analyzing rat%03d-%02d m%d session...'%rds)

            # Set cluster criteria and load session data
            Tetrodes = TetrodeSelect.criterion(dataset, area_query,
                allow_ambiguous=True)
            Criteria = AND(Quality, Tetrodes, PlaceCellCriteria)
            session_data = SessionData.get(rds)
            session_data.cluster_criteria = Criteria
            attrs = session_data.data_group._v_attrs

            # Loop through place cells
            field_count = 0
            event_count = 0
            place_cell_list = session_data.get_clusters()
            self.out.printf('Scanning: ', color='lightgray')
            for tc in place_cell_list:
                cluster = session_data.cluster_data(tc)
                tt, cl = parse_cell_name(tc)
                area = get_unique_row(tetrode_table, '(rat==%d)&(day==%d)&(tt==%d)'%(rat, day, tt))['area']

                test_results = gold_std.field_modulation(rds, tc)
                cell_data = test_results.pop(0)

                for results in test_results:
                    area_counts[area]['fields'] += 1
                    field_count += 1
                    if results['success']:
                        N_events = len(results['events'])
                        if N_events == 1:
                            self.out.printf(u'\u25a1', color='green')
                        elif N_events == 2:
                            self.out.printf(u'\u25a0', color='green')
                        elif N_events > 2:
                            self.out.printf(u'\u25c6', color='green')
                        event_count += N_events
                    else:
                        self.out.printf(u'\u25a1', color='red')
                        continue

                    # Save the successful test results
                    fields.append((rds+(tc,), results))

                    # Update table and spreadsheet records
                    for i, event in enumerate(results['events']):
                        field_row['id'] = event_id
                        field_row['rat'] = rat
                        field_row['day'] = day
                        field_row['session'] = maze
                        field_row['type'] = attrs['type']
                        field_row['tetrode'] = tt
                        field_row['cluster'] = cl
                        field_row['tc'] = tc
                        field_row['area'] = area
                        field_row['fieldnum'] = results['fieldnum']
                        field_row['eventnum'] = i + 1
                        field_row['lap'] = event['lap']
                        field_row['tlim'] = event['tlim']
                        field_row['tracklim'] = event['tracklim']
                        field_row['COM'] = event['COM']
                        field_row['COM_field'] = results['COM']
                        field_row['baseline'] = event['DP_baseline']
                        field_row['strength'] = event['DP_potentiation']
                        field_row.append()
                        event_id += 1
                        area_counts[area]['events'] += 1

            self.out.printf('\t(%d/%d)\n'%(event_count, field_count),
                color='lightgray')
            field_table.flush()

            # Update session data accumulators
            fields_tested.append(field_count)
            positives.append(event_count)

            rat_number.append(rat)
            day_number.append(day)
            session_type.append(attrs['type'])
            mismatch_angle.append(attrs['parameter'])
            maze_number.append(maze)
            type_number.append(attrs['number'])

        # Create data arrays
        self.results['fields_tested'] = np.asarray(fields_tested)
        self.results['positives'] = np.asarray(positives)
        self.results['rat_number'] = np.asarray(rat_number)
        self.results['day_number'] = np.asarray(day_number)
        self.results['session_type'] = np.asarray(session_type)
        self.results['mismatch_angle'] = np.asarray(mismatch_angle)
        self.results['maze_number'] = np.asarray(maze_number)
        self.results['type_number'] = np.asarray(type_number)

        # Save out area counts
        for a in area_list:
            for which in ('fields', 'events'):
                key = '%s_%s' % (a, which)
                self.results[key] = area_counts[a][which]
                self.out('%s %s = %d' % (a, which, self.results[key]))

        # Flush the new table and dump to a spreadsheet
        flush_file()
        # dump_table('/physiology', name=table,
        #     filename=os.path.join(self.datadir, '%s.csv'%field_table._v_name))

        self.out('All done!')

    def process_data(self):
        self.out.outfd = file(os.path.join(self.datadir, 'figure.log'), 'w')
        self.out.timestamp = False

        self.figure = {}
        plt.ioff()

        # Load up session data
        rat_number = self.results['rat_number']
        day_number = self.results['day_number']
        session_type = self.results['session_type']
        mismatch_angle = self.results['mismatch_angle']
        maze_number = self.results['maze_number']
        type_number = self.results['type_number']

        # Output totals
        overall_fields_tested = self.results['fields_tested'].sum()
        overall_positives = self.results['positives'].sum()
        self.out('Number cell-sessions tested: %d'%overall_fields_tested)
        self.out('Number positive cell-sessions: %d'%overall_positives)
        self.out('Overall prevalence: %.2f%%'%(
            100*overall_positives/float(overall_fields_tested)))

        # Set up positive detection bar plots
        def do_positives_bar_plot(ax, values, get_selection, label_str):
            data = np.zeros(len(values), 'd')
            self.out('-'*50)
            for j, value_color in enumerate(values):
                if type(value_color) is tuple:
                    value = value_color[0]
                else:
                    value = value_color
                ix = get_selection(value)
                NP = self.results['positives'][ix]
                N = self.results['fields_tested'][ix]
                valid = N != 0
                if not float(N[valid].sum()):
                    continue
                NPsum = NP[valid].sum()
                Nsum = N[valid].sum()
                fraction = NPsum / float(Nsum)
                self.out('%s: %d / %d = %.4f'%(label_str%value, NPsum, Nsum,
                    fraction))
                data[j] = fraction

            grouped_bar_plot(data, 'Sessions', values, ax=ax, label_str=label_str)
            ax.set_ylim(bottom=0)
            ax.set_ylabel('Positive Fraction')

        def make_positives_figures(figname, breakdown, values, selector,
            label_str='%s'):
            self.figure[figname] = f = plt.figure(figsize=(9, 7))
            f.suptitle('Novel Field Fraction Across %s'%breakdown)
            do_positives_bar_plot(plt.axes(), values, selector, label_str)
            return

        # STD/MIS figures
        make_positives_figures('maze_type', 'Session Type',
            [('STD', 'b'), ('MIS', 'g'), ('FAM', 'c'), ('NOV', 'y')],
            lambda value: session_type == value)
                #np.logical_and(maze_number != 1, session_type == value))

        # Maze number figures (DR)
        make_positives_figures('maze_number_DR', 'DR Maze Number',
            [(1,'b'), (2,'g'), (3,'r'), (4,'c'), (5,'m')],
            lambda value: np.logical_and(maze_number == value,
                np.logical_or(session_type == 'STD', session_type == 'MIS')),
            label_str='m%d')

        # Maze number figures (NOV)
        make_positives_figures('maze_number_NOV', 'Novelty Maze Number',
            [(1,'b'), (2,'g'), (3,'r')],
            lambda value: np.logical_and(maze_number == value,
                np.logical_or(session_type == 'FAM', session_type == 'NOV')),
            label_str='m%d')

        # Mismatch angle figures
        make_positives_figures('mismatch_angle', 'Mismatch Angle',
            [(45,'b'), (90,'g'), (135,'r'), (180,'c')],
            lambda value: (mismatch_angle == value),
            label_str='MIS-%d')

        # STD repetition number figures
        make_positives_figures('std_number', 'STD Repetitions',
            [(1,'b'), (2,'g'), (3,'r')],
            lambda num: np.logical_and(session_type=='STD', type_number==num),
            label_str='STD-%d')

        # Rat number figure
        make_positives_figures('rat_number', 'Rats',
            np.sort(np.unique(rat_number)),
            lambda num: (rat_number==num),
            label_str='Rat %03d')

        # Day number figure
        make_positives_figures('day_number', 'Days',
            range(1, 9),
            lambda num: (day_number==num),
            label_str='Day %d')

        plt.ion()
        plt.show()
        self.out.outfd.close()
        self.out('All done!')


def diff_modulation_tables(table1, table2):
    """Given two place-field modulation (ana.field_modulation.FieldModDescr)
    Table objects, report presence/absence differences between them

    Return tuple of number of events missing from (table2, table1).
    """
    write = lambda msg: sys.stdout.write(msg + '\n')

    write('Table 1: %s :: %s' % (table1._v_file.filename, table1._v_pathname))
    write('Table 2: %s :: %s' % (table2._v_file.filename, table2._v_pathname))
    write('')

    def unique_events(table):
        events = set()
        for row in table.iterrows():
            rds = row['rat'], row['day'], row['session']
            tc = row['tc']
            field = row['fieldnum']
            lap = row['lap']
            events.add((rds, tc, field, lap))
        return events

    events1 = unique_events(table1)
    events2 = unique_events(table2)

    diff1 = events1.difference(events2)
    diff2 = events2.difference(events1)

    def print_diffs(diff):
        for i, event in enumerate(sorted(diff)):
            session = 'Rat%03d-%02d-m%d' % event[0]
            write(' - %s %s field %d lap %2d' % (
                session, event[1].ljust(6), event[2], event[3]))

    write('Present in Table 1, but not Table 2: %s events' % len(diff1))
    print_diffs(diff1)

    write('\nPresent in Table 2, but not Table 1: %s events' % len(diff2))
    print_diffs(diff2)

    return len(diff1), len(diff2)


class ModulationSpeedReport(ForwardSpeedReport):

    """
    Show statistics of forward-running speed for the datasets where novel
    fields were detected
    """

    xnorm = False

    def collect_data(self, table_name='potentiation'):
        """Display statistics of forward-running speed distributions for each
        lap of the datasets with novel field formation detected.
        """
        # Get the unique set of datasets
        datasets = []
        ftable = get_node('/physiology', table_name)
        for row in ftable.iterrows():
            rds = tuple(int(v) for v in (row['rat'], row['day'],
                row['session']))
            if rds not in datasets:
                datasets.append(rds)

        # Call the superclass with the session list
        super(PlaceFieldSpeedReport, self).collect_data(datasets)


class EnsembleSpikesReport(AbstractAnalysis):

    """
    Full display of lap-spikes plots for a simultaneous ensemble of cells
    """

    label = 'ensemble'
    figwidth = Float(7.4409448819) #double-column width
    figheight = Float(9.7244094488) #column height
    nrows = Int(5)
    ncols = Int(5)

    def collect_data(self, rds, cell_list, lap_cut=0.0, elev=18.0, azim=108.0,
        z_pow=0.1):
        """Plot stacked 3D plots of cells in the ensemble specified by a given
        session and list of cell names.
        """
        rat, day, session = rds
        ensemble = dict(rat=rat, day=day, session=session, celldesc=', '.join(cell_list))

        session_data = SessionData.get(rds)
        session_data._compute_laps(cut=lap_cut)
        filters = session_data.running_filter()
        traj = session_data.trajectory
        clusters = [session_data.cluster_data(tc) for tc in cell_list]
        N_cells = len(clusters)

        title_desc = 'Rat %(rat)d, Day %(day)d, Maze %(session)d\nCells %(celldesc)s'%ensemble
        desc = '%(rat)03d-%(day)02d-m%(session)d'%ensemble

        # 3D stacked trajectory/spike plots
        self.out('Creating 3D stacked spike plots...')

        plt.ioff()
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure[desc] = f = \
            plt.figure(figsize=(self.figwidth, self.figheight))
        f.suptitle(title_desc)

        N_laps = session_data.N_laps
        lap_list = range(1, N_laps+1)

        # Plot style formatting dictionaries
        track_fmt = dict(ec='0.7', lw=0.4, ls='solid')

        traj_fmt = TRAJ_FMT.copy()
        traj_fmt.update(zorder=-50, lw=0.35, c='0.3', alpha=0.8)

        old_colors = dict(pause=SpikePartition.color_dict['pause'],
            other=SpikePartition.color_dict['other'])
        SpikePartition.color_dict['pause'] = 'none'
        SpikePartition.color_dict['other'] = 'none'

        for i, lap in enumerate(lap_list):

            subp = 4, 5, lap
            ax = f.add_subplot(*subp, projection='3d')
            ax.view_init(elev=elev, azim=azim)

            tlim = session_data.laps[lap-1:lap+1]
            self.out('Plotting lap %d: from %d to %d'%(lap, tlim[0], tlim[1]))

            # Plot track underlay
            plot_track_underlay(ax, **track_fmt)

            # Plot lap trajectory
            traj_ix = np.logical_and(traj.ts>=tlim[0], traj.ts<=tlim[1])
            traj_x, traj_y = traj.x[traj_ix], traj.y[traj_ix]

            for j, cluster in enumerate(clusters):
                zplane = -0.013 * np.power(j, 2.0) - 0.25 * j + 1.0
                self.out('%d: z = %f'%(j, zplane))

                ax.plot(traj_x, traj_y, zs=zplane, **traj_fmt)

                # Plot spike partition limited to the current lap
                spike_ix = np.logical_and(
                    cluster.spikes>=tlim[0], cluster.spikes<=tlim[1])
                txy = cluster.spikes, cluster.x, cluster.y
                cluster.spikes, cluster.x, cluster.y = (
                    cluster.spikes[spike_ix], cluster.x[spike_ix], cluster.y[spike_ix])
                SpikePartition.plot(session_data, cluster, ax=ax,
                    z=np.zeros_like(cluster.x) + zplane,
                    scan_extensions=True, mod_table=None,
                    marker=',', linewidths=2, facecolor='none')

                ax.axis([-50, 50, -50, 50])
                ax.set_zlim(zplane, 1.0)
                ax.set_axis_off()

                # Restore full cluster data
                cluster.spikes, cluster.x, cluster.y = txy

        # Unwrapped spike raster plots
        self.out('Creating spike raster plots...')

        self.figure[desc+'_raster'] = f = \
            plt.figure(figsize=(self.figwidth, self.figheight))
        f.suptitle(title_desc + '\nSpike Raster')
        ax = f.add_subplot(511)
        bins = np.linspace(0, 1, N_cells+1)
        centers = (bins[:-1] + bins[1:]) / 2

        ax.hlines(centers, 0, N_laps, colors='0.3', lw=0.5, zorder=-1)
        ax.vlines(lap_list, 0, 1, colors='0.75', lw=0.5, zorder=-2)
        for j, cluster in enumerate(clusters[::-1]):
            self.out('Plotting cell %s...'%cluster.name)
            SpikePartition.raster(session_data, cluster, ax=ax,
                ylim=(bins[j]+0.1/N_cells, bins[j+1]-0.1/N_cells), lw=0.35)

        ax.set_xlim(0.0, N_laps)
        ax.set_xlabel('Unwrapped Laps')
        ax.set_xticks(range(N_laps))
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('Cells')
        ax.set_yticks(centers)
        ax.set_yticklabels([cluster.name for cluster in clusters][::-1], size='small')
        ax.tick_params(left=False, bottom=False, top=False, right=False)

        SpikePartition.color_dict.update(old_colors)

        self.results['there_are_no_results'] = True
        self.out('All done!')
        plt.ion()
        plt.show()

    def laps_by_cells_figure(self, rds, cell_list, lap_cut=0.0, ycompress=1.0):
        """Plot stacked lap plots for each cell in the ensemble specified by a given
        session and list of cell names.
        """
        rat, day, session = rds
        ensemble = dict(rat=rat, day=day, session=session, celldesc=', '.join(cell_list))

        session_data = SessionData(rds=rds)
        session_data._compute_laps(cut=lap_cut)
        filters = session_data.running_filter()
        traj = session_data.trajectory
        clusters = [session_data.cluster_data(tc) for tc in cell_list]
        self.ncols = N_cells = len(clusters)

        title_desc = 'Rat %(rat)d, Day %(day)d, Maze %(session)d\nCells %(celldesc)s'%ensemble
        desc = 'laps-%(rat)03d-%(day)02d-m%(session)d'%ensemble

        # Stacked trajectory/spike plots
        self.out('Creating stacked laps-by-cells stacked spike plots...')

        plt.ioff()
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure[desc] = f = \
            plt.figure(figsize=(self.figwidth, self.figheight))
        f.suptitle(title_desc)

        N_laps = session_data.N_laps
        lap_list = range(1, N_laps+1)
        self.nrows = 1

        # Plot style formatting dictionaries
        track_fmt = dict(ec='0.7', lw=0.4, ls='solid')
        traj_fmt = TRAJ_FMT.copy()
        traj_fmt.update(zorder=-50, lw=0.35, c='0.3', alpha=0.8)
        old_colors = dict(pause=SpikePartition.color_dict['pause'],
            other=SpikePartition.color_dict['other'])
        SpikePartition.color_dict['pause'] = 'none'
        SpikePartition.color_dict['other'] = 'none'

        for j, cluster in enumerate(clusters):
            self.out('Plotting cell %d: %s'%(j+1, cluster.name))
            ax = f.add_subplot(1, N_cells, j+1)

            y0 = 0.0
            for i, lap in enumerate(lap_list):
                ytrans = lambda y: ycompress * y + y0

                # Plot lap trajectory
                tlim = session_data.laps[lap-1:lap+1]
                traj_ix = np.logical_and(traj.ts>=tlim[0], traj.ts<=tlim[1])
                traj_x, traj_y = traj.x[traj_ix], ytrans(traj.y[traj_ix])

                ax.plot(traj_x, traj_y, **traj_fmt)

                # Plot spike partition limited to the current lap
                spike_ix = np.logical_and(
                    cluster.spikes>=tlim[0], cluster.spikes<=tlim[1])
                cluster_lap = ClusterData(spikes=cluster.spikes[spike_ix],
                    x=cluster.x[spike_ix], y=cluster.y[spike_ix],
                    name=cluster.name)
                SpikePartition.plot(session_data, cluster_lap,
                    y=ytrans(cluster_lap.y), ax=ax)

                # Step down the y-axis for the next lap plot
                y0 -= 100

            ax.set_xlim(-50, 50)
            ax.set_ylim(y0 + 25, 75)
            ax.set_axis_off()
            quicktitle(ax, cluster.name)

        SpikePartition.color_dict.update(old_colors)

        self.out('All done!')
        plt.ion()
        plt.show()
        self.finished = True
        self.results['created_stacked_plots'] = True


class FullSpikesReport(AbstractAnalysis):

    """
    Full display of lap-spikes plots of a single potentiation event
    """

    label = 'full example'
    figwidth = Float(7.4409448819) #double-column width
    figheight = Float(9.7244094488) #column height
    nrows = Int(6)
    ncols = Int(6)

    def collect_data(self, event_key=(119,8,2,'t18c1'), COM=None, zoom_lap=6, rotation=180):
        """Specify the place-field potentiation event as either a cell-ID tuple
        structured as (rat, day, session, tc) or an integer index into the
        /physiology/potentiation events table.

        If using a tuple to specify an arbitrary field, the COM parameter must
        be passed in specifying the center track-angle position of the field
        in radians.
        """
        if type(event_key) is int:
            event = get_node('/physiology', 'potentiation')[event_key]
            tc = event['tc']
            rds = int(event['rat']), int(event['day']), int(event['session'])

        elif type(event_key) is tuple and len(event_key) == 4:
            rat, day, session, tc = event_key
            rds = rat, day, session
            if COM is None:
                self.out('warning: COM not specified, using 0.0')
                COM = 0.0
            event = dict(rat=rat, day=day, session=session, tc=tc, COM_field=COM)

        else:
            raise TypeError, 'event_key must be potentiation table index or (r, d, s, tc) tuple'

        session_data = SessionData.get(rds)
        session_data._compute_laps(cut=event['COM_field'] - np.pi)
        filters = session_data.running_filter()
        scan_filters = dict(select=session_data.scan_list, velocity_filter=False)
        cluster = session_data.cluster_data(event['tc'])
        traj = session_data.trajectory

        title_desc = 'Rat %(rat)d, Day %(day)d, Maze %(session)d, Cell %(tc)s'%event
        desc = '%(rat)03d-%(day)02d-m%(session)d-%(tc)s'%event

        plt.ioff()
        f = self.new_figure(desc, title_desc, figsize=(self.figwidth, self.figheight))

        N_laps = session_data.N_laps
        plot_desc_list = ([ 'full', 'running', 'scanning',
                            'sawtooth_running',
                            'sawtooth_scanning',
                            'unrolled_running', 'unrolled_scanning' ] +
            range(1, N_laps+1) +
            ['zoom', 'unwrapped', 'rate'])

        # Plot style formatting dictionaries
        track_fmt = dict(ec='0.7', lw=0.4, ls='solid')

        traj_fmt = TRAJ_FMT.copy()
        traj_fmt.update(zorder=0, lw=0.5, c='0.33', alpha=1.0)
        traj_dimmed = traj_fmt.copy()
        traj_dimmed.update(alpha=0.4, zorder=-1, lw=0.4)

        spike_fmt = SpikePartition.spike_fmt.copy()
        spike_fmt.update(marker='.', s=12, zorder=1, edgecolors='none', facecolors='r', alpha=0.8)

        rate_fmt = dict(c='k', lw=1.5, zorder=1)

        def _get_behavior_segments(which):
            beh_filter = { 'running':filters, 'scanning':scan_filters }[which]
            return np.array(
                filter(lambda g: g[1] != traj.ts.size,
                    contiguous_groups(
                        session_data.filter_tracking_data(
                            traj.ts, boolean_index=True, **beh_filter))))

        for i, plot_desc in enumerate(plot_desc_list):
            subp = self.nrows, self.ncols, i + 1

            if plot_desc == 'full':
                ax = f.add_subplot(*subp)

                # Get trajectory and spike data
                traj_x, traj_y = traj.x, traj.y
                spike_x, spike_y = cluster.x[cluster.x!=0], cluster.y[cluster.x!=0]

                # Plot trajectory and spikes
                plot_track_underlay(ax, **track_fmt)
                ax.plot(traj_x, traj_y, **traj_fmt)
                ax.scatter(spike_x, spike_y, **spike_fmt)

                # Axis set-up
                ax.axis('scaled')
                ax.axis([-50, 50, -50, 50])
                ax.set_axis_off()

            elif plot_desc in ('running', 'scanning'):
                ax = f.add_subplot(*subp)

                # Plot base trajectory but dimmed
                plot_track_underlay(ax, **track_fmt)
                valid = cluster.x!=0
                spike_ts, spike_x, spike_y = (cluster.spikes[valid],
                    cluster.x[valid], cluster.y[valid])
                ax.plot(traj.x, traj.y, **traj_dimmed)

                # Plot behavioral segments on top of base trajectory
                segments = _get_behavior_segments(plot_desc)
                for start, end in segments:
                    ax.plot(traj.x[start:end], traj.y[start:end], **traj_fmt)

                # Plot behavior-specific spikes
                beh_ix = select_from(spike_ts, traj.ts[segments])
                ax.scatter(spike_x[beh_ix], spike_y[beh_ix], **spike_fmt)

                # Axis set-up
                ax.axis('scaled')
                ax.axis([-50, 50, -50, 50])
                ax.set_axis_off()
                quicktitle(ax, plot_desc)

            elif plot_desc in ('unrolled_scanning', 'unrolled_running'):
                label = desc + '_unrolled'
                if label in self.figure:
                    g = self.figure[label]
                else:
                    g = self.new_figure(label, title_desc, figsize=(self.figwidth, self.figwidth/2))
                which = plot_desc.endswith('scanning') and 'scanning' or 'running'
                subp = 211 + int(which == 'scanning')

                from mpl_toolkits.mplot3d import Axes3D
                ax = g.add_subplot(subp, projection='3d')
                ax.view_init(elev=0.0, azim=0.0)

                # Plot base trajectory but dimmed
                valid = cluster.x!=0
                spike_ts, spike_x, spike_y = (cluster.spikes[valid],
                    cluster.x[valid], cluster.y[valid])

                # Get angles and apply optional rotational offset
                d2r = (2 * np.pi / 360.0)
                spike_alpha = session_data.F_('alpha_unwrapped')(session_data.T_(spike_ts)) - rotation
                traj_alpha = traj.alpha_unwrapped - rotation
                traj_x, traj_y = rot2D_vec(np.vstack((traj.x, traj.y)), -d2r*rotation)
                spike_x, spike_y = rot2D_vec(np.vstack((spike_x, spike_y)), -d2r*rotation)

                # Plot spiral track edges
                track_3d_fmt = dict(c='0.7', lw=0.5, zorder=-50)
                alpha = d2r * np.linspace(traj_alpha[0], traj_alpha[-1], 512)
                ax.plot3D(28*np.cos(alpha), -alpha / d2r, 28*np.sin(alpha), **track_3d_fmt)
                ax.plot3D(38*np.cos(alpha), -alpha / d2r, 38*np.sin(alpha), **track_3d_fmt)

                # Plot base trajectory, dimmed
                old_lw = traj_dimmed.pop('lw')
                traj_dimmed.update(lw=0.7)
                ax.plot3D(traj_x, -1*traj_alpha, traj_y, **traj_dimmed)
                traj_dimmed.update(lw=old_lw)

                # Plot behavioral segments on top of base trajectory
                segments = _get_behavior_segments(which)
                old_lw = traj_fmt.pop('lw')
                traj_fmt.update(lw=1)
                for start, end in segments:
                    ax.plot3D(traj_x[start:end], -1*traj_alpha[start:end], traj_y[start:end], **traj_fmt)
                traj_fmt.update(lw=old_lw)

                # Plot behavior-specific spikes
                spike_3d_fmt = dict(s=144, alpha=1, marker='.', linewidths=0, edgecolor='none', facecolor='r', zorder=50)
                beh_ix = select_from(spike_ts, traj.ts[segments])
                ax.scatter3D(spike_x[beh_ix], -1*spike_alpha[beh_ix], spike_y[beh_ix], **spike_3d_fmt)

                ax.set_axis_off()
                quicktitle(ax, which)

            elif plot_desc in ('sawtooth_scanning', 'sawtooth_running'):
                label = desc + '_sawtooth'
                if label in self.figure:
                    g = self.figure[label]
                    plt.figure(g.number)
                else:
                    g = self.new_figure(label, title_desc, figsize=(self.figwidth, self.figwidth/2))
                which = plot_desc.endswith('scanning') and 'scanning' or 'running'
                subp = 211 + int(which == 'scanning')
                ax = g.add_subplot(subp)

                # Plot base trajectory but dimmed
                valid = cluster.x!=0
                spike_ts = cluster.spikes[valid]
                spike_t = session_data.T_(spike_ts)

                # Get angles and apply optional rotational offset
                spike_alpha = 360 - session_data.F_('alpha')(spike_t)
                spike_unwrapped = 360 - session_data.F_('alpha_unwrapped')(spike_t)
                traj_alpha = 360 - traj.alpha
                traj_unwrapped = 360 - traj.alpha_unwrapped

                # Plot base trajectory, dimmed
                old_lw = traj_dimmed.pop('lw')
                traj_dimmed.update(lw=0.8)
                ax.plot(traj_unwrapped, traj_alpha, **traj_dimmed)
                traj_dimmed.update(lw=old_lw)

                # Plot behavioral segments on top of base trajectory
                segments = _get_behavior_segments(which)
                old_lw = traj_fmt.pop('lw')
                traj_fmt.update(lw=1)
                for start, end in segments:
                    ax.plot(traj_unwrapped[start:end], traj_alpha[start:end], **traj_fmt)
                traj_fmt.update(lw=old_lw)

                # Plot behavior-specific spikes
                spike_saw_fmt = dict(s=64, alpha=1, marker='.', linewidths=0, edgecolor='none', facecolor='r', zorder=50)
                beh_ix = select_from(spike_ts, traj.ts[segments])
                ax.scatter(spike_unwrapped[beh_ix], spike_alpha[beh_ix], **spike_saw_fmt)

                ax.set_axis_off()
                quicktitle(ax, which)

            elif type(plot_desc) is int or plot_desc == 'zoom':
                ax = f.add_subplot(*subp)

                if plot_desc == 'zoom':
                    lap = zoom_lap
                else:
                    lap = plot_desc
                tlim = session_data.laps[lap-1:lap+1]

                # Plot track underlay
                plot_track_underlay(ax, **track_fmt)

                # Plot lap trajectory
                traj_ix = np.logical_and(traj.ts>=tlim[0], traj.ts<=tlim[1])
                ax.plot(traj.x[traj_ix], traj.y[traj_ix], **traj_fmt)

                # Plot spike partition limited to the current lap
                spike_ix = np.logical_and(cluster.x != 0,
                    np.logical_and(
                        cluster.spikes>=tlim[0], cluster.spikes<=tlim[1]))
                ax.scatter(cluster.x[spike_ix], cluster.y[spike_ix], **spike_fmt)

                # Axis set-up
                ax.axis('scaled')
                ax.axis([-50, 50, -50, 50])
                if plot_desc == 'zoom':
                    quicktitle(ax, 'zoom lap %d'%zoom_lap)
                ax.set_axis_off()

            elif plot_desc.startswith('unwrapped'):
                g = self.new_figure('%s_unwrapped'%desc, 'Unwrapped: %s'%title_desc,
                    figsize=(self.figwidth, self.figwidth/2))
                ax = g.add_subplot(221)

                Uscan = session_data.get_unwrapped_cluster_ratemap(cluster, **scan_filters)
                U = session_data.get_unwrapped_cluster_ratemap(cluster, **filters)
                centers = (U['bins'][1:] + U['bins'][:-1]) / 2
                ax.plot(centers, U['R'], c='k', lw=1.2, aa=False, zorder=10)
                ax.plot(centers, Uscan['R'], c='b', lw=1.2, aa=False, zorder=5)
                ax.set_xlim(U['tracklim'])
                track_laps = np.arange(U['tracklim'][1]/360.0)
                ax.set_xticks(360 * track_laps)
                ax.set_xticklabels(map(lambda l: (l%2==0) and str(int(l)) or '', track_laps))
                ax.set_xlabel('Unwrapped Laps')
                ax.set_ylabel('Firing Rate (spk/s)')
                ax.tick_params(right=False, top=False)

            elif plot_desc == 'rate':
                ax = f.add_subplot(*subp, projection='polar')

                # Get peak overall firing rate, plus before/after ratemaps
                R = session_data.get_cluster_ratemap(event['tc'], **filters).squeeze()
                R = np.r_[R, R[0]]
                bins = R.size

                # Plot the before/after ratemap on a polar plot
                ax.plot(np.linspace(0, 2*np.pi, bins), np.ones(bins), c='0.7', lw=0.5, zorder=-1)
                ax.plot(np.linspace(0, 2*np.pi, bins), 1.0 + R / R.max(), **rate_fmt)

                # Axis set-up
                ax.set_rscale('linear')
                ax.set_rlim(0, 2.1)

                ax.text(0.5, 0.5, '%.1f'%R.max(), ha='center', va='center',
                    transform=ax.transAxes, size='xx-small')
                self.out('Peak rate = %f'%R.max())
                ax.set_axis_off()

        self.results['there_is_no_spoon'] = True
        self.out('All done!')
        plt.ion()
        plt.show()


class CompactLapSpikesReport(BaseReport):

    """
    Compact display of lap-spikes plots of potentiation events for figures
    """

    label = 'compact events'
    nrows = 8
    ncols = 5
    xnorm = False
    ynorm = False
    figwidth = Float(7.4409448819) #double-column width
    figheight = Float(9.7244094488) #column height

    def get_plot(self, mod_table, condn_or_ids, bins=48):
        self._start_report()
        if type(condn_or_ids) is str:
            ix = mod_table.getWhereList(condn_or_ids)
        else:
            ix = condn_or_ids

        N = len(ix)
        w = int(self.ncols / 2)
        i = 0

        new_style_table = hasattr(mod_table.cols, 'COM_field')

        for event_ix in ix:
            if np.isscalar(event_ix):
                event = mod_table[event_ix]
            elif type(event_ix) is dict:
                event = event_ix

            rds = event['rat'], event['day'], event['session']
            tc = event['tc']
            session_data = SessionData.get(rds)

            # Cut laps for field depending on modulation table type
            if new_style_table:
                session_data._compute_laps(
                    cut=(np.pi/180) * (event['COM_field'] - 180))
            else:
                cut_laps_opposite_field(session_data, tc)

            l = event['lap']
            lap_intervals = np.array(
                zip(session_data.laps[:-1], session_data.laps[1:]))

            plot_type = 'rate'
            for lap in xrange(l - w + 0, l + w + 1): # adjust displayed laps here
                if lap < 0 or lap > session_data.N_laps - 1:
                    tlim = None
                elif lap == l - w:
                    tlim = (session_data.start, lap_intervals[lap][1])
                    plot_type = 'before_rate'
                elif lap == l + w:
                    tlim = (lap_intervals[lap][0], session_data.end)
                    plot_type = 'after_rate'
                else:
                    tlim = lap_intervals[lap]

                self.lastpanel = i == N - 1
                ax = self._create_new_axes(polar=(plot_type.endswith('rate')))
                yield (session_data, tc, tlim, plot_type, ax)

                # Set the axis titles and output to console
                if lap == l - w + 0: # also adjust here for axis titling
                    axlabel = 'rat%03d-%02d-m%d-%s\nlaps %d-%d'%(rds + (tc,) + (1, lap+1))
                elif lap == l + w:
                    axlabel = 'laps %d-%d'%(lap+1, session_data.N_laps)
                else:
                    axlabel = 'lap %d'%(lap+1)
                self.out(axlabel.title())
                quicktitle(ax, axlabel, size='x-small')

                self._advance(ax)
                i += 1

                if tlim is not None:
                    plot_type = 'spikes'

            self._finish_chunk()
        self._finish_report()

    def collect_data(self, table='potentiation', condn_or_ids='rat!=0'):
        """Potentiation events can be a string query on the potentiation table,
        a list of ids (indexes) into the potentiation table and/or dictionaries
        with 'rat', 'day', 'sesssion', 'tc', 'lap', and 'COM_field' items.
        """
        mod_table = get_node('/physiology', table)
        self.desc = '%s Events' % snake2title(table)

        for session_data, tc, tlim, plot_type, ax in self.get_plot(mod_table, condn_or_ids):
            if tlim is None:
                ax.set_axis_off()
                continue

            if plot_type == 'spikes':

                # Plot track underlay
                plot_track_underlay(ax, ec='0.8', lw=0.4)

                # Plot lap trajectory
                traj = session_data.trajectory
                traj_ix = np.logical_and(traj.ts>=tlim[0], traj.ts<=tlim[1])
                x, y = traj.x[traj_ix], traj.y[traj_ix]
                ax.plot(x, y, **TRAJ_FMT)

                # Plot spike partition limited to the current lap
                cluster = session_data.cluster_data(tc)
                txy = cluster.spikes, cluster.x, cluster.y # save to restore later
                spike_ix = np.logical_and(
                    cluster.spikes>=tlim[0], cluster.spikes<=tlim[1])
                cluster.spikes = cluster.spikes[spike_ix]
                cluster.x = cluster.x[spike_ix]
                cluster.y = cluster.y[spike_ix]
                SpikePartition.plot(session_data, cluster, ax=ax,
                    scan_extensions=True, mod_table=mod_table, s=16)

                # Axis set-up
                ax.axis('scaled')
                ax.axis([-50, 50, -50, 50])
                ax.set_axis_off()

                # Restore spike data to cluster object
                cluster.spikes, cluster.x, cluster.y = txy

            elif plot_type == 'before_rate':

                # Store time bounds and axis for later plotting
                self.current_before_rate = (tlim, ax)

            elif plot_type == 'after_rate':

                # Unpack time bounds and axis for "before" polar ratemap
                tlim_before, ax1 = self.current_before_rate
                ax2 = ax

                # Get circular ratemaps for before and after potentation laps
                filters = session_data.running_filter()
                R1 = session_data.get_cluster_ratemap(tc, tlim=tlim_before, **filters).squeeze()
                R1 = np.r_[R1, R1[0]]
                R2 = session_data.get_cluster_ratemap(tc, tlim=tlim, **filters).squeeze()
                R2 = np.r_[R2, R2[0]]
                bins = R1.size

                # Normalize from 1 to max of before/after firing rate peaks
                R_max = max(R1.max(), R2.max())
                R_before = 1.0 + R1 / R_max
                R_after = 1.0 + R2 / R_max

                # Plot the before/after ratemap on a polar plot
                angles = np.linspace(0, 2*np.pi, bins)
                ax1.plot(angles, np.ones(bins), c='0.7', lw=0.5, zorder=0, solid_capstyle='round')
                ax2.plot(angles, np.ones(bins), c='0.7', lw=0.5, zorder=0, solid_capstyle='round')
                ax1.plot(angles, R_before, c='k', lw=1.5, zorder=1, solid_capstyle='round')
                ax2.plot(angles, R_after, c='k', lw=1.5, zorder=1, solid_capstyle='round')

                # Axis set-up
                ax1.set_rscale('linear')
                ax1.set_rlim(0, 2.1)
                ax1.set_axis_off()
                ax2.set_rscale('linear')
                ax2.set_rlim(0, 2.1)
                ax2.set_axis_off()

                ax1.text(0.5, 0.5, '%.2f'%R1.max(), ha='center', va='center',
                    transform=ax1.transAxes, size='xx-small')
                self.out('Peak rate before = %f'%R1.max())
                ax2.text(0.5, 0.5, '%.2f'%R2.max(), ha='center', va='center',
                    transform=ax2.transAxes, size='xx-small')
                self.out('Peak rate after = %f'%R2.max())

        self.out('All done!')


class ModulationLapSpikesReport(BaseReport):

    """
    Show series of lap-spikes plots centered on detected place-field events
    """

    label = 'event lap report'
    nrows = 5
    ncols = 7
    figwidth = Float(11.0)
    figheight = Float(8.5)

    def get_plot(self, mod_table, condn, bins=48):
        self._start_report()
        N = len(mod_table.getWhereList(condn))
        w = int(self.ncols / 2)
        i = 0

        new_style_table = hasattr(mod_table.cols, 'COM_field')

        for event in mod_table.where(condn):
            rds = event['rat'], event['day'], event['session']
            tc = event['tc']
            session_data = SessionData.get(rds)

            # Cut laps for field depending on modulation table type
            if new_style_table:
                session_data._compute_laps(
                    cut=(np.pi/180) * (event['COM_field'] - 180))
            else:
                cut_laps_opposite_field(session_data, tc)

            l = event['lap']
            lap_intervals = np.array(
                zip(session_data.laps[:-1], session_data.laps[1:]))

            for lap in xrange(l - w + 0, l + w + 1): # adjust displayed laps here
                if lap < 0 or lap > session_data.N_laps - 1:
                    tlim = None
                else:
                    tlim = lap_intervals[lap]

                self.lastpanel = i == N - 1
                ax = self._create_new_axes()
                yield (session_data, tc, tlim, ax)

                # Set the axis titles and output to console
                if lap == l - w + 0: # also adjust here for axis titling
                    axlabel = 'rat%03d-%02d-m%d %s'%(rds + (tc,))
                else:
                    axlabel = 'lap %d'%(lap+1)
                self.out(axlabel.title())
                quicktitle(ax, axlabel)

                self._advance(ax)
                i += 1

            self._finish_chunk()
        self._finish_report()

    def collect_data(self, show_mod=True, table_name='potentiation', condn='rat!=0'):
        mod_table = get_node('/physiology', table_name)
        self.desc = snake2title(table_name)

        mod_table_plot = None
        if show_mod:
            mod_table_plot = mod_table

        for session_data, tc, tlim, ax in self.get_plot(mod_table, condn):
            if tlim is None:
                ax.set_axis_off()
                continue

            # Plot track underlay
            plot_track_underlay(ax, ec='0.8', lw=0.4)

            # Plot lap trajectory
            traj = session_data.trajectory
            traj_ix = np.logical_and(traj.ts>=tlim[0], traj.ts<=tlim[1])
            x, y = traj.x[traj_ix], traj.y[traj_ix]
            ax.plot(x, y, **TRAJ_FMT)

            # Plot spike partition limited to the current lap
            cluster = session_data.cluster_data(tc)
            txy = cluster.spikes, cluster.x, cluster.y # save to restore later
            spike_ix = np.logical_and(
                cluster.spikes>=tlim[0], cluster.spikes<=tlim[1])
            cluster.spikes = cluster.spikes[spike_ix]
            cluster.x = cluster.x[spike_ix]
            cluster.y = cluster.y[spike_ix]
            SpikePartition.plot(session_data, cluster, ax=ax,
                scan_extensions=True, mod_table=mod_table_plot, s=18)

            # Axis set-up
            ax.axis([-48, 48, -48, 48])
            ax.axis('equal')
            ax.set_axis_off()

            # Restore spike data to cluster object
            cluster.spikes, cluster.x, cluster.y = txy

        self.out('All done!')


class ModulationSpikesReport(BaseReport):

    """
    Show each detected novel field on a time x track-angle plot with spikes
    """

    label = 'spike report'
    xnorm = False
    ynorm = False

    def collect_data(self, table_name='potentiation'):
        """Display track-angle and spike plots of novel fields that form in
        the middle of session, data from /physiology/potentiation.
        """
        self.save_file_stem = table_name + '_spikes_report'

        COM_fmt = dict(c='m', alpha=0.5, ls='-', lw=1, zorder=-5)

        ftable = get_node('/physiology', table_name)
        cell_list = unique_cells('/physiology', table_name)

        for cell_id, ax in self.get_plot(cell_list):
            rds, tc = cell_id
            self.out('Plotting rat%03d-%02d-m%d cell %s...'%(rds+(tc,)))

            # Retrieve session data
            data = SessionData.get(rds)
            alpha = data.trajectory.alpha
            t = data.T_(data.trajectory.ts)

            # Get separate index arrays for field, scan, and regular spikes
            cluster = data.cluster_data(tc)
            t_spikes = data.T_(cluster.spikes)
            alpha_spikes = xy_to_deg_vec(cluster.x, cluster.y)

            ax.plot(t, alpha, **TRAJ_FMT)
            SpikePartition.plot(data, tc, ax=ax, mod_table=ftable,
                x=t_spikes, y=alpha_spikes, linewidths=0.5, s=5)

            mod_ix = SpikePartition.ix['mod']
            if mod_ix.any():
                ax.axhline(alpha_spikes[mod_ix.nonzero()[0][0]], **COM_fmt)

            quicktitle(ax, '%d-%02d-m%d-%s'%(rds+(tc,)), size='x-small')
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(0, 360)
            if self.firstonpage:
                ax.set_yticks([0, 180, 360])
                ax.set_ylabel(u'Track Angle (CCW\u00b0)')
            else:
                ax.set_yticklabels([])
            if self.lastonpage:
                ax.set_xlabel('Time in Session')
            ax.set_xticklabels([])

        self.out('All done!')


class ModulationRatemapReport(BaseReport):

    """
    Show each detected field modulations as ratemap lap matrices
    """

    label = 'ratemap report'
    xnorm = False
    ynorm = False

    def collect_data(self, table_name='potentiation', bins=48):
        """Display track-angle and spike plots of novel fields that form in
        the middle of session, data from /physiology/potentiation.
        """
        self.save_file_stem = table_name + '_ratemap_report'

        c = 'w'
        crosshairs = dict(c=c, lw=1, alpha=0.8)
        dots = dict(marker='o', ms=3, mew=0, mec=c, mfc=c, alpha=0.8)

        filter_kwds = dict(exclude_off_track=True, velocity_filter=True)
        ratemap_kwds = dict(bins=bins, blur_width=360./bins)
        ratemap_kwds.update(filter_kwds)

        ftable = get_node('/physiology', table_name)
        cell_list = unique_cells('/physiology', table_name)

        for cell_id, ax in self.get_plot(cell_list):
            rds, tc = cell_id
            self.out('Plotting rat%03d-%02d-m%d cell %s...'%(rds+(tc,)))

            # Retrieve session data
            data = SessionData.get(rds)
            alpha = data.trajectory.alpha
            t = data.T_(data.trajectory.ts)

            # Compute and plot the ratemap lap matrix
            ratemap_kwds.update(exclude=data.extended_scan_and_pause_list)
            R = data.get_population_lap_matrix(clusters=tc,
                **ratemap_kwds).squeeze()
            N_laps = R.shape[1]
            ax.imshow(R, origin='lower', aspect='auto',
                interpolation='nearest',
                extent=[0.5, N_laps+0.5, 0, 360])

            # Plot crosshairs and points at detected field modulation events
            for event in ftable.where(data.session_query+'&(tc=="%s")'%tc):
                lap = event['lap']+1
                angle = event['COM']
                ax.axvline(lap, **crosshairs)
                ax.axhline(angle, **crosshairs)
                ax.plot([lap], [angle], **dots)

            quicktitle(ax, '%d-%02d-m%d-%s'%(rds+(tc,)), size='x-small')
            ax.set(ylim=(0, 360), xlim=(0.5, N_laps+0.5), xticks=[],
                yticks=[0, 180, 360])
            if self.firstonpage:
                ax.set_ylabel(u'Track Angle (CCW\u00b0)')
            else:
                ax.set_yticklabels([])
            if self.lastonpage:
                ax.set_xlabel('Laps')

        self.out('All done!')


class ScanFieldDistributions(AbstractAnalysis):

    """
    Spatiotemporal distributions of scanning activity around novel fields
    """

    label = 'field distros'

    def collect_data(self, table_name='potentiation', expt='ALL', max_laps=16, bins=32):
        """For every detected novel field, compute relative timing and spatial
        positions of head scans and head scan spiking.

        Keyword arguments:
        table_name -- table under /physiology with FindFieldModulation results
        expt -- restrict by experiment type, set to 'DR', 'NOV', or 'ALL'
        bins -- number of bins for ratemaps
        """
        session_table = get_node('/metadata', 'sessions')
        field_table = get_node('/physiology', table_name)
        self.results['table_name'] = table_name

        new_style_table = hasattr(field_table.cols, 'COM_field')
        COM_field_key = new_style_table and 'COM_field' or 'COM_max'

        angles = get_angle_array(bins, degrees=True)
        if expt == 'ALL':
            N_events = field_table.nrows
        else:
            N_events = 0
            for row in field_table.iterrows():
                rds = row['rat'], row['day'], row['session']
                data = SessionData.get(rds)
                session = get_unique_row(session_table, data.session_query)
                if session['expt_type'] == expt:
                    N_events += 1
            self.out('Found %d fields in %s experiments.'%(N_events, expt))
        self.results['max_laps'] = max_laps
        N_unwrap = int(max_laps*bins)

        # Initialize data storage and accumulators
        cell_info = self.results['cell_info'] = []
        field_matrix = self.results['field_matrix'] = np.empty((N_events, bins), 'd')
        # dp_matrix = self.results['dp_matrix'] = np.zeros((N_events, max_laps), 'd')-1
        unwrap_field_matrix = self.results['unwrap_field_matrix'] = \
            np.zeros((N_events, N_unwrap), 'd')-1
        unwrap_scan_matrix = self.results['unwrap_scan_matrix'] = \
            np.zeros((N_events, N_unwrap), 'd')-1
        lap = self.results['lap'] = np.zeros((N_events,), 'i')
        first_COM = self.results['first_COM'] = np.empty(N_events, 'd')
        field_COM = self.results['field_COM'] = np.zeros(N_events, 'd')
        start = self.results['start'] = np.zeros(N_events, 'd')
        end = self.results['end'] = np.zeros(N_events, 'd')
        track_start = self.results['track_start'] = np.zeros(N_events, 'd')
        track_end = self.results['track_end'] = np.zeros(N_events, 'd')
        track_start_unwrap = self.results['track_start_unwrap'] = np.zeros(N_events, 'd')
        track_end_unwrap = self.results['track_end_unwrap'] = np.zeros(N_events, 'd')
        scan_times = self.results['scan_times'] = []
        scan_alpha = self.results['scan_alpha'] = []
        scan_alpha_bar = self.results['scan_alpha_unwrap'] = []
        spike_counts = self.results['spike_counts'] = []
        event_ids = self.results['event_ids'] = []

        i = 0
        for cell_id in unique_cells(field_table):
            rds, tc = cell_id
            cell_id = rds + (tc,) # flatten
            self.out('Analyzing rat%03d-%02d-m%d for cell %s'%cell_id)

            # Filter based on experiment type if necessary
            session_data = SessionData.get(rds)
            if expt != 'ALL':
                session = get_unique_row(session_table, session_data.session_query)
                if session['expt_type'] != expt:
                    self.out('Skipping because %s experiment...'%session['expt_type'])
                    continue

            # Load the recording session data
            cell_info.append(cell_id)
            cluster = session_data.cluster_data(tc)

            # Compute ratemap of the place field
            ratemap_kwds = dict(exclude_off_track=True,
                exclude=session_data.extended_scan_and_pause_list, bins=bins,
                velocity_filter=True, blur_width=360./bins)
            R = session_data.get_cluster_ratemap(tc, **ratemap_kwds)

            # Construct the masked unwrapped firing ratemap
            P = session_data.get_unwrapped_cluster_ratemap(cluster,
                **ratemap_kwds)
            pvalid = (True-P['mask'])[:N_unwrap]

            # Construct the masked unwrapped scan firing ratemap
            S = session_data.get_unwrapped_cluster_ratemap(cluster,
                bins=bins, blur_width=360./bins, velocity_filter=False,
                exclude_off_track=False, select=session_data.scan_list)
            svalid = (True-S['mask'])[:N_unwrap]

            # Get timing and wrapped/unwrapped track positions of head scans and
            # initial field formation
            traj = session_data.trajectory
            t = session_data.T_(traj.ts)
            if len(session_data.scan_list):
                t_scans = session_data.T_(np.array(session_data.scan_list)[:,0])
            else:
                t_scans = np.array([])
            F_alpha_bar = session_data.F_('alpha_unwrapped')

            # Get initial activity info from data table
            cell_query = session_data.session_query+'&(tc=="%s")'%tc
            for event in field_table.where(cell_query):

                #FIXME: Old dot-product code kind of updated, but needs work?
                # COM_field = event[COM_field_key]
                # session_data._compute_laps(cut=(np.pi/180) * (COM_field-180))
                # R_lap = session_data.get_population_lap_matrix(clusters=tc,
                #     **ratemap_kwds).squeeze().T
                # dot_product = np.dot(R_lap, R) / np.dot(R, R)
                # if dot_product[-2] >= 0.1 and dot_product[-1] < 0.1:
                #     dot_product = dot_product[:-1]
                #     R_lap = R_lap[:-1]
                # dp_matrix[i,:dot_product.size] = dot_product[:max_laps]

                lap[i] = event['lap']
                first_COM[i] = float(event['COM'])
                field_COM[i] = float(event[COM_field_key])
                start[i], end[i] = session_data.T_(event['tlim'])
                if new_style_table:
                    track_end[i], track_start[i] = event['tracklim']
                else:
                    track_start[i], track_end[i] = event['tracklim']
                field_matrix[i] = R
                event_ids.append(event['id'])
                unwrap_field_matrix[i,pvalid] = P['R'][pvalid]
                unwrap_scan_matrix[i,svalid] = S['R'][svalid]
                track_start_unwrap[i] = F_alpha_bar(start[i])
                track_end_unwrap[i] = F_alpha_bar(end[i])
                scan_times.append(t_scans)
                scan_alpha_bar.append(F_alpha_bar(t_scans))
                scan_alpha.append(np.array([a%360 for a in scan_alpha_bar[-1]]))
                spike_counts.append([count_spikes(cluster.spikes, scan[0], scan[1])
                    for scan in session_data.scan_list])
                i += 1

        # Good-bye!
        self.out('All done!')

    def process_data(self, unwrap_norm=True, unwrap_mask_color='k',
        ratemap_cmap='jet', time_window=10, track_cycles=4, bins_per_cycle=10,
        predictive_panel=False):
        res = Reify(self.results)

        # Set up plot size data
        self.figure = {}
        self.figure['main'] = f = plt.figure(num=50, figsize=(8.5,11))
        plt.clf()
        f.suptitle(snake2title(res.table_name))
        N_fields, bins = res.field_matrix.shape
        self.panel = 0
        nrows = 4
        ncols = 3

        self.out('Found %d-bin data for %d place-field events.'%(bins, N_fields))

        def nextaxis(skip=1):
            self.panel += skip
            return plt.subplot(nrows, ncols, self.panel)

        # Sorted index arrays for the novel field data
        sort_field_COM = np.argsort(res.field_COM)
        sort_first_COM = np.argsort(res.first_COM)
        sort_track_start = np.argsort(res.track_start)
        sort_track_start_unwrap = np.argsort(res.track_start_unwrap)[::-1]
        sort_start = np.argsort(res.start)
        sort_end = np.argsort(res.end)

        # Draw place fields
        ax = nextaxis()
        Rsort = res.field_matrix[sort_field_COM]
        Rnorm = Rsort / np.trapz(Rsort, axis=1)[:,np.newaxis]
        Rnormmax = Rsort / np.max(Rsort, axis=1)[:,np.newaxis]
        ax.imshow(Rnorm, origin='upper', interpolation='nearest', aspect='auto',
            extent=[0, 360, 0, N_fields])
        ax.set_xticks([0, 360])
        ax.set_ylabel('Fields [COM Sort]')
        quicktitle(ax, 'Place Fields')
        array_to_image(Rnorm, os.path.join(self.datadir, 'place_fields_norm.png'),
            cmap=ratemap_cmap)
        array_to_image(Rnormmax, os.path.join(self.datadir, 'place_fields_max.png'),
            cmap=ratemap_cmap)

        # First COM vs full max field COM, first scatter
        ax = nextaxis()
        ax.scatter(res.field_COM, res.first_COM, s=10, marker='x', color='b')
        ax.plot([0, 360], [0, 360], 'k--')
        ax.set(xlim=(0, 360), xticks=(0,360), #xlabel='Field COM',
            ylim=(0, 360), yticks=(0,360))#, ylabel='Initial COM')
        quicktitle(ax, 'Full vs. Initial COM')

        # ..., then histogram
        ax = nextaxis()
        ax.hist(circle_diff_vec_deg(res.first_COM, res.field_COM),
            np.linspace(-180,180,32), histtype='step')
        ax.axvline(0, c='k', ls='--')
        ax.set(xlim=(-180, 180), xticks=(-180,-90,0,90,180),
            xlabel='Initial Offset')
        quicktitle(ax, 'COM Histogram')

        # Unwrapped place field matrix
        ax = plt.subplot(nrows, 1, 2)
        unwrapped_matrix = res.unwrap_field_matrix[sort_track_start_unwrap]
        mask = unwrapped_matrix == -1
        if unwrap_norm:
            unwrapped_matrix = unwrapped_matrix / unwrapped_matrix.max(axis=1).reshape(N_fields,1)
        M = masked_array_to_rgba(
                unwrapped_matrix,
                mask=mask, mask_color=unwrap_mask_color, cmap=ratemap_cmap,
                norm=(not unwrap_norm))
        ax.imshow(
            M,
            origin='upper',
            interpolation='nearest', aspect='auto',
            extent=[0, res.max_laps, 0, N_fields])
        ax.set_xticks(np.arange(res.max_laps+1))
        # ax.set_xlabel('Unwrapped Track (laps)')
        ax.set_ylabel('Fields [Unwrapped Angle Sort]')
        quicktitle(ax, 'Unwrapped Place Fields')
        rgba_to_image(M, os.path.join(self.datadir, 'unwrapped_place_fields.png'))

        # Output file with sort index for unwrapped matrix and corresponding event IDs
        self.out('Writing sort index -> event IDs file...')
        fd = file(os.path.join(self.datadir, 'unwrapped_sort_event_ids.csv'), 'w')
        fd.write('row,matrix_index,event_id\n')
        for i, row_ix in enumerate(sort_track_start_unwrap):
            fd.write('%d,%d,%d\n' % (i,row_ix,res.event_ids[row_ix]))
        fd.close()
        self.out('...done.')

        # Unwrapped scan field matrix
        ax = plt.subplot(nrows, 1, 3)
        unwrapped_scan_matrix = res.unwrap_scan_matrix[sort_track_start_unwrap]
        mask = unwrapped_scan_matrix == -1
        if unwrap_norm:
            unwrapped_scan_matrix = unwrapped_scan_matrix / unwrapped_scan_matrix.max(axis=1).reshape(N_fields,1)
        M = masked_array_to_rgba(
                unwrapped_scan_matrix,
                mask=mask, mask_color=unwrap_mask_color, cmap=ratemap_cmap,
                norm=(not unwrap_norm))
        ax.imshow(
            M,
            origin='upper',
            interpolation='nearest', aspect='auto',
            extent=[0, res.max_laps, 0, N_fields])
        ax.set_xticks(np.arange(res.max_laps+1))
        ax.set_xlabel(u'Unwrapped Track (laps)')
        ax.set_ylabel('Scans')
        # quicktitle(ax, 'Unwrapped Place Fields')
        rgba_to_image(M, os.path.join(self.datadir, 'unwrapped_scan_fields.png'))

        # self.panel += 2*ncols

        # Dot-product matrix, sorted by field onset time
        # ax = nextaxis()
        # # mask = res.dp_matrix == -1
        # # DP = res.dp_matrix.copy()
        # DP[DP>1] = 1 # truncate at 1
        # DP[DP<0] = 0 # convert mask values to 0 floor
        # ax.imshow(
        #     masked_array_to_rgba(DP[sort_start], mask=mask[sort_start]),
        #     origin='upper', interpolation='nearest', aspect='auto',
        #     extent=[0, res.max_laps, 0, N_fields])
        # ax.set_xticks([0, res.max_laps])
        # ax.set_xlabel('Laps')
        # ax.set_ylabel('Fields [Onset Sort]')
        # quicktitle(ax, 'D.P. x Laps')

        # Generic scan histogram plotting function
        def plot_scan_histogram(min_spikes=0, bins=32, xlim=None, sortix=None,
            field_values=None, scan_values=None, sort_label='', abscissa='',
            label=''):
            ax = nextaxis()
            scan_hist = np.empty((N_fields, bins), 'i')
            for i,ix in enumerate(sortix):
                N_spikes = np.array(res.spike_counts[ix])
                values = scan_values[ix][N_spikes>=min_spikes]
                scan_hist[i] = np.histogram(values,
                    bins=np.linspace(xlim[0], xlim[1], bins+1))[0]
            ax.imshow(scan_hist!=0, origin='upper', interpolation='nearest',
                aspect='auto', extent=[xlim[0], xlim[1], 0, N_fields],
                cmap=plt.cm.gray_r)
            ax.plot(field_values[sortix], np.mgrid[N_fields-1:-1:-1], 'b-',
                zorder=10)
            ax.set(xlim=xlim, xticks=xlim, xlabel='', yticks=[], #xlabel=abscissa
                ylim=(0,N_fields))
            if self.panel == 1:
                ax.set_ylabel('Cells [%s Sort]'%sort_label)
            quicktitle(ax, u'%s Scan Distrib. \u2265%d Spks'%(label,min_spikes))
            textlabel(ax, '%d'%scan_hist.max())

        # Generic whole-dataset cross-correlation histogram function
        def plot_xcorr(ax, min_spikes=0, bins=32, xlim=None, field_values=None,
            scan_values=None, cmp_fn=None):
            xcorr = np.zeros(bins, 'i')
            bin_edges = np.linspace(xlim[0], xlim[1], bins+1)
            for i in xrange(N_fields):
                N_spikes = np.array(res.spike_counts[i])
                values = scan_values[i][N_spikes>=min_spikes]
                xcorr += np.histogram(
                    cmp_fn(field_values[i], values), bins=bin_edges)[0]
            bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
            ax.plot(bin_centers, xcorr, ls='-', #drawstyle='steps-pre',
                label=u'\u2265%d spikes'%min_spikes)

        # Spatial scan histograms
        self.figure['spatial'] = f = plt.figure(num=51, figsize=(8.5,11))
        plt.clf()
        f.suptitle('Spatial Distribution of Scans')
        self.panel = 0
        for spike_threshold in (0,1,2,4,8,12):
            plot_scan_histogram(min_spikes=spike_threshold,
                bins=16, xlim=(0,360),
                sortix=sort_track_start, field_values=res.track_start,
                scan_values=res.scan_alpha, sort_label='Track Start',
                abscissa='Track', label='Spatial')

        # Spatial xcorr histograms
        ax = nextaxis()
        for spike_threshold in (0,1,2,4,8,12):
            plot_xcorr(ax, min_spikes=spike_threshold,
                bins=32, xlim=(-180,180), field_values=res.track_start,
                scan_values=res.scan_alpha, cmp_fn=circle_diff_vec_deg)
        ax.axvline(0, ls='--', c='k', zorder=-1)
        ax.set(xlim=(-180,180), xticks=(-180,-90,0,90,180), xlabel='Track')
        quicktitle(ax, 'Spatial Xcorr')

        # Spatio-temporal (unwrapped angle) scan histograms
        ax = nextaxis()
        ubins = bins_per_cycle*track_cycles
        dab = 360*track_cycles
        for spike_threshold in (0,1,2,4,8,12):
            plot_xcorr(ax, min_spikes=spike_threshold,
                bins=ubins, xlim=(-dab,dab), field_values=res.track_start_unwrap,
                scan_values=res.scan_alpha_unwrap,
                cmp_fn=lambda a,b: a-b)
        ax.axvline(0, ls='--', c='k', zorder=-1)
        ax.set(xlim=(-dab,dab), xticks=(-dab,-dab/2.,0,dab/2.,dab),
            xlabel='Unwrapped Track (degs)')
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        quicktitle(ax, 'Unwrapped Xcorr')

        # Temporal scan histograms
        self.figure['temporal'] = f = plt.figure(num=52, figsize=(8.5,11))
        plt.clf()
        f.suptitle('Temporal Distribution of Scans')
        self.panel = 0
        for spike_threshold in (0,1,2,4,8,12):
            plot_scan_histogram(min_spikes=spike_threshold,
                bins=32, xlim=(0,330),
                sortix=sort_start, field_values=res.start,
                scan_values=res.scan_times, sort_label='Onset',
                abscissa='Time (s)', label='Temporal')

        # Temporal xcorr histograms
        ax = nextaxis()
        dt = time_window
        for spike_threshold in (0,1,2,4,8,12):
            plot_xcorr(ax, min_spikes=spike_threshold,
                bins=32, xlim=(-dt,dt), field_values=res.start,
                scan_values=res.scan_times, cmp_fn=lambda a,b: b-a)
        ax.set(xlim=(-dt,dt), xticks=(-dt,-dt/2.,0,dt/2.,dt), xlabel='Time (s)')
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        quicktitle(ax, 'Temporal Xcorr')

        if not predictive_panel:
            plt.ion(); plt.show();
            return

        # Generic function for separating out predictive/non-predictive scanning
        # activity into a complementary pair of plots
        def plot_predictive_diptych(min_spikes=0, window=180, cycle=1, laplim=(-5,5)):
            ax1 = nextaxis()
            ax2 = nextaxis()
            for i in xrange(N_fields):
                N_spikes = np.array(res.spike_counts[i])
                scan_alpha = res.scan_alpha_unwrap[i][N_spikes>=min_spikes]
                field_alpha = res.track_start_unwrap[i]
                delta = field_alpha - scan_alpha
                amin = (field_alpha + cycle*360) - window/2.
                amax = (field_alpha + cycle*360) + window/2.
                matched = np.logical_and(delta>=amin, delta<=amax).any()
                # valid = res.dp_matrix[i] != -1
                # DP = res.dp_matrix[i][valid] # leave off extra laps
                lap = res.lap[i] # appearance lap index
                index = np.arange(-lap, -lap+len(DP))
                ax = matched and ax1 or ax2
                ax.scatter(index, DP, marker='x', color='k')
                ax.plot(index, DP, 'k-', alpha=0.2, zorder=-1)
            ax_traits = dict(yticks=[0,5,10], ylim=[0,10], ylabel='D.P.',
                xticks=[laplim[0],0,laplim[1]], xticklabels=[], xlim=laplim)
            for ax in (ax1, ax2):
                ax.set(**ax_traits)
                ax.axvline(0, ls='--', c='k', lw=1, zorder=-2)
                ax.axhline(1, ls='--', c='k', lw=1, zorder=-2)
            title = u'\u2265%d spks, \u2206\u003c%d\u00b0, cyc%d'%(min_spikes,
                window/2, cycle)
            quicktitle(ax1, u'True, %s'%title)
            quicktitle(ax2, u'False, %s'%title)

        # Predictive vs non-predictive field onset: scan-spike threshold
        self.figure['predictive_Nspks'] = f = plt.figure(num=53, figsize=(8.5,11))
        plt.clf()
        f.suptitle('Predictive vs. Non-predictive Fields: Spike Threshold')
        self.panel = 0
        nrows = 6
        ncols = 2
        for spike_threshold in (0,1,2,4,8,12):
            plot_predictive_diptych(min_spikes=spike_threshold)
            if spike_threshold == 12:
                ax = plt.gca()
                xlim = ax.get_xlim()
                ax.set(xticks=[xlim[0],0,xlim[1]], xlabel=u'\u2206laps')

        # Predictive vs non-predictive field onset: cycle (lap) occurrence
        self.figure['predictive_cycle'] = f = plt.figure(num=54, figsize=(8.5,11))
        plt.clf()
        f.suptitle('Predictive vs. Non-predictive Fields: Cycle')
        self.panel = 0
        nrows = 6
        ncols = 2
        for cycle_check in (0,1,2,3):
            plot_predictive_diptych(min_spikes=1, cycle=cycle_check)
            if cycle_check == 3:
                ax = plt.gca()
                ax.set(xticks=[xlim[0],0,xlim[1]], xlabel=u'\u2206laps')

        # Predictive vs non-predictive field onset: track-angle window
        self.figure['predictive_track'] = f = plt.figure(num=55, figsize=(8.5,11))
        plt.clf()
        f.suptitle('Predictive vs. Non-predictive Fields: Track Position Window')
        self.panel = 0
        nrows = 6
        ncols = 2
        for trackwin in (360, 180, 90, 45, 22.5):
            plot_predictive_diptych(min_spikes=1, window=trackwin)
            if trackwin == 22.5:
                ax = plt.gca()
                ax.set(xticks=[xlim[0],0,xlim[1]], xlabel=u'\u2206laps')

        plt.ion()
        plt.show()


class PredictiveValueAnalysis(AbstractAnalysis, TetrodeSelect):

    """
    Computing predictive value of firing during scanning as a putative cause of
    place field modulation
    """

    label = 'predictive value'

    xnorm = False
    ynorm = False
    angle_window = Float(45.0)
    scan_rate_threshold = Float(0.25)
    gold_standard = Instance(GoldStandard)

    def _gold_standard_default(self):
        return GoldStandard()

    def run_tests(self, scan_rate_list, field_time, field_angle, session,
        shift=0.0):
        success = False
        for start, rate in scan_rate_list:
            if rate < self.scan_rate_threshold:
                continue
            test_result = self.gold_standard.backward_test(
                start, field_time, field_angle, session,
                scan_angle=self.angle_window, shift=shift)
            if test_result:
                success = True
                break

        return success

    def compute_predictive_values(self, W):
        # Initialize data accumulators and per-cell predictive values
        results = {}
        results['true_positives'] = 0
        results['positives'] = 1
        results['predictive_cells'] = []
        results['false_positive_cells'] = []
        results['shuffled_true_positives'] = 0
        results['shuffled_positives'] = 0

        # Get cluster name and novel field data
        tc = W.record['tc']
        appeared_ts = W.record['tlim'][0]
        appeared_angle = W.record['COM']
        self.out('Testing place cell %s... '%tc)

        # Compute firing rates of all scans
        scan_rate_list = []
        spikes = W.session_data.clusts[tc].spikes
        for start, end in W.scan_list:
            duration = time.elapsed(start, end)
            N_spikes = count_spikes(spikes, start, end)
            scan_rate_list.append((start, N_spikes/duration))

        # Test for prior occurrence of scan
        obs_success = self.run_tests(scan_rate_list, appeared_ts,
            appeared_angle, W.session_data)
        if obs_success:
            results['true_positives'] += 1
        else:
            results['false_positive_cells'].append((W.rds, tc))

        # Computed a shuffled distribution of predictive success
        shuffle_success = 0
        if obs_success:
            self.out.printf('%s: '%tc, color='lightgreen')
        else:
            self.out.printf('%s: '%tc, color='lightred')
        for i in xrange(W.shuffle_tries):
            success = self.run_tests(scan_rate_list, appeared_ts,
                appeared_angle, W.session_data,
                shift=360*np.random.rand())
            if success:
                results['shuffled_true_positives'] += 1
                shuffle_success += 1
                self.out.printf(u'\u25a0', color='green')
            else:
                self.out.printf(u'\u25a0', color='red')
        self.out.printf('\n')

        if obs_success:
            pval = shuffle_success / float(W.shuffle_tries)
            cell_info = (W.rds, tc, pval)
            results['predictive_cells'].append(cell_info)
            if pval <= W.sig_level:
                self.out('*** Rat%03d-%02d-m%d, %s: PV p < %.4f'%(
                    W.rds+(tc, pval)))

        results['shuffled_positives'] = W.shuffle_tries * results['positives']
        return results

    def collect_data(self, shuffle_tries=200, sig_level=0.05):
        """For every place cell in the specified area (or all hippocampal areas)
        compute whether coincident spiking activity and scanning activity is
        predicitive of place field modulation and/or potentiation

        Arguments:
        area -- string name or list of names of hippocampal recording areas,
            or 'ALL' for all hippocampal areas, defaults to 'ALL'
        min_quality -- minimum cluster isolation quality for including cells
        shuffle_tries -- number of random shuffles for computing p-values on
            predictive vlaues
        sig_level -- statistical significance level for reporting and saving
            predictive cells
        """
        self.results['shuffle_tries'] = shuffle_tries

        # Predictive value per-session accumulators
        session_true_positives = []
        session_positives = []
        shuffled_session_true_positives = []
        shuffled_session_positives = []

        # Session information per-session accumulators
        rat_number = []
        day_number = []
        session_type = []
        mismatch_angle = []
        maze_number = []
        type_number = []

        # True and false positive cells across the dataset
        cells_tested = []
        self.results['predictive_cells'] = predictive_cells = []
        self.results['false_positive_cells'] = false_positive_cells = []

        # Get the table of novel fields
        field_table = get_node('/physiology', 'potentiation')
        session_data = None

        for record in field_table.iterrows():
            rds = record['rat'], record['day'], record['session']
            rat, day, maze = rds
            dataset = (rat, day)
            self.out('Analyzing rat%03d-%02d m%d session...'%rds)

            # Load session data and scan list
            if not (session_data and session_data.rds == rds):
                session_data = SessionData.get(rds)
                maze_start, maze_end = session_data.start, session_data.end
                maze_size = maze_end - maze_start
                scan_list = session_data.scan_list

            # Call to method which computes predictive values for each cell in
            # the current dataset, and runs bootstraps for p-values
            workspace = Reify(locals())
            cell_results = self.compute_predictive_values(workspace)
            if cell_results is None:
                continue

            # Add all test results for this session to accumulators
            session_true_positives.append(cell_results['true_positives'])
            session_positives.append(cell_results['positives'])
            shuffled_session_true_positives.append(cell_results['shuffled_true_positives'])
            shuffled_session_positives.append(cell_results['shuffled_positives'])

            predictive_cells.extend(cell_results['predictive_cells'])
            false_positive_cells.extend(cell_results['false_positive_cells'])

            attrs = session_data.data_group._v_attrs
            rat_number.append(rat)
            day_number.append(day)
            session_type.append(attrs['type'])
            mismatch_angle.append(attrs['parameter'])
            maze_number.append(maze)
            type_number.append(attrs['number'])

        self.results['session_true_positives'] = np.array(session_true_positives)
        self.results['session_positives'] = np.array(session_positives)
        self.results['shuffled_session_true_positives'] = np.array(shuffled_session_true_positives)
        self.results['shuffled_session_positives'] = np.array(shuffled_session_positives)

        self.results['rat_number'] = np.asarray(rat_number)
        self.results['day_number'] = np.asarray(day_number)
        self.results['session_type'] = np.asarray(session_type)
        self.results['mismatch_angle'] = np.asarray(mismatch_angle)
        self.results['maze_number'] = np.asarray(maze_number)
        self.results['type_number'] = np.asarray(type_number)

        # Good-bye!
        self.out('All done!')

    def process_data(self, ymax=1.0, plot_ratemaps=False):
        self.out.outfd = file(os.path.join(self.datadir, 'figure.log'), 'w')

        # Load session information data
        rat_number = self.results['rat_number']
        day_number = self.results['day_number']
        session_type = self.results['session_type']
        mismatch_angle = self.results['mismatch_angle']
        maze_number = self.results['maze_number']
        type_number = self.results['type_number']

        overall_positives = self.results['session_positives'].sum()
        overall_true_positives = self.results['session_true_positives'].sum()
        overall_PV = overall_true_positives / float(overall_positives)
        self.out('Number positive cell-sessions: %d'%overall_positives)
        self.out('Number true positive cell-sessions: %d'%overall_true_positives)
        self.out('Overall predictive value: %.4f'%overall_PV)

        shuffled_positives = self.results['shuffled_session_positives'].sum()
        shuffled_true_positives = self.results['shuffled_session_true_positives'].sum()
        shuffled_PV = shuffled_true_positives / float(shuffled_positives)
        self.out('Number shuffled positive cell-sessions: %d'%shuffled_positives)
        self.out('Number shuffled true positive cell-sessions: %d'%shuffled_true_positives)
        self.out('Shuffled predictive value: %.4f'%shuffled_PV)

        plt.ioff()
        self.figure = {}

        def do_bar_plot(ax, values, get_selection, label_str):
            data = np.zeros(len(values), 'd')
            baseline = np.zeros(len(values), 'd')
            for j, value_color in enumerate(values):
                if type(value_color) is tuple:
                    ix = get_selection(value_color[0])
                else:
                    ix = get_selection(value_color)
                for D, prefix in [(data, ''), (baseline, 'shuffled_')]:
                    trues = self.results['%ssession_true_positives'%prefix][ix]
                    total = self.results['%ssession_positives'%prefix][ix]
                    valid = total != 0
                    if not float(total[valid].sum()):
                        continue
                    PV = trues[valid].sum() / float(total[valid].sum())
                    D[j] = PV

            grouped_bar_plot(data, 'PPV', values, ax=ax, baselines=baseline,
                label_str=label_str)
            ax.set_ylim(0, ymax)
            ax.set_ylabel('Predictive Value')

        def make_pv_figures(figname, breakdown, values, selector,
            label_str='%s'):
            self.figure[figname] = f = plt.figure(figsize=(9, 7))
            f.suptitle('Predictive Value Across %s'%breakdown)
            do_bar_plot(plt.axes(), values, selector, label_str)
            return

        # STD/MIS figures
        make_pv_figures('maze_type', 'Session Type',
            [('STD', 'b'), ('MIS', 'g'), ('FAM', 'c'), ('NOV', 'y')],
            lambda value: np.logical_and(maze_number != 1, session_type == value))

        # Maze number figures
        make_pv_figures('maze_number', 'Maze Number',
            [(1,'b'), (2,'g'), (3,'r'), (4,'c'), (5,'m')],
            lambda value: (maze_number == value),
            label_str='m%d')

        # Mismatch angle figures
        make_pv_figures('mismatch_angle', 'Mismatch Angle',
            [(45,'b'), (90,'g'), (135,'r'), (180,'c')],
            lambda value: (mismatch_angle == value),
            label_str='MIS-%d')

        # STD repetition number figures
        make_pv_figures('std_number', 'STD Repetitions',
            [(1,'b'), (2,'g'), (3,'r')],
            lambda num: np.logical_and(session_type=='STD', type_number==num),
            label_str='STD-%d')

        # Rat number figure
        make_pv_figures('rat_number', 'Rats',
            np.sort(np.unique(rat_number)),
            lambda num: (rat_number==num),
            label_str='Rat %03d')

        # Day number figure
        make_pv_figures('day_number', 'Days',
            range(1, 5),
            lambda num: (day_number==num),
            label_str='Day %d')

        # Plot ratemaps of true and false positive cells
        def plot_cell_ratemaps(key):
            self.figure[key] = f = plt.figure(figsize=(11,8.5))
            cell_list = self.results[key][:64]
            axlist = AxesList()
            axlist.make_grid(len(cell_list))
            f.suptitle('%s - Lap Ratemaps'%(key.replace('_', ' ').title()))
            for i, ax in enumerate(axlist):
                rds, tc = cell_list[i][:2]
                data = SessionData.get(rds)
                R = data.get_population_lap_matrix(clusters=tc, bins=36,
                    exclude=data.scan_list, exclude_off_track=True)
                R = np.flipud(np.squeeze(R).T)
                ax.imshow(R, origin='lower', aspect='auto', interpolation='nearest')
                ax.set_title('%03d-%02d-m%d-%s'%(rds+(tc,)), size='xx-small')
                ax.set_xticks([])
                ax.set_yticks([])
            axlist.tight()

        if plot_ratemaps:
            plot_cell_ratemaps('false_positive_cells')
            plot_cell_ratemaps('predictive_cells')

        # Finish up
        plt.ion()
        plt.show()
        self.out.outfd.close()


def run_predictive_value_series(rootdir, quality='fair', shuffles=1000, **kwds):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    for areas in ('CA1', 'CA3', ['CA1', 'CA3']):
        if type(areas) is list:
            desc = "_".join(areas)
        else:
            desc = areas
        anadir = os.path.join(rootdir, desc)
        if os.path.exists(anadir):
            report = PredictiveValueAnalysis.load_data(anadir)
        else:
            report = PredictiveValueAnalysis(desc=desc,
                datadir=anadir, **kwds)
            report(area=areas, min_quality=quality, shuffle_tries=shuffles)
        report.process_data()
        report.save_plots_and_close()


class PredictiveMetaAnalysis(AbstractAnalysis):

    def collect_data(self, rate_threshold=0.5, angle_window=60.0,
        areas=['CA1', 'CA3'], quality='fair', shuffles=1000, save_figures=True):
        """Run a series of PredictiveValueAnalysis analyses and collate the
        data for some meta-analytic plots.

        NOTE: One of rate_threshold and angle_window should be a fixed value,
        while the other must be an iterable list of values to sweep through.
        """
        self.results['scan_rate_threshold'] = rate_threshold
        self.results['angle_window'] = angle_window

        self.results['pval_distros'] = pval_distros = []
        self.results['N_predictive'] = N_predictive = []

        self.results['overall_PV'] = overall_PV = []
        self.results['shuffled_PV'] = shuffled_PV = []
        self.results['baseline_diff'] = baseline_diff = []

        kwds = {}
        if np.iterable(rate_threshold):
            value_list = rate_threshold
            self.results['value_key'] = value_key = 'scan_rate_threshold'
            self.results['value_units'] = 'spikes/s'
            kwds['angle_window'] = float(angle_window)
        elif np.iterable(angle_window):
            value_list = angle_window
            self.results['value_key'] = value_key = 'angle_window'
            self.results['value_units'] = 'degs'
            kwds['scan_rate_threshold'] = float(rate_threshold)
        else:
            raise ValueError, \
                "either rate_threshold or angle_window must be a sequence"

        for value in value_list:
            desc = '%s %.2f'%(value_key, value)
            anadir = os.path.join(self.datadir, desc)
            if os.path.exists(anadir):
                ana = PredictiveValueAnalysis.load_data(anadir)
            else:
                kwds[value_key] = value
                ana = PredictiveValueAnalysis(desc=desc,
                    datadir=anadir, **kwds)
                ana(area=areas, min_quality=quality, shuffle_tries=shuffles)
            ana.process_data(plot_ratemaps=True)
            if save_figures:
                ana.save_plots_and_close()
            else:
                plt.close('all')

            pvalues = [p for rds, tc, p in ana.results['predictive_cells']]
            pval_distros.append(pvalues)
            N_predictive.append(len(pvalues))

            overall_positives = ana.results['session_positives'].sum()
            overall_true_positives = ana.results['session_true_positives'].sum()
            this_overall = overall_true_positives / float(overall_positives)
            overall_PV.append(this_overall)

            shuffled_positives = ana.results['shuffled_session_positives'].sum()
            shuffled_true_positives = ana.results['shuffled_session_true_positives'].sum()
            this_shuffled = shuffled_true_positives / float(shuffled_positives)
            shuffled_PV.append(this_shuffled)

            baseline_diff.append(
                (this_overall-this_shuffled)/(this_overall+this_shuffled))

        self.results['pval_distros'] = np.array(pval_distros)
        self.results['N_predictive'] = np.array(N_predictive)
        self.results['overall_PV'] = np.array(overall_PV)
        self.results['shuffled_PV'] = np.array(shuffled_PV)
        self.results['baseline_diff'] = np.array(baseline_diff)

        self.out('All done!')

    def process_data(self):
        key = self.results['value_key']
        values = self.results[key]
        label = key.replace('_', ' ').title() + ', ' + self.results['value_units']

        self.figure = {}
        self.figure['pvalues'] = f = plt.figure(figsize=(7,6))
        f.suptitle('P-value Distributions of Predictive Cell-Sessions')
        ax = plt.axes()
        pvalues = [np.array(p) for p in self.results['pval_distros']]
        ax.boxplot(pvalues)
        ax.set_xticklabels(values)
        ax.set_xlabel(label)
        ax.set_ylabel('Bootstrap p-values')

        self.figure['predictive_value'] = f = plt.figure(figsize=(7,6))
        f.suptitle('Observed and Shuffled Predictive Values')
        ax = plt.axes()
        ax.plot(values, self.results['overall_PV'], 'b-o', label='Observed')
        ax.plot(values, self.results['shuffled_PV'], 'r-', label='Expected')
        ax.plot(values, self.results['baseline_diff'], 'k--', label='Norm. Diff.')
        ax.legend(loc=0)
        ax.set_xlabel(label)

        ax.set_ylabel('Predictive Values')
