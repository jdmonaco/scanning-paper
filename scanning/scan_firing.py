# encoding: utf-8
"""
scan_firing.py -- Firing rates during scan vs. non-scan

Created by Joe Monaco on 2011-05-06.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
import matplotlib.pylab as plt
import os

# Package imports
from scanr.lib import *
from .reports import BaseDatasetReport
from scanr.spike import TetrodeSelect, parse_cell_name
from scanr.data import session_list
from scanr.cluster import (AND, get_tetrode_restriction_criterion,
    get_min_quality_criterion, PrincipalCellCriteria, PlaceCellCriteria)
from scanr.spike import (get_session_clusters, interval_firing_rate,
    plot_correlogram, xcorr)
from scanr.time import select_from

# Local imports
from .core.analysis import AbstractAnalysis
from .tools.stats import zscore, t_one_tailed
from .tools.plot import grouped_bar_plot
from .tools.misc import DataSpreadsheet

# Scan type labels
AMBIG = Config['scanning']['amb_type']
INTERIOR = Config['scanning']['in_type']
EXTERIOR = Config['scanning']['out_type']


class ScanPhaseFiring(AbstractAnalysis):

    """
    Updated analysis for examining firing rates across scan type and phase
    """

    label = 'scan firing report'

    def collect_data(self, area_query='(area=="CA3")|(area=="CA1")'):
        """Collect firing rate data across scans
        """
        datasets = TetrodeSelect.datasets(area_query)
        tetrode_table = get_node('/metadata', 'tetrodes')
        scan_table = get_node('/behavior', 'scans')

        epochs = (  'rate', 'running_rate', 'pause_rate', 'scan_rate',
                    'interior_rate', 'exterior_rate',
                    'outbound_rate', 'inbound_rate',
                    'ext_out_rate', 'ext_in_rate',
                    'int_out_rate', 'int_in_rate' )

        spreadsheet = DataSpreadsheet(
            os.path.join(self.datadir, 'scan_firing_rates.csv'),
            [   ('dataset', 's'), ('rat', 'd'), ('day', 'd'),
                ('area', 's'), ('area_sub', 's'), ('cell', 's') ] +
            map(lambda n: (n, 'f'), epochs))
        self.out('Record string: %s'%spreadsheet.get_record_string())
        record = spreadsheet.get_record()

        # Index labels for the scan data
        PRE, START, MAX, END = 0, 1, 2, 3

        for dataset in datasets:

            rat, day = dataset
            dataset_str = 'rat%03d-%02d'%dataset
            self.out('Calculating scan firing rates for %s...'%dataset_str)

            # Set dataset info
            record['dataset'] = dataset_str
            record['rat'] = rat
            record['day'] = day

            # Cell accumulators
            collated_cells = []
            N = {}
            T = {}

            def increment(tc, which, count, duration):
                N[tc][which] += count
                T[tc][which] += duration

            for maze in get_maze_list(rat, day):
                rds = rat, day, maze
                data = SessionData.get(rds)
                traj = data.trajectory

                def occupancy(traj_occupied):
                    return \
                        data.duration * (np.sum(traj_occupied) / float(traj.N))

                Criteria = AND(PlaceCellCriteria,
                    TetrodeSelect.criterion(dataset, area_query))

                for tc in data.get_clusters(Criteria):
                    cluster = data.cluster_data(tc)

                    if tc not in collated_cells:
                        collated_cells.append(tc)
                        N[tc] = { k: 0 for k in epochs }
                        T[tc] = { k: 0.0 for k in epochs }

                    spikes = cluster.spikes

                    increment(tc, 'rate', cluster.N, data.duration)
                    increment(tc, 'running_rate',
                        data.velocity_filter(spikes).sum(),
                        occupancy(data.velocity_filter(traj.ts)))

                    increment(tc, 'scan_rate',
                        np.sum(select_from(spikes, data.scan_list)),
                        occupancy(select_from(traj.ts, data.scan_list)))

                    increment(tc, 'pause_rate',
                        np.sum(select_from(spikes, data.pause_list)),
                        occupancy(select_from(traj.ts, data.pause_list)))

                    ext_scan_list = np.array(
                        [(rec['prepause'], rec['start'], rec['max'], rec['end'])
                            for rec in scan_table.where(data.session_query +
                                '&(type=="%s")'%EXTERIOR)])

                    int_scan_list = np.array(
                        [(rec['prepause'], rec['start'], rec['max'], rec['end'])
                            for rec in scan_table.where(data.session_query +
                                '&(type=="%s")'%INTERIOR)])

                    both_scan_list = np.array(
                        [(rec['prepause'], rec['start'], rec['max'], rec['end'])
                            for rec in scan_table.where(data.session_query +
                                '&(type!="%s")'%AMBIG)])

                    if ext_scan_list.shape[0]:
                        increment(tc, 'exterior_rate',
                            np.sum(select_from(spikes, ext_scan_list[:,(START,END)])),
                            occupancy(select_from(traj.ts, ext_scan_list[:,(START,END)])))

                        increment(tc, 'ext_out_rate',
                            np.sum(select_from(spikes, ext_scan_list[:,(START,MAX)])),
                            occupancy(select_from(traj.ts, ext_scan_list[:,(START,MAX)])))

                        increment(tc, 'ext_in_rate',
                            np.sum(select_from(spikes, ext_scan_list[:,(MAX,END)])),
                            occupancy(select_from(traj.ts, ext_scan_list[:,(MAX,END)])))

                    if int_scan_list.shape[0]:
                        increment(tc, 'interior_rate',
                            np.sum(select_from(spikes, int_scan_list[:,(START,END)])),
                            occupancy(select_from(traj.ts, int_scan_list[:,(START,END)])))

                        increment(tc, 'int_out_rate',
                            np.sum(select_from(spikes, int_scan_list[:,(START,MAX)])),
                            occupancy(select_from(traj.ts, int_scan_list[:,(START,MAX)])))

                        increment(tc, 'int_in_rate',
                            np.sum(select_from(spikes, int_scan_list[:,(MAX,END)])),
                            occupancy(select_from(traj.ts, int_scan_list[:,(MAX,END)])))

                    if both_scan_list.shape[0]:
                        increment(tc, 'outbound_rate',
                            np.sum(select_from(spikes, both_scan_list[:,(START,MAX)])),
                            occupancy(select_from(traj.ts, both_scan_list[:,(START,MAX)])))

                        increment(tc, 'inbound_rate',
                            np.sum(select_from(spikes, both_scan_list[:,(MAX,END)])),
                            occupancy(select_from(traj.ts, both_scan_list[:,(MAX,END)])))

            def firing_rate(tc, k):
                if T[tc][k]:
                    return N[tc][k] / T[tc][k]
                return 0.0

            self.out('Writing out spreadsheet records...')
            for tc in collated_cells:
                self.out.printf('.')

                tt, cl = parse_cell_name(tc)
                tetrode = get_unique_row(tetrode_table,
                    '(rat==%d)&(day==%d)&(tt==%d)'%(rat, day, tt))

                record['area'] = tetrode['area']
                record['area_sub'] = tetrode['subdiv']
                record['cell'] = tc

                record.update({ k: firing_rate(tc, k) for k in epochs })

                spreadsheet.write_record(record)
            self.out.printf('\n')

        # Finish up
        spreadsheet.close()
        self.out('All done!')


class ScanRateAnalysis(AbstractAnalysis):

    """
    Firing-rate averages across recordings areas for scan type x phase
    """

    label = 'scan rate'

    def collect_data(self, min_quality='fair', exclude_bad=False, ROI=None,
        exclude_zero_scans=False, only_m1=False, use_ec_table=False):
        """Collect normalized firing-rate information across region, scan type,
        and scan phase, for all valid clusters

        Arguments:
        min_quality -- minimum cluster isolation quality (string name or
            ClusterQuality value) for inclusion of data
        """
        self.results['quality'] = min_quality
        Quality = get_min_quality_criterion(min_quality)
        self.out('Cluster quality: at least %s'%min_quality)

        tetrode_table = get_node('/metadata', 'tetrodes')
        scan_table = get_node('/behavior', 'scans')

        # Only entorhinal if using the EC clusters table
        if use_ec_table:
            ec_table = get_node('/metadata', 'ec_clusters')
            ROI_areas = ['LEC', 'MEC']
        elif ROI:
            ROI_areas = ROI
        else:
            ROI_areas = spike.PrimaryAreas

        # Initialize main cluster data accumulator
        def new_arrays():
            return dict(INT=dict(   inbound=[], inbound_rate=[],
                                    inbound_rate_norm=[], inbound_diff=[],
                                    outbound=[], outbound_rate=[],
                                    outbound_rate_norm=[], outbound_diff=[]),
                        EXT=dict(   inbound=[], inbound_rate=[],
                                    inbound_rate_norm=[], inbound_diff=[],
                                    outbound=[], outbound_rate=[],
                                    outbound_rate_norm=[], outbound_diff=[]))
        cluster_data = { area: new_arrays() for area in ROI_areas }
        self.results['cluster_data'] = cluster_data

        scan_rate_data = { area: [] for area in ROI_areas }
        self.results['scan_rate_data'] = scan_rate_data

        scan_nonscan_diff = { area: [] for area in ROI_areas }
        self.results['scan_nonscan_diff'] = scan_nonscan_diff

        type_by_phase = [[x, y] for x in ('INT', 'EXT')
            for y in ('inbound', 'outbound')]

        cell_counts = { area: 0 for area in ROI_areas }
        self.results['cell_counts'] = cell_counts

        Criteria = AND(PrincipalCellCriteria, Quality)

        spreadsheet = DataSpreadsheet(
            os.path.join(self.datadir, 'scan_firing_rates.csv'),
            [   ('rat', 'd'), ('day', 'd'), ('area', 's'), ('cell', 's'),
                ('rate', 'f'), ('scan_rate', 'f'), ('nonscan_rate', 'f'),
                ('scan_nonscan_diff', 'f'), ('INT_outbound_diff', 'f'),
                ('INT_inbound_diff', 'f'), ('EXT_outbound_diff', 'f'),
                ('EXT_inbound_diff', 'f') ])
        self.out('Record string: %s'%spreadsheet.get_record_string())
        record = spreadsheet.get_record()

        def count_spikes(ts, start, end):
            return len(time.time_slice_sample(ts, start=start, end=end))

        for dataset in meta.walk_days():
            rat, day = dataset
            record['rat'], record['day'] = rat, day
            self.out('Analyzing dataset rat%03d-%02d.'%dataset)

            dataset_duration = 0.0
            dataset_spikes = {}

            N = dict(   INT=dict(inbound={}, outbound={}),
                        EXT=dict(inbound={}, outbound={})   )

            duration = dict(   INT=dict(inbound={}, outbound={}),
                               EXT=dict(inbound={}, outbound={})   )

            collated = []

            for maze in meta.get_maze_list(rat, day):
                if only_m1 and maze != 1:
                    continue

                rds = rat, day, maze
                grp = get_group(rds=rds)
                if exclude_bad:
                    attrs = grp._v_attrs
                    if attrs['HD_missing'] or attrs['timing_jumps']:
                        self.out('...bad dataset, skipping...')
                        continue

                if use_ec_table:
                    request = ['t%dc%d'%(rec['tt'], rec['cluster']) for
                        rec in ec_table.where(
                            '(rat==%d)&(day==%d)&(session==%d)'%rds)]
                    if request:
                        self.out('Found EC clusters: %s'%str(request)[1:-1])
                    else:
                        self.out('...no EC table clusters, skipping...')
                        continue
                else:
                    request = Criteria

                session_data = session.SessionData(rds=rds)
                dataset_duration += session_data.duration
                scan_list = \
                    [(scan['type'], scan['start'], scan['max'], scan['end'])
                        for scan in scan_table.where(session_data.session_query)
                        if scan['type'] != 'AMB']

                # Collate the spike counts, accumulating cells across sessions
                for tc in session_data.get_clusters(request):
                    cl_data = session_data.clusts[tc]
                    area = spike.get_tetrode_area(rat, day, cl_data.tt)
                    if area not in ROI_areas:
                        continue

                    # Initialize accumulators if first occurence of this cell
                    if (tc, area) not in collated:
                        collated.append((tc, area))
                        dataset_spikes[tc] = 0
                        for t, p in type_by_phase:
                            N[t][p][tc] = 0
                            duration[t][p][tc] = 0.0

                    t_spikes = cl_data.spikes
                    dataset_spikes[tc] += t_spikes.size

                    for scan_type, start, scan_max, end in scan_list:
                        if exclude_zero_scans and (count_spikes(t_spikes, start,
                            end) == 0):
                            continue

                        N[scan_type]['outbound'][tc] += \
                            count_spikes(t_spikes, start, scan_max)
                        N[scan_type]['inbound'][tc] += \
                            count_spikes(t_spikes, scan_max, end)
                        duration[scan_type]['outbound'][tc] += \
                            time.elapsed(start, scan_max)
                        duration[scan_type]['inbound'][tc] += \
                            time.elapsed(scan_max, end)

            self.out('Computing firing rates for %d cells...'%len(collated))
            for tc, area in collated:
                N_total = float(sum([N[t][p][tc] for t, p in type_by_phase]))
                duration_total = sum([duration[t][p][tc] for t, p in type_by_phase])
                if not duration_total:
                    continue

                record['cell'] = tc
                record['area'] = area

                scan_rate = N_total / duration_total
                scan_rate_data[area].append(scan_rate)
                record['scan_rate'] = scan_rate

                overall_rate = dataset_spikes[tc] / dataset_duration
                overall_nonscan_rate = \
                    (dataset_spikes[tc] - N_total) / (dataset_duration - duration_total)
                record['rate'] = overall_rate
                record['nonscan_rate'] = overall_nonscan_rate

                scan_nonscan_diff[area].append((scan_rate - overall_nonscan_rate) /
                    (scan_rate + overall_nonscan_rate))
                record['scan_nonscan_diff'] = scan_nonscan_diff[area][-1]

                cell_counts[area] += 1
                c_dict = cluster_data[area]
                for t, p in type_by_phase:
                    key = '%s_%s_diff'%(t, p)
                    record[key] = -99
                    if not (N_total and duration[t][p][tc]):
                        continue
                    c_dict[t][p].append(N[t][p][tc]/N_total)
                    this_scan_rate = N[t][p][tc]/duration[t][p][tc]
                    c_dict[t][p+'_rate'].append(this_scan_rate)
                    if this_scan_rate + scan_rate != 0:
                        c_dict[t][p+'_rate_norm'].append(
                            (this_scan_rate-scan_rate) /
                                (this_scan_rate+scan_rate))
                    if this_scan_rate + overall_nonscan_rate != 0:
                        c_dict[t][p+'_diff'].append(
                            (this_scan_rate-overall_nonscan_rate) /
                                (this_scan_rate+overall_nonscan_rate))
                        record[key] = c_dict[t][p+'_diff'][-1]
                spreadsheet.write_record(record)

        # Convert data to arrays
        for area in ROI_areas:
            self.out('Total cell count for %s: %d cells'%(area,
                cell_counts[area]))
            scan_rate_data[area] = np.array(scan_rate_data[area])
            scan_nonscan_diff[area] = np.array(scan_nonscan_diff[area])
            for scan_type in 'INT', 'EXT':
                c_dict = cluster_data[area][scan_type]
                for k in c_dict:
                    c_dict[k] = np.array(c_dict[k])

        spreadsheet.close()
        self.out("All done!")

    def process_data(self, ROI=None, norm_ymax=1.1):
        """Plot bar graphs of scan firing-rate data
        """
        self.out.outfd = file(os.path.join(self.datadir, 'figure.log'), 'w')
        self.figure = {}

        # Load data
        cluster_data = self.results['cluster_data']
        scan_rate_data = self.results['scan_rate_data']
        scan_nonscan_diff = self.results['scan_nonscan_diff']
        if ROI:
            area_list = ROI
        else:
            area_list = cluster_data.keys()
            area_list.sort()

        # Set up type x phase values and colors for bar plots
        type_by_phase = [[x, y] for x in ('INT', 'EXT')
            for y in ('outbound', 'inbound')]
        phase_names = [t+"_"+p for t, p in type_by_phase]
        phase_colors = 'b', 'c', 'r', 'm'
        phase_values = zip(phase_names, phase_colors)

        # Print out cell counts
        for i, area in enumerate(area_list):
            self.out('For %s: N=%d cells'%(area,
                len(cluster_data[area]['INT']['inbound'])))

        def type_phase_bar_plot(ax, suffix='', legend=True):
            color_dict = dict(EXT_outbound='r', EXT_inbound='m',
                INT_outbound='b', INT_inbound='c')

            mu = np.empty((len(area_list), len(type_by_phase)), 'd')
            sem = np.empty((len(area_list), len(type_by_phase)), 'd')
            medians = np.empty((len(area_list), len(type_by_phase)), 'd')
            for i, area in enumerate(area_list):
                for j, tp in enumerate(type_by_phase):
                    scan_type, scan_phase = tp
                    the_data = cluster_data[area][scan_type][scan_phase+suffix]
                    mu[i,j] = the_data.mean()
                    sem[i,j] = the_data.std()/np.sqrt(the_data.size)
                    medians[i, j] = np.median(the_data)
                    self.out('%s: %s scan, %s phase = %.3f +/- %.4f'%(area,
                        scan_type, scan_phase, mu[i,j], sem[i,j]))

            h = grouped_bar_plot(mu, area_list, phase_values, errors=sem, ax=ax,
                baselines=medians, legend=legend, legend_loc=2)
            ax.set_xlabel('Recording Area')
            ax.set_ylim(bottom=0)
            return ax

        # Spike fraction bar plot
        self.figure['fraction'] = f = plt.figure(figsize=(11, 7))
        f.suptitle('Scan Type x Phase Firing Fraction')
        ax = type_phase_bar_plot(plt.axes(), legend=False)
        ax.set_ylabel('Fraction of Total Scan Firing')
        ax.set_ylim(top=1)

        # Firing rate bar plot
        self.figure['firing_rates'] = f = plt.figure(figsize=(11, 7))
        f.suptitle('Scan Type x Phase Firing Rates')
        ax = type_phase_bar_plot(plt.axes(), '_rate')
        ax.set_ylabel('Firing Rate (spikes/s)')

        # Normalized firing rate bar plot
        self.figure['scan_normalized_rates'] = f = plt.figure(figsize=(11, 7))
        f.suptitle('Normalized Scan Type x Phase Firing Rates')
        ax = type_phase_bar_plot(plt.axes(), '_rate_norm')
        ax.set_ylabel('Normalized Firing Rate')
        ax.set_ylim(-norm_ymax, norm_ymax)

        # Scan - Nonscan normalized difference bar plot
        self.figure['scan_nonscan_diff'] = f = plt.figure(figsize=(11, 7))
        f.suptitle('Normalized Scan Type x Phase to Non-Scan Rate Difference')
        ax = type_phase_bar_plot(plt.axes(), '_diff')
        ax.set_ylabel('Scan - Nonscan, normalized')
        ax.set_ylim(-norm_ymax, norm_ymax)

        # Overall scan rate bar plot
        self.figure['overall_scan_rates'] = f = plt.figure(figsize=(5, 6))
        f.suptitle('Overall Scan Firing Rates')
        ax = plt.axes()
        mu = [scan_rate_data[area].mean() for area in area_list]
        sem = [scan_rate_data[area].std()/np.sqrt(scan_rate_data[area].size)
            for area in area_list]
        grouped_bar_plot(mu, 'Areas', area_list, errors=sem, ax=ax,
            legend_loc=2)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Firing Rate (spikes/s)')

        # Overall scan - nonscan difference rate bar plot
        self.figure['overall_scan_nonscan_diff'] = f = plt.figure(figsize=(5, 6))
        f.suptitle('[Scan - Non-scan] Normalized Difference')
        ax = plt.axes()
        mu = [scan_nonscan_diff[area].mean() for area in area_list]
        sem = [scan_nonscan_diff[area].std()/np.sqrt(scan_nonscan_diff[area].size)
            for area in area_list]
        medians = [np.median(scan_nonscan_diff[area]) for area in area_list]
        grouped_bar_plot(mu, 'Areas', area_list, errors=sem, ax=ax,
            baselines=medians, legend_loc=0)
        ax.set_ylim(-norm_ymax, norm_ymax)
        ax.set_ylabel('Scan - Nonscan, normalized')

        # Close log file
        self.out.outfd.close()


class ScanSpikeCorrelationReport(BaseDatasetReport, TetrodeSelect):

    label = "psth report"

    def collect_data(self, tetrode_query="area=='CA1'", scan_time="max",
        scan_type=None, min_quality="fair", shuffle=False, **kwds):
        """Perform scan-firing xcorrs on selected tetrodes

        Arguments:
        tetrode_query -- query string against the /metadata/tetrodes table
        scan_time -- time point of scan to lock onto (must be "start", "end",
            "mid", "max")
        min_quality -- minimum cluster isolation quality (string name or
            ClusterQuality value) for inclusion of data
        shuffle -- whether to randomly shuffle scan times

        Keywords are passed to the spike.xcorr function.
        """
        assert scan_time in ("start", "end", "mid", "max"), \
            "bad scan_time: %s"%scan_time

        ttable = get_node('/metadata', 'tetrodes')
        stable = get_node('/behavior', 'scans')

        units = ('use_millis' in kwds and kwds['use_millis']) and 'ms' or 's'
        overall_psth = None
        rat_psth = {}

        Quality = get_min_quality_criterion(min_quality)
        self.out("Quality filter: at least %s"%min_quality)

        for ratday, ax in self.get_plot(self._get_datasets(tetrode_query)):
            valid_tetrodes = self._get_valid_tetrodes(ratday, tetrode_query)
            Tetrodes = get_tetrode_restriction_criterion(valid_tetrodes)
            rat, day = ratday
            psth = {}

            ClusterFilter = AND(Tetrodes, Quality, PrincipalCellCriteria)

            for maze in session_list(rat, day, exclude_timing_issue=True):
                data = session.SessionData(rds=(rat, day, maze))
                clusters = data.get_clusters(ClusterFilter)

                if shuffle:
                    scan_times = data.random_timestamp_array(
                        size=len(stable.getWhereList(data.session_query)))
                else:
                    query = data.session_query
                    if scan_type is not None:
                        query += "&(type==\'%s\')"%scan_type.upper()
                    scan_times = [s[scan_time] for s in stable.where(query)]
                t_scan = data.to_time(scan_times)
                for cl in clusters:
                    t_spikes = data.to_time(data.get_spike_train(cl))
                    C, lags = xcorr(t_scan, t_spikes, **kwds)
                    if cl not in psth:
                        psth[cl] = np.zeros_like(C)
                    psth[cl] += C

            if not psth:
                ax.set_axis_off()
                continue

            # Initialize aggregators for overall psth and per-rat psth
            if overall_psth is None:
                overall_psth = np.zeros_like(C)
            if rat not in rat_psth:
                rat_psth[rat] = np.zeros_like(C)

            drawn = False
            fmt = dict(lw=0.5, c='b')
            max_corr = max([max(corr) for corr in psth.values()])
            for cl in psth:
                overall_psth += psth[cl]
                rat_psth[rat] += psth[cl]
                psth[cl] /= max_corr
                plot_correlogram((psth[cl], lags), is_corr=True, ax=ax,
                    plot_type="lines", fmt=fmt, zero_line=not drawn)
                drawn = True
            ax.set_yticks([])
            if self.lastrow:
                ax.set_xlabel('Spike Lag (%s)'%units)
            else:
                ax.set_xticks([])

        # For plotting overall and per-rat cross-correlograms
        self.results['lags'] = lags
        self.results['C_overall'] = overall_psth
        self.results['C_rat'] = rat_psth
        self.results['query'] = tetrode_query
        self.results['scan_time'] = scan_time
        self.results['shuffled'] = shuffle
        self.results['units'] = units
        self.results['scan_type'] = scan_type

        self.out('All done!')

    def process_data(self, **kwds):
        """Keywords are passed spike.plot_correlogram as style formatting.
        """
        plt.ioff()
        self.figure = {}
        res = self.results
        lags = res['lags']
        C = res['C_overall']
        C_rat = res['C_rat']

        # Set an informative title
        title = "Scan " + res['scan_time'] + ": " + res['query']
        if res['shuffled']:
            title += ' [shuffled]'

        fmt = dict(lw=1)
        fmt.update(kwds)
        corr_args = dict(is_corr=True, norm=False, plot_type="lines", fmt=fmt)

        # Plot the per-rat correlograms normalized in different ways
        C_rat_norm = []
        for norm in 0, 1, 2:
            f = plt.figure(figsize=(8, 7))
            if norm == 0:
                self.figure['rat'] = f
                rat_title = title + ' [per rat]'
            elif norm == 1:
                self.figure['rat_norm_max'] = f
                rat_title = title + ' [per rat] [norm max]'
            elif norm == 2:
                self.figure['rat_norm_zero'] = f
                rat_title = title + ' [per rat] [norm zero]'
            ax = plt.axes()
            for i, psth in enumerate(C_rat.values()):
                if norm == 1:
                    psth /= psth.max()
                    C_rat_norm.append(psth)
                elif norm == 2:
                    psth /= psth[lags==0]
                plot_correlogram((psth, lags), **corr_args)
            ax.set_xlabel('Spike Lag (%s)'%res['units'])
            ax.set_ylabel('Correlation')
            ax.set_title(rat_title)
        # if len(C_rat_norm):
        #     valid = filter(lambda x: np.isfinite(x).all(), C_rat_norm)
        #     C_rat_norm = np.array(C_rat_norm)
        #     mu = C_rat_norm.mean(axis=0)
        #     std = C_rat_norm.std(axis=0)
        #     ci = 1.96*std/np.sqrt(C_rat_norm.shape[0])

        # Plot overal correlogram with min/max per-rat envelope
        self.figure['overall'] = plt.figure(figsize=(8, 7))
        ax = plt.axes()
        fmt['lw'] = 2
        fmt['c'] = 'k'
        corr_args['norm'] = True
        h = plot_correlogram((C, lags), **corr_args)
        h.set_label('Overall Spikes')
        # if len(C_rat_norm):
        #     ax.plot(lags, mu, 'b-', zorder=-1, label='Rat Average')
        #     ax.plot(lags, mu+ci, 'r--', zorder=-2, label='Rat 95% CI')
        #     ax.plot(lags, mu-ci, 'r--', zorder=-2)
        ax.set_xlabel('Spike Lag (%s)'%res['units'])
        ax.set_ylabel('Correlation')
        ax.set_title(title)
        ax.legend(loc=4)

        plt.ion()
        plt.show()


class ECScanFiring(AbstractAnalysis, TetrodeSelect):

    """
    Analysis of scan vs non-scan firing rate differences between LEC and MEC
    clusters
    """

    label = 'entorhinal scans'

    def collect_data(self, min_quality='fair', shuffle_samples=200,
        shuffle_retry=20):
        """Collate and summarize statistics of scan vs. non-scan activity for
        MEC and LEC clusters

        Keyword arguments:
        min_quality -- if quality_filter is False, then this is the threshold for
            cluster isolation quality used to choose cells
        shuffle_samples -- number of randomized samples for empirical p-values
        """
        self.results['quality'] = min_quality
        Quality = get_min_quality_criterion(min_quality)
        self.out('Cluster quality: at least %s'%min_quality)

        scan_table = get_node('/behavior', 'scans')
        scan_firing_data = dict(MEC=[], LEC=[])

        LEC_datasets = self._get_datasets("area=='LEC'")
        MEC_datasets = self._get_datasets("area=='MEC'")
        area_list = ['LEC']*len(LEC_datasets) + ['MEC']*len(MEC_datasets)
        dataset_list = LEC_datasets+MEC_datasets

        # Set up spreadsheet output
        fn = os.path.join(self.datadir, 'entorhinal_scan_firing.csv')
        cols = [('dataset', 's'), ('maze', 'd'), ('type', 's'),
                ('parameter', 'd'), ('area', 's'), ('cluster', 's'),
                ('N_scans', 'd'), ('scan_rate_mean', 'f'), ('scan_rate_sd', 'f'),
                ('scan_rate_overall', 'f'), ('nonscan_rate_overall', 'f'),
                ('scan_nonscan_ratio', 'f') ]
        spreadsheet = DataSpreadsheet(fn, cols)
        record = spreadsheet.get_record()
        self.out('Record string: ' + spreadsheet.get_record_string())

        for area, dataset in zip(area_list, dataset_list):
            dataset_str = 'rat%03d-%02d'%dataset
            rat, day = dataset
            self.out('Analyzing %s for area %s.'%(dataset_str, area))
            area_query = "area=='%s'"%area

            record['dataset'] = dataset_str
            record['area'] = area

            for maze in data.session_list(rat, day):
                rds = rat, day, maze
                record['maze'] = maze

                Tetrodes = get_tetrode_restriction_criterion(
                    self._get_valid_tetrodes(dataset, area_query))
                Criteria = AND(PrincipalCellCriteria, Quality, Tetrodes)

                session_data = session.SessionData(rds=rds,
                    cluster_criteria=Criteria)
                total_time = session_data.duration
                record['type'] = session_data.data_group._v_attrs['type']
                record['parameter'] = session_data.data_group._v_attrs['parameter']

                scan_list = [tuple(scan['tlim']) for scan in
                    scan_table.where(session_data.session_query)]
                scan_list.sort()
                if len(scan_list) == 0:
                    continue
                record['N_scans'] = len(scan_list)

                for tc in session_data.get_clusters():
                    record['cluster'] = tc
                    cl_data = session_data.clusts[tc]
                    ts_spikes = cl_data.spikes

                    # Cell-session statistics
                    total_spikes = ts_spikes.size
                    session_firing_rate = total_spikes / total_time

                    # Initialize per-scan accumulators
                    scan_counts = []
                    scan_durations = []
                    scan_rates = []
                    scan_pvals = []
                    scan_norm = []

                    for start, end in scan_list:
                        scan_spikes = time.time_slice_sample(ts_spikes,
                            start=start, end=end)

                        scan_counts.append(scan_spikes.size)
                        this_scan_duration = time.elapsed(start, end)
                        scan_durations.append(this_scan_duration)
                        firing_rate = scan_spikes.size / this_scan_duration
                        scan_rates.append(firing_rate)

                        # Randomized firing-distributions for one-sided p-values
                        delta_ts = end - start
                        shuffle = np.empty((shuffle_samples,), 'd')
                        for i in xrange(shuffle_samples):
                            c = 0
                            while c < shuffle_retry:
                                # Get random time segment of same length in session
                                rand_start = long(session_data.start +
                                    plt.randint(session_data.end -
                                        session_data.start - delta_ts))
                                rand_end = long(rand_start + delta_ts)

                                # Only accept if not colliding with another scan...
                                hit = False
                                for s, e in scan_list:
                                    if ((rand_start <= s <= rand_end) or
                                        (rand_start <= e <= rand_end) or
                                        (s < rand_start and e > rand_start)):
                                        hit = True
                                        break
                                if hit:
                                    c += 1
                                else:
                                    break
                            rand_spikes = time.time_slice_sample(ts_spikes,
                                start=rand_start, end=rand_end)
                            shuffle[i] = rand_spikes.size / this_scan_duration
                        p_val = (1+(shuffle > firing_rate).sum()) / float(1+shuffle_samples)
                        scan_pvals.append(p_val)
                        scan_norm.append(firing_rate / session_firing_rate)

                    # Overall scan firing rate
                    overall_scan_rate = np.sum(scan_counts) / np.sum(scan_durations)

                    # Finish spreadsheet entry for this cluster
                    record['scan_rate_mean'] = np.mean(scan_rates)
                    record['scan_rate_sd'] = np.std(scan_rates)
                    record['scan_rate_overall'] = overall_scan_rate
                    record['nonscan_rate_overall'] = \
                        (total_spikes-np.sum(scan_counts)) / (total_time-np.sum(scan_durations))
                    record['scan_nonscan_ratio'] = 0.0
                    if record['nonscan_rate_overall']:
                        record['scan_nonscan_ratio'] = overall_scan_rate / record['nonscan_rate_overall']
                    spreadsheet.write_record(record)

                    # Create the final record for this cell-session
                    scan_row = (  np.mean(scan_rates),
                                  np.median(scan_rates),
                                  overall_scan_rate,
                                  np.mean(scan_norm),
                                  np.median(scan_norm),
                                  record['scan_nonscan_ratio'],
                                  np.mean(scan_pvals),
                                  np.median(scan_pvals)   )

                    # Store the record in an area-specific list
                    scan_firing_data[area].append(scan_row)

        # Save data as numpy record arrays
        firing_data = [ ('mean_rate', float), ('median_rate', float), ('overall_rate', float),
                        ('mean_norm', float), ('median_norm', float), ('overall_norm', float),
                        ('mean_pval', float), ('median_pval', float)  ]
        self.results['LEC'] = np.rec.fromrecords(scan_firing_data['LEC'],
            dtype=firing_data)
        self.results['MEC'] = np.rec.fromrecords(scan_firing_data['MEC'],
            dtype=firing_data)

        spreadsheet.close()
        self.out('All done!')

    def process_data(self):
        """Create figure plotting distributions of scan-firing measures and
        compute various relevant statistics
        """
        from .tools.stats import smooth_pdf, t_one_tailed
        from .tools.string import snake2title
        from scipy.stats import (ttest_ind as ttest, t as t_dist, ks_2samp as kstest)

        os.chdir(self.datadir)
        self.out.outfd = file('figure.log', 'w')
        self.out.timestamp = False

        self.figure = {}
        self.figure['distros'] = f = plt.figure(figsize=(10, 16))
        f.suptitle('Distributions of LEC/MEC Scan/Non-scan Firing')

        LEC = self.results['LEC']
        MEC = self.results['MEC']

        data_types = LEC.dtype.names
        N = len(data_types)

        def mean_pm_sem(a):
            return '%.4f +/- %.4f'%(a.mean(), a.std()/np.sqrt(a.size))

        plt.ioff()
        kw = dict(lw=2, aa=True)
        for i, data in enumerate(data_types):
            ax = plt.subplot(N, 1, i+1)
            label = snake2title(data)

            kw.update(label='LEC', c='b')
            ax.plot(*smooth_pdf(LEC[data]), **kw)

            kw.update(label='MEC', c='g')
            ax.plot(*smooth_pdf(MEC[data]), **kw)

            ax.axis('tight')
            v = list(ax.axis())
            v[3] *= 1.1
            ax.axis(v)

            med_LEC = np.median(LEC[data])
            med_MEC = np.median(MEC[data])
            ax.plot([med_LEC]*2, [v[2], v[3]], 'b--')
            ax.plot([med_MEC]*2, [v[2], v[3]], 'g--')

            self.out(label.center(50, '-'))
            N_LEC = LEC[data].size
            N_MEC = MEC[data].size
            self.out('Median LEC(%s) = %.4f'%(label, med_LEC))
            self.out('Mean/SEM LEC(%s) = %s'%(label, mean_pm_sem(LEC[data])))
            if data.endswith('norm'):
                self.out('T(LEC > 1) = %.4f, p = %.8f'%t_one_tailed(LEC[data]))
                self.out('T_cstv(LEC > 1) = %.4f, p = %.8f'%t_one_tailed(
                    LEC[data], df=N_LEC/float(5)-1))
            self.out('N LEC cell-sessions = %d'%N_LEC)
            self.out('Median MEC(%s) = %.4f'%(label, med_MEC))
            self.out('Mean/SEM MEC(%s) = %s'%(label, mean_pm_sem(MEC[data])))
            if data.endswith('norm'):
                self.out('T(MEC > 1) = %.4f, p = %.8f'%t_one_tailed(MEC[data]))
                self.out('T_cstv(MEC > 1) = %.4f, p = %.8f'%t_one_tailed(
                    MEC[data], df=N_MEC/float(5)-1))
            self.out('N MEC cell-sessions = %d'%N_MEC)

            t = ttest(LEC[data], MEC[data])
            k = kstest(LEC[data], MEC[data])
            self.out('T-test(%s): t = %.4f, p = %.8f'%(label, t[0], t[1]))
            self.out('KS-test(%s): D = %.4f, p = %.8f'%(label, k[0], k[1]))
            if t[1] < 0.05:
                ax.text(0.025, 0.8, '*t', size='x-large', transform=ax.transAxes)
            if k[1] < 0.05:
                ax.text(0.025, 0.6, '*KS', size='x-large', transform=ax.transAxes)

            ax.set_ylabel('p[ %s ]'%label)

            if i == 0:
                ax.legend(loc=1)

        plt.ion()
        plt.show()
        self.out.outfd.close()


def run_sweep():
    """Run primary areas by scan-lock timing by shuffling
    """
    for area in spike.PrimaryAreas:
        desc1 = area
        for scan_time in ['max']: #'start', 'end', 'mid', 'max':
            desc2 = desc1 + " " + scan_time
            for shuffled in [False]: #False, True:
                if shuffled:
                    desc3 = desc2 + " shuffle"
                else:
                    desc3 = desc2
                report = ScanSpikeCorrelationReport(desc=desc3)
                report( tetrode_query="area=='%s'"%area, scan_time=scan_time,
                        shuffle=shuffled, maxlag=3, bins=32)
                report.process_data()
                report.save_plots()
                plt.close('all')
