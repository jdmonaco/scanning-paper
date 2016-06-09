# encoding: utf-8
"""
spike_probability.py -- Probabilistic data-driven modeling of spike counts for
    head scanning events

Created by Joe Monaco on August 1, 2012.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

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
from scanr.field import mark_max_field, field_extent, cut_laps_opposite_field
from scanr.time import select_from
from .core.analysis import AbstractAnalysis
from scanr.tools.radians import xy_to_deg_vec
from scanr.tools.stats import integer_hist, SampleDistribution
from scanr.tools.plot import AxesList, textlabel, quicktitle

# Constants
RATEMAP_BINS = 48
MIN_SCAN_MAGNITUDE = 10
MIN_FIELD_RATE = 1.5
MIN_FIELD_SIZE = 15
MIN_TRAVERSAL_SIZE = 15
MIN_TRAVERSAL_SPIKES = 3

# Table descriptions
ScanDescr =     {   'rat'               :   tb.UInt16Col(pos=1),
                    'day'               :   tb.UInt16Col(pos=2),
                    'session'           :   tb.UInt16Col(pos=3),
                    'tc'                :   tb.StringCol(itemsize=8, pos=4),
                    'scan'              :   tb.UInt16Col(pos=5),
                    'field_distance'    :   tb.FloatCol(pos=6),
                    'traversal_distance':   tb.FloatCol(pos=7),
                    'strength'          :   tb.FloatCol(pos=8),
                    'field_size'        :   tb.FloatCol(pos=9),
                    'traversal_size'    :   tb.FloatCol(pos=10),
                    'out_spikes'        :   tb.UInt16Col(pos=11),
                    'in_spikes'         :   tb.UInt16Col(pos=12),
                    'spikes'            :   tb.UInt16Col(pos=13) }


class ScanFiringProbability(AbstractAnalysis):

    """
    Assess probability of within-field scans to contain spikes
    """

    label = 'scan field bias'

    def collect_data(self, min_quality='fair',
        exclude_table_names=('novel_fields', 'potentiation_0.5_tol_0.2')):
        """Excluding place fields from the tables listed in exclude_table_names
        under /physiology, categorize whether scans within the field fire or
        not based on the normalized distance through the field where the
        scan occurs. This should illustrate any field-based scan firing bias
        that may explain the observed bias in the scan-field cross-correlation
        data in scanr.ana.predictive.TrackAngleCorrelation.
        """
        area_query = '(area=="CA1")|(area=="CA3")'

        # Check modulation event table
        self.results['exclude_table_names'] = exclude_table_names
        exclude_tables = map(lambda t: get_node('/physiology', t), exclude_table_names)
        sessions_table = get_node('/metadata', 'sessions')
        scan_table = get_node('/behavior', 'scans')

        # Place-field tables and iterator
        data_file = self.open_data_file()
        scan_spike_table = data_file.createTable('/', 'field_scans',
            ScanDescr, title='In-field Scan Spiking Data')
        row = scan_spike_table.row

        # Quality criterion
        Quality = get_min_quality_criterion(min_quality)

        self.out('Gathering place field scanning data...')
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

                # Get timing of scan start, max, and end
                scan_timing = session_data.T_(np.array(
                    [(rec['start'], rec['max'], rec['end'])
                        for rec in scan_table.where(session_data.session_query)]))
                scan_magnitude = np.array([rec['magnitude']
                    for rec in scan_table.where(session_data.session_query)])

                if not scan_timing.size:
                    continue

                self.out.printf('Scanning: ', color='lightgray')
                for tc in session_data.get_clusters():

                    # Check for any events for this cell, skip if found
                    skip_unstable = False
                    for table in exclude_tables:
                        found = table.getWhereList(session_data.session_query +
                            '&(tc=="%s")'%tc)
                        if len(found):
                            skip_unstable = True
                            break
                    if skip_unstable:
                        self.out.printf(u'\u25a0', color='red')
                        continue

                    # Get pooled ratemap and discard weak place fields
                    ratemap_kwds = dict(bins=RATEMAP_BINS,
                        blur_width=360/RATEMAP_BINS)
                    ratemap_kwds.update(session_data.running_filter())
                    R_pooled = session_data.get_cluster_ratemap(tc, **ratemap_kwds)
                    if R_pooled.max() < MIN_FIELD_RATE:
                        self.out.printf(u'\u25a1', color='red')
                        continue

                    # Mark pooled field and discard small place fields
                    field = mark_max_field(R_pooled, floor=0.1, kill_on=2)
                    start, end = field_extent(field)
                    wrapped = start > end
                    field_size = wrapped and (360 - start + end) or (end - start)
                    if field_size < MIN_FIELD_SIZE:
                        self.out.printf(u'\u25a1', color='red')
                        continue

                    # Output indication that we are processing a place field
                    self.out.printf(u'\u25a1', color='green')

                    # Cut laps opposite COM, get spike trains, spike angles
                    cut_laps_opposite_field(session_data, tc, R=R_pooled)
                    cdata = session_data.cluster_data(tc)
                    run_ix = session_data.filter_tracking_data(cdata.spikes, cdata.x, cdata.y,
                        boolean_index=True, **session_data.running_filter())
                    t_all_spikes = session_data.T_(cdata.spikes)
                    t_run_spikes = t_all_spikes[run_ix]
                    alpha_run_spikes = xy_to_deg_vec(cdata.x[run_ix], cdata.y[run_ix])
                    in_field = (wrapped and np.logical_or or np.logical_and)(
                        alpha_run_spikes >= start, alpha_run_spikes <= end)

                    for i in xrange(1, session_data.N_laps - 1):

                        # this loop skips first and last laps to avoid problems with finding
                        # complete traversals on incomplete laps

                        lap_interval = [session_data.T_(session_data.laps[i:i+2])]

                        # Find traversal spikes on this lap, ignore if smaller than threshold
                        in_lap = select_from(t_run_spikes, lap_interval)
                        in_traversal = np.logical_and(in_lap, in_field)
                        if in_traversal.sum() < MIN_TRAVERSAL_SPIKES:
                            continue
                        alpha_traversal_spikes = alpha_run_spikes[in_traversal]
                        start_traversal, end_traversal = alpha_traversal_spikes[-1], alpha_traversal_spikes[0]
                        wrapped_traversal = start_traversal > end_traversal
                        if wrapped_traversal:
                            traversal_size = 360 - start_traversal + end_traversal
                        else:
                            traversal_size = end_traversal - start_traversal
                        if traversal_size < MIN_TRAVERSAL_SIZE:
                            continue

                        strength = in_traversal.sum() / t_run_spikes[in_traversal].ptp() # rough firing rate

                        # Indices of scans on this lap meeting the minimum magnitude threshold
                        lap_scan_ix = np.logical_and(
                            select_from(scan_timing[:,0], lap_interval),
                            scan_magnitude >= MIN_SCAN_MAGNITUDE
                            ).nonzero()[0]

                        for scan_ix in lap_scan_ix:
                            scan = session_data.F_('alpha_unwrapped')(
                                scan_timing[scan_ix,0]) % 360

                            # Compute field traversal-normalized scan locations for wrapped and
                            # not-wrapped linear fields; skip non-field scans
                            if wrapped_traversal:
                                if scan >= start_traversal:
                                    norm_dist_traversal = (scan - start_traversal) / traversal_size
                                elif scan <= end_traversal:
                                    norm_dist_traversal = (360 - start_traversal + scan) / traversal_size
                                else:
                                    continue
                            else:
                                if start_traversal <= scan <= end_traversal:
                                    norm_dist_traversal = (scan - start_traversal) / traversal_size
                                else:
                                    continue

                            # ...and for the pooled field
                            if wrapped:
                                if scan >= start:
                                    norm_dist_field = (scan - start) / field_size
                                elif scan <= end:
                                    norm_dist_field = (360 - start + scan) / field_size
                            else:
                                norm_dist_field = (scan - start) / field_size

                            # Convert to running direction (CCW -> CW)
                            norm_dist_field = 1 - norm_dist_field
                            norm_dist_traversal = 1 - norm_dist_traversal

                            # Count the number of scan spikes
                            N_out_spikes = select_from(t_all_spikes, [scan_timing[scan_ix,:2]]).sum()
                            N_in_spikes = select_from(t_all_spikes, [scan_timing[scan_ix,1:]]).sum()
                            N_spikes = select_from(t_all_spikes, [scan_timing[scan_ix,(0,2)]]).sum()

                            # Add row to field-scan table
                            row['rat'] = rat
                            row['day'] = day
                            row['session'] = session['session']
                            row['tc'] = tc
                            row['scan'] = scan_ix + 1
                            row['field_distance'] = norm_dist_field
                            row['traversal_distance'] = norm_dist_traversal
                            row['strength'] = strength
                            row['field_size'] = field_size
                            row['traversal_size'] = traversal_size
                            row['out_spikes'] = N_out_spikes
                            row['in_spikes'] = N_in_spikes
                            row['spikes'] = N_spikes
                            row.append()

                self.out.printf('\n')
            scan_spike_table.flush()
        self.out('All done!')

    def process_data(self, disp_max_k=19.2, max_k=25, drange=(0, 1, 11), rrange=(0, 15, 4),
        srange=(0, 120, 3), which='spikes'):
        """Multi-variate likelihood estimation of scan spike counts up to max_k,
        for ranges specified in normalized field-traversal distance (drange),
        firing rate (rrange), and size (srange). Ranges specified as inclusive
        bin-edge tuples (low, high, N). Select spike counts from outbound,
        inbound, or whole-scan by setting *which* to 'out_spikes', 'in_spikes',
        or 'spikes'.
        """
        assert which in ('out_spikes', 'in_spikes', 'spikes'), \
            "invalid scan phase spikes argument"

        self.out.outfd = file(os.path.join(self.datadir, 'likelihood.log'), 'w')
        self.out.timestamp = False

        # Load results data
        data_file = self.get_data_file()
        scan_data = data_file.root.field_scans
        # 'field_distance'
        # 'traversal_distance'
        # 'spikes'
        # 'field_size'
        # 'traversal_size'
        # 'strength'

        # Compute empirical likelihood functions of scan spikes based on field
        # spiking data
        dbins = np.linspace(*drange)
        rbins = np.r_[np.linspace(*rrange), np.inf] # no upper bound on final bins
        sbins = np.r_[np.linspace(*srange), np.inf] # no upper bound on final bins
        edges = (dbins, rbins, sbins)
        N = (dbins.size - 1, rbins.size - 1, sbins.size - 1, max_k + 1)
        H = np.zeros(N, 'd')
        P = np.empty(N[:-1], object)

        def make_query(name, low, high):
            q = '(%s>=%f)'%(name, low)
            if np.isfinite(high):
                q += '&(%s<%f)'%(name, high)
            return q

        # Compute the multi-variate conditional probability distributions
        for d in xrange(N[0]):
            dquery = make_query('traversal_distance', dbins[d], dbins[d+1])
            self.out('%d: %s'%(d, dquery))
            for r in xrange(N[1]):
                rquery = make_query('strength', rbins[r], rbins[r+1])
                self.out('%d: %s'%(r, rquery))
                for s in xrange(N[2]):
                    squery = make_query('traversal_size', sbins[s], sbins[s+1])
                    self.out('%d: %s'%(s, squery))
                    query = '&'.join([dquery, rquery, squery])
                    spikes = [rec[which] for rec in scan_data.where(query)]
                    k, histo = integer_hist(spikes,
                        int_range=(0, max_k), open_range=True, relative=False)
                    if not np.any(histo):
                        histo[0] = 1 # set 0-mode for unsampled bins
                    H[d,r,s] = histo
                    P[d,r,s] = SampleDistribution(histo, k)
                self.out('-'*3)
            self.out('-'*3)

        self.out.outfd.close()
        self.close_data_file()

        # Save the likelihood data
        self.out('Saving edges...')
        edges_fd = file(os.path.join(self.datadir, 'edges.pickle'), 'w')
        cPickle.dump(edges, edges_fd)
        edges_fd.close()

        self.out('Saving histograms...')
        H_fd = file(os.path.join(self.datadir, 'H.pickle'), 'w')
        cPickle.dump(H, H_fd)
        H_fd.close()

        self.out('Saving distributions...')
        P_fd = file(os.path.join(self.datadir, 'P.pickle'), 'w')
        cPickle.dump(P, P_fd)
        P_fd.close()

        # Create distance x firing-rate figure
        kvals = np.arange(max_k + 1)
        self.figure = {}
        self.out('Bringing up distance x rate figure...')
        self.figure['bias_distance_rate'] = f = plt.figure(figsize=(10,8))
        axlist = AxesList()
        axlist.make_grid(N[2])
        f.suptitle('Distribution Means for Scan-%s: Traversal Distance x Firing Rate'%
            which.replace('_','-').title())

        for s, ax in enumerate(axlist):

            M = np.empty((N[0], N[1]), 'd')
            for d in xrange(N[0]):
                for r in xrange(N[1]):
                    M[d,r] = (kvals * (H[d,r,s] / H[d,r,s].sum())).sum()

            lrbt = [dbins[0], dbins[-1],
                    rbins[0], 2 * rbins[-2] - rbins[-3]]
            ax.imshow(M.T, aspect='auto', origin='lower', vmin=(0, disp_max_k),
                interpolation='nearest', extent=lrbt)

            ax.set(xlim=lrbt[:2], ylim=lrbt[2:])
            if s == 0:
                ax.set_ylabel('Traversal Rate')
            else:
                ax.set_yticklabels([])
            if s == N[2] - 1:
                ax.set_xlabel('Traversal Distance')
            else:
                ax.set_xticklabels([])
            textlabel(ax, '%.1f'%M.max())
            quicktitle(ax, 'size > %d'%int(sbins[s]))

        # Create distance x size figure
        self.out('Bringing up distance x size figure...')
        self.figure['bias_distance_size'] = f = plt.figure(figsize=(10,8))
        axlist = AxesList()
        axlist.make_grid(N[1])
        f.suptitle('Distribution Means for Scan-%s: Traversal Distance x Size'%
            which.replace('_','-').title())

        for r, ax in enumerate(axlist):

            M = np.empty((N[0], N[2]), 'd')
            for d in xrange(N[0]):
                for s in xrange(N[2]):
                    M[d,s] = (kvals * (H[d,r,s] / H[d,r,s].sum())).sum()

            lrbt = [dbins[0], dbins[-1],
                    sbins[0], 2 * sbins[-2] - sbins[-3]]
            ax.imshow(M.T, aspect='auto', origin='lower', vmin=(0, disp_max_k),
                interpolation='nearest', extent=lrbt)

            ax.set(xlim=lrbt[:2], ylim=lrbt[2:])
            if r == 0:
                ax.set_ylabel('Traversal Size')
            else:
                ax.set_yticklabels([])
            if r == N[1] - 1:
                ax.set_xlabel('Traversal Distance')
            else:
                ax.set_xticklabels([])
            textlabel(ax, '%.1f'%M.max())
            quicktitle(ax, 'rate > %.1f'%rbins[r])

        # Create distance x size figure
        self.out('Bringing up rate x size figure...')
        self.figure['bias_rate_size'] = f = plt.figure(figsize=(10,8))
        axlist = AxesList()
        axlist.make_grid(N[0])
        f.suptitle('Distribution Means for Scan-%s: Traversal Firing-rate x Size'%
            which.replace('_','-').title())

        for d, ax in enumerate(axlist):

            M = np.empty((N[1], N[2]), 'd')
            for r in xrange(N[1]):
                for s in xrange(N[2]):
                    M[r,s] = (kvals * (H[d,r,s] / H[d,r,s].sum())).sum()

            lrbt = [rbins[0], 2 * rbins[-2] - rbins[-3],
                    sbins[0], 2 * sbins[-2] - sbins[-3]]
            ax.imshow(M.T, aspect='auto', origin='lower', vmin=(0, disp_max_k),
                interpolation='nearest', extent=lrbt)

            ax.set(xlim=lrbt[:2], ylim=lrbt[2:])
            if d == 0:
                ax.set_ylabel('Traversal Size')
            else:
                ax.set_yticklabels([])
            if d == N[0] - 1:
                ax.set_xlabel('Traversal Firing-Rate')
            else:
                ax.set_xticklabels([])
            textlabel(ax, '%.1f'%M.max())
            quicktitle(ax, 'distance > %.2f'%dbins[d])


        # View flattened distributions
        CDF_flat = np.empty((H.shape[0]*H.shape[1]*H.shape[2], H.shape[3]), 'd')
        PDF_flat = np.empty((H.shape[0]*H.shape[1]*H.shape[2], H.shape[3]), 'd')
        i = 0
        for d in xrange(N[0]):
            for r in xrange(N[1]):
                for s in xrange(N[2]):
                    CDF_flat[i] = np.cumsum(H[d,r,s]) / np.sum(H[d,r,s])
                    PDF_flat[i] = H[d,r,s] / np.max(H[d,r,s])
                    i += 1

        self.figure['all_cdfs'] = f = plt.figure(figsize=(10,8))
        f.suptitle('Empirical CDFs for All %s Distributions'%
            which.replace('_','-').title())
        ax = f.add_subplot(111)
        ax.imshow(CDF_flat[:,:-1], origin='upper', interpolation='nearest',
            aspect='auto', vmin=(0, 1))
        ax.set_xlabel('Spike Count')
        ax.set_ylabel('(distance, rate, size)')
        quicktitle(ax, 'Scan-Spike Bias ECDFs')

        self.figure['all_pdfs'] = f = plt.figure(figsize=(10,8))
        f.suptitle('Empirical PDFs for All %s Distributions'%
            which.replace('_','-').title())
        ax = f.add_subplot(111)
        ax.imshow(PDF_flat, origin='upper', interpolation='nearest',
            aspect='auto', vmin=(0, 1))
        ax.set_xlabel('Spike Count')
        ax.set_ylabel('(distance, rate, size)')
        quicktitle(ax, 'Scan-Spike Bias EPDFs')

        self.out('All done!')


class SimulatedTrackAngleCorrelation(AbstractAnalysis):

    """
    Modified data collection for null scan-spike distributions based
    on simulations of scan-spike likelihood data from a ScanSpikingModel.
    """

    def collect_data(self,
        tables=('novel_fields', 'potentiation_0.5_tol_0.2'),
        bias_datadir='/Users/joe/projects/output/scan_field_bias-bug_fixes-00/',
        scan_phase='all', shuffles=DEFAULT_SHUFFLES, id_fmt='%08d'):
        """Collect unwrapped track-angle data for scan trains and field
        potentiation events

        bias_datadir -- directory containing ScanFiringProbability results
        scan_phase -- either 'out', 'in', or 'all', to specify for which phase
            of scans spikes should be counted
        tables -- names of modulation tables under /physiology that contain all
            modulation events for analysis
        shuffles -- number of simulated shuffle samples
        """
        scan_table = get_node('/behavior', 'scans')
        session_table = get_node('/metadata', 'sessions')
        tetrode_table = get_node('/metadata', 'tetrodes')

        self.results['tables'] = tables
        self.results['shuffles'] = shuffles

        # Create tables and data groups for observed and shuffled data
        data_file = self.open_data_file()
        observed_table = data_file.createTable('/', 'observed', XCorrDescr,
            title='Observed Scan-Field Cross-Correlation Data')
        observed_group = data_file.createGroup('/', 'observed_data',
            title='Object Database of Observed Data Arrays')
        shuffled_group = data_file.createGroup('/', 'shuffled_data',
            title='Object Database of Shuffled Data Arrays')

        angle_fmt = 'angles_' + id_fmt
        count_fmt = 'counts_' + id_fmt

        if scan_phase == 'all':
            phase_ix = (0,2)
        elif scan_phase == 'out':
            phase_ix = (0,1)
        elif scan_phase == 'in':
            phase_ix = (1,2)
        else:
            raise ValueError, 'invalid scan phase: %s'%str(scan_phase)

        i = 0
        row = observed_table.row
        for table_name in tables:
            field_table = get_node('/physiology', table_name)
            self.out('Analyzing %s...'%field_table._v_pathname)

            for rds, tc in unique_cells(field_table):
                session_data = SessionData.get(rds, load_clusters=False)
                F_alpha = session_data.F_('alpha_unwrapped')
                attrs = get_unique_row(session_table, session_data.session_query)
                cluster = session_data.cluster_data(tc)
                tt, cl = parse_cell_name(tc)
                tetrode = get_unique_row(tetrode_table, '(rat==%d)&(day==%d)&(tt==%d)'%(
                    rds[0], rds[1], tt))

                # Get timing of scan start, max, and end
                scan_timing = np.array(
                    [(rec['start'], rec['max'], rec['end'])
                        for rec in scan_table.where(session_data.session_query +
                            '&(magnitude>=%f)'%MIN_SCAN_MAGNITUDE)])

                # Get initial activity and scan info from data table
                cell_query = session_data.session_query + '&(tc=="%s")'%tc
                for event in field_table.where(cell_query):
                    self.out.printf('.')
                    row['event_type'] = table_name
                    row['id'] = i
                    row['rat'], row['day'], row['session'] = rds
                    row['tc'] = tc
                    row['type'] = attrs['type']
                    row['expt_type'] = attrs['expt_type']
                    row['area'] = tetrode['area']
                    row['field_start'] = F_alpha(session_data.T_(event['tlim'][0]))
                    row['field_end'] = F_alpha(session_data.T_(event['tlim'][1]))
                    angle_key, count_key = angle_fmt%i, count_fmt%i
                    row['scan_angle_id'] = angle_key
                    row['scan_count_id'] = count_key
                    row.append()

                    if len(scan_timing):
                        data_file.createArray(observed_group, angle_key,
                            F_alpha(session_data.T_(scan_timing[:,0])))
                        data_file.createArray(observed_group, count_key,
                            np.array([np.sum(select_from(cluster.spikes, [(u,v)]))
                                for u,v in scan_timing[:,phase_ix]]))
                    else:
                        data_file.createArray(observed_group, angle_key,
                            np.array([]))
                        data_file.createArray(observed_group, count_key,
                            np.array([]))

                    i += 1

                if i % 10 == 0:
                    self.out.printf('|', color='cyan')
                    observed_table.flush()

            self.out.printf('\n')

        data_file.flush()
        self.results['N_events'] = N_events = i
        self.out('Found %d fields in %d field-modulation tables.'%(N_events, len(tables)))

        # Simulated shuffled-scan-timing data for every observed event
        self.out('Loading scan-field bias data for null simulations...')
        spike_model = ScanSpikingModel(bias_datadir)
        pmin, pmax = POOL_RANGE
        scan_offset = lambda r: 2*(pmax - pmin)*(r - 0.5) + pmin*np.sign(r - 0.5)
        i_event = 0
        for event in observed_table.iterrows():
            rat, day, session = event['rat'], event['day'], event['session']
            rds = rat, day, session
            tc = event['tc']

            # Load session and cluster data
            session_data = SessionData.get(rds, load_clusters=False)
            cluster = session_data.cluster_data(tc)
            t_spikes = session_data.T_(cluster.spikes)
            scan_timing = session_data.T_(
                np.array([(rec['start'], rec['max'], rec['end'])
                    for rec in scan_table.where(
                        session_data.session_query + '&(magnitude>=%f)'%
                            MIN_SCAN_MAGNITUDE)]))
            scan_list = scan_timing[:,phase_ix]
            N_scans = len(scan_list)

            # Accumulate pool of all scan information in all *other sessions*
            # of the *same dataset* for random resampling
            scan_list_pool = np.array([0,0], 'd') # start with dummy row
            scan_start_pool = np.array([], 'd')
            for maze in get_maze_list(rat, day):
                if maze == session:
                    continue
                data = SessionData.get((rat, day, maze), load_clusters=False)
                scan_list = data.T_([rec['tlim'] for rec in
                    scan_table.where(data.session_query +
                        '&(magnitude>=%f)'%MIN_SCAN_MAGNITUDE)])
                if len(scan_list):
                    scan_list_pool = np.vstack((scan_list_pool, scan_list))
                    scan_start_pool = np.r_[scan_start_pool,
                        session_data.F_('alpha_unwrapped')(scan_list[:,0])]
            N_pool = scan_start_pool.size
            scan_list_pool = scan_list_pool[1:] # remove first dummy row

            # Place field information for scan-spike model inputs
            ratemap_kwds = dict(bins=RATEMAP_BINS, blur_width=360/RATEMAP_BINS)
            ratemap_kwds.update(session_data.running_filter())
            R_pooled = session_data.get_cluster_ratemap(tc, **ratemap_kwds)
            field = mark_max_field(R_pooled, floor=0.1, kill_on=2)
            start, end = field_extent(field)
            wrapped = start > end
            field_size = wrapped and (360 - start + end) or (end - start)
            no_traversal_simulation = (
                field_size < MIN_FIELD_SIZE or R_pooled.max() < MIN_FIELD_RATE)

            # Cut laps opposite COM, get spike trains, spike angles
            cut_laps_opposite_field(session_data, tc, R=R_pooled)
            run_ix = session_data.filter_tracking_data(
                cluster.spikes, cluster.x, cluster.y,
                boolean_index=True, **session_data.running_filter())
            t_run_spikes = t_spikes[run_ix]
            unwrapped_run_spikes = session_data.F_('alpha_unwrapped')(
                t_run_spikes)

            # Create shuffle samples of scan angles, allow simple extrapolation
            # of spikes as scan-spikes by default; following loop will find
            # and simulate scans occuring during field traversals
            self.out.printf('.', color='red')
            shuffled_scan_start = np.zeros((shuffles, N_scans), 'd')
            shuffled_scan_spike_counts = np.zeros((shuffles, N_scans), 'i')
            if N_scans:
                for j in xrange(shuffles):
                    ix = permutation(N_pool)[:N_scans] # choose without replacement
                    six = ix[np.argsort(scan_start_pool[ix])][::-1] # descending start-angle index sort
                    shuffled_scan_start[j] = scan_start_pool[six]
                    shuffled_scan_spike_counts[j] = [
                        ((t_spikes >= u) * (t_spikes <= v)).sum()
                        for u, v in scan_list_pool[six]]

            # Find all the pooled scan occuring during place-field traversals,
            # except for last lap due to irregular lap size
            self.out.printf('.', color='cyan')
            scans_in_traversal = []
            t_laps = session_data.T_(session_data.laps[:-1]) # exclude last lap
            for lap_interval in zip(t_laps[:-1], t_laps[1:]):
                if no_traversal_simulation:
                    continue

                # Complicated algorithm that finds traversal spikes and size,
                # rejecting traversals with <2 spikes or <10 degrees
                alpha_lap_end = session_data.F_('alpha_unwrapped')(lap_interval[1])
                in_traversal = np.logical_and(
                    unwrapped_run_spikes <= alpha_lap_end + 360,
                    unwrapped_run_spikes > alpha_lap_end)
                if in_traversal.sum() < MIN_TRAVERSAL_SPIKES:
                    continue
                unwrapped_traversal_spikes = unwrapped_run_spikes[in_traversal]
                alpha_traversal_spikes = unwrapped_traversal_spikes % 360
                start_traversal, end_traversal = (alpha_traversal_spikes[-1],
                    alpha_traversal_spikes[0])
                wrapped_traversal = start_traversal > end_traversal
                if wrapped_traversal:
                    size = 360 - start_traversal + end_traversal
                else:
                    size = end_traversal - start_traversal
                if size < MIN_TRAVERSAL_SIZE:
                    continue

                # Traversal firing rate
                rate = in_traversal.sum() / t_run_spikes[in_traversal].ptp()

                # Store data about scans that occurred during this traversal
                for j, unwrapped_scan in enumerate(scan_start_pool):
                    if unwrapped_scan > unwrapped_traversal_spikes.max() or \
                        unwrapped_scan < unwrapped_traversal_spikes.min():
                        continue

                    # Compute field traversal-normalized scan locations for wrapped and
                    # not-wrapped linear fields; skip non-field scans
                    scan = unwrapped_scan % 360
                    if wrapped_traversal:
                        if scan >= start_traversal:
                            norm_dist = (scan - start_traversal) / size
                        else:
                            norm_dist = (360 - start_traversal + scan) / size
                    else:
                        norm_dist = (scan - start_traversal) / size

                    # Convert to running direction (CCW -> CW)
                    norm_dist = 1 - norm_dist

                    drs = dict(distance=norm_dist, rate=rate, size=size)
                    scans_in_traversal.append((j, drs))

            # Simulate spike-counts for traversal scans
            for scan_ix, drs in scans_in_traversal:
                match = (shuffled_scan_start == scan_start_pool[scan_ix])
                K = spike_model.get_distro(**drs)(match.sum())
                shuffled_scan_spike_counts[match] = K

            self.out.printf('.', color='yellow')
            if N_scans:
                data_file.createArray(shuffled_group, event['scan_angle_id'],
                    shuffled_scan_start)
                data_file.createArray(shuffled_group, event['scan_count_id'],
                    shuffled_scan_spike_counts)
            else:
                data_file.createArray(shuffled_group, event['scan_angle_id'],
                    np.array([]))
                data_file.createArray(shuffled_group, event['scan_count_id'],
                    np.array([]))

            i_event += 1
            if i_event % 10 == 0:
                self.out.printf('|', color='green')
                shuffled_group._f_flush()

        self.out.printf('\n')


# Scan spiking simulator based on interpolation and empirical sampling
# of bias distributions

def _floor(xe):
    x, e = xe
    if x < e[0]:
        return 0
    elif x >= e[-1]:
        return e.size - 2
    return (x >= e).nonzero()[0][-1]


class ScanSpikingModel(object):

    """
    Simulate the number of spikes for a scan during a place-field traversal
    """

    def __init__(self, bias):
        if hasattr(bias, 'datadir'):
            loaddir = bias.datadir
        elif os.path.isdir(bias):
            loaddir = bias
        else:
            raise ValueError, 'invalid spike-data specification: %s'%str(bias)
        self.P = np.load(os.path.join(bias, 'P.pickle'))
        self.edges = tuple(map(np.float32,
            np.load(os.path.join(loaddir, 'edges.pickle'))))
        self.bias = bias

    def __call__(self, **kwds):
        """As in get_distro, but returns a sample from the distribution
        """
        return self.get_distro(**kwds)()

    def _interpolate(self, *values):
        return tuple(map(_floor, zip(values, self.edges)))

    def _distro(self, index):
        return self.P[index]

    def get_distro(self, distance=None, rate=None, size=None):
        """Call this model object with the place-field traversal normalized
        distance, firing rate, and size in degrees.

        Returns probability distribution.
        """
        assert None not in (distance, rate, size), """traversal distance,
            firing rate, and size required"""
        return self._distro(self._interpolate(distance, rate, size))

