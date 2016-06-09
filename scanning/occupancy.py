# encoding: utf-8
"""
occupancy.py -- Compute behavior-space occupancy distributions

Created by Joe Monaco on 2011-10-24.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Package imports
from scanr.lib import *
from .firing_modulation import get_bin_edges, display_modulation, GainFunction
from .core.analysis import AbstractAnalysis
from scanr.tools.interp import linear_upsample

# Constants
CfgScan = Config['scanning']
CfgMod = Config['modulation']
SCAN_TYPES = [CfgScan['in_type'], CfgScan['out_type']]
SCAN_PHASES = ['all', 'outbound', 'inbound']
MOMENTS = filter(lambda x: x != 'alpha', behavior.Moment.keys())


class BehaviorOccupancy(AbstractAnalysis):

    """
    Behavioral occupancy distributions for type x phase taxonomy of head scan
    events
    """

    label = 'occupancy'

    @classmethod
    def munge_data_key(cls, scan_type, scan_phase, moment):
        """Convert [type, phase] x moment to unique dictionary key
        """
        return '%s-%s-%s'%(scan_type.lower(), scan_phase, moment)

    def collect_data(self, upsampling=4):
        """Collect baseline behavioral data for head scanning events

        Arguments:
        upsampling -- factor for upsampling baseline data for better
            coverage (less data loss due to resolution mismatch)
        """
        scan_table = get_node('/behavior', 'scans')
        sessions_table = get_node('/metadata', 'sessions')
        session_query = 'missing_HD==False'

        # Flatten scan type, scan phase, and moment pairs
        tpm = [[], [], []]
        for scan_type in SCAN_TYPES:
            for scan_phase in SCAN_PHASES:
                for moment in MOMENTS:
                    tpm[0].append(scan_type)
                    tpm[1].append(scan_phase)
                    tpm[2].append(moment)
        self.results['type_phase_moments'] = tpm

        # Initialize data accumulators
        scan_data = {}
        for selector in zip(*tpm):
            key = self.munge_data_key(*selector)
            scan_data[key] = np.array([], 'd')

        # Iterator through various data selection criteria to collect data
        for session in sessions_table.where(session_query):
            rat, day, maze = session['rat'], session['day'], session['session']
            rds = rat, day, maze

            self.out('Collecting behavior data for rat%03d-%02d m%d...'%rds)
            traj = tracking.TrajectoryData(rds=rds)
            M = Moment.get(traj)
            factor = int(upsampling / (traj.fs/30.0))

            scan_query = "(rat==%d)&(day==%d)&(session==%d)&(type!='AMB')"%rds
            for scan in scan_table.where(scan_query):
                ix = dict(  all=slice(*scan['slice']),
                            outbound=slice(*scan['outbound']),
                            inbound=slice(*scan['inbound'])  )

                for scan_phase in SCAN_PHASES:
                    for moment in MOMENTS:
                        key = self.munge_data_key(scan['type'], scan_phase,
                            moment)
                        scan_data[key] = np.r_[scan_data[key],
                            linear_upsample(M[moment][ix[scan_phase]],
                                factor)]

        # Save data
        savedir = os.path.join(self.datadir, 'data')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.results['data_files'] = []
        for k in scan_data:
            save_fn = os.path.join(savedir, '%s.npy'%k)
            self.results['data_files'].append(save_fn)
            np.save(save_fn, scan_data[k])
            self.out('Saved: %s'%save_fn)
        self.out('All done!')

    def process_data(self, show_area=None, no_scan_data=False, bins=None):
        """Plot occupancy histograms for behavioral data by scan type and phase

        Keyword arguments:
        show_area -- whether to show area-based firing modulation plots as
            the background with scan data as foreground contour plots; this
            results in half as many figures, as each scan type is plotted on
            the same figure in different color contours
        no_scan_data -- in combination with show_area, this shows just the
            firing modulation plots with no scanning data contours, in order
            to take advantage of the similarity in pair-wise plotting code here
        bins -- override the default number of bins for histograms and
            firing modulation plots if show_area is specified
        """
        if show_area:
            assert show_area in spike.PrimaryAreas, 'bad area: %s'%show_area

        self.figure = {}
        plt.ioff()

        savedir = os.path.join(self.datadir, 'data')
        nrows = ncols = len(MOMENTS) - 1
        mod_plotted = False

        colors = dict(INT='b', EXT='r')
        if show_area:
            self.out('Contour colors: %s'%str(colors))

        for scan_type in SCAN_TYPES:
            for scan_phase in SCAN_PHASES:

                if show_area:
                    if no_scan_data:
                        fname = "_".join([show_area, 'modulation'])
                    else:
                        fname = "_".join([show_area, scan_phase])
                    if fname not in self.figure:
                        self.out('Creating %s figure...'%fname)
                        self.figure[fname] = f = plt.figure(figsize=(9,8))
                        mod_plotted = False
                    else:
                        f = self.figure[fname]
                        plt.figure(f.number)
                        mod_plotted = True
                else:
                    fname = scan_type.lower() + "_" + scan_phase
                    self.out('Creating %s figure...'%fname)
                    self.figure[fname] = f = plt.figure(figsize=(9,8))

                if show_area:
                    if no_scan_data:
                        figtitle = "%s: Firing Modulation"%show_area
                    else:
                        figtitle = "%s: %s Phase"%(show_area, scan_phase.title())
                else:
                    figtitle = '%s. Scans, %s Phase'%(scan_type.title(),
                        scan_phase.title())
                if not (show_area and mod_plotted):
                    f.suptitle(figtitle)

                self.out('Generating pause mask...')
                fwd_speed = np.load(os.path.join(savedir,
                    '%s.npy'%self.munge_data_key(scan_type, scan_phase,
                        'fwd_speed'))) / 50
                hd_speed = np.load(os.path.join(savedir,
                    '%s.npy'%self.munge_data_key(scan_type, scan_phase,
                        'hd_speed'))) / 200
                rad_speed = np.load(os.path.join(savedir,
                    '%s.npy'%self.munge_data_key(scan_type, scan_phase,
                        'rad_speed'))) / 10
                speed_mask = \
                    np.sqrt(fwd_speed**2 + hd_speed**2 + rad_speed**2) > 1

                self.out('Computing occupancy...')
                for row, ordinate in enumerate(MOMENTS[1:]):
                    for col, abscissa in enumerate(MOMENTS[:row+1]):
                        ax = plt.subplot(nrows, ncols, ncols*row + col + 1)

                        if show_area and not mod_plotted:
                            gain = GainFunction(show_area, abscissa, ordinate)
                            cmap = no_scan_data and 'jet' or 'bone'
                            gain.show(ax=ax, cmap=cmap, cmax=1.25, bins=bins)

                        abs_data = np.load(os.path.join(savedir,
                            '%s.npy'%self.munge_data_key(scan_type, scan_phase,
                                abscissa)))[speed_mask]
                        ord_data = np.load(os.path.join(savedir,
                            '%s.npy'%self.munge_data_key(scan_type, scan_phase,
                                ordinate)))[speed_mask]

                        bounds = \
                            [CfgMod[abscissa+'_min'], CfgMod[abscissa+'_max'],
                             CfgMod[ordinate+'_min'], CfgMod[ordinate+'_max']]
                        hist_bins = get_bin_edges(bounds, bins)

                        H, xe, ye = np.histogram2d(abs_data, ord_data,
                            bins=hist_bins)
                        H = H.T.astype('d')
                        H /= H.sum()

                        if not no_scan_data:
                            if show_area:
                                X = (xe[:-1]+xe[1:])/2
                                Y = (ye[:-1]+ye[1:])/2
                                levels = np.linspace(0.0002, H.max(), 6)
                                ch = plt.contour(X, Y, H, levels,
                                    colors=colors[scan_type],
                                    extend='neither')
                            else:
                                mask = H == 0
                                xwrap = behavior.Moment.Wrapped[abscissa]
                                ywrap = behavior.Moment.Wrapped[ordinate]
                                display_modulation(ax, H, mask, bounds,
                                    xwrap=xwrap, ywrap=ywrap)

                        if not (show_area and mod_plotted):
                            ax.set_xticks(ax.get_xticks()[::2])
                            ax.set_xlim(bounds[:2])
                            if behavior.Moment.Wrapped[abscissa]:
                                ax.set_xlim(right=bounds[1] -
                                    np.diff(hist_bins[0][:2]))
                            ax.set_yticks(ax.get_yticks()[::2])
                            ax.set_ylim(bounds[2:])
                            if behavior.Moment.Wrapped[abscissa]:
                                ax.set_ylim(top=bounds[3] -
                                    np.diff(hist_bins[1][:2]))

                        if col == 0:
                            ax.set_ylabel(behavior.Moment.Names[ordinate])
                        else:
                            ax.set_yticklabels([])
                        if row == nrows - 1:
                            ax.set_xlabel(behavior.Moment.Names[abscissa])
                        else:
                            ax.set_xticklabels([])

                if no_scan_data:
                    break
            if no_scan_data:
                break

        plt.ion()
        plt.show()


def generate_all_possible_figures_please(occ_analysis):
    """Create occupancy contour figures with backgrounds corresponding to the
    firing modulation plots of each major recording area
    """
    for area in ['CA3', 'CA1', None]: #spike.PrimaryAreas + [None]:
        if area:
            no_scan_data = [False, True]
        else:
            no_scan_data = [False]
        for no_scan in no_scan_data:
            occ_analysis.process_data(show_area=area, no_scan_data=no_scan)
            occ_analysis.save_plots_and_close()
