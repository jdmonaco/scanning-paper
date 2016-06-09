# encoding: utf-8
"""
behavior_reports.py -- Scan tracking data for candidate head scans and print out
    complete pdf report

Created by Joe Monaco on 2011-04-08.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
from tables import NodeError
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import numpy as np
import os
from scipy.signal import medfilt2d

# Package imports
from .core.analysis import AbstractAnalysis
from .reports import BaseSessionReport, BaseLapReport
from scanr.lib import (Config, get_kdata_file, close_file, get_group, get_node,
    get_group_path, get_tracking, new_array, SessionData)
from scanr.tracking import plot_track_underlay
from scanr.time import time_slice, time_slice_sample
from scanr.behavior import ScanPhasePoints
from scanr.session import SessionData
from .core.report import BaseReport
from .tools.misc import DataSpreadsheet
from .tools.radians import xy_to_rad_vec
from .tools.stats import smooth_pdf, IQR
from .tools.colormaps import diffmap
from .tools.plot import quicktitle

# Constants
CfgData = Config['h5']
CfgScan = Config['scanning']
TRAJ_LW = 0.25
SCAN_LW = 1.1


class HighlightDatasetScans(AbstractAnalysis):

    """
    Quick plot of scans on a time-line for all sessions in a dataset
    """

    label = 'highlight scans'

    def collect_data(self, rat, day, cell_spikes=None, spacing=36,
        highlight_radius=16, event_session=1, no_highlights=False):
        """Create the highlighted scans plot
        """
        from matplotlib.patches import Rectangle
        self.out("Plotting scans from rat%03d-%02d..." % (rat, day))

        plt.ioff()
        self.figure = {}
        self.figure['scans-%d-%d' % (rat, day)] = f = plt.figure(figsize=(19,8))
        f.suptitle('Highlighting Scans for Rat %d, Day %d' % (rat, day))

        dataset = SessionData.get_session_list(rat, day)
        N_sessions = len(dataset)
        ax = f.add_subplot(111)

        traj_fmt = dict(c='k', alpha=0.8, lw=1.2, zorder=0)
        scan_fmt = dict(ec='none', fc='g', alpha=0.5, fill=True, zorder=-1)
        spike_fmt = dict(linewidths=0, c='r', marker='o', s=20, facecolor='r',
            edgecolor='none', alpha=0.7)

        r0 = 0.0
        tmax = 0
        yticks = []
        yticklabels = []
        start_times = None

        for i, session in enumerate(dataset):
            traj = session.trajectory
            t = session.T_(traj.ts)
            ax.plot(t, traj.radius + r0, **traj_fmt)

            if cell_spikes is None:
                cluster = None
            else:
                cluster = session.cluster_data(cell_spikes)
                t_spikes = session.T_(cluster.spikes)
                r_spikes = session.F_('radius')(t_spikes)
                ax.scatter(t_spikes, r_spikes + r0, **spike_fmt)

            yticks.append(r0)
            yticklabels.append(session.session)

            if t.max() > tmax:
                tmax = t.max()

            scans = session.T_(session.scan_list)
            for start, end in scans:
                if no_highlights:
                    continue
                scan_rect = Rectangle(
                    (start, r0 - highlight_radius),
                    end - start, 2 * highlight_radius,
                    clip_box=ax.bbox, **scan_fmt)
                ax.add_artist(scan_rect)

            if start_times is None:
                start_times = scans[:,0]
            elif session.session != event_session:
                start_times = np.r_[start_times, scans[:,0]]

            r0 -= spacing

        ax.set_xlim(0, tmax)
        ax.set_ylim(-N_sessions * spacing, spacing)
        ax.tick_params(top=False, right=False, direction='out')
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('Session time (s)')
        ax.set_ylabel('Sessions')

        self.figure['timing-%d-%d' % (rat, day)] = f = plt.figure(figsize=(19,8))
        f.suptitle('Non-Event Session Scan Timing Distribution for Rat %d, Day %d' % (rat, day))

        ax = f.add_subplot(111)
        ax.plot(*smooth_pdf(start_times), c='k', lw=2)
        ax.set_xlim(0, tmax)
        ax.set_xlabel('Session time (s)')
        ax.set_ylabel('Scan timing probability')

        plt.ion()
        plt.show()

        self.results['there_are_no_results'] = True
        self.out("All done!")


class HeadScanReport(BaseSessionReport):

    """
    Print a complete report of head scan events
    """

    label = 'head scans'

    def collect_data(self, datasets=None, phasing=False):
        """Get head scan table and print out reports

        Keyword arguments:
        datasets -- restrict report to particular datasets specified as a list
            of (rat, day) tuples
        """
        kfile = get_kdata_file()
        bpath = get_group_path(tree='behavior')

        ts_file = file(os.path.join(self.datadir, 'head_scan_bounds.csv'), 'w')
        ts_file.write('rat,day,maze,scan,t0,t1\n')

        table_name = CfgData['scan_table']
        try:
            scan_table = kfile.getNode(bpath, table_name)
        except NodeError, e:
            raise e, 'could not find valid scan table'

        self.out('Reporting scans found in %s...'%scan_table._v_pathname)

        fmt = dict(lw=SCAN_LW, solid_capstyle='round', alpha=0.7)
        for rds, ax in self.get_plot(datasets=datasets, missing_HD=True, timing_issue=True):
            ts, x, y, hd = get_tracking(*rds)
            ax.plot(x, y, '-', c='k', alpha=0.85, lw=0.25) #c='0.4', lw=TRAJ_LW)
            plot_track_underlay(ax, ec='0.8', lw=0.4) #lw=0.5, ls='solid')

            for scan in scan_table.where('(rat==%d)&(day==%d)&(session==%d)'%rds):
                if phasing:
                    outslice = slice(*scan['outbound'])
                    inslice = slice(*scan['inbound'])
                    if scan['type'] == CfgScan['out_type']:
                        colors = 'r', 'm'
                    elif scan['type'] == CfgScan['in_type']:
                        colors = 'b', 'c'
                    else:
                        colors = 'g', 'g'
                    ax.plot(x[outslice], y[outslice], c=colors[0], zorder=10, **fmt)
                    ax.plot(x[inslice], y[inslice], c=colors[1], zorder=5, **fmt)
                else:
                    tslice = slice(*scan['slice'])
                    ax.plot(x[tslice], y[tslice], 'r-', zorder=5, **fmt)

                # Write out the timestamp bounds of the scan
                ts_file.write('%d,%d,%d,%d,%d,%d\n'%(rds+(scan['number'],
                    scan['start'], scan['end'])))

            ax.axis([-55, 55, -53,53])
            # ax.axis('equal')
            ax.axis('off')

        ts_file.close()

    # def _get_title(self, rds):
    #     return 'rat%d-%02d-m%d'%rds


class HeadScanLapReport(BaseLapReport):

    """
    Print a complete by-lap report of head scan events
    """

    label = 'head scan laps'

    def collect_data(self, datasets=None, phasing=True):
        """Get head scan table and print out reports

        Keyword arguments:
        datasets -- restrict report to particular datasets specified as a list
            of (rat, day) tuples
        """
        kfile = get_kdata_file()
        bpath = get_group_path(tree='behavior')

        table_name = CfgData['scan_table']
        try:
            scan_table = kfile.getNode(bpath, table_name)
        except NodeError, e:
            raise e, 'could not find valid scan table'

        self.out('Reporting scans found in %s...'%scan_table._v_pathname)

        colors = dict(prefix=('g',0), outbound=('r',2), dwell=('y',0), inbound=('m',1), postfix=('g',0))
        phase_names = colors.keys()

        fmt = dict(lw=SCAN_LW, solid_capstyle='round', alpha=0.7)
        for rds, lap, ax in self.get_plot(datasets=datasets, missing_HD=True):
            ts, x, y, hd = get_tracking(*rds)
            lapslice = time_slice(ts, start=lap[0], end=lap[1])
            ax.plot(x[lapslice], y[lapslice], '-', c='0.4', lw=TRAJ_LW, zorder=-1)
            plot_track_underlay(ax, lw=0.5, ls='dotted')

            for scan in scan_table.where('(rat==%d)&(day==%d)&(session==%d)'%rds):
                if not (lap[0] <= scan['start'] <= lap[1]):
                    continue

                # Plot the scan on top of trajectory
                if phasing:
                    for phase in phase_names:
                        start, end = ScanPhasePoints[phase]
                        tslice = time_slice(ts, start=scan[start], end=scan[end])
                        col, zord = colors[phase]
                        ax.plot(x[tslice], y[tslice], c=col, zorder=zord, **fmt)
                else:
                    tslice = slice(*scan['slice'])
                    ax.plot(x[tslice], y[tslice], 'r-', zorder=5, **fmt)

            ax.axis([-50, 50, -48, 48])
            # ax.axis('equal')
            ax.axis('off')


class PauseReport(BaseSessionReport):

    """
    Print a complete report of pause events
    """

    label = 'pauses'

    def collect_data(self, datasets=None):
        """Get pause table and print out full report

        Keyword arguments:
        datasets -- restrict report to particular datasets specified as a list
            of (rat, day) tuples
        """
        # Set up spreadsheet with time bounds
        cols = ['rat', 'day', 'maze', 'number','t0','t1']
        coltypes = ['d']*len(cols)
        spreadsheet = DataSpreadsheet(
            os.path.join(self.datadir, 'pause_bounds.csv'),
            zip(cols, coltypes))
        record = spreadsheet.get_record()

        # Get the pause table
        pause_table = get_node('/behavior', 'pauses')
        self.out('Reporting pauses found in %s...'%pause_table._v_pathname)

        fmt = dict(lw=SCAN_LW, solid_capstyle='round', alpha=0.7)
        for rds, ax in self.get_plot(datasets=datasets, missing_HD=True, timing_issue=True):
            record['rat'], record['day'], record['maze'] = rds
            ts, x, y, hd = get_tracking(*rds)
            ax.plot(x, y, '-', c='0.4', lw=TRAJ_LW)
            plot_track_underlay(ax, lw=0.5, ls='dotted')

            for pause in pause_table.where('(rat==%d)&(day==%d)&(session==%d)'%rds):
                tslice = slice(*pause['slice'])
                ax.plot(x[tslice], y[tslice], 'r-', zorder=5, **fmt)

                # Write out the timestamp bounds of the pause
                spreadsheet.write_record(record)

            ax.axis([-50, 50,-48,48])
            # ax.axis('equal')
            ax.axis('off')

        spreadsheet.close()


class PauseLapReport(BaseLapReport):

    """
    Print a complete by-lap report of pause events
    """

    label = 'pause laps'

    def collect_data(self, datasets=None):
        """Get head scan table and print out reports

        Keyword arguments:
        datasets -- restrict report to particular datasets specified as a list
            of (rat, day) tuples
        """
        pause_table = get_node('/behavior', 'pauses')
        self.out('Reporting pauses found in %s...'%pause_table._v_pathname)

        fmt = dict(lw=SCAN_LW, solid_capstyle='round', alpha=0.7)
        for rds, lap, ax in self.get_plot(datasets=datasets, timing_issue=True, missing_HD=True):
            ts, x, y, hd = get_tracking(*rds)
            lapslice = time_slice(ts, start=lap[0], end=lap[1])
            ax.plot(x[lapslice], y[lapslice], '-', c='0.4', lw=TRAJ_LW)
            plot_track_underlay(ax, lw=0.5, ls='dotted')

            for pause in pause_table.where('(rat==%d)&(day==%d)&(session==%d)'%rds):
                if not (lap[0] <= pause['start'] <= lap[1]):
                    continue

                tslice = slice(*pause['slice'])
                ax.plot(x[tslice], y[tslice], 'r-', zorder=5, **fmt)

            ax.axis([-50, 50, -48, 48])
            # ax.axis('equal')
            ax.axis('off')


class TrajectorySpeedReport(BaseSessionReport):

    label = "tracking speed"

    def collect_data(self, speed_lim=(0,75), fwd_speed_lim=(-75, 125)):
        """Plot speed vs. fwd velocity for all trajectories
        """
        for rds, ax in self.get_plot(missing_HD=True, timing_issue=True):
            self.out('Plotting speed heat map for rat%03d-%02d-m%d...'%rds)

            TrajectoryData(rds=rds).speed_plot(ax=ax, labels=False,
                range=[speed_lim, fwd_speed_lim])

            ax.set_xticks(ax.get_xticks()[::2])
            if self.firstonpage:
                ax.set_ylabel(u'Fwd. Speed (\u00b0CW/s)')
            else:
                ax.set_yticklabels([])
            if self.lastonpage:
                ax.set_xlabel('Running Speed (cm/s)')
            else:
                ax.set_xticklabels([])


class BaseForwardSpeed(object):

    """
    Base class for producing reports of occupancy heat maps of forward-running
    speed vs. overall tracking speed
    """

    xnorm = False
    ynorm = False

    def collect_data(self, sessions):
        """Display forward-running speed vs. tracking speed distributions for
        the sessions provided as a list of rds triplets.
        """
        for rds, ax in self.get_plot(sessions):
            data = SessionData(rds=rds, load_clusters=False)

            # Trajectory data
            ts = data.trajectory.ts
            omega = data.trajectory.forward_velocity

            # Data initialization
            lower = np.empty((data.N_laps,), 'd')
            upper = np.empty((data.N_laps,), 'd')
            median = np.empty((data.N_laps,), 'd')
            mean = np.empty((data.N_laps,), 'd')

            # Compute the distribution stats
            for lap in xrange(data.N_laps):
                _t, lap_speed = time_slice_sample(ts, omega,
                    start=data.laps[lap], end=data.laps[lap+1])
                lower[lap], upper[lap] = IQR(lap_speed, factor=0)
                median[lap] = np.median(lap_speed)
                mean[lap] = np.mean(lap_speed)

            # Plot the stats
            laps = np.arange(1, data.N_laps+1)
            p = Polygon(
                np.c_[np.r_[laps, laps[::-1]], np.r_[lower, upper[::-1]]],
                alpha=0.5, ec='none', fc='c', zorder=-1)
            ax.add_artist(p)
            ax.plot(laps, median, 'k-', lw=2)
            ax.plot(laps, mean, 'k--', lw=1)
            ax.axhline(0, ls='-', c='k')
            ax.set_xticks(laps)
            ax.set_xlim(1, data.N_laps)
            ax.set_ylim(-50, 100)
            ax.set_xticklabels([])
            quicktitle(ax, '%d-%02d-m%d'%rds, size='x-small')

            if self.firstonpage:
                ax.set_ylabel(u'Forward Speed (\u00b0CW/s)')
            else:
                ax.set_yticklabels([])
            if self.lastonpage:
                ax.set_xlabel('Laps')

        self.out('All done!')


class ForwardSpeedReport(BaseForwardSpeed, BaseReport):
    label = 'speed report'

class SessionForwardSpeedReport(BaseForwardSpeed, BaseSessionReport):
    label = 'speed report'

