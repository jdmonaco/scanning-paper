# encoding: utf-8
"""
data_reports.py -- Reporting subclasses for various types of session data

Created by Joe Monaco on 2011-09-22.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
import matplotlib.pyplot as plt

# Package imports
from .reports import BaseSessionReport, BaseClusterReport, BaseLapClusterReport
from scanr.spike import plot_correlogram
from scanr.time import time_slice_sample
from scanr.tracking import TrajectoryData, plot_track_underlay
from scanr.cluster import (AND, get_min_quality_criterion, get_tetrode_restriction_criterion,
    PrincipalCellCriteria)
from .tools.radians import xy_to_deg_vec

# Plotting styles
TRAJ_FMT = dict(ls='-', c='k', alpha=0.75, lw=0.5, aa=True, zorder=0)
SPIKE_FMT = dict(s=5, marker='o', edgecolor='r', facecolor='none',
    linewidth=0.5, alpha=0.75, zorder=1)


class ClusterCorrReport(BaseClusterReport):

    label = "corr report"

    def collect_data(self, session, clusters=None, **kwds):
        """Generate per-cluster plots of autocorrelograms

        Keywords arguments are passed to spike.plot_correlogram.
        """
        for tc, data, ax in self.get_plot(session, clusters):
            plot_correlogram(data.spikes, ax=ax, **kwds)
            ax.set_yticks([])
            ax.set_ylim(ymin=0)
            if not self.lastrow:
                ax.set_xticks([])


class LapClusterSpikesReport(BaseLapClusterReport):

    label = "lap spikes"

    def collect_data(self, session, min_quality='fair', alpha_color=False,
        tetrodes=None, plot_track=True):
        """Generate per-lap plots for each cluster of spikes on a trajectory
        """
        if type(session) is tuple and len(session) == 3:
            from scanr.session import SessionData
            session = SessionData(rds=session)

        traj = session.trajectory
        ts, x, y = traj.ts, traj.x, traj.y

        self.out("Quality filter: at least %s"%min_quality)
        criteria = AND(PrincipalCellCriteria,
            get_min_quality_criterion(min_quality))

        if tetrodes is not None:
            criteria = AND(criteria,
                get_tetrode_restriction_criterion(tetrodes))

        for tc, data, lap, ax in self.get_plot(session, criteria):

            start, end = lap
            t, xy = time_slice_sample(ts, np.c_[x, y], start=start, end=end)
            lap_x, lap_y = xy.T

            t, xy = time_slice_sample(data.spikes, np.c_[data.x, data.y],
                start=start, end=end)
            spike_x, spike_y = xy.T

            ax.plot(lap_x, lap_y, **TRAJ_FMT)

            if len(spike_x):
                if alpha_color:
                    alpha = xy_to_deg_vec(spike_x, spike_y)
                    ax.scatter(spike_x, spike_y, c=alpha, vmin=0, vmax=360,
                        **SPIKE_FMT)
                else:
                    ax.scatter(spike_x, spike_y, **SPIKE_FMT)

            if plot_track:
                plot_track_underlay(ax, lw=0.5, ls='dotted')

            ax.axis('equal')
            ax.set_axis_off()
