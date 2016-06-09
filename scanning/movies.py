# encoding: utf-8
"""
movies.py -- Analysis classes for generating dataviz movies

Created by Joe Monaco on March 24, 2014.
Copyright (c) 2014 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division

# Library imports
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Package imports
from scanr.lib import *
from scanr.time import select_from
from .data_reports import TRAJ_FMT, SPIKE_FMT

# Local imports
from .core.analysis import AbstractAnalysis
from .tools.interp import linear_upsample
from .tools.plot import heatmap

# Constants
SCAN_PROJECT = "/Users/joe/Archives/Project-Archives/scan-potentiation-project"
RAT_RUNNING = os.path.join(SCAN_PROJECT, "rat-running.png")
RAT_SCANNING_LEFT = os.path.join(SCAN_PROJECT, "rat-scanning-left.png")
RAT_SCANNING_RIGHT = os.path.join(SCAN_PROJECT, "rat-scanning-right.png")


SPIKE_FMT = dict(edgecolors='k', facecolors='r', s=20, marker='o', alpha=0.7,
        linewidths=0.15, zorder=10)

# Plot the full-session data plots for comparison with the movie
def plot_full_session_references(sess, ax=None, cell=None, dpi=150.0, dest='.'):

    if ax is None:
        ax = plt.gca()

    x = sess.trajectory.x
    y = sess.trajectory.y

    def _reset_axes():
        ax.cla()
        plot_track_underlay(ax, lw=0.5)
        ax.axis([-50,50,-50,50])
        ax.plot(x, y, c='k', alpha=0.75, solid_capstyle='round')

    def _save_fig(name):
        _p = os.path.join(dest, '%s.png' % name)
        ax.set_axis_off()
        plt.savefig(_p, dpi=dpi)

    _reset_axes()
    _save_fig('full-traj')

    if cell:
        cluster = sess.cluster_data(cell)

        run = sess.running_filter()
        running_xy = sess.filter_tracking_data(cluster.spikes, **run)

        scan_spikes = select_from(cluster.spikes, sess.scan_list)
        scanning_xy = (cluster.x[scan_spikes], cluster.y[scan_spikes])

        _reset_axes()
        ax.scatter(cluster.x, cluster.y, **SPIKE_FMT)
        _save_fig('traj-allspikes-%s' % cell)

        _reset_axes()
        ax.scatter(*running_xy, **SPIKE_FMT)
        _save_fig('traj-runspikes-%s' % cell)

        _reset_axes()
        ax.scatter(*scanning_xy, **SPIKE_FMT)
        _save_fig('traj-scanspikes-%s' % cell)


class RealtimeTrajectoryMovie(AbstractAnalysis):

    """
    Generate real-time (or x-fold time) movie of a trajectory plot
    """

    label = 'realtime movie'

    def collect_data(self, rds=(72,1,1), cell_spikes=None, cmap='gray_r', trace=2.0,
        trace_dt=5.0, speedup=1.0, fps=30, w=384, tlim=None, upsample=10,
        inset=None):
        """Render frames of position on a trajectory along with a historical
        trailing line that is colored by time into the past

        inset -- None, 'histo', or 'icon'
        """
        self.results['desc'] = 'traj-rat%03d-%02d-m%d' % rds
        self.results['fps'] = fps = int(fps)
        self.results['frame_dir'] = frame_dir = os.path.join(self.datadir, 'frames')
        os.makedirs(frame_dir)

        if type(cmap) is str:
            cmap = plt.cm.cmap_d[cmap]

        w_inches = 3.0
        figsize = (w_inches, w_inches)

        session_data = SessionData.get(rds=rds)
        traj = session_data.trajectory

        plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1])
        if inset:
            if inset == 'histo':
                ax2 = plt.axes([0.38,0.38,0.24,0.24])
                rmax = 1.01 * np.max(traj.radius)
                smax = 1.01 * np.max(traj.radial_velocity)
                rbins = 32
                hist_bins = [np.linspace(-rmax, rmax, rbins), np.linspace(-smax, smax, rbins)]
            elif inset == 'icon':
                x0 = 0.31
                sz = 1 - 2 * x0
                ax2 = plt.axes([x0,x0+0.03,sz,sz])
                scan_table = get_node('/behavior', 'scans')
                rat_running_image = plt.imread(RAT_RUNNING)
                rat_scanning_images = dict(EXT=plt.imread(RAT_SCANNING_LEFT),
                    INT=plt.imread(RAT_SCANNING_RIGHT))
                scan_direction = []
                for scan in scan_table.where(session_data.session_query):
                    t_start, t_end = session_data.T_((scan['start'], scan['end']))
                    scan_direction.append((t_start, t_end, scan['type']))

            else:
                raise ValueError, 'inset must be None, "histo", or "icon"'

        tlim = (tlim is None) and (0, session_data.duration) or tuple(tlim)
        dur = tlim[1] - tlim[0]
        frame_timing = np.linspace(tlim[0], tlim[1], dur*fps/speedup)

        t = t_rad = session_data.T_(traj.ts)
        x, y, radius, radial_speed = traj.x, traj.y, traj.radius, \
            traj.radial_velocity

        spikes = None
        if cell_spikes:
            cluster = session_data.cluster_data(cell_spikes)
            spikes = session_data.T_(cluster.spikes)

        if upsample:
            t_rad, radius, radial_speed = map(
                lambda z: linear_upsample(z, int(upsample)),
                (t_rad, radius, radial_speed))

        def _frame_path(i):
            return os.path.join(frame_dir, 'frame-%04d.png' % i)

        for i, frame_t in enumerate(frame_timing):
            self.out('frame %04d: t = %f' % (i, frame_t))

            ax.cla()
            plot_track_underlay(ax, lw=0.5)
            ax.axis([-50,50,-50,50])

            ix = (t < frame_t).nonzero()[0]

            j = 0
            while t[j] < frame_t and j < t.size - 1:
                dt = frame_t - t[j]
                if dt <= trace_dt:
                    ax.plot(x[j:j+2], y[j:j+2], c=cmap(np.exp(-dt/trace)), solid_capstyle='round')
                j += 1

            if inset:
                ax2.cla()
                ax2.set_axis_off()
                if inset == 'histo':
                    r_ix = np.logical_and(t_rad >= frame_t - 4.0, t_rad <= frame_t)
                    heatmap(radius[r_ix], radial_speed[r_ix], ax=ax2, bins=hist_bins, normed=True)
                elif inset == 'icon':
                    rat_img = rat_running_image
                    for start, end, direction in scan_direction:
                        if frame_t >= start and frame_t <= end:
                            if direction in ('EXT', 'INT'):
                                rat_img = rat_scanning_images[direction]
                                break
                    ax2.imshow(rat_img, interpolation='nearest')

            if spikes is not None:
                spike_ix = np.logical_and(spikes >= frame_t - trace_dt, spikes <= frame_t)
                if np.any(spike_ix):
                    ax.scatter(cluster.x[spike_ix], cluster.y[spike_ix], **SPIKE_FMT)

            ax.text(-48, -48, '%.1f s' % frame_t, ha='left', va='bottom')

            fp = _frame_path(i)

            ax.set_axis_off()
            plt.savefig(fp, dpi=w/w_inches)

        plot_full_session_references(session_data, ax=ax, cell=cell_spikes,
                dpi=w/w_inches, dest=self.datadir)

        plt.ion()

    def process_data(self):

        fp = self.results['frame_dir']
        fps = self.results['fps']
        desc = self.results['desc']

        output_path = os.path.join(self.datadir, '%s.avi' % desc)

        menc_call = ['mencoder', 'mf://%s/*.png' % fp, '-mf', 'fps=%d:type=png' % fps,
            '-o', output_path, '-ovc', 'lavc', '-lavcopts', 'vcodec=msmpeg4v2:vbitrate=4800']

        self.out('Calling mencoder:\n%s' % ' '.join(menc_call))
        subprocess.call(menc_call)

        self.out('Opening %s...' % output_path)
        subprocess.call(['open', output_path])


