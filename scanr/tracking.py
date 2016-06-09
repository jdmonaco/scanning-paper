# encoding: utf-8
"""
scanr.tracking -- Utility functions for handling position tracking data

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division

# Library imports
import numpy as np
from numpy import pi
import sys
import copy
import warnings
import tables as tb
from os import path
from scipy.interpolate import interp1d
from scipy.signal import firwin, medfilt
from traits.api import (HasTraits, Int, Float, Array, Dict, Tuple,
    Bool, String, Trait)

# Package imports
from .config import Config
from .paths import get_data_file_path
from .data import (get_node, get_group, get_unique_row, unique_cells,
    get_kdata_file, flush_file, close_file)
from .meta import (get_start_end, walk_mazes, get_maze_list, walk_days,
    get_session_metadata)
from .time import (find_sample_rate, time_slice, time_slice_sample,
    stamp_to_time, elapsed)
from .spike import get_session_clusters
from .tools.plot import heatmap
from .tools.stats import pvalue, integer_hist, KT_estimate
from .tools.misc import contiguous_groups, memoize
from .tools.filters import quick_boxcar, circular_blur, filtfilt
from .tools.radians import (radian, xy_to_deg_vec, xy_to_rad_vec,
    get_angle_array, get_angle_histogram, circle_diff, circle_diff_vec_deg,
    unwrap_deg)

# Constants
DEBUG = Config['debug_mode']
INVALID = Config['track']['invalid_sample']
BOXCAR_KERNEL = Config['track']['boxcar_kernel']
INNER_DIAMETER = Config['track']['inner_diameter']
OUTER_DIAMETER = Config['track']['outer_diameter']
TRACK_RADIUS = (INNER_DIAMETER + OUTER_DIAMETER) / 4
TRACK_WIDTH = (OUTER_DIAMETER - INNER_DIAMETER) / 2

# Spatial information data table
CellInfoDescr = {       'id'                : tb.UInt64Col(pos=1),
                        'rat'               : tb.UInt16Col(pos=2),
                        'day'               : tb.UInt8Col(pos=3),
                        'session'           : tb.UInt8Col(pos=4),
                        'tc'                : tb.StringCol(itemsize=8, pos=5),
                        'area'              : tb.StringCol(itemsize=16, pos=6),
                        'quality'           : tb.StringCol(itemsize=16, pos=7),
                        'spike_width'       : tb.FloatCol(pos=8),
                        'N_running'         : tb.UInt16Col(pos=9),
                        'I'                 : tb.FloatCol(pos=10),
                        'p_value'           : tb.FloatCol(pos=11)   }


class TrajectoryData(HasTraits):

    """
    Smart container for the trajectory tracking data of a recording session
    """

    rds = Trait(Tuple((int,int,int)), desc='Required constructor argument')

    # Loaded at initialization based on session specifier
    ts = Array(desc='timestamps')
    x = Array(desc='x position (cm)')
    y = Array(desc='y postiion (cm)')
    hd = Array(desc='head direction (deg)')

    # Traits automatically computed as needed
    fs = Float(desc='sample rate (Hz)')
    N = Int(desc='number of tracking samples')
    alpha = Array(desc='track-angle position (deg)')
    alpha_unwrapped = Array(desc='unwrapped track angle (deg)')
    omega = Array(desc='track-angle speed (degs/s)')
    forward_velocity = Array(desc='forward-running speed (degs/s)')
    speed = Array(desc='tracking path-length speed (cm/s)')
    radius = Array(desc='track-center-relative radius (cm)')
    radial_velocity = Array(desc='radial velocity (cm/s)')
    forward_hd = Array(desc='forward-relative head direction (degs)')
    hd_velocity = Array(desc='head-direction velocity (degs/s)')
    moments = Dict(desc='behavior moments dictionary')

    def __init__(self, rds=None, **traits):
        HasTraits.__init__(self, **traits)
        if rds and len(rds) == 3:
            self.rds = rds
            self.ts, self.x, self.y, self.hd = get_tracking(*rds)
            self.t = stamp_to_time(self.ts,
                zero_stamp=get_session_metadata(*rds)['start'])
        else:
            raise ValueError, 'bad session specifier: %s'%str(rds)

    def __str__(self):
        return 'TrajectoryData(rds=%s)'%str(self.rds)
    def __repr__(self):
        return str(self)

    def _fs_default(self):
        return tracking_sample_rate(self.ts)

    def _N_default(self):
        return self.ts.size

    def _alpha_default(self):
        return xy_to_deg_vec(self.x, self.y)

    def _alpha_unwrapped_default(self):
        return unwrap_deg(self.alpha)

    def _omega_default(self):
        return np.r_[0.0, np.diff(self.alpha_unwrapped) * self.fs]

    def _forward_velocity_default(self):
        return -1*self.omega

    def _speed_default(self):
        return np.r_[0.0,
            np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2) * self.fs]

    def _radius_default(self):
        return np.sqrt(self.x**2 + self.y**2) - TRACK_RADIUS

    def _radial_velocity_default(self):
        return np.r_[0.0, np.diff(self.radius) * self.fs]

    def _forward_hd_default(self):
        """Baseline-adjusted forward-relative head direction
        """
        return circle_diff_vec_deg(
            self.hd - Config['track']['baseline_hd'], self.alpha - 90)

    def _hd_velocity_default(self):
         return np.r_[0.0, circle_diff_vec_deg(self.forward_hd[1:],
            self.forward_hd[:-1]) * self.fs]

    def _moments_default(self):
        from .behavior import Moment
        return Moment.get(self)

    # Convenience or output methods

    def tsxyhd(self):
        return self.ts, self.x, self.y, self.hd

    def plot_speed_scatter(self, ax=None, labels=True, **kwds):
        import matplotlib.pyplot as plt
        t = stamp_to_time(self.ts)
        if 'bins' not in kwds:
            kwds.update(bins=48)
        heatmap(self.speed, self.forward_velocity, ax=ax, **kwds)
        ax = plt.gca()
        ax.axhline(Config['pauses']['max_fwd_speed'], ls='--', c='k')
        ax.axvline(Config['pauses']['max_speed'], ls='--', c='k')
        ax.axhline(0, ls='-', c='k')
        if labels:
            ax.set(xlabel='Speed (cm/s)', ylabel=u'Fwd. Speed (\u00b0CW/s)')
        plt.draw()
        return ax

    def plot_laps(self, **kwds):
        """Plot trajectory laps for the specified recording session

        Keywords are passed to the line plot command.
        """
        from .tools.images import tiling_dims
        from .tools.plot import AxesList
        import matplotlib.pyplot as plt
        rds = rat, day, session
        ts, x, y, hd = self.tsxyhd()
        laps = find_laps(ts, x, y) + [get_start_end(*rds)[1]]
        N_laps = len(laps)-1
        plt.ioff()
        f = plt.figure(figsize=(9,9))
        axlist = AxesList()
        axlist.make_grid(*tiling_dims(N_laps))
        f.suptitle('Rat %d, Day %d, Maze %d'%rds)
        for i in xrange(N_laps):
            lap = time_slice(ts, start=laps[i], end=laps[i+1])
            axlist[i].plot(x[lap], y[lap], **kwds)
            axlist[i].set_title('lap %d'%(i+1))
        axlist.gallery()
        plt.ion()
        plt.show()


class SpatialInformationScore(HasTraits):

    """
    Compute the Skaggs spatial information rate and empirical p-value
    """

    # P-value sampling parameters (set random to False to use fixed spacing,
    # or True to use N random shuffles)
    random = Bool(False)
    random_shuffles = Int(1000)
    fixed_spacing = Float(0.5)
    min_offset = Float(15.0)
    measure = Trait('skaggs', 'olypher', 'shannon')
    reverse_for_shuffle = Bool(True)

    # Skaggs test parameters
    bins = Int(360)
    duration = Dict

    # Olypher/Shannon test parameters
    dt = Float(0.1)
    x_bins = Int(48)

    # Filter tracking data flags

    @memoize
    def _filter_keywords(self, session):
        return dict(
            boolean_index=True,
            velocity_filter=True,
            speed_filter=False,
            exclude=session.scan_and_pause_list,
            exclude_off_track=True)

    # Skaggs support methods

    @memoize
    def get_occupancy(self, session):
        traj = session.trajectory
        running = session.filter_tracking_data(traj.ts, traj.x, traj.y,
            **self._filter_keywords(session))
        alpha = traj.alpha[running]
        H = np.histogram(alpha, range=(0,360), bins=self.bins)[0]
        self.duration[hash(session)] = session.duration * (
            running.sum() / float(running.size)) # running fraction
        return H.astype('d') / traj.fs # -> seconds

    def get_duration(self, session):
        sess_id = hash(session)
        if sess_id not in self.duration:
            occ = self.get_occupancy(session)
        return self.duration[sess_id]

    # Olypher support methods

    @memoize
    def get_time_samples_and_filter(self, session):
        dts = long(self.dt * Config['sample_rate']['time'])
        ts = np.arange(session.start, session.end, dts)
        t = session.T_(ts)
        running = session.filter_tracking_data(ts,
            session.F_('x')(t), session.F_('y')(t),
            **self._filter_keywords(session))

        edge_filter = []
        i = 0
        while i < ts.size - 1:
            edge_filter.append(running[i] and running[i+1])
            i += 1
        return ts, np.nonzero(edge_filter)

    @memoize
    def get_alpha(self, session):
        F = session.F_('alpha_unwrapped')
        T = session.T_
        t, running = self.get_time_samples_and_filter(session)
        bins = ((t[1:] + t[:-1]) / 2).astype(long)
        alpha = F(T(bins)) % 360
        return alpha[running]

    def get_spike_counts(self, session, cluster):
        t, running = self.get_time_samples_and_filter(session)
        return np.histogram(cluster.spikes, bins=t)[0][running]

    @memoize
    def P_x(self, session):
        return KT_estimate(np.histogram(self.get_alpha(session),
            bins=np.linspace(0, 360, self.x_bins+1))[0],
            zero_trim=False)

    def H_xk(self, session, cluster):
        X = self.get_alpha(session)
        K = self.get_spike_counts(session, cluster)
        return np.histogram2d(X, K,
            bins=[np.linspace(0, 360, self.x_bins+1), np.arange(max(K)+2)])[0]

    # Spatial information methods

    def compute(self, session, cluster):
        """Compute the spatial information of a given spike-cluster and session
        based on the current parameters

        Arguments:
        session -- SessionData object representing the recording session
        cluster -- ClusterData object representing the cell activity

        This returns the information rate and sets the I_spike/I_pos/I_mutual
        attribute on the cluster object. Additionally, the N_running cluster
        attribute is set based on the tracking-filtered spikes.
        """
        spike_ix = session.filter_tracking_data(
            cluster.spikes, cluster.x, cluster.y,
            **self._filter_keywords(session))
        cluster.N_running = N = spike_ix.sum()

        if N == 0:
            I = 0.0

        elif self.measure == 'skaggs':
            occ = self.get_occupancy(session)
            duration = self.get_duration(session)
            p = occ / occ.sum()
            x_spikes, y_spikes = cluster.x[spike_ix], cluster.y[spike_ix]
            with warnings.catch_warnings(): # ignoring div-by-zero runtime warnings
                warnings.simplefilter("ignore")
                f = get_angle_histogram(x_spikes, y_spikes, self.bins) / occ
                F = N / duration
                I = p * f * np.log2(f/F)
                I[np.isnan(I)] = 0.0 # handle zero-spike bins
                I = I.sum()/F
            cluster.I_spike = I

        elif self.measure == 'olypher':
            H_xk = self.H_xk(session, cluster)
            P_k = KT_estimate(H_xk.sum(axis=0))
            P_k_x = (H_xk.astype('d') + 0.5) / ( # K-T estimate
                H_xk.sum(axis=1)[:,np.newaxis] + 0.5*H_xk.shape[1])
            I_pos = np.sum(P_k_x * np.log2(P_k_x / P_k), axis=1)
            I = circular_blur(I_pos, 360./self.x_bins).max()
            cluster.I_pos = I

        elif self.measure == 'shannon':
            P_x = self.P_x(session)
            P_k = KT_estimate(H_xk.sum(axis=0))
            H_xk = self.H_xk(session, cluster)
            P_xk = (H_xk + 0.5) / (H_xk.sum() + 0.5*H_xk.size) # 2D K-T estimate
            E_xk = np.dot(P_x[:,np.newaxis], P_k[np.newaxis])
            I = np.sum(P_xk * np.log2(P_xk / E_xk))
            cluster.I_mutual = I

        return I

    def pval(self, session, cluster):
        """Perform spike time shuffling to compute expected spatial information
        scores and corresponding p-value.

        Keywords are passed to the compute method.

        This returns the information p-value and sets the spatial_p_value
        attribute on the cluster object.
        """
        I_obs = self.compute(session, cluster)

        if self.reverse_for_shuffle:
            spikes = (session.start + session.end - cluster.spikes)[::-1]
        else:
            spikes = cluster.spikes.copy()

        margin = long(self.min_offset * Config['sample_rate']['time'])
        delta = session.end - session.start
        if self.random:
            cuts = np.random.randint(margin, delta - margin,
                self.random_shuffles)
        else:
            dt = long(self.fixed_spacing * Config['sample_rate']['time'])
            cuts = np.arange(margin, delta - margin, dt)

        shuffled = copy.copy(cluster)
        I = np.empty_like(cuts, dtype='d')
        for i, cut in enumerate(cuts):
            new_spikes = spikes - cut
            new_spikes[new_spikes<session.start] += delta
            shuffled.spikes = new_spikes
            t_spikes = session.T_(shuffled.spikes)
            shuffled.x = session.F_('x')(t_spikes)
            shuffled.y = session.F_('y')(t_spikes)
            I[i] = self.compute(session, shuffled)

        p = pvalue(I_obs, I)
        if self.measure == 'skaggs':
            cluster.spike_p_value = p
        elif self.measure == 'olypher':
            cluster.pos_p_value = p
        elif self.measure == 'shannon':
            cluster.mutual_p_value = p

        return p

def merge_spatial_information_scores(temporal_table="cell_information_olypher",
    spatial_table="cell_information_skaggs"):
    """Merge the spatial information scores and p-values from cell spatial
    information tables (a temporal reliability measure like Olypher, plus a
    purely spatial measure like Skaggs) under /physiology into the cluster
    nodes of the main data tree. The temporal measure is used as the first-pass
    place field criterion, with the spatial measure as the second-pass fallback.
    The temporal measure results are stored on cell node attributes as 'I_pos'
    and 'pos_p_value'; spatial measures results are stored as 'I_spike' and
    'spike_p_value'.
    """
    ttable = get_node('/physiology', temporal_table)
    stable = get_node('/physiology', spatial_table)

    for rds in walk_mazes():
        grp = get_group(rds=rds)
        rds_query = '(rat==%d)&(day==%d)&(session==%d)'%rds

        for tc in get_session_clusters(grp):
            cell_name = 't%dc%d'%tc
            attrs = getattr(getattr(grp, cell_name), '_v_attrs')

            # Query tables for cell records
            cell_query = rds_query + '&(tc=="%s")'%cell_name
            pos_rec = get_unique_row(ttable, cell_query)
            spike_rec = get_unique_row(stable, cell_query)

            # Update cell node attributes with measure-specific names
            attrs['I_pos'] = pos_rec['I']
            attrs['pos_p_value'] = pos_rec['p_value']
            attrs['I_spike'] = spike_rec['I']
            attrs['spike_p_value'] = spike_rec['p_value']
            attrs['N_running'] = spike_rec['N_running']

            sys.stdout.write('Merged cell %s from %s.\n'%(cell_name,
                grp._v_pathname))
        grp._f_flush()
    flush_file()

def smooth(s, fs=30.0):
    """Median-filter denoising of a trajectory signal, where the kernel size
    adapts to the sampling frequency
    """
    return medfilt(s, kernel_size=max(3,8*int(fs/30.0)-1))

def plot_track_underlay(ax=None, **kwds):
    """Draw schematic representation of the circle track on an axis plot

    Keyword arguments are passed to the patches.Circle constructor.
    """
    import matplotlib as mpl
    if ax is None:
        ax = mpl.pyplot.gca()
    fmt = dict(clip_box=ax.bbox, fc='none', ec='b', lw=2, aa=True, zorder=-100)
    fmt.update(**kwds)
    inner_fmt = fmt.copy()
    # inner_fmt.update(fc='w', zorder=(fmt['zorder']+1)) # force white interior
    inner_fmt.update(fc='none')
    ax.add_artist(mpl.patches.Circle((0, 0), radius=INNER_DIAMETER/2, **inner_fmt))
    ax.add_artist(mpl.patches.Circle((0, 0), radius=OUTER_DIAMETER/2, **fmt))
    mpl.pyplot.draw()

def find_laps(ts, x, y, start=None, alpha_cut=None):
    """Compute circle-track laps based on initial track position

    Arguments:
    ts, x, y -- the position tracking data
    start -- timestamp for start of session; default to first tracking stamp
    alpha_cut -- specify track angle in radians at which laps should be cut;
        defaults to initial track position

    Returns a list of timestamps corresponding to the start of detected laps.
    """
    if start is None:
        start = ts[0]
    else:
        start = long(start)

    # Initialize logic-state and loop variables
    alpha = xy_to_rad_vec(x, y)
    if alpha_cut is None:
        alpha0 = alpha[0]
    else:
        alpha0 = alpha_cut
    alpha_halfwave = float(radian(alpha0 - pi))
    alpha_hw_flag = False
    laps_list = [long(start)]
    for i in xrange(1, ts.size):
        # Filter out any 0/2pi jumps
        if alpha[i-1] - alpha[i] > pi:
            continue
        # Clockwise -> less-than radian comparisons
        if not alpha_hw_flag:
            if abs(circle_diff(alpha[i], alpha0)) > pi/2:
                if circle_diff(alpha[i-1], alpha_halfwave) > 0 and \
                    circle_diff(alpha[i], alpha_halfwave) <= 0:
                    alpha_hw_flag = True
        elif abs(circle_diff(alpha[i], alpha_halfwave)) > pi/2:
            if circle_diff(alpha[i-1], alpha0) > 0 and \
                circle_diff(alpha[i], alpha0) <= 0:
                alpha_hw_flag = False
                laps_list.append(long(ts[i]))
    return laps_list

def get_tracking_slice(start=None, end=None, rat=None, day=None, session=None):
    """Time-sliced tracking data

    Keyword arguments:
    start/end -- timestamp bounds of the slice
    rat/day/session -- specify the session

    Returns (t, x, y, hd) position tracking arrays.
    """
    maze = get_group(rat=rat, day=day, session=session)
    if not maze or (None in (rat, day, session)):
        raise ValueError, 'no session found for %s'%str((rat, day, session))
    t_slice = time_slice(maze.t_pos[:], start=start, end=end)
    return maze.t_pos[t_slice], maze.x[t_slice], maze.y[t_slice], \
        maze.hd[t_slice]

def get_tracking(rat, day, session=None):
    """Retrieve tracking data for specified session from H5 file

    If session is not specified, the data for the entire day is returned.

    Returns (ts, x, y, angle) tuple of arrays.
    """
    if session is not None:
        maze = get_group(rat=rat, day=day, session=session)
        if not maze:
            raise ValueError, 'no session found for %s'%str((rat, day, session))
        t_pos, x, y, hd = maze.t_pos[:], maze.x[:], maze.y[:], maze.hd[:]
    else:
        t_pos, x, y, hd = np.array([], 'l'), np.array([], 'f'), np.array([], 'f'), \
            np.array([], 'f')
        for session in get_maze_list(rat, day):
            maze = get_group(rat=rat, day=day, session=session)
            if not maze:
                raise ValueError, 'no session found for %s'%str((rat, day, session))
            t_pos = np.r_[t_pos, maze.t_pos[:]]
            x = np.r_[x, maze.x[:]]
            y = np.r_[y, maze.y[:]]
            hd = np.r_[hd, maze.hd[:]]
    return t_pos, x, y, hd

def load_tracking_data(rat=None, day=None, session=None, status={}):
    """Calibrated tracking data for the specified experimental session

    Optional: pass in *status* namespace to extract fs, N_samples,
    hd_fix_success, hd_invalid, or hd_discontinuous.

    Returns (timestamp, x, y, direction) tuple of arrays.
    """
    if not (rat and day and session):
        raise ValueError, 'need (rat, day, session) to specifiy maze run'

    center_file = get_data_file_path('center', rat=rat, day=day, session=session)
    pos_file = get_data_file_path('tracking', rat=rat, day=day)

    xy0 = read_center_file(center_file)
    ts, x, y, angle = read_position_file(pos_file)

    start, end = get_start_end(rat, day, session)
    ts, samples = time_slice_sample(ts, np.c_[x, y, angle], start=start, end=end)
    x, y, angle = samples.T

    ts, x, y, angle = remove_dupes(ts, x, y, angle)
    fs = tracking_sample_rate(ts, verbose=False)

    status['fs'] = fs
    status['N_samples'] = ts.size

    calibrate_tracking(xy0, x, y)
    fix_head_direction(angle, fs=fs, status=status)
    smooth_tracking(x, y, fs=fs)

    return ts, x, y, angle

def remove_dupes(ts, x, y, hd):
    """Remove all tracking records with duplicate timestamps, treating the
    first occurrence of a given timestamp as the valid sample.

    Valid tracking arrays are returned.
    """
    valid = np.r_[True, (np.diff(ts) != 0)]
    if not valid.all():
        sys.stderr.write('remove_dupes: found %d dupes\n'%
            (np.sum(True - valid)))
        ts = ts[valid]
        x = x[valid]
        y = y[valid]
        hd = hd[valid]
    return ts, x, y, hd

def smooth_tracking(x, y, fs=30.0):
    """In-place boxcar smoothing of tracking data to remove artifacting
    """
    box = int(round(BOXCAR_KERNEL * (fs / 30.0))) # sample frequency adjustment
    if box > 1:
        x[:] = quick_boxcar(x, M=box, centered=True)
        y[:] = quick_boxcar(y, M=box, centered=True)
    return

def tracking_sample_rate(ts, verbose=False):
    """Determine the sampling rate of tracking data based on the corresponding
    timestamp array
    """
    r_bar = r0 = find_sample_rate(ts)
    if 29.5 < r0 < 30.5:
        r_bar = 30.0
    elif 59.0 < r0 < 61.0:
        r_bar = 60.0
        if DEBUG and verbose:
            sys.stdout.write('trajectory: detected 60Hz sampling\n')
    else:
        r_bar = r0
        if DEBUG and verbose:
            sys.stdout.write(
                'trajectory: warning: found %0.2fHz sampling\n'%r_bar)
    return r_bar

def calibrate_tracking(center, x, y, inplace=True):
    """Center and scale raw position data to origin-center and centimeters

    Y-axis inversion is fixed.

    Arguments:
    center -- the center point of the track (from center file)
    x, y -- raw position data for a single recording session

    Returns calibrated (x, y) ONLY IF inplace set to False.
    """
    if len(center) != 2:
        raise ValueError, 'center must specify a 2D point'
    if not inplace:
        x, y = x.copy(), y.copy()

    # Fix glitches in tracking data
    glitchy_samples = np.logical_or(x==0, y==0)
    fix_glitches(x, glitchy_samples, func='tracking_data')
    fix_glitches(y, glitchy_samples, func='tracking_data')

    # Center coordinate frame
    x -= center[0]
    y -= center[1]
    fix_centering(x, y)

    # Compute tracking resolution empirically
    dots_per_cm = np.median(np.sqrt(x**2 + y**2)) / TRACK_RADIUS

    # Scale the tracking data
    x /= dots_per_cm
    y /= -1 * dots_per_cm # fix y-axis inversion

    if not inplace:
        return x, y

def fix_centering(x, y, bins=32):
    """Recalibrate the centering of a circle-track trajectory
    """
    alpha = xy_to_deg_vec(x, y)
    radius = np.sqrt(x**2 + y**2)
    perimeter = np.empty((bins, 2), 'd')
    angles = np.linspace(0.0, 360.0, bins+1)

    for b in xrange(bins):
        ix = np.logical_and(alpha >= angles[b], alpha < angles[b+1])
        r_bar = np.median(radius[ix])
        angle_bar = (pi/180) * (angles[b] + angles[b+1]) / 2
        perimeter[b] = r_bar*np.cos(angle_bar), r_bar*np.sin(angle_bar)

    x_bar, y_bar = perimeter.mean(axis=0)
    x -= x_bar
    y -= y_bar

    if DEBUG:
        sys.stdout.write('fix_centering: dx = (%.4f, %.4f) cm\n'%
            (-x_bar, -y_bar))

    return perimeter

def fix_head_direction(angle, fs=30.0, inplace=True, status={}):
    """Fix 0-angle reference and glitches in the head direction signal

    Note: If head direction not recorded in position file (all -99s), then
    nothing is fixed but "append_tracking.py" should mark this session
    as "missing_HD==True".

    Keyword arguments:
    fs -- tracking sample rate for the position data
    inplace -- fix the array in place, otherwise new copy is returned
    """
    func='fix_head_direction'

    invalid_samples = angle == INVALID
    discontinuities = np.r_[False, np.abs(circle_diff_vec_deg(angle[1:],
        angle[:-1]))>=60/(fs/30.0)]
    glitchy = np.logical_or(invalid_samples, discontinuities)

    status['hd_invalid'] = np.sum(invalid_samples)
    status['hd_discontinuous'] = np.sum(discontinuities)
    status['hd_fix_success'] = False

    if glitchy.all():
        sys.stdout.write('%s: bad values\n'%func)
        return

    # Original 0-angle reference is WEST, make it EAST
    angle -= 180
    angle[angle<0] += 360

    # glitch fixing can fail if many glitches: if it does fail, just assume it will
    # get marked as HD_missing, and don't do anything
    try:
        fix_glitches(angle, glitchy, wrapped=True, func=func)
    except IndexError:
        if DEBUG:
            sys.stderr.write('%s: failed to fix glitches\n'%func)
    else:
        status['hd_fix_success'] = True

def fix_glitches(z, glitchy, wrapped=False, func='fix_glitches'):
    """Use linear interpolation to fix glitches in position sampling arrays

    Arguments:
    z -- array data to be fixed (x, y, or angle data)
    glitchy -- boolean array indicating which samples are glitches

    Array is fixed in place.
    """
    glitches = contiguous_groups(glitchy)
    for g in glitches:
        n = g[1] - g[0]
        if g[0] == 0:
            z0 = z[g[1]] - (n+1)*(z[g[1]+1]-z[g[1]])
            z1 = z[g[1]]
            if DEBUG:
                sys.stderr.write('%s: initial glitch: %d samples\n'%(func, n))
        else:
            z0 = z[g[0]-1]
            if g[1] == z.shape[0]:
                z1 = z0 + (n+1)*(z[g[0]-1]-z[g[0]-2])
                if DEBUG:
                    sys.stderr.write('%s: terminal glitch: %d samples\n'%(func, n))
            else:
                z1 = z[g[1]]
            if wrapped:
                if z1 - z0 >= 180:
                    z1 -= 360
                if z1 - z0 < -180:
                    z1 += 360
        z[slice(*g)] = np.linspace(z0, z1, n+2)[1:-1]
    if wrapped:
        z[z<0] += 360
        z[z>=360] -= 360

    if DEBUG:
        sys.stderr.write('%s: fixed %d tracking glitches\n'%
            (func, len(glitches)))

@memoize
def read_center_file(fn):
    """Load track center coordinates from file

    Returns (x_center, y_center) point array.
    """
    if not path.exists(fn):
        raise ValueError, 'invalid center file specified: %s.'%fn
    return np.loadtxt(fn, delimiter=',')

@memoize
def read_position_file(fn):
    """Load position tracking data from Pos.p (or Pos.p.ascii) file

    Returns (timestamp, x, y, direction) tuple of arrays.
    """
    if fn.endswith('.p'):
        import neuralynx
        ts, x, y, angle = neuralynx.read_position_file(fn)
    elif fn.endswith('.ascii'):
        ts, x, y, angle = np.loadtxt(fn, comments='%', delimiter=',', unpack=True)
    else:
        raise ValueError, 'bad position file name: %s'%fn
    ts = ts.astype(long)
    return ts, x, y, angle
