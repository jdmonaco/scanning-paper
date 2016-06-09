#encoding: utf-8
"""
session.py -- Components for loading and representing recording data from the
    circle-track mismatch experiments.

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import sys
import warnings
from os import path, makedirs
import operator as op
import numpy as np
from numpy import pi
import tables as tb
from scipy.interpolate import interp1d
from traits.api import (HasTraits, Float, Int, String, Array, Long,
    Directory, Trait, Instance, Dict, List, false, true)

# Package imports
from .config import Config
from .meta import get_maze_list
from .cluster import (ClusterData, ClusterCriteria, PrincipalCellCriteria,
    string_to_quality)
from .behavior import Moment
from .spike import get_session_clusters, parse_cell_name
from .data import get_group, get_node
from .paths import get_path
from .time import (elapsed, stamp_to_time, select_from, exclude_from,
    select_from, time_slice_sample)
from .tracking import TrajectoryData, find_laps, TRACK_RADIUS, TRACK_WIDTH
from .tools.bash import CPrint
from .tools.filters import circular_blur, unwrapped_blur
from .tools.radians import get_angle_histogram
from .tools.misc import contiguous_groups, memoize

# Constants
DEBUG = Config['debug_mode']
DEFAULT_MIN_VEL = Config['ratemap']['min_velocity']
DEFAULT_MIN_SPEED = Config['ratemap']['min_speed']
DEFAULT_BINS = Config['ratemap']['bins']
DEFAULT_BLUR = Config['ratemap']['gaussian_blur']


class SessionData(HasTraits):

    """
    Load and compute responses from recording data from a specified session

    Ratemap discretization (default values cf. Lee et al 2004):
    default_bins -- default number of bins to use in computing ratemaps
    default_blur_width -- default width gaussian blur filter

    Cluster filtering traits:
    cluster_criteria -- scanr.cluster.ClusterCriteria object that acts as the
        default cluster filter for calls to get_criterion_clusters

    Convenience methods:
    extract_session_data -- output simplified data files in text format

    Methods to compute spatial responses:
    get_cluster_ratemap -- radial ratemap of a cluster's activity
    get_population_matrix -- matrix of ratemaps for a defined ensemble
    get_population_lap_matrix -- population matrix expanded into lap dimension
    """

    out = Instance(CPrint)

    # Data attributes
    rat = Int
    day = Int
    session = Int
    # rds = Trait(Tuple((int, int, int)))
    clusts = Trait(dict)
    start = Long
    end = Long
    duration = Float
    ripple_list = List
    trajectory = Instance(TrajectoryData)
    load_clusters = true
    quiet = false

    # Automatic lists of scan and pause intervals
    scan_list = List
    scan_extensions_list = List
    extended_scan_list = List
    pause_list = List
    scan_and_pause_list = List
    extended_scan_and_pause_list = List

    # Session data directories and files
    data_path = Directory
    data_group = Instance(tb.Group)
    data_group_path = String
    attrs = Instance(tb.attributeset.AttributeSet)
    session_query = String
    interpolants = Dict

    # Dataset problems
    missing_HD = false
    timing_issue = false

    # Lap traits
    is_circle_track = true
    laps = Array
    N_laps = Int

    # Ratemap properties
    min_velocity = Float(DEFAULT_MIN_VEL, desc='spike velocity filter')
    min_speed = Float(DEFAULT_MIN_SPEED, desc='spike path-speed filter')
    default_bins = Int(DEFAULT_BINS)
    default_blur_width = Float(DEFAULT_BLUR)
    traj_hist = Array

    # Cluster criteria for including cell data
    cluster_criteria = Instance(ClusterCriteria)

    def __init__(self, rds=None, **traits):
        HasTraits.__init__(self, **traits)
        assert type(rds) is tuple and len(rds) == 3, 'bad session specification: %s'%str(rds)
        self.rds = tuple(int(v) for v in rds)
        self.out = CPrint(prefix=self.__class__.__name__, color='purple')
        if type(self.data_group) is tb.Group:
            attrs = self.data_group._v_attrs
            self.rat = attrs['rat']
            self.day = attrs['day']
            self.session = attrs['session']
        else:
            if np.any(self.rds):
                self.rat, self.day, self.session = self.rds
            self.data_group = get_group(rat=self.rat, day=self.day,
                session=self.session)
            if self.data_group is None:
                raise ValueError, 'bad session group specification'

        self._load_data()

        if self.is_circle_track:
            self._compute_laps()

    def __str__(self):
        return str(self.rds)
    def __repr__(self):
        return str(self)

    # Cached session data loading

    @classmethod
    @memoize
    def get(klass, rds, **traits):
        """Class method for auto-cached retrieval of a SessionData object based
        on a required rds-triplet argument. Keyword arguments set trait values
        on the SessionData object.
        """
        return klass(rds=rds, **traits)

    # Data loading and initialization methods

    def _cluster_criteria_default(self):
        return PrincipalCellCriteria

    def _load_data(self):
        """Load data from H5 group containing the session data
        """
        # Store session metadata and data paths
        # self.rds = SessionSpec(self.rat, self.day, self.session)
        self.attrs = self.data_group._v_attrs
        self.start = self.attrs['start']
        self.end = self.attrs['end']
        self.duration = elapsed(self.start, self.end)
        self.data_path = get_path(rat=self.rat, day=self.day)
        self.data_group_path = self.data_group._v_pathname

        # Retrieve the position tracking data
        self.trajectory = TrajectoryData(rds=self.rds)

        # Load cluster data if specified
        if self.load_clusters:
            self._load_cluster_data()

        # Cache track angle histogram for computing ratemaps
        self.traj_hist = get_angle_histogram(
            self.trajectory.x, self.trajectory.y, self.default_bins)

        # Get problems from sessions table
        stable = get_node('/metadata', 'sessions')
        self.missing_HD = \
            [S['missing_HD'] for S in stable.where(self.session_query)][0]
        self.timing_issue = \
            [S['timing_issue'] for S in stable.where(self.session_query)][0]

        if not self.quiet:
            self.out("Loaded session data (%d clusters): %s"%(len(self.clusts),
                self.data_group_path))

    def _load_cluster_data(self):
        """Store spike trains and positions for all recorded clusters
        """
        for tc in get_session_clusters(self.data_group):
            self._load_cluster(tc)

    def _load_cluster(self, tc):
        tt, cl = tc
        ts_spike = self.get_spike_train(tt, cl)
        t_spike = self.T_(ts_spike)
        x_spike = self.F_('x')(t_spike)
        y_spike = self.F_('y')(t_spike)

        width = self.get_cluster_attribute(tc, "maxwidth")
        quality = self.get_cluster_attribute(tc, "quality")
        comment = self.get_cluster_attribute(tc, "comment")
        I_pos = self.get_cluster_attribute(tc, "I_pos")
        p_pos = self.get_cluster_attribute(tc, "pos_p_value")
        I_spike = self.get_cluster_attribute(tc, "I_spike")
        p_spike = self.get_cluster_attribute(tc, "spike_p_value")
        N = self.get_cluster_attribute(tc, "N_running")

        def _none_guard(u):
            if u is None:
                return 0.0
            return float(u)

        def _none_guard_int(u):
            if u is None:
                return 0
            return int(u)

        self.clusts['t%dc%d'%tc] = ClusterData(
            x=x_spike, y=y_spike,
            tt=tt, cl=cl,
            spikes=ts_spike,
            T=self.duration,
            spike_width=float(width),
            quality=string_to_quality(quality),
            comment=str(comment),
            I_pos=_none_guard(I_pos),
            I_spike=_none_guard(I_spike),
            pos_p_value=_none_guard(p_pos),
            spike_p_value=_none_guard(p_spike),
            N_running=_none_guard_int(N)
        )

    def _compute_laps(self, cut=None):
        """Compute the lap partition of the trajectory
        """
        lap_starts = find_laps(self.trajectory.ts, self.trajectory.x,
            self.trajectory.y, start=self.start, alpha_cut=cut)
        self.N_laps = len(lap_starts)
        self.laps = np.asarray(lap_starts + [self.end], 'i8')

    def _scan_extensions_list_default(self):
        scan_table = get_node('/behavior', 'scans')
        pre = [(rec['prefix'], rec['start'])
            for rec in scan_table.where(self.session_query)]
        post = [(rec['end'], rec['postfix'])
            for rec in scan_table.where(self.session_query)]
        tlim = pre + post
        tlim.sort()
        return tlim

    def _extended_scan_list_default(self):
        scan_table = get_node('/behavior', 'scans')
        tlim = [(rec['prefix'], rec['postfix'])
            for rec in scan_table.where(self.session_query)]
        tlim.sort()
        return tlim

    def _scan_list_default(self):
        return self._get_tlim_list('scans')

    def _pause_list_default(self):
        return self._get_tlim_list('pauses')

    def _ripple_list_default(self):
        return self._get_tlim_list('ripples', parent='/physiology')

    def _get_tlim_list(self, table_name, parent='/behavior'):
        table = get_node(parent, table_name)
        tlim = [tuple(rec['tlim']) for rec in table.where(self.session_query)]
        tlim.sort()
        return tlim

    def _extended_scan_and_pause_list_default(self):
        events = self.extended_scan_list + self.pause_list
        events.sort()
        return events

    def _scan_and_pause_list_default(self):
        events = self.scan_list + self.pause_list
        events.sort()
        return events

    # Convenience functions for working with session data

    def cluster_data(self, descr):
        """Cluster data object is loaded on demand and returned, specified by
        string ('tXcY') or tuple (tt, cl) descriptions, or cluster data object
        is simply passed through.
        """
        if type(descr) is tuple and len(descr) == 2:
            descr = 't%dc%d'%descr
        if type(descr) is np.string_:
            descr = str(descr)
        if type(descr) is ClusterData:
            cluster = descr
        else:
            if descr not in self.clusts:
                self._load_cluster(parse_cell_name(descr))
            cluster = self.clusts[descr]
        return cluster

    def to_time(self, ts):
        """Return any time-stamp array as same-sized array of elapsed times
        """
        return stamp_to_time(ts, zero_stamp=self.start)
    T_ = to_time # short-hand synonym

    def get_trajectory_interpolant(self, name):
        """Get a callable interpolant function mapping elapsed time (seconds)
        within the session to trajectory-based values specified by name of the
        TrajectoryData attribute or Moment

        Interpolants are cached.
        """
        key = name
        moment = not hasattr(self.trajectory, name) and name in Moment.Names
        if moment:
            key = 'moments-%s'%key
        if key not in self.interpolants:
            if moment:
                values = self.trajectory.moments[name]
            elif hasattr(self.trajectory, name):
                values = getattr(self.trajectory, name)
            else:
                raise ValueError, 'bad trajectory data name: %s'%name
            self.interpolants[key] = interp1d(self.T_(self.trajectory.ts),
                values, fill_value=0.0, bounds_error=False)
        return self.interpolants[key]
    F_ = get_trajectory_interpolant # short-hand synonym

    def get_spike_train(self, tt_or_tc, cl=None):
        """Get the spike train (as timestamp array) for specified cluster
        """
        if type(tt_or_tc) is str:
            tc = tt_or_tc
        elif cl is not None:
            tc = 't%dc%d'%(tt_or_tc,cl)
        else:
            raise ValueError, "missing cluster number"

        # Load spike train and restrict spike times to session bounds
        _ts = getattr(self.data_group, tc).read()
        _ix = select_from(_ts, [(self.start, self.end)])
        ts = _ts[_ix]

        return ts

    def running_filter(self):
        """Return a consistent dictionary of keywords that define the tracking-
        based filter that selects for forward-running behavior.
        """
        return dict(velocity_filter=True,
                    exclude_off_track=True,
                    exclude=self.extended_scan_and_pause_list)

    def velocity_filter(self, ts, min_velocity=None):
        """Get boolean index array for given timestamp array that filters based
        on a minimum CW track-angle velocity.
        """
        if min_velocity is None:
            min_velocity = self.min_velocity
        return self.F_('forward_velocity')(self.T_(ts)) >= self.min_velocity

    def path_speed_filter(self, ts, min_speed=None):
        """Non-directional minimum path-speed filter
        """
        if min_speed is None:
            min_speed = self.min_speed
        return self.F_('speed')(self.T_(ts)) >= self.min_speed

    def pause_filter(self, *args, **kwargs):
        """Get boolean index array that is the inverse of velocity_filter.
        """
        return True - self.velocity_filter(*args, **kwargs)

    def get_cluster_attribute(self, tc, attr):
        """Get the node attribute for the specified cluster name
        """
        if type(tc) is tuple and len(tc) == 2:
            tc = 't%dc%d'%tc
        clnode = getattr(self.data_group, tc)
        value = None
        try:
            value = getattr(clnode._v_attrs, attr)
        except AttributeError:
            self.out('Missing \'%s\' for cluster %s'%(attr, tc), error=True)
        return value

    def extract_session_data(self, destdir='.'):
        """Save (x, y, t) data for each cluster to simplified data files
        """
        # Set and create destination paths
        new_rat_dir = path.join(path.realpath(destdir), DATA_DIR%self.rat)
        new_day_dir = path.join(new_rat_dir, SESSION_DIR%(self.rat, self.day))
        sess_dir = path.join(new_day_dir, 'session-%02d'%self.session)
        if not path.exists(sess_dir):
            makedirs(sess_dir)

        # Store the trajectory data
        data_fmt = '%.12e'
        traj = self.trajectory
        np.savetxt(path.join(sess_dir, 'pos.xy'),
            np.c_[traj.ts, traj.x, traj.y], fmt=data_fmt)

        # Store spike times for each cluster
        for tc in self.clusts:
            cluster = self.clusts[tc]
            np.savetxt(path.join(sess_dir, '%s.spikes'%tc),
                np.c_[cluster.spikes, cluster.x, cluster.y], fmt=data_fmt)

    # Methods for computing cluster, population and per-lap ratemaps

    def filter_tracking_data(self, ts, x=None, y=None, tlim=None,
        exclude_off_track=False, exclude=None, select=None,
        velocity_filter=True, speed_filter=False, boolean_index=False):
        """Retrieve filtered positional tracking data

        Required arguments:
        ts -- timestamp array for the position data

        Keyword arguments:
        x,y -- position tracking data to be filtered (by default, interpolated
            values are used to compute radius if exclude_off_track is True
            and for return values)
        tlim -- specify time bounds for the data going into the ratemap;
            provide an 1<=int<=N_laps to specify a particular lap; provide a
            tuple of start and end timestamps to specify an arbitrary interval;
            defaults to whole-session ratemap
        exclude -- optional list of timestamp intervals from which to exclude
            spike and trajectory data
        select -- optional list of timestamp intervals from which to select
            spike and trajectory data
        exclude_off_track -- whether to exclude data occurring off the track
        velocity_filter -- optional minimum forward-running velocity threshold
        speed_filter -- optional minimum path-speed threshold
        boolean_index -- get a boolean index array instead of filtered data

        Returns (x, y) tuple of filtered positional data arrays or a boolean
        index array if boolean_index is set to True.
        """
        if not self.is_circle_track and type(tlim) is int:
            tlim = None

        xy_provided = not (x is None or y is None)
        if not xy_provided:
            t = self.T_(ts)
            x, y = self.F_('x')(t), self.F_('y')(t)

        # Get trajectories and spike positions for specified time bounds
        if tlim is not None:
            if type(tlim) is int:
                lap = tlim
                if lap < 1 or lap > self.N_laps:
                    raise ValueError, 'bad lap number specified: %d'%lap
                start, end = self.laps[lap-1], self.laps[lap]
            elif type(tlim) in (list, tuple):
                start, end = tlim
            else:
                raise ValueError, 'bad interval specification: %s'%str(tlim)

            ts, _xy = time_slice_sample(ts, np.c_[x, y], start, end)
            x, y = _xy.T

        filters = [np.ones_like(x, dtype='?')]

        if velocity_filter:
            filters.append(self.velocity_filter(ts))

        if speed_filter:
            filters.append(self.path_speed_filter(ts))

        if exclude is not None:
            filters.append(exclude_from(ts, exclude))

        if select is not None:
            filters.append(select_from(ts, select))

        if exclude_off_track:
            if xy_provided:
                radius = np.sqrt(x**2 + y**2) - TRACK_RADIUS
            else:
                radius = self.F_('radius')(t)
            filters.append(
                np.logical_and(radius <= TRACK_WIDTH/2,
                    radius >= -TRACK_WIDTH/2))

        ix = reduce(op.mul, filters)
        if boolean_index:
            return ix

        return x[ix], y[ix]

    def get_unwrapped_cluster_ratemap(self, cl, bins=None, smoothing=True,
        blur_width=None, **filter_kwds):
        """Compute unwrapped radial ratemap of an individual cluster's activity

        See get_cluster_ratemap doc string for parameter information.

        Other keywords are passed to get_unwrapped_spike_tracking_data.

        Returns (R, mask, tracklim, bins) dict of (N_laps*bins)-shaped arrays of
        the firing rate and occupancy mask data for unwrapped angle bins from 0
        to 360*N_laps degrees CW around the track.
        """
        # Get track angle data for trajectories and spikes
        traj = self.trajectory
        cluster = self.cluster_data(cl)
        cluster_alpha = 360 - self.F_('alpha_unwrapped')(self.T_(cluster.spikes))
        traj_alpha = 360 - traj.alpha_unwrapped

        # Hijack filter_tracking_data to filter track-angle data
        filter_kwds.update(tlim=None, exclude_off_track=False)
        traj_alpha = self.filter_tracking_data(traj.ts, traj_alpha,
            np.zeros_like(traj_alpha), **filter_kwds)[0]
        spike_alpha = self.filter_tracking_data(cluster.spikes, cluster_alpha,
            np.zeros_like(cluster_alpha), **filter_kwds)[0]

        # Set the number of bins for the ratemap
        if bins is None or type(bins) is not int:
            bins = self.default_bins

        # Compute occupancy and spike histograms
        angle_range = 0, 360*self.N_laps
        bin_edges = np.linspace(
            angle_range[0], angle_range[1], bins*self.N_laps+1)
        traj_hist = np.histogram(traj_alpha, bins=bin_edges)[0]
        sp_hist = np.histogram(spike_alpha, bins=bin_edges)[0]

        # Divide spike count by total occupancy for sampled bins
        valid = traj_hist != 0
        mask = True - valid
        sp_hist[valid] *= self.trajectory.fs / traj_hist[valid]
        sp_hist[mask] = 0.0

        # Smooth ratemap with a gaussian blur filter
        if smoothing:
            if blur_width is None:
                blur_width = self.default_blur_width
            sp_hist = unwrapped_blur(sp_hist, blur_width, bins)

        return dict(R=sp_hist, mask=mask, tracklim=angle_range, bins=bin_edges)

    def get_cluster_ratemap(self, cl, bins=None, norm=False, smoothing=True,
        blur_width=None, **filter_kwds):
        """Compute radial ratemap of an individual cluster's activity

        This method should be used as the 'kernel' (bowtie) operation for
        computing place cell responses in the double-rotation dataset.

        Required arguments:
        cl -- cluster key (a 'tXcY' string) specifying a particular cluster

        Keyword arguments:
        bins -- number of bins to use for histograms (determines array size);
            a value of None defaults to default_bins
        norm -- whether to normalize unit responses (by numerical integral)
        smoothing -- whether to smooth ratemaps using gaussian blur filter
        blur_width -- width of gaussian window to use for smoothing; a value
            of None defaults to default_blur_width

        Remaining keywords are using to filter the positional data (see
        filter_tracking_data for parameters).

        Returns (bins,)-shaped firing rate-map array representing angle bins
        from 0 to 360 degrees CCW around the track.
        """
        # Get the trajectory and cluster data
        traj = self.trajectory
        traj_x, traj_y = self.filter_tracking_data(traj.ts, traj.x, traj.y,
            **filter_kwds)
        cluster = self.cluster_data(cl)
        sp_x, sp_y = self.filter_tracking_data(cluster.spikes, cluster.x,
            cluster.y, **filter_kwds)

        # Set the number of bins for the ratemap
        if bins is None or type(bins) is not int:
            bins = self.default_bins

        # Compute occupancy and spike histograms
        traj_hist = get_angle_histogram(traj_x, traj_y, bins)
        sp_hist = get_angle_histogram(sp_x, sp_y, bins)

        # Divide spike count by total occupancy for sampled bins
        valid = (traj_hist != 0)
        sp_hist[valid] *= self.trajectory.fs / traj_hist[valid]
        sp_hist[True - valid] = 0.0

        # Smooth ratemap with a gaussian blur filter
        if smoothing:
            if blur_width is None:
                blur_width = self.default_blur_width
            sp_hist = circular_blur(sp_hist, blur_width)

        # Normalize the rate profile if specified
        if norm and sp_hist.sum():
            sp_hist /= np.trapz(sp_hist)

        return sp_hist

    def get_population_matrix(self, clusters=None, unwrapped=False, **kwargs):
        """Compute a matrix of radial ratemaps for all clusters

        To restrict the population to a particular list of clusters, pass the list
        of cluster names in with the *clusters* keyword. Other keyword arguments
        are passed onto get_cluster_ratemap().

        The rows of the population matrix are sorted by firing rate position unless
        *clusters* is specified.

        Returns (N_clusts, bins) response matrix normally, but a track-angle
        unwrapped (N_clusts, N_laps*bins) matrix if *unwrapped* is specified.
        """
        # Set the list of clusters to be used as the population
        try:
            clusts = self.get_clusters(clusters)
        except ValueError:
            return np.array([])

        # Initialize the population matrix
        bins = kwargs.get('bins', self.default_bins)
        cols = bins
        if unwrapped:
            cols *= self.N_laps
            mask = np.empty((len(clusts), cols), '?')
        R = np.empty((len(clusts), cols), 'd')

        # Construct the population matrix of ratemaps
        for i,cl in enumerate(clusts):
            if unwrapped:
                R_data = self.get_unwrapped_cluster_ratemap(cl, **kwargs)
                R[i], mask[i] = R_data['R'], R_data['mask']
            else:
                R[i] = self.get_cluster_ratemap(cl, **kwargs)

        # Return the maps sorted by peak firing-rate location only if a particular
        # list of clusters was NOT passed in.
        if clusters is None:
            R = R[np.argsort(np.argmax(R[:,:bins], axis=1))]

        if unwrapped:
            res = dict(R=R, mask=mask, tracklim=R_data['tracklim'],
                bins=R_data['bins'])
        else:
            res = R

        return res

    def get_population_lap_matrix(self, session_sort=False, **kwargs):
        """Construct concatentation of per-lap population response matrices

        Keyword arguments:
        session_sort -- additionally compute session-wide population map in order
            to sort the output matrix by overall peak locations

        Returns (N_clusts, bins, N_laps) response matrix.
        """
        if not self.is_circle_track:
            raise ValueError, 'lap matrix only available for circle track data'

        # Set clusters list to maintain cluster order across laps
        if 'clusters' not in kwargs:
            kwargs['clusters'] = self.get_criterion_clusters()

        # Create lap matrix
        for lap in xrange(1, self.N_laps+1):
            kwargs['tlim'] = lap
            if lap == 1:
                R = self.get_population_matrix(**kwargs)
                L = np.empty(R.shape+(self.N_laps,), 'd')
                L[..., 0] = R
            else:
                L[..., lap-1] = self.get_population_matrix(**kwargs)

        # Return ratemap, sorted by session peaks if specified
        if session_sort:
            kwargs['lap'] = 'all'
            Rsess = self.get_population_matrix(**kwargs)
            return L[np.argsort(np.argmax(Rsess, axis=1))]
        else:
            return L

    def get_clusters(self, request=None):
        """Get a list of cluster names, either all criterion clusters or specify
        a list that will be checked against the stored clusters.
        """
        if type(request) is ClusterCriteria or request is None:
            clusts = self.get_criterion_clusters(criteria=request)
        elif type(request) is str and request in self.clusts:
            clusts = [request]
        elif np.iterable(request):
            clusts = filter(lambda c: c in self.clusts, request)
            if len(clusts) < len(request):
                self.out("Warning: %d requested clusters not found"%
                    (len(request)-len(clusts)))
        else:
            raise ValueError, "bad cluster specification: %s"%str(request)
        return clusts

    def get_criterion_clusters(self, criteria=None):
        """Return a list of clusters matching filter criteria specified by a
        ClusterCriteria argument or by the current value of the instance
        variable *cluster_criteria*.
        """
        if criteria is None:
            criteria = self.cluster_criteria
        elif type(criteria) is not ClusterCriteria:
            raise TypeError, "requires ClusterCriteria argument"
        if criteria is None:
            self.out('warning: no cluster criteria selected, using all cells')
            clusts = self.clusts.keys()
        else:
            clusts = [cl for cl in self.clusts if criteria.filter(self.clusts[cl])]
        clusts.sort()
        return clusts

    # Convenience functions

    def _session_query_default(self):
        """Returns a session query string for tables with rat/day/session columns
        """
        return '(rat==%d)&(day==%d)&(session==%d)'%self.rds

    def random_timestamp_array(self, size, avoid_scans=False):
        """Generate an array of random timestamps acrosss this session
        """
        delta = self.end - self.start
        a = np.random.randint(delta, size=size).astype('int64')
        a += self.start
        a.sort()
        if avoid_scans:
            stable = get_node('/behavior', 'scans')
            scans = np.array([rec['tlim'] for rec in
                stable.where(self.session_query)])
            N = len(scans)
            gtr_start = a >= scans[:,0].reshape(N,1).repeat(size, axis=1)
            less_end = a <= scans[:,1].reshape(N,1).repeat(size, axis=1)
            hit = np.logical_and(gtr_start, less_end).sum(axis=0).nonzero()[0]
            for ix in hit:
                c = 0
                while c < 200:
                    a[ix] = self.start + np.random.randint(delta)
                    min_ix = np.max((scans[:,0]<=a[ix]).nonzero()[0])
                    if a[ix] > scans[min_ix, 1]:
                        break
        return a

    # Convenience functions for loading session data

    @classmethod
    def get_session_list(cls, rat, day):
        """Get a list of SessionData objects for a given data set
        """
        return [cls(rds=(rat, day, m)) for m in get_maze_list(rat, day)]
