# encoding: utf-8
"""
scanr.spike -- Utility functions for handling discrete spiking data

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import sys, re
from os import path
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

# Package imports
from .config import Config
from .paths import get_path, get_data_file_path
from .data import get_node, new_table, get_unique_row, flush_file, close_file
from .time import stamp_to_time, select_from
from .cluster import (ClusterQuality, string_to_quality, PrincipalCellCriteria,
    get_tetrode_restriction_criterion, AND)
from .meta import walk_tetrodes, get_tetrode_comment, get_maze_list
from .tools.bash import CPrint

# Define anatomical regions that have been recorded
AreaSubdivs = dict( CA1=['proximal', 'intermediate', 'distal'],
                    CA3=['a', 'b', 'c'],
                    DG=['lower blade', 'upper blade'],
                    MEC=['super', 'mid', 'deep'],
                    LEC=['super', 'mid', 'deep']    )
PrimaryAreas = sorted(AreaSubdivs.keys())
SecondaryAreas = ['CA2', 'fasciola', 'subiculum', 'thalamus',
    'perirhinal', 'postrhinal', 'para', 'fissure', 'TeA', 'AD', 'LD', 'AV']
AllAreas = PrimaryAreas + SecondaryAreas


# Table descriptors for spiking data
TetrodeDescr = {    'rat'               : tb.UInt16Col(pos=1),
                    'day'               : tb.UInt8Col(pos=2),
                    'tt'                : tb.UInt8Col(pos=3),
                    'expt_type'         : tb.StringCol(itemsize=8, pos=4),
                    'EEG'               : tb.BoolCol(pos=5),
                    'relative_theta'    : tb.FloatCol(pos=6),
                    'comment'           : tb.StringCol(itemsize=64, pos=7),
                    'area'              : tb.StringCol(itemsize=16, pos=8),
                    'ambiguous'         : tb.BoolCol(pos=9),
                    'subdiv'            : tb.StringCol(itemsize=16, pos=10),
                    'layer'             : tb.StringCol(itemsize=16, pos=11),
                    'unsure'            : tb.BoolCol(pos=12) }

ECQualityDescr = {  'area'              : tb.StringCol(itemsize=4, pos=1),
                    'rat'               : tb.UInt16Col(pos=2),
                    'day'               : tb.UInt8Col(pos=3),
                    'session'           : tb.UInt8Col(pos=4),
                    'tt'                : tb.UInt8Col(pos=5),
                    'cluster'           : tb.UInt8Col(pos=6)  }


class SpikePartition(object):

    raster_fmt = dict(lw=0.5)
    spike_fmt = dict(s=30, marker='o', facecolor='none', linewidths=1,
        alpha=0.8)
    color_dict = dict(other='k', pause='#E0BB26', scan_extension='#0DD6D6', scan='g',
        field='r', full_field='r', mod='b')

    plot_order = list(['other', 'pause', 'scan_extension', 'full_field', 'scan'])
    mod_plot_order = list(['other', 'pause', 'scan_extension', 'field', 'mod', 'scan'])
    ix = None

    @classmethod
    def split(cls, session, tc, scan_extensions=False, mod_table=None):
        cluster = session.cluster_data(tc)
        ix = dict()
        ix['scan'] = select_from(cluster.spikes, session.scan_list)
        ix['pause'] = np.logical_and(
            np.logical_or(
                select_from(cluster.spikes, session.pause_list),
                session.pause_filter(cluster.spikes)),
            True - ix['scan'])
        ix['full_field'] = session.filter_tracking_data(
            cluster.spikes, cluster.x, cluster.y,
            exclude=session.scan_and_pause_list,
            exclude_off_track=True, velocity_filter=True,
            boolean_index=True)
        if scan_extensions:
            ix['scan_extension'] = select_from(cluster.spikes,
                session.scan_extensions_list)
            ix['pause'] = np.logical_and(
                ix['pause'], True - ix['scan_extension'])
            ix['full_field'] = np.logical_and(
                ix['full_field'], True - ix['scan_extension'])
        else:
            ix['scan_extension'] = np.zeros_like(ix['scan'])
        if mod_table is None:
            ix['field'] = ix['full_field']
            ix['mod'] = np.zeros_like(ix['field'], dtype='?')
        else:
            mod_spikes = select_from(cluster.spikes,
                [rec['tlim'] for rec in
                    mod_table.where(
                        session.session_query + '&(tc=="%s")'%cluster.name)])
            ix['mod'] = ix['full_field'] * mod_spikes
            ix['field'] = ix['full_field'] * (True - mod_spikes)
        ix['other'] = True - (
            ix['scan'] + ix['pause'] + ix['mod'] + ix['field'] + ix['scan_extension'])
        cls.ix = ix
        return ix

    @classmethod
    def plot(cls, session, tc, ax=None, x=None, y=None, z=None,
        scan_extensions=True, mod_table=None, **kwds):
        ix = cls.split(session, tc, scan_extensions=scan_extensions,
            mod_table=mod_table)
        segments = (mod_table is None) and cls.plot_order or cls.mod_plot_order
        colors = [cls.color_dict[k] for k in segments]

        is_3d = hasattr(ax, 'plot3D')
        cluster = session.cluster_data(tc)
        if x is None:
            x = cluster.x
        if y is None:
            y = cluster.y
        if is_3d:
            if z is None:
                z = session.T_(cluster.spikes)
        if ax is None:
            ax = plt.gca()

        plt.ioff()
        fmt = cls.spike_fmt.copy()
        fmt.update(kwds)
        if is_3d:
            fmt.update(marker='.', linewidths=0, edgecolor='none')
            del fmt['facecolor']
        else:
            fmt.update(zorder=1)
        for k, col in zip(segments, colors):
            if not ix[k].any():
                continue
            if is_3d:
                ax.scatter3D(x[ix[k]], y[ix[k]], zs=z[ix[k]],
                    facecolor=col, **fmt)
            else:
                ax.scatter(x[ix[k]], y[ix[k]], edgecolor=col, **fmt)
                fmt['zorder'] += 1
        plt.ion()
        plt.draw()

    @classmethod
    def raster(cls, session, tc, ax=None, ylim=(0, 1),
        scan_extensions=True, mod_table=None, **kwds):
        """Plot spike rasters along the unwrapped track measured in laps
        """
        ix = cls.split(session, tc, scan_extensions=scan_extensions,
            mod_table=mod_table)
        segments = (mod_table is None) and cls.plot_order or cls.mod_plot_order
        colors = [cls.color_dict[k] for k in segments]

        cluster = session.cluster_data(tc)
        x = 1.0 - session.F_('alpha_unwrapped')(session.T_(cluster.spikes)) / 360.0
        if ax is None:
            ax = plt.gca()

        plt.ioff()
        fmt = cls.raster_fmt.copy()
        fmt.update(kwds, zorder=1)
        for k, col in zip(segments, colors):
            if not ix[k].any():
                continue
            ax.vlines(x[ix[k]], ylim[0], ylim[1], color=col, **fmt)
            fmt['zorder'] += 1
        plt.ion()
        plt.draw()


class TetrodeSelect(object):

    """
    Mix-in or static capability for choosing datasets and tetrodes that match a
    certain query on /metadata/tetrodes, such as 'area=="DG"'.

    Class/instance methods:
    datasets/_get_datasets -- unique datasets with matching tetrodes
    tetrodes/_get_valid_tetrodes -- matching tetrode list
    criterion -- get a ClusterCriteria object for the matching tetrodes

    Keywords passed to datasets/tetrodes methods are processed into query
    enhancements by _process_query:
    allow_ambiguous -- include tetrodes with ambiguous area, default False
    debug -- print a debug statement with the selected tetrodes, default False
    """

    _tetrode_out = CPrint(prefix='TetrodeSelect', color='cyan')

    @classmethod
    def datasets(cls, query, quiet=False, **kwds):
        datasets = []
        ttable = get_node('/metadata', 'tetrodes')
        for rec in ttable.where(cls._process_query(query, **kwds)):
            day = rec['rat'], rec['day']
            if day not in datasets:
                datasets.append(day)
        datasets.sort()
        if not quiet:
            cls._tetrode_out("Matching datasets: %s"%datasets)
        return datasets

    @classmethod
    def tetrodes(cls, dataset, query, quiet=False, **kwds):
        valid = []
        rat, day = dataset
        ttable = get_node('/metadata', 'tetrodes')
        valid = [int(rec['tt']) for rec in
            ttable.where(cls._process_query(query, **kwds))
            if rec['rat'] == rat and rec['day'] == day]
        valid.sort()
        if not quiet:
            cls._tetrode_out("Valid tetrodes: %s"%valid)
        return valid

    @classmethod
    def criterion(cls, *args, **kwds):
        return get_tetrode_restriction_criterion(cls.tetrodes(*args, **kwds))

    @classmethod
    def _process_query(cls, query, debug=False, allow_ambiguous=False):
        orig = query
        if not allow_ambiguous:
            query = '(%s)&(ambiguous==False)'%orig
        if debug:
            cls._tetrode_out('Tetrode query: %s'%query)
        return query

    # Instance methods (these may be deprecated at some point)

    def _get_datasets(self, *args, **kwds):
        TetrodeSelect.datasets(*args, **kwds)

    def _get_valid_tetrodes(self, *args, **kwds):
        TetrodeSelect.tetrodes(*args, **kwds)


def acorr(t, **kwds):
    """Compute the spike train autocorrelogram of a spike train array
    """
    return xcorr(t, t, **kwds)

def xcorr(a, b, maxlag=1.0, bins=128):
    """Compute the spike train correlogram of two spike train arrays

    Arguments:
    a, b -- compute correlations of spike train b relative to spike train a
    maxlag -- range of lag times (+/-) to be returned
    bins -- number of discrete lag bins to use for the histogram
    use_millis -- lags in milliseconds (including *maxlag*), instead of seconds

    Returns correlogram as (counts, bin_centers) tuple. The counts array is
    an array of doubles, to allow easy normalization.
    """
    dot, ones = np.dot, np.ones
    na, nb = a.size, b.size
    D = dot(b.reshape(nb,1), ones((1,na))) - dot(ones((nb,1)), a.reshape(1,na))
    if bins % 2 == 0:
        bins += 1
    H, edges = np.histogram(D, range=(-maxlag, maxlag), bins=bins, density=False)
    centers = edges[:-1] + np.diff(edges)/2
    return H.astype('d'), centers

def interval_firing_rate(t, duration=None):
    """ISI-based firing rate computation for a given spike train

    Arguments:
    t -- spike train (or segment) for which to compute ISI firing rate
    duration -- total sampling duration of the spike train; defaults to elapsed
        time of the spike train
    """
    if duration is None:
        duration = t.ptp()
    if t.size == 0:
        return 0.0
    elif t.size == 1:
        return 1 / duration
    else:
        return 1 / np.diff(t).mean()

def plot_correlogram(data, ax=None, fmt=None, plot_type='verts', zero_line=True,
    norm=False, is_corr=False, **kwds):
    """Plot spike train correlogram to the specified axes

    Required arguments:
    data -- spike train array (acorr) or tuple of two arrays (xcorr)
    ax -- axes object to draw autocorrelogram into
    fmt -- an optional style formatting dictionary (vlines or plot)
    plot_type -- plot type: can be "verts" (vlines), "steps", or "lines"
    zero_line -- draw a vertical dotted line at zero lag for reference
    norm -- normalize so that the peak correlation is 1
    is_corr -- whether data argument is pre-computed correlogram in the form of
        a (C, lags) tuple

    Remaining keyword arguments are passed to the xcorr function. The handle to
    the plot (not the axis) is returned.
    """
    t = data
    if is_corr:
        assert len(t) == 2, "is_corr failed: data must be length 2"
        assert len(t[0]) == len(t[1]), "is_corr failed, length mismatch"
        C, lags = t
    elif type(t) is tuple:
        if len(t) == 2:
            C, lags = xcorr(*t, **kwds)
        else:
            raise ValueError, "spike train arg must be len 2 (was %d)"%len(t)
    else:
        C, lags = acorr(t, **kwds)
    if norm:
        C = C.astype(float) / C.max()

    if ax is None:
        ax = plt.gca()
    if plot_type in ("lines", "steps"):
        line_fmt = dict(lw=2)
        if plot_type == "steps":
            line_fmt.update(drawstyle='steps-mid')
        if fmt:
            line_fmt.update(fmt)
        h = ax.plot(lags, C, **line_fmt)
        h = h[0]
    elif plot_type == "verts":
        vlines_fmt = dict(colors='k', lw=2)
        if fmt:
            vlines_fmt.update(fmt)
        h = ax.vlines(lags, [0], C, **vlines_fmt)

    ax.axhline(color='k')
    if zero_line:
        ax.axvline(color='k', ls=':')
    ax.set_xlim(lags.min(), lags.max())

    plt.draw()
    return h # return plot handle

def get_multiunit_activity(rds, tetrodes, min_quality='marginal', dt=0.01,
    smoothing=None):
    """Compute normalized population activity across the specified tetrodes

    Arguments:
    rds -- session to analyze
    tetrodes -- list of tetrode numbers composing the population
    dt -- temporal bin width (seconds) for quantizing spikes
    smoothing -- sigma of gaussian kernel for smoothing (default dt)

    Returns (t, MUA) timeseries.
    """
    #TODO: collate spike trains and create a smoothed MUA signal

def find_theta_tetrode(dataset, condn="EEG==True", ambiguous=True,
    verbose=False):
    """Return (tt, rel_theta) tuple for the tetrode from given dataset with
    highest relative theta power

    Arguments:
    dataset -- (rat, day) tuple specifying dataset
    condn -- /metadata/tetrodes conditional query, defaults to 'EEG==True'

    Returns None if no matching tetrodes.
    """
    all_tetrodes = TetrodeSelect.tetrodes(dataset, condn,
        allow_ambiguous=ambiguous, quiet=(not verbose))
    tetrode_table = get_node('/metadata', 'tetrodes')

    rat, day = dataset
    rel_theta = [get_unique_row(tetrode_table,
            '(rat==%d)&(day==%d)&(tt==%d)'%(rat, day, tt))['relative_theta']
            for tt in all_tetrodes]

    if not rel_theta:
        return None
    return all_tetrodes[np.argmax(rel_theta)], np.max(rel_theta)

def parse_cell_name(tc):
    """Return (tetrode, cluster) tuple from string name of cell
    """
    search = re.match('t(\d+)c(\d+)$', tc)
    if not search:
        raise ValueError, 'bad cell name: %s'%str(tc)
    tokens = search.groups()
    return tuple(int(tok) for tok in tokens)

def get_tetrode_area(rat, day, tt):
    """Get the primary recording area associated with a particular tetrode-day
    """
    tetrode_table = get_node('/metadata', 'tetrodes')
    tt_query = '(rat==%d)&(day==%d)&(tt==%d)'%(rat, day, tt)
    return get_unique_row(tetrode_table, tt_query)['area']

def get_session_clusters(maze, tt=None):
    """Get list of child spike-train arrays for parent maze group

    Keyword argumnets:
    tt -- restrict clusters to a particular tetrode or list of tetrodes

    Returns list of (t, c) tetrode-cluster tuples.
    """
    tc_list = []
    if tt is not None:
        if not np.iterable(tt):
            tt = [tt]
    for name in maze._v_children.keys():
        tc_match = re.match('t(\d+)c(\d+)$', name)
        if tc_match:
            tc = tc_match.groups()
            t, c = int(tc[0]), int(tc[1])
            if tt and t not in tt:
                continue
            tc_list.append((t, c))
    tc_list.sort()
    return tc_list

def create_EC_cluster_table():
    """Create /metadata/ec_clusters table node for high-quality MEC/LEC cells
    """
    ectable = new_table('/metadata', 'ec_cluster_table',
        ECQualityDescr, title='MEC-LEC High-Quality Clusters')

    qfile_path = path.join(get_path(), 'MEC_LEC_quality.csv')
    try:
        csv_file = file(qfile_path, 'r')
    except IOError, e:
        raise e, 'could not open EC cluster file %s'%qfile_path

    # Column names (areas)
    areas = [tok.strip() for tok in csv_file.readline().split(',')]

    # Parse rows
    pattern = re.compile('(\d\d\d)-(\d\d)_+(\d+)_+m (\d).(\d+)')
    row = ectable.row
    for line in csv_file.readlines():
        cl = [tok.strip() for tok in line.split(',')]
        for i, desc in enumerate(cl):
            match = re.match(pattern, desc)
            if not match:
                continue
            tokens = match.groups()
            row['area'] = areas[i]
            row['rat'] = int(tokens[0])
            row['day'] = int(tokens[1])
            row['session'] = int(tokens[3])
            row['tt'] = int(tokens[2])
            row['cluster'] = int(tokens[4])
            row.append()
    csv_file.close()
    ectable.flush()

def create_tetrode_table():
    """Create /metadata/tetrodes table based on tetrode location metadata

    Called by append_theta script which also populates the EEG and relative_theta
    columns of the table. Thus, running append_theta performs a complete refresh
    of /metadata/tetrodes.
    """
    pos = len(TetrodeDescr) + 1
    for area in AllAreas:
        TetrodeDescr[area] = tb.BoolCol(pos=pos)
        pos += 1

    ttable = new_table('/metadata', 'tetrode_table',
        TetrodeDescr, title='Tetrode Locations Per Dataset')
    stable = get_node('/metadata', 'sessions')

    row = ttable.row
    count = 0
    for rat, day, tt in walk_tetrodes():
        comment = get_tetrode_comment(rat, day, tt)
        _comment = comment.lower()
        row['rat'] = rat
        row['day'] = day
        row['tt'] = tt
        row['expt_type'] = get_unique_row(stable,
            '(rat==%d)&(day==%d)'%(rat, day), raise_on_fail=False)['expt_type']
        row['EEG'] = False # populated by append_theta script
        row['relative_theta'] = 0.0 # ibid.
        row['comment'] = comment
        row['layer'] = '' #TODO
        row['subdiv'] = '' # Set below if primary area is found
        row['ambiguous'] = 'ambiguous' in _comment
        row['unsure'] = '?' in _comment

        for area in AllAreas:
            row[area] = area.lower() in _comment

        for area in PrimaryAreas:
            if comment.startswith(area):
                row['area'] = area
                for subdiv in AreaSubdivs[area]:
                    qualifier = str(_comment).split('/')[0][len(area):]
                    if subdiv.lower() in qualifier:
                        row['subdiv'] = subdiv
                        break

        row.append()
        sys.stdout.write('tetrode_table: rat %d, day %d, tt %02d: %s\n'%
            (rat, day, tt, comment))

        count += 1
        if count % 10 == 0:
            ttable.flush()

    flush_file()

def find_spike_files(rat, day, session, tetrode):
    """Find and return a list of filenames and cluster id numbers (ints) for
    the specified recording tetrode and session

    Returns (filenames, clusters) tuple of lists.
    """
    glob = get_data_file_path('cluster', rat=rat, day=day, session=session,
        tetrode=tetrode, do_glob=True)
    tmaze = re.compile('maze%d.(\d+)$'%session)
    filenames = []
    clusters = []
    for fn in glob:
        if fn.startswith('dummy'):
            continue
        match = tmaze.search(fn)
        if match:
            filenames.append(fn)
            clusters.append(int(match.groups()[0]))
    return filenames, clusters

def load_mean_spike_width(rat, day, session, tt, cl):
    """Retrieve the spike maxwidth data from a cluster file

    Returns mean spike width for the cluster in milliseconds.
    """
    fn = get_data_file_path('cluster', rat=rat, day=day, session=session,
        tetrode=tt, cluster=cl)
    if not fn:
        sys.stderr.write('mean_spike_width: could not find cluster file\n')
        return 0.0
    fs = Config['sample_rate']['spikes']
    maxwidth = 0.0
    try:
        cldata = np.loadtxt(fn, comments='%', delimiter=',', skiprows=13,
            usecols=[14]) # maxwidth is 15th column in cl-maze files
    except IOError:
        sys.stderr.write('mean_spike_width: error loading %s\n'%fn)
    else:
        nspikes = cldata.size
        if nspikes != 0:
            if nspikes == 1:
                nsamples = float(cldata)
            else:
                nsamples = cldata.mean()
            maxwidth = nsamples * (1000/fs)
    return maxwidth

def load_cluster_quality(rat, day, tt, cl):
    """Retrieve the cluster isolation quality rating from a ClNotes file

    Returns (quality, comment) tuple of a ClusterQuality value and the string
    comment left by the experimenter.
    """
    fn = get_data_file_path('clnotes', rat=rat, day=day, tetrode=tt,
        search=False)
    quality = ClusterQuality.None
    comment = "n/a"
    try:
        fd = file(fn, 'r')
    except IOError:
        sys.stderr.write('cluster_quality: error opening %s\n'%fn)
        return quality, comment
    try:
        lines = fd.readlines()
    except IOError:
        sys.stderr.write('cluster_quality: error reading %s\n'%fn)
    else:
        found = False
        quality_pattern = re.compile('(\d+),"(.*)","(.*)"')
        for line in lines:
            search = re.match(quality_pattern, line)
            if not search:
                continue
            cl_str, quality_str, comment = search.groups()
            found_cl = int(cl_str)
            if found_cl == cl:
                quality = string_to_quality(quality_str)
                found = True
                break
        if not found:
            sys.stderr.write('cluster_quality: missing cluster %d in %s\n'%
                (cl,fn))
    finally:
        fd.close()
    return quality, comment

def read_spike_file(fn):
    """Load spike train timestamp array from a cluster file

    Array of spike-time timestamps is returned. If an error occurs during
    load, then None is returned unless *error* keyword is set to True.
    """
    if not path.exists(fn):
        raise ValueError, 'invalid cluster file specified: %s.'%fn
    spikes = None
    try:
        spikes = np.loadtxt(fn, skiprows=13, delimiter=',', usecols=[-1])
    except IOError:
        sys.stderr.write('read_spike_file: error loading %s\n'%fn)
    else:
        spikes = spikes.astype('uint64')
        if spikes.size and not np.iterable(spikes): # single spike case
            spikes = spikes.reshape((1,))
    return spikes

def session_timing_from_cluster_file(fn):
    """Read the start/end timestamps of the recording session listed in the
    given cluster file
    """
    skip = 11
    fd = file(fn, 'r')
    while skip:
        fd.readline()
        skip -= 1
    start = long(fd.readline())
    end = long(fd.readline())
    fd.close()
    return start, end
