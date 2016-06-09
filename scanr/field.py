# encoding: utf-8
"""
scanr.field -- Utility functions for operating on place field data

Created by Joe Monaco, April 5, 2012
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
import matplotlib.pyplot as plt

# Package imports
from .config import Config
from .session import SessionData
from .spike import SpikePartition
from .tools.radians import get_angle_array, xy_to_rad_vec
from .tools.filters import find_maxima
from .tools import circstat

# Place field criteria
FieldCfg = Config['placefield']
MIN_PEAK_RATE = FieldCfg['min_peak_rate']
MIN_SIZE_DEGREES = FieldCfg['min_size_degrees']
MAX_SIZE_DEGREES = FieldCfg['max_size_degrees']
DEFAULT_BINS = FieldCfg['default_bins']
FLOOR_FRACTION = FieldCfg['floor_fraction']
KILL_AFTER = FieldCfg['kill_after']


def plot_concentric_lap_spikes(session, tc, ax=None, start='inner', lmin=0.3,
    **fmt):
    """Plot place cell spiking across laps represented as concentric circles

    Keyword arguments:
    ax -- optional pre-existing polar axis, otherwise display new figure
    start -- ['inner'|'outer'] specifying where the first lap goes on the inner
        or outer ring, with appropriate directionality
    lmin -- fraction along lap (radial) axis for inner-most lap to start

    To draw into a pre-existing axis, *ax* must be a polar projection axis.
    Remaining keywords are passed to SpikePartition.plot() to format the
    scatter plot.
    """
    plt.ioff()
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111, projection='polar')

    # Recut lap partition
    COM = cut_laps_opposite_field(session, tc)
    laps = session.laps

    # Get laps and track angles for spikes
    cluster = session.cluster_data(tc)
    L = np.array(map(lambda t: (t >= laps).sum(), cluster.spikes)) - 1
    if start == 'outer':
        L = L.max() - L
    spike_laps = lmin + ((1 - lmin) / L.max()) * L
    spike_alpha = xy_to_rad_vec(cluster.x, cluster.y)

    # Draw concentric rings
    max_lap = L.max()
    rgrid = lmin + ((1 - lmin) / max_lap) * np.arange(max_lap + 1)
    theta = np.linspace(0, 2*np.pi, 360)
    zed = np.zeros(360, 'd')
    for r in rgrid:
        ax.plot(theta, r + zed, 'k-', lw=0.5, zorder=-1)

    # Plot spikes and turn off axis
    SpikePartition.plot(session, tc, ax=ax, x=spike_alpha, y=spike_laps, **fmt)
    ax.set_axis_off()

    plt.ion()
    plt.show()
    plt.draw()

def plot_fields(rds, tc, bins=DEFAULT_BINS, **field_kwds):
    """Plot linear rate-map with detected fields highlighted

    Keyword arguments are passed to mark_all_fields.
    """
    data = SessionData.get(rds)
    R = data.get_cluster_ratemap(tc, bins=bins, blur_width=360.0 / bins,
        **data.running_filter())
    plt.ioff()
    f = plt.figure()
    ax = f.add_subplot(111)
    alpha = track_bins(bins)
    plt.plot(alpha, R, 'k-')
    ax.axhline(R.max() * FLOOR_FRACTION, c='k', ls='--')
    for field in mark_all_fields(R, **field_kwds):
        R_field = np.ma.masked_where(True - field, R)
        ax.plot(alpha, R_field, '-o', lw=2)
    ax.set_title('Rat %d, Day %d, M%d, Cell %s'%(rds + (tc,)))
    ax.set_xlim(0, 360)
    plt.ion()
    plt.show()

def track_bins(R, centers=True, degrees=True):
    """Get the [0, 360) bin centers for a field array (or number of bins)

    Keyword arguments:
    centers -- whether to return bin centers (True) or left edges (False)
    degrees -- values in degrees or radians
    """
    if type(R) is int:
        bins = R
    else:
        bins = np.squeeze(R).shape[-1]
    dx = 0.0
    if centers:
        dx = degrees and 180./bins or np.pi/bins
    return get_angle_array(bins, degrees=degrees) + dx

def mark_max_field(R, **kwds):
    """Get maximal field marked in a boolean array

    Keyword arguments as in mark_all_fields.
    """
    kwds.update(min_peak=0, primary_only=True)
    return np.squeeze(mark_all_fields(R, **kwds))

def mark_all_fields(R, min_peak=MIN_PEAK_RATE, floor_frac=FLOOR_FRACTION,
    kill_on=KILL_AFTER, primary_only=False):
    """Get boolean arrays the same length as the given linearized ratemap with
    the place fields marked by contiguous True elements

    Arguments:
    min_peak -- minimum peak rate for a field to count
    floor -- fraction of field peak rate to serve as the tail cutoff
    kill_on -- number of sub-floor bins to stop search for edges
    primary_only -- just the maximal-peak-rate field, ignoring others

    Returns (N_fields, N_bins)
    """
    R = np.squeeze(R)
    maxima = find_maxima(R, wrapped=True)
    floor = floor_frac * R.max()
    subthreshold = R < floor
    putative = []
    fields = []

    if primary_only:
        maxima = maxima[np.argmax(R[maxima])],

    for max_ix in maxima:

        # Place-field peak-rate filter
        if R[max_ix] < max(min_peak, floor):
            continue

        # Mark off the field in both directions
        F = np.zeros_like(R, dtype='?')
        for inc in -1, +1:
            i = max_ix
            hit = 0
            while hit < kill_on and not F.all():
                if subthreshold[i]:
                    hit += 1
                F[i] = True
                i = (i + inc) % R.size

        # Check for collisions
        hit = False
        for G in putative:
            if (F * G).sum():
                G[:] = F + G # aggregate the fields
                hit = True
                break

        if not hit:
            putative.append(F)

    # Size filter on aggregated fields
    for field in putative:
        size = field.sum() * (360.0 / R.size)
        if MIN_SIZE_DEGREES <= size <= MAX_SIZE_DEGREES:
            fields.append(field)

    return np.array(fields)

def field_extent(marked_field):
    """For a boolean marked field array, return track-angle bounds in degrees
    """
    F = np.squeeze(marked_field).astype('?')
    if F.all():
        start, end = 0, 360
    elif not F.any():
        start, end = 360, 0
    else:
        bins = F.size
        centers = track_bins(bins, centers=True, degrees=True)
        d_field = np.diff(np.r_[F[-1], F].astype('i'))
        start = centers[(d_field == +1).nonzero()[0][0]]
        end = centers[(d_field == -1).nonzero()[0][0]]
    return start, end

def center_of_mass(R, F=None, degrees=True):
    """Get the center-of-mass of a linearized firing-rate map, optionally
    restricted to a marked field (via *F*)
    """
    if F is None:
        F = slice(None)
    angles = track_bins(R, degrees=False)
    COM = circstat.mean(angles[F], w=R[F])
    if degrees:
        COM *= (180 / np.pi)
    return COM

def cut_laps_opposite_field(session, tc, R=None, bins=DEFAULT_BINS):
    """Recompute a session's laps by cutting at the track location opposite to
    the center-of-mass of the maximal field for the given cell name.

    Returns place field COM in degrees.
    """
    if R is None:
        R = session.get_cluster_ratemap(tc, bins=bins, blur_width=360./bins,
            **session.running_filter())
    else:
        R = np.squeeze(R)

    COM = center_of_mass(R, F=mark_max_field(R), degrees=False)
    session._compute_laps(cut=COM-np.pi)

    return (180 / np.pi) * COM
