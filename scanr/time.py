# encoding: utf-8
"""
scanr.time -- Utility functions for handling time and timestamp conversions

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
import sys

# Package imports
from .config import Config
from .tools.compress import DeltaCompress

# Config
STAMP_RATE = Config['sample_rate']['time']
DEBUG = Config['debug_mode']


def compress_timestamps(ts):
    """Compress an array of timestamps to a delta matrix
    """
    return DeltaCompress.encode(ts)

def extract_timestamps(delta_ts):
    """Extract an array of timestamps from a delta matrix
    """
    return DeltaCompress.decode(delta_ts)

def find_sample_rate(ts):
    """Determine the sampling rate of the given timestamp array
    """
    dt = np.diff(ts)
    dt = dt[dt!=0]
    return float(STAMP_RATE/np.median(dt))

def select_from(ts, intervals):
    """Get boolean index array that selects timestamp intervals from the
    specified list from the given timestamp array

    Arguments:
    ts -- timestamp array from which times are selected
    intervals -- list of (start, end) tuples of timestamp intervals
    """
    ts = np.asarray(ts)
    ix = np.zeros_like(ts, dtype='?')
    for start, end in intervals:
        ix[np.logical_and(ts>=start, ts<=end)] = True
    return ix

def exclude_from(ts, intervals):
    """Get boolean index array that excludes timestamp intervals from the
    specified list from the given timestamp array

    Arguments:
    ts -- timestamp array from which times are excluded
    intervals -- list of (start, end) tuples of timestamp intervals
    """
    return True - select_from(ts, intervals)

def time_slice_sample(t, samples=None, start=None, end=None):
    """Slice an interval out of time/timestamp and sample arrays

    Arguments:
    t -- time/timestamp array (length must match first dimension of samples)
    samples -- optional array (or column-wise matrix) of data samples
    start -- time/timestamp of start (default to first)
    end -- time/timestamp of end (default to last)

    Returns slice of time array, or (t, samples) tuple of sliced time and
    sample arrays if additional sample arrays were passed in.
    """
    t = np.asarray(t)
    interval = time_slice(t, start, end)
    if samples is None:
        return t[interval]
    else:
        samples = np.asarray(samples)
        if t.shape[0] != samples.shape[0]:
            raise ValueError, 'shape mismatch for timestamp and sample arrays'
        return t[interval], samples[interval]

def time_slice(t, start=None, end=None):
    """Index slice object returned for given timestamp bounds

    Arguments:
    t -- time/timestamp array
    start -- time/timestamp of start (default to first)
    end -- time/timestamp of end (default to last)

    Returns slice object which indexes the specified time/timestamp range if
    it exists.
    """
    t = np.asarray(t).flatten()
    t0, t1 = 0, t.size
    if start is not None:
        if t[t0] <= start <= t[t1 - 1]:
            t0 = (t>=start).nonzero()[0].min()
        elif start > t[t1 - 1]:
            return slice(0, 0, None)
    if end is not None:
        if t[t0] <= end <= t[t1 - 1]:
            t1 = (t<=end).nonzero()[0].max() + 1
        elif end < t[t0]:
            return slice(0, 0, None)
    if DEBUG and t1 < t0:
        sys.stderr.write('time_slice: slice ends before start (%s < %s)\n'%
            (str(t1),str(t0)))
    return slice(t0, t1, None)

def stamp_to_time(ts, zero_stamp=None):
    """Convert a timestamp array to a time array

    Arguments:
    ts -- timestamp array to be converted
    zero_stamp -- timestamp defining t=0s for this interval (default to first);
        can be set to 'middle' or any timestamp value

    Returns elapsed time array.
    """
    ts = np.asarray(ts, dtype='i8')
    if zero_stamp is None:
        if ts.size == 0:
            return np.array([], 'd')
        t0 = ts[0]
    else:
        t0 = long(zero_stamp)
    return (ts - t0) / STAMP_RATE

def elapsed(start, end):
    """Get elapsed time in seconds between two timestamps
    """
    return float(end - start) / STAMP_RATE

