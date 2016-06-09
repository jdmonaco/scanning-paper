# encoding: utf-8
"""
filters.py -- A collection of custom signal filtering functions.

Created by Joe Monaco on 2007-11-15. Updated 2009-09-11.
Copyright (c) 2007 Columbia University. All rights reserved.
Copyright (c) 2009 Johns Hopkins University. All rights reserved.
"""

import numpy as np
import scipy.signal


def find_minima(s, wrapped=False):
    """Index array of the local minima of a continuous signal
    """
    return _extrema(s, lambda x: x == +1, wrapped)
    
def find_maxima(s, wrapped=False):
    """Index array of the local maxima of a continuous signal
    """
    return _extrema(s, lambda x: x == -1, wrapped)
    
def find_peaks(s, wrapped=False):
    """Index array of local extrema of a continuous signal
    """
    return _extrema(s, lambda x: x != 0, wrapped)
    
def _extrema(s, which, wrapped):
    if wrapped:
        s = np.r_[s[-1], s, s[0]]

    ex = np.r_[0, np.diff((np.diff(s) >= 0).astype('i')), 0]

    if wrapped:
        ex = ex[1:-1]

    return np.nonzero(which(ex))[0]


def smart_medfilt2d(M, base=20, xwrap=False, ywrap=False):
    """Median filter the given matrix based on its rank size and optionally
    wrapping the filter around the x or y dimension
    """
    kernel = 2*int(np.sqrt(M.shape[0]*M.shape[1])/base)+1
    if kernel <= 1:
        return M
    
    if xwrap:
        M = np.c_[M[:,-kernel:], M, M[:,:kernel]]
    if ywrap:
        M = np.r_[M[-kernel:], M, M[:kernel]]
    
    M = scipy.signal.medfilt2d(M, kernel_size=kernel)
    
    if xwrap:
        M = M[:,kernel:-kernel]
    if ywrap:
        M = M[kernel:-kernel]
    
    return M
    

def filtfilt(b, a, s):
    """Forward-backward filter: linear filtering that preserves phase
     
    Modified from: http://www.scipy.org/Cookbook/FiltFilt
    """
    from numpy import r_, flipud, zeros
    
    if type(a) is type(0):
        len_a = 1
    else:
        len_a = len(a)
    ntaps = max(len_a, len(b))
    wrap = 3 * ntaps

    if s.ndim != 1:
        raise ValueError, "filtfilt: requires a 1D signal vector"

    # x must be bigger than edge
    if s.size < wrap:
        raise ValueError, "filtfilt: signal not big enough for filter"

    # pad b coefficients if necessary
    if len_a > len(b):
        b = r_[b, zeros(len_a - len(b))]
    elif len_a < len(b):
        a = 1

    # reflect-wrap the signal for filter stability
    s = r_[2*s[0] - s[wrap:0:-1], s, 2*s[-1] - s[-1:-wrap-1:-1]]

    # filter forward, filter backward
    y = scipy.signal.lfilter(b, a, s, -1)
    y = scipy.signal.lfilter(b, a, flipud(y), -1)

    return flipud(y[wrap:-wrap])
    
def quick_boxcar(s, M=4, centered=True):
    """Returns a boxcar-filtered version of the input signal
    
    Keyword arguments:
    M -- number of averaged samples (default 4)
    centered -- recenter the filtered signal to reduce lag (default False)
    """
    # Sanity check on signal and filter window
    length = s.shape[0]
    if length <= 2*M:
        raise ValueError, 'signal too short for specified filter window'
    
    # Set up staggered arrays for vectorized average
    z = np.empty((M, length+M-1), 'd')
    for i in xrange(M):
        z[i] = np.r_[np.zeros(i)+s[0], s, np.zeros(M-i-1)+s[-1]]
    
    # Center the average if specified
    start_ix = 0
    end_ix = length
    if centered:
        start_ix += int(M/2)
        end_ix += int(M/2)
    
    return z.mean(axis=0)[start_ix:end_ix]
    
def circular_blur(s, blur_width):
    """Return a wrapped gaussian smoothed (blur_width in degrees) signal for 
    data binned on a full circle range [0, 2PI/360).
    """
    bins = s.shape[0]
    width = blur_width / (360.0/bins)
    size = np.ceil(8*width)
    if size > bins:
        size = bins
    wrapped = np.r_[s[-size:], s, s[:size]]
    G = scipy.signal.gaussian(size, width)
    G /= np.trapz(G)
    S = scipy.signal.convolve(wrapped, G, mode='same')
    return S[size+1:-size+1]

def unwrapped_blur(s, blur_width, bins_per_cycle):
    """Return a gaussian smoothed (blur_width in degrees) signal for 
    unwrapped angle data across multiple cycles.
    """
    width = blur_width / (360.0/bins_per_cycle)
    size = np.ceil(8*width)
    G = scipy.signal.gaussian(size, width)
    G /= np.trapz(G)
    S = scipy.signal.convolve(s, G, mode='same')
    return S
