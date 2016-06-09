# encoding: utf-8
"""circstat.py -- module for circular statistics functions

   Currently implmented functions:
     -- mean, std, var

   Note: all functions take an array of radian-angle values
   on [0, 2*pi] as input.

   Written by Joe Monaco, 4/17/2007
   Center for Theoretical Neuroscience
"""

def mean(theta, w=None):
    """first circular moment

       Input: theta - array of radian angle values
          w - optional weighting if angle values are binned
       Returns: scalar circular mean of theta

       See: http://en.wikipedia.org/wiki/Directional_statistics
    """
    from numpy import ones, dot, sin, cos
    from ..tools.radians import xy_to_rad
    sz = theta.shape[0]
    if w is None:
        w = ones(sz, 'd')
    elif w.size != sz:
        raise ValueError, 'weight array size mismatch'
    s_bar = dot(w, sin(theta))
    c_bar = dot(w, cos(theta))
    return xy_to_rad(c_bar, s_bar)

def std(theta):
    """sample circular deviation

       Input: theta - array of radian angle values
       Returns: circular standard deviation
    """
    return (var(theta))**0.5

def var(theta, Nbins=360):
    """sample circular variance, second moment

       Calculated using the minimum variance method with moving cut points.
       See: Weber RO (1997). J. Appl. Meteorol. 36(10), 1403-1415.

       Input: theta - array of radian angle values
          numbins - number of intervals across [0, 2pi] to minimize
       Returns: circular variance
    """
    from scipy.stats import histogram
    from numpy import empty, arange, pi

    N = len(theta)
    delta_t = 2 * pi / Nbins
    lims = (0, 2 * pi)
    x = arange(delta_t, 2*pi + delta_t, delta_t)
    n, xmin, w, extra = histogram(theta, numbins=Nbins, defaultlimits=lims)

    tbar = empty((Nbins,), 'd')
    S = empty((Nbins,), 'd')
    s2 = empty((Nbins,), 'd')

    tbar[0] = (x*n).sum() / N                                               # A1
    S[0] = ((x**2)*n).sum() / (N - 1)                                       # A2
    s2[0] = S[0] - N * (tbar[0]**2) / (N - 1)                               # A3

    for k in xrange(1, Nbins):
        tbar[k] = tbar[k-1] + (2*pi) * n[k-1] / N                           # A4
        S[k] = S[k-1] + (2*pi) * (2*pi + 2*x[k-1]) * n[k-1] / (N - 1)   # A5
        s2[k] = S[k] - N * (tbar[k]**2) / (N - 1)                           # A6

    return s2.min()
