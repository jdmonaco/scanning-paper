# encoding: utf-8
"""
scanr.compare -- Functions for comparing double-rotation sessions

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
from scipy.stats import pearsonr

# Package imports
from .session import SessionData
from .tools.filters import circular_blur
from .tools.radians import get_angle_array, circle_diff


# Functions to compare population response matrices

def unwrapped_correlation(U, R):
    """For an unwrapped population response matrix U, compute the population
    vector correlation for every bin with the wrapped population reference
    matrix R. The number of columns in U should be a multiple (i.e., number of
    laps) of the number of columns in R.
    """
    assert U.shape[1] % R.shape[1] == 0, \
        'unwrapped response shape must be a column-wise multiple of reference'
    bins = R.shape[1]
    R = R[:,::-1] # reverse reference so it is CW like unwrapped
    C = np.empty(U.shape[1], 'd')
    for i, PV in enumerate(U.T):
        C[i] = pearsonr(PV, R[:, i % bins])[0]
    C[np.isnan(C)] = 0.0 # handle unsampled bins in U
    return C

def mismatch_response_tally(STD, MIS, mismatch, angle_tol=0.5, min_corr=0.4,
    **kwargs):
    """Categorize and tally spatial response changes in mismatch session

    Both angle tolerance and minimum correlation criteria must be met for a
    response change to classify as coherent.

    Required arguments:
    STD, MIS -- SessionData objects for STD and MIS session pair
    mismatch -- total mismatch angle for the cue rotation (in degrees)

    Keyword arguments:
    angle_tol -- proportional tolerance for matching a cue rotation
    min_corr -- minimum maximal correlation value across rotation

    Returns dictionary of tallies: local, distal, ambiguous, on, off.
    """
    if not hasattr(STD, 'get_criterion_clusters') or \
        type(MIS) is not type(STD):
        raise ValueError, 'invalid or non-matching session data inputs'

    # Count coherent vs non-coherent response changes
    tally = dict(local=0, distal=0, amb=0)
    rotations = mismatch_rotation(STD, MIS, degrees=False, **kwargs)
    mismatch *= (np.pi/180) / 2 # cue rotation in radians
    for rotcorr in rotations.T:
        rot, corr = rotcorr
        if corr < min_corr:
            tally['amb'] += 1
        else:
            angle_dev = min(
                abs(circle_diff(rot, mismatch)),
                abs(circle_diff(rot, -mismatch)))
            if angle_dev <= angle_tol * mismatch:
                if rot < np.pi:
                    tally['local'] += 1
                else:
                    tally['distal'] += 1
            else:
                tally['amb'] += 1

    # Count remapping (on/off) response changes
    STDclusts = set(STD.get_criterion_clusters())
    MISclusts = set(MIS.get_criterion_clusters())
    tally['off'] = len(STDclusts.difference(MISclusts))
    tally['on'] = len(MISclusts.difference(STDclusts))

    return tally

def mismatch_rotation(STD, MIS, degrees=True, **kwargs):
    """Computes rotation angle and peak correlation for active clusters between
    two sessions

    Returns two-row per-cluster array: angle, correlation.
    """
    Rstd, Rmis = comparison_matrices(STD, MIS, **kwargs)
    units, bins = Rstd.shape
    return np.array([cluster_mismatch_rotation(Rstd[c], Rmis[c], degrees=degrees)
        for c in xrange(units)]).T

def cluster_mismatch_rotation(Rstd, Rmis, degrees=True):
    """Find the rotation angle for a single cluster ratemap

    Ratemap inputs must be one-dimensional arrays representing the whole track.

    Keyword arguments:
    degrees -- whether to specify angle in degrees or radians

    Returns (angle, correlation) tuple.
    """
    bins = Rstd.shape[0]
    angle = get_angle_array(bins, degrees=degrees)
    corr = np.empty(bins, 'd')
    for offset in xrange(bins):
        MISrot = np.concatenate((Rmis[offset:], Rmis[:offset]))
        corr[offset] = pearsonr(Rstd, MISrot)[0]
    return angle[np.argmax(corr)], corr.max()

def population_spatial_correlation(STD, MIS, **kwargs):
    """Return whole population spatial correlation between two matrices
    """
    A, B = comparison_matrices(STD, MIS, **kwargs)
    return pearsonr(A.flatten(), B.flatten())[0]


# Functions to compute and operate on correlation matrices

def correlation_matrix(SD, cross=None, **kwargs):
    """Compute a spatial correlation matrix of population-rate vectors

    Returns (bins, bins) correlation matrix.
    """
    # Validate arguments and compute population response matrices
    R, R_ = comparison_matrices(SD, cross, **kwargs)

    # Compute the correlation matrix
    N_units, bins = R.shape
    C = np.empty((bins, bins), 'd')
    for i in xrange(bins):
        R_i = R[:,i] / np.sqrt(np.dot(R[:,i], R[:,i]))
        for j in xrange(bins):
            R_j = R_[:,j] / np.sqrt(np.dot(R_[:,j], R_[:,j]))
            C[i,j] = np.dot(R_i, R_j)

    # Fix any NaN's resulting from silent population responses (rare!)
    C[np.isnan(C)] = 0.0
    return C

def lap_correlation_matrix(PLM, cluster=None):
    """Lap-by-lap spatial correlation matrix for population or single cluster

    PLM must be a matrix as returned by the get_population_lap_matrix method.

    Keyword arguments:
    cluster -- the row index of a single cluster represented in the PLM; if not
        specified, population lap correlation are computed by default

    Returns a (N_laps, N_laps) lap correlation matrix.
    """
    N_clusts, bins, N_laps = PLM.shape
    C = np.empty((N_laps, N_laps), 'd')
    R = PLM[cluster]
    for i in xrange(N_laps):
        R_i = R[...,i].flatten()
        R_i /= np.sqrt(np.dot(R_i, R_i))
        for j in xrange(i, N_laps):
            R_j = R[...,j].flatten()
            R_j /= np.sqrt(np.dot(R_j, R_j))
            C[i,j] = C[j,i] = np.dot(R_i, R_j)

    # Fix any NaN's resulting from no spikes on certain laps
    C[np.isnan(C)] = 0.0
    return C

def correlation_diagonals(C, use_median=True, centered=False, blur=None):
    """Return the angle bins and diagonals of a correlation matrix

    Keyword arguments:
    use_median -- whether to use the median diagonal correlation to collapse
        the diagonals; if use_median=False, the average is used
    centered -- whether to center the diagonals on [-180, 180]
    blur -- if not None, specifies width in degrees of gaussian blur to be
        applied to diagonal array
    """
    bins = C.shape[0]
    if C.shape != (bins, bins):
        raise ValueError, 'correlation matrix must be square'
    f = use_median and np.median or np.mean
    D = np.empty(bins+1, 'd')
    d = np.empty(bins, 'd')
    offset = 0
    if centered:
        offset = int(bins/2)

    # Loop through and collapse correlation diagonals
    for b0 in xrange(bins):
        for b1 in xrange(bins):
            d[b1] = C[b1, np.fmod(offset+b0+b1, bins)]
        D[b0] = f(d)
    if blur is not None:
        D[:bins] = circular_blur(D[:bins], blur)

    # Wrap the last point around to the beginning
    D[-1] = D[0]
    last = centered and 180 or 360
    a = np.r_[get_angle_array(bins, degrees=True, zero_center=centered), last]
    return np.array([a, D])


# Functions on lists of SessionData objects (e.g., full five-session double-
# rotation experiments)

def comparison_matrices(SD, cross, **kwargs):
    """Validate multiple types of arguments for use as comparanda

    SD and cross must be the same type of object (unless cross is None for an
    autocomparison): SessionData instances or previously computed population
    matrices.

    For SessionData objects, a clusters list is automatically created to be
    passed in for the get_population_matrix call. This may be overriden by
    passing in your own clusters list as a keyword argument. Additional keyword
    arguments are passed to get_population_matrix.

    Returns two valid (units, bins) population matrix references.
    """
    if type(SD) is np.ndarray:
        if SD.ndim != 2:
            raise ValueError, 'expecting 2-dim population matrix'
        R = R_ = SD
        if type(cross) is np.ndarray:
            if cross.shape == SD.shape:
                R_ = cross
            else:
                raise ValueError, 'non-matching population matrices'
    elif hasattr(SD, 'get_criterion_clusters'):
        kwargs['norm'] = False
        if type(cross) is type(SD):
            if 'clusters' not in kwargs:
                kwargs['clusters'] = common_clusters(SD, cross)
        elif cross is not None:
            raise ValueError, 'non-matching session object types'
        R = R_ = SD.get_population_matrix(**kwargs)
        if cross is not None:
            R_ = cross.get_population_matrix(**kwargs)
            if R_.shape != R.shape:
                raise ValueError, 'population matrix size mismatch'
    return R, R_

def common_clusters(*SD_list):
    """Get a list of the criterion clusters are common to a set of sessions
    """
    # Allow a single python list to be passed in
    if len(SD_list) == 1 and type(SD_list[0]) is list:
        SD_list = SD_list[0]

    # Get a list of sets of clusters for each data object
    clust_list = []
    for SD in SD_list:
        clust_list.append(set(SD.get_criterion_clusters()))

    # Find the common clusters
    common = clust_list[0]
    for i in xrange(1, len(SD_list)):
        common = common.intersection(clust_list[i])

    return list(common)
