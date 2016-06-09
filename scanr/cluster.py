#encoding: utf-8
"""
scanr.cluster -- Module for cluster data manipulation and filtering

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import sys
from traits.api import HasTraits, Int, Float, String, Array, Instance

# Package imports
from .config import Config
from .tools.enum import Enum

# Constants
MIN_I_POS = Config['placefield']['min_positional_info']
MIN_I_SPIKE = Config['placefield']['min_skaggs_info']
MIN_SPIKE_COUNT = Config['placefield']['min_spike_count']
PLACE_SIGLEVEL = Config['placefield']['info_significance']

# Cluster quality ratings enumeration
ClusterQuality = Enum("None", "Poor", "Marginal", "Fair", "PrettyGood", "Good")


def string_to_quality(rating):
    if type(rating) is type(ClusterQuality.None):
        return rating
    try:
        rating = str(rating)
    except TypeError:
        sys.stderr.write("string_to_quality: requires string argument\n")
        raise
    q = ClusterQuality.None
    if rating != "":
        if "pretty" in rating.lower():
            rating = 'Pretty Good'
        try:
            q = getattr(ClusterQuality, rating.title().replace(" ", ""))
        except AttributeError:
            sys.stderr.write('string_to_quality: bad rating: \"%s\"\n'%rating)
    return q


class ClusterData(HasTraits):

    """
    Session data for a single cluster
    """

    # Automatically calculated
    N = Int(desc='number of spikes')
    mean_firing_rate = Float
    name = String

    # These should be set on instantiation
    tt = Int(desc='tetrode')
    cl = Int(desc='cluster')
    spikes = Array(desc='timestamp spike train')
    x = Array
    y = Array
    T = Float(desc='total session duration')
    spike_width = Float
    quality = Instance(ClusterQuality.None, ClusterQuality.None)
    comment = String
    N_running = Int(desc='total forward-running spikes')
    I_pos = Float
    I_spike = Float
    pos_p_value = Float
    spike_p_value = Float

    # Traits defaults

    def _N_default(self):
        return self.spikes.size

    def _mean_firing_rate_default(self):
        return self.N / self.T

    def _name_default(self):
        return 't%dc%d'%(self.tt, self.cl)


class ClusterCriteria(list):

    """
    A collection of customizable filters for selecting clusters
    """

    def __init__(self, *args, **kwds):
        list.__init__(self, *args, **kwds)
        for item in self:
            if not callable(item):
                raise ValueError, "only callables can be added"

    def append(self, value):
        """Add a new callable filter to the list of cluster criteria

        Filters must receive a ClusterData object as the only argument and
        return True or False.
        """
        assert callable(value), "only callables can be cluster criteria"
        list.append(self, value)

    def filter(self, cluster):
        """Test the given ClusterData object against the current set of cluster
        criterion filters. Return True or False to indicate pass or fail.
        """
        assert type(cluster) is ClusterData, "requires ClusterData object"
        for test in self:
            if not test(cluster):
                return False
        return True


# Operations on clustser criteria

def AND(*args):
    """Return union of all criteria passed in arguments
    """
    union = ClusterCriteria()
    for criteria in args:
        assert type(criteria) is ClusterCriteria
        for item in criteria:
            if item not in union:
                union.append(item)
    return union

def NOT(criteria):
    """Return negative of given criteria
    """
    return ClusterCriteria([lambda c: not criteria.filter(c)])


# Cluster criteria definitions

principal_cell_line = \
    lambda cluster: 20*cluster.spike_width - 0.6*cluster.mean_firing_rate + 1
comment_contains = \
    lambda cluster, token: token.lower() in cluster.comment.lower()

PrincipalCellCriteria = ClusterCriteria([
    lambda cluster: principal_cell_line(cluster) >= 0,
    lambda cluster: not comment_contains(cluster, 'interneuron')
    ])
InterneuronCriteria = NOT(PrincipalCellCriteria)

OlypherCriteria = ClusterCriteria([
    lambda cluster: cluster.I_pos >= MIN_I_POS,
    lambda cluster: cluster.pos_p_value <= PLACE_SIGLEVEL
    ])
SkaggsCriteria = ClusterCriteria([
    lambda cluster: cluster.I_spike >= MIN_I_SPIKE,
    lambda cluster: cluster.spike_p_value <= PLACE_SIGLEVEL
    ])
SpatialInformationCriteria = ClusterCriteria([
    lambda cluster: OlypherCriteria.filter(cluster) or SkaggsCriteria.filter(cluster)
    ])

SpikeCountCriteria = ClusterCriteria([
    lambda cluster: cluster.N_running >= MIN_SPIKE_COUNT
    ])
PlaceCellCriteria = AND(
    PrincipalCellCriteria, SpikeCountCriteria, SpatialInformationCriteria)


# Cluster criteria factory functions

def get_min_quality_criterion(min_quality):
    """Get a ClusterCriteria object that enforces a minimum cluster quality
    """
    if type(min_quality) is str:
        min_quality = string_to_quality(min_quality)
    return ClusterCriteria([lambda cluster: cluster.quality >= min_quality])

def get_tetrode_restriction_criterion(tt_list):
    """Get a ClusterCriteria object that restricts clusters to a certain set of
    valid tetrodes
    """
    if type(tt_list) is int:
        tt_list = [tt_list]
    return ClusterCriteria([lambda cluster: cluster.tt in tt_list])
