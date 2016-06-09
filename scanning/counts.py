#encoding: utf-8

"""
counts.py -- Walk the data tree for place-field counts and breakdowns

Written by Joe Monaco, April 11, 2012.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

import sys
import numpy as np

from .core.analysis import AbstractAnalysis
from scanr.meta import get_maze_list
from scanr.data import get_node
from scanr.session import SessionData
from scanr.cluster import (SpatialInformationCriteria, SkaggsCriteria,
    OlypherCriteria, SpikeCountCriteria, SpatialInformationCriteria,
    PrincipalCellCriteria, get_min_quality_criterion,
    get_tetrode_restriction_criterion, AND, NOT)
from scanr.spike import TetrodeSelect
from .tools.bash import CPrint
from .tools.misc import outer_pairs

# Constants
AREAS = { 'CA1':['all', 'proximal', 'intermediate', 'distal'], 'CA3':['all', 'a', 'b', 'c'] }


class CountsByArea(AbstractAnalysis):

    """
    Counts for all place fields broken down by primary area and optional subdivision
    """

    label = 'field counts'

    def collect_data(self, test='place', place_field='pass', min_quality='fair',
        allow_ambiguous=True):
        """Tally place fields across areas

        Keyword arguments similar to info_scores.InfoScoreData. Remaining
        keywords are passed to TetrodeSelect.
        """
        # Metadata for determining valid fields
        self.results['test'] = test
        self.results['place_field'] = place_field
        self.results['min_quality'] = min_quality
        self.results['allow_ambiguous'] = allow_ambiguous
        if place_field == 'all':
            self.test = 'place'

        # Construct place cell selection criteria based on keyword arguments
        if test == 'place':
            SpatialTest = SpatialInformationCriteria
        elif test == 'skaggs':
            SpatialTest = SkaggsCriteria
        elif test == 'olypher':
            SpatialTest = OlypherCriteria
        else:
            raise ValueError, 'bad test value: %s' % test
        MinQuality = get_min_quality_criterion(min_quality)
        CellCriteria = AND(PrincipalCellCriteria, SpikeCountCriteria, MinQuality)
        if place_field == 'pass':
            CellCriteria = AND(CellCriteria, SpatialTest)
        elif place_field == 'fail':
            CellCriteria = AND(CellCriteria, NOT(SpatialTest))
        elif place_field != 'all':
            raise ValueError, 'bad place_field value: %s' % place_field

        # Walk the tree and count place fields
        N = {}
        N_cells = {}
        N_sessions = {}
        sessions = set()
        tetrodes = get_node('/metadata', 'tetrodes')
        for area in AREAS.keys():
            for subdiv in (['all'] + AREAS[area]):
                self.out('Walking datasets for %s %s...' % (area, subdiv))
                key = '%s_%s' % (area, subdiv)
                N[key] = 0
                N_cells[key] = 0
                N_sessions[key] = 0

                area_query = 'area=="%s"' % area
                if subdiv != 'all':
                    area_query = '(%s)&(subdiv=="%s")' % (area_query, subdiv)

                for dataset in TetrodeSelect.datasets(area_query, allow_ambiguous=allow_ambiguous):
                    Criteria = AND(CellCriteria,
                        TetrodeSelect.criterion(dataset, area_query,
                            allow_ambiguous=allow_ambiguous))
                    dataset_cells = set()

                    for maze in get_maze_list(*dataset):
                        rds = dataset + (maze,)
                        data = SessionData.get(rds)
                        sessions.add(rds)
                        place_cell_clusters = data.get_clusters(request=Criteria)
                        N[key] += len(place_cell_clusters)
                        dataset_cells.update(place_cell_clusters)
                        N_sessions[key] += 1

                    N_cells[key] += len(dataset_cells)

        self.out.timestamp = False
        self.results['N'] = N
        self.out('Total number of sessions = %d' % len(sessions))
        for key in sorted(N.keys()):
            self.out('N_cells[%s] = %d cells' % (key, N_cells[key]))
            self.out('N_sessions[%s] = %d sessions' % (key, N_sessions[key]))
            self.out('N_cell_sessions[%s] = %d cell-sessions' % (key, N[key]))

        # Good-bye
        self.out('All done!')
