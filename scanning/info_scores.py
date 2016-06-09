#encoding: utf-8

"""
info_scores.py -- Collect pre-computed spatial information scores for units
    for comparison with place-cell criteria testing

Written by Joe Monaco, March 27, 2012.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
import operator as op

from scanr.lib import Config
from scanr.meta import get_maze_list
from scanr.session import SessionData
from scanr.data import get_data_file_path, close_file, unique_cells, get_unique_row
from scanr.cluster import (SpatialInformationCriteria, SkaggsCriteria, OlypherCriteria,
    SpikeCountCriteria, SpatialInformationCriteria, PrincipalCellCriteria,
    get_min_quality_criterion, get_tetrode_restriction_criterion, AND, NOT)
from scanr.spike import TetrodeSelect

from .tools.bash import CPrint, lightgreen
from .tools.plot import heatmap, quicktitle, AxesList
from .tools.misc import outer_pairs


class InfoScoreData(TetrodeSelect):

    """
    Component spatial information scores for analyzing place field criteria
    """

    def run(self, test='place', place_field='pass', min_quality='fair', **kwds):
        """Compute I_pos and I_spike across all criterion place cells in CA3/CA1

        Keyword arguments:
        place_field -- 'pass', 'fail', or 'all' to restrict responses based on place
            field criterion test results
        test -- 'place', 'skaggs', or 'olypher' to use either the full place field test or
            one of the component tests for the cell filtering
        min_quality -- isolation quality threshold for filtering cells

        Remaining keywords are passed to TetrodeSelect.

        Returns (I_pos, I_spike) tuple of arrays for selected cell clusters.
        """
        self.out = CPrint(prefix='ScatterInfo')
        area_query = '(area=="CA3")|(area=="CA1")'

        # Metadata for the plot title
        self.place_field = place_field
        self.test = test
        self.quality = min_quality
        if place_field == 'all':
            self.test = 'place'

        if test == 'place':
            SpatialTest = SpatialInformationCriteria
        elif test == 'skaggs':
            SpatialTest = SkaggsCriteria
        elif test == 'olypher':
            SpatialTest = OlypherCriteria
        else:
            raise ValueError, 'bad test value: %s'%test

        MinQuality = get_min_quality_criterion(min_quality)
        CellCriteria = AND(PrincipalCellCriteria, SpikeCountCriteria, MinQuality)
        if place_field == 'pass':
            CellCriteria = AND(CellCriteria, SpatialTest)
        elif place_field == 'fail':
            CellCriteria = AND(CellCriteria, NOT(SpatialTest))
        elif place_field != 'all':
            raise ValueError, 'bad place_field value: %s'%place_field

        I = []
        for dataset in TetrodeSelect.datasets(area_query):
            rat, day = dataset
            Criteria = AND(CellCriteria,
                TetrodeSelect.criterion(dataset, area_query, **kwds))

            for maze in get_maze_list(*dataset):
                data = SessionData.get((rat, day, maze))

                for tc in data.get_clusters(request=Criteria):
                    cluster = data.cluster_data(tc)
                    I.append((cluster.I_pos, cluster.I_spike))

        self.I = I = np.array(I).T
        self.out('%d cell-sessions counted.'%I.shape[1])
        return I[0], I[1]

    def plot(self, ax=None):
        """Plot the last run of info scores
        """
        I_pos, I_spike = self.I
        if ax is None:
            f = plt.figure()
            ax = plt.gca()
        heatmap(I_pos, I_spike, ax=ax, bins=32, range=[[0.1, 3], [0.1, 10]])

        ax.set_title('[%s] %s Fields (N = %d, Q > %s) - Olypher Peak vs. Skaggs'%(
            self.place_field.title(), self.test.title(), I_pos.size,
            self.quality.title()))
        ax.set_xlabel('max(I_pos), bits')
        ax.set_ylabel('I_spike, bits/spike')
        ax.axhline(Config['placefield']['min_skaggs_info'], c='k', ls='--')
        ax.axvline(Config['placefield']['min_positional_info'], c='k', ls='--')
        plt.draw()


def info_scores_plot(**kwargs):
    """Walk the cluster nodes of the data tree to get scatter data of Olypher
    and Skaggs information scores for all cells that meet the place field
    criteria and plot the results.

    Keyword arguments are passed to InfoScoreData.run().

    Saves I_pos.npy and I_spike.npy array files.
    """
    scores = InfoScoreData()
    I_pos, I_spike = scores.run(**kwargs)
    np.save("I_pos", I_pos)
    np.save("I_spike", I_spike)
    sys.stdout.write('Wrote I_pos.npy and I_spike.npy.\n')
    scores.plot()


class CompareInformationScores(object):

    def __init__(self):
        self.I_skaggs1 = self.I_skaggs2 = self.I_olypher1 = self.I_olypher2 = None
        self.kfilepath1 = self.kfilepath2 = ''

    def compare_tables_in_files(self, kfilepath1, kfilepath2=None):
        """If one kdata file path is specified, it is compared to the currently
        active file.
        """
        self.kfilepath1 = kfilepath1
        kfile1 = tb.openFile(kfilepath1, mode='r')
        if kfilepath2 is None:
            kfilepath2 = get_data_file_path('data')
            close_file()
        self.kfilepath2 = kfilepath2
        kfile2 = tb.openFile(kfilepath2, mode='r')

        skaggs1 = kfile1.root.physiology.cell_information_skaggs
        skaggs2 = kfile2.root.physiology.cell_information_skaggs
        olypher1 = kfile1.root.physiology.cell_information_olypher
        olypher2 = kfile2.root.physiology.cell_information_olypher

        cells = unique_cells(skaggs1)
        self.I_skaggs1 = I_skaggs1 = np.empty(len(cells), 'd')
        self.I_skaggs2 = I_skaggs2 = np.empty_like(I_skaggs1)
        self.I_olypher1 = I_olypher1 = np.empty_like(I_skaggs1)
        self.I_olypher2 = I_olypher2 = np.empty_like(I_skaggs1)

        sys.stdout.write('Comparing information for %d cell-sessions.\n' % len(cells))

        def get_I(table, query):
            row = get_unique_row(table, query, raise_on_fail=False)
            if row is None:
                return 0.0
            return row['I']

        for i, cell_id in enumerate(cells):
            rds, tc = cell_id
            cell_query = '(rat==%d)&(day==%d)&(session==%d)&(tc=="%s")' % (rds + (tc,))
            I_skaggs1[i] = get_I(skaggs1, cell_query)
            I_skaggs2[i] = get_I(skaggs2, cell_query)
            I_olypher1[i] = get_I(olypher1, cell_query)
            I_olypher2[i] = get_I(olypher2, cell_query)
            if i % 100 == 0:
                sys.stdout.write(lightgreen('.'))
                sys.stdout.flush()
        sys.stdout.write('\n')

        kfile1.close()
        kfile2.close()

    def plot(self, cmap='jet'):
        assert self.I_skaggs1 is not None, 'please run compare_tables_in_files first'

        f = plt.figure(figsize=(10,10))
        axlist = AxesList()
        axlist.make_grid((4,4))
        f.suptitle('1) %s\n2) %s' % (self.kfilepath1, self.kfilepath2))

        I = (self.I_skaggs1, self.I_skaggs2, self.I_olypher1, self.I_olypher2)
        name = ('sk1', 'sk2', 'oly1', 'oly2')
        I_range = range(len(I))

        D = reduce(op.mul, map(np.isfinite, I))

        for i, j in outer_pairs(I_range, I_range):
            cmp_name = '%s - %s' % (name[j], name[i])
            sys.stdout.write('Plotting %s...\n' % cmp_name)
            sys.stdout.flush()

            ax = axlist[j + len(I) * i]
            heatmap(I[j][D], I[i][D], bins=128, ax=ax, cmap=cmap)
            quicktitle(ax, cmp_name, size='x-small')

        return f

