# encoding: utf-8
"""
reports.py -- Base classes for various styles of report generators

Created by Joe Monaco on 2011-04-12.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os, sys
import pylab as plt
import numpy as np
from numpy import inf
from traits.api import List, Dict

# Package imports
from scanr.lib import Config
from scanr.tracking import find_laps, get_tracking
from scanr.meta import walk_days, get_day_list, get_start_end
from scanr.data import session_list, get_node
from .core.report import BaseReport
from scanr.tools.path import unique_path

# Costants
DEBUG = Config['debug_mode']


class BaseDatasetReport(BaseReport):

    """
    Base class for printing out a series of PDF reports of plots for each
    experiment day (i.e., dataset) in a specified list.

    Defines the get_plot() generator method which subclasses should use to
    create reports based on per-session plots.
    """

    label = 'dataset report'

    dataset_list = List
    ncols = 4

    # Session plot generator

    def get_plot(self, datasets=None):
        """Generator method that walks the data tree and returns a tuple
        containing a (rat, day) tuple and an axes handle.

        Keyword arguments:
        datasets -- specify datasets to report as a list of rats and/or
            (rat, day) tuples; default to entire dataset
        """
        self._start_report()
        self._set_dataset_list(datasets)
        N = len(self.dataset_list)

        for i, ratday in enumerate(self.dataset_list):
            if DEBUG:
                self.out('Reporting rat %d, day %d...'%ratday)
            if i == N - 1:
                self.lastpanel = True

            ax = self._create_new_axes()
            yield (ratday, ax)

            ax.set_title('rat%d-%02d'%ratday, size='small')
            self._advance(ax)
        self._finish_chunk()
        self._finish_report()

    # Support methods

    def _set_dataset_list(self, datasets):
        if datasets is None:
            self.dataset_list = list(walk_days())
        elif np.iterable(datasets):
            for item in datasets:
                if type(item) is tuple and len(item) == 2:
                    self.dataset_list.append(item)
                elif type(item) is int:
                    self.dataset_list += \
                        [(item, day) for day in get_day_list(item)]
        else:
            raise ValueError, "bad dataset specification: %s"%datasets


class BaseSessionReport(BaseDatasetReport):

    """
    Base class for printing out a series of PDF reports of plots for each
    experimental session in a specified list of data sets.

    Defines the get_plot() generator method which subclasses should use to
    create reports based on per-session plots.
    """

    label = 'session report'
    ncols = 5
    dataset_filter = Dict

    # Session plot generator

    def _start_report(self, datasets, **kwds):
        """Add dataset and session filters to report start
        """
        BaseDatasetReport._start_report(self)
        self._set_dataset_list(datasets)
        self._set_dataset_filter(**kwds)

    def get_plot(self, datasets=None, **kwds):
        """Generator method that walks the data tree and returns a tuple
        containing an rds triplet and an axes handle.

        Keyword arguments:
        datasets -- specify datasets to report as a list of rats and/or
            (rat, day) tuples; default to entire dataset
        missing_hd -- whether to include sessions with missing HD data
        timing_issue -- whether to include sessions with Neuralynx timing issue
        """
        self._start_report(datasets, **kwds)

        for rat, day in self.dataset_list:
            if DEBUG:
                self.out('Reporting rat %d, day %d...'%(rat, day))
            for maze in session_list(rat, day, **self.dataset_filter):
                ax = self._create_new_axes()

                rds = (rat, day, maze)
                yield (rds, ax)

                if self.firstcol:
                    ax.set_title(self._get_title(rds), size='small')
                self._advance(ax)
            self._finish_chunk()
        self._finish_report()

    # Support methods

    def _get_title(self, rds):
        return 'rat%d-%02d'%(rds[0], rds[1])

    def _set_dataset_filter(self, missing_HD=False, timing_issue=False):
        sf = dict(  exclude_missing_HD=(not missing_HD),
                    exclude_timing_issue=(not timing_issue))
        self.out('Session filters:\n' + '\n'.join(
            ['%s = %s'%(k, sf[k]) for k in sf]))
        self.dataset_filter = sf


class BaseLapReport(BaseSessionReport):

    """
    Base class for printing out a series of PDF reports of plots for each
    lap of an experimental session.

    Defines the get_plot() generator method which subclasses should use to
    create per-lap axis plots.
    """

    label = 'lap report'
    ncols = 5

    def get_plot(self, datasets=None, **kwds):
        """Generator method that walks the data tree and returns a tuple
        containing an rds triplet, lap bounds, and an axes handle.

        Keyword arguments:
        datasets -- specify datasets to report as a list of rats and/or
            (rat, day) tuples; default to entire dataset
        missing_hd -- whether to include sessions with missing HD data
        timing_issue -- whether to include sessions with Neuralynx timing issue
        """
        self._start_report(datasets, **kwds)

        for rat, day in self.dataset_list:
            for maze in session_list(rat, day, **self.dataset_filter):

                # Get lap partitions
                rds = rat, day, maze
                ts, x, y, hd = get_tracking(*rds)
                start, end = get_start_end(*rds)
                laps = find_laps(ts, x, y, start=start)
                N_laps = len(laps)
                laps += [end]

                self.out('Reporting rat %d, day %d, maze %d: '%rds)
                for lap in xrange(N_laps):
                    if lap > 14:
                        continue

                    self.out.printf('.')
                    ax = self._create_new_axes()

                    yield (rds, (laps[lap], laps[lap+1]), ax)

                    if self.firstcol:
                        ax.set_title('rat%d-%02d m%d lap%d'%(rds+(lap+1,)),
                            size='small')
                    self._advance(ax)
                self.out.printf('\n')
                self._finish_chunk()
        self._finish_report()


class BaseClusterReport(BaseReport):

    label = "cluster report"

    ynorm = False
    nrows = 5
    ncols = 3

    def get_plot(self, session, cluster_criteria=None):
        """Plot generator for showing plots of cluster data

        Yields (tc_string, cluster_data, ax) tuple.
        """
        self._start_report()
        clusters = session.get_clusters(cluster_criteria)
        N = len(clusters)
        for i, tc in enumerate(clusters):
            if i == N - 1:
                self.lastpanel = True
            ax = self._create_new_axes()
            yield (tc, session.clusts[tc], ax)
            ax.set_title(tc, size='small')
            self._advance(ax)
        self._finish_chunk()
        self._finish_report()


class BaseLapClusterReport(BaseReport):

    label = "lap cluster report"

    figwidth = 8.5
    figheight = 8.5
    nrows = 4
    ncols = 4

    def get_plot(self, session, cluster_criteria=None):
        """Plot generator for showing per-lap plots of cluster data

        Yields (tc_string, cluster_data, (lap_start, lap_end), ax) tuple.
        """
        self._start_report()
        self.out('Reporting rat %d, day %d, maze %d...'%session.rds)

        for tc in session.get_clusters(cluster_criteria):
            self.f.suptitle('Rat %d, Day %d, Maze %d - Cluster %s'%
                (session.rds+(tc,)))
            self.out('Generating plots for cluster %s'%tc)
            for lap in xrange(session.N_laps):
                if lap > 15:
                    continue
                self.lastpanel = (lap == session.N_laps - 1)
                ax = self._create_new_axes()
                yield (tc, session.clusts[tc],
                    (session.laps[lap], session.laps[lap+1]), ax)
                ax.set_title('lap %d'%(lap+1), size='small')
                self._advance(ax)
            self._finish_chunk()
        self._finish_report()

