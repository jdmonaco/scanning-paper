#encoding: utf-8
"""
report.py -- Automatic PDF report generation functionality

Subclasses should implement the get_plot() method generator to provide the
behavior relevant to the data set and plot types that are being reported.

Written by Joe Monaco.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import sys
from numpy import inf
import matplotlib.pylab as plt
from traits.api import (Directory, Bool, Int, Float, List, String,
    Instance)

# Package imports
from .analysis import AbstractAnalysis
from scanr.tools.path import unique_path
from scanr.tools.string import snake2title


class BaseReport(AbstractAnalysis):

    """
    Base analysis subclass for creating a multi-page PDF report of data plots
    for any series of data sets. This class provides a generator method
    get_plot that yields an axes object for plotting as well as necessary
    metadata for retrieving the appropriate data. This method must be
    overridden by subclasses to provide additional report generation behaviors.

    Constructor arguments:
    polar -- create subplots with polar projection instead of linear
    nrows, ncols -- number of rows/columns of plots per page of the report
    xnorm, ynorm -- whether to normalize the x or y axis limits across all plots
        on a given page

    Individual report pages are created in the report/ subdirectory. If the pdftk
    software is installed, will attempt to combine pages into single multi-page
    PDF report stored in analysis directory.
    """

    label = 'report'

    polar = Bool(False, user=True)

    nrows = Int(7, user=True)
    ncols = Int(5, user=True)
    xnorm = Bool(True, user=True)
    ynorm = Bool(True, user=True)

    pageno = Int(1)
    firstrow = Bool(False)
    lasttrow = Bool(False)
    firstcol = Bool(False)
    lastcol = Bool(False)
    lastpanel = Bool(False, desc='must be set by get_plot implementation')
    firstonpage = Bool(False)
    lastonpage = Bool(False)

    figwidth = Float(8.5)
    figheight = Float(11.0)

    # Initialize file names and save directory

    def __init__(self, **traits):
        super(AbstractAnalysis, self).__init__(**traits)
        self.savedir = os.path.join(self.datadir, 'report')
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.save_file_stem = '_'.join(self.label.split())
        self.save_file_list = []
        self.subp_list = []
        self.row = 0
        self.col = 0
        self.panel = 0
        self._reset_lims()

    # Generator definition: override this for different behaviors

    def get_plot(self, data_list):
        """Plot generator that yields tuples consisting of an item from the
        given list of metadata and the corresponding axes handle.

        This allows for simple reports of list of data plots. Report generation
        subclasses can override this method to provide other behaviors. All
        commented calls here must be preset in subclass implementations.
        """
        self._start_report() # initialize the report
        N = len(data_list)
        for i, item in enumerate(data_list): # iterate through provided metadata
            self.lastpanel = i == N - 1 # set last-panel flag
            ax = self._create_new_axes() # create the new axis to plot into
            yield (item, ax) # yield metadata and axis handle
            self._advance(ax) # advance to next plot in chunk
        self._finish_chunk() # finish up a related chunk of plots
        self._finish_report() # finish up the report

    # Plotting, figure, and document creation: support functions

    def _start_report(self):
        self.was_interactive = plt.isinteractive()
        plt.ioff()
        self.f = plt.figure(figsize=(self.figwidth, self.figheight))
        self.pageno = 1

    def _create_new_axes(self, polar=None):
        self.panel = self.ncols*self.row + self.col + 1
        if polar is None:
            do_polar = self.polar
        else:
            do_polar = polar
        ax = plt.subplot(self.nrows, self.ncols, self.panel, polar=do_polar)
        self.subp_list.append((self.nrows, self.ncols, self.panel))
        self.firstrow = (self.row == 0)
        self.lastrow = (self.row == self.nrows-1)
        self.firstcol = (self.col == 0)
        self.lastcol = (self.col == self.ncols-1)
        self.firstonpage = self.firstcol and self.firstrow
        self.lastonpage = (self.lastcol and self.lastrow) or self.lastpanel
        return ax

    def _advance(self, ax):
        self._set_lims(ax)
        self._advance_panel()
        self._check_page_and_clear()

    def _finish_chunk(self):
        if self.col:
            self.row += 1
            self._check_page_and_clear()
        self.col = 0

    def _finish_report(self):
        self._do_last_page()
        self.results['report_pages'] = list(self.save_file_list)
        self._combine_pages()
        self._close_figure()

    # Helper methods for support functions

    def _advance_panel(self):
        self.col += 1
        if self.col > self.ncols - 1:
            self.col = 0
            self.row += 1

    def _set_lims(self, ax):
        if self.polar:
            return
        axlim = ax.axis()
        if self.xnorm:
            self.xlim[0] = min(axlim[0], self.xlim[0])
            self.xlim[1] = max(axlim[1], self.xlim[1])
        if self.ynorm:
            self.ylim[0] = min(axlim[2], self.ylim[0])
            self.ylim[1] = max(axlim[3], self.ylim[1])

    def _reset_lims(self):
        self.xlim = [inf, -inf]
        self.ylim = [inf, -inf]

    def _check_page_and_clear(self):
        if self.row > self.nrows - 1:
            self._normalize_axes()
            self._dump_page()
            self.row = self.col = 0
            self._reset_lims()
            self.subp_list = []

    def _dump_page(self):
        savefile = os.path.join(self.savedir,
            '%s_%03d.pdf'%(self.save_file_stem, self.pageno))
        figtitle = snake2title(self.save_file_stem)
        if self.desc:
            figtitle += ' - %s'%self.desc.title()
        self.f.suptitle(figtitle)
        self.f.text(0.5, 0.04, 'Page %d'%self.pageno, size='small', ha='center')
        plt.savefig(savefile)
        self.out('Saved file:\n\t%s'%savefile)
        self.save_file_list.append(savefile)
        self.pageno += 1
        plt.clf()

    def _normalize_axes(self):
        for sp in self.subp_list:
            ax = plt.subplot(*sp)
            if self.xnorm:
                ax.set_xlim(self.xlim)
            if self.ynorm:
                ax.set_ylim(self.ylim)

    def _do_last_page(self):
        if self.subp_list:
            self._normalize_axes()
            self._dump_page()

    def _close_figure(self):
        plt.close(self.f)
        if self.was_interactive:
            plt.ion()

    def _combine_pages(self):
        if sys.platform == 'win32':
            return
        if os.system('pdftk > /dev/null') != 0:
            self.out('Install pdftk to allow automatic report merging!')
            return
        self.out('Merging %d report files into complete report...'%len(
            self.save_file_list))
        merged_fn = os.path.join(self.datadir, self.save_file_stem + '.pdf')
        cmd_str = 'pdftk %s cat output %s'%(' '.join(self.save_file_list),
            merged_fn)
        cmd_str = cmd_str.strip()
        if os.system(cmd_str) == 0:
            self.out('Report pages successfully combined:\n%s'%merged_fn)
            self.results['report'] = str(merged_fn)
        else:
            self.out('Failed to combine report pages!', error=True)

    # Convenience functions

    def open(self):
        if 'report' in self.results:
            os.system('open %s'%self.results['report'])
    def process_data(self):
        self.open()
