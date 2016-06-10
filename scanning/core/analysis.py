#encoding: utf-8
"""
analysis.py -- Parallelized data collection and figure creation

AbstractAnalysis provides general functionality for running many 'embarrassingly
parallel' long-running calls in an IPython.kernel ipcluster in order to collect
data (e.g., running model simulations for various parameter sets and comparing
results across parameters).

Subclasses provide all relevant data collection and graphics code. Figures
can be created as either Matplotlib figures or Chaco containers.

Written by Joe Monaco, 03/30/2008.
Updated to use IPython.kernel, 01/26/2010.
Copyright (c) 2008-2009 Columbia University. All rights reserved.
Copyright (c) 2010-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import pdb
import sys
import subprocess
import os
import tables as tb
import cPickle
from numpy import ndarray
from time import sleep
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from functools import wraps

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # IPython imports
    from IPython.parallel import Client as IPclient
    from IPython.parallel import LoadBalancedView as IPtc
    from IPython.parallel import DirectView as IPmec

    # Enthought imports
    # from chaco.api import BasePlotContainer
    # from enable.component_editor import ComponentEditor
    from traitsui.api import View, Group, Item
    from traits.api import (HasTraits, String, Directory, Either,
        Trait, Tuple, List, Instance, false, true)

# Package imports
from . import ANA_DIR
from scanr.tools.bash import CPrint
from scanr.tools.path import unique_path
from scanr.tools.string import snake2title

# Constants
PICKLE_FILE_NAME = 'analysis.pickle'

# Globals
pytables_file_handles = {}


class AbstractAnalysis(HasTraits):

    """
    Base functionality for data analysis and plotting of model results

    Data collection is initiated by calling an instance of a AbstractAnalysis
    subclass with arguments that get passed to the subclass's collect_data()
    method. Afterwards, the associated figure can be created and displayed
    by calling the view() method.

    Subclasses should override:
    collect_data -- do everything necessary to create the arrays of data that
        you wish to be plotted; these arrays should be stored in the *results*
        dictionary attribute
    process_data -- all the graphics code for appropriately plotting the data
        collected in *results*; for Matplotlib code, *figure* must be set to a
        figure handle or a dict of figure handles keyed by filename stubs; for
        Chaco code, *figure* must be an instance of a BasePlotContainer
        subclass and can be shown as a Traits View. In use, this should not be
        called directly (use view() and save_plots()).
    label -- override the trait default to be more descriptive

    Constructor keyword arguments:
    desc -- short phrase describing the analysis being performed
    datadir -- specify path where all the data should be saved
    figure_size -- tuple (width, height) of figure in pixels (600, 800)
    log -- whether to log output messages to analysis.log in the datadir (True)

    Public methods for data collection:
    open_data_file -- open a new HDF-5 PyTables file for collecting data
    get_data_file -- load the HDF-5 file created during collect_data
    close_data_file -- close the HDF-5 file (called automatically following
        data collection in collect_data)
    get_multiengine_client -- gets an ipcontroller multi-engine client and
        verifies the presence of valid ipengines
    get_task_client -- like above, but returns a task client for queueing
        a sequence of tasks to be farmed out to the ipcluster

    Public methods:
    reset -- cleans this analysis object so that it may be called again
    view -- brings up the figure(s) created by process_data()
    save_plots -- saves the figure as an image file in *datadir*
    save_data -- pickles the *results* dictionary in *datadir*
    """

    # Data location
    label = String(__name__.split('.')[-1])
    desc = String
    datadir = Directory
    autosave = true
    last_saved_files = List

    # Chaco container, a matplotlib figure handle, or dict of handles
    figure = Trait(None, dict, Instance(Figure))#, Instance(BasePlotContainer))
    figure_size = Tuple((600, 800))

    # IPython.kernel clients
    mec = Trait(None, Instance(IPmec))
    tc = Trait(None, Instance(IPtc))

    # PyTables file path
    tables_file_path = String

    # Console and log output
    log = true
    logfd = Instance(file)
    out = Instance(CPrint)

    # Dictionary for storing collected data; a "finished" flag
    results = Trait(dict)
    finished = false

    # View for Chaco plots (this can be overriden to customize the view)
    traits_view = \
        View(
            Item("figure", show_label=False),#, editor=ComponentEditor()),
            title='Analysis View',
            resizable=True,
            width=0.5,
            height=0.5,
            kind='nonmodal',
            buttons=['Cancel', 'OK'])

    def __init__(self, **traits):
        super(HasTraits, self).__init__(**traits)

        try:
            if not os.path.exists(self.datadir):
                os.makedirs(self.datadir)
        except OSError:
            self.out('Reverting to base directory:\n%s'%ANA_DIR, error=True)
            self.datadir = ANA_DIR
        finally:
            self.datadir = os.path.abspath(self.datadir)

        self.out('%s initialized:\n%s'%(self.__class__.__name__, str(self)))

    def __call__(self, *args, **kw):
        """Execute data collection; this is a wrapper for collect_data
        """
        if self.finished:
            self.out('Already completed. Create new instance or reset.')
            return

        self.out('Running data collection...', popup=True)
        try:
            self.collect_data(*args, **kw)
        except Exception, e:
            self.out('Unhandled exception:\n%s: %s'%(e.__class__.__name__,
                e.message), error=True, popup=True)
            pdb.post_mortem(sys.exc_info()[2])
        else:
            self.finished = True
            if len(self.results):
                self._save_call_args(*args, **kw)
                self.out('Finished collecting data:\n%s'%'\n'.join(['%s: %s'%
                    (k, self.results[k]) for k in self.results
                        if type(self.results[k]) is ndarray]), popup=True)
                if self.autosave:
                    self.save_data()
            else:
                self.out('Warning: No results found! Analysis incomplete?')
        finally:
            self.close_data_file()
            if self.logfd and not self.logfd.closed:
                self.logfd.close()

    def __str__(self):
        """Column-formatted output of information about this analysis object
        """
        col_w = 16
        s =  ['Subclass:'.ljust(col_w) + self.__class__.__name__]
        if self.desc:
            s += ['Description:'.ljust(col_w) + self.desc]
        s += ['Directory:'.ljust(col_w) + self.datadir]
        if self.mec is not None:
            s += ['Engines:'.ljust(col_w) + str(self.mec.get_ids())]
        else:
            s += ['Engines:'.ljust(col_w) + 'N/A']
        s += ['Autosave:'.ljust(col_w) + str(self.autosave)]
        s += ['Log output:'.ljust(col_w) + str(self.log)]
        s += ['Completed:'.ljust(col_w) + str(self.finished)]
        if self.results:
            s += ['Results:'.ljust(col_w) + '%d items:'%len(self.results)]
            res_list = str(self.results.keys())[1:-1]
            if len(res_list) < 60:
                s += [' '*col_w + res_list]
            else:
                res_split = res_list[:60].split(',')
                res_split[-1] = ' etc.'
                res_list = ','.join(res_split)
                s += [' '*col_w + res_list]
        else:
            s += ['Results:'.ljust(col_w) + 'None']
        return '\n'.join(s)

    # Subclass override methods

    def collect_data(self, *args, **kw):
        """Subclass override; set the results dictionary
        """
        raise NotImplementedError

    def process_data(self, *args, **kw):
        """Subclass override; create figure object
        """
        raise NotImplementedError

    # Public methods

    def _tables_file_path_default(self):
        return os.path.abspath(os.path.join(self.datadir, 'data.h5'))

    def open_data_file(self):
        """Open a new HDF-5 data file and return the PyTables File object that
        represents it. Data should be stored during collection, and can then be
        accessed by opening this file during processing with get_data_file.
        """
        return self.get_data_file(mode='w')

    def get_data_file(self, mode='r'):
        """Open the HDF-5 data file associated with this analysis
        """
        global pytables_file_handles
        if (self.tables_file_path in pytables_file_handles
            and (pytables_file_handles[self.tables_file_path].mode != mode
                or mode == 'w')):
            self.close_data_file()

        if self.tables_file_path not in pytables_file_handles:
            tables_file = tb.openFile(self.tables_file_path, mode=mode)
            pytables_file_handles[self.tables_file_path] = tables_file
            self.out('Loaded data file:\n%s'%self.tables_file_path)

        return pytables_file_handles[self.tables_file_path]

    def close_data_file(self):
        """Close the HDF-5 data file if it is still open
        """
        global pytables_file_handles
        if self.tables_file_path in pytables_file_handles:
            tables_file = pytables_file_handles[self.tables_file_path]
            tries = 0
            while tables_file.isopen and tries < 20:
                tables_file.close()
                sleep(0.05)
                tries += 1
            if tables_file.isopen:
                self.out('Failed to close file:\n%s'%self.tables_file_path)
            else:
                self.out('Closed data file:\n%s'%self.tables_file_path)
                del pytables_file_handles[self.tables_file_path]

    def get_multiengine_client(self):
        """Gets a multi-engine client for an ipcontroller

        Returns None if a valid connection could not be established.
        """
        if self.mec is not None:
            return self.mec

        # Create and return new multi-engine client
        mec = None
        try:
            mec = IPclient.MultiEngineClient()
        except Exception, e:
            self.out('Could not connect to ipcontroller:\n%s: %s'%
                (e.__class__.__name__, e.message), error=True)
        else:
            engines = mec.get_ids()
            N = len(engines)
            if N:
                self.out('Connected to %d ipengine instances:\n%s'%(N,
                    str(engines)))
            else:
                self.out('No ipengines connected to controller', error=True)
        finally:
            self.mec = mec
        return mec

    def get_task_client(self):
        """Gets a task client for an ipcontroller

        Returns None if a valid connection could not be established.
        """
        if self.tc is not None:
            return self.tc

        # Create and return new task client
        tc = None
        try:
            tc = IPclient.TaskClient()
        except Exception, e:
            self.out('Could not connect to ipcontroller:\n%s: %s'%
                (e.__class__.__name__, e.message), error=True)
        finally:
            self.tc = tc
        return tc

    def reset(self):
        """Reset analysis state so that it can be called again
        """
        self.finished = False
        self.results = {}
        self.datadir = self._datadir_default()
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)
        self.log = False    # close old log file
        self.log = True     # open new log file
        return True

    def execute(self, func, *args, **kw):
        """Wrapper for executing long-running function calls in serial
        """
        if not callable(func):
            self.out("Function argument to execute() must be callable",
                error=True)
            return

        # Log and execute the call
        self.out('Running %s():\n%s\nArgs: %s\nKeywords: %s'%
            (func.__name__, str(func), args, kw))
        func(*args, **kw)
        return

    def view(self):
        """Bring up the figure for this analysis
        """
        if self._no_data():
            return

        self.process_data()
        success = False
        if isinstance(self.figure, dict) or isinstance(self.figure, Figure):
            self.out('Bringing up MPL figure(s)...')
            from pylab import isinteractive, ion, show
            if not isinteractive():
                ion()
            show()
            success = True
        # elif isinstance(self.figure, BasePlotContainer):
        #     self.out('Bringing up Chaco view...')
        #     self.configure_traits()
        #     success = True
        else:
            self.out('No valid figure object found!', error=True)

        return success

    def save_data(self):
        """Saves the results data for a completed analysis
        """
        if self._no_data():
            return

        filename = os.path.join(self.datadir, PICKLE_FILE_NAME)
        try:
            fd = open(filename, 'w')
        except IOError:
            self.out('Could not open save file!', error=True)
        else:
            try:
                cPickle.dump(dict(self.results), fd)
            except cPickle.PicklingError, e:
                self.out('PicklingError: %s'%str(e), error=True)
            except TypeError, e:
                self.out('TypeError: %s'%str(e), error=True)
            else:
                self.out('Analysis data save to file:\n%s'%filename)
            finally:
                fd.close()
        return

    @classmethod
    def load_data(cls, picklepath='.'):
        """Gets a new analysis object containing saved results data
        """
        if not picklepath.endswith(PICKLE_FILE_NAME.split('.')[-1]):
            picklepath = os.path.join(picklepath, PICKLE_FILE_NAME)

        picklepath = os.path.abspath(picklepath)
        datadir = os.path.split(picklepath)[0]
        out = CPrint(prefix=cls.__name__)

        if os.path.exists(picklepath):
            fd = file(picklepath, 'r')
            results = cPickle.load(fd)
            fd.close()
            out('Results loaded from path:\n%s' % picklepath)
        else:
            results = {}
            out('Warning: analysis pickle not found. Results are empty.')

        return cls(results=results, datadir=datadir, finished=True, log=False)

    def open_analysis_directory(self):
        """Open the analysis directory in a Finder window (OS X only)
        """
        if not os.path.isdir(self.datadir):
            self.out('Directory does not exist:\n%s'%self.datadir, error=True)
            return
        if sys.platform != 'darwin':
            self.out('Finder open only available on OS X.', error=True)
            return
        if subprocess.call(['open', self.datadir]) == 0:
            self.out('Opened directory in Finder:\n%s'%self.datadir)
        else:
            self.out('Unable to open directory in Finder.', error=True)
        return

    def save_plots_and_close(self, *args, **kwds):
        """Convenience method to save and then close all figures
        """
        self.save_plots(*args, **kwds)
        self.close_figures()

    def close_figures(self):
        """Close all open figure windows
        """
        if isinstance(self.figure, Figure):
            plt.close(self.figure)
        elif isinstance(self.figure, dict):
            map(lambda f: plt.close(f), self.figure.values())
            self.figure = {}

    def save_plots(self, stem='figure', fmt='pdf'):
        """Saves the current results plots as image file(s)

        For Chaco plots: if you want to save as a PDF, then the reportlab
        library is required since kiva.backend_pdf requires a Canvas object.
        (Available at http://www.reportlab.org/)

        Optional keyword arguments:
        stem -- base filename for image file (default 'figure')
        fmt -- specifies image format for saving, either 'pdf' or 'png'
        """
        if self._no_data():
            return

        # Validate format specification
        if fmt not in ('pdf', 'png'):
            self.out("Image format must be either 'pdf' or 'png'", error=True)
            return

        # Inline function for creating unique image filenames
        get_filename = \
            lambda stem: unique_path(os.path.join(self.datadir, stem),
                fmt="%s_%02d", ext=fmt)
        filename_list = []

        # Create and save figure(s) as specified
        figure_saved = False
        if isinstance(self.figure, dict):
            for stem in self.figure:
                f = self.figure[stem]
                if isinstance(stem, str) and isinstance(f, Figure):
                    fn = get_filename(stem)
                    f.savefig(fn)
                    filename_list.append(fn)
            filename_list.sort()
            figure_saved = True
        elif isinstance(self.figure, Figure):
            dpi = self.figure.get_dpi()
            self.figure.set_size_inches(
                (self.figure_size[0]/dpi, self.figure_size[1]/dpi))
            fn = get_filename(stem)
            self.figure.savefig(fn)
            filename_list.append(fn)
            figure_saved = True
        # elif isinstance(self.figure, BasePlotContainer):
        #             container = self.figure
        #             fn = get_filename(stem)
        #             container.bounds = list(self.figure_size)
        #             container.do_layout(force=True)
        #             if fmt is 'png':
        #                 from kiva.backend_image import GraphicsContext
        #                 gc = GraphicsContext(
        #                     (container.bounds[0]+1, container.bounds[1]+1))
        #                 container.draw(gc)
        #                 gc.save(fn)
        #                 figure_saved = True
        #             elif fmt is 'pdf':
        #                 from kiva.backend_pdf import GraphicsContext
        #                 try:
        #                     from reportlab.pdfgen.canvas import Canvas
        #                 except ImportError:
        #                     self.out('Chaco plot PDF generation requires reportlab!',
        #                         error=True)
        #                     return
        #                 gc = GraphicsContext(Canvas(fn))
        #                 container.draw(gc)
        #                 gc.save()
        #                 figure_saved = True
        #             filename_list.append(fn)
        else:
            self.out('Figure object is not valid. Please recreate.', error=True)

        # Output results of save operation and return
        if figure_saved:
            self.out('Figure(s) saved as:\n%s'%('\n'.join(filename_list)))
            self.last_saved_files = filename_list
        else:
            self.out('Plots have not been created!', error=True)
        return figure_saved

    def new_figure(self, label, title=None, **kwds):
        size = kwds.pop('figsize', (9,8))
        if self.figure is None:
            self.figure = {}
        self.figure[label] = f = plt.figure(figsize=size, **kwds)
        plt.clf()
        if title:
            f.suptitle(title)
        return f

    def logcall(self, f):
        """Method (or support function) decorator that logs a title-case version
        of the decorated function's name
        """
        wraps(f)
        def wrapper(*args, **kwargs):
            self.out(snake2title(f.__name__))
            return f(*args, **kwargs)
        return wrapper

    # Auxiliary log file creation

    def start_logfile(self, stem, timestamps=False):
        """Start a separate log file with the specified filename stem, and with
        timestamping indicated by *timestamps* keyword
        """
        self.out.outfd = file(os.path.join(self.datadir, '%s.log' % stem), 'w')
        self.out.timestamp = timestamps

    def close_logfile(self):
        """Close the most recent auxiliary logfile opened with start_logfile
        """
        if self.out.outfd:
            if not self.out.outfd.closed:
                self.out.outfd.close()
                self.out('Wrote logfile:\n%s' % self.out.outfd.name)

    # Support methods

    def _no_data(self):
        """Whether analysis contains incomplete results data
        """
        good = self.finished and self.results
        if not good:
            self.out('Run data collection first!', error=True)
        return not good

    def _save_call_args(self, *args, **kw):
        """Saves call arguments to a log file in the analysis directory
        """
        try:
            fd = file(os.path.join(self.datadir, 'call_args.log'), 'a')
        except IOError:
            self.out('Failed to open file for saving call arguments',
                error='True')
            fd = None
        else:
            s = []
            if args:
                s += ['Arguments: %s\n'%str(args)[1:-1]]
            if kw:
                keys = kw.keys()
                keys.sort()
                s += ['Keywords:']
                s += ['%s = %s'%(k, kw[k]) for k in keys]
            if not (args or kw):
                s = ['-'*60, 'No arguments passed to call', '-'*60]
            fd.write('\n'.join(s) + '\n')
        finally:
            if fd and not fd.closed:
                fd.close()
        return

    # Traits change notification handlers

    def _logfd_changed(self, old, new):
        if old and not old.closed:
            old.close()
        self.out.outfd = new

    def _log_changed(self, logging):
        if logging:
            self.logfd = self._logfd_default()
        else:
            self.logfd = None

    # Traits defaults

    def _out_default(self):
        return CPrint(prefix=''.join(self.label.title().split()),
            outfd=self.logfd)

    def _logfd_default(self):
        if self.log:
            try:
                fd = file(os.path.join(self.datadir, 'analysis.log'), 'a')
            except IOError:
                self.log = False
            else:
                return fd
        return None

    def _datadir_default(self):
        """Set a subdirectory path for the data based on label and description
        """
        munge = lambda s: '_'.join(s.strip().lower().split())

        sd_stem = munge(self.label)
        if self.desc:
            sd_stem = os.path.join(sd_stem, munge(self.desc))
        stem = os.path.join(ANA_DIR, sd_stem + os.path.sep)

        return unique_path(stem)
