# encoding: utf-8
"""
interneurons.py -- Analysis of interneuron firing in relation to scanning

Created by Joe Monaco on January 10, 2014.
Copyright (c) 2013 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

from __future__ import division

# Library imports
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Package imports
from scanr.lib import *
from scanr.spike import xcorr

# Local imports
from .core.analysis import AbstractAnalysis, ANA_DIR
from .tools.misc import AutoVivification
from .tools.string import snake2title
from .tools.plot import quicktitle, shaded_error
from .tools.colormaps import get_diffmap_for
from .tools.path import unique_path


class ScanInterneuronCorrelation(AbstractAnalysis):

    """
    Examine cross-correlation of interneuronal activity with scan timing
    """

    label = 'interneuron xcorr'

    def collect_data(self, potentiation_rats=True, areas=['CA1', 'CA3'], scan_point='start',
        time_window=3.0, time_bins=64, min_cell_count=1):
        """Compute scan-spike cross-correlations for interneurons, restricted to rats
        with detected potentiation events.

        Arguments:
        areas -- list of area names from which to include interneurons
        potentiation_rats -- whether to exclude rats without potentiation events
        scan_point -- string name of scan point for time-locking the correlation
        time_window -- maximum lag of the correlation (half-window)
        time_bins -- number of discrete bins used to construct the curves
        min_cell_count -- minimum number of cells for animal to be included
        """
        assert scan_point in Scan.PointNames

        scan_table = get_node('/behavior', 'scans')
        session_table = get_node('/metadata', 'sessions')

        class Interneuron(object):
            def __init__(self):
                self.name = ''
                self.dataset = (0,0)
                self.N_spikes = 0
                self.mean_rate = 0.0
                self.recording_dt = 0.0
                self.spikes = np.array([])
                self.scans = np.array([])

        @self.logcall
        def _collect_spike_scan_trains():
            data = AutoVivification()
            if potentiation_rats:
                def _get_all_datasets_from_potentiation_rats():
                    rats = unique_rats('/physiology', 'potentiation')
                    self.out('Found %d potentiation rats' % len(rats))
                    rat_query = '|'.join(['(rat==%d)' % rat for rat in rats])
                    return unique_datasets(session_table, condn=rat_query)
                datasets = _get_all_datasets_from_potentiation_rats()
            else:
                area_query = '|'.join(['(area=="%s")' % a for a in areas])
                self.out('Area query: %s' % area_query)
                datasets = TetrodeSelect.datasets(area_query, quiet=True)

            for rat, day in datasets:
                if potentiation_rats:
                    Criteria = InterneuronCriteria # grab ALL interneurons
                else:
                    Criteria = AND(InterneuronCriteria, # grab interneuron in specified areas
                        TetrodeSelect.criterion((rat, day), area_query, quiet=True))

                for session in session_table.where('(rat==%d)&(day==%d)' % (rat, day)):
                    rds = (rat, day, int(session['session']))

                    session_data = SessionData.get(rds)
                    scan_times = np.array([scan[scan_point]
                        for scan in scan_table.where(session_data.session_query)])

                    for tc in session_data.get_clusters(Criteria):
                        cluster = session_data.cluster_data(tc)
                        cell_id = 'd%d%s' % (session['day'], tc)

                        if cell_id not in data[rat]:
                            data[rat][cell_id] = Interneuron()
                            data[rat][cell_id].dataset = rds[:2]
                            data[rat][cell_id].name = tc

                        cell = data[rat][cell_id]
                        cell.spikes = np.concatenate((cell.spikes, cluster.spikes))
                        cell.scans = np.concatenate((cell.scans, scan_times))
                        cell.recording_dt += session_data.duration
            return dict(data)
        interneuron_data = _collect_spike_scan_trains()

        @self.logcall
        def _compute_mean_rates(data):
            for rat in data:
                for cell_id in data[rat]:
                    cell = data[rat][cell_id]
                    cell.N_spikes = cell.spikes.size
                    cell.mean_rate = cell.N_spikes / cell.recording_dt
        _compute_mean_rates(interneuron_data)

        @self.logcall
        def _compute_cross_correlations(data):
            X = AutoVivification()
            for rat in sorted(data.keys()):
                cell_count = len(data[rat].keys())
                if cell_count < min_cell_count:
                    self.out('Rat %d only has %d cell(s), skipping...' % (rat, cell_count))
                    continue
                self.out.printf('Rat %03d: ' % rat, color='lightgray')
                for cell_id in data[rat]:
                    self.out.printf('.')
                    cell = data[rat][cell_id]
                    C = xcorr(cell.scans, cell.spikes, maxlag=time_window*1e6, bins=time_bins)[0]
                    C /= cell.scans.size
                    if rat in X:
                        X[rat]['xcorr'] = np.vstack((X[rat]['xcorr'], C))
                        X[rat]['rate'] = np.hstack((X[rat]['rate'], cell.mean_rate))
                    else:
                        X[rat]['xcorr'] = np.atleast_2d(C)
                        X[rat]['rate'] = np.atleast_1d(cell.mean_rate)
                self.out.printf('\n')
            return dict(X)
        spike_scan_xcorrs = _compute_cross_correlations(interneuron_data)

        @self.logcall
        def _save_correlation_data(X):
            fileh = self.open_data_file()
            for rat in X:
                fileh.createArray('/rat%03d' % rat, 'xcorr', X[rat]['xcorr'], createparents=True)
                fileh.createArray('/rat%03d' % rat, 'rate', X[rat]['rate'], createparents=True)

            bins = (time_bins % 2 == 0) and time_bins + 1 or time_bins
            lags = (lambda b: (b[:-1] + b[1:])/2)(
                np.linspace(-time_window, time_window, bins + 1))
            lh = fileh.createArray('/', 'lags', lags, title='Lags - Bin Centers')
            fileh.setNodeAttr(lh, 'time_window', time_window)
            fileh.setNodeAttr(lh, 'time_bins', bins)
            fileh.setNodeAttr(lh, 'scan_point', scan_point)

            fileh.createArray('/', 'rats', np.array(sorted(X.keys())), title='List of Rats')

            self.close_data_file()
        _save_correlation_data(spike_scan_xcorrs)

        self.results['results_are_in_data_file'] = True
        self.out('Good-bye!')

    def process_data(self, ylim=(0.0, 1.4), clim=(0.3, 1.8), show_fraction=1.0, smoothing=3):

        self.out('Loading data...')
        fileh = self.get_data_file()
        root = fileh.root
        rats = root.rats.read()
        lags = root.lags.read()
        dt = lags[1] - lags[0]
        bins = lags.size
        assert bins == fileh.getNodeAttr('/lags', 'time_bins'), 'bins mismatch'
        maxlag = fileh.getNodeAttr('/lags', 'time_window')
        scan_point = fileh.getNodeAttr('/lags', 'scan_point')
        tlim = (-maxlag, maxlag)

        title = 'Interneuron Spike-Scan [%s] Cross-Correlations\n%d rats' % (scan_point.title(), len(rats))
        f = self.new_figure('xcorrs_%s' % scan_point, title, figsize=(8.5, 11), num=10)

        @self.logcall
        def _compute_rat_average_xcorrs():
            C = np.empty((rats.size, bins), 'd')
            for i, rat in enumerate(rats):
                X = fileh.getNode('/rat%03d' % rat, 'xcorr').read()
                r = fileh.getNode('/rat%03d' % rat, 'rate').read()
                Xbar = X / (r[:,np.newaxis] * dt) # normalized by interneuron mean firing rate
                C[i] = Xbar.mean(axis=0)
            return C
        C_rat = _compute_rat_average_xcorrs()

        plt.ioff()
        line_fmt = dict(c='k', ls='-', lw=1, alpha=0.5, zorder=-5)

        @self.logcall
        def _plot_rat_average_correlations():
            ax = f.add_subplot(321)
            ax.plot(lags, C_rat.T, 'k-', lw=1.5, alpha=0.5)
            ax.axhline(y=1.0, **line_fmt)
            ax.set(xlim=tlim, ylim=ylim, ylabel='Correlation')
            ax.tick_params(top=False, right=False)
            quicktitle(ax, 'per rat')

            ax = f.add_subplot(322)
            C_rat_star = medfilt(C_rat, kernel_size=[1,smoothing])

            mu = C_rat_star.mean(axis=0)
            err = C_rat_star.std(axis=0) / np.sqrt(C_rat_star.shape[0])
            ax.axhline(y=1.0, **line_fmt)
            ax.plot(lags, mu, 'k-', lw=2)
            shaded_error(lags, mu, err, alpha=0.4, ec='none', fc='slateblue')
            ax.tick_params(top=False, right=False)
            ax.set(xlim=tlim, ylim=ylim)
            quicktitle(ax, r'mean $\pm$ s.e.m.')
        _plot_rat_average_correlations()

        @self.logcall
        def _create_ordered_cell_matrix():
            Xbar = []
            for i, rat in enumerate(rats):
                X = fileh.getNode('/rat%03d' % rat, 'xcorr').read()
                r = fileh.getNode('/rat%03d' % rat, 'rate').read()
                Xbar.append(X / (r[:,np.newaxis] * dt))
            M = np.vstack(tuple(Xbar))
            M = medfilt(M, kernel_size=[1,smoothing])
            mod = np.trapz((M - 1)**2)
            hide = int((1 - show_fraction) * M.shape[0])
            if hide > 0:
                def _plot_hidden_cells_separately():
                    f_hidden = plt.figure(figsize=(8,8), num=f.number+1)
                    plt.clf()
                    ax = f_hidden.add_subplot(111)
                    ax.plot(lags, M[np.argsort(mod)][-hide:].T, 'k-', alpha=0.5)
                    plt.figure(f.number)
                _plot_hidden_cells_separately()
                return M[np.argsort(mod)][:-hide]
            return M[np.argsort(mod)]
        sorted_xcorrs = _create_ordered_cell_matrix()
        self.out('Plotting %d interneuron responses in cell matrix...' % sorted_xcorrs.shape[0])

        @self.logcall
        def _plot_cell_matrix(M, subp, cmap='jet'):
            ax = f.add_subplot(subp)
            im = ax.imshow(M, origin='lower', aspect='auto', interpolation='nearest', cmap=cmap,
                vmin=clim[0], vmax=clim[1], zorder=-10,
                extent=[tlim[0], tlim[1], 0, M.shape[0]])
            ax.axvline(x=0.0, **line_fmt)
            ax.set(xlim=tlim, ylim=(0, M.shape[0]))
            ax.tick_params(top=False, right=False)
            plt.colorbar(im)
            return ax

        _plot_cell_matrix(sorted_xcorrs, 323).set_ylabel('Interneurons')
        _plot_cell_matrix(sorted_xcorrs, 324, get_diffmap_for(np.array(clim), 1.0, use_black=False))
        _plot_cell_matrix(sorted_xcorrs, 325, 'prism').set_xlabel(r'$\Delta{t}$ lag, s')
        _plot_cell_matrix(sorted_xcorrs, 326, 'flag').set_xlabel(r'$\Delta{t}$ lag, s')

        plt.ion()
        plt.show()

        self.close_data_file()

#
# Scripting functions
#

def run_all_scan_points(**kwargs):
    """Run data collection for all scan timing points, pass in collect_data() args
    """
    kwargs.pop('scan_point', None)
    root = unique_path(os.path.join(ANA_DIR, 'interneuron_xcorr_series/'))
    for point in Scan.PointNames:
        ana = ScanInterneuronCorrelation(desc='scan point %s' % point, datadir=os.path.join(root, point))
        ana(scan_point=point, **kwargs)

def process_all_scan_points(root_dir=os.path.join(ANA_DIR, 'interneuron_xcorr_series/00'), **kwargs):
    """Run data processing for all scan timing points, pass in process_data() args
    """
    process_args = dict(ylim=(0.75, 1.25), clim=(0.3, 1.7), smoothing=5, show_fraction=0.95)
    process_args.update(kwargs)
    for i, point in enumerate(Scan.PointNames):
        ana = ScanInterneuronCorrelation.load_data(os.path.join(root_dir, point))
        ana.process_data(**process_args)
        ana.save_plots_and_close()
        subprocess.call(["cp", ana.last_saved_files[0], os.path.join(root_dir, '%02d_xcorr_%s.pdf' % (i, point))])
    subprocess.call(["open", root_dir])
