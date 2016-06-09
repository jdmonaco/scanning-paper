# encoding: utf-8
"""
lfp.py -- Distributions of theta (or other band) frequency/power during head
    scanning movements compared to running and non-scan pausing

Created by Joe Monaco on May 7, 2012.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Package imports
from scanr.lib import Config
from scanr.session import SessionData
from scanr.spike import TetrodeSelect, find_theta_tetrode
from scanr.time import select_from
from scanr.meta import get_maze_list
from scanr.eeg import get_eeg_timeseries, get_filter

# Local imports
from .core.analysis import AbstractAnalysis
from .tools.misc import Reify
from .tools.plot import quicktitle, shaded_error
from .tools.stats import KT_estimate, t_paired, zscore

CfgBand = Config['band']
from .scan_ripples import THETA_FREQ_SMOOTHING, THETA_POWER_SMOOTHING


class ScanLFPDistros(AbstractAnalysis):

    """
    Analyze distributions of head scanning events and LFP frequency or power
    """

    label = "LFP Distros"

    def collect_data(self, band='theta', distro='frequency', drange=None,
        nbins=128):
        """Collate frequency and head-scan events across CA1 datasets
        """
        self.results['distro'] = distro
        tetrode_query = '(area=="CA1")&(EEG==True)'
        dataset_list = TetrodeSelect.datasets(tetrode_query,
            allow_ambiguous=True)

        Band = get_filter(band)
        was_zero_lag = Band.zero_lag
        Band.zero_lag = True
        self.results['band'] = band

        # Rat accumulators
        rat_number = []
        running = {}
        scan = {}
        pause = {}

        #if distro == 'frequency':
        #    smoothing = THETA_FREQ_SMOOTHING
        #elif distro == 'power':
        #    self.out('warning: power smoothing not implemented')
        #    smoothing = THETA_POWER_SMOOTHING
        #else:
        #    raise ValueError, 'distro must be frequency or power'

        for dataset in dataset_list:
            rat, day = dataset

            roi_tt, base_theta = find_theta_tetrode(dataset, condn=tetrode_query, ambiguous=True)
            self.out('Rat%03d-%02d: using tetrode Sc%d'%(rat, day, roi_tt))

            if rat not in rat_number:
                rat_number.append(rat)
                running[rat] = np.array([], 'd')
                scan[rat] = np.array([], 'd')
                pause[rat] = np.array([], 'd')

            for session in get_maze_list(rat, day):
                rds = rat, day, session
                data = SessionData.get(rds, load_clusters=False)

                self.out('Collating %s data for rat%03d-%02d-m%d...' % (distro, rat, day, session))
                ts, x = Band.timeseries(*get_eeg_timeseries(rds, roi_tt))

                if distro == 'frequency':
                    sig = Band.frequency(x, filtered=True)
                elif distro == 'power':
                    sig = zscore(Band.power(x, filtered=True))

                running_ix = data.filter_tracking_data(ts, boolean_index=True, **data.running_filter())

                running[rat] = np.r_[running[rat], sig[running_ix]]
                scan[rat] = np.r_[scan[rat], sig[select_from(ts, data.scan_list)]]
                pause[rat] = np.r_[pause[rat], sig[select_from(ts, data.pause_list)]]

        # Initialize data storage and accumulators
        running_pdf = []
        running_cdf = []
        running_mu = []
        scan_pdf = []
        scan_cdf = []
        scan_mu = []
        scan_p = []
        pause_pdf = []
        pause_cdf = []
        pause_mu = []
        pause_p = []
        scan_pause_p = []

        # Setup distribution bins
        if drange is not None:
            bins = np.linspace(drange[0], drange[1], nbins+1)
        elif distro == 'frequency':
            bins = np.linspace(CfgBand[band][0], CfgBand[band][1], nbins+1)
        elif distro == 'power':
            bins = np.linspace(-2, 3, nbins+1)
        self.results['centers'] = (bins[1:] + bins[:-1]) / 2

        def sig_distro(data, cdf=False):
            distro = KT_estimate(np.histogram(data, bins=bins)[0])
            if cdf:
                distro = np.cumsum(distro) / np.sum(distro)
            return distro

        for rat in rat_number:
            self.out('Computing distributions and stats for rat %d...'%rat)
            running_pdf.append(sig_distro(running[rat]))
            scan_pdf.append(sig_distro(scan[rat]))
            pause_pdf.append(sig_distro(pause[rat]))

            running_cdf.append(sig_distro(running[rat], cdf=True))
            scan_cdf.append(sig_distro(scan[rat], cdf=True))
            pause_cdf.append(sig_distro(pause[rat], cdf=True))

            running_mu.append(running[rat].mean())

            scan_mu.append(scan[rat].mean())
            D, pval = st.ks_2samp(scan[rat], running[rat])
            scan_p.append(pval)

            pause_mu.append(pause[rat].mean())
            D, pval = st.ks_2samp(pause[rat], running[rat])
            pause_p.append(pval)

            D, pval = st.ks_2samp(scan[rat], pause[rat])
            scan_pause_p.append(pval)

        # Store results data
        self.results['rat_number'] = np.array(rat_number)
        self.results['running_pdf'] = np.array(running_pdf)
        self.results['scan_pdf'] = np.array(scan_pdf)
        self.results['pause_pdf'] = np.array(pause_pdf)
        self.results['running_cdf'] = np.array(running_cdf)
        self.results['scan_cdf'] = np.array(scan_cdf)
        self.results['pause_cdf'] = np.array(pause_cdf)

        self.results['running_mu'] = np.array(running_mu)
        self.results['scan_mu'] = np.array(scan_mu)
        self.results['scan_p'] = np.array(scan_p)
        self.results['pause_mu'] = np.array(pause_mu)
        self.results['pause_p'] = np.array(pause_p)
        self.results['scan_pause_p'] = np.array(scan_pause_p)

        # Good-bye!
        Band.zero_lag = was_zero_lag
        self.out('All done!')

    def process_data(self, cdf=False, sig_thresh=0.05):
        """Display LFP distributions and statistics for scan/pause events
        """
        plt.ioff()

        # Load results data
        res = Reify(self.results)
        N = len(res.rat_number)
        F = res.centers
        self.out('Found data for %d rats.'%N)

        # Choose distribution function
        if cdf:
            running_df = res.running_cdf
            scan_df = res.scan_cdf
            pause_df = res.pause_cdf
        else:
            running_df = res.running_pdf
            scan_df = res.scan_pdf
            pause_df = res.pause_pdf

        def CI(P):
            return P.mean(axis=0), 1.96 * P.std(axis=0) / np.sqrt(P.shape[0])

        def plot_rat_distros(ax, data, label):
            rats = res.rat_number
            rat_distro = np.empty((rats.size, F.size), 'd')
            norm = lambda P: P / P.max()
            for i, rat in enumerate(rats):
                rat_distro[i] = norm(data[res.rat_number == rat].mean(axis=0))
            ax.imshow(rat_distro, interpolation='nearest', origin='upper', aspect='auto',
                extent=[F[0], F[-1]+(F[-1]-F[-2]), 0, rats.size])
            ax.set(yticks=(np.arange(N)+0.5), yticklabels=map(str, rats[::-1]))
            ax.tick_params(axis='y', right=False, labelsize='xx-small', direction='out', length=3)
            quicktitle(ax, '%s p[%s]' % (label, res.distro))

        # Create the frequency distributions figure
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure['%s_pdfs' % res.distro] = f = plt.figure(figsize=(9,8))
        f.suptitle('%s Distributions Across Rats During Behaviors' % res.distro.title())
        running_label = 'Running'
        ax = f.add_subplot(221)
        ax.plot(F, running_df.mean(axis=0), 'k-', lw=1, label=running_label)
        ax.plot(F, pause_df.mean(axis=0), 'r-', lw=1, label='Pause')
        ax.plot(F, scan_df.mean(axis=0), 'b-', lw=1, label='Scan')
        shaded_error(*((F,) + CI(running_df)), ax=ax, fc='k', alpha=0.4)
        shaded_error(*((F,) + CI(pause_df)), ax=ax, fc='r', alpha=0.4)
        shaded_error(*((F,) + CI(scan_df)), ax=ax, fc='b', alpha=0.4)
        ax.legend(loc=4)
        quicktitle(ax, 'all sessions')

        plot_rat_distros(f.add_subplot(222), scan_df, 'scan')
        plot_rat_distros(f.add_subplot(223), pause_df, 'pause')
        plot_rat_distros(f.add_subplot(224), running_df, 'running')

        # Create the statistical summary figure
        self.figure['%s_rats' % res.distro] = f = plt.figure(figsize=(10,8))
        f.suptitle('Statistical Summary of %s %s Differences' % (res.band.title(), res.distro.title()))

        all_labels = ['scan-running', 'pause-running', 'scan-pause']
        all_df = [  (res.scan_mu, res.running_mu),
                    (res.pause_mu, res.running_mu),
                    (res.scan_mu, res.pause_mu) ]

        ax = f.add_subplot(221)
        for i, d in enumerate(zip(all_labels, all_df)):
            label, mu = d
            delta = mu[0] - mu[1]
            t, p = t_paired(mu[0], mu[1], 0)
            c = (p <= sig_thresh) and 'b' or 'r'
            ax.errorbar([i], [delta.mean()], yerr=CI(delta)[1], fmt=c+'s', ecolor='k', ms=6, mew=0)
            self.out('Rat means, %s differences: %s = %.4f, p < %.6f'%(res.distro, label, delta.mean(), p))
        ax.axhline(0, c='k', ls='-')
        ax.set_xlim(-0.5, 2.5)
        ax.set(ylabel=r'$\Delta$f', xticks=[0,1,2], xticklabels=all_labels)
        ax.tick_params(axis='x', labelsize='x-small')
        quicktitle(ax, 'across rats')

        for i, d in enumerate(zip(all_labels, all_df)):
            label, mu = d
            df = mu[0] - mu[1]
            ax = f.add_subplot(222+i)
            sig_ix = res.scan_p <= sig_thresh
            rats = np.arange(N)
            if np.any(sig_ix):
                h = ax.stem(rats[sig_ix], df[sig_ix], linefmt='k-', basefmt='k-', markerfmt='bo')
                h[0].set_zorder(100)
            if not np.all(sig_ix):
                h = ax.stem(rats[True-sig_ix], df[True-sig_ix], linefmt='k-', basefmt='k-', markerfmt='ro')
                h[0].set_zorder(100)
            ax.set(xticks=rats[::3], xticklabels=map(str,res.rat_number)[::3], xlim=(-1, N))
            ax.tick_params(top=False, right=False)
            ax.tick_params(axis='x', direction='out', labelsize='small')
            if i == 1:
                ax.set_ylabel(r'$\Delta$f')
            quicktitle(ax, label)

        plt.draw()
        plt.show()
