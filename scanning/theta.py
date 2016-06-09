# encoding: utf-8
"""
theta.py -- Analysis of theta power spectra during head scanning movements

Created by Joe Monaco on April 24, 2012.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from scipy.interpolate import interp1d
from scipy.signal import convolve, gaussian

# Package imports
from scanr.session import SessionData
from scanr.spike import TetrodeSelect, find_theta_tetrode
from scanr.data import get_node, unique_rats
from scanr.time import time_slice, select_from, exclude_from, stamp_to_time
from scanr.meta import get_maze_list, get_start_end
from scanr.eeg import get_eeg_timeseries, Theta, FullBand, total_power

# Local imports
from .core.analysis import AbstractAnalysis
from .core.report import BaseReport
from .tools.misc import Reify, AutoVivification
from .tools.plot import quicktitle, shaded_error
from .tools.stats import friedman_str, t_paired, zscore, CI
from .tools.string import snake2title


class ScanCrossCorrelations(AbstractAnalysis):

    """
    Compute cross-correlations between head scanning events and continuous
    signals such as behavioral and LFP signals
    """

    label = "scan xcorrs"

    def collect_data(self, area='CA1'):
        SessionDescr =  {   'id'                :   tb.UInt16Col(pos=1),
                            'rat'               :   tb.UInt16Col(pos=2),
                            'day'               :   tb.UInt16Col(pos=3),
                            'session'           :   tb.UInt16Col(pos=4),
                            'start'             :   tb.UInt64Col(pos=5),
                            'type'              :   tb.StringCol(itemsize=4, pos=6),
                            't_theta'           :   tb.StringCol(itemsize=16, pos=7),
                            'P_theta'           :   tb.StringCol(itemsize=16, pos=8),
                            'f_theta'           :   tb.StringCol(itemsize=16, pos=9),
                            'speed'             :   tb.StringCol(itemsize=16, pos=10),
                            'radial_velocity'   :   tb.StringCol(itemsize=16, pos=11),
                            'hd_velocity'       :   tb.StringCol(itemsize=16, pos=12)    }

        def get_area_query(area):
            if area == "CAX":
                return '(area=="CA1")|(area=="CA3")'
            return 'area=="%s"' % area
        tetrode_query = '(%s)&(EEG==True)' % get_area_query(area)
        self.out('Using tetrode query: %s' % tetrode_query)

        self.results['scan_points'] = ('start', 'max', 'return', 'end')
        dataset_list = TetrodeSelect.datasets(tetrode_query)
        def get_dataset_sessions():
            sessions = []
            for dataset in dataset_list:
                for maze in get_maze_list(*dataset):
                    sessions.append(dataset + (maze,))
            return sessions
        session_list = get_dataset_sessions()
        self.results['rats'] = rat_list = sorted(list(set(map(lambda d: d[0], dataset_list))))
        self.results['N_rats'] = len(rat_list)

        data_file = self.open_data_file()
        array_group = data_file.createGroup('/', 'arrays', title='Scan and Signal Arrays')
        session_table = data_file.createTable('/', 'sessions', SessionDescr,
            'Sessions for Scan Cross-Correlation Analysis')

        id_fmt = 'data_%06d'
        array_id = 0
        session_id = 0
        row = session_table.row
        remove = []

        for rds in session_list:

            rds_str = 'rat%d-%02d-m%d' % rds
            data = SessionData.get(rds)
            theta_tt = find_theta_tetrode(rds[:2], condn=tetrode_query)
            if theta_tt is None:
                remove.append(rds)
                continue
            theta_tt = theta_tt[0]

            row['id'] = session_id
            row['rat'], row['day'], row['session'] = rds
            row['type'] = (data.attrs['type'] in ('STD', 'MIS')) and 'DR' or 'NOV'
            row['start'] = data.start

            EEG = get_eeg_timeseries(rds, theta_tt)
            if EEG is None:
                remove.append(rds)
                continue

            ts_theta, x_theta = Theta.timeseries(*EEG)
            t_theta = data.T_(ts_theta)
            P_theta = zscore(Theta.power(x_theta, filtered=True))
            f_theta = Theta.frequency(x_theta, filtered=True)

            speed = data.F_('speed')(t_theta)
            radial_velocity = np.abs(data.F_('radial_velocity')(t_theta))
            hd_velocity = np.abs(data.F_('hd_velocity')(t_theta))

            session_signals = [
                    ('t_theta', t_theta),
                    ('P_theta', P_theta),
                    ('f_theta', f_theta),
                    ('speed', speed),
                    ('radial_velocity', radial_velocity),
                    ('hd_velocity', hd_velocity)  ]

            for k, d in session_signals:
                data_file.createArray(array_group, id_fmt % array_id, d,
                    title='%s : %s' % (rds_str, k))
                row[k] = id_fmt % array_id
                array_id += 1

            self.out('Saved data from %s.' % rds_str)

            row.append()
            if array_id % 10 == 0:
                session_table.flush()
            session_id += 1

        for rds in remove:
            session_list.remove(rds)

        self.results['sessions'] = session_list
        self.results['N_sessions'] = len(session_list)
        self.results['signals'] = ('P_theta', 'f_theta', 'speed',
            'radial_velocity', 'hd_velocity')

        session_table.flush()
        self.close_data_file()
        self.out('All done!')

    def _get_xcorr_array_name(self, signal, point):
        return 'xcorr_%s_%s' % (signal, point)

    def generate_signal_xcorrs(self, lag=3):

        scan_table = get_node('/behavior', 'scans')

        data_file = self.get_data_file(mode='a')
        session_table = data_file.root.sessions
        signals = self.results['signals']
        scan_points = self.results['scan_points']

        lag_samples = int(lag * Theta.fs)

        def compute_xcorr_averages(signal):
            rat_slices = { k: {} for k in scan_points }
            for session in session_table.iterrows():
                rat = session['rat']
                session_query = '(rat==%(rat)d)&(day==%(day)d)&(session==%(session)d)' % session

                t_theta = data_file.getNode('/arrays', session['t_theta']).read()
                x_theta = data_file.getNode('/arrays', session[signal]).read()

                # matrix of session timing of scan points (rows) for each scan (columns)
                t_event_points = stamp_to_time(
                    np.array([
                        tuple(scan[k] for k in scan_points)
                            for scan in scan_table.where(session_query)]),
                    zero_stamp=session['start']).T
                if t_event_points.size == 0:
                    continue

                for i, point in enumerate(scan_points):
                    t_point_scans = t_event_points[i]

                    for t_scan in t_point_scans:
                        scan_ix = np.argmin(np.abs(t_theta - t_scan))
                        start_ix, end_ix = scan_ix - lag_samples, scan_ix + lag_samples
                        if start_ix < 0 or end_ix > x_theta.size:
                            continue

                        signal_slice = x_theta[start_ix:end_ix+1]

                        if rat in rat_slices[point]:
                            if signal_slice.size != rat_slices[point][rat].shape[1]:
                                continue
                            rat_slices[point][rat] = np.vstack((rat_slices[point][rat], signal_slice))
                        else:
                            rat_slices[point][rat] = signal_slice[np.newaxis]
                self.out.printf('.', color='lightgreen')
            self.out.printf('\n')

            def compute_rat_averages():
                mu = {}
                for point in scan_points:
                    N_rats = len(rat_slices[point].keys())
                    averages = []
                    for rat in rat_slices[point].keys():
                        averages.append(rat_slices[point][rat].mean(axis=0))
                    mu[point] = np.array(averages)
                return mu
            return compute_rat_averages()

        if hasattr(data_file.root, 'xcorr_data'):
            data_file.removeNode(data_file.root, 'xcorr_data', recursive=True)
        xcorr_data_group = data_file.createGroup('/', 'xcorr_data',
            title='Rat Averages for Continous Signal Cross-Correlations')

        for signal in signals:
            self.out('Generating %s xcorr data...' % signal)
            averages = compute_xcorr_averages(signal)
            for point in scan_points:
                data_file.createArray(
                    xcorr_data_group, self._get_xcorr_array_name(signal, point),
                    averages[point],
                    title='Rat Averages for %s and Scan %s' % (signal, point.title()))
            data_file.flush()

        xcorr_data_group._v_attrs['lag'] = float(lag)
        self.close_data_file()

    def plot_signal_xcorrs(self, fc='slateblue'):

        data_file = self.get_data_file()
        xcorr_data_group = data_file.root.xcorr_data
        lag = xcorr_data_group._v_attrs['lag']

        ylim = {
            'P_theta' : (-0.3, 0.3),
            'f_theta' : (7.2, 7.8),
            'speed' : (0, 30),
            'hd_velocity' : (0, 200),
            'radial_velocity' : (0, 17)
        }
        nrows, ncols = 2, 3
        figsize = (11,6.5)

        def create_xcorr_figure(signal):
            f = self.new_figure('%s_xcorrs' % signal, 'Scan x %s Cross-Correlations' % signal,
                figsize=figsize)

            for i, point in enumerate(self.results['scan_points']):
                rat_averages = data_file.getNode(
                    xcorr_data_group, self._get_xcorr_array_name(signal, point)).read()
                N_rats, N_t = rat_averages.shape
                mu = rat_averages.mean(axis=0)
                sem = rat_averages.std(axis=0) / np.sqrt(N_rats)
                t = np.linspace(-lag, lag, N_t)
                ax = f.add_subplot(nrows, ncols, i+1)
                ax.plot(t, mu, '-', c=fc, zorder=1)
                shaded_error(t, mu, sem, ax=ax, fc=fc, alpha=0.3, zorder=0)
                ax.axhline(0.0, c='k', ls='-', zorder=-2)
                ax.set_xlim(-lag, lag)
                ax.set_ylim(ylim[signal])
                ax.tick_params(top=False, right=False, left=False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                quicktitle(ax, point.title())

        plt.ioff()

        for signal in self.results['signals']:
            self.out('Plotting %s figure...' % signal)
            create_xcorr_figure(signal)

        plt.ion()
        plt.show()


class ScanVelocityModulation(AbstractAnalysis):

    """
    Compute velocity modulation curves for theta power and frequency during
    head scanning events
    """

    label = "theta velocity"

    def collect_data(self):
        """Create a data structure with theta power/frequency samples with
        corresponding instantaneous velocity measurements such as path
        speed, head direction velocity, and radial velocity
        """
        velocity_moments = ('speed', 'radial_velocity', 'hd_velocity')
        self.results['velocity_moments'] = velocity_moments

        tetrode_query = '(area=="CA1")&(EEG==True)'
        dataset_list = TetrodeSelect.datasets(tetrode_query, allow_ambiguous=True)

        samples = AutoVivification()

        def initialize_rat_samples(rat):
            for v_name in velocity_moments:
                samples[rat][v_name] = np.array([], float)
            samples[rat]['power'] = np.array([], float)
            samples[rat]['frequency'] = np.array([], float)

        def add_velocity_samples(rat, session, t):
            for moment in velocity_moments:
                add_data_sample(rat, moment, session.F_(moment)(t))

        def add_data_sample(rat, key, data):
            samples[rat][key] = np.r_[samples[rat][key], data]

        for rat, day in dataset_list:
            theta_tt, base_theta = find_theta_tetrode((rat, day),
                condn=tetrode_query, ambiguous=True)

            if rat not in samples:
                initialize_rat_samples(rat)

            for maze in get_maze_list(rat, day):
                rds = rat, day, maze
                self.out('Session rat%03d-%02d-m%d: tetrode Sc%02d' % (rds + (theta_tt,)))

                session = SessionData.get(rds, load_clusters=False)

                EEG = get_eeg_timeseries(rds, theta_tt)
                if EEG is None:
                    continue

                ts, x = EEG
                ts_theta, x_theta = Theta.timeseries(ts, x)

                P_theta = zscore(Theta.power(x_theta, filtered=True))
                f_theta = Theta.frequency(x_theta, filtered=True)

                ix_scanning = select_from(ts_theta, session.scan_list)
                t_theta_scanning = session.T_(ts_theta[ix_scanning])

                add_velocity_samples(rat, session, t_theta_scanning)
                add_data_sample(rat, 'power', P_theta[ix_scanning])
                add_data_sample(rat, 'frequency', f_theta[ix_scanning])

        rat_list = sorted(list(set(samples.keys())))
        self.out('Finished collected data for %d rats.' % len(rat_list))

        sample_description = { k: tb.FloatCol() for k in velocity_moments }
        sample_description.update(rat=tb.UInt16Col(), power=tb.FloatCol(), frequency=tb.FloatCol())

        data_file = self.open_data_file()
        results_table = data_file.createTable('/', 'theta_velocity', sample_description,
            title='Theta and Velocity Data Across Rats')
        row = results_table.row

        self.out('Generating results table...')

        c = 0
        for rat in rat_list:
            N = samples[rat]['power'].size
            self.out('Adding rat %d, with %d samples.' % (rat, N))

            assert len(set(samples[rat][k].size for k in samples[rat].keys())) == 1

            for i in xrange(N):
                row['rat'] = rat
                row['power'] = samples[rat]['power'][i]
                row['frequency'] = samples[rat]['frequency'][i]
                for moment in velocity_moments:
                    row[moment] = samples[rat][moment][i]
                row.append()

                if c % 100 == 0:
                    results_table.flush()
                if c % 500 == 0:
                    self.out.printf('.')
                c += 1

            self.out.printf('\n')
        self.out('Done!')

        self.close_data_file()

    def process_data(self, bins=16):

        self.out.outfd = file(os.path.join(self.datadir, 'stats_%d.log' % bins), 'w')
        self.out.timestamp = False

        data_file = self.get_data_file()
        results = data_file.root.theta_velocity
        rat_list = unique_rats(results)
        velocity_moments = self.results['velocity_moments']

        def find_velocity_bins():
            vbins = {}
            for moment in velocity_moments:
                data = results.col(moment)
                lo, hi = CI(data, alpha=0.02)
                self.out('%s: range %f to %f' % (moment, lo, hi))
                vbins[moment] = np.linspace(lo, hi, bins+1)
            return vbins
        velocity_bins = find_velocity_bins()

        def generate_velocity_modulation_curves():
            self.out('Generating velocity modulation curves...')
            P_v = np.empty((len(velocity_moments), len(rat_list), bins), 'd')
            f_v = np.empty_like(P_v)

            power = results.col('power')
            frequency = results.col('frequency')

            for i, moment in enumerate(velocity_moments):
                self.out('Computing %s...' % moment)

                for j, rat in enumerate(rat_list):
                    self.out.printf('[%d] ' % rat, color='lightgreen')

                    for k in xrange(bins):
                        v_lo, v_hi = velocity_bins[moment][k:k+2]

                        query = '(rat==%d)&(%s>=%f)&(%s<%f)' % (rat,
                            moment, v_lo, moment, v_hi)

                        ix = results.getWhereList(query)
                        power_sample = power[ix]
                        frequency_sample = frequency[ix]

                        P_v[i,j,k] = np.median(power_sample)
                        f_v[i,j,k] = np.median(frequency_sample)

                self.out.printf('\n')
            return P_v, f_v

        P_v_fn = os.path.join(self.datadir, 'P_v_%d.npy' % bins)
        f_v_fn = os.path.join(self.datadir, 'f_v_%d.npy' % bins)
        if os.path.exists(P_v_fn) and os.path.exists(f_v_fn):
            self.out('Loading previous velocity modulation data...')
            P_v, f_v = map(np.load, (P_v_fn, f_v_fn))
        else:
            P_v, f_v = generate_velocity_modulation_curves()
            np.save(P_v_fn, P_v)
            np.save(f_v_fn, f_v)

        plt.ioff()
        if type(self.figure) is not dict:
            self.figure = {}
        self.figure['velocity_modulation_curves'] = f = plt.figure(num=20, figsize=(9,10))
        plt.clf()
        f.suptitle('Theta (Z-)Power, Frequency - Velocity Modulation')

        N_moments = len(velocity_moments)
        line_fmt = dict(ls='-', c='k', lw=2)
        shade_fmt = dict(ec='none', alpha=0.3, fill=True, fc='k', zorder=-1)

        def compute_error(M):
            return 1.96 * M.std(axis=0) / np.sqrt(M.shape[0])

        def get_errlim(mu, err, factor=0.1):
            errmin, errmax = (mu-sem).min(), (mu+sem).max()
            derr = errmax - errmin
            return errmin - (factor/2)*derr, errmax + (factor/2)*derr

        labels = ('Z-Power', 'Frequency')

        for i, moment in enumerate(velocity_moments[:2]):
            centers = (lambda b: (b[1:] + b[:-1]) / 2)(velocity_bins[moment])

            for j, X_v in enumerate([P_v, f_v]):

                ax = f.add_subplot(N_moments, 2, i*2 + j + 1)
                mu = X_v[i].mean(axis=0)
                sem = compute_error(X_v[i])

                ax.plot(centers, mu, **line_fmt)
                shaded_error(centers, mu, sem, ax=ax, **shade_fmt)

                self.out('%s: %s %s' % (moment.title(), labels[j], friedman_str(X_v[i])))

                if i == 0:
                    quicktitle(ax, labels[j], size='x-small')

                ax.set_xlabel(snake2title(moment))
                ax.set_xlim(centers[0], centers[-1])
                ax.set_ylim(get_errlim(mu, sem))

        plt.ion()
        plt.show()
        self.close_data_file()
        self.out.outfd.close()


class ScanThetaPower(AbstractAnalysis):

    """
    Analyze relationship between head scanning event and EEG theta power
    """

    label = "scan theta"

    def collect_data(self):
        """Collate theta power and head-scan events across CA1 datasets
        """
        tetrode_query = '(area=="CA1")&(EEG==True)'
        scan_table = get_node('/behavior', 'scans')
        potentiation_table = get_node('/physiology', 'potentiation')
        dataset_list = TetrodeSelect.datasets(tetrode_query,
            allow_ambiguous=True)

        psd_kwds = dict(Fs=1001.0, NFFT=2048, noverlap=1024, scale_by_freq=True)
        # psd_kwds = dict(Fs=FullBand.fs, NFFT=256, noverlap=0, scale_by_freq=True)

        scan_psd = {}
        pause_psd = {}
        running_psd = {}

        for rat, day in dataset_list:
            theta_tt, base_theta = find_theta_tetrode((rat, day), condn=tetrode_query,
                ambiguous=True)
            self.out('Rat%03d-%02d: using tetrode Sc%d'%(rat, day, theta_tt))

            lfp = np.array([])
            scan_lfp = np.array([])
            pause_lfp = np.array([])
            running_lfp = np.array([])

            for session in get_maze_list(rat, day):
                self.out('Adding data from session %d...' % session)
                rds = rat, day, session
                data = SessionData.get(rds, load_clusters=False)

                ts, EEG = get_eeg_timeseries(rds, theta_tt)
                ts_full, x_full = ts, EEG #FullBand._downsample(ts), FullBand._decimate(EEG)

                running_ix = data.filter_tracking_data(ts_full, boolean_index=True, **data.running_filter())

                lfp = np.r_[lfp, x_full]
                scan_lfp = np.r_[scan_lfp, x_full[select_from(ts_full, data.scan_list)]]
                pause_lfp = np.r_[pause_lfp, x_full[select_from(ts_full, data.pause_list)]]
                running_lfp = np.r_[running_lfp, x_full[running_ix]]

            self.out('Computing and normalizing spectra...')
            Pxx, freqs = psd(lfp, **psd_kwds)
            Pxx_scan = np.squeeze(psd(scan_lfp, **psd_kwds)[0])
            Pxx_pause = np.squeeze(psd(pause_lfp, **psd_kwds)[0])
            Pxx_running = np.squeeze(psd(running_lfp, **psd_kwds)[0])
            if 'freqs' not in self.results:
                self.results['freqs'] = freqs

            full_power = np.trapz(Pxx, x=freqs)
            for P in Pxx_scan, Pxx_pause, Pxx_running:
                P /= full_power

            if rat in scan_psd:
                scan_psd[rat] = np.vstack((scan_psd[rat], Pxx_scan))
                pause_psd[rat] = np.vstack((pause_psd[rat], Pxx_pause))
                running_psd[rat] = np.vstack((running_psd[rat], Pxx_running))
            else:
                scan_psd[rat] = Pxx_scan[np.newaxis]
                pause_psd[rat] = Pxx_pause[np.newaxis]
                running_psd[rat] = Pxx_running[np.newaxis]

        rat_list = sorted(scan_psd.keys())
        self.out('Averaging spectra for %d rats...' % len(rat_list))

        scan_spectra = np.empty((len(rat_list), len(freqs)), 'd')
        pause_spectra = np.empty_like(scan_spectra)
        running_spectra = np.empty_like(scan_spectra)
        for i, rat in enumerate(rat_list):
            scan_spectra[i] = scan_psd[rat].mean(axis=0)
            pause_spectra[i] = pause_psd[rat].mean(axis=0)
            running_spectra[i] = running_psd[rat].mean(axis=0)

        self.results['rat_list'] = np.array(rat_list)
        self.results['scan_psd'] = scan_spectra
        self.results['pause_psd'] = pause_spectra
        self.results['running_psd'] = running_spectra

        self.out('All done!')

    def process_data(self, modelim=(0.0, 0.55), ymax=0.3, flim=(4,12), fcrop=(2,14)):
        """Display PSDs and statistics for scan/pause events
        """
        self.out.outfd = file(os.path.join(self.datadir, 'figure.log'), 'w')
        self.out.timestamp = False
        plt.ioff()

        res = Reify(self.results)
        F = res.freqs
        rats = res.rat_list

        crop = (F>=fcrop[0]) * (F<=fcrop[1])
        F_crop = F[crop]
        P_scan = res.scan_psd[:,crop]
        P_pause = res.pause_psd[:,crop]
        P_running = res.running_psd[:,crop]

        self.out('Found data for %d rats:\n%s' % (len(rats), str(rats)))

        c_scan = 'r'
        c_pause = 'b'
        c_running = 'k'

        def CI(P):
            return 1.96 * P.std(axis=0) / np.sqrt(P.shape[0])
        def freq_mean(P):
            return F_crop, P.mean(axis=0)
        def freq_mean_error(P):
            return freq_mean(P) + (CI(P),)

        if type(self.figure) is not dict:
            self.figure = {}
        self.figure['theta_power'] = f = plt.figure(num=25, figsize=(8,10))
        plt.clf()
        f.suptitle('Power Spectra During Scanning and Non-Scanning Behaviors')
        ax = f.add_subplot(311)

        ax.plot(*freq_mean(P_scan), c=c_scan, lw=2, label='Scan', zorder=10)
        shaded_error(*freq_mean_error(P_scan), ax=ax, fc=c_scan, alpha=0.3, zorder=10)

        ax.plot(*freq_mean(P_pause), c=c_pause, lw=2, label='Pause', zorder=5)
        shaded_error(*freq_mean_error(P_pause), ax=ax, fc=c_pause, alpha=0.3, zorder=5)

        ax.plot(*freq_mean(P_running), c=c_running, lw=2, label='Running', zorder=1)
        shaded_error(*freq_mean_error(P_running), ax=ax, fc=c_running, alpha=0.3, zorder=1)

        ax.set_xlim(flim)
        ax.set_ylim(0, ymax)
        ax.tick_params(top=False, right=False, direction='out')
        ax.legend(loc='upper right')

        theta = (F>5) * (F<12)
        scan_mode = np.max(res.scan_psd[:,theta], axis=1)
        pause_mode = np.max(res.pause_psd[:,theta], axis=1)
        running_mode = np.max(res.running_psd[:,theta], axis=1)

        self.out('Scan - running mode: T = %.3f, p < %e' % t_paired(scan_mode, running_mode))
        self.out('Scan < running #: %d/%d rats' % (np.sum(scan_mode-running_mode<0), len(rats)))
        self.out('Pause - running mode: T = %.3f, p < %e' % t_paired(pause_mode, running_mode))
        self.out('Pause < running #: %d/%d rats' % (np.sum(pause_mode-running_mode<0), len(rats)))
        self.out('Scan - pause mode: T = %.3f, p < %e' % t_paired(scan_mode, pause_mode))
        self.out('Scan > pause #: %d/%d rats' % (np.sum(scan_mode-pause_mode>0), len(rats)))

        ax = f.add_subplot(323)
        ax.scatter(running_mode, scan_mode, c=c_scan, s=24, marker='o', edgecolor='k', linewidths=0.5, zorder=5)
        for i in xrange(len(rats)):
            ax.plot([running_mode[i]]*2, [scan_mode[i], pause_mode[i]], '-', c=c_pause, lw=0.75, zorder=-1)
        ax.scatter(running_mode, pause_mode, c=c_pause, s=18, marker='v', linewidths=0, zorder=0)
        ax.plot(modelim, modelim, '-', c='0.3', zorder=-2)
        ax.set_xlabel(r'running $\theta$ peak')
        # ax.tick_params(direction='out')
        ax.axis(modelim*2)
        quicktitle(ax, r'scan/pause $\theta$ peak')

        self.figure['power_spectra_rats'] = f = plt.figure(num=26, figsize=(9,8))
        plt.clf()
        f.suptitle('Power Spectra For Individual Rats (N = %d)' % len(rats))

        def plot_rat_spectra(ax, data, label):
            ax.imshow(data, interpolation='nearest', origin='upper', aspect='auto',
                extent=[F[0], F[-1]+(F[-1]-F[-2]), 0, rats.size])
            ax.set(yticks=(np.arange(rats.size)+0.5), yticklabels=map(str, rats[::-1]))
            ax.set_xlim(flim)
            ax.tick_params(axis='y', right=False, direction='out', labelsize='xx-small', length=3)
            quicktitle(ax, '%s spectra' % label, size='x-small')

        plot_rat_spectra(f.add_subplot(221), res.scan_psd, 'scan')
        plot_rat_spectra(f.add_subplot(222), res.pause_psd, 'pause')
        plot_rat_spectra(f.add_subplot(223), res.running_psd, 'running')

        plt.draw()
        plt.show()

        self.out.outfd.close()

