"""
scanr.eeg -- Filters and utility functions for handling continuous EEG data

Available filters: Delta, Theta, SlowGamma, Gamma, FastGamma, Ripple

The Ripple filter has a detect() method for finding putative ripple events.

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import sys
import tables as tb
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists as there
from functools import wraps
from scipy.signal import firwin, lfilter, filtfilt, freqz, hilbert
from scipy.interpolate import interp1d
from traits.api import (HasTraits, List, Int, Float, Bool, Array,
    Callable, Enum)

# Package imports
from .config import Config
from .data import get_group
from .paths import get_data_file_path
from .time import extract_timestamps
from .tools.interp import linear_upsample
from .tools.filters import quick_boxcar, find_peaks
from .tools.radians import xy_to_deg_vec, unwrap_deg

# Constants
CfgBands = Config['band']
FS = Config['sample_rate']['eeg']
DTS_STUB = 'DTS%02d'
DATA_STUB = 'EEG%02d'
DEFAULT_NTAPS = 69
DEFAULT_LP_NTAPS = 53
CLIP = 2047

# Top-level pointer to the H5 EEG file object
if 'eeg_file' not in locals():
    eeg_file = None


def _open_file():
    return bool(type(eeg_file) is tb.File and eeg_file.isopen)

def close_eeg_file():
    """Close the EEG file if it is still open
    """
    if _open_file():
        eeg_file.close()
        sys.stdout.write('eeg_file: closed EEG file\n')

def flush_eeg_file():
    """Close the EEG file if it is still open
    """
    if _open_file():
        eeg_file.flush()
        sys.stdout.write('eeg_file: flushed EEG file\n')

def get_eeg_file(readonly=True):
    """Return an open H5 file object linked to the main data file
    """
    global eeg_file

    # Close the data file if it is open and there is a mode mismatch
    if _open_file():
        if      (eeg_file._isWritable() and readonly) or \
            not (eeg_file._isWritable() or readonly):
            close_eeg_file()

    # If the file is closed, open it with the specified mode ('r' or 'a')
    if not _open_file():
        eeg_path = get_data_file_path('eeg', search=False)
        mode = (readonly and there(eeg_path)) and 'r' or 'a'
        eeg_file = tb.openFile(eeg_path, mode=mode, title='KData EEG')
        sys.stdout.write('eeg_file: opened data file (mode=\'%s\')\n' % eeg_file.mode)

    return eeg_file

def get_eeg_data(rds, tt):
    """Retrieve the LFP data from the given recording session and tetrode

    Arguments:
    rds -- session (rat, day, maze) tuple
    tt -- tetrode number

    Returns data array if found (otherwise None).
    """
    results = None
    grp = get_group(eeg=True, rds=rds)
    if grp is not None:
        try:
            data = getattr(grp, DATA_STUB%tt)
        except AttributeError:
            sys.stderr.write('eeg_data: Failed to load EEG%02d from %s\n'%(
                tt, grp._v_pathname))
        else:
            results = data.read()
    return results

def get_eeg_timeseries(rds, tt):
    """Retrieve the LFP timeseries from the given recording session and tetrode

    Returns (ts, samples) tuple of the data if found (otherwise None).
    """
    results = None
    grp = get_group(eeg=True, rds=rds)
    if grp is not None:
        try:
            dts = getattr(grp, DTS_STUB%tt)
            data = getattr(grp, DATA_STUB%tt)
        except AttributeError:
            sys.stderr.write('eeg_timeseries: Failed to load DTS/EEG%02d from %s\n'%(
                tt, grp._v_pathname))
        else:
            results = (extract_timestamps(dts.read()), data.read())
    return results

def get_eeg_sample_rate(rds, tt):
    """Retrieve the EEG sampling rate for the recording session and tetrode

    Returns sample rate or None if data not found.
    """
    results = None
    grp = get_group(eeg=True, rds=rds)
    if grp is not None:
        try:
            data = getattr(grp, DATA_STUB%tt)
        except AttributeError:
            sys.stderr.write('eeg_sample_rate: Failed to load EEG%02d from %s\n'%(
                tt, grp._v_pathname))
        else:
            try:
                results = data._v_attrs['sample_rate']
            except KeyError:
                close_eeg_file()
                eeg_file = get_eeg_file(False)
                results = fs = find_sample_rate(get_eeg_timeseries(rds, tt)[0])
                data._v_attrs['sample_rate'] = fs
                getattr(grp, DTS_STUB%tt)._v_attrs['sample_rate'] = fs
                close_eeg_file()
    return results

def safe_hilbert(x):
    """Pad signal to nearest power of 2 for efficient fast Fourier transforms
    within the Hilbert transformer function
    """
    N = x.size
    M = pow(2, int(np.ceil(np.log2(N))))
    return hilbert(np.r_[x, np.zeros(M - N)])[:N]

def signal_power(x, fs=FS):
    """Compute a smoothed instantaneuous power of continuous EEG signal data
    """
    P_x = x**2 + np.imag(safe_hilbert(x))**2
    M = int(CfgBands['power_window'] * fs)
    if M < 2:
        return P_x
    return quick_boxcar(P_x, M)

def total_power(P_x, fs=FS):
    """Integrate an instantaneous power signal to compute the total power
    """
    return np.trapz(P_x, dx=1./fs)


class AbstractFilter(HasTraits):

    """
    Time-series filtering capability for EEG data

    Parameters:
    zero_lag -- whether to use zero-lag or causal filtering; default zero-lag

    Signal filtering methods:
    filter -- band-pass filtered signal of continuous EEG data
    timeseries -- get the band-pass filtered timeseries

    Methods computing signal characteristics:
    power -- the instantaneous power of an EEG signal
    phase -- the instantaneous phase of an EEG signal
    frequency -- the instantaneous frequency of an EEG signal

    The above methods receive a **filtered** keyword to specify if the input is
    a signal that has been already filtered. Otherwise, the required first
    argument is assumed to be full-band EEG data.
    """

    # User/subclass-defined parameters
    zero_lag = Bool(True)
    band = Enum('theta', CfgBands.keys())
    ntaps = Int(DEFAULT_NTAPS)
    _ntaps = Int(DEFAULT_LP_NTAPS)
    _decimate_factor = Int(1, desc='set to <3 to disable downsampling')

    # Automatic traits
    b = Array(desc='eeg filter')
    _b = Array(desc='decimation filter')
    F = Callable
    fs = Float(FS)

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.fs = FS
        if self._decimate_factor > 2:
            self.fs /= self._decimate_factor
            self._b = firwin(self._ntaps, self.fs, nyq=FS/2, pass_zero=True)
        self.b = firwin(self.ntaps, CfgBands[self.band], nyq=self.fs/2,
            pass_zero=False)

    def _F_default(self):
        if self.zero_lag:
            return filtfilt
        return lfilter

    def _zero_lag_changed(self):
        self.F = self._F_default()

    # Method decorators

    def filtered(func):
        @wraps(func)
        def wrapped(self, *args, **kwds):
            filtered = kwds.pop('filtered', False)
            if not filtered:
                args = (self.filter(args[0]),) + args[1:]
            return func(self, *args, **kwds)
        return wrapped

    def decimation_guard(func):
        @wraps(func)
        def wrapped(self, *args, **kwds):
            if self._decimate_factor < 3:
                return args[0]
            return func(self, *args, **kwds)
        return wrapped

    # Methods for computing signal characteristics, where decorator @filtered
    # provides for a boolean **filtered** keyword to indicate that the EEG
    # signal is already a bandpass-filtered signal (default False). These
    # methods do not take a time argument, just the signal data.

    @filtered
    def power(self, x):
        """Get the instantaneous power of an EEG signal
        """
        return signal_power(x, fs=self.fs)

    @filtered
    def phase(self, x):
        """Get the instantaneous phase of an EEG signal
        """
        return xy_to_deg_vec(x, np.imag(safe_hilbert(x)))

    @filtered
    def frequency(self, x):
        """Get the instantaneous frequency of an EEG signal
        """
        return np.r_[0,
            (self.fs / 360) * np.diff(unwrap_deg(self.phase(x, filtered=True)))]

    @filtered
    def ptp(self, x):
        """Get the cycle peak-to-peak frequency of an EEG signal
        """
        ix = np.arange(x.size)
        p = find_peaks(x)
        ptp = self.fs / (2 * np.diff(p))
        f = interp1d(p, np.r_[ptp[0], ptp], bounds_error=False, fill_value=ptp.mean())(ix)
        return quick_boxcar(f, M=max(2, 3 * int(self.fs / np.mean(CfgBands[self.band]))))

    # Methods for performing the bandpass filter on EEG data

    def timeseries(self, t, x):
        """For EEG timeseries (t, x), get the band-pass filtered timeseries
        """
        assert t.size == x.size, 'size mismatch for time and data'
        return self._downsample(t), self.filter(x)

    def filter(self, x):
        """Get the band-pass filtered signal for continuous EEG data
        """
        return self.F(self.b, [1], self._decimate(x))

    @decimation_guard
    def _decimate(self, x):
        return self._downsample(self.F(self._b, [1], x))

    @decimation_guard
    def _downsample(self, x):
        return x[::self._decimate_factor]

    # Convenience methods

    def plot(self):
        plt.ioff()
        cols = 1 + int(self._decimate_factor > 2)
        f = plt.figure(figsize=(cols*6, 8))
        f.suptitle('%s band filter'%self.band)
        def draw_plots(iax, rax, b, fs):
            ntaps = b.size
            iax.stem(np.arange(ntaps)-int(ntaps/2), b)
            iax.set_title('taps')
            h, w = freqz(b)
            rax.plot((fs/2) * (h/np.pi), np.abs(w), 'b-')
            rax.set_xlim(0, fs/2)
            rax.set_title('response')
        if self._decimate_factor > 2:
            draw_plots(plt.subplot(221), plt.subplot(223), self._b, FS)
            draw_plots(plt.subplot(222), plt.subplot(224), self.b, self.fs)
        else:
            draw_plots(plt.subplot(211), plt.subplot(212), self.b, self.fs)
        plt.ion()
        plt.show()


class Resample2KFilter(AbstractFilter):

    """
    Filter for resampling 2kHz data back down to 1kHz (cf. resample_eeg script)
    """

    _ntaps = 105
    _decimate_factor = 2

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.fs = FS
        self._b = firwin(self._ntaps, self.fs/2, nyq=FS, pass_zero=True)

    def timeseries(self, t, x):
        return t[::self._decimate_factor], self.filter(x)

    def filter(self, x):
        return self.F(self._b, [1], x)[::self._decimate_factor]

Resample2K = Resample2KFilter()


class FullBandFilter(AbstractFilter):

    """
    Full-band reference for "relative" power calculations; somewhat kludged
    together hi- and low-pass filtering for 1-50Hz band-pass.
    """

    band = 'full'
    ntaps = 129
    zero_lag = False
    _decimate_factor = 10 # 50Hz low-pass
    def __init__(self, **traits):
        AbstractFilter.__init__(self, **traits)
        self.b = firwin(self.ntaps, [1.0], nyq=self.fs/2, # 1Hz high-pass
            pass_zero=False)

FullBand = FullBandFilter()


class DeltaFilter(AbstractFilter):
    band = 'delta'
    _decimate_factor = 35
    _ntaps = 77
Delta = DeltaFilter()


class ThetaFilter(AbstractFilter):
    band = 'theta'
    _decimate_factor = 15
Theta = ThetaFilter()


class SlowGammaFilter(AbstractFilter):
    band = 'slow_gamma'
    _decimate_factor = 4
SlowGamma = SlowGammaFilter()


class GammaFilter(AbstractFilter):
    band = 'gamma'
    _decimate_factor = 3
Gamma = GammaFilter()


class FastGammaFilter(AbstractFilter):
    band = 'fast_gamma'
    _decimate_factor = 3
FastGamma = FastGammaFilter()


from .ripples import RippleFilter
Ripple = RippleFilter()


# Convenience functions

def get_filter(which):
    return dict(
        full=FullBand,
        theta=Theta,
        delta=Delta,
        slow_gamma=SlowGamma,
        gamma=Gamma,
        fast_gamma=FastGamma,
        ripple=Ripple)[which]

def theta_delta_ratio(t, EEG, kernel_t=1.0):
    """Z-scored smoothed theta/delta ratio that signals non-SWS when >1
    """
    theta = Theta.timeseries(t, EEG)
    P_theta = Theta.power(theta[1], filtered=True)
    delta = Delta.timeseries(t, EEG)
    P_delta = Delta.power(delta[1], filtered=True)

    P_theta_bar = quick_boxcar(P_theta, M=int(Theta.fs * kernel_t))
    P_delta_bar = quick_boxcar(P_delta, M=int(Delta.fs * kernel_t))

    F = lambda x, y: interp1d(x, y, fill_value=0.0, bounds_error=False)
    P_theta_up = F(theta[0], P_theta_bar)(t)
    P_delta_up = F(delta[0], P_delta_bar)(t)

    zerodiv = (P_delta_up == 0.0)
    P_delta_up[zerodiv] = 1.0

    TDratio = P_theta_up / P_delta_up
    TDratio[zerodiv] = 0.0

    return (TDratio - np.median(TDratio)) / TDratio.std()



# Functions for computing phase-amplitude modulation
#
# Method from:
# Tort AB, Komorowski R, Eichenbaum H, Kopell N (2010) Measuirng phase-amplitude
# coupling between neuronal oscillations of different frequencies.
# J Neurophysiol 104:1195-1210

def phase_modulation_timeseries(t, x, phase='theta', amp='gamma'):
    F_phase, F_amp = get_filter(phase), get_filter(amp)
    F_phase.zero_lag = F_amp.zero_lag = True

    t_phase, x_phase = F_phase.timeseries(t, x)
    phi_x = F_phase.phase(x_phase, filtered=True)

    t_amp, x_amp = F_amp.timeseries(t, x)
    A_x = np.sqrt(x_amp**2 + np.imag(safe_hilbert(x_amp))**2)

    return (t_phase, phi_x), (t_amp, A_x)

def phase_amplitude_distribution(phase_data, amp_data, nbins=72):
    interp_kwds = dict(fill_value=0.0, bounds_error=False)
    bins = np.linspace(0, 360, nbins + 1)

    t_phase, phi_x = phase_data
    t_amp, A_x = amp_data

    phi_amp = \
        np.fmod(
            interp1d(t_phase, unwrap_deg(phi_x), **interp_kwds)(t_amp),
            360)

    p = np.empty(nbins, 'd')
    for i in xrange(nbins):
        p[i] = np.mean(A_x[np.logical_and(phi_amp>=bins[i], phi_amp<bins[i+1])])

    return p / p.sum()

def modulation_index(phase_distro):
    N = phase_distro.size
    U = 1 / float(N)
    return np.sum(phase_distro * np.log(phase_distro / U)) / np.log(N)

def plottable_phase_distribution(phase_distro, cycles=2):
    bins = np.arange(0, cycles*360, 360./phase_distro.size)
    return bins, np.tile(phase_distro, cycles)

def get_phase_distro(t, x, phase='theta', amp='gamma'):
    return phase_amplitude_distribution(
        *phase_modulation_timeseries(t, x, phase, amp))

def get_modulation_index(t, x, phase='theta', amp='gamma'):
    return modulation_index(get_phase_distro(t, x, phase, amp))
