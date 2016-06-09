# encoding: utf-8
"""
firing_modulation.py -- Compute dependence of firing rate on instantaneous
    behavioral moments such as forward velocity and head-direction velocity

Created by Joe Monaco on 2011-09-23.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import glob, re, os, sys
import numpy as np
import tables as tb
import networkx as nx
from os import path
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt2d
from scipy.stats import pearsonr
from .core.analysis import AbstractAnalysis
from .tools.images import tiling_dims, masked_array_to_rgba
from .tools.plot import textlabel
from .tools.stats import smooth_pdf, zscore
from .tools.filters import smart_medfilt2d
from .tools.interp import BilinearInterp2D, linear_upsample
from .tools.string import snake2title
from .tools.misc import unique_pairs

# Package imports
from scanr.lib import Config, flush_file
from .reports import BaseDatasetReport, BaseClusterReport, BaseLapClusterReport
from .data_reports import SPIKE_FMT
from scanr.behavior import Moment
from scanr.data import get_node, new_array
from scanr.cluster import (AND, get_tetrode_restriction_criterion,
    get_min_quality_criterion, PrincipalCellCriteria)
from scanr.session import SessionData
from scanr.time import time_slice_sample
from scanr.spike import PrimaryAreas, TetrodeSelect

# Constants
ModCfg = Config['modulation']
MODULATION_GROUP = ModCfg['group_path']
ND_BINS_NAME = "ND_bins"
DEFAULT_BINS = ModCfg['bins']
FILTER_BASE = 22


def area_from_query(query):
    area_match = re.search("area=='(\w+)'", query)
    if area_match is None:
        from .tools.bash import red
        sys.stderr.write(red('Failed to parse area (\'%s\'), using UNK.\n'%
            self.result['query']))
        area = 'UNK'
    else:
        area = area_match.groups()[0]
    return area

def get_bin_edges(bounds=None, bins=None, absolute=False):
    """Get histogram bin edges with defaults if necessary
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    if bins is None:
        bins = DEFAULT_BINS
    if not (np.iterable(bounds) and len(bounds) == 4):
        raise ValueError, "need (left, right, bottom, top) bounds"
    if absolute:
        bounds = (0, bounds[1], 0, bounds[3])
    return [np.linspace(bounds[0], bounds[1], bins+1),
            np.linspace(bounds[2], bounds[3], bins+1)]

def display_modulation(ax, M, mask, bounds, xwrap=False, ywrap=False):
    """Display a filtered and masked image of a modulation matrix
    """
    smoothed = smart_medfilt2d(M, base=FILTER_BASE, xwrap=xwrap, ywrap=ywrap)
    ax.imshow(
        masked_array_to_rgba(smoothed, mask),
        interpolation='nearest', origin='lower', aspect='auto', extent=bounds)
    textlabel(ax, '%.2f'%smoothed.max(), size='small')
    return ax

def compute_firing_modulation(data, baseline, hist_bins=None, norm=True,
    ax=None, xwrap=False, ywrap=False):
    """Compute (and optionally plot) a firing modulation function for point
    data as 2D baseline-normalized histogram

    Arguments:
    data -- a two-row array for which each column corresponds to a data
        point in (moment_i, moment_j)-space that represents the instan-
        taneous behavioral moment at the time of a spike
    baseline -- a two-row array as above but which contains the entire
        behavioral profile for the entire time the spiking data were
        collected
    hist_bins -- a sequence containing the moment_i and moment_j bin
        edges to use for the histogram
    norm -- whether to normalize the modulation function to the median of
        the non-zero modulation values
    ax -- if specified, the modulation will be plotted to this axis
    xwrap/ywrap -- if plotting, whether to wrap the median filter around
        the x or y dimension, respectively

    Returns firing modulation, baseline occupancy, and sampling mask where
    True signals a pixel in moemnt-space that was not sampled.
    """
    # Set hist_bins if not specified
    if hist_bins is None:
        hist_bins = get_bin_edges()

    # Compute baseline and point data histograms
    H_baseline, x, y = np.histogram2d(baseline[0], baseline[1], bins=hist_bins)
    H_baseline = H_baseline.T.astype('d')
    H_data, x, y = np.histogram2d(data[0], data[1], bins=hist_bins)
    H_data = H_data.T.astype('d')

    # Compute occupancy and no-sampling mask
    H_mask = H_baseline == 0

    # Normalize occupancy and point data to compute firing modulation
    H_baseline /= H_baseline.sum()
    H_data /= H_data.sum()
    H_mod = H_data / H_baseline

    # Zero out un-sampled pixels (these are stored in the mask array)
    H_mod[True - np.isfinite(H_mod)] = 0.0

    # Normalize the firing modulation function to median of non-zero values
    if norm:
        median_nonzero = np.median(H_mod[H_mod!=0])
        if median_nonzero != 0:
            H_mod /= median_nonzero
        else:
            sys.stderr.write('firing_modulation: median_nonzero was zero!\n')

    # Plot an image if an axis was supplied
    if ax is not None:
        display_modulation(ax, H_mod, H_mask,
            [   hist_bins[0][0], hist_bins[0][-1],
                hist_bins[1][0], hist_bins[1][-1]   ],
            xwrap=xwrap, ywrap=ywrap)

    assert np.isfinite(H_mod).all(), 'non-finite modulation matrix'
    return H_mod, H_baseline, H_mask


class BaseScanModulation(TetrodeSelect):

    """
    A collection of support methods for the FiringModulation*Report classes
    """

    def _display_modulation(self, *args, **kwds):
        kwds.update(
            xwrap=Moment.Wrapped[self.results['abscissa']],
            ywrap=Moment.Wrapped[self.results['ordinate']])
        return display_modulation(*args, **kwds)

    def _compute_firing_modulation(self, *args, **kwds):
        kwds.update(
            xwrap=Moment.Wrapped[self.results['abscissa']],
            ywrap=Moment.Wrapped[self.results['ordinate']])
        return compute_firing_modulation(*args, **kwds)

    def _process_args(self, abscissa, ordinate, session, min_quality, tetrodes,
        bins, bounds):
        self._check_moments(abscissa, ordinate)
        session = self._get_session(session)
        criteria = self._get_criteria(min_quality, tetrodes)
        hist_bins, bounds = self._get_bins_bounds(bins, bounds, abscissa,
            ordinate)
        return session, criteria, hist_bins, bounds

    def _check_moments(self, abscissa, ordinate):
        assert abscissa in Moment.Names, "invalid abscissa: %s"%abscissa
        assert ordinate in Moment.Names, "invalid ordinate: %s"%ordinate
        self.out('Analyzing %s x %s firing modulation.'%
            (Moment.Names[abscissa], Moment.Names[ordinate]))
        self.results['abscissa'] = abscissa
        self.results['ordinate'] = ordinate
        return True

    def _get_session(self, session):
        if type(session) is tuple and len(session) == 3:
            session = SessionData(rds=session)
        return session

    def _get_criteria(self, quality, tetrodes=None):
        self.out("Quality filter: at least %s"%quality)
        QualityFilter = get_min_quality_criterion(quality)
        criteria = AND(PrincipalCellCriteria, QualityFilter)
        if tetrodes is not None:
            criteria.append(get_tetrode_restriction_criterion(tetrodes))
        return criteria

    def _get_bins_bounds(self, bins, bounds, abscissa, ordinate):
        if bounds is None:
            bounds = [ModCfg[abscissa+'_min'], ModCfg[abscissa+'_max'],
                ModCfg[ordinate+'_min'], ModCfg[ordinate+'_max']]
        if bins is None:
            hist_bins = None
        else:
            hist_bins = get_bin_edges(bounds, bins)
            bounds = [hist_bins[0][0], hist_bins[0][-1], hist_bins[1][0],
                hist_bins[1][-1]]

        self.out('%s range = %d to %d %s'%(Moment.Names[abscissa],
            bounds[0], bounds[1], Moment.Units[abscissa]))
        self.out('%s range = %d to %d %s'%(Moment.Names[ordinate],
            bounds[2], bounds[3], Moment.Units[ordinate]))

        self.results['hist_bins'] = hist_bins
        self.results['bounds'] = bounds

        return hist_bins, bounds

    def _process_plot_axis(self, ax, bounds, abscissa, ordinate):
        ax.set_xticks(np.linspace(bounds[0], bounds[1], 5))
        ax.set_yticks(np.linspace(bounds[2], bounds[3], 5))
        ax.axis(bounds)
        if self.firstonpage:
            ax.set_ylabel(Moment.Labels[ordinate])
        else:
            ax.set_yticklabels([])
        if self.lastonpage:
            ax.set_xlabel(Moment.Labels[abscissa])
        else:
            ax.set_xticklabels([])

    def _save_data_baseline(self, rat_data, rat_baseline):
        self.out("Saving modulation baselines and point data...")
        savedir = path.join(self.datadir, 'data')
        os.makedirs(savedir)
        for rat in rat_data:
            np.save(path.join(savedir, 'rat_data_%03d.npy'%rat),
                rat_data[rat])
            np.save(path.join(savedir, 'rat_baseline_%03d.npy'%rat),
                rat_baseline[rat])

    def _load_data_baseline(self):
        self.out('Loading rat data and baselines...')
        savedir = path.join(self.datadir, 'data')
        rat_data = {}
        rat_baseline = {}
        rat_data_files = glob.glob(path.join(savedir, "rat_data_*"))
        rat_baseline_files = glob.glob(path.join(savedir, "rat_baseline_*"))
        rat_pattern = re.compile("(\d\d\d)")
        for data, files in \
            [(rat_data, rat_data_files), (rat_baseline, rat_baseline_files)]:
            for fn in files:
                rat_search = re.search(rat_pattern, fn)
                rat = int(rat_search.groups()[0])
                self.out('Loading rat %d data from %s'%(rat, path.split(fn)[1]))
                data[rat] = np.load(fn)
        return rat_data, rat_baseline


class FiringModulationLapClusterReport(BaseLapClusterReport, BaseScanModulation):

    label = "lap cluster modulation"
    xnorm = False

    def collect_data(self, session, min_quality='fair', tetrodes=None,
        bounds=None, abscissa='fwd_speed', ordinate='hd_speed',
        alpha_color=False):
        """Generate per-cluster plots of the firing modulation for each cluster
        in the given session

        Arguments:
        session -- the SessionData object or rds triplet for the session
        tetrodes -- restrict cluster output to a given tetrode or tetrode list
        alpha_color -- color spike points by track angle

        Remaining arguments similar to those for FiringModulationReport.
        """
        session, criteria, bins, bounds = self._process_args(
            abscissa, ordinate, session, min_quality, tetrodes, None, bounds)

        # Compute behavioral moments
        traj = session.trajectory
        M = Moment.get(traj)

        # Interpolators for spike times
        kw = dict(bounds_error=False, fill_value=0.0)
        t = session.T_(traj_ts)
        F_abs = interp1d(t, M[abscissa], **kw)
        F_ord = interp1d(t, M[ordinate], **kw)
        if alpha_color:
            F_alpha = interp1d(t, M['alpha'], **kw)

        for tc, data, lap, ax in self.get_plot(session, criteria):
            start, end = lap

            t_spikes = session.T_(
                time_slice_sample(data.spikes, start=start, end=end))
            if len(t_spikes):
                if alpha_color:
                    ax.scatter(F_abs(t_spikes), F_ord(t_spikes),
                        c=F_alpha(t_spikes), vmin=0, vmax=360, **SPIKE_FMT)
                else:
                    ax.scatter(F_abs(t_spikes), F_ord(t_spikes), **SPIKE_FMT)

            self._process_plot_axis(ax, bounds, abscissa, ordinate)


class FiringModulationClusterReport(BaseClusterReport, BaseScanModulation):

    label = "cluster modulation"
    xnorm = False

    def collect_data(self, session, min_quality='fair', tetrodes=None,
        upsampling=10, bins=DEFAULT_BINS, bounds=None, abscissa='fwd_speed',
        ordinate='hd_speed', scatter=False):
        """Generate per-cluster plots of the firing modulation for each cluster
        in the given session

        Arguments:
        session -- the SessionData object or rds triplet for the session
        tetrodes -- restrict cluster output to a given tetrode or tetrode list
        scatter -- visualize data as scatter plot or as modulation histogram

        Remaining arguments similar to those for FiringModulationReport.
        """
        session, criteria, hist_bins, bounds = self._process_args(
            abscissa, ordinate, session, min_quality, tetrodes, bins, bounds)

        # Compute baseline momentary data
        traj = session.trajectory
        M = Moment.get(traj)
        baseline = np.vstack((  linear_upsample(M[abscissa], upsampling),
                                linear_upsample(M[ordinate], upsampling)))

        # Interpolators for spike times
        kw = dict(bounds_error=False, fill_value=0.0)
        t = session.T_(traj.ts)
        F_abs = interp1d(t, M[abscissa], **kw)
        F_ord = interp1d(t, M[ordinate], **kw)
        if scatter:
            F_alpha = interp1d(t, M['alpha'], **kw)

        for tc, data, ax in self.get_plot(session, criteria):
            self.out('Computing firing modulation for cluster %s...'%tc)

            # Get interpolated spiking data
            t_spikes = session.T_(data.spikes)
            spike_data = np.vstack((F_abs(t_spikes), F_ord(t_spikes)))

            if scatter:
                ax.scatter(spike_data[0], spike_data[1], c=F_alpha(t_spikes),
                    vmin=0, vmax=360, **SPIKE_FMT)
                ax.set_xlim(bounds[:2])
                ax.set_ylim(bounds[2:])
            else:
                self._compute_firing_modulation(spike_data, baseline,
                    hist_bins, ax=ax)

            self._process_plot_axis(ax, bounds, abscissa, ordinate)


class NDFiringModulationAnalysis(AbstractAnalysis, BaseScanModulation):

    label = "ND firing modulation"

    def collect_data(self, tetrode_query="area=='CA1'", min_quality="fair",
        upsampling=5, exclude_alpha=True):
        """Collect N-D baseline and point data for computing the behavioral
        modulation of firing rates

        Arguments same as FiringModulationReport.
        """
        self.results['query'] = tetrode_query
        MainCriteria = self._get_criteria(min_quality)

        moments = Moment.keys()
        if exclude_alpha:
            moments = tuple(filter(lambda k: k != 'alpha', moments))
        self.results['moments'] = moments

        rat_data = {}
        rat_baseline = {}
        for ratday in self._get_datasets(tetrode_query):
            self.out("Creating dataset for rat %d, day %d..."%ratday)
            session_list = SessionData.get_session_list(*ratday)

            # Reject bad datasets
            attrs = session_list[0].data_group._v_attrs
            if attrs['timing_jumps'] or attrs['HD_missing']:
                self.out('...skipping due to dataset problems')
                continue

            Tetrodes = get_tetrode_restriction_criterion(
                self._get_valid_tetrodes(ratday, tetrode_query))

            ClusterFilter = AND(Tetrodes, MainCriteria)

            # Initialize dictionary of data and baseline accumulators
            data = { k: np.array([], 'd') for k in moments }
            baseline = { k: np.array([], 'd') for k in moments }

            # Accumulate behavioral point data across sessions
            for session in session_list:
                # Skip sessions with regular backward jumps in timestamps
                if session.data_group._v_attrs['timing_jumps']:
                    continue

                # Get behavioral moments
                traj = session.trajectory
                M = Moment.get(traj)

                # Upsample the instantaneous moments of the baseline data
                for moment in moments:
                    baseline[moment] = np.r_[baseline[moment], linear_upsample(
                        M[moment], upsampling)]

                kw = dict(bounds_error=False, fill_value=0.0)
                t = session.T_(traj.ts)
                F = { k: interp1d(t, M[k], **kw) for k in moments }

                for cl in session.get_clusters(ClusterFilter):
                    t_spikes = session.T_(session.get_spike_train(cl))
                    for moment in moments:
                        data[moment] = np.r_[data[moment], F[moment](t_spikes)]

            if len(data.values()[0]) == 0:
                continue

            # Concatenate data sets
            rat = ratday[0]
            stacked = np.vstack((data[k] for k in moments))
            if rat not in rat_data:
                rat_data[rat] = stacked
            else:
                rat_data[rat] = np.hstack((rat_data[rat], stacked))

            # Concatenate baseline data sets
            stacked_baseline = np.vstack((baseline[k] for k in moments))
            if rat not in rat_baseline:
                rat_baseline[rat] = stacked_baseline
            else:
                rat_baseline[rat] = np.hstack(
                    (rat_baseline[rat], stacked_baseline))

        self._save_data_baseline(rat_data, rat_baseline)
        self.out("Done!")

    def process_data(self, bins=17, overwrite_bins=False):
        """Create N-D histogram-based firing modulation tensor
        """
        moments = self.results['moments']
        area = area_from_query(self.results['query'])
        save_label = area + "_ND"

        if overwrite_bins:
            save_bounds = True
        else:
            loaded_bounds = get_node(MODULATION_GROUP, ND_BINS_NAME,
                raise_on_fail=False)
            save_bounds = loaded_bounds is None

        if save_bounds:
            hist_bins = [np.linspace(ModCfg[m+"_min"], ModCfg[m+"_max"],
                bins+1) for m in moments]
        else:
            hist_bins = loaded_bounds.read()

        # Load data and compute per-rat firing modulation funcitons
        rat_data, rat_baseline = self._load_data_baseline()
        rat_list = rat_data.keys()
        rat_list.sort()
        H_rat = []
        H_mask = []
        for rat in rat_list:
            self.out.printf('Computing rat %d: '%rat, color='lightgray')
            H_data = np.histogramdd(rat_data[rat].T, bins=hist_bins)[0]
            if H_data.sum() == 0.0:
                self.out("Zero-value modulation for rat %d"%rat)
                continue
            self.out.printf("[data] ")

            H_baseline = np.histogramdd(rat_baseline[rat].T, bins=hist_bins)[0]
            self.out.printf("[baseline] ")

            H_data /= H_data.sum()
            H_baseline /= H_baseline.sum()
            H_mod = H_data / H_baseline

            H_unsampled = True - np.isfinite(H_mod)
            H_mod[H_unsampled] = 0.0

            median_nonzero = np.median(H_mod[H_mod!=0].flatten())
            if median_nonzero != 0:
                H_mod /= median_nonzero
                self.out.printf("[norm] ")
            else:
                H_mod /= H_mod.max() / 2
                self.out.printf("[norm] ", color='lightred')

            H_rat.append(H_mod)
            H_mask.append(H_unsampled)
            self.out.printf("[done]\n")

        # Equally-weighted speed modulation, average across rats
        self.out('Averaging %d per-rat modulations...'%len(H_rat))
        H_avg = np.array(H_rat).mean(axis=0)

        # Compute measure of individual variability
        avg_corr = []
        for i, j in unique_pairs(range(len(H_rat))):
            joint_sampled = True - np.logical_or(H_mask[i], H_mask[j])
            new_corr = pearsonr(H_rat[i][joint_sampled].flatten(),
                H_rat[j][joint_sampled].flatten())[0]
            if np.isfinite(new_corr):
                avg_corr.append(new_corr)
        del H_mask
        if len(avg_corr):
            avg_corr = np.array(avg_corr)
            avg_corr_stats = (avg_corr.mean(),
                avg_corr.std()/np.sqrt(avg_corr.size))
        else:
            avg_corr_stats = 0., 0.
        self.out('Individual variability with avg = %.3f +/- %.3f'%
            avg_corr_stats)

        # Save out the data matrices
        self.out('Saving data files to %s'%MODULATION_GROUP)
        new_arrays = [
            new_array(MODULATION_GROUP, save_label, H_avg,
                title=save_label+' ND Modulation', force=True) ]
        if save_bounds:
            new_arrays.append(
                new_array(MODULATION_GROUP, ND_BINS_NAME, np.array(hist_bins),
                    title='Firing Modulation Bounds', force=True))
        ND_array = new_arrays[0]
        ND_array.attrs.avg_corr_stats = avg_corr_stats
        ND_array.attrs.moments = moments
        self.out('Saved modulation arrays: %s'%
            (', '.join(a._v_pathname for a in new_arrays)))

        flush_file()
        self.out('Done!')


class FiringModulationReport(BaseDatasetReport, BaseScanModulation):

    label = "firing modulation"

    xnorm = False
    ynorm = False

    def collect_data(self, tetrode_query="area=='CA1'", min_quality="fair",
        bins=DEFAULT_BINS, bounds=None, upsampling=5, abscissa='fwd_speed',
        ordinate='hd_speed'):
        """Collect baseline and point data for computing the behavioral
        modulation of firing rates

        Arguments:
        tetrode_query -- query string against the /metadata/tetrodes table
        min_quality -- minimum cluster isolation quality (string name or
            ClusterQuality value) for inclusion of data
        bins -- number of bins in each dimension of the histograms
        bounds -- range bounds of the modulation histogram
        upsampling -- factor for upsampling baseline data for better
            coverage (less data loss due to resolution mismatch)
        abscissa -- name of positional moment to plot on abscissa; defaults to
            'fwd_speed'.
        ordinate -- name of positional moment to plot on ordinate axis;
            defaults to 'hd_speed'.
        """
        self.results['query'] = tetrode_query
        self._check_moments(abscissa, ordinate)
        MainCriteria = self._get_criteria(min_quality)
        hist_bins, bounds = self._get_bins_bounds(bins, bounds, abscissa,
            ordinate)

        rat_data = {}
        rat_baseline = {}
        for ratday, ax in self.get_plot(self._get_datasets(tetrode_query)):
            session_list = SessionData.get_session_list(*ratday)

            # Reject bad datasets
            attrs = session_list[0].data_group._v_attrs
            if attrs['timing_jumps'] or attrs['HD_missing']:
                self.out('...skipping due to dataset problems')
                ax.set_axis_off()
                continue

            Tetrodes = get_tetrode_restriction_criterion(
                self._get_valid_tetrodes(ratday, tetrode_query))

            ClusterFilter = AND(Tetrodes, MainCriteria)

            # Initialize accumulators for this data set
            x_data = np.array([], 'd')
            y_data = np.array([], 'd')
            x_baseline = np.array([], 'd')
            y_baseline = np.array([], 'd')

            # Accumulate behavioral point data across sessions
            for session in session_list:
                clusters = session.get_clusters(ClusterFilter)

                traj = session.trajectory
                M = Moment.get(traj)

                # Upsample the instantaneous moments of the baseline data
                x_baseline = np.r_[x_baseline,
                    linear_upsample(M[abscissa], upsampling)]
                y_baseline = np.r_[y_baseline,
                    linear_upsample(M[ordinate], upsampling)]

                kw = dict(bounds_error=False, fill_value=0.0)
                t = session.T_(traj.ts)
                F_abs = interp1d(t, M[abscissa], **kw)
                F_ord = interp1d(t, M[ordinate], **kw)

                for cl in clusters:
                    t_spikes = session.T_(session.get_spike_train(cl))
                    x_data = np.r_[x_data, F_abs(t_spikes)]
                    y_data = np.r_[y_data, F_ord(t_spikes)]

            if len(x_data) == 0:
                ax.set_axis_off()
                continue

            # Concatenate data sets
            rat = ratday[0]
            stacked = np.vstack((x_data, y_data))
            if rat not in rat_data:
                rat_data[rat] = stacked
            else:
                rat_data[rat] = np.hstack((rat_data[rat], stacked))

            # Concatenate baseline data sets
            stacked_baseline = np.vstack((x_baseline, y_baseline))
            if rat not in rat_baseline:
                rat_baseline[rat] = stacked_baseline
            else:
                rat_baseline[rat] = np.hstack(
                    (rat_baseline[rat], stacked_baseline))

            # Plot the speed modulation
            H, occ, mask = self._compute_firing_modulation(
                [x_data, y_data], [x_baseline, y_baseline],
                hist_bins=hist_bins, ax=ax)
            if H.sum() == 0.0:
                self.out('Zero-value modulation for rat%d-%02d...'%ratday)
                plt.cla()
                ax.set_axis_off()
                continue

            self._process_plot_axis(ax, bounds, abscissa, ordinate)

        self._save_data_baseline(rat_data, rat_baseline)
        self.out("Done!")

    def process_data(self, bins=DEFAULT_BINS, bounds=None, save_label=None):
        """Plot overall and per-rat speed modulation histograms

        Specify string save_label in order to save the averaged per-rat speed
        modulation as gain, occupancy, and is_sampled matrixes to the
        /behavior/speed_gain group.

        Override histogram bounds by passing in *bins* and *bounds* arguments.
        """
        # Set speed modulation histogram bounds
        save_bounds = False
        if save_label is not None:
            bounds_name = "_".join(["bounds", self.results['abscissa'],
                self.results['ordinate']])
            loaded_bounds = get_node(MODULATION_GROUP, bounds_name,
                raise_on_fail=False)
            if loaded_bounds is None:
                save_bounds = True
            else:
                self.out('Retrieving bounds from %s'%loaded_bounds._v_pathname)
                bounds = loaded_bounds
        if bounds is None:
            bounds = self.results['bounds']
        self.out('Using bounds: %s'%str(tuple(bounds))[1:-1])
        hist_bins = get_bin_edges(bounds, bins)

        # Load the data
        rat_data, rat_baseline = self._load_data_baseline()

        # Per-rat histograms
        plt.ioff()
        self.figure = {}
        self.figure['rats'] = f = plt.figure(figsize=(11, 8.5))
        f.suptitle('Firing Modulation - %s'%self.results['query'])
        rat_list = rat_data.keys()
        rat_list.sort()
        nrows, ncols = tiling_dims(len(rat_list))
        panel = 1
        H_rat = []
        H_occ_rat = []
        H_mask_rat = []
        for rat in rat_list:
            ax = plt.subplot(nrows, ncols, panel)
            H, occ, mask = self._compute_firing_modulation(rat_data[rat],
                rat_baseline[rat], hist_bins=hist_bins, norm=True, ax=ax)
            if H.sum() == 0.0:
                self.out("Zero-value modulation for rat %d"%rat)
                continue

            H_rat.append(H)
            H_occ_rat.append(occ)
            H_mask_rat.append(mask)

            ax.set_xticks(ax.get_xticks()[::2])
            ax.set_xlim(bounds[:2])
            ax.set_yticks(ax.get_yticks()[::2])
            ax.set_ylim(bounds[2:])

            ax.set_xticks(np.linspace(bounds[0], bounds[1], 5))
            ax.set_yticks(np.linspace(bounds[2], bounds[3], 5))
            ax.axis(bounds)
            if panel == 1:
                ax.set_ylabel(Moment.Labels[self.results['ordinate']])
            else:
                ax.set_yticklabels([])
            if panel == len(rat_list):
                ax.set_xlabel(Moment.Labels[self.results['abscissa']])
            else:
                ax.set_xticklabels([])
            ax.set_title('rat %03d'%rat, size='x-small')
            panel += 1

        # Equally-weighted speed modulation, average across rats
        self.figure['average'] = f = plt.figure(figsize=(7,6))
        f.suptitle('Average Firing Modulation - %s'%self.results['query'])
        ax = plt.axes()
        H_occ_avg = np.array(H_occ_rat).mean(axis=0)
        H_occ_avg /= H_occ_avg.sum()
        H_mask = H_occ_avg == 0
        H_avg = np.array(H_rat).mean(axis=0)
        self._display_modulation(ax, H_avg, H_mask, bounds)
        ax.set_xlabel(Moment.Labels[self.results['abscissa']])
        ax.set_ylabel(Moment.Labels[self.results['ordinate']])

        # Equally-weighted speed occupancy, average across rats
        self.figure['average_occ'] = f = plt.figure(figsize=(7,6))
        f.suptitle('Average Occupancy - %s'%self.results['query'])
        ax = plt.axes()
        self._display_modulation(ax, H_occ_avg, H_mask, bounds)
        ax.set_xlabel(Moment.Labels[self.results['abscissa']])
        ax.set_ylabel(Moment.Labels[self.results['ordinate']])

        # Compute measure of individual variability
        avg_corr = []
        M = lambda mat: smart_medfilt2d(mat,
            base=FILTER_BASE,
            xwrap=Moment.Wrapped[self.results['abscissa']],
            ywrap=Moment.Wrapped[self.results['ordinate']])
        for i, j in unique_pairs(range(len(H_rat))):
            joint_sampled = True - np.logical_or(H_mask_rat[i], H_mask_rat[j])
            new_corr = pearsonr(M(H_rat[i])[joint_sampled].flatten(),
                M(H_rat[j])[joint_sampled].flatten())[0]
            if np.isfinite(new_corr):
                avg_corr.append(new_corr)
        if len(avg_corr):
            avg_corr = np.array(avg_corr)
            avg_corr_stats = (avg_corr.mean(),
                avg_corr.std()/np.sqrt(avg_corr.size))
        else:
            avg_corr_stats = 0., 0.
        self.out('Individual variability with avg = %.3f +/- %.3f'%
            avg_corr_stats)

        # Save out the data matrices
        self.out('Saving data files to %s'%self.datadir)
        np.save(path.join(self.datadir, 'bounds.npy'), np.array(bounds))
        np.save(path.join(self.datadir, 'average.npy'), H_avg)
        np.save(path.join(self.datadir, 'average_occ.npy'), H_occ_avg)
        np.save(path.join(self.datadir, 'average_mask.npy'), H_mask)

        if save_label:
            try:
                new_arrays = [
                new_array(MODULATION_GROUP, save_label, H_avg,
                    title=save_label+' Firing Modulation', force=True),
                new_array(MODULATION_GROUP, save_label+'_occ', H_occ_avg,
                    title=save_label+' Baseline Occupancy', force=True),
                new_array(MODULATION_GROUP, save_label+'_mask', H_mask,
                    title=save_label+' Sampling Mask', force=True) ]
                mod_array = new_arrays[0]
                mod_array.attrs.avg_corr_stats = avg_corr_stats
                if save_bounds:
                    new_arrays.append(
                    new_array(MODULATION_GROUP, bounds_name, np.array(bounds),
                        title='Firing Modulation Bounds'))
                self.out('Saved modulation arrays: %s'%
                    (', '.join(a._v_pathname for a in new_arrays)))
            except tb.NodeError:
                self.out('Data save failed for %s.'%save_label, error=True)
            finally:
                flush_file()

        plt.ion()
        plt.show()
        self.out('Done!')


class IndividualVariabilityAnalysis(AbstractAnalysis):

    """
    Bar graphs of individual variability across region
    """

    label = 'individual variability'

    def collect_data(self):
        """Collect individual variability data from the firing modulation
        analysis for plotting
        """
        V = {}
        V_area = {}
        V_ND = {}
        for area in PrimaryAreas:
            self.out("Collecting variability data for %s..."%area)
            V[area] = {}
            V_area[area] = []
            V_ND[area] = get_node(MODULATION_GROUP,
                area+"_ND").attrs.avg_corr_stats

            for abscissa, ordinate in Moment.pairs():
                key = ",".join([abscissa, ordinate])
                V[area][key] = get_node(MODULATION_GROUP, "_".join([area,
                    abscissa, ordinate])).attrs.avg_corr_stats
                V_area[area].append(V[area][key][0]) # append the mean

        self.results['V'] = V
        self.results['V_area'] = V_area
        self.results['V_ND'] = V_ND
        self.out("All done!")

    def process_data(self):
        N_areas = len(PrimaryAreas)
        V = self.results['V']
        V_area = self.results['V_area']
        V_ND = self.results['V_ND']

        bar_width = 0.6
        xticks = np.arange(N_areas)

        def process_plot():
            ax = plt.gca()
            ax.set_xticks(xticks)
            ax.set_xticklabels(PrimaryAreas, size='small', rotation=45)
            ax.set_xlim(-0.5, N_areas-0.5)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Recording Area')
            ax.set_ylabel('Correlation')

        self.figure = {}
        self.figure['average'] = f = plt.figure(figsize=(5.5, 7))
        f.suptitle("Average Between-Subject Correlations of Firing Modulation")

        # Averages of 2D modulation between-rat averages
        mean, sem = [], []
        for area in PrimaryAreas:
            mean.append(np.mean(V_area[area]))
            sem.append(np.std(V_area[area])/np.sqrt(len(V_area[area])))

        plt.bar(xticks-bar_width/2, mean, yerr=sem, width=bar_width, ecolor='k',
            ec='none', fc='0.6')
        process_plot()

        # Averages of pair-wise correlations of the ND modulations
        self.figure['ND'] = f = plt.figure(figsize=(5.5, 7))
        f.suptitle("Between-Subject Correlations of ND Firing Modulation")

        mean, sem = [], []
        for area in PrimaryAreas:
            mean.append(V_ND[area][0])
            sem.append(V_ND[area][1])

        plt.bar(xticks-bar_width/2, mean, yerr=sem, width=bar_width, ecolor='k',
            ec='none', fc='0.6')
        process_plot()


class ModulationGraphAnalysis(AbstractAnalysis):

    """
    Create a graph representation of firing modulation correlations across
    recording sites
    """

    label = "modulation graph"

    def collect_data(self, use_ND=False, alpha=0.05):
        """Compute pair-wise modulation correlations between areas

        Arguments:
        use_ND -- get correlations from ND behavioral modulation matrixes
            instead of pair-wise momentary interactions
        alpha -- statistical significance level threshold for collecting r's
        """
        self.results['ND'] = use_ND
        C = {}
        pop_C = []
        for first, second in unique_pairs(PrimaryAreas):
            if first not in C:
                C[first] = {}

            if use_ND:
                self.out('Computing <%s, %s> for ND modulations...'%(first,
                    second))
                H_first = get_node(MODULATION_GROUP, '%s_ND'%first).read()
                H_second = get_node(MODULATION_GROUP, '%s_ND'%second).read()
                joint_sampled = np.logical_or(H_first!=0, H_second!=0)

                C[first][second] = pearsonr(H_first[joint_sampled].flatten(),
                    H_second[joint_sampled].flatten())[0]
                pop_C.append(C[first][second])
            else:
                C[first][second] = []
                for abscissa, ordinate in Moment.pairs():
                    self.out('Computing <%s, %s> for %s x %s...'%(first,
                        second, abscissa, ordinate))

                    M_mask = lambda area: get_node(MODULATION_GROUP,
                        "_".join([area, abscissa, ordinate, 'mask'])).read()

                    M = lambda area: smart_medfilt2d(
                        get_node(MODULATION_GROUP,
                            "_".join([area, abscissa, ordinate])).read(),
                        base=FILTER_BASE,
                        xwrap=Moment.Wrapped[abscissa],
                        ywrap=Moment.Wrapped[ordinate])

                    joint_sampled = True - np.logical_or(M_mask(first),
                        M_mask(second))

                    r, p = pearsonr(
                        M(first)[joint_sampled].flatten(),
                        M(second)[joint_sampled].flatten()  )

                    if p < alpha:
                        C[first][second].append(r)
                        pop_C.append(r)

                C[first][second] = np.array(C[first][second], 'd')
                self.out('--> %d significant correlations'%(
                    C[first][second].size))

        self.results['C'] = C
        self.results['C_mean'] = np.mean(pop_C)
        self.results['C_std'] = np.std(pop_C)

    def process_data(self, show_weak=False):
        self.figure = dict(correlation_graph=plt.figure(figsize=(7,7),
            facecolor='w', frameon=False))

        self.out.outfd = file(os.path.join(self.datadir, 'figure.log'), 'w')

        C = self.results['C']
        C_mean = self.results['C_mean']
        C_std = self.results['C_std']

        G = nx.Graph()

        # Add autoloops so all nodes get drawn
        for area in PrimaryAreas:
            G.add_edge(area, area, weight=0)

        # Add edges with weights corresponding to correlation Z-score
        for first, second in unique_pairs(PrimaryAreas):
            Z = zscore(C[first][second], population=(C_mean, C_std))
            if self.results['ND']:
                weight = float(Z)
            else:
                weight = Z.mean()
            if weight > 0 or show_weak:
                G.add_edge(first, second, weight=weight)

        pos = dict( MEC=[1/3., 1],
                    LEC=[2/3., 1],
                    CA1=[0, 0],
                    CA3=[1, 0],
                    DG=[1, 0.5] )

        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='w',
            node_shape='s', width=2)

        for u, v, data in G.edges(data=True):
            self.out("%s <==> %s = %.4f"%(u, v, data['weight']))
            edge_kwds = dict(edgelist=[(u,v)], width=25*abs(data['weight']),
                edge_color='k')
            if abs(data['weight']) < 0.5:
                edge_kwds.update(style='dashed')
            if data['weight'] < 0.0:
                edge_kwds.update(edge_color='r')
            edge = nx.draw_networkx_edges(G, pos, **edge_kwds)
            edge.set_zorder(data['weight']-5)

        nx.draw_networkx_labels(G, pos, font_size=16, font_family='sans-serif')

        plt.axis('off')
        plt.show()

        self.out.outfd.close()


class NDGainFunction(object):

    """
    Container and interpolator for ND behavior modulation data stored under the
    /behavior/modulation data group.
    """

    def __init__(self, area):
        self.area = area
        self.label = area + "_ND"
        sys.stdout.write('Loading data from label %s.\n'%self.label)

        node = get_node(MODULATION_GROUP, self.label)
        self.moments = node.attrs.moments
        self.ND = node.read()
        self._init_interpolator()

    def __call__(self, *args):
        """Perform nearest neighbor interpolation over a set of moments arrays
        or a moments dictionary returned from behavior.compute_moments().
        """
        if len(args) == 1 and type(args[0]) is dict:
            args = tuple(args[0][k] for k in self.moments)

        assert len(args) == self.ndim, "dimension mismatch"
        assert len(np.unique((len(v) for v in args)))==1, \
            "unequal length inputs"

        index = []
        for d in xrange(self.ndim):
            index.append([])
            for value in args[d]:
                index[d].append(np.argmin(np.abs(value-self.points[d])))
        return self.ND[index]

    def _init_interpolator(self):
        bin_edges = get_node(MODULATION_GROUP, ND_BINS_NAME).read()
        self.ndim, N = bin_edges.shape
        self.bins = N - 1
        assert self.ndim == self.ND.ndim, "bounds ndim mismatch"
        assert self.bins == self.ND.shape[0], "bounds bin mismatch"

        bin_centers = []
        self.bounds = []
        for d in xrange(self.ndim):
            edges = bin_edges[d]
            dmin, dmax = edges[0], edges[-1]
            self.bounds.append((dmin, dmax))
            dx = np.diff(edges[:2])
            bin_centers.append(np.linspace(dmin+dx/2, dmax-dx/2, self.bins))
        self.points = np.array(bin_centers)

    def smoothed(self, args, fs=30.0):
        """Given sampling rate, args are passed to call and result low-passed
        """
        from scipy.signal import firwin, filtfilt
        b = firwin( 32+int(16*(fs/30.0)),
                    2/float(Config['scanning']['min_time']),
                    nyq=fs/2    )
        return filtfilt(b, [1], self(*args))


class GainFunction(object):

    """
    Container, visualizer, and interpolator for pair-wise momentary behavioral
    modulation data stored under the /behavior/modulation data group.
    """

    def __init__(self, area, abscissa='fwd_speed', ordinate='hd_speed'):
        self.area = area
        self.abscissa = abscissa
        self.ordinate = ordinate

        self.label = "_".join([area, abscissa, ordinate])
        sys.stdout.write('Loading data from label %s.\n'%self.label)

        self.M = get_node(MODULATION_GROUP, self.label).read()
        self.M_occ = get_node(MODULATION_GROUP, self.label+"_occ").read()
        self.M_mask = get_node(MODULATION_GROUP, self.label+"_mask").read()

        self.bounds = get_node(MODULATION_GROUP, "_".join(['bounds', abscissa,
            ordinate])).read()
        self.bins = self.M.shape[0]

        self.F = self._init_interpolator(self.M)
        self.F_occ = self._init_interpolator(self.M_occ)
        self.F_mask = self._init_interpolator(self.M_mask.astype('f'),
            fill_value=1.0)

    def __call__(self, *args):
        return self.F(*args)

    def _init_interpolator(self, M, fill_value=0.0):
        """Get a 2D interpolator function for the given speed modulation matrix
        over the given defined bounds.
        """
        dx = np.diff(self.bounds[:2]) / (2*self.bins)
        dy = np.diff(self.bounds[2:]) / (2*self.bins)
        X = np.linspace(self.bounds[0]+dx, self.bounds[1]-dx, self.bins)
        Y = np.linspace(self.bounds[2]+dy, self.bounds[3]-dy, self.bins)
        Z = M.flatten()
        return BilinearInterp2D(x=X, y=Y, z=Z, fill_value=fill_value)

    def get_image(self, occ=False, bins=None, cmax=None, cmap='jet'):
        """Generate an RGBA image matrix of the speed gain modulation or the
        speed occupancy.

        Arguments:
        occ -- whether to plot the occupancy instead of the speed modulation
        bins -- override the default number of bins to use for the interpolated
            image of the distribution
        cmax -- optional maximum cutoff value for the colormap
        cmap -- name of the MPL colormap to use to display the image

        Returns bins x bins x 4 RGBA image matrix, maximum value
        """
        if bins is None:
            bins = self.bins
        b = self.bounds
        dx = np.diff(b[:2]) / bins
        dy = np.diff(b[2:]) / bins
        XY = np.meshgrid(   np.arange(b[0], b[1], dx) + dx/2,
                            np.arange(b[2], b[3], dy) + dy/2   )
        mask = self.F_mask(*XY).astype('?')
        if occ:
            Z = self.F_occ(*XY)
        else:
            Z = self.F(*XY)
        Z = smart_medfilt2d(Z, base=FILTER_BASE,
                xwrap=Moment.Wrapped[self.abscissa],
                ywrap=Moment.Wrapped[self.ordinate] )
        rgba_kwds = dict(cmap=getattr(plt.cm, cmap))
        if cmax is None:
            cmax = Z.max()
        else:
            rgba_kwds.update(norm=False, cmax=cmax)
        return masked_array_to_rgba(Z, mask, **rgba_kwds), cmax

    def show(self, ax=None, **kwds):
        """Plot the behavioral modulation or occupancy functions contained by
        this object

        Keywords are passed to get_image to generate the image.
        """
        do_occ = 'occ' in kwds and kwds['occ']
        new_figure = ax is None
        if new_figure:
            f = plt.figure(figsize=(8,6))
            if do_occ:
                title = self.area + ' Behavioral Occupancy'
            else:
                title = self.area + ' Behavioral Modulation'
            f.suptitle(title)
            ax = plt.axes()
        else:
            title = self.area
            if do_occ:
                title += ' Occupancy'
            ax.set_title(title, size='x-small')

        img_style = dict(interpolation='nearest', origin='lower', aspect='auto',
            extent=self.bounds)
        image, max_value = self.get_image(**kwds)
        ax.imshow(image, **img_style)

        # Set the axis bounds, adjusting for wrapped quantites
        ax.axis(self.bounds)
        bins = self.bins
        if 'bins' in kwds and type(kwds['bins']) is int:
            bins = kwds['bins']
        if Moment.Wrapped[self.abscissa]:
            ax.set_xlim(right=self.bounds[1]-np.diff(self.bounds[:2])/bins)
        if Moment.Wrapped[self.ordinate]:
            ax.set_ylim(top=self.bounds[3]-np.diff(self.bounds[2:])/bins)

        if new_figure:
            ax.set_xlabel(Moment.Labels[self.abscissa])
            ax.set_ylabel(Moment.Labels[self.ordinate])
            textlabel(ax, '%.3f'%max_value)
        else:
            textlabel(ax, '%.3f'%max_value, size='x-small')

        return ax


def run_session_sweep(session, savedir):
    """Run a series of detailed per-cluster, per-session, and per-lap analyses
    of behavioral modulation for the specified experimental sessions

    Arguments:
    session -- either a SessionData object or *rds* triplet
    savedir -- directory for saving all analyses
    """
    from .data_reports import LapClusterSpikesReport

    def run_analysis(name, klass, args, kwds):
        anadir = os.path.join(savedir, name)
        os.makedirs(anadir)
        report = klass(datadir=anadir)
        report(*args, **kwds)
        os.rename(report.results['report'], os.path.join(savedir, name+'.pdf'))

    if type(session) is tuple:
        session = SessionData(rds=session)

    run_analysis('lap_spikes', LapClusterSpikesReport, (session,), {})
    run_analysis('lap_spikes_color', LapClusterSpikesReport, (session,),
        dict(alpha_color=True))

    run_analysis('lap_speed_scatter', FiringModulationLapClusterReport,
        (session,), dict(alpha_color=True))
    run_analysis('lap_position_scatter', FiringModulationLapClusterReport,
        (session,), dict(abscissa='alpha', ordinate='radius', alpha_color=True))

    run_analysis('speed_scatter', FiringModulationClusterReport,
        (session,), dict(scatter=True))
    run_analysis('speed_modulation', FiringModulationClusterReport,
        (session,), dict(upsampling=2))
    run_analysis('position_scatter', FiringModulationClusterReport,
        (session,), dict(abscissa='alpha', ordinate='radius', scatter=True))
    run_analysis('position_modulation', FiringModulationClusterReport,
        (session,), dict(abscissa='alpha', ordinate='radius', upsampling=2))

    run_analysis('dataset_speed_modulation', FiringModulationReport, (),
        dict(tetrode_query='(rat==%d)&(day==%d)'%(session.rat, session.day)))
    run_analysis('dataset_position_modulation', FiringModulationReport, (),
        dict(tetrode_query='(rat==%d)&(day==%d)'%(session.rat, session.day),
            abscissa='alpha', ordinate='radius'))

def run_sweep(rootdir, quality='fair', **kwds):
    """Run each primary area through every pair-wise firing modulation analysis

    Keywords are passed into the data collection call.
    """
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    for abscissa, ordinate in Moment.pairs():
        for area in PrimaryAreas:
            desc = "_".join([area, abscissa, ordinate])
            anadir = os.path.join(rootdir, "firing_modulation_%s"%desc)
            if os.path.exists(anadir):
                sys.stdout.write('Found %s, skipping...\n'%anadir)
                continue
            report = FiringModulationReport(desc=desc, datadir=anadir)
            report( tetrode_query="area=='%s'"%area, min_quality=quality,
                    abscissa=abscissa, ordinate=ordinate, **kwds)
            report.process_data(save_label=desc)
            report.save_plots_and_close()

def run_postprocessing(rootdir, save=False, overwrite=False, figures=False,
    **kwds):
    """Load a collection of firing modulation analyses for processing

    All analysis subdirectories from the specified *rootdir* are loaded for
    processing. Specify *save* to store results into kdata file. Specify
    *figures* if plots should be saved.

    Remaining keywords are passed into the post-processing call.
    """
    from glob import glob
    subdirs = glob(os.path.join(rootdir, 'firing_modulation*'))

    for anadir in subdirs:
        sys.stdout.write('Loading %s...\n'%anadir)
        report = FiringModulationReport.load_data(anadir)

        res = report.results
        area = area_from_query(res['query'])
        if area == 'UNK':
            continue

        abscissa, ordinate = res['abscissa'], res['ordinate']
        sys.stdout.write('Found area %s for %s x %s modulation.\n'%(area,
            abscissa, ordinate))

        if save:
            label = "_".join([area, abscissa, ordinate])
            node = get_node(MODULATION_GROUP, label, raise_on_fail=False)
            if node is not None and not overwrite:
                sys.stdout.write('Found %s, skipping...\n'%node._v_pathname)
                continue
            kwds.update(save_label=label)

        report.process_data(**kwds)

        if figures:
            report.save_plots_and_close()
        else:
            plt.close('all')

def run_ND_sweep(rootdir, bins=33):
    """Run NDFiringModulationAnalysis for each major recording area, saving the
    tensor data to the kdata tables file
    """
    os.makedirs(rootdir)
    for area in PrimaryAreas:
        anadir = os.path.join(rootdir, area)
        ndmod = NDFiringModulationAnalysis(desc=area, datadir=anadir)
        ndmod(tetrode_query="area=='%s'"%area)
        ndmod.process_data(bins=bins)

def run_ND_postprocessing(rootdir, **kwds):
    """Load a collection of ND firing modulation analyses for processing

    Keywords are passed into the post-processing call.
    """
    from glob import glob
    subdirs = glob(os.path.join(rootdir, '*'))

    for anadir in subdirs:
        sys.stdout.write('Loading %s...\n'%anadir)
        try:
            report = NDFiringModulationAnalysis.load_data(anadir)
        except:
            continue

        area = area_from_query(report.results['query'])
        if area == 'UNK':
            continue

        sys.stdout.write('Found area %s ND modulation.\n'%area)
        report.process_data(**kwds)

def show_modulations(save_dir=None, mod_dict={}, occ_dict={}):
    """Plot occupancy and modulation for each pair-wise type of modulation

    A pdf report will be saved out to *save_dir* if specified.

    Modulation-specfiic options can be passed in as mod_dict, while occupancy-
    specific options can be passed in as occ_dict. List of figure handles is
    returned.
    """
    figures = []
    saved = []
    nrows = len(PrimaryAreas)
    ncols = 2

    if save_dir:
        os.makedirs(save_dir)
    for abscissa, ordinate in Moment.pairs():
        f = plt.figure(figsize=(8.5, 11))
        figures.append(f)
        f.suptitle('Firing Modulation: %s x %s'%(Moment.Names[abscissa],
            Moment.Names[ordinate]))

        panel = 1
        for area in PrimaryAreas:
            gain = GainFunction(area, abscissa=abscissa,
                ordinate=ordinate)

            ax = plt.subplot(nrows, ncols, panel)
            gain.show(ax=ax, **mod_dict)
            if panel == 1:
                ax.set_ylabel(Moment.Labels[ordinate])
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([])

            ax = plt.subplot(nrows, ncols, panel+1)
            gain.show(occ=True, ax=ax, **occ_dict)
            if panel+1 == 2*nrows:
                ax.set_xlabel(Moment.Labels[abscissa])
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels([])

            panel += 2

        if save_dir:
            save_fn = os.path.join(save_dir, '%s-%s.pdf'%(abscissa,
                ordinate))
            saved.append(save_fn)
            sys.stdout.write('Saving figure %s\n'%save_fn)
            plt.savefig(save_fn)

    if save_dir:
        sys.stdout.write('Saving final report... ')
        os.system('pdftk %s cat output %s'%(' '.join(saved),
            os.path.join(save_dir, 'all_modulations.pdf')))
        sys.stdout.write('done.\n')

    return figures

def run_series(rootdir, start_fresh=False):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)

    if start_fresh:
        from scanr.data import get_root_group, get_kdata_file
        kfile = get_kdata_file(False)
        try:
            get_root_group().behavior.modulation
        except AttributeError:
            pass
        else:
            kfile.removeNode('/behavior/modulation', recursive=True)
        finally:
            kfile.createGroup('/behavior', 'modulation')

    run_sweep(os.path.join(rootdir, 'full_sweep'))

    show_modulations(save_dir=os.path.join(rootdir, 'mod_report_uncapped'))
    show_modulations(save_dir=os.path.join(rootdir, 'mod_report_1.25'), mod_dict=dict(cmax=1.25), occ_dict=dict(cmap='flag'))

    run_ND_sweep(os.path.join(rootdir, 'full_ND_sweep'), bins=11)

    ana = IndividualVariabilityAnalysis(datadir=os.path.join(rootdir, 'indiv_variability'))
    ana()
    ana.process_data()
    ana.save_plots_and_close()

    graph = ModulationGraphAnalysis(datadir=os.path.join(rootdir, 'ND_graph'))
    graph(use_ND=True)
    graph.process_data()
    graph.save_plots_and_close()
    graph.process_data(show_weak=True)
    graph.save_plots_and_close()

