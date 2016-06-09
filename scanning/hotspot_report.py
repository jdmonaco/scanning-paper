#!/usr/bin/env python
# encoding: utf-8
"""
hotspot_report.py -- Look for modes in the distribution of place fields and head
    scan locations.

Created by Joe Monaco on 2012-02-03.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scanr.lib import get_node, CirclePlaceMap
from scanr.spike import HippocampalAreas, TetrodeSelect
from scanr.cluster import (AND, get_min_quality_criterion, PlaceCellCriteria,
    get_tetrode_restriction_criterion)
from .reports import BaseDatasetReport
from .tools.radians import xy_to_rad_vec


def process_area_argument(area):
    """Get list of areas from an optional recording area argument
    """
    if area is None or area == 'ALL':
        area = HippocampalAreas
    elif type(area) is str and area in spike.PrimaryAreas:
        area = [area]
    else:
        assert type(area) is list, "bad area specification"
    return area

def get_area_query(area_list):
    return '|'.join(["(area=='%s')"%area for area in area_list])

def display_histograms(ax, bins, scan_data, field_data):
    """Display radial histograms, returning scanning mode
    """
    H_scans, edges = np.histogram(scan_data, bins=bins)
    H_scans = H_scans.astype('d')
    H_scans = np.r_[H_scans, H_scans[0]] / H_scans.max()

    H_COMs, edges = np.histogram(field_data, bins=bins)
    H_COMs = H_COMs.astype('d')
    H_COMs = np.r_[H_COMs, H_COMs[0]] / H_COMs.max()

    plt.polar(bins, H_scans, 'b-', drawstyle='steps-pre')
    plt.polar(bins, H_COMs, 'r-', drawstyle='steps-pre')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_rlim(0, 1.1)
    ax.set_rgrids([0.5, 1])
    ax.set_thetagrids([45, 135, 225, 315])

    return bins[np.argmax(H_scans)]


class FieldScanDistributionReport(BaseDatasetReport, TetrodeSelect):

    """
    Compute radial distributions of head scanning events and place fields
    """

    label = 'hotspot report'

    nrows = 5
    polar = True
    xnorm = False
    ynorm = False

    def collect_data(self, area='ALL', min_quality='fair', bins=16):
        """Collect distributions of place field COMs and scan locations

        Arguments:
        area -- string name or list of names of hippocampal recording areas,
            or 'ALL' for all hippocampal areas, defaults to 'ALL'
        min_quality -- minimum cluster isolation quality for including cells
        bins -- numbers of bins for the histogram outputs
        """
        area = process_area_argument(area)
        self.out('Processing recording areas: %s'%str(area)[1:-1])
        self.results['area'] = area
        self.results['quality'] = min_quality

        scan_table = get_node('/behavior', 'scans')
        area_query = get_area_query(area)

        Quality = get_min_quality_criterion(min_quality)
        self.out('Quality threshold: at least %s'%min_quality)

        hist_bins = np.linspace(0, 2*np.pi, bins+1)

        overall_scans = np.array([], 'd')
        overall_fields = np.array([], 'd')
        aligned_scans = np.array([], 'd')
        aligned_fields = np.array([], 'd')

        for dataset, ax in self.get_plot(self._get_datasets(area_query)):
            rat, day = dataset
            self.out('Collating distributions for rat%03d-%02d dataset...'%dataset)

            # Initialize dataset accumulators
            alpha_scans_all = np.array([], 'd')
            alpha_COMs_all = np.array([], 'd')

            # Set cluster criteria
            Tetrodes = get_tetrode_restriction_criterion(
                self._get_valid_tetrodes(dataset, area_query))
            Criteria = AND(Quality, Tetrodes, PlaceCellCriteria)

            for session in data.session_list(rat, day):

                # Load session data
                rds = rat, day, session
                session_data = SessionData(rds=rds,
                    cluster_criteria=Criteria)
                if len(session_data.get_clusters()) == 0:
                    self.out('No matching clusters, skipping...')
                    continue
                trajectory = session_data.trajectory

                # Interpolate scan-start track-angle positions
                t = session_data.to_time(trajectory.ts)
                alpha = xy_to_rad_vec(trajectory.x, trajectory.y)
                F_alpha = session_data.F_('alpha')
                ts_scans = [scan['start'] for scan in
                    scan_table.where(session_data.session_query)]
                t_scans = session_data.to_time(ts_scans)
                alpha_scans = F_alpha(t_scans)

                # Get place-field COM track-angle positions
                pmap = CirclePlaceMap(session_data)
                alpha_COMs = pmap.COM_fields

                # Collate data into dataset accumulators
                alpha_scans_all = np.r_[alpha_scans_all, alpha_scans]
                alpha_COMs_all = np.r_[alpha_COMs_all, alpha_COMs]

            N_scans = len(alpha_scans_all)
            N_fields = len(alpha_COMs_all)
            self.out('Processed %d scans and %d place fields.'%(
                N_scans, N_fields))
            if N_scans == 0 or N_fields == 0:
                ax.set_axis_off()
            else:
                mode = display_histograms(ax, hist_bins, alpha_scans_all,
                    alpha_COMs_all)
                overall_scans = np.r_[overall_scans, alpha_scans_all]
                overall_fields = np.r_[overall_fields, alpha_COMs_all]

                alpha_scans0 = alpha_scans_all - mode
                alpha_scans0[alpha_scans0<0] += 2*np.pi
                alpha_fields0 = alpha_COMs_all - mode
                alpha_fields0[alpha_fields0<0] += 2*np.pi

                aligned_scans = np.r_[aligned_scans, alpha_scans0]
                aligned_fields = np.r_[aligned_fields, alpha_fields0]

        # Save data
        self.out('Saving overall and mode-aligned data...')
        savedir = os.path.join(self.datadir, 'data')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(os.path.join(savedir, 'overall_scans.npy'), overall_scans)
        np.save(os.path.join(savedir, 'overall_fields.npy'), overall_fields)
        np.save(os.path.join(savedir, 'aligned_scans.npy'), aligned_scans)
        np.save(os.path.join(savedir, 'aligned_fields.npy'), aligned_fields)

        # Good-bye
        self.out('All done!')

    def process_data(self, bins=36):
        plt.ioff()
        hist_bins = np.linspace(0, 2*np.pi, bins+1)
        self.figure = {}
        self.figure['overall'] = f = plt.figure(figsize=(11, 7))

        # Load data
        self.out('Loading overall and mode-aligned data...')
        savedir = os.path.join(self.datadir, 'data')
        overall_scans = np.load(os.path.join(savedir, 'overall_scans.npy'))
        overall_fields = np.load(os.path.join(savedir, 'overall_fields.npy'))
        aligned_scans = np.load(os.path.join(savedir, 'aligned_scans.npy'))
        aligned_fields = np.load(os.path.join(savedir, 'aligned_fields.npy'))

        # Plot overall radial histograms
        ax = plt.subplot(1, 2, 1, polar=True)
        display_histograms(ax, hist_bins, overall_scans, overall_fields)
        ax.set_title('Overall')

        # Plot aligned radial histograms
        ax = plt.subplot(1, 2, 2, polar=True)
        display_histograms(ax, hist_bins, aligned_scans, aligned_fields)
        ax.set_title('Overall - Mode Aligned')

        plt.ion()
        plt.show()


if __name__ == "__main__":
    for area in 'DG', 'CA1', 'CA3', 'ALL':
        report = FieldScanDistributionReport(desc=area)
        report(area=area)
        report.process_data()
        report.save_plots_and_close()
