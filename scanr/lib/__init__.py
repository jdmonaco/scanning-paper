# encoding: utf-8
"""
SCANR Package -- Scanning-Coordinated Analysis of Neurophysiological Recordings

Examining behavioral coordination of neurophysiological signatures in the 
Knierim Lab dataset of double cue-rotation and novelty experiments.

Author:  Joe Monaco
Contact: jmonaco@jhu.edu

Copyright 2011-2013 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

from ..config import Config
from ..meta import metadata, get_session_metadata, get_duration, get_start_end, walk_tetrodes, walk_mazes, walk_days, get_rat_list, get_day_list, get_maze_list, get_tetrode_list
from ..data import get_kdata_file, flush_file, close_file, get_root_group, get_group, new_array, new_table, dump_table, get_unique_row, get_node, unique_rats, unique_datasets, unique_sessions, unique_cells, unique_values
from ..paths import get_path, get_group_path, get_data_file_path
from ..time import time_slice, time_slice_sample, stamp_to_time, elapsed, compress_timestamps, extract_timestamps
from ..tracking import TrajectoryData, plot_track_underlay, find_laps, load_tracking_data, read_position_file, get_tracking, get_tracking_slice
from ..behavior import Moment, BehaviorDetector, Scan
from ..session import SessionData
from ..spike import get_tetrode_area, get_session_clusters, plot_correlogram, parse_cell_name, SpikePartition, TetrodeSelect
from ..cluster import string_to_quality, AND, NOT, ClusterQuality, ClusterCriteria, PlaceCellCriteria, PrincipalCellCriteria, InterneuronCriteria, get_min_quality_criterion, get_tetrode_restriction_criterion
from ..placemap import CirclePlaceMap
from ..compare import correlation_matrix, lap_correlation_matrix, correlation_diagonals
from ..neuralynx import read_event_file, read_ncs_file
from ..eeg import get_eeg_data, get_eeg_timeseries, get_eeg_file, flush_eeg_file, close_eeg_file, signal_power, total_power, FullBand, Delta, Theta, Gamma, SlowGamma, FastGamma, Ripple
from ..gui import ScanrGUI
