# encoding: utf-8
"""
scanr.behavior -- Functions for finding and characterizing movement behaviors
    in position tracking data

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os, sys
import tables as tb
import numpy as np
from numpy import logical_and as AND, logical_or as OR
from scipy.signal import firwin, medfilt
from collections import deque
from traits.api import HasTraits, Instance, Int

# Package imports
from .config import Config
from .paths import get_group_path
from .meta import walk_mazes, get_maze_list, get_day_list
from .tracking import TrajectoryData, TRACK_RADIUS, TRACK_WIDTH, smooth
from .time import elapsed, exclude_from
from .data import get_kdata_file, flush_file, get_group, new_table, get_node
from .tools.bash import CPrint
from .tools.misc import (contiguous_groups, merge_adjacent_groups, unique_pairs,
    DataSpreadsheet)
from .tools.radians import xy_to_deg_vec, circle_diff_vec_deg, shortcut_deg
from .tools.filters import filtfilt, find_minima, quick_boxcar
from .tools.stats import IQR

# Constants
DEBUG = Config['debug_mode']
CfgScan = Config['scanning']
CfgPause = Config['pauses']

ScanDescr =     {   'id'        :   tb.UInt16Col(pos=1),
                    'rat'       :   tb.UInt16Col(pos=2),
                    'day'       :   tb.UInt16Col(pos=3),
                    'session'   :   tb.UInt16Col(pos=4),
                    'type'      :   tb.StringCol(itemsize=16, pos=5),
                    'prefix'    :   tb.UInt64Col(pos=6),
                    'prepause'  :   tb.UInt64Col(pos=7),
                    'start'     :   tb.UInt64Col(pos=8),
                    'max'       :   tb.UInt64Col(pos=9),
                    'mid'       :   tb.UInt64Col(pos=10),
                    'return'    :   tb.UInt64Col(pos=11),
                    'end'       :   tb.UInt64Col(pos=12),
                    'postfix'   :   tb.UInt64Col(pos=13),
                    'tlim'      :   tb.UInt64Col(shape=(2,), pos=14),
                    'slice'     :   tb.UInt32Col(shape=(2,), pos=15),
                    'outbound'  :   tb.UInt32Col(shape=(2,), pos=16),
                    'inbound'   :   tb.UInt32Col(shape=(2,), pos=17),
                    'number'    :   tb.UInt16Col(pos=18),
                    'duration'  :   tb.Float32Col(pos=19),
                    'magnitude' :   tb.Float32Col(pos=20)   }

PauseDescr =    {   'id'        :   tb.UInt16Col(pos=1),
                    'rat'       :   tb.UInt16Col(pos=2),
                    'day'       :   tb.UInt16Col(pos=3),
                    'session'   :   tb.UInt16Col(pos=4),
                    'type'      :   tb.StringCol(itemsize=16, pos=5),
                    'start'     :   tb.UInt64Col(pos=6),
                    'end'       :   tb.UInt64Col(pos=7),
                    'tlim'      :   tb.UInt64Col(shape=(2,), pos=8),
                    'slice'     :   tb.UInt32Col(shape=(2,), pos=9),
                    'number'    :   tb.UInt16Col(pos=10),
                    'duration'  :   tb.Float32Col(pos=11)   }


# Scan points and phase definitions
ScanPoints = ('prefix', 'prepause', 'start', 'max', 'return', 'end', 'postfix')
ScanPhaseNames = ('related', 'prefix', 'pscan', 'scan', 'launch', 'outbound', 'dwell', 'inbound', 'reentry', 'postfix')
ScanPhases = {'related'   : (0,6),
              'prefix'    : (0,2),
              'pscan'     : (1,5),
              'scan'      : (2,5),
              'launch'    : (0,3),
              'outbound'  : (2,3),
              'dwell'     : (3,4),
              'inbound'   : (4,5),
              'reentry'   : (4,6),
              'postfix'   : (5,6) }

class Scan(object):

    """
    Lightweight class with scan-structure attributes
    """

    PointNames = ScanPoints
    PhaseNames = ScanPhaseNames
    PhaseIndexes = ScanPhases
    PhasePoints = { k: (ScanPoints[v[0]], ScanPoints[v[1]])
        for k, v in ScanPhases.items() }

ScanPhasePoints = Scan.PhasePoints


# Definitions of higher-order moments of behavior

MomentNames = {'hd'        :   u"Head Direction",
               'hd_speed'  :   u"HD Speed",
               'radius'    :   u"Radius",
               'rad_speed' :   u"Radial Speed",
               'fwd_speed' :   u"Forward Speed",
               'alpha'     :   u"Track Angle" }
MomentUnits = {'hd'        :   u"CCW\u00b0",
               'hd_speed'  :   u"CCW\u00b0/s",
               'radius'    :   u"cm",
               'rad_speed' :   u"cm/s",
               'alpha'     :   u"CCW\u00b0",
               'fwd_speed' :   u"CW\u00b0/s" }


class Moment(object):

    """
    Metadata of the behavioral moments computed by behavior.compute_moments()
    """

    Names = MomentNames
    Units = MomentUnits
    Labels = { k: MomentNames[k] + ", " + MomentUnits[k] for k in MomentNames }
    Wrapped = {'hd'        :   True,
               'hd_speed'  :   False,
               'radius'    :   False,
               'rad_speed' :   False,
               'fwd_speed' :   False,
               'alpha'     :   True  }

    @classmethod
    def pairs(cls):
        """Generator for pairs of moments
        """
        for abscissa, ordinate in unique_pairs(cls.keys()):
            yield (abscissa, ordinate)

    @classmethod
    def keys(cls):
        """Get sorted list of the keys for the Names, Units, Labels, and
        Wrapped class attribute dictionaries.
        """
        k = cls.Names.keys()
        k.sort()
        return tuple(k)

    @classmethod
    def get(cls, trajectory):
        """Get a dictionary of behavioral moments of the given trajectory or
        rds triplet session specifier.
        """
        if type(trajectory) is tuple and len(trajectory) == 3:
            trajectory = TrajectoryData(rds=trajectory)
        return dict(alpha=trajectory.alpha,
                    fwd_speed=trajectory.forward_velocity,
                    radius=trajectory.radius,
                    rad_speed=trajectory.radial_velocity,
                    hd=trajectory.forward_hd,
                    hd_speed=trajectory.hd_velocity)


class BehaviorDetector(object):

    """
    Index-based detection of head scanning and pausing behavior

    Methods:
    create_pause_table -- detect pauses in locomotion across the dataset,
        storing results in the /behavior/pauses table.
    create_scan_table -- detect head scans across entire KData dataset, storing
        results in the /behavior/scans table.

    Subclass-defined methods:
    compute_pause_index -- based on a trajectory, compute an index indicating
        whether each sample meets the velocity thresholds for pausing
    compute_scan_index -- given a dict of instantaneous moments computed by
        behavior.compute_moments(), return a zero-centered index that positively
        indicates whether each sample may be part of a head scan behavior
    find_max_excursion -- given the tracking indices of a detected scan, locate
        the index corresponding to the maximal excursion of the scan
    find_scan_type -- based on the relative location of the maximal excursion,
        specify the corresponding scan type (interior, exterior, ambiguous)
    """

    def create_pause_table(self, thresh=0.0, min_time=None, append=False,
        **kwds):
        """Create a table of behavioral pausing events

        Keyword arguments:
        thresh -- scan metric threshold for determining scan events
        min_time -- minimum duration for a putative pause event to be
            consided a pause; also serves as the adjacency tolerance for
            merging temporally contiguous events
        append -- whether to keep the current scans and add new ones

        Results are stored in /behavior/pauses.
        """
        out = CPrint(prefix='PauseDetector')

        if append:
            kfile = get_kdata_file()
            prev_table = kfile.copyNode('/behavior', name='pauses',
                newname='scans_old')
            kfile.removeNode('/behavior', 'pauses')
            kfile.flush()

        ptable = new_table('/behavior', 'pause_table', PauseDescr,
            title='Pause Events')
        stable = get_node('/behavior', 'scans')

        # Pause event parameter configuration
        if min_time is None:
            min_time = CfgPause['min_time']

        pause_id = 0
        pause = ptable.row
        for rds in walk_mazes():
            session_query = '(rat==%d)&(day==%d)&(session==%d)'%rds
            out.printf('Detecting pauses for rat%03d-%02d-m%d... '%rds,
                color='lightgreen')

            # Configuration and tracking data
            traj = TrajectoryData(rds=rds)
            min_samples = round(min_time*traj.fs)

            # Compute the pause index and get putative pause events
            nonscan = exclude_from(traj.ts,
                [(rec['prefix'], rec['postfix']) for rec in
                    stable.where(session_query)])
            pause_ix = self.compute_pause_index(traj, nonscan, **kwds)
            putative = contiguous_groups(pause_ix >= thresh)
            if append:
                prev_pauses = [tuple(event['slice']) for event in
                    prev_table.where(session_query)]
                putative.extend(prev_pauses)

            # Merge adjacent events, with adaptive adjacency tolerance
            pauses = merge_adjacent_groups(putative,
                tol=max(2, int(min_samples/2)))
            out.printf('%d merged... '%len(pauses), color='lightgray')

            # Store scan events in scan table
            N = 1
            ts = traj.ts
            for start, end in pauses:
                pause['id'] = pause_id
                pause['rat'] = rds[0]
                pause['day'] = rds[1]
                pause['session'] = rds[2]
                pause['start'] = ts[start]
                pause['end'] = ts[end-1]
                pause['tlim'] = ts[start], ts[end-1]
                pause['slice'] = start, end
                pause['number'] = N
                pause['duration'] = elapsed(ts[start], ts[end-1]) + 1/traj.fs
                if pause['duration'] < min_time:
                    continue
                pause.append()
                N += 1
                pause_id += 1
            if append:
                out.printf('%d new pauses\n'%((N-1)-len(prev_pauses)),
                    color='lightgray')
            else:
                out.printf('%d pauses\n'%(N-1), color='lightgray')
            ptable.flush()

        if append:
            prev_table.remove()

        flush_file()

    def compute_pause_index(self, trajectory, nonscan, max_speed=None,
        max_fwd_speed=None):
        speed = trajectory.speed
        fwd_velocity = trajectory.forward_velocity
        d_center = np.abs(trajectory.radius)

        if max_speed is None:
            max_speed = CfgPause['max_speed']
        if max_fwd_speed is None:
            max_fwd_speed = CfgPause['max_fwd_speed']

        pausing = \
            AND(nonscan,
                AND(d_center < TRACK_WIDTH/2,
                    np.sqrt((speed/max_speed)**2 + (fwd_velocity/max_fwd_speed)**2) < 1))

        pause_ix = 2*pausing.astype('i') - 1
        return pause_ix

    def create_scan_table(self, append=False, debug=False, kill_turns=True, **kwds):
        """Test a new scan detection algorithm based on a simple first-order metric

        Keyword arguments:
        append -- whether new events should be appended to the existing table
        debug -- set to rds-tuple of session to debug (no data is stored)
        kill_turns -- whether to exclude body turns as detected scan events

        Scanning event data are stored in /behavior/scans.
        """
        out = CPrint(prefix='ScanDetector')

        if debug:
            record = dict()
        else:
            if append:
                kfile = get_kdata_file()
                prev_table = kfile.copyNode('/behavior', name='scans',
                    newname='scans_old')
                kfile.removeNode('/behavior', 'scans')
                kfile.flush()

            stable = new_table('/behavior', 'scan_table', ScanDescr,
                title='Head Scan Events')

            # Spreadsheet of rejected scans
            spreadsheet = DataSpreadsheet(
                os.path.join(Config['data_root'], 'rejected_scans.csv'),
                [('rat', 'd'), ('day', 'd'), ('session', 'd'),
                ('start', 'd'), ('end', 'd'), ('reason', 's'), ('value', 'f')])
            record = spreadsheet.get_record()

            scan_id = 0
            scan = stable.row

        # Scan detection configuration
        max_angle = CfgScan['max_angle']
        min_rad = CfgScan['min_rad']

        session_list = debug and [debug] or map(tuple, walk_mazes())
        for rds in session_list:
            record['rat'], record['day'], record['session'] = rds

            # Tracking data and configured values for scan size
            traj = TrajectoryData(rds=rds)
            ts, x, y, hd = traj.tsxyhd()
            alpha = traj.alpha
            radius = traj.radius
            fs = traj.fs
            min_samples = int(CfgScan['min_time']*fs)
            max_samples = int(CfgScan['max_time']*fs)

            # Compute the head scan index for the session trajectory data
            out.printf('Detecting scans for rat%03d-%02d-m%d... '%rds)
            scan_ix = self.compute_scan_index(traj, **kwds)

            # Get the putative event list and append to previous events if
            # specified by the *append* argument
            putative_events = contiguous_groups(scan_ix >= 0.0)
            if append and not debug:
                prev_scans = [tuple(scan['slice']) for scan in
                    prev_table.where('(rat==%d)&(day==%d)&(session==%d)'%rds)]
                putative_events.extend(prev_scans)
            if debug:
                out.printf('%d putative... '%len(putative_events))

            # Expand putative scan events so start/end are at least on-track
            # NOTE: Inbound phase only expanded to track edge, so that the
            # max_angle constraint does not reject scans where animal has
            # started moving forward during inbound phase
            events = self.extend_to_track(traj, putative_events)

            # Merge overlapping events after the back-to-track extensions
            merged = merge_adjacent_groups(events, tol=min_samples)
            if debug:
                out.printf('%d merged... '%len(merged))

            # Extend inbound phase so that it most closely approaches the scan
            # starting point
            for s in merged:
                start_dist = lambda x1, y1: np.sqrt((x[s[0]]-x1)**2 + (y[s[0]]-y1)**2)
                d_last = start_dist(x[s[1]-1], y[s[1]-1])
                while ((s[1] < traj.N - 1) and (start_dist(x[s[1]], y[s[1]]) <= d_last)):
                    s[1] += 1
                    d_last = start_dist(x[s[1]-1], y[s[1]-1])

            # Merge overlapping events after the back-to-start extensions
            merged = merge_adjacent_groups(merged, tol=min_samples)

            # Impose sample size, track-angle, and radial distance constraints
            # on merged events, followed by center-crossing detection of
            # full-body turns
            side = (radius>0).astype(int)
            kill = []
            kill_descrs = []
            for s in merged:
                sl = slice(*s)
                sz = np.diff(s)
                d_alpha = shortcut_deg(alpha[s[0]], alpha[s[1]-1])
                d_rad = np.ptp(radius[sl])

                excluded_by_filters = False
                if sz < min_samples or sz > max_samples:
                    record['reason'] = 'duration'
                    record['value'] = float(sz/fs)
                    excluded_by_filters = True
                elif d_alpha > max_angle:
                    record['reason'] = 'angle_spread'
                    record['value'] = d_alpha
                    excluded_by_filters = True
                elif d_rad < min_rad:
                    record['reason'] = 'radial_spread'
                    record['value'] = d_rad
                    excluded_by_filters = True
                elif kill_turns:
                    # Detect and remove full body turns by looking for center-
                    # line crossings far from the scan initiation zone
                    crossings = np.diff(side[sl]).nonzero()[0]
                    for ix in crossings:
                        d_alpha_x = shortcut_deg(alpha[s[0]], alpha[s[0]+ix+1])
                        if d_alpha_x > max_angle:
                            record['reason'] = 'body_turn'
                            record['value'] = d_alpha_x
                            excluded_by_filters = True
                            break

                if excluded_by_filters:
                    record['start'], record['end'] = ts[s[0]], ts[s[1]-1]
                    if debug:
                        kill_descrs.append('Scan [%d, %d] killed by %s = %.3f'%(
                            record['start'], record['end'], record['reason'], record['value']))
                    else:
                        spreadsheet.write_record(record)
                    kill.append(s)

            for bad in kill:
                merged.remove(bad)

            # Final list of head scanning events
            scans = map(tuple, merged)
            if debug:
                out.printf('%d filtered...\n'%len(merged))
                out('\n'.join(kill_descrs))
                return scans
            del events, merged

            # Store scan events in scan table
            N = 1
            M_smooth = self.smooth_tracking(traj)
            for start, end in scans:
                scan['id'] = scan_id
                scan['rat'] = rds[0]
                scan['day'] = rds[1]
                scan['session'] = rds[2]
                pre_ix = self.find_scan_related(traj, start, end, which='prefix')
                scan['prefix'] = ts[pre_ix]
                scan['start'] = ts[start]
                max_ix = self.find_max_excursion(M_smooth, start, end, reverse=False)
                scan['max'] = ts[max_ix]
                scan['type'] = self.find_scan_type(radius, start, max_ix)
                scan['mid'] = long(ts[start]+(ts[end-1]-ts[start])/2)
                ret_ix = self.find_max_excursion(M_smooth, start, end, reverse=True)
                scan['return'] = ts[ret_ix]
                scan['end'] = ts[end-1]
                post_ix = self.find_scan_related(traj, start, end, which='postfix')
                scan['postfix'] = ts[post_ix]
                scan['tlim'] = ts[start], ts[end-1]
                scan['slice'] = start, end
                scan['outbound'] = start, max_ix+1
                scan['inbound'] = ret_ix, end
                scan['number'] = N
                scan['duration'] = elapsed(ts[start], ts[end-1])
                scan['magnitude'] = np.ptp(radius[start:end])
                scan.append()
                N += 1
                scan_id += 1
            if append:
                out.printf('%d new scans\n'%(len(scans) - len(prev_scans)),
                    color='lightgray')
            else:
                out.printf('%d scans\n'%len(scans), color='lightgray')
            stable.flush()

        if append:
            prev_table.remove()

        spreadsheet.close()
        flush_file()

    def compute_scan_index(self, trajectory, min_running=None, iqr_factor=1.0,
        baseline_duration=4.0):
        fwd_speed = trajectory.forward_velocity
        radius = trajectory.radius
        rad_speed = trajectory.radial_velocity

        scan_ix = np.zeros_like(trajectory.x) - 1
        base_sample = int(baseline_duration*trajectory.fs)

        if min_running is None:
            min_running = CfgPause['max_fwd_speed']

        on_track = np.abs(radius) < TRACK_WIDTH / 2
        forward_running = AND(fwd_speed > min_running, on_track)

        # Initialize queues of baseline radial behavior
        init_running = forward_running.nonzero()[0][:base_sample]
        rad_queue = deque(radius[init_running])
        rs_queue = deque(rad_speed[init_running])

        for i, running in enumerate(forward_running):

            # If running forward, add baseline samples to queues
            if running:
                rad_queue.append(radius[i])
                rs_queue.append(rad_speed[i])
                if len(rad_queue) > base_sample:
                    rad_queue.popleft()
                    rs_queue.popleft()

            # ...otherwise look for outlier lateral activity as evidence of
            # scanning behavior
            else:
                rad_iqr = IQR(rad_queue, factor=iqr_factor)
                rs_iqr = IQR(rs_queue, factor=iqr_factor)
                rad_outlier = not (rad_iqr[0] <= radius[i] <= rad_iqr[1])
                rs_outlier = not (rs_iqr[0] <= rad_speed[i] <= rs_iqr[1])
                off_track = not on_track[i]
                if rad_outlier or rs_outlier or off_track:
                    scan_ix[i] = 1

        return scan_ix

    def smooth_tracking(self, trajectory, default_ntaps=37, radius_tol=5,
        rs_tol=8, fwd_tol=10):
        """Perform adaptive low-pass (~8Hz) filter and magnitude normalization
        of several tracking signals to help detection of maximal excursion and
        scan type.
        """
        # Set up the low-pass filter for turn-around phases
        fs = trajectory.fs
        ntaps = int(default_ntaps*(1+(fs/30.0-1)/3))
        if np.fmod(ntaps, 2) == 0:
            ntaps += 1
        if fs < 10.0:
            T = 0.25
        else:
            T = CfgScan['lp_time']
        b = firwin(ntaps, 2/float(T*fs))

        # Filter and normalize moments for max excursion and type detection
        return dict(
            radius=filtfilt(b, 1, trajectory.radius) / radius_tol,
            rad_speed=filtfilt(b, 1, trajectory.radial_velocity) / rs_tol,
            fwd_speed=filtfilt(b, 1, trajectory.forward_velocity) / fwd_tol)

    def find_scan_related(self, traj, i, j, which='prefix', max_angle=7.5, max_time=1.5):
        """Find the points before (prefix) or after (postfix) a scan event
        that indicate the de/acceleration around the scan
        """
        if which == 'prefix':
            ix = i
            inc = -1
        elif which == 'postfix':
            ix = j - 1
            inc = +1
        else:
            raise ValueError, 'which must be "prefix" or "postfix": \"%s\"'%which

        alpha = traj.alpha_unwrapped
        alpha_scan = alpha[ix]

        t = traj.ts
        t_scan = t[ix]

        while True:
            if not 0 < ix < traj.N - 1:
                break

            ix += inc
            if inc * (alpha_scan - alpha[ix]) > max_angle or abs(elapsed(t_scan, t[ix])) > max_time:
                break

        return ix

    def find_max_excursion(self, M, i, j, max_tol=0.2, reverse=False):
        """Determine the point of maximal excursion for the scan specified by
        the (i, j) index pair and the pre-processing tracking data in M.
        """
        scan = slice(i, j, None)
        dr = M['radius'][scan] - M['radius'][i]
        rs = M['rad_speed'][scan]
        fwd = M['fwd_speed'][scan]

        max_ix = np.argmax(np.abs(dr))
        max_dr = dr[max_ix]
        sign_max_dr = np.sign(max_dr)
        thresh = np.abs((1-max_tol) * max_dr)

        sort = reverse and reversed or sorted

        for ix in sort(find_minima(np.sqrt(fwd**2 + rs**2))):
            if np.sign(dr[ix]) == sign_max_dr and np.abs(dr[ix]) >= thresh:
                max_ix = ix
                break

        return i + max_ix

    def find_scan_type(self, radius, start, max_ix):
        """Set the scan type based on max. excursion index
        """
        delta = radius[max_ix] - radius[start]
        if delta >= CfgScan['min_rad']:
            scan_type = CfgScan['out_type']
        elif delta <= -1*CfgScan['min_rad']:
            scan_type = CfgScan['in_type']
        else:
            scan_type = CfgScan['amb_type']
        return scan_type

    def extend_to_track(self, trajectory, events):
        d_center = np.abs(trajectory.radius)
        extended = []
        for start, end in events:
            i = j = start + np.argmax(d_center[start:end])
            while i > 0 and (d_center[i] > TRACK_WIDTH/2):
                i -= 1
            while j < trajectory.N - 1 and (d_center[j] > TRACK_WIDTH/2):
                j += 1
            extended.append([min(i, start), max(j, end)])
        return extended

    def find_prescan_pauses(self, max_time=None):
        """After initially creating the scans and pauses data tables, populate
        the prepause column of the scans table with the onset timestamp of
        a pause (if any) preceding each scan; otherwise, prepause is set to the
        scan onset so that the pre-scan interval is 0-s duration.
        """
        if max_time is None:
            max_time = CfgScan['min_time']
        max_delta = long(max_time * Config['sample_rate']['time'])

        pauses = get_node('/behavior', 'pauses')
        scans = get_node('/behavior', 'scans')

        total = 0
        found = 0
        for rds in walk_mazes():
            query = '(rat==%d)&(day==%d)&(session==%d)'%rds
            pause_list = np.array([rec['tlim'] for rec in
                pauses.where(query)], 'i8')

            for scan in scans.where(query):
                total += 1
                scan_start = long(scan['start'])
                scan['prepause'] = scan_start

                if pause_list.size:

                    delta = scan_start - pause_list[:,1]
                    match = (delta >= 0) * (delta <= max_delta)

                    if match.any():
                        found += 1
                        scan['prepause'] = np.min(pause_list[match.nonzero()[0], 0])
                        sys.stdout.write(
                            'Scan %d prepause in rat%03d-%02d-m%d is %dL.\n'%(
                                (scan['number'],)+rds+(scan['prepause'],)))

                scan.update()
            scans.flush()

        sys.stdout.write('Found %d scans with pre-pause out of %d total.\n'%(
            found, total))

        flush_file()


def fix_tracking_problems():
    """Fix problems due to bad tracking in rat 119
    """
    scan_table = get_node('/behavior', 'scans')
    for scan in scan_table.where('rat==119'):
        if scan['prefix'] > scan['start']:
            sys.stdout.write('Fixing rat%(rat)d-%(day)02d-m%(session)d '
                'prefix for scan %(number)d\n'%scan)
            scan['prefix'] = scan['start']
        if scan['postfix'] < scan['end']:
            sys.stdout.write('Fixing rat%(rat)d-%(day)02d-m%(session)d '
                'postfix for scan %(number)d\n'%scan)
            scan['postfix'] = scan['end']
        scan.update()
    scan_table.flush()


def run_behavior_detection(include_body_turns=False, iqr_threshold=0.5):
    """Recipe for recreating the behavior tables by rescanning the data set
    """
    detector = BehaviorDetector()
    detector.create_scan_table(kill_turns=(not include_body_turns), iqr_factor=iqr_threshold)
    detector.create_pause_table()
    detector.find_prescan_pauses()
    fix_tracking_problems()
