# encoding: utf-8
"""
gui.py -- A Chaco GUI for exploring head scan and spiking data

Created by Joe Monaco on 2011-10-10.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import sys
import numpy as np
import scipy

# SCANr package
from .config import Config
from .session import SessionData
from .behavior import Moment
from .cluster import (ClusterQuality, ClusterCriteria, PrincipalCellCriteria,
    PlaceCellCriteria, get_tetrode_restriction_criterion,
    get_min_quality_criterion, AND)
from .tracking import INNER_DIAMETER, OUTER_DIAMETER
from .time import time_slice, select_from
from .data import get_node
from .meta import get_rat_list, get_day_list, get_maze_list

# Tools
from .tools.bash import CPrint

# Enthought imports
from traits.api import *
from traitsui.api import (View, Item, Group, HGroup, VGroup, VFlow,
    ListStrEditor, EnumEditor, RangeEditor)
from chaco.base import n_gon
from chaco.api import (ArrayPlotData, Plot, BasePlotContainer,
    HPlotContainer)
from enable.component_editor import ComponentEditor

# Constants
CfgMod = Config['modulation']
MOMENT_KEYS = Moment.keys()
SPIKE_FMT = dict(type='scatter', marker='circle', marker_size=3.0, color='none',
    outline_color='red', line_width=1.5)
SCAN_FMT = dict(type='scatter', marker='circle', marker_size=3.5, color='violet',
    outline_color='none', line_width=0.0, alpha=0.5)
PAUSE_FMT = SCAN_FMT.copy()
PAUSE_FMT.update(color='lime')
LIST_END = ['---']


class ScanrGUI(HasTraits):

    """
    A GUI interface for looking at scanning and spiking in dimensions of space,
    time, and behavior.
    """

    out = Instance(CPrint)
    debug = true

    # Session selection traits
    rat = Int
    rat_ix = Int
    current_rat_list = List
    day = Int
    day_ix = Int
    current_day_list = List
    session = Int
    session_ix = Int
    current_session_list = List
    session_data = Instance(SessionData)

    # Cluster data selection traits
    cluster = Str
    current_cluster_list = List

    # Other selectors
    quality = Enum(ClusterQuality.__slots__)
    area = Enum(('ALL', 'CA1', 'CA3', 'DG', 'LEC', 'MEC'))

    # Criteria
    quality_criteria = Instance(ClusterCriteria)
    tetrode_criteria = Instance(ClusterCriteria)

    # Plot traits
    plot_data = Instance(ArrayPlotData)
    main_plots = Instance(HPlotContainer)
    space_plot = Instance(Plot)
    time_plot = Instance(Plot)
    scan_plot_names = List
    pause_plot_names = List
    behavior_plot = Instance(Plot)
    data_to_keep = List

    # Behavior data
    M = Dict
    abscissa = Enum(MOMENT_KEYS)
    ordinate = Enum(MOMENT_KEYS)

    # Scans and laps traits
    N_pauses = Int(1)
    N_scans = Int(1)
    N_laps = Int(1)
    pause_ix = Array(dtype='?')
    scan_ix = Array(dtype='?')
    scan_select = Bool(False)
    lap_select = Bool(False)
    scan_number = Range(low=1, high='N_scans', exclusive_low=False, exclusive_high=False)
    lap_number = Range(low=1, high='N_laps', exclusive_low=False, exclusive_high=False)
    pauses = Dict
    scans = Dict
    laps = Dict
    show_scans = Bool(True)
    show_pauses = Bool(True)

    # Cluster data
    name = String('-')
    isolation_quality = String('-')
    spike_count = Int
    running_spikes = Int
    firing_rate = String('-')
    spike_width = String('-')
    information_spike = String('-')
    p_value_spike = String('-')
    information_pos = String('-')
    p_value_pos = String('-')
    comment = String('-')
    is_place_field = String('-')

    # Time slice traits
    lock_duration = Bool(False)
    lock_adjustment = Bool(False)
    lock_delta = Long
    start = Long
    end = Long
    start_label = Str
    end_label = Str
    ts_start = Long
    ts_end = Long
    start_slice = Range(low='start', high='end', value=0)
    end_slice = Range(low='start', high='end', value=1)
    reset = Button(label='Full Session', height_padding=15)

    traits_view = \
        View(
            VGroup(
                Item('main_plots', editor=ComponentEditor(), show_label=False),
                HGroup(
                    VGroup(
                        Item('start_slice', editor=RangeEditor(mode='slider',
                            low_name='start', high_name='end',
                            label_width=100), label='Selection Start',
                            enabled_when="session_data is not None"),
                        Item('end_slice', editor=RangeEditor(mode='slider',
                            low_name='start', high_name='end',
                            label_width=100), label='Selection End',
                            enabled_when="session_data is not None"),
                        HGroup(
                            Item('start_label', label='Start Timestamp',
                                style='readonly', width=144),
                            Item('end_label', label='End Timestamp',
                                style='readonly', width=144),
                            Item('lock_duration', show_label=True,
                                enabled_when="session_data is not None")
                        ),
                        HGroup(
                            Item('current_rat_list',
                                editor=ListStrEditor(editable=False, selected_index='rat_ix',
                                title='Rat'), show_label=False),
                            VGroup(
                                Item('current_day_list',
                                    editor=ListStrEditor(editable=False, selected_index='day_ix',
                                    title='Day'), show_label=False),
                                Item('current_session_list',
                                    editor=ListStrEditor(editable=False, selected_index='session_ix',
                                    title='Session'), show_label=False)
                            ),
                            Item('current_cluster_list',
                                editor=ListStrEditor(editable=False, selected='cluster',
                                title='Clusters'), show_label=False)
                        ),
                        show_border=True,
                        label='Data Selection'
                    ),
                    HGroup(
                        VFlow(
                            Item('quality',
                                editor=EnumEditor(cols=1,
                                    values={
                                    'Good'          : '1:Good',
                                    'PrettyGood'    : '2:PrettyGood',
                                    'Fair'          : '3:Fair',
                                    'Marginal'      : '4:Marginal',
                                    'Poor'          : '5:Poor',
                                    'None'          : '6:None' }), style='simple', show_label=True),
                            Item('area',
                                editor=EnumEditor(cols=1,
                                    values={
                                    'ALL'    : '1:All',
                                    'CA1'    : '2:CA1',
                                    'CA3'    : '3:CA3',
                                    'DG'     : '4:DG',
                                    'LEC'    : '5:LEC',
                                    'MEC'    : '6:MEC' }), style='simple', show_label=True),
                            Item('lap_select', show_label=True,
                                enabled_when="session_data is not None"),
                            Item('lap_number', label='Lap#', show_label=True,
                                editor=RangeEditor(mode='spinner', low=1, high_name='N_laps'),
                                enabled_when="session_data is not None"),
                            Item('scan_select', show_label=True,
                                enabled_when="session_data is not None"),
                            Item('scan_number', label='Scan#', show_label=True,
                                editor=RangeEditor(mode='spinner', low=1, high_name='N_scans'),
                                enabled_when="session_data is not None"),
                            Item('reset', show_label=False,
                                enabled_when="session_data is not None"),
                            Item('abscissa', label='X-moment', show_label=True, style='simple',
                                enabled_when="session_data is not None"),
                            Item('ordinate', label='Y-moment', show_label=True, style='simple',
                                enabled_when="session_data is not None"),
                            Item('show_scans', show_label=True, style='simple',
                                enabled_when="session_data is not None"),
                            Item('show_pauses', show_label=True, style='simple',
                                enabled_when="session_data is not None"),
                            show_border=True,
                            label='Controls'
                        ),
                        VGroup(
                            Item('name', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')", width=-100),
                            Item('isolation_quality', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('spike_count', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('running_spikes', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('firing_rate', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('spike_width', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('information_spike', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('p_value_spike', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('information_pos', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('p_value_pos', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('comment', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            Item('is_place_field', show_label=True, style='readonly',
                                enabled_when="cluster.startswith('t')"),
                            show_border=True,
                            label='Cluster Information'
                        )
                    )
                )
            ),

            title='SCANR Data Visualizer',
            resizable=True,
            height=1,
            width=1,
            kind='live'
        )

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.debug_print('__init__')
        self.quality = "Fair"
        self.area = "ALL"
        self.abscissa = 'fwd_speed'
        self.ordinate = 'radius'
        self._reset_time()

    # Debug messages to console

    def debug_print(self, msg):
        if self.debug:
            _oldcolor = self.out.color
            self.out.color = 'green'
            self.out(msg, prefix='ScanrGuiDebug')
            self.out.color = _oldcolor

    # Plot handling

    def _plot_data_default(self):
        self.debug_print('_plot_data_default')
        data = ArrayPlotData()
        data['abscissa'] = []
        data['ordinate'] = []
        data['time'] = []
        data['axspan_y_lim'] = [-800, 800, 800, -800]
        self.data_to_keep.append('axspan_y_lim')
        data['x'] = []
        data['y'] = []
        data['scan_x'] = []
        data['scan_y'] = []
        data['pause_x'] = []
        data['pause_y'] = []
        data['cl_abscissa'] = []
        data['cl_ordinate'] = []
        data['cl_time'] = []
        data['cl_x'] = []
        data['cl_y'] = []
        return data

    def _space_plot_default(self):
        self.debug_print('_space_plot_default')
        p = Plot(self.plot_data)

        # Plot track underlay
        x_outer, y_outer = np.transpose(n_gon((0,0), OUTER_DIAMETER/2, 48))
        x_inner, y_inner = np.transpose(n_gon((0,0), INNER_DIAMETER/2, 48))
        self.plot_data['x_outer'] = x_outer
        self.plot_data['y_outer'] = y_outer
        self.plot_data['x_inner'] = x_inner
        self.plot_data['y_inner'] = y_inner
        self.data_to_keep.extend(['x_outer', 'y_outer', 'x_inner', 'y_inner'])
        p.plot(('x_outer', 'y_outer'), type='polygon', edge_width=1.0,
            edge_color='darkgrey', face_color='linen')
        p.plot(('x_inner', 'y_inner'), type='polygon', edge_width=1.0,
            edge_color='darkgrey', face_color='white')

        p.plot(('pause_x', 'pause_y'), **PAUSE_FMT)
        p.plot(('scan_x', 'scan_y'), **SCAN_FMT)
        p.plot(('x', 'y'))
        p.plot(('cl_x', 'cl_y'), **SPIKE_FMT)
        p.title = 'Space'
        p.x_axis.title = 'X (cm)'
        p.y_axis.title = 'Y (cm)'
        p.padding_bottom = 55
        p.range2d.set_bounds((-50, -50), (50, 50))
        return p

    def _time_plot_default(self):
        self.debug_print('_time_plot_default')
        p = Plot(self.plot_data)
        p.plot(('time', 'ordinate'))
        p.plot(('cl_time', 'cl_ordinate'), **SPIKE_FMT)
        p.title = 'Time'
        p.x_axis.title = 'Time (s)'
        p.padding_bottom = 55
        return p

    def _behavior_plot_default(self):
        self.debug_print('_behavior_plot_default')
        p = Plot(self.plot_data)
        p.plot(('abscissa', 'ordinate'), type='scatter', marker='dot',
            marker_size=2.0, line_width=0, color='black')
        p.plot(('cl_abscissa', 'cl_ordinate'), **SPIKE_FMT)
        p.title = 'Behavior'
        p.padding_bottom = 55
        return p

    def _main_plots_default(self):
        self.debug_print('_main_plots_default')
        return HPlotContainer(self.space_plot, self.time_plot,
            self.behavior_plot)

    def _update_space_plot(self):
        self.debug_print('_update_space_plot')
        self.space_plot.request_redraw()

    def _delete_pause_plots(self):
        self.debug_print('_delete_pause_plots')
        if self.pause_plot_names:
            self.time_plot.delplot(*self.pause_plot_names)
            self.pause_plot_names = []

    def _update_pause_plots(self):
        if not self.show_pauses:
            return
        self.debug_print('_update_pause_plots')
        self._delete_pause_plots()
        if self.session_data is None:
            return
        salt = hash(self.session_data)
        for i, pause in enumerate(self.session_data.pause_list):
            pause_x_lim = 'pausex_%d%02d'%(salt,i)
            t_start, t_end = self.session_data.T_(pause)
            self.plot_data[pause_x_lim] = [t_start, t_start, t_end, t_end]
            plot_name = 'pause%02d'%i
            self.pause_plot_names.append(plot_name)
            self.time_plot.plot(
                (pause_x_lim, 'axspan_y_lim'),
                type='polygon', name=plot_name,
                edge_width=0.0, face_color=PAUSE_FMT['color'],
                alpha=PAUSE_FMT['alpha'])

    def _delete_scan_plots(self):
        self.debug_print('_delete_scan_plots')
        if self.scan_plot_names:
            self.time_plot.delplot(*self.scan_plot_names)
            self.scan_plot_names = []

    def _update_scan_plots(self):
        if not self.show_scans:
            return
        self.debug_print('_update_scan_plots')
        self._delete_scan_plots()
        if self.session_data is None:
            return
        salt = hash(self.session_data)
        for i, scan in enumerate(self.session_data.scan_list):
            scan_x_lim = 'scanx_%d%02d'%(salt,i)
            t_start, t_end = self.session_data.T_(scan)
            self.plot_data[scan_x_lim] = [t_start, t_start, t_end, t_end]
            plot_name = 'scan%02d'%i
            self.scan_plot_names.append(plot_name)
            self.time_plot.plot(
                (scan_x_lim, 'axspan_y_lim'),
                type='polygon', name=plot_name,
                edge_width=0.0, face_color=SCAN_FMT['color'],
                alpha=SCAN_FMT['alpha'])

    def _update_time_plot(self):
        self.debug_print('_update_time_plot')
        self.time_plot.y_axis.title = '%s (%s)'%(
            Moment.Names[self.ordinate],
            Moment.Units[self.ordinate])
        try:
            self.time_plot.range2d.x_range.set_bounds(
                self.plot_data['time'][0], self.plot_data['time'][-1])
        except IndexError:
            pass
        self.time_plot.range2d.y_range.set_bounds(
            CfgMod[self.ordinate+'_min'], CfgMod[self.ordinate+'_max'])
        self.time_plot.request_redraw()

    def _update_behavior_plot(self):
        self.debug_print('_update_behavior_plot')
        self.behavior_plot.x_axis.title = '%s (%s)'%(
            Moment.Names[self.abscissa],
            Moment.Units[self.abscissa])
        self.behavior_plot.y_axis.title = '%s (%s)'%(
            Moment.Names[self.ordinate],
            Moment.Units[self.ordinate])
        self.behavior_plot.range2d.set_bounds(
            (CfgMod[self.abscissa+'_min'], CfgMod[self.ordinate+'_min']),
            (CfgMod[self.abscissa+'_max'], CfgMod[self.ordinate+'_max']))
        self.behavior_plot.request_redraw()

    def _update_plots(self):
        self.debug_print('_update_plots')
        if self.session_data is None:
            return

        traj = self.session_data.trajectory
        tslice = time_slice(traj.ts, start=self.ts_start, end=self.ts_end)

        # Trajectory data
        self.plot_data['abscissa'] = self.M[self.abscissa][tslice]
        self.plot_data['ordinate'] = self.M[self.ordinate][tslice]
        self.plot_data['time'] = self.session_data.T_(traj.ts[tslice])
        self.plot_data['x'] = traj.x[tslice]
        self.plot_data['y'] = traj.y[tslice]
        if self.show_scans:
            self.plot_data['scan_x'] = self.plot_data['x'][self.scan_ix[tslice]]
            self.plot_data['scan_y'] = self.plot_data['y'][self.scan_ix[tslice]]
        else:
            self.plot_data['scan_x'] = self.plot_data['scan_y'] = []
        if self.show_pauses:
            self.plot_data['pause_x'] = self.plot_data['x'][self.pause_ix[tslice]]
            self.plot_data['pause_y'] = self.plot_data['y'][self.pause_ix[tslice]]
        else:
            self.plot_data['pause_x'] = self.plot_data['pause_y'] = []

        # Cluster data
        if self.cluster.startswith('t'):
            cluster = self.session_data.cluster_data(self.cluster)
            sp_slice = time_slice(cluster.spikes,
                start=self.ts_start, end=self.ts_end)
            t_spikes = self.session_data.T_(cluster.spikes[sp_slice])

            F_abs = self.session_data.F_(self.abscissa)
            F_ord = self.session_data.F_(self.ordinate)

            self.plot_data['cl_abscissa'] = F_abs(t_spikes)
            self.plot_data['cl_ordinate'] = F_ord(t_spikes)
            self.plot_data['cl_time'] = t_spikes
            self.plot_data['cl_x'] = cluster.x[sp_slice]
            self.plot_data['cl_y'] = cluster.y[sp_slice]
        else:
            self.plot_data['cl_abscissa'] = []
            self.plot_data['cl_ordinate'] = []
            self.plot_data['cl_time'] = []
            self.plot_data['cl_x'] = []
            self.plot_data['cl_y'] = []

        self._update_space_plot()
        self._update_time_plot()
        self._update_behavior_plot()

    # Data slicing

    def _lock_duration_changed(self, locked):
        if locked:
            self.lock_delta = self.end_slice - self.start_slice

    def _lap_select_changed(self, new):
        if new:
            self.scan_select = False
            self.lap_number = 2
            self.lap_number = 1

    def _scan_select_changed(self, new):
        if new:
            self.lap_select = False
            self.scan_number = 2
            self.scan_number = 1

    def _lap_number_changed(self, old, new):
        self.debug_print('_lap_number_changed')
        if self.lap_select:
            self.lock_duration = False
            lap_start, lap_end = self.laps[self.lap_number]
            if old > new:
                self.start_slice = lap_start - self.session_data.start
                self.end_slice = lap_end - self.session_data.start
            else:
                self.end_slice = lap_end - self.session_data.start
                self.start_slice = lap_start - self.session_data.start
            self.out('Lap %d: %i, %i'%(self.lap_number,
                self.ts_start, self.ts_end))

    def _scan_number_changed(self, new):
        self.debug_print('_scan_number_changed')
        if self.scan_select:
            self.lock_duration = False
            scan_start, scan_end = self.scans[self.scan_number]
            self.end_slice = self.session_data.end - self.session_data.start
            self.start_slice = scan_start - self.session_data.start
            self.end_slice = scan_end - self.session_data.start
            self.out('Scan %d: %i, %i'%(self.scan_number,
                self.ts_start, self.ts_end))

    def _update_scans(self):
        self.debug_print('_update_scans')
        stable = get_node('/behavior', 'scans')
        for scan in stable.where(self.session_data.session_query):
            self.scans[scan['number']] = tuple(scan['tlim'])
        self.N_scans = len(self.scans)
        self.out('Found %d scans'%self.N_scans)

    def _update_laps(self):
        self.debug_print('_update_laps')
        self.N_laps = self.session_data.N_laps
        for lap in xrange(1, self.N_laps+1):
            self.laps[lap] = (self.session_data.laps[lap-1],
                self.session_data.laps[lap])
        self.out('Found %d laps'%self.N_laps)

    # Data visibility toggline

    def _show_scans_changed(self, new):
        if new:
            self._update_scan_plots()
        else:
            self._delete_scan_plots()
        self._update_plots()

    def _show_pauses_changed(self, new):
        if new:
            self._update_pause_plots()
        else:
            self._delete_pause_plots()
        self._update_plots()

    # Data selector handling

    def _current_rat_list_default(self):
        self.debug_print('_current_rat_list_default')
        return get_rat_list()

    def _rat_ix_changed(self):
        self.debug_print('_rat_ix_changed')
        self.rat = int(self.current_rat_list[self.rat_ix])
        self.out("Selected rat %d"%self.rat)
        self.current_day_list = get_day_list(self.rat) + LIST_END
        self.day = 0
        self.current_session_list = LIST_END
        self._reset_session()

    def _day_ix_changed(self):
        self.debug_print('_day_ix_changed')
        if self.current_day_list[self.day_ix] == LIST_END[0]:
            return
        self.day = int(self.current_day_list[self.day_ix])
        self.out("Selected day %d"%self.day)
        self.current_session_list = \
            get_maze_list(self.rat, self.day) + LIST_END
        self._reset_session()

    def _session_ix_changed(self):
        self.debug_print('_session_ix_changed')
        if self.current_session_list[self.session_ix] == LIST_END[0]:
            return
        self.session = int(self.current_session_list[self.session_ix])
        self.out("Loading data from rat%03d-%02d maze %d"%(self.rat, self.day,
            self.session))
        self.session_data = SessionData.get((self.rat, self.day, self.session))
        if self.cluster not in self.session_data.clusts:
            self.cluster = ''
        self._reset_time(self.session_data.end - self.session_data.start)
        self._update_cluster_information()
        self._update_cluster_list()
        self._update_behavior()
        self._update_scans()
        self._update_laps()
        self._update_plots()

    def _reset_session(self):
        self.debug_print('_reset_session')
        self.session = 0
        self.current_cluster_list = []
        self.cluster = ''
        self._update_cluster_information()
        self.lap_number = self.scan_number = 1
        self.N_laps = self.N_scans = 1
        self.scans = {}
        self.laps = {}
        self._delete_scan_plots()
        self._delete_pause_plots()
        self.session_data = None
        self._reset_time()
        for k in self.plot_data.arrays.keys():
            if k not in self.data_to_keep:
                self.plot_data[k] = []

    def _reset_time(self, high=1000):
        self.debug_print('_reset_time')
        self.lock_duration = False
        self.start, self.end = 0, high
        self.start_slice, self.end_slice = 1, self.end
        self.start_slice = 0

    def _start_slice_changed(self, old, new):
        self.debug_print('_start_slice_changed')
        if self.session_data:
            if new < self.end_slice:
                self.ts_start = long(self.session_data.start + new)
            if self.lock_adjustment:
                self.lock_adjustment = False
            elif self.lock_duration:
                self.lock_adjustment = True
                self.end_slice = min(self.end, new+self.lock_delta)
        else:
            self.ts_start = self.start
        self.start_label = '%iL'%self.ts_start
        if self.session_data is not None and self.abscissa in self.M:
            self._update_plots()

    def _end_slice_changed(self, old, new):
        self.debug_print('_end_slice_changed')
        if self.session_data:
            if new > self.start_slice:
                self.ts_end = long(self.session_data.start + new)
            if self.lock_adjustment:
                self.lock_adjustment = False
            elif self.lock_duration:
                self.lock_adjustment = True
                self.start_slice = max(0, new-self.lock_delta)
        else:
            self.ts_end = self.end
        self.end_label = '%iL'%self.ts_end
        if self.session_data is not None and self.abscissa in self.M:
            self._update_plots()

    def _reset_fired(self):
        self.debug_print('_reset_fired')
        self.lap_select = self.scan_select = False
        self._reset_time(self.end)

    def _update_behavior(self):
        self.debug_print('_update_behavior')
        traj = self.session_data.trajectory
        self.M = Moment.get(traj)
        self.scan_ix = select_from(traj.ts, self.session_data.scan_list)
        self.pause_ix = select_from(traj.ts, self.session_data.pause_list)
        self._update_scan_plots()
        self._update_pause_plots()

    def _abscissa_changed(self):
        self.debug_print('_abscissa_changed')
        self.out("Setting X-moment to %s"%self.abscissa)
        self._update_plots()

    def _ordinate_changed(self):
        self.debug_print('_ordinate_changed')
        self.out("Setting Y-moment to %s"%self.ordinate)
        self._update_plots()

    # Cluster criteria handling

    def _cluster_changed(self):
        self.debug_print('_cluster_changed')
        if self.cluster.startswith('t'):
            self.out("Cluster selection: %s"%self.cluster)
            self._update_cluster_information()
        self._update_plots()

    def _update_cluster_information(self):
        self.debug_print('_update_cluster_information')
        if self.cluster.startswith('t'):
            cl = self.session_data.cluster_data(self.cluster)
            self.name = cl.name
            self.spike_count = cl.N
            self.running_spikes = cl.N_running
            self.isolation_quality = str(cl.quality)
            self.firing_rate = '%.3f spks/s'%cl.mean_firing_rate
            self.spike_width = '%.3f ms'%cl.spike_width
            self.information_spike = '%.3f bits/spk'%cl.I_spike
            self.p_value_spike = '<%.5f'%cl.spike_p_value
            self.information_pos = '%.3f bits'%cl.I_pos
            self.p_value_pos = '<%.5f'%cl.pos_p_value
            self.comment = cl.comment
            self.is_place_field = PlaceCellCriteria.filter(cl) and 'Yes' or 'No'
        else:
            self.name = self.isolation_quality = self.comment = '-'
            self.spike_count = self.running_spikes = 0
            self.firing_rate = self.spike_width = self.information_spike = \
                self.information_pos = self.is_place_field = \
                self.p_value_spike = self.p_value_pos = '-'

    def _quality_changed(self):
        self.debug_print('_quality_changed')
        self._update_cluster_list()

    def _area_changed(self):
        self.debug_print('_area_changed')
        self._update_cluster_list()

    def _get_quality_criterion(self):
        self.debug_print('_get_quality_criterion')
        return get_min_quality_criterion(self.quality)

    def _get_tetrode_criterion(self):
        self.debug_print('_get_tetrode_criterion')
        ttable = get_node('/metadata', 'tetrodes')
        def do_query(condn):
            tetrodes = [rec['tt'] for rec in ttable.where(condn)]
            return tetrodes
        query = "(rat==%d)&(day==%d)"%(self.rat, self.day)
        if self.area != "ALL":
            query += "&(area==\'%s\')"%self.area
        return get_tetrode_restriction_criterion(do_query(query))

    def _update_cluster_list(self):
        self.debug_print('_update_cluster_list')
        if self.session_data:
            self.current_cluster_list = \
                self.session_data.get_clusters(
                    AND(PrincipalCellCriteria,
                        self._get_quality_criterion(),
                        self._get_tetrode_criterion())) + LIST_END

    def _out_default(self):
        return CPrint(prefix='ScanrGUI', color='cyan')
