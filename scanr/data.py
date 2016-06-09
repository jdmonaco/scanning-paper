# encoding: utf-8
"""
scanr.data -- Interfaces, schema definitions and functions for handling HDF data

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import sys
import tables as tb
import numpy as np
from os.path import exists as there

# Package imports
from .config import Config
from .paths import get_data_file_path, get_group_path
from .meta import metadata, get_rat_list, get_day_list, get_maze_list, walk_mazes
from .tools.misc import DataSpreadsheet

# Constants
CfgData = Config['h5']

# Top-level pointer to the H5 data file object
if 'kdata_file' not in locals():
    kdata_file = None

# Table schema
SessionDescr = {    "rat"           :   tb.UInt16Col(pos=1),
                    "day"           :   tb.UInt16Col(pos=2),
                    "session"       :   tb.UInt16Col(pos=3),
                    "start"         :   tb.UInt64Col(pos=4),
                    "end"           :   tb.UInt64Col(pos=5),
                    "exptr"         :   tb.StringCol(itemsize=8, pos=6),
                    "expt_type"     :   tb.StringCol(itemsize=8, pos=7), # DR | NOV
                    "type"          :   tb.StringCol(itemsize=16, pos=8),
                    "track"         :   tb.StringCol(itemsize=8, pos=9), # circle | hexagon
                    "parameter"     :   tb.FloatCol(pos=10),
                    "number"        :   tb.UInt16Col(pos=11),
                    "start_comment" :   tb.StringCol(itemsize=128, pos=12),
                    "end_comment"   :   tb.StringCol(itemsize=128, pos=13),
                    "missing_HD"    :   tb.BoolCol(pos=14), # tracking HD?
                    "timing_issue"  :   tb.BoolCol(pos=15)  # timestamps in order?
                }


def _open_file():
    return bool(type(kdata_file) is tb.File and kdata_file.isopen)

def dump_table(where, name=None, condn=None, filename=None, sep=','):
    """Dump data table out to a spreadsheet

    Optionally use a table search query by setting *condn* keyword to either
    a query string or callable that maps row->Bool.
    """
    if type(where) is tb.Table:
        table = where
    else:
        table = get_node(where, name)

    if filename is None:
        filename = "%s_table.csv"%table.name

    colnames = table[0].dtype.names
    coltypes = []
    for col in colnames:
        typename = table.coltypes[col]
        if bool(len(table.coldtypes[col].shape)): coltypes.append('s')
        elif typename.find('int') >= 0: coltypes.append('d')
        elif typename.find('bool') >= 0: coltypes.append('d')
        elif typename.find('float') >= 0: coltypes.append('f')
        elif typename.find('str') >= 0: coltypes.append('s')
        else: coltypes.append('s')
    spreadsheet = DataSpreadsheet(filename, zip(colnames, coltypes))
    record = spreadsheet.get_record()

    def save_row(row):
        for name, coltype in zip(colnames, coltypes):
            if coltype == 's' or type(row[name]) is np.ndarray:
                record[name] = str(row[name]).replace(',', ';')
            elif coltype == 'd':
                record[name] = int(row[name])
            elif coltype == 'f':
                record[name] = float(row[name])
        spreadsheet.write_record(record)

    if condn is None:
        for row in table.iterrows():
            save_row(row)
    elif type(condn) is str:
        for row in table.where(condn):
            save_row(row)
    elif callable(condn):
        for row in table.iterrows():
            if condn(row):
                save_row(row)

    sys.stdout.write('Dumped %s to %s.\n'%(table._v_pathname, filename))
    spreadsheet.close()

def new_array(where, name, obj, title='', force=False, eeg=False):
    """Create a new Array leaf in the tree, erasing an old one if it exists

    Arguments:
    where -- group path or node where array should be created
    name -- array name
    obj -- array object to store in the data file
    force -- force erase previous array without asking

    Returns new Array node object.
    """
    if eeg:
        from .eeg import get_eeg_file
        kfile = get_eeg_file(False)
        if kfile.mode == 'r':
            kfile.close()
            kfile = get_eeg_file(False)
    else:
        kfile = get_kdata_file()

    array = None
    try:
        array = kfile.getNode(where, name)
    except tb.NodeError:
        pass
    else:
        if force:
            sys.stdout.write('new_array: Erasing previous \'%s\' array.\n'%name)
            do_erase = 'y'
        else:
            do_erase = raw_input('Found previous \'%s\' array. Erase? (y/N) '%name)
        if do_erase.lower().strip().startswith('y'):
            kfile.removeNode(array)
            array = None
    finally:
        if not array:
            array = kfile.createArray(where, name, obj, title=title)
    return array

def new_table(where, name, descr, title='', force=False):
    """Create a new Table leaf in the tree, erasing an old one if it exists

    Arguments:
    where -- group path or node where table should be created
    name -- key value from config 'h5' section for a table name
    descr -- table column descriptor
    title -- optional title for the new table
    force -- force erasure of a pre-existing table without asking

    Returns new Table node object.
    """
    kfile = get_kdata_file()
    table = None

    # Check for configured table name first, check with user otherwise
    try:
        table_name = CfgData[name]
    except KeyError:
        try_name = name.lower().strip()
        do_create = \
            raw_input('Not a configured table.\nCreate \'%s\' under \'%s\'? (y/N) '%
                (try_name, str(where)))
        if do_create.lower().strip().startswith('y'):
            table_name = try_name
        else:
            return None

    # Check for pre-existing table and query user for deletion if not forced
    try:
        table = kfile.getNode(where, table_name)
    except tb.NodeError:
        pass
    else:
        if force:
            sys.stdout.write('new_table: Erasing previous \'%s\' table.\n'%
                table_name)
            do_erase = 'y'
        else:
            do_erase = raw_input('Found previous \'%s\' table. Erase? (y/N) '%
                table_name)
        if do_erase.lower().strip().startswith('y'):
            kfile.removeNode(table)
            table = None
    finally:
        if not table:
            table = kfile.createTable(where, table_name, descr, title=title)
        kfile.flush()

    return table

def get_unique_row(table, query, raise_on_fail=True):
    """Get the singular, unique record in the table that matches the query

    If raise_on_fail is set to False, then multiple record results will return
    the first record, and empty results will return None. Otherwise, an
    exception will be raised in those cases.
    """
    rec = None
    where = table.getWhereList(query)
    if len(where):
        rec = table[where[0]]
        if len(where) > 1 and raise_on_fail:
            raise RuntimeError, '%s: found >1 record: \"%s\"'%(table._v_pathname,
                query)
    elif raise_on_fail:
        raise RuntimeError, '%s: found no records: \"%s\"'%(table._v_pathname,
            query)
    return rec

def unique_rats(where, name=None, condn=None):
    """Get a list of unique rats listed across the rows of the specified table.
    """
    return unique_values(where, name=name, column='rat', condn=condn)

def unique_values(where, name=None, column=None, condn=None):
    """Get a list of unique column values listed across the rows of the
    specified table.
    """
    values = []
    node = get_node(where, name)
    if column is None:
        sys.stderr.write('unique_values(%s): no column specified\n'%(
            node._v_pathname))
        return values
    try:
        col = node.colinstances[column]
    except KeyError:
        sys.stderr.write('unique_values(%s): unknown column: \'%s\'\n'%(
            node._v_pathname, col))
        return values
    converter = object
    if col.type == 'string':
        converter = str
    elif 'int' in col.type:
        converter = int
    elif 'float' in col.type:
        converter = float
    if condn is None:
        rec_gen = node.iterrows()
    else:
        rec_gen = node.where(condn)
    for rec in rec_gen:
        value = converter(rec[column])
        if value not in values:
            values.append(value)
    values.sort()
    return values

def unique_datasets(where, name=None, condn=None):
    """Get a list of unique dataset (rat, day)-tuples for the rat and day
    columns in the specified data table.
    """
    datasets = []
    node = get_node(where, name)
    if condn is None:
        rec_gen = node.iterrows()
    else:
        rec_gen = node.where(condn)
    for rec in rec_gen:
        ratday = tuple(int(rec[k]) for k in ('rat', 'day'))
        if ratday not in datasets:
            datasets.append(ratday)
    datasets.sort()
    return datasets

def unique_sessions(where, name=None, condn=None):
    """Get a list of unique sessions (rat, day, maze)-tuples for the rat, day,
    and session columns in the specified data table.
    """
    sessions = []
    node = get_node(where, name)
    if condn is None:
        rec_gen = node.iterrows()
    else:
        rec_gen = node.where(condn)
    for rec in rec_gen:
        rds = (int(rec['rat']), int(rec['day']), int(rec['session']))
        if rds not in sessions:
            sessions.append(rds)
    sessions.sort()
    return sessions

def unique_cells(where, name=None, condn=None):
    """Get a list of unique cells (rds, tc)-tuples based on rat, day,
    session, and tc columns in the specified data table.
    """
    cells = []
    node = get_node(where, name)
    if condn is None:
        rec_gen = node.iterrows()
    else:
        rec_gen = node.where(condn)
    for rec in rec_gen:
        rds = (int(rec['rat']), int(rec['day']), int(rec['session']))
        tc = str(rec['tc'])
        cell_id = (rds, tc)
        if cell_id not in cells:
            cells.append(cell_id)
    cells.sort()
    return cells

def get_node(where, name=None, eeg=False, raise_on_fail=True):
    """Get the specified Table or Array node object from the H5 tree

    Arguments:
    where -- group pathname containing node, or Node object for pass-through
    name -- node name required if where is a pathname
    raise_on_fail -- throw exception if table not found; if False, return None

    Returns Table node object.
    """
    from .eeg import get_eeg_file
    node = None
    if issubclass(where.__class__, tb.Node):
        node = where
    elif type(where) is str and type(name) is str:
        kfile = eeg and get_eeg_file() or get_kdata_file()
        try:
            node = kfile.getNode(where, CfgData[name])
        except KeyError:
            try:
                node = kfile.getNode(where, name)
            except tb.NodeError:
                sys.stderr.write(
                    'get_node: node \'%s\' is not a child of %s\n'%(name, where))
        except tb.NodeError:
            sys.stderr.write(
                'get_node: table \'%s\' does not exist\n'%CfgData[name])
        if not node and raise_on_fail:
            raise tb.NodeError, 'could not find node %s'%name
    else:
        raise ValueError, 'node object or name is required'
    return node

def _get_writeable_data_file(eeg=False):
    """Get a writeable File object for the kdata/eeg H5 data files
    """
    if eeg:
        from .eeg import get_eeg_file
        get_file = get_eeg_file
    else:
        get_file = get_kdata_file
    return get_file(False)

def _node_exists(h5file, where, name=None):
    """Test whether a given node exists in the specified H5 file
    """
    try:
        h5file.getNode(where, name)
    except tb.NoSuchNodeError:
        return False
    return True

def initialize_groups(eeg=False, overwrite=True):
    """Create the primary data group structure of the H5 file based on the
    master metadata file. Specify overwrite=False to preserve the existing
    tree and only create new groups.

    Note: run update_group_metadata after this to populate group attributes
    with the appropriate metadata.
    """
    d = dict(rat=None, day=None, session=None)
    f = _get_writeable_data_file(eeg)

    def create_group(where, name, **kw):
        if _node_exists(f, where, name):
            g = f.getNode(where, name)
            sys.stdout.write('initialize_groups: found node %s\n'%g)
        else:
            g = f.createGroup(where, name, **kw)
            sys.stdout.write('initialize_groups: created node %s\n'%g)
        return g

    # Create the hierarchical root group (if necessary)
    if _node_exists(f, '/kdata') and overwrite:
        f.removeNode('/kdata', recursive=True)
    if _node_exists(f, '/kdata'):
        kdata = f.getNode('/kdata')
    else:
        kdata = create_group('/', 'kdata', title='Knierim Lab Data')

    # Create the other top-level groups (e.g., '/metadata', '/physiology')
    if not eeg:
        if not _node_exists(f, '/metadata'):
            create_group('/', 'metadata', title='All Metadata')
        if not _node_exists(f, '/physiology'):
            create_group('/', 'physiology', title='Physiology Data')
        if not _node_exists(f, '/behavior'):
            create_group('/', 'behavior', title='Behavioral Data')

    # Walk the sessions, creating group nodes for each session
    for rat in get_rat_list():
        d['rat'] = rat
        rat_group = create_group(kdata, CfgData['rat']%d, title='Rat %d'%rat)
        for day in get_day_list(rat):
            d['day'] = day
            day_group = create_group(rat_group, CfgData['day']%d,
                title='Rat %d, Day %d'%(rat, day))
            for maze in get_maze_list(rat, day):
                d['session'] = maze
                maze_group = create_group(day_group, CfgData['session']%d,
                    title='Rat %d, Day %d, Maze %d'%(rat, day, maze))
            d['session'] = None
        d['day'] = None
    d['rat'] = None

    f.flush()
    f.close()

def update_group_metadata(eeg=False, write=True):
    """Update the group metadata attributes of main data tree in the H5 file

    Metadata is available as the _v_attrs dictionary attribute of any group.

    Note: this used to be done as part of initialize_groups(), but the metadata
    setting has been refactored out to allow for easy updating of existing
    data trees without destructively initializing the tree structure.
    """
    def update_attr(grp, key, value):
        attrs = grp._v_attrs
        if key in attrs:
            if attrs[key] != value:
                if write:
                    attrs[key] = value
                sys.stdout.write('%s: updated %s to %s\n' % (
                    grp._v_pathname, key, str(value)))
        else:
            if write:
                attrs[key] = value
            sys.stdout.write('%s: added key %s with value %s\n' % (
                grp._v_pathname, key, str(value)))

    # Walk the sessions, updating metadata attributes of group nodes
    for rat in get_rat_list():
        rat_group = get_group(eeg=eeg, rat=rat)
        update_attr(rat_group, 'exptr', metadata[rat]['exptr'])
        for day in get_day_list(rat):
            day_group = get_group(eeg=eeg, rat=rat, day=day)
            update_attr(day_group, 'type', metadata[rat]['days'][day]['type'])
            for maze in get_maze_list(rat, day):
                maze_group = get_group(eeg=eeg, rds=(rat, day, maze))
                update_attr(maze_group, 'rat', rat)
                update_attr(maze_group, 'day', day)
                update_attr(maze_group, 'session', maze)
                session_meta = metadata[rat]['days'][day]['sessions'][maze]
                for k in session_meta.keys():
                    update_attr(maze_group, k, session_meta[k])
                maze_group._f_flush()
            day_group._f_flush()
        rat_group._f_flush()
    sys.stdout.write('Finished!\n')

def add_track_geometry(do_write=True):
    """Add a 'track' keyword attribute to the session groups in the data tree
    with values of either 'circle' or 'hexagon'
    """
    for rds in walk_mazes():
        attrs = get_group(rds=rds)._v_attrs
        snote = attrs['start_comment'].lower()
        enote = attrs['end_comment'].lower()

        track = 'circle'
        if snote.find('circle') != -1 or enote.find('circle') != -1:
            pass
        elif snote.find('hex') != -1 or enote.find('hex') != -1:
            track = 'hexagon'
            sys.stdout.write('Rat %d, day %d, session %s'%rds + ' is a hexagon\n')

        if do_write:
            attrs['track'] = track

    if do_write:
        flush_file()

def create_sessions_table():
    """Create table describing all sessions in the data set in /metadata tree

    Run this *after* any metadata updates from update_group_metadata().
    """
    session_table = new_table('/metadata', 'session_table',
        SessionDescr, 'Recording Sessions Metadata')

    row = session_table.row
    for rat in get_rat_list():
        exptr = metadata[rat]['exptr']

        for day in get_day_list(rat):
            t = metadata[rat]['days'][day]['type']
            if t == "double rotation":
                expt_type = "DR"
            elif t == "novelty":
                expt_type = "NOV"

            for maze in get_maze_list(rat, day):
                attrs = get_group(rds=(rat, day, maze))._v_attrs

                row['rat'] = rat
                row['day'] = day
                row['session'] = maze
                row['start'] = attrs['start']
                row['end'] = attrs['end']
                row['exptr'] = exptr
                row['expt_type'] = expt_type
                row['type'] = attrs['type']
                row['track'] = attrs['track']
                row['parameter'] = attrs['parameter']
                row['number'] = attrs['number']
                row['start_comment'] = attrs['start_comment']
                row['end_comment'] = attrs['end_comment']
                row['missing_HD'] = attrs['HD_missing']
                row['timing_issue'] = attrs['timing_jumps']
                row.append()

    session_table.flush()

def session_list(rat, day, exclude_missing_HD=False, exclude_timing_issue=False):
    """Sorted list of session numbers for given rat and day

    Keyword arguments:
    exclude_missing_HD -- exclude sessions which are missing HD data
    exclude_timing_issue -- exclude sessions with Neuralynx timing issues
    """
    sessions = get_node('/metadata', 'sessions')
    query = '(rat==%d)&(day==%d)'%(rat, day)
    if exclude_missing_HD:
        query += '&(missing_HD==False)'
    if exclude_timing_issue:
        query += '&(timing_issue==False)'
    return sorted([int(rec['session']) for rec in sessions.where(query)])

def get_kdata_file(readonly=False):
    """Return an open H5 file object linked to the main data file
    """
    global kdata_file

    # Close the data file if it is open and there is a mode mismatch
    if _open_file():
        if      (kdata_file._isWritable() and readonly) or \
            not (kdata_file._isWritable() or readonly):
            close_file()

    # If the file is closed, open it with the specified mode ('r' or 'a')
    if not _open_file():
        kdata_path = get_data_file_path('data', search=False)
        mode = (readonly and there(kdata_path)) and 'r' or 'a'
        kdata_file = tb.openFile(kdata_path, mode=mode, title='KData')
        sys.stdout.write('kdata_file: opened data file (mode=\'%s\')\n' % kdata_file.mode)

    return kdata_file

def get_root_group():
    """Return root group for the H5 data set
    """
    return get_kdata_file(True).root

def get_group(eeg=False, **kwds):
    """Return the group node corresponding the specified group

    Specify alternative pytable File object *data_file* with data tree that is
    isomorphic to the standard kdata File.

    Keyword arguments same as for get_group_path().
    """
    # Allow an rds keyword with tuple value to be passed in
    rds = kwds.pop('rds', None)
    if rds:
        kwds.update(tree='kdata', rat=rds[0], day=rds[1], session=rds[2])

    if eeg:
        from .eeg import get_eeg_file
        data_file = get_eeg_file(False)
    else:
        data_file = get_kdata_file()

    # Get the path and return the group node if it exists
    gpath = get_group_path(**kwds)
    group = None
    try:
        group = data_file.getNode(gpath)
    except tb.NodeError:
        sys.stderr.write('get_group: group node %s does not exist\n'%gpath)
    return group

def close_file():
    """Close the H5 file if it is still open
    """
    if _open_file():
        kdata_file.close()
        sys.stdout.write('kdata_file: closed data file\n')

def flush_file():
    """Flush the H5 file if it is open
    """
    if _open_file():
        kdata_file.flush()
        sys.stdout.write('kdata_file: flushed data file\n')
