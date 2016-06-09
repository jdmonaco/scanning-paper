# encoding: utf-8
"""
scanr.meta -- Load experiment condition and tetrode location metadata

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import sys
import yaml
from os.path import exists as there

# Package imports
from .config import Config
from .paths import get_data_file_path
from .time import elapsed

# Local imports
from .tools.misc import AutoVivification
from .tools.path import unique_path


if 'metadata' not in locals():
    master_fn = get_data_file_path('master')
    try:
        fd = file(master_fn, 'r')
    except IOError, e:
        raise e, 'could not open master file: %s' % master_fn
    else:
        metadata = yaml.load(fd)
        fd.close()

def get_duration(*rds):
    """Return the total duration of the specified session
    """
    return elapsed(*get_start_end(*rds))

def get_start_end(*rds):
    """Return the start and end time-stamps for the specified recording session
    """
    meta = get_session_metadata(*rds)
    return meta['start'], meta['end']

def get_session_metadata(*rds):
    """Return the session metadata dictionary for the specified recording session
    """
    if None in rds or len(rds) != 3:
        raise ValueError, 'incomplete session specification: %s' % str(rds)
    rat, day, session = rds
    try:
        the_session = metadata[rat]['days'][day]['sessions'][session]
    except KeyError, e:
        raise e, 'session not found'
    return the_session

def get_rat_list():
    """Sorted list of rat id numbers
    """
    return sorted(metadata.keys())

def walk_rats():
    """Generator for walking rat nodes in the metadata tree
    """
    for rat in get_rat_list():
        yield rat

def get_day_list(rat):
    """Sorted list of recording day numbers for given rat
    """
    return sorted(metadata[rat]['days'].keys())

def walk_days():
    """Generator for walking day nodes in the metadata tree
    """
    for rat in walk_rats():
        for day in get_day_list(rat):
            yield (rat, day)

def get_maze_list(rat, day):
    """Sorted list of session numbers for given rat and day
    """
    return sorted(metadata[rat]['days'][day]['sessions'].keys())
get_session_list = get_maze_list

def walk_sessions(*args, **kwds):
    return walk_mazes(*args, **kwds)

def walk_mazes():
    """Generator for walking maze nodes in the metadata tree

    Keywords are passed to get_maze_list().
    """
    for rat, day in walk_days():
        for maze in get_maze_list(rat, day):
            yield (rat, day, maze)

def get_tetrode_list(rat, day):
    """Sorted list of tetrode numbers for given rat and day
    """
    return sorted(metadata[rat]['days'][day]['tetrodes'].keys())

def walk_tetrodes():
    """Generator for walking tetrode nodes in the metadata tree
    """
    for rat, day in walk_days():
        for tt in get_tetrode_list(rat, day):
            yield (rat, day, tt)

def get_tetrode_comment(rat, day, tt):
    """Return the metadata string comment associated with the specified tetrode
    """
    try:
        ttstr = metadata[rat]['days'][day]['tetrodes'][tt]
    except KeyError, e:
        sys.stderr.write('tetrode_comment: bad tetrode specification: %s' %
            str((rat, day, tt)))
        raise e
    else:
        if ttstr is None:
            ttstr = ''
        return ttstr

def load_yaml_file(which, required=True):
    """Load the metadata tree from the specified config file (one of
    'experiment', 'location', 'timing', or 'master')
    """
    fn = get_data_file_path(which)
    if not fn:
        if required:
            raise IOError, 'could not find \'%s\' file at %s' % (which, fn)
        else:
            return None
    try:
        fd = file(fn, 'r')
    except IOError:
        raise IOError, 'could not open \'%s\' file at %s' % (which, fn)
    else:
        tree = yaml.load(fd)
        fd.close()
    return tree

def merge_location_files(locfile):
    """Combine a previous location.yaml file with tetrode location information
    with the active location.yaml file
    """
    loc_tree = load_yaml_file('location') # active location file
    with file(locfile, 'r') as fd:
        old_tree = yaml.load(fd)

    for rat in loc_tree.keys():
        for day in loc_tree[rat]['days'].keys():
            try:
                tetrodes = old_tree[rat]['days'][day]['tetrodes']
            except KeyError:
                pass
            else:
                loc_tree[rat]['days'][day]['tetrodes'] = tetrodes

    write_yaml_file(loc_tree, 'location')

def merge_master_file():
    """Combine experiment and location YAML files into single master file
    """
    exp_tree = load_yaml_file('experiment')
    loc_tree = load_yaml_file('location')
    add_tree = load_yaml_file('addins', required=False)
    timing_tree = load_yaml_file('timing', required=False)

    try:

        for rat in exp_tree:
            exp = exp_tree[rat]
            loc = loc_tree[rat]

            sys.stdout.write('Rat %d:\n' % rat)
            sys.stdout.write('  1) Experiment <- Location\n')
            for day in exp['days']:
                rdstr = 'rat%03d-%02d' % (rat, day)
                exp['days'][day]['tetrodes'] = loc['days'][day]['tetrodes']
                sys.stdout.write('   - Merged tetrode locations for %s\n' % rdstr)

            sys.stdout.write('  2) Master <- Addins (First Pass)\n')
            if add_tree and rat in add_tree:
                add = add_tree[rat]
                for day in add['days']:
                    rdstr = 'rat%03d-%02d' % (rat, day)
                    exp['days'][day] = add['days'][day]
                    sys.stdout.write('   - Merged %s from addins\n' % rdstr)

            sys.stdout.write('  3) Master <- Timing\n')
            if timing_tree and rat in timing_tree:
                timing = timing_tree[rat]
                for day in timing['days'].keys():
                    rdstr = 'rat%03d-%02d' % (rat, day)
                    sys.stdout.write('   - Updating timing for %s\n' % rdstr)
                    for maze in timing['days'][day]['sessions'].keys():
                        for k in ('start', 'end'):
                            master_value = exp['days'][day]['sessions'][maze][k]
                            timing_value = timing['days'][day]['sessions'][maze][k]
                            if master_value != timing_value:
                                exp['days'][day]['sessions'][maze][k] = timing_value

            sys.stdout.write('  4) Master <- Addins (Final Pass)\n')
            if add_tree and rat in add_tree:
                add = add_tree[rat]
                for day in add['days']:
                    rdstr = 'rat%03d-%02d' % (rat, day)
                    exp['days'][day] = add['days'][day]
                    sys.stdout.write('   - Merged %s from addins\n' % rdstr)

            sys.stdout.flush()

    except KeyError:
        import pdb
        pdb.set_trace()

    write_yaml_file(exp_tree, 'master')
    return exp_tree

def write_yaml_file(tree, which):
    """Write out a nested-dict tree to a YAML file specfied by *which*

    Arguments:
    tree -- nested dictionary containing hierarchical metadata
    which -- either 'experiment', 'location', 'timing', or 'master'
    """
    filename = get_data_file_path(which, search=False)
    if there(filename):
        backup = unique_path(filename, fmt='%s.%d')
        os.rename(filename, backup)
        sys.stdout.write('Backup: moved old %s file:\n\tFrom: %s\n\tTo: %s\n' % (
            which, filename, backup))
    try:
        fd = file(filename, 'w')
    except IOError:
        raise IOError, 'could not open \'%s\' file: %s' % (which, filename)
    else:
        sys.stdout.write('Writing %s file: %s\n' % (which, filename))
        if type(tree) is AutoVivification:
            tree_dict = tree.dict_copy()
        elif type(tree) is dict:
            tree_dict = tree
        yaml.dump(tree_dict, stream=fd, default_flow_style=False)
        fd.close()
