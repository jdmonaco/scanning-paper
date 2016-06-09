#!/usr/bin/env python
# encoding: utf-8

"""
master-scan.py -- Create master metadata files for a collection of datasets

Created by Joe Monaco on 2011-03-25.
Copyright (c) 2011-2014 Johns Hopkins University. All rights reserved.
"""

USAGE = """Please specify arguments:

Usage: master-scan.py [merge [location_file] | [experiment] [location]]

merge -- combine experiment and location files into master file; optionally
    specify a previous location file to merge into the location metadata
experiment / location -- output experimental session and/or tetrode location
     assignment file after running a full scan of the data directory tree
"""

# Library imports
import os
from os import path
from os.path import join, exists as there
import sys
import re
import glob
import yaml

# Package imports
from ..tools.misc import AutoVivification
from ..tools.bash import CPrint
from ..tools.path import unique_path

# SCANR package imports
from scanr.config import Config
from scanr.meta import merge_master_file, merge_location_files, write_yaml_file
from scanr.neuralynx import read_event_file

# Globals
DEBUG = Config['debug_mode']
ROOT = Config['data_root']
events_file = Config['path']['events_file']
tracking_file = Config['path']['tracking_file']
log_fd = file('master-scan.log', 'w')
out = CPrint(prefix='MasterScan', color='lightgreen', outfd=log_fd, timestamp=False)

def warning(s):
    out('Warning: ' + s, error=True)

def main(exp_out, loc_out):

    exp_tree = AutoVivification()
    loc_tree = AutoVivification()
    unknowns = []

    out('Scanning root directory: %s' % ROOT)
    for dpath, dirs, files in os.walk(ROOT, followlinks=True):

        this_events_file = join(dpath, events_file)
        this_pos_file = join(dpath, tracking_file)
        if not (there(this_events_file) and there(this_pos_file)):
            continue

        # Make sure we're in a ratXXX-DD directory
        wayup, up, parent = dpath.split(path.sep)[-3:]
        if not parent:
            up, parent = wayup, up
        ratdaysearch = re.match('[Rr]at(\d+)-(\d+)$', parent)
        if not ratdaysearch:
            continue

        # Get rat and day number
        rat, day = [int(n) for n in ratdaysearch.groups()]
        if DEBUG:
            out('Found directory for rat %d, day %d' % (rat, day))

        # Get experimenter name
        fullpath = path.realpath(dpath)
        exptr = fullpath.split(path.sep)[-3][:4] # Truncate exptr name after 4 chars
        if 'exptr' not in exp_tree[rat]:
            exp_tree[rat]['exptr'] = exptr
        if DEBUG:
            out('Experimenter: %s'%exptr)

        # Get tetrodes from ScXX subdirectories
        tetrodes = []
        for Sc in glob.glob(join(dpath, 'Sc*')):
            Sc_tail = path.split(Sc)[1]
            if re.match('Sc\d+$', Sc_tail):
                tetrodes.append(int(Sc_tail[2:]))
        tetrodes.sort()
        for tt in tetrodes:
            loc_tree[rat]['days'][day]['tetrodes'][tt] = None
        if DEBUG:
            out('Tetrodes: %s' % str(tetrodes))

        # Search patterns for session information in event comment strings
        drot_search_pattern = re.compile('double[-\s]+rot')
        maze_search_pattern = re.compile('(m(aze)??|ses(sion)??)\s*(\d+)')
        mismatch_search_pattern = re.compile('[^a-z]mis(match)?')
        angle_search_pattern = re.compile('(45|90|135|180)')
        standard_search_pattern = re.compile('st(andar)?d')
        novelty_search_pattern = re.compile('(nov|fam)\w*')

        # Parse events file
        timestamps, strings = read_event_file(this_events_file)
        exp_tree[rat]['days'][day]['type'] = 'unknown'
        sessions = exp_tree[rat]['days'][day]['sessions']
        counts = {'std':0, 'mis':0, 'nov':0, 'fam':0, 'unk':0}
        in_session = False
        start = end = 0l
        start_comment = end_comment = ''
        maze = 0
        session_type = ''
        parameter = -1

        for ts, evstr in zip(timestamps, strings):
            s = evstr.strip().lower()
            s.replace('_', ' ')

            # Crop off redundant ratXX-DD info at beginning of event string
            ratday = 'rat%d-%02d ' % (rat, day)
            if s.startswith(ratday):
                s = s[len(ratday):]

            # Skip any events relating to sleep, foraging, open field, or box sessions
            if re.search('sleep|forag|field|box', s):
                continue

            # Match a maze number or session type with start/endto proceed
            drot_search = re.search(drot_search_pattern, s)
            maze_search = re.search(maze_search_pattern, s)
            standard_search = re.search(standard_search_pattern, s)
            mismatch_search = re.search(mismatch_search_pattern, s)
            novelty_search = re.search(novelty_search_pattern, s)
            angle_search = re.search(angle_search_pattern, s)
            start_search  = re.search('start|begin', s)
            end_search = re.search('end|finish', s)
            if not (maze_search or
                ((start_search or end_search) and
                    (standard_search or mismatch_search or angle_search or novelty_search))):
                continue

            if maze_search:
                new_maze = int(maze_search.groups()[3])
            else:
                if in_session:
                    new_maze = maze
                else:
                    new_maze = maze + 1
            if DEBUG:
                out('Found maze: \'%s\' -> %d' % (s, new_maze))

            if in_session:
                if new_maze != maze:
                    warning('Found new maze before old one ended: \'%s\'' % s)
                    maze = new_maze
                if end_search:
                    in_session = False
                    end = ts
                    end_comment = s
                    sessions[maze]['start'] = int(start)
                    sessions[maze]['start_comment'] = start_comment
                    sessions[maze]['end'] = int(end)
                    sessions[maze]['end_comment'] = end_comment
                    sessions[maze]['type'] = session_type.upper()
                    sessions[maze]['number'] = counts[session_type]
                    sessions[maze]['parameter'] = parameter
                    if exp_tree[rat]['days'][day]['type'] == 'unknown':
                        if counts['std'] + counts['mis'] > 0:
                            exp_tree[rat]['days'][day]['type'] = 'double rotation'
                        elif counts['nov'] + counts['fam'] > 0:
                            exp_tree[rat]['days'][day]['type'] = 'novelty'
                    sys.stdout.write('\n')
                elif start_search:
                    start = ts
                    start_comment = s

            else:
                if not end_search:
                    overwrite = False
                    if new_maze == maze:
                        warning('Starting maze %d again: \'%s\'' % (maze, s))
                        overwrite = True
                    elif new_maze - maze != 1:
                        warning('Maze %d follows maze %d: \'%s\'' % (new_maze, maze, s))
                    if not start_search:
                        warning('New maze without start match: \'%s\'' % s)

                    in_session = True
                    maze = new_maze
                    start = ts
                    start_comment = s

                    if standard_search:
                        session_type = 'std'
                        parameter = 0
                    elif mismatch_search:
                        session_type = 'mis'
                        if angle_search:
                            parameter = int(angle_search.groups()[0])
                        else:
                            warning('No angle found for mismatch session: \'%s\'' % s)
                    elif angle_search:
                        session_type = 'mis'
                        parameter = int(angle_search.groups()[0])
                    else:
                        if novelty_search:
                            session_type = novelty_search.groups()[0]
                            parameter = 0
                        else:
                            session_type = 'unk'
                            parameter = 0
                            warning('Could not classify session type: \'%s\'' % s)
                    if drot_search:
                        exp_tree[rat]['days'][day]['type'] = 'double rotation'
                    if not overwrite:
                        counts[session_type] += 1
                    if session_type == 'unk':
                        unknowns.append((rat, day, maze))
                    if DEBUG:
                        out('-> %s session, parameter %d, number %d' % (session_type.upper(),
                            parameter, counts[session_type]))

    # Second pass to interpolate missing data
    def remove_from_unknowns(rds):
        try:
            unknowns.remove(rds)
        except ValueError:
            warning('Could not remove %s from unknowns list' % str(rds))

    out('Interpolation and pruning of metadata tree:')
    for rat in exp_tree.keys():
        for day in exp_tree[rat]['days'].keys():

            sessions = exp_tree[rat]['days'][day]['sessions']
            exp_type = exp_tree[rat]['days'][day]['type']
            n_mazes = len(sessions.keys())

            # Set unknown session types based on number of sessions
            if exp_type == 'unknown':
                if n_mazes < 4:
                    if n_mazes == 3:
                        exp_type = exp_tree[rat]['days'][day]['type'] = 'novelty'
                else:
                    exp_type = exp_tree[rat]['days'][day]['type'] = 'double rotation'

            # Remove sessions with too few maze runs (training, incomplete, etc?)
            prune = False
            if exp_type == 'unknown':
                prune = True
            if prune:
                out(' - Removing unknown/incomplete day (rat %d, day %d)' % (rat, day))

            # Set session types based on inferred experiment type
            fixed = 0
            for maze in sessions.keys():
                rds = (rat, day, maze)

                if prune:
                    remove_from_unknowns(rds)
                    continue
                if exp_type == 'novelty' and maze > 3:
                    remove_from_unknowns(rds)
                    del exp_tree[rat]['days'][day]['sessions'][maze]
                    continue

                if sessions[maze]['type'] == 'UNK':
                    if exp_type == 'double rotation':
                        if n_mazes == 5:
                            sessions[maze]['type'] = ['MIS', 'STD'][maze % 2]
                        elif n_mazes == 6:
                            if maze == 1:
                                sessions[maze]['type'] = 'STD'
                            else:
                                sessions[maze]['type'] = ['STD', 'MIS'][maze % 2]
                        elif maze < n_mazes and sessions[maze + 1]['type'] == 'MIS':
                            sessions[maze]['type'] = 'STD'
                        if sessions[maze]['type'] != 'UNK':
                            remove_from_unknowns(rds)
                            fixed += 1
                    elif exp_type == 'novelty':
                        sessions[maze]['type'] = ['NOV', 'FAM'][maze % 2]
                        remove_from_unknowns(rds)
                        fixed += 1

                    if DEBUG and fixed:
                        out(' - Setting %s to session type %s' % (str(rds), sessions[maze]['type']))

            if prune:
                del exp_tree[rat]['days'][day]
                continue

            # Set session numbers
            if fixed:
                for maze in sessions.keys():
                    sessions[maze]['number'] = \
                        sum([sessions[m]['type'] == sessions[maze]['type'] for m in range(1, maze + 1)])

    if unknowns:
        warning('%d sessions were unable to be classified:\n%s' % (len(unknowns),
            '\n'.join(['\t%d) rat: %d, day: %d, session: %d' % ((i + 1,) + rds)
                            for i, rds in enumerate(unknowns)])))

    print_tree_data_counts(exp_tree)

    if exp_out:
        write_yaml_file(exp_tree, 'experiment')
    if loc_out:
        write_yaml_file(loc_tree, 'location')

    log_fd.close()


def print_tree_data_counts(tree):
    n_days = 0
    n_mazes = 0
    for rat in tree.keys():
        n_days += len(tree[rat]['days'].keys())
        for day in tree[rat]['days'].keys():
            n_mazes += len(tree[rat]['days'][day]['sessions'].keys())
    out('Total: %d rats, %d recording days, %d maze runs' % (len(tree.keys()), n_days, n_mazes))


if __name__ == '__main__':
    do_merge = exp = loc = False
    old_locfile = None

    if len(sys.argv) > 1:
        outputs = sys.argv[1:]
        if 'experiment' in outputs:
            exp = True
        if 'location' in outputs:
            loc = True
        if outputs[0] == 'merge':
            do_merge = True
            if len(outputs) > 1:
                if there(outputs[1]):
                    old_locfile = outputs[1]
                else:
                    warning('Location file does not exist: %s' % outputs[1])
                    sys.exit(1)
    else:
        print USAGE
        sys.exit(0)

    if do_merge:
        if old_locfile:
            merge_location_files(old_locfile)
        master_tree = merge_master_file()
        print_tree_data_counts(master_tree)
    else:
        main(exp, loc)

    sys.exit(0)
