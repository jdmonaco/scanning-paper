#!/usr/bin/env python
# encoding: utf-8
"""
kdata_listing.py

Created by Joe Monaco on 2011-05-16.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.
"""

import sys
import os
import re

from scanr.lib import Config, get_path, get_group, close_file
from ..tools.bash import CPrint

PREFIX = ' -> '

def write_row_str(fd, dirname, ev, pos, EEG, missingEEG, incl):
    fd.write(
        ','.join(    [
            os.path.realpath(dirname),
            str(int(ev)),
            str(int(pos)),
            str(int(EEG)),
            str(int(missingEEG)),
            str(int(incl))  ]   ) + '\n'    )

def main():
    root_dir = get_path()
    log_fd = file('kdata-listing.txt', 'w')
    out = CPrint(prefix='KDataListing', color='lightgreen', outfd=log_fd)
    out.timestamp = False

    csv_fd = file('kdata-listing.csv', 'w')
    csv_complete_fd = file('kdata-complete-listing.csv', 'w')
    csv_fd.write('directory,events,tracking,EEG,missing_EEG,included\n')
    csv_complete_fd.write('directory,events,tracking,EEG,missing_EEG,included\n')

    N = dict(included=0, possibles=0, with_eeg=0, missing_some_eeg=0)

    for dir_path, dir_names, file_names in os.walk(root_dir, followlinks=True):

        # Directory must have some indication that this is a data set directory
        wayup, up, parent = dir_path.split(os.path.sep)[-3:]
        if not parent:
            up, parent = wayup, up
        ratdaysearch = re.search('[Rr]at(\d+)-(\d+)', parent)
        if not ratdaysearch:
            continue

        N['possibles'] += 1

        # Check for Events and Pos files
        events_check = Config['path']['events_file'] in file_names
        pos_check = Config['path']['tracking_file'] in file_names

        # Check for EEG data files
        has_EEG = False
        missing_EEG = False
        for sub_dir in dir_names:
            sub_match = re.match('Sc(\d+)$', sub_dir)
            if not sub_match:
                continue

            tt = int(sub_match.groups()[0])
            if 'CSC%d.Ncs'%tt in file_names:
                has_EEG = True
            else:
                missing_EEG = True
        if not has_EEG:
            missing_EEG = False

        # Check to see if included in database
        ratday = ratdaysearch.groups()
        rat = int(ratday[0])
        day = int(ratday[1])
        is_group = get_group(rat=rat, day=day) is not None
        is_path = get_path(rat=rat, day=day) == dir_path
        included = is_group and is_path

        if included:
            N['included'] += 1
            N['with_eeg'] += int(has_EEG)
            N['missing_some_eeg'] += int(missing_EEG)

        # Write out a row to csv file
        write_row_str(csv_complete_fd, dir_path, events_check, pos_check, has_EEG,
            missing_EEG, included)

        if not (events_check or pos_check):
            continue

        write_row_str(csv_fd, dir_path, events_check, pos_check, has_EEG,
            missing_EEG, included)

        # Write out log file record for data sets with at least events or
        # tracking file
        out('Checking %s:'%dir_path)
        if events_check:
            out(PREFIX + 'Found events file')
        else:
            out(PREFIX + 'Missing events file!')
        if pos_check:
            out(PREFIX + 'Found tracking file')
        else:
            out(PREFIX + 'Missing Tracking file!')
        if has_EEG:
            if missing_EEG:
                out(PREFIX + 'Found EEG data files, but some missing')
            else:
                out(PREFIX + 'Found all EEG data files')
        else:
            out(PREFIX + 'Could not find any EEG data files!')
        if included:
            out(PREFIX + 'Data set (rat %d, day %d) is included'%(rat, day))
        else:
            out(PREFIX + 'Data set is NOT included!')
        out('')

    out('Possible dataset folders:   %d' % N['possibles'])
    out('Datasets actually included: %d' % N['included'])
    out('Datasets with EEG data:     %d' % N['with_eeg'])
    out('...But missing some EEG:    %d' % N['missing_some_eeg'])

    log_fd.close()
    csv_fd.close()
    csv_complete_fd.close()
    close_file()

if __name__ == '__main__':
    main()

