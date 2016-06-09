#!/usr/bin/env python
# encoding: utf-8
"""
update_quality.py -- Update the isolation quality attributes for cells
    collected by Inah based on the up-to-date list

Usage: update_quality.py

Created by Joe Monaco on September 26, 2012.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.
"""

# Library imports
import os
import re

# Package imports
from ..tools.bash import CPrint

# SCANR package imports
from scanr.lib import (Config, walk_days, get_maze_list, get_tetrode_list,
    get_data_file_path, get_group, get_maze_list, flush_file, close_file)
from scanr.cluster import string_to_quality

# Quality file name
QUALITY_FILES = ["updated_fair_cell_list_inah.csv", "cell_quality_shan.csv"]


def main(out, qfile):

    qual_fd = file(os.path.join(Config['data_root'], qfile), 'r')
    id_pattern = re.compile('rat(\d+)-(\d+)\DSc(\d+)\Dcl-maze(\d)\D(\d+)')

    updated = 0
    for line in qual_fd.readlines():
        tokens = line.split(',')
        if tokens[0] == "RatID":
            continue
        elif len(tokens) != 3:
            out('Weird line: "%s"'%line, error=True)
            continue

        id_string, region_string, quality_string = tokens
        search = re.search(id_pattern, id_string)
        if search is None:
            out('Weird ID string: "%s"'%id_string, error=True)
            continue

        rat, day, tt, maze, cl = tuple(map(int, search.groups()))

        sgroup = get_group(rat=rat, day=day, session=maze)
        cell_name = 't%dc%d'%(tt, cl)

        try:
            cell = getattr(sgroup, cell_name)
        except AttributeError:
            out('Could not find "%s" under "%s"'%(cell_name, sgroup._v_pathname), error=True)
        else:
            update_quality = string_to_quality(quality_string.strip().title())
            found_quality = string_to_quality(cell._v_attrs['quality'])
            if found_quality < update_quality:
                cell._v_attrs['quality'] = str(update_quality)
                out('%s: Updated %s -> %s'%(cell._v_pathname, found_quality, update_quality))
                updated += 1

    out('Updated a total of %d cell quality attributes.'%updated)

    # Close the files
    qual_fd.close()
    flush_file()
    close_file()


if __name__ == '__main__':

    log_fd = file('update_quality.log', 'w')
    out = CPrint(prefix='UpdateQuality', color='lightgreen', outfd=log_fd, timestamp=False)

    for quality_file in QUALITY_FILES:
        main(out, quality_file)

    log_fd.close()

