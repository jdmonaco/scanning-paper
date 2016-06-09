#!/usr/bin/env python

"""
preprocess-sundata.py -- Turn TET-* directories into Sc%02d
"""

import os
import sys
import re
import subprocess


SUNDATA_DIR = "/kdata/Sun_data_rat18_to_44"
OLD_TET = "TET_(\d+)"
NEW_TET = "Sc%02d"

def main():

    nchanged = 0
    total = 0
    for dirpath, dnames, fnames in os.walk(SUNDATA_DIR):
        head, tail = os.path.split(dirpath)
        search = re.match(OLD_TET, tail)
        if search is None:
            continue
        tet_no = int(search.groups()[0])
        newpath = os.path.join(head, NEW_TET % tet_no)
        cmd_line = ["mv", dirpath, newpath]

        total += 1
        retcode = subprocess.call(cmd_line)
        if retcode == 0:
            nchanged += 1
        else:
            print "Error: something wrong with %s..." % dirpath

    print "Succesfully change %d/%d tetrode directories!" % (nchanged, total)


if __name__ == "__main__":
    print "Changing TET to Sc for %s..." % SUNDATA_DIR
    main()
    sys.exit(0)

