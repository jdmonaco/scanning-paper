#!/usr/bin/env python

"""
preprocess-sundata.py -- Turn TET-* directories into Sc%02d
"""

import os
import sys
import re
import subprocess


SUNDATA_DIR = "/kdata/Sun_data_rat18_to_44"
OLD_RAT = "rat(\d+)-(\d+)-proc"

def main():

    nchanged = 0
    total = 0
    for dirpath, dnames, fnames in os.walk(SUNDATA_DIR):
        head, tail = os.path.split(dirpath)
        search = re.match(OLD_RAT, tail)
        if search is None:
            continue

        rat, day = map(int, search.groups())
        rat_dir = os.path.join(head, 'Rat%d' % rat)
        if not os.path.exists(rat_dir):
            os.makedirs(rat_dir)

        newpath = os.path.join(rat_dir, 'rat%d-%02d' % (rat, day))
        cmd_line = ["mv", dirpath, newpath]
        print "Changing: %s" % " ".join(cmd_line)

        total += 1
        retcode = subprocess.call(cmd_line)
        if retcode == 0:
            nchanged += 1
        else:
            print "Error: something wrong with %s..." % dirpath

    print "Succesfully change %d/%d rat directories!" % (nchanged, total)


if __name__ == "__main__":
    print "Changing rat*-proc to rat* for %s..." % SUNDATA_DIR
    main()
    sys.exit(0)

