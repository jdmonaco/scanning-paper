#!/usr/bin/env python

"""
preprocess-sachin.py -- Change Sachin's day directories to ratXX-YY'
"""

import os
import sys
import re
import subprocess

DATA_DIR = "/kdata/Sachin_proximal_distal_CA1"
OLD_RAT = "rat(\d+)-(\d+)"

def main():

    nchanged = 0
    total = 0
    for do_rat_dir in (True, False):
        for dirpath, dnames, fnames in os.walk(DATA_DIR):
            head, tail = os.path.split(dirpath)
            search = re.match(OLD_RAT, tail)
            if search is None:
                continue
            rat, day = map(int, search.groups())
            rat_dir = os.path.join(head, 'Rat%d' % rat)

            if do_rat_dir:
                if not os.path.exists(rat_dir):
                    os.makedirs(rat_dir)
            else:
                newpath = os.path.join(rat_dir,tail)
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
    print "Changing rat*-proc to rat* for %s..." % DATA_DIR
    main()
    sys.exit(0)

