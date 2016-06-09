#!/usr/bin/env python

"""
preprocess-sundata-heekyung.py -- Change Heekyung's day directories to ratXX-YY
"""

import os
import sys
import re
import subprocess


SUNDATA_DIR = "/kdata/Heekyung_CA3"
OLD_RAT = "rat(\d+)"
OLD_DAY = "Day(\d+).*"


def main():

    nchanged = 0
    total = 0
    for do_rat_dir in (True, False):
        for dirpath, dnames, fnames in os.walk(SUNDATA_DIR):
            head, tail = os.path.split(dirpath)
            search = None
            if do_rat_dir:
                search = re.match(OLD_RAT, tail)
                newpath = dirpath
                if search is not None:
                    newpath = os.path.join(head, tail.title())
            else:
                search = re.search(OLD_DAY, tail)
                if search is not None:
                    day = int(search.groups()[0])
                    phead, ptail = os.path.split(head)
                    rat = int(ptail[3:])
                    newpath = os.path.join(head, 'rat%d-%02d' % (rat, day))

            if search is None:
                continue

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
    print "Titling rat dirs and fixing Day1 day dirs for %s..." % SUNDATA_DIR
    main()
    sys.exit(0)

