#!/usr/bin/env python

import tables as tb

from scanr.lib import get_node

scan_table = f.get_node('/behavior/scans')
session_table = f.get_node('/sessions')

colnames = table.colnames
descr = scantable.coldescrs

descr['x'] = tb.FloatCol(pos=len(descr))
descr['y'] = tb.FloatCol(pos=len(descr))


# Dummy data file

testfile = tb.open_file('test.h5', 'w')

# Create a new table with expt_type column

newtable = f.create_table('/behavior', 'newscans', descr, title='Extended Scans Table')
row = newtable.row

for scan in table.iterrows():
    for key in colnames:
        row[key] = scan[key]
    row['x'] = 0.0
    row['y'] = 0.0
    row.append()


# Filter by expt_type while iterating over scans



f.close()
