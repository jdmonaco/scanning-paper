#!/usr/bin/env python

import tables as tb

f = tb.open_file('/Users/joe/Archives/Projects/scan-potentiation-project/09 kdata/kdata.h5')
table = f.get_node('/behavior/scans')
colnames = table.colnames
descr = table.coldescrs

descr['x'] = tb.FloatCol(pos=len(descr))
descr['y'] = tb.FloatCol(pos=len(descr))

newtable = f.create_table('/behavior', 'newscans', descr, title='Extended Scans Table')
row = newtable.row

for scan in table.iterrows():
    for key in colnames:
        row[key] = scan[key]
    row['x'] = 0.0
    row['y'] = 0.0
    row.append()

f.close()
