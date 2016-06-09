#!/usr/bin/env python

import os
import sys
import tables

from scanr.tracking import CellInfoDescr
from scanr.ana.field_modulation import FieldModDescr
from scanr.data import get_kdata_file, get_node, close_file

from ..tools.bash import lightgreen


def printdot():
    sys.stdout.write(lightgreen('.'))
def println():
    sys.stdout.write('\n')


def remove_old_cell_information_tables():
    kfile = get_kdata_file()
    kfile.removeNode(kfile.root.physiology.cell_information_skaggs)
    kfile.removeNode(kfile.root.physiology.cell_information_olypher)
    kfile.flush()

def create_new_cell_information_tables():
    """Returns new (skaggs_table, olypher_table) Table objects
    """
    kfile = get_kdata_file()
    sktable = kfile.createTable(kfile.root.physiology, 'cell_information_skaggs', CellInfoDescr, title='Skaggs Spatial Information')
    olytable = kfile.createTable(kfile.root.physiology, 'cell_information_olypher', CellInfoDescr, title='Olypher Spatial Information')
    return sktable, olytable

def copy_information_table_from_file(src_path, which):
    """Use this to copy over a potentiation events table from another file into the current h5 file
    """
    assert which in ('skaggs', 'olypher'), 'bad information measure: %s' % which
    table_name = 'cell_information_%s' % which
    srcfile = tables.openFile(src_path, mode='r')
    kfile = get_kdata_file()

    src_table = srcfile.getNode(srcfile.root.physiology, table_name)
    dest_table = kfile.getNode(kfile.root.physiology, table_name)
    dest_row = dest_table.row

    for src_row in src_table.iterrows():
        for k in CellInfoDescr.keys():
            dest_row[k] = src_row[k]
        printdot()
        dest_row.append()
    println()

    srcfile.close()
    close_file()

def copy_potention_table_from_file(bfile):
    """Use this to copy over a potentiation events table from another file into the current h5 file
    """
    kfile = get_kdata_file()
    row = kfile.root.physiology.potentiation.row
    for orig_row in bfile.root.physiology.potentiation.iterrows():
        for k in FieldModDescr.keys():
            if k == 'area':
                row[k] = get_tetrode_area(orig_row['rat'], orig_row['day'], orig_row['tetrode'])
            else:
                row[k] = orig_row[k]
        print 'rat%03d-%02d-m%d %s in %s field %d lap %d' % (row['rat'], row['day'], row['session'], row['tc'], row['area'], row['fieldnum'], row['lap'])
        row.append()


if __name__ == "__main__":
    # remove_old_cell_information_tables()
    # create_new_cell_information_tables()

    if len(sys.argv) == 3:
        src = sys.argv[1]
        if os.path.exists(src):
            which = sys.argv[2]
            copy_information_table_from_file(src, which)
        else:
            sys.stderr.write('path does not exist: %s' % src)


