#!/usr/bin/env python
# encoding: utf-8
"""
append_spatial_info.py -- Walk the metadata tree and compute spatial information rates
    and bootstrap significance values. Write out a complete table of cell information
    to /physiology/cell_information.

Usage: append_spatial_info.py [skaggs|olypher|shannon]

Created by Joe Monaco on 2012-01-28.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

# Library imports
import os, sys
import gc
import cPickle
import numpy as np
import tables as tb
from IPython.parallel import Client

# Package imports
from ..tools.bash import CPrint
from ..tools.misc import view_wait_with_status

# SCANR package imports
from scanr.config import Config
from scanr.meta import walk_mazes
from scanr.data import flush_file, close_file, new_table, get_group, dump_table
from scanr.cluster import PlaceCellCriteria, PrincipalCellCriteria
from scanr.session import SessionData
from scanr.spike import get_tetrode_area, parse_cell_name
from scanr.tracking import CellInfoDescr

# Hack to adjust preliminary place-cell test output
import scanr.cluster
scanr.cluster.DEFAULT_MIN_INFO_RATE = 0.1

# Script constants
STATUS_PERIOD = 60
PICKLE_FN = 'cell-information-%s.pickle'


def main(out, measure='skaggs'):
    save_fn = os.path.join(Config['data_root'], PICKLE_FN%measure)

    if os.path.exists(save_fn):
        out('Loading saved pickle from:\n%s'%save_fn)
        fd = file(save_fn, 'r')
        results = cPickle.load(fd)
        fd.close()

    else:
        # Set up the parallel engines
        raw_input('Make sure ipcluster has started... ')
        rc = Client()
        dview = rc[:]
        dview.block = True
        dview.execute('from scanr.tracking import SpatialInformationScore')
        dview.execute('from scanr.session import SessionData')

        lview = rc.load_balanced_view()

        @lview.remote(block=False)
        def compute_spatial_information(rds, which):
            data = SessionData(rds=rds)
            score = SpatialInformationScore(random=False, fixed_spacing=0.5,
                min_offset=15.0, reverse_for_shuffle=True,
                measure=which, x_bins=48)
            res = []
            for tc in sorted(data.clusts.keys()):
                cell_res = dict(rds=rds, tc=tc)
                cluster = data.cluster_data(tc)
                cell_res['I'] = score.compute(data, cluster)
                cell_res['p'] = score.pval(data, cluster)
                cell_res['N_running'] = cluster.N_running
                res.append(cell_res)
            return res

        # Send out compute tasks and wait for completion
        out('Sending out tasks to the cluster...')
        async_results = []
        for rds in walk_mazes():
            async_results.append(compute_spatial_information(rds, measure))
        view_wait_with_status(lview, out, timeout=STATUS_PERIOD)

        # Collate results into flattened list of cell info dictionaries
        out('Collating results...')
        results = []
        for async in async_results:
            results.extend(async.get())

        # Save a pickle
        fd = file(save_fn, 'w')
        cPickle.dump(results, fd)
        fd.close()
        out('Saved intermediate pickle to:\n%s'%save_fn)

    itable = new_table('/physiology', 'cell_information_%s'%measure, CellInfoDescr,
        title='%s Cell Information' % measure.title())

    cell_id = 0
    row = itable.row
    for cell_data in results:
        rds = tuple(int(k) for k in cell_data['rds'])
        cell_name = str(cell_data['tc'])

        node = getattr(get_group(rds=rds), cell_name)
        attrs = node._v_attrs

        # Get spatial information results from cell data dictionary
        I = float(cell_data['I'])
        p = float(cell_data['p'])
        N_running = int(cell_data['N_running'])

        # Color-coded console output for this cell
        out.printf('Rat%03d-%02d-m%d-%s: '%(rds+(cell_name,)), color='lightgray')
        if I >= 0.5:
            out.printf('%.2f bits, '%I, color='green')
        else:
            out.printf('%.2f bits, '%I, color='red')
        if p <= 0.01:
            out.printf('p < %.4f, '%p, color='green')
        else:
            out.printf('p < %.4f, '%p, color='red')
        if N_running >= 30:
            out.printf('N = %d\n'%N_running, color='green')
        else:
            out.printf('N = %d\n'%N_running, color='red')

        # Write out the table record
        tt, cl = parse_cell_name(cell_name)
        row['id'] = cell_id
        row['rat'], row['day'], row['session'] = rds
        row['tc'] = cell_name
        row['area'] = get_tetrode_area(rds[0], rds[1], tt)
        row['quality'] = str(attrs['quality'])
        row['spike_width'] = attrs['maxwidth']
        row['N_running'] = N_running
        row['I'] = I
        row['p_value'] = p
        row.append()

        cell_id += 1

        if cell_id and cell_id % 15 == 0:
            out('Flushing table and garbage collecting...')
            itable.flush()
            gc.collect()

    flush_file()
    close_file()


def partition_responses(rmap_bins=36, table='cell_information', quality='None',
    smooth=True, just_bins=False):
    from scanr.cluster import PrincipalCellCriteria, get_min_quality_criterion, AND
    from scanr.data import get_node

    pbins = np.array([1.1, 0.5, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.0])
    ibins = np.linspace(0, 6, 13)
    if just_bins:
        return pbins, ibins
    N_pbins, N_ibins = len(pbins)-1, len(ibins)-1

    R = [[[] for j in xrange(N_ibins)] for i in xrange(N_pbins)]
    cell_table = get_node('/physiology', table)

    cell_criteria = AND(PrincipalCellCriteria, get_min_quality_criterion(quality))

    for cell in cell_table.where('(area=="CA3")|(area=="CA1")'):

        session = SessionData.get((cell['rat'], cell['day'], cell['session']),
            load_clusters=False)
        cluster = session.cluster_data(cell['tc'])

        if not (cell['N_running'] > 30 and cell_criteria.filter(cluster)):
            continue

        pix = (cell['p_value'] <= pbins).nonzero()[0]
        iix = (cell['I'] >= ibins).nonzero()[0]

        if not len(pix) or not (0 <= pix[-1] < N_pbins):
            continue
        if not len(iix) or not (0 <= iix[-1] < N_ibins):
            continue

        pix = pix[-1]
        iix = iix[-1]

        R[pix][iix].append(session.get_cluster_ratemap(
            cluster, bins=rmap_bins, smoothing=smooth, blur_width=360./rmap_bins,
            exclude_off_track=True,
            exclude=session.scan_and_pause_list))

        print '...added %s to p-value bin %.4f, info bin %.2f...'%(
            cell['tc'], pbins[pix], ibins[iix])

    for i, row in enumerate(R):
        for j, rmap in enumerate(row):
            R[i][j] = r = np.asarray(rmap)
            if not len(rmap):
                continue
            R[i][j] = r[np.argsort(np.argmax(r, axis=1))]

    return R


def print_bins():
    pbins, ibins = partition_responses(just_bins=True)
    for p in pbins:
        print "p < %.4f"%p
    print '--'
    for i in ibins:
        print "%.2f"%i


def partition_response_image(R, img_file, centered=False, norm='max', rmap_bins=36):
    from ..tools.images import masked_array_to_rgba, rgba_to_image
    from ..tools.misc import align_center

    pbins, ibins = partition_responses(just_bins=True)

    N = np.array([np.max([rmap.shape[0] for rmap in row]) for row in R]) + 1
    image = np.zeros((N.sum(), (len(ibins)-1) * (rmap_bins+1)), 'd') - 1

    print "Generating matrix ..."
    for i, row in enumerate(R):
        row_start = N[:i].sum()
        for j, rmap in enumerate(row):
            nrows = rmap.shape[0]
            if not nrows:
                continue
            if centered:
                rmap = align_center(rmap, rmap.argmax(axis=1))
                rmap = rmap[np.trapz(rmap, axis=1).argsort()]
            if norm in ('max', 'binary'):
                rmap = rmap / rmap.max(axis=1)[:, np.newaxis]
                if norm == 'binary':
                    rmap[rmap<0.2] = 0.0
                    rmap[rmap>=0.2] = 1.0
            elif norm == 'binmax':
                rmap = rmap / rmap.max()
            elif norm == 'rel':
                rmap = rmap / np.trapz(rmap, axis=1)[:, np.newaxis]
            image[row_start:row_start+nrows, j*(rmap_bins+1):j*(rmap_bins+1)+rmap_bins] = rmap

    print "Masking image ..."
    mask = image == -1
    image[mask] = 0
    if norm == 'binary':
        cmap = 'gray'
    else:
        cmap = 'jet'
    rgba = masked_array_to_rgba(image, mask=mask, cmap=cmap)

    print "Saving image to", img_file, '...'
    rgba_to_image(rgba, img_file)

    return rgba


def dump_cells_missing_quality(table='cell_information_skaggs'):
    cell_list = []
    def missing_quality(row):
        rds = row['rat'], row['day'], row['session']
        cell_id = (rds[0], rds[1], row['tc'])
        if row['quality'] != 'None' or cell_id in cell_list:
            return False
        data = SessionData.get(rds, load_clusters=False)
        cluster = data.cluster_data(row['tc'])
        if PrincipalCellCriteria.filter(cluster):
            cell_list.append(cell_id)
            return True
        return False
    dump_table('/physiology', name=table,
        condn=missing_quality,
        filename='cells-with-missing-quality.csv')


if __name__ == '__main__':

    log_fd = file('append-spatial-info.log', 'w')
    out = CPrint(prefix='AppendSpatialInfo', outfd=log_fd, timestamp=False)

    if len(sys.argv) == 2:
        measure = sys.argv[1].strip().lower()
    else:
        out('Please specify a spatial information measure: \"skaggs\" or \"olypher\"')
        sys.exit(1)

    if measure not in ('skaggs', 'olypher', 'shannon'):
        out('You specified an invalid spatial information measure: \"%s\"' % measure)
        sys.exit(1)

    main(out, measure=measure)

    log_fd.close()

