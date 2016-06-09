#!/usr/bin/env python
# encoding: utf-8
"""
append_theta.py -- Walk tetrodes and compute overall relative theta power to store
    in the /metadata/tetrodes column 'relative_theta'.

Usage: append_theta.py

Created by Joe Monaco on 2012-03-30.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

# Library imports
import os, sys, cPickle
import numpy as np
from IPython.parallel import Client

# Package imports
from ..tools.bash import CPrint
from ..tools.misc import view_wait_with_status

# SCANR package imports
from scanr.lib import Config, walk_days, flush_file, close_file, get_node
from scanr.spike import create_tetrode_table

# Script constants
STATUS_PERIOD = 60
PICKLE_FN = 'tetrodes-relative-theta.pickle'


def main(out):
    save_fn = os.path.join(Config['data_root'], PICKLE_FN)

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
        dview.execute('import cPickle, os, gc')
        dview.execute('from scanr.config import Config')
        dview.execute('from scanr.meta import get_tetrode_list, get_maze_list')
        dview.execute('from scanr.eeg import get_eeg_data, total_power, Theta, FullBand')

        lview = rc.load_balanced_view()

        @lview.remote(block=False)
        def compute_relative_theta(dataset):
            save_fn = os.path.join(Config['data_root'], 'rat%03d-%02d.pickle'%dataset)
            if os.path.exists(save_fn):
                fd = file(save_fn, 'r')
                res = cPickle.load(fd)
                fd.close()
            else:
                res = {}
                Theta.zero_lag = False
                for tt in get_tetrode_list(*dataset):
                    tt_id = tuple(dataset) + (tt,)
                    rtheta = []
                    for maze in get_maze_list(*dataset):
                        rds = tuple(dataset) + (maze,)
                        X = get_eeg_data(rds, tt)
                        if X is None:
                            continue
                        rtheta.append(
                            total_power(Theta.power(X), fs=Theta.fs) /
                                total_power(FullBand.power(X), fs=FullBand.fs)
                        )
                    res[tt_id] = rtheta
                    gc.collect()
                fd = file(save_fn, 'w')
                cPickle.dump(res, fd)
                fd.close()
            return res

        # Send out compute tasks and wait for completion
        out('Sending out tasks to the cluster...')
        async_results = map(compute_relative_theta, list(walk_days()))
        view_wait_with_status(lview, out, timeout=STATUS_PERIOD)

        # Collate results into flattened list of cell info dictionaries
        out('Collating results...')
        results = {}
        for async in async_results:
            results.update(async.get())

        # Save a pickle
        fd = file(save_fn, 'w')
        cPickle.dump(results, fd)
        fd.close()
        out('Saved intermediate pickle to:\n%s'%save_fn)

    # Recreate a fresh tetrodes table
    create_tetrode_table()
    tetrodes_table = get_node('/metadata', 'tetrodes')

    updated = 0
    out('Now updating %s with relative theta power data...'%tetrodes_table._v_pathname)

    for row in tetrodes_table.iterrows():
        tt_id = tuple(int(row[k]) for k in ('rat', 'day', 'tt'))
        tt_str = "rat%03d-%02d-Sc%d"%tt_id
        try:
            rtheta = results[tt_id]
        except KeyError:
            out('Results did not contain %s.'%tt_str, error=True)
            continue

        if len(rtheta) == 0:
            out('No EEG found for tetrode %s'%tt_str)
        else:
            rtheta_score = np.median(rtheta)
            out('Found %d sessions for tetrode %s, median = %.4f'%(
                len(rtheta), tt_str, rtheta_score))

            row['EEG'] = True
            row['relative_theta'] = rtheta_score
            row.update()

            updated += 1
            if updated % 100 == 0:
                out('Flushing updated table data...')
                tetrodes_table.flush()

    flush_file()
    close_file()


if __name__ == '__main__':

    log_fd = file('append-theta.log', 'w')
    out = CPrint(prefix='AppendTheta', outfd=log_fd, timestamp=False)

    main(out)

    log_fd.close()

