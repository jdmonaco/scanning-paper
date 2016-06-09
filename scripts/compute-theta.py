#!/usr/bin/env python
#encoding: utf-8

"""
Testing the compute_relative_theta remote function for the append_theta script.
"""

import cPickle, os, gc
from scanr.meta import get_tetrode_list, get_maze_list
from scanr.eeg import get_eeg_data, total_power, Theta, FullBand
from ..tools.bash import CPrint

out = CPrint(prefix='RelativeTheta', color='cyan')


def compute_relative_theta(dataset):
    out('Computing rat%03d-%02d dataset...'%dataset)
    res = {}
    Theta.zero_lag = False

    for tt in get_tetrode_list(*dataset):
        out('Starting tetrode %d:'%tt)
        tt_id = tuple(dataset) + (tt,)
        rtheta = []

        for maze in get_maze_list(*dataset):
            out.printf('m%d [L'%maze)
            rds = tuple(dataset) + (maze,)
            X = get_eeg_data(rds, tt)
            if X is None:
                continue

            out.printf('Pt')
            P_theta = Theta.power(X)

            out.printf('Px')
            P_X = FullBand.power(X)

            out.printf('Tt')
            tot_theta = total_power(P_theta, fs=Theta.fs)

            out.printf('Tx')
            tot_X = total_power(P_X, fs=FullBand.fs)

            out.printf('Ap')
            rtheta.append(tot_theta / tot_X)

            out.printf('] ')
        out.printf('\n')

        res[tt_id] = rtheta
        out('Relative theta values: %s'%str(rtheta))

        gc.collect()

    return res


if __name__ == '__main__':
    compute_relative_theta((72,1))

