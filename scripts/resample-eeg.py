#!/usr/bin/env python

from scanr.eeg import (DTS_STUB, DATA_STUB, Resample2K, close_eeg_file,
    get_eeg_file, get_eeg_sample_rate, get_eeg_timeseries)
from scanr.time import compress_timestamps, find_sample_rate
from scanr.data import get_group, new_array, get_node
from scanr.meta import walk_mazes, get_tetrode_list


def show_sample_rates():
    fs_list = []
    tetrodes = get_node('/metadata', 'tetrodes')

    for rec in tetrodes.where('EEG==True'):
        maze = 1
        rds = (rec['rat'], rec['day'], maze)
        fs = get_eeg_sample_rate(rds, rec['tt'])

        if fs not in fs_list:
            fs_list.append(fs)

        if fs is not None and not (990 < fs < 1010):
            print 'Rat %d, Day %d, M%d, Sc%02d = %.1f Hz'%(rds + (rec['tt'], fs))

    return fs_list

def resample_all_2k_eeg_data():
    eeg_file = get_eeg_file(False)

    for rat, day, maze in walk_mazes():
        rds = (rat,day,maze)
        data_group = get_group(eeg=True, rds=rds)

        for tt in get_tetrode_list(rat, day):
            fs = get_eeg_sample_rate(rds, tt)
            if fs is None or 990 < fs < 1010:
                continue

            print 'Resampling: Rat%03d-%02d-m%d Sc%02d has %.1f Hz'%(
                rat, day, maze, tt, fs)

            time_name = DTS_STUB%tt
            array_name = DATA_STUB%tt

            ts, EEG = Resample2K.timeseries(*get_eeg_timeseries(rds, tt))
            delta_ts = compress_timestamps(ts)

            time_array = new_array(data_group, time_name, delta_ts,
                title='Delta-Compressed Timestamps: Tetrode %d'%tt,
                force=True, eeg=True)
            data_array = new_array(data_group, array_name, EEG,
                title='Continuous EEG Data: Tetrode %d'%tt,
                force=True, eeg=True)

            fs = find_sample_rate(ts)
            time_array._v_attrs['sample_rate'] = fs
            data_array._v_attrs['sample_rate'] = fs

        eeg_file.flush()
    close_eeg_file()


if __name__ == '__main__':
    # show_sample_rates()
    resample_all_2k_eeg_data()

