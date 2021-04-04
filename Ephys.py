import numpy as np
import os
from tqdm import trange

class Ephys():

    def __init__(self,
                 fname_binary,
                 n_channels,
                 n_times):

        #
        self.fname_binary = fname_binary

        #
        self.fname_npy = fname_binary[:-4]+".npy"

        #
        self.n_channels = n_channels

        #
        self.n_times = n_times

    def find_match(self,
                   spikes_unit,
                   spikes_gt,
                   max_diff = 100):
        ctr = 0
        for s in spikes_unit:
            diffs = np.abs(spikes_gt-s)
            idx = np.where(diffs<=max_diff)[0]
            if idx.shape[0]>0:
                ctr+=1

        return ctr

    def match_unit_to_gt(self, spikes_unit, spikes_gt):

        matches = []
        for k in trange(len(spikes_gt)):
            matches.append(self.find_match(spikes_unit, spikes_gt[k]))

        return matches


    def get_wfs(self, spikes):

        self.binary_reader_waveforms(spikes)

    def get_ptp(self, spikes, n_max = 500):

        idx = np.random.choice(np.arange(spikes.shape[0]),
                               min(n_max, spikes.shape[0]),
                               replace=False)

        self.get_wfs_from_npy(spikes[idx])

        # template
        self.ptp = self.wfs.mean(0).ptp(0).max(0)


    def get_wfs_from_npy(self, spikes):

        #
        data = np.load(self.fname_npy, mmap_mode='r')

        #
        wfs = []
        for k in range(spikes.shape[0]):
            wfs.append(data[:,spikes[k]-self.n_times//2:spikes[k]+self.n_times//2])

        wfs = np.array(wfs)

        self.wfs = wfs


    # visualize recomputed templates over time;
    def binary_reader_waveforms(self,
                                spikes,
                                channels=None,
                                data_type='float32'):
        ''' Reader for loading raw binaries

            standardized_filename:  name of file contianing the raw binary
            n_channels:  number of channels in the raw binary recording
            n_times:  length of waveform
            spikes: 1D array containing spike times in sample rate of raw data
            channels: load specific channels only
            data_type: float32 for standardized data

        '''

        # ***** LOAD RAW RECORDING *****
        if channels is None:
            wfs = np.zeros((spikes.shape[0], self.n_times, self.n_channels), data_type)
        else:
            wfs = np.zeros((spikes.shape[0], self.n_times, self.channels.shape[0]), data_type)

        if data_type =='float32':
            data_len = 4
        else:
            data_len = 2

        with open(self.fname_binary, "rb") as fin:
            for ctr,s in enumerate(spikes):
                #print (ctr,s)
                # index into binary file: time steps * 4  4byte floats * n_channels
                fin.seek(s * data_len * self.n_channels, os.SEEK_SET)
                wfs[ctr] = np.fromfile(
                    fin,
                    dtype=data_type,
                    count=(self.n_times * self.n_channels)).reshape(
                                    self.n_times, self.n_channels)
        fin.close()

        self.wfs = wfs

