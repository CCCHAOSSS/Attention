import torch
from DVSGestures.DVS_gesture_data_process.DVS_Gesture_dataloders import create_datasets
from DVSGestures.CNN.Config import configs
from DVSGestures.DVS_gesture_data_process.events_timeslices import get_tmad_slice, my_chunk_evs_pol_dvs
import os
import h5py
import numpy as np


def create_data(config):
    # Data set
    config.train_dataset = create_datasets(
        config.savePath,
        train=True,
        is_train_Enhanced=config.is_train_Enhanced,
        ds=config.ds,
        dt=config.dt * 1000,
        chunk_size_train=config.T,
        is_spike=config.is_spike,
        interval_scaling=config.interval_scaling,
        input_polarity=config.in_channels
    )

    config.test_dataset = create_datasets(
        config.savePath,
        train=False,
        ds=config.ds,
        dt=config.dt * 1000,
        chunk_size_test=config.T,
        clip=config.clip,
        is_spike=config.is_spike,
        interval_scaling=config.interval_scaling,
        input_polarity=config.in_channels
    )
    # Data loader
    config.train_loader = torch.utils.data.DataLoader(
        config.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        num_workers=config.num_work,
        pin_memory=config.pip_memory)
    config.test_loader = torch.utils.data.DataLoader(
        config.test_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_work,
        pin_memory=config.pip_memory)

if __name__ == '__main__':
    def sample_train(hdf5_file,
                     T=60,
                     dt=1000,
                     is_train_Enhanced=False
                     ):
        label = hdf5_file['labels'][()]

        tbegin = hdf5_file['times'][0]
        tend = np.maximum(0, hdf5_file['times'][-1] - T * dt)

        start_time = np.random.randint(tbegin, tend) if is_train_Enhanced else 0

        tmad = get_tmad_slice(hdf5_file['times'][()],
                              hdf5_file['addrs'][()],
                              start_time,
                              T * dt)
        tmad[:, 0] -= tmad[0, 0]
        return tmad[:, [0, 3, 1, 2]], label


    chunk_size = 30
    is_train_Enhanced = False
    dt = 1000
    ds = [4, 4]
    size = [4, 32 // ds[0], 32 // ds[1]]

    config = configs()
    path = config.savePath
    root_test = os.path.join(path, 'train_TA')
    files_list = os.listdir(root_test)
    with h5py.File(root_test + '/' + files_list[0], 'r', swmr=True, libver="latest") as f:
        data, target = sample_train(f, T=chunk_size, is_train_Enhanced=is_train_Enhanced, dt=dt)

    data = my_chunk_evs_pol_dvs(data=data, dt=dt, T=chunk_size, size=size, ds=ds)

