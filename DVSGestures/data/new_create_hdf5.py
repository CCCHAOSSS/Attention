import os
import sys
import scipy.io as io
rootPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(rootPath)[0]
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

import tarfile
import os
import h5py
import numpy as np
import struct
from DVSGestures.DVS_gesture_data_process.events_timeslices import *


def my_create_hdf5(path, save_path):
    """

    Args:
        path: dataset path
        save_path: hdf5 results path

    Returns: null

    """


    # Train
    print('processing train data...')
    save_path_train = os.path.join(save_path, 'train')
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)

    # 获取目录下的所有文件
    train_path = os.path.join(path, 'train')
    train_sample_folds = os.listdir(train_path)
    print(len(train_sample_folds))
    n = len(train_sample_folds)
    i = 0
    for lbl in range(n):
        sample_fold = os.path.join(train_path, str(lbl))
        for item in os.listdir(sample_fold):
            print('.')
            evs = os.path.join(sample_fold, item)
            events = io.loadmat(evs, squeeze_me=True, struct_as_record=False)
            events['TSout'].ts = events['TSout'].ts - events['TSout'].ts[0]
            times = events['TSout'].ts
            x = events['TSout'].x - 1
            y = events['TSout'].y - 1
            p = events['TSout'].p
            if min(p) > 0:
                p = p - 1

            addrs = np.stack((x, y, p), axis=1).tolist()

            with h5py.File(save_path_train + os.sep + 'DVS-Gesture-train' + str(i) + '.hdf5',
                           'w') as f:
                tm_dset = f.create_dataset('times', data=times, dtype=np.uint32)
                ad_dset = f.create_dataset('addrs', data=addrs, dtype=np.uint8)
                lbl_dset = f.create_dataset('labels', data=lbl, dtype=np.uint8)
            i = i + 1
    print('train finished')



    # Test
    print('processing test data...')
    save_path_test = os.path.join(save_path, 'test')
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)

    # 获取目录下的所有文件
    test_path = os.path.join(path, 'test')
    test_sample_folds = os.listdir(test_path)
    print(len(test_sample_folds))
    n = len(test_sample_folds)
    i = 0
    for lbl in range(n):
        sample_fold = os.path.join(test_path, str(lbl))
        for item in os.listdir(sample_fold):
            print('.')
            evs = os.path.join(sample_fold, item)
            events = io.loadmat(evs, squeeze_me=True, struct_as_record=False)
            events['TSout'].ts = events['TSout'].ts - events['TSout'].ts[0]
            times = events['TSout'].ts

            x = events['TSout'].x - 1
            y = events['TSout'].y - 1
            p = events['TSout'].p
            if min(p) > 0:
                p = p - 1
            addrs = np.stack((x, y, p), axis=1).tolist()

            with h5py.File(save_path_test + os.sep + 'DVS-Gesture-train' + str(i) + '.hdf5',
                           'w') as f:
                tm_dset = f.create_dataset('times', data=times, dtype=np.uint32)
                ad_dset = f.create_dataset('addrs', data=addrs, dtype=np.uint8)
                lbl_dset = f.create_dataset('labels', data=lbl, dtype=np.uint8)
            i = i + 1
    print('test finished')