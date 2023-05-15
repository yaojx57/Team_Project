from __future__ import print_function, absolute_import, division

import os.path as path
import pandas as pd
import json 
from json import JSONEncoder

import numpy as np
from torch.utils.data import DataLoader

from common.data_loader import PoseDataSet, PoseBuffer
from utils.data_utils import fetch, read_3d_data, create_2d_data

'''
this code is used for prepare data loader
'''
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def data_preparation(args):
    ############################################
    # load dataset
    ############################################
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        if args.s1only:
            subjects_train = ['S1']
        else:
            subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    # print(action_filter)
    ############################################
    # general 2D-3D pair dataset
    ############################################
    poses_train, poses_train_2d, actions_train, cams_train = fetch(subjects_train, dataset, keypoints, action_filter,
                                                                   stride)
    poses_valid, poses_valid_2d, actions_valid, cams_valid = fetch(subjects_test, dataset, keypoints, action_filter,
                                                                   stride)
    

    train_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train),
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(PoseDataSet(poses_valid, poses_valid_2d, actions_valid, cams_valid),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ############################################
    # prepare cross dataset validation
    ############################################
    mpi3d_npz = np.load('data_extra/test_set/test_3dhp.npz')
    tmp = mpi3d_npz
    mpi3d_loader = DataLoader(PoseBuffer([tmp['pose3d']], [tmp['pose2d']]),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # print(type())


    # train_loader_dict = {}
    # train_loader_dict['3d'] = poses_train
    # train_loader_dict['2d'] = poses_train_2d
    # with open("train_loader.json", "w") as outfile:
    #     json.dump(train_loader_dict, outfile, cls=NumpyArrayEncoder)

    return {
        'dataset': dataset,
        'train_loader': train_loader,
        'H36M_test': valid_loader,
        '3DHP_test': mpi3d_loader,
        'action_filter': action_filter,
        'subjects_test': subjects_test,
        'keypoints': keypoints
    }
