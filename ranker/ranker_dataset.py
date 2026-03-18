import os
import os.path as osp
import glob
import pickle
import numpy as np
import torch
from torch.utils import data as data

from opencood.utils import dataset_utils


class RankerDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        data_dir="exp/ranker_training_data",
        train_split="subset2",
        data_split="train",
        car_id="0",
        num_points=1000,
        num_annotate_frames=2,
        use_ground=True,
        random_drop_points=False,
    ):
        self.num_points = num_points
        self.data_dir = data_dir
        self.use_ground = use_ground
        self.random_drop_points = random_drop_points

        self.scenario_folders = dataset_utils.get_scenario_folders(
            root_dir=root_dir, train_split=train_split, data_split=data_split)

        self.all_data_path = []
        for sf in self.scenario_folders:
            traintest, scenario = sf.split("/")[-2:]
            data_path = osp.join(data_dir, traintest, scenario, car_id)
            _data_path = sorted(glob.glob(osp.join(data_path, "*", "*.pkl")))
            _data_path = [dp for dp in _data_path if int(dp.split('/')[-2]) < num_annotate_frames]
            self.all_data_path.extend(_data_path)

        print(f"RankerDataset: {len(self.all_data_path)} samples")

    def __getitem__(self, index):
        data_path = self.all_data_path[index]
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        ptc = data_dict['pcd']       # (N, 3) cropped/center-normalized points
        ground = data_dict['ground'] # (N,) ground point mask
        bbox_center = data_dict['object']  # (7,) xyzlwh-yaw of sampled box
        iou3d = data_dict['iou3d']   # scalar IoU with GT box
        offset = data_dict['offset'] # (7,) offset from GT box
        distance = data_dict['l2norm']  # scalar L2 distance from ego

        if self.random_drop_points:
            num_drop = np.random.randint(int(ptc.shape[0] * 0.5))
            keep_mask = np.ones(ptc.shape[0], dtype=bool)
            keep_mask[np.random.choice(ptc.shape[0], num_drop, replace=False)] = False
            ptc = ptc[keep_mask]
            if self.use_ground:
                ground = ground[keep_mask]

        choice = np.random.choice(len(ptc), self.num_points, replace=True)
        ptc = ptc[choice, :]
        ground = ground[choice]
        ground = np.expand_dims(ground, 1)

        return {
            'bbox_center': bbox_center,
            'ptc': ptc,
            'iou': iou3d,
            'distance': distance,
            'offset': offset,
            'ground': ground,
        }

    def __len__(self):
        return len(self.all_data_path)
