import os
import random
import os.path as osp
from collections import OrderedDict
import glob
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.utils import yaml_utils
from opencood.utils.dataset_utils import get_scenario_folders
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.yaml_utils import load_yaml
#from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, dist_two_pose


class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(
        self,
        params,
        #is_train=True,
        data_split="train",
        split_val_test=False,
        reverse_traintest=False,
        ego_car_id='0',
        distance_filtering=False,
        min_distance=0.,
        max_distance=90.,
        return_original_points=False,
        return_original_ground=False,
        no_augment=False,
        #is_16_beam=False,
        num_lidar_beams=32,
        min_timestamp=0,
        max_timestamp=9999,
    ):  
        params['dataset_name'] = params.get("dataset_name", "v2v4real")
        print(f"dataset name: {params['dataset_name']}")
        self.params = params
        #self.visualize = visualize

        
        assert params["resplit_data"], "resplit_data must be True"

        self.data_split = data_split
        self.split_val_test = split_val_test
        self.is_train = (data_split == "train")

        self.return_original_points = return_original_points
        self.return_original_ground = return_original_ground

        self.num_lidar_beams = num_lidar_beams
        #self.is_32_beam = is_16_beam
        print(f"num lidar beams: {self.num_lidar_beams}")

        self.return_oracle_label = False

        self.distance_filtering = distance_filtering
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.min_timestamp = min_timestamp
        self.max_timestamp = max_timestamp

        self.ego_car_id = ego_car_id
        self.ref_car_id = '1' if ego_car_id == '0' else '0'
        print(f"Ego-car id: {self.ego_car_id}, ref-car id: {self.ref_car_id}")

        self.npy_label_idx = params["npy_label_idx"]
        #self.npy_label_order = params["adaptation"]["npy_label_order"]

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(
            params["data_augment"], self.is_train and not no_augment)

        if "wild_setting" in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0
        else:
            print("wild setting disabled..")
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = "sim"
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        print("Re-splitting data..")
        root_dir = params["data_path"]
        
        self.scenario_folders = get_scenario_folders(
            root_dir, params["train_split"], data_split, split_val_test=split_val_test)

        self.scenario_folders_ground = [
            osp.join(root_dir, "above_ground_ransac", p.split('/')[-2], p.split('/')[-1]) 
            for p in self.scenario_folders
        ]

        if self.is_train and self.params["load_npy_label"]:
            self.scenario_folders_npy_label = [
                osp.join(self.params["npy_label_path"], p.split('/')[-2], p.split('/')[-1]) 
                for p in self.scenario_folders
            ]
            print('Pseudo labels loaded')

        #self.num_train_scenario = params.get("num_train_scenario", len(self.scenario_folders))
        #print(f"in total have {len(self.scenario_folders)} scenarios, but will only use {self.num_train_scenario} scenarios")
        self.num_train_scenario = len(self.scenario_folders)

        # print min and max timestamp
        print(f"min timestamp: {self.min_timestamp}, max timestamp: {self.max_timestamp}")

        self.reinitialize()

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []



        num_data = 0
        # loop over all scenarios
        for i, scenario_folder in enumerate(self.scenario_folders):
            if i >= self.num_train_scenario:
                break
            self.scenario_database.update({i: OrderedDict()})
            self.scenario_database[i][self.ego_car_id] = OrderedDict()

            # save all yaml files to the dictionary
            cav_path = osp.join(scenario_folder, self.ego_car_id)

            # use the frame number as key, the full path as the values
            # todo: hardcoded to remove additional yamls. no need to worry
            # about this for users.
            yaml_files = sorted([
                osp.join(cav_path, x)
                for x in os.listdir(cav_path)
                if x.endswith(".yaml")
                and "additional" not in x
                and "camera_gt" not in x
            ])
            timestamps = self.extract_timestamps(yaml_files)

            # timestamps = ['000000', ..., '001234']
            valid_timestamps = []
            for timestamp_idx, timestamp in enumerate(timestamps):
                if not (self.min_timestamp <= timestamp_idx <= self.max_timestamp):
                    continue
                # calculate distance
                # redundant but for clarity
                if self.is_train and self.distance_filtering:
                    ego_params = yaml_utils.load_yaml(osp.join(scenario_folder, self.ego_car_id, timestamp + ".yaml"))
                    ref_params = yaml_utils.load_yaml(osp.join(scenario_folder, self.ref_car_id, timestamp + ".yaml"))
                    ego_lidar_pose = ego_params['lidar_pose']
                    ref_lidar_pose = ref_params['lidar_pose']
                    distance = dist_two_pose(ego_lidar_pose, ref_lidar_pose)
                    
                    if (distance < self.min_distance) or (self.max_distance < distance):
                        print(f'distance {distance} filtered')
                        continue
                
                self.scenario_database[i][self.ego_car_id][timestamp] = OrderedDict()

                yaml_file = osp.join(cav_path, timestamp + ".yaml")
                lidar_file = osp.join(cav_path, timestamp + ".pcd")
                ground_file = osp.join(
                    self.scenario_folders_ground[i],
                    self.ego_car_id,
                    f"{timestamp}.pkl",
                )
                #camera_files = self.load_camera_files(cav_path, timestamp)
                if self.num_lidar_beams != 32:
                    lidar_file = lidar_file.replace("test/", f"test_{self.num_lidar_beams}/")
                    lidar_file = lidar_file.replace("train/", f"train_{self.num_lidar_beams}/")
                    ground_file = ground_file.replace("test/", f"test_{self.num_lidar_beams}/")
                    ground_file = ground_file.replace("train/", f"train_{self.num_lidar_beams}/")

                ref_yaml_file = osp.join(scenario_folder, self.ref_car_id, timestamp + ".yaml")
                
                self.scenario_database[i][self.ego_car_id][timestamp]["yaml"] = yaml_file
                self.scenario_database[i][self.ego_car_id][timestamp]["ref_yaml"] = ref_yaml_file
                #self.scenario_database[i][self.ego_car_id][timestamp]["camera0"] = camera_files
                
                self.scenario_database[i][self.ego_car_id][timestamp]["lidar"] = lidar_file
                self.scenario_database[i][self.ego_car_id][timestamp]["ground"] = ground_file

                if self.params["load_npy_label"] and self.is_train:
                    self.scenario_database[i][self.ego_car_id][timestamp]["pl"] = osp.join(
                        self.scenario_folders_npy_label[i],
                        self.ego_car_id,
                        f"{timestamp}_{self.npy_label_idx}",
                    )
                valid_timestamps.append(timestamp)
                num_data += 1

                
            
            # Assume all cavs will have the same timestamps length. Thus
            # we only need to calculate for the first vehicle in the scene.
            self.scenario_database[i][self.ego_car_id]["ego"] = True
            if not self.len_record:
                self.len_record.append(len(valid_timestamps))
            else:
                prev_last = self.len_record[-1]
                self.len_record.append(prev_last + len(valid_timestamps))

        print(f"Total number of training scenarios: {len(self.scenario_database)}, total number of data: {num_data}")


    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = (
            idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        )
        #print(f"scenario index: {scenario_index}, timestamp index: {timestamp_index}")
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        #print(f"timestamp key: {timestamp_key}")
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        #ref_cav_xyz = scenario_database[1][self.ego_car_id][timestamp_key]["yaml"]

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]["ego"] = cav_content["ego"]

            # calculate delay for this vehicle
            timestamp_delay = self.time_delay_calculation(cav_content["ego"])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index

            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(
                scenario_database, timestamp_index_delay
            )
            # add time delay to vehicle parameters
            data[cav_id]["time_delay"] = timestamp_delay

            # load the corresponding data into the dictionary
            data[cav_id]["params"] = self.reform_param(
                cav_content,
                ego_cav_content,
                timestamp_key,
                timestamp_key_delay,
                cur_ego_pose_flag,
                cav_id,
            )
            # 16-beam only contains xyz while the original pcd contains intensity, xyz
            data[cav_id]["lidar_np"] = pcd_utils.pcd_to_np(
                cav_content[timestamp_key_delay]["lidar"], 
                concat_intensity=(self.num_lidar_beams == 32) 
            )
            data[cav_id]["folder_name"] = cav_content[timestamp_key_delay][
                "lidar"
            ].split("/")[-3]
            data[cav_id]["traintest"] = cav_content[timestamp_key_delay][
                "lidar"
            ].split("/")[-4]
            data[cav_id]["index"] = timestamp_index
            data[cav_id]["timestamp"] = timestamp_key
            data[cav_id]["cav_id"] = int(cav_id)

            data[cav_id]["ego_yaml"] = cav_content[timestamp_key_delay]["yaml"]
            data[cav_id]["ref_yaml"] = cav_content[timestamp_key_delay]["ref_yaml"]

            #data[cav_id]["ground_np"] = np.load(
            #    cav_content[timestamp_key_delay]["ground"]
            #)

            ground_path = cav_content[timestamp_key_delay]["ground"]
            if osp.exists(ground_path):
                with open(ground_path, 'rb') as f:
                    above_plane_dict = pickle.load(f)
                    data[cav_id]["ground_np"] = np.array(above_plane_dict['mask_above_plane'])
            else:
                data[cav_id]["ground_np"] = None

            if self.params["load_npy_label"] and self.is_train:
                data[cav_id]["pl"] = cav_content[timestamp_key_delay]["pl"]

        return data

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split("/")[-1]

            timestamp = res.replace(".yaml", "")
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content["ego"]:
                ego_cav_content = cav_content
                ego_lidar_pose = load_yaml(cav_content[timestamp_key]["yaml"])[
                    "lidar_pose"
                ]
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = load_yaml(cav_content[timestamp_key]["yaml"])["lidar_pose"]
            distance = dist_two_pose(cur_lidar_pose, ego_lidar_pose)
            cav_content["distance_to_ego"] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == "real":
            # noise/time is in ms unit
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == "sim":
            time_delay = np.abs(self.async_overhead)

        # todo: current 10hz, we may consider 20hz in the future
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        if not self.is_train:
            np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [
            pose[0] + xyz_noise[0],
            pose[1] + xyz_noise[1],
            pose[2] + xyz_noise[2],
            pose[3],
            pose[4] + ryp_std[1],
            pose[5],
        ]
        return noise_pose

    def reform_param(
        self,
        cav_content,
        ego_content,
        timestamp_cur,
        timestamp_delay,
        cur_ego_pose_flag,
        use_pseudo_label=False,
        ego_car_id=0,
        ref_car_id=0,
    ):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]["yaml"])
        delay_params = load_yaml(cav_content[timestamp_delay]["yaml"])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]["yaml"])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]["yaml"])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params["lidar_pose"]
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params["lidar_pose"]
        cur_cav_lidar_pose = cur_params["lidar_pose"]
        # NOTE: 여기서 tranformation_matrix_pl 해야함 ego 는 불러지고, self.ref_id 이용해서 ref lidar_pose 불러와서
        if not cav_content["ego"] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(
                delay_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std
            )
            cur_cav_lidar_pose = self.add_loc_noise(
                cur_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std
            )

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose, cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose, delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(
                delay_ego_lidar_pose, cur_ego_lidar_pose
            )
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose, cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params["vehicles"] = cur_params["vehicles"]
        delay_params["transformation_matrix"] = transformation_matrix
        delay_params["gt_transformation_matrix"] = gt_transformation_matrix
        delay_params["spatial_correction_matrix"] = spatial_correction_matrix

        return delay_params

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(
        self,
        lidar_np,
        object_bbx_center,
        object_bbx_mask,
        ground_np=None,
        flip=None,
        rotation=None,
        scale=None,
    ):
        """ """
        tmp_dict = {
            "lidar_np": lidar_np,
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "ground_np": ground_np,
            "flip": flip,
            "noise_rotation": rotation,
            "noise_scale": scale,
        }
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict["lidar_np"]
        object_bbx_center = tmp_dict["object_bbx_center"]
        object_bbx_mask = tmp_dict["object_bbx_mask"]

        return lidar_np, object_bbx_center, object_bbx_mask, ground_np

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        #object_bbx_center_truck = []
        #object_bbx_mask_truck = []
        processed_lidar_list = []
        label_dict_list = []
        #label_dict_list_truck = []

        if self.return_original_points:
            origin_lidar = []
            ground = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            processed_lidar_list.append(ego_dict["processed_lidar"])
            label_dict_list.append(ego_dict["label_dict"])

            #object_bbx_center_truck.append(ego_dict['object_bbx_center_truck'])
            #object_bbx_mask_truck.append(ego_dict['object_bbx_mask_truck'])
            #label_dict_list_truck.append(ego_dict['label_dict_truck'])
            if self.return_original_points: #or self.is_self_train:
                origin_lidar.append(torch.from_numpy(ego_dict["origin_lidar"]))
                ground.append(torch.from_numpy(ego_dict["origin_ground"]))

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        #object_bbx_center_truck = torch.from_numpy(np.array(object_bbx_center_truck))
        #object_bbx_mask_truck = torch.from_numpy(np.array(object_bbx_mask_truck))

        if self.return_original_points:
            origin_lidar = torch.from_numpy(np.array(origin_lidar))
            ground = torch.from_numpy(np.array(ground))
            origin_lidar = torch.from_numpy(np.vstack(origin_lidar))
            ground = torch.from_numpy(np.vstack(ground))

        processed_lidar_torch_dict = self.pre_processor.collate_batch(
            processed_lidar_list
        )
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        #label_torch_dict_truck = self.post_processor.collate_batch(label_dict_list_truck)

        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "label_dict": label_torch_dict,

                #"object_bbx_center_truck": object_bbx_center_truck,
                #"object_bbx_mask_truck": object_bbx_mask_truck,
                #"label_dict_truck": label_torch_dict_truck,
            }
        )

        # list of numpy array
        if self.return_original_points: 
            output_dict["ego"].update(
                {"origin_lidar": origin_lidar, "origin_ground": ground}
            )

        return output_dict
    
    def collate_batch_train_rebuttal(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_bbx_center_truck = []
        object_bbx_mask_truck = []
        processed_lidar_list = []
        label_dict_list = []
        label_dict_list_truck = []

        object_stack = []
        object_labels = []

        if self.return_original_points:
            origin_lidar = []
            ground = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            #object_bbx_center.append(ego_dict["object_bbx_center"])
            #object_bbx_mask.append(ego_dict["object_bbx_mask"])
            processed_lidar_list.append(ego_dict["processed_lidar"])
            #label_dict_list.append(ego_dict["label_dict"])

            object_stack.append(torch.from_numpy(ego_dict["object_stack"]))
            object_labels.append(torch.from_numpy(ego_dict["object_labels"]))

            if self.return_original_points: #or self.is_self_train:
                origin_lidar.append(torch.from_numpy(ego_dict["origin_lidar"]))
                ground.append(torch.from_numpy(ego_dict["origin_ground"]))

        # convert to numpy, (B, max_num, 7)
        #object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        #object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        #object_stack = torch.from_numpy(object_stack) #np.array(object_stack))
        #object_labels = torch.from_numpy(object_labels) #np.array(object_labels))
        #processed_lidar_t = torch.from_numpy(np.array(processed_lidar_list))

        if self.return_original_points:
            origin_lidar = torch.from_numpy(np.array(origin_lidar))
            ground = torch.from_numpy(np.array(ground))
            origin_lidar = torch.from_numpy(np.vstack(origin_lidar))
            ground = torch.from_numpy(np.vstack(ground))

        processed_lidar_torch_dict = self.pre_processor.collate_batch(
            processed_lidar_list
        )
        #label_torch_dict = self.post_processor.collate_batch(label_dict_list)



        output_dict["ego"].update(
            {
                #"object_bbx_center": object_bbx_center,
                #"object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "object_stack": object_stack,
                "object_labels": object_labels,

                #"label_dict": label_torch_dict,
            }
        )

        # list of numpy array
        if self.return_original_points: 
            output_dict["ego"].update(
                {"origin_lidar": origin_lidar, "origin_ground": ground}
            )

        return output_dict


if __name__ == "__main__":
    dataset = BaseDataset(
        params=load_yaml("config.yaml"),
        dataset_name="v2v4real",
        data_split="train",
        split_val_test=True,
        reverse_traintest=False,
        ego_car_id="0",
        distance_filtering=False,
        min_distance=0.0,
        max_distance=90.0,
        return_original_points=False,
        return_original_ground=False,
        no_augment=False,
        num_lidar_beams=32,
        min_timestamp=0,
        max_timestamp=9999,
    )

    print("Dataset length: ", len(dataset))
