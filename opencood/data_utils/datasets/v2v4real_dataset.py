"""
Dataset class for early fusion
"""
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils import box_utils, yaml_utils, transformation_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)


class V2V4RealDataset(basedataset.BaseDataset):
    def __init__(
        self,
        params,
        #is_train=True,
        data_split='train',
        split_val_test=False,
        reverse_traintest=False,
        ego_car_id='0',
        distance_filtering=False,
        min_distance=0.,
        max_distance=90.,
        return_original_points=False,
        return_original_ground=False,
        no_augment=False,
        #is_internal_exp_for_ranker=False,
        #is_16_beam=False,
        num_lidar_beams=32,
        min_timestamp=0,
        max_timestamp=9999,
    ):
        super(V2V4RealDataset, self).__init__(
            params,
            #is_train=is_train,
            data_split=data_split,
            split_val_test=split_val_test,
            reverse_traintest=reverse_traintest,
            ego_car_id=ego_car_id,
            distance_filtering=distance_filtering,
            min_distance=min_distance,
            max_distance=max_distance,
            return_original_points=return_original_points,
            return_original_ground=return_original_ground,
            no_augment=no_augment,
            #is_internal_exp_for_ranker=is_internal_exp_for_ranker,
            #is_16_beam=is_16_beam,
            num_lidar_beams=num_lidar_beams,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
        )

        self.data_split = data_split
        self.is_train = (data_split == 'train')

        self.pre_processor = build_preprocessor(params["preprocess"], self.is_train)
        self.post_processor = build_postprocessor(params["postprocess"], self.is_train)

        self.multi_class = False
        self.specific_object = self.params.get("specific_object", None)
        print("self.specific_object: ", self.specific_object, "self.multi_class: ", self.multi_class)

        #self.is_16_beam = is_16_beam
        self.num_lidar_beams = num_lidar_beams

        self.return_original_points = return_original_points
        self.return_original_ground = return_original_ground

        self.return_oracle_label = False

        self.load_npy_label = self.params["load_npy_label"] and self.is_train
        self.npy_label_path = self.params["npy_label_path"]
        self.npy_label_idx = self.params["npy_label_idx"]
        #self.npy_label_order = self.params["npy_label_order"]

        msg = (
            '------------------------------------\n'
            '# Dataset Config:\n'
            '------------------------------------\n'
            f"# data split     : {self.data_split} \n"
            f"# load npy label : {self.load_npy_label} \n"
            f"# Ego-car id     : {self.ego_car_id} \n"
            f"# load npy label : {self.load_npy_label} \n"
            f"# npy label path : {self.npy_label_path} \n"
            f"# npy label idx  : {self.npy_label_idx} \n"
            f"# distance filtering : {distance_filtering} \n"
            '------------------------------------'
        )
        print(msg)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict["ego"] = {}

        # folder_name = base_data_dict['ego']['folder_name']
        # timestep = base_data_dict['ego']['index']

        # first find the ego vehicle's lidar pose
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["lidar_pose"]  # 4x4 matrix
                folder_name = cav_content["folder_name"]
                traintest = cav_content["traintest"]
                #timestep = cav_content["index"]
                timestep = cav_content["timestamp"]
                ego_yaml = cav_content["ego_yaml"]
                ref_yaml = cav_content["ref_yaml"]
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        projected_lidar_stack = []
        ground_stack = []
        object_stack = []
        object_id_stack = []
        if self.return_oracle_label:
            oracle_object_stack = []
            oracle_object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base, ego_lidar_pose
            )
            # all these lidar and object coordinates are projected to ego already
            if int(cav_id) == int(self.ego_car_id):
                projected_lidar_stack.append(selected_cav_processed["projected_lidar"])
                ground_stack.append(selected_cav_processed["ground"])
                object_stack.append(selected_cav_processed["object_bbx_center"])
                object_id_stack += selected_cav_processed["object_ids"]
                if self.return_oracle_label:
                    oracle_object_stack.append(selected_cav_processed["oracle_object_bbx_center"])
                    oracle_object_id_stack += selected_cav_processed["oracle_object_ids"]

        # exclude all repetitive objects
        # object_id_stack = [1,2,3,1,2,3] -> set(o_i_s) {1, 2, 3} -> uniq_ind = [0,1,2]
        # they always select bbox from 0 > 1 > 2 > ... if overlapped/repetitive
        unique_indices = [
            object_id_stack.index(x) for x in set(object_id_stack)
        ]  # NOTE: is this ok in pseudo data?
        if not unique_indices == []:
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]
        #print("object_stack.shape: ", object_stack.shape)
        # make sure bounding boxes across all frames have the same number
        if self.multi_class:
            label_dim = 8
        else:
            label_dim = 7
        object_bbx_center = np.zeros(
            (self.params["postprocess"]["max_num"], label_dim)
        )  # (100 x 7)
        mask = np.zeros(self.params["postprocess"]["max_num"])  # (100 x 7)
        if not unique_indices == []:
            object_bbx_center[: object_stack.shape[0], :] = object_stack
            mask[: object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # NOTE: possible issue if using multisensor simultaneously
        ground_stack = np.vstack(ground_stack).transpose(1, 0)

        # data augmentation
        projected_lidar_stack, object_bbx_center, mask, ground_stack = self.augment(
            projected_lidar_stack, object_bbx_center, mask, ground_np=ground_stack
        )

        # we do lidar filtering in the stacked lidar
        projected_lidar_stack, ground_stack = mask_points_by_range(
            projected_lidar_stack,
            self.params["preprocess"]["cav_lidar_range"],
            ground_stack,
        )
        
        #print("projected_lidar_stack.shape: ", projected_lidar_stack.shape, "ground_stack.shape: ", ground_stack.shape, "object_bbx_center.shape: ", object_bbx_center.shape, "mask.shape: ", mask.shape)

        if self.num_lidar_beams != 32:
            # augmentation may remove some of the bbx out of range
            # if not 32 beam, remove boxes that includes points less than 5
            object_bbx_center_valid = object_bbx_center[mask == 1]
            object_bbx_center_valid, valid_mask = box_utils.mask_boxes_outside_range_and_num_points_numpy(
                object_bbx_center_valid, projected_lidar_stack,
                self.params["preprocess"]["cav_lidar_range"],
                self.params["postprocess"]["order"],
                min_num_points=5,
            )
        else:
            # augmentation may remove some of the bbx out of range
            object_bbx_center_valid = object_bbx_center[mask == 1]
            object_bbx_center_valid, valid_mask = box_utils.mask_boxes_outside_range_numpy(
                object_bbx_center_valid,
                self.params["preprocess"]["cav_lidar_range"],
                self.params["postprocess"]["order"],
            )

        mask[object_bbx_center_valid.shape[0] :] = 0
        object_bbx_center[: object_bbx_center_valid.shape[0]] = object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0] :] = 0

        # update unique indices
        unique_indices = [
            unique_indices[i] for i, n in enumerate(list(valid_mask)) if n
        ]

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        if not self.multi_class:
            anchor_box = self.post_processor.generate_anchor_box()
        else:
            anchor_box = None

        # generate targets label
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask
        )
        
        processed_data_dict["ego"].update({
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": [object_id_stack[i] for i in unique_indices],
            "anchor_box": anchor_box,
            "processed_lidar": lidar_dict,
            "label_dict": label_dict,
            "folder_name": folder_name,
            "traintest": traintest,
            "timestep": timestep,
            "ego_yaml": ego_yaml,
            "ref_yaml": ref_yaml,
            'object_stack': object_stack,
        })

        if self.return_original_points: #o r self.is_self_train:
            processed_data_dict['ego'].update({
                'origin_lidar': projected_lidar_stack,
                'origin_ground': ground_stack
            })

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose, ground_np=None):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base["params"]["transformation_matrix"]
        #print("transformation_matrix: ", transformation_matrix)

        #if self.params["dataset_name"] == "opv2v":
        #    transformation_matrix = selected_cav_base["params"]["lidar_pose"]
        #    # list to numpy array
        #    transformation_matrix = np.array(transformation_matrix)
        #    #print("updated transformation_matrix: ", transformation_matrix)
        
        if self.params["dataset_name"] == "opv2v":
            transformation_matrix_for_generate_object_center = ego_pose #selected_cav_base["params"]["lidar_pose"]
        elif self.params["dataset_name"] == "v2v4real":
            transformation_matrix_for_generate_object_center = selected_cav_base["params"]["transformation_matrix"]

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center(
                [selected_cav_base],
                transformation_matrix_for_generate_object_center,
                is_train=self.is_train,
                use_pseudo_label=(self.load_npy_label and self.is_train),
                multi_class=self.multi_class,
                specific_object=self.specific_object,
            )

        #print(object_bbx_center.shape, object_bbx_mask.shape, object_ids)

        if self.return_oracle_label:
            oracle_object_bbx_center, oracle_object_bbx_mask, oracle_object_ids = \
                self.post_processor.generate_object_center(
                    [selected_cav_base],
                    transformation_matrix,
                    is_train=self.is_train,
                    use_pseudo_label=False, # always return oracle
                )
            selected_cav_processed.update({
                "oracle_object_bbx_center": oracle_object_bbx_center[oracle_object_bbx_mask == 1],
                "oracle_object_ids": oracle_object_ids
            })

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]
        ground_np = selected_cav_base["ground_np"]
        if ground_np is None:
            ground_np = np.zeros(lidar_np.shape[0], dtype=bool)
        lidar_np, ground_np = shuffle_points(lidar_np, ground_np)
        # remove points that hit itself
        lidar_np, ground_np = mask_ego_points(lidar_np, ground_np)
        # project the lidar to ego space
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(
            lidar_np[:, :3], transformation_matrix
        )

        selected_cav_processed.update({
            "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
            "object_ids": object_ids,
            "projected_lidar": lidar_np,
            "ground": ground_np,
        })

        return selected_cav_processed

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """

        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]
        # print(scenario_path)

        output_dict = {}

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = torch.from_numpy(
                np.array([cav_content["object_bbx_center"]])
            )
            object_bbx_mask = torch.from_numpy(
                np.array([cav_content["object_bbx_mask"]])
            )
            object_ids = cav_content["object_ids"]

            object_stack = torch.from_numpy(
                np.array([cav_content["object_stack"]])
            )

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content["anchor_box"] is not None:
                output_dict[cav_id].update({
                    "anchor_box": torch.from_numpy(
                        np.array(cav_content["anchor_box"])
                    )
                })
            if self.return_original_points:
                origin_lidar = [cav_content["origin_lidar"]]
            if self.return_original_ground:
                origin_ground = [cav_content["origin_ground"]]

            # processed lidar dictionary
            processed_lidar_torch_dict = self.pre_processor.collate_batch(
                [cav_content["processed_lidar"]]
            )
            # label dictionary
            label_torch_dict = self.post_processor.collate_batch(
                [cav_content["label_dict"]]
            )

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()

            # object_ids = torch.as_tensor(object_ids)

            ego_params = yaml_utils.load_yaml(cav_content["ego_yaml"])
            ref_params = yaml_utils.load_yaml(cav_content["ref_yaml"])
            ego_lidar_pose = ego_params['lidar_pose']
            ref_lidar_pose = ref_params['lidar_pose']
            transformation_matrix = transformation_utils.x1_to_x2(ref_lidar_pose, ego_lidar_pose)
            refcar_bbox_np = np.array([[0., 0., -1.3, 5., 2., 1.5, 0.]])
            refcar_bbox_np = box_utils.boxes_to_corners_3d(refcar_bbox_np, order='lwh')
            refcar_bbox_np = box_utils.project_box3d(refcar_bbox_np, transformation_matrix)
            refcar_bbox_np = box_utils.corner_to_center(refcar_bbox_np, order='lwh')
            refcar_bbox_np = torch.from_numpy(refcar_bbox_np)

            output_dict[cav_id].update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": object_bbx_mask,
                    "processed_lidar": processed_lidar_torch_dict,
                    "label_dict": label_torch_dict,
                    "object_ids": object_ids,
                    "transformation_matrix": transformation_matrix_torch,
                    "folder_name": cav_content["folder_name"],
                    "traintest": cav_content["traintest"],
                    "timestep": cav_content["timestep"],
                    "ego_yaml": cav_content["ego_yaml"],
                    "ref_yaml": cav_content["ref_yaml"],
                    "refcar_bbox": refcar_bbox_np,
                    "object_stack": object_stack,
                }
            )

            if self.return_original_points:
                origin_lidar = torch.from_numpy(np.array(origin_lidar))
                output_dict[cav_id].update({"origin_lidar": origin_lidar})
            if self.return_original_ground:
                origin_ground = torch.from_numpy(np.array(origin_ground))
                output_dict[cav_id].update({"origin_ground": origin_ground})

        return output_dict

    def post_process(
        self, data_dict, output_dict, 
        dynamic_score_threshold=False, refcar_bbox=None, a=1,b=1,c=1
    ):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = self.post_processor.post_process(
            data_dict, output_dict, dynamic_score_threshold, refcar_bbox, a, b, c
        )
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_gt(
        self, data_dict, 
        dynamic_score_threshold=False, refcar_bbox=None, a=1,b=1,c=1
    ):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        #pred_box_tensor, pred_score = self.post_processor.post_process(
        #    data_dict, output_dict, dynamic_score_threshold, refcar_bbox, a, b, c
        #)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return gt_box_tensor

if __name__ == "__main__":
    from opencood.utils.yaml_utils import load_yaml
    from tqdm import tqdm

    params = load_yaml("/users/PAS2099/jinsuyoo/cad-v2v/configs/basic_cfg_opv2v_refcar.yaml")

    """
    hypes_yaml configs/basic_cfg.yaml \
        --save_path ${scratch_save_dir}-${local_save_dir} \
        --wandb_project ${scratch_save_dir} --wandb_note ${local_save_dir}_stage_${stage} \
        --pretrained_model_path "" \
        --load_npy_label \
        --npy_label_path ${root_dir}/${scratch_save_dir}/${local_save_dir}/stage_${stage}_2_filtered \
        --npy_label_idx pred.npy \
        --n_epoches ${num_epochs_per_stage} --batch_size ${batch_size} --warmup_epoches ${warmup_epoches} \
        --stage ${stage} \
        --lr 0.002 --warmup_lr 2e-4 --lr_min 2e-5 \
        --distance_filtering --min_distance 0 --max_distance ${max_distance[$stage-1]};
    """

    # TODO: cleanup
    params["load_npy_label"] = False
    params["npy_label_path"] = None
    params["npy_label_idx"] = None

    params["specific_object"] = None

    dataset = V2V4RealDataset(
        params,
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

    for i in tqdm(range(10)):
        dataset.__getitem__(i)
