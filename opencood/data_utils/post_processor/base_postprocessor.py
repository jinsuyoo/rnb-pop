"""
Template for AnchorGenerator
"""

import numpy as np
import torch

##### need to return it to the original!!!
from opencood.utils import box_utils


class BasePostprocessor(object):
    """
    Template for Anchor generator.

    Parameters
    ----------
    anchor_params : dict
        The dictionary containing all anchor-related parameters.
    train : bool
        Indicate train or test mode.

    Attributes
    ----------
    bbx_dict : dictionary
        Contain all objects information across the cav, key: id, value: bbx
        coordinates (1, 7)
    """

    def __init__(self, anchor_params, train=True):
        self.params = anchor_params
        self.bbx_dict = {}
        self.train = train

    def generate_anchor_box(self):
        # needs to be overloaded
        return None

    def generate_label(self, *argv):
        return None

    def generate_gt_bbx(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []
        #print(data_dict['ego']['object_bbx_center'])
        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space.
            # the transformation matrix for gt should always be based on
            # current timestamp (object transformation matrix is for
            # late fusion only since other fusion method already did
            #  the transformation in the preprocess)
            transformation_matrix = cav_content['transformation_matrix'] \
                if 'gt_transformation_matrix' not in cav_content \
                else cav_content['gt_transformation_matrix']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)

            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the id list to filter
        #gt_box3d_selected_indices = [i for i, n in enumerate(object_id_list) if n == -1]
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

        return gt_box3d_tensor

    def generate_gt_bbx_multi(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        gt_box3d_list_truck = []
        # used to avoid repetitive bounding box
        object_id_list = []
        object_id_list_truck = []
        #print(data_dict['ego']['object_bbx_center'])
        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space.
            # the transformation matrix for gt should always be based on
            # current timestamp (object transformation matrix is for
            # late fusion only since other fusion method already did
            #  the transformation in the preprocess)
            transformation_matrix = cav_content['transformation_matrix'] \
                if 'gt_transformation_matrix' not in cav_content \
                else cav_content['gt_transformation_matrix']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            object_bbx_center_truck = cav_content['object_bbx_center_truck']
            object_bbx_mask_truck = cav_content['object_bbx_mask_truck']
            object_ids_truck = cav_content['object_ids_truck']
            object_bbx_center_truck = object_bbx_center_truck[object_bbx_mask_truck == 1]

            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)

            # convert center to corner
            object_bbx_corner_truck = \
                box_utils.boxes_to_corners_3d(object_bbx_center_truck,
                                                self.params['order'])
            projected_object_bbx_corner_truck = \
                box_utils.project_box3d(object_bbx_corner_truck.float(),
                                        transformation_matrix)
            gt_box3d_list_truck.append(projected_object_bbx_corner_truck)


            # append the corresponding ids
            object_id_list += object_ids
            object_id_list_truck += object_ids_truck

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        gt_box3d_list_truck = torch.vstack(gt_box3d_list_truck)
        # some of the bbx may be repetitive, use the id list to filter
        #gt_box3d_selected_indices = [i for i, n in enumerate(object_id_list) if n == -1]
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]
        gt_box3d_selected_indices_truck = \
            [object_id_list_truck.index(x) for x in set(object_id_list_truck)]
        gt_box3d_tensor_truck = gt_box3d_list_truck[gt_box3d_selected_indices_truck]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

        mask_truck = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor_truck)
        gt_box3d_tensor_truck = gt_box3d_tensor_truck[mask_truck, :, :]

        return gt_box3d_tensor, gt_box3d_tensor_truck

    def generate_gt_bbx_ref(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []
        #print(data_dict['ego']['object_bbx_center'])
        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space.
            # the transformation matrix for gt should always be based on
            # current timestamp (object transformation matrix is for
            # late fusion only since other fusion method already did
            #  the transformation in the preprocess)
            transformation_matrix = cav_content['transformation_matrix'] \
                if 'gt_transformation_matrix' not in cav_content \
                else cav_content['gt_transformation_matrix']

            object_bbx_center = cav_content['object_bbx_center_ref']
            object_bbx_mask = cav_content['object_bbx_mask_ref']
            object_ids = cav_content['object_ids_ref']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)

            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the id list to filter
        #gt_box3d_selected_indices = [i for i, n in enumerate(object_id_list) if n == -1]
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

        return gt_box3d_tensor
    
    def generate_object_center(
        self, cav_contents, reference_lidar_pose, is_train, use_pseudo_label=False, multi_class=False, specific_object=None):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        reference_lidar_pose : np.ndarray
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        from opencood.data_utils.datasets import GT_RANGE # gt range is +-100, +-40, -3~5
        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])
            #print(cav_content['pl'])
            #print(use_pseudo_label, cav_content['cav_id'])
            
            if use_pseudo_label: # need update for check label... test ok
            #if use_pseudo_label and cav_content['cav_id'] == 1: # need update for check label... test ok
                #from pprint import pprint
                #pprint(cav_content)
                pseudo_label_path = cav_content['pl']
            else:
                pseudo_label_path = ''
            cav_id = cav_content['cav_id']
        #print(cav_contents)
        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else GT_RANGE

        box_utils.project_world_objects(
            tmp_object_dict, output_dict, 
            reference_lidar_pose, filter_range, self.params['order'], 
            pseudo_label_path=pseudo_label_path,
            use_pseudo_label=use_pseudo_label,
            npy_label_order="lwh", 
            detector=self.params['detector'],
            specific_object=specific_object) # hardcoded for now..

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []
        object_labels = np.zeros(self.params['max_num'])

        #object_truck_np = np.zeros((self.params['max_num'], 7))
        #mask_truck = np.zeros(self.params['max_num'])
        #object_ids_truck = []

        # output_dict is a dictionary with key as object id and value as
        # keys: 'coord', 'obj_type'
        # 'coord' is a numpy array of shape (1, 7) with x, y, z, l, w, h, yaw
        # 'obj_type' is the object type 'Car', 'Truck', etc.

        object_type = []

        for i, (object_id, object_content) in enumerate(output_dict.items()):
            object_np[i] = object_content['coord'][0, :]
            mask[i] = 1
            #object_ids.append(object_id)
            if object_content['ass_id'] != -1:
                object_ids.append(object_content['ass_id'])
            else:
                object_ids.append(object_id + 100 * cav_id)

            #object_type = object_content['obj_type']
            #if object_type == 'Car':
            #    class_idx = 1
            #elif object_type == 'Truck':
            #    class_idx = 2
            
            #object_labels[i] = class_idx

        #if multi_class:
        #    object_labels = np.expand_dims(object_labels, axis=-1)
        #    object_np = np.concatenate([object_np, object_labels], axis=-1)

        return object_np, mask, object_ids #, object_labels
