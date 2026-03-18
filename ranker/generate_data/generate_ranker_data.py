"""
Generate ranker training data from ego-car ground truth annotations.

For each annotated frame, this script:
  1. Loads ego-car LiDAR point cloud and GT bounding boxes
  2. Samples N perturbed boxes around each GT box
  3. Computes IoU3D between each sampled box and the GT box
  4. Saves (point cloud, box, IoU, offset, distance) tuples as .pkl files

Only the first `--num_annotate_frames` frames per scenario are used,
following the <1% labeled data setting from the paper.
"""

import os
import glob
import math
import random
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
import os.path as osp

from ranker.ranker_utils import get_transform, get_scale
from opencood.utils import pcd_utils, yaml_utils, transformation_utils, box_utils, dataset_utils
from opencood.utils.iou3d_nms import iou3d_nms_utils


def parse_config():
    parser = argparse.ArgumentParser(description="Generate ranker training data from GT annotations")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the V2V4Real dataset")
    parser.add_argument("--num_annotate_frames", type=int, default=2,
                        help="Number of annotated frames to use per scenario")
    parser.add_argument("--num_samples_per_box", type=int, default=1000,
                        help="Number of perturbed boxes to sample per GT box")
    parser.add_argument("--num_labels_per_box", type=int, default=4,
                        help="Max samples to save per IoU bin per GT box")
    parser.add_argument("--car_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="exp/ranker_training_data",
                        help="Output directory for ranker training data")
    parser.add_argument("--debug", action="store_true",
                        help="Run on a single scenario/timestamp for testing")

    args = parser.parse_args()

    random.seed(310)
    np.random.seed(310)
    torch.manual_seed(310)

    return args


def get_bbox_center(params, transformation_matrix, order='lwh'):
    bbox_list = []
    for object_id, object_content in params['vehicles'].items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']
        obj_type = object_content.get('obj_type', 'Car')

        if obj_type == 'Pedestrian':
            continue

        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            rotation[0], rotation[1], rotation[2],
        ]
        object2lidar = transformation_utils.x1_to_x2(object_pose, transformation_matrix)
        bbx = box_utils.create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = box_utils.corner_to_center(bbx_lidar, order=order)
        bbx_lidar, _ = box_utils.mask_boxes_outside_range_numpy(
            bbx_lidar, [-80, -40, -5, 80, 40, 3], order, 2
        )
        if bbx_lidar.shape[0] > 0:
            bbox_list.append(bbx_lidar)

    if not bbox_list:
        return np.empty([0, 7])
    return np.concatenate(bbox_list, axis=0)


def sample_bboxs(boxes, num_samples,
                 noise_x=1., noise_y=0.5, noise_z=0.2,
                 noise_l=0.5, noise_w=0.2, noise_h=0.1, noise_angle=0.1):
    """Sample perturbed boxes around the given boxes."""
    assert boxes.shape[-1] == 7
    num_boxes = boxes.shape[0]

    noise_x = torch.clamp(torch.randn(num_samples).to(boxes.dtype).to(boxes.device) * noise_x, -2., 2.)
    noise_y = torch.clamp(torch.randn(num_samples).to(boxes.dtype).to(boxes.device) * noise_y, -1., 1.)
    noise_z = torch.clamp(torch.randn(num_samples).to(boxes.dtype).to(boxes.device) * noise_z, -0.5, 0.5)
    noise_l = torch.clamp(torch.randn(num_samples).to(boxes.dtype).to(boxes.device) * noise_l, -1., 1.)
    noise_w = torch.clamp(torch.randn(num_samples).to(boxes.dtype).to(boxes.device) * noise_w, -0.5, 0.5)
    noise_h = torch.clamp(torch.randn(num_samples).to(boxes.dtype).to(boxes.device) * noise_h, -0.2, 0.2)
    noise_angle = (2. * torch.rand(num_samples) - 1.).to(boxes.dtype).to(boxes.device) * noise_angle

    offset = torch.stack([noise_x, noise_y, noise_z, noise_l, noise_w, noise_h, noise_angle], dim=1)

    sample_idx = np.random.choice(num_boxes, num_samples, replace=True)
    boxes_new = boxes.clone()[sample_idx]
    boxes_new += offset
    return boxes_new, offset


def main():
    args = parse_config()

    scenario_folders = dataset_utils.get_scenario_folders(
        root_dir=args.root_dir, train_split='subset2', data_split='train',
    )

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    print(f"Save path: {save_path}")

    car_id = str(args.car_id)
    order = 'lwh'
    label_counter_all = np.zeros(10, dtype=int)
    targets = []

    for scen_idx, scenario_path in tqdm(enumerate(scenario_folders)):
        if args.debug and scen_idx > 0:
            break

        print(f"Scenario: {scenario_path}")
        num_timestamps = len(glob.glob(osp.join(scenario_path, car_id, '*.pcd')))
        traintest, scenario = scenario_path.split('/')[-2:]

        for timestamp in range(min(num_timestamps, args.num_annotate_frames)):
            if args.debug and timestamp > 0:
                break

            pcd_path = osp.join(scenario_path, car_id, f'{timestamp:06d}.pcd')
            ego_pcd = pcd_utils.pcd_to_np(pcd_path)

            ego_yml_path = osp.join(scenario_path, car_id, f'{timestamp:06d}.yaml')
            ego_params = yaml_utils.load_yaml(ego_yml_path)

            ground_path = osp.join(
                args.root_dir, 'above_ground_ransac',
                traintest, scenario, car_id, f'{timestamp:06d}.pkl'
            )
            with open(ground_path, 'rb') as f:
                above_plane_dict = pickle.load(f)
                ground_np = np.array(above_plane_dict['mask_above_plane'])

            bbox_center_np = get_bbox_center(ego_params, transformation_matrix=np.eye(4), order=order)
            if bbox_center_np.shape[0] == 0:
                continue

            print(f"  Timestamp {timestamp}: {bbox_center_np.shape[0]} GT boxes")

            for i in range(bbox_center_np.shape[0]):
                label_counter = np.zeros(10, dtype=int)

                single_box_np = bbox_center_np[i][None]  # (1, 7)
                single_box_t = torch.from_numpy(single_box_np).cuda().float()

                new_boxes_t, new_offset_t = sample_bboxs(single_box_t, args.num_samples_per_box)
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(single_box_t, new_boxes_t)
                new_targets_np = iou3d[0].cpu().numpy()

                new_dist_from_ego_np = torch.linalg.norm(
                    new_boxes_t[:, :3], dim=1, ord=2).cpu().numpy()

                new_boxes_np = new_boxes_t.cpu().numpy()
                new_offset_np = new_offset_t.cpu().numpy()

                ego_pcd_t = torch.from_numpy(ego_pcd).float()
                gt_bbox_t = torch.from_numpy(new_boxes_np).float()
                scale, ptc, trs, original_ptc = get_scale(ego_pcd_t, gt_bbox_t)

                valid = (scale < 3.0).to(scale.dtype)

                obj_corner_np = box_utils.boxes_to_corners_3d(new_boxes_np, order)
                ones = np.ones([obj_corner_np.shape[0], 8, 1])
                obj_corner_np = np.concatenate([obj_corner_np, ones], axis=2)
                obj_corner_t = torch.from_numpy(obj_corner_np).float()

                new_obj_corner_t = torch.bmm(obj_corner_t, trs)[..., :3]
                new_obj_center_np = box_utils.corner_to_center(new_obj_corner_t.numpy())

                valid_ptc_list = []
                valid_grd_list = []
                for v in range(valid.shape[0]):
                    valid_ptc_list.append(original_ptc[v][valid[v].bool()])
                    valid_grd_list.append(ground_np[valid[v].bool()])

                if not args.debug:
                    for ss in range(valid.shape[0]):
                        single_ptc = valid_ptc_list[ss]
                        single_grd = valid_grd_list[ss]
                        single_obj = new_obj_center_np[ss]
                        single_tgt = new_targets_np[ss]
                        single_offset = new_offset_np[ss]
                        single_dist = new_dist_from_ego_np[ss]

                        if single_tgt > 1.1:
                            continue
                        if not (10 < single_ptc.shape[0] < 7500):
                            continue

                        label_idx = min(math.floor(single_tgt * 10), 9)
                        if label_counter[label_idx] < args.num_labels_per_box:
                            label_counter[label_idx] += 1
                            label_counter_all[label_idx] += 1

                            cur_data_save_dict = {
                                'pcd': single_ptc,
                                'ground': single_grd,
                                'object': single_obj,
                                'iou3d': single_tgt,
                                'offset': single_offset,
                                'l2norm': single_dist,
                            }
                            cur_data_save_path = osp.join(
                                save_path, traintest, scenario, car_id,
                                f'{timestamp:06d}', f'objects_{i:03d}_{ss:03d}.pkl'
                            )
                            os.makedirs(osp.dirname(cur_data_save_path), exist_ok=True)
                            with open(cur_data_save_path, 'wb') as f:
                                pickle.dump(cur_data_save_dict, f)

                            targets.append(single_tgt)

    print(f"Total samples saved: {len(targets)}")
    print(f"IoU bin distribution: {label_counter_all}")


if __name__ == "__main__":
    main()
