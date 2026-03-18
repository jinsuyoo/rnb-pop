"""
Preprocess reference car detector predictions for use as initial pseudo-labels.

For each frame, this script:
  1. Loads reference car detector predictions (in ref car LiDAR frame)
  2. Projects bounding boxes into the ego car LiDAR frame
  3. Adds the reference car's own bounding box (it is a labeled vehicle)
  4. Removes the ego car detection (closest detection to origin)
  5. Adjusts box z-coordinates using the RANSAC ground plane estimate
  6. Removes boxes farther than 80m from the ego car
  7. Removes boxes containing fewer than `--min_num_points` LiDAR points

The output is stored as:
  <root_dir>/<save_dir>/{traintest}/{scenario}/0/{timestamp}_pred.npy

This must be run before the R&B-POP pipeline. Set `initial_label_path` in the
pipeline scripts to <root_dir>/<save_dir>.
"""

import argparse
import glob
import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from opencood.utils import yaml_utils, transformation_utils, box_utils, dataset_utils, pcd_utils


def parse_config():
    parser = argparse.ArgumentParser(
        description="Preprocess reference car predictions into ego car frame"
    )
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Root directory of the V2V4Real dataset")
    parser.add_argument('--label_dir', type=str, required=True,
                        help="Directory containing refcar detector .npy predictions "
                             "(relative to root_dir), e.g. 'refcar_predictions/npy'")
    parser.add_argument('--save_dir', type=str, default='refcar_predictions_preprocessed',
                        help="Output directory name (relative to root_dir)")
    parser.add_argument('--train_split', type=str, default='subset2',
                        choices=['subset1', 'subset2'])
    parser.add_argument('--ego_car_id', type=int, default=0)
    parser.add_argument('--ref_car_id', type=int, default=1)
    parser.add_argument('--min_num_points', type=int, default=10,
                        help="Minimum number of LiDAR points required to keep a box")
    parser.add_argument('--no_adjust_z', action='store_true',
                        help="Skip z-coordinate adjustment using ground plane. "
                             "Use this if above_ground_ransac/ is not available.")
    return parser.parse_args()


def main():
    args = parse_config()

    root_dir = args.root_dir
    ego_car_id = str(args.ego_car_id)
    ref_car_id = str(args.ref_car_id)

    scenario_folders = dataset_utils.get_scenario_folders(
        root_dir=root_dir, train_split=args.train_split, data_split='train'
    )

    keys = []
    for sf in scenario_folders:
        keys.extend([
            osp.join(sf, ref_car_id, x[:-4])
            for x in os.listdir(osp.join(sf, ref_car_id))
            if x.endswith(".pcd")
        ])
    print(f"Total frames: {len(keys)}")

    for key in tqdm(keys):
        traintest, scenario, _, timestamp = key.split("/")[-4:]

        pred_path = osp.join(root_dir, args.label_dir, traintest, scenario, ref_car_id, f"{timestamp}_pred.npy")
        ref_pred_np = np.load(pred_path)

        if ref_pred_np.ndim == 3:
            ref_pred_np = ref_pred_np.squeeze(1)
        assert ref_pred_np.ndim == 2

        # Load extrinsics to project from ref car frame to ego car frame
        ego_yml_path = osp.join(root_dir, traintest, scenario, ego_car_id, f'{timestamp}.yaml')
        ref_yml_path = osp.join(root_dir, traintest, scenario, ref_car_id, f'{timestamp}.yaml')
        ego_params = yaml_utils.load_yaml(ego_yml_path)
        ref_params = yaml_utils.load_yaml(ref_yml_path)
        ego_lidar_pose = ego_params['lidar_pose']
        ref_lidar_pose = ref_params['lidar_pose']
        transformation_matrix = transformation_utils.x1_to_x2(ref_lidar_pose, ego_lidar_pose)

        # Project boxes from ref car frame into ego car frame
        ref_pred_np = box_utils.boxes_to_corners_3d(ref_pred_np, order='lwh')
        ref_pred_np = box_utils.project_box3d(ref_pred_np, transformation_matrix)
        ref_pred_np = box_utils.corner_to_center(ref_pred_np, order='lwh')
        assert ref_pred_np.ndim == 2

        # Add the reference car itself as a labeled box
        refcar_bbox_np = np.array([[0., 0., -1.3, 5., 2., 1.5, 0.]])
        refcar_bbox_np = box_utils.boxes_to_corners_3d(refcar_bbox_np, order='lwh')
        refcar_bbox_np = box_utils.project_box3d(refcar_bbox_np, transformation_matrix)
        refcar_bbox_np = box_utils.corner_to_center(refcar_bbox_np, order='lwh')
        ref_pred_np = np.concatenate([ref_pred_np, refcar_bbox_np], axis=0)
        assert ref_pred_np.ndim == 2

        # Remove ego car detection (closest box to the LiDAR origin)
        if ref_pred_np.shape[0] > 0:
            xy = np.sqrt(ref_pred_np[:, 0] ** 2 + ref_pred_np[:, 1] ** 2)
            ego_car_idx = np.argmin(xy)
            if xy[ego_car_idx] < 3.0:
                mask = np.ones(ref_pred_np.shape[0], dtype=bool)
                mask[ego_car_idx] = False
                ref_pred_np = ref_pred_np[mask]
        assert ref_pred_np.ndim == 2

        # Adjust z using ground plane estimate
        if not args.no_adjust_z:
            above_plane_path = osp.join(
                root_dir, "above_ground_ransac", traintest, scenario, ego_car_id, f"{timestamp}.pkl"
            )
            with open(above_plane_path, 'rb') as f:
                above_plane_dict = pickle.load(f)
                coef = above_plane_dict['coef']
            ref_pred_np[:, 2] = (
                (-coef[3] - coef[0] * ref_pred_np[:, 0] - coef[1] * ref_pred_np[:, 1]) / coef[2]
            ) + 1.0
        assert ref_pred_np.ndim == 2

        # Remove far-away boxes (> 80m)
        if ref_pred_np.shape[0] > 0:
            xy = np.sqrt(ref_pred_np[:, 0] ** 2 + ref_pred_np[:, 1] ** 2)
            ref_pred_np = ref_pred_np[xy < 80.0]
        assert ref_pred_np.ndim == 2

        # Remove boxes with too few LiDAR points
        if ref_pred_np.shape[0] > 0:
            ptc_path = osp.join(root_dir, traintest, scenario, ego_car_id, f'{timestamp}.pcd')
            ptc_np = pcd_utils.pcd_to_np(ptc_path)

            ref_pred_np_3d = box_utils.boxes_to_corners_3d(ref_pred_np, order='lwh')
            ref_pred_np_2d = box_utils.box3d_to_2d(ref_pred_np_3d)

            mask_valid = [
                box_utils.get_points_in_rotated_box(ptc_np[:, :2], ref_pred_np_2d[j]).shape[0]
                > args.min_num_points
                for j in range(ref_pred_np.shape[0])
            ]
            ref_pred_np = ref_pred_np[mask_valid]

        if ref_pred_np.shape[0] == 0:
            ref_pred_np = np.empty((0, 7))

        save_path = osp.join(
            root_dir, args.save_dir, traintest, scenario, ego_car_id, f"{timestamp}_pred.npy"
        )
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        np.save(save_path, ref_pred_np)


if __name__ == "__main__":
    main()
