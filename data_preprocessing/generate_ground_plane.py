"""
Generate per-frame ground plane estimates using RANSAC.

For each LiDAR frame, fits a ground plane to low-lying points and saves:
  - 'coef': plane equation coefficients (4,)
  - 'mask_above_plane': boolean mask (N,) indicating above-ground points

Output is saved to:
  <root_dir>/above_ground_ransac/{traintest}/{scenario}/{car_id}/{timestamp}.pkl

Run for each car separately (ego car: car_id=0, reference car: car_id=1):

    python data_preprocessing/generate_ground_plane.py \\
        --root_dir /path/to/v2v4real \\
        --train_split subset2 \\
        --car_id 0

    python data_preprocessing/generate_ground_plane.py \\
        --root_dir /path/to/v2v4real \\
        --train_split subset2 \\
        --car_id 1
"""

import argparse
import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
from opencood.utils import pcd_utils, dataset_utils
from sklearn.linear_model import RANSACRegressor


def estimate_plane(origin_ptc, max_hs=-1.5, it=1, ptc_range=((-20, 70), (-20, 20))):
    mask = (
        (origin_ptc[:, 2] < max_hs)
        & (origin_ptc[:, 0] > ptc_range[0][0])
        & (origin_ptc[:, 0] < ptc_range[0][1])
        & (origin_ptc[:, 1] > ptc_range[1][0])
        & (origin_ptc[:, 1] < ptc_range[1][1])
    )
    for _ in range(it):
        ptc = origin_ptc[mask]
        reg = RANSACRegressor().fit(ptc[:, [0, 1]], ptc[:, 2])
        w = np.zeros(3)
        w[0] = reg.estimator_.coef_[0]
        w[1] = reg.estimator_.coef_[1]
        w[2] = -1.0
        h = reg.estimator_.intercept_
        norm = np.linalg.norm(w)
        w /= norm
        h = h / norm
        result = np.array((w[0], w[1], w[2], h))
        result *= -1
        mask = np.logical_not(above_plane(origin_ptc[:, :3], result, offset=0.2))
    return result


def above_plane(ptc, plane, offset=0.05, only_range=((-30, 30), (-30, 30))):
    mask = distance_to_plane(ptc, plane, directional=True) < offset
    if only_range is not None:
        range_mask = (
            (ptc[:, 0] < only_range[0][1])
            * (ptc[:, 0] > only_range[0][0])
            * (ptc[:, 1] < only_range[1][1])
            * (ptc[:, 1] > only_range[1][0])
        )
        mask *= range_mask
    return np.logical_not(mask)


def distance_to_plane(ptc, plane, directional=False):
    d = ptc @ plane[:3] + plane[3]
    if not directional:
        d = np.abs(d)
    d /= np.sqrt((plane[:3] ** 2).sum())
    return d


def main(args):
    root_dir = args.root_dir
    save_dir = args.save_dir
    car_id = str(args.car_id)
    print(f"car_id: {car_id}")

    scenario_folders = dataset_utils.get_scenario_folders(
        root_dir=root_dir, train_split=args.train_split, data_split="train", split_val_test=False
    )

    keys = []
    for sf in scenario_folders:
        keys.extend([
            osp.join(sf, car_id, x[:-4])
            for x in os.listdir(osp.join(sf, car_id))
            if x.endswith(".pcd")
        ])
    print(f"Total frames: {len(keys)}")

    for key in tqdm(keys):
        traintest, scenario, _, timestamp = key.split("/")[-4:]

        ptc_path = osp.join(root_dir, traintest, scenario, car_id, f'{timestamp}.pcd')
        ptc_np = pcd_utils.pcd_to_np(ptc_path)

        ransac = estimate_plane(ptc_np, max_hs=-1.5, it=5, ptc_range=((-100, 100), (-100, 100)))
        mask_above_plane = above_plane(ptc_np[:, :3], ransac, offset=0.5)

        ransac_dict = {
            'coef': ransac,
            'mask_above_plane': mask_above_plane,
        }

        save_path = osp.join(root_dir, save_dir, traintest, scenario, car_id, f"{timestamp}.pkl")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(ransac_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-frame ground plane estimates using RANSAC")
    parser.add_argument('--root_dir', type=str, required=True,
                        help="Root directory of the V2V4Real dataset")
    parser.add_argument('--save_dir', type=str, default='above_ground_ransac',
                        help="Output subdirectory name under root_dir")
    parser.add_argument('--train_split', type=str, default='subset2',
                        choices=['subset1', 'subset2'],
                        help="Dataset split to process")
    parser.add_argument('--car_id', type=int, default=0,
                        help="Car ID to process (0=ego, 1=reference)")

    args = parser.parse_args()
    main(args)
