"""
Filter and optionally combine pseudo-labels using a trained box ranker (Sec. 3.3).

For each frame, this script:
  1. Loads pseudo-labels from the current (and optionally previous) stage
  2. Scores candidate boxes with the box ranker
  3. Applies NMS and filters by score threshold
  4. Saves the filtered labels
"""

import os
import glob
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
import argparse

from opencood.utils import pcd_utils, box_utils, dataset_utils
from ranker.ranker_utils import get_scale


def parse_config():
    parser = argparse.ArgumentParser(
        description="Filter pseudo-labels using the trained box ranker"
    )

    parser.add_argument("--prev_label_dir", type=str, required=True,
                        help="Directory containing previous-stage pseudo-labels (relative to root_dir)")
    parser.add_argument("--prev_label_prefix", type=str, default="pred",
                        help="Filename suffix of previous-stage label .npy files")

    parser.add_argument("--cur_label_dir", type=str, required=True,
                        help="Directory containing current-stage predictions (relative to root_dir)")
    parser.add_argument("--cur_label_prefix", type=str, default="pred",
                        help="Filename suffix of current-stage label .npy files")

    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the V2V4Real dataset")

    parser.add_argument("--combine_method", type=str, default="current_only",
                        choices=["current_only", "current_and_previous"],
                        help="How to combine previous and current labels")

    parser.add_argument("--filtering_method", type=str, default="threshold",
                        choices=["topk", "threshold", "none"],
                        help="Score-based filtering strategy")

    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for filtered labels (relative to root_dir)")
    parser.add_argument("--save_prefix", type=str, default="pred",
                        help="Filename suffix for saved label files")

    parser.add_argument("--ego_car_id", type=int, default=0)

    parser.add_argument("--top_k", type=int, default=70,
                        help="Percentage of boxes to keep (for topk filtering)")
    parser.add_argument("--ranker_score_threshold", type=float, default=0.5,
                        help="Minimum ranker score to keep a box (for threshold filtering)")

    parser.add_argument("--ranker_path", type=str, default=None,
                        help="Path to pretrained ranker checkpoint (.pth)")
    parser.add_argument("--use_ground", action="store_true",
                        help="Pass ground point mask to ranker")
    parser.add_argument("--use_offset", action="store_true",
                        help="Use ranker's predicted box offset")

    args = parser.parse_args()
    return args


def main():
    args = parse_config()

    scenario_folders = dataset_utils.get_scenario_folders(
        root_dir=args.root_dir, train_split="subset2", data_split="train",
    )

    ego_car_id = str(args.ego_car_id)

    keys = []
    for sf in scenario_folders:
        num_timestamps = len(glob.glob(osp.join(sf, ego_car_id, "*.pcd")))
        for timestamp in range(num_timestamps):
            keys.append(osp.join(sf, ego_car_id, f"{timestamp:06d}"))

    print(f"Total frames: {len(keys)}")
    print(f"Combine: {args.combine_method}, Filtering: {args.filtering_method}")

    ranker = None
    if args.ranker_path is not None:
        from ranker.pointnet import PointNetRanker
        print("Loading ranker...")
        ranker = PointNetRanker(use_ground=args.use_ground, use_offset=args.use_offset)
        ranker.cuda()
        ranker.eval()
        ranker.load_state_dict(torch.load(args.ranker_path))
        print("Ranker loaded.")

    for i in tqdm(range(len(keys))):
        key = keys[i]
        traintest, folder_name, _, timestep = key.split("/")[-4:]

        save_path = osp.join(args.root_dir, args.save_dir, traintest, folder_name, ego_car_id)
        os.makedirs(save_path, exist_ok=True)
        save_filename = osp.join(save_path, f"{timestep}_{args.save_prefix}.npy")

        ptc_path = osp.join(args.root_dir, traintest, folder_name, ego_car_id, f"{timestep}.pcd")
        ptc = pcd_utils.pcd_to_np(ptc_path)

        prev_bbox_np = np.load(osp.join(
            args.root_dir, args.prev_label_dir,
            traintest, folder_name, ego_car_id,
            f"{timestep}_{args.prev_label_prefix}.npy"
        ))
        if prev_bbox_np.ndim == 3:
            prev_bbox_np = prev_bbox_np.squeeze(1)

        cur_bbox_np = np.load(osp.join(
            args.root_dir, args.cur_label_dir,
            traintest, folder_name, ego_car_id,
            f"{timestep}_{args.cur_label_prefix}.npy"
        ))
        if cur_bbox_np.ndim == 3:
            cur_bbox_np = cur_bbox_np.squeeze(1)

        if prev_bbox_np.shape[0] == 0 and cur_bbox_np.shape[0] == 0:
            np.save(save_filename, np.empty([0, 7]))
            continue

        if args.combine_method == "current_only":
            improved_label = cur_bbox_np if cur_bbox_np.shape[0] > 0 else np.empty([0, 7])
        elif args.combine_method == "current_and_previous":
            if prev_bbox_np.shape[0] == 0:
                improved_label = cur_bbox_np
            elif cur_bbox_np.shape[0] == 0:
                improved_label = prev_bbox_np
            else:
                improved_label = np.concatenate([prev_bbox_np, cur_bbox_np], axis=0)
        else:
            raise ValueError(f"Invalid combine_method: {args.combine_method}")

        if improved_label.shape[0] == 0:
            np.save(save_filename, np.empty([0, 7]))
            continue

        ptc_t = torch.from_numpy(ptc).float().cuda()
        improved_label_t = torch.from_numpy(improved_label).float().cuda()

        ref_scale, ref_ptc, ref_trs, ref_original_ptc = get_scale(ptc_t, improved_label_t)
        valid = (ref_scale < 3.0).to(ref_scale.dtype)

        obj_corner = box_utils.boxes_to_corners_3d(improved_label_t, "lwh")
        obj_corner = torch.cat(
            [obj_corner, torch.ones([obj_corner.shape[0], 8, 1]).to(obj_corner.device)], dim=2
        )
        new_obj_corner_t = torch.bmm(obj_corner.float(), ref_trs.float())[..., :3]
        new_obj_center_np = box_utils.corner_to_center(new_obj_corner_t.cpu().numpy())

        valid_ptc_list = []
        for v in range(valid.shape[0]):
            valid_ptc = ref_original_ptc[v][valid[v].bool()]
            if len(valid_ptc) == 0:
                valid_ptc_list.append(torch.zeros([1, 1000, 3], device=obj_corner.device))
                continue
            choice = np.random.choice(len(valid_ptc), 1000, replace=True)
            valid_ptc_list.append(valid_ptc[choice, :].unsqueeze(0))

        ptc_ranker_input = torch.cat(valid_ptc_list, dim=0).float()
        box_ranker_input = torch.from_numpy(new_obj_center_np).cuda().float()
        sampled_bbox_distance = torch.sqrt(
            torch.sum(torch.pow(improved_label_t[..., :2], 2), dim=-1)
        )

        with torch.no_grad():
            if ranker is not None and args.use_offset:
                estimated_iou, _, _, _ = ranker(
                    box_ranker_input, ptc_ranker_input, None, sampled_bbox_distance)
            elif ranker is not None:
                estimated_iou, _, _ = ranker(
                    box_ranker_input, ptc_ranker_input, None, sampled_bbox_distance)
            else:
                estimated_iou = torch.ones(improved_label_t.shape[0], 1, device="cuda")

        estimated_iou[torch.isnan(estimated_iou)] = 0.0
        estimated_iou[estimated_iou > 2.0] = 0.0
        estimated_iou = estimated_iou.squeeze(1)

        num_boxes_before = improved_label_t.shape[0]

        keep_index = box_utils.nms_rotated(
            box_utils.boxes_to_corners_3d(improved_label_t, "lwh"), estimated_iou, 0.01
        )
        improved_label_nms_t = improved_label_t[keep_index]
        estimated_iou = estimated_iou[keep_index]

        if args.filtering_method == "topk":
            num_selection = min(max(int(improved_label_nms_t.shape[0] * args.top_k / 100.), 1), 30)
            top_scores = estimated_iou.topk(num_selection, dim=0).indices
            improved_label_nms_t = improved_label_nms_t[top_scores]
        elif args.filtering_method == "threshold":
            final_mask = estimated_iou > args.ranker_score_threshold
            improved_label_nms_t = improved_label_nms_t[final_mask]
        elif args.filtering_method == "none":
            pass

        num_boxes_after = improved_label_nms_t.shape[0]
        print(f"[{i:05d}] boxes before/after filtering: {num_boxes_before}/{num_boxes_after}")

        final_label = improved_label_nms_t.cpu().numpy()
        assert final_label.ndim == 2 and final_label.shape[1] == 7
        np.save(save_filename, final_label)


if __name__ == "__main__":
    main()
