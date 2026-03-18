"""
Refine noisy pseudo-labels from a reference car using a trained box ranker (Sec. 3.3).

For each detected box, this script:
  1. Samples N candidate boxes around the initial (noisy) detection
  2. Uses coarse-to-fine (C2F) sampling to handle large localization errors
  3. Scores candidates using the PointNet-based ranker
  4. Saves the highest-scoring candidate as the refined pseudo-label
"""

import argparse
import os
import os.path as osp
import logging
import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from opencood.utils import yaml_utils, box_utils, multi_gpu_utils
from opencood.data_utils.datasets.v2v4real_dataset import V2V4RealDataset
from opencood.utils.logger import get_root_logger
from ranker.ranker_utils import (
    get_scale,
    sample_bboxs,
    sample_bboxes_coarse,
    sample_bboxes_fine,
)


def parse_config():
    parser = argparse.ArgumentParser(
        description="Refine pseudo-labels using the trained box ranker"
    )

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the detector config YAML file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained ranker checkpoint (.pth)")

    parser.add_argument("--initial_label_path", type=str, required=True,
                        help="Directory containing initial pseudo-labels (.npy files)")
    parser.add_argument("--npy_label_idx", type=str, default="pred.npy",
                        help="Filename suffix of the label .npy files")

    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory to save refined pseudo-labels")
    parser.add_argument("--ranker_save_prefix", type=str, default="ranker",
                        help="Prefix for saved refined label filenames")

    parser.add_argument("--data_split", type=str, default="train",
                        choices=["train", "val", "test"])

    parser.add_argument("--sampling_method", type=str, default="coarse_to_fine",
                        choices=["baseline", "coarse_to_fine"],
                        help="Box sampling strategy (coarse_to_fine recommended)")
    parser.add_argument("--num_samples", type=int, default=512,
                        help="Total number of candidate boxes to sample per detection")
    parser.add_argument("--top_k_coarse", type=int, default=3,
                        help="Top-K boxes from coarse stage for fine refinement")
    parser.add_argument("--batch_size_ranker", type=int, default=512)

    parser.add_argument("--use_ground", action="store_true",
                        help="Include ground point mask as ranker input feature")
    parser.add_argument("--use_offset", action="store_true",
                        help="Use ranker's predicted box offset to further correct location")
    parser.add_argument("--adjust_with_estimated_offset", action="store_true",
                        help="Apply the predicted offset to the selected best box")

    parser.add_argument("--min_timestep", type=int, default=0)
    parser.add_argument("--iou_threshold", type=float, default=0.25)

    parser.add_argument("--no_dist", action="store_true",
                        help="Disable distributed processing")
    parser.add_argument("--dist_url", default="env://",
                        help="URL used to set up distributed training")

    args = parser.parse_args()

    random.seed(310)
    np.random.seed(310)
    torch.manual_seed(310)

    cfg = yaml_utils.load_yaml(None, args, yaml_path=args.model_dir)
    return args, cfg


def _score_candidates_with_ranker(
    ranker, sampled_bbox_center, ptc, ground, sampled_bbox_distance,
    batch_size_ranker, num_steps, use_ground, use_offset,
):
    """
    Run the ranker on a batch of candidate boxes and return scores and offsets.

    Args:
        ranker: PointNetRanker model
        sampled_bbox_center: (N, 7) candidate boxes in ego frame
        ptc: (P, 4) full scene point cloud
        ground: (P,) ground point mask
        sampled_bbox_distance: (N,) distance from ego to each candidate box center
        batch_size_ranker: batch size for ranker inference
        num_steps: number of batches
        use_ground: whether to pass ground mask to ranker
        use_offset: whether ranker outputs box offset
    Returns:
        bbox_scores: (N, 1) predicted IoU scores
        bbox_offsets: (N, 7) predicted offsets (None if use_offset is False)
    """
    ref_scale, ref_ptc, ref_trs, ref_original_ptc = get_scale(ptc, sampled_bbox_center)
    scale_range = 3.0
    valid = (ref_scale < scale_range).to(ref_scale.dtype)

    obj_corner = box_utils.boxes_to_corners_3d(sampled_bbox_center, 'lwh')
    obj_corner = torch.cat(
        [obj_corner, torch.ones([obj_corner.shape[0], 8, 1]).to(obj_corner.device)], dim=2
    )
    new_obj_corner_t = torch.bmm(obj_corner.float(), ref_trs.float())[..., :3]
    new_obj_center_np = box_utils.corner_to_center(new_obj_corner_t.cpu().numpy())

    valid_ptc_list = []
    valid_ground_list = []
    for v in range(valid.shape[0]):
        valid_ptc = ref_original_ptc[v][valid[v].bool()]
        if len(valid_ptc) == 0:
            valid_ptc_list.append(torch.zeros([1, 1000, 3], device=obj_corner.device))
            valid_ground_list.append(torch.zeros([1, 1000], device=obj_corner.device))
            continue
        choice = np.random.choice(len(valid_ptc), 1000, replace=True)
        valid_ptc_list.append(valid_ptc[choice, :].unsqueeze(0))
        valid_ground_list.append(ground[choice].unsqueeze(0))

    ptc_ranker_input = torch.cat(valid_ptc_list, dim=0).float()
    ground_ranker_input = torch.cat(valid_ground_list, dim=0).unsqueeze(-1).float()
    box_ranker_input = torch.from_numpy(new_obj_center_np).cuda().float()

    bbox_scores = []
    bbox_offsets = []
    for bb in range(num_steps):
        start = bb * batch_size_ranker
        end = start + batch_size_ranker
        with torch.no_grad():
            ground_input = ground_ranker_input[start:end] if use_ground else None
            if use_offset:
                iou_pred, offset_pred, _, _ = ranker(
                    box_ranker_input[start:end],
                    ptc_ranker_input[start:end],
                    ground_input,
                    sampled_bbox_distance[start:end],
                )
                bbox_offsets.append(offset_pred)
            else:
                iou_pred, _, _ = ranker(
                    box_ranker_input[start:end],
                    ptc_ranker_input[start:end],
                    ground_input,
                    sampled_bbox_distance[start:end],
                )
        iou_pred = iou_pred.clone()
        iou_pred[torch.isnan(iou_pred)] = 0.0
        iou_pred[iou_pred > 2.0] = 0.0
        bbox_scores.append(iou_pred)

    bbox_scores = torch.cat(bbox_scores, dim=0)
    bbox_offsets = torch.cat(bbox_offsets, dim=0) if use_offset else None
    return bbox_scores, bbox_offsets


def main():
    args, hypes = parse_config()

    multi_gpu_utils.init_distributed_mode(args, print_master_only=False)
    print(f'World size: {args.world_size}, Rank: {args.rank}')

    hypes["load_npy_label"] = True
    hypes["npy_label_idx"] = args.npy_label_idx
    hypes["npy_label_order"] = "lwh"
    hypes["npy_label_path"] = args.initial_label_path

    log_file = "refine_labels.log"
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    logger.info("Building dataloader..")
    dataset = V2V4RealDataset(
        params=hypes,
        data_split=args.data_split,
        return_original_points=True,
        return_original_ground=True,
        no_augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        drop_last=False,
    )
    logger.info("Dataloader ready.")

    from ranker.pointnet import PointNetRanker
    ranker = PointNetRanker(use_ground=args.use_ground, use_offset=args.use_offset)
    ranker.cuda()
    ranker.eval()
    ranker.load_state_dict(torch.load(args.model_path))
    logger.info("Ranker loaded.")

    for i, batch in enumerate(loader):
        # Distributed: each process handles a subset of frames
        if not (i % args.world_size == args.rank):
            continue

        folder_name = batch["ego"]["folder_name"]
        traintest = batch["ego"]["traintest"]
        timestep = batch["ego"]["timestep"]

        if int(timestep) < args.min_timestep:
            continue

        bbox_center = batch["ego"]["object_bbx_center"].cuda()
        bbox_mask = batch["ego"]["object_bbx_mask"].cuda()
        ptc = batch["ego"]["origin_lidar"].squeeze(0).cuda()  # (P, 4)
        ground = batch["ego"]["origin_ground"].squeeze(0).squeeze(1).cuda()  # (P,)

        cur_save_path = osp.join(args.save_dir, traintest, folder_name, "0")
        os.makedirs(cur_save_path, exist_ok=True)
        save_prefix = osp.join(cur_save_path, f"{timestep}")

        bbox_center = bbox_center[bbox_mask.bool()]  # (M, 7)
        bbox_center[:, 3:6] = bbox_center[:, [5, 4, 3]]  # hwl -> lwh

        if bbox_center.shape[0] == 0:
            np.save(save_prefix + f"_ranker_{args.ranker_save_prefix}.npy", np.empty([0, 7]))
            logger.info(f"Frame {i:05d}/{len(loader):05d}: no boxes, skipped")
            continue

        refined_bbox_list = []

        for j in range(bbox_center.shape[0]):
            cur_bbox = bbox_center[j].unsqueeze(0)

            # ---------- Coarse sampling ----------
            if args.sampling_method == "baseline":
                sampled = sample_bboxs(cur_bbox, args.num_samples - 1,
                                       noise_xyz=1., noise_lwh=.1, noise_angle=.1)
            else:  # coarse_to_fine
                sampled = sample_bboxes_coarse(cur_bbox, args.num_samples // 2 - 1,
                                               noise_xy=1.5, noise_z=0.5)

            sampled = torch.cat([cur_bbox, sampled], dim=0)
            sampled[:, 3:6] = torch.clamp(sampled[:, 3:6], min=0.01)
            dist = torch.sqrt(torch.sum(sampled[:, :3] ** 2, dim=1)).float()

            if args.sampling_method == "baseline":
                num_steps = sampled.shape[0] // args.batch_size_ranker
            else:
                num_steps = sampled.shape[0] // (args.batch_size_ranker // 2)

            scores, offsets = _score_candidates_with_ranker(
                ranker, sampled, ptc, ground, dist,
                args.batch_size_ranker, num_steps,
                args.use_ground, args.use_offset,
            )

            if args.sampling_method == "baseline":
                _, best_idx = scores.topk(1, dim=0)
                best_box = sampled[best_idx, :]
                if args.use_offset and args.adjust_with_estimated_offset:
                    best_box = best_box - offsets[best_idx, :]
                refined_bbox_list.append(best_box.cpu().numpy())

            else:  # coarse_to_fine
                # Select top-K from coarse stage
                _, top_k_idx = scores.topk(args.top_k_coarse, dim=0)
                top_k_boxes = sampled[torch.squeeze(top_k_idx)]  # (K, 7)
                if args.use_offset and args.adjust_with_estimated_offset:
                    top_k_boxes = top_k_boxes - offsets[torch.squeeze(top_k_idx)]

                # ---------- Fine sampling ----------
                sampled_fine = sample_bboxes_fine(
                    top_k_boxes, args.num_samples // 2 - 1,
                    noise_xyz=0.25, noise_hw=.2, noise_l=.4, noise_angle=0.1,
                )
                sampled_fine = torch.cat([cur_bbox, sampled_fine], dim=0)
                sampled_fine[:, 3:6] = torch.clamp(sampled_fine[:, 3:6], min=0.01)
                dist_fine = torch.sqrt(torch.sum(sampled_fine[:, :3] ** 2, dim=1)).float()

                scores_fine, offsets_fine = _score_candidates_with_ranker(
                    ranker, sampled_fine, ptc, ground, dist_fine,
                    args.batch_size_ranker, num_steps,
                    args.use_ground, args.use_offset,
                )

                _, best_idx = scores_fine.topk(1, dim=0)
                best_box = sampled_fine[best_idx, :]
                if args.use_offset and args.adjust_with_estimated_offset:
                    best_box = best_box - offsets_fine[best_idx, :]
                refined_bbox_list.append(best_box.cpu().numpy())

        refined_np = np.concatenate(refined_bbox_list, axis=0)
        np.save(save_prefix + f"_ranker_{args.ranker_save_prefix}.npy", refined_np)

        logger.info(f"Frame {i:05d}/{len(loader):05d} processed")


if __name__ == "__main__":
    main()
