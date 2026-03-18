import argparse
import os
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from opencood.utils import eval_utils, train_utils, inference_utils, yaml_utils, box_utils, transformation_utils
from opencood.data_utils.datasets.v2v4real_dataset import V2V4RealDataset


eval_iou_list = [0.3, 0.5, 0.7]


def test_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained 3D object detector")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the model directory (contains config yaml and checkpoints)")
    parser.add_argument("--strict_model_path", type=str, default=None,
                        help="Direct path to a specific model checkpoint .pth file")
    parser.add_argument("--load_epoch", type=int, default=None,
                        help="Which epoch checkpoint to load (if --strict_model_path is not set)")

    parser.add_argument("--root_dir", type=str, default=None,
                        help="Root directory of the dataset (overrides config if set)")
    parser.add_argument("--data_split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--split_val_test", action="store_true",
                        help="Use val/test split (80%% test, 20%% val) of the first 20 clips")

    parser.add_argument("--ego_car_id", type=str, default='0')
    parser.add_argument("--num_lidar_beams", type=int, default=32, choices=[8, 16, 32])

    parser.add_argument("--min_timestamp", type=int, default=0)
    parser.add_argument("--max_timestamp", type=int, default=9999)

    # Dynamic score threshold: Tc + a / distance(ego, ref)
    parser.add_argument("--use_dynamic_score_threshold", action="store_true",
                        help="Use distance-based dynamic confidence threshold (Sec. 3.4)")
    parser.add_argument("--fixed_score_threshold", type=float, default=None,
                        help="Override the fixed confidence threshold from config")
    parser.add_argument("--a", type=float, default=1,
                        help="Coefficient for dynamic threshold: score_thresh = fixed + a/distance")
    parser.add_argument("--b", type=float, default=0)
    parser.add_argument("--c", type=float, default=0)

    # Save predictions as .npy for downstream label processing
    parser.add_argument("--save_npy", action="store_true",
                        help="Save predictions as .npy files for pseudo-label generation")
    parser.add_argument("--save_npy_n", type=int, default=5000,
                        help="Maximum number of frames to save as .npy")

    parser.add_argument("--save_path", type=str, default=None,
                        help="Output directory for saved predictions/visualizations")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save BEV visualization images")
    parser.add_argument("--save_vis_n", type=int, default=10)
    parser.add_argument("--print_stats", action="store_true",
                        help="Print per-frame GT/pred box counts")

    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    hypes["load_npy_label"] = False
    hypes["npy_label_path"] = None
    hypes["npy_label_idx"] = None

    if opt.fixed_score_threshold is not None:
        hypes["postprocess"]["target_args"]["score_threshold"] = opt.fixed_score_threshold
        print('Setting fixed score threshold:', opt.fixed_score_threshold)

    if hypes["model"]["core_method"] == "second":
        hypes["model"]["batch_size"] = 1
    elif hypes["model"]["core_method"] == "voxel_net":
        hypes["model"]["args"]["N"] = 1

    print("Building dataloader..")
    test_set = V2V4RealDataset(
        params=hypes,
        data_split=opt.data_split,
        split_val_test=opt.split_val_test,
        ego_car_id=opt.ego_car_id,
        return_original_points=True,
        num_lidar_beams=opt.num_lidar_beams,
        min_timestamp=opt.min_timestamp,
        max_timestamp=opt.max_timestamp,
        no_augment=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=16,
        collate_fn=test_set.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    print("Creating model..")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading model checkpoint..")
    if opt.strict_model_path:
        print(f"Loading from: {opt.strict_model_path}")
        model.load_state_dict(torch.load(opt.strict_model_path), strict=False)
    else:
        _, model = train_utils.load_saved_model(opt.model_dir, model, opt.load_epoch)

    model.eval()
    if hypes["model"]["core_method"] == "second":
        model.batch_size = 1

    save_path = opt.save_path if opt.save_path else opt.model_dir
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved to: {save_path}")

    result_stat = {}
    result_stat_short = {}
    result_stat_middle = {}
    result_stat_long = {}
    for cur_thresh in eval_iou_list:
        result_stat[cur_thresh] = {"tp": [], "fp": [], "gt": 0}
        result_stat_short[cur_thresh] = {"tp": [], "fp": [], "gt": 0}
        result_stat_middle[cur_thresh] = {"tp": [], "fp": [], "gt": 0}
        result_stat_long[cur_thresh] = {"tp": [], "fp": [], "gt": 0}

    if opt.save_vis:
        saver = inference_utils.SaverV2()
        saver.begin_background()

    num_gt_box_total = 0
    num_frames = len(test_loader.dataset)
    num_frames_object_exist = 0
    if opt.use_dynamic_score_threshold:
        print("Using distance-based dynamic score threshold")
    print(f"Number of frames: {num_frames}")

    for i, batch_data in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)

            output_dict = OrderedDict()
            cav_content = batch_data['ego']
            output_dict['ego'] = model(cav_content)

            if opt.use_dynamic_score_threshold:
                ego_yml_path = batch_data["ego"]["ego_yaml"]
                ref_yml_path = batch_data["ego"]["ref_yaml"]
                ego_params = yaml_utils.load_yaml(ego_yml_path)
                ref_params = yaml_utils.load_yaml(ref_yml_path)
                ego_lidar_pose = ego_params['lidar_pose']
                ref_lidar_pose = ref_params['lidar_pose']
                transformation_matrix = transformation_utils.x1_to_x2(ref_lidar_pose, ego_lidar_pose)
                refcar_bbox_np = np.array([[0., 0., -1.3, 5., 2., 1.5, 0.]])
                refcar_bbox_np = box_utils.boxes_to_corners_3d(refcar_bbox_np, order='lwh')
                refcar_bbox_np = box_utils.project_box3d(refcar_bbox_np, transformation_matrix)
                refcar_bbox_np = box_utils.corner_to_center(refcar_bbox_np, order='lwh')
                refcar_bbox_t = torch.from_numpy(refcar_bbox_np).to(device)
                pred_box_t, pred_score, gt_box_t = test_set.post_process(
                    batch_data, output_dict, opt.use_dynamic_score_threshold, refcar_bbox_t,
                    a=opt.a, b=opt.b, c=opt.c,
                )
            else:
                pred_box_t, pred_score, gt_box_t = test_set.post_process(
                    batch_data, output_dict, False, None,
                )

            num_gt_box = gt_box_t.shape[0] if gt_box_t is not None else 0
            num_pred_box = pred_box_t.shape[0] if pred_box_t is not None else 0

            if opt.print_stats:
                print(f"[Frame {i+1:05d}/{num_frames:05d}] GT/Pred: {num_gt_box:03d}/{num_pred_box:03d}")

            num_gt_box_total += num_gt_box
            if num_gt_box > 0:
                num_frames_object_exist += 1

            for cur_thresh in eval_iou_list:
                eval_utils.calculate_tp_fp(
                    pred_box_t, pred_score, gt_box_t, result_stat, cur_thresh
                )
                eval_utils.calculate_tp_fp(
                    pred_box_t, pred_score, gt_box_t, result_stat_short, cur_thresh,
                    left_range=0, right_range=30,
                )
                eval_utils.calculate_tp_fp(
                    pred_box_t, pred_score, gt_box_t, result_stat_middle, cur_thresh,
                    left_range=30, right_range=50,
                )
                eval_utils.calculate_tp_fp(
                    pred_box_t, pred_score, gt_box_t, result_stat_long, cur_thresh,
                    left_range=50, right_range=80,
                )

            current_traintest = batch_data["ego"]["traintest"]
            current_folder_name = batch_data["ego"]["folder_name"]
            current_timestep = batch_data["ego"]["timestep"]

            if opt.save_npy and opt.save_npy_n > i:
                npy_save_path = os.path.join(
                    save_path, "npy", current_traintest, current_folder_name, opt.ego_car_id
                )
                os.makedirs(npy_save_path, exist_ok=True)

                if num_pred_box == 0:
                    pred_box_center_np = np.empty([0, 7])
                else:
                    pred_box_np = pred_box_t.cpu().detach().numpy()
                    pred_box_center_np = box_utils.corner_to_center(pred_box_np, order='lwh')

                gt_box_np = gt_box_t.cpu().detach().numpy()
                gt_box_center_np = box_utils.corner_to_center(gt_box_np, order='lwh')

                np.save(os.path.join(npy_save_path, f'{current_timestep}_pred.npy'), pred_box_center_np)
                np.save(os.path.join(npy_save_path, f'{current_timestep}_gt.npy'), gt_box_center_np)

            if pred_box_t is not None:
                pred_box_t = pred_box_t.cpu().numpy()

            if opt.save_vis and opt.save_vis_n > i:
                saver.save_results(
                    model_dir=save_path,
                    i=i,
                    pred_box_t=pred_box_t,
                    gt_box_t=gt_box_t.cpu().numpy(),
                    origin_lidar=batch_data["ego"]["origin_lidar"][0].cpu().numpy(),
                    gt_range=[-100, -40, -5, 100, 40, 3],
                    folder_name=current_folder_name,
                    timestep=current_timestep,
                )

    eval_utils.eval_final_results(result_stat, save_path, eval_iou_list, "all")
    eval_utils.eval_final_results(result_stat_short, save_path, eval_iou_list, "short")
    eval_utils.eval_final_results(result_stat_middle, save_path, eval_iou_list, "middle")
    eval_utils.eval_final_results(result_stat_long, save_path, eval_iou_list, "long")

    if opt.save_vis:
        saver.end_background()


if __name__ == "__main__":
    main()
