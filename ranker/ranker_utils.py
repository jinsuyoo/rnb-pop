import torch
import numpy as np

from opencood.utils.iou3d_nms import iou3d_nms_utils


def get_transform(boxs: torch.Tensor) -> torch.Tensor:
    center_xyz = boxs[:, :3]
    rotation_xy = boxs[:, 6]

    cos, sin = torch.cos(rotation_xy), torch.sin(rotation_xy)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    rotation_z = torch.stack([cos,  -sin, zero, zero,
                              sin,   cos, zero, zero,
                              zero, zero,  one, zero,
                              zero, zero, zero,  one], dim=-1).view(-1, 4, 4).float()

    translation = torch.eye(4).to(boxs.device).unsqueeze(0).repeat(center_xyz.shape[0], 1, 1)
    translation[:, 3, :3] = -center_xyz
    return torch.matmul(translation, rotation_z)


def get_scale(
        ptc: torch.Tensor,  # (N_points, 4)
        boxs: torch.Tensor  # (M, 7)
        ):

    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)  # [N_points, 4]
    ptc = ptc.unsqueeze(dim=0).expand(boxs.shape[0], ptc.shape[0], 4)  # [M, N_points, 4]

    trs = get_transform(boxs)  # [M, 4, 4]
    ptc = torch.bmm(ptc, trs)[:, :, :3]  # [M, N_points, 3]
    original_ptc = ptc.clone()
    ptc = torch.abs(ptc)  # [M, N_points, 3]

    scale = ptc / (boxs[:, 3:6].unsqueeze(dim=1) * 0.5)  # right-hand side is [M, 1, 3]
    scale = torch.max(scale, dim=2).values
    return scale, ptc, trs, original_ptc


def sample_bboxs(boxes, num_samples, noise_xyz=1., noise_lwh=1., noise_angle=0.1):
    """
    Sample candidate boxes from Gaussian noise around the given boxes.

    Args:
        boxes: (M, 7) reference boxes
        noise_xyz: positional noise std
        noise_lwh: size noise std
        noise_angle: angular noise range
    """
    assert boxes.shape[-1] == 7
    num_boxes = boxes.shape[0]
    noise_xyz = torch.randn(num_samples, 3).to(boxes.dtype).to(boxes.device) * noise_xyz
    noise_lwh = torch.randn(num_samples, 3).to(boxes.dtype).to(boxes.device) * noise_lwh
    noise_angle = (2. * torch.rand(num_samples) - 1.).to(boxes.dtype).to(boxes.device) * noise_angle

    boxes_new = boxes.clone()
    try:
        sample_idx = np.random.choice(num_boxes, num_samples, replace=True)
    except Exception:
        print(num_boxes, num_samples)
    boxes_new = boxes_new[sample_idx]
    boxes_new[:, 3:6] += noise_lwh
    boxes_new[:, :3] += noise_xyz
    boxes_new[:, 6] += noise_angle
    return boxes_new


def sample_bboxes_coarse(boxes, num_samples, noise_xy=3., noise_z=1.):
    """
    Sample candidate boxes with only positional (xy uniform, z Gaussian) noise.

    Args:
        boxes: (M, 7) reference boxes
        noise_xy: uniform noise range for xy
        noise_z: Gaussian noise std for z
    """
    assert boxes.shape[-1] == 7
    num_boxes = boxes.shape[0]
    noise_xy = torch.FloatTensor(num_samples, 2).uniform_(-noise_xy, noise_xy).to(boxes.dtype).to(boxes.device)
    noise_z = torch.randn(num_samples, 1).to(boxes.dtype).to(boxes.device) * noise_z

    boxes_new = boxes.clone()
    try:
        sample_idx = np.random.choice(num_boxes, num_samples, replace=True)
    except Exception:
        print(num_boxes, num_samples)
    boxes_new = boxes_new[sample_idx]
    boxes_new[:, :2] += noise_xy
    boxes_new[:, 2:3] += noise_z
    return boxes_new


def sample_bboxes_fine(boxes, num_samples, noise_xyz=1., noise_hw=1., noise_l=1., noise_angle=0.1):
    """
    Sample candidate boxes with fine-grained positional and size noise.

    Args:
        boxes: (M, 7) reference boxes
        noise_xyz: positional noise std
        noise_hw: height/width noise std
        noise_l: length noise std
        noise_angle: angular noise range
    """
    assert boxes.shape[-1] == 7
    num_boxes = boxes.shape[0]
    noise_xyz = torch.randn(num_samples, 3).to(boxes.dtype).to(boxes.device) * noise_xyz
    noise_hw = torch.randn(num_samples, 2).to(boxes.dtype).to(boxes.device) * noise_hw
    noise_l = torch.randn(num_samples, 1).to(boxes.dtype).to(boxes.device) * noise_l
    noise_angle = (2. * torch.rand(num_samples) - 1.).to(boxes.dtype).to(boxes.device) * noise_angle

    boxes_new = boxes.clone()
    try:
        sample_idx = np.random.choice(num_boxes, num_samples, replace=True)
    except Exception:
        print(num_boxes, num_samples)
    boxes_new = boxes_new[sample_idx]
    boxes_new[:, 4:6] += noise_hw
    boxes_new[:, 3:4] += noise_l
    boxes_new[:, :3] += noise_xyz
    boxes_new[:, 6] += noise_angle
    return boxes_new


def sample_uniform_boxs(boxes, num_samples, noise_xyz=1., noise_lwh=1., noise_angle=0.1):
    """
    Sample candidate boxes from uniform noise around the given boxes.

    Args:
        boxes: (M, 7) reference boxes
        noise_xyz: uniform noise range for position
        noise_lwh: uniform noise range for size
        noise_angle: uniform noise range for angle
    """
    assert boxes.shape[-1] == 7
    num_boxes = boxes.shape[0]
    noise_xyz = torch.FloatTensor(num_samples, 3).uniform_(-noise_xyz, noise_xyz).to(boxes.dtype).to(boxes.device)
    noise_lwh = torch.FloatTensor(num_samples, 3).uniform_(-noise_lwh, noise_lwh).to(boxes.dtype).to(boxes.device)
    noise_angle = torch.FloatTensor(num_samples).uniform_(-noise_angle, noise_angle).to(boxes.dtype).to(boxes.device)

    boxes_new = boxes.clone()
    try:
        sample_idx = np.random.choice(num_boxes, num_samples, replace=True)
    except Exception:
        print(num_boxes, num_samples)
    boxes_new = boxes_new[sample_idx]
    boxes_new[:, :3] += noise_xyz
    boxes_new[:, 3:6] += noise_lwh
    boxes_new[:, 6] += noise_angle
    return boxes_new


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = iou3d_nms_utils.nms_gpu(
            boxes_for_nms[:, 0:7].float(), box_scores_nms.float(), nms_config.NMS_THRESH
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]
