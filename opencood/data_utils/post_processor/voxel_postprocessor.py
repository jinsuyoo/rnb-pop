"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils

from opencood.utils.iou3d_nms import iou3d_nms_utils


MULTI_CLASSES_NMS = False
NMS_TYPE = 'nms_gpu'
NMS_THRESH = 0.0
NMS_PRE_MAXSIZE = 4096
NMS_POST_MAXSIZE = 500

def class_agnostic_nms(box_scores, box_preds, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        #print(box_scores.shape, NMS_PRE_MAXSIZE, min(NMS_PRE_MAXSIZE, box_scores.shape[0]))
        box_scores_nms, indices = torch.topk(box_scores, k=min(NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        #keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
        #        boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        #)
        keep_idx, selected_scores = iou3d_nms_utils.nms_gpu(
                boxes_for_nms[:, 0:7], box_scores_nms, NMS_THRESH
        )
        selected = indices[keep_idx[:NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]

def corner2d_to_standup_box_torch(box2d):
    """
    Find the minmaxx, minmaxy for each 2d box. (N, 4, 2) -> (N, 4)
    x1, y1, x2, y2

    Parameters
    ----------
    box2d : np.ndarray
        (n, 4, 2), four corners of the 2d bounding box.

    Returns
    -------
    standup_box2d : np.ndarray
        (n, 4)
    """
    N = box2d.shape[0]
    standup_boxes2d = torch.zeros((N, 4))

    standup_boxes2d[:, 0] = torch.min(box2d[:, :, 0], dim=1)[0]
    standup_boxes2d[:, 1] = torch.min(box2d[:, :, 1], dim=1)[0]
    standup_boxes2d[:, 2] = torch.max(box2d[:, :, 0], dim=1)[0]
    standup_boxes2d[:, 3] = torch.max(box2d[:, :, 1], dim=1)[0]

    return standup_boxes2d


#def unravel_index(indices, shape):
#    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.
#
#    This is a `torch` implementation of `numpy.unravel_index`.
#
#    Args:
#        indices: A tensor of flat indices, (*,).
#        shape: The target shape.
#
#    Returns:
#        The unraveled coordinates, (*, D).
#    """
#
#    shape = indices.new_tensor((*shape, 1))
#    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
#    print('this:',torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]
#    )
#    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps_torch(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """https://github.com/open-mmlab/mmdetection/
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh']
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2

        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl':
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = \
            box_utils.boxes_to_corners_3d(gt_box_center_valid,
                                          self.params['order'])
        # (H*W*anchor_num, 8, 3)
        anchors_corner = \
            box_utils.boxes_to_corners_3d(anchors,
                                          order=self.params['order'])
        # (H*W*anchor_num, 4)
        anchors_standup_2d = \
            box_utils.corner2d_to_standup_box(anchors_corner)
        # (n, 4)
        gt_standup_2d = \
            box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        # (H*W*anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold'])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        #print('correct:', id_pos, (*feature_map_shape, self.anchor_num))
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict

    def generate_label_torch(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        device = kwargs['device']
        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = torch.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one = torch.zeros((*feature_map_shape, self.anchor_num)).to(device)
        neg_equal_one = torch.zeros((*feature_map_shape, self.anchor_num)).to(device)
        # (H, W, self.anchor_num * 7)
        targets = torch.zeros((*feature_map_shape, self.anchor_num * 7)).to(device)

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(gt_box_center_valid, self.params['order'])
        # (H*W*anchor_num, 8, 3)
        anchors_corner = box_utils.boxes_to_corners_3d(anchors, order=self.params['order'])
        
        # (H*W*anchor_num, 4)
        anchors_standup_2d = corner2d_to_standup_box_torch(anchors_corner)
        # (n, 4)
        gt_standup_2d = corner2d_to_standup_box_torch(gt_box_corner_valid)

        # (H*W*anchor_n)
        #iou = bbox_overlaps(
        #    np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
        #    np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        #)

        iou = bbox_overlaps_torch(anchors_standup_2d.float(), gt_standup_2d.float())

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = torch.argmax(iou.T, dim=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = torch.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = torch.where(iou > self.params['target_args']['pos_threshold'])
        #  find anchors iou  params['neg_iou']
        id_neg = torch.where(torch.sum(iou < self.params['target_args']['neg_threshold'], dim=1) == iou.shape[1])[0]
        id_pos = torch.cat([id_pos, id_highest])
        id_pos_gt = torch.cat([id_pos_gt, id_highest_gt])
        id_pos, index = unique(id_pos) #np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()


        # cal the target and set the equal one
        #print(id_pos, (*feature_map_shape, self.anchor_num))
        index_x, index_y, index_z = unravel_index(id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        #import pdb; pdb.set_trace()

        # calculate the targets
        targets[index_x, index_y, index_z * 7] = ((gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[id_pos]).float()
        targets[index_x, index_y, index_z * 7 + 1] = ((gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[id_pos]).float()
        targets[index_x, index_y, index_z * 7 + 2] = ((gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[id_pos, 3]).float()
        targets[index_x, index_y, index_z * 7 + 3] = (torch.log(gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])).float()
        targets[index_x, index_y, index_z * 7 + 4] = (torch.log(gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])).float()
        targets[index_x, index_y, index_z * 7 + 5] = (torch.log(gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])).float()
        targets[index_x, index_y, index_z * 7 + 6] = ((gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])).float()

        index_x, index_y, index_z = unravel_index(id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = unravel_index(id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = torch.from_numpy(np.array(neg_equal_one))
        targets = torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    def compute_distance_based_score_threshold(
        self, score_threshold, batch_box3d, refcar_bbox,
        a=0.3, b=0.2, c=0.1
    ):
        """
        Compute the dynamic score threshold based on the distance between the
        reference car, ego car, and the predicted car.

        Parameters
        ----------
        score_threshold : float
            The original score threshold.

        batch_box3d : np.ndarray
            The 3D bounding box tensor.

        refcar_bbox : np.ndarray
            The reference car bounding box.

        Returns
        -------
        dynamic_score_threshold : float
            The dynamic score threshold.
        """
        refcar_bbox = refcar_bbox.unsqueeze(1)

        distance_ego_ref = torch.sqrt(torch.sum(
            torch.pow(refcar_bbox[..., :2], 2), dim=-1))
  
        distance_ref_pred = torch.sqrt(torch.sum(
            torch.pow(torch.subtract(refcar_bbox[..., :2], batch_box3d[..., :2]), 2), dim=-1))

        distance_ego_pred = torch.sqrt(torch.sum(
            torch.pow(batch_box3d[..., :2], 2), dim=-1))

        distance_based_score_threshold = score_threshold + (a * (1/distance_ego_ref))
        #distance_based_score_threshold = score_threshold * torch.exp(-(distance_ego_ref / a))

        return distance_based_score_threshold, distance_ego_ref, distance_ref_pred, distance_ego_pred


    def post_process(
        self, 
        data_dict, 
        output_dict, 
        dynamic_score_threshold=False, 
        refcar_bbox=None, 
        a=0.3, b=0.2, c=0.1
    ):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        #if dynamic_score_threshold:
        #    print("# post_process dev mode!!")
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        
        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']
            
            # classification probability
            prob = output_dict[cav_id]['psm']
            prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['rm']

            # convert regression map back to bounding box
            # (N, W*L*anchor_num, 7)
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)

            if dynamic_score_threshold:
                distance_based_score_threshold, distance_ego_ref, distance_ref_pred, distance_ego_pred = \
                    self.compute_distance_based_score_threshold(
                        self.params['target_args']['score_threshold'],
                        batch_box3d, refcar_bbox,
                        a=a, b=b, c=c
                    ) 
                mask = torch.gt(prob, distance_based_score_threshold)
            else:
                mask = torch.gt(prob, self.params['target_args']['score_threshold'])

            # prob.shape: [1, 10000] / self.params['target_args']['score_threshold']: 0.2 (float)
            #mask = torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1, print(batch_box3d.shape)
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])
                # (N, 8, 3)
                projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
                
                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
        
        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)

        #import pdb; pdb.set_trace()
        # nms
        #print(pred_box3d_tensor.shape) 16896 x 8 x 3
        keep_index = box_utils.nms_rotated(
            pred_box3d_tensor, scores, self.params['nms_thresh'])

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        #print("before nms:", scores.shape)
        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range.
        mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]
        #print("after nms", scores.shape)
        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    def post_process_multi(
        self, 
        data_dict, 
        output_dict, 
        dynamic_score_threshold=False, 
        refcar_bbox=None, 
        a=0.3, b=0.2, c=0.1
    ):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        #if dynamic_score_threshold:
        #    print("# post_process dev mode!!")
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []

        pred_box3d_truck_list = []
        pred_box2d_truck_list = []
        
        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            anchor_box_truck = cav_content['anchor_box_truck']
            
            # classification probability
            prob = output_dict[cav_id]['psm']
            prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['rm']

            prob_truck = output_dict[cav_id]['psm_truck']
            prob_truck = torch.sigmoid(prob_truck.permute(0, 2, 3, 1))
            prob_truck = prob_truck.reshape(1, -1)

            reg_truck = output_dict[cav_id]['rm_truck']

            # convert regression map back to bounding box
            # (N, W*L*anchor_num, 7)
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)

            batch_box3d_truck = self.delta_to_boxes3d(reg_truck, anchor_box_truck)

            if dynamic_score_threshold:
                distance_based_score_threshold, distance_ego_ref, distance_ref_pred, distance_ego_pred = \
                    self.compute_distance_based_score_threshold(
                        self.params['target_args']['score_threshold'],
                        batch_box3d, refcar_bbox,
                        a=a, b=b, c=c
                    ) 
                mask = torch.gt(prob, distance_based_score_threshold)
                mask_truck = torch.gt(prob_truck, distance_based_score_threshold)
            else:
                mask = torch.gt(prob, self.params['target_args']['score_threshold'])
                mask_truck = torch.gt(prob_truck, 0.1) #self.params['target_args']['score_threshold'])
            #import pdb; pdb.set_trace()
            # prob.shape: [1, 10000] / self.params['target_args']['score_threshold']: 0.2 (float)
            #mask = torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            mask_truck = mask_truck.view(1, -1)
            mask_reg_truck = mask_truck.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1, print(batch_box3d.shape)
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            boxes3d_truck = torch.masked_select(batch_box3d_truck[0],
                                            mask_reg_truck[0]).view(-1, 7)
            scores_truck = torch.masked_select(prob_truck[0], mask_truck[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])
                # (N, 8, 3)
                projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
                
                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
            
            if len(boxes3d_truck) != 0:
                boxes3d_corner_truck = box_utils.boxes_to_corners_3d(boxes3d_truck, order=self.params['order'])
                projected_boxes3d_truck = box_utils.project_box3d(boxes3d_corner_truck, transformation_matrix)
                projected_boxes2d_truck = box_utils.corner_to_standup_box_torch(projected_boxes3d_truck)
                boxes2d_score_truck = torch.cat((projected_boxes2d_truck, scores_truck.unsqueeze(1)), dim=1)

                pred_box2d_truck_list.append(boxes2d_score_truck)
                pred_box3d_truck_list.append(projected_boxes3d_truck)
        
        car_is_not_detected = len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0
        truck_is_not_detected = len(pred_box2d_truck_list) == 0 or len(pred_box3d_truck_list) == 0
        
        nms_together = False

        if nms_together:
        
            if not car_is_not_detected and not truck_is_not_detected:
                # shape: (N, 5)
                pred_box2d_list = torch.vstack(pred_box2d_list)
                # scores
                scores = pred_box2d_list[:, -1]
                # predicted 3d bbx
                pred_box3d_tensor = torch.vstack(pred_box3d_list)

                #if len(pred_box2d_truck_list) == 0 or len(pred_box3d_truck_list) == 0:
                #    pred_box3d_tensor_truck, scores_truck = None, None
                #else
                pred_box2d_truck_list = torch.vstack(pred_box2d_truck_list)
                scores_truck = pred_box2d_truck_list[:, -1]
                pred_box3d_tensor_truck = torch.vstack(pred_box3d_truck_list)

                # concatenate the car and truck
                pred_box3d_cat_tensor = torch.cat((pred_box3d_tensor, pred_box3d_tensor_truck), dim=0)
                scores_cat = torch.cat((scores, scores_truck), dim=0)

                #import pdb; pdb.set_trace()
                # nms
                #print(pred_box3d_tensor.shape) 16896 x 8 x 3
                keep_index = box_utils.nms_rotated(
                    pred_box3d_cat_tensor, scores_cat, self.params['nms_thresh'])

                # take the car and truck separately
                keep_index_car = keep_index[keep_index < pred_box3d_tensor.shape[0]]
                keep_index_truck = keep_index[keep_index >= pred_box3d_tensor.shape[0]] - pred_box3d_tensor.shape[0]

                pred_box3d_tensor = pred_box3d_tensor[keep_index_car]
                pred_box3d_tensor_truck = pred_box3d_tensor_truck[keep_index_truck]
                
                scores = scores[keep_index_car]
                scores_truck = scores_truck[keep_index_truck]

                # filter out the prediction out of the range.
                mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
                mask_truck = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor_truck)
                pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
                pred_box3d_tensor_truck = pred_box3d_tensor_truck[mask_truck, :, :]
                scores = scores[mask]
                scores_truck = scores_truck[mask_truck]
                #print("after nms", scores.shape)
                assert scores.shape[0] == pred_box3d_tensor.shape[0]
            elif not car_is_not_detected and truck_is_not_detected:
                pred_box3d_tensor_truck, scores_truck = None, None
                # shape: (N, 5)
                pred_box2d_list = torch.vstack(pred_box2d_list)
                # scores
                scores = pred_box2d_list[:, -1]
                # predicted 3d bbx
                pred_box3d_tensor = torch.vstack(pred_box3d_list)

                #import pdb; pdb.set_trace()
                # nms
                #print(pred_box3d_tensor.shape) 16896 x 8 x 3
                keep_index = box_utils.nms_rotated(
                    pred_box3d_tensor, scores, self.params['nms_thresh'])

                pred_box3d_tensor = pred_box3d_tensor[keep_index]

                #print("before nms:", scores.shape)
                # select cooresponding score
                scores = scores[keep_index]

                # filter out the prediction out of the range.
                mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
                pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
                scores = scores[mask]
                #print("after nms", scores.shape)
                assert scores.shape[0] == pred_box3d_tensor.shape[0]
            elif car_is_not_detected and not truck_is_not_detected:
                pred_box3d_tensor, scores = None, None

                pred_box2d_truck_list = torch.vstack(pred_box2d_truck_list)
                scores_truck = pred_box2d_truck_list[:, -1]
                pred_box3d_tensor_truck = torch.vstack(pred_box3d_truck_list)

                #import pdb; pdb.set_trace()
                # nms
                #print(pred_box3d_tensor.shape) 16896 x 8 x 3
                keep_index = box_utils.nms_rotated(
                    pred_box3d_tensor_truck, scores_truck, self.params['nms_thresh'])

                pred_box3d_tensor_truck = pred_box3d_tensor_truck[keep_index]

                scores_truck = scores_truck[keep_index]

                # filter out the prediction out of the range.
                mask_truck = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor_truck)
                pred_box3d_tensor_truck = pred_box3d_tensor_truck[mask_truck, :, :]
                scores_truck = scores_truck[mask_truck]
                #print("after nms", scores.shape)
                assert scores_truck.shape[0] == pred_box3d_tensor_truck.shape[0]
            else:
                pred_box3d_tensor, scores, pred_box3d_tensor_truck, scores_truck = None, None, None, None

        else:
            if car_is_not_detected:
                pred_box3d_tensor, scores = None, None
            else:
                # shape: (N, 5)
                pred_box2d_list = torch.vstack(pred_box2d_list)
                # scores
                scores = pred_box2d_list[:, -1]
                # predicted 3d bbx
                pred_box3d_tensor = torch.vstack(pred_box3d_list)

                #import pdb; pdb.set_trace()
                # nms
                #print(pred_box3d_tensor.shape) 16896 x 8 x 3
                keep_index = box_utils.nms_rotated(
                    pred_box3d_tensor, scores, self.params['nms_thresh'])

                pred_box3d_tensor = pred_box3d_tensor[keep_index]

                #print("before nms:", scores.shape)
                # select cooresponding score
                scores = scores[keep_index]

                # filter out the prediction out of the range.
                mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
                pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
                scores = scores[mask]
                #print("after nms", scores.shape)
                assert scores.shape[0] == pred_box3d_tensor.shape[0]
            if truck_is_not_detected:
                pred_box3d_tensor_truck, scores_truck = None, None
            else:
                pred_box2d_truck_list = torch.vstack(pred_box2d_truck_list)
                scores_truck = pred_box2d_truck_list[:, -1]
                pred_box3d_tensor_truck = torch.vstack(pred_box3d_truck_list)

                #import pdb; pdb.set_trace()
                # nms
                #print(pred_box3d_tensor.shape) 16896 x 8 x 3
                keep_index = box_utils.nms_rotated(
                    pred_box3d_tensor_truck, scores_truck, self.params['nms_thresh'])

                pred_box3d_tensor_truck = pred_box3d_tensor_truck[keep_index]

                scores_truck = scores_truck[keep_index]

                # filter out the prediction out of the range.
                mask_truck = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor_truck)
                pred_box3d_tensor_truck = pred_box3d_tensor_truck[mask_truck, :, :]
                scores_truck = scores_truck[mask_truck]
                #print("after nms", scores.shape)
                assert scores_truck.shape[0] == pred_box3d_tensor_truck.shape[0]

        return pred_box3d_tensor, scores, pred_box3d_tensor_truck, scores_truck

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
