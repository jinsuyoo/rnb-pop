import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.utils.box_utils import corner_to_center
from opencood.utils.iou3d_nms import iou3d_nms_utils
import copy


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()

        self.args = args

        #self.identity = torch.eye(4).cuda() #torch.from_numpy(np.identity(4)).float()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(
            args['cls_head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(
            args['cls_head_dim'],7 * args['anchor_number'], kernel_size=1)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        #print('input shapes: ', voxel_coords.shape, voxel_features.shape, voxel_num_points.shape)
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}
        #print('input shapess: ', batch_dict['voxel_features'].shape, batch_dict['voxel_coords'].shape, batch_dict['voxel_num_points'].shape)
        batch_dict = self.pillar_vfe(batch_dict)
        #print(batch_dict.keys())
        batch_dict = self.scatter(batch_dict)
        #print(batch_dict.keys())
        batch_dict = self.backbone(batch_dict)
        #print(batch_dict.keys())
        #print(batch_dict['spatial_features_2d'].shape)
        spatial_features_2d = batch_dict['spatial_features_2d']

        #print(spatial_features_2d.shape)

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        #print(psm.shape, rm.shape)
        output_dict = {'psm': psm, 'rm': rm}
        
        #batch_dict.clear()

        return output_dict


if __name__ == '__main__':

    from opencood.utils import yaml_utils

    hypes = yaml_utils.load_yaml('/users/PAS2099/jinsuyoo/cad-v2v/sample_pointpillar.yaml', '')

    model = PointPillar(hypes['model']['args'])

    print('# parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


    print('done!')