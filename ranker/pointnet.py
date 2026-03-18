import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        ).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(
            np.eye(self.k, dtype=np.float32).flatten()
        ).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """x : batch x xyz x num_points"""
        num_points = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class RankerEncoder(nn.Module):
    def __init__(self, in_channels, global_feat=True, feature_transform=False):
        super(RankerEncoder, self).__init__()
        self.stn = STNkd(k=in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """x : batch x channels x num_points"""
        num_points = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        return x, trans, trans_feat


class PointNetRanker(nn.Module):
    """
    PointNet-based box quality ranker.

    Given a candidate bounding box and the point cloud within it,
    predicts an IoU score (0-1) and optionally a box offset correction.
    """
    def __init__(
        self,
        use_lwh=True,
        use_angle=True,
        use_depth=True,
        use_ground=True,
        use_dropout=False,
        use_bn=False,
        feature_transform=False,
        use_offset=False,
    ):
        super(PointNetRanker, self).__init__()
        self.use_ground = use_ground
        self.use_offset = use_offset
        self.feature_transform = feature_transform
        self.encoder = RankerEncoder(
            in_channels=3 + 3 + (3 * int(use_lwh)) + int(use_angle) + int(use_depth) + int(use_ground),
            global_feat=True,
            feature_transform=feature_transform,
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        if use_offset:
            self.fc4 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(p=0.3) if use_dropout else nn.Identity()
        self.bn1 = nn.BatchNorm1d(512) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(256) if use_bn else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, bbox_center, ptc, ground=None, distance=None):
        """
        Args:
            bbox_center: (B, 7) box parameters [x, y, z, l, w, h, yaw]
            ptc: (B, N, 3) point cloud within the box
            ground: (B, N, 1) ground point mask (optional)
            distance: (B,) distance from ego to the box center
        Returns:
            score: (B, 1) predicted IoU quality score in [0, 1]
            offset: (B, 7) predicted box offset (only if use_offset=True)
        """
        batch_size, num_points, _ = ptc.shape

        bbox_center = bbox_center.unsqueeze(1).expand(batch_size, num_points, -1)
        distance = distance.unsqueeze(1).unsqueeze(1).expand(batch_size, num_points, -1)

        if ground is not None:
            ground = ground.float()
            input_feat = torch.cat([bbox_center, ptc, ground, distance], dim=-1)
        else:
            input_feat = torch.cat([bbox_center, ptc, distance], dim=-1)

        input_feat = input_feat.transpose(2, 1)

        x, trans, trans_feat = self.encoder(input_feat)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x_score = self.fc3(x)

        if self.use_offset:
            x_offset = self.fc4(x)
            return torch.sigmoid(x_score), x_offset, trans, trans_feat
        else:
            return torch.sigmoid(x_score), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
