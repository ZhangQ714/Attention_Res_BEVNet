import functools
import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from timm.models.layers import trunc_normal_
import time
import sparseconvnet as scn
from fightingcv_attention.attention.CBAM import *
from fightingcv_attention.attention.SEAttention import SEAttention

from spconv_utils import replace_feature, spconv
import fchardnet
import convgru

class VoxelFeatureExtractorV3(nn.Module):
    def __init__(self):
        super(VoxelFeatureExtractorV3, self).__init__()

    def forward(self, features, num_voxels):
        # features: [concated_num_points, num_voxel_size, n_dim]
        # num_voxels: [concated_num_points]
        points_mean = features.sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class VoxelFeatureExtractorV3MultiStep(nn.Module):
    def __init__(self):
        super(VoxelFeatureExtractorV3MultiStep, self).__init__()

    def forward(self, features, num_voxels):
        # features: T x [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: T x [concated_num_points]
        # returns list of T x [num_voxels]
        t = len(features)
        output = []
        for i in range(t):
            features_single = features[i]
            num_single = num_voxels[i]
            points_mean = features_single.sum(
                dim=1, keepdim=False) / num_single.type_as(features_single).view(-1, 1)
            output.append(points_mean.contiguous())
        return output


class SpMiddleNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXY, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),
            BatchNorm1d(64),
            nn.ReLU(),
        )


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        # out = ret.cpu()
        # cbam = CBAMBlock(channel=192, reduction=12, kernel_size=3)
        # out = cbam(out)
        # # print(11112)
        # out = out.cuda()

        return ret


class SpMiddleNoDownsampleXYMultiStep(SpMiddleNoDownsampleXY):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleNoDownsampleXYMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)
                ret = self.middle_conv(ret)
                ret = ret.dense()                # shape [1, 64, 3, 512, 512]       [batch_size, features_dim, z, y, x]
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)    # 3D to 2D
                output.append(ret.detach())
            return output

class SpMiddleNoDownsampleXYSingleFrame(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 num_input_frames,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXYSingleFrame, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.sparse_shape[2] *= num_input_frames

        self.voxel_output_shape = output_shape

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),
            BatchNorm1d(64),
            nn.ReLU(),
        )


class SpMiddleNoDownsampleXYCatFrames(SpMiddleNoDownsampleXYSingleFrame):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleNoDownsampleXYCatFrames, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        voxel_features_cat = voxel_features[0]
        coors_cat = coors[0]
        coors_add = torch.zeros(4).cuda()
        with torch.no_grad():
            t = len(voxel_features)
            for i in range(1, t):
                voxel_features_cat = torch.cat((voxel_features_cat, voxel_features[i]), dim=0)

                coors_add[3] = 512*i
                coors[i][:] += coors_add.int()
                coors_cat = torch.cat((coors_cat, coors[i].int()), dim=0)

            ret = spconv.SparseConvTensor(voxel_features_cat, coors_cat, self.sparse_shape, batch_size)
            ret = self.middle_conv(ret)
            ret = ret.dense()                # shape [1, 64, 3, 512, 512*t]       [batch_size, features_dim, z, y, x]
            N, C, D, H, W = ret.shape
            ret = ret.view(N, C * D, H, W)    # [1, 192, 512, 512*t]

            ret = torch.split(ret, 512, dim=3)
            ret = list(ret)

            return ret


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out

class VoxelResBackBone8x(nn.Module):

    def __init__(self,
                 grid_size,
                 input_channels):
        super(VoxelResBackBone8x, self).__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        sparse_shape = np.array(grid_size[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(

            block(16, 32, 3, norm_fn=norm_fn, stride=(2, 1, 1), padding=(1, 1, 1), indice_key='spconv2', conv_type='spconv'),  # 15
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(

            block(32, 64, 3, norm_fn=norm_fn, stride=(2, 1, 1), padding=(0, 1, 1), indice_key='spconv3', conv_type='spconv'),  # 7
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(

            block(64, 128, 3, norm_fn=norm_fn, stride=(2, 1, 1), padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),  # 3
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        # last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(

            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0),
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, voxel_features, voxel_coords, batch_size):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        out = self.conv_out(x_conv4)

        out = out.dense()
        N, C, D, H, W = out.shape
        out = out.view(N, C * D, H, W)

        # attention  CVAM
        # out = out.cpu()
        # cbam = CBAMBlock(channel=128, reduction=4, kernel_size=7)
        # out = cbam(out)
        # # print(111)
        # out = out.cuda()

        return out

class VoxelResBackBone8xMultiStep(VoxelResBackBone8x):
    def __init__(self, *args, **kwargs):
        super(VoxelResBackBone8xMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, voxel_coords, batch_size):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features[i],
                    indices=voxel_coords[i].int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                x_conv4 = self.conv4(x_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                out = self.conv_out(x_conv4)
                out = out.dense()  # shape [1, 64, 3, 512, 512]       [batch_size, features_dim, z, y, x]
                N, C, D, H, W = out.shape
                out = out.view(N, C * D, H, W)

                # attention  CVAM
                out = out.cpu()
                cbam = CBAMBlock(channel=128, reduction=4, kernel_size=7)
                out = cbam(out)
                # print(111)
                out = out.cuda()

                output.append(out.detach())
            return output

class ResSpNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(ResSpNoDownsampleXY, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape

        self.subm1 = SubMConv3d(num_input_features, 32, 3, indice_key="subm0")
        self.bn1 = BatchNorm1d(32)
        self.relu = nn.ReLU()

        self.subm2 = SubMConv3d(32, 32, 3, indice_key="subm0")
        self.bn2 = BatchNorm1d(32)

        self.spcv1 = SpConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1])
        self.bn3 = BatchNorm1d(64)

        self.subm3 = SubMConv3d(64, 64, 3, indice_key="subm1")
        self.bn4 = BatchNorm1d(64)

        self.subm4 = SubMConv3d(64, 64, 3, indice_key="subm1")
        self.bn5 = BatchNorm1d(64)

        self.subm5 = SubMConv3d(64, 64, 3, indice_key="subm1")
        self.bn6 = BatchNorm1d(64)

        self.spcv2 = SpConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1])
        self.bn7 = BatchNorm1d(64)

        self.subm6 = SubMConv3d(64, 64, 3, indice_key="subm2")
        self.bn8 = BatchNorm1d(64)

        self.subm7 = SubMConv3d(64, 64, 3, indice_key="subm2")
        self.bn9 = BatchNorm1d(64)

        self.subm8 = SubMConv3d(64, 64, 3, indice_key="subm2")
        self.bn10 = BatchNorm1d(64)

        self.spcv3 = SpConv3d(64, 64, (3, 1, 1), (2, 1, 1))
        self.bn11 = BatchNorm1d(64)

        # attention
        self.se = SEAttention(channel=192, reduction=8)
        self.cbam = CBAMBlock(channel=192, reduction=8, kernel_size=7)


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # identity0 = ret

        x1 = self.subm1(ret)
        x1 = replace_feature(x1, self.bn1(x1.features))
        # x1 = replace_feature(x1, x1.features + identity0.features)
        x1 = replace_feature(x1, self.relu(x1.features))
        identity1 = x1

        x2 = self.subm2(x1)
        x2 = replace_feature(x2, self.bn2(x2.features))
        x2 = replace_feature(x2, x2.features + identity1.features)
        x2 = replace_feature(x2, self.relu(x2.features))

        x3 = self.spcv1(x2)
        x3 = replace_feature(x3, self.bn3(x3.features))
        x3 = replace_feature(x3, self.relu(x3.features))
        identity3 = x3

        x4 = self.subm3(x3)
        x4 = replace_feature(x4, self.bn4(x4.features))
        x4 = replace_feature(x4, x4.features + identity3.features)
        x4 = replace_feature(x4, self.relu(x4.features))
        identity4 = x4

        x5 = self.subm4(x4)
        x5 = replace_feature(x5, self.bn5(x5.features))
        x5 = replace_feature(x5, x5.features + identity4.features)
        x5 = replace_feature(x5, self.relu(x5.features))
        identity5 = x5

        x6 = self.subm5(x5)
        x6 = replace_feature(x6, self.bn6(x6.features))
        x6 = replace_feature(x6, x6.features + identity5.features)
        x6 = replace_feature(x6, self.relu(x6.features))

        x7 = self.spcv2(x6)
        x7 = replace_feature(x7, self.bn7(x7.features))
        x7 = replace_feature(x7, self.relu(x7.features))
        identity7 = x7

        x8 = self.subm6(x7)
        x8 = replace_feature(x8, self.bn8(x8.features))
        x8 = replace_feature(x8, x8.features + identity7.features)
        x8 = replace_feature(x8, self.relu(x8.features))
        identity8 = x8

        x9 = self.subm7(x8)
        x9 = replace_feature(x9, self.bn9(x9.features))
        x9 = replace_feature(x9, x9.features + identity8.features)
        x9 = replace_feature(x9, self.relu(x9.features))
        identity9 = x9

        x10 = self.subm8(x9)
        x10 = replace_feature(x10, self.bn10(x10.features))
        x10 = replace_feature(x10, x10.features + identity9.features)
        x10 = replace_feature(x10, self.relu(x10.features))

        x11 = self.spcv3(x10)
        x11 = replace_feature(x11, self.bn11(x11.features))
        ret = replace_feature(x11, self.relu(x11.features))


        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        # attention SE
        # ret = self.se(ret)

        # attention  CBAM
        ret = self.cbam(ret)

        return ret

class ResSpNoDownsampleXYMultiStep(ResSpNoDownsampleXY):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(ResSpNoDownsampleXYMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)

                x1 = self.subm1(ret)
                x1 = replace_feature(x1, self.bn1(x1.features))
                # x1 = replace_feature(x1, x1.features + identity0.features)
                x1 = replace_feature(x1, self.relu(x1.features))
                identity1 = x1

                x2 = self.subm2(x1)
                x2 = replace_feature(x2, self.bn2(x2.features))
                x2 = replace_feature(x2, x2.features + identity1.features)
                x2 = replace_feature(x2, self.relu(x2.features))

                x3 = self.spcv1(x2)
                x3 = replace_feature(x3, self.bn3(x3.features))
                x3 = replace_feature(x3, self.relu(x3.features))
                identity3 = x3

                x4 = self.subm3(x3)
                x4 = replace_feature(x4, self.bn4(x4.features))
                x4 = replace_feature(x4, x4.features + identity3.features)
                x4 = replace_feature(x4, self.relu(x4.features))
                identity4 = x4

                x5 = self.subm4(x4)
                x5 = replace_feature(x5, self.bn5(x5.features))
                x5 = replace_feature(x5, x5.features + identity4.features)
                x5 = replace_feature(x5, self.relu(x5.features))
                identity5 = x5

                x6 = self.subm5(x5)
                x6 = replace_feature(x6, self.bn6(x6.features))
                x6 = replace_feature(x6, x6.features + identity5.features)
                x6 = replace_feature(x6, self.relu(x6.features))

                x7 = self.spcv2(x6)
                x7 = replace_feature(x7, self.bn7(x7.features))
                x7 = replace_feature(x7, self.relu(x7.features))
                identity7 = x7

                x8 = self.subm6(x7)
                x8 = replace_feature(x8, self.bn8(x8.features))
                x8 = replace_feature(x8, x8.features + identity7.features)
                x8 = replace_feature(x8, self.relu(x8.features))
                identity8 = x8

                x9 = self.subm7(x8)
                x9 = replace_feature(x9, self.bn9(x9.features))
                x9 = replace_feature(x9, x9.features + identity8.features)
                x9 = replace_feature(x9, self.relu(x9.features))
                identity9 = x9

                x10 = self.subm8(x9)
                x10 = replace_feature(x10, self.bn10(x10.features))
                x10 = replace_feature(x10, x10.features + identity9.features)
                x10 = replace_feature(x10, self.relu(x10.features))

                x11 = self.spcv3(x10)
                x11 = replace_feature(x11, self.bn11(x11.features))
                ret = replace_feature(x11, self.relu(x11.features))

                ret = ret.dense()                # shape [1, 64, 3, 512, 512]       [batch_size, features_dim, z, y, x]
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)    # 3D to 2D

                # attention  CBAM
                ret = self.cbam(ret)

                output.append(ret.detach())
            return output



class SpatialGroupConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False):
        super(SpatialGroupConv, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int(kernel_size // 2),
            bias=bias,
            indice_key=indice_key,
        )

        self.conv3x3_1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=int(kernel_size // 2) - 1,
            bias=bias,
            dilation=int(kernel_size // 2) - 1,
            indice_key=indice_key + 'conv_3x3_1',
        )
        self._indice_list = []

        if kernel_size == 7:
            _list = [0, 3, 4, 7]
        elif kernel_size == 5:
            _list = [0, 2, 3, 5]
        else:
            raise ValueError('Unknown kernel size %d' % kernel_size)
        for i in range(len(_list) - 1):
            for j in range(len(_list) - 1):
                for k in range(len(_list) - 1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i + 1], _list[j]:_list[j + 1], _list[k]:_list[k + 1]] = 1
                    b = torch.range(0, kernel_size ** 3 - 1, 1)[a.reshape(-1).bool()]
                    self._indice_list.append(b.long())

    def _convert_weight(self, weight):
        weight_reshape = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels,
                                                                          -1).clone()
        weight_return = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels,
                                                                         -1).clone()
        for _indice in self._indice_list:
            _mean_weight = torch.mean(weight_reshape[:, :, _indice], dim=-1, keepdim=True)
            weight_return[:, :, _indice] = _mean_weight
        return weight_return.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size,
                                     self.kernel_size).permute(2, 3, 4, 0, 1)

    def forward(self, x_conv):
        if self.training:
            self.block.weight.data = self._convert_weight(self.block.weight.data)
        x_conv_block = self.block(x_conv)
        x_conv_conv3x3_1 = self.conv3x3_1(x_conv)
        x_conv_block = x_conv_block.replace_feature(x_conv_block.features + x_conv_conv3x3_1.features)
        return x_conv_block


class SpatialGroupConvV2(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, indice_key=None,
                 position_embedding=True, use_relu=False):
        super(SpatialGroupConvV2, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size > 3, "SpatialGroupConv3d requires large kernels."
        _list = [0, int(kernel_size // 2), int(kernel_size // 2) + 1, 7]
        self.group_map = torch.zeros((3 ** 3, int(kernel_size // 2) ** 3)) - 1
        _num = 0
        for i in range(len(_list) - 1):
            for j in range(len(_list) - 1):
                for k in range(len(_list) - 1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i + 1], _list[j]:_list[j + 1], _list[k]:_list[k + 1]] = 1
                    _pos = a.sum()
                    self.group_map[_num][:_pos] = torch.range(0, kernel_size ** 3 - 1, 1)[a.reshape(-1).bool()]
                    _num += 1
        self.group_map = self.group_map.int()

        self.block = spconv.SpatialGroupConv3d(
            in_channels,
            out_channels,
            kernel_size, 3,
            stride=stride,
            padding=int(kernel_size // 2),
            bias=bias,
            indice_key=indice_key,
            algo=ConvAlgo.Native,
            position_embedding=position_embedding,
            use_relu=use_relu,
        )
        if position_embedding:
            trunc_normal_(self.block.position_embedding, std=0.02)

    def forward(self, x_conv):
        x_conv = self.block(x_conv, group_map=self.group_map.to(x_conv.features.device))
        return x_conv


class GroupSparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, norm_fn=None, downsample=None, indice_key=None,
                 conv_type='common'):
        super(GroupSparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        if conv_type == "spatialgroupconv":
            conv_func = SpatialGroupConv
        elif conv_type == 'spatialgroupconvv2':
            conv_func = SpatialGroupConvV2
        elif conv_type == 'common':
            conv_func = spconv.SubMConv3d
        else:
            raise ValueError('Unknown conv type %s.' % conv_type)

        self.conv1 = conv_func(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size // 2), bias=bias,
            indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv_func(
            planes, planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size // 2), bias=bias,
            indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelResBackBone8xLargeKernel3D(nn.Module):
    def __init__(self, input_channels, grid_size, **kwargs):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        kernel_sizes = [7, 5, 5, 3]
        spconv_kernel_sizes = [5, 5]
        conv_types = ['spatialgroupconv', 'common', 'common', 'common']

        # self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        sparse_shape = np.array(grid_size[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            GroupSparseBasicBlock(16, 16, kernel_sizes[0], norm_fn=norm_fn, indice_key='res1', conv_type=conv_types[0]),
            GroupSparseBasicBlock(16, 16, kernel_sizes[0], norm_fn=norm_fn, indice_key='res1', conv_type=conv_types[0]),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=(2, 1, 1), padding=int(spconv_kernel_sizes[0] // 2),
                  indice_key='spconv2', conv_type='spconv'),
            GroupSparseBasicBlock(32, 32, kernel_sizes[1], norm_fn=norm_fn, indice_key='res2', conv_type=conv_types[1]),
            GroupSparseBasicBlock(32, 32, kernel_sizes[1], norm_fn=norm_fn, indice_key='res2', conv_type=conv_types[1]),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=(2, 1, 1), padding=int(spconv_kernel_sizes[1] // 2),
                  indice_key='spconv3', conv_type='spconv'),
            GroupSparseBasicBlock(64, 64, kernel_sizes[2], norm_fn=norm_fn, indice_key='res3', conv_type=conv_types[2]),
            GroupSparseBasicBlock(64, 64, kernel_sizes[2], norm_fn=norm_fn, indice_key='res3', conv_type=conv_types[2]),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 128, 3, norm_fn=norm_fn, stride=(2, 1, 1), padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     GroupSparseBasicBlock(128, 128, kernel_sizes[3], norm_fn=norm_fn, indice_key='res4', conv_type=conv_types[3]),
        #     GroupSparseBasicBlock(128, 128, kernel_sizes[3], norm_fn=norm_fn, indice_key='res4', conv_type=conv_types[3]),
        # )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 64
        # self.backbone_channels = {
        #     'x_conv1': 16,
        #     'x_conv2': 32,
        #     'x_conv3': 64,
        #     'x_conv4': 128
        # }
        # self.forward_ret_dict = {}

    def forward(self, voxel_features, voxel_coords, batch_size):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        # voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        # batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv3)

        out = out.dense()
        N, C, D, H, W = out.shape
        out = out.view(N, C * D, H, W)

        return out

class VoxelResBackBone8xLargeKernel3DMultiStep(VoxelResBackBone8xLargeKernel3D):
    def __init__(self, *args, **kwargs):
        super(VoxelResBackBone8xLargeKernel3DMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, voxel_coords, batch_size):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features[i],
                    indices=voxel_coords[i].int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                # x_conv4 = self.conv4(x_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                out = self.conv_out(x_conv3)
                out = out.dense()  # shape [1, 64, 3, 512, 512]       [batch_size, features_dim, z, y, x]
                N, C, D, H, W = out.shape
                out = out.view(N, C * D, H, W)
                output.append(out.detach())
            return output


class SpMiddleNoDownsampleXYNoExpand(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXYNoExpand, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 64, 3, indice_key="subm0"),
            BatchNorm1d(64),
            nn.ReLU(),
            # SubMConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1]),
            spconv.SparseMaxPool3d(3, (2, 1, 1), padding=[1, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            # SubMConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1]),
            spconv.SparseMaxPool3d(3, (2, 1, 1), padding=[0, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            # SubMConv3d(64, 64, (3, 1, 1), (2, 1, 1)),
            spconv.SparseMaxPool3d((3, 1, 1), (2, 1, 1)),
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleNoDownsampleXYNoExpandMultiStep(SpMiddleNoDownsampleXYNoExpand):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleNoDownsampleXYNoExpandMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)
                ret = self.middle_conv(ret)
                ret = ret.dense()
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)
                output.append(ret.detach())
            return output


class MiddleNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self, output_shape, num_input_features):
        super(MiddleNoDownsampleXY, self).__init__()
        Conv3d = functools.partial(nn.Conv3d, bias=True)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = nn.Sequential(
            Conv3d(num_input_features, 32, 3, padding=1),
            nn.ReLU(),
            Conv3d(32, 64, 3, (2, 1, 1), padding=1),  # Downsample z
            nn.ReLU(),
            Conv3d(64, 64, 3, stride=(2, 1, 1), padding=[0, 1, 1]),  # Downsample z
            nn.ReLU(),
            Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(2, 1, 1)),  # Downsample z
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        inputs = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size).dense()
        ret = self.middle_conv(inputs)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class InpaintingFCHardnetRecurrentBase(object):
    def __init__(self,
                 aggregation_type='pre',
                 gru_input_size=(256, 256),
                 gru_input_dim=448,
                 gru_hidden_dims=[448],
                 gru_cell_type='standard',
                 noisy_pose=False, **kwargs):
        super(InpaintingFCHardnetRecurrentBase, self).__init__(**kwargs)

        assert aggregation_type in ['pre', 'post', 'none'], aggregation_type
        self.aggregation_type = aggregation_type

        if aggregation_type != 'none':
            ### Amirreza: GRU parameters are hardcoded for now
            self.gru = convgru.ConvGRU(input_size=gru_input_size,
                                       input_dim=gru_input_dim,
                                       hidden_dim=gru_hidden_dims,
                                       kernel_size=(3, 3),
                                       num_layers=len(gru_hidden_dims),
                                       dtype=torch.cuda.FloatTensor,
                                       batch_first=True,
                                       bias=True,
                                       return_all_layers=True,
                                       noisy_pose=noisy_pose,
                                       cell_type=gru_cell_type)

            def get_poses(input_pose):
                # convert to matrix
                mat = torch.zeros(input_pose.shape[0], # batch_size
                                  input_pose.shape[1], # t
                                  3, 3, dtype=input_pose.dtype,
                                  device=input_pose.device)

                mat[:, :, 0] = input_pose[:, :, :3]
                mat[:, :, 1] = input_pose[:, :, 3:6]
                mat[:, :, 2, 2] = 1.0

                # We are using two GRU cells with the same poses
                return mat[:, :, None]

            self.get_poses = get_poses

    def forward(self, x, seq_start=None, input_pose=None):
        n, c, h, w = x[0].shape
        t = len(x)

        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        elif isinstance(x, torch.Tensor):
            x = x.view((-1,) + x.size()[2:])  # Fuse dim 0 and 1

        if self.aggregation_type != 'none':
            if seq_start is None:
                self.hidden_state = None
            else:
                # sanity check: only the first index can be True
                assert(torch.any(seq_start[1:]) == False)

                if seq_start[0]:  # start of a new sequence
                    self.hidden_state = None
        if self.aggregation_type == 'pre':
            layer_output_list, last_state_list = self.gru(x[None],
                                                          self.get_poses(input_pose[None]),
                                                          hidden_state=self.hidden_state)
            x = layer_output_list[-1].squeeze(0)           # squeeze(0):若第一个维度为1则去除这个维度    x.shape [6, 192, 512,512]

        out = self.fchardnet(x)            # [6, 5, 512, 512]

        if self.aggregation_type == 'post':
            layer_output_list, last_state_list = self.gru(out[None],
                                                          self.get_poses(input_pose[None]),
                                                          hidden_state=self.hidden_state)
            out = layer_output_list[-1].squeeze(0)

        if self.aggregation_type != 'none':
            self.hidden_state = []
            for state in last_state_list:
                dstate = state[0].detach()
                dstate.requires_grad = True
                self.hidden_state.append(dstate)

        num_class = out.shape[1]
        out = out.reshape((t, n, num_class, h, w))
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class InpaintingFCHardNetSkip1024(nn.Module):
    def __init__(self,
                 num_class=2,
                 num_input_features=128):
        super(InpaintingFCHardNetSkip1024, self).__init__()
        self.fchardnet = fchardnet.HardNet1024Skip(num_input_features, num_class)

    def forward(self, x, *args, **kwargs):
        out = self.fchardnet(x)
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class InpaintingFCHardNetSkipGRU512(InpaintingFCHardnetRecurrentBase, InpaintingFCHardNetSkip1024):
    pass

class SpMiddleAlinPreFramesNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleAlinPreFramesNoDownsampleXY, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape

        self.subm1 = SubMConv3d(num_input_features, 32, 3, indice_key="subm0")
        self.bn1 = BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.subm2 = SubMConv3d(32, 32, 3, indice_key="subm0")
        self.bn2 = BatchNorm1d(32)

        self.spcv1 = SpConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1])
        self.bn3 = BatchNorm1d(64)

        self.subm3 = SubMConv3d(64, 64, 3, indice_key="subm1")
        self.bn4 = BatchNorm1d(64)

        self.subm4 = SubMConv3d(64, 64, 3, indice_key="subm1")
        self.bn5 = BatchNorm1d(64)

        self.subm5 = SubMConv3d(64, 64, 3, indice_key="subm1")
        self.bn6 = BatchNorm1d(64)

        self.spcv2 = SpConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1])
        self.bn7 = BatchNorm1d(64)

        self.subm6 = SubMConv3d(64, 64, 3, indice_key="subm2")
        self.bn8 = BatchNorm1d(64)

        self.subm7 = SubMConv3d(64, 64, 3, indice_key="subm2")
        self.bn9 = BatchNorm1d(64)

        self.subm8 = SubMConv3d(64, 64, 3, indice_key="subm2")
        self.bn10 = BatchNorm1d(64)

        self.spcv3 = SpConv3d(64, 64, (3, 1, 1), (2, 1, 1))
        self.bn11 = BatchNorm1d(64)


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)

        x1 = self.subm1(ret)
        x1 = replace_feature(x1, self.bn1(x1.features))
        x1 = replace_feature(x1, self.relu(x1.features))
        x2 = self.subm2(x1)
        x2 = replace_feature(x2, self.bn2(x2.features))
        x2 = replace_feature(x2, self.relu(x2.features))
        x3 = self.spcv1(x2)
        x3 = replace_feature(x3, self.bn3(x3.features))
        x3 = replace_feature(x3, self.relu(x3.features))
        x4 = self.subm3(x3)
        x4 = replace_feature(x4, self.bn4(x4.features))
        x4 = replace_feature(x4, self.relu(x4.features))
        x5 = self.subm4(x4)
        x5 = replace_feature(x5, self.bn5(x5.features))
        x5 = replace_feature(x5, self.relu(x5.features))
        x6 = self.subm5(x5)
        x6 = replace_feature(x6, self.bn6(x6.features))
        x6 = replace_feature(x6, self.relu(x6.features))
        x7 = self.spcv2(x6)
        x7 = replace_feature(x7, self.bn7(x7.features))
        x7 = replace_feature(x7, self.relu(x7.features))
        x8 = self.subm6(x7)
        x8 = replace_feature(x8, self.bn8(x8.features))
        x8 = replace_feature(x8, self.relu(x8.features))
        x9 = self.subm7(x8)
        x9 = replace_feature(x9, self.bn9(x9.features))
        x9 = replace_feature(x9, self.relu(x9.features))
        x10 = self.subm8(x9)
        x10 = replace_feature(x10, self.bn10(x10.features))
        x10 = replace_feature(x10, self.relu(x10.features))
        x11 = self.spcv3(x10)
        x11 = replace_feature(x11, self.bn11(x11.features))
        ret = replace_feature(x11, self.relu(x11.features))

        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleAlinPreFramesNoDownsampleXYMultiStep(SpMiddleAlinPreFramesNoDownsampleXY):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleAlinPreFramesNoDownsampleXYMultiStep, self).__init__(*args, **kwargs)
        self.x_pre1 = spconv.SparseConvTensor(torch.zeros(20000, 64), torch.zeros(20000, 4).int().cuda(), self.sparse_shape, 1)
        self.x_pre2 = spconv.SparseConvTensor(torch.zeros(20000, 64), torch.zeros(20000, 4).int().cuda(), self.sparse_shape, 1)
        self.x_pre3 = spconv.SparseConvTensor(torch.zeros(20000, 64), torch.zeros(20000, 4).int().cuda(), self.sparse_shape, 1)
        self.xpre = [self.x_pre1, self.x_pre2, self.x_pre3]
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)
        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.adapter_conv_1 = spconv.SparseSequential(
            SubMConv3d(64, 64, 1, indice_key="subm5"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 1, indice_key="subm5"),
            # nn.AvgPool1d((16, 1, 512), 1),
            # nn.Sigmoid(),
        )


    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)


                x1 = self.subm1(ret)
                x1 = replace_feature(x1, self.bn1(x1.features))
                x1 = replace_feature(x1, self.relu(x1.features))
                x2 = self.subm2(x1)
                x2 = replace_feature(x2, self.bn2(x2.features))
                x2 = replace_feature(x2, self.relu(x2.features))

                x3 = self.spcv1(x2)                                    #
                x3 = self.add_pre(x3, self.xpre[0])
                x3 = replace_feature(x3, self.bn3(x3.features))
                x3 = replace_feature(x3, self.relu(x3.features))
                self.xpre[0] = x3

                x4 = self.subm3(x3)
                x4 = replace_feature(x4, self.bn4(x4.features))
                x4 = replace_feature(x4, self.relu(x4.features))
                x5 = self.subm4(x4)
                x5 = replace_feature(x5, self.bn5(x5.features))
                x5 = replace_feature(x5, self.relu(x5.features))
                x6 = self.subm5(x5)
                x6 = replace_feature(x6, self.bn6(x6.features))
                x6 = replace_feature(x6, self.relu(x6.features))

                x7 = self.spcv2(x6)                                   #
                x7 = self.add_pre(x7, self.xpre[1])
                x7 = replace_feature(x7, self.bn7(x7.features))
                x7 = replace_feature(x7, self.relu(x7.features))
                self.xpre[1] = x7

                x8 = self.subm6(x7)
                x8 = replace_feature(x8, self.bn8(x8.features))
                x8 = replace_feature(x8, self.relu(x8.features))
                x9 = self.subm7(x8)
                x9 = replace_feature(x9, self.bn9(x9.features))
                x9 = replace_feature(x9, self.relu(x9.features))
                x10 = self.subm8(x9)
                x10 = replace_feature(x10, self.bn10(x10.features))
                x10 = replace_feature(x10, self.relu(x10.features))

                x11 = self.spcv3(x10)                                  #
                x11 = self.add_pre(x11, self.xpre[2])
                x11 = replace_feature(x11, self.bn11(x11.features))
                ret = replace_feature(x11, self.relu(x11.features))
                self.xpre[2] = ret

                ret = ret.dense()                # shape [1, 64, 3, 512, 512]       [batch_size, features_dim, z, y, x]
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)
                output.append(ret.detach())

                self.xpre = self.pre_adapter(self.xpre, self.adapter_conv_1)

            return output

    def add_pre(self, x_cur, x_pre_i):
        fts_avg = torch.mean(x_pre_i.features, 0).cuda()
        fts_cur = x_cur.features
        for i in range(len(x_cur.features[0])):
           fts_cur[i] += fts_avg
        x_cur = replace_feature(x_cur, fts_cur)
        return x_cur

    def pre_adapter(self, xpre, adapter):
        for i in range(len(xpre)):
            xpre[i] = adapter(xpre[i])
            xpre[i] = replace_feature(xpre[i], xpre[i].features / 10)
        return xpre