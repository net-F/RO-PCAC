import torch
import MinkowskiEngine as ME
from torch import nn
from data_utils import isin, istopk
from .blocks import ResidualBlock, Residual_SparseTransformer_Block


# Encoder_attri
class Encoder_attri_y(torch.nn.Module):
    def __init__(self, channels=[3, 64, 128, 128, 128, 128, 128]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        # self.bnc0 = ME.MinkowskiBatchNorm(channels[1])

        self.color_block0 = nn.Sequential(
            ResidualBlock(channels[1], channels[1], use_res=True, use_bias=True),
            ResidualBlock(channels[1], channels[1], use_res=True, use_bias=True),
            # ResidualBlock(channels[1], channels[1], use_res=True, use_bias=True),
        )

        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        # self.bnd0 = ME.MinkowskiBatchNorm(channels[2])

        self.trans0 = Residual_SparseTransformer_Block(channels[2], channels[2])

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.bnc1 = ME.MinkowskiBatchNorm(channels[3])

        self.color_block1 = nn.Sequential(
            ResidualBlock(channels[3], channels[3], use_res=True, use_bias=True),
            ResidualBlock(channels[3], channels[3], use_res=True, use_bias=True),
            # ResidualBlock(channels[3], channels[3], use_res=True, use_bias=True),

        )

        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.bnd1 = ME.MinkowskiBatchNorm(channels[4])
        self.trans1 = Residual_SparseTransformer_Block(channels[4], channels[4])

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.bnc2 = ME.MinkowskiBatchNorm(channels[5])
        self.color_block2 = nn.Sequential(
            ResidualBlock(channels[5], channels[5], use_res=True, use_bias=True),
            ResidualBlock(channels[5], channels[5], use_res=True, use_bias=True),
            # ResidualBlock(channels[5], channels[5], use_res=True, use_bias=True),
        )

        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[5],
            out_channels=channels[6],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.bnd2 = ME.MinkowskiBatchNorm(channels[6])
        self.trans2 = Residual_SparseTransformer_Block(channels[6], channels[6])

        self.fconv0 = ME.MinkowskiConvolution(
            in_channels=channels[6],
            out_channels=channels[6],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.fconv1 = ME.MinkowskiConvolution(
        #     in_channels=channels[6],
        #     out_channels=channels[6],
        #     kernel_size=3,
        #     stride=2,
        #     bias=True,
        #     dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.color_block0(self.relu(self.conv0(x)))
        out0 = self.relu(self.down0(out0))
        out0 = self.trans0(out0)

        out1 = self.color_block1(self.relu(self.conv1(out0)))
        out1 = self.relu(self.down1(out1))
        out1 = self.trans1(out1)

        out2 = self.color_block2(self.relu(self.conv2(out1)))
        out2 = self.relu(self.down2(out2))
        out2 = self.trans2(out2)

        out2 = self.fconv0(out2)
        # out2 = self.fconv1(out2)

        return [out2, out1, out0]


# Decoder_attri
class Decoder_attri_y(torch.nn.Module):
    """the decoding network with upsampling.
    """

    def __init__(self, channels=[128, 128, 128, 128, 128, 64, 3]):
        super().__init__()

        # self.fconv0 = ME.MinkowskiConvolutionTranspose(
        #     in_channels=channels[0],
        #     out_channels=channels[0],
        #     kernel_size=3,
        #     stride=2,
        #     bias=True,
        #     dimension=3)
        self.fconv1 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.trans0 = Residual_SparseTransformer_Block(channels[0], channels[0])
        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.bnu0 = ME.MinkowskiBatchNorm(channels[1])

        self.color_block0 = nn.Sequential(
            ResidualBlock(channels[1], channels[1], use_res=True, use_bias=True),
            ResidualBlock(channels[1], channels[1], use_res=True, use_bias=True),
            # ResidualBlock(channels[1], channels[1], use_res=True, use_bias=True),
        )

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.bnc0 = ME.MinkowskiBatchNorm(channels[2])
        # self.conv0_cls = ME.MinkowskiConvolution(
        #     in_channels=channels[2],
        #     out_channels=1,
        #     kernel_size=5,
        #     stride=1,
        #     bias=True,
        #     dimension=3)
        self.trans1 = Residual_SparseTransformer_Block(channels[2], channels[2])
        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.bnu1 = ME.MinkowskiBatchNorm(channels[3])
        self.color_block1 = nn.Sequential(
            ResidualBlock(channels[3], channels[3], use_res=True, use_bias=True),
            ResidualBlock(channels[3], channels[3], use_res=True, use_bias=True),
            # ResidualBlock(channels[3], channels[3], use_res=True, use_bias=True),
        )

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.bnc1 = ME.MinkowskiBatchNorm(channels[4])
        # self.conv1_cls = ME.MinkowskiConvolution(
        #     in_channels=channels[4],
        #     out_channels=1,
        #     kernel_size=5,
        #     stride=1,
        #     bias=True,
        #     dimension=3)
        self.trans2 = Residual_SparseTransformer_Block(channels[4], channels[4])
        self.up2 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.bnu2 = ME.MinkowskiBatchNorm(channels[5])
        self.color_block2 = nn.Sequential(
            ResidualBlock(channels[5], channels[5], use_res=True, use_bias=True),
            ResidualBlock(channels[5], channels[5], use_res=True, use_bias=True),
            # ResidualBlock(channels[5], channels[5], use_res=True, use_bias=True),
        )

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[5],
            out_channels=channels[6],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.bnc2 = ME.MinkowskiBatchNorm(channels[6])
        # self.conv2_cls = ME.MinkowskiConvolution(
        #     in_channels=channels[6],
        #     out_channels=1,
        #     kernel_size=5,
        #     stride=1,
        #     bias=True,
        #     dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, nums_list, ground_truth_list, training=True):
        # out = self.fconv0(x)
        out = self.fconv1(x)

        out = self.trans0(out)
        out = self.relu(self.conv0(self.color_block0(self.relu(self.up0(out)))))

        out_cls_0 = out

        # out = self.prune_voxel(out, out_cls_0, nums_list[0], ground_truth_list[0], training)
        #
        out = self.trans1(out)
        out = self.relu(self.conv1(self.color_block1(self.relu(self.up1(out)))))

        out_cls_1 = out
        # out = self.prune_voxel(out, out_cls_1, nums_list[1], ground_truth_list[1], training)
        #
        out = self.trans2(out)
        out = self.conv2(self.color_block2(self.relu(self.up2(out))))

        out_cls_2 = out
        # out = self.prune_voxel(out, out_cls_2, nums_list[2], ground_truth_list[2], training)

        out_cls_list = [out_cls_0, out_cls_1, out_cls_2]

        return out_cls_list, out


class Encoder_attri_z(torch.nn.Module):
    def __init__(self, channels=[3, 64, 128, 128, 128, 128]):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.relu(self.conv0_0(x))

        out1 = self.relu(self.down0(self.conv0(out0)))

        out2 = self.down1(self.conv1(out1))

        return [out2, out1, out0]


# Decoder_attri
class Decoder_attri_z(torch.nn.Module):
    """the decoding network with upsampling.
    """

    def __init__(self, channels=[128, 128, 128, 128, 64, 3]):
        super().__init__()
        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        # self.conv0_cls = ME.MinkowskiConvolution(
        #     in_channels=channels[2],
        #     out_channels=1,
        #     kernel_size=5,
        #     stride=1,
        #     bias=True,
        #     dimension=3)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        # self.conv1_cls = ME.MinkowskiConvolution(
        #     in_channels=channels[4],
        #     out_channels=1,
        #     kernel_size=5,
        #     stride=1,
        #     bias=True,
        #     dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.pruning = ME.MinkowskiPruning()

    # 原来的修剪函数
    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        # mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    # def prune_voxel(self, data, data_cls, nums, ground_truth, training):
    #
    #     if training:
    #         assert not ground_truth is None
    #         mask_true = isin(data_cls.C, ground_truth.C)
    #         mask = mask_true
    #         data_pruned = self.pruning(data, mask.to(data.device))
    #         return data_pruned
    #     else:
    #         return data

    def forward(self, x):
        #
        out = self.relu(self.conv0(self.up0(x)))

        out_cls_0 = out

        # out = self.prune_voxel(out, out_cls_0, nums_list[0], ground_truth_list[0], training)
        #
        out = self.relu(self.conv1(self.up1(out)))

        out_cls_1 = out
        # out = self.prune_voxel(out, out_cls_1, nums_list[1], ground_truth_list[1], training)
        #
        # out = self.prune_voxel(out, out_cls_2, nums_list[2], ground_truth_list[2], training)

        out = self.conv2(out)

        out_cls_list = [out, out_cls_1, out_cls_0]

        return out_cls_list, out
