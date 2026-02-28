import torch
import MinkowskiEngine as ME
import torch.nn as nn
from typing import Any
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Function
import pytorch3d as p3d

from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
import sparse_ops as ops


class LocalSelfAttentionBase(nn.Module):
    def __init__(self, kernel_size, stride, dilation, dimension):
        super(LocalSelfAttentionBase, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension

        self.kernel_generator = KernelGenerator(kernel_size=kernel_size,
                                                stride=stride,
                                                dilation=dilation,
                                                dimension=dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume

    def get_kernel_map_and_out_key(self, stensor):
        cm = stensor.coordinate_manager
        in_key = stensor.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        region_type, region_offset, _ = self.kernel_generator.get_kernel(
            stensor.tensor_stride, False)
        kernel_map = cm.kernel_map(in_key,
                                   out_key,
                                   self.kernel_generator.kernel_stride,
                                   self.kernel_generator.kernel_size,
                                   self.kernel_generator.kernel_dilation,
                                   region_type=region_type,
                                   region_offset=region_offset)
        return kernel_map, out_key

    def key_query_map_from_kernel_map(self, kernel_map):
        kq_map = []
        for kernel_idx, in_out in kernel_map.items():
            in_out[0] = in_out[0] * self.kernel_volume + kernel_idx
            kq_map.append(in_out)
        kq_map = torch.cat(kq_map, -1)
        return kq_map

    def key_query_indices_from_kernel_map(self, kernel_map):
        kq_indices = []
        for _, in_out in kernel_map.items():
            kq_indices.append(in_out)

        kq_indices = torch.cat(kq_indices, -1)
        return kq_indices

    def key_query_indices_from_key_query_map(self, kq_map):
        kq_indices = kq_map.clone()
        kq_indices[0] = kq_indices[0] // self.kernel_volume
        return kq_indices


class SparseTransformer(LocalSelfAttentionBase):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            kernel_size=5,
            stride=1,
            dilation=1,
            num_heads=1,
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0
        assert kernel_size % 2 == 1
        assert stride == 1, "Currently, this layer only supports stride == 1"
        assert dilation == 1, "Currently, this layer only supports dilation == 1"
        super(SparseTransformer, self).__init__(kernel_size, stride, dilation, dimension=3)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        self.to_query = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_key = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_value = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_out = nn.Linear(out_channels, out_channels)

        self.inter_pos_enc = nn.Parameter(torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels))

        nn.init.normal_(self.inter_pos_enc, 0, 1)

    def forward(self, stensor):
        dtype = stensor._F.dtype
        device = stensor._F.device

        q = self.to_query(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()
        k = self.to_key(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()
        v = self.to_value(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()

        norm_q = F.normalize(q, p=2, dim=-1)
        norm_k = F.normalize(k, p=2, dim=-1)
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)

        # key-query map
        kernel_map, out_key = self.get_kernel_map_and_out_key(stensor)

        kq_map = self.key_query_map_from_kernel_map(kernel_map)

        # attention weights with cosine similarity
        attn = torch.zeros((kq_map.shape[1], self.num_heads), dtype=dtype, device=device)
        attn = ops.atten_weights_transfor(norm_q, norm_k, norm_pos_enc, attn, kq_map)

        # aggregation & the output
        kq_indices = self.key_query_indices_from_key_query_map(kq_map)
        out_F = torch.zeros((len(q), self.num_heads, self.attn_channels), dtype=dtype, device=device)
        out_F = ops.scalar_attention_cuda(attn, v, out_F, kq_indices)

        out_F = self.to_out(out_F.view(-1, self.out_channels).contiguous())
        return ME.SparseTensor(out_F, coordinate_map_key=out_key, coordinate_manager=stensor.coordinate_manager) + stensor


class LightweightSelfAttentionLayer(LocalSelfAttentionBase):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            kernel_size=5,
            stride=1,
            dilation=1,
            num_heads=1,
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0
        assert kernel_size % 2 == 1
        assert stride == 1, "Currently, this layer only supports stride == 1"
        assert dilation == 1, "Currently, this layer only supports dilation == 1"
        super(LightweightSelfAttentionLayer, self).__init__(kernel_size, stride, dilation, dimension=3)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        self.to_query = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_value = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_out = nn.Linear(out_channels, out_channels)

        self.inter_pos_enc = nn.Parameter(torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels))
        # self.intra_pos_mlp = nn.Sequential(
        #     nn.Linear(3, 3, bias=False),
        #     nn.BatchNorm1d(3),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(3, in_channels, bias=False),
        #     nn.BatchNorm1d(in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_channels, in_channels)
        # )
        nn.init.normal_(self.inter_pos_enc, 0, 1)

    def forward(self, stensor):
        dtype = stensor._F.dtype
        device = stensor._F.device

        # query, key, value, and relative positional encoding
        # intra_pos_enc = self.intra_pos_mlp(norm_points)
        # stensor = stensor + intra_pos_enc

        q = self.to_query(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()
        v = self.to_value(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()

        # key-query map
        kernel_map, out_key = self.get_kernel_map_and_out_key(stensor)
        kq_map = self.key_query_map_from_kernel_map(kernel_map)

        # attention weights with cosine similarity
        attn = torch.zeros((kq_map.shape[1], self.num_heads), dtype=dtype, device=device)
        norm_q = F.normalize(q, p=2, dim=-1)
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)
        attn = ops.dot_product_cuda(norm_q, norm_pos_enc, attn, kq_map)

        # norm_q_clone = norm_q.clone().requires_grad_(True)
        # norm_pos_enc_clone = norm_pos_enc.clone().requires_grad_(True)
        # attn_clone = attn.clone()
        # kq_map_clone = kq_map.clone()

        # attn2 = torch.zeros((kq_map.shape[1], self.num_heads), dtype=dtype, device=device)
        # for i in range(kq_map.shape[1]):
        #     query_index = kq_map[1, i]
        #     kernel_index = kq_map[0, i] % (self.kernel_size ** 3)
        #
        #     attn2[i, 0] = torch.dot(norm_q[query_index, 0, :], norm_pos_enc[kernel_index, 0, :])

        # aggregation & the output
        out_F = torch.zeros((len(q), self.num_heads, self.attn_channels), dtype=dtype, device=device)
        kq_indices = self.key_query_indices_from_key_query_map(kq_map)
        out_F = ops.scalar_attention_cuda(attn, v, out_F, kq_indices)
        out_F = self.to_out(out_F.view(-1, self.out_channels).contiguous())
        return ME.SparseTensor(out_F, coordinate_map_key=out_key, coordinate_manager=stensor.coordinate_manager) + stensor


class KAttention(nn.Module):
    """
    Args:
        N (int): Number of channels)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 k=16,
                 sub_sample=False,
                 bn_layer=False):
        super(KAttention, self).__init__()
        self.k = k
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数

        # max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.W = conv1x1(in_ch=self.inter_channels,
                         out_ch=self.in_channels)
        nn.init.constant_(self.W.kernel, 0)
        nn.init.constant_(self.W.bias, 0)

        self.g = nn.Linear(in_features=in_channels, out_features=inter_channels, bias=True)

        if bn_layer:
            pass

        self.theta = nn.Linear(in_features=in_channels, out_features=inter_channels, bias=True)
        self.phi = nn.Linear(in_features=in_channels, out_features=inter_channels, bias=True)

        self.pe1 = nn.Sequential(
            nn.Linear(in_features=3, out_features=inter_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels, out_features=inter_channels, bias=True),
        )

        self.pe2 = nn.Sequential(
            nn.Linear(in_features=3, out_features=inter_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels, out_features=inter_channels, bias=True),
        )

        if sub_sample:
            pass
            # self.g = nn.Sequential(self.g, max_pool_layer)
            # self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = len(x.decomposed_features)
        feats_list = []
        for b_idx in range(0, batch_size):
            coordinates = x.decomposed_coordinates[b_idx].float()
            features = x.decomposed_features[b_idx]

            dists, query_idx, nn_coords = p3d.ops.knn_points(coordinates.unsqueeze(0), coordinates.unsqueeze(0), K=self.k, return_nn=True)
            nn_feats = p3d.ops.knn_gather(features.unsqueeze(0), query_idx).squeeze(0)

            theta_feats = self.theta(features.unsqueeze(1))
            phi_feats = self.phi(nn_feats)

            nn_coords = nn_coords.squeeze(0)

            delta_coords = nn_coords - coordinates.unsqueeze(1)
            pe1 = self.pe1(delta_coords)
            pe2 = self.pe1(delta_coords)
            # subtraction
            delta_features = (pe2 * (theta_feats - phi_feats) + pe1) / torch.sqrt(torch.tensor(self.inter_channels, device=x.device))

            # phi_feats = phi_feats.permute(0, 2, 1)
            # sim_vecs = torch.matmul(theta_feats, phi_feats) + pe

            sim_matrix = F.softmax(delta_features, dim=1)

            g_feats = self.g(nn_feats)

            # final_features = torch.matmul(sim_matrix, g_feats).squeeze()

            final_features = torch.sum(sim_matrix * g_feats, dim=1)

            feats_list.append(final_features)

        y_feats = torch.cat(feats_list)
        y = ME.SparseTensor(
            features=y_feats,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)

        W_y = self.W(y)

        return W_y + x


class Edge_conv(nn.Module):
    """
    Args:
        N (int): Number of channels)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 k=16,
                 sub_sample=False,
                 bn_layer=False):
        super(Edge_conv, self).__init__()
        self.k = k
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数

        if bn_layer:
            pass
        else:
            self.W = conv1x1(in_ch=self.inter_channels,
                             out_ch=self.in_channels)
            nn.init.constant_(self.W.kernel, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Sequential(
            nn.Linear(in_features=self.in_channels * 2, out_features=self.inter_channels, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.inter_channels, out_features=self.in_channels, bias=True)
        )

        if sub_sample:
            pass
            # self.g = nn.Sequential(self.g, max_pool_layer)
            # self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = len(x.decomposed_features)
        feats_list = []
        for b_idx in range(0, batch_size):
            x_coordinates = x.decomposed_coordinates[b_idx].float()
            x_features = x.decomposed_features[b_idx]

            dists, query_idx, _ = p3d.ops.knn_points(x_coordinates.unsqueeze(0), x_coordinates.unsqueeze(0), K=self.k)
            nn_feats = p3d.ops.knn_gather(x_features.unsqueeze(0), query_idx).squeeze(0)

            delta_feats = nn_feats - x_features.unsqueeze(1)
            grouped_feats = torch.cat((nn_feats, delta_feats), dim=2)

            theta_feats = self.theta(grouped_feats)

            weights = F.softmax(-dists.squeeze(), dim=1)
            weighted_average_feats = torch.matmul(weights.unsqueeze(1), theta_feats).squeeze()

            feats_list.append(weighted_average_feats)

        new_x = ME.SparseTensor(features=torch.cat(feats_list, dim=0),
                                coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager,
                                device=x.device)
        # torch.cuda.empty_cache()
        return new_x


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_res=True, use_bias=True):
        super().__init__()
        self.use_res = use_res
        self.main = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    bias=use_bias,
                                    dimension=3),
            # ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    bias=use_bias,
                                    dimension=3),
            # ME.MinkowskiBatchNorm(out_channels)
        )
        self.relu = ME.MinkowskiReLU(True)

    def forward(self, x):
        if self.use_res:
            x = self.relu(self.main(x) + x)
        else:
            x = self.relu(self.main(x))
        return x


class Residual_SparseTransformer_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_res=True, use_bias=True):
        super().__init__()
        self.use_res = use_res
        self.main = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    bias=use_bias,
                                    dimension=3),
            # ME.MinkowskiReLU(True),
            # ME.MinkowskiBatchNorm(out_channels),

            SparseTransformer(in_channels=out_channels),
            # ME.MinkowskiReLU(True),
            # ME.MinkowskiBatchNorm(out_channels),

            ME.MinkowskiConvolution(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    bias=use_bias,
                                    dimension=3),
        )
        # self.relu = ME.MinkowskiELU(True)
        # self.bn = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        if self.use_res:
            x = self.main(x) + x
        else:
            x = self.main(x)
        return x

class Residual_Transformer_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_res=True, use_bias=True):
        super().__init__()
        self.use_res = use_res
        self.main = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    bias=use_bias,
                                    dimension=3),
            ME.MinkowskiReLU(True),
            ME.MinkowskiBatchNorm(out_channels),

            KAttention(in_channels=out_channels,
                       inter_channels=out_channels,
                       k=16),
            ME.MinkowskiReLU(True),
            ME.MinkowskiBatchNorm(out_channels),

            ME.MinkowskiConvolution(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    bias=use_bias,
                                    dimension=3),
        )
        self.relu = ME.MinkowskiReLU(True)
        self.bn = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        if self.use_res:
            x = self.bn(self.relu(self.main(x) + x))
        else:
            x = self.bn(self.relu(self.main(x)))
        return x

class maskedConv3D(ME.MinkowskiConvolution):
    def __init__(self, masktype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.kernel.data.clone())
        kD, _, _ = self.kernel.size()
        self.mask.fill_(1)
        self.mask[kD // 2 + (masktype == 'B'):, :, :] = 0
    def forward(self, x):
        self.kernel.data *= self.mask
        return super(maskedConv3D, self).forward(x)


class MaskMEConv(ME.MinkowskiConvolution):

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.kernel.data))
        n, _, _ = self.mask.size()
        self.mask[n // 2 + (mask_type == "B"):, :, :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.kernel.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=3, stride=stride, bias=True, dimension=3)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=1, stride=stride, bias=True, dimension=3)


# 注意力只由卷积得到
class AttentionBlock_convtype(nn.Module):
    """
    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    ME.MinkowskiReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    ME.MinkowskiReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = ME.MinkowskiReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

        self.sigm = ME.MinkowskiSigmoid()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * self.sigm(b)
        out += identity
        return out


class AttentionBlock_convtype_woMB(nn.Module):
    """
    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    ME.MinkowskiReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    ME.MinkowskiReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = ME.MinkowskiReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        # self.conv_b = nn.Sequential(
        #     ResidualUnit(),
        #     ResidualUnit(),
        #     ResidualUnit(),
        #     conv1x1(N, N),
        # )

        # self.sigm = ME.MinkowskiSigmoid()

    def forward(self, x: Tensor) -> Tensor:
        a = self.conv_a(x)
        out = a + x
        return out


# 注意要batch内矩阵相乘
# 注意力由特征矩阵相乘得到（协方差矩阵）
class AttentionBlock_bmmtype(nn.Module):
    """
    Args:
        N (int): Number of channels)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=False,
                 bn_layer=False):
        super(AttentionBlock_bmmtype, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数

        # max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.g = conv1x1(in_ch=self.in_channels,
                         out_ch=self.inter_channels)

        if bn_layer:
            pass
        else:
            self.W = conv1x1(in_ch=self.inter_channels,
                             out_ch=self.in_channels)
            nn.init.constant_(self.W.kernel, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv1x1(in_ch=self.in_channels,
                             out_ch=self.inter_channels)
        self.phi = conv1x1(in_ch=self.in_channels,
                           out_ch=self.inter_channels)

        if sub_sample:
            pass
            # self.g = nn.Sequential(self.g, max_pool_layer)
            # self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        g_x = self.g(x)
        theta_x = self.theta(x)
        phi_x = self.phi(x)
        batch_size = len(x.decomposed_features)
        feats_list = []
        for b_idx in range(0, batch_size):
            g_x_feats = g_x.decomposed_features[b_idx]
            theta_x_feats = theta_x.decomposed_features[b_idx]
            phi_x_feats = phi_x.decomposed_features[b_idx]
            phi_x_feats = phi_x_feats.permute(1, 0)
            f = torch.mm(theta_x_feats, phi_x_feats)
            f_div_C = F.softmax(f, dim=-1)
            y = torch.mm(f_div_C, g_x_feats)
            feats_list.append(y)
        y_feats = torch.cat(feats_list)
        y = ME.SparseTensor(
            features=y_feats,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)
        W_y = self.W(y)
        z = W_y + x
        return z


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """

    def __init__(self,
                 ch,
                 device=torch.device('cuda'),
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)

        self.build(ch, torch.device(device))

    def build(self, ch, device):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=device) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=device)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x):
        inputs_F = x.F

        _, ch = inputs_F.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1)

        # Norm pool calc
        inputs_F = inputs_F.view(1, inputs_F.size(0), inputs_F.size(1))
        inputs_F = inputs_F.permute(0, 2, 1)
        norm_ = nn.functional.conv1d(inputs_F ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs_F = inputs_F * norm_
        else:
            outputs_F = inputs_F / norm_

        outputs_F = outputs_F.permute(0, 2, 1)
        outputs_F = outputs_F.squeeze()
        outputs = ME.SparseTensor(
            features=outputs_F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)

        return outputs


if __name__ == '__main__':
    coords0 = [
        [0, 0, 2.1],  #
        [0, 1, 1.4],  #
        [0, 0, 4.0]
    ]
    feats0 = [[1, 2], [1, 2], [1, 2]]
    coords1 = [
        [1, 0, 0],  #
        [0, 2, 0],  #
        [0, 0, 3]
    ]
    feats1 = [[1, 2], [1, 2], [1, 2]]

    coords, feats = ME.utils.sparse_collate(
        coords=[torch.tensor(coords0), torch.tensor(coords1)], feats=[torch.tensor(feats0), torch.tensor(feats1)])

    # sparse tensors
    A = ME.SparseTensor(coordinates=coords, features=feats.float())
    # conv = ME.MinkowskiConvolution(in_channels=1, out_channels=2, kernel_size=3, stride=2, dimension=3)
    # B = conv(A)

    channels = [2, 256]
    maskMEConv = MaskMEConv(
        in_channels=channels[0],
        out_channels=channels[1],
        kernel_size=3,
        stride=1,
        bias=True,
        dimension=3
    )
    C = maskMEConv(A)

    print(1)