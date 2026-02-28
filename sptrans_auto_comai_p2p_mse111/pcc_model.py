import torch
import torchac
import torch.nn as nn
import MinkowskiEngine as ME
import op
from .autoencoder import Encoder_attri_y, Decoder_attri_y, Encoder_attri_z, Decoder_attri_z
from .entropy_model import EntropyBottleneck
from compressai.entropy_models import EntropyBottleneck as EntropyBottleneck_com
from compressai.entropy_models import GaussianConditional
from .quant import NoiseQuant, SteQuant
from .blocks import maskedConv3D
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
)

from pytorch3d.structures import Pointclouds
from rendering_compare_yuv_new_cofig import Ticker
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PCCModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.encoder = Encoder(channels=[3, 64, 128, 128, 128, 128])
        # self.decoder = Decoder(channels=[128, 128, 128, 64, 3])

        self.encoder = Encoder_attri(channels=[3, 64, 128, 128, 128, 128, 128])
        self.decoder = Decoder_attri(channels=[128, 128, 128, 128, 128, 64, 3])

        self.entropy_bottleneck = EntropyBottleneck(128)

        self.renderers = self.get_renderers()

    def get_renderers(self):
        # 先创建多个render 后面再改参数使得 创建一个render + 多RT
        # Initialize a camera.
        # R, T = look_at_view_transform(250, 0, 60)
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)
        # cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.008,
            points_per_pixel=10,
            bin_size=64,
            max_points_per_bin=40000
        )

        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        # rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        rasterizer = PointsRasterizer(raster_settings=raster_settings)

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        ).to(device)

        return renderer

    def render_imgs(self, renderers, pointcloud, distance, elevation, azimuth):
        R = torch.zeros(len(elevation) * len(azimuth), 3, 3)
        T = torch.zeros(len(elevation) * len(azimuth), 3)
        i = 0
        for ele in elevation:
            for azi in azimuth:
                R[i, ...], T[i, ...] = look_at_view_transform(distance, ele, azi)
                i = i + 1

        cameras = FoVPerspectiveCameras(device=device, R=R.to(device), T=T.to(device), znear=0.01)

        for j, camera in enumerate(cameras):
            img = renderers(pointcloud, cameras=camera)
            if j == 0:
                imgs = img
            else:
                imgs = torch.cat([imgs, img], dim=2)

        return imgs

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F, quantize_mode=quantize_mode)

        data_Q = ME.SparseTensor(
            features=data_F,
            coordinate_map_key=data.coordinate_map_key,
            coordinate_manager=data.coordinate_manager,
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        # generate refed imgs from pytorch3d point cloud rendering
        # input_PCs = Pointclouds(
        #     points=[coords.type(torch.float32) for coords in x.decomposed_coordinates],
        #     features=x.decomposed_features)
        #
        # ref_imgs = self.render_imgs(self.renderers, input_PCs,
        #                             self.config.distance,
        #                             self.config.elevation,
        #                             self.config.azimuth)

        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y, quantize_mode="noise" if training else "symbols")

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'prior': y_q,
                'likelihood': likelihood,
                'ground_truth_list': ground_truth_list}


class PCCModel_Hyper_Com(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.quant_noise = NoiseQuant(table_range=128)
        self.step_noise = SteQuant(table_range=128)

        self.encoder_y = Encoder_attri_y(channels=[1, 64, 128, 128, 128, 128, 128])
        self.decoder_y = Decoder_attri_y(channels=[128, 128, 128, 128, 128, 64, 1])

        self.gaussian_conditional = GaussianConditional(None)

        self.encoder_z = Encoder_attri_z(channels=[128, 128, 128, 128, 128, 128])
        self.decoder_z = Decoder_attri_z(channels=[128, 128, 128, 128, 128, 128])

        self.entropy_bottleneck = EntropyBottleneck_com(128)

    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder_y(x)
        y = y_list[0]

        # y = ME.SparseTensor(
        #     features=torch.abs(y.F),
        #     coordinate_map_key=y.coordinate_map_key,
        #     coordinate_manager=y.coordinate_manager,
        #     device=y.device)

        z_list = self.encoder_z(y)
        z = z_list[0]

        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]

        z_hat_F, z_likelihoods = self.entropy_bottleneck(z.F, training=training)

        z_hat = ME.SparseTensor(
            features=z_hat_F,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)

        _, scales_hat = self.decoder_z(z_hat)

        _, y_likelihoods = self.gaussian_conditional(y.F, scales_hat.F, training=training)
        y_hat_F = self.step_noise(y.F)

        y_hat = ME.SparseTensor(
            features=y_hat_F,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)

        # Decoder
        out_cls_list, out = self.decoder_y(y_hat, nums_list, ground_truth_list, training)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'prior': y_hat,
                'likelihood': [y_likelihoods, z_likelihoods],
                'ground_truth_list': ground_truth_list}


class PCCModel_Mean_Hyper_Com(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.quant_noise = NoiseQuant(table_range=128)
        self.step_noise = SteQuant(table_range=128)

        self.encoder_y = Encoder_attri_y(channels=[3, 64, 128, 128, 128, 128, 128])
        self.decoder_y = Decoder_attri_y(channels=[128, 128, 128, 128, 128, 64, 3])

        self.gaussian_conditional = GaussianConditional(None)

        self.encoder_z = Encoder_attri_z(channels=[128, 128, 128, 128, 128, 128])
        self.decoder_z = Decoder_attri_z(channels=[128, 128, 128, 128, 128, 256])

        self.entropy_bottleneck = EntropyBottleneck_com(128)

    def forward(self, x, training=True):

        y_list = self.encoder_y(x)
        y = y_list[0]

        # y = ME.SparseTensor(
        #     features=torch.abs(y.F),
        #     coordinate_map_key=y.coordinate_map_key,
        #     coordinate_manager=y.coordinate_manager,
        #     device=y.device)

        z_list = self.encoder_z(y)
        z = z_list[0]

        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]

        z_hat_F, z_likelihoods = self.entropy_bottleneck(z.F, training=training)

        z_hat = ME.SparseTensor(
            features=z_hat_F,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)

        _, gaussian_params = self.decoder_z(z_hat)

        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(y.F, scales_hat, means=means_hat, training=training)

        # y_bpp = -torch.sum(torch.log2(y_likelihoods + 1e-10)) / float(x.__len__())

        y_hat_F = self.quant_noise(y.F, training=training)

        # test compress
        # min_v_value, max_v_value = y_hat_F.min().to(torch.int16), means_hat.max().to(torch.int16)
        # y_bytestream = torchac.encode_int16_normalized_cdf(
        #     op._convert_to_int_and_normalize(
        #         op.get_cdf_min_max_v(means_hat - min_v_value, scales_hat, L=max_v_value - min_v_value + 1),
        #         needs_normalization = True).cpu(),
        #         (y_hat_F - min_v_value).cpu().to(torch.int16)
        # )
        #
        # y_bpp_compress = len(y_bytestream) * 8 / float(x.__len__())

        y_hat = ME.SparseTensor(
            features=y_hat_F,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)

        # Decoder
        out_cls_list, out = self.decoder_y(y_hat, nums_list, ground_truth_list, training)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'prior': y_hat,
                'likelihood': [y_likelihoods, z_likelihoods],
                'ground_truth_list': ground_truth_list}

    def compress_decompressV2(self, x, training=False):
        # Encoder
        y_list = self.encoder_y(x)
        y = y_list[0]

        z_list = self.encoder_z(y)
        z = z_list[0]

        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]

        z_hat_F, z_likelihoods = self.entropy_bottleneck(z.F, training=training)

        self.entropy_bottleneck.update(force=True)
        z_size = z_hat_F.size()[2:]
        z_strings = self.entropy_bottleneck.compress(z_hat_F)

        z_hat_F_decompressed = self.entropy_bottleneck.decompress(z_strings, z_size)

        z_hat = ME.SparseTensor(
            features=z_hat_F_decompressed,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)

        _, gaussian_params = self.decoder_z(z_hat)

        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)

        y_hat_F_comai, y_likelihoods = self.gaussian_conditional(y.F, scales_hat, means=means_hat, training=False)

        # y_hat_F = self.quant_noise(y.F, training=False)

        y_hat_F_comai_minusmean = torch.round(y_hat_F_comai - means_hat)

        # test compress
        min_v_value, max_v_value = y_hat_F_comai_minusmean.min().to(torch.int16), y_hat_F_comai_minusmean.max().to(torch.int16)

        y_bytestream = torchac.encode_int16_normalized_cdf(
            op._convert_to_int_and_normalize(
                op.get_cdf_min_max_v(0 - min_v_value.cpu(), scales_hat.cpu(), L=max_v_value.cpu() - min_v_value.cpu() + 1),
                needs_normalization=True),
            (y_hat_F_comai_minusmean - min_v_value).cpu().to(torch.int16)
        )

        # cdf = op.get_cdf_min_max_v(torch.zeros(scales_hat.shape), scales_hat.cpu(), max_v_value.cpu() - min_v_value.cpu() + 1)
        # y_strings = torchac.encode_float_cdf(cdf, torch.round(y_hat_F_comai_minusmean - min_v_value).cpu(), check_input_bounds=True)

        # bpp test
        # y_bpp = -torch.sum(torch.log2(y_likelihoods + 1e-10)) / float(x.__len__())
        # y_bpp_compress = len(y_bytestream) * 8 / float(x.__len__())

        y_hat_F_decompressed = torchac.decode_int16_normalized_cdf(
            op._convert_to_int_and_normalize(
                op.get_cdf_min_max_v(0 - min_v_value.cpu(), scales_hat.cpu(), L=max_v_value.cpu() - min_v_value.cpu() + 1),
                needs_normalization=True),
            y_bytestream
        ) + min_v_value.cpu() + means_hat.cpu()

        y_hat_F_decompressed = y_hat_F_decompressed.float().to(y.device)

        y_hat = ME.SparseTensor(
            features=y_hat_F_decompressed,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)

        # Decoder
        out_cls_list, out = self.decoder_y(y_hat, nums_list, ground_truth_list, False)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'prior': y_hat,
                'likelihood': [y_likelihoods, z_likelihoods],
                'ground_truth_list': ground_truth_list,
                'time': [],
                'bpps': [len(y_bytestream) * 8 / float(x.__len__()), len(z_strings) * 8 / float(x.__len__())]}

    def compress_decompress(self, x, training=False):

        for i in range(1):

            ticker = op.Ticker()
            # Encoder
            ticker.start_count('encoder')
            y_list = self.encoder_y(x)
            y = y_list[0]

            z_list = self.encoder_z(y)
            z = z_list[0]
            ticker.end_count('encoder')

            ground_truth_list = y_list[1:] + [x]
            nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]

            z_hat_F, z_likelihoods = self.entropy_bottleneck(z.F, training=training)

            self.entropy_bottleneck.update(force=True)
            z_size = z_hat_F.size()[2:]

            ticker.start_count('z_compress')
            z_strings = self.entropy_bottleneck.compress(z_hat_F)
            ticker.end_count('z_compress')

            ticker.start_count('z_decompress')
            z_hat_F_decompressed = self.entropy_bottleneck.decompress(z_strings, z_size)

            z_hat = ME.SparseTensor(
                features=z_hat_F_decompressed,
                coordinate_map_key=z.coordinate_map_key,
                coordinate_manager=z.coordinate_manager,
                device=z.device)
            ticker.end_count('z_decompress')

            ticker.start_count('z_decoder')
            _, gaussian_params = self.decoder_z(z_hat)
            scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
            ticker.end_count('z_decoder')

            _, y_likelihoods = self.gaussian_conditional(y.F, scales_hat, means=means_hat, training=False)

            y_hat_F = self.quant_noise(y.F, training=False)

            # compress
            ticker.start_count('y_compress')
            min_v_value, max_v_value = y_hat_F.min().to(torch.int16), y_hat_F.max().to(torch.int16)

            y_bytestream = torchac.encode_int16_normalized_cdf(
                op._convert_to_int_and_normalize(
                    op.get_cdf_min_max_v(means_hat.cpu() - min_v_value.cpu(), scales_hat.cpu(), L=max_v_value.cpu() - min_v_value.cpu() + 1),
                    needs_normalization=True),
                (y_hat_F - min_v_value).cpu().to(torch.int16)
            )
            ticker.end_count('y_compress')

            # bpp test
            # y_bpp = -torch.sum(torch.log2(y_likelihoods + 1e-10)) / float(x.__len__())
            # y_bpp_compress = len(y_bytestream) * 8 / float(x.__len__())

            ticker.start_count('y_decompress')
            y_hat_F_decompressed = torchac.decode_int16_normalized_cdf(
                op._convert_to_int_and_normalize(
                    op.get_cdf_min_max_v(means_hat.cpu() - min_v_value.cpu(), scales_hat.cpu(), L=max_v_value.cpu() - min_v_value.cpu() + 1),
                    needs_normalization=True),
                y_bytestream
            ) + min_v_value.cpu()
            ticker.end_count('y_decompress')

            ticker.start_count('datatransfer_time')
            time1, time2, time3, time4, time5 = means_hat.cpu(), min_v_value.cpu(), scales_hat.cpu(), max_v_value.cpu(), min_v_value.cpu()
            ticker.end_count('datatransfer_time')

            y_hat_F_decompressed = y_hat_F_decompressed.float().cuda()

            ticker.start_count('y_decoder')
            y_hat = ME.SparseTensor(
                features=y_hat_F_decompressed,
                coordinate_map_key=y.coordinate_map_key,
                coordinate_manager=y.coordinate_manager,
                device=y.device)

            # Decoder
            out_cls_list, out = self.decoder_y(y_hat, nums_list, ground_truth_list, False)
            ticker.end_count('y_decoder')

            encode_time = 0
            decode_time = 0
            for key in ticker.dict.keys():
                t = ticker.dict[key]
                if key in ['encoder', 'y_compress']:
                    encode_time += t
                if key in ['z_decompress', 'z_decoder', 'y_decompress', 'y_decoder']:
                    decode_time += t
                if key in ['datatransfer_time']:
                    decode_time -= t

            encode_time = round(encode_time, 3)
            decode_time = round(decode_time, 3)

            print('encode_time', encode_time)
            print('decode_time', decode_time)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'prior': y_hat,
                'likelihood': [y_likelihoods, z_likelihoods],
                'ground_truth_list': ground_truth_list,
                'time': [encode_time, decode_time],
                'bpps': [len(y_bytestream) * 8 / float(x.__len__()), len(z_strings) * 8 / float(x.__len__())]}


class PCCModel_Mean_Hyper_Auto_Com(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.quant_noise = NoiseQuant(table_range=128)
        self.step_noise = SteQuant(table_range=128)

        self.encoder_y = Encoder_attri_y(channels=[3, 64, 128, 128, 128, 128, 128])
        self.decoder_y = Decoder_attri_y(channels=[128, 128, 128, 128, 128, 64, 3])

        self.gaussian_conditional = GaussianConditional(None)

        self.encoder_z_auto = Encoder_attri_z(channels=[128, 128, 128, 128, 128, 128])
        self.decoder_z_auto = Decoder_attri_z(channels=[128, 128, 128, 128, 128, 128])

        self.entropy_bottleneck = EntropyBottleneck_com(128)

        self.predict = maskedConv3D(masktype='A', in_channels=128, out_channels=256, kernel_size=5, dimension=3)
        self.entropy_parameters = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=384,
                                    out_channels=384,
                                    kernel_size=1,
                                    stride=1,
                                    bias=True,
                                    dimension=3),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(in_channels=384,
                                    out_channels=256,
                                    kernel_size=1,
                                    stride=1,
                                    bias=True,
                                    dimension=3),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(in_channels=256,
                                    out_channels=256,
                                    kernel_size=1,
                                    stride=1,
                                    bias=True,
                                    dimension=3), )

    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder_y(x)
        y = y_list[0]

        # y = ME.SparseTensor(
        #     features=torch.abs(y.F),
        #     coordinate_map_key=y.coordinate_map_key,
        #     coordinate_manager=y.coordinate_manager,
        #     device=y.device)

        z_list = self.encoder_z_auto(y)
        z = z_list[0]

        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]

        z_hat_F, z_likelihoods = self.entropy_bottleneck(z.F, training=training)

        z_hat = ME.SparseTensor(
            features=z_hat_F,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)

        _, params = self.decoder_z_auto(z_hat)

        y_hat_F = self.quant_noise(y.F, training=training)

        y_hat = ME.SparseTensor(
            features=y_hat_F,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)

        ctx_params = self.predict(y_hat)

        gaussian_params = self.entropy_parameters(ME.cat(ctx_params, params))

        means_hat, scales_hat = gaussian_params.F.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(y.F, scales_hat, means=means_hat, training=training)

        # y_bpp = -torch.sum(torch.log2(y_likelihoods + 1e-10)) / float(x.__len__())

        # test compress
        # min_v_value, max_v_value = y_hat_F.min().to(torch.int16), means_hat.max().to(torch.int16)
        # y_bytestream = torchac.encode_int16_normalized_cdf(
        #     op._convert_to_int_and_normalize(
        #         op.get_cdf_min_max_v(means_hat - min_v_value, scales_hat, L=max_v_value - min_v_value + 1),
        #         needs_normalization = True).cpu(),
        #         (y_hat_F - min_v_value).cpu().to(torch.int16)
        # )
        #
        # y_bpp_compress = len(y_bytestream) * 8 / float(x.__len__())

        # Decoder
        out_cls_list, out = self.decoder_y(y_hat, nums_list, ground_truth_list, training)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'prior': y_hat,
                'likelihood': [y_likelihoods, z_likelihoods],
                'ground_truth_list': ground_truth_list}


class PCCModel_Mean_Hyper_torchac(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_y = Encoder_attri_y(channels=[3, 64, 128, 128, 128, 128, 128])
        self.decoder_y = Decoder_attri_y(channels=[128, 128, 128, 128, 128, 64, 3])

        self.encoder_z = Encoder_attri_z(channels=[128, 128, 128, 128, 128, 128])
        self.decoder_z = Decoder_attri_z(channels=[128, 128, 128, 128, 128, 256])

        self.entropy_bottleneck = EntropyBottleneck(128)

    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder_y(x)
        y = y_list[0]

        # y = ME.SparseTensor(
        #     features=torch.abs(y.F),
        #     coordinate_map_key=y.coordinate_map_key,
        #     coordinate_manager=y.coordinate_manager,
        #     device=y.device)

        z_list = self.encoder_z(y)
        z = z_list[0]

        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]

        z_hat_F, z_likelihoods = self.entropy_bottleneck(z.F, quantize_mode="noise" if training else "symbols")

        z_hat = ME.SparseTensor(
            features=z_hat_F,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)

        _, gaussian_params = self.decoder_z(z_hat)

        mu, sigma = gaussian_params.F.chunk(2, 1)
        sigma = sigma.abs()
        sigma = torch.clamp(sigma, min=1e-8)

        if training == True:
            y_hat_F = self.entropy_bottleneck._quantize(y.F, 'noise')
        else:
            y_hat_F = self.entropy_bottleneck._quantize(y.F, 'symbols')

        _, y_likelihoods = op.feature_probs_based_mu_sigma(y_hat_F, mu, sigma)

        y_hat = ME.SparseTensor(
            features=y_hat_F,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)

        # Decoder
        out_cls_list, out = self.decoder_y(y_hat, nums_list, ground_truth_list, training)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'prior': y_hat,
                'likelihood': [y_likelihoods, z_likelihoods],
                'ground_truth_list': ground_truth_list}


if __name__ == '__main__':
    model = PCCModel()
    print(model)
