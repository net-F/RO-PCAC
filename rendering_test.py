import os
import subprocess
import time
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    look_at_view_transform,
)
from pytorch3d.renderer.compositing import alpha_composite
from pytorch3d.structures import Pointclouds
from pytorch_msssim import ms_ssim

from data_loader import collate_pointcloud_fn
from sptrans_auto_comai_p2p_mse111.pcc_model import (
    PCCModel_Mean_Hyper_Com as PCCModel_hyper2_comai_sptrans_mse111,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rootdir = os.path.split(__file__)[0]


class AlphaCompositor(nn.Module):
    def __init__(
        self, background_color: Optional[Union[Tuple, List, torch.Tensor]] = None
    ) -> None:
        super().__init__()
        self.background_color = background_color

    def forward(self, fragments, alphas, ptclds, **kwargs):
        background_color = kwargs.get("background_color", self.background_color)
        images = alpha_composite(fragments, alphas, ptclds)
        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images


def _add_background_color_to_images(pix_idxs, images, background_color):
    background_mask = pix_idxs[:, 0] < 0

    if not torch.is_tensor(background_color):
        background_color = images.new_tensor(background_color)
    if background_color.ndim == 0:
        background_color = background_color.expand(images.shape[1])
    if background_color.ndim > 1:
        raise ValueError("Wrong shape of background_color")

    background_color = background_color.to(images)
    if background_color.shape[0] + 1 == images.shape[1]:
        background_color = torch.cat([background_color, images.new_ones(1)])
    if images.shape[1] != background_color.shape[0]:
        raise ValueError(
            f"Background color has {background_color.shape[0]} channels not {images.shape[1]}"
        )

    num_background_pixels = background_mask.sum()
    masked_images = images.permute(0, 2, 3, 1).masked_scatter(
        background_mask[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )
    return masked_images.permute(0, 3, 1, 2), ~background_mask


class PointsRenderer(nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, to_device):
        self.rasterizer = self.rasterizer.to(to_device)
        self.compositor = self.compositor.to(to_device)
        return self

    def forward(self, point_clouds, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)
        radius = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (radius * radius)
        images, mask = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        return images.permute(0, 2, 3, 1), mask


def get_bits(likelihood):
    return -torch.sum(torch.log2(likelihood))


def get_mse_611(imgs1, imgs2, masks):
    num_true_pixels = torch.sum(masks)
    channelwise_mse = (
        torch.sum(torch.square(imgs1 * 255 - imgs2 * 255), dim=(0, 1, 2)) / num_true_pixels
    )
    return (6 * channelwise_mse[0] + channelwise_mse[1] + channelwise_mse[2]) / 8.0


def get_mse_100(imgs1, imgs2, masks):
    num_true_pixels = torch.sum(masks)
    channelwise_mse = (
        torch.sum(torch.square(imgs1 * 255 - imgs2 * 255), dim=(0, 1, 2)) / num_true_pixels
    )
    return channelwise_mse[0]


def rgb2yuv(rgb):
    rgb = 255 * rgb
    yuv = rgb.copy()
    yuv[:, 0] = 0.257 * rgb[:, 0] + 0.504 * rgb[:, 1] + 0.098 * rgb[:, 2] + 16
    yuv[:, 1] = -0.148 * rgb[:, 0] - 0.291 * rgb[:, 1] + 0.439 * rgb[:, 2] + 128
    yuv[:, 2] = 0.439 * rgb[:, 0] - 0.368 * rgb[:, 1] - 0.071 * rgb[:, 2] + 128
    yuv[:, 0] = (yuv[:, 0] - 16) / (235 - 16)
    yuv[:, 1] = (yuv[:, 1] - 16) / (240 - 16)
    yuv[:, 2] = (yuv[:, 2] - 16) / (240 - 16)
    return yuv


def yuv2rgb(yuv):
    yuv[:, 0] = (235 - 16) * yuv[:, 0] + 16
    yuv[:, 1] = (240 - 16) * yuv[:, 1] + 16
    yuv[:, 2] = (240 - 16) * yuv[:, 2] + 16
    rgb = yuv.copy()
    rgb[:, 0] = 1.164 * (yuv[:, 0] - 16) + 1.596 * (yuv[:, 2] - 128)
    rgb[:, 1] = 1.164 * (yuv[:, 0] - 16) - 0.813 * (yuv[:, 2] - 128) - 0.392 * (yuv[:, 1] - 128)
    rgb[:, 2] = 1.164 * (yuv[:, 0] - 16) + 2.017 * (yuv[:, 1] - 128)
    return rgb / 255


def read_ply_ascii_mpeg(filedir):
    with open(filedir, "r", encoding="utf-8", errors="ignore") as files:
        data = []
        for line in files:
            wordslist = line.split(" ")
            try:
                line_values = []
                for v in wordslist:
                    if v == "\n":
                        continue
                    line_values.append(float(v))
            except ValueError:
                continue
            data.append(line_values)
    data = np.array(data)
    coords = data[:, 0:3].astype("int16")
    attri = data[:, 6:9].astype("float32")
    return coords, attri


def write_ply_mpeg(filedir, coords, feats):
    if os.path.exists(filedir):
        os.remove(filedir)
    with open(filedir, "a+", encoding="utf-8") as f:
        f.writelines(["ply\n", "format ascii 1.0\n"])
        f.write("element vertex " + str(coords.shape[0]) + "\n")
        f.writelines(["property float x\n", "property float y\n", "property float z\n"])
        f.writelines(["property uchar red\n", "property uchar green\n", "property uchar blue\n"])
        f.write("end_header\n")
        data = np.concatenate((coords, feats), axis=1).astype("int")
        for p in data:
            f.writelines([str(p[0]), " ", str(p[1]), " ", str(p[2]), " ", str(p[3]), " ", str(p[4]), " ", str(p[5]), "\n"])


def number_in_line(line):
    number = None
    for item in line.split(" "):
        try:
            number = float(item)
        except ValueError:
            continue
    if number is None:
        raise ValueError(f"No number found in line: {line}")
    return number


def pc_error(infile1, infile2, normal=False, show=False):
    headers_f = ["mseF      (p2point)", "mseF,PSNR (p2point)", "h.        (p2point)", "h.,PSNR   (p2point)"]
    headers_c = ["c[0],PSNRF", "c[1],PSNRF", "c[2],PSNRF"]
    headers = headers_f + headers_c

    command = (
        rootdir
        + "/pc_error_d"
        + " -a "
        + infile1
        + " -b "
        + infile2
        + " --color=1"
    )
    if normal:
        headers += ["mse1      (p2plane)", "mse1,PSNR (p2plane)", "mse2      (p2plane)", "mse2,PSNR (p2plane)", "mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command += " -n " + infile1

    results = {}
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    _ = time.time()
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding="utf-8")
        if show:
            print(line)
        for key in headers:
            if key in line:
                results[key] = number_in_line(line)
        c = subp.stdout.readline()
    return pd.DataFrame([results])


def render_imgs(renderers, pointcloud, distance, elevation, azimuth):
    num_views = len(elevation) * len(azimuth)
    r_mat = torch.zeros(num_views, 3, 3)
    t_mat = torch.zeros(num_views, 3)

    i = 0
    for ele in elevation:
        for azi in azimuth:
            r_mat[i, ...], t_mat[i, ...] = look_at_view_transform(distance, ele, azi)
            i += 1

    cameras = FoVPerspectiveCameras(
        device=device,
        R=r_mat.to(device),
        T=t_mat.to(device),
        znear=0.01,
    )

    for j, camera in enumerate(cameras):
        img, mask = renderers(pointcloud, cameras=camera)
        if j == 0:
            imgs = img
            masks = mask
        else:
            imgs = torch.cat([imgs, img], dim=2)
            masks = torch.cat([masks, mask], dim=2)
    return imgs, masks


def YUV611(input_imgs, decoded_imgs, ref_masks):
    mse = get_mse_611(input_imgs, decoded_imgs, ref_masks)
    return 20.0 * torch.log10(255.0 / torch.sqrt(mse))


def YUV100(input_imgs, decoded_imgs, ref_masks):
    mse = get_mse_100(input_imgs, decoded_imgs, ref_masks)
    return 20.0 * torch.log10(255.0 / torch.sqrt(mse))


filedirs_input = ["/home/hx/pycharm_projet/GPCC_data/long4/longdress.ply"]

ckpts_hyper2_comai_sptrans_imgmse111 = [
    ["./ckpts_new/sptrans_hyper2_comai_p2p_imgmse111/ckpt5/epoch_100_r05.pth"],
    ["./ckpts_new/sptrans_hyper2_comai_p2p_imgmse111/ckpt4/epoch_131_r04.pth"],
    ["./ckpts_new/sptrans_hyper2_comai_p2p_imgmse111/ckpt3/epoch_131_r03.pth"],
    ["./ckpts_new/sptrans_hyper2_comai_p2p_imgmse111/ckpt2/epoch_134_r02.pth"],
    ["./ckpts_new/sptrans_hyper2_comai_p2p_imgmse111/ckpt1/epoch_134_r01.pth"],
]

model_hyper2_comai_sptrans_mse111 = PCCModel_hyper2_comai_sptrans_mse111().to(device)


def get_renderers():
    raster_settings = PointsRasterizationSettings(
        image_size=1024,
        radius=0.002,
        points_per_pixel=8,
        bin_size=60,
        max_points_per_bin=80000,
    )
    rasterizer = PointsRasterizer(raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
    ).to(device)
    return renderer


def ai_performance_hyper(
    filedir_inputpc,
    file_name,
    centroid,
    input_imgs,
    renderer,
    input_pc,
    model,
    ckpts,
    metric_choice,
    version,
    distance,
    elevation,
    azimuth,
):
    metrics_ai = []
    bpp_ai = []

    with torch.no_grad():
        for ckptdir in ckpts:
            best_metric = 0
            for ckpt_now in ckptdir:
                assert os.path.exists(ckpt_now)
                ckpt = torch.load(ckpt_now)
                new_state_dict = model.state_dict()
                for name, param in ckpt["model"].items():
                    if name in new_state_dict and param.shape == new_state_dict[name].shape:
                        new_state_dict[name] = param
                model.load_state_dict(new_state_dict)

                print("load checkpoint from\t", ckpt_now)
                out_set_v2 = model.compress_decompressV2(input_pc, training=False)
                bpp = out_set_v2["bpps"][0] + out_set_v2["bpps"][1]
                output_pc = out_set_v2["out"]

                bpp_est_v2 = (get_bits(out_set_v2["likelihood"][0]) + get_bits(out_set_v2["likelihood"][1])) / float(len(input_pc.C))
                if bpp > (bpp_est_v2 + bpp_est_v2 * 0.05):
                    bpp = (bpp_est_v2 + bpp_est_v2 * 0.05).cpu().numpy()

                output_pc_yuv = Pointclouds(
                    points=[coords.type(torch.float32) for coords in output_pc.decomposed_coordinates],
                    features=output_pc.decomposed_features,
                )
                output_pc_rgb = Pointclouds(
                    points=[coords.type(torch.float32) for coords in output_pc.decomposed_coordinates],
                    features=[torch.tensor(yuv2rgb(feats.detach().cpu().numpy()), device=device) for feats in output_pc.decomposed_features],
                )

                output_imgs, renderer_masks = render_imgs(renderer, output_pc_yuv, distance, elevation, azimuth)
                output_imgs_rgb, _ = render_imgs(renderer, output_pc_rgb, distance, elevation, azimuth)
                output_img_rgb = np.clip(output_imgs_rgb[0, ..., :3].detach().cpu().numpy(), 0.0, 1.0)

                ckpt_name = os.path.splitext(os.path.basename(ckpt_now))[0]
                out_img_path = os.path.join("imgs_show", "compare", file_name, f"ai_dec_img_{file_name}_{ckpt_name}.png")
                matplotlib.image.imsave(out_img_path, output_img_rgb)

                filedir_decoded = os.path.join("imgs_show", "compare", file_name, f"ai_dec_img_{file_name}_{ckpt_name}.ply")
                coords = (output_pc.C.detach().cpu().numpy()[:, 1:] + centroid).astype("int16")
                attri = (yuv2rgb(output_pc.F.detach().cpu().numpy()) * 255).astype("int16")
                write_ply_mpeg(filedir_decoded, coords, attri)

                if metric_choice == "yuv611":
                    metric = YUV611(input_imgs, output_imgs, renderer_masks).detach().cpu().numpy()
                elif metric_choice == "y":
                    metric = YUV100(input_imgs, output_imgs, renderer_masks).detach().cpu().numpy()
                elif metric_choice == "ms-ssim":
                    metric = ms_ssim(
                        input_imgs.permute(0, 3, 1, 2),
                        output_imgs_rgb.permute(0, 3, 1, 2),
                        data_range=1.0,
                    ).detach().cpu().numpy()
                elif metric_choice == "p2p_y":
                    pc_error_metrics = pc_error(filedir_inputpc, filedir_decoded)
                    metric = pc_error_metrics["c[0],PSNRF"][0]
                elif metric_choice == "p2p_yuv611":
                    pc_error_metrics = pc_error(filedir_inputpc, filedir_decoded)
                    metric = (
                        6 * pc_error_metrics["c[0],PSNRF"][0]
                        + pc_error_metrics["c[1],PSNRF"][0]
                        + pc_error_metrics["c[2],PSNRF"][0]
                    ) / 8.0
                else:
                    raise ValueError(f"Unsupported metric: {metric_choice}")

                if metric > best_metric:
                    best_metric = metric
                    metrics_ai.append(best_metric)
                    bpp_ai.append(float(bpp))

                torch.cuda.empty_cache()
                print(version)

    return metrics_ai, bpp_ai


if __name__ == "__main__":
    renderer = get_renderers()
    metric = "y"
    distance = 1000
    elevation = [0]
    azimuth = [30, 90, 150, 210, 270, 330]

    for filedir_input in filedirs_input:
        input_coords, input_feats = read_ply_ascii_mpeg(filedir_input)
        input_feats_rgb = input_feats / 255.0
        input_feats_yuv = rgb2yuv(input_feats / 255.0)

        centroid = np.round(np.mean(input_coords, axis=0))
        input_coords = input_coords - centroid
        input_pc = collate_pointcloud_fn([(input_coords, input_feats_yuv)])

        input_pc_yuv = Pointclouds(
            points=[torch.tensor(input_coords.astype(np.float32), device=device)],
            features=[torch.tensor(input_feats_yuv.astype(np.float32), device=device)],
        )
        input_pc_rgb = Pointclouds(
            points=[torch.tensor(input_coords.astype(np.float32), device=device)],
            features=[torch.tensor(input_feats_rgb.astype(np.float32), device=device)],
        )

        input_imgs_yuv, _ = render_imgs(renderer, input_pc_yuv, distance, elevation, azimuth)
        input_imgs_rgb, _ = render_imgs(renderer, input_pc_rgb, distance, elevation, azimuth)
        input_imgs = input_imgs_rgb if metric == "ms-ssim" else input_imgs_yuv

        file_name = os.path.splitext(os.path.basename(filedir_input))[0]
        out_dir = os.path.join("imgs_show", "compare", file_name)
        os.makedirs(out_dir, exist_ok=True)
        ref_img_rgb = np.clip(input_imgs_rgb[0, ..., :3].detach().cpu().numpy(), 0.0, 1.0)
        matplotlib.image.imsave(os.path.join(out_dir, f"ref_img_{file_name}.png"), ref_img_rgb)

        metric_vals, bpp_vals = ai_performance_hyper(
            filedir_inputpc=filedir_input,
            file_name=file_name,
            centroid=centroid,
            input_imgs=input_imgs,
            renderer=renderer,
            input_pc=input_pc,
            model=model_hyper2_comai_sptrans_mse111,
            ckpts=ckpts_hyper2_comai_sptrans_imgmse111,
            metric_choice=metric,
            version="hyper2_comai_sptrans_imgmse111",
            distance=distance,
            elevation=elevation,
            azimuth=azimuth,
        )

        print("finish one file:", file_name)
        print("bpp:", bpp_vals)
        print("metric:", metric_vals)
