# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Beijing Mechanical Equipment Institute. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys

import warnings

warnings.filterwarnings("ignore")

import argparse
import os

import onnx
import torch
from onnxsim import simplify
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

from torch import nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import lean.quantize as quantize
import torch.nn.functional as F
import einops


def trilinear_grid_sample(im, grid, align_corners=False):
    """
    Imitate F.grid_sample for 5D inputs (N, C, D, H, W) with 3D grid (N, Dg, Hg, Wg, 3).

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, D, H, W)
        grid (torch.Tensor): Sampling grid, shape (N, Dg, Hg, Wg, 3), values in [-1,1]
        align_corners (bool): coordinate mapping mode

    Returns:
        torch.Tensor: Sampled output, shape (N, C, Dg, Hg, Wg)
    """
    N, C, D, H, W = im.shape
    _, Dg, Hg, Wg, _ = grid.shape
    # extract normalized coords
    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2]

    if align_corners:
        x = ((x + 1) / 2) * (W - 1)
        y = ((y + 1) / 2) * (H - 1)
        z = ((z + 1) / 2) * (D - 1)
    else:
        x = ((x + 1) * W - 1) / 2
        y = ((y + 1) * H - 1) / 2
        z = ((z + 1) * D - 1) / 2

    # flatten grid
    x = x.view(N, -1)
    y = y.view(N, -1)
    z = z.view(N, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    z0 = torch.floor(z).long()
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # distances
    xd = (x - x0.float())
    yd = (y - y0.float())
    zd = (z - z0.float())

    # corner weights
    w000 = ((1 - xd) * (1 - yd) * (1 - zd)).unsqueeze(1)
    w001 = ((1 - xd) * (1 - yd) * zd).unsqueeze(1)
    w010 = ((1 - xd) * yd * (1 - zd)).unsqueeze(1)
    w011 = ((1 - xd) * yd * zd).unsqueeze(1)
    w100 = (xd * (1 - yd) * (1 - zd)).unsqueeze(1)
    w101 = (xd * (1 - yd) * zd).unsqueeze(1)
    w110 = (xd * yd * (1 - zd)).unsqueeze(1)
    w111 = (xd * yd * zd).unsqueeze(1)

    # pad im with zeros
    im_padded = F.pad(im, pad=[1, 1, 1, 1, 1, 1], mode='constant', value=0)
    Dp, Hp, Wp = D + 2, H + 2, W + 2

    # shift indices after padding
    x0p, x1p = x0 + 1, x1 + 1
    y0p, y1p = y0 + 1, y1 + 1
    z0p, z1p = z0 + 1, z1 + 1

    # clip indices
    x0p = x0p.clamp(0, Wp - 1)
    x1p = x1p.clamp(0, Wp - 1)
    y0p = y0p.clamp(0, Hp - 1)
    y1p = y1p.clamp(0, Hp - 1)
    z0p = z0p.clamp(0, Dp - 1)
    z1p = z1p.clamp(0, Dp - 1)

    # flatten im_padded spatial dims
    im_flat = im_padded.view(N, C, -1)

    # compute linear indices for corners
    def lin_idx(xi, yi, zi):
        return (zi * Hp * Wp + yi * Wp + xi).unsqueeze(1).expand(-1, C, -1)

    idx000 = lin_idx(x0p, y0p, z0p)
    idx001 = lin_idx(x0p, y0p, z1p)
    idx010 = lin_idx(x0p, y1p, z0p)
    idx011 = lin_idx(x0p, y1p, z1p)
    idx100 = lin_idx(x1p, y0p, z0p)
    idx101 = lin_idx(x1p, y0p, z1p)
    idx110 = lin_idx(x1p, y1p, z0p)
    idx111 = lin_idx(x1p, y1p, z1p)

    Ia = torch.gather(im_flat, 2, idx000)
    Ib = torch.gather(im_flat, 2, idx001)
    Ic = torch.gather(im_flat, 2, idx010)
    Id = torch.gather(im_flat, 2, idx011)
    Ie = torch.gather(im_flat, 2, idx100)
    If = torch.gather(im_flat, 2, idx101)
    Ig = torch.gather(im_flat, 2, idx110)
    Ih = torch.gather(im_flat, 2, idx111)

    out = (Ia * w000 + Ib * w001 + Ic * w010 + Id * w011 +
           Ie * w100 + If * w101 + Ig * w110 + Ih * w111)

    return out.view(N, C, Dg, Hg, Wg)


class SubclassCameraModule(nn.Module):
    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.model = model

    def forward(self, x, img_ref_points):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.model.encoders.camera.backbone(x)
        x = self.model.encoders.camera.neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        B = x.shape[0]
        if self.model.encoders.camera.vtransform.transfer_conv is not None:
            x = einops.rearrange(x, 'B N C H W -> (B N) C H W')
            x = self.model.encoders.camera.vtransform.transfer_conv(x)
            x = einops.rearrange(x, '(B N) C H W -> B C N H W', B=B)
        else:
            x = einops.rearrange(x, 'B N C H W -> B C N H W')

        sampling_feats = trilinear_grid_sample(x, img_ref_points, align_corners=True)
        sampling_feats = einops.rearrange(sampling_feats,
                                          'b c num_points 1 1 -> b num_points c',
                                          b=B)

        feats_volume = einops.rearrange(sampling_feats,
                                        'bs (z h w) c -> bs (z c) w h',
                                        z=self.model.encoders.camera.vtransform.volume_size[2],
                                        h=self.model.encoders.camera.vtransform.volume_size[0],
                                        w=self.model.encoders.camera.vtransform.volume_size[1])
        if self.model.encoders.camera.vtransform.down_sample is not None:
            feats_volume = self.model.encoders.camera.vtransform.down_sample(feats_volume)
        return feats_volume


def parse_args():
    parser = argparse.ArgumentParser(description="Export DAOcc model")
    parser.add_argument('--ckpt', type=str, default='model/daocc_ptq.pth')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = torch.load(args.ckpt).module
    save_root = f"model/resnet50-int8"
    if args.fp16:
        save_root = f"model/resnet50"
        quantize.disable_quantization(model).apply()
    os.makedirs(save_root, exist_ok=True)

    data = torch.load("example-data/example-data.pth")
    img = data["img"].data[0].cuda()
    img_ref_points = torch.zeros([1, 324000, 1, 1, 3]).cuda()

    camera_model = SubclassCameraModule(model)
    camera_model.cuda().eval()

    with torch.no_grad():
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"
        TensorQuantizer.use_fb_fake_quant = True
        torch.onnx.export(
            camera_model,
            (img, img_ref_points),
            camera_backbone_onnx,
            input_names=["img", "img_ref_points"],
            output_names=["camera_feature"],
            opset_version=13,
            do_constant_folding=True,
            # verbose=True,
        )
        onnx_orig = onnx.load(camera_backbone_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, camera_backbone_onnx)
        print(f"🚀 The export is completed. ONNX save as {camera_backbone_onnx} 🤗, Have a nice day~")


if __name__ == "__main__":
    main()
