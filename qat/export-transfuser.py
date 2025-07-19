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

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner.fp16_utils import auto_fp16
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmcv.runner import wrap_fp16_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import lean.quantize as quantize
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import einops
# from mmcv.ops.point_sample import bilinear_grid_sample
import numpy as np


class SubclassHeadObject(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        assert (len(self.parent.heads['object'].task_heads) == 6)

    @staticmethod
    def object_forward(self, x):
        """Forward function for CenterPoint.
        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].
        Returns:
            list[dict]: Output results for tasks.
        """
        x = self.shared_conv(x)

        pred = [task(x) for task in self.task_heads]

        return pred[0]['reg'], pred[0]['height'], pred[0]['dim'], pred[0]['rot'], pred[0]['vel'], pred[0]['heatmap'], \
        pred[1]['reg'], pred[1]['height'], pred[1]['dim'], pred[1]['rot'], pred[1]['vel'], pred[1]['heatmap'], \
        pred[2]['reg'], pred[2]['height'], pred[2]['dim'], pred[2]['rot'], pred[2]['vel'], pred[2]['heatmap'], \
        pred[3]['reg'], pred[3]['height'], pred[3]['dim'], pred[3]['rot'], pred[3]['vel'], pred[3]['heatmap'], \
        pred[4]['reg'], pred[4]['height'], pred[4]['dim'], pred[4]['rot'], pred[4]['vel'], pred[4]['heatmap'], \
        pred[5]['reg'], pred[5]['height'], pred[5]['dim'], pred[5]['rot'], pred[5]['vel'], pred[5]['heatmap']

    def forward(self, x):
        for name, head in self.parent.heads.items():
            if name == "object":
                return self.object_forward(head, x)
            elif name == "occ":
                continue
            else:
                raise ValueError(f"unsupported head: {name}")


def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0, device=x0.device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1, device=x0.device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0, device=x1.device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1, device=x1.device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0, device=y0.device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1, device=y0.device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0, device=y1.device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1, device=y1.device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


class SubclassHeadOcc(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.ref_lidar = torch.from_numpy(np.load('qat/ref_lidar.npy'))

    @staticmethod
    def occ_forward(self, inputs, ref_lidar):
        x = einops.rearrange(inputs, 'bs c w h -> bs c h w')
        B = x.shape[0]

        x = self.coordinate_transform.transfer_conv(x)
        ref_lidar = ref_lidar.cuda()
        lidar_x_length, lidar_y_length = self.coordinate_transform.lidar_x_max-self.coordinate_transform.lidar_x_min, self.coordinate_transform.lidar_y_max-self.coordinate_transform.lidar_y_min
        ref_lidar[..., 0] = (ref_lidar[..., 0] - self.coordinate_transform.lidar_x_min) / lidar_x_length
        ref_lidar[..., 1] = (ref_lidar[..., 1] - self.coordinate_transform.lidar_y_min) / lidar_y_length
        ref_lidar = ref_lidar * 2 - 1

        x = bilinear_grid_sample(x, ref_lidar)
        x = einops.rearrange(x.squeeze(-1), 'bs c (h w) -> bs c h w', h=self.coordinate_transform.output_h, w=self.coordinate_transform.output_w)

        occ_pred = self.final_conv(x).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        occ_pred = self.predicter(occ_pred)
        occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        return occ_pred

    @staticmethod
    def get_occ(occ_pred):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1).int()
        return occ_res

    def forward(self, x):
        for name, head in self.parent.heads.items():
            if name == "occ":
                occ_pred = self.occ_forward(head, x, self.ref_lidar)
                return self.get_occ(occ_pred)
            elif name == "object":
                continue
            else:
                raise ValueError(f"unsupported head: {name}")


class SubclassFuser(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @auto_fp16(apply_to=("features",))
    def forward(self, features):
        if self.parent.fuser is not None:
            x = self.parent.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.parent.decoder["backbone"](x)
        x = self.parent.decoder["neck"](x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export transfusion to onnx file")
    parser.add_argument("--ckpt", type=str, default="model/daocc_ptq.pth", help="Pretrain model")
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    model = torch.load(args.ckpt).module
    
    save_root = f"model/resnet50-int8"
    if args.fp16:
        save_root = f"model/resnet50"
        quantize.disable_quantization(model).apply()
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    fuser = SubclassFuser(model).cuda()

    headocc = SubclassHeadOcc(model).cuda()
    headobject = SubclassHeadObject(model).cuda()

    TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        camera_features = torch.randn(1, 1280, 180, 180).cuda()
        lidar_features = torch.randn(1, 256, 180, 180).cuda()

        fuser_onnx_path = f"{save_root}/fuser.onnx"
        torch.onnx.export(fuser, [camera_features, lidar_features], fuser_onnx_path, opset_version=13,
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"🚀 The export is completed. ONNX save as {fuser_onnx_path} 🤗, Have a nice day~")

        occhead_onnx_path = f"{save_root}/head.occ.onnx"
        head_input = torch.randn(1, 512, 180, 180).cuda()
        torch.onnx.export(headocc, head_input, occhead_onnx_path, opset_version=13,
            input_names=["middle"],
            output_names=["occ"],
        )
        print(f"🚀 The export is completed. ONNX save as {occhead_onnx_path} 🤗, Have a nice day~")

        objecthead_onnx_path = f"{save_root}/head.object.onnx"
        head_input = torch.randn(1, 512, 180, 180).cuda()
        torch.onnx.export(headobject, head_input, objecthead_onnx_path, opset_version=13,
            input_names=["middle"],
            output_names=['reg_0', 'height_0', 'dim_0', 'rot_0', 'vel_0', 'hm_0',
                          'reg_1', 'height_1', 'dim_1', 'rot_1', 'vel_1', 'hm_1',
                          'reg_2', 'height_2', 'dim_2', 'rot_2', 'vel_2', 'hm_2',
                          'reg_3', 'height_3', 'dim_3', 'rot_3', 'vel_3', 'hm_3',
                          'reg_4', 'height_4', 'dim_4', 'rot_4', 'vel_4', 'hm_4',
                          'reg_5', 'height_5', 'dim_5', 'rot_5', 'vel_5', 'hm_5'],
        )
        print(f"🚀 The export is completed. ONNX save as {objecthead_onnx_path} 🤗, Have a nice day~")
