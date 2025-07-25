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

import argparse
import copy
import os
import warnings

import mmcv
import numpy as np
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from mmdet3d.models.utils import get_reference_points
from mmdet3d.models import BEVTransformV2


def single_cuda_test(data_loader):
    import os
    import sys
    import libpybev
    model = os.environ["DEBUG_MODEL"]
    precision = os.environ["DEBUG_PRECISION"]
    print(f"Model: {model}, Precision: {precision}")

    core = libpybev.load_bevfusion(
        f"model/{model}/build/camera.backbone.plan",
        f"model/{model}/build/camera.vtransform.plan",
        f"model/{model}/lidar.backbone.xyz.onnx",
        f"model/{model}/build/fuser.plan",
        f"model/{model}/build/head.object.plan",
        f"model/{model}/build/head.occ.plan",
        precision
    )

    vt = BEVTransformV2(x=[-54.0, 54.0], y=[-54.0, 54.0], z=[-5.0, 3.0], xs=180, ys=180, zs=10,
                        input_size=[256, 704], in_channels=256, out_channels=128, top_type='lidar')

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        images = data["img"].data[0].numpy()
        camera_intrinsics = data["camera_intrinsics"].data[0].data
        camera2lidar = data["camera2lidar"].data[0].data
        lidar2image = data["lidar2image"].data[0].data.numpy()
        img_aug_matrix = data["img_aug_matrix"].data[0].data
        lidar_aug_matrix = data["lidar_aug_matrix"].data[0].data
        points = data["points"].data[0][0].half().data.numpy()

        # input_size = [928, 1600] # input_size is [256, 704], not [928, 1600]!!!

        reference_points_img, volume_mask = vt.point_sampling(camera2lidar, camera_intrinsics[..., :3, :3],
                                                              img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
                                                              lidar_aug_matrix)
        bs, num_points, _ = reference_points_img.shape
        reference_points_img = reference_points_img * volume_mask
        reference_points_img = reference_points_img.view(bs, num_points, 1, 1, 3)
        reference_points_img = 2 * reference_points_img - 1

        volume_mask = ~volume_mask

        # Make sure any points not projected onto the camera plane have coordinates outside [-1, 1]
        volume_mask = volume_mask.view(bs, num_points, 1, 1, 1) * 10
        reference_points_img += volume_mask

        img_ref_points = reference_points_img.numpy()

        # core.update(
        #     camera2lidar,
        #     camera_intrinsics,
        #     lidar2image,
        #     img_aug_matrix
        # )
        outputs = core.forward(images, img_ref_points, points, with_normalization=False)

        boxes = outputs['det_outputs']
        boxes = torch.from_numpy(boxes)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 5] * 0.5
        bbs = data["metas"].data[0][0]["box_type_3d"](boxes[:, :-2], 9)
        # TODO: - h/2; vx, vy, rt -> rt, vx, vy; remove nms;
        labels = boxes[:, -2].int()
        scores = boxes[:, -1]

        occ = outputs['occ_outputs'].reshape([200, 200, 16])

        results.append({
            "boxes_3d": bbs,
            "scores_3d": scores,
            "labels_3d": labels,
            "occ_pred": occ,
        })
        prog_bar.update()

    print("Step1 Done.")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("--config",
                        default="configs/nuscenes/occ3d/daocc_occ3d_nus_wo_mask_v2.yaml",
                        help="test config file path")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
             "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
             "useful when you want to format the result to a specific format and "
             "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="bbox",
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
             "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file. If the value to "
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             "Note that the quotation marks are necessary and that no white space "
             "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
             "format will be kwargs for dataset.evaluate() function (deprecate), "
             "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
             "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )
    outputs = single_cuda_test(data_loader)

    if args.out:
        print(f"\nwriting results to {args.out}")
        mmcv.dump(outputs, args.out)

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
            "interval",
            "tmpdir",
            "start",
            "gpu_collect",
            "save_best",
            "rule",
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()
