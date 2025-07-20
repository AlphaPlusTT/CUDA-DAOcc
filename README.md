# CUDA-DAOcc

This repository contains sources and model for [DAOcc](https://github.com/AlphaPlusTT/DAOcc) inference using CUDA & TensorRT.

## 3D Object Detection and Occupancy prediction
- We use the [daocc_occ3d_wo_mask_v2.yaml](https://github.com/AlphaPlusTT/DAOcc/blob/master/configs/nuscenes/occ3d/daocc_occ3d_wo_mask_v2.yaml) configuration for TensorRT.

| **Model** | **Framework** | **Precision** | **mAP** | **NDS** | **mIoU** | **FPS** | **FPS(wo det)** |
|:---------:|:-------------:|:-------------:|:-------:|:-------:|:--------:|:-------:|:---------------:|
|   DAOcc   |    PyTorch    |     FP32      |  59.43  |  64.70  |  54.33   |   7.1   |       7.8       |
|   DAOcc   |   TensorRT    |     FP16      |  60.27  |  65.22  |  54.25   |  83.0   |      104.9      |
- Note: Inference speed was measured on an RTX 4090 GPU using 50 nuScenes validation samples (average result).
  - Since the number of lidar points is the main reason that affects the FPS. 
  - Please refer to the readme of [3DSparseConvolution](/libraries/3DSparseConvolution/README.md) for more details.

## Model and Data
- The model trained with [daocc_occ3d_wo_mask_v2.yaml](https://github.com/AlphaPlusTT/DAOcc/blob/master/configs/nuscenes/occ3d/daocc_occ3d_wo_mask_v2.yaml) can be downloaded from [Google Drive](https://drive.google.com/file/d/1C4pRy1eaHgulYPi2qFo8kJrqmBKhxeYo/view?usp=sharing).
- For quick practice, we provide example data from nuScenes. You can download it from [Google Drive](https://drive.google.com/file/d/158ChxGu1Q27ED3SmCBbw4REjbl4KNrWI/view?usp=sharing). It contains the following:
  1. Camera images from 6 directions.
  2. Projected fixed 3D voxel center points into the surround-view camera pixel coordinate system and normalized to the [-1, 1] range `img_ref_points.bin`.
  3. Single-frame point cloud binary file `*.bin`.
  4. 10-sweep aggregated point cloud binary file `sweeps_*.bin`.

## Prerequisites
To build bevfusion, we need to depend on the following libraries:
- CUDA >= 11.0
- CUDNN >= 8.2
- TensorRT >= 8.5.0
- libprotobuf-dev
- [Compute Capability](https://developer.nvidia.com/cuda-gpus#compute) >= sm_80
- Python >= 3.6

The data in the performance table was obtained using TensorRT-8.5.3.1, cuda-11.8 and cudnn8.9.7 statistics.

## GetStart
```
git clone --recursive https://github.com/AlphaPlusTT/CUDA-DAOcc
cd CUDA-DAOcc
```

### Export onnx and PTQ

#### 1. Configuring the DAOcc runtime environment

[Here](https://github.com/AlphaPlusTT/DAOcc/blob/master/docs/install.md) is the official configuration guide.
> For the final step in the official guide linked above, there's no need to clone the DAOcc codebase, as it already exists in this project path.

After following the above official installation instructions, install these additional dependencies:

```bash
apt install libprotobuf-dev
pip install nvidia-pyindex
pip install pytorch-quantization==2.1.3
pip install trimesh
pip install onnx
pip install onnxsim
pip install onnxruntime
```

#### 2. Download model and nuScenes-example-data.zip
- download pytorch model from [Google Drive](https://drive.google.com/file/d/1C4pRy1eaHgulYPi2qFo8kJrqmBKhxeYo/view?usp=sharing).
- download nuScenes-example-data.tar.gz from [Google Drive](https://drive.google.com/file/d/158ChxGu1Q27ED3SmCBbw4REjbl4KNrWI/view?usp=sharing).

```bash
# download models and datas to CUDA-DAOcc
# unzip datas
tar xzf nuScenes-example-data.tar.gz
# mkdir
mkdir model && mv epoch_6_ema.pth model/daocc.pth
```

#### 3. Generate PTQ model
- This code uses the [nuScenes Dataset](https://www.nuscenes.org/). You need to download it in order to run PTQ.
- You need follow the tips [here](https://github.com/AlphaPlusTT/DAOcc/blob/master/docs/data.md) to prepare the data.
```bash
ln -s DAOcc/data
python qat/ptq.py --config=DAOcc/configs/nuscenes/occ3d/daocc_occ3d_nus_wo_mask_v2.yaml --ckpt=model/daocc.pth --calibrate_batch 300
```

#### 3. Export INT8 model (not test)

```bash
python qat/export-camera.py --ckpt=model/daocc_ptq.pth
python qat/export-transfuser.py --ckpt=model/daocc_ptq.pth
python qat/export-scn.py --ckpt=model/daocc_ptq.pth --save=model/resnet50-int8/lidar.backbone.onnx
```

#### 4. Export FP16 model
```bash
python qat/export-camera.py --ckpt=model/daocc_ptq.pth --fp16
python qat/export-transfuser.py --ckpt=model/daocc_ptq.pth --fp16
python qat/export-scn.py --ckpt=model/daocc_ptq.pth --save=model/resnet50/lidar.backbone.onnx
```

### TensorRT Inference

#### 1. Configure the environment.sh
Modify the TensorRT/CUDA/CUDNN/DAOcc variable values in the tool/environment.sh file.
```bash
# change the path to the directory you are currently using
export TensorRT_Lib=/path/to/TensorRT/lib
export TensorRT_Inc=/path/to/TensorRT/include
export TensorRT_Bin=/path/to/TensorRT/bin

export CUDA_Lib=/path/to/cuda/lib64
export CUDA_Inc=/path/to/cuda/include
export CUDA_Bin=/path/to/cuda/bin
export CUDA_HOME=/path/to/cuda

export CUDNN_Lib=/path/to/cudnn/lib

# For CUDA-11.x:    SPCONV_CUDA_VERSION=11.4
# For CUDA-12.x:    SPCONV_CUDA_VERSION=12.6
export SPCONV_CUDA_VERSION=11.4

# resnet50/resnet50int8/swint
export DEBUG_MODEL=resnet50

# fp16/int8
export DEBUG_PRECISION=fp16
export DEBUG_DATA=example-data
export USE_Python=OFF
```

- Apply the environment to the current terminal.
```bash
. tool/environment.sh
```

#### 5. Compile and run

1. Building the models for tensorRT
```bash
bash tool/build_trt_engine.sh
```

2. Compile and run the program
```bash
# Generate the protobuf code
bash src/onnx/make_pb.sh

# Compile and run
bash tool/run.sh
```

### Evaluating mAP and mIoU with PyBEV
1. Modify `USE_Python=ON` in environment.sh to enable compilation of python.
2. Run `bash tool/run.sh` to build the libpybev.so.
3. Run following command to evaluate mAP and mIoU.
```bash
cp qat/test-mAP-for-cuda.py DAOcc/tools
cd DAOcc
ln -s ../build/libpybev.so
ln -s ../model
python tools/test-mAP-for-cuda.py
```

## Thanks
This project makes use of a number of awesome open source libraries, including:

- [Lidar_AI_Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution) for TensorRT inference.
- [stb_image](https://github.com/nothings/stb) for PNG and JPEG support
- [pybind11](https://github.com/pybind/pybind11) for seamless C++ / Python interop
- and others! See the dependencies folder.

Many thanks to the authors of these brilliant projects!
