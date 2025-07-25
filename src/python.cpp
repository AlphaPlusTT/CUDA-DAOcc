/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 # SPDX-FileCopyrightText: Copyright (c) 2025 Beijing Mechanical Equipment Institute. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>

#include "bevfusion/bevfusion.hpp"
#include "common/tensor.hpp"
#include "common/tensorrt.hpp"
#include "common/timer.hpp"
#include <dlpack/dlpack.h>
#include <cassert>
#include <iostream>

using namespace std;
namespace py = pybind11;

// void loadPluginLibrary(const std::string& plugin_path) {
//     void* handle = dlopen(plugin_path.c_str(), RTLD_NOW);
//     if (!handle) {
//         std::cerr << "Failed to load plugin library: " << dlerror() << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

nv::Tensor convert_to(py::array& array) {
  vector<size_t> tshape(array.shape(), array.shape() + array.ndim());
  vector<int64_t> shape(tshape.size());
  std::transform(tshape.begin(), tshape.end(), shape.begin(), [](size_t v) { return v; });

  nv::DataType nvdtype = nv::DataType::None;
  if (array.dtype() == py::dtype::of<float>())
    nvdtype = nv::DataType::Float32;
  else if (array.dtype() == py::dtype::of<unsigned char>())
    nvdtype = nv::DataType::UInt8;
  else if (array.dtype() == py::dtype("half"))
    nvdtype = nv::DataType::Float16;
  else if (array.dtype() == py::dtype::of<int>())
    nvdtype = nv::DataType::Int32;
  else {
    Assertf(false, "Unsupported data type: %s", std::string(py::str(array.dtype())).c_str());
  }
  return nv::Tensor::from_data_reference((void*)array.data(), shape, nvdtype, false);
}

class BEVFusion {
 public:
  std::shared_ptr<bevfusion::Core> core_;
  cudaStream_t stream_ = nullptr;

  static std::shared_ptr<BEVFusion> load_instance(string camera, string vtransform, string lidar, string fuser, string headbbox, string headocc, string precision) {
    std::shared_ptr<BEVFusion> instance(new BEVFusion());
    if (!instance->load(camera, vtransform, lidar, fuser, headbbox, headocc, precision)) {
      instance.reset();
    }
    return instance;
  }

  virtual ~BEVFusion() {
    if (stream_) checkRuntime(cudaStreamDestroy(stream_));
  }

  bool load(string camera, string vtransform, string lidar, string fuser, string headbbox, string headocc, string precision) {
    bevfusion::camera::NormalizationParameter normalization;
    normalization.image_width = 1600;
    normalization.image_height = 900;
    normalization.output_width = 704;
    normalization.output_height = 256;
    normalization.num_camera = 6;
    normalization.resize_lim = 0.48f;
    normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

    bevfusion::lidar::VoxelizationParameter voxelization;
    voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
    voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
    voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
    voxelization.grid_size =
        voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
    voxelization.max_points_per_voxel = 10;
    voxelization.max_points = 300000;
    voxelization.max_voxels = 160000;
    voxelization.num_feature = 5;

    bevfusion::lidar::SCNParameter scn;
    scn.voxelization = voxelization;
    scn.model = lidar;
    scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

    if (precision == "int8") {
      scn.precision = bevfusion::lidar::Precision::Int8;
    } else {
      scn.precision = bevfusion::lidar::Precision::Float16;
    }

    // bevfusion::camera::GeometryParameter geometry;
    // geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    // geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    // geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
    // geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
    // geometry.image_width = 704;
    // geometry.image_height = 256;
    // geometry.feat_width = 88;
    // geometry.feat_height = 32;
    // geometry.num_camera = 6;
    // geometry.geometry_dim = nvtype::Int3(360, 360, 80);

    // bevfusion::head::transbbox::TransBBoxParameter transbbox;
    // transbbox.out_size_factor = 8;
    // transbbox.pc_range = {-54.0f, -54.0f};
    // transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
    // transbbox.post_center_range_end = {61.2, 61.2, 10.0};
    // transbbox.voxel_size = {0.075, 0.075};
    // transbbox.model = headbbox;
    // transbbox.confidence_threshold = 0.0f;
    // transbbox.sorted_bboxes = true;

    bevfusion::CoreParameter param;
    param.camera_model = camera;
    param.normalize = normalization;
    param.lidar_scn = scn;
    // param.geometry = geometry;
    param.transfusion = fuser;
    // param.transbbox = transbbox;
    // param.camera_vtransform = vtransform;

    bevfusion::head::occ::OccParameter occ;
    occ.model = headocc;
    occ.x_res = 200;
    occ.y_res = 200;
    occ.z_res = 16;
    param.occ = occ;

    bevfusion::head::object::CenterHeadParameter object;
    object.model = headbbox;
    object.score_threshold = 0.1;
    param.object = object;

    core_ = bevfusion::create_core(param);
    if (core_ == nullptr) return false;

    checkRuntime(cudaStreamCreate(&stream_));
    return true;
  }

  void print() { core_->print(); }

  // void update(py::array camera2lidar, py::array camera_intrinsics, py::array lidar2image, py::array img_aug_matrix) {
  //   auto t_lidar2image = convert_to(lidar2image);
  //   auto t_img_aug_matrix = convert_to(img_aug_matrix);
  //   auto t_camera2lidar = convert_to(camera2lidar);
  //   auto t_camera_intrinsics = convert_to(camera_intrinsics);
  //   core_->update(t_camera2lidar.ptr<float>(), t_camera_intrinsics.ptr<float>(), t_lidar2image.ptr<float>(),
  //                 t_img_aug_matrix.ptr<float>(), stream_);
  // }

  // py::dict forward_with_normalization(py::array points) {
  //   auto t_points = convert_to(points);
  //   // auto t_images = convert_to(images);

  //   // int64_t volumn = std::accumulate(t_images.shape.begin() + 2, t_images.shape.end(), 1, std::multiplies<int64_t>());
  //   // std::vector<unsigned char*> image_pointers(t_images.size(1));
  //   // for (size_t i = 0; i < image_pointers.size(); ++i) image_pointers[i] = t_images.ptr<unsigned char>() + i * volumn;

  //   core_->forward(t_points.ptr<nvtype::half>(), t_points.size(0), stream_);

  //   nv::Tensor object_output(std::vector<int>{static_cast<int>(core_->object_output_.size()), 11}, nv::DataType::Float32, false);
  //   for (size_t i = 0; i < core_->object_output_.size(); ++i) {
  //     auto& box = core_->object_output_[i];
  //     float* row = object_output.ptr<float>() + object_output.size(1) * i;
  //     // memcpy(row + 0, &box.position, sizeof(box.position));
  //     // memcpy(row + 3, &box.size, sizeof(box.size));
  //     // row[6] = box.z_rotation;
  //     // memcpy(row + 7, &box.velocity, sizeof(box.velocity));
  //     // row[9] = box.id;
  //     // row[10] = box.score;
  //     row[0] = box.x;
  //     row[1] = box.y;
  //     row[2] = box.z;
  //     row[3] = box.w;
  //     row[4] = box.l;
  //     row[5] = box.h;
  //     row[6] = box.rt;
  //     row[7] = box.vx;
  //     row[8] = box.vy;
  //     row[9] = box.id;
  //     row[10] = box.score;
  //   }
  //   // return py::array(py::dtype("float32"), object_output.shape, object_output.ptr());
  //   py::array det_array = py::array(py::dtype("float32"), object_output.shape, object_output.ptr());
  //   py::array occ_array = py::array(py::dtype("int32"), {core_->occ_output_.size()}, core_->occ_output_.data());
  //   py::dict result;
  //   result["det_outputs"] = det_array;
  //   result["occ_outputs"] = occ_array;
  //   return result;
  // }

  py::dict forward_without_normalization(py::array images, py::array img_ref_points, py::array points) {
    auto t_points = convert_to(points);
    auto t_images = convert_to(images);
    t_images.to_device_();
    t_images = t_images.to_half();

    auto t_img_ref_points = convert_to(img_ref_points);
    t_img_ref_points.to_device_();
    t_img_ref_points = t_img_ref_points.to_half();

    core_->forward_no_normalize(t_images.ptr<nvtype::half>(), t_img_ref_points.ptr<nvtype::half>(), t_points.ptr<nvtype::half>(), t_points.size(0), stream_);

    nv::Tensor object_output(std::vector<int>{static_cast<int>(core_->object_output_.size()), 11}, nv::DataType::Float32, false);
    for (size_t i = 0; i < core_->object_output_.size(); ++i) {
      auto& box = core_->object_output_[i];
      float* row = object_output.ptr<float>() + object_output.size(1) * i;
      // memcpy(row + 0, &box.position, sizeof(box.position));
      // memcpy(row + 3, &box.size, sizeof(box.size));
      // row[6] = box.z_rotation;
      // memcpy(row + 7, &box.velocity, sizeof(box.velocity));
      // row[9] = box.id;
      // row[10] = box.score;
      row[0] = box.x;
      row[1] = box.y;
      row[2] = box.z;
      row[3] = box.w;
      row[4] = box.l;
      row[5] = box.h;
      row[6] = box.rt;
      row[7] = box.vx;
      row[8] = box.vy;
      row[9] = box.id;
      row[10] = box.score;
    }
    // return py::array(py::dtype("float32"), object_output.shape, object_output.ptr());
    py::array det_array = py::array(py::dtype("float32"), object_output.shape, object_output.ptr());
    py::array occ_array = py::array(py::dtype("int32"), {core_->occ_output_.size()}, core_->occ_output_.data());
    py::dict result;
    result["det_outputs"] = det_array;
    result["occ_outputs"] = occ_array;
    return result;
  }

  // py::array forward(py::object images, py::object points, bool with_normalization, bool with_dlpack){
  //   if(with_normalization){
  //     return this->forward_with_normalization(images, points);
  //   }else{
  //     if(with_dlpack){
  //       return this->forward_without_normalization_dlpack(images, points);
  //     }else{
  //       return this->forward_without_normalization(images, points);
  //     }
  //   }
  // }

  py::dict forward(py::object images, py::object img_ref_points, py::object points, bool with_normalization, bool with_dlpack){
    assert(!with_normalization);
    return this->forward_without_normalization(images, img_ref_points, points);
  }
};

PYBIND11_MODULE(libpybev, m) {
  py::class_<BEVFusion, shared_ptr<BEVFusion>>(m, "BEVFusion")
      .def("forward", &BEVFusion::forward, py::arg("images"), py::arg("img_ref_points"), py::arg("points"), py::arg("with_normalization")=true, py::arg("with_dlpack")=false)
      .def("print", &BEVFusion::print);

  m.def("load_bevfusion", BEVFusion::load_instance);
  // dlopen("libcustom_layernorm.so", RTLD_NOW);
  // 显式加载插件库
  // loadPluginLibrary("/home/cf206/project/Lidar_AI_Solution/amirstan_plugin/build/lib/libamirstan_plugin.so");
};