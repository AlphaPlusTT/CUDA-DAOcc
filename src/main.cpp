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

#include <cuda_runtime.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include <stb_image_write.h>

#include <fstream> // 用于文件输入输出
#include <iostream>

#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"
#include "bevfusion/custom-op.hpp"

// void loadPluginLibrary(const std::string& plugin_path) {
//     void* handle = dlopen(plugin_path.c_str(), RTLD_NOW);
//     if (!handle) {
//         std::cerr << "Failed to load plugin library: " << dlerror() << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                              "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    char path[200];
    sprintf(path, "%s/%s", root.c_str(), file_names[i]);

    int width, height, channels;
    images.push_back(stbi_load(path, &width, &height, &channels, 0));
    printf("Image info[%d]: %d x %d : %d\n", i, width, height, channels);
  }
  return images;
}

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}

// static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
//                       const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path,
//                       cudaStream_t stream) {
//   std::vector<nv::Prediction> predictions(bboxes.size());
//   memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

//   int padding = 300;
//   int lidar_size = 1024;
//   int content_width = lidar_size + padding * 3;
//   int content_height = 1080;
//   nv::SceneArtistParameter scene_artist_param;
//   scene_artist_param.width = content_width;
//   scene_artist_param.height = content_height;
//   scene_artist_param.stride = scene_artist_param.width * 3;

//   nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
//   scene_device_image.memset(0x00, stream);

//   scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
//   auto scene = nv::create_scene_artist(scene_artist_param);

//   nv::BEVArtistParameter bev_artist_param;
//   bev_artist_param.image_width = content_width;
//   bev_artist_param.image_height = content_height;
//   bev_artist_param.rotate_x = 70.0f;
//   bev_artist_param.norm_size = lidar_size * 0.5f;
//   bev_artist_param.cx = content_width * 0.5f;
//   bev_artist_param.cy = content_height * 0.5f;
//   bev_artist_param.image_stride = scene_artist_param.stride;

//   auto points = lidar_points.to_device();
//   auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
//   bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
//   bev_visualizer->draw_prediction(predictions, false);
//   bev_visualizer->draw_ego();
//   bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

//   nv::ImageArtistParameter image_artist_param;
//   image_artist_param.num_camera = images.size();
//   image_artist_param.image_width = 1600;
//   image_artist_param.image_height = 900;
//   image_artist_param.image_stride = image_artist_param.image_width * 3;
//   image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
//   memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
//          sizeof(float) * image_artist_param.viewport_nx4x4.size());

//   int gap = 0;
//   int camera_width = 500;
//   int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
//   int offset_cameras[][3] = {
//       {-camera_width / 2, -content_height / 2 + gap, 0},
//       {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
//       {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
//       {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
//       {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
//       {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

//   auto visualizer = nv::create_image_artist(image_artist_param);
//   for (size_t icamera = 0; icamera < images.size(); ++icamera) {
//     int ox = offset_cameras[icamera][0] + content_width / 2;
//     int oy = offset_cameras[icamera][1] + content_height / 2;
//     bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
//     visualizer->draw_prediction(icamera, predictions, xflip);

//     nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
//     device_image.copy_from_host(images[icamera], stream);

//     if (xflip) {
//       auto clone = device_image.clone(stream);
//       scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
//                    device_image.size(1) * 3, stream);
//       checkRuntime(cudaStreamSynchronize(stream));
//     }
//     visualizer->apply(device_image.ptr<unsigned char>(), stream);

//     scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
//                      device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
//     checkRuntime(cudaStreamSynchronize(stream));
//   }

//   printf("Save to %s\n", save_path.c_str());
//   stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
//                  scene_device_image.to_host(stream).ptr(), 100);
// }

std::shared_ptr<bevfusion::Core> create_core(const std::string& model, const std::string& precision) {
  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.44f;
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
  scn.model = nv::format("model/%s/lidar.backbone.xyz.onnx", model.c_str());
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
  // transbbox.model = nv::format("model/%s/build/head.bbox.plan", model.c_str());

  // // if you got an inaccurate boundingbox result please turn on the layernormplugin plan.
  // // transbbox.model = nv::format("model/%s/build/head.bbox.layernormplugin.plan", model.c_str());
  // transbbox.confidence_threshold = 0.12f;
  // transbbox.sorted_bboxes = true;

  bevfusion::CoreParameter param;
  param.camera_model = nv::format("model/%s/build/camera.backbone.plan", model.c_str());
  param.normalize = normalization;
  param.lidar_scn = scn;
  // param.geometry = geometry;
  param.transfusion = nv::format("model/%s/build/fuser.plan", model.c_str());
  // param.transbbox = transbbox;
  // param.camera_vtransform = nv::format("model/%s/build/camera.vtransform.plan", model.c_str());

  bevfusion::head::occ::OccParameter occ;
  occ.model = nv::format("model/%s/build/head.occ.plan", model.c_str());
  occ.x_res = 200;
  occ.y_res = 200;
  occ.z_res = 16;
  param.occ = occ;

  bevfusion::head::object::CenterHeadParameter object;
  object.model = nv::format("model/%s/build/head.object.plan", model.c_str());
  object.score_threshold = 0.45;
  param.object = object;

  return bevfusion::create_core(param);
}

void SaveBoxPred(std::vector<bevfusion::head::object::Bndbox> boxes, std::string file_name)
{
  std::ofstream ofs;
  ofs.open(file_name, std::ios::out);
  ofs.setf(std::ios::fixed, std::ios::floatfield);
  ofs.precision(5);
  if (ofs.is_open())
  {
    for (const auto box : boxes)
    {
      ofs << box.x << " ";
      ofs << box.y << " ";
      ofs << box.z << " ";
      ofs << box.w << " ";
      ofs << box.l << " ";
      ofs << box.h << " ";
      ofs << box.vx << " ";
      ofs << box.vy << " ";
      ofs << box.rt << " ";
      ofs << box.id << " ";
      ofs << box.score << " ";
      ofs << "\n";
    }
  }
  else
  {
    std::cerr << "Output file cannot be opened!" << std::endl;
  }
  ofs.close();
  std::cout << "Saved det prediction in: " << file_name << std::endl;
  return;
}

void SaveOccPred(std::vector<int> occ_data, std::string file_name)
{
  std::ofstream ofs;
  ofs.open(file_name, std::ios::out);
  if (ofs.is_open())
  {
    for (size_t i = 0; i < occ_data.size(); ++i)
    {
      ofs << occ_data[i];
      if (i < occ_data.size() - 1)
      {
        ofs << ","; // 使用逗号作为分隔符
      }
    }
    ofs << std::endl;
  }
  else
  {
    std::cerr << "Output file cannot be opened!" << std::endl;
  }
  ofs.close();
  std::cout << "Saved occ prediction in: " << file_name << std::endl;
  return;
}

int main(int argc, char **argv)
{
  const char *data = "example-data";
  const char *model = "resnet50int8";
  const char *precision = "int8";

  if (argc > 1)
    data = argv[1];
  if (argc > 2)
    model = argv[2];
  if (argc > 3)
    precision = argv[3];
  // dlopen("libcustom_layernorm.so", RTLD_NOW);
  // 显式加载插件库
  // loadPluginLibrary("/home/cf206/project/Lidar_AI_Solution/amirstan_plugin/build/lib/libamirstan_plugin.so");

  auto core = create_core(model, precision);
  if (core == nullptr)
  {
    printf("Core has been failed.\n");
    return -1;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  core->print();
  core->set_timer(true);

  // Load matrix to host
  // auto camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", data));
  // auto camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", data));
  // auto lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", data), false);
  // auto img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix_daocc.tensor", data));
  // core->update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(), img_aug_matrix.ptr<float>(),
  //             stream);
  // core->free_excess_memory();
  // camera2lidar = camera2lidar.to_half(stream);
  // camera_intrinsics = camera_intrinsics.to_half(stream);
  // img_aug_matrix = img_aug_matrix.to_half(stream);

  // Load image and lidar to host
  auto images = load_images(data);
  // auto lidar_points = nv::Tensor::load(nv::format("%s/points.tensor", data), false);

  auto lidar_points = nv::Tensor::loadbinary(nv::format("%s/sweeps_n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281439800013.pcd.bin", data), std::vector<int64_t>{-1, 5}, nv::DataType::Float32);
  // auto lidar_points = nv::Tensor::loadbinary(nv::format("%s/dataflow_0.bin", data), std::vector<int64_t>{-1, 5}, nv::DataType::Float32);
  // lidar_points = lidar_points.to_half(stream);
  // bevfusion::custom::set_last_feature_to_zero(lidar_points, stream);
  lidar_points = lidar_points.to_half(stream);

  // auto img_ref_points = nv::Tensor::load(nv::format("%s/img_ref_points.tensor", data));
  auto img_ref_points = nv::Tensor::loadbinary(nv::format("%s/img_ref_points.bin", data), std::vector<int64_t>{1, -1, 1, 1, 3}, nv::DataType::Float32);
  img_ref_points = img_ref_points.to_half(stream);

  // warmup
  for (int i = 0; i < 3; ++i)
  {
    core->forward((const unsigned char**)images.data(), img_ref_points.ptr<nvtype::half>(), lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
  }
  std::cout << "[CUSTOM] warm up end." << std::endl;

  for (int i = 0; i < 10; ++i)
  {
    core->forward((const unsigned char**)images.data(), img_ref_points.ptr<nvtype::half>(), lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
  }

  // evaluate inference time
  // std::vector<int> occ_pred = core->forward(lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
  // core->forward(lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);

  // std::ofstream outFile("occ_pred.txt");
  // if (outFile.is_open()) {
  //     for (size_t i = 0; i < core->occ_output_.size(); ++i) {
  //         outFile << core->occ_output_[i];
  //         if (i < core->occ_output_.size() - 1) {
  //             outFile << ","; // 使用逗号作为分隔符
  //         }
  //     }
  //     outFile << std::endl;
  //     outFile.close();
  // } else {
  //     std::cerr << "文件打开失败";
  // }

  std::string save_file_name = "object_pred.txt";
  SaveBoxPred(core->object_output_, save_file_name);

  save_file_name = "occ_pred.txt";
  SaveOccPred(core->occ_output_, save_file_name);

  // evaluate inference time
  // for (int i = 0; i < 5; ++i) {
  //   core->forward(lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
  // }

  // visualize and save to jpg
  // visualize(bboxes, lidar_points, images, lidar2image, "build/cuda-bevfusion.jpg", stream);

  // destroy memory
  free_images(images);
  checkRuntime(cudaStreamDestroy(stream));

  // printf("[Warning]: If you got an inaccurate boundingbox result please turn on the layernormplugin plan. (main.cpp:207)\n");
  return 0;
}