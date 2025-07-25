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

#ifndef __HEAD_OBJECT_HPP__
#define __HEAD_OBJECT_HPP__

#include <memory>
#include <string>
#include <vector>
#include <cuda_fp16.h>

#include "common/dtype.hpp"
#include "postprocess.h"

namespace bevfusion {
namespace head {
namespace object {

typedef struct float11 { float val[11]; } float11;

// struct CenterHeadParameter {
//   std::string model;
//   float out_size_factor = 8;
//   nvtype::Float2 voxel_size{0.075, 0.075};
//   nvtype::Float2 pc_range{-54.0f, -54.0f};
//   nvtype::Float3 post_center_range_start{-61.2, -61.2, -10.0};
//   nvtype::Float3 post_center_range_end{61.2, 61.2, 10.0};
//   float confidence_threshold = 0.0f;
//   bool sorted_bboxes = true;
// };

// class CenterHeadParameter
// {
//   public:
//     std::string model;
//     const unsigned int task_num_stride[NUM_TASKS] = { 0, 1, 3, 5, 6, 8, };
//     static const unsigned int num_classes = 10;
//     const char *class_name[num_classes] = { "car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"};

//     const float out_size_factor = 8;
//     const float voxel_size[2] = { 0.075, 0.075, };
//     const float pc_range[2] = { -54, -54, };
//     const float score_threshold = 0.1;
//     const float post_center_range[6] = { -61.2, -61.2, -10.0, 61.2, 61.2, 10.0, };
//     const float nms_iou_threshold = 0.2;
//     const unsigned int nms_pre_max_size = MAX_DET_NUM;
//     const unsigned int nms_post_max_size= 83;

//     const float min_x_range = -54;
//     const float max_x_range = 54;
//     const float min_y_range = -54;
//     const float max_y_range = 54;
//     const float min_z_range = -5.0;
//     const float max_z_range = 3.0;
//     // the size of a pillar
//     const float pillar_x_size = 0.075;
//     const float pillar_y_size = 0.075;
//     const float pillar_z_size = 0.2;
//     const int max_points_per_voxel = 10;

//     const unsigned int max_voxels = 160000;
//     const unsigned int feature_num = 5;

//     CenterHeadParameter() {};

//     int getGridXSize() {
//       return (int)std::round((max_x_range - min_x_range) / pillar_x_size);
//     }
//     int getGridYSize() {
//       return (int)std::round((max_y_range - min_y_range) / pillar_y_size);
//     }
//     int getGridZSize() {
//       return (int)std::round((max_z_range - min_z_range) / pillar_z_size);
//     }
// };

struct Position {
  float x, y, z;
};

struct Size {
  float w, l, h;  // x, y, z
};

struct Velocity {
  float vx, vy;
};

// struct BoundingBox {
//   Position position;
//   Size size;
//   Velocity velocity;
//   float z_rotation;
//   float score;
//   int id;
// };

class CenterHead {
 public:
  virtual ~CenterHead() = default;
  virtual std::vector<Bndbox> forward(const nvtype::half* transfusion_feature, void* stream, bool sorted_by_conf = false) = 0;
  virtual void print() = 0;
};

std::shared_ptr<CenterHead> create_centerhead(const CenterHeadParameter& param);

};  // namespace object
};  // namespace head
};  // namespace bevfusion

#endif  // __HEAD_OBJEC_HPP__