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
 
#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

#include "kernel.h"

namespace bevfusion {
namespace head {
namespace object {

const unsigned int MAX_DET_NUM = 1000;           // nms_pre_max_size = 1000;
const unsigned int DET_CHANNEL = 11;
const unsigned int NUM_TASKS = 6;

struct CenterHeadParameter
{
  std::string model;
  unsigned int task_num_stride[NUM_TASKS] = { 0, 1, 3, 5, 6, 8, };
  static const unsigned int num_classes = 10;
  char *class_name[num_classes] = {"car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"};

  float out_size_factor = 8;
  float voxel_size[2] = { 0.075, 0.075, };
  float pc_range[2] = { -54, -54, };
  float score_threshold = 0.1;
  float post_center_range[6] = { -61.2, -61.2, -10.0, 61.2, 61.2, 10.0, };
  float nms_iou_threshold = 0.2;
  unsigned int nms_pre_max_size = MAX_DET_NUM;
  unsigned int nms_post_max_size= 83;

  float min_x_range = -54;
  float max_x_range = 54;
  float min_y_range = -54;
  float max_y_range = 54;
  float min_z_range = -5.0;
  float max_z_range = 3.0;
  // the size of a pillar
  float pillar_x_size = 0.075;
  float pillar_y_size = 0.075;
  float pillar_z_size = 0.2;
  int max_points_per_voxel = 10;

  unsigned int max_voxels = 160000;
  unsigned int feature_num = 5;
};

/*
box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
*/
struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float vx;
    float vy;
    float rt;
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float vx_, float vy_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), vx(vx_), vy(vy_), rt(rt_), id(id_), score(score_) {}
};

class PostProcessCuda {
  private:
    CenterHeadParameter params_;
    float* d_post_center_range_ = nullptr;
    float* d_voxel_size_ = nullptr;
    float* d_pc_range_ = nullptr;

  public:
    PostProcessCuda();
    ~PostProcessCuda();

    int doPostDecodeCuda(
      int N,
      int H,
      int W,
      int C_reg,
      int C_height,
      int C_dim,
      int C_rot,
      int C_vel,
      int C_hm,
      const half *reg,
      const half *height,
      const half *dim,
      const half *rot,
      const half *vel,
      const half *hm,
      unsigned int *detection_num,
      float *detections, cudaStream_t stream);

    int doPostNMSCuda(
      unsigned int boxes_num,
      float *boxes_sorted,
      uint64_t* mask, cudaStream_t stream);

    int doPermuteCuda(
        unsigned int boxes_num, 
        const float *boxes_sorted, 
        float * permute_boxes, cudaStream_t stream);
    
    void setParams(CenterHeadParameter params);
};

};  // namespace object
};  // namespace head
};  // namespace bevfusion

#endif  // POSTPROCESS_H_