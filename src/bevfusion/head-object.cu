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

#include <algorithm>
#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "head-object.hpp"
#include <iostream>

namespace bevfusion {
namespace head {
namespace object {

class CenterHeadImplement : public CenterHead {
 public:
  // "middle", "reg", "height", "dim", "rot", "vel", "score"
  const char* BindingMiddle = "middle";  // input
  const char* BindingReg0 = "reg_0";
  const char* BindingReg1 = "reg_1";
  const char* BindingReg2 = "reg_2";
  const char* BindingReg3 = "reg_3";
  const char* BindingReg4 = "reg_4";
  const char* BindingReg5 = "reg_5";     // output
  const char* BindingHeight0 = "height_0";
  const char* BindingHeight1 = "height_1";
  const char* BindingHeight2 = "height_2";
  const char* BindingHeight3 = "height_3";
  const char* BindingHeight4 = "height_4";
  const char* BindingHeight5 = "height_5";
  const char* BindingDim0 = "dim_0";
  const char* BindingDim1 = "dim_1";
  const char* BindingDim2 = "dim_2";
  const char* BindingDim3 = "dim_3";
  const char* BindingDim4 = "dim_4";
  const char* BindingDim5 = "dim_5";
  const char* BindingRot0 = "rot_0";
  const char* BindingRot1 = "rot_1";
  const char* BindingRot2 = "rot_2";
  const char* BindingRot3 = "rot_3";
  const char* BindingRot4 = "rot_4";
  const char* BindingRot5 = "rot_5";
  const char* BindingVel0 = "vel_0";
  const char* BindingVel1 = "vel_1";
  const char* BindingVel2 = "vel_2";
  const char* BindingVel3 = "vel_3";
  const char* BindingVel4 = "vel_4";
  const char* BindingVel5 = "vel_5";
  const char* BindingHeatmap0  = "hm_0";
  const char* BindingHeatmap1  = "hm_1";
  const char* BindingHeatmap2  = "hm_2";
  const char* BindingHeatmap3  = "hm_3";
  const char* BindingHeatmap4  = "hm_4";
  const char* BindingHeatmap5  = "hm_5";

  virtual ~CenterHeadImplement() {
    // CenterHead
    post_.reset();

    checkRuntime(cudaFreeHost(h_detections_num_));
    checkRuntime(cudaFree(d_detections_));
    checkRuntime(cudaFree(d_detections_reshape_));

    for (unsigned int i=0; i < NUM_TASKS; i++) {
        checkRuntime(cudaFree(d_reg_[i]));
        checkRuntime(cudaFree(d_height_[i]));
        checkRuntime(cudaFree(d_dim_[i]));
        checkRuntime(cudaFree(d_rot_[i]));
        checkRuntime(cudaFree(d_vel_[i]));
        checkRuntime(cudaFree(d_hm_[i]));
    }

    checkRuntime(cudaFreeHost(h_mask_));
  }

  virtual bool init(const CenterHeadParameter& param) {
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    params_ = param;

    post_.reset(new PostProcessCuda());
    post_->setParams(param);  // TODO: Write it into the constructor

    checkRuntime(cudaMallocHost((void **)&h_detections_num_, sizeof(unsigned int)));
    checkRuntime(cudaMemset(h_detections_num_, 0, sizeof(unsigned int)));

    checkRuntime(cudaMalloc((void **)&d_detections_, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));
    checkRuntime(cudaMemset(d_detections_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));

    //add d_detections_reshape_
    checkRuntime(cudaMalloc((void **)&d_detections_reshape_, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));
    checkRuntime(cudaMemset(d_detections_reshape_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));

    detections_.resize(MAX_DET_NUM, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  
    for(unsigned int i=0; i < NUM_TASKS; i++) {
        checkRuntime(cudaMalloc((void **)&d_reg_[i], engine_->getBindingNumel("reg_" + std::to_string(i)) * sizeof(half)));
        checkRuntime(cudaMalloc((void **)&d_height_[i], engine_->getBindingNumel("height_" + std::to_string(i)) * sizeof(half)));
        checkRuntime(cudaMalloc((void **)&d_dim_[i], engine_->getBindingNumel("dim_" + std::to_string(i)) * sizeof(half)));
        checkRuntime(cudaMalloc((void **)&d_rot_[i], engine_->getBindingNumel("rot_" + std::to_string(i)) * sizeof(half)));
        checkRuntime(cudaMalloc((void **)&d_vel_[i], engine_->getBindingNumel("vel_" + std::to_string(i)) * sizeof(half)));
        checkRuntime(cudaMalloc((void **)&d_hm_[i], engine_->getBindingNumel("hm_" + std::to_string(i)) * sizeof(half)));

        if(i==0){
            auto d = engine_->getBindingDims("reg_" + std::to_string(i));
            reg_n_ = d[0];
            reg_c_ = d[1];
            reg_h_ = d[2];
            reg_w_ = d[3];

            d = engine_->getBindingDims("height_" + std::to_string(i));
            height_c_ = d[1];
            d = engine_->getBindingDims("dim_" + std::to_string(i));
            dim_c_ = d[1];
            d = engine_->getBindingDims("rot_" + std::to_string(i));
            rot_c_ = d[1];
            d = engine_->getBindingDims("vel_" + std::to_string(i));
            vel_c_ = d[1];
        }
        auto d = engine_->getBindingDims("hm_" + std::to_string(i));
        hm_c_[i] = d[1];
    }
    h_mask_size_ = params_.nms_pre_max_size * DIVUP(params_.nms_pre_max_size, NMS_THREADS_PER_BLOCK) * sizeof(uint64_t);
    checkRuntime(cudaMallocHost((void **)&h_mask_, h_mask_size_));
    checkRuntime(cudaMemset(h_mask_, 0, h_mask_size_));
    return true;
  }

  virtual void print() override { engine_->print("BBox"); }

  virtual std::vector<Bndbox> forward(const nvtype::half* transfusion_feature, void* stream, bool sorted) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    nms_pred_.clear();

    engine_->forward(
      std::unordered_map<std::string, const void *>{{BindingMiddle, transfusion_feature}, 
      {BindingReg0, d_reg_[0]}, {BindingHeight0, d_height_[0]}, {BindingDim0, d_dim_[0]}, {BindingRot0, d_rot_[0]}, {BindingVel0, d_vel_[0]}, {BindingHeatmap0, d_hm_[0]},
      {BindingReg1, d_reg_[1]}, {BindingHeight1, d_height_[1]}, {BindingDim1, d_dim_[1]}, {BindingRot1, d_rot_[1]}, {BindingVel1, d_vel_[1]}, {BindingHeatmap1, d_hm_[1]},
      {BindingReg2, d_reg_[2]}, {BindingHeight2, d_height_[2]}, {BindingDim2, d_dim_[2]}, {BindingRot2, d_rot_[2]}, {BindingVel2, d_vel_[2]}, {BindingHeatmap2, d_hm_[2]},
      {BindingReg3, d_reg_[3]}, {BindingHeight3, d_height_[3]}, {BindingDim3, d_dim_[3]}, {BindingRot3, d_rot_[3]}, {BindingVel3, d_vel_[3]}, {BindingHeatmap3, d_hm_[3]},
      {BindingReg4, d_reg_[4]}, {BindingHeight4, d_height_[4]}, {BindingDim4, d_dim_[4]}, {BindingRot4, d_rot_[4]}, {BindingVel4, d_vel_[4]}, {BindingHeatmap4, d_hm_[4]},
      {BindingReg5, d_reg_[5]}, {BindingHeight5, d_height_[5]}, {BindingDim5, d_dim_[5]}, {BindingRot5, d_rot_[5]}, {BindingVel5, d_vel_[5]}, {BindingHeatmap5, d_hm_[5]}},
      _stream);
    
    for(unsigned int i_task =0; i_task < NUM_TASKS; i_task++) {
      checkRuntime(cudaMemset(h_detections_num_, 0, sizeof(unsigned int)));
      checkRuntime(cudaMemset(d_detections_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));
      checkRuntime(cudaMemset(d_detections_reshape_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));

      // std::cout << "[CUSTOM] " << reg_n_ << " " << reg_h_ << " " << reg_w_ << " " << reg_c_ << " " << height_c_ << " " << dim_c_ << " " << rot_c_ << " " << vel_c_ << " " << hm_c_[i_task] << std::endl;
      // [CUSTOM] 1 180 180 2 1 3 2 2 1
      // [CUSTOM] 1 180 180 2 1 3 2 2 2
      // [CUSTOM] 1 180 180 2 1 3 2 2 2
      // [CUSTOM] 1 180 180 2 1 3 2 2 1
      // [CUSTOM] 1 180 180 2 1 3 2 2 2
      // [CUSTOM] 1 180 180 2 1 3 2 2 2
      post_->doPostDecodeCuda(reg_n_, reg_h_, reg_w_, reg_c_, height_c_, dim_c_, rot_c_, vel_c_, hm_c_[i_task],
                              d_reg_[i_task],
                              d_height_[i_task],
                              d_dim_[i_task],
                              d_rot_[i_task],
                              d_vel_[i_task],
                              d_hm_[i_task],
                              h_detections_num_,
                              d_detections_, _stream);
      if(*h_detections_num_ == 0) continue;

      checkRuntime(cudaMemcpyAsync(detections_.data(), d_detections_, MAX_DET_NUM * DET_CHANNEL * sizeof(float), cudaMemcpyDeviceToHost, _stream));
      checkRuntime(cudaStreamSynchronize(_stream));

      std::sort(detections_.begin(), detections_.end(), [](float11 boxes1, float11 boxes2) { return boxes1.val[10] > boxes2.val[10]; });

      checkRuntime(cudaMemcpyAsync(d_detections_, detections_.data() , MAX_DET_NUM * DET_CHANNEL * sizeof(float), cudaMemcpyHostToDevice, _stream));
      checkRuntime(cudaMemsetAsync(h_mask_, 0, h_mask_size_, _stream));

      post_->doPermuteCuda(*h_detections_num_, d_detections_, d_detections_reshape_, _stream);
      checkRuntime(cudaStreamSynchronize(_stream));

      post_->doPostNMSCuda(*h_detections_num_, d_detections_reshape_, h_mask_, _stream);
      checkRuntime(cudaStreamSynchronize(_stream));

      int col_blocks = DIVUP(*h_detections_num_, NMS_THREADS_PER_BLOCK);
      std::vector<uint64_t> remv(col_blocks, 0);
      std::vector<bool> keep(*h_detections_num_, false);
      int max_keep_size = 0;
      for (unsigned int i_nms = 0; i_nms < *h_detections_num_; i_nms++) {
        unsigned int nblock = i_nms / NMS_THREADS_PER_BLOCK;
        unsigned int inblock = i_nms % NMS_THREADS_PER_BLOCK;

        if (!(remv[nblock] & (1ULL << inblock))) {
          keep[i_nms] = true;
          if (max_keep_size++ < params_.nms_post_max_size) {
              nms_pred_.push_back(Bndbox(detections_[i_nms].val[0], detections_[i_nms].val[1], detections_[i_nms].val[2],
                                  detections_[i_nms].val[3], detections_[i_nms].val[4], detections_[i_nms].val[5],
                                  detections_[i_nms].val[6], detections_[i_nms].val[7], detections_[i_nms].val[8],
                                  params_.task_num_stride[i_task] + static_cast<int>(detections_[i_nms].val[9]), detections_[i_nms].val[10]));
          }
          uint64_t* p = h_mask_ + i_nms * col_blocks;
          for (int j_nms = nblock; j_nms < col_blocks; j_nms++) {
              remv[j_nms] |= p[j_nms];
          }
        }
      }
    }
    return nms_pred_;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  std::vector<std::vector<int>> bindshape_;

  // CenterHead
  std::shared_ptr<PostProcessCuda> post_;
  CenterHeadParameter params_;

  unsigned int* h_detections_num_;
  float* d_detections_;
  float* d_detections_reshape_;     //add d_detections_reshape_

  half* d_reg_[NUM_TASKS];
  half* d_height_[NUM_TASKS];
  half* d_dim_[NUM_TASKS];
  half* d_rot_[NUM_TASKS];
  half* d_vel_[NUM_TASKS];
  half* d_hm_[NUM_TASKS];

  int reg_n_;
  int reg_c_;
  int reg_h_;
  int reg_w_;
  int height_c_;
  int dim_c_;
  int rot_c_;
  int vel_c_;
  int hm_c_[NUM_TASKS];

  std::vector<float11> detections_;
  unsigned int h_mask_size_;
  uint64_t* h_mask_ = nullptr;
  std::vector<Bndbox> nms_pred_;
};

std::shared_ptr<CenterHead> create_centerhead(const CenterHeadParameter& param) {
  std::shared_ptr<CenterHeadImplement> instance(new CenterHeadImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace object
};  // namespace head
};  // namespace bevfusion