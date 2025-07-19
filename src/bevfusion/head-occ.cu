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

#include <algorithm>
#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "head-occ.hpp"

namespace bevfusion {
namespace head {
namespace occ{

class BevOccHeadImplement : public BevOccHead {
 public:
  // const char* BindingCamera = "camera";
  const char* BindingMiddle  = "middle";
  const char* BindingOutput = "occ";

  virtual ~BevOccHeadImplement() {
    if (output_) checkRuntime(cudaFree(output_));
  }

  virtual bool init(const OccParameter& param) {
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    auto shape = engine_->static_dims(BindingOutput);
    Asserts(engine_->dtype(BindingOutput) == TensorRT::DType::INT32, "Invalid binding data type.");

    // TODO: check shape
    volumn_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    // std::cout << "[custom] " << volumn << std::endl;
    // for (auto s: shape){
    //   std::cout << "[custom] " << s << std::endl;
    // }
    // std::cout << "[custom] " << "volumn_" << volumn_ << std::endl;

    checkRuntime(cudaMalloc(&output_, volumn_ * sizeof(int)));
    output_cpu_.resize(volumn_, 17);  // TODO: Make 17 a configurable item
    // param_ = param;
    return true;
  }

  virtual void print() override { engine_->print("BevOccHead"); }

  virtual std::vector<int> forward(const nvtype::half* transfusion_feature, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward(
      {// {BindingCamera, camera_bev},
      {BindingMiddle, transfusion_feature},
      {BindingOutput, output_}},
      _stream);
    // std::cout << "[custom] " << "volumn_" << volumn_ << std::endl;
    checkRuntime(cudaMemcpyAsync(output_cpu_.data(), output_, volumn_ * sizeof(int), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));
    return output_cpu_;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  int* output_ = nullptr;
  std::vector<int> output_cpu_;
  std::vector<std::vector<int>> bindshape_;
  // OccParameter param_;
  size_t volumn_;
};

std::shared_ptr<BevOccHead> create_bevocchead(const OccParameter& param) {
  std::shared_ptr<BevOccHeadImplement> instance(new BevOccHeadImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace occ
};  // namespace head
};  // namespace bevfusion