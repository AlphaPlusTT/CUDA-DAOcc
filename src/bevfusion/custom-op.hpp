#ifndef CUSTOM_HPP
#define CUSTOM_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "bevfusion/bevfusion.hpp"
#include "common/tensor.hpp"

namespace bevfusion{
namespace custom
{

void set_last_feature_to_zero(nv::Tensor lidar_points, void* stream);

} // namespace custom
} // bevfusion


#endif // CUSTOM_HPP