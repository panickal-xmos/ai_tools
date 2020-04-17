// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/pooling.h"

extern "C" {
#include "lib_nn/api/nn_types.h"
}

namespace xcore {
namespace pooling {

//**************************************
//**************************************
//**************************************
// MaxPool
//**************************************
//**************************************
//**************************************
XCoreStatus MaxPool::Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                          int32_t Y_w, int32_t C_out) {
  nn_image_params_t in_params;
  in_params.height = X_h;
  in_params.width = X_w;
  in_params.channels = C_in;

  nn_image_params_t out_params;
  out_params.height = Y_h;
  out_params.width = Y_w;
  out_params.channels = C_out;

  nn_window_op_config_t config;
  nn_window_op_config_simple(&config, &in_params, &out_params, params.pool_h,
                             params.pool_w, params.stride_h,
                             params.stride_w);

  maxpool2d_init(&plan_, &in_params, &out_params, &config);

  return kXCoreOk;
}

XCoreStatus MaxPool::Eval(int8_t* Y, const int8_t* X) {
  maxpool2d(Y, X, &plan_);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// AvgPool
//**************************************
//**************************************
//**************************************
XCoreStatus AvgPool::Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                          int32_t Y_w, int32_t C_out) {
  nn_image_params_t in_params;
  in_params.height = X_h;
  in_params.width = X_w;
  in_params.channels = C_in;

  nn_image_params_t out_params;
  out_params.height = Y_h;
  out_params.width = Y_w;
  out_params.channels = C_out;

  nn_window_op_config_t config;
  nn_window_op_config_simple(&config, &in_params, &out_params, params.pool_h,
                             params.pool_w, params.stride_h,
                             params.stride_w);

  avgpool2d_init(&plan_, &in_params, &out_params, &config);

  return kXCoreOk;
}

XCoreStatus AvgPool::Eval(int8_t* Y, const int8_t* X) {
  avgpool2d(Y, X, &plan_);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// AvgPool_Global
//**************************************
//**************************************
//**************************************
XCoreStatus AvgPool_Global::Init(int32_t bias, int32_t shift, int32_t scale) {
  bias_ = bias;
  shift_ = shift;
  scale_ = scale;

  return kXCoreOk;
}

XCoreStatus AvgPool_Global::Eval(int8_t* Y, const int8_t* X, int32_t X_h,
                                 int32_t X_w, uint32_t C_in) {
  avgpool2d_global(Y, X, X_h, X_w, C_in, bias_, shift_, scale_);

  return kXCoreOk;
}

}  // namespace pooling
}  // namespace xcore