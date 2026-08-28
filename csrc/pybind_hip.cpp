// SPDX-License-Identifier: Apache-2.0
// Modifications Copyright 2026 Hygon Information Technology Co., Ltd.

#include <pybind11/pybind11.h>
#include "mem_kernels_hip.cuh"
#include <torch/torch.h>

namespace py = pybind11;

PYBIND11_MODULE(hcu_c_ops, m) {
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_asymmetric", &multi_layer_kv_transfer_asymmetric);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async);
}
