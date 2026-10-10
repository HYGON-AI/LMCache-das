// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include <pybind11/pybind11.h>
#include "mem_kernels_hip.cuh"
#include <torch/torch.h>

namespace py = pybind11;

PYBIND11_MODULE(hcu_c_ops, m) {
  // TransferDirection / GPUKVFormat are registered by the baseline
  // lmcache.c_ops extension (csrc/pybind.cpp). We deliberately do NOT
  // re-register them here: pybind11 throws
  //   "generic_type: type 'TransferDirection' is already registered!"
  // when a second module binds the same C++ enum type. The HCU kernel
  // functions below still accept baseline enum values because pybind11
  // resolves enum type casters globally by C++ type identity, which is
  // shared across both .so files (same enum class declaration).

  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("gpu_kv_format"), py::arg("block_size") = 0,
        py::arg("skip_prefix_n_tokens") = 0);
  m.def("multi_layer_kv_transfer_asymmetric", &multi_layer_kv_transfer_asymmetric,
        py::arg("key_value"), py::arg("key_cache_ptrs"),
        py::arg("value_cache_ptrs"), py::arg("slot_mapping"),
        py::arg("paged_memory_device"), py::arg("num_heads"),
        py::arg("head_size"), py::arg("block_size"),
        py::arg("direction"), py::arg("skip_prefix_n_tokens") = 0);
  m.def("multi_layer_kv_transfer_unilateral", &multi_layer_kv_transfer_unilateral,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("gpu_kv_format"));
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
        py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("direction"),
        py::arg("gpu_kv_format"), py::arg("token_major") = false);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl,
        py::arg("lmc_key_value_cache"), py::arg("sgl_key_cache"),
        py::arg("sgl_value_cache"), py::arg("slot_mapping"),
        py::arg("direction"), py::arg("token_major") = false);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::arg("dest"), py::arg("src"), py::arg("nbytes"),
        py::arg("direction"), py::arg("host_buffer_offset"),
        py::arg("host_buffer_alignments"));
}
