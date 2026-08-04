#
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

# HCU native extension (hcu_c_ops*.so): only mem-kernel functions live here.
_hcu_mem_kernels = importlib.import_module("lmcache_hcu.hcu_c_ops")

# Baseline lmcache.c_ops module, injected by lmcache_hcu.__init__ via
# _set_base_c_ops() before we're placed into sys.modules.
_base_c_ops: ModuleType | None = None

_MEM_KERNEL_OVERRIDES = {
    "multi_layer_kv_transfer",
    "multi_layer_kv_transfer_asymmetric",
    "multi_layer_kv_transfer_unilateral",
    "single_layer_kv_transfer",
    "single_layer_kv_transfer_sgl",
    "load_and_reshape_flash",
    "reshape_and_cache_back_flash",
    "lmcache_memcpy_async",
}


def _set_base_c_ops(module: ModuleType | None) -> None:
    """Inject the baseline lmcache.c_ops so non-mem-kernel calls can fall back."""
    global _base_c_ops
    _base_c_ops = module


def _load_base_c_ops() -> ModuleType:
    if _base_c_ops is None:
        raise AttributeError(
            "baseline lmcache.c_ops is not available; install/build upstream LMCache "
            "for non-mem-kernels c_ops symbols"
        )
    return _base_c_ops


def __getattr__(name: str) -> Any:
    # mem-kernels: served from the HCU extension.
    if name in _MEM_KERNEL_OVERRIDES:
        return getattr(_hcu_mem_kernels, name)
    # everything else: transparently forwarded to the baseline c_ops.
    return getattr(_load_base_c_ops(), name)


def __dir__() -> list[str]:
    names = set(_MEM_KERNEL_OVERRIDES)
    if _base_c_ops is not None:
        names.update(dir(_base_c_ops))
    return sorted(names)


# Eagerly expose mem-kernel symbols so `from lmcache.c_ops import X` also works.
for _name in _MEM_KERNEL_OVERRIDES:
    globals()[_name] = getattr(_hcu_mem_kernels, _name)
