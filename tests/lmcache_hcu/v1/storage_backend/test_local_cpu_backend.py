# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Based on LMCache-Ascend v0.3.12 (58e0aef462eda7c451f84735b277b35e02dce227).

"""Reuse upstream LocalCPUBackend tests with LMCache-HCU runtime patches active."""
from __future__ import annotations

# Standard
import sys

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.

# Third Party
import lmcache_tests.v1 as upstream_v1
import lmcache_tests.v1.utils as upstream_utils

# Upstream local_cpu tests import helpers through the original tests.v1.utils path.
# Register that path to the aliased upstream tests package before importing them.
sys.modules.setdefault("tests.v1", upstream_v1)
sys.modules.setdefault("tests.v1.utils", upstream_utils)
setattr(sys.modules["tests"], "v1", upstream_v1)

# Reuse upstream tests.
from lmcache_tests.v1.storage_backend.test_local_cpu_backend import (  # noqa: F401
    TestLocalCPUBackend,
    local_cpu_backend,
    local_cpu_backend_disabled,
)
