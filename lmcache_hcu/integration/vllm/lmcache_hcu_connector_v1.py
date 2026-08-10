# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Standard

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger

import lmcache_hcu  # noqa: F401

from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic


logger = init_logger(__name__)


class LMCacheHcuConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)