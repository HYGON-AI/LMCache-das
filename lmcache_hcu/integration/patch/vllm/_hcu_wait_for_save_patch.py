# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monkey-patch ``LMCacheConnectorV1Impl.wait_for_save`` to tolerate
``slot_mapping``/``token_ids`` length mismatches.

Mirrors the upstream fix from lmcache commit ``fa6aa371``
(``[fix]errorHandle for kvcache file not found``), which converted
``assert len(slot_mapping) == len(token_ids)`` into a warn + ``continue``.
The upstream change lives in
``lmcache/integration/vllm/vllm_v1_adapter.py::wait_for_save``; this patch
keeps downstream lmcache_hcu safe even when the installed lmcache version
predates fa6aa371.

Strategy
--------
The patched ``wait_for_save`` runs the original implementation but
temporarily replaces ``connector_metadata.requests`` with a filtered list
that drops any request whose ``slot_mapping`` length does not match its
``token_ids`` length. A warning is logged for each dropped request. After
the original returns, the original list is restored. No upstream code is
mutated.
"""

from __future__ import annotations

from typing import Any, List

from lmcache.logging import init_logger

logger = init_logger(__name__)


def _hcu_wait_for_save(self: Any) -> None:
    """Drop length-mismatched requests, then call the original
    ``wait_for_save``."""

    original = getattr(self.__class__, "_lmcache_hcu_original_wait_for_save", None)
    if original is None:
        # Patch not installed -- call the bound method unchanged.
        return self.wait_for_save()

    connector_metadata = self._parent._get_connector_metadata()
    original_requests = connector_metadata.requests
    if not original_requests:
        return original(self)

    filtered: List[Any] = []
    for request in original_requests:
        slot_mapping = getattr(request, "slot_mapping", None)
        token_ids = getattr(request, "token_ids", None)
        if (
            slot_mapping is not None
            and token_ids is not None
            and len(slot_mapping) != len(token_ids)
        ):
            logger.warning(
                "Skipping KV save for request %s: slot_mapping/token_ids "
                "length mismatch (slot_mapping=%d, token_ids=%d). Likely "
                "an upstream allocation/preemption desync; the engine "
                "stays alive and only this request's save is dropped.",
                getattr(request, "req_id", "<unknown>"),
                len(slot_mapping),
                len(token_ids),
            )
            continue
        filtered.append(request)

    if len(filtered) == len(original_requests):
        # Nothing to filter -- no warning needed, no mutation required.
        return original(self)

    connector_metadata.requests = filtered
    try:
        return original(self)
    finally:
        connector_metadata.requests = original_requests


def patch_wait_for_save(vllm_v1_adapter: Any) -> bool:
    """Install the wait_for_save monkey-patch on
    ``LMCacheConnectorV1Impl``."""

    impl_cls = getattr(vllm_v1_adapter, "LMCacheConnectorV1Impl", None)
    if impl_cls is None:
        logger.warning("Skip wait_for_save length-mismatch patch: LMCacheConnectorV1Impl not found")
        return False

    original = impl_cls.wait_for_save
    if getattr(original, "_lmcache_hcu_wait_for_save_patched", False):
        return False

    impl_cls._lmcache_hcu_original_wait_for_save = original  # type: ignore[attr-defined]
    impl_cls.wait_for_save = _hcu_wait_for_save
    _hcu_wait_for_save._lmcache_hcu_wait_for_save_patched = True  # type: ignore[attr-defined]
    return True
