# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import sys
from types import ModuleType


def ensure_module(monkeypatch, name: str) -> ModuleType:
    """Install a lightweight module and its parent packages in sys.modules."""
    parts = name.split(".")
    parent = None
    full = ""
    for part in parts:
        full = part if not full else f"{full}.{part}"
        mod = sys.modules.get(full)
        if mod is None:
            mod = ModuleType(full)
            monkeypatch.setitem(sys.modules, full, mod)
        if parent is not None and not hasattr(parent, part):
            setattr(parent, part, mod)
        parent = mod
    return sys.modules[name]
