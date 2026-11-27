# SPDX-License-Identifier: Apache-2.0
"""Expose pinned upstream fixtures without adding upstream production sources to sys.path."""
import importlib.util
import os
from pathlib import Path
import sys
import tests

TEST_ALIAS = "lmcache_tests"


def prepare_environment():
    root = Path(os.environ["LMCACHEPATH"]) / "tests"
    if not (root / "__init__.py").is_file():
        raise RuntimeError(f"Missing 0.5.6 upstream test fixtures: {root}")
    # Upstream fixtures themselves also use absolute tests.v1.* imports.
    tests.__path__.append(str(root))
    spec = importlib.util.spec_from_file_location(
        TEST_ALIAS, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[TEST_ALIAS] = module
    spec.loader.exec_module(module)
