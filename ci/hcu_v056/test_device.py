import os


def test_hip_allocation():
    import torch
    assert torch.version.hip and torch.cuda.is_available()
    assert torch.cuda.device_count() == int(os.environ["TEST_TP"])
    for i in range(torch.cuda.device_count()):
        result = torch.tensor([1, 2], device=f"cuda:{i}").sum()
        assert result.item() == 3


def test_mp_connector():
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    assert issubclass(LMCacheMPConnector, KVConnectorBase_V1)
