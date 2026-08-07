from lmcache_hcu import _hcu_flatten_block_ids, _hcu_same_block_prefix


def test_flatten_block_ids_accepts_plain_list():
    assert _hcu_flatten_block_ids([1, 2, 3]) == [1, 2, 3]


def test_flatten_block_ids_uses_first_kv_group_from_tuple():
    assert _hcu_flatten_block_ids(([1, 2, 3], [10, 20, 30])) == [1, 2, 3]


def test_same_block_prefix_detects_full_block_table():
    assert _hcu_same_block_prefix([1, 2, 3], [1, 2, 3, 4, 5]) is True


def test_same_block_prefix_rejects_delta_block_table():
    assert _hcu_same_block_prefix([1, 2, 3], [4, 5]) is False


def test_same_block_prefix_rejects_divergent_block_table():
    assert _hcu_same_block_prefix([1, 2, 3], [1, 2, 9, 4]) is False
