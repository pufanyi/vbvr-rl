from src.data.vbvr_latent_dataset import _RankShardSplitter, _ShardSubsetSplitter


def test_rank_shard_splitter_keeps_normal_rank_stride():
    splitter = _RankShardSplitter(rank=2, world_size=4, num_shards=10)

    assert splitter.rank_shard_count == 2
    assert list(splitter(iter(range(10)))) == [2, 6]


def test_shard_subset_splitter_reads_group_shards():
    splitter = _ShardSubsetSplitter(offset=3, stride=4, num_shards=20)

    assert splitter.rank_shard_count == 5
    assert list(splitter(iter(range(20)))) == [3, 7, 11, 15, 19]
