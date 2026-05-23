from types import SimpleNamespace

import pytest
import sglang.srt.layers.vocab_parallel_embedding as vocab_embedding
import sglang.srt.server_args as server_args
import torch
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def test_virtual_tp_vocab_padding_keeps_real_vocab_size(monkeypatch):
    monkeypatch.setattr(vocab_embedding, "_is_cpu", False)
    monkeypatch.setattr(vocab_embedding, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(
        vocab_embedding, "get_tensor_model_parallel_world_size", lambda: 3
    )
    monkeypatch.setattr(
        server_args,
        "_global_server_args",
        SimpleNamespace(virtual_tp_sharding="b12x-padded"),
    )

    embedding = VocabParallelEmbedding(
        129280,
        16,
        params_dtype=torch.float32,
    )

    assert embedding.org_vocab_size == 129280
    assert embedding.num_embeddings == 129280
    assert embedding.org_vocab_size_padded == 129408
    assert embedding.num_embeddings_padded == 129408
    assert embedding.num_embeddings_per_partition == 43136
    assert embedding.shard_indices.num_org_elements == 43008
    assert embedding.shard_indices.num_org_vocab_padding == 128


@pytest.mark.parametrize(
    (
        "rank",
        "world_size",
        "expected_padded",
        "expected_partition",
        "expected_real_elements",
        "expected_padding",
    ),
    [
        (8, 9, 129600, 14400, 14080, 320),
        (9, 10, 129280, 12928, 12928, 0),
    ],
)
def test_virtual_tp_vocab_padding_for_deepseek_v4_pro_tp9_tp10(
    monkeypatch,
    rank,
    world_size,
    expected_padded,
    expected_partition,
    expected_real_elements,
    expected_padding,
):
    monkeypatch.setattr(vocab_embedding, "_is_cpu", False)
    monkeypatch.setattr(vocab_embedding, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        vocab_embedding, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(
        server_args,
        "_global_server_args",
        SimpleNamespace(virtual_tp_sharding="b12x-padded"),
    )

    embedding = VocabParallelEmbedding(
        129280,
        16,
        params_dtype=torch.float32,
    )

    assert embedding.org_vocab_size == 129280
    assert embedding.org_vocab_size_padded == expected_padded
    assert embedding.num_embeddings_padded == expected_padded
    assert embedding.num_embeddings_per_partition == expected_partition
    assert embedding.shard_indices.num_org_elements == expected_real_elements
    assert embedding.shard_indices.num_org_vocab_padding == expected_padding
