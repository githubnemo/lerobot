from pathlib import Path

import torch

from lerobot.policies.vam.cosmos_layer_mix_cache import (
    COSMOS_LAYER_PROBE_DEPTHS,
    CosmosLayerMixCacheItem,
    layer_mix_artifact_name,
    save_layer_mix_feature_cache,
    verify_layer_mix_feature_cache,
)


def provenance(contexts: dict[int, torch.Tensor]) -> dict:
    return {
        "schema_version": 1,
        "artifact": layer_mix_artifact_name("none"),
        "dataset": {
            "repo_id": "hubnemo/cube_out_of_box_dataset",
            "revision": "r" * 40,
            "episode_index": 0,
            "frame_index": 4,
            "window_indices": [0, 1, 2, 3, 4],
            "window_offsets": [-4, -3, -2, -1, 0],
            "fps": 10,
        },
        "split": {"name": "train"},
        "prompt": {},
        "backbone": {},
        "weights": {"lora": {"path": "best_lora.safetensors"}},
        "extraction": {"noise_seed": 7, "state_t": 2},
        "temporal_contract": {"state_t": 2},
        "output": {
            "backbone": "Cosmos-Predict2-2B",
            "tapped_layers": list(COSMOS_LAYER_PROBE_DEPTHS),
            "deepest_layer": 20,
            "state_t": 2,
            "high_noise_sigma": 80.0,
            "one_forward_pass": True,
            "context_transform": "none",
            "context_tokens_per_layer": 2_400,
            "context_channels": 2_048,
            "context_grid": [2, 30, 40],
            "context_dtype": "bfloat16",
        },
        "tensors": {"contexts": {str(layer): list(contexts[layer].shape) for layer in contexts}},
    }


def test_cosmos_multidepth_cache_stacks_state_t2_contexts_in_tapped_order(tmp_path: Path) -> None:
    contexts = {
        layer: torch.full((1, 2_400, 2_048), float(index), dtype=torch.bfloat16)
        for index, layer in enumerate(COSMOS_LAYER_PROBE_DEPTHS)
    }
    item = CosmosLayerMixCacheItem(
        contexts,
        torch.zeros(1, 1, 6),
        torch.zeros(1, 30, 6),
        torch.zeros(1, 30, dtype=torch.bool),
        provenance(contexts),
    )
    assert item.context.shape == (1, 6, 2_400, 2_048)
    assert torch.equal(item.context[0, 0], contexts[4][0])
    assert torch.equal(item.context[0, -1], contexts[20][0])

    path = tmp_path / "episode-0000-frame-000004.safetensors"
    save_layer_mix_feature_cache(item, path)
    loaded = verify_layer_mix_feature_cache(path)
    assert loaded.context.shape == (1, 6, 2_400, 2_048)
