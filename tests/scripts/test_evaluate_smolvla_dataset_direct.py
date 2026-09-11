import hashlib

import torch

from lerobot.policies.vam.action_rmse import EvaluationBatch, evaluate_backend
from scripts.video_vam.evaluate_smolvla_dataset_direct import StableBackend, stable_seed


class NoiseBackend:
    name = "noise"

    def predict_actions(self, batch, seed):
        return torch.randn((1, 30, 6), generator=torch.Generator().manual_seed(seed))


def batch(ids):
    n = len(ids)
    return EvaluationBatch(
        tuple(ids), torch.zeros(n, 30, 6), torch.arange(30)[None].expand(n, -1) >= 2, torch.zeros(n, 6)
    )


def test_stable_seed_formula():
    assert stable_seed(2, "sample") == int(hashlib.sha256(b"2:sample").hexdigest()[:8], 16) % (2**31 - 1)


def test_batch_and_order_invariance():
    backend = StableBackend(NoiseBackend(), 2)
    full = backend.predict_actions(batch(["a", "b"]), 0)
    singleton = torch.cat([backend.predict_actions(batch([x]), 100000) for x in ["a", "b"]])
    assert torch.equal(full, singleton)
    assert torch.equal(full.flip(0), backend.predict_actions(batch(["b", "a"]), 42))


def test_global_masked_aggregation_is_batch_invariant():
    a = evaluate_backend(StableBackend(NoiseBackend(), 0), [batch(["a", "b"])], seed=0)
    b = evaluate_backend(StableBackend(NoiseBackend(), 0), [batch(["a"]), batch(["b"])], seed=0)
    assert a == b
    assert a["n_valid_scalars"] == 24
