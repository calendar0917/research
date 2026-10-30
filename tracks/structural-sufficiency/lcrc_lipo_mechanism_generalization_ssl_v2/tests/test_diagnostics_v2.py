from __future__ import annotations

import numpy as np
import torch

import v2_common as C
import v2_data as D
import v2_diagnostics as G
import v2_models as M
import v2_folds as F


def _batch(fit_cache):
    return D.collate(fit_cache, list(range(16)))


def _pair_multiset(batch):
    pairs = list(zip(batch["path_cat"].tolist(), batch["path_end63"].tolist()))
    return sorted(pairs)


def test_joint_permutation_preserves_pairs_and_atom_groups(fit_cache):
    batch = _batch(fit_cache)
    out = G._permute_segments(batch, ("path_cat", "path_end63"), same_perm=True, seed=0)
    assert torch.equal(batch["path_atom"], out["path_atom"])
    assert _pair_multiset(batch) == _pair_multiset(out)
    assert torch.equal(batch["path_end"], out["path_end"])
    assert torch.equal(batch["path_closure"], out["path_closure"])


def test_independent_permutation_breaks_pairing(fit_cache):
    batch = _batch(fit_cache)
    out = G._permute_segments(batch, ("path_cat", "path_end63"), same_perm=False, seed=0)
    assert torch.equal(batch["path_atom"], out["path_atom"])
    assert sorted(batch["path_cat"].tolist()) == sorted(out["path_cat"].tolist())
    assert sorted(batch["path_end63"].tolist()) == sorted(out["path_end63"].tolist())


def test_E_is_invariant_to_s2_permutations(spec, norm, fit_cache):
    e, _base, _shared = M.build_e(spec, norm, 0)
    e.eval()
    batch = _batch(fit_cache)
    out = G._permute_segments(batch, ("path_cat", "path_end63", "path_end", "path_closure"), same_perm=True, seed=0)
    with torch.no_grad():
        assert torch.equal(e(batch), e(out))


def test_global_train_mean_shape(fit_cache):
    batch = _batch(fit_cache)
    mean = np.arange(42, dtype=np.float32)
    out = G._global_train_mean(batch, mean)
    assert out["global"].shape == (16, 42)
    assert np.allclose(out["global"].numpy(), np.repeat(mean[None, :], 16, axis=0))
