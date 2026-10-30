from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import v2_common as C
import v2_data as D
import v2_folds as F
import v2_ssl as S


def _view(fit_cache):
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    folds = F.load_folds()
    return D.build_fold_view(0, cache=fit_cache, y_fit=y, folds=folds)


def test_structural_leakage_audit_rejects_root_paths(fit_cache):
    audit = S.structural_leakage_audit(fit_cache)
    assert audit["path_end_never_equals_root"] is True
    assert audit["n_paths_checked"] > 0
    assert audit["atom_type_onehot_channels_zeroed_for_masked_atoms"] is True
    assert audit["s2_path_codes_exclude_root"] is True


def test_pretext_split_is_deterministic_and_label_free(fit_cache):
    view = _view(fit_cache)
    a = S.pretext_split(view)
    b = S.pretext_split(view)
    assert a.audit == b.audit
    assert a.train_cache["n_molecules"] + a.val_cache["n_molecules"] == 1568
    assert set(a.train_cache["molecule_ids"]).isdisjoint(set(a.val_cache["molecule_ids"]))
    assert a.audit["labels_used"] is False
    assert a.audit["monitor_or_reserved_used"] is False


def test_fixed_val_mask_is_fixed(fit_cache):
    view = _view(fit_cache)
    split = S.pretext_split(view)
    m1, audit1 = S.fixed_val_mask(split.val_cache, seed=C.SSL_VAL_MASK_SEED)
    m2, audit2 = S.fixed_val_mask(split.val_cache, seed=C.SSL_VAL_MASK_SEED)
    assert torch.equal(m1, m2)
    assert audit1 == audit2
    assert bool(m1.any())


def test_ssl_smoke_writes_temp_and_freezes_f_rho(tmp_path, fit_cache):
    view = _view(fit_cache)
    payload = S.pretrain(
        S.SSLJob(
            key="ssl_smoke",
            fold=0,
            seed=0,
            view=view,
            out_root=tmp_path,
            device="cpu",
            smoke_limit=128,
            smoke_epochs=2,
        )
    )
    assert payload["pretext"]["gate_min_nat"] == C.SSL_GATE_MIN_NAT
    assert payload["leakage_audit"]["path_end_never_equals_root"] is True
    assert payload["gradient_audit"]["f_rho_unchanged"] is True
    assert payload["gradient_audit"]["all_atom_bond_local_head_params_have_gradient"] is True
    assert payload["monitor_labels_read"] is False
    ckpt = torch.load(tmp_path / "ssl_smoke" / "ssl_checkpoint.pt", map_location="cpu", weights_only=False)
    keys = set(ckpt["module_state"])
    assert keys
    assert all(k.startswith(("atom_encoder.", "bond_encoder.", "local.")) for k in keys)
    assert not any(k.startswith(("f.", "rho.")) for k in keys)
    assert "head_state" in ckpt
    assert ckpt["gate_pass"] == payload["pretext"]["gate_pass"]


def test_gate_arithmetic():
    heldout = 1.30
    prior = 1.36
    assert prior - heldout >= C.SSL_GATE_MIN_NAT
    assert prior - 1.32 < C.SSL_GATE_MIN_NAT
