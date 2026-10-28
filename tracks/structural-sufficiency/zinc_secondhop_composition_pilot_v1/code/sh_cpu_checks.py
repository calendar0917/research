"""CPU gates for the second-hop composition pilot.

All gates must PASS before any GPU launch (PROTOCOL.md §3):

    G1  fresh path enumeration == frozen LCRC cache, element-wise
    G2  trunk initialisation == historical LCRC-A initialisation (bit-identical)
    G3  Real/Null identical init, first-step gradients on the new columns,
        short training actually moves them, checkpoint round-trip
    G4  atom relabelling / bond reversal / batch split-merge invariance with
        NON-ZERO new-column weights (real and null channels)
    G5  same marginals + same path count, only the pairing differs
        (real data in-root swap + synthetic module-level counterexample)
    G6  input scale / coverage of the new columns
    G7  evaluation path: frozen A checkpoints reproduce their valid MAE
    G8  guards: parameter cap, non-trainable fixed projections, test unreachable
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import sh_common as S
import sh_data as D
import sh_models as M
import sh_train as T

import lc_models as OLD  # frozen LCRC-v1 implementation (read-only import)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_all():
    tw = time.perf_counter()
    train_cache = S.D.load_cache("train")
    valid_cache = S.D.load_cache("valid")
    train = D.build_supplement("train")
    valid = D.build_supplement("valid")
    return {
        "train_cache": train_cache, "valid_cache": valid_cache,
        "sup_train": train["supplement"], "sup_valid": valid["supplement"],
        "train_xc": train["crosscheck"], "valid_xc": valid["crosscheck"],
        "supplement": {"train": {"content_sha256": train["content_sha256"],
                                 "file_sha256": train["file_sha256"],
                                 "path": train["path"],
                                 "n_paths": train["supplement"]["n_paths"]},
                       "valid": {"content_sha256": valid["content_sha256"],
                                 "file_sha256": valid["file_sha256"],
                                 "path": valid["path"],
                                 "n_paths": valid["supplement"]["n_paths"]},
                       "wall_seconds": time.perf_counter() - tw},
    }


def _single_cache(mol: Mapping[str, Any]) -> dict:
    cache = S.D._concat_molecules([mol])
    cache["n_molecules"] = 1
    cache["labels"] = np.zeros(1, dtype=np.float32)
    cache["meta"] = {"feature_version": S.L.FEATURE_VERSION}
    return cache


def _cache_molecule(cache: Mapping[str, Any], m: int) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    a0, a1 = int(ap[m]), int(ap[m + 1])
    b0, b1 = int(bp[m]), int(bp[m + 1])
    return {
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1].copy(),
        "atom_type": np.asarray(cache["atom_type"])[a0:a1].copy(),
        "bond_index": np.asarray(cache["bond_index"])[b0:b1].copy(),
        "bond_type": np.asarray(cache["bond_type"])[b0:b1].copy(),
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1].copy(),
        "global_feat": np.asarray(cache["global"])[m].copy(),
    }


def _perturbed_model(arm: str = "Real", seed: int = 0, scale: float = 0.02):
    norm = S.D.load_norm()
    model, info = M.build_model(arm, norm, seed, "cpu", dict(M.DEFAULT_CONFIG))
    g = torch.Generator().manual_seed(20261105)
    with torch.no_grad():
        W = model.local.mlp.lins[0].weight
        new = torch.randn(W[:, S.BASE_X_DIM:].shape, generator=g) * float(scale)
        W[:, S.BASE_X_DIM:] = new
    model.eval()
    return model, info


@torch.no_grad()
def _forward(model, cache, sup, rows, pair_mode=None):
    batch = D.collate(cache, sup, rows)
    y, aux = model(batch, return_aux=True, pair_mode=pair_mode)
    return y.numpy(), {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in aux.items()}, batch


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# ---------------------------------------------------------------------------
# G1 -- path enumeration
# ---------------------------------------------------------------------------

def gate_g1(ctx) -> dict:
    audit_paths = S.REPO / "tracks/structural-sufficiency/zinc_path_binding_sufficiency_audit_v1/analysis/paths.json"
    audit = S.read_json_or_none(audit_paths) or {}
    out = {}
    for split in ("train", "valid"):
        xc = ctx[f"{split}_xc"]
        expected = (audit.get("elementwise_vs_frozen_cache", {}).get(split, {})
                    if audit else {})
        out[split] = {
            "crosscheck": xc,
            "n_paths": xc["n_paths"],
            "audit_n_paths": expected.get("n_paths"),
            "audit_agrees": (expected.get("n_paths") == xc["n_paths"]) if expected else None,
            "n_atoms": xc["n_atoms"],
            "audit_n_atoms": expected.get("n_atoms"),
        }
    passed = all(ctx[f"{s}_xc"]["passed"] for s in ("train", "valid"))
    passed = passed and all(out[s]["audit_agrees"] in (True, None) for s in ("train", "valid"))
    return {"passed": bool(passed), "splits": out}


# ---------------------------------------------------------------------------
# G2 -- historical initialisation identity
# ---------------------------------------------------------------------------

def gate_g2(ctx) -> dict:
    norm = S.D.load_norm()
    out: dict[str, Any] = {"seeds": {}, "step0": {}}
    for seed in (0, 1):
        old_model, old_info = OLD.build_model("A", norm, seed, "cpu", dict(OLD.DEFAULT_CONFIG))
        base = M.build_base_model(norm, seed)
        old_state = old_model.state_dict()
        base_state = base.state_dict()
        keys_equal = sorted(old_state) == sorted(base_state)
        tensors_equal = all(torch.equal(old_state[k], base_state[k]) for k in old_state) \
            if keys_equal else False
        extended, ext_info = M.build_model("Real", norm, seed, "cpu", dict(M.DEFAULT_CONFIG))
        ext_state = extended.state_dict()
        # compare every tensor except the widened first layer
        diffs = {}
        for k, v in base_state.items():
            if k == "local.mlp.lins.0.weight":
                continue
            if not torch.equal(v, ext_state[k]):
                diffs[k] = float((v - ext_state[k]).abs().max().item())
        w_base = base_state["local.mlp.lins.0.weight"]
        w_ext = ext_state["local.mlp.lins.0.weight"]
        new_cols_zero = bool(torch.count_nonzero(w_ext[:, S.BASE_X_DIM:]).item() == 0)
        old_cols_equal = bool(torch.equal(w_ext[:, :S.BASE_X_DIM], w_base))
        out["seeds"][str(seed)] = {
            "old_initial_state_sha256": old_info["initial_state_sha256"],
            "historical_recorded_sha256": S.HISTORICAL[seed]["initial_state_sha256"],
            "old_matches_recorded": old_info["initial_state_sha256"]
            == S.HISTORICAL[seed]["initial_state_sha256"],
            "new_base_state_sha256": S.L.C.state_hash(base),
            "new_base_matches_old": S.L.C.state_hash(base) == old_info["initial_state_sha256"],
            "state_keys_equal": bool(keys_equal),
            "all_tensors_bit_identical": bool(tensors_equal),
            "non_widened_tensor_diffs": diffs,
            "extended_initial_state_sha256": ext_info["initial_state_sha256"],
            "new_input_columns_zero": new_cols_zero,
            "old_input_columns_equal": old_cols_equal,
            "parameters": ext_info["parameters"],
            "parameters_A": old_info["parameters"],
            "param_cap_ok": ext_info["param_cap_ok"],
            "has_junction_module": ext_info["has_junction_module"],
            "attention_modules": ext_info["attention_modules"],
        }
    # step-0 forward equality on 8 valid molecules (old A vs base vs extended Real/Null)
    rows = np.arange(8)
    old_model, _ = OLD.build_model("A", norm, 0, "cpu", dict(OLD.DEFAULT_CONFIG))
    base = M.build_base_model(norm, 0)
    ext_r, _ = M.build_model("Real", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    ext_n, _ = M.build_model("Null", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    for m in (old_model, base, ext_r, ext_n):
        m.eval()
    batch = D.collate(ctx["valid_cache"], ctx["sup_valid"], rows)
    with torch.no_grad():
        y_old = old_model(batch).numpy()
        y_base = base(batch).numpy()
        y_real = ext_r(batch).numpy()
        y_null = ext_n(batch).numpy()
    out["step0"] = {
        "rows": rows.tolist(),
        "max_abs_old_vs_base": _max_abs(y_old, y_base),
        "max_abs_old_vs_real": _max_abs(y_old, y_real),
        "max_abs_old_vs_null": _max_abs(y_old, y_null),
        "max_abs_real_vs_null": _max_abs(y_real, y_null),
    }
    passed = all(v["old_matches_recorded"] and v["new_base_matches_old"]
                 and v["all_tensors_bit_identical"] and v["new_input_columns_zero"]
                 and v["old_input_columns_equal"] and v["param_cap_ok"]
                 and not v["has_junction_module"] and not v["attention_modules"]
                 for v in out["seeds"].values())
    out["passed"] = bool(passed and out["step0"]["max_abs_old_vs_real"] == 0.0
                         and out["step0"]["max_abs_old_vs_null"] == 0.0
                         and out["step0"]["max_abs_real_vs_null"] == 0.0)
    return out


# ---------------------------------------------------------------------------
# G3 -- Real/Null init, gradients, short training, checkpoint round-trip
# ---------------------------------------------------------------------------

def gate_g3(ctx) -> dict:
    norm = S.D.load_norm()
    model_r, info_r = M.build_model("Real", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    model_n, info_n = M.build_model("Null", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    shapes_r = {k: tuple(v.shape) for k, v in model_r.state_dict().items()}
    shapes_n = {k: tuple(v.shape) for k, v in model_n.state_dict().items()}
    out: dict[str, Any] = {
        "full_state_hash_equal": info_r["initial_state_sha256"] == info_n["initial_state_sha256"],
        "state_shapes_equal": shapes_r == shapes_n,
        "parameters": info_r["parameters"],
        "parameters_breakdown": info_r["parameters_breakdown"],
        "projection_trainable": info_r["projection_trainable"],
        "projections": info_r["projections"],
    }

    train_cache, sup_train = ctx["train_cache"], ctx["sup_train"]
    rows_fixed = np.arange(128)
    batch = D.collate(train_cache, sup_train, rows_fixed)
    y_fixed = np.asarray(train_cache["labels"], dtype=np.float32)[rows_fixed]
    target = torch.as_tensor(y_fixed)
    step_report = {}
    for model in (model_r, model_n):
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss = F.l1_loss(model(batch).view(-1), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gd = T.grad_diagnostics(model)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        step_report[model.arm] = {
            "loss": float(loss.detach().item()),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
            "first_layer": gd,
            "W_new_before": T.weight_diagnostics(model)["W_new_norm"],
        }
        opt.step()
        step_report[model.arm]["W_new_after_step1"] = T.weight_diagnostics(model)["W_new_norm"]
    out["step1"] = step_report

    # 30-step short training on 256 train molecules (Real only; schedule frozen)
    rows_short = np.arange(256)
    short_model, _ = M.build_model("Real", norm, 0, "cpu", dict(M.DEFAULT_CONFIG))
    short_model.train()
    opt = torch.optim.AdamW(short_model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    for step in range(30):
        rows = rows_short[(step * 128) % 256:((step * 128) % 256) + 128]
        b = D.collate(train_cache, sup_train, rows)
        t = torch.as_tensor(np.asarray(train_cache["labels"], dtype=np.float32)[rows])
        loss = F.l1_loss(short_model(b).view(-1), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(short_model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach().item()))
    wd = T.weight_diagnostics(short_model)
    out["short_train"] = {
        "loss_step1": losses[0],
        "loss_step30": losses[-1],
        "loss_decreased": bool(losses[-1] < losses[0]),
        "W_new_after_30": wd["W_new_norm"],
        "W_old_after_30": wd["W_old_norm"],
        "W_new_nonzero_columns": wd["W_new_nonzero_columns"],
    }

    # checkpoint round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pt"
        sha = T._save_checkpoint(path, short_model, arm="Real", seed=0, epoch=1, valid_mae=0.0)
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        reloaded = M.load_model_from_state("Real", norm, state, "cpu")
        reloaded.eval()
        p1, _ = T.evaluate(short_model, ctx["valid_cache"], ctx["sup_valid"], np.arange(32),
                           device="cpu", batch_size=32)
        p2, _ = T.evaluate(reloaded, ctx["valid_cache"], ctx["sup_valid"], np.arange(32),
                           device="cpu", batch_size=32)
        out["checkpoint"] = {
            "sha256": sha,
            "max_abs_pred_diff": _max_abs(p1, p2),
            "bit_identical": bool(np.array_equal(p1, p2)),
        }
    passed = (out["full_state_hash_equal"] and out["state_shapes_equal"]
              and not out["projection_trainable"]
              and all(v["first_layer"]["grad_new_norm"] > 0
                      and v["first_layer"]["grad_new_nonzero_columns"] > 0
                      and v["first_layer"]["all_finite"]
                      and v["W_new_before"] == 0.0
                      and v["W_new_after_step1"] > 0.0
                      for v in step_report.values())
              and out["short_train"]["loss_decreased"]
              and out["short_train"]["W_new_nonzero_columns"] > 0
              and out["checkpoint"]["max_abs_pred_diff"] == 0.0)
    out["passed"] = bool(passed)
    return out


# ---------------------------------------------------------------------------
# G4 -- invariance with non-zero new-column weights
# ---------------------------------------------------------------------------

def gate_g4(ctx) -> dict:
    cache = ctx["valid_cache"]
    sup = ctx["sup_valid"]
    model, _ = _perturbed_model("Real")
    model_n, _ = _perturbed_model("Null")
    rng = np.random.default_rng(20261105)

    # ---- atom relabelling / bond reversal on 8 single molecules ----------
    per_mol = {}
    for m in range(8):
        raw = _cache_molecule(cache, m)
        n = int(np.asarray(cache["atom_ptr"])[m + 1] - np.asarray(cache["atom_ptr"])[m])
        perm = rng.permutation(n)
        variants = {
            "orig": S.D.build_from_arrays(**raw),
            "perm": S.D.build_from_arrays(**raw, perm=perm),
            "reverse": S.D.build_from_arrays(**raw, reverse_bonds=True),
        }
        outs = {}
        for name, mol in variants.items():
            c = _single_cache(mol)
            s = D.supplement_from_cache(c, "valid", write=False)["supplement"]
            y, aux, _ = _forward(model, c, s, np.arange(1))
            outs[name] = {"y": y, **{k: aux[k] for k in ("h", "B", "C_real", "C_null")}}
        entry = {
            "n_atoms": n,
            "y_orig": float(outs["orig"]["y"][0]),
            "max_abs_y_perm": _max_abs(outs["orig"]["y"], outs["perm"]["y"]),
            "max_abs_y_reverse": _max_abs(outs["orig"]["y"], outs["reverse"]["y"]),
        }
        for key in ("h", "B", "C_real", "C_null"):
            entry[f"max_abs_{key}_perm"] = _max_abs(
                outs["orig"][key], outs["perm"][key][perm])
            entry[f"max_abs_{key}_reverse"] = _max_abs(
                outs["orig"][key], outs["reverse"][key])
        per_mol[str(m)] = entry

    # ---- batch split / merge / order on 16 valid molecules ---------------
    rows = np.arange(16)
    order = np.random.default_rng(7).permutation(16)
    batch_out = {}
    for mode in ("real", "null"):
        model_use = model
        y_full, aux_full, _ = _forward(model_use, cache, sup, rows, pair_mode=mode)
        y_split = np.concatenate([_forward(model_use, cache, sup, rows[:8], pair_mode=mode)[0],
                                  _forward(model_use, cache, sup, rows[8:], pair_mode=mode)[0]])
        y_perm, _, _ = _forward(model_use, cache, sup, rows[order], pair_mode=mode)
        inv_order = np.argsort(order)
        singles = [_forward(model_use, cache, sup, [int(r)], pair_mode=mode)[0][0] for r in rows]
        h_singles = np.concatenate([_forward(model_use, cache, sup, [int(r)], pair_mode=mode)[1]["h"]
                                    for r in rows], axis=0)
        batch_out[mode] = {
            "max_abs_y_split": _max_abs(y_full, y_split),
            "max_abs_y_permuted": _max_abs(y_full, y_perm[inv_order]),
            "max_abs_y_singles": _max_abs(y_full, np.asarray(singles)),
            "max_abs_h_singles": _max_abs(aux_full["h"], h_singles),
        }
    # the Null arm itself also exercises C_null
    y_null_arm, aux_null_arm, _ = _forward(model_n, cache, sup, rows)
    y_null_arm_singles = np.concatenate(
        [_forward(model_n, cache, sup, [int(r)])[0] for r in rows])
    batch_out["null_arm"] = {
        "max_abs_y_singles": _max_abs(y_null_arm, y_null_arm_singles)}

    max_diff = max(
        [v for e in per_mol.values() for k, v in e.items() if k.startswith("max_abs_")]
        + [v for e in batch_out.values() for v in e.values()])
    return {
        "passed": bool(max_diff <= S.INVARIANCE_TOL),
        "tolerance": S.INVARIANCE_TOL,
        "max_diff": float(max_diff),
        "per_molecule": per_mol,
        "batch": batch_out,
    }


# ---------------------------------------------------------------------------
# G5 -- marginals matched, only the pairing differs
# ---------------------------------------------------------------------------

def gate_g5(ctx) -> dict:
    cache, sup = ctx["valid_cache"], ctx["sup_valid"]
    model, _ = _perturbed_model("Real", scale=0.05)

    # ---- synthetic module-level counterexample ---------------------------
    mod = M.SecondHopContent(width=4, proj_dim=2, seed=20261105)
    pb, pa = mod.P_b, mod.P_a
    U = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b2 = U @ torch.linalg.pinv(pb)
    a_u = V @ torch.linalg.pinv(pa)
    a = torch.zeros(3, 4)
    a[1] = a_u[0]
    a[2] = a_u[1]
    b = torch.zeros(2, 4)
    b[0] = b2[0]
    b[1] = b2[1]
    base = {"path_atom": torch.tensor([0, 0]), "path_count": torch.tensor([2, 0, 0])}
    batch_a = dict(base, path_end=torch.tensor([1, 2]), path_e2_pos=torch.tensor([0, 1]))
    batch_b = dict(base, path_end=torch.tensor([1, 2]), path_e2_pos=torch.tensor([1, 0]))
    with torch.no_grad():
        Ba, Cra, Cna = mod(batch_a, a, b)
        Bb, Crb, Cnb = mod(batch_b, a, b)
    synth = {
        "B_equal": _max_abs(Ba, Bb),
        "C_null_equal": _max_abs(Cna, Cnb),
        "C_real_diff": _max_abs(Cra, Crb),
        "C_real_a": Cra[0].tolist(),
        "C_real_b": Crb[0].tolist(),
        "C_null": Cna[0].tolist(),
    }
    synth["passed"] = bool(synth["B_equal"] <= 1e-6 and synth["C_null_equal"] <= 1e-6
                           and synth["C_real_diff"] > 1e-3)

    # ---- real-data in-root pairing swap (n_v == 2 roots) -----------------
    rows = np.arange(64)
    batch = D.collate(cache, sup, rows)
    n = batch["path_count"]
    path_atom = batch["path_atom"]
    # find roots with exactly two paths
    counts = torch.zeros(int(batch["atom_feat"].shape[0]), dtype=torch.long)
    counts.index_add_(0, path_atom, torch.ones_like(path_atom))
    two_roots = torch.nonzero(counts == 2).view(-1)[:64]
    idx_by_root = {}
    for a_idx in two_roots.tolist():
        idx_by_root[a_idx] = torch.nonzero(path_atom == a_idx).view(-1).tolist()
    swapped = batch["path_e2_pos"].clone()
    for pair in idx_by_root.values():
        if len(pair) == 2:
            swapped[pair[0]], swapped[pair[1]] = batch["path_e2_pos"][pair[1]], batch["path_e2_pos"][pair[0]]
    batch_swap = dict(batch, path_e2_pos=swapped)
    with torch.no_grad():
        y_real, aux_real = model(batch, return_aux=True, pair_mode="real")
        y_swap, aux_swap = model(batch_swap, return_aux=True, pair_mode="real")
        y_null, aux_null = model(batch, return_aux=True, pair_mode="null")
        y_null_swap, aux_null_swap = model(batch_swap, return_aux=True, pair_mode="null")
    real_batch_ctx = {
        "n_two_path_roots_swapped": int(len(idx_by_root)),
        "B_swap_diff": _max_abs(aux_real["B"], aux_swap["B"]),
        "C_null_swap_diff": _max_abs(aux_real["C_null"], aux_swap["C_null"]),
        "C_real_swap_diff": _max_abs(aux_real["C_real"], aux_swap["C_real"]),
        "y_real_swap_diff": _max_abs(y_real, y_swap),
        "y_null_swap_diff": _max_abs(y_null, y_null_swap),
        "null_channel_C_real_vs_null_diff": _max_abs(aux_real["C_real"], aux_real["C_null"]),
    }
    real_batch_ctx["passed"] = bool(
        real_batch_ctx["n_two_path_roots_swapped"] > 0
        and real_batch_ctx["B_swap_diff"] <= 1e-5
        and real_batch_ctx["C_null_swap_diff"] <= 1e-5
        and real_batch_ctx["C_real_swap_diff"] > 1e-3
        and real_batch_ctx["y_real_swap_diff"] > 1e-4
        and real_batch_ctx["y_null_swap_diff"] <= 1e-5)

    # ---- n_v <= 1 identity on real data ----------------------------------
    n = batch["path_count"].to(torch.float64)
    one_or_zero = n <= 1
    cr = aux_real["C_real"]
    cn = aux_real["C_null"]
    identity = {
        "atoms_n_le_1": int(one_or_zero.sum().item()),
        "max_abs_C_real_minus_null_n_le_1": float((cr[one_or_zero] - cn[one_or_zero]).abs().max().item())
        if int(one_or_zero.sum().item()) else 0.0,
        "max_abs_C_real_minus_null_n_ge_2": float((cr[~one_or_zero] - cn[~one_or_zero]).abs().max().item())
        if int((~one_or_zero).sum().item()) else 0.0,
        "atoms_n_ge_2": int((~one_or_zero).sum().item()),
    }
    identity["passed"] = bool(identity["atoms_n_le_1"] > 0
                              and identity["max_abs_C_real_minus_null_n_le_1"] == 0.0
                              and identity["max_abs_C_real_minus_null_n_ge_2"] > 1e-3)
    passed = synth["passed"] and real_batch_ctx["passed"] and identity["passed"]
    return {"passed": bool(passed), "synthetic": synth, "real_swap": real_batch_ctx,
            "n_le_1_identity": identity}


# ---------------------------------------------------------------------------
# G6 -- scale and coverage
# ---------------------------------------------------------------------------

def gate_g6(ctx) -> dict:
    model, _ = _perturbed_model("Real", scale=0.02)
    out = {}
    for split, cache, sup in (("train", ctx["train_cache"], ctx["sup_train"]),
                              ("valid", ctx["valid_cache"], ctx["sup_valid"])):
        rows = np.arange(256)
        batch = D.collate(cache, sup, rows)
        with torch.no_grad():
            a, b, _h, aux = model._encode(batch)
            scal = (batch["atom_scalar"] - model.scalar_mean) / model.scalar_scale
            x = model.local.build_input(batch, a, b, scal, aux["second_hop_extra"])
            n = batch["path_count"]

        def rms(t):
            return float(t.pow(2).mean().sqrt().item())

        out[split] = {
            "n_molecules": len(rows),
            "n_atoms": int(x.shape[0]),
            "x_old_rms": rms(x[:, :S.BASE_X_DIM]),
            "B_rms": rms(aux["B"]),
            "C_real_rms": rms(aux["C_real"]),
            "C_null_rms": rms(aux["C_null"]),
            "new_block_rms": rms(aux["second_hop_extra"]),
            "C_real_over_C_null": rms(aux["C_real"]) / max(rms(aux["C_null"]), 1e-12),
            "atoms_path_count_0": int((n == 0).sum().item()),
            "atoms_path_count_1": int((n == 1).sum().item()),
            "atoms_path_count_ge2": int((n >= 2).sum().item()),
            "mean_path_count": float(n.to(torch.float64).mean().item()),
            "max_path_count": int(n.max().item()),
        }
    ratios = [out[s]["C_real_over_C_null"] for s in out]
    out["scale_ratio_band"] = list(S.SCALE_RATIO_BAND)
    out["scale_ratio_ok"] = all(S.SCALE_RATIO_BAND[0] <= r <= S.SCALE_RATIO_BAND[1]
                                for r in ratios)
    out["passed"] = bool(out["scale_ratio_ok"])
    return out


# ---------------------------------------------------------------------------
# G7 -- evaluation path / frozen A reproduction
# ---------------------------------------------------------------------------

def gate_g7(ctx) -> dict:
    norm = S.D.load_norm()
    rows = np.arange(int(ctx["valid_cache"]["n_molecules"]))
    y = np.asarray(ctx["valid_cache"]["labels"], dtype=np.float64)
    out: dict[str, Any] = {"seeds": {}}
    for seed in (0, 1):
        run = S.historical_dir(seed)
        npz = np.load(run / "predictions" / "valid_best.npz")
        pred_frozen = np.asarray(npz["pred"], dtype=np.float64)
        mae_frozen = float(np.mean(np.abs(pred_frozen - y[np.asarray(npz["row"])])))
        train_json = S.L.read_json(run / "train.json")
        ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model, info = M.build_model("Real", norm, seed, "cpu", dict(M.DEFAULT_CONFIG))
        load_info = M.load_historical_A(model, ckpt["state_dict"])
        pred, mae = T.evaluate(model, ctx["valid_cache"], ctx["sup_valid"], rows,
                               device="cpu", batch_size=256)
        out["seeds"][str(seed)] = {
            "checkpoint": S.track_rel(run / "best.pt"),
            "checkpoint_sha256_prefix": S.L.C.sha256_file(run / "best.pt")[:16],
            "recorded_sha256_prefix": S.HISTORICAL[seed]["checkpoint_sha256_prefix"],
            "frozen_npz_mae": mae_frozen,
            "recorded_best_valid_mae_epoch_loop": train_json["best"]["valid_mae"],
            "recorded_best_valid_mae_replay": train_json["best_valid_mae_eval"],
            "npz_matches_replay": abs(mae_frozen - train_json["best_valid_mae_eval"]) <= 1e-9,
            "epoch_loop_minus_replay": float(train_json["best"]["valid_mae"]
                                             - train_json["best_valid_mae_eval"]),
            "cpu_reloaded_mae": float(mae),
            "delta_mae_cpu_vs_frozen": float(mae - mae_frozen),
            "max_abs_pred_diff_cpu_vs_frozen": _max_abs(pred, pred_frozen),
            "load_info": load_info,
        }
    passed = all(abs(v["delta_mae_cpu_vs_frozen"]) <= 1e-6
                 and v["npz_matches_replay"]
                 and v["checkpoint_sha256_prefix"] == v["recorded_sha256_prefix"]
                 for v in out["seeds"].values())
    out["passed"] = bool(passed)
    return out


# ---------------------------------------------------------------------------
# G8 -- guards
# ---------------------------------------------------------------------------

def gate_g8(ctx) -> dict:
    norm = S.D.load_norm()
    projs = {}
    for arm, seed in (("Real", 0), ("Null", 0), ("Real", 1)):
        _, info = M.build_model(arm, norm, seed, "cpu", dict(M.DEFAULT_CONFIG))
        projs[f"{arm}_s{seed}"] = {"sha": info["projections"], "params": info["parameters"],
                                   "trainable": info["projection_trainable"],
                                   "cap_ok": info["param_cap_ok"]}
    ref = projs["Real_s0"]["sha"]
    identical = all(v["sha"] == ref for v in projs.values())
    test_refused = None
    try:
        D.build_supplement("test", write=False)
        test_refused = False
    except ValueError:
        test_refused = True
    except FileNotFoundError:
        test_refused = True
    out = {
        "parameter_cap": S.PARAM_CAP,
        "parameters": {k: v["params"] for k, v in projs.items()},
        "cap_ok": all(v["cap_ok"] for v in projs.values()),
        "projections_identical_across_arms_and_seeds": identical,
        "projections_trainable": any(v["trainable"] for v in projs.values()),
        "projection_seed": S.PROJ_SEED,
        "projection_shas": ref,
        "test_split_refused": bool(test_refused),
        "official_test_loaded": False,
        "budget": {"target_gpu_hours": 1.0, "hard_gpu_hours": 3.0,
                   "max_concurrent_gpus": 2},
        "labels_used_for_selection": False,
    }
    out["passed"] = bool(out["cap_ok"] and identical and not out["projections_trainable"]
                         and out["test_split_refused"])
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

GATES = {"G1": gate_g1, "G2": gate_g2, "G3": gate_g3, "G4": gate_g4,
         "G5": gate_g5, "G6": gate_g6, "G7": gate_g7, "G8": gate_g8}
GATE_NAMES = {
    "G1": "G1_paths", "G2": "G2_init_identity", "G3": "G3_real_null",
    "G4": "G4_invariance", "G5": "G5_pairing_semantics", "G6": "G6_scale_coverage",
    "G7": "G7_eval_path", "G8": "G8_guards",
}


def _sha(path: Path) -> str | None:
    return S.L.C.sha256_file(path) if path.exists() else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="second-hop pilot CPU gates")
    parser.add_argument("--gates", default="G1,G2,G3,G4,G5,G6,G7,G8")
    args = parser.parse_args(argv)
    S.L.ensure_dir(S.ANALYSIS)

    ctx = _load_all()
    S.L.write_json(S.ANALYSIS / "supplement.json", ctx["supplement"])

    results = {}
    for key in args.gates.split(","):
        key = key.strip().upper()
        if key not in GATES:
            raise SystemExit(f"unknown gate {key}")
        t0 = time.perf_counter()
        S.L.log(f"[cpu-gate] {key} start")
        try:
            res = GATES[key](ctx)
        except Exception as exc:  # noqa: BLE001 - a crashing gate is a FAIL
            res = {"passed": False, "error": repr(exc)}
        res["seconds"] = time.perf_counter() - t0
        results[GATE_NAMES[key]] = res
        S.L.log(f"[cpu-gate] {key} {'PASS' if res['passed'] else 'FAIL'} ({res['seconds']:.1f}s)")

    all_pass = all(v.get("passed") for v in results.values())
    payload = {
        "round": S.ROUND_ID,
        "created": S.L.now(),
        "git_commit": S.L.C.git_commit(),
        "environment": S.L.env_summary(),
        "gates": results,
        "all_pass": bool(all_pass),
        "gpu_hours": 0.0,
        "official_test_loaded": False,
        "supplement": ctx["supplement"],
    }
    S.L.write_json(S.ANALYSIS / "cpu_checks.json", payload)

    provenance = {
        "round": S.ROUND_ID,
        "created": S.L.now(),
        "git_commit": S.L.C.git_commit(),
        "environment": S.L.env_summary(),
        "protocol_sha256": _sha(S.EXP_DIR / "PROTOCOL.md"),
        "config_sha256": _sha(S.CONFIG_PATH),
        "cpu_checks_sha256": _sha(S.ANALYSIS / "cpu_checks.json"),
        "train_cache": {
            "path": S.track_rel(S.D.cache_path("train")),
            "file_sha256": _sha(S.D.cache_path("train")),
            "content_sha256": S.D.content_fingerprint(ctx["train_cache"]),
        },
        "valid_cache": {
            "path": S.track_rel(S.D.cache_path("valid")),
            "file_sha256": _sha(S.D.cache_path("valid")),
            "content_sha256": S.D.content_fingerprint(ctx["valid_cache"]),
        },
        "supplement": ctx["supplement"],
        "historical_checkpoints": {
            f"A_s{seed}": {
                "path": S.track_rel(S.historical_dir(seed) / "best.pt"),
                "sha256": _sha(S.historical_dir(seed) / "best.pt"),
                "best_valid_mae": S.HISTORICAL[seed]["best_valid_mae"],
            } for seed in (0, 1)
        },
        "projection_seed": S.PROJ_SEED,
        "gpu_hours": 0.0,
        "official_test_loaded": False,
        "labels_used": "official train/valid only",
        "all_gates_pass": bool(all_pass),
    }
    S.L.write_json(S.ANALYSIS / "provenance.json", provenance)
    S.L.log(f"[cpu-gate] all_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
