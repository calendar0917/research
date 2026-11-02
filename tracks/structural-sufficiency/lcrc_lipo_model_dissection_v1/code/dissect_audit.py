"""CPU verification of the high-risk points, before any formal training.

Checks (mirroring section 8 of the executed protocol):

1. source/split hashes, partition, cache<->row<->label alignment, Train-only norm
2. FULL behavioural equivalence, parameter counts, single shared dropout
3. removed-module / removed-column mapping, shared initial state, shuffle order
4. B4/B5 exact marginal algebra in float64 (n=0, n=1, duplicates, bt2, batch
   offset, bias counting) and intervention coverage on real data
5. gradients for every retained learned branch (B4/B5 end-to-end, no detach)
6. best-epoch save/reload/argmin, denormalisation, no Test target in the view

Writes ``analysis/audit_cpu.json`` and exits non-zero if anything fails.
"""

from __future__ import annotations

import argparse
import copy
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import dissect_common as C
import dissect_data as D
import dissect_model as M
import dissect_source as S

import mb_common as MC
import mb_model as MM
import mb_raw_artifact as RAW
import mb_rdkit as R

import lcrc_data as PREP_DATA
import molecular_graph as MG

FAILS: list[str] = []
CHECKS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    if not ok:
        FAILS.append(name)
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    return {"name": name, "ok": bool(ok), "detail": detail}


# ---------------------------------------------------------------------------
# 1. data
# ---------------------------------------------------------------------------


def audit_data(view: D.View) -> dict[str, Any]:
    df = pd.read_csv(C.LIPOP_CSV)
    raw = np.asarray(df[C.LABEL_COLUMN].to_numpy(), dtype=np.float64)
    out: dict[str, Any] = {"splits": {}}
    for repeat in C.REPEATS:
        src = S.load_lipop(repeat, labels="all")
        rows = src["rows"]
        out["splits"][f"r{repeat}"] = {
            "sizes": src["sizes"],
            "file_sha256_match": C.sha256_file(C.CHEMBENCH_LIPOP / C.SPLIT_FILES[repeat]) == C.SPLIT_FILE_SHA256[repeat],
            "row_sha256_match": [
                C.row_sha256(rows[k]) == C.SPLIT_ROWS_SHA256[repeat][i] for i, k in enumerate(("train", "valid", "test"))
            ],
            "disjoint": not (set(rows["train"]) & set(rows["valid"]))
            and not (set(rows["train"]) & set(rows["test"]))
            and not (set(rows["valid"]) & set(rows["test"])),
            "union_all_rows": len(set(rows["train"]) | set(rows["valid"]) | set(rows["test"])) == C.N_MOLECULES,
            "label_alignment": {
                k: bool(np.array_equal(np.asarray(src["y"][k]), raw[np.asarray(rows[k])])) for k in ("train", "valid", "test")
            },
        }
        out["splits"][f"r{repeat}"]["all_ok"] = bool(
            out["splits"][f"r{repeat}"]["file_sha256_match"]
            and all(out["splits"][f"r{repeat}"]["row_sha256_match"])
            and out["splits"][f"r{repeat}"]["disjoint"]
            and out["splits"][f"r{repeat}"]["union_all_rows"]
            and all(out["splits"][f"r{repeat}"]["label_alignment"].values())
        )
        check(f"data/r{repeat}", out["splits"][f"r{repeat}"]["all_ok"], out["splits"][f"r{repeat}"])

    # cache <-> split <-> label alignment for repeat 122
    train = view.train_cache
    valid = view.valid_cache
    out["cache_rows_match"] = bool(
        list(np.asarray(train["rows"]).tolist()) == view.rows["train"]
        and list(np.asarray(valid["rows"]).tolist()) == view.rows["valid"]
        and train["molecule_ids"] == [str(r) for r in view.rows["train"]]
        and valid["molecule_ids"] == [str(r) for r in view.rows["valid"]]
    )
    check("data/cache_rows", out["cache_rows_match"], None)
    out["cache_label_free"] = bool(not train["meta"]["labels_present"] and not valid["meta"]["labels_present"])
    check("data/cache_label_free", out["cache_label_free"], None)

    # Train-only normalisation: recompute and compare, and show it is not Valid's
    train_norm = PREP_DATA.fit_norm(train, view.spec, source="audit_train_only")
    valid_norm = PREP_DATA.fit_norm(valid, view.spec, source="audit_valid_for_contrast")
    def _arrays_equal(a, b):
        return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a if k.endswith(("mean", "scale")))
    out["norm_train_only"] = {
        "matches_train_fit": bool(_arrays_equal(train_norm, view.train_norm)),
        "differs_from_valid_fit": bool(not _arrays_equal(valid_norm, view.train_norm)),
        "global_mean_max_abs_diff_vs_valid": float(
            np.abs(np.asarray(train_norm["global_mean"]) - np.asarray(valid_norm["global_mean"])).max()
        ),
        "scalar_mean_max_abs_diff_vs_valid": float(
            np.abs(np.asarray(train_norm["scalar_mean"]) - np.asarray(valid_norm["scalar_mean"])).max()
        ),
    }
    check(
        "data/norm_train_only",
        out["norm_train_only"]["matches_train_fit"] and out["norm_train_only"]["differs_from_valid_fit"],
        out["norm_train_only"],
    )

    # scalar field order (closure_ratio must be index 2)
    names = [f.name for f in view.spec.atom_scalar_fields]
    out["scalar_field_order"] = names
    check("data/scalar_field_order", names[2] == "closure_ratio" and len(names) == 9, names)

    # batch-order generator identity for the same seed
    g1 = torch.Generator().manual_seed(0 + C.SHUFFLE_OFFSET)
    g2 = torch.Generator().manual_seed(C.SHUFFLE_OFFSET)
    p1 = torch.randperm(3360, generator=g1).tolist()
    p2 = torch.randperm(3360, generator=g2).tolist()
    out["shuffle_identity"] = {"seed0": p1[:5], "same": p1 == p2}
    check("data/shuffle_identity", p1 == p2, out["shuffle_identity"])
    return out


# ---------------------------------------------------------------------------
# 2. FULL equivalence
# ---------------------------------------------------------------------------


def audit_full(view: D.View) -> dict[str, Any]:
    spec = view.spec
    norm = view.train_norm
    out: dict[str, Any] = {}

    # (a) initial state is bitwise identical to the frozen upstream model
    torch.manual_seed(0)
    old = MM.PREP_MODEL.LCRC(spec, norm, MM.PREP_MODEL.DEFAULT_CONFIG)
    old_state = {k: v.detach().clone() for k, v in old.state_dict().items()}
    new_state = M.build_full_state(spec, norm, 0)
    same_keys = set(old_state) == set(new_state)
    max_init_diff = max(
        (float((old_state[k] - new_state[k]).abs().max()) for k in old_state if old_state[k].shape == new_state[k].shape),
        default=float("inf"),
    )
    out["init_state"] = {
        "same_keys": same_keys,
        "n_entries": len(old_state),
        "max_abs_diff": max_init_diff,
        "bitwise_identical": bool(same_keys and max_init_diff == 0.0),
    }
    check("full/init_state", out["init_state"]["bitwise_identical"], out["init_state"])

    # (b) replay the frozen public122 FULL seed0 checkpoint
    ref_path = C.REPO / C.REFERENCE["checkpoint"]
    ckpt = torch.load(ref_path, map_location="cpu", weights_only=False)
    new_model = M.load_checkpoint_model(spec, norm, "A0", ckpt["state_dict"], torch.device("cpu"))
    old_model = MM.load_checkpoint_model(spec, norm, ckpt["state_dict"], torch.device("cpu"))
    valid_std = D.predict_scores(new_model, view.valid_cache, torch.device("cpu"))
    old_valid_std = D.predict_scores(old_model, view.valid_cache, torch.device("cpu"))
    tn = ckpt["target_norm"]
    valid_raw = valid_std * float(tn["scale"]) + float(tn["mean"])
    rmse = float(np.sqrt(((valid_raw - view.y_valid) ** 2).mean()))
    out["reference_replay"] = {
        "checkpoint": C.REFERENCE["checkpoint"],
        "checkpoint_sha256": C.sha256_file(ref_path),
        "best_epoch": int(ckpt["best_epoch"]),
        "recorded_valid_rmse": C.REFERENCE["valid_rmse"],
        "replayed_valid_rmse": rmse,
        "abs_diff": abs(rmse - C.REFERENCE["valid_rmse"]),
        "tolerance": C.REPRO_TOL,
        "old_vs_new_max_abs_prediction_diff": float(np.abs(valid_std - old_valid_std).max()),
    }
    check(
        "full/reference_replay",
        out["reference_replay"]["abs_diff"] <= C.REPRO_TOL
        and out["reference_replay"]["old_vs_new_max_abs_prediction_diff"] == 0.0,
        out["reference_replay"],
    )

    # (c) eval-mode aux equality on a fixed batch
    batch = D.to_device(PREP_DATA.collate(view.valid_cache, list(range(64))), torch.device("cpu"))
    old_model.eval()
    new_model.eval()
    with torch.no_grad():
        y_old, aux_old = old_model(batch, return_aux=True)
        y_new, aux_new = new_model(batch, return_aux=True)
    aux_keys = ("contrib", "rho", "contrib_atom", "mean_h", "mean_h2", "h", "count", "a", "b")
    diffs = {k: float((aux_old[k] - aux_new[k]).abs().max()) for k in aux_keys}
    out["eval_aux"] = {
        "y_max_abs_diff": float((y_old - y_new).abs().max()),
        "aux_max_abs_diff": diffs,
        "max": max([float((y_old - y_new).abs().max())] + list(diffs.values())),
    }
    check("full/eval_aux", out["eval_aux"]["max"] <= C.CPU_EQUIV_TOL, out["eval_aux"])

    # (d) train-mode dropout: same RNG -> same dropped h, loss, grads, one step
    full_state = M.build_full_state(spec, norm, 3)
    old_tr = MM.PREP_MODEL.LCRC(spec, norm, MM.PREP_MODEL.DEFAULT_CONFIG)
    MM.DropoutAtLocal(old_tr, float(C.RECIPE["dropout"]))
    old_tr.load_state_dict({k: v.clone() for k, v in full_state.items()})
    new_tr = M.DissectLCRC(spec, norm, "A0")
    new_tr.load_state_dict({k: v.clone() for k, v in full_state.items()})
    for m in (old_tr, new_tr):
        m.train()
    y_target = torch.as_tensor(np.random.RandomState(0).randn(64).astype(np.float32)).reshape(-1)
    torch.manual_seed(1234)
    y_o, aux_o = old_tr(batch, return_aux=True)
    torch.manual_seed(1234)
    y_n, aux_n = new_tr(batch, return_aux=True)
    loss_o = torch.nn.functional.mse_loss(y_o.reshape(-1), y_target)
    loss_n = torch.nn.functional.mse_loss(y_n.reshape(-1), y_target)
    loss_o.backward()
    loss_n.backward()
    grad_pairs = [(n, p) for n, p in old_tr.named_parameters() if p.grad is not None]
    grad_diffs = {
        n: float((p.grad - dict(new_tr.named_parameters())[n].grad).abs().max()) for n, p in grad_pairs
    }
    opt_o = torch.optim.AdamW(old_tr.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    opt_n = torch.optim.AdamW(new_tr.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    opt_o.step()
    opt_n.step()
    step_diffs = {
        n: float((p.detach() - dict(new_tr.named_parameters())[n].detach()).abs().max())
        for n, p in old_tr.named_parameters()
    }
    out["train_mode"] = {
        "h_max_abs_diff": float((aux_o["h"] - aux_n["h"]).abs().max()),
        "loss_abs_diff": float(abs(loss_o.item() - loss_n.item())),
        "grad_max_abs_diff": max(grad_diffs.values()) if grad_diffs else float("inf"),
        "step_max_abs_diff": max(step_diffs.values()) if step_diffs else float("inf"),
        "n_grad_entries": len(grad_pairs),
    }
    # prove dropout was really applied: the pre-dropout local output differs
    with torch.no_grad():
        a_enc, b_enc, scal = new_tr._encode(batch)
        h_pre, _ = new_tr.local(batch, a_enc, b_enc, scal)
    out["train_mode"]["dropout_effective"] = bool(not torch.equal(h_pre, aux_n["h"]))
    out["train_mode"]["all_ok"] = bool(
        out["train_mode"]["h_max_abs_diff"] <= C.CPU_EQUIV_TOL
        and out["train_mode"]["loss_abs_diff"] <= C.CPU_EQUIV_TOL
        and out["train_mode"]["grad_max_abs_diff"] <= C.CPU_EQUIV_TOL
        and out["train_mode"]["step_max_abs_diff"] <= C.CPU_EQUIV_TOL
        and out["train_mode"]["dropout_effective"]
    )
    check("full/train_mode", out["train_mode"]["all_ok"], out["train_mode"])

    # (e) checkpoint save/reload is bitwise stable
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ck.pt"
        torch.save({"state_dict": {k: v.cpu() for k, v in new_tr.state_dict().items()}}, p)
        reload = M.load_checkpoint_model(
            spec, norm, "A0", torch.load(p, map_location="cpu", weights_only=False)["state_dict"], torch.device("cpu")
        )
    reload.eval()
    with torch.no_grad():
        y_reload = reload(batch)
        y_ref = new_tr.eval()(batch)
    out["save_reload"] = {"max_abs_diff": float((y_reload - y_ref).abs().max())}
    check("full/save_reload", out["save_reload"]["max_abs_diff"] == 0.0, out["save_reload"])

    # (f) batch split / merge invariance
    rows = list(range(8))
    with torch.no_grad():
        merged = new_model(
            D.to_device(PREP_DATA.collate(view.valid_cache, rows), torch.device("cpu"))
        ).reshape(-1).numpy()
        split_pred = []
        for r in rows:
            split_pred.append(
                float(new_model(D.to_device(PREP_DATA.collate(view.valid_cache, [r]), torch.device("cpu"))).reshape(-1)[0])
            )
    out["batch_split_merge"] = {"max_abs_diff": float(np.abs(merged - np.asarray(split_pred)).max())}
    check("full/batch_split_merge", out["batch_split_merge"]["max_abs_diff"] <= 1e-5, out["batch_split_merge"])

    # (g) atom renumbering invariance on a real molecule
    mol_index = int(view.rows["valid"][3])
    payload = RAW.load(verify=True)
    raw_row = payload["rows"]["Lipophilicity"][mol_index]
    graph = R.raw_row_to_graph(raw_row, mol_index, spec)
    n = int(graph.atom_type.shape[0])
    rng = np.random.RandomState(20261102)
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    new_row = dict(raw_row)
    new_row["atom_class"] = np.asarray(raw_row["atom_class"])[perm]
    new_row["atom_charge"] = np.asarray(raw_row["atom_charge"])[perm]
    new_row["atom_h"] = np.asarray(raw_row["atom_h"])[perm]
    ei = np.asarray(raw_row["edge_index"])
    new_row["edge_index"] = np.stack([inv[ei[0]], inv[ei[1]]], axis=0)
    graph_p = R.raw_row_to_graph(new_row, mol_index, spec)
    mol = PREP_DATA.build_molecule(graph, spec)
    mol_p = PREP_DATA.build_molecule(graph_p, spec)
    cache1 = PREP_DATA.concat_molecules([mol], spec)
    cache2 = PREP_DATA.concat_molecules([mol_p], spec)
    with torch.no_grad():
        y1, aux1 = new_model(D.to_device(PREP_DATA.collate(cache1, [0]), torch.device("cpu")), return_aux=True)
        y2, aux2 = new_model(D.to_device(PREP_DATA.collate(cache2, [0]), torch.device("cpu")), return_aux=True)
    perm_ok = bool(torch.allclose(aux1["contrib_atom"][perm], aux2["contrib_atom"], atol=1e-5, rtol=0))
    out["atom_renumbering"] = {
        "molecule_row": mol_index,
        "n_atoms": n,
        "prediction_abs_diff": float((y1 - y2).abs().max()),
        "per_atom_contribution_matches_permutation": perm_ok,
    }
    check(
        "full/atom_renumbering",
        out["atom_renumbering"]["prediction_abs_diff"] <= 1e-5 and perm_ok,
        out["atom_renumbering"],
    )

    # (h) isolated atom (single-atom molecule) and empty S2 (ethane)
    from mb_rdkit import smiles_to_raw_row

    for smiles, tag in (("C", "single_atom"), ("CC", "empty_s2")):
        rr = smiles_to_raw_row(smiles, 0)
        g = R.raw_row_to_graph(rr, 0, spec)
        m = PREP_DATA.build_molecule(g, spec)
        cache = PREP_DATA.concat_molecules([m], spec)
        path_count = int(cache["meta"]["n_s2_paths"])
        with torch.no_grad():
            y, aux = new_model(D.to_device(PREP_DATA.collate(cache, [0]), torch.device("cpu")), return_aux=True)
        finite = bool(torch.isfinite(y).all() and torch.isfinite(aux["h"]).all())
        out[tag] = {"n_atoms": int(cache["meta"]["n_atoms_total"]), "n_s2_paths": path_count, "finite": finite}
        check(f"full/{tag}", finite, out[tag])
    out["empty_s2"]["has_no_paths"] = out["empty_s2"]["n_s2_paths"] == 0
    check("full/empty_s2_has_no_paths", out["empty_s2"]["has_no_paths"], out["empty_s2"])
    return out


# ---------------------------------------------------------------------------
# 3. variants / column mapping
# ---------------------------------------------------------------------------


def _perturb(batch: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    if key == "global":
        out["global"] = out["global"] * 0.5 + 0.25
    elif key == "atom_scalar":
        out["atom_scalar"] = out["atom_scalar"] + 0.37
    elif key == "atom_feat":
        out["atom_feat"] = torch.flip(out["atom_feat"], dims=[1])
    elif key == "bond_feat":
        out["bond_feat"] = torch.flip(out["bond_feat"], dims=[1])
    elif key == "nbr":
        out["nbr_code"] = (out["nbr_code"] + 7) % C.N_NBR_CODES
    elif key == "path":
        out["path_cat"] = (out["path_cat"] + 25) % C.N_PATH_CODES
        out["path_end63"] = (out["path_end63"] + 5) % C.N_NBR_CODES
    elif key == "path_closure":
        out["path_closure"] = 1 - out["path_closure"]
    elif key == "path_end":
        out["path_end"] = (out["path_end"] + 1) % int(batch["atom_feat"].shape[0])
    else:
        raise KeyError(key)
    return out


def audit_variants(view: D.View) -> dict[str, Any]:
    spec = view.spec
    norm = view.train_norm
    state = M.build_full_state(spec, norm, 0)
    batch = D.to_device(PREP_DATA.collate(view.valid_cache, list(range(24))), torch.device("cpu"))
    out: dict[str, Any] = {}
    for variant in C.VARIANT_ORDER:
        model = M.build_variant(spec, norm, variant, state, torch.device("cpu"))
        model.eval()
        params = M.parameter_count(model)
        shared = M.shared_state_equal(variant, state, model.state_dict())
        module_names = [n for n, _ in model.named_modules()]
        with torch.no_grad():
            y0 = model(batch)
        local_cols, rho_cols = C.variant_cols(variant)
        expected_absent = {
            "A1": ["local", "f", "atom_encoder", "bond_encoder"],
            "A5": ["rho"],
            "B0": ["local.path_emb", "local.end_emb", "local.path_u", "local.path_v", "local.joint_proj",
                   "local.close_emb", "local.path_gate"],
            "B1": ["local.path_u", "local.path_v", "local.joint_proj"],
            "B2": ["local.path_gate"],
            "B3": ["local.path_u", "local.path_v", "local.joint_proj", "local.path_gate"],
            "B6": ["local.close_emb"],
            "C0": ["local.nbr_emb"],
        }.get(variant, [])
        absent_ok = all(name not in module_names for name in expected_absent)

        invariances: dict[str, float] = {}
        for key in ("global", "atom_scalar", "atom_feat", "bond_feat", "nbr", "path", "path_closure", "path_end"):
            with torch.no_grad():
                y1 = model(_perturb(batch, key))
            invariances[key] = float((y0 - y1).abs().max())

        should_be_invariant = {
            "A1": ["atom_scalar", "atom_feat", "bond_feat", "nbr", "path", "path_closure", "path_end"],
            "A5": ["global"],
            "B0": ["path", "path_closure", "path_end"],
            "B2": [],
            "B6": ["path_closure"],
            "C0": ["nbr"],
            "C2": ["atom_scalar"],
        }.get(variant, [])
        invariance_ok = all(invariances[k] == 0.0 for k in should_be_invariant if k in invariances)

        # for B6 the removed scalar column is index 2: perturb only that column
        if variant == "B6":
            b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            b["atom_scalar"][:, 2] += 5.0
            with torch.no_grad():
                y2 = model(b)
            b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            b2["atom_scalar"][:, 3] += 5.0
            with torch.no_grad():
                y3 = model(b2)
            out["B6_scalar_columns"] = {
                "removed_col2_abs_delta": float((y0 - y2).abs().max()),
                "kept_col3_abs_delta": float((y0 - y3).abs().max()),
            }
            invariance_ok = invariance_ok and out["B6_scalar_columns"]["removed_col2_abs_delta"] == 0.0
            invariance_ok = invariance_ok and out["B6_scalar_columns"]["kept_col3_abs_delta"] > 0.0

        out[variant] = {
            "parameters": params,
            "expected_parameters": C.EXPECTED_PARAMS[variant],
            "parameters_match": params == C.EXPECTED_PARAMS[variant],
            "n_local_cols": len(local_cols),
            "n_rho_cols": len(rho_cols),
            "shared_init_bitwise_equal": shared["all_equal"],
            "shared_checks": shared["n_checked"],
            "expected_absent_modules": expected_absent,
            "absent_modules_ok": absent_ok,
            "invariance_max_abs_delta": invariances,
            "required_invariances": should_be_invariant,
            "required_invariance_ok": invariance_ok,
            "finite_forward": bool(torch.isfinite(y0).all()),
        }
        out[variant]["ok"] = bool(
            out[variant]["parameters_match"]
            and out[variant]["shared_init_bitwise_equal"]
            and out[variant]["absent_modules_ok"]
            and out[variant]["required_invariance_ok"]
            and out[variant]["finite_forward"]
        )
        check(f"variant/{variant}", out[variant]["ok"], out[variant])
    return out


# ---------------------------------------------------------------------------
# 4. B4 / B5 marginal algebra
# ---------------------------------------------------------------------------


def _ref_joint_groups(u: np.ndarray, w: np.ndarray, idx_groups: list[np.ndarray], W: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(idx_groups), u.shape[1]))
    for g, idx in enumerate(idx_groups):
        n = len(idx)
        if n == 0:
            continue
        out[g] = W @ (n * u[idx].mean(0) * w[idx].mean(0)) + n * b
    return out


def _ref_joint_full(u: np.ndarray, w: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (u * w) @ W.T + b


def _ref_content_groups(t: np.ndarray, z: np.ndarray, idx_groups: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(idx_groups), z.shape[1]))
    for g, idx in enumerate(idx_groups):
        n = len(idx)
        if n == 0:
            continue
        out[g] = n * t[idx].mean() * z[idx].mean(0)
    return out


def audit_marginal(view: D.View) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rs = np.random.RandomState(20261102)
    rank, n_groups = 4, 3
    u = rs.randn(6, rank)
    w = rs.randn(6, rank)
    W = rs.randn(rank, rank)
    b = rs.randn(rank)

    Wt = torch.as_tensor(W, dtype=torch.float64)
    bt = torch.as_tensor(b, dtype=torch.float64)

    # ---- n=0, n=1, duplicates, distinct pairings, bias counting ----------
    key = torch.as_tensor([1, 2, 2, 2], dtype=torch.int64)
    got = M.marginal_joint_agg(
        torch.as_tensor(u[1:5], dtype=torch.float64),
        torch.as_tensor(w[1:5], dtype=torch.float64),
        key,
        n_groups,
        Wt,
        bt,
    ).numpy()
    ref = np.zeros((n_groups, rank))
    ref[0] = 0.0
    ref[1] = _ref_joint_full(u[[1]], w[[1]], W, b)[0]
    ref[2] = _ref_joint_groups(u, w, [np.asarray([2, 3, 4])], W, b)[0]
    out["float64_joint"] = {
        "max_abs_diff_vs_reference": float(np.abs(got - ref).max()),
        "n0_is_zero": bool(np.allclose(got[0], 0.0)),
        "n1_equals_full": bool(np.allclose(got[1], _ref_joint_full(u[[1]], w[[1]], W, b)[0])),
        "n3_distinct_equals_marginal_reference": bool(np.allclose(got[2], ref[2])),
        "n3_distinct_differs_from_full_sum": bool(
            not np.allclose(got[2], _ref_joint_full(u[2:5], w[2:5], W, b).sum(0))
        ),
        "duplicate_occurrences_equal_full": bool(
            np.allclose(
                _ref_joint_groups(u, w, [np.asarray([1, 1, 1])], W, b)[0],
                _ref_joint_full(u[[1, 1, 1]], w[[1, 1, 1]], W, b).sum(0),
            )
        ),
        "bias_counts_n_times": bool(
            np.allclose(
                _ref_joint_groups(u, w, [np.asarray([1, 1, 1])], W, b)[0],
                3.0 * (W @ (u[1] * w[1]) + b),
            )
        ),
    }
    check(
        "marginal/float64_joint",
        out["float64_joint"]["max_abs_diff_vs_reference"] <= 1e-12
        and out["float64_joint"]["n0_is_zero"]
        and out["float64_joint"]["n1_equals_full"]
        and out["float64_joint"]["n3_distinct_equals_marginal_reference"]
        and out["float64_joint"]["n3_distinct_differs_from_full_sum"]
        and out["float64_joint"]["duplicate_occurrences_equal_full"]
        and out["float64_joint"]["bias_counts_n_times"],
        out["float64_joint"],
    )

    # ---- content aggregation --------------------------------------------
    t = rs.rand(6)
    z = rs.randn(6, 3)
    got_c = M.marginal_content_agg(
        torch.as_tensor(t[1:5], dtype=torch.float64).unsqueeze(1),
        torch.as_tensor(z[1:5], dtype=torch.float64),
        key,
        n_groups,
    ).numpy()
    ref_c = np.zeros((n_groups, 3))
    ref_c[1] = t[1] * z[1]
    ref_c[2] = 3.0 * t[2:5].mean() * z[2:5].mean(0)
    out["float64_content"] = {
        "max_abs_diff_vs_reference": float(np.abs(got_c - ref_c).max()),
        "n0_is_zero": bool(np.allclose(got_c[0], 0.0)),
        "n1_equals_full": bool(np.allclose(got_c[1], (t[1] * z[1]))),
        "n3_distinct_differs_from_full_sum": bool(
            not np.allclose(got_c[2], (t[2:5, None] * z[2:5]).sum(0))
        ),
    }
    check(
        "marginal/float64_content",
        out["float64_content"]["max_abs_diff_vs_reference"] <= 1e-12
        and out["float64_content"]["n0_is_zero"]
        and out["float64_content"]["n1_equals_full"]
        and out["float64_content"]["n3_distinct_differs_from_full_sum"],
        out["float64_content"],
    )

    # ---- bt2 grouping must not average across different bt2 groups ------
    u2 = np.asarray([[1.0, 0.0], [3.0, 0.0]])
    w2 = np.asarray([[2.0, 0.0], [5.0, 0.0]])
    W2 = np.eye(2)
    b2 = np.zeros(2)
    ref_bt2 = _ref_joint_groups(u2, w2, [np.asarray([0]), np.asarray([1])], W2, b2).sum(0)
    ref_all = _ref_joint_groups(u2, w2, [np.asarray([0, 1])], W2, b2)[0]
    out["bt2_separation"] = {
        "bt2_result": ref_bt2.tolist(),
        "collapsed_result": ref_all.tolist(),
        "differs": bool(not np.allclose(ref_bt2, ref_all)),
        "bt2_equals_full_when_groups_singleton": bool(np.allclose(ref_bt2, _ref_joint_full(u2, w2, W2, b2).sum(0))),
    }
    check("marginal/bt2_separation", out["bt2_separation"]["differs"], out["bt2_separation"])

    # ---- batch offset: marginal must not mix roots across molecules -----
    st = M.build_full_state(view.spec, view.train_norm, 0)
    m4 = M.build_variant(view.spec, view.train_norm, "B4", st, torch.device("cpu")).eval()
    m5 = M.build_variant(view.spec, view.train_norm, "B5", st, torch.device("cpu")).eval()
    rows = list(range(6))
    batch = D.to_device(PREP_DATA.collate(view.valid_cache, rows), torch.device("cpu"))
    with torch.no_grad():
        _, a4 = m4(batch, return_aux=True)
    joint_slice = slice(64, 96)  # [type(32), end(32), joint(32), closure(8)]
    merged_joint = a4["path_agg"].numpy()[:, joint_slice]
    merged_parts: list[np.ndarray] = []
    single_parts: list[np.ndarray] = []
    offset = 0
    for r in rows:
        b1 = D.to_device(PREP_DATA.collate(view.valid_cache, [r]), torch.device("cpu"))
        with torch.no_grad():
            _, a1 = m4(b1, return_aux=True)
        n_atoms = int(b1["atom_feat"].shape[0])
        merged_parts.append(merged_joint[offset : offset + n_atoms])
        single_parts.append(a1["path_agg"].numpy()[:, joint_slice])
        offset += n_atoms
    out["batch_offset"] = {
        "max_abs_diff": float(np.abs(np.concatenate(merged_parts) - np.concatenate(single_parts)).max())
    }
    check("marginal/batch_offset", out["batch_offset"]["max_abs_diff"] <= 1e-6, out["batch_offset"])

    # ---- intervention coverage on real public122 Valid and Test ---------
    coverage: dict[str, Any] = {}
    for split_name, cache in (("valid", view.valid_cache), ("test", D.build_test_cache(view.repeat))):
        cov = _coverage(m4, m5, cache)
        coverage[split_name] = cov
        C.log(
            f"[coverage {split_name}] B4 changes {cov['B4']['molecules_changed_fraction']:.3f} molecules / "
            f"{cov['B4']['roots_changed_fraction']:.3f} roots; "
            f"B5 {cov['B5']['molecules_changed_fraction']:.3f} / {cov['B5']['roots_changed_fraction']:.3f}"
        )
    out["coverage"] = coverage

    # gate probe: path_gate starts at 1, so the content half of B4/B5 is exactly
    # FULL at initialisation.  A deterministic non-constant gate shows the
    # structural opportunity of the content control.
    m4p = M.build_variant(view.spec, view.train_norm, "B4", st, torch.device("cpu")).eval()
    m5p = M.build_variant(view.spec, view.train_norm, "B5", st, torch.device("cpu")).eval()
    with torch.no_grad():
        probe = torch.linspace(-1.0, 1.0, C.N_PATH_CODES).unsqueeze(1)
        m4p.local.path_gate.weight.copy_(probe)
        m5p.local.path_gate.weight.copy_(probe)
    probe_cov = _coverage(m4p, m5p, view.valid_cache)
    out["coverage_gate_probe_valid"] = probe_cov
    C.log(
        f"[coverage gate-probe] B4 joint {probe_cov['B4']['joint_roots_changed_fraction']:.3f} / "
        f"content {probe_cov['B4']['content_roots_changed_fraction']:.3f}; "
        f"B5 joint {probe_cov['B5']['joint_roots_changed_fraction']:.3f} / "
        f"content {probe_cov['B5']['content_roots_changed_fraction']:.3f}"
    )

    # ---- bt2 == path_end63 % 5 on every published repeat ----------------
    bt2_id = True
    for repeat in C.REPEATS:
        src = S.load_lipop(repeat, labels="none")
        cache = D.load_or_build_split_cache(repeat, "valid", src["rows"]["valid"], src["smiles"], view.spec)
        pc = torch.as_tensor(np.asarray(cache["path_cat"]), dtype=torch.int64)
        pe = torch.as_tensor(np.asarray(cache["path_end63"]), dtype=torch.int64)
        bt2_id = bt2_id and bool(torch.equal(pc % C.N_BOND_TYPES, pe % C.N_BOND_TYPES))
    out["bt2_identity_on_data"] = bt2_id
    check("marginal/bt2_identity", bt2_id, None)
    return out


def _coverage(m4: M.DissectLCRC, m5: M.DissectLCRC, cache: dict[str, Any]) -> dict[str, Any]:
    """Fraction of roots/molecules where B4/B5 change the joint/content block.

    The content part of this control is degenerate while ``path_gate`` is the
    frozen all-ones initialisation (n*mean(1)*mean(z) == sum(z)); the audit
    therefore also runs a deterministic non-constant gate probe.
    """
    n_mol = int(cache["n_molecules"])
    rows = list(range(n_mol))
    batch = D.to_device(PREP_DATA.collate(cache, rows), torch.device("cpu"))
    atom_batch = batch["atom_batch"].numpy()
    n_atoms = int(batch["atom_feat"].shape[0])
    spec, norm = m4.spec, _norm_of(m4)
    st = M.build_full_state(spec, norm, 0)
    full = M.build_variant(spec, norm, "A0", st, torch.device("cpu")).eval()
    with torch.no_grad():
        _, aF = full(batch, return_aux=True)
        _, a4 = m4(batch, return_aux=True)
        _, a5 = m5(batch, return_aux=True)
    tol = 1e-5
    out: dict[str, Any] = {}
    for tag, ax in (("B4", a4), ("B5", a5)):
        d_joint = (ax["path_agg"][:, 64:96] - aF["path_agg"][:, 64:96]).abs().max(dim=1).values
        d_content = (ax["content_agg"] - aF["content_agg"]).abs().max(dim=1).values
        d = torch.maximum(d_joint, d_content)
        changed = (d > tol).numpy()
        per_mol = np.zeros(n_mol, dtype=bool)
        for a, m in zip(changed, atom_batch):
            if a:
                per_mol[m] = True
        out[tag] = {
            "roots_total": int(n_atoms),
            "roots_changed": int(changed.sum()),
            "roots_changed_fraction": float(changed.mean()),
            "molecules_total": n_mol,
            "molecules_changed": int(per_mol.sum()),
            "molecules_changed_fraction": float(per_mol.mean()),
            "max_abs_aggregate_delta": float(d.max()),
            "joint_roots_changed": int((d_joint > tol).sum()),
            "joint_roots_changed_fraction": float((d_joint > tol).float().mean()),
            "content_roots_changed": int((d_content > tol).sum()),
            "content_roots_changed_fraction": float((d_content > tol).float().mean()),
            "max_abs_joint_delta": float(d_joint.max()),
            "max_abs_content_delta": float(d_content.max()),
        }

    path_atom = batch["path_atom"].numpy().astype(np.int64)
    path_cat = batch["path_cat"].numpy().astype(np.int64)
    roots = np.bincount(path_atom, minlength=n_atoms)
    out["path_occurrences_per_root"] = {
        "mean": float(roots.mean()),
        "max": int(roots.max()),
        "n_roots_with_ge2": int((roots >= 2).sum()),
        "fraction_roots_with_ge2": float((roots >= 2).mean()),
    }
    if path_atom.size:
        keys = path_atom * C.N_BOND_TYPES + (path_cat % C.N_BOND_TYPES)
        uniq, counts = np.unique(keys, return_counts=True)
        groups_per_root = np.bincount(uniq // C.N_BOND_TYPES, minlength=n_atoms)
        out["B5"]["bt2_groups"] = {
            "n_nonempty_groups": int(uniq.size),
            "group_size_histogram": {str(k): int((counts == k).sum()) for k in sorted(set(counts.tolist()))},
            "roots_with_multiple_bt2_groups": int((groups_per_root >= 2).sum()),
            "fraction_roots_with_multiple_bt2_groups": float((groups_per_root >= 2).mean()),
        }
    else:
        out["B5"]["bt2_groups"] = {"n_nonempty_groups": 0}
    return out


def _norm_of(model: M.DissectLCRC):
    return {
        "global_mean": model.global_mean.detach(),
        "global_scale": model.global_scale.detach(),
        "scalar_mean": model.scalar_mean.detach(),
        "scalar_scale": model.scalar_scale.detach(),
    }


# ---------------------------------------------------------------------------
# 5. gradients
# ---------------------------------------------------------------------------


def audit_gradients(view: D.View) -> dict[str, Any]:
    spec = view.spec
    state = M.build_full_state(spec, view.train_norm, 0)
    batch = D.to_device(PREP_DATA.collate(view.valid_cache, list(range(32))), torch.device("cpu"))
    out: dict[str, Any] = {}
    for variant in C.VARIANT_ORDER:
        model = M.build_variant(spec, view.train_norm, variant, state, torch.device("cpu"))
        model.train()
        y = model(batch).sum()
        model.zero_grad(set_to_none=True)
        y.backward()
        grads = {n: (p.grad.abs().max().item() if p.grad is not None else None) for n, p in model.named_parameters()}
        missing = [n for n, g in grads.items() if g is None]
        zero = [n for n, g in grads.items() if g == 0.0]
        branch_ok = True
        if variant in ("B4", "B5"):
            for key in ("local.path_u.weight", "local.path_v.weight", "local.joint_proj.weight", "local.path_gate.weight"):
                branch_ok = branch_ok and grads.get(key, 0.0) > 0.0
        if variant == "A5":
            branch_ok = branch_ok and grads.get("b0", 0.0) > 0.0
        out[variant] = {
            "n_parameters": len(grads),
            "missing_grad": missing,
            "exactly_zero_grad": zero,
            "marginal_branches_have_gradient": branch_ok,
        }
        out[variant]["ok"] = bool(not missing and branch_ok)
        check(f"grad/{variant}", out[variant]["ok"], out[variant])
    return out


# ---------------------------------------------------------------------------
# 6. selection / denormalisation / no Test in the view
# ---------------------------------------------------------------------------


def audit_selection(view: D.View) -> dict[str, Any]:
    from dissect_train import BestEpochTracker, raw_metrics

    out: dict[str, Any] = {}
    tracker = BestEpochTracker()
    seq = [(1, 0.9), (2, 0.8), (3, 0.8), (4, 0.85), (5, 0.8)]
    for epoch, value in seq:
        tracker.consider(epoch, {"rmse": value})
    out["tie_keeps_earlier"] = tracker.best_epoch == 2 and tracker.best_primary == 0.8
    check("selection/tie_keeps_earlier", out["tie_keeps_earlier"], tracker.best_epoch)

    # denormalisation identity
    y = np.asarray([1.0, 2.0, 3.5], dtype=np.float64)
    tn = {"mean": 2.0, "scale": 1.5}
    std = (y - tn["mean"]) / tn["scale"]
    m = raw_metrics(std, y, tn)
    out["denormalisation"] = {"rmse": m["rmse"], "identity": abs(m["rmse"]) < 1e-12}
    check("selection/denormalisation", out["denormalisation"]["identity"], out["denormalisation"])

    # view must not contain Test targets
    audit = view.audit()
    out["view_no_test"] = {
        "test_targets_loaded": audit["test_targets_loaded"],
        "test_structure_cache_built": audit["test_structure_cache_built"],
        "valid_or_test_used_in_norm": view.train_norm.get("valid_or_test_used"),
        "target_norm_source": view.target_norm.get("source"),
        "has_y_test": hasattr(view, "y_test"),
    }
    check(
        "selection/view_no_test",
        (not audit["test_targets_loaded"])
        and (not audit["test_structure_cache_built"])
        and (view.train_norm.get("valid_or_test_used") is False)
        and (view.target_norm.get("source") == "train_split_only")
        and (not hasattr(view, "y_test")),
        out["view_no_test"],
    )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPU verification before formal training")
    parser.add_argument("--out", default=str(C.ANALYSIS / "audit_cpu.json"))
    args = parser.parse_args(argv)
    view = D.build_view(C.MAIN_REPEAT)
    report: dict[str, Any] = {
        "schema": "lcrc-lipo-dissection-cpu-audit-v1",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "checks": [],
    }
    for section, fn in (
        ("data", lambda: audit_data(view)),
        ("full", lambda: audit_full(view)),
        ("variants", lambda: audit_variants(view)),
        ("marginal", lambda: audit_marginal(view)),
        ("gradients", lambda: audit_gradients(view)),
        ("selection", lambda: audit_selection(view)),
    ):
        C.log(f"[audit] {section}")
        try:
            report[section] = fn()
        except Exception as exc:  # keep going so one section cannot hide another
            FAILS.append(f"{section}: exception {type(exc).__name__}: {exc}")
            report[section] = {"exception": f"{type(exc).__name__}: {exc}"}
    report["checks"] = CHECKS
    report["failures"] = list(FAILS)
    report["ok"] = not FAILS
    C.write_json(Path(args.out), report)
    if FAILS:
        for f in FAILS:
            print(f"FAIL: {f}")
        return 1
    C.log("all CPU audits passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
