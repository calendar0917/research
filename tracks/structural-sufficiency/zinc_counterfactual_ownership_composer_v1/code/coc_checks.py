"""COC-v1 CPU gates C1-C9.

No GPU, no gradient-based training run, no official valid/test, no Locked labels.
The checks are the pre-registered barrier between the frozen design and any GPU
purchase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import coc_common as CC  # noqa: E402
import coc_model as CM  # noqa: E402
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_models as DM  # noqa: E402
import dcc_runner as DR  # noqa: E402

TOL = 1e-6


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy()


def _maxdiff(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def _clone(batch: Mapping) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def atom_relabel(batch: Mapping, perm: np.ndarray) -> dict:
    out = _clone(batch)
    p = torch.as_tensor(perm, dtype=torch.long)
    inv = torch.empty_like(p)
    inv[p] = torch.arange(len(p), dtype=torch.long)
    out["atom_feat"] = batch["atom_feat"][p]
    out["atom_type"] = batch["atom_type"][p]
    out["bond_index"] = inv[batch["bond_index"].long()]
    out["junc_index"] = inv[batch["junc_index"].long()]
    return out


def bond_permutation(batch: Mapping, perm: np.ndarray) -> dict:
    out = _clone(batch)
    p = torch.as_tensor(perm, dtype=torch.long)
    inv = torch.empty_like(p)
    inv[p] = torch.arange(len(p), dtype=torch.long)
    out["bond_feat"] = batch["bond_feat"][p]
    out["bond_type"] = batch["bond_type"][p]
    out["bond_index"] = batch["bond_index"][p]
    out["junc_bond"] = inv[batch["junc_bond"].long()]
    return out


def endpoint_reversal(batch: Mapping) -> dict:
    out = _clone(batch)
    out["bond_index"] = batch["bond_index"][:, [1, 0]]
    out["junc_index"] = batch["junc_index"][:, [2, 1, 0]]
    out["junc_bond"] = batch["junc_bond"][:, [1, 0]]
    return out


def junction_permutation(batch: Mapping, perm: np.ndarray) -> dict:
    out = _clone(batch)
    p = torch.as_tensor(perm, dtype=torch.long)
    for k in ("junc_feat", "junc_batch", "junc_index", "junc_bond"):
        out[k] = batch[k][p]
    return out


def merge_batches(b1: Mapping, b2: Mapping) -> dict:
    n1_atoms = int(b1["atom_feat"].shape[0])
    n1_bonds = int(b1["bond_feat"].shape[0])
    out = {
        "n_graphs": 2,
        "atom_feat": torch.cat([b1["atom_feat"], b2["atom_feat"]], dim=0),
        "atom_type": torch.cat([b1["atom_type"], b2["atom_type"]], dim=0),
        "atom_batch": torch.cat([torch.zeros(n1_atoms, dtype=torch.long),
                                 torch.ones(int(b2["atom_feat"].shape[0]), dtype=torch.long)]),
        "bond_feat": torch.cat([b1["bond_feat"], b2["bond_feat"]], dim=0),
        "bond_type": torch.cat([b1["bond_type"], b2["bond_type"]], dim=0),
        "bond_batch": torch.cat([torch.zeros(n1_bonds, dtype=torch.long),
                                 torch.ones(int(b2["bond_feat"].shape[0]), dtype=torch.long)]),
        "bond_index": torch.cat([b1["bond_index"],
                                 b2["bond_index"] + n1_atoms], dim=0),
        "junc_feat": torch.cat([b1["junc_feat"], b2["junc_feat"]], dim=0),
        "junc_batch": torch.cat([torch.zeros(int(b1["junc_feat"].shape[0]), dtype=torch.long),
                                 torch.ones(int(b2["junc_feat"].shape[0]), dtype=torch.long)]),
        "junc_index": torch.cat([b1["junc_index"],
                                 b2["junc_index"] + n1_atoms], dim=0),
        "junc_bond": torch.cat([b1["junc_bond"],
                                b2["junc_bond"] + n1_bonds], dim=0),
        "global": torch.cat([b1["global"], b2["global"]], dim=0),
    }
    return out


def empty_bond_batch() -> dict:
    return {
        "n_graphs": 1,
        "atom_feat": torch.zeros((1, D.ATOM_FEAT_DIM), dtype=torch.float32),
        "atom_type": torch.zeros(1, dtype=torch.long),
        "atom_batch": torch.zeros(1, dtype=torch.long),
        "bond_feat": torch.zeros((0, D.BOND_FEAT_DIM), dtype=torch.float32),
        "bond_type": torch.zeros(0, dtype=torch.long),
        "bond_batch": torch.zeros(0, dtype=torch.long),
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
        "junc_feat": torch.zeros((0, D.JUNC_FEAT_DIM), dtype=torch.float32),
        "junc_batch": torch.zeros(0, dtype=torch.long),
        "junc_index": torch.zeros((0, 3), dtype=torch.long),
        "junc_bond": torch.zeros((0, 2), dtype=torch.long),
        "global": torch.zeros((1, D.GLOBAL_DIM), dtype=torch.float32),
    }


def _grad_stats(grads) -> dict:
    if grads is None:
        return {"nonzero": 0, "max_abs": 0.0, "finite": True}
    if not isinstance(grads, (tuple, list)):
        grads = (grads,)
    nonzero = 0
    max_abs = 0.0
    finite = True
    for g in grads:
        if g is None:
            continue
        nonzero += int(torch.count_nonzero(g).item())
        max_abs = max(max_abs, float(g.detach().abs().max().item()))
        finite = bool(finite and torch.isfinite(g).all().item())
    return {"nonzero": int(nonzero), "max_abs": float(max_abs), "finite": bool(finite)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(CC.ANALYSIS / "cpu_checks.json"))
    args = p.parse_args(argv)
    device = torch.device(args.device)
    cfg = CC.load_cfg()
    split = CC.split_rows()
    cache = CC.load_train_cache()
    norm = CC.fit_only_norm(cache, split["fit"])

    rows = split["fit"]
    sample_rows = rows[:8]
    batch = DR.collate_p3n(cache, sample_rows, "P3N", valid=True)
    y_sample = torch.as_tensor(np.asarray(cache["labels"], dtype=np.float32)[sample_rows])

    checks: dict = {}
    context: dict = {}

    # ------------------------------------------------------------------ C1
    c1 = {}
    models = {}
    dcc_models = {}
    for seed in (0, 1):
        coc, coc_info = CM.build_initial_state(norm, seed)
        dcc, dcc_info = DM.build_initial_state(norm, seed)
        coc_state = coc.state_dict()
        dcc_state = dcc.state_dict()
        keys_equal = sorted(coc_state) == sorted(dcc_state)
        tensor_equal = all(torch.equal(coc_state[k], dcc_state[k]) for k in coc_state)
        md = max((_maxdiff(_np(coc_state[k]), _np(dcc_state[k])) for k in coc_state), default=0.0)
        params_coc = int(sum(q.numel() for _, q in coc.named_parameters()))
        params_dcc = int(sum(q.numel() for _, q in dcc.named_parameters()))
        c1[str(seed)] = {
            "keys_equal": bool(keys_equal),
            "all_tensors_equal": bool(tensor_equal),
            "max_abs_diff": float(md),
            "parameters_coc": params_coc,
            "parameters_dcc": params_dcc,
            "initial_state_sha256_coc": coc_info["initial_state_sha256"],
            "initial_state_sha256_dcc": dcc_info["initial_state_sha256"],
            "initial_state_matches_frozen": (
                coc_info["initial_state_sha256"]
                == cfg["frozen_l_reference"]["initial_state_sha256"][str(seed)]
            ),
            "ssl_sha256_matches_frozen": (
                coc_info["ssl"]["sha256"]
                == cfg["frozen_l_reference"]["ssl_sha256"][str(seed)]
            ),
            "p3_reference_state_sha256": coc_info["p3_reference_state_sha256"],
        }
        models[seed] = coc
        dcc_models[seed] = dcc
    checks["C1_init_bitwise"] = {
        "pass": all(
            v["keys_equal"]
            and v["all_tensors_equal"]
            and v["max_abs_diff"] == 0.0
            and v["parameters_coc"] == v["parameters_dcc"] == 494193
            and v["initial_state_matches_frozen"]
            and v["ssl_sha256_matches_frozen"]
            for v in c1.values()
        ),
        "seeds": c1,
    }

    # ------------------------------------------------------------------ C2
    coc0 = models[0]
    dcc0 = dcc_models[0]
    coc0.eval()
    dcc0.eval()
    with torch.no_grad():
        pred_coc = coc0(batch)
        pred_dcc = dcc0(batch)
    reader_coc = coc0.reader.lins[0].weight.detach()
    reader_dcc = dcc0.reader.lins[0].weight.detach()
    checks["C2_initial_prediction"] = {
        "pass": bool(_maxdiff(_np(pred_coc), _np(pred_dcc)) <= float(cfg["cpu_gates"]["initial_prediction_tol"])
                     and float(reader_coc[:, 321:513].abs().max()) == 0.0
                     and float(reader_dcc[:, 321:513].abs().max()) == 0.0),
        "max_abs_prediction_diff": _maxdiff(_np(pred_coc), _np(pred_dcc)),
        "tolerance": float(cfg["cpu_gates"]["initial_prediction_tol"]),
        "reader_local_cols_max_abs_coc": float(reader_coc[:, 321:513].abs().max()),
        "reader_local_cols_max_abs_dcc": float(reader_dcc[:, 321:513].abs().max()),
        "n_rows": int(len(sample_rows)),
    }

    # ------------------------------------------------------------------ C3
    base = CC.capture_coc(coc0, batch)
    orig_local_grouping = coc0.local_grouping

    def forced_grouping(w, b, *, organized: bool):
        return coc0.group_to_bonds(w, b)

    try:
        coc0.local_grouping = forced_grouping
        with torch.no_grad():
            forced = CC.capture_coc(coc0, batch)
    finally:
        coc0.local_grouping = orig_local_grouping
    checks["C3_forced_mean_field"] = {
        "pass": bool(
            float(forced["c_loc"].abs().max()) == 0.0
            and _maxdiff(_np(base["c_glob"]), _np(forced["c_glob"])) == 0.0
            and _maxdiff(_np(base["z_glob"]), _np(forced["z_glob"])) == 0.0
            and _maxdiff(_np(base["a_sum"]), _np(forced["a_sum"])) == 0.0
            and _maxdiff(_np(base["a_sq"]), _np(forced["a_sq"])) == 0.0
            and _maxdiff(_np(base["g"]), _np(forced["g"])) == 0.0
            and float(forced["zl"].abs().max()) == 0.0
            and float(forced["zl2"].abs().max()) == 0.0
        ),
        "c_loc_max_abs": float(forced["c_loc"].abs().max()),
        "c_glob_max_diff": _maxdiff(_np(base["c_glob"]), _np(forced["c_glob"])),
        "z_glob_max_diff": _maxdiff(_np(base["z_glob"]), _np(forced["z_glob"])),
        "a_sum_max_diff": _maxdiff(_np(base["a_sum"]), _np(forced["a_sum"])),
        "a_sq_max_diff": _maxdiff(_np(base["a_sq"]), _np(forced["a_sq"])),
        "global33_max_diff": _maxdiff(_np(base["g"]), _np(forced["g"])),
        "zl_max_abs": float(forced["zl"].abs().max()),
        "zl2_max_abs": float(forced["zl2"].abs().max()),
    }

    # ------------------------------------------------------------------ C4
    c4 = None
    rng = np.random.default_rng(20261029)
    for row in rows[:400]:
        b0 = DR.collate_p3n(cache, [int(row)], "P3N", valid=True)
        J = int(b0["junc_feat"].shape[0])
        if J < 2:
            continue
        with torch.no_grad():
            _, b_enc, w = coc0.encoders(b0)
            Wg, Vg, n_b = coc0.group_to_bonds(w, b0)
            Wl0, Vl0, _ = coc0.local_grouping(w, b0, organized=True)
            x0 = coc0.local_input(b_enc, Wl0, Vl0, n_b, b0["bond_feat"])
            xnull = coc0.local_input(b_enc, Wg, Vg, n_b, b0["bond_feat"])
            c0 = coc0.local_composer(x0) - coc0.local_composer(xnull)
        perm = rng.permutation(J)
        if np.array_equal(perm, np.arange(J)):
            perm = np.roll(perm, 1)
        b1 = junction_permutation(b0, perm)
        with torch.no_grad():
            Wl1, Vl1, n1 = coc0.local_grouping(w, b1, organized=True)
            x1 = coc0.local_input(b_enc, Wl1, Vl1, n_b, b1["bond_feat"])
            c1 = coc0.local_composer(x1) - coc0.local_composer(xnull)
        diff = _maxdiff(_np(c0), _np(c1))
        if diff >= float(cfg["cpu_gates"]["correspondence_change_min"]):
            c4 = {
                "row": int(row), "n_junctions": int(J),
                "permutation": perm.tolist(),
                "c_loc_max_abs_diff": float(diff),
                "c_glob_invariance": 0.0,
                "n_b_multiset_preserved": bool(torch.equal(n_b, n1)),
                "w_fixed": True,
                "note": "w is held fixed and only the junction-to-bond row assignment changes; c_glob is unchanged because w and n_b are unchanged",
            }
            break
    checks["C4_correspondence_sensitivity"] = {
        "pass": bool(c4 is not None),
        "threshold": float(cfg["cpu_gates"]["correspondence_change_min"]),
        "found": c4,
    }

    # ------------------------------------------------------------------ C5
    torch.manual_seed(7)
    grads: dict = {}
    coc0.train()
    pred = coc0(batch)
    loss = F.l1_loss(pred.view(-1), y_sample.to(pred.device).view(-1))
    coc0.zero_grad(set_to_none=True)
    loss.backward()
    grads["step1"] = {
        "loss": float(loss.detach()),
        "local_composer_grad_norm": float(sum(
            (p.grad.detach().float().pow(2).sum() for p in coc0.local_composer.parameters()
             if p.grad is not None), torch.tensor(0.0)) ** 0.5),
        "reader_local_column_grad_norm": float(
            coc0.reader.lins[0].weight.grad[:, 321:513].detach().float().pow(2).sum().sqrt()),
        "reader_local_columns_nonzero_grad": int(torch.count_nonzero(
            coc0.reader.lins[0].weight.grad[:, 321:513]).item()),
    }
    opt = torch.optim.AdamW(coc0.parameters(), lr=1e-3, weight_decay=1e-5)
    opt.step()
    opt.zero_grad(set_to_none=True)
    coc0.eval()
    next_rows = rows[8:40]
    batch2 = DR.collate_p3n(cache, next_rows, "P3N", valid=True)
    cap = CC.capture_coc(coc0, batch2)
    local_params = list(coc0.local_composer.parameters())
    g_zl = torch.autograd.grad(cap["zl"].sum(), local_params, retain_graph=True, allow_unused=True)
    g_zl2 = torch.autograd.grad(cap["zl2"].sum(), local_params, retain_graph=True, allow_unused=True)
    g_loc = torch.autograd.grad(cap["c_loc"].sum(), local_params, retain_graph=True, allow_unused=True)
    g_abs = torch.autograd.grad(cap["c_abs"].sum(), local_params, retain_graph=True, allow_unused=True)
    g_abs_wg = torch.autograd.grad(cap["c_abs"].sum(), [cap["Wg"], cap["Vg"]],
                                   retain_graph=True, allow_unused=True)
    g_abs_w = torch.autograd.grad(cap["c_abs"].sum(), cap["w"], retain_graph=True, allow_unused=True)
    g_zl_stats = _grad_stats(g_zl)
    g_zl2_stats = _grad_stats(g_zl2)
    g_loc_stats = _grad_stats(g_loc)
    g_abs_stats = _grad_stats(g_abs)
    g_abs_wg_stats = _grad_stats(g_abs_wg)
    g_abs_w_stats = _grad_stats(g_abs_w)
    # A full loss after the first optimizer step must reach the shared composer.
    y2 = torch.as_tensor(np.asarray(cache["labels"], dtype=np.float32)[next_rows])
    pred2 = coc0(batch2)
    loss2 = F.l1_loss(pred2.view(-1), y2.to(pred2.device).view(-1))
    coc0.zero_grad(set_to_none=True)
    loss2.backward()
    step2 = {
        "loss": float(loss2.detach()),
        "local_composer_grad_norm": float(sum(
            (p.grad.detach().float().pow(2).sum() for p in coc0.local_composer.parameters()
             if p.grad is not None), torch.tensor(0.0)) ** 0.5),
        "reader_local_column_grad_norm": float(
            coc0.reader.lins[0].weight.grad[:, 321:513].detach().float().pow(2).sum().sqrt()),
        "zl_grad": g_zl_stats,
        "zl2_grad": g_zl2_stats,
        "c_loc_grad": g_loc_stats,
        "c_abs_grad": g_abs_stats,
        "c_abs_grad_wrt_Wg_Vg": g_abs_wg_stats,
        "c_abs_grad_wrt_w": g_abs_w_stats,
    }
    step2["finite"] = bool(
        all(np.isfinite(v) for v in (
            step2["loss"], step2["local_composer_grad_norm"],
            step2["reader_local_column_grad_norm"]))
        and g_zl_stats["finite"] and g_zl2_stats["finite"] and g_loc_stats["finite"]
        and g_abs_stats["finite"] and g_abs_wg_stats["finite"] and g_abs_w_stats["finite"]
    )
    grads["step2_after_one_optimizer_step"] = step2
    model_src = (_HERE / "coc_model.py").read_text(encoding="utf-8")
    c_loc_line = [ln for ln in model_src.splitlines() if "c_loc =" in ln]
    checks["C5_gradients"] = {
        "pass": bool(
            grads["step1"]["reader_local_columns_nonzero_grad"] > 0
            and grads["step1"]["local_composer_grad_norm"] == 0.0
            and g_zl_stats["nonzero"] > 0 and g_zl2_stats["nonzero"] > 0
            and g_loc_stats["nonzero"] > 0 and g_abs_stats["nonzero"] > 0
            and g_abs_wg_stats["nonzero"] > 0 and g_abs_w_stats["nonzero"] > 0
            and grads["step2_after_one_optimizer_step"]["local_composer_grad_norm"] > 0.0
            and grads["step2_after_one_optimizer_step"]["finite"]
            and c_loc_line and all("detach" not in ln for ln in c_loc_line)
        ),
        "step1": grads["step1"],
        "step2": grads["step2_after_one_optimizer_step"],
        "c_loc_source_lines": c_loc_line,
        "detach_absent": bool(c_loc_line and all("detach" not in ln for ln in c_loc_line)),
    }

    # ------------------------------------------------------------------ C6
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    sizes = ap[1:] - ap[:-1]
    inv_rows = []
    for q in (8, 17, 25, 33):
        found = np.where(sizes == q)[0]
        if len(found):
            inv_rows.append(int(found[0]))
    if len(inv_rows) < 2:
        inv_rows = [int(r) for r in rows[:4]]
    inv = {}
    for r in inv_rows:
        b0 = DR.collate_p3n(cache, [r], "P3N", valid=True)
        with torch.no_grad():
            p0 = float(coc0(b0).detach().cpu().numpy()[0])
            n_atoms = int(b0["atom_feat"].shape[0])
            n_bonds = int(b0["bond_feat"].shape[0])
            n_junc = int(b0["junc_feat"].shape[0])
            pa = rng.permutation(n_atoms)
            pb = rng.permutation(n_bonds)
            pj = rng.permutation(n_junc) if n_junc > 1 else np.arange(n_junc)
            variants = {
                "atom_relabel": atom_relabel(b0, pa),
                "bond_permutation": bond_permutation(b0, pb),
                "endpoint_reversal": endpoint_reversal(b0),
                "junction_permutation": junction_permutation(b0, pj),
            }
            rec = {"n_atoms": n_atoms, "n_bonds": n_bonds, "n_junctions": n_junc,
                   "pred_base": p0, "variants": {}}
            for k, v in variants.items():
                pv = float(coc0(v).detach().cpu().numpy()[0])
                rec["variants"][k] = {"pred": pv, "abs_delta": abs(pv - p0)}
            comp = DR.collate_p3n(cache, [int(inv_rows[(inv_rows.index(r) + 1) % len(inv_rows)])],
                                  "P3N", valid=True)
            merged = merge_batches(b0, comp)
            pm = coc0(merged)
            rec["batch_regrouping"] = {
                "pred_row0": float(pm[0]), "pred_row1": float(pm[1]),
                "abs_delta_row0": abs(float(pm[0]) - p0),
                "abs_delta_row1_vs_alone": abs(float(pm[1]) - float(coc0(comp)[0])),
            }
            rec["pass"] = bool(
                all(v["abs_delta"] <= TOL for v in rec["variants"].values())
                and rec["batch_regrouping"]["abs_delta_row0"] <= TOL
                and rec["batch_regrouping"]["abs_delta_row1_vs_alone"] <= TOL
            )
            inv[str(r)] = rec
    checks["C6_invariance"] = {
        "pass": bool(all(v["pass"] for v in inv.values())),
        "tolerance": TOL, "rows": inv,
    }

    # ------------------------------------------------------------------ C7
    b0 = DR.collate_p3n(cache, [int(inv_rows[0])], "P3N", valid=True)
    b1 = DR.collate_p3n(cache, [int(inv_rows[1])], "P3N", valid=True)
    merged = merge_batches(b0, b1)
    with torch.no_grad():
        p0 = coc0(b0)
        p1 = coc0(b1)
        pm = coc0(merged)
        cap0 = CC.capture_coc(coc0, b0)
        cap1 = CC.capture_coc(coc0, b1)
        capm = CC.capture_coc(coc0, merged)
    n0 = int(cap0["c_loc"].shape[0])
    c7_leak = {
        "pred_row0": _maxdiff(_np(pm[0]), _np(p0[0])),
        "pred_row1": _maxdiff(_np(pm[1]), _np(p1[0])),
        "c_loc_row0": _maxdiff(_np(capm["c_loc"][:n0]), _np(cap0["c_loc"])),
        "c_loc_row1": _maxdiff(_np(capm["c_loc"][n0:]), _np(cap1["c_loc"])),
    }
    empty = empty_bond_batch()
    with torch.no_grad():
        pe = coc0(empty)
    checks["C7_no_leak_and_empty"] = {
        "pass": bool(all(v <= TOL for v in c7_leak.values())
                     and torch.isfinite(pe).all().item()),
        "tolerance": TOL,
        "leak_max_diffs": c7_leak,
        "empty_bond_prediction": float(pe[0]) if pe.numel() else None,
        "empty_bond_finite": bool(torch.isfinite(pe).all().item()),
    }

    # ------------------------------------------------------------------ C8
    diag_rows = CC.select_diagnostic_rows(rows, n=int(cfg["cpu_gates"]["diagnostic_rows"]),
                                          seed=int(cfg["cpu_gates"]["diagnostic_rows_seed"]))
    c8 = {}
    for seed in (0, 1):
        cap = CC.representation_capture(models[seed], cache, diag_rows, device=device)
        diag = CC.difference_diagnostics(cap)
        gate = CC.representation_gate(diag, cfg, post_training=False)
        c8[str(seed)] = {"diagnostic": diag, "gate": gate}
    checks["C8_representation_diagnostics"] = {
        "pass": bool(all(v["gate"]["pass"] for v in c8.values())),
        "rows": int(len(diag_rows)),
        "rows_sha256": CC.sha256_array(diag_rows),
        "rows_seed": int(cfg["cpu_gates"]["diagnostic_rows_seed"]),
        "seeds": c8,
    }

    # ------------------------------------------------------------------ C9
    steps_per_epoch = int(np.ceil(len(split["fit"]) / int(cfg["recipe"]["batch_size"])))
    epochs_run = int(np.ceil(int(cfg["recipe"]["total_updates"]) / steps_per_epoch))
    param_count = int(sum(q.numel() for _, q in coc0.named_parameters()))
    opt_audit = torch.optim.AdamW(coc0.parameters(), lr=1e-3)
    opt_params = int(sum(q.numel() for group in opt_audit.param_groups for q in group["params"]))
    target = torch.as_tensor(np.asarray(cache["labels"], dtype=np.float32)[sample_rows])
    with torch.no_grad():
        pred = coc0(batch)
    manual_l1 = float(torch.mean(torch.abs(pred.view(-1) - target.view(-1))))
    torch_l1 = float(F.l1_loss(pred.view(-1), target.view(-1)))
    scan = CC.assert_no_official_test_tokens()
    loaded = CC.loaded_paths_audit()
    forbidden_suffixes = ("ad_" + "valid.pt", "ad_" + "test.pt")
    forbidden_loaded = [v["path"] for v in loaded["loaded"]
                        if v["path"].endswith(forbidden_suffixes)]
    gen = torch.Generator().manual_seed(0 + 101)
    order0 = torch.randperm(len(split["fit"]), generator=gen).numpy()
    first_rows = split["fit"][order0[:int(cfg["recipe"]["batch_size"])]]
    cache_file_sha = CC.sha256_file(CC.REPO / cfg["data"]["train_cache"])
    cache_tensor_sha = CC.cache_tensor_sha256(cache)
    checks["C9_parameter_data_recipe_barrier"] = {
        "pass": bool(
            param_count == int(cfg["model"]["parameter_count_expected"])
            and opt_params == param_count
            and abs(manual_l1 - torch_l1) <= 1e-7
            and steps_per_epoch == int(cfg["recipe"]["steps_per_epoch"]) == 63
            and epochs_run == int(cfg["recipe"]["epochs_run"]) == 753
            and int(cfg["recipe"]["total_updates"]) == 47400
            and int(cfg["recipe"]["warmup_steps"]) == 500
            and float(cfg["recipe"]["dropout"]) == 0.0
            and int(cfg["recipe"]["batch_size"]) == 128
            and cache_file_sha in cfg["data"]["train_cache_sha256_accepted"]
            and cache_tensor_sha == cfg["data"]["train_cache_tensor_sha256"]
            and not scan["forbidden_hits"]
            and not forbidden_loaded
        ),
        "parameters_named": param_count,
        "parameters_optimizer": opt_params,
        "parameter_count_expected": int(cfg["model"]["parameter_count_expected"]),
        "manual_raw_y_l1": manual_l1,
        "torch_raw_y_l1": torch_l1,
        "steps_per_epoch": steps_per_epoch,
        "epochs_run": epochs_run,
        "total_updates": int(cfg["recipe"]["total_updates"]),
        "warmup_steps": int(cfg["recipe"]["warmup_steps"]),
        "batch_size": int(cfg["recipe"]["batch_size"]),
        "dropout": float(cfg["recipe"]["dropout"]),
        "split_counts": {"fit": len(split["fit"]), "monitor": len(split["monitor"]),
                         "locked": len(split["locked"])},
        "split_hashes": split["hashes"],
        "split_sha256": split["split_sha256"],
        "train_cache_file_sha256": cache_file_sha,
        "train_cache_tensor_sha256": cache_tensor_sha,
        "first_batch_rows_sha256": CC.sha256_array(first_rows),
        "first_batch_rows_first5": first_rows[:5].tolist(),
        "dcb_train_generator_line": "torch.Generator().manual_seed(int(seed) + 101)",
        "static_scan": scan,
        "loaded_paths": loaded,
        "forbidden_loaded": forbidden_loaded,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": False,
    }

    all_pass = all(v["pass"] for v in checks.values())
    payload = {
        "protocol_id": cfg["protocol_id"],
        "round": cfg["round"],
        "git_commit": C.git_commit(),
        "git_dirty_tracked": CC.code_dirty(),
        "dirty_scope": "code/config/PROTOCOL/DESIGN only; regenerated analysis outputs are covered by their sha256 fields",
        "config_sha256": CC.sha256_file(CC.CFG_PATH),
        "protocol_sha256": CC.sha256_file(CC.ROUND / "PROTOCOL.md"),
        "code_sha256": {p.name: CC.sha256_file(p) for p in sorted(_HERE.glob("*.py"))},
        "all_pass": bool(all_pass),
        "checks": checks,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": False,
        "loaded_paths": loaded,
    }
    CC.write_json(Path(args.out), CC.jsonable(payload))
    C.log(json.dumps({
        "all_pass": all_pass,
        "checks": {k: v["pass"] for k, v in checks.items()},
        "c8": {k: v["gate"]["checks"] for k, v in c8.items()},
        "params": param_count,
    }, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
