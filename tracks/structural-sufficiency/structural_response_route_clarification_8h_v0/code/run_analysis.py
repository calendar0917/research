#!/usr/bin/env python
"""Phase B / C analysis: molecule tables, strict-edit response tables, contrasts.

Runs only after every D0 checkpoint is frozen (the checkpoint SHA-256 values are
hashed into ``checkpoints/manifest.json`` by :mod:`freeze_checkpoints`).  Labels
of the pair endpoints are attached here for the first time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import sr_common as C
import sr_context as SX
import sr_fgcr as SF
import sr_gine as SG
import sr_pairs as SP
import sr_resp as SR

B_ROOT = C.RESULTS_DIR / "B"
SUBSETS = ["dev_dev_all", "dev_dev_supported119", "dev_dev_strong97",
           "dev_dev_unsupported157"]
DIRECTION_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# predict-from-checkpoint
# ---------------------------------------------------------------------------


def _fgcr_predictor(ckpt_path: Path, ctx, meta: Mapping[str, Any], device) -> Callable:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    summary = payload["summary"]
    h_mode = meta["h_mode"]
    eta_mode = meta["eta_mode"]
    if h_mode not in ("IND", "FULL") or eta_mode not in ("STAR", "FULL"):
        raise RuntimeError(f"checkpoint does not declare a response mode: {meta}")
    model, _ = SF.build_variant(ctx.bundle.atom_dims, ctx.bundle.bond_dims,
                                h_mode=h_mode, eta_mode=eta_mode, init_state=state,
                                device=device)
    model.eval()

    def predict(rows: np.ndarray) -> np.ndarray:
        return SF.predict_std(model, ctx.bundle, ctx.plan, ctx.row_to_plan, rows,
                              batch_size=256, device=device)

    return predict


def _resp_predictor(ckpt_path: Path, ctx, meta: Mapping[str, Any], fit_rows, scaler,
                    device, *, universe=None) -> Callable:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    operator = meta["operator"]
    if operator not in ("PRODUCT", "FUSION_MLP"):
        raise RuntimeError(f"checkpoint does not declare a RESP operator: {meta}")
    rows_for_bundle = fit_rows if universe is None else universe
    bundle = SR.RespBundle(ctx.dataset, rows_for_bundle, scaler, split_name="train")
    dims = bundle.dims
    model = SR.build_model(dims, SR.canonical_init(dims), operator, C.RESP_RECIPE)
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    model.to(device)
    model.eval()
    subset = bundle.fit

    def predict(rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        return SR.predict_std(model, bundle, subset, subset.local(rows), device,
                              batch_size=256)

    return predict


def _gine_predictor(ckpt_path: Path, ctx, meta: Mapping[str, Any], device) -> Callable:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    model = SG.build_model(ctx.bundle.gine_categories, state).to(device)
    model.eval()

    def predict(rows: np.ndarray) -> np.ndarray:
        return SG.predict_std(model, ctx.bundle, np.asarray(rows, dtype=np.int64), device,
                              batch_size=256)

    return predict


def make_predictor(ckpt_path: Path, ctx, meta: Mapping[str, Any], fit_rows, scaler,
                   device, *, universe=None) -> Callable:
    family = meta["family"]
    if family == "FGCR":
        return _fgcr_predictor(ckpt_path, ctx, meta, device)
    if family == "RESP":
        return _resp_predictor(ckpt_path, ctx, meta, fit_rows, scaler, device,
                               universe=universe)
    if family == "GINE":
        return _gine_predictor(ckpt_path, ctx, meta, device)
    raise KeyError(family)


# ---------------------------------------------------------------------------
# edit metrics
# ---------------------------------------------------------------------------


def edit_metrics(delta_hat: np.ndarray, delta_y: np.ndarray, component: np.ndarray,
                 *, direction_threshold: float = DIRECTION_THRESHOLD) -> dict[str, Any]:
    err = np.abs(delta_hat - delta_y)
    sq = (delta_hat - delta_y) ** 2
    ce, n_comp = C.component_equal(err, component)
    ce_rmse, _ = C.component_equal(np.sqrt(sq), component)
    big = np.abs(delta_y) >= direction_threshold
    out = {
        "n_pairs": int(delta_y.size),
        "n_components": int(n_comp),
        "ce_edit_mae": float(ce),
        "pair_edit_mae": float(err.mean()) if err.size else float("nan"),
        "pair_edit_rmse": float(np.sqrt(sq.mean())) if sq.size else float("nan"),
        "direction_n": int(big.sum()),
        "direction_accuracy": (float((np.sign(delta_hat[big]) == np.sign(delta_y[big])).mean())
                               if big.any() else float("nan")),
        "mean_delta_hat": float(delta_hat.mean()) if delta_hat.size else float("nan"),
        "mean_delta_y": float(delta_y.mean()) if delta_y.size else float("nan"),
    }
    return out


def transform_mean_reference(index: SP.PairIndex, labels: np.ndarray,
                            delta_y: np.ndarray, component: np.ndarray) -> np.ndarray:
    """``TRAIN_TRANSFORM_MEAN``: Fit-Fit deltas of the same transform, excluding the
    evaluated pair's own 2133-graph component; NA where no other component supports."""
    fit_mask = index.fit_pair_mask
    table: dict[str, dict[int, list[float]]] = {}
    for i in np.nonzero(fit_mask)[0].tolist():
        table.setdefault(index.rows[i]["transform_id"], {}).setdefault(
            int(component[i]), []).append(float(delta_y[i]))
    out = np.full(delta_y.size, np.nan)
    for i, row in enumerate(index.rows):
        per_comp = table.get(row["transform_id"])
        if not per_comp:
            continue
        values = [float(np.mean(v)) for c, v in per_comp.items() if c != int(component[i])]
        if not values:
            continue
        out[i] = float(np.mean(values))
    return out


def _paired_component(delta_a: np.ndarray, delta_b: np.ndarray, delta_y: np.ndarray,
                      component: np.ndarray, *, seed: int) -> dict[str, Any]:
    err_a = np.abs(delta_a - delta_y)
    err_b = np.abs(delta_b - delta_y)
    out = C.paired_component_bootstrap(err_a, err_b, component, seed=seed)
    out["drop_worst_component"] = C.drop_worst_component(err_a, err_b, component)
    out["delta_ce_pair"] = float(out["delta_mean"])
    ce_a, _ = C.component_equal(err_a, component)
    ce_b, _ = C.component_equal(err_b, component)
    out["ce_a"] = float(ce_a)
    out["ce_b"] = float(ce_b)
    out["ce_delta"] = float(ce_b - ce_a)
    return out


def _decompose(e_a: np.ndarray, e_b: np.ndarray, component: np.ndarray) -> dict[str, Any]:
    """Raw-residual decomposition of E[(e_B - e_A)^2]."""

    def block(weights: str) -> dict[str, float]:
        if weights == "pair_equal":
            ea, eb = e_a, e_b
        else:
            uniq, inv = np.unique(component, return_inverse=True)
            cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
            w = 1.0 / np.maximum(cnt[inv], 1.0) / uniq.size
            ea, eb = e_a * np.sqrt(w), e_b * np.sqrt(w)
        ea2 = float((ea ** 2).sum())
        eb2 = float((eb ** 2).sum())
        cross = float((ea * eb).sum())
        n_eff = float((ea ** 2 + eb ** 2).sum() / max(1e-30, np.mean(ea ** 2 + eb ** 2)))
        return {
            "E_eA2": ea2 / n_eff, "E_eB2": eb2 / n_eff,
            "E_eA_eB": cross / n_eff,
            "E_eB_minus_eA_sq": eb2 / n_eff + ea2 / n_eff - 2.0 * cross / n_eff,
            "mean_bias_delta": float(ea.mean() - eb.mean()),
            "centered_covariance": float(((ea - ea.mean()) * (eb - eb.mean())).mean()),
        }

    return {"pair_equal": block("pair_equal"), "component_equal": block("component_equal"),
            "n_pairs": int(e_a.size)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", default="all")
    args = parser.parse_args()
    device = args.device

    ctx = SX.build_lipo(include_valid=False)
    fit_rows = np.asarray(ctx.d0["fit_rows"], dtype=np.int64)
    dev_rows = np.asarray(ctx.d0["dev_rows"], dtype=np.int64)
    ctx.bundle.set_scaler(fit_rows)
    fit_scaler = dict(ctx.bundle.scaler)
    index = ctx.pairs
    labels = ctx.bundle.labels.astype(np.float64)
    lab = SP.attach_labels(index, labels)
    delta_y_all = lab["delta_y"]
    component = index.component_ids
    masks = SP.subset_masks(index)
    dd = index.dev_dev_index

    pair_union = np.union1d(lab["row_a"], lab["row_b"])
    resp_universe = np.union1d(fit_rows, dev_rows).astype(np.int64)  # RESP view subset
    print(f"pair row union: {pair_union.size} molecules", flush=True)

    # ---------------- frozen checkpoint manifest ----------------------
    manifest_path = C.CKPT_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("run freeze_checkpoints.py before opening the strict-edit evaluation")
    manifest = C.read_json(manifest_path)
    manifest_sha = {e["path"]: e["sha256"] for e in manifest["entries"]}
    for path, expected in manifest_sha.items():
        actual = C.sha256_file(C.REPO_ROOT / path)
        if actual != expected:
            raise SystemExit(f"checkpoint {path} changed after the freeze "
                             f"({actual} != {expected})")
    print(f"checkpoint manifest verified: {len(manifest_sha)} files", flush=True)

    # ---------------- checkpoints -------------------------------------
    frozen = C.read_json(C.ANALYSIS_DIR / "branch_decision.json")
    rows_family = frozen["d0_rows"]
    units: list[tuple[str, str]] = []
    for key in ("1", "2", "G"):
        row = rows_family[key]
        for target in ("BASE", "ENDPOINT", "DELTA"):
            units.append((row, target))

    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    molecule_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for row, target in units:
        out_dir = B_ROOT / row / target
        if not (out_dir / "summary.json").is_file():
            print(f"missing unit {row}/{target}", flush=True)
            continue
        summary = C.read_json(out_dir / "summary.json")
        for ckpt in ("best", "final"):
            path = out_dir / f"{ckpt}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            meta = dict(payload["extra"])
            meta.setdefault("family", summary["family"])
            predictor = make_predictor(path, ctx, meta, fit_rows, fit_scaler, device,
                                       universe=resp_universe)
            std_union = predictor(pair_union)
            dev_std = predictor(dev_rows)
            pred = {
                "std_union": std_union,
                "union_rows": pair_union,
                "dev_std": dev_std,
            }
            predictions[(row, target, ckpt)] = pred
            checkpoint_rows.append({
                "row": row, "target": target, "checkpoint": ckpt,
                "path": str(path.relative_to(C.REPO_ROOT)),
                "sha256": C.sha256_file(path),
                "best_epoch": int(summary["best_epoch"]),
                "dev_rmse": float(summary["best_eval_rmse"] if ckpt == "best"
                                  else summary["final_eval_rmse"]),
            })
            dev_raw = dev_std * fit_scaler["std"] + fit_scaler["mean"]
            molecule_rows.append({
                "row": row, "target": target, "checkpoint": ckpt,
                "split": "dev504", "n": int(dev_rows.size),
                "rmse": C.rmse(dev_raw, labels[dev_rows]),
                "mae": C.mae(dev_raw, labels[dev_rows]),
                "best_epoch": int(summary["best_epoch"]),
                "last20_rmse": float(summary["last20_eval_rmse"]),
                "last20_mae": float(summary["last20_eval_mae"]),
                "final_rmse": float(summary["final_eval_rmse"]),
                "n_parameters": int(summary["n_parameters"]),
                "seconds": float(summary["seconds"]),
            })
            np.savez_compressed(
                C.PRED_DIR / f"edit_response__{row}__{target}__{ckpt}.npz",
                pair_id=np.asarray([r["pair_id"] for r in index.rows]),
                row_a=lab["row_a"], row_b=lab["row_b"],
                component_id=component, cell=index.cell.astype(str),
                transform_id=np.asarray([r["transform_id"] for r in index.rows]),
                core_uid=np.asarray([r["core_uid"] for r in index.rows]),
                core_size=np.asarray([r["core_size"] for r in index.rows]),
                cut_bond_type=np.asarray([r["cut_bond_type"] for r in index.rows]),
                s_key_a=np.asarray([r["s_key_a"] for r in index.rows]),
                s_key_b=np.asarray([r["s_key_b"] for r in index.rows]),
                s_size_a=np.asarray([r["s_size_a"] for r in index.rows]),
                s_size_b=np.asarray([r["s_size_b"] for r in index.rows]),
                support_other_component=index.support_other_component.astype(np.int8),
                support_two_components=index.support_two_components.astype(np.int8),
                delta_y=delta_y_all,
                delta_hat=_delta_hat(pred, lab, pair_union, index, fit_scaler),
                dev_row_ids=dev_rows, dev_y_true=labels[dev_rows],
                dev_y_pred=dev_raw, fit_scaler_mean=np.float64(fit_scaler["mean"]),
                fit_scaler_std=np.float64(fit_scaler["std"]),
            )
        print(f"scored {row}/{target}", flush=True)

    C.write_json(C.ANALYSIS_DIR / "checkpoint_hashes.json", checkpoint_rows)
    C.write_csv(C.ANALYSIS_DIR / "main_molecule_table.csv", molecule_rows)

    # ---------------- edit tables -------------------------------------
    edit_rows: list[dict[str, Any]] = []
    delta_cache: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    for (row, target, ckpt), pred in predictions.items():
        dh_all = _delta_hat(pred, lab, pair_union, index, fit_scaler)
        for subset_name in SUBSETS:
            sel = dd[masks[subset_name]]
            m = edit_metrics(dh_all[sel], delta_y_all[sel], component[sel])
            m.update({"row": row, "target": target, "checkpoint": ckpt,
                      "subset": subset_name})
            n_transforms = len({index.rows[i]["transform_id"] for i in sel.tolist()})
            n_molecules = len(set(lab["row_a"][sel].tolist()) | set(lab["row_b"][sel].tolist()))
            m["n_transforms"] = int(n_transforms)
            m["n_molecules"] = int(n_molecules)
            edit_rows.append(m)
            delta_cache[(row, target, ckpt, subset_name)] = {
                "delta_hat": dh_all[sel], "delta_y": delta_y_all[sel],
                "component": component[sel]}

    # ZERO and TRAIN_TRANSFORM_MEAN references on the same pair sets
    tmean_all = transform_mean_reference(index, labels, delta_y_all, component)
    for subset_name in SUBSETS:
        sel = dd[masks[subset_name]]
        zero = np.zeros(sel.size)
        m = edit_metrics(zero, delta_y_all[sel], component[sel])
        m.update({"row": "ZERO", "target": "-", "checkpoint": "-", "subset": subset_name,
                  "n_transforms": len({index.rows[i]["transform_id"] for i in sel.tolist()}),
                  "n_molecules": len(set(lab["row_a"][sel].tolist())
                                     | set(lab["row_b"][sel].tolist()))})
        edit_rows.append(m)
        tm = tmean_all[sel]
        ok = ~np.isnan(tm)
        if ok.any():
            m = edit_metrics(tm[ok], delta_y_all[sel][ok], component[sel][ok])
            m.update({"row": "TRAIN_TRANSFORM_MEAN", "target": "-", "checkpoint": "-",
                      "subset": subset_name,
                      "n_pairs_supported": int(ok.sum()),
                      "n_transforms": len({index.rows[i]["transform_id"]
                                           for i in sel[ok].tolist()}),
                      "n_molecules": len(set(lab["row_a"][sel][ok].tolist())
                                         | set(lab["row_b"][sel][ok].tolist())),
                      "note": "defined only on pairs with another-component support"})
            edit_rows.append(m)

    # Cross238 (secondary) and Fit-Fit (training fit) for the nine units
    extra_sets = {"cross238": index.fit_dev_index, "fit_fit1619": index.fit_fit_index}
    for (row, target, ckpt), pred in predictions.items():
        dh_all = _delta_hat(pred, lab, pair_union, index, fit_scaler)
        for name, sel in extra_sets.items():
            m = edit_metrics(dh_all[sel], delta_y_all[sel], component[sel])
            m.update({"row": row, "target": target, "checkpoint": ckpt, "subset": name,
                      "n_transforms": len({index.rows[i]["transform_id"] for i in sel.tolist()}),
                      "n_molecules": len(set(lab["row_a"][sel].tolist())
                                         | set(lab["row_b"][sel].tolist()))})
            edit_rows.append(m)

    C.write_csv(C.ANALYSIS_DIR / "edit_response_table.csv", edit_rows)

    # ---------------- paired contrasts --------------------------------
    contrast_rows: list[dict[str, Any]] = []
    r1, r2 = rows_family["1"], rows_family["2"]
    pairs_to_compare = [
        (f"{r1}_D_vs_{r2}_D", (r1, "DELTA"), (r2, "DELTA")),
        (f"{r1}_D_vs_E", (r1, "DELTA"), (r1, "ENDPOINT")),
        (f"{r1}_D_vs_BASE", (r1, "DELTA"), (r1, "BASE")),
        (f"{r1}_E_vs_BASE", (r1, "ENDPOINT"), (r1, "BASE")),
        (f"{r2}_D_vs_E", (r2, "DELTA"), (r2, "ENDPOINT")),
        (f"{r2}_D_vs_BASE", (r2, "DELTA"), (r2, "BASE")),
        ("GINE_D_vs_E", ("GINE_G0", "DELTA"), ("GINE_G0", "ENDPOINT")),
        ("GINE_D_vs_BASE", ("GINE_G0", "DELTA"), ("GINE_G0", "BASE")),
        ("GINE_E_vs_BASE", ("GINE_G0", "ENDPOINT"), ("GINE_G0", "BASE")),
        (f"{r1}_D_vs_GINE_D", (r1, "DELTA"), ("GINE_G0", "DELTA")),
        (f"{r2}_D_vs_GINE_D", (r2, "DELTA"), ("GINE_G0", "DELTA")),
        (f"{r1}_B_vs_GINE_B", (r1, "BASE"), ("GINE_G0", "BASE")),
        (f"{r2}_B_vs_GINE_B", (r2, "BASE"), ("GINE_G0", "BASE")),
        (f"{r1}_D_vs_TRAIN_TRANSFORM_MEAN", (r1, "DELTA"), None),
        ("GINE_D_vs_TRAIN_TRANSFORM_MEAN", ("GINE_G0", "DELTA"), None),
    ]
    for ckpt in ("best", "final"):
        for name, a, b in pairs_to_compare:
            for subset_name in SUBSETS:
                key_a = (a[0], a[1], ckpt, subset_name)
                if key_a not in delta_cache:
                    continue
                da = delta_cache[key_a]
                sel = dd[masks[subset_name]]
                if b is None:
                    tm = tmean_all[sel]
                    ok = ~np.isnan(tm)
                    if not ok.any():
                        continue
                    res = _paired_component(da["delta_hat"][ok], tm[ok], da["delta_y"][ok],
                                            da["component"][ok],
                                            seed=C.BOOT_SEED)
                else:
                    key_b = (b[0], b[1], ckpt, subset_name)
                    if key_b not in delta_cache:
                        continue
                    db = delta_cache[key_b]
                    res = _paired_component(da["delta_hat"], db["delta_hat"], da["delta_y"],
                                            da["component"], seed=C.BOOT_SEED)
                row_out = {"contrast": name, "checkpoint": ckpt, "subset": subset_name,
                           "a": subset_label(a), "b": subset_label(b)}
                row_out.update({k: v for k, v in res.items() if not isinstance(v, dict)})
                row_out["drop_worst_component"] = json.dumps(C.jsonable(
                    res["drop_worst_component"]))
                contrast_rows.append(row_out)

    C.write_csv(C.ANALYSIS_DIR / "paired_contrasts.csv", contrast_rows)

    # ---------------- error decomposition -----------------------------
    decomposition = {}
    for ckpt in ("best", "final"):
        for row, target in ((r1, "DELTA"), (r1, "ENDPOINT"), (r2, "DELTA"),
                            ("GINE_G0", "DELTA"), ("GINE_G0", "ENDPOINT")):
            key = (row, target, ckpt, "dev_dev_all")
            if key not in delta_cache:
                continue
            pred_std = predictions[(row, target, ckpt)]["std_union"]
            pos = np.searchsorted(pair_union, lab["row_a"])
            pos_b = np.searchsorted(pair_union, lab["row_b"])
            f_a = pred_std[pos] * fit_scaler["std"] + fit_scaler["mean"]
            f_b = pred_std[pos_b] * fit_scaler["std"] + fit_scaler["mean"]
            sel = dd[masks["dev_dev_all"]]
            e_a = f_a[sel] - labels[lab["row_a"][sel]]
            e_b = f_b[sel] - labels[lab["row_b"][sel]]
            decomposition[f"{row}|{target}|{ckpt}"] = _decompose(e_a, e_b, component[sel])
    C.write_json(C.ANALYSIS_DIR / "error_decomposition.json", decomposition)

    # ---------------- response cards ----------------------------------
    cards = _response_cards(index, lab, delta_y_all, component, delta_cache, masks,
                            rows_family, predictions, pair_union, fit_scaler, ckpt="best")
    C.write_json(C.ANALYSIS_DIR / "response_cards.json", cards)

    support = {
        "generated_at": C.utc_stamp(),
        "pair_index": index.meta,
        "subsets": {name: {"n_pairs": int(masks[name].sum()),
                           "n_components": int(len(np.unique(component[dd[masks[name]]]))),
                           "n_transforms": int(len({index.rows[i]["transform_id"]
                                                    for i in dd[masks[name]].tolist()}))}
                    for name in SUBSETS},
        "orientation_sign_all_plus_one": bool(np.all(index.orientation_sign == 1)),
        "fit_only_schedule": {"steps": 9000, "pairs_per_step": 64, "seed": C.PAIR_SEED},
        "fit_scaler": fit_scaler,
        "dev_rows_sha256": C.sha256_array(dev_rows),
        "fit_rows_sha256": C.sha256_array(fit_rows),
        "train_transform_mean_supported_pairs": int((~np.isnan(tmean_all)).sum()),
        "n_molecules_in_pair_union": int(pair_union.size),
        "test_untouched": ctx.audit["test_statement"],
    }
    C.write_json(C.ANALYSIS_DIR / "support.json", support)
    print(json.dumps({"n_edit_rows": len(edit_rows), "n_contrasts": len(contrast_rows),
                      "n_molecule_rows": len(molecule_rows)}, indent=1))
    return 0


def subset_label(a) -> str:
    return "TRAIN_TRANSFORM_MEAN" if a is None else f"{a[0]}|{a[1]}"


def _delta_hat(pred: Mapping[str, Any], lab: Mapping[str, np.ndarray],
               pair_union: np.ndarray, index: SP.PairIndex,
               scaler: Mapping[str, float]) -> np.ndarray:
    std = pred["std_union"]
    pos_a = np.searchsorted(pair_union, lab["row_a"])
    pos_b = np.searchsorted(pair_union, lab["row_b"])
    f_a = std[pos_a] * float(scaler["std"]) + float(scaler["mean"])
    f_b = std[pos_b] * float(scaler["std"]) + float(scaler["mean"])
    return f_b - f_a


def _response_cards(index, lab, delta_y_all, component, delta_cache, masks, rows_family,
                    predictions, pair_union, scaler, ckpt="best") -> dict[str, Any]:
    dd = index.dev_dev_index
    supported = dd[masks["dev_dev_supported119"]]
    # pre-selected rule (independent of any result): first pair of each of the six
    # smallest supported components, ordered by (component id, pair id)
    order = sorted(supported.tolist(),
                   key=lambda i: (int(component[i]), index.rows[i]["pair_id"]))
    seen: dict[int, int] = {}
    chosen = []
    for i in order:
        c = int(component[i])
        if c in seen:
            continue
        seen[c] = i
        chosen.append(i)
        if len(chosen) == 6:
            break

    r1, r2 = rows_family["1"], rows_family["2"]
    base = r1
    selected = set(chosen)
    r1_pred = predictions.get((r1, "DELTA", ckpt))
    r1_end = predictions.get((r1, "ENDPOINT", ckpt))
    if r1_pred is not None and r1_end is not None:
        dh_d = _delta_hat(r1_pred, lab, pair_union, index, scaler)
        dh_e = _delta_hat(r1_end, lab, pair_union, index, scaler)
        gain = np.abs(dh_e - delta_y_all) - np.abs(dh_d - delta_y_all)
        pool = [i for i in dd[masks["dev_dev_all"]].tolist() if i not in selected]
        ranked = sorted(pool, key=lambda i: -float(gain[i]))
        best3 = ranked[:3]
        worst3 = ranked[-3:]
        diagnostic = ("D beats E", "D lags E")
    else:
        best3, worst3 = [], []
        diagnostic = ("n/a", "n/a")

    def one(i: int, group: str) -> dict[str, Any]:
        row = index.rows[i]
        out = {
            "group": group, "pair_id": row["pair_id"], "row_a": int(row["row_a"]),
            "row_b": int(row["row_b"]), "component_id": int(component[i]),
            "transform_id": row["transform_id"], "core_uid": row["core_uid"],
            "core_size": int(row["core_size"]), "cut_bond_type": int(row["cut_bond_type"]),
            "s_key_a": row["s_key_a"], "s_key_b": row["s_key_b"],
            "s_size_a": int(row["s_size_a"]), "s_size_b": int(row["s_size_b"]),
            "supported_by_other_component": bool(index.support_other_component[i]),
            "supported_by_two_components": bool(index.support_two_components[i]),
            "delta_y": float(delta_y_all[i]),
        }
        for (r, t, c), pred in predictions.items():
            if c != ckpt:
                continue
            dh = _delta_hat(pred, lab, pair_union, index, scaler)
            out[f"delta_hat_{r}_{t}"] = float(dh[i])
            out[f"abs_err_{r}_{t}"] = float(abs(dh[i] - delta_y_all[i]))
        return out

    return {
        "rule": "6 pre-selected: one pair from each of the six smallest supported "
                "components, ordered by (component id, pair id), chosen before looking at "
                "any prediction or residual. 3 best / 3 worst: ranking by "
                "|err_ENDPOINT| - |err_DELTA| among the remaining 270 Dev-Dev pairs of the "
                "structural row 1. The last six are result-selected diagnostics, not "
                "population evidence.",
        "structural_row_1": r1, "structural_row_2": r2,
        "checkpoint": ckpt,
        "preselected_supported": [one(i, "preselected_supported") for i in chosen],
        "best_d_vs_e": [one(i, f"diagnostic_{diagnostic[0]}") for i in best3],
        "worst_d_vs_e": [one(i, f"diagnostic_{diagnostic[1]}") for i in worst3],
    }


if __name__ == "__main__":
    raise SystemExit(main())
