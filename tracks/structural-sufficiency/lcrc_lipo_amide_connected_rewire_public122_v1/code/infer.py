"""Frozen-checkpoint inference over O / T / K rewired views (no training).

Run from the repository root on the compute host::

    uv run python tracks/.../code/run.py --stages infer --device cuda:0

Inference only: ``eval()`` + ``torch.inference_mode``, frozen weights, frozen
Train-only normalisation, zero optimiser, zero backward, zero new seed.  The step
reads the *frozen* manifests written by ``prepare.py`` (their sha256 is verified)
and never re-selects a pair, a view, a population or a range flag.

It writes ``per_view_predictions.csv`` (one row per view and scope),
``infer_checks.json`` and ``runtime.json``.  Labels are joined later, locally, by
``row_id`` only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import graphlib as G
import feats as F
import rewire as R


# ---------------------------------------------------------------------------
# frozen-input gate
# ---------------------------------------------------------------------------


def verify_freeze(analysis_dir: Path) -> dict[str, Any]:
    freeze = C.read_json(analysis_dir / "freeze.json")
    observed: dict[str, str] = {}
    missing: list[str] = []
    for name, want in (freeze.get("files") or {}).items():
        path = analysis_dir / name
        got = C.sha256_file(path) if path.exists() else "missing"
        observed[name] = got
        if got != want:
            missing.append(name)
    return {"pass": not missing, "mismatched": missing, "observed": observed, "frozen": freeze.get("files") or {}}


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def collate_rows(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return C.DD.PREP_DATA.collate(cache, [int(r) for r in rows])


def predict_views(
    model,
    cache: Mapping[str, Any],
    *,
    device,
    o_global: Mapping[int, np.ndarray],
    batch_size: int = 256,
) -> dict[str, Any]:
    """Predict every view under both frozen scopes.

    ``FULL`` recomputes every input block from the rewired graph.  ``G_FIXED``
    keeps the rewired local inputs but restores the molecule's own original
    (un-normalised) ``global42`` before the frozen normalisation, which isolates
    "local statistical path" from "global statistics moved".
    """
    view_keys = list(cache["view_ids"])
    n = len(view_keys)
    full = np.zeros(n, dtype=np.float64)
    fixed = np.zeros(n, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            idx = list(range(start, min(start + batch_size, n)))
            batch = collate_rows(cache, idx)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            full[start : start + len(idx)] = model(batch).detach().float().cpu().numpy().reshape(-1)
            g = batch["global"].clone()
            for j, i in enumerate(idx):
                rid = int(view_keys[i].split(":")[0])
                g[j] = torch.from_numpy(
                    np.asarray(o_global[rid], dtype=np.float32).reshape(-1)
                ).to(device)
            batch_fixed = dict(batch)
            batch_fixed["global"] = g
            fixed[start : start + len(idx)] = model(batch_fixed).detach().float().cpu().numpy().reshape(-1)
    return {"full": full, "fixed": fixed}


# ---------------------------------------------------------------------------
# per-view structural / input gates
# ---------------------------------------------------------------------------


def _instances_by_row(raw_rows: Sequence[Mapping[str, Any]], row_ids: Sequence[int]) -> dict[int, list[G.Instance]]:
    out: dict[int, list[G.Instance]] = {}
    for rid in sorted(set(int(r) for r in row_ids)):
        out[rid] = G.find_instances(G.summarise_raw(raw_rows[rid], rid))
    return out


def verify_views(
    *,
    rows_meta: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    train_min: np.ndarray,
    train_max: np.ndarray,
) -> dict[str, Any]:
    """Re-prove, on the exact tensors fed to the model, every structural invariant."""
    failures: list[dict[str, Any]] = []
    af51_failures: list[dict[str, Any]] = []
    n_cut = 0
    n_role = 0
    max_af51_diff = 0.0
    max_scalar_diff = 0.0
    idx_by_key = {str(k): i for i, k in enumerate(cache["view_ids"])}
    o_index = {int(m["row_id"]): idx_by_key[str(m["view_key"])] for m in rows_meta if m["kind"] == "O"}
    inst_cache = _instances_by_row(raw_rows, [int(m["row_id"]) for m in rows_meta])
    atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
    bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
    o_atoms = {rid: int(atom_counts[i]) for rid, i in o_index.items()}
    o_bonds = {rid: int(bond_counts[i]) for rid, i in o_index.items()}
    count_failures: list[dict[str, Any]] = []
    comp_bad: list[str] = []
    in_range: dict[str, bool] = {}
    n_global_changed = 0
    max_global_change = 0.0
    glob = np.asarray(cache["global"], dtype=np.float64)
    for meta in rows_meta:
        key = str(meta["view_key"])
        i = idx_by_key[key]
        rid = int(meta["row_id"])
        ok_range, _bad = _range_verdict(glob[i], train_min, train_max)
        in_range[key] = bool(ok_range)
        if int(np.asarray(cache["n_components"])[i]) != 1:
            comp_bad.append(key)
        if meta["kind"] == "O":
            continue
        n_cut += 1
        remove = _parse_bonds(str(meta["remove"]))
        add = _parse_bonds(str(meta["add"]))
        g0 = G.summarise_raw(raw_rows[rid], rid)
        view_row = F.row_from_artifact({**dict(raw_rows[rid]), "mol_index": rid})
        view_row, _info = R.edit_row(view_row, remove, add)
        rem_sets = [frozenset(b) for b in remove]
        cand = [
            inst
            for inst in inst_cache[rid]
            if frozenset((inst.c, inst.n)) in rem_sets or frozenset((inst.c, inst.acyl_c)) in rem_sets
        ]
        if not cand:
            failures.append({"view_key": key, "reason": "instance_not_found"})
            continue
        inst = cand[0]
        kind = meta["kind"]
        chk = R.verify_view_structure(g0, view_row, remove=remove, add=add, kind=str(kind), inst=inst)
        n_role += 1
        if not chk["pass"]:
            failures.append(
                {
                    "view_key": key,
                    "kind": kind,
                    "failed": [k for k, v in chk.items() if v is False],
                }
            )
        g_delta = np.abs(glob[i] - glob[o_index[rid]])
        if float(g_delta.max()) > 0.0:
            n_global_changed += 1
        max_global_change = max(max_global_change, float(g_delta.max()))
        dif = F.af51_difference(cache, o_index[rid], i)
        max_af51_diff = max(max_af51_diff, float(dif["max_abs_atom_feat_diff"]))
        max_scalar_diff = max(max_scalar_diff, float(dif["max_abs_atom_scalar_diff"]))
        if not dif["atom_feat_equal"]:
            af51_failures.append({"view_key": key, **dif})
        if int(atom_counts[i]) != o_atoms[rid] or int(bond_counts[i]) != o_bonds[rid]:
            count_failures.append(
                {"view_key": key, "atoms": int(atom_counts[i]), "bonds": int(bond_counts[i])}
            )
    return {
        "n_views": int(len(rows_meta)),
        "n_cut_views_checked": n_cut,
        "n_role_checked": n_role,
        "structural_failures": failures,
        "af51_failures": af51_failures,
        "af51_checked": n_cut,
        "af51_max_abs_atom_feat_diff": max_af51_diff,
        "atom_scalar_max_abs_diff": max_scalar_diff,
        "input_count_failures": count_failures,
        "n_components_not_one": comp_bad,
        "n_cut_views_with_global_change": int(n_global_changed),
        "max_abs_global_change": float(max_global_change),
        "views_in_train_range": in_range,
        "n_views_out_of_train_range": int(sum(1 for v in in_range.values() if not v)),
        "pass": not failures and not af51_failures and not count_failures and not comp_bad,
        "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
        "n_atoms_total": int(np.asarray(cache["atom_ptr"])[-1]),
        "n_bonds_total": int(np.asarray(cache["bond_ptr"])[-1]),
        "n_s2_paths_total": int(np.asarray(cache["path_ptr"])[-1]),
    }


def _range_verdict(vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[bool, list[int]]:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    bad = [int(i) for i in np.flatnonzero((v < lo - C.RANGE_TOL) | (v > hi + C.RANGE_TOL))]
    return (not bad), bad


def _parse_bonds(code: str) -> list[tuple[int, int]]:
    code = str(code).strip()
    if not code:
        return []
    out = []
    for chunk in code.split("."):
        a, b = chunk.split("-")
        out.append((int(a), int(b)))
    return out


# ---------------------------------------------------------------------------
# frozen baseline reproduction
# ---------------------------------------------------------------------------


def baseline_check(
    view_keys: Sequence[str],
    pred_std: np.ndarray,
    *,
    target_norm: Mapping[str, float],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    y = previous["y"]
    o_pred = previous["o_pred"]
    split = previous["split"]
    per_split: dict[str, Any] = {}
    for name in ("valid", "test"):
        rows, raw, std, ref_raw = [], [], [], []
        for key, s in zip(view_keys, pred_std):
            if ":O" not in key:
                continue
            rid = int(key.split(":")[0])
            if split.get(rid) != name:
                continue
            rows.append(rid)
            std.append(float(s))
            raw.append(float(target_norm["mean"] + target_norm["scale"] * float(s)))
            ref_raw.append(float(o_pred[rid]))
        if not rows:
            continue
        raw_arr = np.asarray(raw)
        yy = np.asarray([float(y[r]) for r in rows])
        ref = np.asarray(ref_raw)
        rmse = float(np.sqrt(np.mean((raw_arr - yy) ** 2)))
        mae = float(np.mean(np.abs(raw_arr - yy)))
        per_split[name] = {
            "n": len(rows),
            "rmse_raw": rmse,
            "mae_raw": mae,
            "reference_rmse": {
                "valid": C.REFERENCE["valid_rmse"],
                "test": C.REFERENCE["test_rmse"],
            }[name],
            "reference_mae": C.REFERENCE["test_mae"] if name == "test" else None,
            "rmse_diff": rmse
            - {"valid": C.REFERENCE["valid_rmse"], "test": C.REFERENCE["test_rmse"]}[name],
            "mae_diff": (mae - C.REFERENCE["test_mae"]) if name == "test" else None,
            "max_abs_pred_diff_vs_previous_round": float(np.abs(raw_arr - ref).max()),
            "rows_sha256": C.row_sha256(rows),
        }
    ok = True
    for name, d in per_split.items():
        ok = ok and abs(d["rmse_diff"]) <= C.REPRO_TOL
        ok = ok and d["max_abs_pred_diff_vs_previous_round"] <= C.REPRO_TOL
        if d["mae_diff"] is not None:
            ok = ok and abs(d["mae_diff"]) <= C.REPRO_TOL
    return {"per_split": per_split, "tolerance": C.REPRO_TOL, "pass": bool(ok)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    analysis_dir = Path(args.analysis_dir)
    C.log(f"infer: device={args.device} analysis_dir={C.track_rel(analysis_dir)}")

    gate_freeze = verify_freeze(analysis_dir)
    if not gate_freeze["pass"]:
        raise SystemExit(f"FROZEN_INPUT_MISMATCH: {gate_freeze['mismatched']}")

    manifest = C.read_json(analysis_dir / "inputs_manifest.json")
    train_min = np.asarray(manifest["train_global_range"]["min"], dtype=np.float64)
    train_max = np.asarray(manifest["train_global_range"]["max"], dtype=np.float64)
    view_rows = [
        {
            "view_key": r["view_key"],
            "row_id": int(r["row_id"]),
            "split": r["split"],
            "kind": r["kind"],
            "remove": r["remove"],
            "add": r["add"],
            "scopes": r["scopes"],
        }
        for r in _read_csv(analysis_dir / "view_manifest.csv")
    ]
    if not (analysis_dir / "donor_catalog.csv").exists():
        raise SystemExit("FROZEN_INPUT_MISMATCH: donor_catalog.csv missing")

    keys = [r["view_key"] for r in view_rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("VIEW_MANIFEST_NOT_UNIQUE")
    for r in view_rows:
        if r["kind"] != "O" and (not r["remove"] or not r["add"]):
            raise SystemExit(f"VIEW_KEY_INCOMPLETE: {r['view_key']}")

    spec = C.DD.frozen_spec()
    ck = C.load_checkpoint()
    tgt = C.target_norm(ck)
    model, spec = C.build_frozen_model(ck, device=args.device)
    state_before = C.DM.state_fingerprint(model.state_dict())
    norm_before = C.DD.PREP_DATA.norm_fingerprint(C.frozen_norm(ck))
    C.log(f"frozen model ready: {C.VARIANT} params={C.EXPECTED_PARAMS} state={C.STATE_HASH[:12]}")

    payload = C.load_raw_rows()
    raw_rows = payload["rows"]["Lipophilicity"]
    previous = C.load_previous_inputs()

    timings: dict[str, float] = {}
    predictions: list[dict[str, Any]] = []
    checks: dict[str, Any] = {"freeze": gate_freeze, "views": {}, "pass": True}
    fingerprints: dict[str, str] = {}

    for split in ("valid", "test"):
        t0 = time.perf_counter()
        rows_here = [r for r in view_rows if r["split"] == split]
        entries: list[tuple[str, dict[str, Any]]] = []
        for r in rows_here:
            rid = int(r["row_id"])
            base = F.row_from_artifact({**dict(raw_rows[rid]), "mol_index": rid})
            base["n_components"] = G.summarise_raw(raw_rows[rid], rid).n_components
            if r["kind"] == "O":
                entries.append((r["view_key"], base))
            else:
                cut, _info = R.edit_row(base, _parse_bonds(r["remove"]), _parse_bonds(r["add"]))
                entries.append((r["view_key"], cut))
        C.log(f"{split}: building {len(entries)} view graphs ...")
        cache = F.build_cache(entries, spec, split=f"rewire_views_{split}")
        fingerprints[split] = str(cache["meta"].get("content_fingerprint"))
        o_global = {}
        for i, (key, _row) in enumerate(entries):
            if ":O" in key:
                o_global[int(key.split(":")[0])] = np.asarray(cache["global"][i], dtype=np.float32)
        C.log(f"{split}: predicting ...")
        preds = predict_views(
            model, cache, device=args.device, o_global=o_global, batch_size=args.batch_size
        )
        o_mask = np.asarray([":O" in key for key, _row in entries], dtype=bool)
        o_diffs = np.abs(preds["full"][o_mask] - preds["fixed"][o_mask]) if o_mask.any() else np.zeros(1)
        o_max_diff = float(o_diffs.max())
        atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
        bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
        for i, (key, _row) in enumerate(entries):
            meta = rows_here[i]
            for scope, arr in (("FULL", preds["full"]), ("G_FIXED", preds["fixed"])):
                if meta["kind"] == "O" and scope == "G_FIXED":
                    continue
                predictions.append(
                    {
                        "view_key": key,
                        "row_id": int(key.split(":")[0]),
                        "split": split,
                        "kind": meta["kind"],
                        "remove": meta["remove"],
                        "add": meta["add"],
                        "scope": scope,
                        "pred_standardised": float(arr[i]),
                        "pred_raw": float(tgt["mean"] + tgt["scale"] * float(arr[i])),
                        "n_atoms": int(atom_counts[i]),
                        "n_bonds": int(bond_counts[i]),
                        "n_components": int(np.asarray(cache["n_components"])[i]),
                    }
                )
        blk = verify_views(
            rows_meta=rows_here,
            cache=cache,
            raw_rows=raw_rows,
            train_min=train_min,
            train_max=train_max,
        )
        blk["o_view_n"] = int(o_mask.sum())
        blk["o_view_max_abs_diff_full_vs_gfixed"] = o_max_diff
        blk["o_view_tolerance"] = C.REPRO_TOL
        blk["o_view_gfixed_equals_full_within_tol"] = bool(o_max_diff <= C.REPRO_TOL)
        blk["pass"] = bool(blk["pass"] and o_max_diff <= C.REPRO_TOL)
        checks["views"][split] = blk
        timings[f"build_and_predict_{split}"] = float(time.perf_counter() - t0)
        C.log(f"{split}: done in {timings[f'build_and_predict_{split}']:.1f}s")

    # baseline reproduction against the executed previous round
    base = baseline_check(
        [p["view_key"] for p in predictions if p["scope"] == "FULL"],
        np.asarray([p["pred_standardised"] for p in predictions if p["scope"] == "FULL"], dtype=np.float64),
        target_norm=tgt,
        previous=previous,
    )
    checks["baseline"] = base
    checks["pass"] = bool(checks["pass"] and base["pass"])
    view_gates = {split: bool(blk.get("pass")) for split, blk in checks["views"].items()}
    checks["view_gates"] = view_gates
    checks["pass"] = bool(checks["pass"] and all(view_gates.values()))

    state_after = C.DM.state_fingerprint(model.state_dict())
    norm_after = C.DD.PREP_DATA.norm_fingerprint(C.frozen_norm(ck))
    checks["state_identity"] = {
        "state_hash_before": state_before,
        "state_hash_after": state_after,
        "expected_state_hash": C.STATE_HASH,
        "norm_fingerprint_before": norm_before,
        "norm_fingerprint_after": norm_after,
        "unchanged": bool(state_before == state_after == C.STATE_HASH and norm_before == norm_after),
        "model_training_steps": 0,
        "optimizer_created": False,
        "backward_calls": 0,
    }
    checks["pass"] = bool(checks["pass"] and checks["state_identity"]["unchanged"])

    _write_csv(
        analysis_dir / "per_view_predictions.csv",
        predictions,
        [
            "view_key",
            "row_id",
            "split",
            "kind",
            "remove",
            "add",
            "scope",
            "pred_standardised",
            "pred_raw",
            "n_atoms",
            "n_bonds",
            "n_components",
        ],
    )
    total = float(time.perf_counter() - t_start)
    runtime = {
        "round_id": C.ROUND_ID,
        "stage": "infer",
        "device": str(args.device),
        "environment": C.env_info(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "n_views": len(view_rows),
        "n_predictions": len(predictions),
        "n_forward_passes": int(len(view_rows) * 2),
        "timings_seconds": timings,
        "wall_seconds": total,
        "gpu_hours": float(total / 3600.0) if str(args.device).startswith("cuda") else 0.0,
        "view_cache_fingerprints": fingerprints,
        "python": C.env_info()["python"],
    }
    C.write_json(analysis_dir / "runtime.json", runtime)
    checks["view_cache_fingerprints"] = fingerprints
    C.write_json(analysis_dir / "infer_checks.json", checks)
    C.log(f"infer done: {len(predictions)} predictions in {total:.1f}s; pass={checks['pass']}")
    if not checks["pass"]:
        C.log("INFERENCE_CHECKS_FAILED - see infer_checks.json")
        return 2
    return 0


def _read_csv(path: Path | str) -> list[dict[str, str]]:
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
    return p


if __name__ == "__main__":
    raise SystemExit(main())
