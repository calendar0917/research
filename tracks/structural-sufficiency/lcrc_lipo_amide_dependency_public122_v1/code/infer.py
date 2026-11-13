"""Frozen-checkpoint inference over O / T / K / R views (no training).

Run from the repository root on the compute host::

    uv run python tracks/.../code/infer.py --device cuda:0 --out <analysis dir>

The step is **inference only**: ``eval()`` + ``inference_mode``, frozen weights,
frozen Train-only normalisation, zero optimiser, zero backward, zero new seeds
(only the fixed atom-permutation RNG of the protocol check).

It writes ``per_view_predictions.csv`` (one row per view and scope) plus
``infer_checks.json`` and ``runtime.json``.  The label join and every statistical
quantity happen in ``analysis.py`` on the local host.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import motif as M
import prepare as P
import views as V

import lcrc_data as PREP_DATA
import torch


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def collate_rows(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return PREP_DATA.collate(cache, [int(r) for r in rows])


def predict_views(
    model,
    cache: Mapping[str, Any],
    *,
    device,
    o_global: Mapping[int, np.ndarray],
    batch_size: int = 256,
) -> dict[str, Any]:
    """Predict every view; returns FULL and G_FIXED standardised scores.

    ``G_FIXED`` keeps the T/K local inputs (atom features, scalars, S1/S2) but
    restores the *un-normalised* ``global42`` of the original molecule before
    the frozen normalisation is applied inside the model.
    """
    view_ids = list(cache["view_ids"])
    n = len(view_ids)
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
                rid = int(view_ids[i].split(":")[0])
                g[j] = torch.from_numpy(np.asarray(o_global[rid], dtype=np.float32).reshape(-1)).to(device)
            batch_fixed = dict(batch)
            batch_fixed["global"] = g
            fixed[start : start + len(idx)] = model(batch_fixed).detach().float().cpu().numpy().reshape(-1)
    return {"full": full, "fixed": fixed}


# ---------------------------------------------------------------------------
# structural verification of a built view cache
# ---------------------------------------------------------------------------


def verify_view_cache(
    entries: Sequence[tuple[str, dict[str, Any]]],
    cache: Mapping[str, Any],
    catalogs: Mapping[str, list[dict[str, Any]]],
    spec,
) -> dict[str, Any]:
    """Per-view structural checks (protocol section 8 items 2 and 3)."""

    inst_by_row: dict[int, list[dict[str, int]]] = {}
    for r in catalogs["instances"]:
        inst_by_row.setdefault(int(r["row_id"]), []).append(
            {"c": int(r["c"]), "o": int(r["o"]), "n": int(r["n"]), "acyl_c": int(r["acyl_c"])}
        )
    failures: list[dict[str, Any]] = []
    n_checked = 0
    s2_change_checked = 0
    for i, (vid, row) in enumerate(entries):
        rid = int(vid.split(":")[0])
        if ":O" in vid:
            continue
        cut_a = int(row["_cut_a"])
        cut_b = int(row["_cut_b"])
        base_row = V.row_from_artifact(catalogs["rows"][rid])
        base_row["n_components"] = V.n_components_of_row(base_row)
        chk = V.verify_cut(base_row, row, cut_a=cut_a, cut_b=cut_b, spec=spec)
        ok = (
            chk["removed_is_requested_cut"]
            and not chk["added_physical_bonds"]
            and not chk["changed_bond_attributes"]
            and all(chk["atom_attributes_unchanged"].values())
            and chk["directed_records_after_equal_before_minus_2"]
            and chk["components_increased_by_one"]
            and chk["raw_n_components_matches_graph"]
            and chk["s1_degree_sum_matches_bonds"]
        )
        n_checked += 1
        # role-specific path expectations: any instance whose cut matches
        role_ok = None
        role_kind = None
        for inst in inst_by_row.get(rid, []):
            pairs = {
                "T": ({inst["c"], inst["n"]}, "T"),
                "K": ({inst["c"], inst["acyl_c"]}, "K"),
            }
            for kind, (pair, kind_name) in pairs.items():
                if pair == {cut_a, cut_b}:
                    summ = V.cut_effect_summary(
                        chk,
                        kind=kind_name,
                        c=int(inst["c"]),
                        n=int(inst["n"]),
                        o=int(inst["o"]),
                        acyl_c=int(inst["acyl_c"]),
                    )
                    role_ok = bool(summ["pass"]) if role_ok is None else (role_ok and bool(summ["pass"]))
                    role_kind = kind_name
        if role_ok is not None:
            s2_change_checked += 1
            ok = ok and role_ok
        # the cached molecule must be finite / valid (validate_cache already ran)
        sl = V.cache_slice(cache, i)
        ok = ok and bool(np.isfinite(sl["atom_feat"]).all() and np.isfinite(sl["atom_scalar"]).all())
        if not ok:
            failures.append(
                {
                    "view_id": vid,
                    "role": role_kind,
                    "removed": chk["removed_physical_bonds"],
                    "components": [chk["n_components_before"], chk["n_components_after"]],
                    "role_ok": role_ok,
                }
            )
    # I/O-level identity: physical bond count changed by exactly one for cut views
    bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
    atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
    o_bonds = {}
    o_atoms = {}
    for i, vid in enumerate(cache["view_ids"]):
        if ":O" in vid:
            o_bonds[int(vid.split(":")[0])] = int(bond_counts[i])
            o_atoms[int(vid.split(":")[0])] = int(atom_counts[i])
    count_failures = []
    for i, vid in enumerate(cache["view_ids"]):
        if ":O" in vid:
            continue
        rid = int(vid.split(":")[0])
        if int(atom_counts[i]) != o_atoms[rid] or int(bond_counts[i]) != o_bonds[rid] - 1:
            count_failures.append(
                {"view_id": vid, "atom_counts": int(atom_counts[i]), "bond_counts": int(bond_counts[i])}
            )
    return {
        "n_cut_views_checked": n_checked,
        "n_role_checked": s2_change_checked,
        "failures": failures,
        "input_count_failures": count_failures,
        "bond_count_equal_O_minus_one": not count_failures,
        "pass": not failures and not count_failures,
        "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
        "n_views": int(cache["n_molecules"]),
        "n_atoms_total": int(np.asarray(cache["atom_ptr"])[-1]),
        "n_bonds_total": int(np.asarray(cache["bond_ptr"])[-1]),
        "n_s2_paths_total": int(np.asarray(cache["path_ptr"])[-1]),
    }


# ---------------------------------------------------------------------------
# frozen baseline reproduction
# ---------------------------------------------------------------------------


def baseline_check(
    view_ids: Sequence[str],
    pred_std: np.ndarray,
    *,
    target_norm: Mapping[str, float],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_row = {int(r["row_id"]): r for r in reference}
    per_split: dict[str, Any] = {}
    for split in ("valid", "test"):
        rows, std, raw, y = [], [], [], []
        for vid, s in zip(view_ids, pred_std):
            if ":O" not in vid:
                continue
            rid = int(vid.split(":")[0])
            ref = by_row.get(rid)
            if ref is None or ref["split"] != split:
                continue
            rows.append(rid)
            std.append(float(s))
            raw.append(float(target_norm["mean"] + target_norm["scale"] * float(s)))
            y.append(float(ref["y"]))
        if not rows:
            continue
        std = np.asarray(std)
        raw = np.asarray(raw)
        y = np.asarray(y)
        ref_raw = np.asarray([float(by_row[r]["pred_raw"]) for r in rows])
        ref_std = np.asarray([float(by_row[r]["pred_standardised"]) for r in rows])
        rmse = float(np.sqrt(np.mean((raw - y) ** 2)))
        mae = float(np.mean(np.abs(raw - y)))
        per_split[split] = {
            "n": len(rows),
            "rmse_raw": rmse,
            "mae_raw": mae,
            "max_abs_pred_diff_vs_reference": float(np.abs(raw - ref_raw).max()),
            "max_abs_std_diff_vs_reference": float(np.abs(std - ref_std).max()),
            "reference_rmse": {
                "valid": C.REFERENCE["valid_rmse"],
                "test": C.REFERENCE["test_rmse"],
            }[split],
            "reference_mae": C.REFERENCE["test_mae"] if split == "test" else None,
            "rmse_diff": rmse
            - {
                "valid": C.REFERENCE["valid_rmse"],
                "test": C.REFERENCE["test_rmse"],
            }[split],
            "mae_diff": (mae - C.REFERENCE["test_mae"]) if split == "test" else None,
            "row_order_matches_frozen_split": [int(x) for x in rows],
        }
    ok = True
    for split, d in per_split.items():
        ok = ok and abs(d["rmse_diff"]) <= C.REPRO_TOL
        ok = ok and d["max_abs_pred_diff_vs_reference"] <= C.REPRO_TOL
        if d["mae_diff"] is not None:
            ok = ok and abs(d["mae_diff"]) <= C.REPRO_TOL
    return {"per_split": per_split, "tolerance": C.REPRO_TOL, "pass": bool(ok)}


# ---------------------------------------------------------------------------
# fixed atom-relabelling check
# ---------------------------------------------------------------------------


def permutation_check(
    model,
    *,
    spec,
    device,
    raw_rows: Sequence[Mapping[str, Any]],
    instances: Sequence[Mapping[str, Any]],
    pred_by_view_scope: Mapping[tuple[str, str], float],
    seed: int = C.BOOTSTRAP_SEED,
    n_molecules: int = 3,
) -> dict[str, Any]:
    """Relabel atoms on the 3 smallest MATCHED-Test rows; predictions must match.

    The permutation is fixed by ``seed``; the structural catalog, the T/K cut
    views, the per-view predictions and the per-molecule aggregate contrasts are
    all compared after mapping atoms back through the permutation.
    """
    matched_test_rows = sorted(
        {int(r["row_id"]) for r in instances if r["split"] == "test" and int(r["matched"]) == 1}
    )
    take = matched_test_rows[:n_molecules]
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for rid in take:
        base = V.row_from_artifact(raw_rows[rid])
        base["n_components"] = V.n_components_of_row(base)
        n = int(base["atom_class"].shape[0])
        perm = rng.permutation(n)
        inv = np.argsort(perm)
        prow = V.permute_row(base, perm)
        g = M.summarise_raw(prow, rid)
        found = {(i.c, i.o, i.n, i.acyl_c) for i in M.find_instances(g)}
        inst_rows = [r for r in instances if int(r["row_id"]) == rid]
        expected = {
            (
                int(perm[int(r["c"])]),
                int(perm[int(r["o"])]),
                int(perm[int(r["n"])]),
                int(perm[int(r["acyl_c"])]),
            )
            for r in inst_rows
        }
        lut: dict[frozenset, str] = {}
        for r in inst_rows:
            lut[frozenset((int(r["c"]), int(r["n"])))] = str(r["view_t"])
            lut[frozenset((int(r["c"]), int(r["acyl_c"])))] = str(r["view_k"])
        entries: list[tuple[str, dict[str, Any]]] = [(P.view_id(rid, "O"), prow)]
        view_to_original: dict[str, str] = {}
        for r in inst_rows:
            pc, pn, pa = int(perm[int(r["c"])]), int(perm[int(r["n"])]), int(perm[int(r["acyl_c"])])
            t_row, _ = V.cut_row(prow, pc, pn)
            k_row, _ = V.cut_row(prow, pc, pa)
            vt = P.view_id(rid, "T", pc, pn)
            vk = P.view_id(rid, "K", pc, pa)
            entries.append((vt, _with_cut(t_row, pc, pn)))
            entries.append((vk, _with_cut(k_row, pc, pa)))
            view_to_original[vt] = lut[frozenset((int(inv[pc]), int(inv[pn])))]
            view_to_original[vk] = lut[frozenset((int(inv[pc]), int(inv[pa])))]
        cache = V.build_cache(entries, spec, split="permutation_probe")
        o_global = {rid: np.asarray(cache["global"][0], dtype=np.float32)}
        with torch.inference_mode():
            preds = predict_views(model, cache, device=device, o_global=o_global)
        diffs = []
        for i, (vid, _row) in enumerate(entries):
            orig_vid = P.view_id(rid, "O") if ":O" in vid else view_to_original[vid]
            ref = pred_by_view_scope.get((orig_vid, "FULL"))
            if ref is None:
                continue
            diffs.append(abs(float(preds["full"][i]) - float(ref)))
        out.append(
            {
                "row_id": int(rid),
                "n_atoms": n,
                "permutation": perm.tolist(),
                "catalog_instances_match": found == expected,
                "n_views": len(entries),
                "n_compared": len(diffs),
                "max_abs_pred_diff": float(max(diffs)) if diffs else None,
            }
        )
    ok = all(
        o["catalog_instances_match"]
        and (o["max_abs_pred_diff"] is not None)
        and o["max_abs_pred_diff"] <= C.REPRO_TOL
        for o in out
    )
    return {"molecules": out, "tolerance": C.REPRO_TOL, "pass": bool(ok)}


def _with_cut(row: dict[str, Any], a: int, b: int) -> dict[str, Any]:
    row = dict(row)
    row["_cut_a"] = int(a)
    row["_cut_b"] = int(b)
    return row


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

    spec = C.DD.frozen_spec()
    ck = C.load_checkpoint()
    tgt = C.target_norm(ck)
    model, spec = C.build_frozen_model(ck, device=args.device)
    C.log(f"frozen model ready: {C.VARIANT} params={C.EXPECTED_PARAMS} state={C.STATE_HASH[:12]}")
    state_before = C.DM.state_fingerprint(model.state_dict())
    norm_before = C.DD.PREP_DATA.norm_fingerprint(C.frozen_norm(ck))

    payload = C.load_raw_rows()
    raw_rows = payload["rows"]["Lipophilicity"]
    instances = P.read_csv(analysis_dir / "motif_catalog.csv")
    view_rows = P.read_csv(analysis_dir / "view_manifest.csv")
    reference = P.read_csv(analysis_dir / "reference_o_predictions.csv")

    # dedupe gate: (row_id, cut set) must be unique
    keys = []
    for r in view_rows:
        rid = int(r["row_id"])
        key = (
            (rid, "none") if int(r["cut_a"]) < 0 else (rid, tuple(sorted((int(r["cut_a"]), int(r["cut_b"])))))
        )
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise SystemExit("VIEW_MANIFEST_NOT_UNIQUE: (row_id, cut set) repeats")

    timings: dict[str, float] = {}
    predictions: list[dict[str, Any]] = []
    checks: dict[str, Any] = {"views": {}, "baseline": {}, "pass": True}
    view_cache_fingerprints: dict[str, str] = {}

    for split in ("train", "valid", "test"):
        t0 = time.perf_counter()
        rows_here = [r for r in view_rows if r["split"] == split]
        meta_by_view = {str(r["view_id"]): r for r in rows_here}
        entries: list[tuple[str, dict[str, Any]]] = []
        for r in rows_here:
            rid = int(r["row_id"])
            base = V.row_from_artifact(raw_rows[rid])
            base["n_components"] = V.n_components_of_row(base)
            if int(r["cut_a"]) < 0:
                entries.append((str(r["view_id"]), base))
            else:
                cut, _info = V.cut_row(base, int(r["cut_a"]), int(r["cut_b"]))
                entries.append((str(r["view_id"]), _with_cut(cut, int(r["cut_a"]), int(r["cut_b"]))))
        C.log(f"{split}: building {len(entries)} view graphs ...")
        cache = V.build_cache(entries, spec, split=f"amide_views_{split}")
        view_cache_fingerprints[split] = str(cache["meta"].get("content_fingerprint"))
        o_global = {}
        for i, (vid, _row) in enumerate(entries):
            if ":O" in vid:
                o_global[int(vid.split(":")[0])] = np.asarray(cache["global"][i], dtype=np.float32)
        C.log(f"{split}: predicting ...")
        preds = predict_views(model, cache, device=args.device, o_global=o_global, batch_size=args.batch_size)
        # O-view G_FIXED must be numerically identical to FULL
        atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
        bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
        o_pred_identical = True
        for i, (vid, _row) in enumerate(entries):
            if ":O" in vid and not np.array_equal(preds["full"][i], preds["fixed"][i]):
                o_pred_identical = False
            for scope, arr in (("FULL", preds["full"]), ("G_FIXED", preds["fixed"])):
                if ":O" in vid and scope == "G_FIXED":
                    continue
                m = meta_by_view[vid]
                predictions.append(
                    {
                        "view_id": vid,
                        "row_id": int(vid.split(":")[0]),
                        "split": split,
                        "kinds": str(m["kinds"]),
                        "cut_a": int(m["cut_a"]),
                        "cut_b": int(m["cut_b"]),
                        "scope": scope,
                        "pred_standardised": float(arr[i]),
                        "pred_raw": float(tgt["mean"] + tgt["scale"] * float(arr[i])),
                        "n_atoms": int(atom_counts[i]),
                        "n_bonds": int(bond_counts[i]),
                        "n_components": int(cache["n_components"][i]),
                    }
                )
        checks["views"][split] = verify_view_cache(
            entries, cache, {"instances": instances, "rows": raw_rows}, spec
        )
        checks["views"][split]["o_view_gfixed_equals_full"] = bool(o_pred_identical)
        checks["views"][split]["pass"] = bool(checks["views"][split]["pass"] and o_pred_identical)
        timings[f"build_and_predict_{split}"] = float(time.perf_counter() - t0)
        C.log(f"{split}: done in {timings[f'build_and_predict_{split}']:.1f}s")

    # baseline reproduction (uses the O views of both splits)
    base = baseline_check(
        [p["view_id"] for p in predictions if p["scope"] == "FULL"],
        np.asarray([p["pred_standardised"] for p in predictions if p["scope"] == "FULL"], dtype=np.float64),
        target_norm=tgt,
        reference=reference,
    )
    checks["baseline"] = base
    checks["pass"] = bool(checks["pass"] and base["pass"])
    order_ok = True
    for split in ("valid", "test"):
        expected_rows = [int(r["row_id"]) for r in view_rows if r["split"] == split and int(r["cut_a"]) < 0]
        observed_rows = [
            int(p["row_id"])
            for p in predictions
            if p["split"] == split and int(p["cut_a"]) < 0 and p["scope"] == "FULL"
        ]
        ok = expected_rows == observed_rows
        checks.setdefault("o_view_row_order", {})[split] = {
            "n": len(expected_rows),
            "matches_manifest_order": bool(ok),
            "rows_sha256": C.row_sha256(observed_rows),
        }
        order_ok = order_ok and ok
    checks["o_view_row_order_pass"] = bool(order_ok)
    checks["pass"] = bool(checks["pass"] and order_ok)

    # fixed atom-relabelling gate
    pred_map = {(p["view_id"], p["scope"]): p["pred_standardised"] for p in predictions}
    t0 = time.perf_counter()
    perm = permutation_check(
        model,
        spec=spec,
        device=args.device,
        raw_rows=raw_rows,
        instances=instances,
        pred_by_view_scope=pred_map,
    )
    timings["permutation_check"] = float(time.perf_counter() - t0)
    checks["permutation"] = perm
    checks["pass"] = bool(checks["pass"] and perm["pass"])

    P.write_csv(
        analysis_dir / "per_view_predictions.csv",
        predictions,
        [
            "view_id",
            "row_id",
            "split",
            "kinds",
            "cut_a",
            "cut_b",
            "scope",
            "pred_standardised",
            "pred_raw",
            "n_atoms",
            "n_bonds",
            "n_components",
        ],
    )

    # state / normalisation identity before vs after the whole inference run
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
        "view_cache_fingerprints": view_cache_fingerprints,
    }
    C.write_json(analysis_dir / "runtime.json", runtime)
    checks["view_cache_fingerprints"] = view_cache_fingerprints
    C.write_json(analysis_dir / "infer_checks.json", checks)
    C.log(f"infer done: {len(predictions)} predictions in {total:.1f}s; pass={checks['pass']}")
    if not checks["pass"]:
        C.log("INFERENCE_CHECKS_FAILED - see infer_checks.json")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
