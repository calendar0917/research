"""Screening, mechanism, seed and diagnostic tables for discovery v1."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import dc_common as C
import dc_data as D
import dc_models as M

import m2_evaluate as E  # noqa: E402
import r42_metrics as MET  # noqa: E402


def _read_train(key: str) -> dict[str, Any]:
    path = C.RUNS / key / "train.json"
    return C.read_json(path) if path.exists() else {}


def _read_pred(key: str, split: str) -> dict[str, np.ndarray] | None:
    path = C.PREDICTIONS / f"{key}_{split}420.npz"
    if not path.exists():
        return None
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _eval_metrics_path() -> Path | None:
    path = C.EVAL_DIR / "metrics.json"
    return path if path.exists() else None


def _screen_metric_path() -> Path | None:
    for path in (C.EVAL_DIR / "metrics.json", C.ANALYSIS / "eval_screen" / "metrics.json"):
        if path.exists():
            return path
    return None


def _model_summary(key: str) -> dict[str, Any] | None:
    train = _read_train(key)
    if key in C.FROZEN_CHECKPOINTS:
        entry = C.FROZEN_CHECKPOINTS[key]
        train = C.read_json(entry["train_json"])
        family, seed = entry["family"], entry["seed"]
        params = train.get("model_audit", {}).get("parameters")
        if params is None and key == "base_s0":
            params = C.LCRC_BASE_PARAMS_REFERENCE
        elif params is None:
            params = C.CONTEXT_TOTAL_PARAMS_REFERENCE
        train_rmse = train.get("metrics", {}).get("train", {}).get("rmse", float("nan"))
        wall = train.get("timing", {}).get("train_wall_seconds", float("nan"))
    else:
        if not train:
            return None
        family, seed = train["family"], train["seed"]
        params = train["model_audit"]["parameters"]
        train_rmse = train["metrics"]["train"]["rmse"]
        wall = train["timing"]["train_wall_seconds"]
    pred = _read_pred(key, "valid")
    if pred is None:
        return None
    metrics = MET.all_metrics(pred["pred"], pred["y_true"])
    return {
        "model": key,
        "family": family,
        "seed": int(seed),
        "params": int(params),
        "train_rmse": float(train_rmse),
        "valid_rmse": float(metrics["rmse"]),
        "valid_mae": float(metrics["mae"]),
        "valid_r2": float(metrics["r2"]),
        "valid_sse": float(np.sum((pred["pred"] - pred["y_true"]) ** 2)),
        "wall_seconds": float(wall),
        "rows": pred["rows"],
        "y": pred["y_true"],
        "pred": pred["pred"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------


def _scaffold_groups(split: Mapping[str, Any]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def scaffold(smiles: str) -> str:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""
        return str(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))

    train_rows = [int(r) for r in split["rows"]["train"]]
    valid_rows = [int(r) for r in split["rows"]["valid"]]
    train_scaffolds = {scaffold(split["smiles"][r]) for r in train_rows}
    valid_scaffolds = [scaffold(split["smiles"][r]) for r in valid_rows]
    seen = np.asarray([s in train_scaffolds for s in valid_scaffolds], dtype=bool)
    return {
        "valid_seen": seen,
        "valid_scaffolds": valid_scaffolds,
        "n_seen": int(seen.sum()),
        "n_unseen": int((~seen).sum()),
        "n_train_unique_scaffolds": int(len(train_scaffolds)),
        "n_valid_unique_scaffolds": int(len(set(valid_scaffolds))),
    }


def _property_groups(split: Mapping[str, Any], valid_cache: Mapping[str, Any]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    train_rows = [int(r) for r in split["rows"]["train"]]
    valid_rows = [int(r) for r in split["rows"]["valid"]]
    train_tpsa = np.asarray([Descriptors.TPSA(Chem.MolFromSmiles(str(split["smiles"][r]))) for r in train_rows], dtype=np.float64)
    valid_tpsa = np.asarray([Descriptors.TPSA(Chem.MolFromSmiles(str(split["smiles"][r]))) for r in valid_rows], dtype=np.float64)
    atom_ptr = np.asarray(valid_cache["atom_ptr"], dtype=np.int64)
    valid_n_atoms = np.diff(atom_ptr)
    train_ptr = np.asarray(split.get("atom_ptr", []), dtype=np.int64) if "atom_ptr" in split else None
    # Train atom counts are not needed for a valid-only split by the train median:
    # use the frozen Train3360 cache through the same loader when available.
    train_cache = D.MD.build_views(include_test=False)["train_cache"]
    train_n_atoms = np.diff(np.asarray(train_cache["atom_ptr"], dtype=np.int64))
    return {
        "valid_tpsa": valid_tpsa,
        "valid_n_atoms": valid_n_atoms,
        "tpsa_low": valid_tpsa <= float(np.median(train_tpsa)),
        "size_small": valid_n_atoms <= float(np.median(train_n_atoms)),
        "train_tpsa_median": float(np.median(train_tpsa)),
        "train_n_atoms_median": float(np.median(train_n_atoms)),
    }


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def build_screen_table() -> list[dict[str, Any]]:
    eval_path = _screen_metric_path()
    keys: list[str] = []
    if eval_path is not None:
        keys = list(C.read_json(eval_path)["models"].keys())
    if not keys:
        keys = ["base_s0", "roleblind_s0", *C.STAGE1_ARMS]
    base = _model_summary("roleblind_s0")
    rows = []
    for key in keys:
        s = _model_summary(key)
        if s is None:
            continue
        gain = float(base["valid_rmse"] - s["valid_rmse"]) if base else float("nan")
        mae_change = float(s["valid_mae"] - base["valid_mae"]) if base else float("nan")
        rows.append(
            {
                "model": key,
                "family": s["family"],
                "seed": s["seed"],
                "split": "valid420",
                "params": s["params"],
                "train_rmse": s["train_rmse"],
                "valid_rmse": s["valid_rmse"],
                "valid_mae": s["valid_mae"],
                "valid_r2": s["valid_r2"],
                "gain_vs_roleblind_s0": gain,
                "mae_change_vs_roleblind_s0": mae_change,
                "wall_seconds": s["wall_seconds"],
            }
        )
    rows.sort(key=lambda r: (r["valid_rmse"], r["model"]))
    _write_csv(
        C.TABLE_SCREEN,
        rows,
        [
            "model",
            "family",
            "seed",
            "split",
            "params",
            "train_rmse",
            "valid_rmse",
            "valid_mae",
            "valid_r2",
            "gain_vs_roleblind_s0",
            "mae_change_vs_roleblind_s0",
            "wall_seconds",
        ],
    )
    return rows


def _tail_rows_for_pair(ref_pred, other_pred, y, rows) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    tail_mask = np.isin(rows, np.asarray(C.TAIL_ROWS, dtype=np.int64))
    non = ~tail_mask
    err_ref = ref_pred - y
    err_other = other_pred - y
    sse_ref_tail = float(np.sum(err_ref[tail_mask] ** 2))
    sse_other_tail = float(np.sum(err_other[tail_mask] ** 2))
    rmse_ref_non = float(np.sqrt(np.mean(err_ref[non] ** 2)))
    rmse_other_non = float(np.sqrt(np.mean(err_other[non] ** 2)))
    return {
        "n_tail": int(tail_mask.sum()),
        "tail_sse_ref": sse_ref_tail,
        "tail_sse_other": sse_other_tail,
        "tail_sse_gain_other": float(sse_ref_tail - sse_other_tail),
        "tail_net_sse_other_minus_ref": float(sse_other_tail - sse_ref_tail),
        "non_tail_rmse_ref": rmse_ref_non,
        "non_tail_rmse_other": rmse_other_non,
        "non_tail_rmse_gain_other": float(rmse_ref_non - rmse_other_non),
        "non_tail_n": int(non.sum()),
    }


def build_mechanism_table() -> list[dict[str, Any]]:
    eval_path = _eval_metrics_path() or _screen_metric_path()
    if eval_path is None:
        return []
    payload = C.read_json(eval_path)
    splits = [s for s in ("valid", "test") if any(s in payload["models"][k] for k in payload["models"])]
    rows = []
    for pair_name, (fam_a, fam_b, _la, _lb) in C.MECHANISM_PAIRS.items():
        candidates_a = [k for k in payload["models"] if payload["models"][k]["family"] == fam_a]
        candidates_b = [k for k in payload["models"] if payload["models"][k]["family"] == fam_b]
        # Match seed when both sides have the same seed; otherwise use seed 0.
        for a in sorted(candidates_a):
            for b in sorted(candidates_b):
                if payload["models"][a]["seed"] != payload["models"][b]["seed"]:
                    continue
                for split in splits:
                    ma = payload["models"][a][split]
                    mb = payload["models"][b][split]
                    pred_a = _read_pred(a, split)
                    pred_b = _read_pred(b, split)
                    if pred_a is None or pred_b is None:
                        continue
                    y = pred_a["y_true"]
                    boot = E.paired_bootstrap(pred_a["pred"], pred_b["pred"], y)
                    tail = _tail_rows_for_pair(pred_a["pred"], pred_b["pred"], y, pred_a["rows"])
                    rows.append(
                        {
                            "pair": pair_name,
                            "true_model": a,
                            "control_model": b,
                            "seed": payload["models"][a]["seed"],
                            "split": f"{split}420",
                            "rmse_true": ma["rmse"],
                            "rmse_control": mb["rmse"],
                            "gain_true_vs_control": float(mb["rmse"] - ma["rmse"]),
                            "mae_true": ma["mae"],
                            "mae_control": mb["mae"],
                            "mae_change_true_vs_control": float(ma["mae"] - mb["mae"]),
                            "sse_gain_true_vs_control": float(mb["sse"] - ma["sse"]),
                            "n_true_better_abs": boot["observed"]["n_other_better_abs"],
                            "mean_abs_prediction_difference": boot["observed"]["mean_abs_prediction_difference"],
                            "gain_ci_low": boot["interval"]["gain_rmse"][0],
                            "gain_ci_median": boot["interval"]["gain_rmse"][1],
                            "gain_ci_high": boot["interval"]["gain_rmse"][2],
                            "tail_net_sse_true_minus_control": tail["tail_net_sse_other_minus_ref"],
                            "non_tail_rmse_true": tail["non_tail_rmse_ref"],
                            "non_tail_rmse_control": tail["non_tail_rmse_other"],
                            "non_tail_gain_true_vs_control": tail["non_tail_rmse_gain_other"],
                        }
                    )
    _write_csv(
        C.TABLE_MECHANISM,
        rows,
        [
            "pair",
            "true_model",
            "control_model",
            "seed",
            "split",
            "rmse_true",
            "rmse_control",
            "gain_true_vs_control",
            "mae_true",
            "mae_control",
            "mae_change_true_vs_control",
            "sse_gain_true_vs_control",
            "n_true_better_abs",
            "mean_abs_prediction_difference",
            "gain_ci_low",
            "gain_ci_median",
            "gain_ci_high",
            "tail_net_sse_true_minus_control",
            "non_tail_rmse_true",
            "non_tail_rmse_control",
            "non_tail_gain_true_vs_control",
        ],
    )
    return rows


def build_seed_table() -> list[dict[str, Any]]:
    eval_path = _eval_metrics_path() or _screen_metric_path()
    if eval_path is None:
        return []
    payload = C.read_json(eval_path)
    rows = []
    keys = list(payload["models"])
    for key in keys:
        meta = payload["models"][key]
        for split in ("valid", "test"):
            if split not in meta:
                continue
            rows.append(
                {
                    "model": key,
                    "family": meta["family"],
                    "seed": meta["seed"],
                    "split": f"{split}420",
                    "params": meta.get("parameters"),
                    "train_rmse": meta.get("train_rmse"),
                    "rmse": meta[split]["rmse"],
                    "mae": meta[split]["mae"],
                    "r2": meta[split]["r2"],
                }
            )
    # Seed-pair differences for any family observed at both seeds.
    by_family: dict[str, dict[int, str]] = {}
    for key in keys:
        meta = payload["models"][key]
        by_family.setdefault(meta["family"], {})[int(meta["seed"])] = key
    for family, seeds in by_family.items():
        if 0 in seeds and 1 in seeds:
            a, b = seeds[1], seeds[0]
            for split in ("valid", "test"):
                if split not in payload["models"][a] or split not in payload["models"][b]:
                    continue
                # Attach the seed-1 minus seed-0 difference as an extra row is not natural;
                # the screening table and REPORT describe it.  Keep this table tidy.
    _write_csv(
        C.TABLE_SEED,
        rows,
        ["model", "family", "seed", "split", "params", "train_rmse", "rmse", "mae", "r2"],
    )
    return rows


def build_group_table(split: Mapping[str, Any], valid_cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    eval_path = _eval_metrics_path() or _screen_metric_path()
    if eval_path is None:
        return []
    payload = C.read_json(eval_path)
    scaffolds = _scaffold_groups(split)
    props = _property_groups(split, valid_cache)
    groups = {
        "scaffold_seen": scaffolds["valid_seen"],
        "scaffold_unseen": ~scaffolds["valid_seen"],
        "small_molecules": props["size_small"],
        "large_molecules": ~props["size_small"],
        "low_tpsa": props["tpsa_low"],
        "high_tpsa": ~props["tpsa_low"],
    }
    rows = []
    for key, meta in payload["models"].items():
        pred = _read_pred(key, "valid")
        if pred is None:
            continue
        y = pred["y_true"]
        for group, mask in groups.items():
            if int(mask.sum()) == 0:
                continue
            rows.append(
                {
                    "model": key,
                    "family": meta["family"],
                    "seed": meta["seed"],
                    "group": group,
                    "n": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean((pred["pred"][mask] - y[mask]) ** 2))),
                    "mae": float(np.mean(np.abs(pred["pred"][mask] - y[mask]))),
                }
            )
    path = C.ANALYSIS / "table_groups.csv"
    _write_csv(path, rows, ["model", "family", "seed", "group", "n", "rmse", "mae"])
    C.write_json(
        C.ANALYSIS / "group_metadata.json",
        {
            "scaffold_seen_n": scaffolds["n_seen"],
            "scaffold_unseen_n": scaffolds["n_unseen"],
            "n_train_unique_scaffolds": scaffolds["n_train_unique_scaffolds"],
            "n_valid_unique_scaffolds": scaffolds["n_valid_unique_scaffolds"],
            "train_tpsa_median": props["train_tpsa_median"],
            "train_n_atoms_median": props["train_n_atoms_median"],
            "note": "Murcko scaffolds and TPSA are descriptive groups only; no model was reselected on them.",
        },
    )
    return rows


def stage2_selection(screen_rows: Sequence[Mapping[str, Any]], mechanism_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_model = {r["model"]: r for r in screen_rows}
    base = by_model.get("roleblind_s0")
    candidates = []
    if base is not None:
        for r in screen_rows:
            if r["model"] in ("base_s0", "roleblind_s0") or r["seed"] != 0:
                continue
            gain = float(r["gain_vs_roleblind_s0"])
            mae_change = float(r["mae_change_vs_roleblind_s0"])
            candidates.append(
                {
                    "model": r["model"],
                    "family": r["family"],
                    "valid_rmse": r["valid_rmse"],
                    "gain_vs_roleblind_s0": gain,
                    "mae_change_vs_roleblind_s0": mae_change,
                    "performance_gate": bool(gain >= C.STAGE2_RMSE_GAIN_THRESHOLD and mae_change <= 0.005),
                }
            )
    mechanism = []
    for r in mechanism_rows:
        if r["split"] != "valid420":
            continue
        mechanism.append(
            {
                "pair": r["pair"],
                "true_model": r["true_model"],
                "control_model": r["control_model"],
                "gain_true_vs_control": r["gain_true_vs_control"],
                "gate": bool(float(r["gain_true_vs_control"]) >= C.STAGE2_MECHANISM_GAIN_THRESHOLD),
            }
        )
    selected: list[str] = []
    # Prefer a mechanism-supported true arm, then a performance candidate from a different family.
    for m in sorted(mechanism, key=lambda x: -float(x["gain_true_vs_control"])):
        if m["gate"] and m["true_model"] not in selected:
            selected.append(m["true_model"])
            break
    for c in sorted(candidates, key=lambda x: -float(x["gain_vs_roleblind_s0"])):
        if c["performance_gate"] and c["model"] not in selected:
            family = c["family"]
            if not any(by_model.get(s, {}).get("family") == family for s in selected):
                selected.append(c["model"])
                break
    if len(selected) < 2:
        for c in sorted(candidates, key=lambda x: -float(x["gain_vs_roleblind_s0"])):
            if c["performance_gate"] and c["model"] not in selected:
                selected.append(c["model"])
                break
    payload = {
        "created_utc": C.utcnow(),
        "performance_gate": f"valid gain vs ROLEBLIND_s0 >= {C.STAGE2_RMSE_GAIN_THRESHOLD} and MAE change <= 0.005",
        "mechanism_gate": f"true-vs-matched-control valid gain >= {C.STAGE2_MECHANISM_GAIN_THRESHOLD}",
        "candidates": candidates,
        "mechanism": mechanism,
        "selected": selected[:2],
        "note": "Exploratory resource gate, not a statistical significance test.",
    }
    C.write_json(C.ANALYSIS / "stage2_selection.json", payload)
    return payload


def analyze() -> dict[str, Any]:
    views = D.load_views(include_test=False)
    screen = build_screen_table()
    mechanism = build_mechanism_table()
    seed = build_seed_table()
    groups = build_group_table(views["split"], views["valid_cache"])
    selection = stage2_selection(screen, mechanism)
    summary = {
        "created_utc": C.utcnow(),
        "protocol_id": C.PROTOCOL_ID,
        "screen": screen,
        "mechanism": mechanism,
        "seed": seed,
        "groups": groups,
        "stage2_selection": selection,
    }
    C.write_json(C.ANALYSIS / "analysis_summary.json", summary)
    return summary


__all__ = [
    "build_screen_table",
    "build_mechanism_table",
    "build_seed_table",
    "build_group_table",
    "stage2_selection",
    "analyze",
]
