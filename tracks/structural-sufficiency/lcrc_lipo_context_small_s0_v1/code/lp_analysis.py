"""Analysis: main tables, paired comparisons, judgements, scaffold bootstrap,
the three predefined structural groups and the frozen-D2 source interventions.

Everything in this module is CPU post-processing of pulled per-arm outputs
(``analysis/results/<arm>/``).  It never trains anything, and it never selects a
checkpoint: the best-Val point and the last epoch were already frozen by the
training runs.

Design notes
------------
* Comparisons treat ``B`` (and ``S`` / ``R2``) as the control and report the
  continuous quantity ``(RMSE_control - RMSE_candidate) / RMSE_control`` at the
  best-Val point, the last epoch and the late-epoch window, together with MAE,
  per-molecule absolute-error differences, win rate and the tail-trimmed
  sensitivity sets (rows removed are chosen from the *control's* worst absolute
  errors and applied to both sides).
* The 500-replicate interval is a Murcko-scaffold **cluster** bootstrap with
  Exp(1) weights shared by every model in a comparison (paired).  It describes
  sample uncertainty of the fixed models, not training-seed variance.
* The group step freezes the group on B's best-Val point (size-standardised MAE
  gap >= 0.05 raw logD) and only then reads Test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import lp_common as C
import lp_data as D
import lp_source as S

MAIN_ARMS = ("B", "S", "R1", "R2", "D2", "G")
CONTROLS = {
    "D2_vs_B": ("B", "D2"),
    "D2_vs_S": ("S", "D2"),
    "D2_vs_R2": ("R2", "D2"),
    "D2_vs_R1": ("R1", "D2"),
    "R2_vs_B": ("B", "R2"),
    "G_vs_B": ("B", "G"),
    "G_vs_D2": ("D2", "G"),
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def arm_dir(arm: str) -> Path:
    return C.RESULTS / arm


def load_arm(arm: str) -> dict[str, Any]:
    path = arm_dir(arm) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {C.track_rel(path)}; pull the {arm} run first")
    return C.read_json(path)


def load_curve(arm: str) -> list[dict[str, Any]]:
    import csv as _csv

    path = arm_dir(arm) / "curve.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in _csv.DictReader(handle)]


def load_pred(arm: str, split: str, tag: str) -> dict[str, np.ndarray]:
    path = arm_dir(arm) / f"pred_{split}_{tag}.npz"
    with np.load(path, allow_pickle=False) as handle:
        return {k: np.asarray(handle[k]) for k in handle.files}


def available_arms() -> list[str]:
    return [arm for arm in MAIN_ARMS if (arm_dir(arm) / "metrics.json").is_file()]


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------


def rel(control: float, candidate: float) -> float:
    """Relative improvement of ``candidate`` over ``control`` (positive = better)."""
    return float((control - candidate) / control)


def rmse(pred: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if weights is None:
        return float(np.sqrt(np.mean(err**2)))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.sum(w * err**2) / np.sum(w)))


def mae(pred: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    err = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    if weights is None:
        return float(np.mean(err))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * err) / np.sum(w))


# ---------------------------------------------------------------------------
# split-level predictions (best-Val and last epoch) for every arm
# ---------------------------------------------------------------------------


def split_table(arm: str) -> dict[str, Any]:
    payload = load_arm(arm)
    summary = payload["summary"]
    return {
        "arm": arm,
        "epochs": payload["recipe"]["epochs"],
        "parameters": payload["model"]["parameters"],
        "best_epoch": summary["best_valid"]["epoch"],
        "best_valid": {split: summary["best_valid"][split] for split in C.SPLITS},
        "last_epoch": {"epoch": summary["last_epoch"]["epoch"], **{split: summary["last_epoch"][split] for split in C.SPLITS}},
        "late_window": summary["late_window"],
        "train_online_mse_last": float(
            [row for row in load_curve(arm)][-1]["train_mse_standardized_online"]
        ),
        "gpu_hours": payload["timing"].get("gpu_hours_charged"),
    }


# ---------------------------------------------------------------------------
# paired comparison at the best-Val / last / late window points
# ---------------------------------------------------------------------------


def _tail_sets(control_err: np.ndarray, fraction: float) -> np.ndarray:
    """Row mask keeping everything except the control's worst ``fraction`` rows."""
    n = control_err.size
    k = int(math.floor(fraction * n))
    keep = np.ones(n, dtype=bool)
    if k > 0:
        order = np.argsort(-control_err, kind="stable")
        keep[order[:k]] = False
    return keep


def compare(control: str, candidate: str) -> dict[str, Any]:
    cb = load_arm(control)
    cd = load_arm(candidate)
    out: dict[str, Any] = {"control": control, "candidate": candidate}

    for point in ("best_valid", "last_epoch"):
        c = cb["summary"][point]
        d = cd["summary"][point]
        entry = {}
        for split in C.SPLITS:
            entry[split] = {
                "control_rmse": float(c[split]["rmse"]),
                "candidate_rmse": float(d[split]["rmse"]),
                "rel_improvement": rel(float(c[split]["rmse"]), float(d[split]["rmse"])),
                "control_mae": float(c[split]["mae"]),
                "candidate_mae": float(d[split]["mae"]),
                "rel_mae_improvement": rel(float(c[split]["mae"]), float(d[split]["mae"])),
                "abs_rmse_delta": float(c[split]["rmse"]) - float(d[split]["rmse"]),
            }
        out[point] = entry

    late_c = cb["summary"]["late_window"]
    late_d = cd["summary"]["late_window"]
    out["late_window"] = {
        "window": int(late_c["window"]),
        "epochs": late_c["epochs"],
        **{
            f"{split}_rel_improvement": rel(
                float(late_c[f"{split}_rmse"]), float(late_d[f"{split}_rmse"])
            )
            for split in C.SPLITS
        },
        **{
            f"{split}_control_rmse": float(late_c[f"{split}_rmse"])
            for split in C.SPLITS
        },
        **{
            f"{split}_candidate_rmse": float(late_d[f"{split}_rmse"])
            for split in C.SPLITS
        },
    }

    # ---- per-molecule (best-Val) and tail sensitivity --------------------
    tail: dict[str, Any] = {}
    for split in ("valid", "test"):
        pb = load_pred(control, split, "best")
        pd = load_pred(candidate, split, "best")
        if not np.array_equal(pb["rows"], pd["rows"]):
            raise ValueError(f"{control}/{candidate}: {split} row order differs")
        y = pb["y_true"]
        err_c = np.abs(pb["raw"] - y)
        err_d = np.abs(pd["raw"] - y)
        diff = err_c - err_d  # positive = candidate closer
        entry = {
            "mean_abs_error_difference": float(diff.mean()),
            "median_abs_error_difference": float(np.median(diff)),
            "win_rate": float((diff > 0).mean()),
            "tie_rate": float((diff == 0).mean()),
            "n": int(diff.size),
            "control_rmse": rmse(pb["raw"], y),
            "candidate_rmse": rmse(pd["raw"], y),
        }
        for fraction in C.TAIL_TRIM_FRACTIONS:
            keep = _tail_sets(err_c, fraction)
            entry[f"trim_{fraction}"] = {
                "rows_removed": int((~keep).sum()),
                "control_rmse": rmse(pb["raw"][keep], y[keep]),
                "candidate_rmse": rmse(pd["raw"][keep], y[keep]),
                "rel_improvement": rel(rmse(pb["raw"][keep], y[keep]), rmse(pd["raw"][keep], y[keep])),
                "control_mae": mae(pb["raw"][keep], y[keep]),
                "candidate_mae": mae(pd["raw"][keep], y[keep]),
            }
        tail[split] = entry
    out["per_molecule"] = tail
    return out


# ---------------------------------------------------------------------------
# judgement
# ---------------------------------------------------------------------------


def _channel_used() -> dict[str, Any]:
    path = C.ANALYSIS / "preflight.json"
    if not path.is_file():
        return {"available": False, "channel_used": None}
    payload = C.read_json(path)
    section = payload.get("sections", {}).get("c", {})
    return {
        "available": True,
        "channel_used": bool(section.get("channel_used")),
        "checks": section.get("checks", {}),
        "watched_steps": section.get("watched_steps", []),
    }


def judge(comp: Mapping[str, Any], channel: Mapping[str, Any]) -> dict[str, Any]:
    best = comp["best_valid"]
    last = comp["last_epoch"]
    late = comp["late_window"]
    test = comp["per_molecule"]["test"]

    rel_val_best = float(best["valid"]["rel_improvement"])
    rel_test_best = float(best["test"]["rel_improvement"])
    rel_val_last = float(last["valid"]["rel_improvement"])
    rel_test_last = float(last["test"]["rel_improvement"])
    rel_test_late = float(late["test_rel_improvement"])
    rel_train_best = float(best["train"]["rel_improvement"])
    mae_val_ok = float(best["valid"]["rel_mae_improvement"]) >= -C.PROMISE_MAE_TOLERANCE
    mae_test_ok = float(best["test"]["rel_mae_improvement"]) >= -C.PROMISE_MAE_TOLERANCE
    tail1 = float(test["trim_0.01"]["rel_improvement"])
    tail5 = float(test["trim_0.05"]["rel_improvement"])
    used = channel.get("channel_used")
    conditions = {
        "val_best_ge_2pct": rel_val_best >= C.PROMISE_REL_IMPROVEMENT,
        "test_best_ge_2pct": rel_test_best >= C.PROMISE_REL_IMPROVEMENT,
        "mae_val_not_worse_1pct": mae_val_ok,
        "mae_test_not_worse_1pct": mae_test_ok,
        "last_epoch_same_direction": rel_test_last > 0,
        "late_window_same_direction": rel_test_late > 0,
        "tail_trim_1pct_test_positive": tail1 > 0,
        "channel_used": bool(used) if used is not None else False,
    }
    fit_only = bool(rel_train_best > C.PROMISE_REL_IMPROVEMENT and rel_val_best <= 0 and rel_test_best <= 0)
    signs = {
        "val_best": np.sign(rel_val_best),
        "test_best": np.sign(rel_test_best),
        "test_last": np.sign(rel_test_last),
        "test_late": np.sign(rel_test_late),
    }
    direction_conflict = len({int(v) for v in signs.values()}) > 1
    tail_specific = bool(tail5 >= C.PROMISE_REL_IMPROVEMENT and rel_test_best < tail5 - C.PROMISE_REL_IMPROVEMENT)

    promising = all(conditions.values())
    if fit_only:
        label = "FIT_ONLY"
    elif promising:
        label = "PROMISING_THIS_TASK"
    elif direction_conflict:
        label = "SELECTION_OR_SPLIT_SENSITIVE"
    elif tail_specific:
        label = "TAIL_SPECIFIC"
    else:
        label = "NO_CLEAR_INCREMENT_THIS_TASK"
    return {
        "label": label,
        "conditions": conditions,
        "values": {
            "rel_val_best": rel_val_best,
            "rel_test_best": rel_test_best,
            "rel_val_last": rel_val_last,
            "rel_test_last": rel_test_last,
            "rel_test_late_window": rel_test_late,
            "rel_train_best": rel_train_best,
            "rel_mae_val_best": float(best["valid"]["rel_mae_improvement"]),
            "rel_mae_test_best": float(best["test"]["rel_mae_improvement"]),
            "tail_trim_1pct_test": tail1,
            "tail_trim_5pct_test": tail5,
            "test_win_rate": float(test["win_rate"]),
        },
        "signs": {k: int(v) for k, v in signs.items()},
        "fit_only": fit_only,
        "tail_specific": tail_specific,
        "direction_conflict": direction_conflict,
    }


def arm_level_label(judgements: Mapping[str, Mapping[str, Any]], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-registered mechanical arm-level judgement."""

    def _separated(delta_interval: Mapping[str, float], rel_value: float) -> bool:
        lo, hi = float(delta_interval["lo"]), float(delta_interval["hi"])
        excludes_zero = lo > 0 or hi < 0
        return bool(excludes_zero and abs(rel_value) >= C.PROMISE_REL_IMPROVEMENT)

    d2_s = judgements["D2_vs_S"]
    d2_r2 = judgements["D2_vs_R2"]
    d2_b = judgements["D2_vs_B"]

    intervals = bootstrap.get("paired_deltas", {})
    sep = {
        "D2_vs_S": _separated(intervals.get("D2_vs_S", {}).get("test", {"lo": 1.0, "hi": -1.0}), d2_s["values"]["rel_test_best"]),
        "D2_vs_R2": _separated(intervals.get("D2_vs_R2", {}).get("test", {"lo": 1.0, "hi": -1.0}), d2_r2["values"]["rel_test_best"]),
        "D2_vs_B": _separated(intervals.get("D2_vs_B", {}).get("test", {"lo": 1.0, "hi": -1.0}), d2_b["values"]["rel_test_best"]),
        "R2_vs_B": _separated(intervals.get("R2_vs_B", {}).get("test", {"lo": 1.0, "hi": -1.0}), judgements["R2_vs_B"]["values"]["rel_test_best"]),
    }
    strong = [name for name in ("D2_vs_S", "D2_vs_R2", "R2_vs_B") if judgements[name]["label"] == "PROMISING_THIS_TASK"]

    if judgements["D2_vs_S"]["label"] == "PROMISING_THIS_TASK" and judgements["D2_vs_R2"]["label"] == "PROMISING_THIS_TASK" and sep["D2_vs_R2"]:
        label = "LEARNED_CONTEXT_CANDIDATE_ON_LIPO"
    elif strong and not sep["D2_vs_R2"]:
        label = "FIXED_CONTEXT_COMPETITIVE_ON_LIPO"
    elif not sep["D2_vs_S"]:
        label = "NO_SEPARATED_NEIGHBOR_GAIN_ON_LIPO"
    else:
        label = "WEAK_CHANNEL_THIS_CONFIG"
    return {
        "label": label,
        "separated": sep,
        "promising_arms": strong,
        "note": (
            "mechanical application of the pre-registered rules; 'separated' means the paired "
            "scaffold-bootstrap interval of the Test RMSE delta excludes 0 AND the relative "
            "improvement reaches the 2% investment threshold"
        ),
    }


# ---------------------------------------------------------------------------
# scaffold bootstrap
# ---------------------------------------------------------------------------


def scaffold_clusters(smiles: Sequence[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    keys: list[str] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            keys.append(str(smi))
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        keys.append(scaffold if scaffold else Chem.MolToSmiles(mol, canonical=True))
    unique = {key: i for i, key in enumerate(sorted(set(keys)))}
    return np.asarray([unique[key] for key in keys], dtype=np.int64)


def bootstrap_payload(arms: Sequence[str], *, replicates: int | None = None) -> dict[str, Any]:
    replicates = int(C.BOOTSTRAP_REPLICATES if replicates is None else replicates)
    split_rows = S.load_split()["rows"]
    smiles = read_mol_csv_from_manifest()
    rng = np.random.default_rng(int(C.BOOTSTRAP_SEED))
    out: dict[str, Any] = {"replicates": replicates, "seed": int(C.BOOTSTRAP_SEED), "weight_dist": "Exp(1)", "splits": {}}
    errs: dict[str, dict[str, np.ndarray]] = {}
    ys: dict[str, dict[str, np.ndarray]] = {}
    clusters: dict[str, np.ndarray] = {}
    for split in ("valid", "test"):
        rows = split_rows[split]
        keys = scaffold_clusters([smiles[int(r)] for r in rows])
        clusters[split] = keys
        n_clusters = int(keys.max()) + 1
        weights = rng.exponential(scale=1.0, size=(replicates, n_clusters))
        split_out: dict[str, Any] = {"n": int(len(rows)), "n_scaffold_clusters": n_clusters}
        per_arm: dict[str, dict[str, float]] = {}
        for arm in arms:
            pred = load_pred(arm, split, "best")
            errs.setdefault(split, {})[arm] = np.asarray(pred["raw"], dtype=np.float64) - np.asarray(pred["y_true"], dtype=np.float64)
            ys.setdefault(split, {})[arm] = np.asarray(pred["y_true"], dtype=np.float64)
            samples_rmse = np.empty(replicates)
            samples_mae = np.empty(replicates)
            for i in range(replicates):
                w = weights[i][keys]
                samples_rmse[i] = math.sqrt(float(np.sum(w * errs[split][arm] ** 2) / np.sum(w)))
                samples_mae[i] = float(np.sum(w * np.abs(errs[split][arm])) / np.sum(w))
            per_arm[arm] = {
                "rmse_point": rmse(errs[split][arm] + ys[split][arm], ys[split][arm]),
                "rmse_lo": float(np.percentile(samples_rmse, 2.5)),
                "rmse_hi": float(np.percentile(samples_rmse, 97.5)),
                "mae_point": mae(errs[split][arm] + ys[split][arm], ys[split][arm]),
                "mae_lo": float(np.percentile(samples_mae, 2.5)),
                "mae_hi": float(np.percentile(samples_mae, 97.5)),
            }
        split_out["models"] = per_arm
        out["splits"][split] = split_out
        # paired deltas
        paired: dict[str, Any] = {}
        for name, (control, candidate) in CONTROLS.items():
            if control not in arms or candidate not in arms:
                continue
            delta_rmse = np.empty(replicates)
            delta_mae = np.empty(replicates)
            for i in range(replicates):
                w = weights[i][keys]
                e_c = errs[split][control]
                e_d = errs[split][candidate]
                delta_rmse[i] = math.sqrt(float(np.sum(w * e_c**2) / np.sum(w))) - math.sqrt(
                    float(np.sum(w * e_d**2) / np.sum(w))
                )
                delta_mae[i] = float(np.sum(w * (np.abs(e_c) - np.abs(e_d))) / np.sum(w))
            paired[name] = {
                "delta_rmse_point": rmse(errs[split][control] + ys[split][control], ys[split][control])
                - rmse(errs[split][candidate] + ys[split][candidate], ys[split][candidate]),
                "lo": float(np.percentile(delta_rmse, 2.5)),
                "hi": float(np.percentile(delta_rmse, 97.5)),
                "delta_mae_point": mae(errs[split][control] + ys[split][control], ys[split][control])
                - mae(errs[split][candidate] + ys[split][candidate], ys[split][candidate]),
                "mae_lo": float(np.percentile(delta_mae, 2.5)),
                "mae_hi": float(np.percentile(delta_mae, 97.5)),
                "interpretation": "positive delta = candidate better",
            }
        out.setdefault("paired_deltas", {})[split] = paired
    # restructure paired_deltas as {comparison: {split: {...}}}
    restructured: dict[str, Any] = {}
    for split, payload in out["paired_deltas"].items():
        for name, entry in payload.items():
            restructured.setdefault(name, {})[split] = entry
    out["paired_deltas"] = restructured
    return out


def read_mol_csv_from_manifest() -> list[str]:
    """SMILES in official row order, from the verified downloaded archive."""
    manifest = C.read_json(C.DATA_MANIFEST)
    root = C.DATA / "download" / "extracted" / C.OGB_DOWNLOAD_NAME
    if not root.is_dir():
        raise FileNotFoundError(
            f"the extracted OGB archive is missing at {C.track_rel(root)}; "
            "run `uv run python code/lp_source.py` first (local RDKit step)"
        )
    mol = S.read_mol_csv(root)
    if C.sha256_file(root / "mapping/mol.csv.gz") != manifest["files"]["mol_csv"]["sha256"]:
        raise ValueError("mol.csv.gz drifted from the recorded manifest")
    return list(mol["smiles"])


# ---------------------------------------------------------------------------
# structural groups (H1 / H2 / H3) and size-standardised gap
# ---------------------------------------------------------------------------


def morgan_support(smiles: Sequence[str], rows: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Train-document-frequency of (root, radius 2/3, identifier) environment tokens."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    train_rows = [int(r) for r in rows["train"]]
    tokens_per_mol: dict[int, set[tuple[int, int, int]]] = {}
    roots_per_mol: dict[int, set[tuple[int, int]]] = {}
    for row in train_rows:
        mol = Chem.MolFromSmiles(smiles[row])
        info: dict[int, Any] = {}
        rdMolDescriptors.GetMorganFingerprint(
            mol,
            C.MORGAN_RADIUS,
            useChirality=False,
            useBondTypes=True,
            useFeatures=False,
            useCounts=True,
            bitInfo=info,
            includeRedundantEnvironments=True,
        )
        tokens: set[tuple[int, int, int]] = set()
        roots: set[tuple[int, int]] = set()
        for identifier, environments in info.items():
            for atom_idx, radius in environments:
                if radius in C.MORGAN_RADIUS_FINE:
                    tokens.add((int(atom_idx), int(radius), int(identifier)))
                    roots.add((int(atom_idx), int(radius)))
        tokens_per_mol[row] = tokens
        roots_per_mol[row] = roots
    df: dict[tuple[int, int, int], int] = {}
    for row, tokens in tokens_per_mol.items():
        for token in tokens:
            df[token] = df.get(token, 0) + 1
    return {
        "df": df,
        "tokens_per_mol": tokens_per_mol,
        "roots_per_mol": roots_per_mol,
        "n_train": len(train_rows),
        "radius": C.MORGAN_RADIUS,
        "fine_radii": list(C.MORGAN_RADIUS_FINE),
    }


def row_env_support(smiles: str, support: Mapping[str, Any], *, is_train: bool, mol_row: int | None = None) -> dict[str, Any]:
    """Support of every environment of one molecule, with Train leave-one-out."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False}
    info: dict[int, Any] = {}
    rdMolDescriptors.GetMorganFingerprint(
        mol,
        C.MORGAN_RADIUS,
        useChirality=False,
        useBondTypes=True,
        useFeatures=False,
        useCounts=True,
        bitInfo=info,
        includeRedundantEnvironments=True,
    )
    df = support["df"]
    own = support["tokens_per_mol"].get(int(mol_row), set()) if is_train and mol_row is not None else set()
    per_root: dict[int, dict[int, int]] = {2: {}, 3: {}}
    valid_roots: dict[int, set[int]] = {2: set(), 3: set()}
    for identifier, environments in info.items():
        for atom_idx, radius in environments:
            if radius not in C.MORGAN_RADIUS_FINE:
                continue
            token = (int(atom_idx), int(radius), int(identifier))
            count = df.get(token, 0)
            if token in own:
                count -= 1
            per_root[radius].setdefault(int(atom_idx), []).append(int(count))
            valid_roots[radius].add(int(atom_idx))
    return {"valid": True, "per_root": per_root, "valid_roots": valid_roots, "mol": mol}


def group_membership(smiles: Sequence[str], rows: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Row-level membership of the three frozen groups plus per-atom root sets."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    support = morgan_support(smiles, rows)
    amide = Chem.MolFromSmarts("[NX3]-[CX3](=[OX1])")
    train_rows = set(int(r) for r in rows["train"])
    out: dict[str, Any] = {
        "support": {
            "n_train": support["n_train"],
            "n_tokens_r2": int(sum(1 for k in support["df"] if k[1] == 2)),
            "n_tokens_r3": int(sum(1 for k in support["df"] if k[1] == 3)),
        },
        "splits": {},
    }
    for split in C.SPLITS:
        entries = []
        for row in [int(r) for r in rows[split]]:
            env = row_env_support(smiles[row], support, is_train=split == "train", mol_row=row)
            if not env["valid"]:
                entries.append({"row": row, "status": "PARSE_FAILED"})
                continue
            r2_roots = env["valid_roots"][2]
            r3_roots = env["valid_roots"][3]
            missing = sorted(set(range(env["mol"].GetNumAtoms())) - r2_roots)
            h1_roots = sorted(a for a in r2_roots if max(env["per_root"][2][a]) == 0)
            h2_roots = sorted(
                a
                for a in r2_roots
                if max(env["per_root"][2][a]) >= C.GROUP_SUPPORT_MIN
                and (a not in r3_roots or max(env["per_root"][3][a]) == 0)
            )
            amide_atoms = sorted({int(i) for match in env["mol"].GetSubstructMatches(amide) for i in match})
            entries.append(
                {
                    "row": row,
                    "n_atoms": int(env["mol"].GetNumAtoms()),
                    "n_heavy": int(env["mol"].GetNumHeavyAtoms()),
                    "missing_r2_roots": missing,
                    "h1_roots": h1_roots,
                    "h2_roots": h2_roots,
                    "h3_roots": amide_atoms,
                    "h1": bool(h1_roots),
                    "h2": bool(h2_roots),
                    "h3": bool(amide_atoms),
                }
            )
        out["splits"][split] = entries
    return out


def size_bins(train_heavy: np.ndarray) -> dict[str, Any]:
    q1, q2 = np.quantile(train_heavy, [1 / 3, 2 / 3])
    return {"q1": float(q1), "q2": float(q2), "edges": [float(q1), float(q2)]}


def assign_bins(heavy: np.ndarray, edges: Mapping[str, Any]) -> np.ndarray:
    q1, q2 = float(edges["q1"]), float(edges["q2"])
    return np.where(heavy <= q1, 0, np.where(heavy <= q2, 1, 2))


def size_standardised_gap(
    errors: np.ndarray,
    membership: np.ndarray,
    heavy: np.ndarray,
    edges: Mapping[str, Any],
    train_marginal: np.ndarray,
) -> dict[str, Any]:
    bins = assign_bins(heavy, edges)
    per_bin = []
    common = []
    for b in (0, 1, 2):
        in_bin = bins == b
        pos = in_bin & membership
        neg = in_bin & ~membership
        if int(pos.sum()) < C.GROUP_MIN_BIN_SIDE or int(neg.sum()) < C.GROUP_MIN_BIN_SIDE:
            per_bin.append(
                {"bin": b, "n_group": int(pos.sum()), "n_other": int(neg.sum()), "usable": False}
            )
            continue
        gap = float(errors[pos].mean() - errors[neg].mean())
        per_bin.append(
            {
                "bin": b,
                "n_group": int(pos.sum()),
                "n_other": int(neg.sum()),
                "mae_group": float(errors[pos].mean()),
                "mae_other": float(errors[neg].mean()),
                "gap": gap,
                "usable": True,
            }
        )
        common.append(b)
    if len(common) < C.GROUP_MIN_BINS or int(membership.sum()) < C.GROUP_MIN_SIDE:
        return {"estimable": False, "per_bin": per_bin, "n_group": int(membership.sum()), "n_other": int((~membership).sum())}
    weights = np.asarray([train_marginal[b] for b in common], dtype=np.float64)
    weights = weights / weights.sum()
    gaps = np.asarray([per_bin[b]["gap"] for b in common], dtype=np.float64)
    return {
        "estimable": True,
        "gap": float(np.sum(weights * gaps)),
        "per_bin": per_bin,
        "common_bins": common,
        "bin_weights": weights.tolist(),
        "n_group": int(membership.sum()),
        "n_other": int((~membership).sum()),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_all(*, make_figures: bool = True) -> dict[str, Any]:
    arms = available_arms()
    if len(arms) < len(MAIN_ARMS):
        C.log(f"analysis over the available arms only: {arms}")
    C.TABLES.mkdir(parents=True, exist_ok=True)
    C.GROUPS.mkdir(parents=True, exist_ok=True)
    C.INTERVENTIONS.mkdir(parents=True, exist_ok=True)

    tables = {arm: split_table(arm) for arm in arms}
    C.write_json(C.TABLES / "main.json", tables)

    comparisons: dict[str, Any] = {}
    for name, (control, candidate) in CONTROLS.items():
        if control in arms and candidate in arms:
            comparisons[name] = compare(control, candidate)
    C.write_json(C.TABLES / "comparisons.json", comparisons)

    channel = _channel_used()
    judgements = {name: judge(comp, channel) for name, comp in comparisons.items()}
    bootstrap: dict[str, Any] = {}
    if all(a in arms for a in ("B", "D2")):
        bootstrap = bootstrap_payload(arms)
        C.write_json(C.TABLES / "bootstrap.json", bootstrap)
    arm_label = (
        arm_level_label(judgements, bootstrap)
        if bootstrap and "D2_vs_S" in judgements and "D2_vs_R2" in judgements
        else {"label": "INCOMPLETE"}
    )
    C.write_json(
        C.TABLES / "judgements.json",
        {"pairs": judgements, "arm_level": arm_label, "channel": channel},
    )

    groups: dict[str, Any] = {}
    interventions: dict[str, Any] = {}
    try:
        import lp_mechanism as MC

        groups = MC.run_groups()
        interventions = MC.run_interventions(groups.get("selection", {}))
    except Exception as exc:  # noqa: BLE001 - analysis must not lose the main tables
        C.log(f"mechanism stage incomplete: {type(exc).__name__}: {exc}")
        C.write_json(C.GROUPS / "error.json", {"error": f"{type(exc).__name__}: {exc}"})

    artifacts = [
        C.track_rel(C.TABLES / "main.json"),
        C.track_rel(C.TABLES / "comparisons.json"),
        C.track_rel(C.TABLES / "judgements.json"),
    ]
    if bootstrap:
        artifacts.append(C.track_rel(C.TABLES / "bootstrap.json"))
    if groups:
        artifacts.append(C.track_rel(C.GROUPS / "groups.json"))
    if interventions:
        artifacts.append(C.track_rel(C.INTERVENTIONS / "interventions.json"))

    try:
        import lp_report as R

        artifacts += R.write_tables(tables, comparisons, {"pairs": judgements, "arm_level": arm_label}, bootstrap, groups, interventions)
        if make_figures:
            artifacts += R.figures(tables, comparisons, groups, interventions)
    except Exception as exc:  # noqa: BLE001
        C.log(f"report stage incomplete: {type(exc).__name__}: {exc}")

    log_path = C.ANALYSIS / "test_access_log.json"
    log = C.read_json(log_path) if log_path.is_file() else {"reads": []}
    log["reads"].append(
        {
            "utc": C.utcnow(),
            "stage": "analyze",
            "what": "official Test predictions and labels of every trained arm",
            "n_test_rows": int(len(S.load_split()["rows"]["test"])),
            "purpose": "development-evaluation reporting; never used for selection",
            "selection_rule": "validation-only (lowest Val RMSE, ties keep the earlier epoch)",
        }
    )
    C.write_json(log_path, log)
    artifacts.append(C.track_rel(log_path))

    metrics = {
        "n_arms": len(arms),
        "arm_level_label": arm_label.get("label"),
        "judgements": {name: payload["label"] for name, payload in judgements.items()},
        "bootstrap_replicates": bootstrap.get("replicates"),
        "group_label": (groups.get("selection") or {}).get("label"),
        "n_figures": len([a for a in artifacts if a.endswith(".png")]),
    }
    return {"metrics": metrics, "artifacts": artifacts}
