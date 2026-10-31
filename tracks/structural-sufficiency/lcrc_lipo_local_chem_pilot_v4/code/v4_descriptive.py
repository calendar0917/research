"""Fold-1 data diagnostics on the frozen Fit2352 structures (CPU, label-aware only in Fit).

Scope
-----
Every structural quantity (formal charge, HBD/HBA, TPSA, MolLogP, atom count,
aromatic atom fraction, duplicate encodings) is computed from the label-free
frozen SMILES.  Labels are read only from the frozen **Fit** label artifact and
only to (a) describe the target distribution, (b) count exact boundary labels,
(c) compare labels of duplicated structure encodings, and (d) report the
frozen A-R3 Fold-1 dev residuals by group.  Monitor, Reserved and official
valid/test labels are never opened.

Nothing here selects a model or changes a pre-registered threshold.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import v4_common as C


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - y
    sse = float(np.sum(err * err))
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt(np.mean(err * err))) if y.size else None,
        "mae": float(np.mean(np.abs(err))) if y.size else None,
        "bias": float(np.mean(err)) if y.size else None,
        "sum_squared_error": sse,
        "share_of_total_sse": None,
    }


def _attach_shares(groups: dict[str, dict[str, Any]], total_n: int) -> None:
    total = sum(g["sum_squared_error"] for g in groups.values())
    for g in groups.values():
        g["share_of_total_sse"] = (g["sum_squared_error"] / total) if total > 0 else None
        g["row_share"] = (g["n"] / total_n) if total_n else None


def _dist(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    q = np.quantile(x, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(q[0]),
        "q01": float(q[1]),
        "q05": float(q[2]),
        "q25": float(q[3]),
        "median": float(q[4]),
        "q75": float(q[5]),
        "q95": float(q[6]),
        "q99": float(q[7]),
        "max": float(q[8]),
    }


# ---------------------------------------------------------------------------
# per-molecule structure table
# ---------------------------------------------------------------------------


def molecule_table() -> dict[str, Any]:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

    cache_info = C.load_fit_cache()
    cache = cache_info["cache"]
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    folds = C.F.load_folds()
    train_pos = [int(p) for p in C.F.train_positions(folds, C.FOLD)]
    dev_pos = [int(p) for p in C.F.fold_positions(folds, C.FOLD)]
    y_fit = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    n = int(cache["n_molecules"])
    if n != len(fit_rows):
        raise ValueError("cache/Fit size mismatch")

    charges = np.zeros(n, dtype=np.int64)
    hbd = np.zeros(n, dtype=np.int64)
    hba = np.zeros(n, dtype=np.int64)
    tpsa = np.zeros(n, dtype=np.float64)
    logp = np.zeros(n, dtype=np.float64)
    n_atoms = np.zeros(n, dtype=np.int64)
    aromatic_fraction = np.zeros(n, dtype=np.float64)
    morgan = []
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    canonical = []
    raw_smiles = []
    for p in range(n):
        smiles = str(cache["smiles"][p])
        raw_smiles.append(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit cannot parse cache SMILES at {p}")
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
        charges[p] = int(Chem.GetFormalCharge(mol))
        hbd[p] = int(Descriptors.NumHDonors(mol))
        hba[p] = int(Descriptors.NumHAcceptors(mol))
        tpsa[p] = float(Descriptors.TPSA(mol))
        logp[p] = float(Crippen.MolLogP(mol))
        n_atoms[p] = int(mol.GetNumAtoms())
        aromatic_fraction[p] = float(np.mean([a.GetIsAromatic() for a in mol.GetAtoms()]))
        fp = fp_gen.GetFingerprint(mol)
        morgan.append(C.sha256_text("".join("1" if b else "0" for b in fp)))
    table = {
        "positions": np.arange(n, dtype=np.int64),
        "rows": np.asarray(fit_rows, dtype=np.int64),
        "y": y_fit,
        "formal_charge": charges,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "mol_logp": logp,
        "n_atoms": n_atoms,
        "aromatic_atom_fraction": aromatic_fraction,
        "canonical_smiles_sha256": [C.sha256_text(s) for s in canonical],
        "raw_smiles_sha256": [C.sha256_text(s) for s in raw_smiles],
        "morgan2_2048_sha256": morgan,
        "train_positions": train_pos,
        "dev_positions": dev_pos,
        "rdkit_version": str(rdkit.__version__),
    }
    return table


# ---------------------------------------------------------------------------
# duplicate structure encodings
# ---------------------------------------------------------------------------


def _duplicate_report(keys: Sequence[str], y: np.ndarray, split_of: Sequence[str], label: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(str(key), []).append(i)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    conflicting = {}
    for k, idx in dup.items():
        values = y[np.asarray(idx, dtype=np.int64)]
        if float(values.max() - values.min()) > 1e-9:
            conflicting[k] = {
                "n": len(idx),
                "positions": idx,
                "rows": [int(x) for x in np.asarray(idx)],
                "labels": [float(x) for x in values],
                "label_range": float(values.max() - values.min()),
                "splits": [split_of[i] for i in idx],
            }
    return {
        "encoding": label,
        "n_molecules": int(len(keys)),
        "n_unique_keys": int(len(groups)),
        "n_duplicate_keys": int(len(dup)),
        "n_molecules_in_duplicate_keys": int(sum(len(v) for v in dup.values())),
        "n_keys_with_conflicting_labels": int(len(conflicting)),
        "conflicting": list(conflicting.values())[:20],
        "all_conflicts": conflicting,
    }


# ---------------------------------------------------------------------------
# group diagnostics for the frozen A-R3 residuals
# ---------------------------------------------------------------------------


def _ar3_dev() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = C.HISTORICAL["key"]
    train = C.read_json(C.V2_RUNS / key / "train.json")
    data = np.load(C.V2_RUNS / key / "predictions.npz")
    rows = [int(x) for x in data["rows"].tolist()]
    pred = np.asarray(data["pred"], dtype=np.float64)
    folds = C.F.load_folds()
    dev_pos = [int(p) for p in C.F.fold_positions(folds, C.FOLD)]
    y = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)[np.asarray(dev_pos, dtype=np.int64)]
    if rows != dev_pos:
        raise ValueError("historical A-R3 prediction rows do not match Fold-1 dev positions")
    if abs(float(train["metrics"]["dev_rmse_raw"]) - C.HISTORICAL["dev_rmse"]) > 1e-9:
        raise ValueError("historical A-R3 dev RMSE constant changed")
    return np.asarray(dev_pos, dtype=np.int64), y, pred


def _group_metrics(mask: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if not mask.any():
        return {"n": 0, "rmse": None, "mae": None, "bias": None, "sum_squared_error": 0.0, "row_share": None}
    out = _metrics(y[mask], pred[mask])
    out["row_share"] = float(mask.mean())
    return out


def error_groups(table: dict[str, Any]) -> dict[str, Any]:
    dev_pos, y, pred = _ar3_dev()
    pos_index = {int(p): i for i, p in enumerate(dev_pos)}
    dev_local = np.asarray([pos_index[int(p)] for p in table["dev_positions"]], dtype=np.int64)
    # table rows are indexed by cache position == dev_positions order
    if not np.array_equal(np.asarray(table["dev_positions"], dtype=np.int64), dev_pos):
        raise ValueError("dev positions disagree between the table and the fold file")
    train_pos = np.asarray(table["train_positions"], dtype=np.int64)
    tpsa = np.asarray(table["tpsa"], dtype=np.float64)
    hbd = np.asarray(table["hbd"], dtype=np.int64)
    hba = np.asarray(table["hba"], dtype=np.int64)
    charge = np.asarray(table["formal_charge"], dtype=np.int64)
    n_atoms = np.asarray(table["n_atoms"], dtype=np.int64)
    dev_tpsa = tpsa[dev_local]
    dev_tpsa_train = tpsa[train_pos]
    cuts = np.unique(np.quantile(dev_tpsa_train, [0.25, 0.5, 0.75, 0.9]))
    groups: dict[str, Any] = {}

    def add(name: str, values: np.ndarray, definitions: Sequence[tuple[str, Any]]) -> None:
        block = {}
        for label, fn in definitions:
            mask = np.asarray([bool(fn(v)) for v in values])
            block[label] = _group_metrics(mask, y, pred)
        _attach_shares(block, len(values))
        groups[name] = block

    add(
        "formal_charge",
        charge[dev_local],
        [
            ("neutral", lambda v: v == 0),
            ("charged", lambda v: v != 0),
            ("positive", lambda v: v > 0),
            ("negative", lambda v: v < 0),
        ],
    )
    add(
        "hbd_lipinski",
        hbd[dev_local],
        [
            ("0", lambda v: v == 0),
            ("1", lambda v: v == 1),
            ("2", lambda v: v == 2),
            ("3", lambda v: v == 3),
            (">=4", lambda v: v >= 4),
        ],
    )
    add(
        "hba_lipinski",
        hba[dev_local],
        [
            ("0-1", lambda v: v <= 1),
            ("2-3", lambda v: 2 <= v <= 3),
            ("4-5", lambda v: 4 <= v <= 5),
            (">=6", lambda v: v >= 6),
        ],
    )
    tpsa_defs = []
    edges = [-np.inf, *cuts.tolist(), np.inf]
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if np.isneginf(lo):
            tpsa_defs.append((f"<= {hi:.4g}", lambda v, hi=hi: v <= hi))
        elif np.isposinf(hi):
            tpsa_defs.append((f"> {lo:.4g}", lambda v, lo=lo: v > lo))
        else:
            tpsa_defs.append((f"({lo:.4g}, {hi:.4g}]", lambda v, lo=lo, hi=hi: lo < v <= hi))
    add("tpsa_train_quantiles", dev_tpsa, tpsa_defs)
    add(
        "polarity_flags",
        dev_tpsa,
        [
            ("tpsa_ge_train_q75", lambda v: v >= float(cuts[2])),
            ("tpsa_ge_train_q90", lambda v: v >= float(cuts[3])),
        ],
    )
    add(
        "combinations",
        np.stack([dev_tpsa, hbd[dev_local].astype(float), charge[dev_local].astype(float)], axis=1),
        [
            ("charged_or_hbd_ge_3", lambda v: v[2] != 0 or v[1] >= 3),
            ("tpsa_ge_q75_and_hbd_ge_2", lambda v: v[0] >= float(cuts[2]) and v[1] >= 2),
            ("uncharged_and_tpsa_below_q75", lambda v: v[2] == 0 and v[0] < float(cuts[2])),
        ],
    )
    return {
        "source": C.HISTORICAL["key"],
        "split": "fold1_dev",
        "n": int(y.size),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "tpsa_train_cuts": [float(x) for x in cuts],
        "groups": groups,
        "dev_label_stats": _dist(y),
        "dev_atom_count_stats": _dist(n_atoms[dev_local]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_descriptive() -> dict[str, Any]:
    table = molecule_table()
    y = np.asarray(table["y"], dtype=np.float64)
    train_pos = np.asarray(table["train_positions"], dtype=np.int64)
    dev_pos = np.asarray(table["dev_positions"], dtype=np.int64)
    split_of = np.asarray(["train" if p in set(train_pos.tolist()) else "dev" for p in range(y.size)])
    logp = np.asarray(table["mol_logp"], dtype=np.float64)
    charge = np.asarray(table["formal_charge"], dtype=np.int64)
    hbd = np.asarray(table["hbd"], dtype=np.int64)
    hba = np.asarray(table["hba"], dtype=np.int64)
    tpsa = np.asarray(table["tpsa"], dtype=np.float64)
    n_atoms = np.asarray(table["n_atoms"], dtype=np.int64)
    arom = np.asarray(table["aromatic_atom_fraction"], dtype=np.float64)
    diff = y - logp

    def block(idx: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(idx.size),
            "atom_count": _dist(n_atoms[idx]),
            "aromatic_atom_fraction": _dist(arom[idx]),
            "tpsa": _dist(tpsa[idx]),
            "mol_logp": _dist(logp[idx]),
            "hbd": {str(k): int((hbd[idx] == k).sum()) for k in range(0, 7)},
            "hbd_ge4": int((hbd[idx] >= 4).sum()),
            "hba": {str(k): int((hba[idx] == k).sum()) for k in range(0, 12)},
            "formal_charge": {str(k): int((charge[idx] == k).sum()) for k in np.unique(np.concatenate([charge[idx], [0]]))},
            "formal_charge_nonzero": int((charge[idx] != 0).sum()),
            "formal_charge_abs_ge2": int((np.abs(charge[idx]) >= 2).sum()),
            "label": _dist(y[idx]),
            "label_minus_mol_logp": _dist(diff[idx]),
            "label_at_minus_1p5": int(np.isclose(y[idx], -1.5, atol=1e-9).sum()),
            "label_at_4p5": int(np.isclose(y[idx], 4.5, atol=1e-9).sum()),
            "label_at_boundary_any": int((np.isclose(y[idx], -1.5, atol=1e-9) | np.isclose(y[idx], 4.5, atol=1e-9)).sum()),
            "label_min": float(y[idx].min()),
            "label_max": float(y[idx].max()),
            "distinct_labels": int(np.unique(y[idx]).size),
        }

    duplicates = [
        _duplicate_report(table["canonical_smiles_sha256"], y, split_of, "canonical_smiles"),
        _duplicate_report(table["raw_smiles_sha256"], y, split_of, "raw_smiles_string"),
        _duplicate_report(table["morgan2_2048_sha256"], y, split_of, "morgan_radius2_2048bit"),
    ]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "rdkit_version": table["rdkit_version"],
        "scope": {
            "fit_rows": 2352,
            "fold": C.FOLD,
            "train_n": int(train_pos.size),
            "dev_n": int(dev_pos.size),
            "labels_source": "frozen Fit-only label artifact (phase1a labels_fit_monitor, fit values only)",
            "monitor_used": False,
            "reserved_used": False,
            "official_valid_test_used": False,
            "label_used_for_thresholds": False,
        },
        "all_fit": block(np.arange(y.size)),
        "fold1_train": block(train_pos),
        "fold1_dev": block(dev_pos),
        "duplicate_encodings": [{k: v for k, v in d.items() if k != "all_conflicts"} for d in duplicates],
        "duplicate_conflicts_full": {d["encoding"]: d["all_conflicts"] for d in duplicates},
        "ar3_dev_error_groups": error_groups(table),
        "dataset_notes": {
            "label_semantics": (
                "the frozen LCRC lipophilicity label is the octanol/water distribution coefficient logD; there is no "
                "separate experimental logP column in the frozen artifacts"
            ),
            "logd_minus_mollogp": (
                "label_minus_mol_logp is therefore literally logD - RDKit Crippen MolLogP computed on the frozen "
                "heavy-atom graph; it is a descriptive chemistry offset, not a model input"
            ),
            "boundary_labels": (
                "only 3 of 2352 Fit molecules sit on a reported clipping boundary (4.5); no -1.5 boundary labels"
            ),
            "formal_charge_coverage": (
                "2351 of 2352 Fit molecules are neutral and the single charged molecule is in the Fold-1 train split, "
                "so the formal-charge subgroup is empty on dev and cannot be evaluated in this round"
            ),
            "duplicate_structure_conflicts": (
                "morgan_radius2_2048bit collisions are recorded; identical encodings with different labels bound the "
                "achievable RMSE for any representation built only from the frozen graph"
            ),
        },
        "interpretation_guard": (
            "descriptive only: these numbers rank error concentration on the frozen A-R3 dev residuals; "
            "they do not select a model or alter any pre-registered threshold"
        ),
    }
    C.write_json(C.ANALYSIS / "descriptive_v4.json", payload)
    return payload


__all__ = ["molecule_table", "error_groups", "run_descriptive"]
