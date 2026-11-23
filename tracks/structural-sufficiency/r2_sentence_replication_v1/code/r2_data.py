"""Split-parameterised, label-safe data layer for R2-SENTENCE-REPLICATION-v1.

GWIC-v0's ``gwic_data.load_bundle()`` is hard-wired to the D0 split.  This module
re-uses its *helpers* (raw/label loading, molecule views, deterministic scalars,
category support) unchanged and only parameterises the choice of Fit/held-out
rows, so D0 and D1 differ by nothing except the split membership.

Per-run statistics (target scaler, graph scalars, atom scalars, category
support) are always recomputed from the current run's Fit rows.  Official
Valid/Test labels are never loaded: the raw whitelist has no ``labels`` and no
``test_idx`` member, and every Valid label stays NaN in the training object.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import r2_common as RC
import gwic_data as GD  # frozen helper module
import gwic_structure as GS
import gwic_tensor as GT

ATOM_SCALAR_MODES = ("fit", "v0_fit_dev")


def _official(raw: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (np.sort(np.asarray(raw["train_idx"], np.int64)),
            np.sort(np.asarray(raw["valid_idx"], np.int64)))


def _check_partition(name: str, fit: np.ndarray, ev: np.ndarray,
                     official_train: np.ndarray) -> None:
    if fit.size == 0 or ev.size == 0:
        raise ValueError(f"{name}: empty split side")
    if np.unique(fit).size != fit.size or np.unique(ev).size != ev.size:
        raise ValueError(f"{name}: duplicate rows inside a side")
    if np.intersect1d(fit, ev).size:
        raise ValueError(f"{name}: fit/held-out overlap")
    if not np.isin(fit, official_train).all() or not np.isin(ev, official_train).all():
        raise ValueError(f"{name}: a row is outside official Train")
    if np.union1d(fit, ev).size != official_train.size:
        raise ValueError(f"{name}: fit + held-out do not cover official Train")


def load_split(name: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Load, verify and describe one frozen split.  Never reads a label."""
    pin = RC.split_pin(name)
    official_train, _ = _official(raw)
    if name == "D0":
        payload = GD._load_split(raw)          # frozen identity + coverage checks
        fit = np.asarray(payload["train_sub_rows"], np.int64)
        ev = np.asarray(payload["dev_rows"], np.int64)
        identity = str(payload["recomputed_sha256"])
        file_sha = str(payload["file_sha256"])
        if identity != RC.D0_IDENTITY or file_sha != RC.D0_FILE_SHA256:
            raise ValueError("D0 identity mismatch")
        extra = {"seed": int(payload["seed"]),
                 "n_scaffolds_fit": int(payload["n_scaffolds_train_sub"]),
                 "n_scaffolds_eval": int(payload["n_scaffolds_dev"]),
                 "n_scaffolds_official_train": int(payload["n_scaffolds_official_train"]),
                 "scaffold_definition": str(payload["scaffold_definition"]),
                 "assignment_rule": str(payload["assignment_rule"])}
        source = str(RC.D0_SPLIT.relative_to(RC.REPO_ROOT))
    elif name == "D1":
        if RC.sha256_file(RC.D1_SPLIT) != RC.D1_FILE_SHA256:
            raise ValueError("D1 split file sha mismatch")
        payload = RC.read_json(RC.D1_SPLIT)
        identity = RC.sha256_json({k: payload[k] for k in ("fit_rows", "heldout_rows", "seed")})
        if identity != payload.get("split_sha256") or identity != RC.D1_IDENTITY:
            raise ValueError("D1 identity mismatch")
        fit = np.asarray(payload["fit_rows"], np.int64)
        ev = np.asarray(payload["heldout_rows"], np.int64)
        scaf = payload["scaffold_by_row"]
        missing = [r for r in official_train.tolist() if str(r) not in scaf]
        if missing:
            raise ValueError(f"D1 scaffold map misses {len(missing)} official Train rows")
        fit_scaf = {scaf[str(r)] for r in fit.tolist()}
        ev_scaf = {scaf[str(r)] for r in ev.tolist()}
        if fit_scaf & ev_scaf:
            raise ValueError(f"D1 scaffold leakage: {len(fit_scaf & ev_scaf)} shared keys")
        if len(fit_scaf) != int(payload["n_scaffolds_fit"]) or \
                len(ev_scaf) != int(payload["n_scaffolds_heldout"]):
            raise ValueError("D1 scaffold counts do not match the frozen file")
        if len(fit_scaf) + len(ev_scaf) != int(payload["n_scaffolds_official_train"]):
            raise ValueError("D1 scaffold partition is not exhaustive")
        if payload.get("labels_read") is not False or payload.get("label_free") is not True:
            raise ValueError("D1 split is not flagged label-free")
        file_sha = RC.D1_FILE_SHA256
        extra = {"seed": int(payload["seed"]),
                 "target_heldout": int(payload["target_heldout"]),
                 "n_scaffolds_fit": len(fit_scaf),
                 "n_scaffolds_eval": len(ev_scaf),
                 "n_scaffolds_official_train": int(payload["n_scaffolds_official_train"]),
                 "scaffold_overlap": 0,
                 "scaffold_definition": str(payload["scaffold_definition"]),
                 "assignment_rule": str(payload["assignment_rule"]),
                 "role": str(payload["role"])}
        source = str(RC.D1_SPLIT.relative_to(RC.REPO_ROOT))
    else:
        raise KeyError(f"unknown split {name!r}")

    _check_partition(name, fit, ev, official_train)
    fit_sha = RC.array_sha256(fit)
    ev_sha = RC.array_sha256(ev)
    expected_fit = RC.D0_FIT_ROWS_SHA256 if name == "D0" else RC.D1_FIT_ROWS_SHA256
    expected_ev = RC.D0_EVAL_ROWS_SHA256 if name == "D0" else RC.D1_EVAL_ROWS_SHA256
    if fit_sha != expected_fit or ev_sha != expected_ev:
        raise ValueError(f"{name}: row hashes do not match the frozen pins")
    if fit.size != int(pin["n_fit"]) or ev.size != int(pin["n_eval"]):
        raise ValueError(f"{name}: unexpected split sizes {fit.size}/{ev.size}")
    spec = {
        "name": name, "source": source, "file_sha256": file_sha,
        "identity_sha256": identity, "fit_rows_sha256": fit_sha,
        "eval_rows_sha256": ev_sha, "n_fit": int(fit.size), "n_eval": int(ev.size),
        "role": pin.get("role", "development split (D0)"),
        "official_train_size": int(official_train.size),
        **extra,
    }
    return {"fit_rows": fit, "eval_rows": ev, "spec": spec}


def d1_recompute_scaffolds(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the D1 scaffold map and the whole split from SMILES (checks only).

    Uses exactly the frozen PSC algorithm: RDKit Bemis-Murcko
    (``includeChirality=False``), acyclic molecules keyed by their own SMILES,
    sorted keys, ``default_rng(seed).permutation``, greedy whole-group
    assignment until the held-out target is reached.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    payload = RC.read_json(RC.D1_SPLIT)
    official_train, _ = _official(raw)
    smiles = np.asarray(raw["smiles"])

    def key_of(sm: str) -> str:
        mol = Chem.MolFromSmiles(str(sm))
        if mol is None:
            return f"__UNPARSED__{sm}"
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaf if scaf else f"__ACYCLIC__{sm}"

    scaffold_of = {int(r): key_of(str(smiles[int(r)])) for r in official_train.tolist()}
    stored = payload["scaffold_by_row"]
    mismatched = [r for r, k in scaffold_of.items() if str(stored[str(r)]) != k]
    groups: dict[str, list[int]] = {}
    for row, key in scaffold_of.items():
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    order = np.random.default_rng(int(payload["seed"])).permutation(len(keys))
    held: list[int] = []
    target = int(payload["target_heldout"])
    held_scaffolds: list[str] = []
    for i in order.tolist():
        key = keys[i]
        if len(held) + len(groups[key]) <= target:
            held.extend(groups[key])
            held_scaffolds.append(key)
        if len(held) == target:
            break
    held = np.asarray(sorted(held), np.int64)
    held_set = set(held.tolist())
    fit = np.asarray([r for r in official_train.tolist() if r not in held_set], np.int64)
    fit_scaf = {scaffold_of[r] for r in fit.tolist()}
    return {
        "n_scaffold_map_mismatch": int(len(mismatched)),
        "n_official_train": int(official_train.size),
        "n_regenerated_fit": int(fit.size), "n_regenerated_heldout": int(held.size),
        "fit_rows_equal": bool(np.array_equal(fit, np.asarray(payload["fit_rows"], np.int64))),
        "heldout_rows_equal": bool(np.array_equal(held, np.asarray(payload["heldout_rows"], np.int64))),
        "scaffold_overlap_recomputed": int(len(fit_scaf & set(held_scaffolds))),
        "n_scaffolds_fit_recomputed": int(len(fit_scaf)),
        "n_scaffolds_heldout_recomputed": int(len(held_scaffolds)),
        "n_scaffolds_official_train_recomputed": int(len(keys)),
    }


def load_bundle(split: str = "D0", atom_scalar_mode: str = "fit") -> GD.DataBundle:
    """The GWIC-v0 training object, but for ``split`` and fit-only statistics."""
    if atom_scalar_mode not in ATOM_SCALAR_MODES:
        raise KeyError(f"unknown atom scalar mode {atom_scalar_mode!r}")
    raw = GD._load_raw()
    labels_file = GD._load_train_labels()
    official_train, official_valid = _official(raw)
    n_mol = int(np.asarray(raw["node_ptr"]).size - 1)
    if not np.array_equal(np.asarray(raw["row_ids"]), np.arange(n_mol)):
        raise ValueError("raw row ids are not 0..N-1")
    if not np.array_equal(np.sort(labels_file["row_id"]), official_train):
        raise ValueError("train label rows are not the official Train rows")
    labels = np.full(n_mol, np.nan, np.float64)
    labels[labels_file["row_id"]] = labels_file["label"]
    if not np.isfinite(labels[official_train]).all():
        raise ValueError("missing finite label on an official Train row")
    if np.isfinite(labels[official_valid]).any():
        raise ValueError("official Valid row carries a finite label")

    loaded = load_split(split, raw)
    fit, ev = loaded["fit_rows"], loaded["eval_rows"]
    if not np.isfinite(labels[fit]).all() or not np.isfinite(labels[ev]).all():
        raise ValueError("non-finite label on a fit or held-out row")
    y = labels[fit]
    scaler = {"mean": float(y.mean()), "std": float(y.std(ddof=0)), "n": int(y.size)}
    if not (scaler["std"] > 0):
        raise ValueError("zero target scale")

    graph = np.zeros((n_mol, 8), np.float64)
    for row in range(n_mol):
        graph[row] = GD.graph_scalars(GD.molecule_view(raw, row))
    graph_mean = graph[fit].mean(axis=0)
    graph_scale = graph[fit].std(axis=0)
    graph_scale = np.where(graph_scale > 1e-12, graph_scale, 1.0)

    if atom_scalar_mode == "fit":
        scalar_rows = fit
    else:  # v0_fit_dev: the GWIC-v0 implementation, kept only for audit parity
        scalar_rows = np.union1d(fit, ev)
    atom_rows = [GD.atom_scalars(GD.molecule_view(raw, int(r)))
                 for r in scalar_rows.tolist()]
    all_atom_scalars = np.concatenate(atom_rows, axis=0)
    atom_mean = all_atom_scalars.mean(axis=0)
    atom_scale = all_atom_scalars.std(axis=0)
    atom_scale = np.where(atom_scale > 1e-12, atom_scale, 1.0)

    hashes = {
        "raw_cache": {"path": str(RC.RAW_CACHE.relative_to(RC.REPO_ROOT)),
                      "sha256": raw["cache_sha256"],
                      "members": list(RC.RAW_STRUCT_KEYS)},
        "train_labels": {"path": str(RC.TRAIN_LABELS.relative_to(RC.REPO_ROOT)),
                         "sha256": labels_file["sha256"],
                         "rows": int(labels_file["row_id"].size)},
        "split": {"path": loaded["spec"]["source"],
                  "sha256": loaded["spec"]["identity_sha256"],
                  "file_sha256": loaded["spec"]["file_sha256"],
                  "n_fit": int(fit.size), "n_eval": int(ev.size),
                  "fit_rows_sha256": loaded["spec"]["fit_rows_sha256"],
                  "eval_rows_sha256": loaded["spec"]["eval_rows_sha256"]},
        "atom_scalar_mode": atom_scalar_mode,
    }
    return GD.DataBundle(raw=raw, labels=labels,
                         row_id=np.asarray(raw["row_ids"], np.int64),
                         official_train=official_train, official_valid=official_valid,
                         fit_rows=fit, eval_rows=ev, split=loaded["spec"],
                         scaler=scaler, graph_scalars=graph,
                         graph_mean=graph_mean, graph_scale=graph_scale,
                         atom_scalar_mean=atom_mean, atom_scalar_scale=atom_scale,
                         hashes=hashes)


def norm_of(bundle: GD.DataBundle) -> dict[str, np.ndarray]:
    return {"atom_scalar_mean": bundle.atom_scalar_mean,
            "atom_scalar_scale": bundle.atom_scalar_scale,
            "graph_mean": bundle.graph_mean,
            "graph_scale": bundle.graph_scale}


def batch_from_views(views: Sequence[Mapping[str, Any]],
                     device: Any = "cpu") -> dict[str, Any]:
    """Collate arbitrary explicit molecule views (used by the invariance checks)."""
    import gwic_synthetic as SYN

    views = [dict(v) for v in views]
    for view in views:
        view["atom_scalars"] = GD.atom_scalars(view)
    raw = SYN.raw_from_views(views)
    structs = {i: GS.build_molecule_struct(view) for i, view in enumerate(views)}
    return GT.collate(list(range(len(views))), structs, raw, device)


def permuted_view(view: Mapping[str, Any], perm: Sequence[int],
                  bond_order: Sequence[int] | None = None) -> dict[str, Any]:
    import gwic_synthetic as SYN

    return SYN.permute_view(view, perm, bond_order)