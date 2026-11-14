"""Dataset/split loading, targets, Train-only support tables and aux targets.

Caches are the frozen label-free ``lcrc_molnet_benchmark_pilot_v2`` structure
caches (same ``lipo_v1`` FeatureSpec for all five datasets).  They are consumed
read-only; this round only records their hashes and row ids.

Everything fitted here uses the Train split only:

* regression target mean/std,
* the S-family support table ``c(key)``, ``kappa`` and ``a_bar``,
* the auxiliary targets (WC_LOGP / MolWt / one fixed shuffle).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lcpr_common as C

import lcrc_data as PREP_DATA
import mb2_source as SRC

CACHE_VERSION = "lcrc-performance-route-v1"


# ---------------------------------------------------------------------------
# source / cache
# ---------------------------------------------------------------------------


def dataset_meta(dataset: str) -> dict[str, Any]:
    if dataset not in C.DATASETS:
        raise KeyError(dataset)
    return dict(C.DATASETS[dataset])


def load_source(dataset: str, labels: str = "train_valid") -> dict[str, Any]:
    return SRC.load_dataset(dataset, verify=True, labels=labels)


def cache_path(dataset: str, split: str) -> Path:
    return C.UPSTREAM_CACHE / f"{dataset}_{split}.pt"


def frozen_spec():
    from mb2_data import frozen_spec as mb2_spec

    spec = mb2_spec()
    if spec.hash() != C.FEATURE_SPEC_HASH:
        raise ValueError(f"lipo_v1 spec hash {spec.hash()} != frozen {C.FEATURE_SPEC_HASH}")
    return spec


def load_cache(dataset: str, split: str, rows: Sequence[int] | None = None) -> dict[str, Any]:
    path = cache_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"missing label-free structure cache {path}; run prepare on a host with it")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache["meta"].get("feature_spec_hash") != C.FEATURE_SPEC_HASH:
        raise ValueError(f"{path}: feature spec hash differs from the frozen lipo_v1 spec")
    if cache["meta"].get("labels_present"):
        raise ValueError(f"{path}: structure cache must be label-free")
    got = [int(x) for x in np.asarray(cache["rows"]).tolist()]
    if rows is not None and got != [int(r) for r in rows]:
        raise ValueError(f"{path}: cached row ids differ from the published split")
    if int(cache["n_molecules"]) != len(got):
        raise ValueError(f"{path}: cache size differs")
    return cache


def verify_dataset(dataset: str) -> dict[str, Any]:
    """Check split sizes, row hashes and cache/source agreement."""
    meta = dataset_meta(dataset)
    src = load_source(dataset, labels="none")
    sizes = tuple(len(src["rows"][s]) for s in C.SPLITS)
    if sizes != tuple(meta["sizes"]):
        raise RuntimeError(f"{dataset}: published split sizes {sizes} != frozen {meta['sizes']}")
    out = {
        "dataset": dataset,
        "task": meta["task"],
        "split_method": meta["split_method"],
        "sizes": list(sizes),
        "rows_sha256": {s: C.row_sha256(src["rows"][s]) for s in C.SPLITS},
        "n_total": int(len(src["smiles"])),
        "label_column": src["meta"]["label_columns"],
        "units": src["meta"]["units"],
        "splits": {},
    }
    for split in C.SPLITS:
        cache = load_cache(dataset, split, src["rows"][split])
        out["splits"][split] = {
            "rows_sha256": C.row_sha256(np.asarray(cache["rows"]).tolist()),
            "content_fingerprint": cache["meta"]["content_fingerprint"],
            "n_atoms": int(np.asarray(cache["atom_ptr"])[-1]),
            "n_paths": int(len(cache["path_cat"])),
            "cache_sha256": C.sha256_file(cache_path(dataset, split)),
            "labels_present": bool(cache["meta"]["labels_present"]),
        }
    return out


def targets(dataset: str, *, include_test: bool) -> dict[str, Any]:
    """Published targets; Test is only decoded by the caller when it asks for it."""
    labels = "all" if include_test else "train_valid"
    src = load_source(dataset, labels=labels)
    meta = dataset_meta(dataset)
    out: dict[str, Any] = {
        "task": meta["task"],
        "primary": meta["primary"],
        "units": meta["units"],
        "split_method": meta["split_method"],
        "rows": {s: [int(x) for x in src["rows"][s]] for s in C.SPLITS},
        "y": {},
    }
    for split in ("train", "valid") + (("test",) if include_test else ()):
        out["y"][split] = np.asarray(src["y"][split], dtype=np.float64).reshape(-1)
    if meta["task"] == "regression":
        y = out["y"]["train"]
        out["target_norm"] = {
            "mean": float(np.mean(y)),
            "scale": float(np.std(y, ddof=0)),
            "n": int(y.size),
            "source": "train_split_only",
            "formula": "z = (y - mean)/scale; y_raw = mean + scale*z",
        }
        if out["target_norm"]["scale"] <= 0:
            raise ValueError("degenerate Train target distribution")
    else:
        pos = float(np.mean(out["y"]["train"] > 0.5))
        out["target_norm"] = {"pos_weight": 1.0, "train_positive_rate": pos,
                              "loss": "BCEWithLogits, no pos_weight, raw 0/1 labels"}
    return out


# ---------------------------------------------------------------------------
# chem8 atom channels (historical frozen 8 columns, label-free)
# ---------------------------------------------------------------------------

CHEM8_PATH = C.LONGRUN / "analysis/source/chem8/{dataset}.npz"
CHEM8_CHANNELS = (
    "CIP_R", "CIP_S", "CIP_UNASSIGNED_TETRA", "HBD", "HBA", "HYB_SP", "HYB_SP2", "HYB_SP3",
)


def chem8_atoms(dataset: str, split: str, cache: Mapping[str, Any]) -> np.ndarray:
    """Per-atom 8 columns in the cache's own atom order."""
    path = Path(str(CHEM8_PATH).format(dataset=dataset))
    if not path.exists():
        raise FileNotFoundError(f"missing chem8 artefact {path}")
    z = np.load(path, allow_pickle=False)
    if tuple(str(x) for x in z["channel_names"]) != CHEM8_CHANNELS:
        raise ValueError(f"{path}: chem8 channel order changed")
    atom_ptr = np.asarray(z["atom_ptr"])
    chem8 = np.asarray(z["chem8"])
    rows = [int(x) for x in np.asarray(cache["rows"]).tolist()]
    cache_atom_ptr = np.asarray(cache["atom_ptr"])
    blocks = []
    for i, row in enumerate(rows):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        c0, c1 = int(cache_atom_ptr[i]), int(cache_atom_ptr[i + 1])
        if (a1 - a0) != (c1 - c0):
            raise ValueError(f"{dataset}/{split}: chem8 atom count differs at row {row}")
        blocks.append(chem8[a0:a1].astype(np.float32))
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 8), np.float32)


# ---------------------------------------------------------------------------
# S-family support table (Train structures only, label-blind)
# ---------------------------------------------------------------------------


def support_path(dataset: str) -> Path:
    return C.CACHE / f"support_{dataset}.npz"


def build_support(dataset: str) -> dict[str, Any]:
    """Frozen support table from Train *structures* only.

    ``key = (path_cat, path_end63)``; ``c(key)`` = number of distinct Train
    molecules that contain the key at least once.
    ``kappa = max(1, median_{occupied keys} c(key))``.
    ``a_bar`` = mean over all Train root atoms of ``mean_{paths at r} c/(c+kappa)``
    (an atom with no path contributes 1).
    """
    train = load_cache(dataset, "train")
    path_ptr = np.asarray(train["path_ptr"])
    cat = np.asarray(train["path_cat"]).astype(np.int64)
    end = np.asarray(train["path_end63"]).astype(np.int64)
    n_atoms = int(np.asarray(train["atom_ptr"])[-1])
    counts: dict[tuple[int, int], int] = {}
    for a in range(n_atoms):
        p0, p1 = int(path_ptr[a]), int(path_ptr[a + 1])
        if p1 <= p0:
            continue
        for k in {(int(cat[i]), int(end[i])) for i in range(p0, p1)}:
            counts[k] = counts.get(k, 0) + 1
    occ = np.asarray(sorted(counts.values()), dtype=np.float64)
    kappa = float(max(1.0, np.median(occ))) if occ.size else 1.0
    keys = np.asarray(sorted(counts.keys()), dtype=np.int64).reshape(-1, 2)
    vals = np.asarray([counts[(int(k[0]), int(k[1]))] for k in keys], dtype=np.float32)

    def _key_counts(cache) -> np.ndarray:
        pc = np.asarray(cache["path_cat"]).astype(np.int64)
        pe = np.asarray(cache["path_end63"]).astype(np.int64)
        idx = np.stack([pc, pe], axis=1) if len(pc) else np.zeros((0, 2), np.int64)
        return np.searchsorted(
            keys[:, 0] * (int(C.N_PATH_CODES) * int(C.N_NBR_CODES)) + keys[:, 1],
            idx[:, 0] * (int(C.N_PATH_CODES) * int(C.N_NBR_CODES)) + idx[:, 1],
            side="left",
        ).astype(np.int32) if len(idx) else np.zeros(0, np.int32)

    flat_keys = keys[:, 0] * (int(C.N_PATH_CODES) * int(C.N_NBR_CODES)) + keys[:, 1]

    def _path_counts(cache) -> np.ndarray:
        pc = np.asarray(cache["path_cat"]).astype(np.int64)
        pe = np.asarray(cache["path_end63"]).astype(np.int64)
        flat = pc * (int(C.N_PATH_CODES) * int(C.N_NBR_CODES)) + pe
        out = np.zeros(len(flat), dtype=np.float32)
        if len(flat) == 0:
            return out
        pos = np.clip(np.searchsorted(flat_keys, flat), 0, max(len(vals) - 1, 0))
        ok = flat_keys[pos] == flat
        out[ok] = vals[pos[ok]]
        return out

    def _mean_root_coeff(coeff: np.ndarray) -> float:
        total = 0.0
        for a in range(n_atoms):
            p0, p1 = int(path_ptr[a]), int(path_ptr[a + 1])
            if p1 <= p0:
                total += 1.0
                continue
            flat = cat[p0:p1] * (int(C.N_PATH_CODES) * int(C.N_NBR_CODES)) + end[p0:p1]
            pos = np.clip(np.searchsorted(flat_keys, flat), 0, max(len(vals) - 1, 0))
            ok = flat_keys[pos] == flat
            cur = np.ones(len(flat))
            cur[ok] = coeff[pos[ok]]
            total += float(np.mean(cur))
        return float(total / max(n_atoms, 1))

    a_bar = _mean_root_coeff(vals / (vals + kappa))
    a_bar_k4 = _mean_root_coeff(vals / (vals + 4.0 * kappa))

    per_split: dict[str, np.ndarray] = {}
    for split in C.SPLITS:
        cache = load_cache(dataset, split)
        per_split[split] = _path_counts(cache)
    np.savez_compressed(
        support_path(dataset),
        keys=keys, counts=vals, kappa=np.float64(kappa), a_bar=np.float64(a_bar),
        a_bar_k1=np.float64(a_bar), a_bar_k4=np.float64(a_bar_k4),
        **{f"path_count_{s}": per_split[s] for s in C.SPLITS},
    )
    return {
        "dataset": dataset,
        "n_occupied_keys": int(len(vals)),
        "n_train_molecules": int(train["n_molecules"]),
        "kappa": kappa,
        "a_bar": a_bar,
        "a_bar_k1": a_bar,
        "a_bar_k4": a_bar_k4,
        "count_quantiles": {
            "min": float(vals.min()), "q25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)), "q75": float(np.percentile(vals, 75)),
            "max": float(vals.max()),
        },
        "per_split_paths": {s: int(len(per_split[s])) for s in C.SPLITS},
        "per_split_frac_count_le_1": {s: float(np.mean(per_split[s] <= 1)) if len(per_split[s]) else 0.0
                                      for s in C.SPLITS},
        "label_free": True,
        "identical_train_and_eval_rule": True,
    }


def load_support(dataset: str) -> dict[str, Any]:
    path = support_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing support table {path}; run prepare first")
    z = np.load(path, allow_pickle=False)
    return {
        "kappa": float(z["kappa"]),
        "a_bar": float(z["a_bar"]),
        "a_bar_k1": float(z["a_bar_k1"]),
        "a_bar_k4": float(z["a_bar_k4"]),
        "path_count": {s: np.asarray(z[f"path_count_{s}"]) for s in C.SPLITS},
        "keys": np.asarray(z["keys"]),
        "counts": np.asarray(z["counts"]),
    }


# ---------------------------------------------------------------------------
# support table v2: DISTINCT-MOLECULE support (the definition the round stated)
# ---------------------------------------------------------------------------


def support_v2_path(dataset: str) -> Path:
    return C.CACHE / f"support_v2_{dataset}.npz"


def _mol_key_counts(atom_ptr, path_ptr, cat, end, n_mol: int) -> dict[tuple[int, int], int]:
    """``c(key)`` = number of distinct molecules that contain ``key`` at least once.

    A molecule's roots are the atoms in ``[atom_ptr[m], atom_ptr[m+1])``; the
    paths rooted at those atoms are contiguous in ``path_ptr`` (per-atom prefix
    sums), so the molecule's whole path block is ``[path_ptr[a0], path_ptr[a1])``.
    Keys are de-duplicated per molecule *before* counting, so extra roots that
    carry the same key do not inflate the count.
    """
    atom_ptr = np.asarray(atom_ptr, dtype=np.int64)
    path_ptr = np.asarray(path_ptr, dtype=np.int64)
    cat = np.asarray(cat, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    counts: dict[tuple[int, int], int] = {}
    for m in range(int(n_mol)):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        if a1 <= a0:
            continue
        p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
        keys = {(int(cat[i]), int(end[i])) for i in range(p0, p1)}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    return counts


def build_support_v2(dataset: str) -> dict[str, Any]:
    """Corrected support table: **distinct Train molecules**, not distinct roots.

    ``build_support`` walks ``for a in range(n_atoms)`` and accumulates per root
    atom, so its ``c`` is a *root* support (one molecule with the same key on many
    roots is counted many times).  The round's stated definition is the number of
    distinct Train molecules that contain the key; this function implements it and
    writes a separate file so the legacy cache stays byte-identical.
    """
    train = load_cache(dataset, "train")
    atom_ptr = np.asarray(train["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(train["path_ptr"], dtype=np.int64)
    cat = np.asarray(train["path_cat"], dtype=np.int64)
    end = np.asarray(train["path_end63"], dtype=np.int64)
    n_mol = int(train["n_molecules"])
    n_atoms = int(atom_ptr[-1])
    counts = _mol_key_counts(atom_ptr, path_ptr, cat, end, n_mol)
    keys = np.asarray(sorted(counts), dtype=np.int64).reshape(-1, 2)
    vals = np.asarray([counts[(int(k[0]), int(k[1]))] for k in keys], dtype=np.float32)
    n_codes = int(C.N_PATH_CODES) * int(C.N_NBR_CODES)
    flat_keys = keys[:, 0] * n_codes + keys[:, 1]

    # ---- sanity assertions that encode the definition ---------------------
    if vals.size:
        assert float(vals.min()) >= 1.0, "a table key must be present in >=1 molecule"
        assert float(vals.max()) <= n_mol, f"c_mol {vals.max()} > n_Train {n_mol}"
    legacy = support_path(dataset)
    root_max = None
    if legacy.exists():
        old = np.load(legacy, allow_pickle=False)
        old_keys = np.asarray(old["keys"]).reshape(-1, 2)
        old_vals = np.asarray(old["counts"]).astype(np.float64)
        root_max = float(old_vals.max())
        old_flat = old_keys[:, 0] * n_codes + old_keys[:, 1]
        pos = np.clip(np.searchsorted(old_flat, flat_keys), 0, max(len(old_vals) - 1, 0))
        ok = old_flat[pos] == flat_keys
        if bool(ok.all()):
            assert bool((vals.astype(np.float64) <= old_vals[pos] + 1e-6).all()), \
                "molecule support must never exceed root support"

    occ = np.asarray(sorted(counts.values()), dtype=np.float64)
    kappa = float(max(1.0, np.median(occ))) if occ.size else 1.0

    def _coeff(c: np.ndarray) -> np.ndarray:
        return c / (c + kappa)

    def _counts_for(cache) -> np.ndarray:
        pc = np.asarray(cache["path_cat"], dtype=np.int64)
        pe = np.asarray(cache["path_end63"], dtype=np.int64)
        flat = pc * n_codes + pe
        out = np.zeros(len(flat), dtype=np.float32)
        if len(flat) == 0 or not len(vals):
            return out
        pos = np.clip(np.searchsorted(flat_keys, flat), 0, max(len(vals) - 1, 0))
        ok = flat_keys[pos] == flat
        out[ok] = vals[pos[ok]]
        return out

    def _mean_root_coeff(mult: float) -> float:
        """Mean over all Train root atoms of the mean coefficient of its paths."""
        k_eff = kappa * float(mult)
        coeff = vals / (vals + k_eff)
        total = 0.0
        for a in range(n_atoms):
            p0, p1 = int(path_ptr[a]), int(path_ptr[a + 1])
            if p1 <= p0:
                total += 1.0          # atom with no path contributes 1 (protocol)
                continue
            flat = cat[p0:p1] * n_codes + end[p0:p1]
            pos = np.clip(np.searchsorted(flat_keys, flat), 0, max(len(coeff) - 1, 0))
            ok = flat_keys[pos] == flat
            cur = np.ones(len(flat))
            cur[ok] = coeff[pos[ok]]
            total += float(np.mean(cur))
        return float(total / max(n_atoms, 1))

    a_bar_k1 = _mean_root_coeff(1.0)
    a_bar_k4 = _mean_root_coeff(4.0)
    per_split = {s: _counts_for(load_cache(dataset, s)) for s in C.SPLITS}
    alpha: dict[str, dict[str, float]] = {}
    for s in C.SPLITS:
        c = per_split[s].astype(np.float64)
        a = _coeff(c)
        alpha[s] = {
            "n_tokens": int(c.size),
            "frac_c_le_1": float(np.mean(c <= 1)) if c.size else 0.0,
            "alpha_mean": float(np.mean(a)) if c.size else 1.0,
            "alpha_p05": float(np.percentile(a, 5)) if c.size else 1.0,
            "alpha_p25": float(np.percentile(a, 25)) if c.size else 1.0,
            "alpha_median": float(np.median(a)) if c.size else 1.0,
            "alpha_p75": float(np.percentile(a, 75)) if c.size else 1.0,
            "alpha_p95": float(np.percentile(a, 95)) if c.size else 1.0,
        }
    np.savez_compressed(
        support_v2_path(dataset),
        keys=keys, counts=vals, kappa=np.float64(kappa), a_bar=np.float64(a_bar_k1),
        a_bar_k1=np.float64(a_bar_k1), a_bar_k4=np.float64(a_bar_k4),
        **{f"path_count_{s}": per_split[s] for s in C.SPLITS},
    )
    return {
        "dataset": dataset,
        "definition": "c(key) = # distinct Train molecules containing key >= once",
        "n_occupied_keys": int(len(vals)),
        "n_train_molecules": int(n_mol),
        "n_train_root_atoms": int(n_atoms),
        "c_min": float(vals.min()) if vals.size else None,
        "c_median": float(np.median(vals)) if vals.size else None,
        "c_max": float(vals.max()) if vals.size else None,
        "legacy_root_support_max": root_max,
        "kappa": kappa,
        "a_bar": a_bar_k1,
        "a_bar_k1": a_bar_k1,
        "a_bar_k4": a_bar_k4,
        "alpha_by_split": alpha,
        "label_free": True,
        "fitted_on": "Train split only",
    }


def load_support_v2(dataset: str) -> dict[str, Any]:
    path = support_v2_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing v2 support table {path}; run the correct stage first")
    z = np.load(path, allow_pickle=False)
    return {
        "kappa": float(z["kappa"]),
        "a_bar": float(z["a_bar"]),
        "a_bar_k1": float(z["a_bar_k1"]),
        "a_bar_k4": float(z["a_bar_k4"]),
        "path_count": {s: np.asarray(z[f"path_count_{s}"]) for s in C.SPLITS},
        "keys": np.asarray(z["keys"]),
        "counts": np.asarray(z["counts"]),
    }


# ---------------------------------------------------------------------------
# auxiliary targets (Train only, plus reporting-only values elsewhere)
# ---------------------------------------------------------------------------


def aux_target_path(dataset: str) -> Path:
    return C.CACHE / f"aux_{dataset}.npz"


def build_aux_targets(dataset: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    src = load_source(dataset, labels="none")
    smiles = [str(x) for x in src["smiles"]]
    wc = np.full(len(smiles), np.nan, dtype=np.float64)
    mw = np.full(len(smiles), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"{dataset}: RDKit cannot parse row {i}")
        Chem.SanitizeMol(mol)
        wc[i] = float(rdMolDescriptors.CalcCrippenDescriptors(mol, includeHs=True, force=True)[0])
        mw[i] = float(Descriptors.MolWt(mol))
    train_rows = np.asarray(src["rows"]["train"], dtype=np.int64)
    rng = np.random.default_rng(int(C.AUX_SHUFFLE_SEED))
    shuffled = wc[train_rows].copy()
    rng.shuffle(shuffled)
    wc_shuf = np.full(len(smiles), np.nan, dtype=np.float64)
    wc_shuf[train_rows] = shuffled

    def _norm(x: np.ndarray) -> dict[str, float]:
        v = np.asarray(x, dtype=np.float64)
        return {"mean": float(np.mean(v)), "scale": float(np.std(v, ddof=0)), "n": int(v.size)}

    np.savez_compressed(
        aux_target_path(dataset),
        wc_logp=wc, molwt=mw, wc_logp_shuffled=wc_shuf,
        train_rows=train_rows, shuffle_seed=np.int64(C.AUX_SHUFFLE_SEED),
    )
    return {
        "dataset": dataset,
        "train_rows_sha256": C.row_sha256(train_rows),
        "wc_logp_norm_train": _norm(wc[train_rows]),
        "molwt_norm_train": _norm(mw[train_rows]),
        "wc_logp_shuffled_matches_multiset": bool(
            np.allclose(np.sort(shuffled), np.sort(wc[train_rows])) and len(np.unique(shuffled)) > 1
        ),
        "shuffle_scope": "Train rows only, one fixed private permutation (seed 90217)",
        "main_task_labels_shuffled": False,
    }


def load_aux_targets(dataset: str) -> dict[str, Any]:
    path = aux_target_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing aux target table {path}; run prepare first")
    z = np.load(path, allow_pickle=False)
    out = {
        "wc_logp": np.asarray(z["wc_logp"]),
        "molwt": np.asarray(z["molwt"]),
        "wc_logp_shuffled": np.asarray(z["wc_logp_shuffled"]),
        "train_rows": np.asarray(z["train_rows"]),
    }
    for key, name in (("wc_logp", "wc_logp"), ("molwt", "molwt"), ("wc_logp_shuffled", "wc_logp")):
        v = out[key][out["train_rows"]]
        out[f"{name}_norm"] = {
            "mean": float(np.mean(v)), "scale": float(np.std(v, ddof=0)), "n": int(v.size),
        }
    return out


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def collate(cache: Mapping[str, Any], rows: Sequence[int], *, path_count: np.ndarray | None = None):
    batch = PREP_DATA.collate(cache, list(rows))
    if path_count is not None:
        idx = _path_selection(cache, rows)
        batch["path_key_count"] = torch.from_numpy(np.asarray(path_count)[idx].astype(np.float32))
    return batch


def _path_selection(cache: Mapping[str, Any], rows: Sequence[int]) -> np.ndarray:
    """Exactly the path index selection ``lcrc_data.collate`` performs."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    rows = np.asarray(list(rows), dtype=np.int64)
    atom_idx = np.concatenate([
        np.arange(int(atom_ptr[r]), int(atom_ptr[r + 1]), dtype=np.int64) for r in rows
    ]) if len(rows) else np.zeros(0, np.int64)
    counts = path_ptr[atom_idx + 1] - path_ptr[atom_idx] if len(atom_idx) else np.zeros(0, np.int64)
    if not len(atom_idx):
        return np.zeros(0, np.int64)
    return np.concatenate([
        np.arange(int(path_ptr[a]), int(path_ptr[a] + c), dtype=np.int64) for a, c in zip(atom_idx, counts)
    ])


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def perms_path(dataset: str) -> Path:
    return C.CACHE / f"perms_{dataset}.npz"


def build_perms(dataset: str, max_epochs: int) -> np.ndarray:
    n = int(dataset_meta(dataset)["sizes"][0])
    gen = torch.Generator().manual_seed(int(C.SHUFFLE_SEED))
    out = np.empty((int(max_epochs), n), dtype=np.int64)
    for e in range(int(max_epochs)):
        out[e] = torch.randperm(n, generator=gen).numpy().astype(np.int64)
    np.savez_compressed(perms_path(dataset), perms=out, seed=np.int64(C.SHUFFLE_SEED))
    return out


def load_perms(dataset: str, epochs: int) -> np.ndarray:
    path = perms_path(dataset)
    if not path.exists():
        return build_perms(dataset, max(epochs, C.LONG_EPOCHS))
    z = np.load(path, allow_pickle=False)
    perms = np.asarray(z["perms"])
    if int(z["seed"]) != int(C.SHUFFLE_SEED):
        raise ValueError(f"{path}: shuffle seed differs from the frozen offset")
    if perms.shape[0] < int(epochs):
        perms = build_perms(dataset, max(int(epochs), C.LONG_EPOCHS))
    n = int(dataset_meta(dataset)["sizes"][0])
    if perms.shape[1] != n:
        raise ValueError(f"{path}: permutation length differs from Train size")
    return perms[: int(epochs)]


__all__ = [n for n in dir() if not n.startswith("_")]
