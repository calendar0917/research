"""Shared loaders/checks for the ZINC generalization audit.

Only official train/valid artifacts are read.  The official test split is
never referenced or constructed anywhere in this audit.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TRACK = Path(__file__).resolve().parents[1]
SUFF = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[4]
EXP = SUFF
V1 = EXP / "zinc_arch_discovery_v1"
V2 = EXP / "zinc_recipe_trunk_v2"
ZIP = EXP / "zinc_induced_pattern_pilot_v1"
PERF = EXP / "zinc_perf_protocol_v1"
ANALYSIS = TRACK / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

# Pre-fixed model set.  Recipes/initialisation differ by family; these are not
# an architecture-controlled experiment and are never pooled as one treatment.
MODELS = OrderedDict(
    [
        (
            "P3B_s0",
            {
                "family": "P3",
                "seed": 0,
                "recipe": "B (AdamW, 600 ep, 500-step warmup)",
                "ssl": "train-only masked SSL s0 (006fabc1...)",
                "selection": "best-valid single checkpoint",
                "params": 413505,
                "run": V2 / "runs" / "xBssl_P3N_s0",
            },
        ),
        (
            "J_s0",
            {
                "family": "J (full typed rooted induced paths)",
                "seed": 0,
                "recipe": "B (AdamW, 600 ep, 500-step warmup)",
                "ssl": "train-only masked SSL s0 (006fabc1...)",
                "selection": "best-valid single checkpoint",
                "params": 484281,
                "run": ZIP / "runs" / "J_s0",
            },
        ),
        (
            "Q_s0",
            {
                "family": "Q (typed path sequence control)",
                "seed": 0,
                "recipe": "B (AdamW, 600 ep, 500-step warmup)",
                "ssl": "train-only masked SSL s0 (006fabc1...)",
                "selection": "best-valid single checkpoint",
                "params": 484281,
                "run": ZIP / "runs" / "Q_s0",
            },
        ),
        (
            "P3A_s0",
            {
                "family": "P3",
                "seed": 0,
                "recipe": "A (Adam, 360 ep, no warmup)",
                "ssl": "train-only masked SSL s0 (seed-matched)",
                "selection": "best-valid single checkpoint",
                "params": 413505,
                "run": PERF / "runs" / "p3_s0",
            },
        ),
        (
            "P3A_s1",
            {
                "family": "P3",
                "seed": 1,
                "recipe": "A (Adam, 360 ep, no warmup)",
                "ssl": "train-only masked SSL s1 (seed-matched)",
                "selection": "best-valid single checkpoint",
                "params": 413505,
                "run": PERF / "runs" / "p3_s1",
            },
        ),
        (
            "M_s0",
            {
                "family": "M (global moment context + conditional composition)",
                "seed": 0,
                "recipe": "C (Adam, 360 ep, 500-step warmup)",
                "ssl": "train-only masked SSL s0 (006fabc1...)",
                "selection": "best-valid single checkpoint",
                "params": 435681,
                "run": V2 / "runs" / "m_M_s0",
            },
        ),
        (
            "M_s1",
            {
                "family": "M (global moment context + conditional composition)",
                "seed": 1,
                "recipe": "C (Adam, 360 ep, 500-step warmup)",
                "ssl": "train-only masked SSL s0 (006fabc1...)",
                "selection": "best-valid single checkpoint",
                "params": 435681,
                "run": V2 / "runs" / "m_M_s1",
            },
        ),
    ]
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_cache(split: str) -> dict:
    assert split in ("train", "valid"), "audit is train/valid only"
    return torch.load(V1 / "cache" / f"ad_{split}.pt", map_location="cpu", weights_only=False)


def load_path_cache(split: str) -> dict:
    assert split in ("train", "valid"), "audit is train/valid only"
    return torch.load(ZIP / "cache" / f"path_{split}.pt", map_location="cpu", weights_only=False)


def load_predictions(name: str) -> dict:
    meta = MODELS[name]
    run = meta["run"]
    npz = np.load(run / "predictions" / "valid_best.npz", allow_pickle=False)
    pred = np.asarray(npz["pred"], dtype=np.float64).reshape(-1)
    row = np.asarray(npz["row"], dtype=np.int64).reshape(-1)
    info = json.loads((run / "train.json").read_text())
    return {"pred": pred, "row": row, "info": info, "run": run, "meta": meta}


def load_all_predictions() -> tuple[dict, np.ndarray, dict]:
    valid = load_cache("valid")
    y = np.asarray(valid["labels"], dtype=np.float64)
    out = {}
    for name in MODELS:
        out[name] = load_predictions(name)
        r = out[name]
        if not np.array_equal(r["row"], np.arange(len(y), dtype=np.int64)):
            raise RuntimeError(f"{name}: valid rows are not 0..{len(y)-1}")
        if not np.isfinite(r["pred"]).all():
            raise RuntimeError(f"{name}: non-finite predictions")
        mae = float(np.abs(r["pred"] - y).mean())
        stored = float(r["info"]["fixed_objects"]["best"]["valid_mae"])
        if abs(mae - stored) > 1e-5:
            raise RuntimeError(f"{name}: recomputed MAE {mae:.12f} != stored {stored:.12f}")
        r["mae_recomputed"] = mae
    return out, y, valid


def atom_cats(cache: dict, i: int) -> np.ndarray:
    ap = np.asarray(cache["atom_ptr"])
    return np.asarray(cache["atom_type"])[ap[i] : ap[i + 1]].astype(np.int64)


def bond_arrays(cache: dict, i: int):
    bp = np.asarray(cache["bond_ptr"])
    bi = np.asarray(cache["bond_index"])[bp[i] : bp[i + 1]].astype(np.int64)
    bt = np.asarray(cache["bond_type"])[bp[i] : bp[i + 1]].astype(np.int64)
    return bi, bt


def graph_sha256(cache: dict, i: int) -> str:
    """Order-sensitive content fingerprint of the exact cached graph row.

    It is *not* an isomorphism-invariant graph id and is never used by a model;
    it only makes the per-molecule table traceable to a cache row.
    """
    cats = atom_cats(cache, i)
    bi, bt = bond_arrays(cache, i)
    order = np.lexsort((bi[:, 1], bi[:, 0], bt))
    payload = (
        int(len(cats)).to_bytes(8, "little")
        + cats.astype(np.uint8).tobytes()
        + int(len(bi)).to_bytes(8, "little")
        + bi[order].astype(np.int32).tobytes()
        + bt[order].astype(np.uint8).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def graph_stats(cache: dict, i: int) -> dict:
    """Raw-graph descriptors only; no labels and no learned embedding."""
    import networkx as nx

    cats = atom_cats(cache, i)
    bi, bt = bond_arrays(cache, i)
    n_atoms = int(len(cats))
    g = nx.Graph()
    g.add_nodes_from(range(n_atoms))
    for (u, v) in bi:
        g.add_edge(int(u), int(v))
    n_comp = int(nx.number_connected_components(g))
    cycle_rank = int(len(bi) - n_atoms + n_comp)
    rings = nx.cycle_basis(g)
    ring_sizes = sorted(len(r) for r in rings)
    max_ring = int(max(ring_sizes)) if ring_sizes else 0
    deg = np.zeros(n_atoms, dtype=np.int64)
    for (u, v) in bi:
        deg[int(u)] += 1
        deg[int(v)] += 1
    atom_counts = np.bincount(cats, minlength=21).astype(int)
    bond_counts = np.bincount(bt, minlength=4).astype(int)
    return {
        "n_atoms": n_atoms,
        "n_bonds": int(len(bi)),
        "n_components": n_comp,
        "cycle_rank": cycle_rank,
        "n_rings": int(len(rings)),
        "max_ring_size": max_ring,
        "ring_sizes": ring_sizes,
        "degree_max": int(deg.max()) if n_atoms else 0,
        "atom_counts_21": atom_counts.tolist(),
        "bond_counts_1_3": bond_counts[1:4].tolist(),
    }


def p3_object_from_cache(cache: dict, i: int):
    """Minimal object exposing the fields used by ip_data.enumerate_rooted_paths."""
    cats = atom_cats(cache, i)
    bi, bt = bond_arrays(cache, i)
    feat = np.zeros((len(cats), 21), dtype=np.float32)
    feat[np.arange(len(cats)), cats] = 1.0
    return SimpleNamespace(
        n_atoms=int(len(cats)),
        atom_feat=feat,
        bond_index=bi,
        bond_type=bt,
    )


def cached_path_entries(path_cache: dict, i: int) -> dict:
    """Reconstruct cached {length: {key: {root: multiplicity}}} for one molecule."""
    ptr = np.asarray(path_cache["ent_ptr"])
    ent_len = np.asarray(path_cache["ent_len"])
    ent_pat = np.asarray(path_cache["ent_pat_j"])
    ent_root = np.asarray(path_cache["ent_root"])
    ent_count = np.asarray(path_cache["ent_count"])
    pat_atom = np.asarray(path_cache["pat_atom_j"])
    pat_pair = np.asarray(path_cache["pat_pair_j"])
    out = {int(l): {} for l in path_cache["lengths"]}
    for e in range(int(ptr[i]), int(ptr[i + 1])):
        length = int(path_cache["lengths"][int(ent_len[e])])
        key = pat_atom[ent_pat[e]].astype(np.uint8).tobytes() + pat_pair[ent_pat[e]].astype(np.uint8).tobytes()
        out[length].setdefault(key, {})
        root = int(ent_root[e])
        out[length][key][root] = out[length][key].get(root, 0) + int(ent_count[e])
    return out
