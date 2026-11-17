"""Automatic connected-subgraph discovery, controls and external localisation.

The discovery functions receive a molecule's structure and a gradient field.
They never receive a real label, the paired molecule, the edit region or an
oracle contribution.  Edit/oracle quantities are attached only afterwards by
``evaluate_*``.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import xr_common as C


# ---------------------------------------------------------------------------
# candidate discovery
# ---------------------------------------------------------------------------


def discover_from_gradient(mol, grad_raw: np.ndarray, *, score_name: str = "A_discovery") -> dict[str, Any]:
    """Frozen search driven only by ``g_e`` and the molecular graph."""
    mods = C.source_modules()
    AS = mods["as_search"]
    res = AS.search_subgraphs(mol, lambda bonds: AS.score_discovery(bonds, grad_raw), score_name=score_name)
    topk, rejected = AS.select_topk(res.pool())
    return {
        "score_name": score_name,
        "topk": [{"bonds": list(c.bonds), "score": float(c.score), "k": len(c.bonds)} for c in topk],
        "rejected_overlap": rejected,
        "n_unique_evaluated": int(res.n_unique_evaluated),
        "levels": {int(k): len(v) for k, v in sorted(res.levels.items())},
        "log": res.log,
        "score_pool_size": len(res.pool()),
    }


def discover_local_heuristic(mol, contrib_abs: np.ndarray) -> dict[str, Any]:
    mods = C.source_modules()
    AS = mods["as_search"]
    res = AS.search_subgraphs(
        mol,
        lambda bonds: AS.score_local_heuristic(bonds, contrib_abs, mol),
        score_name="B_local_output",
    )
    topk, rejected = AS.select_topk(res.pool())
    return {
        "score_name": "B_local_output",
        "topk": [{"bonds": list(c.bonds), "score": float(c.score), "k": len(c.bonds)} for c in topk],
        "rejected_overlap": rejected,
        "n_unique_evaluated": int(res.n_unique_evaluated),
        "levels": {int(k): len(v) for k, v in sorted(res.levels.items())},
        "log": res.log,
        "score_pool_size": len(res.pool()),
    }


# ---------------------------------------------------------------------------
# random controls
# ---------------------------------------------------------------------------


def same_size_random_controls(
    mol,
    top1_bonds: Sequence[int],
    *,
    seed: int,
    n_target: int = C.N_RANDOM,
    max_attempts: int = C.MAX_GROWTH_ATTEMPTS,
) -> dict[str, Any]:
    mods = C.source_modules()
    K = mods["as_controls"]
    top1 = tuple(sorted(int(b) for b in top1_bonds))
    k = len(top1)
    rng = np.random.default_rng(seed)
    seen = {top1}
    accepted: list[dict[str, Any]] = []
    attempts = 0
    while attempts < int(max_attempts) and len(accepted) < int(n_target):
        attempts += 1
        cand = K.grow_connected_subgraph(mol, k, rng)
        if cand is None:
            break
        if cand in seen:
            continue
        seen.add(cand)
        if not mol.atoms_connected(cand):
            continue
        accepted.append({"bonds": list(cand), "atoms": mol.bond_atoms(cand), "n_bonds": len(cand)})
    status = "OK" if len(accepted) >= int(C.MIN_RANDOM) else "INSUFFICIENT_RANDOM"
    return {
        "status": status,
        "seed": seed if isinstance(seed, (int, np.integer)) else list(seed),
        "n_target": int(n_target),
        "n_random": len(accepted),
        "attempts": int(attempts),
        "rule": "same molecule, same physical-bond count, connected, exclude Top-1, fixed seed",
        "controls": accepted,
    }


def s2_matched_random_controls(mol, top1_bonds: Sequence[int], *, seed: int) -> dict[str, Any]:
    mods = C.source_modules()
    K = mods["as_controls"]
    rng = np.random.default_rng(seed)
    return K.matched_random_controls(mol, top1_bonds, rng, n_target=C.N_RANDOM, max_attempts=C.MAX_GROWTH_ATTEMPTS, s2_rel_tol=C.S2_MATCH_REL_TOL)


# ---------------------------------------------------------------------------
# external localisation metrics (structure only)
# ---------------------------------------------------------------------------


def _metric_record(
    mol,
    bonds: Sequence[int],
    edit_atoms: set[int],
    edit_bonds: set[tuple[int, int]],
    attach_atom: int,
    attach_bonds: set[tuple[int, int]],
) -> dict[str, Any]:
    cand_ids = set(int(x) for x in bonds)
    cand_bonds = set(
        (min(int(mol.bond_u[b]), int(mol.bond_v[b])), max(int(mol.bond_u[b]), int(mol.bond_v[b])))
        for b in cand_ids
    )
    cand_atoms = set(mol.bond_atoms(sorted(cand_ids)))
    atom_overlap = len(cand_atoms & edit_atoms)
    bond_overlap = len(cand_bonds & edit_bonds)
    atom_precision = atom_overlap / len(cand_atoms) if cand_atoms else 0.0
    atom_recall = atom_overlap / len(edit_atoms) if edit_atoms else 0.0
    atom_jaccard = atom_overlap / len(cand_atoms | edit_atoms) if (cand_atoms | edit_atoms) else 0.0
    bond_precision = bond_overlap / len(cand_bonds) if cand_bonds else 0.0
    bond_recall = bond_overlap / len(edit_bonds) if edit_bonds else 0.0
    bond_jaccard = bond_overlap / len(cand_bonds | edit_bonds) if (cand_bonds | edit_bonds) else 0.0
    return {
        "n_bonds": len(cand_bonds),
        "n_atoms": len(cand_atoms),
        "atom_overlap": int(atom_overlap),
        "atom_precision": float(atom_precision),
        "atom_recall": float(atom_recall),
        "atom_jaccard": float(atom_jaccard),
        "bond_overlap": int(bond_overlap),
        "bond_precision": float(bond_precision),
        "bond_recall": float(bond_recall),
        "bond_jaccard": float(bond_jaccard),
        "hit_attachment_atom": bool(attach_atom in cand_atoms),
        "hit_attachment_bond": bool(bool(cand_bonds & attach_bonds)),
        "hit_attachment_boundary": bool(attach_atom in cand_atoms or bool(cand_bonds & attach_bonds)),
        "candidate_atoms": sorted(cand_atoms),
        "candidate_bonds": sorted([list(x) for x in cand_bonds]),
    }


def _aggregate_controls(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "atom_precision",
        "atom_recall",
        "atom_jaccard",
        "bond_precision",
        "bond_recall",
        "bond_jaccard",
        "hit_attachment_atom",
        "hit_attachment_bond",
        "hit_attachment_boundary",
    )
    out: dict[str, Any] = {"n": len(records)}
    for key in keys:
        vals = np.asarray([float(r[key]) for r in records], dtype=np.float64)
        out[f"mean_{key}"] = float(vals.mean()) if vals.size else None
        out[f"median_{key}"] = float(np.median(vals)) if vals.size else None
    return out


def localisation_for_candidate(
    mol,
    bonds: Sequence[int],
    edit: Mapping[str, Any],
    random_controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    edit_atoms = set(int(v) for v in edit["edit_atoms"])
    edit_bonds = set(tuple(sorted(int(x) for x in b)) for b in edit["edit_bonds"])
    attach_atom = int(edit["attach_atom"])
    attach_bonds = set(tuple(sorted(int(x) for x in b)) for b in edit["attach_bonds"])
    top = _metric_record(mol, bonds, edit_atoms, edit_bonds, attach_atom, attach_bonds)
    ctrl = [_metric_record(mol, rec["bonds"], edit_atoms, edit_bonds, attach_atom, attach_bonds) for rec in random_controls]
    agg = _aggregate_controls(ctrl)
    adv = {}
    for key in ("atom_precision", "atom_recall", "atom_jaccard", "bond_precision", "bond_recall", "bond_jaccard", "hit_attachment_atom", "hit_attachment_bond", "hit_attachment_boundary"):
        base = agg.get(f"mean_{key}")
        adv[key] = float(top[key] - base) if base is not None else None
    return {
        "top1": top,
        "random": agg,
        "random_records": ctrl,
        "advantage": adv,
    }


# ---------------------------------------------------------------------------
# oracle mass capture
# ---------------------------------------------------------------------------


def mass_capture(delta: Sequence[float], atoms: Iterable[int], denom: float | None = None) -> float | None:
    absd = np.abs(np.asarray(delta, dtype=np.float64))
    if denom is None:
        denom = float(absd.sum())
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    atoms = [int(v) for v in atoms if 0 <= int(v) < absd.size]
    if not atoms:
        return 0.0
    return float(absd[atoms].sum() / denom)


def oracle_capture_for_candidate(
    delta: Sequence[float],
    candidate_atoms: Sequence[int],
    random_controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    absd = np.abs(np.asarray(delta, dtype=np.float64))
    denom = float(absd.sum())
    top = mass_capture(delta, candidate_atoms, denom)
    vals = [mass_capture(delta, rec["atoms"], denom) for rec in random_controls]
    vals = [v for v in vals if v is not None]
    median = float(np.median(vals)) if vals else None
    mean = float(np.mean(vals)) if vals else None
    return {
        "denominator": denom,
        "defined": bool(denom > 0.0),
        "top1_capture": top,
        "random_median_capture": median,
        "random_mean_capture": mean,
        "n_random": len(vals),
        "oracle_enrichment": float(top - median) if (top is not None and median is not None) else None,
    }


def oracle_concentration(pair_wc: Mapping[str, Any]) -> dict[str, Any]:
    core = float(pair_wc["core_abs_sum"])
    va = float(pair_wc["var_a_abs_sum"])
    vb = float(pair_wc["var_b_abs_sum"])
    total = core + va + vb
    return {
        "core_abs_mass": core,
        "var_a_abs_mass": va,
        "var_b_abs_mass": vb,
        "total_abs_mass": total,
        "core_fraction": core / total if total > 0 else None,
        "variable_fraction": (va + vb) / total if total > 0 else None,
        "var_a_fraction": va / total if total > 0 else None,
        "var_b_fraction": vb / total if total > 0 else None,
    }


# ---------------------------------------------------------------------------
# candidate log writer
# ---------------------------------------------------------------------------


class CandidateLog:
    """Streaming gzip JSONL candidate log (no full-round in-memory retention)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=6)
        self.n = 0

    def write(self, endpoint_key: str, model: str, result: Mapping[str, Any]) -> None:
        for rec in result["log"]:
            row = {
                "endpoint": endpoint_key,
                "model": model,
                "scorer": result["score_name"],
                "k": int(rec["k"]),
                "bonds": [int(x) for x in rec["bonds"]],
                "score": float(rec["score"]),
            }
            self._handle.write(json.dumps(row, sort_keys=True) + "\n")
            self.n += 1

    def close(self) -> dict[str, Any]:
        self._handle.close()
        return {"path": str(self.path), "n_candidates": int(self.n), "sha256": C.sha256_file(self.path) if self.path.is_file() else None}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


__all__ = [
    "discover_from_gradient",
    "discover_local_heuristic",
    "same_size_random_controls",
    "s2_matched_random_controls",
    "localisation_for_candidate",
    "mass_capture",
    "oracle_capture_for_candidate",
    "oracle_concentration",
    "CandidateLog",
]
