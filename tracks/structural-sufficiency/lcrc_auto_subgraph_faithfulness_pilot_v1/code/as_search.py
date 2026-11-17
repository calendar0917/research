"""Gradient sensitivity, connected-subgraph search and the two discovery scores.

Nothing here pre-selects a functional group, a SMARTS pattern, an atom name or a
task-specific motif.  The search space is *every connected subgraph of 2..6 real
physical bonds* of the molecule, reached by the fixed procedure of protocol §4.4:

``k=1``  rank every physical bond;
``k=2``  enumerate **all** pairs of bonds sharing an endpoint (no k=1 prefilters);
``k>=3`` expand every candidate kept at level ``k-1`` by one adjacent bond.

At most :data:`as_common.BEAM_WIDTH` candidates are kept per size level, with the
frozen tie-break ``(score desc, fewer bonds, sorted bond-id tuple)``.  The whole
evaluation log is persisted so the Top-3 can never be re-picked after seeing the
real masked effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import as_common as C
import as_data as D
import as_gate as G


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------


def gradient_pass(arm: torch.nn.Module, gb: D.GatedBatch, *, sigma: float, mu: float) -> dict[str, Any]:
    """One forward + one backward: ``g[e] = d pred_raw / d m[e]`` at all-ones.

    No training loss is formed and no parameter receives an update; ``M`` is the
    only leaf with ``requires_grad``.
    """
    M = gb.gate_ones(requires_grad=True)
    parts = G.gated_parts(arm, gb, M)
    grad = torch.autograd.grad(parts["pred_std"].sum(), M)[0]
    grad_np = grad.detach().cpu().numpy().astype(np.float64)
    pred_std = parts["pred_std"].detach().cpu().numpy().astype(np.float64)
    local_sum = parts["local_sum"].detach().cpu().numpy().astype(np.float64)
    rho = parts["rho"].detach().cpu().numpy().astype(np.float64)
    contrib = parts["contrib_atom"].detach().cpu().numpy().astype(np.float64)
    return {
        "grad_transformed": grad_np * float(sigma),      # d pred_raw / d m
        "grad_std": grad_np,                             # d pred_std / d m
        "pred_std": pred_std,
        "pred_raw": pred_std * float(sigma) + float(mu),
        "local_sum_std": local_sum,
        "rho_std": rho,
        "local_sum_raw": local_sum * float(sigma),
        "rho_raw": rho * float(sigma),
        "contrib_atom_std": contrib,
        "n_graphs": int(pred_std.size),
        "max_bonds": int(gb.max_bonds),
    }


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------


def score_discovery(bonds: Sequence[int], grad_raw: np.ndarray) -> float:
    """``A(S) = |sum_{e in S} g[e]| / sqrt(|S|)`` -- signed sum, then abs."""
    ids = [int(b) for b in bonds]
    if not ids:
        return 0.0
    total = float(sum(float(grad_raw[e]) for e in ids))
    return abs(total) / math.sqrt(len(ids))


def score_local_heuristic(bonds: Sequence[int], contrib_abs: np.ndarray, mol: G.SingleMolecule) -> float:
    """``B(S) = sum_{v in atoms(S)} |c_v| / sqrt(|S|)``.

    ``c_v = f(h_v)`` is the frozen model's per-atom local output.  It is used
    *only* as a cheap direct comparison heuristic; it is never called a causal
    per-atom contribution.
    """
    ids = [int(b) for b in bonds]
    if not ids:
        return 0.0
    atoms = mol.bond_atoms(ids)
    total = float(sum(float(contrib_abs[v]) for v in atoms))
    return total / math.sqrt(len(ids))


#: Canonical sort key: score desc, fewer bonds, then the sorted bond-id tuple.
def rank_key(score: float, bonds: Sequence[int]) -> tuple[float, int, tuple[int, ...]]:
    return (-float(score), len(bonds), tuple(sorted(int(b) for b in bonds)))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    bonds: tuple[int, ...]
    score: float

    def as_record(self) -> dict[str, Any]:
        return {"bonds": list(self.bonds), "score": float(self.score)}


@dataclass
class SearchResult:
    score_name: str
    levels: dict[int, list[Candidate]] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    n_unique_evaluated: int = 0

    def pool(self, min_k: int = C.SEARCH_MIN_K, max_k: int = C.SEARCH_MAX_K) -> list[Candidate]:
        out: list[Candidate] = []
        for k in range(int(min_k), int(max_k) + 1):
            out.extend(self.levels.get(k, []))
        out.sort(key=lambda c: rank_key(c.score, c.bonds))
        return out


def _keep_top(cands: list[Candidate], width: int) -> list[Candidate]:
    uniq: dict[tuple[int, ...], Candidate] = {}
    for c in cands:
        prev = uniq.get(c.bonds)
        if prev is None or c.score > prev.score:
            uniq[c.bonds] = c
    ordered = sorted(uniq.values(), key=lambda c: rank_key(c.score, c.bonds))
    return ordered[: int(width)]


def search_subgraphs(
    mol: G.SingleMolecule,
    score_fn: Callable[[Sequence[int]], float],
    *,
    score_name: str,
    min_k: int = C.SEARCH_MIN_K,
    max_k: int = C.SEARCH_MAX_K,
    beam_width: int = C.BEAM_WIDTH,
    keep_log: bool = True,
) -> SearchResult:
    """The frozen §4.4 search, driven by an arbitrary deterministic score."""
    res = SearchResult(score_name=score_name)
    evaluated: set[tuple[int, ...]] = set()

    def _eval(bonds: Sequence[int], level: int) -> Candidate:
        key = tuple(sorted(int(b) for b in bonds))
        cand = Candidate(bonds=key, score=float(score_fn(key)))
        if keep_log:
            res.log.append({"k": int(level), "bonds": list(key), "score": float(cand.score)})
        return cand

    # k = 1: every physical bond.
    level1 = [_eval((e,), 1) for e in range(mol.n_bonds)]
    evaluated.update(c.bonds for c in level1)
    level1.sort(key=lambda c: rank_key(c.score, c.bonds))
    res.levels[1] = level1                            # full ranking is kept
    beam = level1[: int(beam_width)]

    # k = 2: every pair sharing an endpoint, independently of the k=1 ranking.
    pairs = [tuple(sorted(p)) for p in mol.shared_endpoint_pairs()]
    pairs = sorted(set(pairs))
    level2 = [_eval(p, 2) for p in pairs]
    evaluated.update(c.bonds for c in level2)
    res.levels[2] = _keep_top(level2, beam_width)
    beam = res.levels[2]

    for k in range(3, int(max_k) + 1):
        expanded: list[Candidate] = []
        for parent in beam:
            for e in mol.adjacent_bonds(parent.bonds):
                child = tuple(sorted(parent.bonds + (int(e),)))
                if child in evaluated:
                    continue
                evaluated.add(child)
                expanded.append(_eval(child, k))
        res.levels[k] = _keep_top(expanded, beam_width)
        beam = res.levels[k]
        if not beam:
            break
    res.n_unique_evaluated = len(evaluated)
    return res


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def select_topk(pool: Sequence[Candidate], k: int = C.TOP_K_SUBGRAPHS,
                max_overlap: float = C.JACCARD_MAX_OVERLAP) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Greedy Top-k with the frozen Jaccard constraint; may return fewer than k."""
    chosen: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    for cand in pool:
        if len(chosen) >= int(k):
            break
        overlaps = [jaccard(cand.bonds, c.bonds) for c in chosen]
        worst = max(overlaps) if overlaps else 0.0
        if worst <= float(max_overlap) + 1e-12:
            chosen.append(cand)
        else:
            rejected.append({"bonds": list(cand.bonds), "max_jaccard": float(worst)})
    return chosen, rejected


# ---------------------------------------------------------------------------
# encoding of the full candidate log
# ---------------------------------------------------------------------------


def encode_search_log(entries: Sequence[tuple[int, SearchResult]]) -> dict[str, np.ndarray]:
    """Compact, replayable encoding of every evaluated candidate.

    ``entries`` is ``(cache_row, SearchResult)`` in a deterministic order.
    """
    arrays: dict[str, list[np.ndarray]] = {
        "log_k": [], "log_score": [], "log_bonds": [], "log_ptr": [],
        "log_molecule": [], "log_scorer": [],
    }
    scorer_names: dict[str, int] = {}
    ptr = 0
    for mol_index, res in entries:
        for rec in res.log:
            scorer = res.score_name
            if scorer not in scorer_names:
                scorer_names[scorer] = len(scorer_names)
            arrays["log_k"].append(np.asarray([rec["k"]], dtype=np.int8))
            arrays["log_score"].append(np.asarray([rec["score"]], dtype=np.float64))
            arrays["log_bonds"].append(np.asarray(rec["bonds"], dtype=np.int16))
            arrays["log_ptr"].append(np.asarray([len(rec["bonds"])], dtype=np.int8))
            arrays["log_molecule"].append(np.asarray([mol_index], dtype=np.int32))
            arrays["log_scorer"].append(np.asarray([scorer_names[scorer]], dtype=np.int8))
            ptr += 1
    out: dict[str, np.ndarray] = {}
    for key, parts in arrays.items():
        out[key] = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    out["scorer_names"] = np.asarray(sorted(scorer_names, key=lambda s: scorer_names[s]), dtype="<U32")
    return out


_LOG_ARRAYS = ("log_k", "log_score", "log_bonds", "log_ptr", "log_molecule", "log_scorer")


def merge_search_logs(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-molecule encodings (constant memory: one small block each)."""
    if not parts:
        return encode_search_log([])
    out: dict[str, np.ndarray] = {}
    for key in _LOG_ARRAYS:
        blocks = [np.asarray(part[key]) for part in parts if np.asarray(part[key]).size]
        out[key] = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)
    names: list[str] = []
    for part in parts:
        for name in np.asarray(part["scorer_names"]).tolist():
            if name not in names:
                names.append(name)
    out["scorer_names"] = np.asarray(names, dtype="<U32")
    return out


def decode_search_log(payload: Mapping[str, np.ndarray], molecule: int) -> list[dict[str, Any]]:
    """Replay one molecule's full candidate log from the frozen arrays."""
    mol = np.asarray(payload["log_molecule"], dtype=np.int64)
    ptr = np.asarray(payload["log_ptr"], dtype=np.int64)
    bonds = np.asarray(payload["log_bonds"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(ptr)[:-1]]).astype(np.int64)
    out: list[dict[str, Any]] = []
    for idx in np.where(mol == int(molecule))[0].tolist():
        s = int(starts[idx])
        out.append(
            {
                "k": int(np.asarray(payload["log_k"])[idx]),
                "score": float(np.asarray(payload["log_score"])[idx]),
                "bonds": bonds[s : s + int(ptr[idx])].tolist(),
            }
        )
    return out


__all__ = [
    "gradient_pass",
    "score_discovery",
    "score_local_heuristic",
    "rank_key",
    "Candidate",
    "SearchResult",
    "search_subgraphs",
    "jaccard",
    "select_topk",
    "encode_search_log",
    "decode_search_log",
]
