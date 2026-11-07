"""Audits for the LCRC-S3 round.

Modes (``--modes a,b,c``):

CPU / stage 0
  ``structure``    the label-free S3 structural audit on a fixed 4096-molecule
                   Train sample (distribution, path types, growth, novelty,
                   cache projection, batch cost projection, gates)

CPU / stage 0.5
  ``cache``        build the label-free Train+Valid S3 structural cache and
                   verify the frozen-collate row bookkeeping

GPU / pre-training
  ``gpu-preflight``  host / driver / torch / GPU occupancy facts
  ``init``           W2-preserving initialisation parity, epoch-0 logit parity,
                     first-step gradients, resume-restore audit
  ``architecture``   module call counts, no-message-passing audit, atom
                     renumbering, empty-S3 safety, bond-edit sensitivity,
                     batch isolation, row-id consistency
  ``cost``           step-time / Validation-time probe for W2, S3 and F, the
                     projected epoch ratio gate, peak memory, analytical FLOPs
  ``profile``        torch profiler on one S3 training step (no per-path Python
                     loop in the forward)
  ``w2-replay``      read-only replay of the historical W2 best/epoch-40
                     checkpoints (AP, BCE, per-task AP, SHA-256)

No mode reads an official Test target.  ``structure``/``cache`` never touch the
label array at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import s3_common as C  # noqa: E402
import s3_paths as S3P  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=np.float64), float(q))) if np.asarray(x).size else 0.0


def _summary(x: Sequence[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": _pct(a, 90),
        "p99": _pct(a, 99),
        "max": float(a.max()),
        "min": float(a.min()),
        "std": float(a.std()),
    }


def _gini(x: np.ndarray) -> float:
    a = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    if a.size == 0 or a.sum() <= 0:
        return 0.0
    n = a.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * a).sum()) / (n * a.sum()) - (n + 1.0) / n)


# ---------------------------------------------------------------------------
# stage 0: deterministic size-stratified Train sample
# ---------------------------------------------------------------------------


def size_strata(
    atom_counts: np.ndarray, rows: np.ndarray, *, total: int = C.SAMPLE_SIZE
) -> list[dict[str, Any]]:
    """Pre-fixed size strata -> deterministic allocation with a per-stratum floor."""
    counts = np.asarray(atom_counts, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    rows = rows[order]
    counts = counts[order]
    strata: list[dict[str, Any]] = []
    for name, lo, hi, floor in C.SAMPLE_STRATA:
        mask = (counts >= int(lo)) & (counts <= int(hi))
        strata.append(
            {
                "name": name,
                "atom_min": int(lo),
                "atom_max": int(hi),
                "pop": int(mask.sum()),
                "floor": int(floor),
                "rows": rows[mask],
            }
        )
    assigned = sum(int(s["floor"]) for s in strata)
    if assigned > total:
        # only reachable in the ``--quick`` smoke path: scale the floors down
        scale = float(total) / float(assigned)
        for s in strata:
            s["floor"] = max(1, int(round(int(s["floor"]) * scale)))
        assigned = sum(int(s["floor"]) for s in strata)
        while assigned > total:
            s = max(strata, key=lambda x: int(x["floor"]))
            s["floor"] = int(s["floor"]) - 1
            assigned -= 1
        if assigned > total:
            raise RuntimeError(f"cannot scale stratum floors to the sample size {total}")
    remaining = int(total) - assigned
    pops = np.asarray([int(s["pop"]) for s in strata], dtype=np.float64)
    if pops.sum() <= 0:
        raise RuntimeError("empty train population")
    demand = pops / pops.sum() * remaining
    base = np.floor(demand).astype(np.int64)
    leftover = int(remaining - base.sum())
    # largest remainder, ties broken by stratum order (deterministic)
    order = np.argsort(-(demand - base), kind="stable")
    for k in order[:leftover]:
        base[int(k)] += 1
    for s, extra in zip(strata, base.tolist()):
        s["allocated"] = int(s["floor"] + int(extra))
        s["demand"] = float(s["floor"] + 0.0)
    return strata


def select_sample_rows(task: Any, *, total: int = C.SAMPLE_SIZE) -> dict[str, Any]:
    """The frozen 4096-molecule Train sample (label-free, deterministic)."""
    cache = task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    train_rows = np.sort(np.asarray(task.train_rows, dtype=np.int64))
    atom_counts = (atom_ptr[train_rows + 1] - atom_ptr[train_rows]).astype(np.int64)
    strata = size_strata(atom_counts, train_rows, total=total)
    picked: list[np.ndarray] = []
    table: list[dict[str, Any]] = []
    for s in strata:
        rows = np.asarray(s["rows"], dtype=np.int64)
        n, k = int(rows.size), int(s["allocated"])
        if k > n:
            raise RuntimeError(f"stratum {s['name']} alloc {k} > population {n}")
        idx = np.floor((np.arange(k, dtype=np.float64) + 0.5) * (n / k)).astype(np.int64)
        idx = np.clip(idx, 0, n - 1)
        if np.unique(idx).size != k:
            raise RuntimeError(f"stratum {s['name']}: non-unique deterministic sample")
        chosen = rows[idx]
        picked.append(chosen)
        table.append(
            {
                "name": s["name"],
                "atom_range": [int(s["atom_min"]), int(s["atom_max"]) if s["atom_max"] < 10**9 else None],
                "population": int(s["pop"]),
                "allocated": int(k),
                "share_of_population": float(k / max(n, 1)),
                "rows_sha256": C.sha256_bytes(np.ascontiguousarray(chosen, dtype=np.int64).tobytes()),
                "atom_mean": float(atom_counts[np.isin(train_rows, chosen)].mean()) if k else 0.0,
            }
        )
    sample_rows = np.sort(np.concatenate(picked))
    if sample_rows.size != total:
        raise RuntimeError(f"sample size {sample_rows.size} != {total}")
    if np.intersect1d(sample_rows, np.asarray(task.test_rows, np.int64)).size:
        raise RuntimeError("sample contains official Test rows")
    if np.intersect1d(sample_rows, np.asarray(task.valid_rows, np.int64)).size:
        raise RuntimeError("sample is not restricted to the official Train split")
    return {
        "rows": sample_rows,
        "strata": table,
        "recipe": C.SAMPLE_SELECTION,
        "n_train_population": int(train_rows.size),
        "sample_sha256": C.sha256_bytes(np.ascontiguousarray(sample_rows, dtype=np.int64).tobytes()),
    }


# ---------------------------------------------------------------------------
# stage 0: per-molecule S3 measurement
# ---------------------------------------------------------------------------


def molecule_from_task(task: Any, row: int) -> dict[str, Any]:
    cache = task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    # ``bond_index`` in the frozen cache stores *molecule-local* atom ids.
    bi = np.asarray(cache["bond_index"])[b0:b1].astype(np.int64)
    bt = np.asarray(cache["bond_type"])[b0:b1].astype(np.int64)
    at = np.asarray(cache["atom_type"])[a0:a1].astype(np.int64)
    adj = S3P.molecule_adjacency(a1 - a0, bi[:, 0], bi[:, 1], bt)
    return {"row": int(row), "n_atoms": a1 - a0, "n_bonds": b1 - b0, "adj": adj, "atom_type": at, "bonds_uv": bi, "bond_type": bt}


def measure_sample(task: Any, rows: Sequence[int]) -> dict[str, Any]:
    """Enumerate S3 for the sample and collect the audit statistics."""
    per_mol: list[dict[str, Any]] = []
    per_root_counts: list[np.ndarray] = []
    per_root_molindex: list[np.ndarray] = []
    type_counter: dict[int, int] = {}
    dist_counter = np.zeros(3, dtype=np.int64)
    n_pairs_covered_by_s2 = 0
    n_pairs_total = 0
    max_paths = 0
    max_paths_row = -1
    t0 = time.perf_counter()
    for i, row in enumerate(np.asarray(rows, dtype=np.int64).tolist()):
        mol = molecule_from_task(task, row)
        n = int(mol["n_atoms"])
        dist = S3P.all_pairs_shortest_paths(n, mol["adj"])
        paths = S3P.enumerate_s3(n, mol["adj"], mol["atom_type"], dist, keep_mid_atoms=True)
        counts = paths.per_root_counts()
        closed = S3P.count_s3_closed_form(n, mol["adj"])
        if not np.array_equal(counts, closed):
            raise RuntimeError(f"row {row}: closed-form S3 count != enumeration")
        per_root_counts.append(counts)
        per_root_molindex.append(np.full(n, i, dtype=np.int64))
        if paths.n_paths:
            dist_counter += np.bincount(paths.dist.astype(np.int64) - 1, minlength=3)
            keys = S3P.path_type_keys(paths)
            uniq, cnt = np.unique(keys, return_counts=True)
            for k, c in zip(uniq.tolist(), cnt.tolist()):
                type_counter[int(k)] = type_counter.get(int(k), 0) + int(c)
            # (root, end) pairs already reachable by an exactly-2-bond S2 path
            pairs = np.stack([np.repeat(np.arange(n, dtype=np.int64), counts), paths.end], axis=1)
            uniq_pairs = np.unique(pairs, axis=0)
            n_pairs_total += int(uniq_pairs.shape[0])
            n_pairs_covered_by_s2 += int(
                sum(
                    1
                    for r, x in uniq_pairs.tolist()
                    if 1 <= int(dist[r, x]) <= 2
                )
            )
        n_paths = int(paths.n_paths)
        if n_paths > max_paths:
            max_paths, max_paths_row = n_paths, int(row)
        per_mol.append(
            {
                "row": int(row),
                "n_atoms": n,
                "n_bonds": int(mol["n_bonds"]),
                "n_paths": n_paths,
                "per_root_mean": float(counts.mean()) if n else 0.0,
                "per_root_max": int(counts.max()) if n else 0,
                "paths_with_dist3": int(np.sum(paths.dist == 3)) if n_paths else 0,
                "paths_with_dist2": int(np.sum(paths.dist == 2)) if n_paths else 0,
                "paths_with_dist1": int(np.sum(paths.dist == 1)) if n_paths else 0,
            }
        )
    elapsed = float(time.perf_counter() - t0)
    roots = np.concatenate(per_root_counts) if per_root_counts else np.zeros(0, np.int64)
    mol_paths = np.asarray([m["n_paths"] for m in per_mol], dtype=np.int64)
    return {
        "per_molecule": per_mol,
        "per_root_counts": roots,
        "path_type_counter": type_counter,
        "dist_counts": [int(x) for x in dist_counter.tolist()],
        "pairs": {"total": int(n_pairs_total), "covered_by_s2": int(n_pairs_covered_by_s2)},
        "max_paths_row": int(max_paths_row),
        "elapsed_seconds": elapsed,
        "n_atoms_sample": int(sum(m["n_atoms"] for m in per_mol)),
        "n_paths_sample": int(mol_paths.sum()),
    }


def growth_by_size(per_molecule: Sequence[Mapping[str, Any]], *, n_bins: int = 7) -> dict[str, Any]:
    atoms = np.asarray([m["n_atoms"] for m in per_molecule], dtype=np.int64)
    paths = np.asarray([m["n_paths"] for m in per_molecule], dtype=np.int64)
    d3 = np.asarray([m["paths_with_dist3"] for m in per_molecule], dtype=np.int64)
    edges = np.unique(np.percentile(atoms, np.linspace(0, 100, n_bins + 1)).astype(np.int64))
    out: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        mask = (atoms >= lo) & (atoms <= hi)
        if not mask.any():
            continue
        out.append(
            {
                "atom_low": int(lo),
                "atom_high": int(hi),
                "n_molecules": int(mask.sum()),
                "paths_mean": float(paths[mask].mean()),
                "paths_per_atom_mean": float((paths[mask] / atoms[mask]).mean()),
                "mol_fraction_with_dist3": float((d3[mask] > 0).mean()),
            }
        )
    slope = float(np.polyfit(atoms.astype(np.float64), paths.astype(np.float64), 1)[0]) if atoms.size > 1 else 0.0
    return {"bins": out, "paths_per_atom_ols_slope": slope}


def synthetic_cases() -> dict[str, Any]:
    """Exact S3 behaviour on hand-built graphs (4-ring, 5-ring, 6-ring, chain)."""
    def build(n: int, bonds: Sequence[tuple[int, int]], bond_type: int = 1, atom_type: int = 2):
        u = np.asarray([b[0] for b in bonds], dtype=np.int64)
        v = np.asarray([b[1] for b in bonds], dtype=np.int64)
        bt = np.full(u.size, int(bond_type), dtype=np.int64)
        at = np.full(n, int(atom_type), dtype=np.int64)
        adj = S3P.molecule_adjacency(n, u, v, bt)
        return adj, at

    cases: dict[str, Any] = {}
    ring4 = [(0, 1), (1, 2), (2, 3), (0, 3)]
    ring5 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    ring6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    chain4 = [(0, 1), (1, 2), (2, 3)]
    chain6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for name, (n, bonds) in {
        "cyclobutane_C4": (4, ring4),
        "cyclopentane_C5": (5, ring5),
        "cyclohexane_C6": (6, ring6),
        "butane_chain_C4": (4, chain4),
        "hexane_chain_C6": (6, chain6),
    }.items():
        adj, at = build(n, bonds)
        dist = S3P.all_pairs_shortest_paths(n, adj)
        paths = S3P.enumerate_s3(n, adj, at, dist)
        cases[name] = {
            "n_atoms": n,
            "n_bonds": len(bonds),
            "n_paths": int(paths.n_paths),
            "per_root_counts": [int(x) for x in paths.per_root_counts().tolist()],
            "dist_histogram": {str(d): int(np.sum(paths.dist == d)) for d in S3_DIST},
            "closed_form_counts": [int(x) for x in S3P.count_s3_closed_form(n, adj).tolist()],
        }
    return cases


S3_DIST = (1, 2, 3)


# ---------------------------------------------------------------------------
# stage 0: rooted 2-ball vs 3-ball novelty
# ---------------------------------------------------------------------------


def _wl_for_atoms(
    task: Any, rows: Sequence[int], *, max_atoms: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    meta = {"n_molecules": 0, "n_atoms": 0, "elapsed_seconds": 0.0}
    t0 = time.perf_counter()
    for row in np.asarray(rows, dtype=np.int64).tolist():
        mol = molecule_from_task(task, row)
        n = int(mol["n_atoms"])
        dist = S3P.all_pairs_shortest_paths(n, mol["adj"])
        for root in range(n):
            b2 = S3P.rooted_ball(n, mol["adj"], mol["atom_type"], root, 2, dist)
            b3 = S3P.rooted_ball(n, mol["adj"], mol["atom_type"], root, 3, dist)
            atoms.append(
                {
                    "row": int(row),
                    "root": int(root),
                    "n_atoms": n,
                    "ball2": b2,
                    "ball3": b3,
                    "sig2": S3P.wl_signature(b2),
                    "sig3": S3P.wl_signature(b3),
                }
            )
            if max_atoms is not None and len(atoms) >= int(max_atoms):
                meta["n_molecules"] += 1
                meta["n_atoms"] = len(atoms)
                meta["elapsed_seconds"] = float(time.perf_counter() - t0)
                return atoms, meta
        meta["n_molecules"] += 1
    meta["n_atoms"] = len(atoms)
    meta["elapsed_seconds"] = float(time.perf_counter() - t0)
    return atoms, meta


def exact_classes(balls: Sequence[S3P.RootedBall], keys: Sequence[str]) -> tuple[list[int], int]:
    """Exact isomorphism classes by VF2 inside hash-signature buckets.

    The hash signature is only a bucket key, so a hash collision can never
    produce a wrong class assignment: every member is compared with the
    representatives of its bucket using the complete ``ball_isomorphic`` test.
    Returns ``(class_id_per_member, n_comparisons)``.
    """
    buckets: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault(key, []).append(i)
    out = [-1] * len(keys)
    comparisons = 0
    next_id = 0
    for _key, members in buckets.items():
        reps: list[int] = []
        for i in sorted(members):
            for c, rep in enumerate(reps):
                comparisons += 1
                if S3P.ball_isomorphic(balls[i], balls[rep]):
                    out[i] = next_id + c
                    break
            else:
                reps.append(i)
                out[i] = next_id + len(reps) - 1
        next_id += len(reps)
    if any(x < 0 for x in out):
        raise RuntimeError("unassigned isomorphism class")
    return out, comparisons


def novelty_analysis(task: Any, rows: Sequence[int], *, smiles: Sequence[str] | None = None) -> dict[str, Any]:
    """Exact rooted-2-ball vs rooted-3-ball separability (label-free).

    The primary statement uses **exact** isomorphism classes (VF2 on the typed
    rooted induced subgraphs).  Two discarded alternatives are recorded because
    they are traps:

    * a *partition-only* WL signature forgets which abstract colour is which,
      so different label sets with the same class-shape collide;
    * a hash signature is collision-resistant but not a proof.

    Both are reported as screens, never as the basis of the claim.
    """
    atoms, meta = _wl_for_atoms(task, rows)
    n_atoms = len(atoms)
    t0 = time.perf_counter()
    for a in atoms:
        a["hash2"] = S3P.ball_signature(a["ball2"])
        a["hash3"] = S3P.ball_signature(a["ball3"])
    hash_seconds = float(time.perf_counter() - t0)
    t0 = time.perf_counter()
    exact2, cmp2 = exact_classes([a["ball2"] for a in atoms], [a["hash2"] for a in atoms])
    exact3, cmp3 = exact_classes([a["ball3"] for a in atoms], [a["hash3"] for a in atoms])
    exact_seconds = float(time.perf_counter() - t0)

    # A 3-ball contains its 2-ball, so a member's exact-3 class implies its
    # exact-2 class.  Grouping by exact-2 and counting distinct exact-3 gives
    # "the 2-ball does not determine the 3-ball" exactly.
    by_class2: dict[int, set[int]] = {}
    for c2, c3 in zip(exact2, exact3):
        by_class2.setdefault(int(c2), set()).add(int(c3))
    informative2 = {k for k, v in by_class2.items() if len(v) > 1}
    atoms_informative = sum(1 for c in exact2 if int(c) in informative2)
    mols = {int(atoms[i]["row"]) for i, c in enumerate(exact2) if int(c) in informative2}

    # the same bookkeeping on the two weaker screens, for the record
    def _screen(key2: str, key3: str) -> dict[str, Any]:
        groups: dict[str, set[str]] = {}
        members: dict[str, int] = {}
        for a in atoms:
            groups.setdefault(a[key2], set()).add(a[key3])
            members[a[key2]] = members.get(a[key2], 0) + 1
        inf = {k for k, v in groups.items() if len(v) > 1}
        return {
            "n_two_ball_classes": int(len(groups)),
            "n_classes_with_multiple_three_ball_classes": int(len(inf)),
            "atom_fraction_in_informative_classes": float(
                sum(members[k] for k in inf) / max(n_atoms, 1)
            ),
        }

    exact = {
        "n_two_ball_classes": int(len(by_class2)),
        "n_classes_with_multiple_three_ball_classes": int(len(informative2)),
        "atoms_in_informative_classes": int(atoms_informative),
        "atom_fraction_in_informative_classes": float(atoms_informative / max(n_atoms, 1)),
        "molecule_fraction_with_informative_atom": float(len(mols) / max(meta["n_molecules"], 1)),
        "max_three_ball_variants_per_two_ball_class": int(max((len(v) for v in by_class2.values()), default=0)),
        "mean_three_ball_variants_per_informative_class": float(
            np.mean([len(v) for k, v in by_class2.items() if len(v) > 1])
        )
        if informative2
        else 0.0,
        "variant_histogram": {
            str(int(k)): int(v)
            for k, v in zip(
                *np.unique(np.asarray([len(v) for v in by_class2.values() if len(v) > 1], dtype=np.int64), return_counts=True)
            )
        }
        if informative2
        else {},
        "vf2_comparisons": int(cmp2 + cmp3),
    }
    wl = _screen("sig2", "sig3")
    hash_screen = _screen("hash2", "hash3")

    # deterministic witnesses: same exact 2-ball class, different exact 3-ball class
    members_of: dict[int, list[int]] = {}
    for i, c in enumerate(exact2):
        members_of.setdefault(int(c), []).append(i)
    examples: list[dict[str, Any]] = []
    used_rows: set[int] = set()
    pending: list[list[int]] = []
    for c in sorted(members_of, key=lambda k: min(members_of[k])):
        by_class3: dict[int, int] = {}
        for i in sorted(members_of[c]):
            by_class3.setdefault(int(exact3[i]), i)
        if len(by_class3) > 1:
            pending.append([by_class3[k] for k in sorted(by_class3)[:2]])
    # prefer witnesses that involve molecules not already used (spread the evidence)
    for pair in pending + [
        p for p in pending if {int(atoms[p[0]]["row"]), int(atoms[p[1]]["row"])} & used_rows
    ]:
        rows_here = {int(atoms[pair[0]]["row"]), int(atoms[pair[1]]["row"])}
        if rows_here & used_rows:
            continue
        used_rows |= rows_here
        examples.append(_witness(task, atoms[pair[0]], atoms[pair[1]], smiles))
        if len(examples) >= 6:
            break
    verified = [e for e in examples if e.get("verified_2ball_identical") and e.get("verified_3ball_different")]

    return {
        "n_atoms_analysed": int(n_atoms),
        "n_molecules_analysed": int(meta["n_molecules"]),
        "exact": exact,
        "hash_screen": hash_screen,
        "wl_partition_screen": wl,
        "screens_agree_with_exact": bool(
            hash_screen["n_two_ball_classes"] == exact["n_two_ball_classes"]
            and hash_screen["n_classes_with_multiple_three_ball_classes"]
            == exact["n_classes_with_multiple_three_ball_classes"]
        ),
        "hash_seconds": hash_seconds,
        "exact_seconds": exact_seconds,
        "elapsed_seconds": float(meta["elapsed_seconds"] + hash_seconds + exact_seconds),
        "exact_method": (
            "VF2 isomorphism on the exact rooted induced subgraph, node labels (atom_type, is_root), "
            "edge labels bond_type; bucketed by a SHA-256 content signature (a bucket key only, so a "
            "hash collision can add work but cannot change a class)"
        ),
        "wl_method": (
            "partition-only rooted 1-WL: sound but incomplete AND label-oblivious (colour ids are "
            "renumbered by sorted uniqueness), which is why it under-separates; reported for the record"
        ),
        "exact_witness_check": "the same VF2 test applied directly to the two witness balls",
        "examples": examples,
        "n_examples": len(examples),
        "n_verified_examples": len(verified),
        "verified_pair_present": bool(verified),
        "claim": (
            "S3 exposes explicit local structure that the rooted 2-ball does not determine.  This is "
            "a statement about explicit local structural INPUT, not about assay information, and not "
            "about what a trained W2 can or cannot represent implicitly."
        ),
    }


def _witness(task: Any, a: Mapping[str, Any], b: Mapping[str, Any], smiles: Sequence[str] | None) -> dict[str, Any]:
    same2 = S3P.ball_isomorphic(a["ball2"], b["ball2"])
    same3 = S3P.ball_isomorphic(a["ball3"], b["ball3"])
    smiles_cache = task.cache.get("smiles") if isinstance(task.cache, Mapping) else None

    def _smiles(row: int) -> str | None:
        try:
            if smiles_cache is not None:
                entry = smiles_cache[int(row)]
                if isinstance(entry, Mapping):
                    return str(entry.get("smiles", "")) or None
                return str(entry)
        except Exception:  # pragma: no cover - diagnostic only
            return None
        return None

    return {
        "atom_a": {"row": int(a["row"]), "root": int(a["root"]), "n_atoms": int(a["n_atoms"]), "smiles": _smiles(a["row"])},
        "atom_b": {"row": int(b["row"]), "root": int(b["root"]), "n_atoms": int(b["n_atoms"]), "smiles": _smiles(b["row"])},
        "two_ball_sizes": [len(a["ball2"].nodes), len(b["ball2"].nodes)],
        "three_ball_sizes": [len(a["ball3"].nodes), len(b["ball3"].nodes)],
        "wl_signature_equal_at_radius_2": bool(a["sig2"] == b["sig2"]),
        "wl_signature_equal_at_radius_3": bool(a["sig3"] == b["sig3"]),
        "verified_2ball_identical": bool(same2),
        "verified_3ball_different": bool(not same3),
        "ball2_a": _ball_json(a["ball2"]),
        "ball3_a": _ball_json(a["ball3"]),
    }


def _ball_json(ball: S3P.RootedBall) -> dict[str, Any]:
    return {
        "radius": int(ball.radius),
        "root": int(ball.root),
        "n_nodes": len(ball.nodes),
        "labels": [int(x) for x in ball.labels],
        "is_root": [bool(x) for x in ball.is_root],
        "dist_from_root": [int(x) for x in ball.dist_from_root],
        "edges": [[int(i), int(j), int(bt)] for i, j, bt in ball.edges],
    }


def synthetic_novelty_witness() -> dict[str, Any]:
    """Cyclohexane vs n-hexane, rooted at the centre atom: exact witness."""
    def graph(n: int, bonds: Sequence[tuple[int, int]]):
        u = np.asarray([b[0] for b in bonds], dtype=np.int64)
        v = np.asarray([b[1] for b in bonds], dtype=np.int64)
        bt = np.ones(u.size, dtype=np.int64)
        at = np.full(n, 2, dtype=np.int64)  # carbon
        return S3P.molecule_adjacency(n, u, v, bt), at

    ring = graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
    chain = graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    root = 0
    b2_ring = S3P.rooted_ball(6, ring[0], ring[1], root, 2)
    b3_ring = S3P.rooted_ball(6, ring[0], ring[1], root, 3)
    b2_chain = S3P.rooted_ball(6, chain[0], chain[1], 2, 2)  # centre of the chain
    b3_chain = S3P.rooted_ball(6, chain[0], chain[1], 2, 3)
    p_ring = S3P.enumerate_s3(6, ring[0], ring[1])
    p_chain = S3P.enumerate_s3(6, chain[0], chain[1])
    return {
        "case": "cyclohexane (root 0) vs n-hexane (root 2, the centre atom)",
        "two_ball_rooted_isomorphic": bool(S3P.ball_isomorphic(b2_ring, b2_chain)),
        "three_ball_rooted_isomorphic": bool(S3P.ball_isomorphic(b3_ring, b3_chain)),
        "two_ball_wl_equal": bool(S3P.wl_signature(b2_ring) == S3P.wl_signature(b2_chain)),
        "three_ball_wl_equal": bool(S3P.wl_signature(b3_ring) == S3P.wl_signature(b3_chain)),
        "ring_ball2": _ball_json(b2_ring),
        "chain_ball2": _ball_json(b2_chain),
        "ring_ball3": _ball_json(b3_ring),
        "chain_ball3": _ball_json(b3_chain),
        "s3_paths_ring": int(p_ring.n_paths),
        "s3_paths_chain": int(p_chain.n_paths),
        "s3_per_root_ring": [int(x) for x in p_ring.per_root_counts().tolist()],
        "s3_per_root_chain": [int(x) for x in p_chain.per_root_counts().tolist()],
        "ring_root0_paths": p_ring.records(0),
        "chain_root2_paths": [r for r in p_chain.records(2)],
        "dist_ring": {str(d): int(np.sum(p_ring.dist == d)) for d in S3_DIST},
        "dist_chain": {str(d): int(np.sum(p_chain.dist == d)) for d in S3_DIST},
    }


# ---------------------------------------------------------------------------
# stage 0: cache + batch cost projection
# ---------------------------------------------------------------------------


def cache_projection(
    task: Any, sample: Mapping[str, Any], measured: Mapping[str, Any]
) -> dict[str, Any]:
    """Exact full-split S3 counts (closed form) plus measured throughput."""
    cache = task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    bond_index = np.asarray(cache["bond_index"])
    bond_type = np.asarray(cache["bond_type"])
    scope = np.sort(np.concatenate([np.asarray(task.train_rows, np.int64), np.asarray(task.valid_rows, np.int64)]))
    t0 = time.perf_counter()
    total_paths = 0
    total_atoms = 0
    total_bonds = 0
    per_row_paths = np.zeros(scope.size, dtype=np.int64)
    for i, row in enumerate(scope.tolist()):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        total_atoms += n
        total_bonds += b1 - b0
        if n < 4 or b1 == b0:
            continue
        bi = bond_index[b0:b1].astype(np.int64)  # molecule-local ids
        adj = S3P.molecule_adjacency(n, bi[:, 0], bi[:, 1], bond_type[b0:b1].astype(np.int64))
        per_row_paths[i] = int(S3P.count_s3_closed_form(n, adj).sum())
        total_paths += int(per_row_paths[i])
    count_seconds = float(time.perf_counter() - t0)

    # bytes per path in the planned cache layout
    bytes_per_path = 4 + 6 * 1  # int32 end + 6 int8 codes
    n_scope_atoms = int(total_atoms)
    cache_bytes = int(total_paths) * bytes_per_path + (n_scope_atoms + 1) * 8 + scope.size * 16
    sample_seconds = float(measured["elapsed_seconds"])
    sample_mols = len(measured["per_molecule"])
    sample_paths = int(measured["n_paths_sample"])
    # measured per-molecule overhead and per-path cost from the sample run
    per_mol_seconds = sample_seconds / max(sample_mols, 1)
    projected_seconds = per_mol_seconds * float(scope.size)
    return {
        "scope": "official Train + Valid rows (394,136); official Test excluded",
        "n_rows": int(scope.size),
        "n_atoms": int(total_atoms),
        "n_bonds": int(total_bonds),
        "n_paths_exact": int(total_paths),
        "paths_per_row_mean": float(per_row_paths.mean()),
        "paths_per_row_median": float(np.median(per_row_paths)),
        "paths_per_row_p90": _pct(per_row_paths, 90),
        "paths_per_row_p99": _pct(per_row_paths, 99),
        "paths_per_row_max": int(per_row_paths.max()),
        "rows_with_dist3_upper_bound_fraction": float((per_row_paths > 0).mean()),
        "count_only_scan_seconds": count_seconds,
        "bytes_per_path": int(bytes_per_path),
        "cache_bytes_projected": int(cache_bytes),
        "cache_gib_projected": float(cache_bytes / 1024**3),
        "sample_seconds": sample_seconds,
        "sample_molecules": int(sample_mols),
        "sample_paths": int(sample_paths),
        "sample_seconds_per_molecule": per_mol_seconds,
        "sample_paths_per_second": float(sample_paths / max(sample_seconds, 1e-9)),
        "build_seconds_projected_from_sample": float(projected_seconds),
        "build_minutes_projected_from_sample": float(projected_seconds / 60.0),
        "projection_method": (
            "full-split exact S3 count via the closed form (no path materialisation, no label read); "
            "wall-clock from the measured sample enumeration (per-molecule rate, which includes the "
            "path materialisation that the count-only pass does not)"
        ),
        "per_row_paths": per_row_paths,
    }


def batch_cost_projection(measured: Mapping[str, Any]) -> dict[str, Any]:
    per_mol = np.asarray([m["n_paths"] for m in measured["per_molecule"]], dtype=np.float64)
    dist = np.asarray(measured["dist_counts"], dtype=np.float64)
    total = float(dist.sum()) if dist.sum() else 1.0
    paths_per_batch_mean = float(per_mol.mean()) * C.BATCH_SIZE
    paths_per_batch_p99 = _pct(per_mol, 99) * C.BATCH_SIZE
    paths_per_epoch = float(per_mol.mean()) * C.N_TRAIN
    d = C.S3_DIM
    a_out = int(C.WIDTHS["atom_out"])
    # per atom: one W_e (192->48) and one out Linear (97->128); per path: cheap gathers/adds.
    flops_per_path = 14.0 * d
    flops_per_atom = 2.0 * a_out * d + 2.0 * C.S3_INPUT_DIM * C.S3_OUT_WIDTH + 2.0 * d
    atoms_per_epoch = float(np.mean([m["n_atoms"] for m in measured["per_molecule"]])) * C.N_TRAIN
    return {
        "batch_size": int(C.BATCH_SIZE),
        "paths_per_batch_mean": float(paths_per_batch_mean),
        "paths_per_batch_p99": float(paths_per_batch_p99),
        "paths_per_train_epoch": float(paths_per_epoch),
        "atom_dim": int(C.S3_DIM),
        "dist_fractions": {
            "1_closure_4ring": float(dist[0] / total),
            "2_shortcut": float(dist[1] / total),
            "3_true_third_shell": float(dist[2] / total),
        },
        "forward_flops_per_path_estimate": float(flops_per_path),
        "forward_flops_per_atom_estimate": float(flops_per_atom),
        "forward_flops_per_epoch_estimate": float(paths_per_epoch * flops_per_path + atoms_per_epoch * flops_per_atom),
        "batch_tensor_bytes_mean": int(
            paths_per_batch_mean * (7 * 8 + 4 * d)  # 7 int64 index tensors + z
        ),
        "path_index_tensor_bytes_p99": int(paths_per_batch_p99 * 7 * 8),
        "measured_peak_gpu_bytes_note": "filled by the GPU cost probe (analysis/architecture_audit.json)",
    }


# ---------------------------------------------------------------------------
# stage 0 driver
# ---------------------------------------------------------------------------


def run_structure(task: Any, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = select_sample_rows(task)
    rows = sample["rows"]
    C.log(f"[structure] sample {rows.size} train molecules sha {sample['sample_sha256'][:12]}")

    measured = measure_sample(task, rows)
    per_mol = measured["per_molecule"]
    mol_paths = np.asarray([m["n_paths"] for m in per_mol], dtype=np.int64)
    dist_counts = np.asarray(measured["dist_counts"], dtype=np.int64)
    dist_total = int(dist_counts.sum())

    # -- path count distribution -------------------------------------------
    per_root = np.asarray(measured["per_root_counts"], dtype=np.int64)
    mol_with_dist3 = int(np.sum([m["paths_with_dist3"] > 0 for m in per_mol]))
    mol_with_dist2 = int(np.sum([m["paths_with_dist2"] > 0 for m in per_mol]))
    mol_with_dist1 = int(np.sum([m["paths_with_dist1"] > 0 for m in per_mol]))
    type_counter = measured["path_type_counter"]
    type_counts = np.sort(np.asarray(list(type_counter.values()), dtype=np.int64))[::-1] if type_counter else np.zeros(0, np.int64)
    top_types = sorted(type_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:40]
    top_paths = np.sort(mol_paths)[::-1]
    top_decile_n = max(int(0.1 * mol_paths.size), 1)
    distribution = {
        "sample": {
            "n_molecules": int(rows.size),
            "sample_sha256": sample["sample_sha256"],
            "recipe": sample["recipe"],
            "strata": sample["strata"],
            "n_atoms": int(measured["n_atoms_sample"]),
            "n_paths": int(measured["n_paths_sample"]),
        },
        "paths_per_molecule": _summary(mol_paths),
        "paths_per_root_atom": _summary(per_root),
        "molecule_fraction_with_true_dist3_path": float(mol_with_dist3 / len(per_mol)),
        "molecule_fraction_with_dist2_path": float(mol_with_dist2 / len(per_mol)),
        "molecule_fraction_with_closure_dist1_path": float(mol_with_dist1 / len(per_mol)),
        "distance_class_counts": {"1": int(dist_counts[0]), "2": int(dist_counts[1]), "3": int(dist_counts[2])},
        "distance_class_fractions": (
            {
                "1": float(dist_counts[0] / dist_total),
                "2": float(dist_counts[1] / dist_total),
                "3": float(dist_counts[2] / dist_total),
            }
            if dist_total
            else {"1": 0.0, "2": 0.0, "3": 0.0}
        ),
        "distinct_path_types": int(len(type_counter)),
        "path_type_concentration": {
            "top1_share": float(type_counts[0] / dist_total) if type_counts.size and dist_total else 0.0,
            "top10_share": float(type_counts[:10].sum() / dist_total) if type_counts.size and dist_total else 0.0,
            "top100_share": float(type_counts[:100].sum() / dist_total) if type_counts.size and dist_total else 0.0,
            "types_with_single_occurrence": int((type_counts == 1).sum()) if type_counts.size else 0,
            "top40": [
                {"key": int(k), "count": int(v), "share": float(v / dist_total) if dist_total else 0.0, "type": S3P.decode_path_type(k)}
                for k, v in top_types
            ],
        },
        "root_endpoint_pairs": {
            **measured["pairs"],
            "fraction_already_reachable_within_two_bonds": float(
                measured["pairs"]["covered_by_s2"] / max(measured["pairs"]["total"], 1)
            ),
            "note": (
                "every dist=3 path reaches an atom that no exactly-2-bond S2 path can reach; "
                "dist=1/2 paths re-use S2-reachable endpoints and add route multiplicity "
                "(4-ring closure at distance 1, chorded/shortcut routes at distance 2)"
            ),
        },
        "concentration": {
            "gini_paths_per_molecule": _gini(mol_paths),
            "top_decile_share_of_all_paths": float(top_paths[:top_decile_n].sum() / max(int(mol_paths.sum()), 1)),
            "paths_from_top_5pct_molecules": float(
                top_paths[: max(int(0.05 * mol_paths.size), 1)].sum() / max(int(mol_paths.sum()), 1)
            ),
            "max_paths_molecule": int(mol_paths.max()) if mol_paths.size else 0,
            "max_paths_row": int(measured["max_paths_row"]),
            "max_molecule_share": float(mol_paths.max() / max(int(mol_paths.sum()), 1)) if mol_paths.size else 0.0,
        },
        "growth_by_size": growth_by_size(per_mol),
        "enumeration_seconds": float(measured["elapsed_seconds"]),
    }
    C.write_json(out_dir / "path_count_distribution.json", distribution)

    # -- synthetic cases ----------------------------------------------------
    cases = synthetic_cases()
    expected = {
        "cyclobutane_C4": {"n_paths": 8, "dist": {1: 8, 2: 0, 3: 0}},
        "cyclopentane_C5": {"n_paths": 10, "dist": {1: 0, 2: 10, 3: 0}},
        "cyclohexane_C6": {"n_paths": 12, "dist": {1: 0, 2: 0, 3: 12}},
        "butane_chain_C4": {"n_paths": 2, "dist": {1: 0, 2: 0, 3: 2}},
        "hexane_chain_C6": {"n_paths": 6, "dist": {1: 0, 2: 0, 3: 6}},
    }
    cases_ok = True
    for name, exp in expected.items():
        c = cases[name]
        ok = int(c["n_paths"]) == exp["n_paths"] and all(
            int(c["dist_histogram"][str(d)]) == exp["dist"][d] for d in (1, 2, 3)
        )
        c["expected"] = exp
        c["passed"] = bool(ok)
        cases_ok = cases_ok and ok

    # -- local distinguishability ------------------------------------------
    C.log("[structure] rooted 2-ball vs 3-ball novelty scan")
    novelty = novelty_analysis(task, rows)
    novelty["synthetic_witness"] = synthetic_novelty_witness()
    novelty["synthetic_cases"] = cases
    novelty["synthetic_cases_all_passed"] = bool(cases_ok)
    novelty["placement"] = (
        "real witness atoms are drawn from the same fixed 4096-molecule Train sample; "
        "no label is read at any point"
    )
    novelty["synthetic_witness_verified"] = bool(
        novelty["synthetic_witness"]["two_ball_rooted_isomorphic"]
        and not novelty["synthetic_witness"]["three_ball_rooted_isomorphic"]
    )
    C.write_json(out_dir / "structure_novelty.json", novelty)

    # -- cache + batch projection ------------------------------------------
    proj = cache_projection(task, sample, measured)
    per_row = np.asarray(proj.pop("per_row_paths"))
    batch = batch_cost_projection(measured)
    projection = {
        **proj,
        "batch": batch,
        "per_row_paths_summary": _summary(per_row),
        "gates": {
            "cache_gib_max": float(C.GATES["max_cache_gib"]),
            "preprocess_minutes_max": float(C.GATES["max_preprocess_minutes"]),
            "cache_gate_pass": bool(proj["cache_gib_projected"] <= C.GATES["max_cache_gib"]),
            "preprocess_gate_pass": bool(
                proj["build_minutes_projected_from_sample"] <= C.GATES["max_preprocess_minutes"]
            ),
        },
        "code_env": C.environment(),
    }
    C.write_json(out_dir / "cache_cost_projection.json", projection)

    # -- gate summary -------------------------------------------------------
    gate_rows = {
        "enumeration_matches_raw_graph": True,  # verified path-by-path in verify_sample_paths
        "atom_permutation_invariance": True,  # verified in permutation_check
        "molecule_fraction_with_dist3": float(mol_with_dist3 / len(per_mol)),
        "molecule_fraction_with_dist3_min": float(C.GATES["min_molecule_fraction_with_dist3"]),
        "dist3_coverage_pass": bool(mol_with_dist3 / len(per_mol) >= C.GATES["min_molecule_fraction_with_dist3"]),
        "witness_pair_present": bool(novelty["verified_pair_present"] and novelty["synthetic_witness_verified"]),
        "cache_gib_projected": float(proj["cache_gib_projected"]),
        "cache_gate_pass": bool(proj["cache_gib_projected"] <= C.GATES["max_cache_gib"]),
        "preprocess_minutes_projected": float(proj["build_minutes_projected_from_sample"]),
        "preprocess_gate_pass": bool(
            proj["build_minutes_projected_from_sample"] <= C.GATES["max_preprocess_minutes"]
        ),
        "synthetic_cases_pass": bool(cases_ok),
        "no_truncation_required": True,
        "no_truncation_evidence": (
            "the enumerator, the cache and the collate have no path cap and no sampling; "
            f"max {int(proj['paths_per_row_max'])} paths in one molecule, "
            f"{float(proj['paths_per_row_mean']):.1f} mean, projected cache "
            f"{float(proj['cache_gib_projected']):.3f} GiB"
        ),
        "path_cap_in_code": None,
        "paths_per_row_max": int(proj["paths_per_row_max"]),
    }
    return {"distribution": distribution, "novelty": novelty, "projection": projection, "gates": gate_rows}


def verify_sample_paths(task: Any, rows: Sequence[int], *, n_molecules: int = 256) -> dict[str, Any]:
    """Path-by-path verification: three real bonds, four distinct atoms."""
    checked = 0
    n_paths = 0
    for row in np.asarray(rows, dtype=np.int64)[: int(n_molecules)].tolist():
        mol = molecule_from_task(task, row)
        adj = mol["adj"]
        n = int(mol["n_atoms"])
        dist = S3P.all_pairs_shortest_paths(n, adj)
        paths = S3P.enumerate_s3(n, adj, mol["atom_type"], dist, keep_mid_atoms=True)
        bond = {(int(u), int(v)): int(t) for (u, v), t in zip(mol["bonds_uv"].tolist(), mol["bond_type"].tolist())}
        bond.update({(v, u): t for (u, v), t in list(bond.items())})
        for r in range(n):
            for rec in paths.records(r):
                u, v, x = rec["mid_atom_1"], rec["mid_atom_2"], rec["end_atom"]
                if len({r, u, v, x}) != 4:
                    raise RuntimeError(f"row {row}: S3 path with repeated atoms {rec}")
                if (r, u) not in bond or (u, v) not in bond or (v, x) not in bond:
                    raise RuntimeError(f"row {row}: S3 path over a non-bond {rec}")
                if bond[(r, u)] != rec["bond_type_1"] or bond[(u, v)] != rec["bond_type_2"] or bond[(v, x)] != rec["bond_type_3"]:
                    raise RuntimeError(f"row {row}: S3 path bond type mismatch {rec}")
                if int(dist[r, x]) != rec["shortest_distance"]:
                    raise RuntimeError(f"row {row}: S3 shortest_distance mismatch {rec}")
                if int(mol["atom_type"][u]) != rec["mid_type_1"] or int(mol["atom_type"][v]) != rec["mid_type_2"]:
                    raise RuntimeError(f"row {row}: S3 mid atom type mismatch {rec}")
                n_paths += 1
        checked += 1
    return {"molecules_checked": int(checked), "paths_checked": int(n_paths), "passed": True}


def permutation_check(task: Any, rows: Sequence[int], *, n_molecules: int = 128, seed: int = 20261109) -> dict[str, Any]:
    """Atom renumbering must leave the per-root S3 record multisets invariant."""
    gen = np.random.default_rng(int(seed))
    checked = 0
    total_paths = 0
    for row in np.asarray(rows, dtype=np.int64)[: int(n_molecules)]:
        mol = molecule_from_task(task, int(row))
        n = int(mol["n_atoms"])
        perm = gen.permutation(n)  # perm[i] = new id of old atom i
        inv = np.empty(n, dtype=np.int64)
        inv[perm] = np.arange(n, dtype=np.int64)
        adj = mol["adj"]
        atom_type = mol["atom_type"]
        base = S3P.enumerate_s3(n, adj, atom_type, keep_mid_atoms=True)
        # rebuild the permuted graph with identical labels
        new_adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for u_old, v_old, bt in zip(mol["bonds_uv"][:, 0].tolist(), mol["bonds_uv"][:, 1].tolist(), mol["bond_type"].tolist()):
            u_new, v_new = int(perm[u_old]), int(perm[v_old])
            new_adj[u_new].append((v_new, int(bt)))
            new_adj[v_new].append((u_new, int(bt)))
        for r in new_adj:
            r.sort()
        new_at = atom_type[inv]  # label of new atom j is the label of old atom inv[j]
        permuted = S3P.enumerate_s3(n, new_adj, new_at, keep_mid_atoms=True)
        if int(base.n_paths) != int(permuted.n_paths):
            raise RuntimeError(f"row {row}: S3 path count changed under atom renumbering")
        for r_old in range(n):
            r_new = int(perm[r_old])
            a = sorted(
                (
                    int(perm[rec["end_atom"]]),
                    int(perm[rec["mid_atom_1"]]),
                    int(perm[rec["mid_atom_2"]]),
                    rec["bond_type_1"],
                    rec["bond_type_2"],
                    rec["bond_type_3"],
                    rec["mid_type_1"],
                    rec["mid_type_2"],
                    rec["shortest_distance"],
                )
                for rec in base.records(r_old)
            )
            b = sorted(
                (
                    rec["end_atom"],
                    rec["mid_atom_1"],
                    rec["mid_atom_2"],
                    rec["bond_type_1"],
                    rec["bond_type_2"],
                    rec["bond_type_3"],
                    rec["mid_type_1"],
                    rec["mid_type_2"],
                    rec["shortest_distance"],
                )
                for rec in permuted.records(r_new)
            )
            if a != b:
                raise RuntimeError(f"row {row}: S3 records are not renumbering-equivariant at root {r_old}")
        total_paths += int(base.n_paths)
        checked += 1
    return {"molecules_checked": int(checked), "paths_checked": int(total_paths), "passed": True, "seed": int(seed)}


__all__ = [
    "batch_cost_projection",
    "cache_projection",
    "growth_by_size",
    "measure_sample",
    "novelty_analysis",
    "permutation_check",
    "run_structure",
    "select_sample_rows",
    "size_strata",
    "synthetic_cases",
    "synthetic_novelty_witness",
    "verify_sample_paths",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", required=True, help="comma separated audit modes")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--s3-cache", default=str(C.S3_CACHE_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-size", type=int, default=C.SAMPLE_SIZE)
    parser.add_argument("--quick", action="store_true", help="tiny sample (smoke only, never a result)")
    args = parser.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else C.ANALYSIS
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for mode in modes:
        C.log(f"=== audit mode {mode} ===")
        t0 = time.perf_counter()
        try:
            payload, path = _run_mode(mode, args, out_dir)
            payload = dict(payload)
            payload.setdefault("audit_mode", mode)
            payload.setdefault("round_id", C.ROUND_ID)
            payload.setdefault("protocol_id", C.PROTOCOL_ID)
            payload.setdefault("created_utc", C.utcnow())
            payload.setdefault("seconds", float(time.perf_counter() - t0))
            payload.setdefault("test_evaluated", False)
            C.write_json(path, payload)
            C.log(f"[{mode}] wrote {C.track_rel(path)} in {payload['seconds']:.1f}s")
        except Exception as exc:  # noqa: BLE001
            C.log(f"[{mode}] FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            C.write_json(
                out_dir / f"{mode}_failed.json",
                {
                    "audit_mode": mode,
                    "round_id": C.ROUND_ID,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "created_utc": C.utcnow(),
                },
            )
            rc = 1
    return rc


def _run_mode(mode: str, args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], Path]:
    from sg_data import load_task

    if mode == "structure":
        task = load_task()
        quick = bool(args.quick)
        if quick:
            sample = select_sample_rows(task, total=min(96, int(args.sample_size)))
            rows = sample["rows"]
            meas = measure_sample(task, rows)
            nov = novelty_analysis(task, rows)
            ver = verify_sample_paths(task, rows)
            perm = permutation_check(task, rows)
            payload = {
                "quick": True,
                "sample": {k: v for k, v in sample.items() if k != "rows"},
                "paths_per_molecule": _summary([m["n_paths"] for m in meas["per_molecule"]]),
                "dist_counts": meas["dist_counts"],
                "novelty_verified_pair_present": nov["verified_pair_present"],
                "synthetic_cases": synthetic_cases(),
                "synthetic_witness_verified": synthetic_novelty_witness()["three_ball_rooted_isomorphic"] is False,
                "path_verification": ver,
                "permutation": perm,
            }
            return payload, out_dir / "structure_quick.json"
        res = run_structure(task, out_dir)
        rows = select_sample_rows(task, total=int(args.sample_size))["rows"]
        ver = verify_sample_paths(task, rows)
        perm = permutation_check(task, rows)
        gates = dict(res["gates"])
        gates.update(
            {
                "path_by_path_verification": ver,
                "enumeration_matches_raw_graph": bool(ver["passed"]),
                "atom_permutation_invariance": bool(perm["passed"]),
                "permutation_check": perm,
            }
        )
        gates["all_structural_gates_pass"] = bool(
            gates["enumeration_matches_raw_graph"]
            and gates["atom_permutation_invariance"]
            and gates["dist3_coverage_pass"]
            and gates["witness_pair_present"]
            and gates["cache_gate_pass"]
            and gates["preprocess_gate_pass"]
            and gates["synthetic_cases_pass"]
            and gates["no_truncation_required"]
        )
        payload = {
            "gates": gates,
            "stage0_pass": bool(gates["all_structural_gates_pass"]),
            "status": "S3_PREFLIGHT_STRUCTURE_PASSED" if gates["all_structural_gates_pass"] else "S3_COST_GATE_PENDING",
            "files": {
                "path_count_distribution": "analysis/path_count_distribution.json",
                "structure_novelty": "analysis/structure_novelty.json",
                "cache_cost_projection": "analysis/cache_cost_projection.json",
            },
        }
        return payload, out_dir / "stage0_gates.json"

    if mode == "cache":
        task = load_task()
        target = Path(args.s3_cache)
        meta = C.build_s3_cache(task, target)
        # verify the frozen-collate row bookkeeping on a fixed batch
        from sg_data import make_plan40

        plan = make_plan40(task, seed=int(C.SEED))
        s3 = C.load_s3_cache(target)
        rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)
        batch = C.collate_with_s3(task, s3, rows.tolist())
        iso = C.check_s3_batch_isolation(batch)
        if not iso["isolated"]:
            raise RuntimeError(f"S3 batch isolation failed: {iso}")
        if not np.array_equal(batch["atom_counts"].numpy(), np.asarray(s3["s3_row_atoms"])[s3["row_index"][rows]]):
            raise RuntimeError("S3 row-id / atom-count bookkeeping mismatch")
        payload = {
            "cache": meta,
            "batch": {
                "rows": int(rows.size),
                "n_paths": int(batch["s3_path_atom"].numel()),
                "max_end": int(batch["s3_path_end"].max().item()) if batch["s3_path_end"].numel() else -1,
                "n_atoms": int(batch["atom_batch"].numel()),
                "isolation": iso,
                "paths_per_molecule": float(batch["s3_path_atom"].numel() / rows.size),
            },
            "row_id_consistency": True,
            "labels_read": False,
            "test_rows_present": False,
        }
        return payload, out_dir / "s3_cache_audit.json"

    if mode == "gpu-preflight":
        return _gpu_preflight(), out_dir / "gpu_preflight.json"

    if mode in ("init", "architecture", "cost", "profile", "w2-replay", "ablation"):
        import torch

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for a GPU audit mode")
        task = C.load_task_with_s3(Path(args.s3_cache))
        if mode == "init":
            return _run_init(task, device), out_dir / "initialization_parity.json"
        if mode == "architecture":
            return _run_architecture(task, device), out_dir / "architecture_audit.json"
        if mode == "cost":
            return _run_cost(task, device), out_dir / "gpu_cost_probe.json"
        if mode == "profile":
            return _run_profile(task, device), out_dir / "gpu_profile.json"
        if mode == "w2-replay":
            return _run_w2_replay(task, device), out_dir / "w2_best_replay.json"
        if mode == "ablation":
            return _run_ablation(task, device), out_dir / "s3_ablation.json"

    raise KeyError(f"unknown audit mode {mode!r}")



# ---------------------------------------------------------------------------
# GPU audit modes (pre-training)
# ---------------------------------------------------------------------------


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _gpu_preflight() -> dict[str, Any]:
    return {
        "host_facts": C.BC.host_facts(),
        "gpu_occupancy": C.BC.gpu_occupancy(),
        "environment": C.environment(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": C.git_state(),
    }


def _batch(task: Any, plan: Sequence[Mapping[str, Any]], n: int, device: Any) -> dict[str, Any]:
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)[: int(n)]
    return _to_device(C.collate_with_s3(task, task.s3, rows.tolist()), device)


def _permute_batch(batch: Mapping[str, Any], *, seed: int) -> tuple[dict[str, Any], Any]:
    """Renumber atoms inside every molecule, including the S3 path tensors."""
    import torch
    import bs_model as BSM

    out, p = BSM.permute_atoms(batch, seed=int(seed))
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(p.numel(), dtype=torch.long)
    for key in ("s3_path_atom", "s3_path_end"):
        value = out[key]
        out[key] = p_inv[value] if value.numel() else value
    out = dict(out)
    # permute_atoms drops unknown keys from the returned copy? it clones all keys
    for key in ("s3_mid1_type", "s3_mid2_type", "s3_bond1", "s3_bond2", "s3_bond3", "s3_dist"):
        out[key] = batch[key]
    return out, p


def _run_init(task: Any, device: Any) -> dict[str, Any]:
    import torch
    import bs_model as BSM
    import bs_train as BT
    import s3_model as M
    import s3_train as ST

    plan = C.BC.SD.make_plan40(task, seed=int(C.SEED))
    batch = _batch(task, plan, 32, device)
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)[:32]
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)

    w2_cpu, w2_audit = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device="cpu")
    w2_state = {k: v.detach().cpu().clone() for k, v in w2_cpu.state_dict().items()}
    base_fp = C.state_fingerprint({k: v for k, v in w2_state.items() if not k.startswith("readout.")})
    w2 = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)[0]
    w2.eval()
    with torch.no_grad():
        logits_w2 = w2(batch).detach()

    report: dict[str, Any] = {
        "base_model": C.BASE_MODEL_KEY,
        "base_parameters": C.BASE_PARAMETERS,
        "base_trunk_init_fingerprint": base_fp,
        "recorded_w2_trunk_init_fingerprint": C.W2["trunk_init_fingerprint"],
        "base_init_fingerprint_matches_history": bool(base_fp == C.W2["trunk_init_fingerprint"]),
        "w2_base_state_sha256": M.state_sha256(w2_state),
        "device": str(device),
        "epoch0_logit_tolerance": 0.0 if device.type == "cpu" else 2e-5,
        "arms": {},
    }
    for key in C.MODEL_ORDER:
        C.seed_everything(int(C.SEED))
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
        model.eval()
        with torch.no_grad():
            logits = model(batch).detach()
        diff = float((logits - logits_w2).abs().max().item()) if logits.shape == logits_w2.shape else float("inf")
        zero_diff = None
        if key == "S3":
            with torch.no_grad():
                zero_diff = float((model(batch, s3_zero=True) - logits_w2).abs().max().item())

        opt, opt_audit = BSM.make_optimizer(model)
        model.train()
        first_grads: dict[str, float] = {}
        grads_after_three: dict[str, float] = {}
        losses: list[float] = []
        for step in range(3):
            out = BT.train_step(model, opt, batch, y, global_step=step, clip=float(C.RECIPE["gradient_clip"]))
            losses.append(float(out["loss"]))
            if step == 0:
                first_grads = M.branch_grad_norm(model)
            if step == 2:
                grads_after_three = M.branch_grad_norm(model)
        if key == "S3":
            first_active = [k for k, v in first_grads.items() if v > 0]
            upstream_first_step = [k for k in first_grads if k != "s3_branch.out.weight" and k != "s3_branch.out.bias"]
            first_step_values = {k: float(first_grads.get(k, 0.0)) for k in sorted(upstream_first_step)}
        else:
            first_active = [k for k, v in first_grads.items() if v > 0]
            first_step_values = {k: float(first_grads.get(k, 0.0)) for k in sorted(first_grads)}

        # resume audit: full payload -> fresh model/optimizer -> identical state + logits
        payload = ST.checkpoint_payload(
            model_key=key,
            epoch=1,
            global_step=3,
            model=model,
            optimizer=opt,
            rng_state=C.capture_rng_state(),
            plan=ST.plan_meta(plan, 1, int(C.SEED)),
            model_audit=audit,
            valid_ap=0.0,
            valid_report=None,
            kind="audit_roundtrip",
        )
        restored, restored_audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
        restored_opt, _ = BSM.make_optimizer(restored)
        resume_info = ST.restore(payload, restored, restored_opt, model_key=key)
        restored.eval()
        with torch.no_grad():
            logits_restored = restored(batch).detach()
        model.eval()
        with torch.no_grad():
            logits_now = model(batch).detach()
        resume_diff = float((logits_restored - logits_now).abs().max().item())
        state_same = M.state_sha256({k: v.detach().cpu() for k, v in restored.state_dict().items()}) == M.state_sha256(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )

        branch_diag = M.branch_diagnostics(model, batch)
        report["arms"][key] = {
            "parameters": audit["total_parameters"],
            "new_parameters": audit["new_parameters"],
            "original_w2_subset_bitwise_equal": audit["original_w2_subset"]["all_bitwise_equal"],
            "base_state_sha256": audit["base_state_sha256"],
            "base_state_matches_w2": bool(audit["base_state_sha256"] == report["w2_base_state_sha256"]),
            "epoch0_logits_max_abs_diff_vs_w2": diff,
            "epoch0_zero_branch_logits_max_abs_diff_vs_w2": zero_diff,
            "epoch0_parity_pass": bool(diff <= (0.0 if device.type == "cpu" else 2e-5)),
            "new_module_init": audit["new_module_init"],
            "first_step_active_new_grads": first_active,
            "first_step_upstream_new_grads": first_step_values,
            "third_step_active_new_grads": [k for k, v in grads_after_three.items() if v > 0],
            "three_step_losses": losses,
            "resume_restore_state_identical": bool(state_same),
            "resume_restore_logits_max_abs_diff": resume_diff,
            "resume_restore_info": resume_info,
            "branch_diagnostics_after_three_steps": {
                k: branch_diag[k] for k in ("residual_over_h_raw", "nonzero_fraction", "n_paths") if k in branch_diag
            },
            "optimizer": opt_audit,
        }
        del model, opt, restored, restored_opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["checks"] = {
        "base_init_matches_history": bool(report["base_init_fingerprint_matches_history"]),
        "both_arms_epoch0_equal_w2": bool(all(v["epoch0_parity_pass"] for v in report["arms"].values())),
        "both_arms_base_bitwise_equal": bool(all(v["base_state_matches_w2"] for v in report["arms"].values())),
        "resume_restores_exactly": bool(all(v["resume_restore_state_identical"] for v in report["arms"].values())),
        "new_modules_receive_gradients": bool(
            all(v["third_step_active_new_grads"] for v in report["arms"].values())
        ),
    }
    report["checks"]["all_pass"] = bool(all(report["checks"].values()))
    return report


def _run_architecture(task: Any, device: Any) -> dict[str, Any]:
    import torch
    import bs_model as BSM
    import s3_model as M

    plan = C.BC.SD.make_plan40(task, seed=int(C.SEED))
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)[:32]
    batch = _batch(task, plan, 32, device)
    out: dict[str, Any] = {"device": str(device), "rows": int(rows.size), "n_paths": int(batch["s3_path_atom"].numel())}

    # 1) call counts (no iterative message passing, one local call, one dropout)
    calls: dict[str, Any] = {}
    s3_model, s3_audit = M.build_model("S3", task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    f_model, f_audit = M.build_model("F", task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    calls["S3"] = M.count_module_calls_typed(s3_model, batch)
    calls["F"] = M.count_module_calls_typed(f_model, batch)
    out["module_calls"] = calls
    out["one_local_one_drop"] = {
        k: bool(v["local"] == 1 and v["drop"] == 1 and v["atom_encoder"] == 1 and v["bond_encoder"] == 1)
        for k, v in calls.items()
    }

    # 2) the S3 branch cannot read bonds, S2 paths, h or any propagated state
    a, b, scal = s3_model._encode(batch)
    residual_full, _aux = s3_model.s3_branch(batch, a)
    minimal = {k: v for k, v in batch.items() if k.startswith("s3_")}
    residual_min, _ = s3_model.s3_branch(minimal, a)
    out["s3_branch_reads_only_static_atom_embedding_and_s3"] = {
        "keys_offered_in_minimal_call": sorted(minimal),
        "max_abs_diff": float((residual_full - residual_min).abs().max().item()),
        "identical": bool(torch.equal(residual_full, residual_min)),
    }

    # 3) with the S3 residual forced to zero, the model is exactly W2 on the same base weights
    w2_loaded = BSM.BSModel(task.spec, task.norm, C.N_TASKS, s3_model.cfg)
    base_state = {k: v for k, v in s3_model.state_dict().items() if not (k.startswith("s3_branch."))}
    w2_loaded.load_state_dict(base_state, strict=True)
    w2_loaded.to(device).eval()
    s3_model.eval()
    with torch.no_grad():
        lg_zero = s3_model(batch, s3_zero=True).detach()
        lg_w2 = w2_loaded(batch).detach()
        lg_full = s3_model(batch).detach()
    out["zero_branch_equals_w2"] = {
        "max_abs_diff": float((lg_zero - lg_w2).abs().max().item()),
        "pass": bool(float((lg_zero - lg_w2).abs().max().item()) <= (0.0 if device.type == "cpu" else 2e-5)),
        "branch_effect_max_abs_diff": float((lg_full - lg_w2).abs().max().item()),
    }

    # 4) atom renumbering leaves the graph logits unchanged
    permuted, p = _permute_batch(batch, seed=20261109)
    with torch.no_grad():
        lg_perm = s3_model(permuted).detach()
    out["atom_renumbering"] = {
        "max_abs_logit_diff": float((lg_full - lg_perm).abs().max().item()),
        "tolerance": 2e-5,
        "pass": bool(float((lg_full - lg_perm).abs().max().item()) <= 2e-5),
        "n_atoms": int(batch["atom_batch"].numel()),
    }

    # 5) batch isolation + row-id consistency of the S3 cache
    iso = C.check_s3_batch_isolation(batch)
    per_row_cached = np.asarray(task.s3["s3_row_paths"])[np.asarray(task.s3["row_index"])[rows]]
    out["batch"] = {
        "isolation": iso,
        "paths_per_row_cached_sum": int(per_row_cached.sum()),
        "paths_in_batch": int(batch["s3_path_atom"].numel()),
        "row_id_consistency": bool(int(per_row_cached.sum()) == int(batch["s3_path_atom"].numel())),
    }

    # 6) empty-S3 safety and the pathless-root convention
    empty = dict(batch)
    for key, shape in (
        ("s3_path_atom", (0,)),
        ("s3_path_end", (0,)),
        ("s3_mid1_type", (0,)),
        ("s3_mid2_type", (0,)),
        ("s3_bond1", (0,)),
        ("s3_bond2", (0,)),
        ("s3_bond3", (0,)),
        ("s3_dist", (0,)),
    ):
        empty[key] = torch.zeros(shape, dtype=torch.long, device=device)
    with torch.no_grad():
        lg_empty = s3_model(empty).detach()
        res_empty, aux_empty = s3_model.s3_branch(empty, a)
    with torch.no_grad():
        torch.nn.init.constant_(s3_model.s3_branch.out.bias, 0.25)
        res_bias, _ = s3_model.s3_branch(empty, a)
        torch.nn.init.zeros_(s3_model.s3_branch.out.bias)
    out["empty_s3"] = {
        "logits_finite": bool(torch.isfinite(lg_empty).all().item()),
        "logits_close_to_w2": float((lg_empty - lg_w2).abs().max().item()),
        "residual_all_zero": bool(float(res_empty.abs().max().item()) == 0.0),
        "pathless_residual_equals_bias": bool(
            float((res_bias - 0.25).abs().max().item()) < 1e-6
        ),
        "n_paths": int(aux_empty["n_paths"]),
    }

    # 7) bond edits change the S3 encoding (structural, label-free)
    import s3_paths as S3P

    edits: list[dict[str, Any]] = []
    for row in rows[:6].tolist():
        mol = molecule_from_task(task, int(row))
        n = int(mol["n_atoms"])
        base_dist = S3P.all_pairs_shortest_paths(n, mol["adj"])
        base_paths = S3P.enumerate_s3(n, mol["adj"], mol["atom_type"], base_dist)
        if mol["bonds_uv"].size == 0:
            continue
        # (a) change one bond type
        bi = mol["bonds_uv"].copy()
        bt = mol["bond_type"].copy()
        bt[0] = 2 if int(bt[0]) != 2 else 4
        adj_ret = S3P.molecule_adjacency(n, bi[:, 0], bi[:, 1], bt)
        p_ret = S3P.enumerate_s3(n, adj_ret, mol["atom_type"], S3P.all_pairs_shortest_paths(n, adj_ret))
        # (b) break one bond
        adj_break = S3P.molecule_adjacency(n, bi[1:, 0], bi[1:, 1], mol["bond_type"][1:])
        p_break = S3P.enumerate_s3(n, adj_break, mol["atom_type"], S3P.all_pairs_shortest_paths(n, adj_break))
        edits.append(
            {
                "row": int(row),
                "paths_base": int(base_paths.n_paths),
                "paths_after_bond_type_change": int(p_ret.n_paths),
                "paths_after_bond_removal": int(p_break.n_paths),
                "path_types_changed_by_bond_type": bool(
                    sorted(S3P.path_type_keys(base_paths).tolist()) != sorted(S3P.path_type_keys(p_ret).tolist())
                ),
                "count_changed_by_bond_removal": bool(int(p_break.n_paths) != int(base_paths.n_paths)),
            }
        )
    out["bond_edit_sensitivity"] = {
        "cases": edits,
        "all_bond_type_edits_change_the_encoding": bool(all(e["path_types_changed_by_bond_type"] for e in edits))
        if edits
        else False,
        "all_bond_removals_change_the_count": bool(all(e["count_changed_by_bond_removal"] for e in edits)) if edits else False,
    }

    out["no_message_passing_statement"] = (
        "the trunk runs exactly one atom encoder, one bond encoder and one S1/S2 local block; "
        "the S3 branch consumes only the static atom embedding and the once-enumerated path tensors "
        "(verified by the minimal-input call above), and nothing is propagated along a bond twice. "
        "There is no iterative message passing in either arm."
    )
    out["checks"] = {
        "one_local_one_drop": bool(all(out["one_local_one_drop"].values())),
        "s3_branch_minimal_input_identical": bool(out["s3_branch_reads_only_static_atom_embedding_and_s3"]["identical"]),
        "zero_branch_equals_w2": bool(out["zero_branch_equals_w2"]["pass"]),
        "atom_renumbering_invariant": bool(out["atom_renumbering"]["pass"]),
        "batch_isolated": bool(iso["isolated"]),
        "row_id_consistent": bool(out["batch"]["row_id_consistency"]),
        "empty_s3_safe": bool(out["empty_s3"]["logits_finite"] and out["empty_s3"]["pathless_residual_equals_bias"]),
        "bond_edits_change_s3": bool(
            out["bond_edit_sensitivity"]["all_bond_type_edits_change_the_encoding"]
            and out["bond_edit_sensitivity"]["all_bond_removals_change_the_count"]
        ),
    }
    out["checks"]["all_pass"] = bool(all(out["checks"].values()))
    out["parameter_audit"] = {"S3": s3_audit, "F": f_audit}
    return out


def _timing_probe(task: Any, key: str, device: Any, *, warmup: int, steps: int) -> dict[str, Any]:
    import torch
    import bs_model as BSM
    import bs_train as BT
    import s3_model as M
    from pf_pipeline import PinnedStager

    plan = C.BC.SD.make_plan40(task, seed=int(C.SEED))
    C.seed_everything(int(C.SEED))
    model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    opt, _ = BSM.make_optimizer(model)
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    times: list[float] = []
    collate_times: list[float] = []
    losses: list[float] = []
    rows_iter = [np.asarray(b, dtype=np.int64) for b in plan[0]["batches"]]
    n_paths = 0
    n_atoms = 0
    step = 0
    for rows in rows_iter[: int(warmup) + int(steps)]:
        t_c = time.perf_counter()
        raw = C.collate_with_s3(task, task.s3, rows.tolist())
        collate_times.append(float(time.perf_counter() - t_c))
        batch = stager.batch(raw)
        y = stager.labels(task.labels[rows])
        if step < int(warmup):
            BT.train_step(model, opt, batch, y, global_step=step, clip=clip)
        else:
            t0 = time.perf_counter()
            out = BT.train_step(model, opt, batch, y, global_step=step, clip=clip)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(float(time.perf_counter() - t0))
            losses.append(float(out["loss"]))
            n_paths = int(batch["s3_path_atom"].numel()) if key == "S3" else 0
            n_atoms = int(batch["atom_batch"].numel())
        step += 1
    gpu = C.BC.gpu_peak(device)
    stager.release()
    arr = np.asarray(times, dtype=np.float64)
    coll = np.asarray(collate_times, dtype=np.float64) * 1e3
    out = {
        "key": key,
        "parameters": audit["total_parameters"],
        "new_parameters": audit["new_parameters"],
        "warmup_steps": int(warmup),
        "measured_steps": int(steps),
        "step_ms_mean": float(arr.mean() * 1e3),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "collate_ms_p50": float(np.percentile(coll, 50)),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "projected_epoch_seconds": float(np.percentile(arr, 50) * C.STEPS_PER_EPOCH),
        "projected_epoch_seconds_mean": float(arr.mean() * C.STEPS_PER_EPOCH),
        "loss_mean": float(np.mean(losses)) if losses else None,
        "peak_allocated_bytes": int(gpu["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(gpu["peak_reserved_bytes"]),
        "s3_paths_per_batch": int(n_paths),
        "atoms_per_batch": int(n_atoms),
    }
    if key == "S3":
        out["branch_flops"] = M.branch_flops(model, n_paths=int(out["s3_paths_per_batch"]), n_atoms=int(n_atoms))
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _validation_timing(task: Any, key: str, device: Any) -> dict[str, Any]:
    import torch
    import s3_model as M

    model, _ = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    model.eval()
    rows = np.asarray(task.valid_rows, dtype=np.int64)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, rows.size, 256):
            chunk = rows[start : start + 256]
            batch = _to_device(C.collate_with_s3(task, task.s3, chunk.tolist()), device)
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = float(time.perf_counter() - t0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"key": key, "validation_rows": int(rows.size), "seconds": seconds}


def _run_cost(task: Any, device: Any) -> dict[str, Any]:

    warmup, steps = 5, 20
    timings = {
        key: _timing_probe(task, key, device, warmup=warmup, steps=steps)
        for key in (C.BASE_MODEL_KEY, *C.MODEL_ORDER)
    }
    validations = {key: _validation_timing(task, key, device) for key in (C.BASE_MODEL_KEY, *C.MODEL_ORDER)}
    hist = float(C.W2["epoch_seconds_mean"])
    measured_w2 = float(timings[C.BASE_MODEL_KEY]["projected_epoch_seconds"])
    for key, t in timings.items():
        t["validation_seconds"] = float(validations[key]["seconds"])
        t["validation_fraction_of_projected_epoch"] = float(
            validations[key]["seconds"] / max(t["projected_epoch_seconds"], 1e-9)
        )
        t["epoch_ratio_vs_historical_w2"] = float(t["projected_epoch_seconds"] / hist)
        t["epoch_ratio_vs_measured_w2"] = float(t["projected_epoch_seconds"] / max(measured_w2, 1e-9))
        t["projected_total_seconds_40_epochs_with_validation"] = float(
            40.0 * (t["projected_epoch_seconds"] + validations[key]["seconds"])
        )
        t["projected_total_gpu_hours"] = float(t["projected_total_seconds_40_epochs_with_validation"] / 3600.0)
    s3_ratio = float(timings["S3"]["projected_epoch_seconds"] / hist)
    total_hours = float(sum(v["projected_total_gpu_hours"] for k, v in timings.items() if k != C.BASE_MODEL_KEY))
    out = {
        "device": str(device),
        "warmup_steps": warmup,
        "measured_steps": steps,
        "timings": timings,
        "validation": validations,
        "historical_w2_epoch_seconds": hist,
        "measured_w2_epoch_seconds": measured_w2,
        "s3_epoch_ratio_vs_historical_w2": s3_ratio,
        "s3_epoch_ratio_max": float(C.GATES["max_epoch_ratio_vs_w2"]),
        "epoch_ratio_gate_pass": bool(s3_ratio <= float(C.GATES["max_epoch_ratio_vs_w2"])),
        "projected_formal_gpu_hours_S3_plus_F": total_hours,
        "budget": dict(C.BUDGET),
        "budget_gate_pass": bool(total_hours <= float(C.BUDGET["gpu_target_hours"])),
        "s3_vs_ffn_step_ratio": float(
            timings["S3"]["step_ms_p50"] / max(timings["F"]["step_ms_p50"], 1e-9)
        ),
        "memory_note": "peak_allocated_bytes / peak_reserved_bytes are the real training-step peaks",
        "gpu_occupancy": C.BC.gpu_occupancy(),
    }
    out["all_gates_pass"] = bool(out["epoch_ratio_gate_pass"])
    return out


def _run_profile(task: Any, device: Any) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    plan = C.BC.SD.make_plan40(task, seed=int(C.SEED))
    report: dict[str, Any] = {"device": str(device), "batches": {}}
    for n_rows in (8, 128):
        rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)[:n_rows]
        batch = _to_device(C.collate_with_s3(task, task.s3, rows.tolist()), device)
        import s3_model as M
        import bs_train as BT
        import bs_model as BSM

        model, _ = M.build_model("S3", task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
        opt, _ = BSM.make_optimizer(model)
        y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
        model.train()
        for step in range(3):
            BT.train_step(model, opt, batch, y, global_step=step, clip=5.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for step in range(5):
                BT.train_step(model, opt, batch, y, global_step=100 + step, clip=5.0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        key_averages = prof.key_averages()
        events = list(key_averages)
        total_cuda = sum(float(e.self_device_time_total) for e in events)
        top = sorted(events, key=lambda e: -float(e.self_device_time_total))[:12]
        report["batches"][str(n_rows)] = {
            "rows": int(n_rows),
            "n_paths": int(batch["s3_path_atom"].numel()),
            "n_atoms": int(batch["atom_batch"].numel()),
            "distinct_ops": int(len(events)),
            "total_op_calls_per_step": float(sum(e.count for e in events) / 5.0),
            "total_cuda_time_ms_per_step": float(total_cuda / 5.0 / 1000.0),
            "top_ops": [
                {
                    "key": e.key,
                    "calls_per_step": float(e.count / 5.0),
                    "self_cuda_ms_per_step": float(e.self_device_time_total / 5.0 / 1000.0),
                    "self_cpu_ms_per_step": float(e.self_cpu_time_total / 5.0 / 1000.0),
                }
                for e in top
            ],
        }
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    small = report["batches"]["8"]
    large = report["batches"]["128"]
    report["op_count_ratio_128_over_8"] = float(large["total_op_calls_per_step"] / max(small["total_op_calls_per_step"], 1.0))
    report["path_count_ratio_128_over_8"] = float(large["n_paths"] / max(small["n_paths"], 1))
    report["no_per_path_python_loop"] = bool(report["op_count_ratio_128_over_8"] < 2.0)
    report["statement"] = (
        "The number of dispatched operators per training step is (nearly) independent of the number of "
        "S3 paths, while the path count grows with batch size: the S3 branch is tensorised (gather + "
        "index_add_ + two Linears), not a per-path Python loop."
    )
    return report


def _run_w2_replay(task: Any, device: Any) -> dict[str, Any]:
    import torch
    import bs_model as BSM

    run = json.loads(C.W2_RUN_JSON.read_text(encoding="utf-8"))
    recorded = run["fixed_epoch_reports"]
    out: dict[str, Any] = {
        "source_round": C.W2["round"],
        "run_json": C.track_rel(C.W2_RUN_JSON),
        "recorded_best_epoch": int(run["best_epoch"]),
        "recorded_best_valid_ap": float(run["best_valid_ap"]),
        "recorded_ap40": float(recorded[str(int(run["final_epoch"]))]["ap"]),
        "selection_rule": "argmax official Validation macro AP over epochs 1..40 (read back from the historical run)",
        "test_accessed": False,
    }
    entries = {}
    for name, path, epoch in (
        ("best", C.W2_CHECKPOINT_BEST, int(run["best_epoch"])),
        ("epoch40", C.W2_CHECKPOINT_EPOCH40, int(run["final_epoch"])),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"historical W2 checkpoint missing: {path}")
        sha = C.sha256_file(path)
        expected = C.W2["best_checkpoint_sha256"] if name == "best" else C.W2["epoch40_checkpoint_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model, _ = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        report = C.BC.SD.evaluate_validation(model, task, device, reason=f"w2_{name}_read_only_replay")
        rec = recorded[str(epoch)]
        per_task_diff = [
            abs(float(a) - float(b)) for a, b in zip(report["per_task_ap"], rec["per_task_ap"]) if a is not None and b is not None
        ]
        entries[name] = {
            "checkpoint": C.track_rel(path),
            "checkpoint_sha256": sha,
            "expected_sha256": expected,
            "sha_matches": bool(sha == expected),
            "epoch": int(epoch),
            "ap": float(report["ap"]),
            "recorded_ap": float(rec["ap"]),
            "ap_delta_vs_recorded": float(report["ap"]) - float(rec["ap"]),
            "valid_bce": float(report["valid_bce"]),
            "recorded_valid_bce": float(rec["valid_bce"]),
            "n_tasks_scored": int(report["n_tasks_scored"]),
            "per_task_ap": report["per_task_ap"],
            "observed": report["observed"],
            "positive": report["positive"],
            "max_abs_per_task_ap_delta": float(max(per_task_diff)) if per_task_diff else 0.0,
            "valid_logits_sha256": report["valid_logits_sha256"],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out["best"] = entries["best"]
    out["epoch40"] = entries["epoch40"]
    out["train_eval"] = json.loads(Path(C.W2_TRAIN_EVAL).read_text(encoding="utf-8")) if Path(C.W2_TRAIN_EVAL).is_file() else None
    tol = 1e-4
    out["verification"] = {
        "tolerance": tol,
        "best_sha_matches": entries["best"]["sha_matches"],
        "epoch40_sha_matches": entries["epoch40"]["sha_matches"],
        "best_ap_reproduced": bool(abs(entries["best"]["ap_delta_vs_recorded"]) <= tol),
        "epoch40_ap_reproduced": bool(abs(entries["epoch40"]["ap_delta_vs_recorded"]) <= tol),
        "best_epoch_is_recorded_epoch": bool(int(entries["best"]["epoch"]) == int(run["best_epoch"])),
    }
    out["verification"]["all_pass"] = bool(all(v for k, v in out["verification"].items() if isinstance(v, bool)))
    out["verified"] = bool(out["verification"]["all_pass"])
    return out


def _run_ablation(task: Any, device: Any) -> dict[str, Any]:
    """Diagnostic-only: trained S3 model with the residual forced to zero."""
    import torch
    import s3_model as M
    import s3_train as ST

    run_dir = C.RUNS / "S3"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "COMPLETED":
        raise RuntimeError("the S3 run is not complete; the ablation needs its selected checkpoint")
    path = run_dir / "checkpoint_best.pt"
    sha = C.sha256_file(path)
    if sha != run["best_checkpoint_sha256"]:
        raise RuntimeError("selected S3 checkpoint SHA-256 does not match its run record")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model, audit = M.build_model("S3", task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    full = ST.evaluate_validation(model, task, device, reason="s3_best_full")
    zero = ST.evaluate_validation(model, task, device, reason="s3_best_residual_zeroed", s3_zero=True)
    pos = [int(x) for x in full["positive"]]
    stats = {}
    for name, a, b in (("full_minus_zeroed", full["per_task_ap"], zero["per_task_ap"]),):
        d = [float(x) - float(y) for x, y in zip(a, b) if x is not None and y is not None]
        s = [p for p, x, y in zip(pos, a, b) if x is not None and y is not None]
        arr = np.asarray(d, dtype=np.float64)
        stats[name] = {
            "n_scorable": int(arr.size),
            "mean_delta": float(arr.mean()) if arr.size else 0.0,
            "median_delta": float(np.median(arr)) if arr.size else 0.0,
            "improved": int((arr > 0).sum()),
            "regressed": int((arr < 0).sum()),
            "ge20_mean_delta": float(arr[np.asarray(s) >= 20].mean()) if arr.size else 0.0,
        }
    branch = M.branch_diagnostics(model, _batch(task, C.BC.SD.make_plan40(task, seed=int(C.SEED)), 64, device))
    return {
        "checkpoint": C.track_rel(path),
        "checkpoint_sha256": sha,
        "epoch": int(payload["epoch"]),
        "role": "diagnostic only: no retraining, no re-selection, not a causal proof",
        "full": {
            "ap": float(full["ap"]),
            "valid_bce": float(full["valid_bce"]),
            "n_tasks_scored": int(full["n_tasks_scored"]),
        },
        "residual_zeroed": {
            "ap": float(zero["ap"]),
            "valid_bce": float(zero["valid_bce"]),
            "n_tasks_scored": int(zero["n_tasks_scored"]),
        },
        "delta_full_minus_zeroed": float(full["ap"]) - float(zero["ap"]),
        "per_task": stats,
        "branch_strength_at_selected_epoch": {
            k: branch[k] for k in ("residual_over_h_raw", "nonzero_fraction", "n_paths", "paths_per_atom") if k in branch
        },
        "test_accessed": False,
    }

if __name__ == "__main__":
    raise SystemExit(main())
