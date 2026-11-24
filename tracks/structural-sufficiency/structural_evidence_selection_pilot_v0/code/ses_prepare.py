"""Build the shared candidate artifact for SES-v0.

Run once on the compute host before training.  Produces
``results/prepared/ses_prepared.pkl`` containing, for every official row, the
canonical full-graph struct (DENSE) and the ordered list of candidate structs
(RANDOM / SELECT), plus a fixed label-blind random candidate index.
"""

from __future__ import annotations

import pickle
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

import ses_common as C
import ses_data as D
import ses_structure as S


def _connectivity_ok(struct: S.Struct) -> bool:
    if struct.n_atoms == 0:
        return False
    if struct.n_atoms == 1:
        return True
    seen = {0}
    stack = [0]
    adj, _ = S._adjacency(struct.n_atoms, struct.bond_u, struct.bond_v)
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == struct.n_atoms


def main(out: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    topology = C.assert_frozen_sources()
    bundle = D.load_bundle()
    cfg = C.load_config()
    out_path = Path(out) if out else C.PREPARED_DIR / "ses_prepared.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = bundle.raw
    n_mol = int(bundle.labels.size)
    rng = np.random.default_rng(int(cfg["seed"]))
    full_structs: dict[int, S.Struct] = {}
    candidates: dict[int, list[S.Struct]] = {}
    fixed_index = np.zeros(n_mol, np.int64)
    meta_rows: list[dict[str, Any]] = []
    t_build = time.time()
    for row in range(n_mol):
        full_structs[row] = S.canonical_full_struct(raw, row)
        specs, meta = S.generate_candidates(raw, row)
        structs = [S.build_candidate_struct(raw, row, spec) for spec in specs]
        for st in structs:
            if not _connectivity_ok(st):
                raise RuntimeError(f"row {row}: disconnected candidate")
        if not structs:
            raise RuntimeError(f"row {row}: no candidate")
        candidates[row] = structs
        fixed_index[row] = int(rng.integers(0, len(structs)))
        meta = dict(meta)
        meta["candidate_sizes"] = [int(st.n_atoms) for st in structs]
        meta["candidate_edges"] = [int(st.n_bonds) for st in structs]
        meta["source_ranks"] = [int(st.source_rank) for st in structs]
        meta_rows.append(meta)
    build_seconds = time.time() - t_build

    # coverage and target-budget audit, split by official split
    split_of = np.full(n_mol, -1, np.int64)
    split_of[bundle.train_rows] = 0
    split_of[bundle.valid_rows] = 1
    split_of[bundle.test_rows] = 2

    def split_stats(split: int, name: str) -> dict[str, Any]:
        rows = np.flatnonzero(split_of == split)
        cov, edge_cov, n_cands, k_target, comp_rows, below = [], [], [], [], 0, 0
        n_below_candidates = 0
        for row in rows.tolist():
            sts = candidates[row]
            n = full_structs[row].n_atoms
            n_cands.append(len(sts))
            k_target.append(int(sts[0].target_k))
            if any(int(x) < int(sts[0].target_k) for x in [st.n_atoms for st in sts]):
                below += 1
            n_below_candidates += int(sum(1 for st in sts if int(st.n_atoms) < int(sts[0].target_k)))
            for st in sts:
                cov.append(float(st.n_atoms) / float(n))
                edge_cov.append(float(st.n_bonds) / float(max(1, full_structs[row].n_bonds)))
            if meta_rows[row]["n_components"] > 1:
                comp_rows += 1
        cov = np.asarray(cov, np.float64)
        edge_cov = np.asarray(edge_cov, np.float64)
        return {
            "split": name, "n_molecules": int(rows.size),
            "n_candidates": int(sum(n_cands)),
            "candidates_per_molecule_mean": float(np.mean(n_cands)),
            "candidates_per_molecule_min": int(np.min(n_cands)),
            "candidates_per_molecule_max": int(np.max(n_cands)),
            "node_coverage_mean": float(cov.mean()),
            "node_coverage_min": float(cov.min()),
            "node_coverage_p01": float(np.percentile(cov, 1)),
            "edge_coverage_mean": float(edge_cov.mean()),
            "edge_coverage_min": float(edge_cov.min()),
            "molecules_with_candidate_below_k": int(below),
            "candidates_below_k": int(n_below_candidates),
            "multicomponent_molecules": int(comp_rows),
            "target_k_mean": float(np.mean(k_target)),
        }

    stats = {
        "host": socket.gethostname(),
        "build_seconds": float(build_seconds),
        "topology": topology,
        "n_molecules": int(n_mol),
        "n_full_structs": int(len(full_structs)),
        "n_candidates": int(sum(len(v) for v in candidates.values())),
        "candidate_count_histogram": {
            str(k): int(sum(1 for m in meta_rows if m["n_candidates"] == k))
            for k in sorted({m["n_candidates"] for m in meta_rows})
        },
        "unique_candidate_count_histogram": {
            str(k): int(sum(1 for m in meta_rows if m["n_unique_candidates"] == k))
            for k in sorted({m["n_unique_candidates"] for m in meta_rows})
        },
        "n_atoms_one_rows": int(sum(1 for m in meta_rows if m["n_atoms_special"])),
        "multicomponent_molecules": int(sum(1 for m in meta_rows
                                               if m["n_components"] > 1)),
        "rows_with_component_below_k": int(sum(1 for m in meta_rows
                                                if m["any_component_smaller_than_k"])),
        "candidates_below_k": int(sum(m["candidate_sizes"].count(x) if False else
                                      sum(1 for st in candidates[m["row"]] if
                                          int(st.n_atoms) < int(st.target_k))
                                      for m in meta_rows)),
        "splits": {name: split_stats(i, name) for i, name in enumerate(("train", "valid", "test"))},
        "fixed_index": {"source": "numpy default_rng(seed=0), label-blind, per row in 0..N-1 order",
                        "sha256": C.array_sha256(fixed_index)},
        "raw_cache_sha256": bundle.hashes["raw_cache"]["sha256"],
        "split_identity_sha256": bundle.hashes["split_identity"],
    }
    C.write_json(C.ANALYSIS_DIR / "candidate_stats.json", stats)

    payload = {
        "format": "ses-v0-prepared-v1",
        "commit": C.git_commit(),
        "raw_cache_sha256": C.RAW_CACHE_SHA256,
        "split_identity_sha256": bundle.hashes["split_identity"],
        "config": cfg,
        "full_structs": full_structs,
        "candidates": candidates,
        "fixed_index": fixed_index,
        "meta_rows": meta_rows,
        "stats": stats,
    }
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    tmp.replace(out_path)
    payload["prepared_path"] = str(out_path)
    payload["size_bytes"] = int(out_path.stat().st_size)
    payload.pop("full_structs")
    payload.pop("candidates")
    payload.pop("meta_rows")
    C.write_json(C.ANALYSIS_DIR / "prepare_manifest.json", payload)
    print(f"[prepare] {stats['n_molecules']} molecules, "
          f"{stats['n_candidates']} candidates, {build_seconds:.1f}s, "
          f"{out_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.out)
