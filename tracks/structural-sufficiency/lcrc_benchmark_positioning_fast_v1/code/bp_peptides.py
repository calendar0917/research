"""LRGB Peptides-func loading and label-free LCRC input construction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import bp_common as C
import bp_data as D
import bp_graph as BG
import bp_spec as S
import lcrc_data as PREP_DATA
import molecular_graph as MG
from feature_spec import save_spec


class InputAdapterBlocked(RuntimeError):
    pass


SPLITS = ("train", "valid", "test")


def _load_split(split: str) -> dict[str, Any]:
    from torch_geometric.datasets import LRGBDataset

    loader_split = "val" if split == "valid" else split
    ds = LRGBDataset(root=str(C.LRGB_ROOT), name=C.PEPTIDE_DATASET, split=loader_split)
    records = []
    clean = {"n_directed_records": 0, "n_self_loops_dropped": 0, "n_duplicate_records_merged": 0}
    for gid in range(len(ds)):
        data = ds[gid]
        n = int(data.num_nodes)
        if n < 1:
            raise InputAdapterBlocked(f"peptides {split}[{gid}]: empty graph")
        x = np.asarray(data.x.detach().cpu().numpy())
        ea = np.asarray(data.edge_attr.detach().cpu().numpy())
        ei = np.asarray(data.edge_index.detach().cpu().numpy())
        if x.shape != (n, len(S.PEPTIDE_ATOM_VOCABS)):
            raise InputAdapterBlocked(f"peptides {split}[{gid}]: x shape {x.shape}")
        if ea.shape[1] != len(S.PEPTIDE_BOND_VOCABS):
            raise InputAdapterBlocked(f"peptides {split}[{gid}]: edge_attr shape {ea.shape}")
        if not np.allclose(x, np.round(x)):
            raise InputAdapterBlocked(f"peptides {split}[{gid}]: non-integer atom features")
        if not np.allclose(ea, np.round(ea)):
            raise InputAdapterBlocked(f"peptides {split}[{gid}]: non-integer bond features")
        atom_raw = np.round(x).astype(np.int64)
        bond_raw = np.round(ea).astype(np.int64)
        seen: dict[tuple[int, int], tuple[int, ...]] = {}
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u == v:
                clean["n_self_loops_dropped"] += 1
                continue
            key = (min(u, v), max(u, v))
            val = tuple(int(t) for t in bond_raw[k].tolist())
            if key in seen:
                clean["n_duplicate_records_merged"] += 1
                if seen[key] != val:
                    raise InputAdapterBlocked(
                        f"peptides {split}[{gid}]: reverse/duplicate directed records disagree for {key}"
                    )
            else:
                seen[key] = val
            clean["n_directed_records"] += 1
        keys = sorted(seen)
        pairs = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        bond_feat_raw = np.asarray([seen[k] for k in keys], dtype=np.int64).reshape(-1, len(S.PEPTIDE_BOND_VOCABS))
        y = np.asarray(data.y.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        if y.shape != (10,):
            raise InputAdapterBlocked(f"peptides {split}[{gid}]: y shape {y.shape}")
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise InputAdapterBlocked(f"peptides {split}[{gid}]: target outside [0,1]")
        records.append(
            {
                "gid": gid,
                "n_atoms": n,
                "atom_raw": atom_raw,
                "pairs": pairs,
                "bond_raw": bond_feat_raw,
                "y": y,
            }
        )
    y_all = np.stack([r["y"] for r in records], axis=0)
    return {
        "split": split,
        "n_graphs": len(records),
        "records": records,
        "y": y_all,
        "clean": clean,
        "min_nodes": int(min(r["n_atoms"] for r in records)),
        "max_nodes": int(max(r["n_atoms"] for r in records)),
        "n_nodes_total": int(sum(r["n_atoms"] for r in records)),
    }


def _build_molecule(record: Mapping[str, Any], spec) -> tuple[dict[str, Any], dict[str, Any]]:
    n = int(record["n_atoms"])
    atom_raw = np.asarray(record["atom_raw"], dtype=np.int64)
    pairs = np.asarray(record["pairs"], dtype=np.int64).reshape(-1, 2)
    bond_raw = np.asarray(record["bond_raw"], dtype=np.int64).reshape(-1, len(S.PEPTIDE_BOND_VOCABS))
    m = int(pairs.shape[0])
    atom_type = atom_raw[:, 0]
    bond_type = bond_raw[:, 0] + 1
    if np.any(atom_type < 0) or np.any(atom_type >= spec.n_atom_types):
        raise InputAdapterBlocked("peptides: atom type outside [0,119)")
    if m and (np.any(bond_type < 1) or np.any(bond_type > spec.n_bond_types)):
        raise InputAdapterBlocked("peptides: bond type outside [1,5]")
    placeholder = MG.build_molecular_graph(
        atom_type=atom_type,
        bond_index=pairs,
        bond_type=bond_type if m else np.zeros(0, dtype=np.int64),
        atom_feat=np.zeros((n, spec.atom_feat_dim), dtype=np.float32),
        bond_feat=np.zeros((m, spec.bond_feat_dim), dtype=np.float32),
        molecule_id=f"peptides#{record['gid']}",
    )
    analysis = BG.analyse_graph_fast(placeholder)
    atom_feat, bond_feat = S.peptide_features(atom_raw, bond_raw, atom_type, bond_type, analysis)
    s1_ptr, s1_code = BG.s1_arrays(atom_type, analysis.adj, spec.n_bond_types)
    s2 = BG.s2_arrays(atom_type, analysis.adj, spec.n_bond_types)
    per_root = np.diff(s2["path_ptr"]).astype(np.int64)
    closure_counts = np.zeros(n, dtype=np.int64)
    if int(s2["path_ptr"][-1]):
        closure_counts = np.bincount(
            np.repeat(np.arange(n, dtype=np.int64), per_root),
            weights=np.asarray(s2["path_closure"], dtype=np.float64),
            minlength=n,
        ).astype(np.int64)
    mol = {
        "n_atoms": np.int64(n),
        "n_bonds": np.int64(m),
        "atom_feat": atom_feat,
        "atom_type": atom_type,
        "atom_scalar": S.scalar_features(analysis, per_root, closure_counts),
        "bond_feat": bond_feat,
        "bond_type": bond_type,
        "bond_index": np.stack([pairs[:, 0], pairs[:, 1]], axis=1).astype(np.int64)
        if m
        else np.zeros((0, 2), dtype=np.int64),
        "nbr_ptr": s1_ptr,
        "nbr_code": s1_code,
        "path_ptr": s2["path_ptr"],
        "path_cat": s2["path_cat"],
        "path_end63": s2["path_end63"],
        "path_end": s2["path_end"],
        "path_closure": s2["path_closure"],
        "global": S.global_features(analysis, atom_type),
        "n_components": np.int64(analysis.n_components),
        "n_junctions": np.int64(analysis.n_junctions),
        "n_ring_systems": np.int64(analysis.n_ring_systems),
        "molecule_id": f"peptides#{record['gid']}",
        "smiles": "",
    }
    PREP_DATA.validate_molecule(mol, spec)
    return mol, {
        "gid": int(record["gid"]),
        "n_atoms": int(n),
        "n_bonds": int(m),
        "s1_codes": int(s1_ptr[-1]),
        "s2_paths": int(s2["path_ptr"][-1]),
    }


def build(*, force: bool = False) -> dict[str, Any]:
    spec = S.peptide_spec(
        source_provenance={
            "dataset": "LRGB Peptides-func",
            "loader": "torch_geometric.datasets.LRGBDataset(root, name='Peptides-func', split=...)",
            "atom_vocabs": list(S.PEPTIDE_ATOM_VOCABS),
            "bond_vocabs": list(S.PEPTIDE_BOND_VOCABS),
            "s1_s2_type_sources": "atom feature column 0 and bond feature column 0",
        }
    )
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    spec_path = C.ANALYSIS / "peptides_func_spec.json"
    save_spec(spec, spec_path)
    out: dict[str, Any] = {"dataset": C.PEPTIDE_DATASET, "feature_spec_hash": spec.hash(), "splits": {}}
    for split in SPLITS:
        path = C.CACHE / f"peptides_func_{split}.pt"
        stats_path = C.ANALYSIS / f"peptides_func_{split}_cache_stats.json"
        if not force and path.is_file() and stats_path.is_file() and (C.ANALYSIS / f"peptides_func_y_{split}.npy").is_file():
            cached = D.load_cache(path)
            if str(cached["meta"].get("feature_spec_hash")) == spec.hash():
                out["splits"][split] = C.read_json(stats_path)
                C.log(f"[cache peptides/{split}] reuse {C.track_rel(path)}")
                continue
        t0 = time.perf_counter()
        ds = _load_split(split)
        mols = []
        per_graph = []
        for i, rec in enumerate(ds["records"]):
            mol, pg = _build_molecule(rec, spec)
            mols.append(mol)
            per_graph.append(pg)
            if (i + 1) % 2000 == 0:
                C.log(f"[cache peptides/{split}] {i + 1}/{len(ds['records'])} graphs {time.perf_counter() - t0:.1f}s")
        cache = PREP_DATA.concat_molecules(mols, spec, split=split)
        cache["gids"] = np.arange(len(mols), dtype=np.int64)
        cache["meta"]["build_seconds"] = float(time.perf_counter() - t0)
        PREP_DATA.validate_cache(cache, spec)
        D.save_cache(cache, path)
        np.save(C.ANALYSIS / f"peptides_func_y_{split}.npy", ds["y"])
        arr_n = np.asarray([r["n_atoms"] for r in per_graph], dtype=np.int64)
        arr_m = np.asarray([r["n_bonds"] for r in per_graph], dtype=np.int64)
        arr_s = np.asarray([r["s2_paths"] for r in per_graph], dtype=np.int64)
        np.savez(
            C.ANALYSIS / f"peptides_func_{split}_graph_distribution.npz",
            gids=np.asarray([r["gid"] for r in per_graph], dtype=np.int64),
            n_atoms=arr_n,
            n_bonds=arr_m,
            s2_paths=arr_s,
        )
        out["splits"][split] = {
            "cache": C.track_rel(path),
            "cache_sha256": C.sha256_file(path),
            "cache_bytes": int(path.stat().st_size),
            "build_seconds": float(cache["meta"]["build_seconds"]),
            "n_graphs": int(ds["n_graphs"]),
            "n_nodes_total": int(ds["n_nodes_total"]),
            "min_nodes": int(ds["min_nodes"]),
            "max_nodes": int(ds["max_nodes"]),
            "clean": ds["clean"],
            "n_atoms": _dist(arr_n),
            "n_bonds": _dist(arr_m),
            "s2_paths": _dist(arr_s),
            "s2_paths_total": int(arr_s.sum()),
            "labels_sha256": C.sha256_file(C.ANALYSIS / f"peptides_func_y_{split}.npy"),
            "cache_validation": PREP_DATA.validate_cache(cache, spec),
        }
        C.write_json(C.ANALYSIS / f"peptides_func_{split}_cache_stats.json", out["splits"][split])
    C.write_json(C.ANALYSIS / "peptides_func_cache_stats.json", out)
    return out


def _dist(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": int(a.min()) if a.size else 0,
        "max": int(a.max()) if a.size else 0,
        "mean": float(a.mean()) if a.size else 0.0,
        "p50": float(np.percentile(a, 50)) if a.size else 0.0,
        "p95": float(np.percentile(a, 95)) if a.size else 0.0,
        "p99": float(np.percentile(a, 99)) if a.size else 0.0,
        "total": int(a.sum()) if a.size else 0,
    }


__all__ = ["InputAdapterBlocked", "build"]
