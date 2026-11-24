"""Build the frozen CSS-v0 preparation artifact.

Run once on the compute host before any training.  Produces, per training size:

* the Fit-only sparse vocabulary and SVD projection (``z_v``);
* K = 64 real reference roots, ``tau`` and the precomputed weights ``q_v``;
* the fixed branch normalisation and the shared per-epoch batch plan;
* the label scaler, R2 normalisers and category support masks;
* read-only analysis slices (Valid roots, reference metadata, 8 hash-selected
  Valid examples).

The canonical full-molecule structs (``ses_structure``) are pickled once and
shared by all four training units.  No label, prediction error or test row
influences any fitted statistic.
"""

from __future__ import annotations

import pickle
import platform
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import css_common as C
import css_data as D
import css_descriptors as DD
import css_fit as F
import ses_data as SD
import ses_structure as SS


def _support_summary(struct: SS.Struct, rank: int) -> dict[str, Any]:
    adj, _ = SS._adjacency(struct.n_atoms, struct.bond_u, struct.bond_v)
    dist = SS._bfs(struct.n_atoms, adj, int(rank))
    r2 = np.flatnonzero((dist >= 0) & (dist <= 2))
    bonds = struct.r2_bond[int(struct.r2_bond_ptr[rank]):int(struct.r2_bond_ptr[rank + 1])]
    atom_types = Counter(int(x) for x in struct.atom_feat[r2, 0].tolist())
    bond_types = Counter(int(x) for x in struct.bond_feat[bonds, 0].tolist())
    return {
        "canonical_rank": int(rank),
        "original_atom_id": int(struct.support_atoms[int(rank)]),
        "root_atom_type": int(struct.atom_feat[int(rank), 0]),
        "root_degree": int(struct.degree[int(rank)]),
        "r2_atom_count": int(r2.size),
        "r2_bond_count": int(bonds.size),
        "distance_histogram": {str(d): int((dist == d).sum()) for d in (0, 1, 2)},
        "atom_type_counts": {str(k): int(v) for k, v in sorted(atom_types.items())},
        "bond_type_counts": {str(k): int(v) for k, v in sorted(bond_types.items())},
        "atom_types_in_r2": sorted(int(x) for x in struct.atom_feat[r2, 0].tolist()),
        "bond_types_in_r2": sorted(int(x) for x in struct.bond_feat[bonds, 0].tolist()),
    }


def _example_selection(valid_rows: np.ndarray, structs: Mapping[int, SS.Struct],
                       n_examples: int = 8) -> list[dict[str, int]]:
    rows = sorted(np.asarray(valid_rows, np.int64).tolist(),
                  key=lambda r: C.sha256_bytes(f"css-v0-example/{int(r)}".encode()))
    chosen = []
    for row in rows[:n_examples]:
        n = int(structs[int(row)].n_atoms)
        ranks = range(n)
        rank = min(ranks, key=lambda k: C.sha256_bytes(
            f"css-v0-example-root/{int(row)}/{int(k)}".encode()))
        chosen.append({"row": int(row), "canonical_rank": int(rank), "n_atoms": int(n)})
    return chosen


def main(out_dir: str | None = None) -> dict[str, Any]:
    t_process = time.time()
    topology = C.assert_frozen_sources()
    cfg = C.load_config()
    seed = int(cfg["seed"])
    out_path = Path(out_dir) if out_dir else C.PREP_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "prep.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"host={socket.gethostname()} python={platform.python_version()} "
        f"commit={C.git_commit()}")
    t_data = time.time()
    bundle = SD.load_bundle()
    train = np.sort(np.asarray(bundle.train_rows, np.int64))
    valid = np.sort(np.asarray(bundle.valid_rows, np.int64))
    test = np.sort(np.asarray(bundle.test_rows, np.int64))
    views = {size: D.build_view(bundle, size, train, valid, test) for size in C.SIZES}
    n_mol = int(bundle.labels.size)
    data_seconds = time.time() - t_data
    log(f"data ready: {n_mol} molecules; valid={valid.size} test={test.size} "
        f"fit100={views[100].fit_rows.size} fit50={views[50].fit_rows.size} ({data_seconds:.1f}s)")

    # ---- shared canonical full-molecule structs ---------------------------
    t_struct = time.time()
    structs = {row: SS.canonical_full_struct(bundle.raw, row) for row in range(n_mol)}
    atom_counts = np.asarray([structs[r].n_atoms for r in range(n_mol)], np.int64)
    row_atom_ptr = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)
    n_roots = int(row_atom_ptr[-1])
    cached_atoms = int(np.asarray(bundle.raw["graph_n_atoms"], np.int64).sum())
    if n_roots != cached_atoms:
        raise ValueError(f"root count {n_roots} != cached atom count {cached_atoms}")
    struct_seconds = time.time() - t_struct
    with (out_path / "structs.pkl").open("wb") as handle:
        pickle.dump({"format": "css-v0-full-structs-v1",
                     "raw_cache_sha256": C.RAW_CACHE_SHA256,
                     "structs": structs}, handle, protocol=4)
    struct_size = (out_path / "structs.pkl").stat().st_size
    log(f"structs: {n_roots} atoms, pickle {struct_size / 1e6:.1f} MB ({struct_seconds:.1f}s)")

    # ---- descriptors (label-free, Fit-independent) -----------------------
    t_desc = time.time()
    desc = DD.build_all(bundle.raw, range(n_mol))
    if int(desc["root_row"].size) != n_roots:
        raise ValueError("descriptor root count mismatch")
    if not np.array_equal(np.asarray(desc["root_atom"], np.int64),
                          np.concatenate([np.arange(atom_counts[r]) for r in range(n_mol)])):
        raise ValueError("descriptor root atoms are not molecule-local atom order")
    desc_seconds = time.time() - t_desc
    desc_summary = DD.summary_stats(desc["keys"], desc["counts"], desc["root_ptr"], n_roots)
    log(f"descriptors: {desc_summary['n_entries']} entries / {n_roots} roots "
        f"({desc_summary['entries_per_root_mean']:.1f} per root, {desc_seconds:.1f}s)")

    root_row = np.asarray(desc["root_row"], np.int64)
    root_atom = np.asarray(desc["root_atom"], np.int64)
    canonical_rank = np.zeros(n_roots, np.int64)
    for row in range(n_mol):
        st = structs[row]
        order = np.asarray(st.support_atoms, np.int64)
        pos = np.empty(st.n_atoms, np.int64)
        pos[order] = np.arange(st.n_atoms, dtype=np.int64)
        lo = int(row_atom_ptr[row])
        canonical_rank[lo:lo + st.n_atoms] = pos

    plan_info: dict[str, Any] = {}
    for size in C.SIZES:
        view = views[size]
        plan = F.make_batch_plan(int(view.fit_rows.size), seed=seed)
        np.save(out_path / f"plan_{size}.npy", plan)
        plan_info[size] = {"shape": list(plan.shape),
                           "sha256": C.array_sha256(plan),
                           "epoch0_sha256": C.array_sha256(plan[0])}

    prep: dict[int, dict[str, Any]] = {}
    for size in C.SIZES:
        t_size = time.time()
        view = views[size]
        fit_mask = np.isin(root_row, view.fit_rows)
        frozen, info = F.fit_projection(desc, view.fit_rows, fit_mask, n_roots,
                                        seed=seed, verbose=True)
        projected = frozen.project(desc, n_roots)
        z, q, nearest = projected["z"], projected["q"], projected["nearest"]
        fit_index = np.flatnonzero(fit_mask)
        if C.array_sha256(z[fit_mask]) != info["z_fit_sha256"]:
            raise ValueError(f"size {size}: projection is not reproducible")
        ref_pos = np.asarray(info["ref_positions"], np.int64)
        ref_info = dict(info["reference_selection"])
        ref_root_idx = fit_index[ref_pos]
        z_ref = frozen.z_ref
        tau_info = dict(info["temperature"])
        q_stats = dict(info["q_stats"])
        norm_direct = frozen.norm_direct
        norm_share = frozen.norm_share
        branch_direct, branch_share = F.canonical_branch_tables(
            z, q, norm_direct, norm_share, structs, row_atom_ptr, n_mol)
        projection_vocab = frozen.vocab

        oov = DD.oov_report(desc["keys"], desc["root_ptr"], root_row,
                            projection_vocab,
                            {"fit": view.fit_rows, "valid": view.valid_rows,
                             "test": view.test_rows, "heldout": view.heldout_rows})

        refs = []
        for j, root in enumerate(ref_root_idx.tolist()):
            row = int(root_row[root])
            rank = int(canonical_rank[root])
            st = structs[row]
            refs.append({
                "index": int(j),
                "fit_position": int(ref_pos[j]),
                "global_root": int(root),
                "row": row,
                "canonical_rank": rank,
                "original_atom_id": int(root_atom[root]),
                "smiles": str(np.asarray(bundle.raw["smiles"])[row]),
                "molecule_n_atoms": int(st.n_atoms),
                "z": [float(x) for x in z[root].tolist()],
                "support": _support_summary(st, rank),
            })

        # valid slice for the mechanism analysis (small, pulled locally)
        valid_mask = np.isin(root_row, view.valid_rows)
        np.savez(out_path / f"valid_{size}.npz",
                 root_row=root_row[valid_mask].astype(np.int32),
                 canonical_rank=canonical_rank[valid_mask].astype(np.int32),
                 q=q[valid_mask], z=z[valid_mask], nearest_d2=nearest[valid_mask].astype(np.float32))

        examples_selection = _example_selection(view.valid_rows, structs)
        examples = []
        for item in examples_selection:
            row = int(item["row"])
            rank = int(item["canonical_rank"])
            root = int(row_atom_ptr[row] + rank)
            qrow = q[root]
            top = np.argsort(-qrow)[:3]
            examples.append({
                "row": row,
                "canonical_rank": rank,
                "z_norm": float(np.linalg.norm(z[root])),
                "nearest_d2": float(nearest[root]),
                "n_reference": int(z_ref.shape[0]),
                "smiles": str(np.asarray(bundle.raw["smiles"])[row]),
                "support": _support_summary(structs[row], rank),
                "top_references": [{
                    "rank_in_reference_list": int(j),
                    "weight": float(qrow[int(j)]),
                    "reference": refs[int(j)],
                } for j in top.tolist()],
                "top3_weight_sum": float(qrow[top].sum()),
            })

        np.savez(out_path / f"fit_{size}.npz",
                 branch_DIRECT=branch_direct, branch_SHARE=branch_share,
                 z=z, q=q, nearest_d2=nearest.astype(np.float32),
                 ref_z=z_ref, ref_root_index=ref_root_idx.astype(np.int64),
                 ref_fit_position=ref_pos.astype(np.int64),
                 mean_DIRECT=norm_direct["mean"], mean_SHARE=norm_share["mean"])

        fitstats = {
            "size": int(size),
            "seed": seed,
            "target_scaler": view.scaler,
            "graph_mean": view.graph_mean,
            "graph_scale": view.graph_scale,
            "atom_scalar_mean": view.atom_scalar_mean,
            "atom_scalar_scale": view.atom_scalar_scale,
            "atom_ok": [np.asarray(x, bool) for x in view.atom_ok],
            "bond_ok": [np.asarray(x, bool) for x in view.bond_ok],
            "fit_rows": np.asarray(view.fit_rows, np.int64),
            "heldout_rows": np.asarray(view.heldout_rows, np.int64),
            "valid_rows": np.asarray(view.valid_rows, np.int64),
            "test_rows": np.asarray(view.test_rows, np.int64),
            "branch_norm": {
                "DIRECT": {"mean_sha256": C.array_sha256(norm_direct["mean"]),
                           "rms": float(norm_direct["rms"]),
                           "scale": float(norm_direct["scale"])},
                "SHARE": {"mean_sha256": C.array_sha256(norm_share["mean"]),
                          "rms": float(norm_share["rms"]),
                          "scale": float(norm_share["scale"])},
            },
        }
        with (out_path / f"fitstats_{size}.pkl").open("wb") as handle:
            pickle.dump({"format": "css-v0-fitstats-v1", **fitstats}, handle, protocol=4)

        C.write_json(out_path / f"refs_{size}.json", {
            "size": int(size), "selection": ref_info, "temperature": tau_info,
            "references": refs,
        })
        C.write_json(out_path / f"examples_{size}.json", {
            "size": int(size),
            "selection_rule": "8 official Valid molecules, smallest sha256('css-v0-example/<row_id>'); root = smallest sha256('css-v0-example-root/<row_id>/<rank>')",
            "examples": examples,
        })

        prep[size] = {
            "size": int(size),
            "fit_rows": int(view.fit_rows.size),
            "heldout_rows": int(view.heldout_rows.size),
            "fit_rows_sha256": C.array_sha256(view.fit_rows),
            "heldout_rows_sha256": C.array_sha256(view.heldout_rows),
            "target_scaler": view.scaler,
            "projection": {k: info[k] for k in (
                "n_features", "n_svd_components", "padded_components",
                "effective_components", "singular_values", "explained_variance_ratio",
                "explained_variance_total", "zero_projected_rows", "matrix_stats",
                "vocab_sha256", "components_sha256", "fit_matrix_nnz", "density_fit",
                "fit_matrix_sha256", "z_fit_sha256", "q_fit_sha256")},
            "oov": oov,
            "reference_selection": ref_info,
            "temperature": tau_info,
            "q_stats": {k: v for k, v in q_stats.items() if k != "mean_column_weight"},
            "q_mean_column_weight": q_stats["mean_column_weight"],
            "branch_norm": fitstats["branch_norm"],
            "branch_table_sha256": {
                "DIRECT": C.array_sha256(branch_direct),
                "SHARE": C.array_sha256(branch_share),
            },
            "branch_table_finite": bool(np.isfinite(branch_direct).all()
                                        and np.isfinite(branch_share).all()),
            "seconds": float(time.time() - t_size),
        }
        log(f"size {size}: D={info['n_features']} evr={info['explained_variance_total']:.4f} "
            f"tau={tau_info['tau']:.4f} mean_eff={q_stats['mean_effective_number']:.2f} "
            f"oov_valid={oov['valid']['oov_fraction']:.5f} "
            f"({prep[size]['seconds']:.1f}s)")

    manifest = {
        "format": "css-v0-prep-v1",
        "protocol_id": cfg["protocol_id"],
        "seed": seed,
        "commit": C.git_commit(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": __import__("torch").__version__,
        "cuda": __import__("torch").version.cuda,
        "now_utc": C.now_utc(),
        "config_sha256": C.sha256_json(cfg),
        "code_sha256": C.code_sha([
            "css_common.py", "css_data.py", "css_descriptors.py", "css_fit.py",
            "css_model.py", "css_prepare.py"]),
        "topology": topology,
        "data_hashes": bundle.hashes,
        "data_stats": bundle.stats,
        "n_molecules": n_mol,
        "n_roots": n_roots,
        "official_rows": {"train": C.array_sha256(train), "valid": C.array_sha256(valid),
                          "test": C.array_sha256(test)},
        "descriptor_summary": desc_summary,
        "descriptor_sha256": {"keys": C.array_sha256(desc["keys"]),
                              "counts": C.array_sha256(desc["counts"]),
                              "root_ptr": C.array_sha256(desc["root_ptr"]),
                              "root_row": C.array_sha256(root_row),
                              "root_atom": C.array_sha256(root_atom)},
        "struct_pickle": {"path": str((out_path / "structs.pkl")),
                          "bytes": int(struct_size)},
        "batch_plans": plan_info,
        "sizes": prep,
        "timing_seconds": {
            "data": float(data_seconds), "structs": float(struct_seconds),
            "descriptors": float(desc_seconds),
            "total": float(time.time() - t_process),
        },
        "wall_seconds": float(time.time() - t_process),
    }
    C.write_json(out_path / "prep_manifest.json", manifest)
    log(f"prep done in {manifest['wall_seconds']:.1f}s")
    log_handle.close()
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    payload = main(args.out)
    print(C.jsonable({k: payload[k] for k in ("n_molecules", "n_roots", "timing_seconds")}))
