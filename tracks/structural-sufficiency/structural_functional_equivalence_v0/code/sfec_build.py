"""Parallel structural build: cuts, exact keys, chunk files, merge.

Workers never load labels.  Each chunk is written atomically and can be skipped
on a re-run, so extending the PCQM prefix never re-processes earlier molecules.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import sfec_common as C
import sfec_pcqm as P
import sfec_zinc as Z

CHUNK_SIZE_PCQM = 2000
CHUNK_SIZE_ZINC = 500


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------


def _intern(mapping: dict, keys: list, value: str) -> int:
    idx = mapping.get(value)
    if idx is None:
        idx = len(keys)
        mapping[value] = idx
        keys.append(value)
    return idx


def process_pcqm_chunk(task: dict[str, Any]) -> dict[str, Any]:
    dataset = task["dataset"]
    start = int(task["start"])
    rows = task["rows"]
    max_small = int(task["max_small_side"])
    min_context = int(task["min_context"])
    min_ratio = float(task["min_ratio"])
    out_dir = Path(task["out_dir"])
    sample_every = int(task["reassembly_sample_every"])
    sample_rows = int(task["reassembly_sample_rows"])
    t0 = time.time()

    C_map: dict[str, int] = {}
    C_keys: list[str] = []
    C_meta: list[dict[str, Any]] = []
    S_map: dict[str, int] = {}
    S_keys: list[str] = []
    S_meta: list[dict[str, Any]] = []
    mol_keys: list[str] = []
    mol_meta: list[dict[str, Any]] = []
    mol_map: dict[str, int] = {}

    cols = {k: [] for k in (
        "row_pos", "row_id", "mol_local", "cut_id", "bond_type",
        "C_local", "S_local", "nS", "nC", "reasm_graph", "reasm_stereo",
    )}
    stats = {
        "chunk_start": start,
        "chunk_end": start + len(rows),
        "n_rows": len(rows),
        "n_parse_failed": 0,
        "n_multi_component": 0,
        "n_molecules_used": 0,
        "n_bridges": 0,
        "n_single_bridges": 0,
        "n_cuts": 0,
        "n_reassembly_checked": 0,
        "n_reassembly_graph_bad": 0,
        "n_reassembly_stereo_bad": 0,
    }
    for local_i, (pos, row_id, smiles) in enumerate(rows):
        check = (pos < sample_rows) or (pos % sample_every == 0)
        try:
            rec = P.decompose(
                smiles,
                max_small_side=max_small,
                min_context=min_context,
                min_ratio=min_ratio,
                do_reassembly=check,
            )
        except Exception as exc:  # pragma: no cover - defensive
            stats["n_parse_failed"] += 1
            stats.setdefault("errors", []).append({"pos": int(pos), "error": repr(exc)})
            continue
        if rec is None:
            stats["n_parse_failed"] += 1
            continue
        st = rec["stats"]
        stats["n_bridges"] += st["n_bridges"]
        stats["n_single_bridges"] += st["n_single_bridges"]
        if st["multi_component"]:
            stats["n_multi_component"] += 1
            continue
        mk = rec["mol_key"]
        mid = mol_map.get(mk)
        if mid is None:
            mid = len(mol_keys)
            mol_map[mk] = mid
            mol_keys.append(mk)
            mol_meta.append(
                {
                    "mol_key": mk,
                    "row_pos": int(pos),
                    "row_id": int(row_id),
                    "n_atoms": int(st["n_atoms"]),
                    "n_bonds": int(st["n_bonds"]),
                    "n_components": int(st["n_components"]),
                    "mol_scaffold": rec.get("mol_scaffold", ""),
                }
            )
        stats["n_molecules_used"] += 1
        for cut in rec["cuts"]:
            cid = _intern(C_map, C_keys, cut["C_key"])
            sid = _intern(S_map, S_keys, cut["S_key"])
            # metadata list is parallel to the local key list
            if len(C_meta) < len(C_keys):
                C_meta.extend([None] * (len(C_keys) - len(C_meta)))
                C_meta[cid] = {
                    "q": cut["q"],
                    "nC": int(cut["nC"]),
                    "scaffold": cut.get("C_scaffold", ""),
                }
            if len(S_meta) < len(S_keys):
                S_meta.extend([None] * (len(S_keys) - len(S_meta)))
                S_meta[sid] = {"nS": int(cut["nS"])}
            cols["row_pos"].append(pos)
            cols["row_id"].append(row_id)
            cols["mol_local"].append(mid)
            cols["cut_id"].append(cut["cut_id"])
            cols["bond_type"].append(cut["bond_type"])
            cols["C_local"].append(cid)
            cols["S_local"].append(sid)
            cols["nS"].append(cut["nS"])
            cols["nC"].append(cut["nC"])
            rg = cut.get("reassembly_graph_ok")
            rs = cut.get("reassembly_stereo_ok")
            cols["reasm_graph"].append(-1 if rg is None else int(bool(rg)))
            cols["reasm_stereo"].append(-1 if rs is None else int(bool(rs)))
            if rg is not None:
                stats["n_reassembly_checked"] += 1
                stats["n_reassembly_graph_bad"] += int(not rg)
                stats["n_reassembly_stereo_bad"] += int(not rs)
            stats["n_cuts"] += 1

    arrays = {}
    for key, values in cols.items():
        if key in ("row_pos", "row_id", "mol_local", "C_local", "S_local"):
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key in ("cut_id", "nC"):
            arrays[key] = np.asarray(values, dtype=np.int16)
        elif key == "nS":
            arrays[key] = np.asarray(values, dtype=np.int8)
        elif key == "bond_type":
            arrays[key] = np.asarray(values, dtype=np.int8)
        else:
            arrays[key] = np.asarray(values, dtype=np.int8)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"pcqm_{dataset}_{start:08d}"
    tmp = stem.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(stem.with_suffix(".npz"))
    meta = {
        "dataset": dataset,
        "split": dataset,
        "start": start,
        "n_rows": len(rows),
        "C_keys": C_keys,
        "C_meta": C_meta,
        "S_keys": S_keys,
        "S_meta": S_meta,
        "mol_keys": mol_keys,
        "mol_meta": mol_meta,
        "stats": stats,
        "elapsed_s": round(time.time() - t0, 3),
    }
    tmp = stem.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(stem.with_suffix(".json"))
    gc.collect()
    return {"stem": str(stem), "stats": stats, "elapsed_s": meta["elapsed_s"]}


def process_zinc_chunk(task: dict[str, Any]) -> dict[str, Any]:
    dataset = task["dataset"]
    split = task["split"]
    start = int(task["start"])
    rows = task["rows"]
    max_small = int(task["max_small_side"])
    min_context = int(task["min_context"])
    min_ratio = float(task["min_ratio"])
    out_dir = Path(task["out_dir"])
    t0 = time.time()
    C_map: dict[str, int] = {}
    C_keys: list[str] = []
    C_meta: list[dict[str, Any]] = []
    S_map: dict[str, int] = {}
    S_keys: list[str] = []
    S_meta: list[dict[str, Any]] = []
    mol_keys: list[str] = []
    mol_meta: list[dict[str, Any]] = []
    mol_map: dict[str, int] = {}
    cols = {k: [] for k in (
        "row_id", "mol_local", "cut_id", "bond_type", "C_local", "S_local",
        "nS", "nC",
    )}
    stats = {
        "split": split,
        "chunk_start": start,
        "n_rows": len(rows),
        "n_parse_failed": 0,
        "n_multi_component": 0,
        "n_molecules_used": 0,
        "n_bridges": 0,
        "n_single_bridges": 0,
        "n_cuts": 0,
    }
    for rec in rows:
        if rec.get("skip"):
            stats["n_parse_failed"] += 1
            continue
        out = Z.decompose(
            rec["x"], rec["edge_index"], rec["edge_attr"],
            single_bond_code=1,
            max_small_side=max_small,
            min_context=min_context,
            min_ratio=min_ratio,
        )
        st = out["stats"]
        stats["n_bridges"] += st["n_bridges"]
        stats["n_single_bridges"] += st["n_single_bridges"]
        if st["multi_component"]:
            stats["n_multi_component"] += 1
            continue
        mkey = out["mol_key"]
        mid = mol_map.get(mkey)
        if mid is None:
            mid = len(mol_keys)
            mol_map[mkey] = mid
            mol_keys.append(mkey)
            mol_meta.append(
                {
                    "mol_key": mkey,
                    "row_id": int(rec["row_id"]),
                    "n_atoms": int(st["n_atoms"]),
                    "n_bonds": int(st["n_bonds"]),
                    "n_components": int(st["n_components"]),
                    "mol_scaffold": "",
                }
            )
        stats["n_molecules_used"] += 1
        for cut in out["cuts"]:
            cid = _intern(C_map, C_keys, cut["C_key"])
            sid = _intern(S_map, S_keys, cut["S_key"])
            if len(C_meta) < len(C_keys):
                C_meta.extend([None] * (len(C_keys) - len(C_meta)))
                C_meta[cid] = {"q": cut["q"], "nC": int(cut["nC"]), "scaffold": ""}
            if len(S_meta) < len(S_keys):
                S_meta.extend([None] * (len(S_keys) - len(S_meta)))
                S_meta[sid] = {"nS": int(cut["nS"])}
            cols["row_id"].append(rec["row_id"])
            cols["mol_local"].append(mid)
            cols["cut_id"].append(cut["cut_id"])
            cols["bond_type"].append(cut["bond_type"])
            cols["C_local"].append(cid)
            cols["S_local"].append(sid)
            cols["nS"].append(cut["nS"])
            cols["nC"].append(cut["nC"])
            stats["n_cuts"] += 1
    arrays = {}
    for key, values in cols.items():
        if key in ("row_id", "mol_local", "C_local", "S_local"):
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key in ("cut_id", "nC"):
            arrays[key] = np.asarray(values, dtype=np.int16)
        elif key == "nS":
            arrays[key] = np.asarray(values, dtype=np.int8)
        elif key == "bond_type":
            arrays[key] = np.asarray(values, dtype=np.int8)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"zinc_{split}_{start:08d}"
    tmp = stem.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(stem.with_suffix(".npz"))
    meta = {
        "dataset": "zinc",
        "split": split,
        "start": start,
        "n_rows": len(rows),
        "C_keys": C_keys,
        "C_meta": C_meta,
        "S_keys": S_keys,
        "S_meta": S_meta,
        "mol_keys": mol_keys,
        "mol_meta": mol_meta,
        "stats": stats,
        "elapsed_s": round(time.time() - t0, 3),
    }
    tmp = stem.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(stem.with_suffix(".json"))
    gc.collect()
    return {"stem": str(stem), "stats": stats, "elapsed_s": meta["elapsed_s"]}


# ---------------------------------------------------------------------------
# task generation
# ---------------------------------------------------------------------------


def iter_pcqm_tasks(
    dataset: str,
    cache: Path,
    start: int,
    end: int,
    out_dir: Path,
    params: dict[str, Any],
    chunk_size: int = CHUNK_SIZE_PCQM,
) -> Iterable[dict[str, Any]]:
    gids = C.pcqm_cache_gids(cache)
    buf: list[tuple[int, int, str]] = []
    buf_start = start
    with (cache / "smiles.jsonl").open(encoding="utf-8") as handle:
        for pos, line in enumerate(handle):
            if pos < start:
                continue
            if pos >= end:
                break
            if not buf:
                buf_start = pos
            buf.append((pos, int(gids[pos]), json.loads(line)))
            if len(buf) >= chunk_size:
                yield {
                    "dataset": dataset,
                    "start": buf_start,
                    "rows": buf,
                    "out_dir": str(out_dir),
                    **params,
                }
                buf = []
    if buf:
        yield {
            "dataset": dataset,
            "start": buf_start,
            "rows": buf,
            "out_dir": str(out_dir),
            **params,
        }


def iter_zinc_tasks(
    split: str,
    records: list[dict[str, Any]],
    out_dir: Path,
    params: dict[str, Any],
    chunk_size: int = CHUNK_SIZE_ZINC,
) -> Iterable[dict[str, Any]]:
    for start in range(0, len(records), chunk_size):
        yield {
            "dataset": split,
            "split": split,
            "start": start,
            "rows": records[start:start + chunk_size],
            "out_dir": str(out_dir),
            **params,
        }


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def merge_chunks(
    stems: Sequence[str],
    out_prefix: Path,
    *,
    splits: Sequence[str],
    split_codes: dict[str, int],
) -> dict[str, Any]:
    """Merge chunk files into global ids, key tables and a cell table."""
    C_map: dict[bytes, int] = {}
    C_keys: list[str] = []
    C_q_map: dict[str, int] = {}
    C_q_keys: list[str] = []
    C_qid: list[int] = []
    C_scaf_map: dict[str, int] = {}
    C_scaf_keys: list[str] = []
    C_scafid: list[int] = []
    C_nC: list[int] = []
    C_hash: list[int] = []
    C_hash64: list[int] = []
    S_map: dict[bytes, int] = {}
    S_keys: list[str] = []
    S_n: list[int] = []
    mol_map: dict[bytes, int] = {}
    mol_keys: list[str] = []
    mol_split: list[int] = []
    mol_row: list[int] = []
    mol_n: list[int] = []
    mol_scaf_map: dict[str, int] = {}
    mol_scaf_keys: list[str] = []
    mol_scafid: list[int] = []

    cols: dict[str, list[np.ndarray]] = {k: [] for k in (
        "row_id", "mol_id", "cut_id", "bond_type", "C_id", "S_id",
        "nS", "nC", "split_id", "reasm_graph", "reasm_stereo",
    )}
    total_stats: dict[str, Any] = {}
    for stem in stems:
        stem = Path(stem)
        npz = np.load(stem.with_suffix(".npz"))
        meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        split = str(meta.get("split", "train"))
        if split not in split_codes:
            raise KeyError(f"unknown split {split}")
        split_code = int(split_codes[split])
        local_C = np.asarray(npz["C_local"], dtype=np.int64)
        mapC = np.empty(len(meta["C_keys"]), dtype=np.int32)
        for i, key in enumerate(meta["C_keys"]):
            d = _digest(key)
            gid = C_map.get(d)
            if gid is None:
                gid = len(C_keys)
                C_map[d] = gid
                C_keys.append(key)
                m = meta["C_meta"][i]
                q = m["q"]
                qi = C_q_map.get(q)
                if qi is None:
                    qi = len(C_q_keys)
                    C_q_map[q] = qi
                    C_q_keys.append(q)
                C_qid.append(qi)
                sc = m.get("scaffold", "") or ""
                si = C_scaf_map.get(sc)
                if si is None:
                    si = len(C_scaf_keys)
                    C_scaf_map[sc] = si
                    C_scaf_keys.append(sc)
                C_scafid.append(si)
                C_nC.append(int(m["nC"]))
                hh = C.context_hash_hex(key, 0)
                C_hash.append(int(hh[:16], 16) % 100)
                C_hash64.append(int(hh[:16], 16))
            elif C_keys[gid] != key:
                raise RuntimeError("C key hash collision")
            mapC[i] = gid
        local_S = np.asarray(npz["S_local"], dtype=np.int64)
        mapS = np.empty(len(meta["S_keys"]), dtype=np.int32)
        for i, key in enumerate(meta["S_keys"]):
            d = _digest(key)
            gid = S_map.get(d)
            if gid is None:
                gid = len(S_keys)
                S_map[d] = gid
                S_keys.append(key)
                S_n.append(int(meta["S_meta"][i]["nS"]))
            elif S_keys[gid] != key:
                raise RuntimeError("S key hash collision")
            mapS[i] = gid
        local_mol = np.asarray(npz["mol_local"], dtype=np.int64)
        mapM = np.empty(len(meta["mol_keys"]), dtype=np.int32)
        for i, key in enumerate(meta["mol_keys"]):
            d = _digest(key)
            gid = mol_map.get(d)
            if gid is None:
                gid = len(mol_keys)
                mol_map[d] = gid
                mol_keys.append(key)
                mol_split.append(split_code)
                mol_row.append(int(meta["mol_meta"][i]["row_id"]))
                mol_n.append(int(meta["mol_meta"][i]["n_atoms"]))
                sc = meta["mol_meta"][i].get("mol_scaffold", "") or ""
                si = mol_scaf_map.get(sc)
                if si is None:
                    si = len(mol_scaf_keys)
                    mol_scaf_map[sc] = si
                    mol_scaf_keys.append(sc)
                mol_scafid.append(si)
            elif mol_keys[gid] != key:
                raise RuntimeError("molecule key hash collision")
            mapM[i] = gid
        cols["row_id"].append(np.asarray(npz["row_id"], dtype=np.int64))
        cols["mol_id"].append(mapM[local_mol])
        cols["cut_id"].append(np.asarray(npz["cut_id"], dtype=np.int16))
        cols["bond_type"].append(np.asarray(npz["bond_type"], dtype=np.int8))
        cols["C_id"].append(mapC[local_C])
        cols["S_id"].append(mapS[local_S])
        cols["nS"].append(np.asarray(npz["nS"], dtype=np.int8))
        cols["nC"].append(np.asarray(npz["nC"], dtype=np.int16))
        cols["split_id"].append(np.full(local_C.size, split_code, dtype=np.int8))
        if "reasm_graph" in npz:
            cols["reasm_graph"].append(np.asarray(npz["reasm_graph"], dtype=np.int8))
            cols["reasm_stereo"].append(np.asarray(npz["reasm_stereo"], dtype=np.int8))
        else:
            cols["reasm_graph"].append(np.full(local_C.size, -1, dtype=np.int8))
            cols["reasm_stereo"].append(np.full(local_C.size, -1, dtype=np.int8))
        st = meta.get("stats", {})
        for k, v in st.items():
            if k in ("chunk_start", "chunk_end"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total_stats[k] = total_stats.get(k, 0) + v
        npz.close()
    cells = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_prefix.with_suffix(".npz"), **cells)
    import pyarrow as pa
    import pyarrow.parquet as pq

    C_table = pa.table(
        {
            "C_id": np.arange(len(C_keys), dtype=np.int32),
            "C_key": pa.array(C_keys, type=pa.string()),
            "q_id": np.asarray(C_qid, dtype=np.int32),
            "nC": np.asarray(C_nC, dtype=np.int16),
            "scaffold_id": np.asarray(C_scafid, dtype=np.int32),
            "hash_mod": np.asarray(C_hash, dtype=np.int16),
            "hash64": np.asarray(C_hash64, dtype=np.uint64),
        }
    )
    pq.write_table(C_table, out_prefix.with_suffix(".C.parquet"), compression="zstd")
    S_table = pa.table(
        {
            "S_id": np.arange(len(S_keys), dtype=np.int32),
            "S_key": pa.array(S_keys, type=pa.string()),
            "nS": np.asarray(S_n, dtype=np.int8),
        }
    )
    pq.write_table(S_table, out_prefix.with_suffix(".S.parquet"), compression="zstd")
    M_table = pa.table(
        {
            "mol_id": np.arange(len(mol_keys), dtype=np.int32),
            "mol_key": pa.array(mol_keys, type=pa.string()),
            "split_id": np.asarray(mol_split, dtype=np.int8),
            "row_id": np.asarray(mol_row, dtype=np.int64),
            "n_atoms": np.asarray(mol_n, dtype=np.int16),
            "scaffold_id": np.asarray(mol_scafid, dtype=np.int32),
        }
    )
    pq.write_table(M_table, out_prefix.with_suffix(".M.parquet"), compression="zstd")
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    C.write_json(
        out_prefix.with_suffix(".keys.json"),
        {
            "q_keys": C_q_keys,
            "scaffold_keys": C_scaf_keys,
            "mol_scaffold_keys": mol_scaf_keys,
        },
    )
    return {
        "n_cells": int(cells["C_id"].size),
        "n_C": len(C_keys),
        "n_S": len(S_keys),
        "n_molecules": len(mol_keys),
        "n_q": len(C_q_keys),
        "n_scaffolds": len(C_scaf_keys),
        "total_stats": total_stats,
    }


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def run_tasks(tasks: Iterable[dict[str, Any]], worker, workers: int, label: str) -> list[dict[str, Any]]:
    import multiprocessing as mp

    todo = []
    for task in tasks:
        if task["out_dir"]:
            fname = Path(task["out_dir"]) / _stem_name(task)
            if fname.with_suffix(".npz").exists() and fname.with_suffix(".json").exists():
                continue
        todo.append(task)
    if not todo:
        print(f"[{label}] all chunks present")
        return []
    print(f"[{label}] {len(todo)} chunks, {workers} workers", flush=True)
    ctx = mp.get_context("fork")
    results = []
    t0 = time.time()
    with ctx.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap(worker, todo, chunksize=1), 1):
            results.append(res)
            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - t0
                print(
                    f"[{label}] {i}/{len(todo)} chunks in {elapsed:.1f}s "
                    f"({i/elapsed:.2f} chunks/s) last={res['stats'].get('n_cuts')} cuts",
                    flush=True,
                )
    return results


def _stem_name(task: dict[str, Any]) -> str:
    if task.get("dataset") in ("train", "valid", "test"):
        return f"pcqm_{task['dataset']}_{int(task['start']):08d}"
    return f"zinc_{task['split']}_{int(task['start']):08d}"


def list_chunks(pattern: str) -> list[Path]:
    return sorted(C.CHUNK_DIR.glob(pattern))


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pcqm-build")
    p.add_argument("--dataset", default="train", choices=["train", "valid", "test"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_PCQM)

    p = sub.add_parser("zinc-build")
    p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("merge")
    p.add_argument("--pattern", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", default="train")
    p.add_argument("--max-start", type=int, default=None)

    args = parser.parse_args(argv)
    cfg = C.load_config()
    params = {
        "max_small_side": cfg["cuts"]["max_small_side_atoms"],
        "min_context": cfg["cuts"]["min_context_atoms"],
        "min_ratio": cfg["cuts"]["min_context_to_small_ratio"],
        "reassembly_sample_every": 100,
        "reassembly_sample_rows": 2000,
    }
    if args.cmd == "pcqm-build":
        cache = C.PCQM_TRAIN_CACHE if args.dataset == "train" else C.PCQM_VALID_CACHE
        tasks = iter_pcqm_tasks(
            args.dataset, cache, args.start, args.end, C.CHUNK_DIR, params, args.chunk_size
        )
        run_tasks(tasks, process_pcqm_chunk, args.workers, f"pcqm-{args.dataset}")
        return 0
    if args.cmd == "zinc-build":
        all_records = {}
        for split in ("train", "valid", "test"):
            ds = C.load_zinc_split(split)
            all_records[split] = C.zinc_arrays(ds)
        for split, records in all_records.items():
            tasks = iter_zinc_tasks(split, records, C.CHUNK_DIR, params)
            run_tasks(tasks, process_zinc_chunk, args.workers, f"zinc-{split}")
        return 0
    if args.cmd == "merge":
        stems = list_chunks(args.pattern)
        if args.max_start is not None:
            stems = [p for p in stems if int(p.stem.rsplit("_", 1)[-1]) < args.max_start]
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        split_codes = {"train": 0, "valid": 1, "test": 2}
        split_codes = {s: split_codes[s] for s in splits}
        summary = merge_chunks(stems, Path(args.out), splits=splits, split_codes=split_codes)
        print(json.dumps(summary, indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
