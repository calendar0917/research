"""Official Valid/Test descriptive transfer (labels never used for selection)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import sfec_common as C
import sfec_data as D
from sfec_p1 import compact_ctx, metric, pair_counts_for_keys, rank_key, sign_accuracy
from sfec_p2 import fit_shared


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _key_set(prefix: Path, column: str, kind: str) -> set[bytes]:
    tab = pq.read_table(prefix.with_suffix(f".{kind}.parquet"), columns=[column])
    out = set()
    for value in tab[column].to_pylist():
        out.add(_digest(str(value)))
    return out


def _train_mu(prefix: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    # Reuse the P1 JSON, which already froze the discovery medians.
    path = Path(str(prefix) + ".universe.npz")
    # The P1 CSV has the per-pair medians; read it for a dependency-free lookup.
    import csv
    rows = {}
    csv_path = C.ANALYSIS_DIR / f"response_transfer_{'pcqm' if 'pcqm' in str(prefix) else 'zinc'}.csv"
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["A_rank"]), int(row["B_rank"]))] = {
                "mu": float(row["mu_median"]),
                "n": int(row["n_discovery"]),
            }
    return rows


def run_pcqm_valid(train_prefix: Path, valid_prefix: Path, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    train_cells = D.load_cells(train_prefix)
    train_uni = D.load_universe(train_prefix)
    train_S = D.load_S(train_prefix)
    train_C = D.load_C(train_prefix, columns=["C_id", "hash64", "hash_mod"])
    train_M = D.load_M(train_prefix, columns=["mol_id", "mol_key"])
    valid_cells = D.load_cells(valid_prefix)
    valid_S = D.load_S(valid_prefix)
    valid_M = D.load_M(valid_prefix, columns=["mol_id", "mol_key"])
    labels = C.load_pcqm_labels()
    primary = train_uni["primary"]
    rank = train_uni["rank"]
    s_digest = {_digest(str(k)): int(i) for i, k in enumerate(train_S["S_key"].tolist())}
    vmap = np.asarray([s_digest.get(_digest(str(k)), -1) for k in valid_S["S_key"].tolist()], dtype=np.int64)
    train_hash = set(int(x) for x in train_C["hash64"].tolist())
    train_mol = set(_digest(str(k)) for k in train_M["mol_key"].tolist())
    valid_hash = np.asarray(
        pq.read_table(valid_prefix.with_suffix(".C.parquet"), columns=["hash64"])["hash64"].to_numpy(),
        dtype=np.uint64,
    )
    valid_mol_digest = [_digest(str(k)) for k in valid_M["mol_key"].tolist()]
    valid_mol_dup = np.asarray([d in train_mol for d in valid_mol_digest], dtype=bool)
    # local valid cells with mapped S ids
    cells = {k: v.copy() for k, v in valid_cells.items()}
    mapped = vmap[cells["S_id"].astype(np.int64)]
    cells["S_id"] = mapped
    seen_s = mapped >= 0
    seen_c = np.asarray([int(h) in train_hash for h in valid_hash[cells["C_id"].astype(np.int64)]], dtype=bool)
    seen_c_by_C = np.asarray([int(h) in train_hash for h in valid_hash.tolist()], dtype=bool)
    dup_mol = valid_mol_dup[cells["mol_id"].astype(np.int64)]
    base_mask = seen_s & primary[np.maximum(mapped, 0)] & ~dup_mol
    ctx_v = D.build_contexts(cells, base_mask, labels, len(train_S["S_id"]), primary, rank)
    pairs_v = D.build_pairs(ctx_v, rank)
    train_stats = D.pair_stats(D.build_pairs(
        D.build_contexts(train_cells, train_uni["discovery_mask"], labels, len(train_S["S_id"]), primary, rank),
        rank,
    ))
    mu_map = {}
    for i, key in enumerate(train_stats["pair_key"]):
        mu_map[int(key)] = (float(train_stats["median"][i]), int(train_stats["n"][i]))
    key_v = pairs_v["A"].astype(np.int64) * 64 + pairs_v["B"].astype(np.int64)
    mu = np.full(len(key_v), np.nan)
    n_disc = np.zeros(len(key_v), dtype=np.int64)
    for i, k in enumerate(key_v.tolist()):
        if k in mu_map:
            mu[i], n_disc[i] = mu_map[k]
    use = np.isfinite(mu) & (n_disc >= 10)
    # categories by context/struct (per pair row)
    c_unseen_row = ~seen_c_by_C[ctx_v.C[pairs_v["ctx"]]]
    stats = {}
    for label, mask in (
        ("all_pairs_with_discovery_mu", use),
        ("unseen_exact_context", use & c_unseen_row),
        ("seen_exact_context", use & ~c_unseen_row),
    ):
        if not np.any(mask):
            stats[label] = {"n": 0}
            continue
        delta = pairs_v["delta"][mask]
        pred = mu[mask]
        rc, nc = compact_ctx(pairs_v["ctx"][mask])
        m0 = metric(delta, np.zeros_like(delta)); m1 = metric(delta, pred)
        stats[label] = {
            "n_obs": int(len(delta)), "n_ctx": int(nc),
            "n_pairs": int(len(np.unique(key_v[mask]))),
            "zero_mae": m0["mae"], "pair_mae": m1["mae"],
            "skill": 1.0 - m1["mae"] / max(m0["mae"], 1e-12),
            "sign_accuracy": sign_accuracy(delta, pred),
        }
    payload = {
        "dataset": "pcqm_valid",
        "n_valid_cells": int(len(valid_cells["C_id"])),
        "n_valid_molecules": int(len(valid_M["mol_id"])),
        "n_valid_molecules_duplicate_train_graph": int(valid_mol_dup.sum()),
        "n_cells_seen_s": int(seen_s.sum()),
        "n_cells_unseen_s": int((~seen_s).sum()),
        "n_cells_seen_c": int(seen_c.sum()),
        "n_cells_unseen_c": int((~seen_c).sum()),
        "n_cells_duplicate_graph": int(dup_mol.sum()),
        "n_contexts_used": int(ctx_v.n_ctx),
        "n_pair_observations": int(len(pairs_v["delta"])),
        "n_pair_observations_with_discovery_mu": int(use.sum()),
        "transfer": stats,
        "notes": "PCQM official Test labels were never read; only Train/Valid used.",
    }
    C.write_json(out_dir / "valid_transfer_pcqm.json", payload)
    C.write_csv(out_dir / "valid_transfer_pcqm.csv", [{"subset": k, **v} for k, v in stats.items()])
    return payload


def run_zinc_valid(train_prefix: Path, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    cells = D.load_cells(train_prefix)
    uni = D.load_universe(train_prefix)
    S = D.load_S(train_prefix)
    primary = uni["primary"]
    rank = uni["rank"]
    train_mol = set(cells["mol_id"][cells["split_id"] == 0].tolist())
    train_C = set(cells["C_id"][cells["split_id"] == 0].tolist())
    out = {"dataset": "zinc", "splits": {}}
    for split, code in (("valid", 1), ("test", 2)):
        labels = C.zinc_labels(split)
        dup = np.isin(cells["mol_id"], np.asarray(sorted(train_mol), dtype=np.int64))
        mask = (cells["split_id"] == code) & primary[cells["S_id"]] & ~dup
        ctx = D.build_contexts(cells, mask, labels, len(S["S_id"]), primary, rank)
        pairs = D.build_pairs(ctx, rank)
        key = pairs["A"].astype(np.int64) * 64 + pairs["B"].astype(np.int64)
        # discovery medians from train json/csv
        mu_map = {}
        csv_path = C.ANALYSIS_DIR / "response_transfer_zinc.csv"
        if csv_path.exists():
            import csv
            with csv_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    mu_map[(int(row["A_rank"]), int(row["B_rank"]))] = float(row["mu_median"])
        mu = np.full(len(key), np.nan)
        for i in range(len(key)):
            mu[i] = mu_map.get((int(pairs["A"][i]), int(pairs["B"][i])), np.nan)
        use = np.isfinite(mu)
        if np.any(use):
            delta = pairs["delta"][use]
            pred = mu[use]
            m0 = metric(delta, np.zeros_like(delta)); m1 = metric(delta, pred)
            trans = {"n_obs": int(use.sum()), "zero_mae": m0["mae"], "pair_mae": m1["mae"],
                     "skill": 1.0 - m1["mae"] / max(m0["mae"], 1e-12)}
        else:
            trans = {"n_obs": 0}
        out["splits"][split] = {
            "n_cells": int(mask.sum()),
            "n_contexts": int(ctx.n_ctx),
            "n_pair_observations": int(len(pairs["delta"])),
            "n_pair_observations_with_discovery_mu": int(use.sum()),
            "n_cells_from_duplicate_train_graph": int(np.sum((cells["split_id"] == code) & dup)),
            "n_cells_seen_exact_context": int(np.sum(mask & np.isin(cells["C_id"], list(train_C)))),
            "transfer": trans,
        }
    C.write_json(out_dir / "valid_transfer_zinc.json", out)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pcqm", "zinc"])
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--valid-prefix", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = C.load_config()
    out = Path(args.out) if args.out else C.ANALYSIS_DIR
    if args.dataset == "pcqm":
        payload = run_pcqm_valid(Path(args.train_prefix), Path(args.valid_prefix), out, cfg)
    else:
        payload = run_zinc_valid(Path(args.train_prefix), out, cfg)
    print(json.dumps(payload, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())