"""Prepare the round: label-safe Train labels, D1 split, caches, pair directory.

Everything here is label-safe and deterministic.  It is the only entry point
that writes ``data/``.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import psc_common as C
import psc_data as D
import psc_paths as PP


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    L = int(args.max_len or cfg["paths"]["max_len"])
    C.ROUND_DIR.joinpath("data").mkdir(parents=True, exist_ok=True)
    C.ROUND_DIR.joinpath("analysis").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict = {"protocol_id": cfg["protocol_id"], "generated_utc": C.now_utc(),
                    "git_commit": C.git_commit(), "max_len": L}

    # 1. official Train labels (exactly two members of a frozen prediction file)
    raw = C.load_raw_structure()
    report["train_labels"] = C.build_train_labels(raw)
    bundle = D.load_bundle()
    print(f"train labels: n={bundle.labels.size and report['train_labels']['n']} "
          f"sha={report['train_labels']['sha256']}")

    # 2. F0 split identity and coverage
    f0 = C.load_split_f0()
    fit0 = np.asarray(f0["train_sub_rows"], np.int64)
    dev0 = np.asarray(f0["dev_rows"], np.int64)
    if sorted(np.concatenate([fit0, dev0]).tolist()) != bundle.official_train.tolist():
        raise ValueError("F0 does not cover official Train")
    report["split_f0"] = {
        "file": str(C.SPLIT_F0.relative_to(C.REPO_ROOT)),
        "file_sha256": f0["file_sha256"],
        "internal_sha256": f0["split_sha256"],
        "seed": int(f0["seed"]), "n_fit": int(fit0.size), "n_dev": int(dev0.size),
        "n_scaffolds_fit": int(f0["n_scaffolds_train_sub"]),
        "n_scaffolds_dev": int(f0["n_scaffolds_dev"]),
        "fit_rows_sha256": C.array_sha256(fit0),
        "dev_rows_sha256": C.array_sha256(dev0),
    }
    print(f"F0: {fit0.size}/{dev0.size}")

    # 3. D1 split (frozen before any result of this round is seen)
    if not C.SPLIT_F1.is_file():
        d1cfg = cfg["d1"]
        payload = C.build_split_d1(bundle.raw, int(d1cfg["seed"]),
                                   int(d1cfg["target_heldout"]))
        C.write_json(C.SPLIT_F1, payload)
        print(f"D1 written: fit={payload['n_fit']} heldout={payload['n_heldout']} "
              f"scaffolds fit/held={payload['n_scaffolds_fit']}/"
              f"{payload['n_scaffolds_heldout']}")
    d1 = json.loads(C.SPLIT_F1.read_text(encoding="utf-8"))
    d1_file_sha = C.sha256_file(C.SPLIT_F1)
    fit1 = np.asarray(d1["fit_rows"], np.int64)
    held1 = np.asarray(d1["heldout_rows"], np.int64)
    if sorted(np.concatenate([fit1, held1]).tolist()) != bundle.official_train.tolist():
        raise ValueError("D1 does not cover official Train")
    scaf = d1["scaffold_by_row"]
    overlap = {scaf[str(r)] for r in fit1.tolist()} & {scaf[str(r)] for r in held1.tolist()}
    if overlap:
        raise ValueError(f"D1 scaffold overlap: {len(overlap)}")
    report["split_f1"] = {
        "file": str(C.SPLIT_F1.relative_to(C.REPO_ROOT)), "file_sha256": d1_file_sha,
        "internal_sha256": d1["split_sha256"], "seed": int(d1["seed"]),
        "n_fit": int(fit1.size), "n_heldout": int(held1.size),
        "n_scaffolds_fit": int(d1["n_scaffolds_fit"]),
        "n_scaffolds_heldout": int(d1["n_scaffolds_heldout"]),
        "scaffold_overlap": 0,
        "fit_rows_sha256": C.array_sha256(fit1),
        "heldout_rows_sha256": C.array_sha256(held1),
        "assignment_rule": d1["assignment_rule"],
        "scaffold_definition": d1["scaffold_definition"],
    }
    print(f"D1: {fit1.size}/{held1.size} scaffolds fit/held="
          f"{d1['n_scaffolds_fit']}/{d1['n_scaffolds_heldout']} overlap=0")

    # 4. path cache for Train + Valid structure (no labels, no Test rows)
    cache_rows = np.union1d(bundle.official_train, bundle.official_valid)
    if not args.skip_cache:
        meta = PP.ensure_cache(bundle.raw, cache_rows, L)
        meta.pop("rows", None)
        report["path_cache"] = {k: v for k, v in meta.items() if k != "cache_path"}
        report["path_cache"]["path"] = str(C.PATH_CACHE.relative_to(C.REPO_ROOT))
        report["path_cache"]["rows_include"] = "official Train 3360 + Valid 420 structure only"
        print(f"path cache: {meta['n_paths']} keys, {meta['cache_bytes'] / 1e6:.1f} MB, "
              f"{meta['cache_sha256'][:16]}")

    # 5. matched-pair directory (Train x Train cut rows, label-free members)
    import psc_train as T
    pair_dir = T.ensure_pair_directory(bundle)
    report["pair_directory"] = {
        "path": str(T.PAIR_CACHE.relative_to(C.REPO_ROOT)),
        "sha256": C.sha256_file(T.PAIR_CACHE),
        "n_cut_rows": int(pair_dir.mol_a.size),
        "n_unique_molecules": int(np.unique(np.concatenate([pair_dir.mol_a,
                                                            pair_dir.mol_b])).size),
        "checks": pair_dir.checks,
    }
    print(f"pair directory: {pair_dir.mol_a.size} cut rows, "
          f"{pair_dir.checks['n_duplicate_pair_keys']} duplicate keys")

    # 6. fit-set pair graphs and label-free support summaries for every split
    from psc_train import rows_of
    targets = {}
    for name in ("D0", "D1", "FULL"):
        split = C.split_spec(name, bundle.raw, f0=f0, f1=d1)
        t = D.build_pair_targets(pair_dir, split["fit_rows"], bundle.labels.size)
        targets[name] = {"n_fit": int(np.asarray(split["fit_rows"]).size),
                         "pair_checks": t.checks,
                         "component_metrics": D.component_metrics(t, bundle.labels.size)}
        print(f"{name}: pairs={t.checks['n_unique_pairs']} "
              f"components={t.checks['n_components']}")
    report["pair_targets"] = targets

    report["environment"] = C.environment()
    report["peak_rss_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    report["wall_seconds"] = time.time() - t0
    C.write_json(C.ROUND_DIR / "analysis" / "prepare.json", report)
    print(f"prepare done in {report['wall_seconds']:.1f}s peak_rss="
          f"{report['peak_rss_mb']:.0f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())