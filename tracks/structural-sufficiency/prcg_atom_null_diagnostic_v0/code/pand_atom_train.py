"""PRCG-ATOM-NULL part B: train the single ATOM_ONLY arm.

The arm is the frozen PRCG-v0 model with an *empty* merge grammar:

* every atom is a single-atom word (no ``Compose`` call is ever made),
* every real molecular bond is an inter-word bond,
* atom encoding, Q, f, global readout, raw fields and graph scalars are
  byte-for-byte the PRCG-v0 modules,
* the training recipe, width and initialisation are the frozen PRCG-v0 ones.

No atom-only-specific network is added.  One seed (0), one formal run, D0 only.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import pand_common as P
import prcg_train as PT


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override only for a smoke run; the formal run uses the frozen recipe")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(P.RESULTS_DIR))
    ap.add_argument("--allow-nondeterministic", action="store_true")
    return ap.parse_args(argv)


def build_data() -> dict[str, Any]:
    bundle = P.D.load_bundle()
    rows_fit = np.asarray(bundle.fit_rows, np.int64)
    rows_eval = np.asarray(bundle.eval_rows, np.int64)
    all_rows = np.concatenate([rows_fit, rows_eval])
    molecules = P.D.build_molecules(bundle.raw, all_rows)
    grammar = P.empty_grammar()
    cache = P.T.featurize(molecules, all_rows.tolist(), grammar)
    index = {int(r): i for i, r in enumerate(all_rows.tolist())}
    return {"bundle": bundle, "rows_fit": rows_fit, "rows_eval": rows_eval,
            "all_rows": all_rows, "molecules": molecules, "grammar": grammar,
            "cache": cache, "index": index}


def cache_facts(cache: dict[str, np.ndarray]) -> dict[str, Any]:
    n_atoms = int(np.asarray(cache["w_rule"]).size)
    return {
        "n_words": n_atoms,
        "n_atoms": n_atoms,
        "n_final_words": int(np.asarray(cache["w_final"]).sum()),
        "n_bonds": int(np.asarray(cache["b_core"]).shape[0]),
        "n_inter_word_bonds": int(np.asarray(cache["e_i"]).size),
        "n_crossing_bonds": int(np.asarray(cache["x_word"]).size),
        "all_words_are_atoms": bool(np.all(np.asarray(cache["w_rule"]) < 0)),
        "all_word_sizes_one": bool(np.all(np.asarray(cache["w_size"]) == 1)),
        "word_size_histogram": {str(int(k)): int(v) for k, v in
                                zip(*np.unique(cache["w_size"], return_counts=True))},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    t0 = time.time()
    cfg = P.load_config()
    parent = P.parent_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_data()
    bundle = data["bundle"]
    rows_fit, rows_eval = data["rows_fit"], data["rows_eval"]
    grammar = data["grammar"]

    run_config = dict(parent)
    run_config["protocol_id"] = P.PROTOCOL_ID
    run_config["parent_protocol_id"] = parent["protocol_id"]
    run_config["arms"] = {"ATOM_ONLY": {"grammar": "atom_only", "port": "REAL"}}
    run_config["atom_only"] = dict(cfg["new_arm"])

    deterministic = False
    if not args.allow_nondeterministic:
        deterministic = PT._make_deterministic(int(args.threads))
    result = PT.run_arm(
        "ATOM_ONLY", int(cfg["new_arm"]["seed"]), int(cfg["new_arm"]["init_seed"]),
        run_config, bundle, data["molecules"], {"atom_only": grammar},
        {"atom_only": data["cache"]}, data["index"], rows_fit, rows_eval,
        args.device, args.epochs, out_dir, args.tag, deterministic)
    wall = time.time() - t0
    manifest = {
        "protocol_id": P.PROTOCOL_ID,
        "part": "B_atom_only_training",
        "generated_utc": P.C.now_utc(),
        "git_commit": P.git_commit(),
        "parent_protocol_id": parent["protocol_id"],
        "parent_config_sha256": P.PARENT_CONFIG_SHA256,
        "new_code_hashes": P.new_code_hashes(),
        "parent_code_hashes": P.C.code_hashes(),
        "runtime": {
            "device": args.device,
            "torch_threads": int(torch.get_num_threads()),
            "deterministic": bool(deterministic),
            "wall_seconds_total": wall,
            "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "data": {
            "split_sha256": bundle.hashes["split"]["sha256"],
            "fit_rows": int(rows_fit.size),
            "dev_rows": int(rows_eval.size),
            "fit_rows_sha256": P.C.array_sha256(rows_fit),
            "dev_rows_sha256": P.C.array_sha256(rows_eval),
        },
        "grammar": {"name": "atom_only", "n_rules": len(grammar),
                    "construction": "empty Grammar(); initial singleton partition only"},
        "cache_facts": cache_facts(data["cache"]),
        "parameters": result["parameters"],
        "init_hashes": result["init_hashes"],
        "result_file": str((out_dir / "ATOM_ONLY_seed0.json").relative_to(P.REPO_ROOT)),
        "checkpoint_file": str((out_dir / "ATOM_ONLY_seed0.pt").relative_to(P.REPO_ROOT)),
    }
    P.write_json(out_dir / "ATOM_ONLY_seed0_manifest.json", manifest)
    print(json.dumps({
        "arm": result["arm"], "best_epoch": result["best_epoch"],
        "dev_rmse": result["dev_rmse"], "dev_mae": result["dev_mae"],
        "fit_rmse": result["fit_rmse"], "wall_seconds": result["wall_seconds"],
        "n_rules": len(grammar), "cache": manifest["cache_facts"],
        "peak_rss_kb": manifest["runtime"]["peak_rss_kb"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())