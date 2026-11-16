"""Local data preparation for round sq-pilot-small-v1.

Run from a machine with the official OGB data (``data/ogb``) and RDKit:

    uv run python tracks/structural-query-learning/code/sq_prepare.py

It writes the small, LABEL-FREE caches that ``res`` consumes:

* ``data/ogbg_<task>_sq-pilot-cache-v1.npz`` — raw OGB atom/bond columns,
  canonical physical bonds, unlabeled degree/min-ring/cycle primitives, official
  row ids, official split indices, targets, SMILES;
* ``data/ogbg_<task>_ecfp4_sq-pilot-cache-v1.npz`` — Morgan r=2 / 2048 bits /
  chirality + the ten frozen descriptors + parse-failure flag (E's inputs);
* ``data/ogbg_<task>_sq-pilot-lipo-raw-v1.npz`` — the frozen ``lipo_v1`` raw
  attributes for the C1 reference (L), so no RDKit is needed on the GPU host.

plus ``analysis/prepare_notes.md`` and ``analysis/protocol_freeze.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import sq_common as C  # noqa: E402
import sq_data as D  # noqa: E402


def freeze_payload() -> dict:
    import sq_ast as A
    import sq_bank as B
    import sq_gine as G
    import sq_search as S
    import sq_xgb as X

    return {
        "round_id": C.ROUND_ID,
        "code_files": {
            name: C.sha256_file(CODE_DIR / name)
            for name in sorted(path.name for path in CODE_DIR.glob("*.py"))
        },
        "query_language": {
            "max_nodes": A.MAX_NODES,
            "max_nbr_branch_hops": A.MAX_HOPS,
            "node_kinds": sorted(A.NODE_KINDS),
            "operators": sorted(A.OP_KINDS),
            "feature_transforms": list(A.FEATURE_TRANSFORMS),
        },
        "base_bank": {
            "cap": B.BASE_BANK_CAP,
            "min_train_molecules": B.MIN_TRAIN_MOLECULES,
            "base_columns": list(B.BASE_COLUMN_NAMES),
        },
        "discovery": {
            "proxy_params": S.proxy_params("cuda:0"),
            "proxy_rounds": S.PROXY_ROUNDS,
            "max_outer_rounds": S.MAX_OUTER_ROUNDS,
            "max_new_per_round": S.MAX_NEW_PER_ROUND,
            "max_discovered": S.MAX_DISCOVERED,
            "n_candidates": S.N_CANDIDATES,
            "quota": {
                "and_primitive": S.N_AND_PRIMITIVE,
                "and_pair": S.N_AND_PAIR,
                "nbr": S.N_NBR,
                "branch": S.N_BRANCH,
            },
            "max_generation_attempts": S.MAX_GENERATION_ATTEMPTS,
            "parents": {
                "n": S.N_PARENTS,
                "scored": S.N_PARENTS_SCORED,
                "random": S.N_PARENTS_RANDOM,
            },
            "stump": {
                "quantiles": list(S.QUANTILES),
                "half_integers": list(S.HALF_INTEGER_THRESHOLDS),
                "min_molecules": S.STUMP_MIN_MOLECULES,
                "ridge": S.STUMP_RIDGE,
                "complexity_penalty": S.COMPLEXITY_PENALTY,
            },
            "rng_seed": C.SEARCH_RNG_SEED,
            "time_limit_s": S.DISCOVERY_TIME_LIMIT_S,
        },
        "random_library": {
            "rng_seed": C.RANDOM_LIBRARY_SEED,
            "max_attempts_per_slot": S.MAX_R_ATTEMPTS_PER_SLOT,
            "time_limit_s": S.R_TIME_LIMIT_S,
        },
        "final_heads": {
            "num_boost_round": X.NUM_BOOST_ROUND,
            "presets": X.PRESETS,
            "common": {
                "min_child_weight": X.MIN_CHILD_WEIGHT,
                "reg_lambda": X.REG_LAMBDA,
                "reg_alpha": X.REG_ALPHA,
                "subsample": X.SUBSAMPLE,
                "colsample_bytree": X.COLSAMPLE,
                "max_bin": X.MAX_BIN,
                "nthread": X.NTHREAD,
            },
        },
        "gine": {
            "epochs": G.EPOCHS,
            "batch_size": G.BATCH_SIZE,
            "warmup_epochs": G.WARMUP_EPOCHS,
            "min_lr": G.MIN_LR,
            "weight_decay": G.WEIGHT_DECAY,
            "grad_clip": G.GRAD_CLIP,
            "dropout": G.DROPOUT,
            "hidden": G.HIDDEN,
            "n_layers": G.N_LAYERS,
            "peak_lrs": list(G.PEAK_LRS),
            "dataloader_seed": G.DATALOADER_SEED,
        },
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "seeds": {"train": C.TRAIN_SEED},
    }


def audit_from_committed_cache(dataset: str) -> dict[str, object]:
    """Verify-only path: rebuild the audit from the committed caches (no OGB, no RDKit)."""
    cache_path = D.cache_path(dataset)
    cache = D.load_cache(dataset)
    if cache["cache_sha256"] != C.EXPECTED_CACHE_SHA256[cache_path.name]:
        raise ValueError(f"{dataset}: graph cache sha256 does not match the pinned value")
    packed = D.packed_split(cache, "train")
    if int(packed.n_bonds) != int(packed.graph_n_bonds.sum()):
        raise ValueError(f"{dataset}: physical bond count disagrees with per-graph sums")
    splits = {split: int(D.split_indices(cache, split).size) for split in ("train", "valid", "test")}
    if splits != C.EXPECTED_SPLIT_SIZES[dataset]:
        raise ValueError(f"{dataset}: split sizes {splits} != official {C.EXPECTED_SPLIT_SIZES[dataset]}")
    for name, path in (("ecfp", D.ecfp_path(dataset)), ("lipo_raw", D.lipo_raw_path(dataset))):
        if not path.is_file() or C.sha256_file(path) != C.EXPECTED_CACHE_SHA256[path.name]:
            raise ValueError(f"{dataset}: {name} cache missing or changed")
    labels = cache["labels"]
    return {
        "cache": {
            "path": str(cache_path),
            "sha256": cache["cache_sha256"],
            "bytes": int(cache_path.stat().st_size),
        },
        "ecfp": {
            "path": str(D.ecfp_path(dataset)),
            "sha256": C.sha256_file(D.ecfp_path(dataset)),
            "bytes": int(D.ecfp_path(dataset).stat().st_size),
        },
        "lipo_raw": {
            "path": str(D.lipo_raw_path(dataset)),
            "sha256": C.sha256_file(D.lipo_raw_path(dataset)),
            "bytes": int(D.lipo_raw_path(dataset).stat().st_size),
        },
        "audit": {
            "n_graphs": int(cache["row_ids"].size),
            "split_sizes": splits,
            "n_atoms_total": int(cache["node_ptr"][-1]),
            "n_physical_bonds_total": int(cache["bond_ptr"][-1]),
            "n_multi_component_molecules": int((cache["graph_n_components"] > 1).sum()),
            "ring_bucket_counts": {
                str(b): int((cache["min_ring"] == b).sum()) for b in range(0, 10)
            },
            "label_range": {
                "min": float(np.min(labels)),
                "max": float(np.max(labels)),
            },
            "label_hash": {
                split: C.sha256_array(np.asarray(labels[D.split_indices(cache, split)]))
                for split in ("train", "valid", "test")
            },
            "split_index_hash": {
                split: C.sha256_array(D.split_indices(cache, split))
                for split in ("train", "valid", "test")
            },
        },
    }


def _write_analysis(audit: dict[str, object], *, mode: str) -> None:
    frozen = freeze_payload()
    frozen["cache_sha256"] = dict(C.EXPECTED_CACHE_SHA256)
    frozen["datasets"] = audit
    frozen["provenance"] = {
        "mode": mode,
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "environment": C.environment_snapshot(),
    }
    C.write_json(C.ANALYSIS_DIR / "protocol_freeze.json", frozen)

    notes = ["# Preparation notes (sq-pilot-small-v1)\n"]
    notes.append(f"- provenance: `{mode}`; protocol `{C.PROTOCOL_ID}`; commit `{C.git_commit()}`")
    notes.append(
        "- The GPU host consumes only these committed label-free caches: it needs neither "
        "the OGB raw archives nor RDKit.  `stage=prepare` and every worker re-verify the "
        "SHA-256 values pinned in `code/sq_common.py::EXPECTED_CACHE_SHA256`, the official "
        "split sizes, the split index hashes and the physical-bond/ring invariants."
    )
    for dataset, entry in audit.items():
        info = entry["audit"]
        notes.append(
            f"- `{dataset}`: {info['n_graphs']} molecules, splits {info['split_sizes']}, "
            f"{info['n_atoms_total']} atoms, {info['n_physical_bonds_total']} physical bonds, "
            f"{info['n_multi_component_molecules']} multi-component molecules, "
            f"ring buckets {info['ring_bucket_counts']}, "
            f"labels [{info['label_range']['min']:.3f}, {info['label_range']['max']:.3f}]"
        )
        for name in ("cache", "ecfp", "lipo_raw"):
            notes.append(
                f"  - {name}: `{Path(entry[name]['path']).name}` "
                f"sha256 `{entry[name]['sha256']}` ({entry[name]['bytes']} bytes, committed)"
            )
    notes.append("")
    notes.append("## Reproducing the caches")
    notes.append("")
    notes.append("From a machine with the official OGB trees under `data/ogb` and RDKit:")
    notes.append("")
    notes.append("```")
    notes.append("uv run python tracks/structural-query-learning/code/sq_prepare.py")
    notes.append("```")
    notes.append("")
    notes.append("On any other machine (no OGB raw data, no RDKit) the caches are verified instead:")
    notes.append("")
    notes.append("```")
    notes.append("uv run python tracks/structural-query-learning/code/sq_prepare.py --verify-only")
    notes.append("```")
    notes.append("")
    (C.ANALYSIS_DIR / "prepare_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not rebuild anything; verify the committed caches and rewrite analysis/",
    )
    args = parser.parse_args(argv)
    C.ensure_dir(C.DATA_DIR)
    C.ensure_dir(C.ANALYSIS_DIR)
    if args.verify_only:
        audit = {dataset: audit_from_committed_cache(dataset) for dataset in C.DATASETS}
        _write_analysis(audit, mode="verify-only")
    else:
        audit = {}
        for dataset in C.DATASETS:
            prepared = D.prepare_dataset(dataset, root="data/ogb")
            cache_info = D.save_cache(prepared["payload"], D.cache_path(dataset))
            ecfp_info = D.save_ecfp(D.build_ecfp_payload(dataset), D.ecfp_path(dataset))
            lipo_info = D.save_lipo_raw(D.build_lipo_raw_payload(dataset), D.lipo_raw_path(dataset))
            audit[dataset] = {
                "cache": cache_info,
                "ecfp": ecfp_info,
                "lipo_raw": lipo_info,
                "audit": prepared["audit"],
            }
        _write_analysis(audit, mode="rebuilt from data/ogb")
    printable = {
        dataset: {
            **{k: v for k, v in entry.items() if k != "audit"},
            "split_sizes": entry["audit"]["split_sizes"],
            "n_graphs": entry["audit"]["n_graphs"],
        }
        for dataset, entry in audit.items()
    }
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
