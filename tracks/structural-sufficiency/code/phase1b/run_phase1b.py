"""Phase 1b runner: prep frozen structures/splits/cache, then train one model.

Examples
--------
Prep (CPU, once per dataset):
    python .../run_phase1b.py prep --dataset zinc
    python .../run_phase1b.py prep --dataset lipo

Train one trajectory:
    python .../run_phase1b.py train --dataset zinc --model S0 --seed 0 --device cuda

CPU smoke:
    python .../run_phase1b.py train --dataset lipo --model R1 --seed 0 \
        --device cpu --smoke --epochs 2 --patience 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ssrc_data import (  # noqa: E402
    CACHE_DIR,
    LIPO_LABELS,
    LIPO_RAW,
    LIPO_SPLIT,
    ZINC_ROOT,
    LabelVault,
    build_cache,
    load_cache,
    load_lipo_mols,
    load_lipo_split,
    load_zinc_train,
    row_sha256,
    save_cache,
    sha256_file,
    write_json,
    write_labels_artifact,
    write_zinc_split,
    zinc_internal_split,
)
from ssrc_train import TrainConfig, train_one  # noqa: E402

RESULTS = REPO / "tracks/structural-sufficiency/results/phase1b"
RUNS = REPO / "tracks/structural-sufficiency/runs/phase1b"
SMOKE_RUNS = REPO / "tracks/structural-sufficiency/runs/phase1b_smoke"
MODELS = ["S0", "R1", "M1", "M2", "G_ref"]


def _zinc_paths() -> dict[str, Path]:
    return {
        "split": RESULTS / "split_zinc_phase1b.json",
        "labels": RESULTS / "zinc_labels_fit_monitor.json",
        "cache": CACHE_DIR / "zinc_phase1b.pt",
    }


def _lipo_paths() -> dict[str, Path]:
    return {
        "split": LIPO_SPLIT,
        "labels": LIPO_LABELS,
        "cache": CACHE_DIR / "lipo_phase1b.pt",
    }


def prep(dataset: str) -> int:
    if dataset == "zinc":
        mols, y, data_prov = load_zinc_train(ZINC_ROOT)
        split = zinc_internal_split()
        split_payload = write_zinc_split(_zinc_paths()["split"], split)
        scope = write_labels_artifact(
            _zinc_paths()["labels"], y, split["fit"], split["monitor"], split["reserved"],
            extra_scope={"target": data_prov["target"]},
        )
        cache = build_cache(mols)
        save_cache(cache, _zinc_paths()["cache"])
        manifest = {
            "dataset": "zinc",
            "split": {
                "file": str(_zinc_paths()["split"].relative_to(REPO)),
                "sha256": split_payload["sha256"],
                "counts": split_payload["counts"],
            },
            "label_scope": scope,
            "data_provenance": data_prov,
            "cache": {k: v for k, v in cache["meta"].items()},
        }
        write_json(RESULTS / "prep_zinc_manifest.json", manifest)
        print(json.dumps({"dataset": "zinc", "cache_meta": cache["meta"], "scope": scope}, indent=2))
        return 0

    if dataset == "lipo":
        mols = load_lipo_mols(LIPO_RAW)
        split_info = load_lipo_split(LIPO_SPLIT)
        cache = build_cache(mols)
        save_cache(cache, _lipo_paths()["cache"])
        manifest = {
            "dataset": "lipo",
            "split": {
                "file": str(Path(LIPO_SPLIT).relative_to(REPO)),
                "sha256": split_info["sha256"],
                "counts": {k: len(v) for k, v in split_info["split"].items()},
            },
            "raw_graph_sha256": sha256_file(LIPO_RAW),
            "labels_sha256": sha256_file(LIPO_LABELS),
            "cache": {k: v for k, v in cache["meta"].items()},
        }
        write_json(RESULTS / "prep_lipo_manifest.json", manifest)
        print(json.dumps({"dataset": "lipo", "cache_meta": cache["meta"], "split": split_info["sha256"]}, indent=2))
        return 0

    raise SystemExit(f"unknown dataset {dataset!r}")


def train(args: argparse.Namespace) -> int:
    dataset = args.dataset
    paths = _zinc_paths() if dataset == "zinc" else _lipo_paths()
    if not paths["cache"].exists():
        raise SystemExit(f"missing cache {paths['cache']}; run prep --dataset {dataset} first")
    cache = load_cache(paths["cache"])
    if dataset == "zinc":
        split_payload = json.loads(paths["split"].read_text())
        split = {k: [int(x) for x in v] for k, v in split_payload["rows"].items()}
    else:
        split_info = load_lipo_split(paths["split"])
        split = split_info["split"]

    n_total = int(cache["meta"]["n_molecules"])
    if dataset == "zinc" and n_total != 10000:
        raise SystemExit("unexpected ZINC cache size")
    if dataset == "lipo" and n_total != 4200:
        raise SystemExit("unexpected Lipophilicity cache size")

    vault = LabelVault(paths["labels"], split["fit"], split["monitor"], split["reserved"])
    vault.load()
    y_all = torch.full((n_total,), float("nan"), dtype=torch.float32)
    for rows, purpose in ((split["fit"], "train"), (split["monitor"], "monitor")):
        vals = vault.labels(rows, purpose)
        y_all[torch.as_tensor(rows, dtype=torch.long)] = vals
    if torch.isnan(y_all[torch.as_tensor(split["fit"] + split["monitor"], dtype=torch.long)]).any():
        raise SystemExit("missing allowed labels")

    train_rows = list(split["fit"])
    monitor_rows = list(split["monitor"])
    reserved_rows = list(split["reserved"])
    if args.smoke:
        train_rows = train_rows[:128]
        monitor_rows = monitor_rows[:64]
        if set(train_rows) & set(reserved_rows) or set(monitor_rows) & set(reserved_rows):
            raise SystemExit("smoke rows hit Reserved")

    default_epochs = 300 if dataset == "zinc" else 200
    epochs = int(args.epochs) if args.epochs is not None else default_epochs
    patience = int(args.patience) if args.patience is not None else (30 if not args.smoke else 1)
    batch_size = int(args.batch_size) if args.batch_size is not None else (64 if not args.smoke else 32)
    cfg = TrainConfig(epochs=epochs, patience=patience, batch_size=batch_size)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    out_dir = Path(args.out_dir) if args.out_dir else (SMOKE_RUNS if args.smoke else RUNS)
    out_path = out_dir / dataset / f"{args.model}_seed{args.seed}{'_smoke' if args.smoke else ''}.json"
    provenance = {
        "dataset": dataset,
        "cache_sha256": sha256_file(paths["cache"]),
        "cache_meta": dict(cache["meta"]),
        "split_file": str(paths["split"].relative_to(REPO)) if paths["split"].is_relative_to(REPO) else str(paths["split"]),
        "split_sha256": sha256_file(paths["split"]),
        "labels_file": str(paths["labels"].relative_to(REPO)) if paths["labels"].is_relative_to(REPO) else str(paths["labels"]),
        "labels_sha256": sha256_file(paths["labels"]),
        "label_vault": vault.audit(),
        "row_sha256": {k: row_sha256(v) for k, v in split.items()},
    }
    payload = train_one(
        model_name=args.model,
        dataset=dataset,
        seed=int(args.seed),
        train_rows=train_rows,
        monitor_rows=monitor_rows,
        reserved_rows=reserved_rows,
        y_all=y_all,
        cache=cache,
        cfg=cfg,
        device=device,
        out_path=out_path,
        provenance=provenance,
        smoke=bool(args.smoke),
    )
    print(
        json.dumps(
            {
                "dataset": dataset,
                "model": args.model,
                "seed": int(args.seed),
                "n_train": payload["n_train"],
                "n_monitor": payload["n_monitor"],
                "params": payload["parameters"],
                "best_epoch": payload["best_epoch"],
                "monitor_mae": payload["best_metrics"]["monitor"]["mae"],
                "monitor_rmse": payload["best_metrics"]["monitor"]["rmse"],
                "train_mae": payload["best_metrics"]["train"]["mae"],
                "train_seconds": payload["train_seconds"],
                "gpu_seconds": payload["gpu_seconds"],
                "peak_gpu_mb": payload["peak_gpu_memory_mb_allocated"],
                "label_vault": payload["provenance"]["label_vault"],
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p_prep = sub.add_parser("prep")
    p_prep.add_argument("--dataset", choices=["zinc", "lipo"], required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", choices=["zinc", "lipo"], required=True)
    p_train.add_argument("--model", choices=MODELS, required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--out-dir", default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--patience", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--smoke", action="store_true")

    args = ap.parse_args()
    if args.mode == "prep":
        return prep(args.dataset)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
