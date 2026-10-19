"""Phase 1d runner: additive-incidence control and one capacity-scaling arm.

Only the official ZINC *train* split is read through the frozen Phase 1b
internal split and Fit/Monitor label vault.  Official valid/test and internal
Reserved are never instantiated or scored.

Examples
--------
GPU Phase B:
    python .../run_phase1d.py train --dataset zinc --model R_incidence --seed 0 --device cuda

GPU Phase C (one model chosen by the frozen protocol rule):
    python .../run_phase1d.py train --dataset zinc --model R1_size_cap --seed 0 --device cuda
    python .../run_phase1d.py train --dataset zinc --model R_incidence_cap --seed 0 --device cuda

CPU smoke:
    python .../run_phase1d.py train --dataset zinc --model R_incidence --seed 0 \
        --device cpu --smoke --epochs 2 --patience 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = HERE.parent / "phase1b"
PHASE1C = HERE.parent / "phase1c"
for p in (HERE, PHASE1B, PHASE1C):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssrc_data import (  # noqa: E402
    CACHE_DIR,
    LabelVault,
    load_cache,
    row_sha256,
    sha256_file,
)
from ssrc_models_p1d import (  # noqa: E402
    base_incidence_config,
    incidence_capacity_config,
    r1_size_capacity_config,
    register_phase1d_models,
)
from ssrc_train import TrainConfig, train_one  # noqa: E402

RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RESULTS_1C = REPO / "tracks/structural-sufficiency/results/phase1c"
RESULTS_1D = REPO / "tracks/structural-sufficiency/results/phase1d"
RUNS_1D = REPO / "tracks/structural-sufficiency/runs/phase1d"
CACHE_PATH = CACHE_DIR / "zinc_phase1b.pt"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
LABELS_PATH = RESULTS_1B / "zinc_labels_fit_monitor.json"
SIZE_PATH = RESULTS_1C / "zinc_size_features.pt"
SIZE_MANIFEST = RESULTS_1C / "prep_size_manifest.json"

MODELS = {
    "R_incidence": base_incidence_config,
    "R1_size_cap": r1_size_capacity_config,
    "R_incidence_cap": incidence_capacity_config,
}


def _load_size_feat() -> torch.Tensor:
    if not SIZE_PATH.exists():
        raise SystemExit(f"missing {SIZE_PATH}; run Phase 1c prep --dataset zinc first")
    obj = torch.load(SIZE_PATH, map_location="cpu", weights_only=False)
    feat = torch.as_tensor(obj["size_feat"], dtype=torch.float32)
    if feat.shape != (10000, 3):
        raise SystemExit("unexpected size-feature shape")
    return feat


def train(args: argparse.Namespace) -> int:
    if args.dataset != "zinc":
        raise SystemExit("Phase 1d is frozen to ZINC-12k")
    if args.model not in MODELS:
        raise SystemExit(f"model must be one of {sorted(MODELS)}")
    if args.seed != 0:
        raise SystemExit("Phase 1d is frozen to seed 0 only")
    register_phase1d_models()

    cache = load_cache(CACHE_PATH)
    cache["mol_size_feat"] = _load_size_feat()
    split_payload = json.loads(SPLIT_PATH.read_text())
    split = {k: [int(x) for x in v] for k, v in split_payload["rows"].items()}
    vault = LabelVault(LABELS_PATH, split["fit"], split["monitor"], split["reserved"])
    vault.load()
    y_all = torch.full((10000,), float("nan"), dtype=torch.float32)
    for rows, purpose in ((split["fit"], "train"), (split["monitor"], "monitor")):
        y_all[torch.as_tensor(rows, dtype=torch.long)] = vault.labels(rows, purpose)
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

    epochs = int(args.epochs) if args.epochs is not None else 300
    patience = int(args.patience) if args.patience is not None else (1 if args.smoke else 30)
    batch_size = int(args.batch_size) if args.batch_size is not None else (32 if args.smoke else 64)
    model_cfg = MODELS[args.model]()
    cfg = TrainConfig(epochs=epochs, patience=patience, batch_size=batch_size, model=model_cfg)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    out_dir = Path(args.out_dir) if args.out_dir else RUNS_1D
    suffix = ("_smoke" if args.smoke else "") + (f"_{args.label}" if args.label else "")
    out_path = out_dir / "zinc" / f"{args.model}_seed{args.seed}{suffix}.json"
    protocol_path = REPO / "tracks/structural-sufficiency/notes/phase1d_protocol.md"
    config_path = REPO / "tracks/structural-sufficiency/configs/phase1d.yaml"
    gates_path = RESULTS_1D / "cpu_gates_phase1d.json"
    provenance = {
        "dataset": "zinc",
        "phase1d_protocol": str(protocol_path.relative_to(REPO)),
        "phase1d_protocol_sha256": sha256_file(protocol_path) if protocol_path.exists() else None,
        "phase1d_config": str(config_path.relative_to(REPO)),
        "phase1d_config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "phase1d_cpu_gates_sha256": sha256_file(gates_path) if gates_path.exists() else None,
        "size_manifest_sha256": sha256_file(SIZE_MANIFEST) if SIZE_MANIFEST.exists() else None,
        "cache_sha256": sha256_file(CACHE_PATH),
        "cache_meta": dict(cache["meta"]),
        "size_features_sha256": sha256_file(SIZE_PATH),
        "split_file": str(SPLIT_PATH.relative_to(REPO)),
        "split_sha256": sha256_file(SPLIT_PATH),
        "labels_file": str(LABELS_PATH.relative_to(REPO)),
        "labels_sha256": sha256_file(LABELS_PATH),
        "label_vault": vault.audit(),
        "row_sha256": {k: row_sha256(v) for k, v in split.items()},
        "batch_kwargs": {"include_size": True},
        "edge_rewiring": None,
        "model_config": model_cfg.as_dict(),
        "frozen_anchors": {
            "R1_size_seed0_monitor_mae": 0.236575,
            "G_ref_seed0_monitor_mae": 0.165290,
            "G_ref_params": 439559,
        },
    }
    payload = train_one(
        model_name=args.model,
        model_label=args.label,
        dataset="zinc",
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
        batch_kwargs={"include_size": True},
        protocol_id="ssrc-phase1d-zinc-v1",
        scope_extra={
            "phase": "B_incidence_control" if args.model == "R_incidence" else "C_capacity_scaling",
            "model_config": model_cfg.as_dict(),
            "capacity_target_params": None if args.model == "R_incidence" else 439559,
        },
    )
    print(json.dumps({"out": str(out_path), "monitor_mae": payload["best_monitor_mae"], "params": payload["parameters"]}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", default="zinc", choices=["zinc"])
    tr.add_argument("--model", required=True, choices=sorted(MODELS))
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--patience", type=int, default=None)
    tr.add_argument("--batch-size", type=int, default=None)
    tr.add_argument("--out-dir", default=None)
    tr.add_argument("--label", default=None)
    args = ap.parse_args()
    if args.command == "train":
        return train(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
