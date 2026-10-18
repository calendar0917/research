"""Phase 1c runner: frozen ZINC Fit/Monitor readout/correspondence controls.

Only the official ZINC *train* split is read (through the frozen Phase 1b
internal split and label vault).  Official valid/test and internal Reserved are
never instantiated or scored.

Examples
--------
Prep (CPU, once; builds the explicit size-feature sidecar):
    python .../run_phase1c.py prep --dataset zinc

Train one trajectory:
    python .../run_phase1c.py train --dataset zinc --model R1_size --seed 0 --device cuda

CPU smoke:
    python .../run_phase1c.py train --dataset zinc --model S_cap_size --seed 0 \
        --device cpu --smoke --epochs 2 --patience 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = HERE.parent / "phase1b"
for p in (HERE, PHASE1B):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssrc_data import (  # noqa: E402
    CACHE_DIR,
    LabelVault,
    load_cache,
    row_sha256,
    sha256_file,
    write_json,
)
from ssrc_models_p1c import P1CConfig, register_phase1c_models  # noqa: E402
from ssrc_train import TrainConfig, train_one  # noqa: E402

RESULTS = REPO / "tracks/structural-sufficiency/results/phase1c"
RUNS = REPO / "tracks/structural-sufficiency/runs/phase1c"
RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
CACHE_PATH = CACHE_DIR / "zinc_phase1b.pt"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
LABELS_PATH = RESULTS_1B / "zinc_labels_fit_monitor.json"
SIZE_PATH = RESULTS / "zinc_size_features.pt"
SIZE_MANIFEST = RESULTS / "prep_size_manifest.json"

MODELS = ["S_cap_size", "R1_size", "R1_marginal_size", "M_keep_edge"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _connected_components(n_atoms: int, src: np.ndarray, dst: np.ndarray) -> int:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_atoms)})


def prep(dataset: str) -> int:
    if dataset != "zinc":
        raise SystemExit("Phase 1c is frozen to ZINC-12k")
    cache = load_cache(CACHE_PATH)
    if int(cache["meta"]["n_molecules"]) != 10000:
        raise SystemExit("unexpected ZINC cache size")
    n_atoms = cache["mol_n_atoms"].numpy().astype(np.int64)
    root_edges = cache["root_n_edges"].numpy().astype(np.int64) // 2
    starts = cache["root_edge_start"].numpy()
    src_all = cache["root_edge_src_flat"].numpy()
    dst_all = cache["root_edge_dst_flat"].numpy()
    components = np.empty(n_atoms.size, dtype=np.int64)
    for mol in range(n_atoms.size):
        a, b = int(starts[mol]), int(starts[mol + 1])
        components[mol] = _connected_components(int(n_atoms[mol]), src_all[a:b], dst_all[a:b])
    rings = root_edges - n_atoms + components
    if (rings < 0).any():
        raise RuntimeError("negative cyclomatic ring count")
    size_feat = np.stack(
        [n_atoms.astype(np.float32) / 10.0, root_edges.astype(np.float32) / 10.0, rings.astype(np.float32)],
        axis=1,
    )
    SIZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n_atoms": torch.as_tensor(n_atoms, dtype=torch.int64),
            "n_bonds": torch.as_tensor(root_edges, dtype=torch.int64),
            "n_components": torch.as_tensor(components, dtype=torch.int64),
            "rings": torch.as_tensor(rings, dtype=torch.int64),
            "size_feat": torch.as_tensor(size_feat, dtype=torch.float32),
            "meta": {
                "formula": "[n_atoms/10, n_bonds/10, n_bonds - n_atoms + n_components]",
                "components_from": "union-find over the frozen root adjacency (structure only)",
                "cache_sha256": sha256_file(CACHE_PATH),
                "n_molecules": int(n_atoms.size),
                "all_connected": bool((components == 1).all()),
                "ring_min": int(rings.min()),
                "ring_max": int(rings.max()),
            },
        },
        SIZE_PATH,
    )
    split = json.loads(SPLIT_PATH.read_text())
    manifest = {
        "protocol_id": "ssrc-phase1c-zinc-v1",
        "dataset": "zinc",
        "scope": {
            "labels_read": False,
            "structures_only": True,
            "official_valid_test_read": False,
            "internal_reserved_read": False,
        },
        "cache": {"path": str(CACHE_PATH.relative_to(REPO)), "sha256": sha256_file(CACHE_PATH)},
        "split": {
            "path": str(SPLIT_PATH.relative_to(REPO)),
            "sha256": sha256_file(SPLIT_PATH),
            "row_sha256": split["sha256"],
            "counts": split["counts"],
        },
        "size_features": {
            "path": str(SIZE_PATH.relative_to(REPO)),
            "sha256": sha256_file(SIZE_PATH),
            "meta": {
                "formula": "[n_atoms/10, n_bonds/10, n_bonds - n_atoms + components]",
                "all_connected": bool((components == 1).all()),
                "n_molecules": int(n_atoms.size),
                "ring_min": int(rings.min()),
                "ring_max": int(rings.max()),
            },
        },
    }
    write_json(SIZE_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _load_size_feat() -> torch.Tensor:
    if not SIZE_PATH.exists():
        raise SystemExit(f"missing {SIZE_PATH}; run prep --dataset zinc first")
    obj = torch.load(SIZE_PATH, map_location="cpu", weights_only=False)
    feat = torch.as_tensor(obj["size_feat"], dtype=torch.float32)
    if feat.shape != (10000, 3):
        raise SystemExit("unexpected size-feature shape")
    return feat


def train(args: argparse.Namespace) -> int:
    if args.dataset != "zinc":
        raise SystemExit("Phase 1c is frozen to ZINC-12k")
    register_phase1c_models()
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
    cfg = TrainConfig(epochs=epochs, patience=patience, batch_size=batch_size, model=P1CConfig())

    batch_kwargs: dict[str, bool] = {"include_size": True}
    if args.model == "R1_marginal_size":
        batch_kwargs["include_edge_mol"] = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    out_dir = Path(args.out_dir) if args.out_dir else RUNS
    suffix = ("_smoke" if args.smoke else "") + (f"_{args.label}" if args.label else "")
    out_path = out_dir / "zinc" / f"{args.model}_seed{args.seed}{suffix}.json"
    protocol_path = REPO / "tracks/structural-sufficiency/notes/phase1c_protocol.md"
    config_path = REPO / "tracks/structural-sufficiency/configs/phase1c.yaml"
    gates_path = RESULTS / "cpu_gates_phase1c.json"
    provenance = {
        "dataset": "zinc",
        "phase1c_protocol": str(protocol_path.relative_to(REPO)),
        "phase1c_protocol_sha256": sha256_file(protocol_path) if protocol_path.exists() else None,
        "phase1c_config": str(config_path.relative_to(REPO)),
        "phase1c_config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "phase1c_cpu_gates_sha256": sha256_file(gates_path) if gates_path.exists() else None,
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
        "batch_kwargs": dict(batch_kwargs),
        "edge_rewiring": None,
        "frozen_phase1b_anchors": {
            "S0_zinc_seed0_monitor_mae": 0.317303,
            "R1_zinc_seed0_monitor_mae": 0.260161,
            "G_ref_zinc_seed0_monitor_mae": 0.165290,
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
        batch_kwargs=batch_kwargs,
        protocol_id="ssrc-phase1c-zinc-v1",
        scope_extra={
            "phase": "phase1c",
            "input_scope": "E2 patches + true root edges (S_cap_size: no root edges; R1_marginal_size: no endpoint correspondence)",
            "penalized_logp_components_as_input": False,
        },
    )
    print(
        json.dumps(
            {
                "dataset": "zinc",
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
    p_prep.add_argument("--dataset", choices=["zinc"], required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", choices=["zinc"], required=True)
    p_train.add_argument("--model", choices=MODELS, required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--out-dir", default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--patience", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--smoke", action="store_true")
    p_train.add_argument("--label", default=None, help="suffix for the output file name (e.g. long)")

    args = ap.parse_args()
    if args.mode == "prep":
        return prep(args.dataset)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
