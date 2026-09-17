"""MolHIV local-token channel necessity (pre-registered).

Transfers the ZINC channel-necessity question
(``decision-local-token-null-20260919``) to OGBG-MolHIV: is the dense
molecule-dependent local patch token (the exact-certificate lookup,
839,456 of 1,076,589 params, 78 %) necessary for the frozen recurrent
pair--centre backbone, or does the fixed 237,133-param backbone carry almost
all of the predictive power?

Conditions
----------
``typed_lookup``  frozen reference (839,456-param dense lookup; NOT retrained)
``null``          exact zero 32-D token per patch, **no** local-token generator
``constant``      one trainable graph-wide 32-D vector shared by every patch

Only the local-token channel changes; the patch encoder, parent embedding,
pair/centre core, global encoder, head, T=2 tying, loss, optimizer, batch
size, epoch budget, patience and Top-5 soup rule are identical to the frozen
MolHIV reference model.

Discipline
----------
* pre-registered in ``notes/molhiv_local_token_null_preregistration.md``
  before any candidate training;
* official train / official valid only; **the official test split is never
  loaded** (there is no freeze/test stage here);
* seed 0 only, exactly one architecture at a time;
* fixed Top-5 equal-weight soup on official valid ROC-AUC, no search.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.molhiv_local_token_null <stage>

Stages: ``params data_sanity smoke train witness decision report``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/molhiv_local_token_null"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
CURVE_DIR = RESULTS_DIR / "curves"

PROTOCOL_VERSION = "molhiv_local_token_null_v1"
PROTOCOL_ID = "molhiv-cross-scaffold-interaction"

#: representations that are allowed to be *trained* in this round
TRAINABLE_REPRESENTATIONS = ("null", "constant")

#: pre-registered gate thresholds on D = typed_soup_seed0 - candidate_soup_seed0
GATE_D_STRONG = 0.005
GATE_D_MILD = 0.020
GATE_D_SUBSTANTIAL = 0.050

EXAMPLE_VOCAB = (32769, 4097)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    rpc._write_json(path, payload)


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------


def typed_reference() -> dict[str, Any]:
    """The frozen MolHIV typed-lookup reference (seed 0)."""
    path = rpc.RESULTS_DIR / "run_seed0.json"
    if not path.exists():
        raise RuntimeError(f"missing frozen typed reference {path}")
    run = rpc._read_json(path)
    return {
        "source": str(path),
        "parameters": int(run["parameters"]),
        "best_epoch": int(run["best_epoch"]),
        "raw_valid_auc": float(run["raw_valid_auc_recomputed"]),
        "soup_valid_auc": float(run["soup_valid_auc"]),
        "top5_epochs": [int(e) for e in run["top5_epochs"]],
        "top5_valid_auc": [float(v) for v in run["top5_valid_auc"]],
    }


# ---------------------------------------------------------------------------
# params / sanity
# ---------------------------------------------------------------------------


def params() -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for representation in ("typed_lookup", "null", "constant"):
        model = rpc.build_model(*EXAMPLE_VOCAB, 0, patch_representation=representation)
        breakdown[representation] = {
            "total_params": rpc._n_params(model),
            "parameter_breakdown": model.parameter_breakdown(),
            "typed_embedding_instantiated": model.typed_embedding is not None,
            "local_token_constant_instantiated": model.local_token_constant is not None,
            "patch_encoder_input_width": int(
                model.patch_encoder.layers[0].in_features
            ),
        }
        del model
    reference = typed_reference()
    ref_run = rpc._read_json(rpc.RESULTS_DIR / "run_seed0.json")
    real_vocab = (
        int(ref_run["typed_vocabulary_size_with_oov"]),
        int(ref_run["parent_vocabulary_size_with_oov"]),
    )
    real: dict[str, Any] = {}
    for representation in ("typed_lookup", "null", "constant"):
        model = rpc.build_model(*real_vocab, 0, patch_representation=representation)
        real[representation] = {
            "total_params": rpc._n_params(model),
            "typed_embedding_instantiated": model.typed_embedding is not None,
            "local_token_constant_instantiated": model.local_token_constant is not None,
        }
        del model
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "architecture": {
            "h_dim": rpc.H_DIM,
            "q_dim": rpc.Q_DIM,
            "token_width": rpc.TOKEN_WIDTH,
            "recurrence_rounds": rpc.RECURRENCE_ROUNDS,
            "loss": "BCEWithLogitsLoss (unweighted)",
            "readout": "unary moments + distance-conditioned pair moments + global encoder",
        },
        "example_vocab": {"typed": EXAMPLE_VOCAB[0], "parent": EXAMPLE_VOCAB[1]},
        "real_vocab_from_reference_seed0": {"typed": real_vocab[0], "parent": real_vocab[1]},
        "conditions": breakdown,
        "conditions_real_vocab": real,
        "typed_reference_seed0": reference,
        "reference_matches_real_vocab": bool(
            real["typed_lookup"]["total_params"] == reference["parameters"]
        ),
        "parameter_drop_vs_reference": {
            representation: int(
                reference["parameters"] - real[representation]["total_params"]
            )
            for representation in ("null", "constant")
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


def data_sanity() -> dict[str, Any]:
    bundle = rpc.load_bundle()
    split = {
        name: np.asarray(bundle.split[name]) for name in ("train", "valid", "test")
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset": "ogbg-molhiv",
        "official_split": "OGB scaffold train/valid/test",
        "sizes": {name: int(values.size) for name, values in split.items()},
        "expected_sizes": dict(rpc.SPLIT_SIZES),
        "expected_sizes_match": all(
            int(split[name].size) == rpc.SPLIT_SIZES[name] for name in rpc.SPLIT_SIZES
        ),
        "positive": {name: int(bundle.y[split[name]].sum()) for name in split},
        "record_cache": str(rpc.RECORD_CACHE),
        "official_test_labels_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "data_sanity.json", payload)
    if not payload["expected_sizes_match"]:
        raise RuntimeError("MolHIV split sizes do not match the official scaffold split")
    return payload


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def smoke(
    representation: str = "null", device: str = "cuda", limit: int = 512
) -> dict[str, Any]:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("smoke requested CUDA but CUDA is unavailable")
    records, _ = rpc._load_records("train")
    transforms = rpc.fit_train_transforms(records[: int(limit)])
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    data = rpc.encode_records(records[: int(limit)], transforms)
    del records
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = rpc.build_model(
        typed_size, parent_size, 0, patch_representation=representation
    ).to(dev)
    loader = rpc._train_loader(data, 0)
    batch = next(iter(loader)).to(dev)
    model.train()
    logits = model(batch)
    target = batch.y.view(-1)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    grads = [
        float(parameter.grad.abs().max())
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "representation": representation,
        "device": str(dev),
        "batch_size": int(target.numel()),
        "loss": float(loss.detach()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_std": float(logits.detach().std()) if logits.numel() > 1 else 0.0,
        "grad_max": max(grads) if grads else None,
        "n_params_with_grad": len(grads),
        "total_params": rpc._n_params(model),
        "typed_embedding_instantiated": model.typed_embedding is not None,
        "local_token_constant_instantiated": model.local_token_constant is not None,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated())
            if dev.type == "cuda"
            else None
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"smoke_{representation}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _run_paths(representation: str, seed: int) -> dict[str, Path]:
    return {
        "run": RESULTS_DIR / f"run_{representation}_seed{seed}.json",
        "raw_state": RESULTS_DIR / f"raw_{representation}_seed{seed}.pt",
        "soup_state": RESULTS_DIR / f"soup_{representation}_seed{seed}.pt",
        "curve": CURVE_DIR / f"{representation}_seed{seed}_curve.json",
        "snapshots": SNAPSHOT_DIR / f"{representation}_seed{seed}",
    }


def train_seed(
    representation: str,
    seed: int = 0,
    device: str = "cuda",
    max_epochs: int | None = None,
    patience: int | None = None,
) -> dict[str, Any]:
    if representation not in TRAINABLE_REPRESENTATIONS:
        raise ValueError(
            f"refusing to train representation={representation!r}; "
            f"allowed: {TRAINABLE_REPRESENTATIONS}"
        )
    max_epochs = int(rpc.MAX_EPOCHS if max_epochs is None else max_epochs)
    patience = int(rpc.PATIENCE if patience is None else patience)
    started = time.perf_counter()
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train requested CUDA but CUDA is unavailable")
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats()

    paths = _run_paths(representation, seed)
    paths["snapshots"].mkdir(parents=True, exist_ok=True)

    train_records, _ = rpc._load_records("train")
    valid_records, _ = rpc._load_records("valid")
    transforms = rpc.fit_train_transforms(train_records)
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    train_data = rpc.encode_records(train_records, transforms)
    valid_data = rpc.encode_records(valid_records, transforms)
    del train_records, valid_records
    gc.collect()

    model = rpc.build_model(
        typed_size, parent_size, seed, patch_representation=representation
    ).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=rpc.LEARNING_RATE, weight_decay=rpc.WEIGHT_DECAY
    )
    train_loader = rpc._train_loader(train_data, seed)
    valid_loader = rpc._valid_loader(valid_data, seed)

    manifest: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    best_auc = -float("inf")
    best_epoch = 1
    stale = 0
    epoch_times: list[float] = []
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(dev)
            target = batch.y.view(-1)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rpc.GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_loss = total_loss / max(seen, 1)
        valid_auc = mpp._evaluate_auc(model, valid_loader, dev)
        epoch_time = float(time.perf_counter() - epoch_started)
        epoch_times.append(epoch_time)
        snap_path = paths["snapshots"] / f"epoch_{epoch:03d}.pt"
        torch.save(copy.deepcopy(model.state_dict()), snap_path)
        manifest.append(
            {
                "epoch": int(epoch),
                "path": str(snap_path),
                "train_loss": float(train_loss),
                "valid_auc": float(valid_auc),
            }
        )
        curve.append(
            {
                "epoch": int(epoch),
                "train_bce": float(train_loss),
                "valid_auc": float(valid_auc),
            }
        )
        if valid_auc > best_auc:
            best_auc = float(valid_auc)
            best_epoch = int(epoch)
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or valid_auc == best_auc:
            print(
                f"[molhiv-ltn {representation} seed{seed}] epoch={epoch:03d} "
                f"bce={train_loss:.6f} valid_auc={valid_auc:.6f} "
                f"best={best_auc:.6f}@{best_epoch}",
                flush=True,
            )
        if stale >= patience:
            print(
                f"[molhiv-ltn {representation} seed{seed}] early_stop epoch={epoch}",
                flush=True,
            )
            break

    top_rows = rpc._topk(manifest, rpc.SOUP_K)
    best_row = rpc._topk(manifest, 1)[0]
    raw_state = torch.load(Path(best_row["path"]), map_location="cpu", weights_only=True)
    soup_state = rpc._soup_state(top_rows)
    valid_targets, raw_logits = rpc._predict(model, raw_state, valid_loader, dev)
    _t, soup_logits = rpc._predict(model, soup_state, valid_loader, dev)
    raw_valid_auc = rpc._auc(valid_targets, raw_logits)
    soup_valid_auc = rpc._auc(valid_targets, soup_logits)

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    paths["curve"].write_text(json.dumps(curve, indent=2), encoding="utf-8")
    reference = typed_reference()
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "representation": representation,
        "seed": int(seed),
        "device": str(dev),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if dev.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "parameters": rpc._n_params(model),
        "typed_vocabulary_size_with_oov": int(typed_size),
        "parent_vocabulary_size_with_oov": int(parent_size),
        "typed_embedding_instantiated": model.typed_embedding is not None,
        "local_token_constant_instantiated": model.local_token_constant is not None,
        "epochs_run": int(len(curve)),
        "early_stopped": bool(len(curve) < max_epochs),
        "best_epoch": int(best_epoch),
        "best_valid_auc": float(best_auc),
        "raw_valid_auc_recomputed": float(raw_valid_auc),
        "soup_valid_auc": float(soup_valid_auc),
        "top5_epochs": [int(row["epoch"]) for row in top_rows],
        "top5_valid_auc": [float(row["valid_auc"]) for row in top_rows],
        "soup_parameters": int(sum(v.numel() for v in soup_state.values())),
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated())
            if dev.type == "cuda"
            else None
        ),
        "typed_reference_seed0": reference,
        "delta_soup_vs_typed_seed0": float(
            reference["soup_valid_auc"] - soup_valid_auc
        ),
        "delta_raw_vs_typed_seed0": float(
            reference["raw_valid_auc"] - raw_valid_auc
        ),
        "curve": curve,
        "snapshot_manifest": manifest,
        "valid_targets": valid_targets.tolist(),
        "raw_valid_logits": raw_logits.tolist(),
        "soup_valid_logits": soup_logits.tolist(),
        "loss": "BCEWithLogitsLoss (unweighted)",
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "official_test_loaded": False,
    }
    _write_json(paths["run"], summary)
    torch.save(raw_state, paths["raw_state"])
    torch.save(soup_state, paths["soup_state"])
    return summary


# ---------------------------------------------------------------------------
# mechanism witness
# ---------------------------------------------------------------------------

_GRAD_GROUPS = (
    "patch_encoder",
    "parent_embedding",
    "pair_projection",
    "relation_encoder",
    "distance_gate",
    "pair_encoder",
    "center_update",
    "global_encoder",
    "head",
    "typed_embedding",
    "local_token_constant",
)


def witness(representation: str, seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    if representation not in ("null", "constant"):
        raise ValueError("witness only supports null/constant")
    dev = torch.device(device)
    paths = _run_paths(representation, seed)
    if not paths["soup_state"].exists():
        raise RuntimeError(f"missing soup state {paths['soup_state']}")

    train_records, _ = rpc._load_records("train")
    valid_records, _ = rpc._load_records("valid")
    transforms = rpc.fit_train_transforms(train_records)
    typed_size = len(transforms["typed_vocabulary"]) + 1
    parent_size = len(transforms["parent_vocabulary"]) + 1
    train_data = rpc.encode_records(train_records[:512], transforms)
    valid_data = rpc.encode_records(valid_records[:512], transforms)
    del train_records, valid_records
    gc.collect()

    model = rpc.build_model(
        typed_size, parent_size, seed, patch_representation=representation
    ).to(dev)
    soup_state = torch.load(paths["soup_state"], map_location="cpu", weights_only=True)
    model.load_state_dict(soup_state, strict=True)

    # token audit
    model.eval()
    valid_loader = rpc._valid_loader(valid_data, seed)
    batch = next(iter(valid_loader)).to(dev)
    with torch.no_grad():
        token = model._patch_token_value(batch)
        logits = model(batch)
    token_norm = token.norm(dim=1)
    token_stats = {
        "shape": list(token.shape),
        "n_patches": int(token.shape[0]),
        "mean_norm": float(token_norm.mean()),
        "max_abs": float(token.abs().max()),
        "exactly_zero": bool(torch.all(token == 0).item()),
        "per_dim_std": float(token.std(dim=0).mean()) if token.numel() else 0.0,
    }

    # gradient audit (one train step of gradients)
    model.train()
    train_loader = rpc._train_loader(train_data, seed)
    train_batch = next(iter(train_loader)).to(dev)
    target = train_batch.y.view(-1)
    loss = F.binary_cross_entropy_with_logits(model(train_batch), target)
    model.zero_grad()
    loss.backward()
    grads: dict[str, float | None] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        value = (
            float(parameter.grad.abs().max())
            if parameter.grad is not None
            else None
        )
        previous = grads.get(group)
        if value is None:
            grads.setdefault(group, None)
        elif previous is None:
            grads[group] = value
        else:
            grads[group] = max(previous, value)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "representation": representation,
        "seed": int(seed),
        "soup_state": str(paths["soup_state"]),
        "total_params": rpc._n_params(model),
        "typed_embedding_instantiated": model.typed_embedding is not None,
        "local_token_constant_instantiated": model.local_token_constant is not None,
        "patch_encoder_input_width": int(model.patch_encoder.layers[0].in_features),
        "token": token_stats,
        "forward_finite": bool(torch.isfinite(logits).all()),
        "logits_std": float(logits.std()) if logits.numel() > 1 else 0.0,
        "grad_max_by_group": grads,
        "all_groups_nonzero": bool(
            all(v is not None and v > 0.0 for v in grads.values())
        ),
        "train_step_bce": float(loss.detach()),
        "official_test_loaded": False,
    }
    if model.local_token_constant is not None:
        constant = model.local_token_constant.detach()
        payload["constant"] = {
            "norm": float(constant.norm()),
            "max_abs": float(constant.abs().max()),
            "per_dim_std": float(constant.std()) if constant.numel() > 1 else 0.0,
        }
    _write_json(RESULTS_DIR / f"witness_{representation}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# decision / report
# ---------------------------------------------------------------------------


def _gate(delta: float) -> str:
    if delta <= GATE_D_STRONG:
        return "STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED"
    if delta <= GATE_D_MILD:
        return "MILD_LOCAL_TOKEN_CHANNEL_NEARLY_REDUNDANT"
    if delta <= GATE_D_SUBSTANTIAL:
        return "SUBSTANTIAL_LOCAL_TOKEN_CHANNEL_CONTRIBUTES"
    return "STOP_LOCAL_TOKEN_CHANNEL_REQUIRED"


def decision(seed: int = 0) -> dict[str, Any]:
    reference = typed_reference()
    path = _run_paths("null", seed)["run"]
    if not path.exists():
        raise RuntimeError(f"missing null run {path}")
    run = rpc._read_json(path)
    delta = float(reference["soup_valid_auc"] - run["soup_valid_auc"])
    gate = _gate(delta)
    authorize_constant = gate in (
        "STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED",
        "MILD_LOCAL_TOKEN_CHANNEL_NEARLY_REDUNDANT",
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "seed": int(seed),
        "typed_reference_seed0_soup": float(reference["soup_valid_auc"]),
        "typed_reference_seed0_raw": float(reference["raw_valid_auc"]),
        "null_soup": float(run["soup_valid_auc"]),
        "null_raw": float(run["raw_valid_auc_recomputed"]),
        "delta_soup": delta,
        "delta_raw": float(reference["raw_valid_auc"] - run["raw_valid_auc_recomputed"]),
        "gate": gate,
        "authorize_constant": bool(authorize_constant),
        "thresholds": {
            "strong": GATE_D_STRONG,
            "mild": GATE_D_MILD,
            "substantial": GATE_D_SUBSTANTIAL,
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report(seed: int = 0) -> dict[str, Any]:
    reference = typed_reference()
    runs = {}
    for representation in TRAINABLE_REPRESENTATIONS:
        path = _run_paths(representation, seed)["run"]
        runs[representation] = rpc._read_json(path) if path.exists() else None
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "seed": int(seed),
        "typed_reference_seed0": reference,
        "runs": runs,
        "decision": (
            rpc._read_json(RESULTS_DIR / "decision.json")
            if (RESULTS_DIR / "decision.json").exists()
            else None
        ),
        "witness": {
            representation: (
                rpc._read_json(RESULTS_DIR / f"witness_{representation}_seed{seed}.json")
                if (RESULTS_DIR / f"witness_{representation}_seed{seed}.json").exists()
                else None
            )
            for representation in TRAINABLE_REPRESENTATIONS
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "params",
            "data_sanity",
            "smoke",
            "train",
            "witness",
            "decision",
            "report",
        ],
    )
    parser.add_argument(
        "--representation", choices=list(TRAINABLE_REPRESENTATIONS), default="null"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    rpc._set_deterministic(bool(args.deterministic))
    if args.stage == "params":
        result = params()
    elif args.stage == "data_sanity":
        result = data_sanity()
    elif args.stage == "smoke":
        result = smoke(args.representation, device=args.device)
    elif args.stage == "train":
        result = train_seed(
            args.representation,
            seed=args.seed,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
    elif args.stage == "witness":
        result = witness(args.representation, seed=args.seed, device=args.device)
    elif args.stage == "decision":
        result = decision(seed=args.seed)
    else:
        result = report(seed=args.seed)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
