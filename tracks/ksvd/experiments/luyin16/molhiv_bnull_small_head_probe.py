"""MolHIV B-Null frozen-representation small-head sufficiency probe.

Question
--------
The frozen MolHIV B-Null seed0 Top-5 soup reaches official-valid ROC-AUC
``0.840847`` with 237,133 parameters.  Its graph head is a
``423 -> 192 -> 96 -> 1`` MLP (100,417 params).  Before proposing any
end-to-end compact MolHIV model, this round asks a cheap, pre-registered
question:

    On the *frozen* pre-head graph representation ``R(G) in R^423``, does a
    single fixed ~14k small head recover essentially the same official-valid
    ROC-AUC as a fresh head with the current architecture?

Stages
------
``extract``  freeze the B-Null soup backbone, extract ``R`` for official train
             and official valid only (never test), record hashes.
``h_refit``  fresh head with the current MolHIV head architecture on frozen R.
``h_small``  one pre-registered ``423 -> 32 -> 16 -> 1`` head on frozen R.
``gate``     small-head sufficiency gate against ``H_refit``.
``report``   consolidated payload.

Discipline: seed0 only, BCEWithLogits, Adam lr 1e-3, weight decay 1e-5,
batch 128, max 240 epochs, patience 40, fixed Top-5 soup, official valid only.
Official test is never loaded and no test representation is extracted.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.molhiv_bnull_small_head_probe <stage>
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import molhiv_local_token_null as mltn
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc

# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/molhiv_bnull_small_head_probe"
PROTOCOL_VERSION = "molhiv_bnull_small_head_probe_v1"
PROTOCOL_ID = "molhiv-cross-scaffold-interaction"

BNULL_RESULTS_DIR = TRACK_ROOT / "results/molhiv_local_token_null"
BNULL_SOUP_PATH = BNULL_RESULTS_DIR / "soup_null_seed0.pt"
BNULL_RUN_PATH = BNULL_RESULTS_DIR / "run_null_seed0.json"

R_TRAIN_PATH = RESULTS_DIR / "R_train.pt"
R_VALID_PATH = RESULTS_DIR / "R_valid.pt"
R_META_PATH = RESULTS_DIR / "R_extraction.json"

SEED = 0
BATCH_SIZE = 128
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
MAX_EPOCHS = 240
PATIENCE = 40
SOUP_K = 5
DROPOUT = rpc.DROPOUT

HEAD_REFIT_INPUT_DIM = 423
HEAD_REFIT_HIDDEN = 192

# pre-registered gate thresholds
HEAD_PROBE_INVALID_MARGIN = 0.010
SMALL_HEAD_STRONG = 0.005
SMALL_HEAD_MILD = 0.015
SMALL_HEAD_NO_GO = 0.020


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    head = REPO_ROOT / ".git" / "HEAD"
    if head.exists():
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = REPO_ROOT / ".git" / text.split(" ", 1)[1].strip()
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        return text
    return "unknown"


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()
    ).hexdigest()


def _n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _auc(targets: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(targets, logits))


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------


def build_h_refit(in_dim: int = HEAD_REFIT_INPUT_DIM) -> nn.Module:
    """Exactly the current MolHIV graph head architecture."""
    hidden = max(int(rpc.H_DIM) * 2, 96)
    return nn.Sequential(
        nn.Linear(int(in_dim), hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Dropout(float(DROPOUT)),
        nn.Linear(hidden, int(rpc.H_DIM)),
        nn.ReLU(),
        nn.Linear(int(rpc.H_DIM), 1),
    )


def build_h_small(in_dim: int = HEAD_REFIT_INPUT_DIM) -> nn.Module:
    """The single pre-registered ``in -> 32 -> 16 -> 1`` small head."""
    return nn.Sequential(
        nn.Linear(int(in_dim), 32),
        nn.LayerNorm(32),
        nn.ReLU(),
        nn.Dropout(float(DROPOUT)),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


HEADS: dict[str, Callable[[], nn.Module]] = {
    "H_refit": build_h_refit,
    "H_small32": build_h_small,
}


def params() -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "R_dim": int(HEAD_REFIT_INPUT_DIM),
        "heads": {
            name: int(_n_params(builder()))
            for name, builder in HEADS.items()
        },
        "original_bnull_total_params": None,
        "official_test_loaded": False,
    }
    if BNULL_RUN_PATH.exists():
        payload["original_bnull_total_params"] = int(
            _read_json(BNULL_RUN_PATH)["parameters"]
        )
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def _build_bnull_soup_model(device: torch.device):
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
        typed_size, parent_size, SEED, patch_representation="null"
    ).to(device)
    soup_state = torch.load(BNULL_SOUP_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(soup_state, strict=True)
    model.eval()
    return model, train_data, valid_data, soup_state


@torch.no_grad()
def _extract_R_and_y(model: nn.Module, loader, device: torch.device):
    """Extract frozen ``R`` and labels in a single ordered pass.

    ``R`` and ``y`` must come from the same iteration: the train loader
    shuffles with a stateful generator, so a second pass would reorder the
    molecules and silently misalign representation and label.
    """
    captured: list[torch.Tensor] = []
    representations: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    def hook(_module, args):
        captured.append(args[0].detach().to("cpu"))

    handle = model.head.register_forward_pre_hook(hook)
    model.eval()
    try:
        for batch in loader:
            captured.clear()
            model(batch.to(device))
            representations.append(captured[0])
            targets.append(batch.y.view(-1).detach().to("cpu"))
    finally:
        handle.remove()
    return torch.cat(representations, dim=0), torch.cat(targets, dim=0)


def extract(device: str = "cuda") -> dict[str, Any]:
    if not BNULL_SOUP_PATH.exists():
        raise FileNotFoundError(f"missing frozen B-Null soup {BNULL_SOUP_PATH}")
    dev = torch.device(device)
    started = time.perf_counter()
    model, train_data, valid_data, soup_state = _build_bnull_soup_model(dev)
    train_loader = rpc._train_loader(train_data, SEED)
    valid_loader = rpc._valid_loader(valid_data, SEED)

    # identity check: frozen head applied to R must reproduce model logits
    sample = next(iter(valid_loader)).to(dev)
    captured: list[torch.Tensor] = []

    def hook(_module, args):
        captured.append(args[0].detach())

    handle = model.head.register_forward_pre_hook(hook)
    with torch.no_grad():
        logits = model(sample)
    handle.remove()
    head_logits = model.head(captured[0]).view(-1)
    identity_max_abs = float((logits.view(-1) - head_logits).abs().max())

    R_train, y_train = _extract_R_and_y(model, train_loader, dev)
    R_valid, y_valid = _extract_R_and_y(model, valid_loader, dev)
    del train_data, valid_data
    gc.collect()

    if int(R_train.shape[1]) != HEAD_REFIT_INPUT_DIM:
        raise RuntimeError(
            f"unexpected frozen representation width {int(R_train.shape[1])}; "
            f"expected {HEAD_REFIT_INPUT_DIM}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"R": R_train, "y": y_train}, R_TRAIN_PATH)
    torch.save({"R": R_valid, "y": y_valid}, R_VALID_PATH)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "extract",
        "git_commit": _git_commit(),
        "device": str(dev),
        "R_dim": int(R_train.shape[1]),
        "train_shape": [int(v) for v in R_train.shape],
        "valid_shape": [int(v) for v in R_valid.shape],
        "train_positive": int(y_train.sum()),
        "valid_positive": int(y_valid.sum()),
        "R_train_sha256": _sha256_tensor(R_train),
        "R_valid_sha256": _sha256_tensor(R_valid),
        "y_train_sha256": _sha256_tensor(y_train),
        "y_valid_sha256": _sha256_tensor(y_valid),
        "bnull_soup_sha256": hashlib.sha256(BNULL_SOUP_PATH.read_bytes()).hexdigest(),
        "frozen_head_reproduces_logits_max_abs": identity_max_abs,
        "backbone_frozen": True,
        "official_test_representation_extracted": False,
        "official_test_loaded": False,
        "wall_clock_s": float(time.perf_counter() - started),
    }
    _write_json(R_META_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# head-only training on frozen R
# ---------------------------------------------------------------------------


def _head_forward(head: nn.Module, R: torch.Tensor) -> torch.Tensor:
    return head(R).view(-1)


def _valid_auc(head: nn.Module, R: torch.Tensor, y: torch.Tensor, device) -> float:
    head.eval()
    with torch.no_grad():
        logits = []
        for start in range(0, R.shape[0], BATCH_SIZE):
            batch = R[start:start + BATCH_SIZE].to(device)
            logits.append(_head_forward(head, batch).cpu())
        logits = torch.cat(logits).numpy().astype(np.float64)
    return _auc(y.numpy().astype(np.float64), logits)


def train_head(
    name: str,
    device: str = "cuda",
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    if name not in HEADS:
        raise ValueError(f"unknown head {name!r}")
    train_payload = torch.load(R_TRAIN_PATH, map_location="cpu", weights_only=True)
    valid_payload = torch.load(R_VALID_PATH, map_location="cpu", weights_only=True)
    R_train, y_train = train_payload["R"].float(), train_payload["y"].float()
    R_valid, y_valid = valid_payload["R"].float(), valid_payload["y"].float()

    dev = torch.device(device)
    torch.manual_seed(SEED)
    head = HEADS[name]().to(dev)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    n = int(R_train.shape[0])
    snapshot_dir = RESULTS_DIR / f"snapshots_{name}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    best_auc = -float("inf")
    best_epoch = 1
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        head.train()
        generator = torch.Generator().manual_seed(SEED * 100003 + epoch)
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, n, BATCH_SIZE):
            index = permutation[start:start + BATCH_SIZE]
            batch = R_train[index].to(dev)
            target = y_train[index].to(dev)
            logits = _head_forward(head, batch)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), rpc.GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        train_loss = total_loss / max(seen, 1)
        valid_auc = _valid_auc(head, R_valid, y_valid, dev)
        snap_path = snapshot_dir / f"epoch_{epoch:03d}.pt"
        torch.save(copy.deepcopy(head.state_dict()), snap_path)
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
                f"[{name} seed{SEED}] epoch={epoch:03d} bce={train_loss:.6f} "
                f"valid_auc={valid_auc:.6f} best={best_auc:.6f}@{best_epoch}",
                flush=True,
            )
        if stale >= int(patience):
            print(f"[{name} seed{SEED}] early_stop epoch={epoch}", flush=True)
            break

    top_rows = rpc._topk(manifest, SOUP_K)
    best_row = rpc._topk(manifest, 1)[0]
    raw_state = torch.load(Path(best_row["path"]), map_location="cpu", weights_only=True)
    soup_state = rpc._soup_state(top_rows)

    def _predict_state(state):
        head.load_state_dict(state, strict=True)
        head.eval()
        with torch.no_grad():
            logits = []
            for start in range(0, R_valid.shape[0], BATCH_SIZE):
                logits.append(
                    _head_forward(head, R_valid[start:start + BATCH_SIZE].to(dev)).cpu()
                )
        return torch.cat(logits).numpy().astype(np.float64)

    raw_logits = _predict_state(raw_state)
    soup_logits = _predict_state(soup_state)
    y_np = y_valid.numpy().astype(np.float64)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "head": name,
        "seed": SEED,
        "device": str(dev),
        "parameters": int(_n_params(head)),
        "epochs_run": int(len(curve)),
        "early_stopped": bool(len(curve) < int(max_epochs)),
        "best_epoch": int(best_epoch),
        "best_valid_auc": float(best_auc),
        "raw_valid_auc_recomputed": _auc(y_np, raw_logits),
        "soup_valid_auc": _auc(y_np, soup_logits),
        "top5_epochs": [int(row["epoch"]) for row in top_rows],
        "top5_valid_auc": [float(row["valid_auc"]) for row in top_rows],
        "valid_targets": y_np.tolist(),
        "raw_valid_logits": raw_logits.tolist(),
        "soup_valid_logits": soup_logits.tolist(),
        "curve": curve,
        "wall_clock_s": float(time.perf_counter() - started),
        "loss": "BCEWithLogitsLoss (unweighted)",
        "optimizer": f"Adam lr={LEARNING_RATE} weight_decay={WEIGHT_DECAY}",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"run_{name}_seed{SEED}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# gate / report
# ---------------------------------------------------------------------------


def gate() -> dict[str, Any]:
    refit = _read_json(RESULTS_DIR / f"run_H_refit_seed{SEED}.json")
    small = _read_json(RESULTS_DIR / f"run_H_small32_seed{SEED}.json")
    original = float(_read_json(BNULL_RUN_PATH)["soup_valid_auc"])
    refit_auc = float(refit["soup_valid_auc"])
    small_auc = float(small["soup_valid_auc"])
    delta = refit_auc - small_auc
    probe_margin = original - refit_auc
    if probe_margin > HEAD_PROBE_INVALID_MARGIN:
        probe_gate = "HEAD_PROBE_INVALID"
        small_gate = "NOT_EVALUATED"
    else:
        probe_gate = "HEAD_PROBE_VALID"
        if delta <= SMALL_HEAD_STRONG:
            small_gate = "STRONG_SMALL_HEAD_SUFFICIENCY"
        elif delta <= SMALL_HEAD_MILD:
            small_gate = "MILD_SMALL_HEAD_SIGNAL"
        elif delta <= SMALL_HEAD_NO_GO:
            small_gate = "BORDERLINE_SMALL_HEAD_SIGNAL"
        else:
            small_gate = "SMALL_HEAD_NO_GO"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "original_bnull_soup_auc": original,
        "H_refit_params": int(refit["parameters"]),
        "H_refit_soup_auc": refit_auc,
        "H_small32_params": int(small["parameters"]),
        "H_small32_soup_auc": small_auc,
        "delta_auc_refit_minus_small": float(delta),
        "refit_minus_original_margin": float(probe_margin),
        "probe_gate": probe_gate,
        "small_head_gate": small_gate,
        "thresholds": {
            "head_probe_invalid_margin": HEAD_PROBE_INVALID_MARGIN,
            "strong": SMALL_HEAD_STRONG,
            "mild": SMALL_HEAD_MILD,
            "no_go": SMALL_HEAD_NO_GO,
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "gate.json", payload)
    return payload


def report() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "git_commit": _git_commit(),
        "files": {},
        "official_test_loaded": False,
    }
    for name in (
        "parameter_accounting.json",
        "R_extraction.json",
        "gate.json",
    ):
        path = RESULTS_DIR / name
        if path.exists():
            payload["files"][name] = _read_json(path)
    for name in ("H_refit", "H_small32"):
        path = RESULTS_DIR / f"run_{name}_seed{SEED}.json"
        if path.exists():
            run = _read_json(path)
            payload["files"][f"run_{name}"] = {
                key: run[key]
                for key in (
                    "head",
                    "parameters",
                    "best_epoch",
                    "best_valid_auc",
                    "raw_valid_auc_recomputed",
                    "soup_valid_auc",
                    "top5_epochs",
                    "top5_valid_auc",
                    "epochs_run",
                    "wall_clock_s",
                )
                if key in run
            }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["params", "extract", "h_refit", "h_small", "gate", "report"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args(argv)

    if args.stage == "params":
        result = params()
    elif args.stage == "extract":
        result = extract(device=args.device)
    elif args.stage == "h_refit":
        result = train_head(
            "H_refit", device=args.device, max_epochs=args.max_epochs, patience=args.patience
        )
    elif args.stage == "h_small":
        result = train_head(
            "H_small32", device=args.device, max_epochs=args.max_epochs, patience=args.patience
        )
    elif args.stage == "gate":
        result = gate()
    else:
        result = report()
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
