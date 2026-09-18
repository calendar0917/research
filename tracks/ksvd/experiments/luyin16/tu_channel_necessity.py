"""TU graph-classification channel-necessity screen.

Pre-registered port of the ZINC/MolHIV local-token-null question to TU:

    is the graph-dependent local patch-token channel (the typed certificate
    lookup) necessary, or does the fixed downstream backbone carry almost all
    of the predictive power?

Conditions (only the local-token channel changes):
    ``typed_lookup``  a learned row per exact rooted-patch certificate
    ``null``          exact zero token per patch, no generator
    ``constant``      one trainable graph-wide vector shared by every patch

Strict split protocol (frozen in ``splits.json`` before any training):
    outer stratified k-fold; test fold never used for training or selection;
    train/valid carved only from the non-test folds; identical splits for every
    condition; vocabularies and standardizers fit on the train fold only;
    the test fold is evaluated once per fold after best-valid + Top-5 soup are
    frozen.

Stages: ``prep smoke run report``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import tu_patch_path_pooling as tu

REPO_ROOT = tu.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tu_channel_necessity"
CACHE_DIR = RESULTS_DIR / "cache"
RUNS_DIR = RESULTS_DIR / "runs"

PROTOCOL_VERSION = "tu_channel_necessity_v1"
DATASETS = ("MUTAG", "PROTEINS", "IMDB-BINARY")


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
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------


def _records_path(name: str) -> Path:
    return CACHE_DIR / f"{name}_records.pkl"


def _splits_path(name: str, k: int) -> Path:
    return RESULTS_DIR / f"splits_{name}_k{int(k)}.json"


def prep_dataset(name: str, k: int) -> dict[str, Any]:
    graphs, y, meta = tu.load_dataset(name)
    cache: dict[tuple[int, int, bytes], bytes] = {}
    records = [
        tu.extract_graph(g, meta["n_node_types"], meta["n_edge_types"], cache)
        for g in graphs
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _records_path(name).open("wb") as handle:
        pickle.dump({"meta": meta, "y": y, "records": records}, handle)
    folds = tu.build_folds(y, int(k))
    split_checks = tu.check_folds(folds, len(graphs))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(name),
        "k": int(k),
        "meta": meta,
        "split_seed": tu.SPLIT_SEED,
        "valid_frac": tu.VALID_FRAC,
        "splits_fingerprint": tu.folds_fingerprint(folds),
        "split_checks": split_checks,
        "folds": [
            {
                "fold": int(f["fold"]),
                "train": [int(v) for v in f["train"]],
                "valid": [int(v) for v in f["valid"]],
                "test": [int(v) for v in f["test"]],
            }
            for f in folds
        ],
        "test_never_used_for_selection": True,
        "official_test_loaded": False,
    }
    _write_json(_splits_path(name, k), payload)
    if not (split_checks["all_disjoint"] and split_checks["each_graph_tested_once"]):
        raise RuntimeError(f"strict split checks failed for {name}: {split_checks}")
    return payload


def _load_records(name: str) -> tuple[dict[str, Any], np.ndarray, list[Any]]:
    with _records_path(name).open("rb") as handle:
        blob = pickle.load(handle)
    return blob["meta"], blob["y"], blob["records"]


# ---------------------------------------------------------------------------
# one fold
# ---------------------------------------------------------------------------


def _select(records: Sequence[Any], indices: np.ndarray) -> list[Any]:
    return [records[int(i)] for i in indices]


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _soup(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = list(states[0].keys())
    return {
        key: torch.stack([state[key].float() for state in states], dim=0)
        .mean(dim=0)
        .to(states[0][key].dtype)
        for key in keys
    }


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        targets.extend(batch.y.view(-1).cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)


def _accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float((targets == predictions).mean())


def run_fold(
    name: str,
    representation: str,
    fold: int,
    k: int,
    *,
    device: str = "cpu",
    max_epochs: int | None = None,
    patience: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if representation not in tu.TRAINABLE_REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    max_epochs = int(tu.MAX_EPOCHS if max_epochs is None else max_epochs)
    patience = int(tu.PATIENCE if patience is None else patience)
    dev = torch.device(device)
    started = time.perf_counter()

    meta, y, records = _load_records(name)
    splits = _read_json(_splits_path(name, k))
    fold_row = next(f for f in splits["folds"] if int(f["fold"]) == int(fold))
    train_idx = np.asarray(fold_row["train"], dtype=np.int64)
    valid_idx = np.asarray(fold_row["valid"], dtype=np.int64)
    test_idx = np.asarray(fold_row["test"], dtype=np.int64)

    train_records = _select(records, train_idx)
    valid_records = _select(records, valid_idx)
    test_records = _select(records, test_idx)

    typed_vocab = tu.fit_vocabulary(train_records, "typed_certificate")
    parent_vocab = tu.fit_vocabulary(train_records, "parent_certificate")
    patch_std = tu.Standardizer.fit(
        np.stack([p.shell_descriptor for r in train_records for p in r.patches], axis=0)
    )
    global_std = tu.Standardizer.fit(
        np.stack([r.global_context for r in train_records], axis=0)
    )
    train_data = tu.encode_records(
        train_records, typed_vocab, parent_vocab, patch_std, global_std
    )
    valid_data = tu.encode_records(
        valid_records, typed_vocab, parent_vocab, patch_std, global_std
    )
    test_data = tu.encode_records(
        test_records, typed_vocab, parent_vocab, patch_std, global_std
    )

    torch.manual_seed(int(seed))
    model = tu.TuPatchPathModel(
        len(typed_vocab) + 1,
        len(parent_vocab) + 1,
        shell_width=tu.shell_width(meta["n_node_types"], meta["n_edge_types"]),
        relation_width=tu.relation_width(meta["n_edge_types"]),
        n_classes=meta["n_classes"],
        patch_representation=representation,
    ).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tu.LEARNING_RATE, weight_decay=tu.WEIGHT_DECAY
    )
    train_loader = tu.make_loader(train_data, tu.BATCH_SIZE, True, int(seed) + 1)
    valid_loader = tu.make_loader(valid_data, tu.BATCH_SIZE, False, 0)
    test_loader = tu.make_loader(test_data, tu.BATCH_SIZE, False, 0)

    manifest: list[dict[str, Any]] = []
    best_acc = -1.0
    best_epoch = 1
    stale = 0
    curve: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(dev)
            logits = model(batch)
            target = batch.y.view(-1)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tu.GRAD_CLIP)
            optimizer.step()
            total += float(loss.detach()) * int(target.numel())
            seen += int(target.numel())
        valid_targets, valid_pred = _predict(model, _state_dict_cpu(model), valid_loader, dev)
        valid_acc = _accuracy(valid_targets, valid_pred)
        curve.append(
            {
                "epoch": int(epoch),
                "train_ce": float(total / max(seen, 1)),
                "valid_accuracy": float(valid_acc),
            }
        )
        manifest.append(
            {
                "epoch": int(epoch),
                "state": _state_dict_cpu(model),
                "valid_accuracy": float(valid_acc),
            }
        )
        if valid_acc > best_acc:
            best_acc = float(valid_acc)
            best_epoch = int(epoch)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    top = sorted(
        manifest, key=lambda row: (-float(row["valid_accuracy"]), int(row["epoch"]))
    )[: tu.SOUP_K]
    raw_state = next(
        row["state"] for row in manifest if int(row["epoch"]) == int(best_epoch)
    )
    soup_state = _soup([row["state"] for row in top])

    valid_targets, raw_valid_pred = _predict(model, raw_state, valid_loader, dev)
    _t, soup_valid_pred = _predict(model, soup_state, valid_loader, dev)
    test_targets, raw_test_pred = _predict(model, raw_state, test_loader, dev)
    _t2, soup_test_pred = _predict(model, soup_state, test_loader, dev)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(name),
        "representation": representation,
        "fold": int(fold),
        "k": int(k),
        "device": str(dev),
        "n_params": model.n_params(),
        "typed_vocabulary_size_with_oov": len(typed_vocab) + 1,
        "parent_vocabulary_size_with_oov": len(parent_vocab) + 1,
        "epochs_run": int(len(curve)),
        "best_epoch": int(best_epoch),
        "best_valid_accuracy": float(best_acc),
        "raw_valid_accuracy": _accuracy(valid_targets, raw_valid_pred),
        "soup_valid_accuracy": _accuracy(valid_targets, soup_valid_pred),
        "raw_test_accuracy": _accuracy(test_targets, raw_test_pred),
        "soup_test_accuracy": _accuracy(test_targets, soup_test_pred),
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_accuracy": [float(row["valid_accuracy"]) for row in top],
        "wall_clock_s": float(time.perf_counter() - started),
        "curve": curve,
        "test_never_used_for_selection": True,
        "official_test_loaded": False,
        "splits_fingerprint": splits["splits_fingerprint"],
        "git_commit": _git_commit(),
    }
    _write_json(RUNS_DIR / f"{name}_{representation}_fold{fold}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# smoke / report
# ---------------------------------------------------------------------------


def smoke(
    name: str = "MUTAG",
    k: int = 3,
    folds: int = 1,
    epochs: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    prep_dataset(name, k)
    rows: list[dict[str, Any]] = []
    for representation in tu.TRAINABLE_REPRESENTATIONS:
        for fold in range(int(folds)):
            row = run_fold(
                name,
                representation,
                fold,
                k,
                device=device,
                max_epochs=int(epochs),
                patience=int(epochs),
            )
            rows.append(
                {
                    "representation": representation,
                    "fold": fold,
                    "n_params": row["n_params"],
                    "soup_valid_accuracy": row["soup_valid_accuracy"],
                    "soup_test_accuracy": row["soup_test_accuracy"],
                    "epochs_run": row["epochs_run"],
                }
            )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "smoke": True,
        "dataset": name,
        "k": k,
        "folds": folds,
        "epochs": epochs,
        "rows": rows,
        "splits": _read_json(_splits_path(name, k))["split_checks"],
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / f"smoke_{name}.json", payload)
    return payload


def majority_baseline(name: str, k: int) -> dict[str, Any]:
    """Train-fold-majority classifier applied to each held-out test fold."""
    _meta, y, _records = _load_records(name)
    splits = _read_json(_splits_path(name, k))
    per_fold: list[dict[str, Any]] = []
    for fold_row in splits["folds"]:
        train = np.asarray(fold_row["train"], dtype=np.int64)
        test = np.asarray(fold_row["test"], dtype=np.int64)
        counts = np.bincount(y[train], minlength=int(y.max()) + 1)
        majority = int(counts.argmax())
        accuracy = float((y[test] == majority).mean())
        per_fold.append(
            {
                "fold": int(fold_row["fold"]),
                "majority_class": majority,
                "test_accuracy": accuracy,
            }
        )
    values = np.asarray([row["test_accuracy"] for row in per_fold], dtype=np.float64)
    return {
        "n_folds": int(len(per_fold)),
        "test_mean": float(values.mean()),
        "test_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "per_fold": per_fold,
    }


def screen(
    k: int,
    datasets: Sequence[str] = DATASETS,
    *,
    device: str = "cpu",
    epochs: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run every dataset x condition x fold, skipping finished runs."""
    rows: list[dict[str, Any]] = []
    for name in datasets:
        prep_dataset(name, int(k))
        for representation in tu.TRAINABLE_REPRESENTATIONS:
            for fold in range(int(k)):
                path = RUNS_DIR / f"{name}_{representation}_fold{fold}.json"
                if resume and path.exists():
                    payload = _read_json(path)
                else:
                    payload = run_fold(
                        name, representation, fold, int(k),
                        device=device, max_epochs=epochs,
                    )
                rows.append(
                    {
                        "dataset": name,
                        "representation": representation,
                        "fold": int(fold),
                        "n_params": payload["n_params"],
                        "soup_valid_accuracy": payload["soup_valid_accuracy"],
                        "soup_test_accuracy": payload["soup_test_accuracy"],
                        "raw_test_accuracy": payload["raw_test_accuracy"],
                        "epochs_run": payload["epochs_run"],
                    }
                )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "k": int(k),
        "rows": rows,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }


def report(k: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in DATASETS:
        if not _splits_path(name, k).exists():
            continue
        per_condition: dict[str, Any] = {}
        for representation in tu.TRAINABLE_REPRESENTATIONS:
            rows = []
            for fold in range(int(k)):
                path = RUNS_DIR / f"{name}_{representation}_fold{fold}.json"
                if path.exists():
                    rows.append(_read_json(path))
            if not rows:
                continue
            raw = np.asarray([r["raw_test_accuracy"] for r in rows], dtype=np.float64)
            soup = np.asarray([r["soup_test_accuracy"] for r in rows], dtype=np.float64)
            valid = np.asarray([r["soup_valid_accuracy"] for r in rows], dtype=np.float64)
            per_condition[representation] = {
                "n_folds": int(len(rows)),
                "params": int(rows[0]["n_params"]),
                "valid_soup_mean": float(valid.mean()),
                "test_raw_mean": float(raw.mean()),
                "test_raw_std": float(raw.std(ddof=1)) if len(raw) > 1 else 0.0,
                "test_soup_mean": float(soup.mean()),
                "test_soup_std": float(soup.std(ddof=1)) if len(soup) > 1 else 0.0,
            }
        typed = per_condition.get("typed_lookup")
        null = per_condition.get("null")
        delta = None
        if typed and null:
            delta = {
                "D_typed_minus_null_soup": float(
                    typed["test_soup_mean"] - null["test_soup_mean"]
                ),
                "D_typed_minus_null_raw": float(
                    typed["test_raw_mean"] - null["test_raw_mean"]
                ),
            }
        summary[name] = {
            "per_condition": per_condition,
            "majority_baseline": majority_baseline(name, k),
            "channel_necessity_delta": delta,
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "k": int(k),
        "datasets": summary,
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / f"report_k{k}.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prep", "smoke", "run", "screen", "report"])
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--representation", default="typed_lookup")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.stage == "prep":
        result = prep_dataset(args.dataset, args.k)
    elif args.stage == "smoke":
        result = smoke(
            args.dataset, k=args.k, folds=args.folds, epochs=(args.epochs or 3),
            device=args.device,
        )
    elif args.stage == "run":
        result = run_fold(
            args.dataset,
            args.representation,
            args.fold,
            args.k,
            device=args.device,
            max_epochs=args.epochs,
            seed=args.seed,
        )
    elif args.stage == "screen":
        result = screen(
            args.k,
            datasets=DATASETS,
            device=args.device,
            epochs=args.epochs,
            resume=True,
        )
    else:
        result = report(args.k)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
