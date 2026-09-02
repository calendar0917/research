"""Multi-bank consensus occurrence residual on a frozen prototype-MIL ensemble.

Three audited graph-balanced real-prototype MIL members form the frozen base by
arithmetic mean of probabilities.  Each bank independently measures connected
prototype-support shapes.  Prototype rows are canonicalized without retaining
bank-specific identity and averaged across banks before a single low-capacity,
zero-gated graph-level residual is trained.

This runner is restricted to official-train scaffold folds.  Official valid and
test remain unencoded and unevaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import load_molhiv
from code.run_molhiv_prototype_occurrence_residual import (
    normalize_rows, occurrence_features, prediction_arrays, seed_all,
    sha256, sigmoid_np, sparse_positive_codes, transform_features,
)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--base-results", required=True, help="comma-separated stable member JSONs")
    ap.add_argument("--prototype-seeds", required=True, help="comma-separated seeds aligned to base results")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controls", default="consensus_no_id,consensus_shuffled,global_stats")
    ap.add_argument("--candidate-bank-size", type=int, default=256)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--residual-cap", type=float, default=0.25)
    ap.add_argument("--residual-epochs", type=int, default=20)
    ap.add_argument("--residual-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--residual-l2", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--shuffle-seed", type=int, default=314159)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_paths = [Path(x.strip()) for x in args.base_results.split(",") if x.strip()]
    prototype_seeds = [int(x.strip()) for x in args.prototype_seeds.split(",") if x.strip()]
    controls = [x.strip() for x in args.controls.split(",") if x.strip()]
    known = {"consensus_no_id", "consensus_shuffled", "global_stats"}
    if len(base_paths) < 2 or len(base_paths) != len(prototype_seeds):
        raise ValueError("base-results/prototype-seeds must have the same length >= 2")
    if not controls or len(set(controls)) != len(controls) or not set(controls) <= known:
        raise ValueError(f"invalid controls: {controls}")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(f"missing training dependencies: {exc}") from exc

    t0 = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=False,
    )
    labels = np.asarray(bundle.y, dtype=np.float32)
    original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)
    with np.load(args.fold_cache, allow_pickle=False) as f:
        fit_indices = np.asarray(f[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(f[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if not np.array_equal(np.asarray(f["original_indices"], dtype=np.int64), original):
            raise ValueError("fold cache original mismatch")
        if not np.array_equal(np.asarray(f["official_train_indices"], dtype=np.int64), official_train):
            raise ValueError("fold cache official train mismatch")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("fit/heldout overlap")

    with np.load(args.latent_cache, allow_pickle=False) as f:
        offsets = np.asarray(f["offsets"], dtype=np.int64)
        latents = np.asarray(f["latents"], dtype=np.float32)
        if not np.array_equal(np.asarray(f["original_indices"], dtype=np.int64), original):
            raise ValueError("latent original mismatch")
        if not np.array_equal(np.asarray(f["train_indices"], dtype=np.int64), official_train):
            raise ValueError("latent train mismatch")
    with np.load(args.token_cache, allow_pickle=False) as f:
        if not np.array_equal(np.asarray(f["offsets"], dtype=np.int64), offsets):
            raise ValueError("token/latent offset mismatch")
    nontrain_rows = np.concatenate([
        np.arange(int(offsets[i]), int(offsets[i + 1]), dtype=np.int64)
        for i in np.concatenate([official_valid, official_test])
    ])
    if np.any(latents[nontrain_rows] != 0):
        raise AssertionError("official valid/test latents are encoded")
    token_meta = json.loads(Path(args.token_cache).with_suffix(".json").read_text())
    latent_meta = json.loads(Path(args.latent_cache).with_suffix(".json").read_text())
    if token_meta.get("dictionary_fit_indices_sha256") != sha256(fit_indices):
        raise ValueError("token cache outer-fit mismatch")
    if bool(token_meta.get("encoded_official_valid", True)):
        raise ValueError("token cache encoded official valid")
    if bool(latent_meta.get("official_valid_encoded", True)) or bool(latent_meta.get("official_test_encoded", True)):
        raise ValueError("latent cache encoded official valid/test")

    bases = [json.loads(path.read_text()) for path in base_paths]
    member_fit_prob: list[np.ndarray] = []
    member_held_prob: list[np.ndarray] = []
    member_raw: list[np.ndarray] = []
    common_fit_y: np.ndarray | None = None
    common_held_y: np.ndarray | None = None

    all_graphs = np.concatenate([fit_indices, heldout_indices])
    for bank_i, (path, prototype_seed, base) in enumerate(zip(base_paths, prototype_seeds, bases)):
        cfg = base.get("config", {})
        expected_cfg = {
            "fold": args.fold, "prototype_seed": prototype_seed, "seed": 0,
            "max_graphs": args.max_graphs, "candidate_bank_size": args.candidate_bank_size,
            "n_prototypes": args.n_prototypes, "sparsity": args.sparsity,
        }
        for key, expected in expected_cfg.items():
            if cfg.get(key) != expected:
                raise ValueError(f"{path}: config mismatch {key}")
        fit_graphs, fit_y, fit_logits = prediction_arrays(base, "fit")
        held_graphs, held_y, held_logits = prediction_arrays(base, "heldout")
        if not np.array_equal(fit_graphs, fit_indices) or not np.array_equal(held_graphs, heldout_indices):
            raise ValueError(f"{path}: prediction order mismatch")
        if common_fit_y is None:
            common_fit_y, common_held_y = fit_y, held_y
        elif not np.array_equal(common_fit_y, fit_y) or not np.array_equal(common_held_y, held_y):
            raise ValueError("member labels mismatch")
        member_fit_prob.append(sigmoid_np(fit_logits))
        member_held_prob.append(sigmoid_np(held_logits))

        rng = np.random.default_rng(prototype_seed + 910_003 * (args.fold + 1))
        source_pos = rng.choice(len(fit_indices), size=args.candidate_bank_size, replace=False).astype(np.int64)
        source_graphs = fit_indices[source_pos]
        candidate_rows = np.asarray([
            rng.integers(int(offsets[i]), int(offsets[i + 1])) for i in source_graphs
        ], dtype=np.int64)
        candidate_bank = normalize_rows(latents[candidate_rows])
        selected = rng.choice(args.candidate_bank_size, size=args.n_prototypes, replace=False).astype(np.int64)
        prototypes = candidate_bank[selected]
        selection = base["results"]["fixed_random"]["selection"]
        if selection.get("candidate_indices") != selected.tolist() or selection.get("prototype_sha256") != sha256(prototypes):
            raise ValueError(f"{path}: reconstructed prototype bank mismatch")

        rows: list[np.ndarray] = []
        for j, raw_i in enumerate(all_graphs.tolist()):
            i = int(raw_i)
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            codes = sparse_positive_codes(latents[lo:hi], prototypes, args.sparsity)
            rows.append(occurrence_features(bundle.graphs[i], codes))
        raw_np = np.asarray(rows, dtype=np.float32)
        member_raw.append(raw_np)
        print(f"bank {bank_i + 1}/{len(bases)} features complete: seed={prototype_seed}", flush=True)

    assert common_fit_y is not None and common_held_y is not None
    fit_y = common_fit_y
    held_y = common_held_y
    base_fit_prob = np.mean(np.stack(member_fit_prob, axis=0), axis=0)
    base_held_prob = np.mean(np.stack(member_held_prob, axis=0), axis=0)
    base_fit_logits = logit_np(base_fit_prob).astype(np.float32)
    base_held_logits = logit_np(base_held_prob).astype(np.float32)
    base_auc = float(roc_auc_score(held_y, base_held_prob))
    n_fit = len(fit_indices)

    feature_sets: dict[str, np.ndarray] = {}
    if "consensus_no_id" in controls:
        feature_sets["consensus_no_id"] = np.mean(np.stack([
            transform_features(raw, "no_id", all_graphs, args.shuffle_seed)
            for raw in member_raw
        ], axis=0), axis=0).astype(np.float32)
    if "consensus_shuffled" in controls:
        feature_sets["consensus_shuffled"] = np.mean(np.stack([
            transform_features(raw, "shuffled", all_graphs, args.shuffle_seed + 10_007 * bank_i)
            for bank_i, raw in enumerate(member_raw)
        ], axis=0), axis=0).astype(np.float32)
    if "global_stats" in controls:
        # Identity-free coarse control: for each of six statistics, retain only
        # its prototype-wise mean and maximum, then average across banks.
        globals_by_bank = []
        for raw in member_raw:
            globals_by_bank.append(np.concatenate([raw.mean(axis=1), raw.max(axis=1)], axis=1))
        feature_sets["global_stats"] = np.mean(np.stack(globals_by_bank, axis=0), axis=0).astype(np.float32)

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, args.hidden), nn.LayerNorm(args.hidden), nn.SiLU(),
                nn.Dropout(args.dropout), nn.Linear(args.hidden, 1),
            )
            self.gate = nn.Parameter(torch.zeros(()))
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x):
            amplitude = args.residual_cap * torch.tanh(self.gate)
            return amplitude * torch.tanh(self.net(x).view(-1))

    results: dict[str, Any] = {
        "base": {
            "fit_auc": float(roc_auc_score(fit_y, base_fit_prob)),
            "heldout_auc": base_auc,
            "fit_probabilities": base_fit_prob.astype(np.float32).tolist(),
            "heldout_probabilities": base_held_prob.astype(np.float32).tolist(),
            "fit_logits": base_fit_logits.tolist(),
            "heldout_logits": base_held_logits.tolist(),
        }
    }
    for control in controls:
        features = feature_sets[control]
        fit_x, held_x = features[:n_fit], features[n_fit:]
        mean, std = fit_x.mean(axis=0), fit_x.std(axis=0)
        std = np.where(std >= 1e-5, std, 1.0).astype(np.float32)
        fit_x = np.clip((fit_x - mean) / std, -5.0, 5.0).astype(np.float32)
        held_x = np.clip((held_x - mean) / std, -5.0, 5.0).astype(np.float32)

        seed_all(args.seed, torch)
        model = ResidualMLP(fit_x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.residual_lr, weight_decay=args.weight_decay)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(base_fit_logits), torch.from_numpy(fit_y)),
            batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 1000),
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, args.residual_epochs + 1):
            model.train(); total = bce_total = 0.0; seen = 0
            for x_batch, base_batch, y_batch in loader:
                delta = model(x_batch)
                bce = F.binary_cross_entropy_with_logits(base_batch + delta, y_batch)
                loss = bce + args.residual_l2 * delta.square().mean()
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                n = len(y_batch); total += float(loss.detach()) * n; bce_total += float(bce.detach()) * n; seen += n
            history.append({
                "epoch": epoch, "loss": total / seen, "bce": bce_total / seen,
                "gate": float(model.gate.detach()),
                "amplitude": float(args.residual_cap * torch.tanh(model.gate.detach())),
            })
        model.eval()
        with torch.no_grad():
            fit_delta = model(torch.from_numpy(fit_x)).numpy().astype(np.float32)
            held_delta = model(torch.from_numpy(held_x)).numpy().astype(np.float32)
        fit_logits = base_fit_logits + fit_delta
        held_logits = base_held_logits + held_delta
        fit_prob = sigmoid_np(fit_logits)
        held_prob = sigmoid_np(held_logits)
        auc = float(roc_auc_score(held_y, held_prob))
        results[control] = {
            "fit_auc": float(roc_auc_score(fit_y, fit_prob)),
            "heldout_auc": auc,
            "heldout_delta_auc": auc - base_auc,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "input_dim": int(fit_x.shape[1]),
            "gate": float(model.gate.detach()),
            "effective_amplitude": float(args.residual_cap * torch.tanh(model.gate.detach())),
            "fit_correction_std": float(fit_delta.std()),
            "heldout_correction_std": float(held_delta.std()),
            "heldout_correction_abs_max": float(np.abs(held_delta).max()),
            "fit_probabilities": fit_prob.astype(np.float32).tolist(),
            "heldout_probabilities": held_prob.astype(np.float32).tolist(),
            "fit_logits": fit_logits.astype(np.float32).tolist(),
            "heldout_logits": held_logits.astype(np.float32).tolist(),
            "history": history,
        }
        print(f"{control} heldout_auc={auc:.6f} delta={auc-base_auc:+.6f}", flush=True)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-frozen-multibank-consensus-occurrence-residual-scaffold-v1",
        "date": "2026-07-29",
        "hypothesis": "multi-bank consensus can reduce vocabulary variance in identity-free occurrence morphology",
        "architecture": {
            "base": "arithmetic mean probability of frozen graph-balanced real-prototype MIL banks",
            "occurrence": "radius-1 connected components of positive top-k prototype support",
            "consensus": "canonical identity-free prototype-row ordering followed by arithmetic mean across banks",
            "residual": "one hidden-layer zero-gated MLP capped in logit space",
            "supervised_message_passing_layers": 0,
        },
        "selection_policy": {
            "data": "8000-graph subset official-train only",
            "outer_validation": "one held-out Bemis-Murcko scaffold fold",
            "official_valid_evaluations": 0, "official_test_evaluations": 0,
            "epoch_policy": f"fixed {args.residual_epochs} epochs",
        },
        "config": vars(args),
        "prototype_seeds": prototype_seeds,
        "base_result_paths": [str(x) for x in base_paths],
        "fold": args.fold,
        "fit_graph_indices": fit_indices.tolist(), "heldout_graph_indices": heldout_indices.tolist(),
        "fit_labels": fit_y.tolist(), "heldout_labels": held_y.tolist(),
        "member_raw_feature_sha256": [sha256(x) for x in member_raw],
        "results": results,
        "elapsed_sec": time.time() - t0,
        "output": str(output),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "fold": args.fold,
        "heldout_auc": {k: v["heldout_auc"] for k, v in results.items()},
        "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
