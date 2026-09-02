"""Pilot for differentiable task-aware sparse dictionary learning.

Unlike the witness-prototype pilot, this script updates the dictionary itself.
Each scaffold fold fits an unsupervised K-SVD dictionary on the training
changes, then jointly optimizes normalized atoms, a soft-threshold sparse code,
and a direction classifier.  The loss combines task BCE, reconstruction, and
dictionary incoherence.  Held-out cores are encoded with the frozen learned
dictionary and classifier.

This is a mechanism pilot, not a leaderboard model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ksvd import _omp, ksvd
from .run_molpcba_local_change_compression import (
    _fit_score,
    _random_dictionary,
    _relative_reconstruction,
)
from .run_molpcba_local_change_gate import _fold_for_scaffold, _replace_for_task


def _scale_fit(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values * values, axis=0) + 1e-8)


def _codes(dictionary: np.ndarray, values: np.ndarray, sparsity: int) -> np.ndarray:
    return np.stack([_omp(dictionary, row, sparsity) for row in values], axis=0)


def _fit_differentiable_dictionary(
    dictionary: np.ndarray,
    train_values: np.ndarray,
    *,
    sparsity: int,
    epochs: int,
    lr: float,
    task_weight: float,
    reconstruction_weight: float,
    incoherence_weight: float,
    seed: int,
) -> tuple[np.ndarray, float, dict[str, float]]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x = np.concatenate([train_values, -train_values], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(train_values)), np.zeros(len(train_values))]).astype(np.float32)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    init = torch.from_numpy(dictionary.T.astype(np.float32))
    model = nn.Linear(dictionary.shape[1], 1, bias=True)
    # The linear head starts small; the task loss should shape the dictionary,
    # not immediately overpower the reconstruction basin.
    nn.init.normal_(model.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.bias)
    atom_param = nn.Parameter(init.clone())
    log_tau = nn.Parameter(torch.tensor(-3.0))
    optimizer = torch.optim.Adam([atom_param, *model.parameters(), log_tau], lr=lr)
    bce = nn.BCEWithLogitsLoss()
    last_task = last_reconstruction = last_incoherence = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        atoms = torch.nn.functional.normalize(atom_param, dim=1)
        corr = x_t @ atoms.T
        tau = torch.sigmoid(log_tau) * 0.5
        code = torch.sign(corr) * torch.relu(torch.abs(corr) - tau)
        logits = model(code).squeeze(1)
        reconstructed = code @ atoms
        task_loss = bce(logits, y_t)
        reconstruction_loss = torch.mean((reconstructed - x_t) ** 2)
        gram = atoms @ atoms.T
        offdiag = gram - torch.diag(torch.diag(gram))
        incoherence_loss = torch.mean(offdiag ** 2)
        loss = (
            task_weight * task_loss
            + reconstruction_weight * reconstruction_loss
            + incoherence_weight * incoherence_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([atom_param, *model.parameters(), log_tau], 5.0)
        optimizer.step()
        last_task = float(task_loss.detach())
        last_reconstruction = float(reconstruction_loss.detach())
        last_incoherence = float(incoherence_loss.detach())
    with torch.no_grad():
        atoms = torch.nn.functional.normalize(atom_param, dim=1)
        tau = float((torch.sigmoid(log_tau) * 0.5).cpu())
        learned = atoms.T.cpu().numpy().astype(np.float64)
    return learned, tau, {
        "final_task_bce": last_task,
        "final_reconstruction_mse": last_reconstruction,
        "final_incoherence": last_incoherence,
        "learned_tau": tau,
    }


def _soft_codes(dictionary: np.ndarray, values: np.ndarray, tau: float) -> np.ndarray:
    corr = values @ dictionary
    return np.sign(corr) * np.maximum(np.abs(corr) - tau, 0.0)


def _one_fold(values: np.ndarray, folds: np.ndarray, fold: int, args: argparse.Namespace) -> dict[str, object]:
    train, test = folds != fold, folds == fold
    raw_train, raw_test = values[train], values[test]
    scale = _scale_fit(raw_train)
    train_scaled, test_scaled = raw_train / scale, raw_test / scale
    pca = PCA(n_components=min(args.n_atoms, train_scaled.shape[0], train_scaled.shape[1]), random_state=args.seed).fit(train_scaled)
    random_dictionary = _random_dictionary(train_scaled, args.n_atoms, args.seed + fold)
    ksvd_dictionary, _, ksvd_info = ksvd(
        train_scaled.T, n_atoms=args.n_atoms, T=args.sparsity,
        n_iter=args.ksvd_iter, seed=args.seed + fold,
    )
    task_dictionary, tau, task_fit = _fit_differentiable_dictionary(
        ksvd_dictionary,
        train_scaled,
        sparsity=args.sparsity,
        epochs=args.task_epochs,
        lr=args.task_lr,
        task_weight=args.task_weight,
        reconstruction_weight=args.reconstruction_weight,
        incoherence_weight=args.incoherence_weight,
        seed=args.seed + 1009 * (fold + 1),
    )
    # The task model's learned head is intentionally not reused here: refit
    # the same linear readout on frozen codes, making the representation
    # comparison directly comparable to the existing compression audit.
    representations = {
        "raw_change": (train_scaled, -train_scaled, test_scaled, -test_scaled),
        "pca": (
            pca.transform(train_scaled), pca.transform(-train_scaled),
            pca.transform(test_scaled), pca.transform(-test_scaled),
        ),
        "random_real_change_prototypes": (
            _codes(random_dictionary, train_scaled, args.sparsity),
            _codes(random_dictionary, -train_scaled, args.sparsity),
            _codes(random_dictionary, test_scaled, args.sparsity),
            _codes(random_dictionary, -test_scaled, args.sparsity),
        ),
        "ksvd_change_dictionary": (
            _codes(ksvd_dictionary, train_scaled, args.sparsity),
            _codes(ksvd_dictionary, -train_scaled, args.sparsity),
            _codes(ksvd_dictionary, test_scaled, args.sparsity),
            _codes(ksvd_dictionary, -test_scaled, args.sparsity),
        ),
        "differentiable_task_dictionary": (
            _soft_codes(task_dictionary, train_scaled, tau),
            _soft_codes(task_dictionary, -train_scaled, tau),
            _soft_codes(task_dictionary, test_scaled, tau),
            _soft_codes(task_dictionary, -test_scaled, tau),
        ),
    }
    scores = {name: _fit_score(*arrays) for name, arrays in representations.items()}
    return {
        "fold": fold,
        "n_train_replacements": int(np.sum(train)),
        "n_test_replacements": int(np.sum(test)),
        "scores": scores,
        "reconstruction_relative": {
            "random_real_change_prototypes": _relative_reconstruction(test_scaled, random_dictionary, args.sparsity),
            "ksvd_change_dictionary": _relative_reconstruction(test_scaled, ksvd_dictionary, args.sparsity),
            "differentiable_task_dictionary": float(
                np.linalg.norm(test_scaled - _soft_codes(task_dictionary, test_scaled, tau) @ task_dictionary.T)
                / max(np.linalg.norm(test_scaled), 1e-12)
            ),
        },
        "ksvd_fit": ksvd_info,
        "task_fit": task_fit,
    }


def _summarize(folds: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    names = list(folds[0]["scores"].keys())  # type: ignore[index]
    return {
        name: {
            metric: float(np.mean([row["scores"][name][metric] for row in folds]))  # type: ignore[index]
            for metric in ("direction_accuracy", "paired_roc_auc")
        }
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ogb/ogbg_molpcba")
    parser.add_argument("--task", type=int, default=93)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bits", type=int, default=512)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-atoms", type=int, default=32)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--ksvd-iter", type=int, default=4)
    parser.add_argument("--task-epochs", type=int, default=120)
    parser.add_argument("--task-lr", type=float, default=0.01)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--incoherence-weight", type=float, default=0.05)
    parser.add_argument("--pair-retries", type=int, default=4)
    parser.add_argument("--max-per-scaffold", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root)
    header = pd.read_csv(root / "mapping/mol.csv.gz", nrows=0)
    label_columns = list(header.columns[:128])
    assay = label_columns[args.task]
    table = pd.read_csv(root / "mapping/mol.csv.gz", usecols=["smiles", assay])
    train_indices = pd.read_csv(root / "split/scaffold/train.csv.gz", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    replacements, counts = _replace_for_task(
        table, train_indices, assay,
        n_bits=args.n_bits, radius=args.radius,
        seed=args.seed + args.task, retries=args.pair_retries,
        max_per_scaffold=args.max_per_scaffold,
    )
    if len(replacements) < 300:
        raise RuntimeError("too few clean replacements")
    values = np.stack([
        item.positive.fingerprint.astype(np.float64) - item.negative.fingerprint.astype(np.float64)
        for item in replacements
    ])
    folds = np.asarray([_fold_for_scaffold(item.scaffold, 3) for item in replacements], dtype=np.int64)
    results = [_one_fold(values, folds, fold, args) for fold in range(3)]
    summary = _summarize(results)
    task_name = "differentiable_task_dictionary"
    decision = {
        "task_dictionary_beats_ksvd_mean_auc": bool(summary[task_name]["paired_roc_auc"] > summary["ksvd_change_dictionary"]["paired_roc_auc"]),
        "task_dictionary_beats_ksvd_folds": int(sum(
            row["scores"][task_name]["paired_roc_auc"] > row["scores"]["ksvd_change_dictionary"]["paired_roc_auc"]  # type: ignore[index]
            for row in results
        )),
        "interpretation": "promising_joint_task_dictionary" if summary[task_name]["paired_roc_auc"] > summary["ksvd_change_dictionary"]["paired_roc_auc"] else "no_pilot_evidence_over_reconstruction_ksvd",
    }
    payload = {
        "protocol_id": "molpcba-differentiable-task-dictionary-pilot-v1",
        "scope": "official training split only; three internal Murcko-scaffold folds",
        "purpose": "Jointly rotate a K-SVD dictionary using local-change direction loss while retaining reconstruction.",
        "config": vars(args),
        "assay": assay,
        "counts": counts,
        "folds": results,
        "summary": summary,
        "decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
