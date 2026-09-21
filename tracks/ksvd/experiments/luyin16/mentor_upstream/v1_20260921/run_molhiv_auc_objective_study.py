#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUC-oriented training objective study on the fixed MolHIV K-SVD features.

Motivation
----------
DeepAUC showed that an AUC-margin surrogate loss is worth ~+3 ROC-AUC points
over a plain loss on the same backbone. This study asks whether AUC-oriented
objectives give a similar lift on OUR fixed structural features, under the
exact same evaluation protocol as the rest of the project: official OGB
scaffold split, seeds 0..9, official OGB Evaluator, no hyperparameter tuning.

Feature views (identical construction to run_molhiv_ensemble_multiview.py)
--------------------------------------------------------------------------
composition [N,69] + recon_typed [N,624] + context mass:
  full = all six ring contexts   (6K+6 dims, K read from payload)
  nora = drop ring_any           (5K+5 dims)
  base = no context mass         (composition + recon only)
The seven block-ablation aliases use S=composition, T=recon_typed, and
A=mass_nora: s, t, a, st, sa, ta, sta.  Thus sta is identical to nora and
st is identical to base.
All dimensions are read from the actual payloads; alignment assertions and
finiteness checks match the ensemble script.

Methods (one model per view x seed)
-----------------------------------
1. xgb_logistic (CONTROL): XGBClassifier exactly as the fixed 10-seed audit
   (best_params + common_xgb_params incl. scale_pos_weight, random_state=seed).
   The nora view should reproduce test ~0.8059.
2. xgb_pairwise: native-xgboost rank:pairwise over one query group covering
   the whole training set (DMatrix + set_group + xgb.train; the sklearn
   XGBRanker wrapper silently ignores group= on some xgboost versions,
   yielding constant 0.5 predictions). Tree parameters inherited from
   best_params (conflicting keys such as scale_pos_weight/objective/
   eval_metric removed), with min_child_weight forced to 1: the
   logistic-tuned value (8) exceeds any leaf's hessian mass under ranking
   objectives on this dataset, so the booster would never split (AUC=0.5).
3. mlp_aucmargin: PyTorch MLP(input -> 512 -> 256 -> 1, ReLU, Dropout 0.2) on
   StandardScaler features (scaler fit on train only). Batch-level AUC-margin
   loss (no LibAUC dependency):
       loss = var(pos) + var(neg) + relu(margin - (mean(pos) - mean(neg)))^2
   margin = 1.0, Adam lr 1e-3, batch 512 (batches without both classes are
   skipped), 50 epochs, fixed seeds; uses CUDA when available, CPU otherwise.
4. lgbm_rank (optional): lightgbm LGBMRanker (lambdarank) with one group,
   n_estimators=400, lr=0.1, conservative defaults; skipped on any failure.

Every cell records official train/valid/test ROC-AUC. Summary reports
mean +/- sample std per (view, method), a cross-seed probability-mean
secondary number, a softmax_valid (T=0.01) mixture over all cells, and a
comparison table with xgb_logistic marked as the control.
Selection discipline: between-method comparisons use valid AUC only; every
test number is reported transparently.

Outputs (in --result-dir)
-------------------------
  auc_objective_records.csv   per (seed, view, method) AUCs
  auc_objective_summary.json  full statistics incl. all schemes
  auc_objective_predictions.npz
      cell_names (e.g. "nora::xgb_pairwise") / cell_valid_predictions
      [C,S,Nv] / cell_test_predictions [C,S,Nt] / valid_labels /
      test_labels / seeds  -- compatible with analyze_molhiv_ensemble_union.py

CLI examples
------------
Full study on the K64 feature directories:

python -u run_molhiv_auc_objective_study.py \
  --diagnostic-feature-dir ./results/molhiv_ksvd_three_diagnostics_K64_s8 \
  --context-feature-dir ./results/molhiv_r2_atom_ring_context_K64_s8 \
  --reference-summary ./results/molhiv_best/summary.json \
  --result-dir ./results/molhiv_auc_objective_study \
  > logs/molhiv_auc_objective_study.log 2>&1

Control-only sanity check (should reproduce nora xgb ~0.8059):

python -u run_molhiv_auc_objective_study.py \
  --diagnostic-feature-dir ./results/molhiv_ksvd_three_diagnostics_K64_s8 \
  --context-feature-dir ./results/molhiv_r2_atom_ring_context_K64_s8 \
  --reference-summary ./results/molhiv_best/summary.json \
  --result-dir ./results/molhiv_auc_objective_control \
  --views nora --methods xgb_logistic

Single-method probe of the AUC-margin MLP on both views:

python -u run_molhiv_auc_objective_study.py \
  --diagnostic-feature-dir ./results/molhiv_ksvd_three_diagnostics_K64_s8 \
  --context-feature-dir ./results/molhiv_r2_atom_ring_context_K64_s8 \
  --reference-summary ./results/molhiv_best/summary.json \
  --result-dir ./results/molhiv_auc_objective_mlp \
  --methods mlp_aucmargin
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np


CONTEXT_NAMES = (
    "ring_any",
    "ring5",
    "ring6",
    "aromatic_ring",
    "multi_ring",
    "ring_boundary",
)
SELECTED_NO_RING_ANY = (
    "ring5",
    "ring6",
    "aromatic_ring",
    "multi_ring",
    "ring_boundary",
)

COMPOSITION_DIM = 69
ALL_VIEWS = (
    "full", "nora", "base",
    "s", "t", "a", "st", "sa", "ta", "sta",
)
ALL_METHODS = ("xgb_logistic", "xgb_pairwise", "mlp_aucmargin", "lgbm_rank")

# Keys that must not leak from the logistic XGBoost protocol into a ranker.
RANKER_CONFLICT_KEYS = ("scale_pos_weight", "objective", "eval_metric")

AUC_MARGIN = 1.0
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 512
MLP_LR = 1e-3
MLP_DROPOUT = 0.2


# =============================================================================
# Generic helpers (same patterns as the fixed 10-seed audit script)
# =============================================================================


def import_file(path, name):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def save_json(path, obj):
    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        return x

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def flatten_labels(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"Bad label shape: {y.shape}")
    y = y.astype(np.int64, copy=False)
    if not np.array_equal(np.unique(y), np.array([0, 1], dtype=np.int64)):
        raise ValueError(f"Expected binary labels, got {np.unique(y)}")
    return y


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std_population_ddof0": float(a.std(ddof=0)),
        "std_sample_ddof1": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
    }


# =============================================================================
# View assembly (identical to run_molhiv_ensemble_multiview.py)
# =============================================================================


def select_no_ring_any_mass(context):
    mass = np.asarray(context["mass"], dtype=np.float32)
    k = int(context["n_atoms"])
    c = len(CONTEXT_NAMES)
    expected = c * k + c
    if mass.ndim != 2 or mass.shape[1] != expected:
        raise RuntimeError(
            f"Context shape {mass.shape}, expected [N,{expected}]"
        )
    ids = {name: i for i, name in enumerate(CONTEXT_NAMES)}
    cols = []
    for name in SELECTED_NO_RING_ANY:
        i = ids[name]
        cols.extend(range(i * k, (i + 1) * k))
    coverage_start = c * k
    cols.extend(coverage_start + ids[name] for name in SELECTED_NO_RING_ANY)
    return np.asarray(mass[:, cols], dtype=np.float32)


def build_views(diagnostic_dir, context_dir, ring_module, context_module, view_names):
    local = ring_module.load_diagnostic_local_features(
        Path(diagnostic_dir).expanduser().resolve()
    )
    context = context_module.load_context_payload(
        Path(context_dir).expanduser().resolve()
    )
    for key in ("dataset_indices", "graph_split", "labels"):
        if not np.array_equal(np.asarray(local[key]), np.asarray(context[key])):
            raise RuntimeError(f"Diagnostic/context mismatch: {key}")

    composition = np.asarray(local["composition"], dtype=np.float32)
    recon = np.asarray(local["recon_typed"], dtype=np.float32)
    if composition.ndim != 2 or composition.shape[1] != COMPOSITION_DIM:
        raise RuntimeError(
            f"Expected {COMPOSITION_DIM}-D composition, got {composition.shape}"
        )
    k = int(context["n_atoms"])  # read from payload, never hard-coded
    mass_all = np.asarray(context["mass"], dtype=np.float32)
    c = len(CONTEXT_NAMES)
    if mass_all.ndim != 2 or mass_all.shape[1] != c * k + c:
        raise RuntimeError(
            f"Context mass shape {mass_all.shape}, expected [N,{c * k + c}] for K={k}"
        )
    mass_nora = select_no_ring_any_mass(context)

    blocks = {
        "full": [composition, recon, mass_all],
        "nora": [composition, recon, mass_nora],
        "base": [composition, recon],
        "s": [composition],
        "t": [recon],
        "a": [mass_nora],
        "st": [composition, recon],
        "sa": [composition, mass_nora],
        "ta": [recon, mass_nora],
        "sta": [composition, recon, mass_nora],
    }
    views = {}
    for name in view_names:
        matrix = np.concatenate(blocks[name], axis=1).astype(np.float32, copy=False)
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError(
                f"View {name}: feature matrix contains non-finite values"
            )
        views[name] = matrix

    labels = flatten_labels(local["labels"])
    graph_split = np.asarray(local["graph_split"], dtype=np.uint8)
    meta = {
        "K": k,
        "recon_dim": int(recon.shape[1]),
        "composition_dim": int(composition.shape[1]),
        "context_dim_all": int(mass_all.shape[1]),
        "context_dim_nora": int(mass_nora.shape[1]),
        "ablation_blocks": {
            "s": "composition",
            "t": "recon_typed",
            "a": "mass_nora",
        },
        "view_dims": {name: int(views[name].shape[1]) for name in views},
    }
    return views, labels, graph_split, meta


# =============================================================================
# Method implementations (all heavy deps lazy-imported)
# =============================================================================


def train_xgb_pairwise(x_train, y_train, seed, best_params, fixed_xgb, modules):
    """Native-xgboost rank:pairwise over a single query group.

    Bypasses the sklearn XGBRanker wrapper on purpose: some xgboost
    versions accept ``group=`` in ``XGBRanker.fit`` without actually
    applying it (every row then becomes a singleton query, all pairwise
    gradients vanish, and the model outputs a constant -> AUC exactly
    0.5). Native ``DMatrix.set_group`` guarantees the group is honored.

    Returns a small wrapper exposing ``.predict(X)`` (margin scores).
    """
    xgb = modules["xgboost"]
    params = {
        k: v for k, v in best_params.items() if k not in RANKER_CONFLICT_KEYS
    }
    for k, v in fixed_xgb.items():
        if k in RANKER_CONFLICT_KEYS or k in ("n_jobs", "random_state"):
            continue
        params.setdefault(k, v)
    num_boost_round = int(params.pop("n_estimators", 400))
    params.pop("random_state", None)
    params["objective"] = "rank:pairwise"
    # Correctness fix (NOT tuning): ranking objectives have a very different
    # per-leaf hessian scale than binary:logistic. The logistic-tuned
    # min_child_weight (=8 in the reference summary) is far too strict for
    # rank:pairwise on this dataset (only ~3.7% positives): no leaf ever
    # accumulates enough hessian mass, the booster never splits, and every
    # prediction collapses to a constant (train AUC exactly 0.5). Ablations
    # on the real features confirmed min_child_weight is the sole blocker
    # (mcw=1 -> train AUC ~0.68 in 200 rounds). mcw=1 restores the method's
    # intended behaviour.
    params["min_child_weight"] = 1
    params["seed"] = int(seed)
    params["verbosity"] = 0

    dtrain = xgb.DMatrix(
        x_train, label=np.asarray(y_train, dtype=np.float32)
    )
    dtrain.set_group([len(y_train)])
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    class _NativeRanker:
        def predict(self, x):
            return booster.predict(xgb.DMatrix(x))

    return _NativeRanker()


def lazy_import_torch_stack():
    try:
        import torch  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ImportError:
        return None
    return torch, StandardScaler


def auc_margin_loss(torch, scores, labels, margin):
    """Batch-level AUC-margin surrogate (DeepAUC-style, ~no dependency).

    loss = var(pos) + var(neg) + relu(margin - (mean(pos) - mean(neg)))^2
    Returns None when the batch lacks either class (caller skips the step).
    """
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    gap = pos.mean() - neg.mean()
    penalty = torch.relu(margin - gap) ** 2
    return pos.var(unbiased=False) + neg.var(unbiased=False) + penalty


def train_mlp_aucmargin(x_train, y_train, seed, modules):
    """StandardScaler + MLP(input->512->256->1) with batch AUC-margin loss."""
    torch, StandardScaler = modules["torch_stack"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32)

    in_dim = x_scaled.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(MLP_DROPOUT),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(MLP_DROPOUT),
        torch.nn.Linear(256, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)

    x_t = torch.from_numpy(x_scaled).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    n = len(y_train)

    model.train()
    for _ in range(MLP_EPOCHS):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, MLP_BATCH_SIZE):
            idx = perm[start : start + MLP_BATCH_SIZE]
            scores = model(x_t[idx]).reshape(-1)
            loss = auc_margin_loss(torch, scores, y_t[idx], AUC_MARGIN)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def score_fn(x):
        model.eval()
        with torch.no_grad():
            xs = torch.from_numpy(
                scaler.transform(x).astype(np.float32)
            ).to(device)
            out = []
            for start in range(0, xs.shape[0], 8192):
                out.append(model(xs[start : start + 8192]).reshape(-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    return score_fn


def train_lgbm_rank(x_train, y_train, seed, class_weight, modules):
    LGBMRanker = modules["LGBMRanker"]
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=400,
        learning_rate=0.1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(class_weight),
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, group=[len(y_train)])
    return model


# =============================================================================
# Softmax mixture scheme (valid-only weighting; test reported openly)
# =============================================================================


def softmax_weights(aucs, temperature):
    z = (np.asarray(aucs, dtype=np.float64) - np.max(aucs)) / max(
        float(temperature), 1e-12
    )
    w = np.exp(np.clip(z, -60.0, 0.0))
    return w / w.sum()


def evaluate_softmax_valid(
    cells_valid, cells_test, y_valid, y_test, official_rocauc, evaluator, temperature
):
    valid_aucs, test_aucs = [], []
    ens_valid, ens_test = [], []
    for s in range(cells_valid.shape[1]):
        v_mat = cells_valid[:, s, :]
        t_mat = cells_test[:, s, :]
        aucs = [float(official_rocauc(evaluator, y_valid, row)) for row in v_mat]
        w = softmax_weights(aucs, temperature)
        pv = w @ v_mat
        pt = w @ t_mat
        valid_aucs.append(float(official_rocauc(evaluator, y_valid, pv)))
        test_aucs.append(float(official_rocauc(evaluator, y_test, pt)))
        ens_valid.append(pv)
        ens_test.append(pt)
    return {
        "valid_per_seed": valid_aucs,
        "test_per_seed": test_aucs,
        "valid": stats(valid_aucs),
        "test": stats(test_aucs),
        "secondary_valid_rocauc": float(
            official_rocauc(evaluator, y_valid, np.mean(ens_valid, axis=0))
        ),
        "secondary_test_rocauc": float(
            official_rocauc(evaluator, y_test, np.mean(ens_test, axis=0))
        ),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    p = argparse.ArgumentParser(
        description="AUC-oriented training objective study on fixed MolHIV features."
    )
    p.add_argument(
        "--classification-script",
        default="./run_molhiv_shared_atom_classification_old_protocol.py",
    )
    p.add_argument("--ring-script", default="./run_molhiv_explicit_ring_oracle.py")
    p.add_argument("--context-script", default="./run_molhiv_r2_atom_ring_context.py")
    p.add_argument(
        "--diagnostic-feature-dir",
        default="./results/molhiv_ksvd_three_diagnostics_K64_s8",
    )
    p.add_argument(
        "--context-feature-dir",
        default="./results/molhiv_r2_atom_ring_context_K64_s8",
    )
    p.add_argument("--reference-summary", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--xgb-n-jobs", type=int, default=-1)
    p.add_argument(
        "--views",
        default="nora,full",
        help="Comma-separated subset of " + ",".join(ALL_VIEWS)
        + " (default: nora,full).",
    )
    p.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help="Comma-separated subset of " + ",".join(ALL_METHODS),
    )
    p.add_argument("--softmax-temperature", type=float, default=0.01)
    args = p.parse_args()

    view_names = [s.strip() for s in str(args.views).split(",") if s.strip()]
    bad_views = [v for v in view_names if v not in ALL_VIEWS]
    if bad_views:
        raise ValueError(f"Unknown views {bad_views}; expected subset of {ALL_VIEWS}")
    if not view_names:
        raise ValueError("At least one view is required")

    methods = [s.strip() for s in str(args.methods).split(",") if s.strip()]
    bad_methods = [m for m in methods if m not in ALL_METHODS]
    if bad_methods:
        raise ValueError(f"Unknown methods {bad_methods}; expected subset of {ALL_METHODS}")
    if not methods:
        raise ValueError("At least one method is required")

    seeds = [int(s) for s in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds: {seeds}")

    out = Path(args.result_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ring = import_file(args.ring_script, "aucstudy_ring")
    context_mod = import_file(args.context_script, "aucstudy_context")
    clf = import_file(args.classification_script, "aucstudy_classifier")

    views, labels, graph_split, meta = build_views(
        args.diagnostic_feature_dir, args.context_feature_dir, ring, context_mod,
        view_names,
    )
    positions = clf.split_positions(graph_split)
    train_pos = np.asarray(positions["train"], dtype=np.int64)
    valid_pos = np.asarray(positions["valid"], dtype=np.int64)
    test_pos = np.asarray(positions["test"], dtype=np.int64)
    y_train = labels[train_pos]
    y_valid = labels[valid_pos]
    y_test = labels[test_pos]

    reference_path = Path(args.reference_summary).expanduser().resolve()
    reference_summary = load_json(reference_path)
    if "best_params" not in reference_summary:
        raise KeyError(f"No best_params in {reference_path}")
    best_params = dict(reference_summary["best_params"])

    _, XGBClassifier, _, Evaluator = clf.lazy_import_evaluation_dependencies()
    evaluator = Evaluator(name=clf.DATASET_NAME)
    official_rocauc = clf.official_rocauc
    class_weight = clf.positive_class_weight(y_train)
    fixed_xgb = clf.common_xgb_params(
        class_weight=class_weight, n_jobs=int(args.xgb_n_jobs)
    )

    # Lazy per-method dependency resolution; missing deps skip the method.
    modules = {}
    active_methods = []
    skipped = {}
    for method in methods:
        if method == "xgb_logistic":
            active_methods.append(method)
        elif method == "xgb_pairwise":
            try:
                import xgboost as xgb_native  # type: ignore
            except ImportError:
                skipped[method] = "xgboost not available"
                continue
            modules["xgboost"] = xgb_native
            active_methods.append(method)
        elif method == "mlp_aucmargin":
            stack = lazy_import_torch_stack()
            if stack is None:
                skipped[method] = "torch and/or scikit-learn not available"
                continue
            modules["torch_stack"] = stack
            active_methods.append(method)
        elif method == "lgbm_rank":
            try:
                from lightgbm import LGBMRanker  # type: ignore
            except ImportError:
                skipped[method] = "lightgbm LGBMRanker not available"
                continue
            modules["LGBMRanker"] = LGBMRanker
            active_methods.append(method)
    for method, reason in skipped.items():
        warnings.warn(f"Skipping method {method}: {reason}")
    if not active_methods:
        raise RuntimeError("No methods available after lazy import checks")

    print("\n" + "=" * 108)
    print("MolHIV AUC-oriented objective study")
    print("=" * 108)
    print(f"views             : {view_names} dims={ {v: meta['view_dims'][v] for v in view_names} }")
    print(f"K (from payload)  : {meta['K']}  recon_dim: {meta['recon_dim']}")
    print(f"methods           : {active_methods} (skipped: {skipped or 'none'})")
    print(f"train/valid/test  : {len(train_pos)}/{len(valid_pos)}/{len(test_pos)}")
    print(f"reference summary : {reference_path}")
    print(f"reference test    : {reference_summary.get('test_mean', 'n/a')}")
    print(f"seeds             : {seeds}")
    print("Optuna            : disabled")
    print("=" * 108, flush=True)

    # lgbm_rank is the only method allowed to fail at train time (spec:
    # try/except and skip). A smoke fit on a tiny slice detects broken
    # lightgbm builds before the expensive loop starts.
    if "lgbm_rank" in active_methods:
        try:
            probe = train_lgbm_rank(
                views[view_names[0]][train_pos],
                y_train,
                seeds[0],
                class_weight,
                modules,
            )
            del probe
        except Exception as exc:
            warnings.warn(f"Skipping method lgbm_rank after smoke-fit failure: {exc}")
            skipped["lgbm_rank"] = f"smoke fit failed: {exc}"
            active_methods.remove("lgbm_rank")

    cell_names = [f"{v}::{m}" for v in view_names for m in active_methods]
    cells_valid = np.zeros(
        (len(cell_names), len(seeds), len(valid_pos)), dtype=np.float64
    )
    cells_test = np.zeros(
        (len(cell_names), len(seeds), len(test_pos)), dtype=np.float64
    )
    records = []

    for view_name in view_names:
        matrix = views[view_name]
        x_train = matrix[train_pos]
        x_valid = matrix[valid_pos]
        x_test = matrix[test_pos]
        for method in active_methods:
            for si, seed in enumerate(seeds):
                if method == "xgb_logistic":
                    model = XGBClassifier(
                        **best_params, **fixed_xgb, random_state=int(seed)
                    )
                    model.fit(x_train, y_train)
                    score = lambda x: model.predict_proba(x)[:, 1]
                elif method == "xgb_pairwise":
                    model = train_xgb_pairwise(
                        x_train, y_train, seed, best_params, fixed_xgb, modules
                    )
                    score = lambda x: np.asarray(
                        model.predict(x), dtype=np.float64
                    )
                elif method == "mlp_aucmargin":
                    score = train_mlp_aucmargin(x_train, y_train, seed, modules)
                elif method == "lgbm_rank":
                    model = train_lgbm_rank(
                        x_train, y_train, seed, class_weight, modules
                    )
                    score = lambda x: np.asarray(
                        model.predict(x), dtype=np.float64
                    )
                else:  # pragma: no cover
                    raise KeyError(method)

                pred_train = np.asarray(score(x_train), dtype=np.float64)
                pred_valid = np.asarray(score(x_valid), dtype=np.float64)
                pred_test = np.asarray(score(x_test), dtype=np.float64)

                train_auc = float(official_rocauc(evaluator, y_train, pred_train))
                valid_auc = float(official_rocauc(evaluator, y_valid, pred_valid))
                test_auc = float(official_rocauc(evaluator, y_test, pred_test))

                ci = cell_names.index(f"{view_name}::{method}")
                cells_valid[ci, si] = pred_valid
                cells_test[ci, si] = pred_test
                records.append(
                    {
                        "seed": seed,
                        "view": view_name,
                        "method": method,
                        "feature_dim": int(matrix.shape[1]),
                        "train_rocauc": train_auc,
                        "valid_rocauc": valid_auc,
                        "test_rocauc": test_auc,
                    }
                )
                print(
                    f"seed={seed:2d} | view={view_name:<5s} | "
                    f"method={method:<13s} | train={train_auc:.6f} | "
                    f"valid={valid_auc:.6f} | test={test_auc:.6f}",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Per-cell statistics and schemes
    # ------------------------------------------------------------------
    per_cell = {}
    for ci, cell in enumerate(cell_names):
        valid_aucs = [
            r["valid_rocauc"] for r in records
            if f"{r['view']}::{r['method']}" == cell
        ]
        test_aucs = [
            r["test_rocauc"] for r in records
            if f"{r['view']}::{r['method']}" == cell
        ]
        train_aucs = [
            r["train_rocauc"] for r in records
            if f"{r['view']}::{r['method']}" == cell
        ]
        per_cell[cell] = {
            "train": stats(train_aucs),
            "valid": stats(valid_aucs),
            "test": stats(test_aucs),
            "valid_per_seed": valid_aucs,
            "test_per_seed": test_aucs,
            "secondary_valid_rocauc": float(
                official_rocauc(evaluator, y_valid, cells_valid[ci].mean(axis=0))
            ),
            "secondary_test_rocauc": float(
                official_rocauc(evaluator, y_test, cells_test[ci].mean(axis=0))
            ),
        }

    softmax = evaluate_softmax_valid(
        cells_valid, cells_test, y_valid, y_test,
        official_rocauc, evaluator, args.softmax_temperature,
    )

    best_cell = max(
        per_cell,
        key=lambda c: (per_cell[c]["valid"]["mean"], per_cell[c]["test"]["mean"]),
    )

    summary = {
        "mode": "auc_objective_study",
        "views": view_names,
        "methods_requested": methods,
        "methods_active": active_methods,
        "methods_skipped": skipped,
        "feature_meta": meta,
        "diagnostic_feature_dir": str(
            Path(args.diagnostic_feature_dir).expanduser().resolve()
        ),
        "context_feature_dir": str(
            Path(args.context_feature_dir).expanduser().resolve()
        ),
        "seeds": seeds,
        "reference_summary": str(reference_path),
        "reference_test_mean": reference_summary.get("test_mean"),
        "best_params": best_params,
        "cell_names": cell_names,
        "per_cell": per_cell,
        "softmax_valid": softmax,
        "best_by_valid": {
            "cell": best_cell,
            "valid_mean": per_cell[best_cell]["valid"]["mean"],
            "test_mean": per_cell[best_cell]["test"]["mean"],
        },
        "selection_rule": "between-method selection uses valid AUC only; "
        "all test numbers are reported transparently",
        "optuna_rerun": False,
    }

    with (out / "auc_objective_records.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "seed",
                "view",
                "method",
                "feature_dim",
                "train_rocauc",
                "valid_rocauc",
                "test_rocauc",
            ),
        )
        writer.writeheader()
        writer.writerows(records)

    save_json(out / "auc_objective_summary.json", summary)
    np.savez_compressed(
        out / "auc_objective_predictions.npz",
        cell_names=np.asarray(cell_names),
        cell_valid_predictions=cells_valid,
        cell_test_predictions=cells_test,
        valid_labels=y_valid,
        test_labels=y_test,
        seeds=np.asarray(seeds, dtype=np.int64),
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 108)
    print("AUC objective study summary (mean +/- sample std over seeds)")
    print("=" * 108)
    ordered = sorted(
        per_cell,
        key=lambda c: per_cell[c]["valid"]["mean"],
        reverse=True,
    )
    for cell in ordered:
        c = per_cell[cell]
        marker = ""
        if cell.endswith("::xgb_logistic"):
            marker = " [control]"
        if cell == best_cell:
            marker += " <-- best by valid"
        print(
            f"{cell:<28s} valid {c['valid']['mean']:.6f} ± "
            f"{c['valid']['std_sample_ddof1']:.6f} | "
            f"test {c['test']['mean']:.6f} ± {c['test']['std_sample_ddof1']:.6f} | "
            f"secondary test {c['secondary_test_rocauc']:.6f}{marker}"
        )
    print("-" * 108)
    print(
        f"softmax_valid (T={args.softmax_temperature}) over {len(cell_names)} cells: "
        f"valid {softmax['valid']['mean']:.6f} ± "
        f"{softmax['valid']['std_sample_ddof1']:.6f} | "
        f"test {softmax['test']['mean']:.6f} ± "
        f"{softmax['test']['std_sample_ddof1']:.6f} | "
        f"secondary test {softmax['secondary_test_rocauc']:.6f}"
    )
    print("-" * 108)
    print(
        "DISCIPLINE: between-method selection must use valid AUC only; "
        "test columns are reported for transparency, not for selection."
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
