"""Strict inner-OOF late fusion of NCI1 attributes and Beam8 chain views."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _encode,
    _fit_dictionary,
    _graph_token_feature,
    _normalize_patch_tokens,
    _shuffle_permutation,
    prepare_graph,
)
from .run_beam8_nci1_compact_relation_followup import _compact_feature


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_NCI1_OOF_LATE_FUSION_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_nci1_oof_late_fusion_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_NCI1_OOF_LATE_FUSION_20260813.md"
VARIANTS = (
    "FEATURE_STATS",
    "INIT_BAG",
    "INIT_TRUE",
    "FUSION_INIT_BAG",
    "FUSION_INIT_TRUE",
    "FUSION_INIT_SHUFFLED",
    "FUSION_FINAL_TRUE",
)


def _linear() -> Any:
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, random_state=0)
    )


def _decision(model: Any, values: np.ndarray) -> np.ndarray:
    output = np.asarray(model.decision_function(values), dtype=np.float64)
    return output.reshape(-1, 1) if output.ndim == 1 else output


def _score(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
    }


def _base_score(
    values: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    model = _linear()
    model.fit(values[train_indices], labels[train_indices])
    return _score(labels[test_indices], model.predict(values[test_indices]))


def _oof_fusion(
    primary: np.ndarray,
    secondary: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1729 + seed)
    primary_oof = None
    secondary_oof = None
    for inner_train_local, validation_local in inner.split(
        np.zeros(len(train_indices)), labels[train_indices]
    ):
        inner_train = train_indices[inner_train_local]
        validation = train_indices[validation_local]
        primary_model = _linear()
        secondary_model = _linear()
        primary_model.fit(primary[inner_train], labels[inner_train])
        secondary_model.fit(secondary[inner_train], labels[inner_train])
        p = _decision(primary_model, primary[validation])
        s = _decision(secondary_model, secondary[validation])
        if primary_oof is None:
            primary_oof = np.zeros((len(train_indices), p.shape[1]), dtype=np.float64)
            secondary_oof = np.zeros((len(train_indices), s.shape[1]), dtype=np.float64)
        primary_oof[validation_local] = p
        secondary_oof[validation_local] = s
    assert primary_oof is not None and secondary_oof is not None
    meta_train = np.concatenate([primary_oof, secondary_oof], axis=1)
    meta = _linear()
    meta.fit(meta_train, labels[train_indices])

    primary_model = _linear()
    secondary_model = _linear()
    primary_model.fit(primary[train_indices], labels[train_indices])
    secondary_model.fit(secondary[train_indices], labels[train_indices])
    meta_test = np.concatenate(
        [
            _decision(primary_model, primary[test_indices]),
            _decision(secondary_model, secondary[test_indices]),
        ],
        axis=1,
    )
    return _score(labels[test_indices], meta.predict(meta_test))


def _features(
    prepared: Sequence[Any],
    train_indices: np.ndarray,
    dictionary: dict[str, Any],
    *,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    init_raw = [
        np.concatenate(
            [_encode(item, dictionary["initial"], dictionary["mean"], 3), item.node_histograms],
            axis=1,
        )
        for item in prepared
    ]
    final_raw = [
        np.concatenate(
            [_encode(item, dictionary["final"], dictionary["mean"], 3), item.node_histograms],
            axis=1,
        )
        for item in prepared
    ]
    init = _normalize_patch_tokens(init_raw, train_indices)
    final = _normalize_patch_tokens(final_raw, train_indices)
    rows = {
        "FEATURE_STATS": [], "INIT_BAG": [], "INIT_TRUE": [],
        "INIT_SHUFFLED": [], "FINAL_TRUE": [],
    }
    for index, item in enumerate(prepared):
        rows["FEATURE_STATS"].append(np.concatenate([item.graph_features, item.stats]))
        rows["INIT_BAG"].append(_graph_token_feature(item, init[index], chain=False))
        rows["INIT_TRUE"].append(_compact_feature(item, init[index]))
        permutation = _shuffle_permutation(
            len(init[index]), seed=shuffle_seed + item.index * 1009
        )
        rows["INIT_SHUFFLED"].append(
            _compact_feature(item, init[index], permutation=permutation)
        )
        rows["FINAL_TRUE"].append(_compact_feature(item, final[index]))
    return {name: np.stack(values) for name, values in rows.items()}


def _paired(folds: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [fold["scores"][left]["balanced_accuracy"] - fold["scores"][right]["balanced_accuracy"] for fold in folds]
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
    }


def _summarize(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = {
        variant: {
            "balanced_accuracy_mean": float(np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])),
            "balanced_accuracy_std": float(np.std([fold["scores"][variant]["balanced_accuracy"] for fold in folds])),
            "accuracy_mean": float(np.mean([fold["scores"][variant]["accuracy"] for fold in folds])),
        }
        for variant in VARIANTS
    }
    paired = {
        "fusion_increment": _paired(folds, "FUSION_INIT_TRUE", "FEATURE_STATS"),
        "relation_specific": _paired(folds, "FUSION_INIT_TRUE", "FUSION_INIT_SHUFFLED"),
        "chain_specific": _paired(folds, "FUSION_INIT_TRUE", "FUSION_INIT_BAG"),
        "ksvd_update": _paired(folds, "FUSION_FINAL_TRUE", "FUSION_INIT_TRUE"),
    }
    checks = {
        "fusion_increment": bool(paired["fusion_increment"]["mean"] >= 0.01 and paired["fusion_increment"]["wins"] >= 2),
        "relation_specific_fusion": bool(paired["relation_specific"]["mean"] >= 0.005 and paired["relation_specific"]["wins"] >= 2),
        "chain_specific_fusion": bool(paired["chain_specific"]["mean"] >= 0.005 and paired["chain_specific"]["wins"] >= 2),
        "ksvd_update": bool(paired["ksvd_update"]["mean"] >= 0.01 and paired["ksvd_update"]["wins"] >= 2),
    }
    if all(checks[name] for name in ("fusion_increment", "relation_specific_fusion", "chain_specific_fusion")):
        decision = "BEAM8_CHAIN_MULTIVIEW_FUSION_PASS"
    elif checks["fusion_increment"]:
        decision = "GENERAL_STRUCTURE_FUSION_ONLY"
    else:
        decision = "NCI1_BEAM8_OOF_FUSION_BELOW_GATE"
    return {"variants": variants, "paired": paired, "checks": checks, "decision": decision}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, features, metadata = load_tud(args.dataset, args.dataset_root)
    prepared = []
    for index, (graph, label, node_features) in enumerate(zip(graphs, labels, features)):
        prepared.append(
            prepare_graph(index, graph, int(label), node_features, patch_size=8, overlap=2,
                          retained_beam=8, edge_capacity_multiplier=1.5, seed=20260813)
        )
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"cover {index + 1}/{len(graphs)}", flush=True)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        dictionary = _fit_dictionary(prepared, train_indices, n_atoms=24, sparsity=3,
                                     iterations=5, max_train_patches=3000, seed=fold_index)
        values = _features(prepared, train_indices, dictionary, shuffle_seed=731421 + fold_index)
        scores = {
            "FEATURE_STATS": _base_score(values["FEATURE_STATS"], labels, train_indices, test_indices),
            "INIT_BAG": _base_score(values["INIT_BAG"], labels, train_indices, test_indices),
            "INIT_TRUE": _base_score(values["INIT_TRUE"], labels, train_indices, test_indices),
            "FUSION_INIT_BAG": _oof_fusion(values["FEATURE_STATS"], values["INIT_BAG"], labels, train_indices, test_indices, seed=fold_index),
            "FUSION_INIT_TRUE": _oof_fusion(values["FEATURE_STATS"], values["INIT_TRUE"], labels, train_indices, test_indices, seed=fold_index),
            "FUSION_INIT_SHUFFLED": _oof_fusion(values["FEATURE_STATS"], values["INIT_SHUFFLED"], labels, train_indices, test_indices, seed=fold_index),
            "FUSION_FINAL_TRUE": _oof_fusion(values["FEATURE_STATS"], values["FINAL_TRUE"], labels, train_indices, test_indices, seed=fold_index),
        }
        folds.append({"fold_index": fold_index, "scores": scores})
        print(f"fold {fold_index}: FS={scores['FEATURE_STATS']['balanced_accuracy']:.4f} FUSION_TRUE={scores['FUSION_INIT_TRUE']['balanced_accuracy']:.4f} SHUFFLED={scores['FUSION_INIT_SHUFFLED']['balanced_accuracy']:.4f}", flush=True)
    return {"protocol": PROTOCOL, "dataset": metadata, "folds": folds,
            "summary": _summarize(folds), "seconds": time.time() - started}


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = ["# Beam8/NCI1 strict OOF late fusion", "",
             f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{summary['decision']}`", "",
             "| variant | balanced accuracy | accuracy |", "|---|---:|---:|"]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | {row['accuracy_mean']:.4f} |")
    lines.extend(["", "## Paired attribution", "", "| comparison | mean | W/T/L |", "|---|---:|---:|"])
    for name, row in summary["paired"].items():
        lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(["", "## Boundary", "",
                  "- meta head 只读取 outer-train inner-OOF base logits；outer-test 不参与权重选择。",
                  "- 本轮失败时，NCI1 上停止扩大 Beam8 分类融合模型。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NCI1")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

