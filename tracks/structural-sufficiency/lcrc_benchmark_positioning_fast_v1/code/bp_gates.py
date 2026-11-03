"""Artifact-level audit of the section-7 correctness gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import bp_common as C
import bp_data as D
import bp_graph as BG
import bp_metrics as MET
import bp_model as BM
import bp_tu as T
import molecular_graph as MG
import tu_model as M
from feature_spec import load_spec


def _record(rows: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    rows.append({"gate": name, "ok": bool(ok), "detail": detail})


def _check_tu(rows: list[dict[str, Any]]) -> None:
    for ds in C.TU_DATASETS:
        fp = C.ANALYSIS / f"{ds}_folds.json"
        labels = C.ANALYSIS / f"{ds}_labels.npy"
        cache_path = C.CACHE / f"{ds}.pt"
        if not (fp.is_file() and labels.is_file() and cache_path.is_file()):
            continue
        y = np.load(labels)
        folds = C.read_json(fp)["folds"]
        all_rows = np.arange(len(y))
        ok = True
        detail = []
        for fold in folds:
            tr = set(fold["rows"]["train"])
            va = set(fold["rows"]["valid"])
            te = set(fold["rows"]["test"])
            good = (
                not (tr & va)
                and not (tr & te)
                and not (va & te)
                and np.array_equal(np.sort(np.asarray(list(tr | va | te))), all_rows)
                and len(tr) + len(va) + len(te) == len(y)
            )
            ok = ok and good
            detail.append({"fold": fold["fold"], "sizes": fold["sizes"], "ok": bool(good)})
        _record(rows, f"TU/{ds}/split_disjoint_cover", ok, detail)


def _check_mol(rows: list[dict[str, Any]]) -> None:
    for ds in C.MOL_DATASETS:
        short = ds.replace("ogbg-", "")
        sp = C.ANALYSIS / f"{short}_split.json"
        cp = C.CACHE / f"{short}.pt"
        if not (sp.is_file() and cp.is_file()):
            continue
        split = C.read_json(sp)
        cache = D.load_cache(cp)
        n = int(cache["n_molecules"])
        sets = {k: set(int(x) for x in split["rows"][k]) for k in ("train", "valid", "test")}
        ok = (
            all(min(s) >= 0 and max(s) < n for s in sets.values() if s)
            and not (sets["train"] & sets["valid"])
            and not (sets["train"] & sets["test"])
            and not (sets["valid"] & sets["test"])
            and len(sets["train"] | sets["valid"] | sets["test"]) == n
        )
        _record(
            rows,
            f"OGB/{ds}/official_split_matches_cache",
            ok,
            {"cache_n": n, "sizes": {k: len(v) for k, v in sets.items()}, "source": split.get("source")},
        )
        del cache


def _check_cache_labels(rows: list[dict[str, Any]]) -> None:
    for ds in C.TU_DATASETS:
        p = C.CACHE / f"{ds}.pt"
        if not p.is_file() or p.stat().st_size == 0:
            continue
        cache = D.load_cache(p)
        forbidden = {"y", "label", "labels", "target", "targets"} & set(cache.keys())
        _record(rows, f"TU/{ds}/no_target_in_cache", not forbidden and cache["meta"].get("labels_present") is False, {"forbidden": sorted(forbidden)})
        del cache
    for ds in C.MOL_DATASETS:
        p = C.CACHE / f"{ds.replace('ogbg-', '')}.pt"
        if not p.is_file() or p.stat().st_size == 0:
            continue
        cache = D.load_cache(p)
        forbidden = {"y", "label", "labels", "target", "targets"} & set(cache.keys())
        _record(rows, f"OGB/{ds}/no_target_in_cache", not forbidden and cache["meta"].get("labels_present") is False, {"forbidden": sorted(forbidden)})
        del cache


def _check_train_only(rows: list[dict[str, Any]]) -> None:
    ds = "MUTAG"
    spec = load_spec(C.ANALYSIS / f"{ds}_spec.json")
    cache = D.load_cache(C.CACHE / f"{ds}.pt")
    fold = C.read_json(C.ANALYSIS / f"{ds}_folds.json")["folds"][0]
    train = np.asarray(fold["rows"]["train"], dtype=np.int64)
    norm = D.fit_norm_rows(cache, spec, train, source="gate_train_only")
    atom_rows = D.cache_rows_atoms(cache, train)
    okay = np.allclose(np.asarray(cache["atom_scalar"])[atom_rows].mean(0), norm["scalar_mean"], atol=1e-4)
    okay = okay and np.allclose(np.asarray(cache["global"])[train].mean(0), norm["global_mean"], atol=1e-4)
    _record(rows, "TU/train_only_normalisation", bool(okay), {"dataset": ds, "fit_rows": int(train.size), "fit_rows_sha256": norm["fit_rows_sha256"]})


def _check_s2(rows: list[dict[str, Any]]) -> None:
    graph = MG.build_molecular_graph(
        atom_type=np.asarray([0, 1, 2, 3], dtype=np.int64),
        bond_index=np.asarray([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int64),
        bond_type=np.asarray([1, 1, 1, 2], dtype=np.int64),
        atom_feat=np.zeros((4, 1), dtype=np.float32),
        bond_feat=np.zeros((4, 1), dtype=np.float32),
    )
    up = MG.analyse_graph(graph)
    fast = BG.analyse_graph_fast(graph)
    cmp = BG.compare_with_upstream(graph, fast, up, 2)
    _record(rows, "S2/hand_and_upstream_equality", all(cmp[k] for k in ("s2_counts_equal", "n_d2_equal", "distance_stats_equal")), cmp)


def _check_c1(rows: list[dict[str, Any]]) -> None:
    spec = load_spec(C.ANALYSIS / "MUTAG_spec.json")
    full = M.local_layout(spec, "FULL")
    c1 = M.local_layout(spec, "C1")
    norm = {
        "global_mean": np.zeros(spec.global_dim, np.float32),
        "global_scale": np.ones(spec.global_dim, np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, np.float32),
        "scalar_scale": np.ones(spec.n_scalars, np.float32),
    }
    model, info = BM.build_c1(spec, norm, 2, 123, "cpu")
    ok = (
        set(full) - set(c1) == {"b_mean"}
        and len(M.local_columns(spec, "FULL")) - len(M.local_columns(spec, "C1")) == 96
        and info["shared_with_full"]["all_equal"]
        and "local.b_mean" not in dict(model.named_parameters())
    )
    _record(rows, "C1/full_minus_b_mean_and_seed_mapping", bool(ok), {"full_parts": list(full), "c1_parts": list(c1), "shared": info["shared_with_full"]})


def _check_forward(rows: list[dict[str, Any]]) -> None:
    spec = load_spec(C.ANALYSIS / "MUTAG_spec.json")
    cache = D.load_cache(C.CACHE / "MUTAG.pt")
    norm = {
        "global_mean": np.zeros(spec.global_dim, np.float32),
        "global_scale": np.ones(spec.global_dim, np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, np.float32),
        "scalar_scale": np.ones(spec.n_scalars, np.float32),
    }
    model, _ = BM.build_c1(spec, norm, 2, 321, "cpu")
    model.eval()
    with torch.no_grad():
        one = model(D.collate(cache, [0])).numpy()
        batch = model(D.collate(cache, [0, 1, 2, 3])).numpy()
    _record(rows, "model/single_vs_batch_forward", bool(np.allclose(one[0], batch[0], atol=1e-6)), {"max_abs_delta": float(np.max(np.abs(one[0] - batch[0])))})


def _check_metrics(rows: list[dict[str, Any]]) -> None:
    logits = np.asarray([[2.0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 2, 0]], np.float32)
    y = np.asarray([0, 2, 1, 2], np.int64)
    m = MET.classification_metrics(logits, y, 3)
    mask = np.asarray([[1.0, np.nan], [0.0, 1.0]], np.float64)
    loss = MET.masked_bce_loss(torch.zeros((2, 2)), torch.as_tensor(mask))
    y_auc = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], np.float64).reshape(-1, 1)
    p_auc = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6], np.float64).reshape(-1, 1)
    auc = MET.ogb_rocauc("ogbg-molhiv", y_auc, p_auc)
    ap = MET.macro_ap(np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], np.float64), np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]], np.float64))
    ok = m["accuracy"] == 0.75 and torch.isfinite(loss).item() and 0 <= float(auc["value"]) <= 1 and np.isfinite(ap["value"])
    _record(rows, "metrics/dimensions_missing_and_recompute", bool(ok), {"accuracy": m["accuracy"], "masked_loss": float(loss), "roc_auc": auc["value"], "mean_ap": ap["value"]})


def _check_peptide(rows: list[dict[str, Any]]) -> None:
    stats = C.ANALYSIS / "peptides_func_cache_stats.json"
    if not stats.is_file():
        return
    payload = C.read_json(stats)
    y = np.load(C.ANALYSIS / "peptides_func_y_train.npy")
    ok = y.ndim == 2 and y.shape[1] == 10 and np.isfinite(y).all()
    for split in ("train", "valid", "test"):
        s = payload["splits"][split]
        ok = ok and s["n_graphs"] > 0 and s["s2_paths"]["max"] > 0
    _record(rows, "peptides/AP_dimensions_and_missing_values", bool(ok), {"y_shape": list(y.shape), "splits": {k: payload["splits"][k]["n_graphs"] for k in ("train", "valid", "test")}})


def main() -> int:
    rows: list[dict[str, Any]] = []
    _check_tu(rows)
    _check_mol(rows)
    _check_cache_labels(rows)
    _check_train_only(rows)
    _check_s2(rows)
    _check_c1(rows)
    _check_forward(rows)
    _check_metrics(rows)
    _check_peptide(rows)
    payload = {
        "generated_utc": C.utcnow(),
        "n_checks": len(rows),
        "n_failed": sum(1 for r in rows if not r["ok"]),
        "all_passed": all(r["ok"] for r in rows),
        "checks": rows,
    }
    C.write_json(C.ANALYSIS / "correctness_gates.json", payload)
    print(json.dumps({"n_checks": payload["n_checks"], "n_failed": payload["n_failed"], "all_passed": payload["all_passed"]}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
