from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.classify import eval_logistic_cv  # noqa: E402
from code.data_tud import load_tud, node_feature_readout  # noqa: E402
from code.pipeline import PipelineConfig, embed_dataset  # noqa: E402
from code.synthetic_task import degree_features, make_triangle_vs_longcycle  # noqa: E402


def _pad_stack(rows: list[np.ndarray]) -> np.ndarray:
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, : r.shape[0]] = r
    return X


def run_synth() -> dict:
    graphs, y, meta = make_triangle_vs_longcycle(n_per_class=100, n_nodes=16, seed=0)
    Xdeg = degree_features(graphs)
    out = {
        "meta": meta,
        "degree_stats": eval_logistic_cv(Xdeg, y, n_splits=5, seed=0),
        "methods": {},
    }
    for method, p, q, decay in [
        ("B0", 1.0, 1.0, 1.0),
        ("B1", 1.0, 1.0, 1.0),
        ("M0", 0.5, 2.0, 0.7),
        ("M0_local", 1.0, 2.0, 0.7),  # more BFS-like
    ]:
        cfg = PipelineConfig(
            method="M0" if method.startswith("M0") else method,
            p=p,
            q=q,
            walk_length=10,
            max_nodes=12,
            num_walks=20,
            edge_decay=decay,
            seed=0,
            n_atoms=10,
            T=3,
            T_min=2,
            ksvd_iter=6,
        )
        t0 = time.time()
        X, infos = embed_dataset(graphs, cfg)
        clf = eval_logistic_cv(X, y, n_splits=5, seed=0)
        out["methods"][method] = {
            **clf,
            "mean_recon": float(np.mean([i["recon_rel"] for i in infos])),
            "seconds": time.time() - t0,
        }
        print(
            f"  synth {method}: acc={clf['acc_mean']:.3f}±{clf['acc_std']:.3f} "
            f"recon={out['methods'][method]['mean_recon']:.3f} ({out['methods'][method]['seconds']:.1f}s)"
        )
    return out


def run_mutag_fused() -> dict:
    graphs, y, node_feats, meta = load_tud("MUTAG")
    # attribute-only readout
    attr_rows = [node_feature_readout(xf) for xf in node_feats]
    Xattr = _pad_stack(attr_rows)
    Xdeg = degree_features(graphs)
    Xattr_deg = np.hstack([Xattr, Xdeg])

    out: dict = {
        "meta": meta,
        "baselines": {
            "node_attr_only": eval_logistic_cv(Xattr, y, n_splits=10, seed=0),
            "degree_stats": eval_logistic_cv(Xdeg, y, n_splits=10, seed=0),
            "attr+degree": eval_logistic_cv(Xattr_deg, y, n_splits=10, seed=0),
        },
        "methods": {},
    }
    for k, v in out["baselines"].items():
        print(f"  mutag base {k}: acc={v['acc_mean']:.3f}±{v['acc_std']:.3f}")

    for method, p, q, decay in [
        ("B0", 1.0, 1.0, 1.0),
        ("B1", 1.0, 1.0, 1.0),
        ("M0", 0.5, 2.0, 0.7),
    ]:
        cfg = PipelineConfig(
            method=method,
            p=p,
            q=q,
            walk_length=6,
            max_nodes=10,
            num_walks=12,
            edge_decay=decay,
            seed=0,
            n_atoms=8,
            T=3,
            T_min=2,
            ksvd_iter=5,
        )
        t0 = time.time()
        Xstruct, infos = embed_dataset(graphs, cfg)
        # structure only
        s_only = eval_logistic_cv(Xstruct, y, n_splits=10, seed=0)
        # structure + attr
        Xsa = np.hstack([Xstruct, Xattr])
        s_attr = eval_logistic_cv(Xsa, y, n_splits=10, seed=0)
        # structure + attr + degree
        Xsad = np.hstack([Xstruct, Xattr, Xdeg])
        s_all = eval_logistic_cv(Xsad, y, n_splits=10, seed=0)
        out["methods"][method] = {
            "structure_only": s_only,
            "structure+attr": s_attr,
            "structure+attr+degree": s_all,
            "mean_recon": float(np.mean([i["recon_rel"] for i in infos])),
            "seconds": time.time() - t0,
        }
        print(
            f"  mutag {method}: struct={s_only['acc_mean']:.3f} "
            f"s+attr={s_attr['acc_mean']:.3f} "
            f"s+a+d={s_all['acc_mean']:.3f} ({time.time()-t0:.1f}s)"
        )
    return out


def main() -> int:
    results = {"track": "ksvd", "experiment": "followup_v1"}
    print("=== 1) Synthetic: triangle vs long-cycle (same n, |E|) ===")
    results["synth_triangle"] = run_synth()

    print("=== 2) MUTAG: structure + node attributes ===")
    results["mutag_fused"] = run_mutag_fused()

    out = _TRACK / "results" / "followup.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = _TRACK / "results" / "FOLLOWUP_SUMMARY.md"
    summary.write_text(_render(results), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {summary}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# Follow-up experiments",
        "",
        "## 1. Triangle vs long-cycle (controlled)",
        "",
        f"Meta: `{r['synth_triangle']['meta']}`",
        "",
        "| method | Acc |",
        "|--------|-----|",
    ]
    deg = r["synth_triangle"]["degree_stats"]
    lines.append(f"| degree_stats | {deg['acc_mean']:.3f} ± {deg['acc_std']:.3f} |")
    for m, info in r["synth_triangle"]["methods"].items():
        lines.append(f"| {m} | {info['acc_mean']:.3f} ± {info['acc_std']:.3f} |")
    lines += [
        "",
        "## 2. MUTAG fusion",
        "",
        f"Meta: `{r['mutag_fused']['meta']}`",
        "",
        "### Baselines",
        "",
        "| method | Acc |",
        "|--------|-----|",
    ]
    for m, info in r["mutag_fused"]["baselines"].items():
        lines.append(f"| {m} | {info['acc_mean']:.3f} ± {info['acc_std']:.3f} |")
    lines += ["", "### With KSVD structure", "", "| method | struct | +attr | +attr+deg |", "|--------|--------|-------|-----------|"]
    for m, info in r["mutag_fused"]["methods"].items():
        lines.append(
            f"| {m} | {info['structure_only']['acc_mean']:.3f} | "
            f"{info['structure+attr']['acc_mean']:.3f} | "
            f"{info['structure+attr+degree']['acc_mean']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Synth task: same n and |E|; class0 has triangle, class1 triangle-free longer cycle.",
        "- MUTAG protocol: sklearn Stratified 10-fold **mean** Acc — not CIN paper protocol.",
        "- MolHIV with LR only: see discussion in chat; expect modest AUC unless stronger head.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
