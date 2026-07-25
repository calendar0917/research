"""
molhiv structure-only probe (P0/P1).

Protocol: molhiv-struct-probe-v0 (scaffold splits; LR on s_G; D from train only).

Usage:
  python -m code.run_molhiv_probe --check-only
  python -m code.run_molhiv_probe --max-graphs 2000 --pools mean,attn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import check_env, degree_hist_features, load_molhiv  # noqa: E402
from code.graph_level import (  # noqa: E402
    GraphLevelConfig,
    encode_graph,
    learn_shared_D_graph_level,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def encode_split(
    graphs,
    indices: np.ndarray,
    D: np.ndarray,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
) -> np.ndarray:
    rows = []
    for i in indices:
        s, _ = encode_graph(graphs[int(i)], D, cfg, mode=mode, seed=cfg.seed + int(i) * 17)
        rows.append(s)
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for j, r in enumerate(rows):
        X[j, : r.shape[0]] = r
    return X


def fit_auc(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    seed: int = 0,
) -> dict:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)
    def _auc(X, y):
        if len(np.unique(y)) < 2:
            return float("nan")
        proba = clf.predict_proba(X)
        # positive class column
        classes = list(clf.named_steps["logisticregression"].classes_)
        if 1.0 in classes:
            col = classes.index(1.0)
        elif 1 in classes:
            col = classes.index(1)
        else:
            col = int(np.argmax(classes))
        return float(roc_auc_score(y, proba[:, col]))

    return {
        "train_auc": _auc(Xtr, ytr),
        "valid_auc": _auc(Xva, yva),
        "test_auc": _auc(Xte, yte),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--max-graphs", type=int, default=None)
    ap.add_argument("--pools", type=str, default="mean,max,attn")
    ap.add_argument("--mode", type=str, default="coverage", choices=["coverage", "B0"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", type=str, default=None)
    args = ap.parse_args()

    env = check_env()
    log(f"env: {env}")
    if args.check_only:
        out = {"protocol_id": "ogb-molhiv-v0", "check_only": True, "env": env}
        if env.get("ogb"):
            try:
                bundle = load_molhiv(root=args.data_root, max_graphs=args.max_graphs or 100)
                out["meta"] = bundle.meta
                log(f"load ok: {bundle.meta}")
            except Exception as e:
                out["load_error"] = str(e)
                log(f"load error: {e}")
        path = _TRACK / "results" / "molhiv" / "check.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(f"wrote {path}")
        return

    if not env.get("ogb"):
        log("ERROR: ogb missing. pip install -r tracks/ksvd/configs/requirements-molhiv.txt")
        sys.exit(1)

    t0 = time.time()
    log(f"Loading molhiv max_graphs={args.max_graphs}...")
    bundle = load_molhiv(root=args.data_root, max_graphs=args.max_graphs)
    log(f"  {bundle.meta}")
    graphs, y = bundle.graphs, bundle.y
    tr, va, te = bundle.split["train"], bundle.split["valid"], bundle.split["test"]
    if len(tr) < 10 or len(va) < 5 or len(te) < 5:
        log("ERROR: split too small after max_graphs clip; increase --max-graphs")
        sys.exit(1)

    base = GraphLevelConfig(
        p=0.5,
        q=2.0,
        walk_length=8,
        max_nodes=8,
        max_walks=12,
        edge_decay=0.7,
        n_atoms=16,
        T=3,
        T_min=2,
        ksvd_iter=6,
        seed=args.seed,
        max_train_patches=8000,
    )

    log(f"Learning shared D on train ({len(tr)} graphs), mode={args.mode}...")
    D, dinfo = learn_shared_D_graph_level(graphs, tr, base, mode=args.mode)
    log(f"  recon={dinfo.get('recon_rel'):.4f} patches={dinfo.get('n_patches_used')}")

    results = {
        "protocol_id": "molhiv-struct-probe-v0",
        "parent_protocol": "ogb-molhiv-v0",
        "meta": bundle.meta,
        "mode": args.mode,
        "dinfo": {k: v for k, v in dinfo.items() if not isinstance(v, (list, np.ndarray))},
        "baselines": {},
        "pools": {},
    }

    # degree baseline
    log("Degree histogram baseline...")
    Xdeg = degree_hist_features(graphs)
    results["baselines"]["degree"] = fit_auc(Xdeg[tr], y[tr], Xdeg[va], y[va], Xdeg[te], y[te], args.seed)
    log(f"  degree test_auc={results['baselines']['degree']['test_auc']:.4f}")

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    for pool in pools:
        cfg = GraphLevelConfig(
            **{**base.__dict__, "readout_mode": "pool", "pool": pool, "seed": args.seed}
        )
        log(f"Encode pool={pool}...")
        Xtr = encode_split(graphs, tr, D, cfg, mode=args.mode)
        Xva = encode_split(graphs, va, D, cfg, mode=args.mode)
        Xte = encode_split(graphs, te, D, cfg, mode=args.mode)
        r = fit_auc(Xtr, y[tr], Xva, y[va], Xte, y[te], args.seed)
        results["pools"][pool] = {**r, "dim": int(Xtr.shape[1])}
        log(
            f"  {pool}: val={r['valid_auc']:.4f} test={r['test_auc']:.4f} dim={Xtr.shape[1]}"
        )

    results["elapsed_sec"] = round(time.time() - t0, 2)
    out_dir = _TRACK / "results" / "molhiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"probe_n{bundle.meta['n_used']}" if args.max_graphs else "probe_full"
    json_path = out_dir / f"{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# molhiv structure-only probe",
        "",
        f"Protocol: `molhiv-struct-probe-v0` · n_used={bundle.meta['n_used']}",
        "",
        f"Degree test AUC: **{results['baselines']['degree']['test_auc']:.4f}**",
        "",
        "| pool | valid AUC | test AUC | dim |",
        "|------|-----------|----------|-----|",
    ]
    for k, v in results["pools"].items():
        lines.append(
            f"| {k} | {v['valid_auc']:.4f} | {v['test_auc']:.4f} | {v['dim']} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        "- P1 pass if best pool **test AUC > degree** and not random (~0.5).",
        "- Then run dual-channel (`run_molhiv_dual.py`) for P2.",
        "",
        f"Elapsed: {results['elapsed_sec']}s",
        "",
    ]
    md_path = out_dir / f"{tag}_SUMMARY.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
