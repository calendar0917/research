"""
Pool ablation on synthetic C4 task: mean vs max vs MIL-attn.

Protocol: pool-ablation-v0. No ogb dependency.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.graph_level import (  # noqa: E402
    GraphLevelConfig,
    encode_dataset,
    learn_shared_D_graph_level,
)
from code.synthetic_task import degree_features, make_c4_vs_longcycle  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def cv_acc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return {"acc_mean": float(scores.mean()), "acc_std": float(scores.std()), "scores": scores.tolist()}


def main() -> None:
    t0 = time.time()
    graphs, y, _meta = make_c4_vs_longcycle(n_per_class=100, seed=0)
    y = np.asarray(y)
    idx = np.arange(len(y))
    # single train fold for D (all train via 80% for fair-ish probe)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n_tr = int(0.8 * len(y))
    train_idx = perm[:n_tr]

    base = GraphLevelConfig(
        p=0.5,
        q=2.0,
        walk_length=8,
        max_nodes=8,
        max_walks=12,
        edge_decay=0.7,
        n_atoms=12,
        T=3,
        ksvd_iter=6,
        seed=0,
    )
    log("Learning shared D (coverage)...")
    D, dinfo = learn_shared_D_graph_level(graphs, train_idx, base, mode="coverage")
    log(f"  D recon={dinfo.get('recon_rel'):.4f} patches={dinfo.get('n_patches_used')}")

    results = {
        "protocol_id": "pool-ablation-v0",
        "task": "c4_vs_longcycle",
        "n": int(len(y)),
        "degree": cv_acc(degree_features(graphs), y),
        "pools": {},
        "dinfo": {k: v for k, v in dinfo.items() if not isinstance(v, list)},
    }

    for pool in ("mean", "max", "attn"):
        cfg = GraphLevelConfig(**{**base.__dict__, "readout_mode": "pool", "pool": pool})
        log(f"Encode pool={pool}...")
        X, _ = encode_dataset(graphs, D, cfg, mode="coverage")
        r = cv_acc(X, y)
        results["pools"][pool] = {**r, "dim": int(X.shape[1])}
        log(f"  {pool}: {r['acc_mean']*100:.1f} ± {r['acc_std']*100:.1f}% dim={X.shape[1]}")

    # rich+mean vs rich+attn
    for pool in ("mean", "attn"):
        cfg = GraphLevelConfig(**{**base.__dict__, "readout_mode": "rich", "pool": pool})
        tag = f"rich+{pool}"
        log(f"Encode {tag}...")
        X, _ = encode_dataset(graphs, D, cfg, mode="coverage")
        r = cv_acc(X, y)
        results["pools"][tag] = {**r, "dim": int(X.shape[1])}
        log(f"  {tag}: {r['acc_mean']*100:.1f} ± {r['acc_std']*100:.1f}%")

    results["elapsed_sec"] = round(time.time() - t0, 2)
    out_dir = _TRACK / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pool_ablation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # markdown summary
    lines = [
        "# Pool ablation (synthetic C4)",
        "",
        "Protocol: `pool-ablation-v0`",
        "",
        f"Degree baseline: **{results['degree']['acc_mean']*100:.1f}%**",
        "",
        "| pool | Acc % | dim |",
        "|------|-------|-----|",
    ]
    for k, v in results["pools"].items():
        lines.append(f"| {k} | {v['acc_mean']*100:.1f} ± {v['acc_std']*100:.1f} | {v['dim']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Compare **attn** vs **mean** on same D; gain → keep MIL for molhiv P1.",
        "- No claim on molhiv; synthetic mechanism only.",
        "",
        f"Elapsed: {results['elapsed_sec']}s",
        "",
    ]
    md_path = out_dir / "POOL_ABLATION_SUMMARY.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
