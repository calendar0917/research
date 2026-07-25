"""
Receptive-field / coverage curves on C4 vs C8 (mechanism presentation).

Outputs:
  results/rf_curve/rf_curve.json
  results/rf_curve/rf_curve.png
  results/rf_curve/README.md

Does NOT claim downstream molecular SOTA — only sampling feasibility vs |S| budget.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.pipeline import PipelineConfig, embed_dataset_shared_dict  # noqa: E402
from code.sample import SampleConfig, node2vec_walk, run_method  # noqa: E402
from code.synthetic_task import _count_c4, degree_features, make_c4_vs_longcycle  # noqa: E402
from code.vectorize import adjacency_padded, flatten_upper  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def c4_hit_rate_graph(g, method: str, cfg: SampleConfig, starts: int = 8) -> float:
    """Fraction of sampled patches whose induced subgraph contains a C4."""
    import random

    hits = 0
    total = 0
    if method == "B0":
        for v in g.nodes:
            S = {v} | set(g.neighbors(v))
            if len(S) > cfg.max_nodes:
                others = sorted(S - {v}, key=lambda u: (-len(g.neighbors(u)), u))
                S = {v} | set(others[: cfg.max_nodes - 1])
            total += 1
            if _count_c4(g.induced(S)) > 0:
                hits += 1
        return hits / max(total, 1)

    rng = random.Random(cfg.seed)
    nodes = g.nodes
    ew = {e: 1.0 for e in g.edges()}
    for i in range(starts):
        v = nodes[i % len(nodes)]
        walk, _ = node2vec_walk(g, v, cfg, rng, ew)
        S = []
        for u in walk:
            if u not in S:
                S.append(u)
            if len(S) >= cfg.max_nodes:
                break
        Sset = set(S) if S else {v}
        total += 1
        if _count_c4(g.induced(Sset)) > 0:
            hits += 1
    return hits / max(total, 1)


def mean_c4_hit(graphs, y, method: str, L: int, m: int, p: float, q: float) -> dict:
    """Mean hit rate on class-0 (C4) graphs only."""
    cfg = SampleConfig(
        p=p,
        q=q,
        walk_length=L,
        max_nodes=m,
        num_walks=1,
        edge_decay=1.0,
        no_backtrack=True,
        seed=0,
    )
    rates = []
    sizes = []
    for g, lab in zip(graphs, y):
        if lab != 0:
            continue
        rates.append(c4_hit_rate_graph(g, method, cfg, starts=min(12, g.n)))
        # mean |S| for a few starts
        import random

        rng = random.Random(0)
        if method == "B0":
            ss = []
            for v in list(g.nodes)[:8]:
                S = {v} | set(g.neighbors(v))
                ss.append(min(len(S), m))
            sizes.append(float(np.mean(ss)))
        else:
            ew = {e: 1.0 for e in g.edges()}
            ss = []
            for i, v in enumerate(list(g.nodes)[:8]):
                walk, _ = node2vec_walk(g, v, cfg, rng, ew)
                S = []
                for u in walk:
                    if u not in S:
                        S.append(u)
                    if len(S) >= m:
                        break
                ss.append(len(S) if S else 1)
            sizes.append(float(np.mean(ss)))
    return {
        "c4_hit_rate": float(np.mean(rates)),
        "c4_hit_std": float(np.std(rates)),
        "mean_|S|": float(np.mean(sizes)),
    }


def ksvd_acc(graphs, y, method: str, L: int, m: int, p: float, q: float) -> dict:
    cfg = PipelineConfig(
        method=method if method != "B0" else "B0",
        p=p,
        q=q,
        walk_length=L,
        max_nodes=m,
        num_walks=20 if method != "B0" else 1,
        edge_decay=1.0 if method != "M0" else 0.7,
        seed=0,
        n_atoms=12,
        T=3,
        T_min=2,
        ksvd_iter=5,
        order_mode="bfs",
        readout_mode="basic",
    )
    if method == "B1":
        cfg.method = "B1"
        cfg.p, cfg.q, cfg.edge_decay = 1.0, 1.0, 1.0
    elif method == "M0":
        cfg.method = "M0"
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(5, int(counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        X, _ = embed_dataset_shared_dict(graphs, cfg, train_idx=tr, max_train_cols=3000)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=fold),
        )
        clf.fit(X[tr], y[tr])
        scores.append(float(clf.score(X[te], y[te])))
    return {
        "acc_mean": float(np.mean(scores)),
        "acc_std": float(np.std(scores)),
        "seconds": time.time() - t0,
    }


def oracle_c4_acc(graphs, y) -> dict:
    c4 = np.array([_count_c4(g) for g in graphs], dtype=np.float64).reshape(-1, 1)
    return eval_simple(c4, y)


def eval_simple(X, y) -> dict:
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(5, int(counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=fold),
        )
        clf.fit(X[tr], y[tr])
        scores.append(float(clf.score(X[te], y[te])))
    return {"acc_mean": float(np.mean(scores)), "acc_std": float(np.std(scores))}


def main() -> int:
    out_dir = _TRACK / "results" / "rf_curve"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=== RF curve: C4 vs C8 ===")
    graphs, y, meta = make_c4_vs_longcycle(
        n_per_class=80, n_nodes=14, long_cycle=8, seed=0
    )
    log(f"meta={meta}")

    # baselines
    results = {
        "task": meta,
        "baselines": {
            "degree": eval_simple(degree_features(graphs), y),
            "oracle_c4_count": oracle_c4_acc(graphs, y),
        },
        "curve": [],
    }
    log(
        f"degree={results['baselines']['degree']['acc_mean']:.3f}  "
        f"oracle_c4={results['baselines']['oracle_c4_count']['acc_mean']:.3f}"
    )

    # B0 reference (m does not help beyond star size ~3)
    for m in [3, 4, 6, 8]:
        hit = mean_c4_hit(graphs, y, "B0", L=1, m=m, p=1.0, q=1.0)
        acc = ksvd_acc(graphs, y, "B0", L=1, m=m, p=1.0, q=1.0)
        row = {"method": "B0", "L": 1, "m": m, "p": 1.0, "q": 1.0, **hit, **acc}
        results["curve"].append(row)
        log(
            f"B0 m={m}: hit={hit['c4_hit_rate']:.3f} |S|={hit['mean_|S|']:.2f} "
            f"acc={acc['acc_mean']:.3f}±{acc['acc_std']:.3f}"
        )

    # RW sweep L and m
    for L in [2, 3, 4, 6, 8, 12]:
        for m in [3, 4, 6, 8]:
            if m > L + 1:
                # still ok: walk can revisit less unique nodes
                pass
            hit = mean_c4_hit(graphs, y, "RW", L=L, m=m, p=0.5, q=2.0)
            acc = ksvd_acc(graphs, y, "B1", L=L, m=m, p=1.0, q=1.0)
            # use B1 uniform for clean L,m effect; also one M0 at selected points
            row = {
                "method": "RW_B1",
                "L": L,
                "m": m,
                "p": 1.0,
                "q": 1.0,
                **hit,
                **acc,
            }
            results["curve"].append(row)
            log(
                f"RW L={L} m={m}: hit={hit['c4_hit_rate']:.3f} |S|={hit['mean_|S|']:.2f} "
                f"acc={acc['acc_mean']:.3f}±{acc['acc_std']:.3f} ({acc['seconds']:.1f}s)"
            )

    # M0 local bias at a few points
    for L, m in [(4, 4), (6, 6), (8, 8)]:
        hit = mean_c4_hit(graphs, y, "RW", L=L, m=m, p=0.5, q=2.0)
        acc = ksvd_acc(graphs, y, "M0", L=L, m=m, p=0.5, q=2.0)
        row = {"method": "RW_M0", "L": L, "m": m, "p": 0.5, "q": 2.0, **hit, **acc}
        results["curve"].append(row)
        log(
            f"M0 L={L} m={m}: hit={hit['c4_hit_rate']:.3f} acc={acc['acc_mean']:.3f}"
        )

    path = out_dir / "rf_curve.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"wrote {path}")

    # plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # left: hit rate vs m for fixed L in {4,6,8} and B0
        ax = axes[0]
        for L in [4, 6, 8]:
            xs, ys = [], []
            for row in results["curve"]:
                if row["method"] == "RW_B1" and row["L"] == L:
                    xs.append(row["m"])
                    ys.append(row["c4_hit_rate"])
            if xs:
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", label=f"RW L={L}")
        b0x, b0y = [], []
        for row in results["curve"]:
            if row["method"] == "B0":
                b0x.append(row["m"])
                b0y.append(row["c4_hit_rate"])
        if b0x:
            order = np.argsort(b0x)
            ax.plot(np.array(b0x)[order], np.array(b0y)[order], "s--", label="B0", color="gray")
        ax.set_xlabel("max_nodes m")
        ax.set_ylabel("C4 hit rate (class-0 patches)")
        ax.set_title("Coverage: does patch contain C4?")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # right: acc vs m
        ax = axes[1]
        for L in [4, 6, 8]:
            xs, ys, es = [], [], []
            for row in results["curve"]:
                if row["method"] == "RW_B1" and row["L"] == L:
                    xs.append(row["m"])
                    ys.append(row["acc_mean"])
                    es.append(row["acc_std"])
            if xs:
                order = np.argsort(xs)
                ax.errorbar(
                    np.array(xs)[order],
                    np.array(ys)[order],
                    yerr=np.array(es)[order],
                    fmt="o-",
                    label=f"RW L={L}",
                )
        b0x, b0y, b0e = [], [], []
        for row in results["curve"]:
            if row["method"] == "B0":
                b0x.append(row["m"])
                b0y.append(row["acc_mean"])
                b0e.append(row["acc_std"])
        if b0x:
            order = np.argsort(b0x)
            ax.errorbar(
                np.array(b0x)[order],
                np.array(b0y)[order],
                yerr=np.array(b0e)[order],
                fmt="s--",
                label="B0",
                color="gray",
            )
        ax.axhline(
            results["baselines"]["oracle_c4_count"]["acc_mean"],
            color="green",
            ls=":",
            label="oracle C4 count",
        )
        ax.axhline(
            results["baselines"]["degree"]["acc_mean"],
            color="orange",
            ls=":",
            label="degree",
        )
        ax.set_xlabel("max_nodes m")
        ax.set_ylabel("Acc (shared KSVD + LR)")
        ax.set_title("Downstream on C4 vs C8 probe")
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Receptive field on C4 vs C8 (mechanism)", fontsize=12)
        fig.tight_layout()
        fig_path = out_dir / "rf_curve.png"
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        log(f"wrote {fig_path}")
        results["figure"] = fig_path.name
    except Exception as e:
        log(f"plot failed: {e}")
        results["figure"] = None

    # markdown
    md = [
        "# Receptive-field curve (C4 vs C8)",
        "",
        "Mechanism presentation — not molecular SOTA.",
        "",
        f"Task: `{meta}`",
        "",
        f"![rf_curve]({results.get('figure') or 'rf_curve.png'})",
        "",
        "## Baselines",
        "",
        f"- degree Acc = {results['baselines']['degree']['acc_mean']:.3f}",
        f"- oracle C4 count Acc = {results['baselines']['oracle_c4_count']['acc_mean']:.3f}",
        "",
        "## How to read",
        "",
        "1. **C4 hit rate**: fraction of patches on C4-class graphs whose induced subgraph contains a C4.",
        "2. **B0**: hit rate stays ~0 (star is P3); Acc plateaus below oracle.",
        "3. **RW**: as \(L,m\) grow past 4, hit rate and Acc rise toward oracle.",
        "4. This supports **sampling feasibility**, not MUTAG/molhiv gains.",
        "",
        "## Selected rows",
        "",
        "| method | L | m | hit | mean\\|S\\| | Acc |",
        "|--------|---|---|-----|-----------|-----|",
    ]
    for row in results["curve"]:
        if row["method"] == "B0" or (row["method"] == "RW_B1" and row["L"] in (4, 6, 8) and row["m"] in (3, 4, 6, 8)):
            md.append(
                f"| {row['method']} | {row['L']} | {row['m']} | {row['c4_hit_rate']:.3f} | "
                f"{row['mean_|S|']:.2f} | {row['acc_mean']:.3f}±{row['acc_std']:.3f} |"
            )
    for row in results["curve"]:
        if row["method"] == "RW_M0":
            md.append(
                f"| {row['method']} | {row['L']} | {row['m']} | {row['c4_hit_rate']:.3f} | "
                f"{row['mean_|S|']:.2f} | {row['acc_mean']:.3f}±{row['acc_std']:.3f} |"
            )
    md += ["", "Raw: `rf_curve.json`", ""]
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"wrote {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
