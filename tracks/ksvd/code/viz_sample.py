"""
Visualize B0 vs RW patches on multiple synthetic scenarios.

Usage (from tracks/ksvd):
  python -m code.viz_sample
Outputs: results/viz/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.graph import Graph, from_edges  # noqa: E402
from code.sample import SampleConfig, run_method  # noqa: E402
from code.synthetic_task import make_c4_vs_longcycle, make_triangle_vs_longcycle  # noqa: E402


def spring_layout(g: Graph, seed: int = 0, iters: int = 80) -> dict[int, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    nodes = g.nodes
    n = len(nodes)
    if n == 0:
        return {}
    pos = {u: rng.normal(size=2) for u in nodes}
    for _ in range(iters):
        disp = {u: np.zeros(2) for u in nodes}
        # repulsion
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                d = pos[u] - pos[v]
                dist = np.linalg.norm(d) + 1e-3
                f = (d / dist) * (0.5 / dist)
                disp[u] += f
                disp[v] -= f
        # attraction along edges
        for u, v in g.edges():
            d = pos[v] - pos[u]
            dist = np.linalg.norm(d) + 1e-3
            f = d * 0.05 * dist
            disp[u] += f
            disp[v] -= f
        for u in nodes:
            pos[u] = pos[u] + 0.1 * disp[u]
    # normalize to [0,1]^2
    xs = np.array([pos[u][0] for u in nodes])
    ys = np.array([pos[u][1] for u in nodes])
    xs = (xs - xs.min()) / (np.ptp(xs) + 1e-9)
    ys = (ys - ys.min()) / (np.ptp(ys) + 1e-9)
    return {u: (float(xs[i]), float(ys[i])) for i, u in enumerate(nodes)}


def _draw_graph_ax(ax, g: Graph, pos, highlight: set[int] | None = None, title: str = ""):
    import matplotlib.pyplot as plt

    highlight = highlight or set()
    for u, v in g.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        both = u in highlight and v in highlight
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#e74c3c" if both else "#bdc3c7",
            lw=2.2 if both else 1.0,
            zorder=1,
        )
    for u in g.nodes:
        x, y = pos[u]
        c = "#e74c3c" if u in highlight else "#3498db"
        ax.scatter([x], [y], s=120 if u in highlight else 70, c=c, zorder=2, edgecolors="k", lw=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def scenario_graphs() -> list[tuple[str, Graph, str]]:
    """Diverse scenes for qualitative comparison."""
    out = []
    # 1 pure C4
    out.append(("pure_C4", from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]), "cycle vertex 0"))
    # 2 C8
    out.append(
        (
            "pure_C8",
            from_edges(8, [(i, (i + 1) % 8) for i in range(8)]),
            "cycle vertex 0",
        )
    )
    # 3 C4 + trees from dataset
    gs, y, _ = make_c4_vs_longcycle(n_per_class=5, n_nodes=14, long_cycle=8, seed=1)
    for g, lab in zip(gs, y):
        if lab == 0:
            out.append(("c4_plus_trees", g, "a degree-2 cycle-ish node"))
            break
    for g, lab in zip(gs, y):
        if lab == 1:
            out.append(("c8_plus_trees", g, "on long cycle"))
            break
    # 4 two triangles distant
    # path 0-1-2-3-4-5-6 with triangles at ends
    edges = [(i, i + 1) for i in range(6)]
    edges += [(0, 7), (0, 8), (7, 8), (6, 9), (6, 10), (9, 10)]
    out.append(("distant_two_triangles", from_edges(11, edges), "endpoint 0"))
    # 5 grid-like
    # 3x3 grid
    n = 3
    e = []
    for i in range(n):
        for j in range(n):
            u = i * n + j
            if j + 1 < n:
                e.append((u, u + 1))
            if i + 1 < n:
                e.append((u, u + n))
    out.append(("grid_3x3", from_edges(9, e), "corner 0"))
    # 6 star
    e = [(0, i) for i in range(1, 8)]
    out.append(("star", from_edges(8, e), "hub 0"))
    return out


def pick_start(g: Graph, hint: str) -> int:
    # prefer a degree-2 node if any (cycle-like)
    deg2 = [u for u in g.nodes if len(g.neighbors(u)) == 2]
    if deg2:
        return min(deg2)
    return min(g.nodes, key=lambda u: (-len(g.neighbors(u)), u))


def visualize_all(out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"scenes": []}

    scenes = scenario_graphs()
    for name, g, hint in scenes:
        pos = spring_layout(g, seed=0)
        start = pick_start(g, hint)

        # B0 around start
        S_b0 = {start} | set(g.neighbors(start))
        # RW: force start for fair visual
        cfg = SampleConfig(
            p=1.0,
            q=1.0,
            walk_length=10,
            max_nodes=6,
            num_walks=1,
            edge_decay=1.0,
            no_backtrack=True,
            seed=0,
        )
        # manual one walk from start
        from code.sample import node2vec_walk

        rng = __import__("random").Random(0)
        walk, _ = node2vec_walk(g, start, cfg, rng, {e: 1.0 for e in g.edges()})
        S_rw = set()
        for u in walk:
            S_rw.add(u)
            if len(S_rw) >= cfg.max_nodes:
                break

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        _draw_graph_ax(axes[0], g, pos, set(), f"{name}\nfull graph")
        _draw_graph_ax(
            axes[1],
            g,
            pos,
            S_b0,
            f"B0 1-hop from {start}\n|S|={len(S_b0)}",
        )
        _draw_graph_ax(
            axes[2],
            g,
            pos,
            S_rw,
            f"RW L=10 m=6 from {start}\n|S|={len(S_rw)} walk={walk[:8]}…",
        )
        fig.suptitle(f"Patch comparison · start={start}", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)

        # metrics: does induced contain cycle of length 4?
        def has_c4(S: set[int]) -> bool:
            sub = g.induced(S)
            from code.synthetic_task import _count_c4

            return _count_c4(sub) > 0

        summary["scenes"].append(
            {
                "name": name,
                "n": g.n,
                "e": g.num_edges(),
                "start": start,
                "B0_|S|": len(S_b0),
                "RW_|S|": len(S_rw),
                "B0_induced_has_c4": has_c4(S_b0),
                "RW_induced_has_c4": has_c4(S_rw),
                "png": str(path.name),
            }
        )
        print(
            f"  {name}: B0|S|={len(S_b0)} c4={has_c4(S_b0)} | "
            f"RW|S|={len(S_rw)} c4={has_c4(S_rw)} → {path.name}"
        )

    # multi-panel gallery
    n = len(scenes)
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.6 * n))
    if n == 1:
        axes = np.array([axes])
    for row, (name, g, hint) in enumerate(scenes):
        pos = spring_layout(g, seed=0)
        start = pick_start(g, hint)
        S_b0 = {start} | set(g.neighbors(start))
        cfg = SampleConfig(walk_length=10, max_nodes=6, num_walks=1, seed=0)
        from code.sample import node2vec_walk
        import random

        walk, _ = node2vec_walk(g, start, cfg, random.Random(0), {e: 1.0 for e in g.edges()})
        S_rw = set(list(dict.fromkeys(walk))[:6])
        _draw_graph_ax(axes[row, 0], g, pos, set(), name if row == 0 else name)
        _draw_graph_ax(axes[row, 1], g, pos, S_b0, "B0" if row == 0 else "")
        _draw_graph_ax(axes[row, 2], g, pos, S_rw, "RW" if row == 0 else "")
    fig.suptitle("B0 vs RW patches across scenarios (red = sampled nodes)", fontsize=12)
    fig.tight_layout()
    gal = out_dir / "gallery_all.png"
    fig.savefig(gal, dpi=130)
    plt.close(fig)
    summary["gallery"] = gal.name
    print(f"  gallery → {gal}")
    return summary


def main() -> int:
    out_dir = _TRACK / "results" / "viz"
    summary = visualize_all(out_dir)
    (out_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# RW vs B0 patch visualization",
        "",
        "Red nodes = sampled set S; gray/blue = rest of graph.",
        "",
        f"![gallery]({summary['gallery']})",
        "",
        "| scene | B0 |S| | B0 has C4 | RW |S| | RW has C4 | fig |",
        "|-------|------|-----------|------|-----------|-----|",
    ]
    for s in summary["scenes"]:
        md.append(
            f"| {s['name']} | {s['B0_|S|']} | {s['B0_induced_has_c4']} | "
            f"{s['RW_|S|']} | {s['RW_induced_has_c4']} | `{s['png']}` |"
        )
    md += [
        "",
        "Expected: on pure_C4 / c4_plus_trees, B0 often **no** C4 in induced patch; RW with m≥4 often **yes**.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
