from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# allow `python -m code.run_stage0` from tracks/ksvd
_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.graph import ring_chords  # noqa: E402
from code.metrics import compare_methods, evaluate_bundle  # noqa: E402
from code.sample import SampleConfig, run_method  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _minimal_yaml(text)


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Tiny subset parser for our stage0.yaml if PyYAML missing."""
    # prefer json if user converted; else use defaults + eval-free line parse
    cfg: dict[str, Any] = {
        "protocol_id": "stage0-sample-v0",
        "seed": 0,
        "seeds": [0, 1, 2],
        "graph": {"type": "ring_chords", "n": 20, "chords": [[0, 5], [3, 10], [7, 15]]},
        "sampler": {
            "methods": ["B0", "B1", "M0"],
            "p": 1.0,
            "q": 1.0,
            "walk_length": 8,
            "max_nodes": 12,
            "num_walks": 30,
            "edge_decay": 0.7,
            "no_backtrack": True,
        },
        "output_dir": "results/stage0",
    }
    # override simple scalars if present
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s or s.endswith(":"):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k, v = k.strip(), v.strip()
        if k == "protocol_id":
            cfg["protocol_id"] = v.strip("'\"")
        if k == "output_dir":
            cfg["output_dir"] = v.strip("'\"")
    return cfg


def build_graph(gcfg: dict[str, Any]):
    t = gcfg.get("type", "ring_chords")
    if t == "ring_chords":
        return ring_chords(int(gcfg["n"]), gcfg.get("chords") or [])
    raise ValueError(f"unknown graph type {t}")


def run_one_seed(g, scfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    cfg = SampleConfig(
        p=float(scfg.get("p", 1.0)),
        q=float(scfg.get("q", 1.0)),
        walk_length=int(scfg.get("walk_length", 8)),
        max_nodes=int(scfg.get("max_nodes", 12)),
        num_walks=int(scfg.get("num_walks", 30)),
        edge_decay=float(scfg.get("edge_decay", 0.7)),
        no_backtrack=bool(scfg.get("no_backtrack", True)),
        seed=seed,
    )
    rows = []
    for method in scfg.get("methods", ["B0", "B1", "M0"]):
        bundle = run_method(g, method, cfg)
        row = evaluate_bundle(g, bundle)
        row["seed"] = seed
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KSVD stage0: sampling metrics")
    ap.add_argument(
        "--config",
        type=Path,
        default=_TRACK / "configs" / "stage0.yaml",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = _load_yaml(args.config)
    g = build_graph(cfg.get("graph") or {})
    scfg = cfg.get("sampler") or {}
    seeds = cfg.get("seeds") or [int(cfg.get("seed", 0))]

    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        all_rows.extend(run_one_seed(g, scfg, int(seed)))

    # aggregate by method
    methods = sorted({r["method"] for r in all_rows})
    summary = []
    for m in methods:
        rs = [r for r in all_rows if r["method"] == m]
        keys = [
            "edge_cover",
            "edge_repeat",
            "edge_mean_count",
            "edge_count_per_sg",
            "edge_gini",
            "mean_|S|",
            "p95_|S|",
            "mean_|Es|",
        ]
        agg: dict[str, Any] = {"method": m, "n_seeds": len(rs)}
        for k in keys:
            vals = [float(r[k]) for r in rs]
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1)
            agg[k] = {"mean": mean, "std": var**0.5}
        summary.append(agg)

    cmp = compare_methods([{**s, **{k: s[k]["mean"] for k in s if isinstance(s.get(k), dict)}} for s in summary])
    # fix compare input
    flat = []
    for s in summary:
        flat.append(
            {
                "method": s["method"],
                "edge_cover": s["edge_cover"]["mean"],
                "edge_repeat": s["edge_repeat"]["mean"],
                "edge_count_per_sg": s["edge_count_per_sg"]["mean"],
            }
        )
    cmp = compare_methods(flat)

    payload = {
        "protocol_id": cfg.get("protocol_id", "stage0-sample-v0"),
        "config": cfg,
        "graph": {"n": g.n, "num_edges": g.num_edges()},
        "per_seed": all_rows,
        "summary": summary,
        "compare": cmp,
    }

    out_dir = Path(args.out or (_TRACK / cfg.get("output_dir", "results/stage0")))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage0_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"protocol: {payload['protocol_id']}")
    print(f"graph: n={g.n} edges={g.num_edges()}")
    for s in summary:
        print(
            f"{s['method']}: cover={s['edge_cover']['mean']:.3f} "
            f"cnt/sg={s['edge_count_per_sg']['mean']:.4f} "
            f"|S|={s['mean_|S|']['mean']:.2f}"
        )
    if "M0_count_per_sg_lte_B1" in cmp:
        print(f"M0 cnt/sg ≲ B1: {cmp['M0_count_per_sg_lte_B1']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
