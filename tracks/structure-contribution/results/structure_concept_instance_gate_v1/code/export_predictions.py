"""Export per-graph predictions for every arm and seed (hard and soft modes).

Usage:
    uv run python code/export_predictions.py [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline as P  # noqa: E402

RESULTS = HERE.parent
ARMS = ("A", "B", "C", "D")
SCRAMBLE_ARMS = ("B", "D")


def export_seed(seed: int) -> Path:
    data, _ = P.prepare_data()
    out_path = RESULTS / "predictions" / f"predictions_seed{seed}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for arm in ARMS:
            model = P.load_model(f"{arm}_seed{seed}")
            for gid, d in sorted(data.items()):
                hard_logit, hard_prob = P.predict_graph(model, d, "true", "hard")
                soft_logit, soft_prob = P.predict_graph(model, d, "true", "soft") if arm in ("C", "D") else (None, None)
                fh.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "arm": arm,
                            "relation_source": "true",
                            "graph_id": gid,
                            "family": d.family,
                            "split": d.split,
                            "base_id": d.base_id,
                            "pair_id": d.pair_id,
                            "bag_twin": d.bag_twin,
                            "label": d.label,
                            "n_nodes": d.n_nodes,
                            "n_edges": d.n_edges,
                            "true_distance": d.meta.get("d"),
                            "hard_logit": hard_logit,
                            "hard_prob": hard_prob,
                            "soft_logit": soft_logit,
                            "soft_prob": soft_prob,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        for arm in SCRAMBLE_ARMS:
            model = P.load_model(f"{arm}_scrambled_seed{seed}")
            for gid, d in sorted(data.items()):
                hard_logit, hard_prob = P.predict_graph(model, d, "scrambled", "hard")
                fh.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "arm": f"{arm}_scrambled",
                            "relation_source": "scrambled",
                            "graph_id": gid,
                            "family": d.family,
                            "split": d.split,
                            "base_id": d.base_id,
                            "pair_id": d.pair_id,
                            "bag_twin": d.bag_twin,
                            "label": d.label,
                            "n_nodes": d.n_nodes,
                            "n_edges": d.n_edges,
                            "true_distance": d.meta.get("d"),
                            "hard_logit": hard_logit,
                            "hard_prob": hard_prob,
                            "soft_logit": None,
                            "soft_prob": None,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    for seed in args.seeds:
        print("wrote", export_seed(seed))


if __name__ == "__main__":
    main()
