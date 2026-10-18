"""Driver for the SSRC Phase 0 exact CPU calibration.

Run:

    uv run python tracks/structural-sufficiency/results/phase0/code/run_phase0.py

Outputs ``analysis/phase0_results.json`` next to this directory.  No labels,
no GPU, no training, no official molecular test split.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
PHASE0 = HERE.parent
ANALYSIS = PHASE0 / "analysis"
sys.path.insert(0, str(HERE))

import ssrc_toy as toy  # noqa: E402


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def canonical_crosscheck() -> dict[str, Any]:
    """pynauty key == exhaustive canonical == networkx rooted isomorphism.

    The exhaustive check is only run on patches with n <= 8.  We compare every
    pair of patches within a graph and every cross-graph pair for the key
    d=6/d=9 pair on the C24 family.
    """
    out: dict[str, Any] = {}
    g6 = toy.cycle_graph(24, 6)
    g9 = toy.cycle_graph(24, 9)
    for r in range(0, 4):
        p6 = toy.all_rooted_balls(g6, r)
        p9 = toy.all_rooted_balls(g9, r)
        pairs = 0
        bad = 0
        for p1, p2 in itertools.combinations(p6, 2):
            pairs += 1
            a = toy.exhaustive_canonical(p1) == toy.exhaustive_canonical(p2)
            b = toy.pynauty_key(p1) == toy.pynauty_key(p2)
            c = toy.nx_isomorphic_rooted(p1, p2)
            bad += int(not (a == b == c))
        for p1 in p6:
            for p2 in p9:
                pairs += 1
                a = toy.exhaustive_canonical(p1) == toy.exhaustive_canonical(p2)
                b = toy.pynauty_key(p1) == toy.pynauty_key(p2)
                c = toy.nx_isomorphic_rooted(p1, p2)
                bad += int(not (a == b == c))
        out[f"r{r}"] = {"pairs_checked": pairs, "mismatches": bad, "passed": bad == 0}

    # A random sample for the larger grid/caterpillar patches (exhaustive is
    # intentionally not run there).
    sample: dict[str, Any] = {}
    for name, graph, r in [
        ("caterpillar_r3", toy.caterpillar_graph(n_spine=40, distance=6, margin=6), 3),
        ("grid_r2", toy.grid_graph(distance=6), 2),
        ("grid_r3", toy.grid_graph(distance=9), 3),
    ]:
        patches = toy.all_rooted_balls(graph, r)
        rng = __import__("random").Random(20261018)
        pairs = 0
        bad = 0
        for _ in range(200):
            p1, p2 = rng.sample(patches, 2)
            pairs += 1
            b = toy.pynauty_key(p1) == toy.pynauty_key(p2)
            c = toy.nx_isomorphic_rooted(p1, p2)
            bad += int(b != c)
        sample[name] = {"pairs_checked": pairs, "mismatches": bad, "passed": bad == 0}
    out["large_patch_sample"] = sample
    return out


def collision_table() -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}

    specs: list[tuple[str, Callable[..., toy.AttributedGraph], range, range]] = [
        ("cycle_C24", lambda **kw: toy.cycle_graph(24, **kw), range(1, 13), range(0, 5)),
        (
            "caterpillar_40",
            lambda **kw: toy.caterpillar_graph(n_spine=40, margin=6, **kw),
            range(1, 13),
            range(0, 5),
        ),
        ("grid_20x9", lambda **kw: toy.grid_graph(width=20, height=9, a_x=4, a_y=4, **kw), range(1, 10), range(0, 5)),
    ]
    for name, factory, distances, radii in specs:
        rows: dict[str, Any] = {}
        for r in radii:
            t0 = time.perf_counter()
            encodings = {d: toy.encoding(factory(distance=d), r) for d in distances}
            elapsed = time.perf_counter() - t0
            stats = toy.collision_stats(encodings, label_threshold=4)
            stats["encoding_seconds"] = elapsed
            stats["same_6_9"] = encodings[6] == encodings[9] if 6 in encodings and 9 in encodings else None
            stats["both_marks_count_by_distance"] = {
                str(d): toy.both_marks_count(factory(distance=d), r) for d in distances
            }
            rows[f"r{r}"] = stats
        families[name] = {"distances": list(distances), "rows": rows}
    return families


def key_pair_detail() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, factory in [
        ("cycle_C24", lambda d: toy.cycle_graph(24, d)),
        ("caterpillar_40", lambda d: toy.caterpillar_graph(n_spine=40, distance=d, margin=6)),
        ("grid_20x9", lambda d: toy.grid_graph(width=20, height=9, distance=d, a_x=4, a_y=4)),
    ]:
        g6, g9 = factory(6), factory(9)
        rows: dict[str, Any] = {}
        for r in range(0, 5):
            p6 = toy.all_rooted_balls(g6, r)
            p9 = toy.all_rooted_balls(g9, r)
            rows[f"r{r}"] = {
                "same_multiset": toy.encoding(g6, r) == toy.encoding(g9, r),
                "both_marks_patches_d6": sum(p.has_both_marks() for p in p6),
                "both_marks_patches_d9": sum(p.has_both_marks() for p in p9),
                "n_unique_patch_keys_d6": len({toy.pynauty_key(p) for p in p6}),
                "n_unique_patch_keys_d9": len({toy.pynauty_key(p) for p in p9}),
            }
        out[name] = {
            "actual_distance_d6": toy.graph_distance(g6, *toy.markers(g6)),
            "actual_distance_d9": toy.graph_distance(g9, *toy.markers(g9)),
            "rows": rows,
        }
    return out


def invariance_checks() -> dict[str, Any]:
    out: dict[str, Any] = {}
    cases = [
        ("cycle_C24_d6", toy.cycle_graph(24, 6)),
        ("cycle_C24_d9", toy.cycle_graph(24, 9)),
        ("caterpillar_40_d6", toy.caterpillar_graph(n_spine=40, distance=6, margin=6)),
        ("caterpillar_40_d9", toy.caterpillar_graph(n_spine=40, distance=9, margin=6)),
        ("grid_20x9_d6", toy.grid_graph(distance=6)),
        ("grid_20x9_d9", toy.grid_graph(distance=9)),
    ]
    for name, graph in cases:
        out[name] = toy.check_relabel_invariance(graph, radii=range(0, 5), n_permutations=50)
    return out


def edge_order_invariance() -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = __import__("random").Random(20261018)
    for name, graph in [
        ("cycle_C24_d6", toy.cycle_graph(24, 6)),
        ("caterpillar_40_d6", toy.caterpillar_graph(n_spine=40, distance=6, margin=6)),
        ("grid_20x9_d6", toy.grid_graph(distance=6)),
    ]:
        shuffled = list(graph.edges)
        rng.shuffle(shuffled)
        g2 = toy.AttributedGraph(n=graph.n, node_types=graph.node_types, edges=tuple(sorted(shuffled)))
        out[name] = {
            "same_encoding_r2": toy.encoding(graph, 2) == toy.encoding(g2, 2),
            "same_encoding_r3": toy.encoding(graph, 3) == toy.encoding(g2, 3),
        }
    return out


def oracle_checks() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, factory in [
        ("cycle_C24", lambda d: toy.cycle_graph(24, d)),
        ("caterpillar_40", lambda d: toy.caterpillar_graph(n_spine=40, distance=d, margin=6)),
        ("grid_20x9", lambda d: toy.grid_graph(width=20, height=9, distance=d, a_x=4, a_y=4)),
    ]:
        rows: dict[str, Any] = {}
        for d in (6, 9):
            graph = factory(d)
            rows[f"d{d}"] = {
                "dynamic_detect_by_depth": {str(L): toy.dynamic_oracle_detect(graph, L) for L in range(0, 7)},
                "noniter_ab_distance": toy.noniter_distance_oracle(graph),
                "static_rule_r0": toy.oracle_positive_control(graph, 0, 4),
                "static_rule_r1": toy.oracle_positive_control(graph, 1, 4),
                "static_rule_r2": toy.oracle_positive_control(graph, 2, 4),
                "static_rule_r3": toy.oracle_positive_control(graph, 3, 4),
            }
        out[name] = rows
    return out


def positive_control_summary() -> dict[str, Any]:
    """Label = [d <= 4]; deterministic static rule = [some r-ball has both marks]."""
    out: dict[str, Any] = {}
    for name, factory, distances in [
        ("cycle_C24", lambda d: toy.cycle_graph(24, d), range(1, 13)),
        ("caterpillar_40", lambda d: toy.caterpillar_graph(n_spine=40, distance=d, margin=6), range(1, 13)),
        ("grid_20x9", lambda d: toy.grid_graph(width=20, height=9, distance=d, a_x=4, a_y=4), range(1, 10)),
    ]:
        rows: dict[str, Any] = {}
        for r in range(0, 5):
            correct = 0
            total = 0
            confusion: dict[str, int] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            for d in distances:
                pred, truth = toy.oracle_positive_control(factory(d), r, 4)
                correct += int(pred == truth)
                total += 1
                if pred == 1 and truth == 1:
                    confusion["tp"] += 1
                elif pred == 0 and truth == 0:
                    confusion["tn"] += 1
                elif pred == 1 and truth == 0:
                    confusion["fp"] += 1
                else:
                    confusion["fn"] += 1
            rows[f"r{r}"] = {
                "accuracy": correct / total,
                "n": total,
                "confusion": confusion,
            }
        out[name] = rows
    return out


def main() -> int:
    t0 = time.perf_counter()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "protocol": "ssrc-phase0-cpu-calibration-v1",
        "date_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {
            "ssrc_toy.py": sha256(HERE / "ssrc_toy.py"),
            "run_phase0.py": sha256(HERE / "run_phase0.py"),
        },
        "scope": {
            "cpu_only": True,
            "training": False,
            "labels_read": False,
            "official_valid_or_test_read": False,
            "gpu_used": False,
        },
    }
    payload["canonical_crosscheck"] = canonical_crosscheck()
    payload["collision_table"] = collision_table()
    payload["key_pair_detail"] = key_pair_detail()
    payload["relabel_invariance"] = invariance_checks()
    payload["edge_order_invariance"] = edge_order_invariance()
    payload["oracles"] = oracle_checks()
    payload["positive_control"] = positive_control_summary()
    payload["runtime_seconds"] = time.perf_counter() - t0

    # Compact, explicit conclusions (facts of this run only).
    c24 = payload["key_pair_detail"]["cycle_C24"]["rows"]
    cat = payload["key_pair_detail"]["caterpillar_40"]["rows"]
    grid = payload["key_pair_detail"]["grid_20x9"]["rows"]
    payload["proven_facts"] = {
        "cycle_E0_E1_E2_equal_for_d6_d9": all(c24[f"r{r}"]["same_multiset"] for r in (0, 1, 2)),
        "cycle_E3_differs_for_d6_d9": not c24["r3"]["same_multiset"],
        "caterpillar_E2_equal_for_d6_d9": cat["r2"]["same_multiset"],
        "caterpillar_E3_differs_for_d6_d9": not cat["r3"]["same_multiset"],
        "grid_E2_equal_for_d6_d9": grid["r2"]["same_multiset"],
        "grid_E3_differs_for_d6_d9": not grid["r3"]["same_multiset"],
        "canonical_crosscheck_passed": all(
            payload["canonical_crosscheck"][f"r{r}"]["passed"] for r in range(0, 4)
        )
        and all(v["passed"] for v in payload["canonical_crosscheck"]["large_patch_sample"].values()),
        "relabel_invariance_all_passed": all(v["passed"] for v in payload["relabel_invariance"].values()),
        "one_shot_distance_oracle_separates_d6_d9": all(
            payload["oracles"][k]["d6"]["noniter_ab_distance"] == 6
            and payload["oracles"][k]["d9"]["noniter_ab_distance"] == 9
            for k in ("cycle_C24", "caterpillar_40", "grid_20x9")
        ),
        "depth3_dynamic_oracle_separates_d6_d9": all(
            payload["oracles"][k]["d6"]["dynamic_detect_by_depth"]["3"] is True
            and payload["oracles"][k]["d9"]["dynamic_detect_by_depth"]["3"] is False
            for k in ("cycle_C24", "caterpillar_40", "grid_20x9")
        ),
        "positive_control_r2_perfect": all(
            payload["positive_control"][k]["r2"]["accuracy"] == 1.0
            for k in ("cycle_C24", "caterpillar_40", "grid_20x9")
        ),
    }
    out_path = ANALYSIS / "phase0_results.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("SSRC Phase 0 CPU calibration")
    print(f"  git: {payload['git_commit']}")
    print(f"  output: {out_path}")
    print("  key facts:")
    for k, v in payload["proven_facts"].items():
        print(f"    {k}: {v}")
    print(f"  runtime: {payload['runtime_seconds']:.2f}s")
    print("  collision table (n_distinct / label-mixed groups):")
    for fam, fam_data in payload["collision_table"].items():
        vals = [
            f"r{r}:{row['n_distinct_encodings']}/{row['label_mixed_groups']}"
            for r, row in fam_data["rows"].items()
        ]
        print(f"    {fam}: " + " ".join(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
