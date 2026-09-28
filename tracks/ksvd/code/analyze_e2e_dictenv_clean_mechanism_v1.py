"""Post-hoc analysis tables for round e2e_dictenv_clean_mechanism_v1.

Reads only the JSON artifacts of the round and prints compact markdown tables
(no training, no GPU, official test never loaded).  Safe to run while the
round is in progress; missing artifacts are skipped.

Usage:
    uv run python tracks/ksvd/code/analyze_e2e_dictenv_clean_mechanism_v1.py \
        [--section a|b|c|d|f|all]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_clean_mechanism_v1"
STAGES = {
    "a": RESULTS / "stage_a_cpu_baseline",
    "c": RESULTS / "stage_c_independence",
    "d": RESULTS / "stage_d_relation",
    "f": RESULTS / "stage_f_dictionary_specificity",
}

#: core Stage-B read-out set (distribution-preserving unless noted).
CORE_PROBES = (
    "GS1",
    "EG2",
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "RS1",
    "RS2",
    "RS3",
    "RS4",
    "RS5",
    "N3",
    "N4",
    "N6",
    "N1",
    "N2",
    "A2",
    "A4",
    "A5",
    "T1",
    "EB2",
    "EB3",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:+.{digits}f}" if abs(number) < 1 else f"{number:.{digits}f}"


def section_a() -> None:
    gate = load(STAGES["a"] / "gate_a.json")
    clean = load(RESULTS / "clean_base.json")
    if gate is None:
        print("## Stage A — no gate artifact yet")
        return
    print("## Stage A — BASE vs C6 (soup valid MAE, CPU matched)\n")
    rows = gate["per_arm"]
    seeds = sorted({int(seed) for arm in rows.values() for seed in arm})
    header = "| seed | BASE | C6 | C6-BASE | C1 | C1-BASE |"
    print(header)
    print("|---|---|---|---|---|---|")
    for seed in seeds:
        base = rows.get("BASE", {}).get(str(seed), {}).get("soup_valid_mae")
        c6 = rows.get("C6", {}).get(str(seed), {}).get("soup_valid_mae")
        c1 = rows.get("C1", {}).get(str(seed), {}).get("soup_valid_mae")
        print(
            f"| {seed} | {fmt(base)} | {fmt(c6)} | {fmt(None if (base is None or c6 is None) else c6 - base)} "
            f"| {fmt(c1)} | {fmt(None if (base is None or c1 is None) else c1 - base)} |"
        )
    c6_gate = gate["c6_gate"]
    stats = c6_gate.get("stats", {})
    print(
        f"\nC6 gate: **{c6_gate.get('verdict')}** (adopted={c6_gate.get('adopted')}); "
        f"mean={fmt(stats.get('mean'))} median={fmt(stats.get('median'))} std={fmt(stats.get('std'))} "
        f"range={fmt(stats.get('range'))}"
    )
    if clean:
        print(f"\nCLEAN_BASE = **{clean.get('clean_base')}**")
    if gate.get("c1_fallback_gate"):
        fallback = gate["c1_fallback_gate"]
        print(
            f"C1 fallback gate: **{fallback.get('verdict')}** "
            f"mean={fmt(fallback.get('stats', {}).get('mean'))}"
        )


def _probe_index(payload: Mapping[str, Any]) -> dict[str, float]:
    index: dict[str, float] = {}
    for table, rows in payload.get("tables", {}).items():
        for row in rows:
            index[str(row["probe"])] = float(row["delta_mae"])
    return index


def section_b() -> None:
    summary = load(RESULTS / "stage_b_mechanism" / "mechanism_summary.json")
    if summary is None:
        print("## Stage B — no mechanism summary yet")
        return
    print("## Stage B — core probe deltas by arm/seed (distribution-preserving)\n")
    for tag, payload in sorted(summary.items()):
        print(f"### {tag} (spec {payload['spec']})\n")
        for view in ("own_mask", "identity"):
            deltas = (payload.get("views", {}).get(view) or {}).get("deltas", {})
            core = {key: deltas.get(key) for key in CORE_PROBES if key in deltas}
            line = " | ".join(f"{key} {fmt(value, 4)}" for key, value in core.items())
            print(f"* **{view}**: {line}")
        dictionary = payload.get("dictionary", {})
        print(
            f"* dictionary: active={dictionary.get('active_atoms')} "
            f"effective={fmt(dictionary.get('effective_atoms'), 3)} "
            f"sparsity={fmt(dictionary.get('coord_sparsity_mean'), 4)} "
            f"movement={fmt(dictionary.get('dictionary_movement_frobenius'), 3)}\n"
        )


def section_c() -> None:
    diagnostics = load(RESULTS / "node_edge_independence_diagnostics.json")
    gate = load(STAGES["c"] / "gate_c.json")
    adapt = load(STAGES["c"] / "adaptation_summary_e20.json")
    if diagnostics:
        print("## Stage C1 — frozen binding-replacement diagnostics\n")
        print("| checkpoint | baseline | NODE_INDEP Δ | EDGE_INDEP Δ | BOTH Δ | corr(node) | corr(edge) |")
        print("|---|---|---|---|---|---|---|")
        for tag, payload in sorted(diagnostics.items()):
            base = payload["baseline_valid_mae"]
            variants = payload["variants"]
            node = variants["NODE_INDEP"]
            edge = variants["EDGE_INDEP"]
            both = variants["BOTH_INDEP"]
            print(
                f"| {tag} | {base:.6f} | {node['valid_mae'] - base:+.6f} | "
                f"{edge['valid_mae'] - base:+.6f} | {both['valid_mae'] - base:+.6f} | "
                f"{node.get('prediction_correlation', float('nan')):.4f} | "
                f"{edge.get('prediction_correlation', float('nan')):.4f} |"
            )
        print("\n### per-shell node residual (ratio = ||R||/||U_pair||)\n")
        print("| checkpoint | shell | n_slots | n_multi | ratio_mean | ratio_median | p90 | p95 | zero_frac |")
        print("|---|---|---|---|---|---|---|---|---|")
        for tag, payload in sorted(diagnostics.items()):
            for shell, row in payload["norms"]["node"].items():
                print(
                    f"| {tag} | {shell} | {row['n_slots']} | {row['n_multi']} | "
                    f"{row['ratio_mean']:.4f} | {row['ratio_median']:.4f} | {row['p90']:.4f} | "
                    f"{row['p95']:.4f} | {row.get('ratio_zero_fraction', float('nan')):.4f} |"
                )
        print("\n### per-shellpair edge residual\n")
        print("| checkpoint | shellpair | n_multi | ratio_mean | ratio_median | p90 | p95 | zero_frac |")
        print("|---|---|---|---|---|---|---|---|")
        for tag, payload in sorted(diagnostics.items()):
            for shellpair, row in payload["norms"]["edge"].items():
                print(
                    f"| {tag} | {shellpair} | {row['n_multi']} | {row['ratio_mean']:.4f} | "
                    f"{row['ratio_median']:.4f} | {row['p90']:.4f} | {row['p95']:.4f} | "
                    f"{row.get('ratio_zero_fraction', float('nan')):.4f} |"
                )
    if adapt:
        print("\n## Stage C2 — 20-epoch warm-start adaptation (vs matched continuation)\n")
        control = adapt.get("control")
        print(f"* continuation control ({adapt.get('clean_base')} mask): {fmt(control)}")
        for role in ("node", "edge"):
            row = adapt.get(role)
            if row:
                print(
                    f"* {role}: soup={fmt(row['soup_valid_mae'])} "
                    f"delta_vs_control={fmt(row['delta_vs_control'])} best_epoch={row['best_epoch']}"
                )
    if gate:
        print("\n## Stage C2 — from-scratch independence gates\n")
        for role in ("node", "edge"):
            row = gate.get(role) or {}
            per_seed = (row.get("gate") or {}).get("per_seed", {})
            print(
                f"* {role}: verdict={ (row.get('gate') or {}).get('verdict') } "
                f"per_seed={ {k: round(v, 6) for k, v in per_seed.items()} } "
                f"adaptation={row.get('adaptation') and round(row['adaptation']['delta_vs_control'], 6)}"
            )


def section_d() -> None:
    groups = load(RESULTS / "relation_groups.json")
    decision = load(STAGES["d"] / "screen_decision.json")
    adapt = load(STAGES["d"] / "adaptation_summary_e20.json")
    gate = load(STAGES["d"] / "gate_d.json")
    print("## Stage D — relation simplification\n")
    if groups:
        print(f"* provenance all_passed={groups['provenance_checks'].get('all_passed')}")
    if adapt:
        print(f"* 20-epoch screen control={fmt(adapt.get('control'))}")
        for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
            row = adapt.get(kind)
            if row:
                print(
                    f"  * {kind}: soup={fmt(row['soup_valid_mae'])} "
                    f"delta_vs_control={fmt(row['delta_vs_control'])} best_epoch={row['best_epoch']}"
                )
    if decision:
        print(f"* candidate: **{decision['candidate']}** ({decision['reason']})")
    if gate:
        for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
            row = gate.get(kind) or {}
            print(f"* {kind}: gate={ (row.get('gate') or {}).get('verdict') } raw={row.get('gate')}")


def section_f() -> None:
    gate = load(STAGES["f"] / "gate_f.json")
    final = load(RESULTS / "final_clean.json")
    print("## Stage F — dictionary specificity\n")
    if final:
        print(f"* adopted: {final['adopted']}")
        print(f"* final spec: {final['final_spec']}")
    if gate:
        print(f"* sparse reference: {gate['sparse_reference']}")
        for seed, row in sorted(gate.get("deltas", {}).items(), key=lambda item: int(item[0])):
            print(
                f"  * seed {seed}: sparse={fmt(row['sparse_soup_valid_mae'])} "
                f"dense={fmt(row['dense_soup_valid_mae'])} "
                f"G_dict={fmt(row['delta_dense_minus_sparse'])}"
            )
        print(f"* gate: {gate.get('gate')}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default="all", choices=["a", "b", "c", "d", "f", "all"])
    args = parser.parse_args(argv)
    sections = {
        "a": section_a,
        "b": section_b,
        "c": section_c,
        "d": section_d,
        "f": section_f,
    }
    if args.section == "all":
        for name in ("a", "b", "c", "d", "f"):
            sections[name]()
            print()
    else:
        sections[args.section]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
