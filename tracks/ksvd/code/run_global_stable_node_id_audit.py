"""Audit graph-global stable IDs, patch records, and Beam8 replay."""
from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .canonical_slots import reorder_cover_structurally
from .global_stable_ids import (
    StableNodeIDs,
    ambiguity_audit,
    compute_global_wl_ids,
    compute_rooted_wl_ids,
    mapped_id_audit,
    reorder_by_stable_ids,
)
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import (
    PatchCover,
    audit_cover,
    cover_set_similarity,
    cover_vectors,
    patch_budget,
    remap_cover,
)
from .run_overlap_cover_audit import FAMILIES, generate_graph


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/global_stable_node_id_audit_20260805.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/GLOBAL_STABLE_NODE_ID_AUDIT_20260805.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_GLOBAL_STABLE_NODE_ID_PROTOCOL_20260805.md"
ID_METHODS: dict[str, Callable[[np.ndarray], StableNodeIDs]] = {
    "GLOBAL_WL": compute_global_wl_ids,
    "ROOTED_WL": compute_rooted_wl_ids,
}
SAMPLER_BRANCHES = ("RAW_BEAM8", "STABLE_ID_PREORDER_BEAM8")


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _identity_ids(n_nodes: int) -> StableNodeIDs:
    keys = tuple((node,) for node in range(n_nodes))
    return StableNodeIDs(
        method="input_id",
        canonical_ids=tuple(range(n_nodes)),
        class_ids=tuple(range(n_nodes)),
        unique_mask=tuple(True for _ in range(n_nodes)),
        order=tuple(range(n_nodes)),
        class_sizes=tuple(1 for _ in range(n_nodes)),
        class_keys=keys,
    )


def _true_edges(adjacency: np.ndarray) -> set[tuple[int, int]]:
    return {
        (left, right)
        for left in range(adjacency.shape[0])
        for right in range(left + 1, adjacency.shape[0])
        if adjacency[left, right] != 0
    }


def _observed_pairs(cover: PatchCover) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for patch in cover.patches:
        result.update(tuple(sorted(pair)) for pair in combinations(patch.node_ids, 2))
    return result


def relabeled_stable_to_base_mapping(
    base_ids: StableNodeIDs,
    relabeled_ids: StableNodeIDs,
    permutation: Sequence[int],
) -> np.ndarray:
    """Map relabeled stable coordinates through abstract identity to base coordinates."""
    return np.asarray(
        [
            base_ids.canonical_ids[int(permutation[new_node])]
            for new_node in relabeled_ids.order
        ],
        dtype=np.int64,
    )


def _map_cover_to_stable_namespace(
    adjacency: np.ndarray,
    cover: PatchCover,
    ids: StableNodeIDs,
) -> tuple[np.ndarray, PatchCover]:
    stable_adjacency = reorder_by_stable_ids(adjacency, ids)
    mapping = np.asarray(ids.canonical_ids, dtype=np.int64)
    return stable_adjacency, remap_cover(cover, mapping, stable_adjacency)


def _record_signatures(
    adjacency: np.ndarray,
    cover: PatchCover,
    ids: StableNodeIDs,
) -> dict[str, Any]:
    stable_adjacency, stable_cover = _map_cover_to_stable_namespace(
        adjacency, cover, ids
    )
    rooted_cover, _stats = reorder_cover_structurally(
        stable_adjacency, stable_cover, "rooted_canonical"
    )
    ordered_memberships = tuple(patch.node_ids for patch in stable_cover.patches)
    rooted_memberships = tuple(patch.node_ids for patch in rooted_cover.patches)
    class_memberships = tuple(
        tuple(sorted(ids.class_ids[node] for node in patch.node_ids))
        for patch in cover.patches
    )
    transition_records = []
    for left, right, transition in zip(
        rooted_cover.patches,
        rooted_cover.patches[1:],
        rooted_cover.transitions,
    ):
        transition_records.append(
            tuple(
                (left_slot, right_slot, left.node_ids[left_slot])
                for left_slot, right_slot in transition.left_to_right_slots
            )
        )
    residual = _true_edges(adjacency) - _observed_pairs(cover)
    residual_concrete = tuple(
        sorted(
            tuple(sorted((ids.canonical_ids[left], ids.canonical_ids[right])))
            for left, right in residual
        )
    )
    residual_classes = tuple(
        sorted(
            tuple(sorted((ids.class_ids[left], ids.class_ids[right])))
            for left, right in residual
        )
    )
    return {
        "ordered_memberships": ordered_memberships,
        "rooted_memberships": rooted_memberships,
        "class_memberships": class_memberships,
        "transition_records": tuple(transition_records),
        "residual_concrete": residual_concrete,
        "residual_classes": residual_classes,
    }


def _row_match(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right):
        return 0.0
    if not left:
        return 1.0
    return float(np.mean([a == b for a, b in zip(left, right)]))


def _compare_records(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    return {
        "ordered_membership_row_match": _row_match(
            left["ordered_memberships"], right["ordered_memberships"]
        ),
        "rooted_membership_row_match": _row_match(
            left["rooted_memberships"], right["rooted_memberships"]
        ),
        "class_membership_row_match": _row_match(
            left["class_memberships"], right["class_memberships"]
        ),
        "transition_record_match": _row_match(
            left["transition_records"], right["transition_records"]
        ),
        "residual_concrete_match": float(
            left["residual_concrete"] == right["residual_concrete"]
        ),
        "residual_class_match": float(
            left["residual_classes"] == right["residual_classes"]
        ),
    }


def _class_chain_metrics(
    left_cover: PatchCover,
    left_coord_to_class: Sequence[int],
    right_cover: PatchCover,
    right_coord_to_class: Sequence[int],
) -> dict[str, float]:
    left_rows = tuple(
        tuple(left_coord_to_class[node] for node in patch.node_ids)
        for patch in left_cover.patches
    )
    right_rows = tuple(
        tuple(right_coord_to_class[node] for node in patch.node_ids)
        for patch in right_cover.patches
    )
    exact = float(left_rows == right_rows)
    if len(left_rows) != len(right_rows) or not left_rows:
        jaccard = exact
    else:
        values = []
        for left, right in zip(left_rows, right_rows):
            left_multiset = sorted(left)
            right_multiset = sorted(right)
            common = 0
            work = list(right_multiset)
            for value in left_multiset:
                if value in work:
                    common += 1
                    work.remove(value)
            union = len(left_multiset) + len(right_multiset) - common
            values.append(common / union if union else 1.0)
        jaccard = float(np.mean(values))
    return {
        "equivalence_class_chain_match": exact,
        "equivalence_class_patch_jaccard": jaccard,
    }


def _cover_invariants(audit: dict[str, Any], overlap: int) -> bool:
    return bool(
        audit["patch_connected_rate"] == 1.0
        and audit["continuous_transition_fraction"] == 1.0
        and all(value == overlap for value in audit["transition_overlap_exact"])
    )


def _sampler_comparison(
    left_adjacency: np.ndarray,
    left_cover: PatchCover,
    right_adjacency: np.ndarray,
    right_cover: PatchCover,
    *,
    overlap: int,
) -> dict[str, Any]:
    left_audit = audit_cover(left_adjacency, left_cover)
    right_audit = audit_cover(right_adjacency, right_cover)
    left_rooted, _ = reorder_cover_structurally(
        left_adjacency, left_cover, "rooted_canonical"
    )
    right_rooted, _ = reorder_cover_structurally(
        right_adjacency, right_cover, "rooted_canonical"
    )
    left_vectors = cover_vectors(left_rooted)
    right_vectors = cover_vectors(right_rooted)
    vector_match = (
        float(np.mean(np.all(left_vectors == right_vectors, axis=1)))
        if left_vectors.shape == right_vectors.shape
        else 0.0
    )
    transition_match = _row_match(
        [value.left_to_right_slots for value in left_rooted.transitions],
        [value.left_to_right_slots for value in right_rooted.transitions],
    )
    left_raw_rmse = math.sqrt(
        max(0.0, 1.0 - float(left_audit["full_adjacency_accuracy"]))
    )
    right_raw_rmse = math.sqrt(
        max(0.0, 1.0 - float(right_audit["full_adjacency_accuracy"]))
    )
    exact_chain = float(
        len(left_cover.patches) == len(right_cover.patches)
        and all(
            left.node_ids == right.node_ids
            for left, right in zip(left_cover.patches, right_cover.patches)
        )
    )
    return {
        "exact_ordered_chain_match": exact_chain,
        "patch_set_jaccard": cover_set_similarity(left_cover, right_cover),
        "rooted_vector_row_match": vector_match,
        "rooted_transition_map_match": transition_match,
        "left_edge_coverage": float(left_audit["true_edge_coverage"]),
        "right_edge_coverage": float(right_audit["true_edge_coverage"]),
        "left_pair_coverage": float(left_audit["node_pair_coverage"]),
        "right_pair_coverage": float(right_audit["node_pair_coverage"]),
        "absolute_edge_coverage_delta": abs(
            float(left_audit["true_edge_coverage"])
            - float(right_audit["true_edge_coverage"])
        ),
        "absolute_pair_coverage_delta": abs(
            float(left_audit["node_pair_coverage"])
            - float(right_audit["node_pair_coverage"])
        ),
        "absolute_raw_rmse_delta": abs(left_raw_rmse - right_raw_rmse),
        "left_raw_rmse": left_raw_rmse,
        "right_raw_rmse": right_raw_rmse,
        "left_invariants": _cover_invariants(left_audit, overlap),
        "right_invariants": _cover_invariants(right_audit, overlap),
    }


def _summaries(rows: Sequence[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    names = sorted({str(row[field]) for row in rows})
    metrics = sorted(
        key
        for key, value in rows[0].items()
        if key not in {field, "graph_index", "family", "target_degree", "permutation_seed"}
        and isinstance(value, (int, float, bool))
    )
    return {
        name: {
            metric: _mean([row for row in rows if row[field] == name], metric)
            for metric in metrics
        }
        for name in names
    }


def classify(
    id_rows: Sequence[dict[str, Any]],
    record_rows: Sequence[dict[str, Any]],
    sampler_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    id_summary = _summaries(id_rows, "id_method")
    for method, summary in id_summary.items():
        method_rows = [row for row in id_rows if row["id_method"] == method]
        tied_pairs = sum(int(row["ambiguous_pair_count"]) for row in method_rows)
        summary["ambiguous_pair_count_total"] = tied_pairs
        summary["ambiguous_swap_automorphism_fraction"] = (
            sum(
                int(row["ambiguous_pair_count"])
                * float(row["ambiguous_swap_automorphism_fraction"])
                for row in method_rows
            )
            / tied_pairs
            if tied_pairs
            else 1.0
        )
        summary["largest_class_max"] = max(
            int(row["largest_class"]) for row in method_rows
        )
    record_summary = _summaries(record_rows, "record_method")
    sampler_summary = _summaries(sampler_rows, "sampler_branch")
    rooted = id_summary["ROOTED_WL"]
    global_rooted_order_match_rate = _mean(
        id_rows, "global_rooted_order_match"
    )
    selected_id_method = (
        "GLOBAL_WL"
        if global_rooted_order_match_rate == 1.0
        else "ROOTED_WL"
    )
    raw_sampler = sampler_summary["RAW_BEAM8"]
    stable_sampler = sampler_summary["STABLE_ID_PREORDER_BEAM8"]
    class_invariant = all(
        summary["stable_class_match_rate"] == 1.0
        for summary in id_summary.values()
    )
    id_checks = {
        "singleton_fraction_at_least_090": rooted["singleton_fraction"] >= 0.90,
        "singleton_id_match_at_least_0999": rooted[
            "singleton_unique_id_match_rate"
        ]
        >= 0.999,
        "canonical_adjacency_match_at_least_099": rooted[
            "canonical_adjacency_match"
        ]
        >= 0.99,
    }
    id_gate = class_invariant and all(id_checks.values())
    exact_gain = (
        stable_sampler["exact_ordered_chain_match"]
        - raw_sampler["exact_ordered_chain_match"]
    )
    vector_gain = (
        stable_sampler["rooted_vector_row_match"]
        - raw_sampler["rooted_vector_row_match"]
    )
    sampler_checks = {
        "all_invariants": all(
            bool(row["left_invariants"]) and bool(row["right_invariants"])
            for row in sampler_rows
        ),
        "edge_coverage_not_worse_by_001": stable_sampler[
            "left_edge_coverage"
        ]
        >= raw_sampler["left_edge_coverage"] - 0.01,
        "pair_coverage_not_worse_by_001": stable_sampler[
            "left_pair_coverage"
        ]
        >= raw_sampler["left_pair_coverage"] - 0.01,
        "exact_chain_gain_at_least_050": exact_gain >= 0.50,
        "rooted_vector_gain_at_least_030": vector_gain >= 0.30,
    }
    sampler_gate = all(sampler_checks.values())
    if not class_invariant:
        label = "FAIL_GLOBAL_STABLE_ID_INVARIANTS"
    elif id_gate and sampler_gate:
        label = "ADOPT_GLOBAL_STABLE_ID_PREORDER"
    elif id_gate:
        label = "ADOPT_STABLE_IDS_FOR_RECORDS_ONLY"
    else:
        label = "STABLE_EQUIVALENCE_CLASSES_ONLY"
    return {
        "classification": label,
        "class_invariant_gate": class_invariant,
        "id_gate": id_gate,
        "id_checks": id_checks,
        "sampler_gate": sampler_gate,
        "sampler_checks": sampler_checks,
        "exact_chain_match_gain": exact_gain,
        "rooted_vector_row_match_gain": vector_gain,
        "selected_id_method": selected_id_method,
        "global_rooted_order_match_rate": global_rooted_order_match_rate,
        "id_summaries": id_summary,
        "record_summaries": record_summary,
        "sampler_summaries": sampler_summary,
    }


def _f(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    config = payload["config"]
    lines = [
        "# Graph-global stable node ID 与 Beam8 记录稳定性审计",
        "",
        "> 日期：2026-08-05",
        f"> 协议：`{payload['protocol']}`",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 判定",
        "",
        f"{config['graph_count']} graphs × {len(config['permutation_seeds'])} relabel permutations。",
        "",
        f"- class invariant：`{decision['class_invariant_gate']}`；",
        f"- practical stable-ID gate：`{decision['id_gate']}`；checks：`{decision['id_checks']}`；",
        f"- stable-preorder sampler gate：`{decision['sampler_gate']}`；checks：`{decision['sampler_checks']}`；",
        f"- selected practical ID：`{decision['selected_id_method']}`（GLOBAL/ROOTED order match=`{decision['global_rooted_order_match_rate']:.4f}`）。",
        "",
        "限定：fully-singleton 图可获得具体节点身份的严格 replay；ambiguous 图只应主张 equivalence-class、patch vector 与 transition 稳定，不能把对称节点的 concrete ID 称为唯一。",
        "",
        "## 2. Stable ID",
        "",
        "| method | singleton nodes | fully-singleton graphs | largest class mean/max | class match | singleton ID match | all-node concrete match | canonical adjacency match | tied pairs | tied-pair swap-auto | sec/graph |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in decision["id_summaries"].items():
        lines.append(
            f"| {method} | {_f(row['singleton_fraction'])} | "
            f"{_f(row['fully_singleton'])} | {_f(row['largest_class'])}/{int(row['largest_class_max'])} | "
            f"{_f(row['stable_class_match_rate'])} | "
            f"{_f(row['singleton_unique_id_match_rate'])} | "
            f"{_f(row['all_node_concrete_id_match_rate'])} | "
            f"{_f(row['canonical_adjacency_match'])} | "
            f"{int(row['ambiguous_pair_count_total'])} | "
            f"{_f(row['ambiguous_swap_automorphism_fraction'])} | "
            f"{_f(row['id_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "Singleton ID 才是可证明的 unique stable ID；non-singleton class 内的 concrete ID 只是 opaque handle。",
            "",
            "## 3. Frozen-cover records",
            "",
            "| record IDs | ordered membership | rooted membership | class membership | transition | residual concrete | residual class |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, row in decision["record_summaries"].items():
        lines.append(
            f"| {method} | {_f(row['ordered_membership_row_match'])} | "
            f"{_f(row['rooted_membership_row_match'])} | "
            f"{_f(row['class_membership_row_match'])} | "
            f"{_f(row['transition_record_match'])} | "
            f"{_f(row['residual_concrete_match'])} | "
            f"{_f(row['residual_class_match'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. Relabel + Beam8 resampling",
            "",
            "| branch | abstract exact chain | abstract patch Jaccard | class chain | direct-coordinate replay (control) | rooted vector rows | transition map | edge/pair cover | edge/pair delta | RAW RMSE delta | sec |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for branch, row in decision["sampler_summaries"].items():
        lines.append(
            f"| {branch} | {_f(row['exact_ordered_chain_match'])} | "
            f"{_f(row['patch_set_jaccard'])} | "
            f"{_f(row['equivalence_class_chain_match'])} | "
            f"{_f(row['canonical_coordinate_chain_match'])} | "
            f"{_f(row['rooted_vector_row_match'])} | "
            f"{_f(row['rooted_transition_map_match'])} | "
            f"{_f(row['left_edge_coverage'])}/{_f(row['left_pair_coverage'])} | "
            f"{_f(row['absolute_edge_coverage_delta'])}/{_f(row['absolute_pair_coverage_delta'])} | "
            f"{_f(row['absolute_raw_rmse_delta'])} | "
            f"{_f(row['sampling_seconds'])} |"
        )
    lines.extend(
        [
            "",
            f"Abstract-node exact-chain gain：`{_f(decision['exact_chain_match_gain'])}`；rooted-vector-row gain：`{_f(decision['rooted_vector_row_match_gain'])}`。",
            "",
            "`direct-coordinate replay` 不参与上述 gain：stable-preorder 行表示 canonical preprocessing 后的确定性 consistency；RAW 行没有共同 canonical 坐标，只是未映射 input-index control，不应解释为稳定性指标。",
            "",
            "## 5. 分层",
            "",
            "| family | degree | stable-ID singleton | stable exact chain | stable vector rows | stable edge cover |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["strata"]:
        lines.append(
            f"| {row['family']} | {row['target_degree']} | "
            f"{_f(row['singleton_fraction'])} | {_f(row['exact_chain_match'])} | "
            f"{_f(row['rooted_vector_row_match'])} | {_f(row['edge_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "### 5.1 Fully-singleton 与 ambiguous graphs",
            "",
            "| branch | stratum | trials | abstract exact chain | abstract Jaccard | class chain | canonical consistency | vectors | transition |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["symmetry_strata"]:
        lines.append(
            f"| {row['sampler_branch']} | {row['stratum']} | {row['trial_count']} | "
            f"{_f(row['exact_ordered_chain_match'])} | {_f(row['patch_set_jaccard'])} | "
            f"{_f(row['equivalence_class_chain_match'])} | "
            f"{_f(row['canonical_coordinate_chain_match'])} | "
            f"{_f(row['rooted_vector_row_match'])} | "
            f"{_f(row['rooted_transition_map_match'])} |"
        )
    lines.extend(
        [
            "",
            "## 6. 解释边界",
            "",
            "- global stable ID 用于 membership/transition/stitching/residual；45D KSVD local slots 仍使用 rooted canonical，不按 global ID 排序。",
            "- 本轮检验的是同图 numeric relabel，不是加减边后的 perturbation stability。",
            "- stable ID 只在单张图内部有意义，不赋予跨图相同 ID 共同语义。",
            "- 对称节点只能获得 stable class；具体 labeled identity 仍需 opaque sidecar。要恢复输入的原始 labeled adjacency，仍需 canonical-ID→input-ID 映射；稳定 ID 不会免费消除该 sidecar。",
            "- 使用 SHA-256 digest 承载离散 WL signature；重编号稳定性是实测严格匹配，理论上仍采用密码学碰撞可忽略假设，不把 digest 称为无条件数学证明。",
            "- stable-preorder 不改变 Beam8 objective；收益若存在来自消除输入 numeric-order 与 tie-list ordering 的影响。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_symmetry_strata(
    sampler_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for branch in SAMPLER_BRANCHES:
        for fully_singleton, stratum in (
            (True, "fully_singleton"),
            (False, "ambiguous"),
        ):
            rows = [
                row
                for row in sampler_rows
                if row["sampler_branch"] == branch
                and bool(row["fully_singleton_graph"]) == fully_singleton
            ]
            if not rows:
                continue
            output.append(
                {
                    "sampler_branch": branch,
                    "stratum": stratum,
                    "trial_count": len(rows),
                    "exact_ordered_chain_match": _mean(
                        rows, "exact_ordered_chain_match"
                    ),
                    "patch_set_jaccard": _mean(rows, "patch_set_jaccard"),
                    "equivalence_class_chain_match": _mean(
                        rows, "equivalence_class_chain_match"
                    ),
                    "canonical_coordinate_chain_match": _mean(
                        rows, "canonical_coordinate_chain_match"
                    ),
                    "rooted_vector_row_match": _mean(
                        rows, "rooted_vector_row_match"
                    ),
                    "rooted_transition_map_match": _mean(
                        rows, "rooted_transition_map_match"
                    ),
                }
            )
    return output


def _build_strata(
    id_rows: Sequence[dict[str, Any]], sampler_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["family"], row["target_degree"]) for row in id_rows})
    for family, degree in keys:
        ids = [
            row
            for row in id_rows
            if row["family"] == family
            and row["target_degree"] == degree
            and row["id_method"] == "GLOBAL_WL"
        ]
        samplers = [
            row
            for row in sampler_rows
            if row["family"] == family
            and row["target_degree"] == degree
            and row["sampler_branch"] == "STABLE_ID_PREORDER_BEAM8"
        ]
        output.append(
            {
                "family": family,
                "target_degree": degree,
                "singleton_fraction": _mean(ids, "singleton_fraction"),
                "exact_chain_match": _mean(samplers, "exact_ordered_chain_match"),
                "rooted_vector_row_match": _mean(samplers, "rooted_vector_row_match"),
                "edge_coverage": _mean(samplers, "left_edge_coverage"),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument(
        "--permutation-seeds",
        type=int,
        nargs="+",
        default=[960101, 960102, 960103],
    )
    parser.add_argument("--cover-seed", type=int, default=970101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=3)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    graph_sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    id_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    sampler_rows: list[dict[str, Any]] = []
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    graph_sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.multiplier,
                )
                sampler_seed = int(
                    np.random.SeedSequence(
                        [args.cover_seed, graph_index]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                base_ids: dict[str, StableNodeIDs] = {}
                base_id_seconds: dict[str, float] = {}
                for method, compute in ID_METHODS.items():
                    started = time.perf_counter()
                    base_ids[method] = compute(adjacency)
                    base_id_seconds[method] = time.perf_counter() - started
                global_rooted_order_match = (
                    base_ids["GLOBAL_WL"].canonical_ids
                    == base_ids["ROOTED_WL"].canonical_ids
                )
                started = time.perf_counter()
                base_raw_cover = sample_marginal_candidate_cover(
                    adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                )
                base_raw_seconds = time.perf_counter() - started
                stable_adjacency = reorder_by_stable_ids(
                    adjacency, base_ids["GLOBAL_WL"]
                )
                started = time.perf_counter()
                base_stable_cover = sample_marginal_candidate_cover(
                    stable_adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                )
                base_stable_seconds = time.perf_counter() - started
                base_records = {
                    "INPUT_ID": _record_signatures(
                        adjacency,
                        base_raw_cover,
                        _identity_ids(args.n_nodes),
                    ),
                    **{
                        method: _record_signatures(
                            adjacency, base_raw_cover, base_ids[method]
                        )
                        for method in ID_METHODS
                    },
                }

                for permutation_seed in args.permutation_seeds:
                    permutation = np.random.default_rng(
                        np.random.SeedSequence(
                            [permutation_seed, graph_index]
                        ).generate_state(1, dtype=np.uint32)[0]
                    ).permutation(args.n_nodes)
                    inverse = np.empty(args.n_nodes, dtype=np.int64)
                    inverse[permutation] = np.arange(args.n_nodes)
                    relabeled_adjacency = adjacency[np.ix_(permutation, permutation)]
                    mapped_cover = remap_cover(
                        base_raw_cover, inverse, relabeled_adjacency
                    )
                    relabeled_ids: dict[str, StableNodeIDs] = {}
                    relabeled_id_seconds: dict[str, float] = {}
                    for method, compute in ID_METHODS.items():
                        started = time.perf_counter()
                        relabeled_ids[method] = compute(relabeled_adjacency)
                        relabeled_id_seconds[method] = (
                            time.perf_counter() - started
                        )
                    relabeled_order_match = (
                        relabeled_ids["GLOBAL_WL"].canonical_ids
                        == relabeled_ids["ROOTED_WL"].canonical_ids
                    )
                    for method in ID_METHODS:
                        audit = mapped_id_audit(
                            base_ids[method], relabeled_ids[method], permutation
                        )
                        canonical_match = np.array_equal(
                            reorder_by_stable_ids(adjacency, base_ids[method]),
                            reorder_by_stable_ids(
                                relabeled_adjacency, relabeled_ids[method]
                            ),
                        )
                        ambiguity = ambiguity_audit(
                            adjacency, base_ids[method]
                        )
                        id_rows.append(
                            {
                                "graph_index": graph_index,
                                "family": family,
                                "target_degree": degree,
                                "permutation_seed": int(permutation_seed),
                                "id_method": method,
                                "singleton_fraction": base_ids[
                                    method
                                ].singleton_fraction,
                                "fully_singleton": base_ids[method].fully_singleton,
                                "largest_class": base_ids[method].largest_class,
                                "ambiguous_class_count": base_ids[
                                    method
                                ].ambiguous_class_count,
                                "canonical_adjacency_match": bool(canonical_match),
                                "global_rooted_order_match": bool(
                                    global_rooted_order_match
                                    and relabeled_order_match
                                ),
                                **ambiguity,
                                "id_seconds": float(
                                    (
                                        base_id_seconds[method]
                                        + relabeled_id_seconds[method]
                                    )
                                    / 2.0
                                ),
                                **audit,
                            }
                        )
                    relabeled_record_ids = {
                        "INPUT_ID": _identity_ids(args.n_nodes),
                        **relabeled_ids,
                    }
                    for method, ids in relabeled_record_ids.items():
                        right = _record_signatures(
                            relabeled_adjacency, mapped_cover, ids
                        )
                        record_rows.append(
                            {
                                "graph_index": graph_index,
                                "family": family,
                                "target_degree": degree,
                                "permutation_seed": int(permutation_seed),
                                "record_method": method,
                                **_compare_records(base_records[method], right),
                            }
                        )

                    started = time.perf_counter()
                    relabeled_raw_cover = sample_marginal_candidate_cover(
                        relabeled_adjacency,
                        np.random.default_rng(sampler_seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                        retained_beam=args.retained_beam,
                        candidate_restarts=args.candidate_restarts,
                    )
                    raw_seconds = time.perf_counter() - started
                    mapped_raw_cover = remap_cover(
                        relabeled_raw_cover, permutation, adjacency
                    )
                    raw_comparison = _sampler_comparison(
                        adjacency,
                        base_raw_cover,
                        adjacency,
                        mapped_raw_cover,
                        overlap=args.target_overlap,
                    )
                    raw_direct_exact = float(
                        len(base_raw_cover.patches)
                        == len(relabeled_raw_cover.patches)
                        and all(
                            left.node_ids == right.node_ids
                            for left, right in zip(
                                base_raw_cover.patches,
                                relabeled_raw_cover.patches,
                            )
                        )
                    )
                    raw_classes = tuple(base_ids["GLOBAL_WL"].class_ids)
                    sampler_rows.append(
                        {
                            "graph_index": graph_index,
                            "family": family,
                            "target_degree": degree,
                            "permutation_seed": int(permutation_seed),
                            "sampler_branch": "RAW_BEAM8",
                            "fully_singleton_graph": base_ids[
                                "GLOBAL_WL"
                            ].fully_singleton,
                            "sampling_seconds": float(
                                (base_raw_seconds + raw_seconds) / 2.0
                            ),
                            "canonical_coordinate_chain_match": raw_direct_exact,
                            **_class_chain_metrics(
                                base_raw_cover,
                                raw_classes,
                                mapped_raw_cover,
                                raw_classes,
                            ),
                            **raw_comparison,
                        }
                    )

                    relabeled_stable_adjacency = reorder_by_stable_ids(
                        relabeled_adjacency, relabeled_ids["GLOBAL_WL"]
                    )
                    started = time.perf_counter()
                    relabeled_stable_cover = sample_marginal_candidate_cover(
                        relabeled_stable_adjacency,
                        np.random.default_rng(sampler_seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                        retained_beam=args.retained_beam,
                        candidate_restarts=args.candidate_restarts,
                    )
                    stable_seconds = time.perf_counter() - started
                    # Map relabeled stable coordinates back through abstract node
                    # identity into the base stable coordinate system. Comparing the
                    # two un-mapped covers is only a deterministic consistency check.
                    relabeled_coord_to_base_coord = (
                        relabeled_stable_to_base_mapping(
                            base_ids["GLOBAL_WL"],
                            relabeled_ids["GLOBAL_WL"],
                            permutation,
                        )
                    )
                    mapped_stable_cover = remap_cover(
                        relabeled_stable_cover,
                        relabeled_coord_to_base_coord,
                        stable_adjacency,
                    )
                    canonical_coordinate_exact = float(
                        len(base_stable_cover.patches)
                        == len(relabeled_stable_cover.patches)
                        and all(
                            left.node_ids == right.node_ids
                            for left, right in zip(
                                base_stable_cover.patches,
                                relabeled_stable_cover.patches,
                            )
                        )
                    )
                    base_coord_classes = tuple(
                        base_ids["GLOBAL_WL"].class_ids[node]
                        for node in base_ids["GLOBAL_WL"].order
                    )
                    stable_comparison = _sampler_comparison(
                        stable_adjacency,
                        base_stable_cover,
                        stable_adjacency,
                        mapped_stable_cover,
                        overlap=args.target_overlap,
                    )
                    sampler_rows.append(
                        {
                            "graph_index": graph_index,
                            "family": family,
                            "target_degree": degree,
                            "permutation_seed": int(permutation_seed),
                            "sampler_branch": "STABLE_ID_PREORDER_BEAM8",
                            "fully_singleton_graph": base_ids[
                                "GLOBAL_WL"
                            ].fully_singleton,
                            "sampling_seconds": float(
                                (base_stable_seconds + stable_seconds) / 2.0
                            ),
                            "canonical_coordinate_chain_match": (
                                canonical_coordinate_exact
                            ),
                            **_class_chain_metrics(
                                base_stable_cover,
                                base_coord_classes,
                                mapped_stable_cover,
                                base_coord_classes,
                            ),
                            **stable_comparison,
                        }
                    )
                print(
                    f"graph={graph_index} {family}/d{degree} "
                    f"global/rooted_singleton="
                    f"{base_ids['GLOBAL_WL'].singleton_fraction:.3f}/"
                    f"{base_ids['ROOTED_WL'].singleton_fraction:.3f}",
                    flush=True,
                )
                graph_index += 1

    decision = classify(id_rows, record_rows, sampler_rows)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "permutation_seeds": args.permutation_seeds,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "graph_count": graph_count,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "labels_used": False,
        },
        "id_rows": id_rows,
        "record_rows": record_rows,
        "sampler_rows": sampler_rows,
        "strata": _build_strata(id_rows, sampler_rows),
        "symmetry_strata": _build_symmetry_strata(sampler_rows),
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
