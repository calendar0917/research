"""cscl-distance-pooling-bottleneck-v1 driver — CPU audit, matched GPU arms,
frozen official-validation evaluation.

Protocol: ``configs/distance_pooling_bottleneck_v1.yaml`` (frozen before the
run).  Research note: ``notes/distance_pooling_bottleneck_v1.md``.

Stages
------
``audit``   CPU-only: source-path audit, train-10k distance distribution,
            frozen pair-encoder distance-descriptor audit, Hard5/Soft5 feature
            audit, runtime correctness gates and the pre-frozen purchase gate.
``train``   One fresh seed-0 arm (``--arm hard5`` or ``--arm soft5``) on all
            10000 official-train molecules, 240 epochs, last-5 soup.
``check``   Cross-arm matching audit (initialisation / RNG / rows / prep).
``predict`` Freeze both soup checkpoints, write official-validation predictions
            (labels never read), write ``freeze_manifest.json``.
``score``   One official-validation label read, metrics, paired bootstrap,
            subgroups and the pre-frozen decision rule.
``report``  Aggregate the durable summary (optional convenience).

Discipline
----------
* seed 0 only (argparse enforces ``choices=[0]``).
* official test is never loaded and never scored.
* the training task loss is raw-y L1 + the frozen label-free reconstruction.
* no new trainable parameters, no reader-width change, C6_MASK unchanged.
* the Soft5 kernel constants are frozen in :mod:`distance_pooling_v1`.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import distance_pooling_v1 as dp  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    e2e_dictenv_clean_mechanism_v1 as cm,
    e2e_dictenv_h1_clarity_audit as audit,
    e2e_dictenv_p1 as p1,
    e2e_dictenv_scale_v1 as sc,
    zinc_e2e_dictenv_p1 as p1run,
    zinc_full_cycle_target_decomposition_v1 as zftd,
    zinc_joint_dictionary_decision_v1 as zjd,
    zinc_patch_path_pooling as zpp,
    zinc_static_dictionary_pair as sdp,
)

PROTOCOL_ID = "cscl-distance-pooling-bottleneck-v1"
CONFIG_PATH = REPO_ROOT / "tracks/structure-contribution/configs/distance_pooling_bottleneck_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/distance_pooling_bottleneck_v1"

# ---- frozen artifacts / data ----------------------------------------------
ENCODED_TRAIN = sdp.CACHE_DIR / "encoded_train.pt"
ENCODED_VALID = sdp.CACHE_DIR / "encoded_valid.pt"
KSVD_CONF = REPO_ROOT / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1"
HIST_PREP = KSVD_CONF / "all_train_prep.npz"
HIST_PREP_META = KSVD_CONF / "prep_meta.json"
HIST_Y_SOUP = (
    REPO_ROOT
    / "tracks/ksvd/results/zinc_full_cycle_target_decomposition_v1/Y_seed0_raw_soup_state.pt"
)
BUDGET_FREEZE = (
    REPO_ROOT
    / "tracks/structure-contribution/results/zinc_full_y_budget_control_v1/freeze_manifest.json"
)
V3D_EVAL_F_PREDS = (
    REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d_official_eval/valid_F_preds.npz"
)

# ---- frozen training recipe (task section 7.2) -----------------------------
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
SCALE_SEED = 0
H1_LAMBDA = float(cm.H1_LAMBDA)
FULL_PARAMETERS = int(dp.FULL_PARAMETERS)
EXPECTED_STEPS = 18_960

# ---- CPU audit constants ----------------------------------------------------
DISTANCE_PRESENCE_THRESHOLD = 0.10     # >=10% of molecules with d=5 and d>=7 in bucket 4
FEATURE_DIFF_THRESHOLD = 0.01          # >1% normalised pooled-feature difference
FEATURE_DIFF_FRACTION = 0.25           # on >=25% of training molecules
CONSERVATION_RTOL = 1.0e-4
WEIGHT_SUM_ATOL = 1.0e-6

# ---- evaluation constants (pre-frozen) -------------------------------------
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
EXTREME_VALID_INDEX = 172              # valid:0172, historically extreme row
ARM_NAMES = ("Hard5", "Soft5")
ARM_MODES = {"Hard5": "hard5", "Soft5": "soft5"}
DECISION_DELTA_A = -0.010              # delta <= A  -> case A (clear signal)
DECISION_DELTA_B = -0.003              # (A, B]      -> case B (limited)
EXTREME_KEEP_FRACTION = 0.75           # delta without valid:0172 keeps >=75% magnitude
TARGET_LEVELS = (0.09, 0.08)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(message, flush=True)


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def rng_hash() -> dict[str, str]:
    import hashlib

    payload = {
        "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda": (
            hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
            if torch.cuda.is_available()
            else "cpu"
        ),
    }
    return payload


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_train_rows() -> list[Any]:
    """Official-train encoded cache + env (10000 rows).  Never touches valid/test."""
    rows = list(torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False))
    p1run.attach_env(rows, "train")
    if len(rows) != 10000:
        raise RuntimeError(f"expected 10000 encoded train rows, got {len(rows)}")
    return rows


def load_valid_rows() -> list[Any]:
    """Official-validation encoded cache + env (1000 rows).  Labels stay unread."""
    rows = list(torch.load(ENCODED_VALID, map_location="cpu", weights_only=False))
    p1run.attach_env(rows, "valid")
    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 encoded valid rows, got {len(rows)}")
    return rows


def load_prep() -> dict[str, np.ndarray]:
    return {k: z for k, z in np.load(HIST_PREP, allow_pickle=False).items()}


def apply_prep(rows: Sequence[Any], prep: Mapping[str, np.ndarray]) -> None:
    zftd.apply_prep_train_only(rows, prep)


def train_y(rows: Sequence[Any]) -> np.ndarray:
    return np.asarray([float(row.y.reshape(-1)[0]) for row in rows], dtype=np.float64)


def _batches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool) -> list[list[int]]:
    if shuffle:
        order = torch.randperm(int(n), generator=generator).tolist()
    else:
        order = list(range(int(n)))
    return [order[s : s + batch_size] for s in range(0, int(n), batch_size)]


def _batch(rows: Sequence[Any], indices: Sequence[int], targets: torch.Tensor, device: torch.device):
    batch = p1.env_collate([rows[i] for i in indices])
    batch.y = targets[torch.as_tensor(list(indices), dtype=torch.long)].view(-1).clone()
    return batch.to(device)


@torch.no_grad()
def predict_raw(model: nn.Module, rows: Sequence[Any], device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[torch.Tensor] = []
    for indices in _batches(len(rows), BATCH_SIZE, torch.Generator().manual_seed(0), False):
        batch = _batch(rows, indices, torch.zeros(len(rows)), device)
        outputs.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
    return torch.cat(outputs).numpy().astype(np.float64)


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def decide_case(delta: float) -> str:
    """Pre-frozen decision bands (pure function, unit-tested).

    ``A`` requires the additional conditions (CI < 0, ordinary-y subgroup
    improvement, not single-row driven, engineering gates) checked in
    :func:`stage_score`; ``UNCERTAIN`` is raised there when they fail.
    """
    if delta <= DECISION_DELTA_A:
        return "A"
    if delta <= DECISION_DELTA_B:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# stage: audit — distance distribution
# ---------------------------------------------------------------------------


def _graph_from_edge_index(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u != v:
            adjacency[u, v] = 1.0
            adjacency[v, u] = 1.0
    return adjacency


def distance_distribution(
    rows: Sequence[Any], dataset: Any, bfs_limit: int | None = None, log=log
) -> dict[str, Any]:
    """Label-blind train-10k distance census from the stored pair cache, with an
    all-pairs BFS cross-check against the raw PyG graphs."""
    from scipy.sparse.csgraph import shortest_path

    n_molecules = len(rows)
    bfs_rows = n_molecules if bfs_limit is None else min(int(bfs_limit), n_molecules)
    exact_counts = {d: 0 for d in range(1, 13)}  # 12 = 12+ (aggregated bucket)
    exact_present = {d: 0 for d in range(1, 13)}
    bucket_counts = np.zeros(5, dtype=np.int64)
    last_bucket_diversity = np.zeros(n_molecules, dtype=np.int64)
    last_bucket_span3 = np.zeros(n_molecules, dtype=bool)
    last_bucket_multi = np.zeros(n_molecules, dtype=bool)
    last_bucket_mean = np.full(n_molecules, np.nan)
    last_bucket_var = np.full(n_molecules, np.nan)
    last_bucket_entropy = np.full(n_molecules, np.nan)
    diameter = np.zeros(n_molecules, dtype=np.int64)
    n_pairs = np.zeros(n_molecules, dtype=np.int64)
    bucket4_pairs = np.zeros(n_molecules, dtype=np.int64)
    n_atoms = np.zeros(n_molecules, dtype=np.int64)
    ring_rank = np.zeros(n_molecules, dtype=np.int64)
    bfs_mismatch = 0
    bfs_pairs_checked = 0
    max_abs_rel5_gap = 0.0
    max_abs_dist_gap = 0.0
    has_d5_ge7 = np.zeros(n_molecules, dtype=bool)

    for i, row in enumerate(rows):
        rel5 = row.pair_relation[:, 5].to(torch.float64).numpy()
        distances = np.rint(np.expm1(rel5)).astype(np.int64)
        distances = np.maximum(distances, 1)
        max_abs_rel5_gap = max(max_abs_rel5_gap, float(np.max(np.abs(rel5 - np.log1p(distances)))))
        cached_bucket = row.pair_bucket.to(torch.int64).numpy()
        expected_bucket = np.clip(distances, 1, 5) - 1
        if not np.array_equal(cached_bucket, expected_bucket):
            raise RuntimeError(f"row {i}: cached bucket != min(max(d,1),5)-1")
        source = row.pair_index[0].numpy()
        target = row.pair_index[1].numpy()

        # raw-graph all-pairs BFS cross-check
        data = dataset[i]
        n = int(data.x.shape[0])
        n_pairs[i] = distances.shape[0]
        n_atoms[i] = n
        if i < bfs_rows:
            adjacency = _graph_from_edge_index(data.edge_index.numpy(), n)
            all_dist = shortest_path(adjacency, method="D", unweighted=True, directed=False)
            bfs = all_dist[source, target]
            if not np.all(np.isfinite(bfs)):
                raise RuntimeError(f"row {i}: disconnected pair in the raw graph")
            bfs_int = np.rint(bfs).astype(np.int64)
            gaps = np.abs(bfs_int - distances)
            if int(gaps.max()) > 0:
                bfs_mismatch += int((gaps > 0).sum())
            max_abs_dist_gap = max(max_abs_dist_gap, float(gaps.max()))
            bfs_pairs_checked += int(gaps.shape[0])
            n_unique_edges = int(np.count_nonzero(np.triu(adjacency, 1)))
            ring_rank[i] = n_unique_edges - n + _n_components(adjacency)
        else:
            ring_rank[i] = -1

        for d in range(1, int(distances.max()) + 1):
            count = int((distances == d).sum())
            if count:
                exact_counts[min(d, 12)] += count
                exact_present[min(d, 12)] += 1
        for b in range(5):
            bucket_counts[b] += int((cached_bucket == b).sum())
        diameter[i] = int(distances.max())
        last = distances[distances >= 5]
        bucket4_pairs[i] = int(last.shape[0])
        if last.size:
            unique = np.unique(last)
            last_bucket_diversity[i] = int(unique.size)
            last_bucket_span3[i] = bool(int(unique.max()) - int(unique.min()) >= 3)
            last_bucket_multi[i] = bool(unique.size >= 2)
            last_bucket_mean[i] = float(last.mean())
            last_bucket_var[i] = float(last.var())
            prob = np.bincount(last) / last.size
            prob = prob[prob > 0]
            last_bucket_entropy[i] = float(-(prob * np.log(prob)).sum())
        has_d5_ge7[i] = bool((distances == 5).any() and (distances >= 7).any())

    finite_mean = last_bucket_mean[np.isfinite(last_bucket_mean)]
    finite_var = last_bucket_var[np.isfinite(last_bucket_var)]
    finite_entropy = last_bucket_entropy[np.isfinite(last_bucket_entropy)]

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    atom_strata = n_atoms <= 20
    result = {
        "protocol_id": PROTOCOL_ID,
        "n_molecules": n_molecules,
        "n_pairs_total": int(n_pairs.sum()),
        "distance_pairs": {str(d): int(exact_counts[d]) for d in range(1, 13)},
        "distance_pairs_11plus_in_12plus_bucket": "12 means >=12 in distance_pairs",
        "distance_molecule_presence": {str(d): int(exact_present[d]) for d in range(1, 13)},
        "bucket_pairs": {str(b): int(bucket_counts[b]) for b in range(5)},
        "bucket_fractions": {str(b): float(bucket_counts[b] / bucket_counts.sum()) for b in range(5)},
        "last_bucket_only": {
            "n_molecules_with_bucket4": int((bucket4_pairs > 0).sum()),
            "mean_distinct_distances": float(last_bucket_diversity[bucket4_pairs > 0].mean()),
            "p50_distinct_distances": float(np.percentile(last_bucket_diversity[bucket4_pairs > 0], 50)),
            "p90_distinct_distances": float(np.percentile(last_bucket_diversity[bucket4_pairs > 0], 90)),
            "max_distinct_distances": int(last_bucket_diversity.max()),
            "molecules_multi_distance": int(last_bucket_multi.sum()),
            "frac_molecules_multi_distance": float(last_bucket_multi.mean()),
            "molecules_span_at_least_3": int(last_bucket_span3.sum()),
            "frac_molecules_span_at_least_3": float(last_bucket_span3.mean()),
            "molecules_with_d5_and_ge7": int(has_d5_ge7.sum()),
            "frac_molecules_with_d5_and_ge7": float(has_d5_ge7.mean()),
            "distance_mean_of_means": float(finite_mean.mean()) if finite_mean.size else None,
            "distance_var_of_means": float(finite_var.mean()) if finite_var.size else None,
            "distance_entropy_mean": float(finite_entropy.mean()) if finite_entropy.size else None,
            "distance_entropy_p50": float(np.percentile(finite_entropy, 50)) if finite_entropy.size else None,
        },
        "diameter": {
            "mean": float(diameter.mean()),
            "p50": float(np.percentile(diameter, 50)),
            "p90": float(np.percentile(diameter, 90)),
            "max": int(diameter.max()),
            "shared_max_distance_molecules": int((diameter == diameter.max()).sum()),
        },
        "per_molecule": {
            "n_atoms_mean": float(n_atoms.mean()),
            "ring_rank_mean": float(ring_rank.mean()),
            "bucket4_fraction_mean": float((bucket4_pairs / np.maximum(n_pairs, 1)).mean()),
        },
        "correlations": {
            "bucket4_distance_diversity_vs_n_atoms": _corr(last_bucket_diversity, n_atoms),
            "bucket4_distance_entropy_vs_n_atoms": _corr(
                np.nan_to_num(last_bucket_entropy, nan=0.0), n_atoms
            ),
            "bucket4_span3_vs_n_atoms": _corr(last_bucket_span3.astype(float), n_atoms),
            "bucket4_fraction_vs_n_atoms": _corr(bucket4_pairs / np.maximum(n_pairs, 1), n_atoms),
            "bucket4_fraction_vs_ring_rank": _corr(
                bucket4_pairs / np.maximum(n_pairs, 1), ring_rank
            ),
            "bucket4_distance_mean_vs_n_atoms": _corr(
                np.nan_to_num(last_bucket_mean, nan=0.0), n_atoms
            ),
        },
        "small_molecule_stratum": {
            "n": int(atom_strata.sum()),
            "frac_multi_distance": float(last_bucket_multi[atom_strata].mean()),
            "frac_span_at_least_3": float(last_bucket_span3[atom_strata].mean()),
        },
        "bfs_crosscheck": {
            "molecules_checked": int(bfs_rows),
            "pairs_checked": int(bfs_pairs_checked),
            "pair_distance_mismatches": int(bfs_mismatch),
            "max_abs_distance_gap": float(max_abs_dist_gap),
            "max_abs_log1p_roundtrip_gap": float(max_abs_rel5_gap),
            "note": "all-pairs unweighted shortest paths on the raw PyG graphs",
        },
        "y_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    log(
        f"[dist] pairs={int(n_pairs.sum())} bucket4={bucket_counts[4]} "
        f"multi={result['last_bucket_only']['frac_molecules_multi_distance']:.4f} "
        f"d5&ge7={result['last_bucket_only']['frac_molecules_with_d5_and_ge7']:.4f} "
        f"bfs_mismatch={bfs_mismatch}"
    )
    return result


def _n_components(adjacency: np.ndarray) -> int:
    n = int(adjacency.shape[0])
    seen = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            neighbours = np.nonzero(adjacency[node])[0]
            for nxt in neighbours:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))
    return components


# ---------------------------------------------------------------------------
# stage: audit — source path / pair descriptor
# ---------------------------------------------------------------------------


def source_path_audit(model: nn.Module) -> dict[str, Any]:
    """Static + live audit of the actual Full-Y distance/pooling forward path."""
    zpp_source = Path(zpp.__file__).read_text(encoding="utf-8")
    audit_source = Path(audit.__file__).read_text(encoding="utf-8")
    p1_source = Path(p1.__file__).read_text(encoding="utf-8")
    scale_source = Path(sc.__file__).read_text(encoding="utf-8")

    checks = {
        "bucket_rule_source": "bucket = min(max(int(distance), 1), DISTANCE_BUCKETS) - 1" in zpp_source,
        "distance_one_hot_source": "distance_one_hot[bucket] = 1.0" in zpp_source,
        "log_distance_source": "np.log1p(float(distance))" in zpp_source,
        "distance_gate_is_embedding": "self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, int(pair_hidden))"
        in zpp_source
        or "distance_gate = nn.Embedding(int(p2DISTANCE_BUCKETS), p)" in scale_source,
        "relation_slice_excludes_path_bond_and_adjacent": p1.P1_RELATION_INDICES
        == tuple(range(14)) + (18,),
        "relation_slice_contains_log_distance": int(p1.P1_RELATION_INDICES[5]) == 5,
        "relation_slice_contains_path_count": int(p1.P1_RELATION_INDICES[14]) == 18,
        "pool_pair_moments_first_second_count": (
            "squared.index_add_(0, current_batch, current * current)" in audit_source
            and "torch.log1p(counts)" in audit_source
        ),
        "c6_mask_counts_zeroed": (
            tuple(cm.C6_MASK.unary_zero_blocks) == ("count",)
            and tuple(cm.C6_MASK.pair_zero_blocks) == ("count",)
            and tuple(cm.C6_MASK.relation_zero_groups) == ("path_count",)
        ),
        "c6_mask_topology_not_zeroed": "topology_zero=False" not in Path(cm.__file__).read_text(encoding="utf-8")
        and cm.C6_MASK.topology_zero is False,
        "reader_consumes_unary_pair_global_topology": (
            "torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)" in audit_source
        ),
    }

    # live shapes / values on one real molecule
    live: dict[str, Any] = {}

    def _linears(module: nn.Module) -> list[nn.Linear]:
        return [m for m in module.modules() if isinstance(m, nn.Linear)]

    pair_linears = _linears(model.pair_encoder)
    relation_linears = _linears(model.relation_encoder)
    reader_linears = _linears(model.reader)
    live["distance_gate_weight_shape"] = list(model.distance_gate.weight.shape)
    live["pair_projection_out"] = int(model.pair_projection.out_features)
    live["pair_encoder_in"] = int(pair_linears[0].in_features)
    live["pair_encoder_out"] = int(pair_linears[-1].out_features)
    live["relation_encoder_in"] = int(relation_linears[0].in_features)
    live["reader_in"] = int(reader_linears[0].in_features)
    live["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    live["pair_encoder_expected_in"] = bool(int(pair_linears[0].in_features) == 3 * dp.PAIR_VALUE_DIM + dp.PAIR_VALUE_DIM)
    live["relation_encoder_expected_in"] = bool(int(relation_linears[0].in_features) == 15)
    live["reader_expected_in"] = bool(int(reader_linears[0].in_features) == dp.READER_INPUT_DIM)
    return {
        "protocol_id": PROTOCOL_ID,
        "checks": checks,
        "all_static_checks_pass": all(bool(v) for v in checks.values()),
        "live": live,
        "source_files": {
            "zinc_patch_path_pooling": str(Path(zpp.__file__).relative_to(REPO_ROOT)),
            "e2e_dictenv_h1_clarity_audit": str(Path(audit.__file__).relative_to(REPO_ROOT)),
            "e2e_dictenv_p1": str(Path(p1.__file__).relative_to(REPO_ROOT)),
            "e2e_dictenv_scale_v1": str(Path(sc.__file__).relative_to(REPO_ROOT)),
            "distance_pooling_v1": str(Path(dp.__file__).relative_to(REPO_ROOT)),
        },
    }


def distance_descriptor_audit(
    rows: Sequence[Any], model: nn.Module, max_molecules: int, log=log
) -> dict[str, Any]:
    """Frozen Full-Y pair-state distribution by *exact* shortest-path distance.

    Forward-only; no training, no labels.  Captures ``pair_value`` with a hook
    and checks that the masked relation encoder input keeps the continuous
    ``log1p(distance)`` column while the C6-masked path-count column is zero.
    """
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handles = [
        model.pair_encoder.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("pair_value", out.detach())
        ),
        model.relation_encoder.register_forward_pre_hook(
            lambda _m, inputs: captured.__setitem__("relation_input", inputs[0].detach())
        ),
    ]
    max_distance = 12
    sums: dict[int, np.ndarray] = {}
    counts_by_distance = np.zeros(max_distance + 1, dtype=np.int64)
    sq_norms: dict[int, float] = {}
    relation_col5_ok = True
    relation_col14_zero = True
    gate_within_bucket_gap = 0.0
    n_rows = min(int(max_molecules), len(rows))
    try:
        with torch.no_grad():
            for start in range(0, n_rows, BATCH_SIZE):
                indices = list(range(start, min(start + BATCH_SIZE, n_rows)))
                batch = _batch(rows, indices, torch.zeros(len(rows)), torch.device("cpu"))
                model(batch, mask=cm.C6_MASK)
                pair_value = captured["pair_value"].to(torch.float64).numpy()
                relation_input = captured["relation_input"].to(torch.float64).numpy()
                raw5 = batch.pair_relation[:, 5].to(torch.float64).numpy()
                distances = np.rint(np.expm1(raw5)).astype(np.int64)
                relation_col5_ok &= bool(np.max(np.abs(relation_input[:, 5] - raw5)) < 1e-6)
                relation_col14_zero &= bool(np.max(np.abs(relation_input[:, 14])) == 0.0)
                bucket_np = batch.pair_bucket.numpy()
                weights = model.distance_gate.weight.detach().to(torch.float64).numpy()
                gate = 1.0 + np.tanh(weights[bucket_np])
                for bucket in range(5):
                    in_bucket = bucket_np == bucket
                    if in_bucket.any():
                        reference = gate[in_bucket][0]
                        gate_within_bucket_gap = max(
                            gate_within_bucket_gap,
                            float(np.max(np.abs(gate[in_bucket] - reference))),
                        )
                for d in np.unique(distances):
                    mask = distances == d
                    key = int(min(d, max_distance))
                    counts_by_distance[key] += int(mask.sum())
                    block = pair_value[mask]
                    sums.setdefault(key, np.zeros(block.shape[1], dtype=np.float64))
                    sums[key] += block.sum(0)
                    sq_norms[key] = sq_norms.get(key, 0.0) + float((block * block).sum())
                # distance_gate is constant within a bucket by construction
    finally:
        for handle in handles:
            handle.remove()

    means: dict[str, list[float]] = {}
    mean_norms: dict[str, float] = {}
    for key in sorted(sums):
        means[str(key)] = (sums[key] / max(counts_by_distance[key], 1)).tolist()
        mean_norms[str(key)] = float(
            math.sqrt(sq_norms[key] / max(counts_by_distance[key], 1))
        )
    # consecutive-distance representation change
    consecutive: dict[str, float] = {}
    keys = sorted(sums)
    for a, b in zip(keys, keys[1:]):
        va = np.asarray(means[str(a)])
        vb = np.asarray(means[str(b)])
        denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12
        consecutive[f"{a}->{b}"] = float(1.0 - float(va @ vb) / denom)
    within_last_bucket = {
        str(d): float(
            1.0
            - float(
                np.asarray(means[str(d)]) @ np.asarray(means[str(8)])
                / (np.linalg.norm(means[str(d)]) * np.linalg.norm(means[str(8)]) + 1e-12)
            )
        )
        for d in keys
        if d >= 5
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "n_molecules_forwarded": int(n_rows),
        "distance_counts": {str(k): int(v) for k, v in enumerate(counts_by_distance.tolist())},
        "mean_pair_value_norm_by_distance": mean_norms,
        "mean_pair_value_by_distance": means,
        "consecutive_cosine_difference": consecutive,
        "last_bucket_cosine_difference_vs_d8": within_last_bucket,
        "checks": {
            "masked_relation_input_keeps_log_distance": bool(relation_col5_ok),
            "masked_relation_input_path_count_is_zero": bool(relation_col14_zero),
            "distance_gate_constant_within_bucket": bool(gate_within_bucket_gap == 0.0),
            "distance_gate_is_5_bucket_embedding": bool(
                tuple(model.distance_gate.weight.shape) == (5, int(dp.PAIR_VALUE_DIM))
            ),
            "pair_value_width_48": True,
            "different_distances_can_change_pair_value": bool(
                any(v > 1e-6 for v in consecutive.values())
            ),
            "continuous_change_within_last_bucket": bool(
                any(v > 1e-6 for v in within_last_bucket.values())
            ),
        },
        "interpretation": (
            "the pair encoder consumes the continuous log1p(distance) coordinate and its "
            "output varies across exact distances, including inside the hard >=5 bucket; "
            "this does NOT establish any causal contribution to y"
        ),
        "y_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    log(
        "[descriptor] "
        + ", ".join(f"{k}={v}" for k, v in result["checks"].items())
    )
    return result


# ---------------------------------------------------------------------------
# stage: audit — Hard5/Soft5 pooled feature audit (frozen pair states)
# ---------------------------------------------------------------------------


def hard5_soft5_feature_audit(
    rows: Sequence[Any], model: nn.Module, log=log
) -> dict[str, Any]:
    """Compare Hard5 and Soft5 graph-level pooled moments on identical frozen
    pair states (10000 train molecules, label-blind)."""
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handle = model.pair_encoder.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("pair_value", out.detach())
    )
    n = len(rows)
    rel_diffs = np.zeros(n, dtype=np.float64)
    bucket4_fraction = np.zeros(n, dtype=np.float64)
    n_pairs = np.zeros(n, dtype=np.int64)
    hard_rows: list[np.ndarray] = []
    soft_rows: list[np.ndarray] = []
    hard_model_readout_max_gap = 0.0
    conservation = {"first": 0.0, "second": 0.0, "count": 0.0}
    weight_dev = 0.0
    try:
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                indices = list(range(start, min(start + BATCH_SIZE, n)))
                batch = _batch(rows, indices, torch.zeros(n), torch.device("cpu"))
                prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
                pv = aux["pair_value"]
                pb = aux["pair_batch"]
                logd = batch.pair_relation[:, 5].to(pv.dtype)
                n_graphs = len(indices)
                hard = dp.pool_pair_moments_hard5(
                    pv, pb, batch.pair_bucket, n_graphs, zero_count_blocks=True
                )
                soft = dp.pool_pair_moments_soft5(
                    pv, pb, logd, n_graphs, zero_count_blocks=True
                )
                hard_model_readout_max_gap = max(
                    hard_model_readout_max_gap,
                    float((hard - aux["relation_readout"]).abs().max()),
                )
                weights = dp.soft5_weights_from_log_distance(logd)
                weight_dev = max(weight_dev, float((weights.sum(1) - 1.0).abs().max()))
                # float64 reference conservation
                pv64 = pv.to(torch.float64)
                hard_full = hard.reshape(n_graphs, 5, 97).to(torch.float64)
                soft_full = soft.reshape(n_graphs, 5, 97).to(torch.float64)
                hard96 = hard_full[:, :, :96].reshape(n_graphs, 480)
                soft96 = soft_full[:, :, :96].reshape(n_graphs, 480)
                first_sum = hard_full[:, :, :48].sum(1)
                second_sum = hard_full[:, :, 48:96].sum(1)
                ref_first = torch.zeros_like(first_sum).index_add_(0, pb, pv64)
                ref_second = torch.zeros_like(ref_first).index_add_(0, pb, pv64 * pv64)
                scale_first = ref_first.abs().mean().clamp_min(1e-8)
                scale_second = ref_second.abs().mean().clamp_min(1e-8)
                conservation["first"] = max(
                    conservation["first"],
                    float(((first_sum - ref_first).abs().mean() / scale_first).item()),
                )
                conservation["second"] = max(
                    conservation["second"],
                    float(((second_sum - ref_second).abs().mean() / scale_second).item()),
                )
                # count conservation (pre-mask), soft only
                soft_counts = torch.zeros((n_graphs, 5), dtype=torch.float64)
                for k in range(5):
                    soft_counts[:, k].index_add_(0, pb, weights[:, k].to(torch.float64))
                ref_counts = torch.bincount(pb, minlength=n_graphs).to(torch.float64)
                conservation["count"] = max(
                    conservation["count"],
                    float((soft_counts.sum(1) - ref_counts).abs().max())
                    / max(1.0, float(ref_counts.max())),
                )

                hard_np = hard96.reshape(n_graphs, 480).numpy().astype(np.float64)
                soft_np = soft96.reshape(n_graphs, 480).numpy().astype(np.float64)
                hard_rows.append(hard_np)
                soft_rows.append(soft_np)
                denom = np.linalg.norm(hard_np, axis=1) + 1e-9
                rel_diffs[start : start + n_graphs] = np.linalg.norm(
                    soft_np - hard_np, axis=1
                ) / denom
                for local, row_index in enumerate(indices):
                    row = rows[row_index]
                    distances = np.rint(
                        np.expm1(row.pair_relation[:, 5].to(torch.float64).numpy())
                    ).astype(np.int64)
                    n_pairs[row_index] = distances.shape[0]
                    bucket4_fraction[row_index] = (
                        float((distances >= 5).sum()) / max(distances.shape[0], 1)
                    )
    finally:
        handle.remove()

    hard_matrix = np.concatenate(hard_rows, axis=0)  # [10000, 480]
    soft_matrix = np.concatenate(soft_rows, axis=0)

    def _effective_rank(matrix: np.ndarray) -> dict[str, float]:
        centered = matrix - matrix.mean(0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        energy = singular ** 2
        total = float(energy.sum())
        participation = float(total ** 2 / float((energy ** 2).sum())) if total > 0 else 0.0
        threshold = 1e-6 * float(singular.max()) if singular.size else 0.0
        return {
            "participation_ratio": participation,
            "n_singular_above_1e-6_max": int((singular > threshold).sum()),
            "top1_energy_share": float(energy[0] / total) if total > 0 else 0.0,
        }

    frac_diff_gt_1pct = float((rel_diffs > FEATURE_DIFF_THRESHOLD).mean())
    correlations = float(np.corrcoef(hard_matrix.ravel(), soft_matrix.ravel())[0, 1])
    bucket4_strata = bucket4_fraction >= np.median(bucket4_fraction)
    result = {
        "protocol_id": PROTOCOL_ID,
        "n_molecules": n,
        "feature_layout": {"hard5": [n, 485], "soft5": [n, 485], "compared_slices": [n, 480]},
        "hard_model_readout_max_abs_gap": hard_model_readout_max_gap,
        "conservation_relative": conservation,
        "soft5_weight_row_sum_max_dev": weight_dev,
        "relative_feature_difference": {
            "mean": float(rel_diffs.mean()),
            "p50": float(np.percentile(rel_diffs, 50)),
            "p90": float(np.percentile(rel_diffs, 90)),
            "max": float(rel_diffs.max()),
            "frac_gt_1pct": frac_diff_gt_1pct,
            "threshold": FEATURE_DIFF_THRESHOLD,
            "n_gt_1pct": int((rel_diffs > FEATURE_DIFF_THRESHOLD).sum()),
        },
        "representations": {
            "hard5": {
                "channel_std_mean": float(hard_matrix.std(0).mean()),
                "channel_std_max": float(hard_matrix.std(0).max()),
                "finite": bool(np.isfinite(hard_matrix).all()),
                **_effective_rank(hard_matrix),
            },
            "soft5": {
                "channel_std_mean": float(soft_matrix.std(0).mean()),
                "channel_std_max": float(soft_matrix.std(0).max()),
                "finite": bool(np.isfinite(soft_matrix).all()),
                **_effective_rank(soft_matrix),
            },
            "flattened_correlation": correlations,
            "mutual_information_exactness": "not applicable; deterministic function of hard inputs",
        },
        "difference_concentration": {
            "corr_rel_diff_vs_bucket4_fraction": float(
                np.corrcoef(rel_diffs, bucket4_fraction)[0, 1]
            ),
            "mean_rel_diff_high_bucket4": float(rel_diffs[bucket4_strata].mean()),
            "mean_rel_diff_low_bucket4": float(rel_diffs[~bucket4_strata].mean()),
            "mean_rel_diff_molecules_with_bucket4_pairs": float(
                rel_diffs[bucket4_fraction > 0].mean()
            ),
            "mean_rel_diff_molecules_without_bucket4_pairs": float(
                rel_diffs[bucket4_fraction == 0].mean()
            ),
        },
        "interpretation": (
            "Soft5 reorganises the same frozen pair states through a different fixed "
            "weighting; this is a representation-difference fact, not predictive evidence"
        ),
        "y_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    np.savez_compressed(
        OUT / "feature_audit_arrays.npz",
        rel_diff=rel_diffs.astype(np.float32),
        bucket4_fraction=bucket4_fraction.astype(np.float32),
        n_pairs=n_pairs.astype(np.int64),
        hard_matrix=hard_matrix.astype(np.float32),
        soft_matrix=soft_matrix.astype(np.float32),
    )
    log(
        f"[features] rel_diff mean={result['relative_feature_difference']['mean']:.4f} "
        f"frac>1%={frac_diff_gt_1pct:.4f} conservation={conservation}"
    )
    return result


# ---------------------------------------------------------------------------
# stage: audit — runtime correctness gates
# ---------------------------------------------------------------------------


def correctness_runtime_gates(
    rows: Sequence[Any], prep: Mapping[str, np.ndarray], log=log
) -> dict[str, Any]:
    """Numerical gates on real data for both pooling modes."""
    device = torch.device("cpu")
    gates: dict[str, Any] = {}

    def _build(mode: str) -> nn.Module:
        return dp.build_pooling_model(prep, 0, mode=mode)

    hard = _build("hard5")
    soft = _build("soft5")
    parent = sc.build_scale_model(
        np.asarray(prep["D_fit"], np.float32),
        0,
        zjd.fold_subspace(prep),
        sc.FULL,
        scale_seed=0,
    )
    hard_state = hard.state_dict()
    soft_state = soft.state_dict()
    parent_state = parent.state_dict()
    gates["init_hard_equal_soft"] = bool(
        all(torch.equal(hard_state[k], soft_state[k]) for k in hard_state)
    )
    gates["init_hard_equal_parent"] = bool(
        all(torch.equal(hard_state[k], parent_state[k]) for k in hard_state)
    )
    gates["parameter_names_equal"] = set(hard_state) == set(soft_state)
    gates["parameter_count_408651"] = bool(
        sum(p.numel() for p in hard.parameters()) == FULL_PARAMETERS
        and sum(p.numel() for p in soft.parameters()) == FULL_PARAMETERS
    )
    gates["no_new_trainable_parameters"] = bool(
        sum(p.numel() for p in hard.parameters()) == sum(p.numel() for p in parent.parameters())
        and {k: tuple(v.shape) for k, v in hard_state.items()}
        == {k: tuple(v.shape) for k, v in parent_state.items()}
    )

    batch = _batch(rows, list(range(64)), torch.zeros(len(rows)), device)
    hard.eval()
    soft.eval()
    parent.eval()
    with torch.no_grad():
        hard_pred, hard_aux = hard(batch, mask=cm.C6_MASK, return_aux=True)
        soft_pred, soft_aux = soft(batch, mask=cm.C6_MASK, return_aux=True)
        parent_pred, parent_aux = parent(batch, mask=cm.C6_MASK, return_aux=True)
        batch_y_random = batch.clone()
        batch_y_random.y = torch.randn_like(batch.y)
        soft_pred_b = soft(batch_y_random, mask=cm.C6_MASK)
    gates["hard5_prediction_matches_parent_bitwise"] = bool(
        torch.equal(hard_pred, parent_pred.view(-1))
    )
    gates["hard5_readout_matches_parent_bitwise"] = bool(
        torch.equal(hard_aux["relation_readout"], parent_aux["relation_readout"])
    )
    gates["soft5_forward_changes_prediction"] = bool(
        not torch.equal(hard_pred, soft_pred)
    )
    gates["reader_input_width_814"] = bool(
        int(hard_aux["relation_readout"].shape[1]) == dp.RELATION_READOUT_DIM
    )
    count_blocks_zero = True
    for aux in (hard_aux, soft_aux):
        readout = aux["relation_readout"]
        for k in range(5):
            count_blocks_zero &= bool(float(readout[:, k * 97 + 96 : k * 97 + 97].abs().max()) == 0.0)
    gates["c6_pair_count_blocks_zero"] = bool(count_blocks_zero)
    gates["y_independent_forward"] = bool(torch.equal(soft_pred, soft_pred_b))

    # soft weight properties
    weights = dp.soft5_weights_from_log_distance(batch.pair_relation[:, 5])
    gates["soft5_weights_finite"] = bool(torch.isfinite(weights).all())
    gates["soft5_weights_nonnegative"] = bool((weights >= 0).all())
    gates["soft5_weights_sum_to_one"] = bool(
        float((weights.sum(1) - 1.0).abs().max()) <= WEIGHT_SUM_ATOL
    )

    # conservation against a float64 reference
    pv = soft_aux["pair_value"].to(torch.float64)
    pb = soft_aux["pair_batch"]
    n_graphs = 64
    ref_first = torch.zeros((n_graphs, 48), dtype=torch.float64).index_add_(0, pb, pv)
    ref_second = torch.zeros_like(ref_first).index_add_(0, pb, pv * pv)
    soft_view = soft_aux["relation_readout"].to(torch.float64).reshape(n_graphs, 5, 97)
    first_sum = soft_view[:, :, :48].sum(1)
    second_sum = soft_view[:, :, 48:96].sum(1)
    gates["conservation_first_moment"] = bool(
        float((first_sum - ref_first).abs().mean() / ref_first.abs().mean().clamp_min(1e-8))
        <= CONSERVATION_RTOL
    )
    gates["conservation_second_moment"] = bool(
        float((second_sum - ref_second).abs().mean() / ref_second.abs().mean().clamp_min(1e-8))
        <= CONSERVATION_RTOL
    )
    soft_counts = torch.zeros((n_graphs, 5), dtype=torch.float64)
    for k in range(5):
        soft_counts[:, k].index_add_(0, pb, weights[:, k].to(torch.float64))
    ref_counts = torch.bincount(pb, minlength=n_graphs).to(torch.float64)
    gates["each_pair_weighted_once_count_conservation"] = bool(
        float((soft_counts.sum(1) - ref_counts).abs().max()) <= 1e-3
    )

    # gradient flow under soft5
    soft.train()
    optimizer = torch.optim.Adam(soft.parameters(), lr=LR)
    prediction, aux = soft(batch, mask=cm.C6_MASK, return_aux=True)
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + H1_LAMBDA * (
        soft.reconstruction_loss(aux["phi"], aux["coord"])
    )
    optimizer.zero_grad()
    loss.backward()
    grads = {
        "pair_encoder": float(
            sum(p.grad.abs().sum() for p in soft.pair_encoder.parameters() if p.grad is not None)
        ),
        "relation_encoder": float(
            sum(p.grad.abs().sum() for p in soft.relation_encoder.parameters() if p.grad is not None)
        ),
        "distance_gate": float(soft.distance_gate.weight.grad.abs().sum()),
    }
    gates["soft5_gradients_reach_pair_stack"] = bool(all(v > 0 for v in grads.values()))
    gates["soft5_loss_finite"] = bool(math.isfinite(float(loss)))

    # pair-order invariance (same physical pairs, reversed absorption order)
    order = torch.arange(weights.shape[0] - 1, -1, -1, dtype=torch.long)
    soft_perm = dp.pool_pair_moments_weighted(
        soft_aux["pair_value"][order],
        pb[order],
        weights[order],
        n_graphs,
        zero_count_blocks=True,
    )
    gates["pair_order_invariance_tolerance_1e3"] = bool(
        float((soft_perm - soft_aux["relation_readout"]).abs().max()) <= 1e-3
    )

    # endpoint swap invariance (pair encoder input is symmetric); eval() so
    # dropout cannot masquerade as an invariance violation
    soft.eval()
    swapped = batch.clone()
    swapped.pair_index = torch.stack([batch.pair_index[1], batch.pair_index[0]], dim=0)
    with torch.no_grad():
        _p, swapped_aux = soft(swapped, mask=cm.C6_MASK, return_aux=True)
    gates["endpoint_swap_pair_value_invariant"] = bool(
        float((swapped_aux["pair_value"] - soft_aux["pair_value"]).abs().max()) <= 1e-6
    )

    # batch composition: same molecule alone vs inside the batch
    single = _batch(rows, [0], torch.zeros(len(rows)), device)
    with torch.no_grad():
        _p1, single_aux = soft(single, mask=cm.C6_MASK, return_aux=True)
    gates["single_molecule_pooling_matches_batch"] = bool(
        float((single_aux["relation_readout"][0] - soft_aux["relation_readout"][0]).abs().max())
        <= 1e-4
    )

    gates["_grad_values"] = grads
    gates["all_pass"] = bool(
        all(bool(v) for k, v in gates.items() if not k.startswith("_"))
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "gates": gates,
        "all_correctness_gates_pass": gates["all_pass"],
        "y_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# stage: audit — purchase gate
# ---------------------------------------------------------------------------


def purchase_gate(
    distribution: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> dict[str, Any]:
    presence = float(distribution["last_bucket_only"]["frac_molecules_with_d5_and_ge7"])
    frac_diff = float(feature_audit["relative_feature_difference"]["frac_gt_1pct"])
    gates = {
        "G1_long_distance_heterogeneity": {
            "measured": presence,
            "threshold": DISTANCE_PRESENCE_THRESHOLD,
            "pass": bool(presence >= DISTANCE_PRESENCE_THRESHOLD),
            "definition": "fraction of train molecules holding both d=5 and d>=7 pairs in bucket 4",
        },
        "G2_pooled_feature_difference": {
            "measured": frac_diff,
            "threshold": FEATURE_DIFF_FRACTION,
            "pass": bool(frac_diff >= FEATURE_DIFF_FRACTION),
            "definition": "fraction of train molecules with >1% normalised Hard5/Soft5 difference",
        },
        "G3_correctness": {
            "measured": bool(correctness["all_correctness_gates_pass"]),
            "threshold": True,
            "pass": bool(correctness["all_correctness_gates_pass"]),
            "definition": "all conservation / dimension / gradient / matching gates",
        },
    }
    purchase = bool(all(item["pass"] for item in gates.values()))
    return {
        "protocol_id": PROTOCOL_ID,
        "gates": gates,
        "purchase": purchase,
        "label": "CPU_PURCHASE_GRANTED" if purchase else "CPU_NO_PURCHASE",
        "note": (
            "these gates only show the operator is not a near-identity reparameterisation "
            "acting on molecules with genuine long-distance heterogeneity; they are not "
            "predictive evidence"
        ),
        "y_used_for_gate": False,
    }


# ---------------------------------------------------------------------------
# stage: train
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    *,
    device: torch.device,
    epochs: int = EPOCHS,
    smoke: bool = False,
    log_every: int = 20,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    mode = ARM_MODES[arm]
    rows = load_train_rows()
    prep = load_prep()
    apply_prep(rows, prep)
    y_all = train_y(rows)
    fit_rows = np.arange(len(rows), dtype=np.int64)

    seed_everything(0)
    model = dp.build_pooling_model(prep, 0, mode=mode).to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_hash = state_hash(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(0 + TRAIN_SHUFFLE_OFFSET)
    y_fit = torch.as_tensor(y_all, dtype=torch.float32)

    soup_epochs = set(range(max(1, epochs - SOUP_LAST + 1), epochs + 1))
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    rng_hashes: dict[str, dict[str, str]] = {}
    epoch_seconds: list[float] = []
    steps_done = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps = 0.0, 0.0, 0, 0
        for indices in _batches(len(rows), BATCH_SIZE, train_gen, True):
            batch = _batch(rows, indices, y_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + H1_LAMBDA * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rng_hashes[str(epoch)] = rng_hash()
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "train_rec": float(rec_sum / max(n_steps, 1)),
                "seconds": epoch_seconds[-1],
            }
        )
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            log(
                f"[{arm}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"rec={curve[-1]['train_rec']:.4g} {epoch_seconds[-1]:.1f}s"
            )
        if smoke and epoch >= 3:
            break

    members = sorted(soup)
    expected_members = list(range(max(1, epochs - SOUP_LAST + 1), epochs + 1))
    if not smoke and members != expected_members:
        raise RuntimeError(f"soup members {members} != last-{SOUP_LAST} epochs {expected_members}")
    soup_state = (
        {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
        if members
        else {}
    )
    meta: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "pair_pooling": mode,
        "git_commit": git_commit(),
        "seed": 0,
        "scale_seed": SCALE_SEED,
        "smoke": bool(smoke),
        "epochs": int(epochs),
        "epochs_run": int(len(curve)),
        "steps_done": int(steps_done),
        "batches_per_epoch": int(math.ceil(len(rows) / BATCH_SIZE)),
        "n_train_rows": int(len(rows)),
        "fit_rows_sha256": sha256_rows(fit_rows),
        "soup_members": members,
        "soup_rule": "mean(state_dict at epochs 236..240)",
        "hyper": {
            "optimizer": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "h1_lambda": H1_LAMBDA,
            "train_shuffle_seed": TRAIN_SHUFFLE_OFFSET,
            "scheduler": None,
            "amp": False,
            "ddp": False,
            "precision": "fp32",
            "task_supervision": "raw y L1 only",
            "auxiliary": "label-free structural reconstruction (disclosed)",
        },
        "init_state_sha256": init_hash,
        "final_state_sha256": state_hash({k: v.detach().cpu() for k, v in model.state_dict().items()}),
        "soup_state_sha256": state_hash(soup_state) if soup_state else None,
        "prep_path": str(HIST_PREP.relative_to(REPO_ROOT)),
        "prep_sha256": file_sha256(HIST_PREP),
        "encoded_train_sha256": file_sha256(ENCODED_TRAIN),
        "curve": curve,
        "rng_hash_last": rng_hashes[str(len(curve))] if rng_hashes else {},
        "rng_hash_epoch_1": rng_hashes.get("1", {}),
        "rng_hash_epoch_40": rng_hashes.get("40", {}),
        "rng_hash_epoch_120": rng_hashes.get("120", {}),
        "rng_hash_epoch_240": rng_hashes.get(str(epochs), {}),
        "wall_seconds": float(time.perf_counter() - started),
        "seconds_per_epoch_mean": float(np.mean(epoch_seconds)) if epoch_seconds else float("nan"),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if smoke:
        write_json(OUT / f"{arm}_smoke_meta.json", meta)
        log(f"[{arm}] smoke done")
        return meta

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(soup_state, OUT / f"{arm}_last5_soup_state.pt")
    meta["soup_state_file"] = str((OUT / f"{arm}_last5_soup_state.pt").relative_to(REPO_ROOT))
    meta["soup_state_file_sha256"] = file_sha256(OUT / f"{arm}_last5_soup_state.pt")

    # post-training internal diagnostics (never used for selection)
    model.load_state_dict({k: v.to(device) for k, v in soup_state.items()}, strict=True)
    model.eval()
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    internal = {}
    for name, subset in (
        ("fit_inner", np.asarray(idx["fit_inner"], dtype=np.int64)),
        ("monitor", np.asarray(idx["monitor"], dtype=np.int64)),
        ("dev", np.asarray(idx["dev"], dtype=np.int64)),
        ("fit_all", np.arange(len(rows), dtype=np.int64)),
    ):
        pred = predict_raw(model, [rows[i] for i in subset], device)
        internal[f"{name}_mae"] = mae(pred, y_all[subset])
    meta["auxiliary_internal_mae"] = internal
    write_json(OUT / f"{arm}_train_meta.json", meta)
    log(
        f"[{arm}] soup written; fit_all={internal['fit_all_mae']:.6f} "
        f"monitor={internal['monitor_mae']:.6f} dev={internal['dev_mae']:.6f}"
    )
    return meta


# ---------------------------------------------------------------------------
# stage: check (cross-arm matching)
# ---------------------------------------------------------------------------


def arm_matching_audit() -> dict[str, Any]:
    metas = {arm: read_json(OUT / f"{arm}_train_meta.json") for arm in ARM_NAMES}
    checks: dict[str, Any] = {}
    checks["same_git_commit"] = len({m["git_commit"] for m in metas.values()}) == 1
    checks["same_n_train_rows"] = len({m["n_train_rows"] for m in metas.values()}) == 1
    checks["same_fit_rows_sha256"] = len({m["fit_rows_sha256"] for m in metas.values()}) == 1
    checks["same_prep_sha256"] = len({m["prep_sha256"] for m in metas.values()}) == 1
    checks["same_encoded_train_sha256"] = len({m["encoded_train_sha256"] for m in metas.values()}) == 1
    checks["same_init_state_sha256"] = len({m["init_state_sha256"] for m in metas.values()}) == 1
    checks["same_soup_members"] = len({tuple(m["soup_members"]) for m in metas.values()}) == 1
    checks["same_steps_done"] = len({m["steps_done"] for m in metas.values()}) == 1
    checks["steps_equal_18960"] = all(m["steps_done"] == EXPECTED_STEPS for m in metas.values())
    checks["parameters_408651_both"] = all(m["n_params"] == FULL_PARAMETERS for m in metas.values())
    checks["same_rng_epoch1"] = len({str(m["rng_hash_epoch_1"]) for m in metas.values()}) == 1
    checks["same_rng_epoch40"] = len({str(m["rng_hash_epoch_40"]) for m in metas.values()}) == 1
    checks["same_rng_epoch120"] = len({str(m["rng_hash_epoch_120"]) for m in metas.values()}) == 1
    checks["same_rng_epoch240"] = len({str(m["rng_hash_epoch_240"]) for m in metas.values()}) == 1
    result = {
        "protocol_id": PROTOCOL_ID,
        "checks": checks,
        "all_match": bool(all(bool(v) for v in checks.values())),
        "arms": {arm: {"pair_pooling": m["pair_pooling"], "init_state_sha256": m["init_state_sha256"],
                       "soup_state_sha256": m["soup_state_sha256"], "wall_seconds": m["wall_seconds"]}
                 for arm, m in metas.items()},
        "note": (
            "RNG state hashes are identical because both arms execute the identical op "
            "sequence (only the pooling arithmetic differs); this is evidence of matched "
            "dropout masks and batch streams, not of bitwise training reproducibility (the "
            "GPU reduction order remains nondeterministic, documented in the report)"
        ),
    }
    write_json(OUT / "matching_audit.json", result)
    log(f"[check] all_match={result['all_match']} {checks}")
    return result


# ---------------------------------------------------------------------------
# stage: predict (freeze + official-valid predictions, labels unread)
# ---------------------------------------------------------------------------


def _load_valid_with_alignment() -> tuple[list[Any], dict[str, Any]]:
    """Encoded official-valid rows + a fresh atom-count/type alignment check.

    The ZINC ``val`` PyG dataset is instantiated for the raw graph fields only;
    its labels are never read in this stage (pinned by the fast test that the
    predict stage source contains no ``.y`` access).
    """
    from torch_geometric.datasets import ZINC

    rows = load_valid_rows()
    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
    if not (len(rows) == len(ds) == 1000):
        raise RuntimeError(f"valid length mismatch: {len(rows)} vs {len(ds)}")
    atom_mismatch = type_mismatch = 0
    for i in range(1000):
        if int(rows[i].dict_phi.shape[0]) != int(ds[i].x.shape[0]):
            atom_mismatch += 1
        elif not torch.equal(rows[i].dict_atom.long().reshape(-1), ds[i].x.long().reshape(-1)):
            type_mismatch += 1
    if atom_mismatch or type_mismatch:
        raise RuntimeError(f"valid alignment failed: atom={atom_mismatch} type={type_mismatch}")
    return rows, {"atom_count_mismatches": atom_mismatch, "atom_type_mismatches": type_mismatch}


def stage_predict(device: torch.device) -> dict[str, Any]:
    prep = load_prep()
    rows, alignment = _load_valid_with_alignment()
    apply_prep(rows, prep)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    checkpoint_info: dict[str, Any] = {}
    for arm in ARM_NAMES:
        mode = ARM_MODES[arm]
        state = torch.load(OUT / f"{arm}_last5_soup_state.pt", map_location=device, weights_only=True)
        model = dp.build_pooling_model(prep, 0, mode=mode).to(device)
        model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
        model.eval()
        pred = predict_raw(model, rows, device)
        predictions[arm] = pred
        path = OUT / f"{arm}_valid_preds.npz"
        np.savez_compressed(path, pred=pred.astype(np.float32), row=np.arange(1000, dtype=np.int64))
        checkpoint_info[arm] = {
            "checkpoint_path": str((OUT / f"{arm}_last5_soup_state.pt").relative_to(REPO_ROOT)),
            "checkpoint_file_sha256": file_sha256(OUT / f"{arm}_last5_soup_state.pt"),
            "checkpoint_state_sha256": state_hash(state),
            "prediction_path": str(path.relative_to(REPO_ROOT)),
            "prediction_file_sha256": file_sha256(path),
            "pair_pooling": mode,
        }
    combined = OUT / "valid_predictions.npz"
    np.savez_compressed(
        combined,
        pred_Hard5=predictions["Hard5"].astype(np.float32),
        pred_Soft5=predictions["Soft5"].astype(np.float32),
        row=np.arange(1000, dtype=np.int64),
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_molecules": 1000,
        "device": str(device),
        "valid_alignment": alignment,
        "checkpoints": checkpoint_info,
        "prep_path": str(HIST_PREP.relative_to(REPO_ROOT)),
        "prep_sha256": file_sha256(HIST_PREP),
        "encoded_valid_sha256": file_sha256(ENCODED_VALID),
        "combined_predictions": {
            "path": str(combined.relative_to(REPO_ROOT)),
            "sha256": file_sha256(combined),
        },
        "extraction_program_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "y_in_file": False,
        "official_valid_y_read_during_predict": False,
        "official_test_loaded": False,
        "row_alignment_reference": {
            "method": "encoded-valid row i atom count/type vs official ZINC val row i (no labels)",
            "note": "same alignment audit as the frozen budget-control round",
        },
    }
    write_json(OUT / "freeze_manifest.json", manifest)
    log("[predict] frozen predictions written; y never read")
    return manifest


# ---------------------------------------------------------------------------
# stage: score
# ---------------------------------------------------------------------------


def bootstrap_paired(diff: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    size = diff.shape[0]
    draws = np.empty(int(n), dtype=np.float64)
    for b in range(int(n)):
        sample = rng.integers(0, size, size)
        draws[b] = diff[sample].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "statistic": "paired signed per-molecule diff (|e_Soft5| - |e_Hard5|)",
        "mean_diff": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "n_resamples": int(n),
        "seed": int(seed),
        "frac_soft5_better": float((diff < 0).mean()),
        "n_soft5_better": int((diff < 0).sum()),
        "n_hard5_better": int((diff > 0).sum()),
        "n_tied": int((diff == 0).sum()),
    }


def _load_official_valid_y() -> np.ndarray:
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
    if len(ds) != 1000:
        raise RuntimeError(f"official valid length {len(ds)} != 1000")
    return np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(1000)], dtype=np.float64)


def stage_score() -> dict[str, Any]:
    manifest = read_json(OUT / "freeze_manifest.json")
    # re-verify frozen artifacts before touching y
    for arm in ARM_NAMES:
        info = manifest["checkpoints"][arm]
        if file_sha256(OUT / f"{arm}_valid_preds.npz") != info["prediction_file_sha256"]:
            raise RuntimeError(f"{arm}: prediction file changed after freeze")
        if file_sha256(OUT / f"{arm}_last5_soup_state.pt") != info["checkpoint_file_sha256"]:
            raise RuntimeError(f"{arm}: checkpoint changed after freeze")
    if file_sha256(Path(__file__)) != manifest["extraction_program_sha256"]:
        raise RuntimeError("extraction program changed after freeze")

    preds = np.load(OUT / "valid_predictions.npz")
    pred_hard = preds["pred_Hard5"].astype(np.float64)
    pred_soft = preds["pred_Soft5"].astype(np.float64)
    y = _load_official_valid_y()  # single label read

    errors = {"Hard5": np.abs(pred_hard - y), "Soft5": np.abs(pred_soft - y)}
    metrics: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "n_molecules": 1000,
        "mae": {arm: float(errors[arm].mean()) for arm in ARM_NAMES},
        "rmse": {arm: float(np.sqrt(((errors[arm]) ** 2).mean())) for arm in ARM_NAMES},
        "median_abs": {arm: float(np.median(errors[arm])) for arm in ARM_NAMES},
        "p90_abs": {arm: float(np.percentile(errors[arm], 90)) for arm in ARM_NAMES},
        "p95_abs": {arm: float(np.percentile(errors[arm], 95)) for arm in ARM_NAMES},
        "max_abs": {arm: float(errors[arm].max()) for arm in ARM_NAMES},
        "historical_Hard5_reference": 0.118618,
        "label_reads": 1,
    }
    diff = errors["Soft5"] - errors["Hard5"]
    delta = float(diff.mean())
    metrics["delta_soft5_minus_hard5"] = delta
    metrics["gap_to_target"] = {str(level): float(metrics["mae"]["Soft5"] - level) for level in TARGET_LEVELS}
    paired = bootstrap_paired(diff)

    # subgroups (all pre-declared)
    valid_rows = load_valid_rows()
    subgroup_payload: dict[str, Any] = {}
    subgroup_payload["y_bands"] = _subgroup_report(
        y, errors, np.where(y <= -3.0)[0], np.where(y > -3.0)[0], "y<=-3", "y>-3"
    )
    atom_counts = np.asarray([int(valid_rows[i].dict_phi.shape[0]) for i in range(1000)])
    ring_rank = np.asarray([_valid_ring_rank(valid_rows[i]) for i in range(1000)])
    diameter = np.asarray([_valid_diameter(valid_rows[i]) for i in range(1000)])
    bucket4_fraction = np.asarray([_valid_bucket4_fraction(valid_rows[i]) for i in range(1000)])
    bucket4_diversity = np.asarray([_valid_bucket_diversity(valid_rows[i]) for i in range(1000)])
    subgroup_payload["atoms"] = _stratified_report(
        atom_counts, errors, qs=(0.0, 0.25, 0.5, 0.75, 1.0), label="n_atoms"
    )
    subgroup_payload["ring_rank"] = _stratified_report(
        ring_rank, errors, qs=(0.0, 0.25, 0.5, 0.75, 1.0), label="ring_rank"
    )
    subgroup_payload["diameter"] = _stratified_report(
        diameter, errors, qs=(0.0, 0.25, 0.5, 0.75, 1.0), label="graph_diameter"
    )
    subgroup_payload["bucket4_fraction"] = _stratified_report(
        bucket4_fraction, errors, qs=(0.0, 0.25, 0.5, 0.75, 1.0), label="bucket4_pair_fraction"
    )
    subgroup_payload["bucket4_diversity"] = _stratified_report(
        bucket4_diversity, errors, qs=(0.0, 0.25, 0.5, 0.75, 1.0), label="last_bucket_distinct_distances"
    )
    extreme = {
        "valid_index": int(EXTREME_VALID_INDEX),
        "hard5_abs_error": float(errors["Hard5"][EXTREME_VALID_INDEX]),
        "soft5_abs_error": float(errors["Soft5"][EXTREME_VALID_INDEX]),
        "delta_without_row": float((errors["Soft5"] - errors["Hard5"])[np.arange(1000) != EXTREME_VALID_INDEX].mean()),
        "hard5_error_mass_share": float(errors["Hard5"][EXTREME_VALID_INDEX] / errors["Hard5"].sum()),
        "delta": delta,
        "delta_without_row_keeps_magnitude": float(
            abs((errors["Soft5"] - errors["Hard5"])[np.arange(1000) != EXTREME_VALID_INDEX].mean())
            / max(abs(delta), 1e-12)
        ),
        "top10_hard5_share": float(np.sort(errors["Hard5"])[-10:].sum() / errors["Hard5"].sum()),
        "top10_soft5_share": float(np.sort(errors["Soft5"])[-10:].sum() / errors["Soft5"].sum()),
    }
    y_minus3 = y > -3.0
    diff_ordinary = diff[y_minus3]
    ordinary = bootstrap_paired(diff_ordinary)
    subgroup_payload["ordinary_molecules_y_gt_minus3"] = ordinary
    subgroup_payload["extreme_row"] = extreme
    subgroup_payload["subgroup_definitions_frozen"] = [
        "y<=-3 vs y>-3",
        "n_atoms quartiles",
        "cyclomatic ring rank quartiles",
        "graph diameter quartiles",
        "bucket-4 pair fraction quartiles",
        "last-bucket distinct-distance quartiles",
        "valid:0172 excluded (descriptive only)",
    ]

    # decision (pre-frozen)
    ci_excludes_zero = paired["ci95"][1] < 0.0
    band = decide_case(delta)
    ordinary_ci_excludes_zero = ordinary["ci95"][1] < 0.0
    not_extreme = extreme["delta_without_row_keeps_magnitude"] >= EXTREME_KEEP_FRACTION
    engineering_ok = bool(
        read_json(OUT / "correctness_gates.json")["all_correctness_gates_pass"]
        and read_json(OUT / "matching_audit.json")["all_match"]
    )
    if band == "A":
        if ci_excludes_zero and ordinary_ci_excludes_zero and not_extreme and engineering_ok:
            decision = "SOFT5_PREDICTIVE_SIGNAL"
        elif not ci_excludes_zero:
            decision = "UNCERTAIN"
        else:
            decision = "SOFT5_PREDICTIVE_SIGNAL_UNCERTAIN"
    elif band == "B":
        decision = "SOFT5_LIMITED_SIGNAL"
    else:
        decision = "SOFT5_NO_USEFUL_GAIN"

    decision_payload = {
        "protocol_id": PROTOCOL_ID,
        "delta": delta,
        "band": band,
        "ci95": paired["ci95"],
        "ci_excludes_zero": ci_excludes_zero,
        "ordinary_y_gt_minus3_delta": float(diff_ordinary.mean()),
        "ordinary_y_gt_minus3_ci95": ordinary["ci95"],
        "ordinary_ci_excludes_zero": ordinary_ci_excludes_zero,
        "extreme_row_check": not_extreme,
        "extreme_keep_fraction_threshold": EXTREME_KEEP_FRACTION,
        "engineering_gates_pass": engineering_ok,
        "decision": decision,
        "absolute_gap": {str(level): float(metrics["mae"]["Soft5"] - level) for level in TARGET_LEVELS},
        "thresholds": {
            "case_A": f"delta <= {DECISION_DELTA_A} and CI<0 and y>-3 subgroup improves and not extreme-driven",
            "case_B": f"{DECISION_DELTA_A} < delta <= {DECISION_DELTA_B}",
            "case_C": f"delta > {DECISION_DELTA_B}",
            "ci_conflict_label": "UNCERTAIN",
        },
        "scoring_provenance": {
            "label_reads": 1,
            "official_test_loaded": False,
            "note": "official validation has been exposed in earlier rounds; this is a controlled comparison, not a fresh blind test",
        },
    }

    write_json(OUT / "valid_metrics.json", metrics)
    write_json(OUT / "paired_bootstrap.json", paired)
    write_json(OUT / "subgroup_analysis.json", subgroup_payload)
    write_json(OUT / "decision.json", decision_payload)
    log(
        f"[score] Hard5={metrics['mae']['Hard5']:.6f} Soft5={metrics['mae']['Soft5']:.6f} "
        f"delta={delta:+.6f} CI={paired['ci95']} decision={decision}"
    )
    return decision_payload


def _subgroup_report(
    y: np.ndarray, errors: Mapping[str, np.ndarray], mask_a: np.ndarray, mask_b: np.ndarray, name_a: str, name_b: str
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in ((name_a, mask_a), (name_b, mask_b)):
        diff = errors["Soft5"][mask] - errors["Hard5"][mask]
        out[name] = {
            "n": int(mask.sum()),
            "mae_hard5": float(errors["Hard5"][mask].mean()),
            "mae_soft5": float(errors["Soft5"][mask].mean()),
            "delta": float(diff.mean()),
            "ci95": bootstrap_paired(diff)["ci95"],
        }
    return out


def _stratified_report(
    values: np.ndarray, errors: Mapping[str, np.ndarray], qs: Sequence[float], label: str
) -> dict[str, Any]:
    edges = np.unique(np.percentile(values, [q * 100.0 for q in qs]))
    if edges.size < 2:
        return {"label": label, "strata": []}
    strata = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        diff = errors["Soft5"][mask] - errors["Hard5"][mask]
        strata.append(
            {
                "range": [float(lo), float(hi)],
                "n": int(mask.sum()),
                "mae_hard5": float(errors["Hard5"][mask].mean()) if mask.any() else None,
                "mae_soft5": float(errors["Soft5"][mask].mean()) if mask.any() else None,
                "delta": float(diff.mean()) if mask.any() else None,
            }
        )
    return {"label": label, "strata": strata}


def _valid_ring_rank(row: Any) -> int:
    # cyclomatic number from the cached pair distances: bucket 0 pairs are bonds
    n = int(row.dict_phi.shape[0])
    bonds = int((row.pair_bucket == 0).sum())
    # a connected ZINC molecule has one component
    return int(bonds - n + 1)


def _valid_diameter(row: Any) -> int:
    distances = np.rint(np.expm1(row.pair_relation[:, 5].to(torch.float64).numpy())).astype(np.int64)
    return int(distances.max()) if distances.size else 0


def _valid_bucket4_fraction(row: Any) -> float:
    distances = np.rint(np.expm1(row.pair_relation[:, 5].to(torch.float64).numpy())).astype(np.int64)
    return float((distances >= 5).mean()) if distances.size else 0.0


def _valid_bucket_diversity(row: Any) -> int:
    distances = np.rint(np.expm1(row.pair_relation[:, 5].to(torch.float64).numpy())).astype(np.int64)
    last = distances[distances >= 5]
    return int(np.unique(last).size) if last.size else 0


# ---------------------------------------------------------------------------
# stage: audit orchestration
# ---------------------------------------------------------------------------


def stage_audit(max_descriptor_molecules: int = 10000, max_bfs: int | None = None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = load_train_rows()
    prep = load_prep()
    apply_prep(rows, prep)
    dataset = cf.load_official_train()
    alignment = rpv2.verify_row_alignment(rows, log=log)

    frozen_model = dp.build_pooling_model(prep, 0, mode="hard5")
    state = torch.load(HIST_Y_SOUP, map_location="cpu", weights_only=True)
    frozen_model.load_state_dict({k: v.to("cpu") for k, v in state.items()}, strict=True)
    frozen_model.eval()

    source = source_path_audit(frozen_model)
    write_json(OUT / "source_path_audit.json", source)

    distribution = distance_distribution(rows, dataset, bfs_limit=max_bfs)
    write_json(OUT / "distance_distribution.json", distribution)

    descriptor = distance_descriptor_audit(rows, frozen_model, max_descriptor_molecules)
    write_json(OUT / "distance_descriptor_audit.json", descriptor)

    features = hard5_soft5_feature_audit(rows, frozen_model)
    write_json(OUT / "hard5_soft5_feature_audit.json", features)

    correctness = correctness_runtime_gates(rows, prep)
    write_json(OUT / "correctness_gates.json", correctness)

    gate = purchase_gate(distribution, features, correctness)
    write_json(OUT / "cpu_purchase_gate.json", gate)

    summary = {
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_commit(),
        "alignment": alignment,
        "source_all_pass": source["all_static_checks_pass"],
        "descriptor_checks": descriptor["checks"],
        "feature_audit": {
            "frac_gt_1pct": features["relative_feature_difference"]["frac_gt_1pct"],
            "hard_model_readout_max_abs_gap": features["hard_model_readout_max_abs_gap"],
        },
        "correctness_all_pass": correctness["all_correctness_gates_pass"],
        "purchase": gate["purchase"],
        "purchase_label": gate["label"],
        "seconds": time.perf_counter() - t0,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "audit_summary.json", summary)
    log(json.dumps(_j(summary), indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument(
        "stage", choices=("audit", "train", "check", "predict", "score")
    )
    parser.add_argument("--arm", choices=ARM_NAMES, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-descriptor-molecules", type=int, default=10000)
    parser.add_argument("--max-bfs", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(max(4, min(8, torch.get_num_threads())))
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.stage == "audit":
        stage_audit(args.max_descriptor_molecules, args.max_bfs)
        return 0
    if args.stage == "train":
        if args.arm is None:
            raise SystemExit("--arm Hard5|Soft5 required for train")
        train_arm(args.arm, device=device, epochs=args.epochs, smoke=args.smoke)
        return 0
    if args.stage == "check":
        arm_matching_audit()
        return 0
    if args.stage == "predict":
        stage_predict(device)
        return 0
    if args.stage == "score":
        stage_score()
        return 0
    raise SystemExit(f"unknown stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
