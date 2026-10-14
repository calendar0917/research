#!/usr/bin/env python
"""Driver for protocol ``cscl-bond-random-walk-organization-v1``.

Two-stage, label-disciplined workflow:

  CPU (no GPU):
    verify-inputs, prep-rw, audit-structure, audit-communities, synthetic,
    cpu-gates, freeze

  One frozen GPU pass (after ``freeze``):
    train --arm {R,U} --seed 0
    predict --arm {R,U} --seed 0      (Monitor, no labels)
    freeze-predictions
    score-monitor                     (single Monitor label read)
    mechanism                         (label-blind)
    finalize

Split-D labels are read exactly once, inside ``score-monitor``, after every
prediction file is frozen and hashed.  Locked is never touched here; official
valid/test are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_random_walk_organization_v1 as bwo  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = TRACK / "results" / "bond_random_walk_organization_v1"
PRIOR = TRACK / "results" / "bjoc_task_adaptive_ssl_v1"
JUNC_AUDIT = TRACK / "results" / "bjoc_junction_sufficiency_audit_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
CONFIG_PATH = TRACK / "configs" / "bond_random_walk_organization_v1.yaml"

PROTOCOL_ID = bwo.PROTOCOL_ID
SPLIT_TAG = "D"
SEEDS = (0,)
ARMS = ("R", "U")

SPLIT_FINGERPRINTS = {
    "fit_sha256_rows": "9468656affcc2b7c69409f03698b1b6287b3a42dc6e164efd19dfb0190254525",
    "monitor_sha256_rows": "d196bc75ce898c7cfd61efa133372f25803d5bc1198210f60b6e0f981a400393",
    "locked_sha256_rows": "f3525ae882114350a57f7b0ddb6889656f40c05cebda21b2611a2c99cdfa22d5",
}

N_FIT = 8000
N_MONITOR = 1000
AUDIT_N = 2000
INVARIANCE_N = 200
GPU_BUDGET_CAP_HOURS = 1.5

#: frozen read-only reference Monitor predictions (split D, seed 0)
REFERENCE_PATHS = {
    "A0": {"path": JUNC_AUDIT / "predictions" / "monitor_A0_D_s0.npz", "key": "pred"},
    "A1": {"path": JUNC_AUDIT / "predictions" / "monitor_A1_D_s0.npz", "key": "pred"},
    "A2_P3_ref": {"path": JUNC_AUDIT / "predictions" / "monitor_A2_D_s0.npz", "key": "pred"},
    "P3_s0": {"path": PRIOR / "predictions" / "monitor_P3_D_s0.npz", "key": "pred"},
}

#: mechanism acceptance thresholds (frozen)
MECH_K_EFFECT_MIN_MEAN_PRED_DELTA = 0.02
MECH_C2_RELATIVE_RESIDUAL_MIN = 0.10

DECISIONS = {
    "RW_STRUCTURE_USED_AND_GAIN": "R beats U on the frozen Monitor contrast and C1/C2 mechanism checks pass",
    "RW_INFORMATION_BUT_NO_GAIN": "K measurably changes outputs but R does not beat U",
    "RW_DEGENERATE_NO_GAIN": "K has no measurable effect and/or R is not better than U",
    "RW_STOP_PART_A": "Part A gate failed; no GPU was purchased",
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False, default=_json_default)
        fh.write("\n")


def read_json(path: Path | str) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def file_sha256(path: Path | str) -> str:
    return bp.file_sha256(Path(path))


def rows_sha256(rows: Sequence[int]) -> str:
    return bp.rows_sha256(rows)


def array_sha256(a: np.ndarray) -> str:
    return bp.array_sha256(np.asarray(a))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def config_sha256() -> str:
    return file_sha256(CONFIG_PATH)


def load_yaml_config() -> Mapping[str, Any]:
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# inputs / caches
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return PRIOR / "split_D.json"


def fit_objects_path() -> Path:
    return PRIOR / "objects_fit_D.pt"


def monitor_objects_path() -> Path:
    return PRIOR / "objects_monitor_D.pt"


def ssl_path() -> Path:
    return PRIOR / "pretrain_D_s0.pt"


def rw_cache_path(zone: str) -> Path:
    return OUT / f"rw_{zone}_{SPLIT_TAG}.pt"


def soup_path(arm: str) -> Path:
    return OUT / f"soup_{arm}_{SPLIT_TAG}_s0.pt"


def train_json_path(arm: str) -> Path:
    return OUT / f"train_{arm}_{SPLIT_TAG}_s0.json"


def prediction_path(arm: str) -> Path:
    return OUT / "predictions" / f"monitor_{arm}_{SPLIT_TAG}_s0.npz"


def load_split() -> Mapping[str, Any]:
    return read_json(split_path())


def load_zone(zone: str) -> list[sro.MolObjects]:
    path = fit_objects_path() if zone == "fit" else monitor_objects_path()
    objs = torch.load(path, map_location="cpu", weights_only=False)
    return list(objs)


def load_fit_y() -> np.ndarray:
    rows = [int(i) for i in load_split()["fit_rows"]]
    require(len(rows) == N_FIT, "split D fit row count changed")
    return bp.load_y_rows(ENCODED_TRAIN, rows)


def load_ssl_state() -> Mapping[str, Any]:
    return torch.load(ssl_path(), map_location="cpu", weights_only=False)


def audit_objs() -> list[sro.MolObjects]:
    objs = load_zone("fit")
    rows = [int(i) for i in load_split()["fit_rows"]]
    require(len(objs) == len(rows), "fit object count mismatch")
    audit_rows = rows[:AUDIT_N]
    audit_set = set(audit_rows)
    return [o for o in objs if int(o.mol_index) in audit_set][:AUDIT_N]


def load_rw_triplets(zone: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    payload = torch.load(rw_cache_path(zone), map_location="cpu", weights_only=False)
    return [(np.asarray(s), np.asarray(d), np.asarray(w)) for s, d, w in payload["triplets"]]


def arm_triplets(zone: str, arm: str, objs: Sequence[sro.MolObjects]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if arm == "R":
        return load_rw_triplets(zone)
    if arm == "U":
        return [bwo.uniform_triplets(int(o.bond_index.shape[0])) for o in objs]
    raise SystemExit(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# verify-inputs / prep-rw
# ---------------------------------------------------------------------------


def _input_manifest() -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for name, path in (
        ("split_D.json", split_path()),
        ("objects_fit_D.pt", fit_objects_path()),
        ("objects_monitor_D.pt", monitor_objects_path()),
        ("pretrain_D_s0.pt", ssl_path()),
        ("encoded_train.pt", ENCODED_TRAIN),
        ("config", CONFIG_PATH),
    ):
        require(Path(path).exists(), f"missing input {name}")
        inputs[name] = {"sha256": file_sha256(path), "bytes": int(Path(path).stat().st_size)}
    split = load_split()
    rows = {
        "fit_rows": [int(i) for i in split["fit_rows"]],
        "monitor_rows": [int(i) for i in split["monitor_rows"]],
        "locked_rows": [int(i) for i in split["locked_rows"]],
    }
    observed = {
        "fit_sha256_rows": rows_sha256(rows["fit_rows"]),
        "monitor_sha256_rows": rows_sha256(rows["monitor_rows"]),
        "locked_sha256_rows": rows_sha256(rows["locked_rows"]),
    }
    matches = {k: observed[k] == v for k, v in SPLIT_FINGERPRINTS.items()}
    require(all(matches.values()), f"split D fingerprints drifted: {matches}")
    objs = load_zone("fit")
    require([int(o.mol_index) for o in objs] == rows["fit_rows"], "fit object row order mismatch")
    return {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": inputs,
        "split_fingerprints_match": matches,
        "n_fit": len(rows["fit_rows"]),
        "n_monitor": len(rows["monitor_rows"]),
        "n_locked": len(rows["locked_rows"]),
        "fit_y_loaded": False,
        "monitor_y_loaded": False,
        "locked_y_loaded": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def cmd_verify_inputs(_args: argparse.Namespace) -> int:
    payload = _input_manifest()
    write_json(OUT / "input_audit.json", payload)
    log(f"[verify-inputs] fit={payload['n_fit']} monitor={payload['n_monitor']} locked={payload['n_locked']}")
    return 0


def cmd_prep_rw(_args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "formula": "P=(I+D^-1 A_L)/2, K=(P+P^2+P^3)/3, stored as unordered triplets of Ks=(K+K^T)/2",
        "max_k": bwo.MAX_K,
        "kernel_scales": list(bwo.KERNEL_KS),
        "zones": {},
    }
    for zone in ("fit", "monitor"):
        objs = load_zone(zone)
        triplets = []
        checks = {"row_sum_max_err": 0.0, "sym_max_err": 0.0, "reconstruction_max_err": 0.0}
        rng = np.random.default_rng(0)
        for o in objs:
            A = bwo.line_graph_adjacency(o.bond_index)
            P = bwo.lazy_transition(A)
            checks["row_sum_max_err"] = max(checks["row_sum_max_err"], float(np.abs(P.sum(axis=1) - 1.0).max()))
            powers = bwo.kernel_powers(A, bwo.MAX_K)
            K = bwo.relation_kernel(powers)
            Ks = bwo.symmetrize(K)
            checks["sym_max_err"] = max(checks["sym_max_err"], float(np.abs(K - K.T).max()))
            s, d, w = bwo.unordered_triplets(Ks)
            triplets.append((s, d, w))
            # exact double-sum reconstruction with a random symmetric test function
            # (mirrors the index_add_ diagonal-halving convention of pair_aggregate)
            n = A.shape[0]
            if n:
                B = rng.normal(size=(n, 4))
                phi = B[:, None, :] + B[None, :, :]
                w_eff = np.where(s == d, w * 0.5, w)
                phi_pair = phi[s, d]
                r_trip = np.zeros((n, 4))
                np.add.at(r_trip, s, w_eff[:, None] * phi_pair)
                np.add.at(r_trip, d, w_eff[:, None] * phi_pair)
                r_dense = np.einsum("ef,efk->ek", Ks, phi)
                checks["reconstruction_max_err"] = max(
                    checks["reconstruction_max_err"], float(np.abs(r_dense - r_trip).max())
                )
        torch.save({"triplets": triplets}, rw_cache_path(zone))
        payload["zones"][zone] = {
            "n_molecules": len(objs),
            "n_triplets_total": int(sum(int(s.shape[0]) for s, _, _ in triplets)),
            "checks": checks,
            "file_sha256": file_sha256(rw_cache_path(zone)),
        }
        log(
            f"[prep-rw {zone}] n={len(objs)} triplets={payload['zones'][zone]['n_triplets_total']} "
            f"row_sum_err={checks['row_sum_max_err']:.2e} recon_err={checks['reconstruction_max_err']:.2e}"
        )
        require(checks["row_sum_max_err"] <= 1e-9, "lazy transition rows do not sum to 1")
        require(checks["reconstruction_max_err"] <= 1e-6, "triplet reconstruction mismatch")
    write_json(OUT / "rw_manifest.json", payload)
    return 0


# ---------------------------------------------------------------------------
# Part A audits
# ---------------------------------------------------------------------------


def cmd_audit_structure(_args: argparse.Namespace) -> int:
    objs = audit_objs()
    per_mol = [bwo.molecule_walk_stats(o) | {"category": bwo._molecule_category(o)} for o in objs]
    summary = bwo.summarize_walk_stats(per_mol)
    # invariance on a deterministic subset
    rng = np.random.default_rng(20261014)
    inv_atom, inv_bond, max_stat = [], [], 0.0
    for o in objs[:INVARIANCE_N]:
        inv_atom.append(bwo.atom_relabel_invariance(o, rng))
        bpv = bwo.bond_permutation_equivariance(o, rng)
        inv_bond.append(float(bpv["kernel_conjugation_max_abs"]))
        max_stat = max(max_stat, float(bpv["statistic_max_abs"]))
    # bond-class diffusion
    from collections import defaultdict

    by_class: dict[str, list[float]] = defaultdict(list)
    for o in objs:
        A = bwo.line_graph_adjacency(o.bond_index)
        powers = bwo.kernel_powers(A, bwo.MAX_K)
        n_b = A.shape[0]
        for e in range(n_b):
            row_tv = float(0.5 * np.abs(powers[3][e] - 1.0 / n_b).sum())
            by_class[bwo._bond_class(o, e)].append(row_tv)
    class_stats = {
        k: {"n": len(v), "mean_tv_uniform_k3": float(np.mean(v))} for k, v in sorted(by_class.items())
    }
    # real-molecule pair search: same structural marginal signature, different K spectrum
    pairs = _collision_search(objs)
    invariants = {
        "atom_relabel_max_kernel_diff": float(max(inv_atom)) if inv_atom else 0.0,
        "bond_permutation_max_conjugation_diff": float(max(inv_bond)) if inv_bond else 0.0,
        "bond_permutation_max_statistic_diff": float(max_stat),
    }
    gate_conditions = dict(summary["gate_conditions"])
    gate_conditions["invariance_ok"] = bool(
        max(inv_atom + inv_bond + [0.0]) <= 1e-9 and max_stat <= 1e-9
    )
    decision = "PART_A_PASS_GPU_PURCHASE"
    if not gate_conditions["invariance_ok"]:
        decision = "PART_A_STOP_INVARIANCE"
    elif not gate_conditions["mean_tv_p3_ge_min"]:
        decision = "PART_A_STOP_RW_DEGENERATES_TO_GLOBAL_MEAN"
    elif not (gate_conditions["mean_tv_K_ge_min"] and gate_conditions["scale_distinctness_ge_min"]):
        decision = "PART_A_STOP_SCALES_REDUNDANT"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "audit_rows": "first 2000 split-D Fit rows (sorted split order)",
        "n_molecules": len(objs),
        "definitions": {
            "tv_uniform_k": "mean_e 0.5*sum_f |P^k_ef - 1/N|",
            "increment_cos_12": "cosine of flattened (P^1-U) and (P^2-U)",
            "scale_distinctness_p1p3": "mean_e TV(P^1_e, P^3_e) = mean_e 0.5*sum_f |P^3_ef - P^1_ef|; the raw matrix L1 / (2N^2) is kept as scale_distinctness_raw_p1p3",
            "K_tv_uniform": "mean_e 0.5*sum_f |K_ef - 1/N|",
        },
        "summary": summary,
        "invariants": invariants,
        "bond_class_diffusion": class_stats,
        "real_pair_search": pairs,
        "gate_conditions": gate_conditions,
        "decision": decision,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "structure_audit.json", payload)
    with (OUT / "structure_per_molecule.jsonl").open("w") as fh:
        for rec in per_mol:
            fh.write(json.dumps(rec, default=_json_default) + "\n")
    log(f"[audit-structure] decision={decision}")
    log(
        f"  tv_k1={summary['aggregate']['tv_uniform_k1']:.3f} tv_k3={summary['aggregate']['tv_uniform_k3']:.3f} "
        f"K_tv={summary['aggregate']['K_tv_uniform']:.3f} distinct={summary['aggregate']['scale_distinctness_p1p3']:.3f}"
    )
    return 0


def _collision_search(objs: Sequence[sro.MolObjects], max_pairs: int = 50) -> dict[str, Any]:
    """Molecules with identical coarse *marginal* summaries but different K.

    The signature uses only quantities visible to an unstructured pair-summary
    control: the number of bonds, the multiset of bond types, the multiset of
    line-graph degrees, and the multiset of per-bond junction counts
    (equivalently the atom-degree sequence).  Two molecules in the same bucket
    therefore have identical raw marginals; their K differs only through how
    those marginals are organised.  K equality is tested permutation-invariantly
    via the sorted vector of all K entries (rounded to 6 decimals); K difference
    is the max abs difference of those sorted vectors.

    A looser ``degree-multiset only`` bucket count is reported as context.
    """
    from collections import defaultdict

    buckets: dict[tuple, list[int]] = defaultdict(list)
    loose: dict[tuple, list[int]] = defaultdict(list)
    spectra: dict[int, np.ndarray] = {}
    for i, o in enumerate(objs):
        A = bwo.line_graph_adjacency(o.bond_index)
        n_b = int(o.bond_index.shape[0])
        deg = tuple(sorted(int(x) for x in A.sum(axis=1)))
        per_bond_junc = np.zeros(n_b, dtype=np.int64)
        for jb in o.junc_bond:
            per_bond_junc[jb] += 1
        jmult = tuple(sorted(int(x) for x in per_bond_junc))
        sig = (n_b, tuple(sorted(int(t) for t in o.bond_type)), deg, jmult)
        buckets[sig].append(i)
        loose[(n_b, deg)].append(i)
        K = bwo.relation_kernel(bwo.kernel_powers(A, bwo.MAX_K))
        spectra[i] = np.round(np.sort(K, axis=None), 6)
    groups, pair_diffs = [], []
    for sig, members in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        best, best_diff = None, -1.0
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                si, sj = spectra[i], spectra[j]
                if si.shape != sj.shape:
                    diff = float("inf")
                else:
                    diff = float(np.abs(si - sj).max())
                if diff <= 1e-9:
                    continue
                pair_diffs.append(diff)
                if diff > best_diff:
                    best, best_diff = (i, j), diff
        if best is not None:
            groups.append(
                {
                    "bucket_size": len(members),
                    "n_bonds": int(sig[0]),
                    "example_pair": {
                        "i": int(best[0]),
                        "j": int(best[1]),
                        "K_sorted_max_abs_diff": float(best_diff),
                    },
                }
            )
            if len(groups) >= int(max_pairs):
                break
    n_loose_multi = sum(1 for v in loose.values() if len(v) > 1)
    return {
        "signature": "(n_bonds, sorted bond types, sorted line-graph degrees, sorted per-bond junction counts)",
        "n_molecules": len(objs),
        "n_signature_buckets": len(buckets),
        "n_multi_molecule_buckets": sum(1 for v in buckets.values() if len(v) > 1),
        "n_pairs_with_identical_marginals_and_different_K": len(pair_diffs),
        "K_equality_test": "permutation-invariant sorted K entries, 1e-9 tolerance",
        "mean_K_sorted_max_abs_diff_among_matched_pairs": float(np.mean(pair_diffs)) if pair_diffs else None,
        "max_K_sorted_max_abs_diff_among_matched_pairs": float(np.max(pair_diffs)) if pair_diffs else None,
        "examples": groups[:10],
        "loose_signature": "(n_bonds, sorted line-graph degrees)",
        "n_loose_multi_molecule_buckets": int(n_loose_multi),
        "note": "marginal signature is coarse (raw types/degrees/junction counts); K differs within the bucket",
    }


def cmd_audit_communities(args: argparse.Namespace) -> int:
    objs = audit_objs()
    per_mol = [bwo.community_analysis_molecule(o, relabel_trials=2, seed=0) for o in objs]
    summary = bwo.summarize_communities(per_mol)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_molecules": len(objs),
        "walktrap_steps": bwo.WALKTRAP_STEPS,
        "summary": summary,
        "per_molecule": per_mol,
        "claims_boundary": (
            "community counts/sizes are structural observations, not chemistry; bond communities "
            "guarantee unique bond membership but not unique atom membership"
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "community_audit.json", payload)
    log(
        f"[audit-communities] walktrap mean={summary['mean_communities']['walktrap']:.2f} "
        f"tasu={summary['mean_communities']['tasu_2ecc_bridge']:.2f} "
        f"ARI(wt,tasu)={summary['mean_ari']['walktrap_vs_tasu']:.3f}"
    )
    return 0


# ---------------------------------------------------------------------------
# synthetic witnesses + cpu gates
# ---------------------------------------------------------------------------


def _random_pair_net(in_dim: int, out_dim: int, hidden: int, seed: int) -> torch.nn.Module:
    net = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, out_dim))
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.as_tensor(rng.normal(scale=0.5, size=tuple(p.shape)), dtype=torch.float32))
    return net


def cmd_synthetic(_args: argparse.Namespace) -> int:
    pair_ab = _random_pair_net(24, 8, 8, seed=0)
    pair_res = _random_pair_net(3 * bwo.BondRWConfig().pair_out, bwo.BondRWConfig().pair_out, bwo.PAIR_HIDDEN, seed=1)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "junction_moment_counterexample": bwo.junction_moment_counterexample(),
        "pair_block_ablation_random_phi": bwo.pair_block_ablation(pair_ab, seed=0, dim=8),
        "pair_linear_summary_residual_random_phi": bwo.pair_linear_summary_residual(
            pair_res, n_molecules=48, seed=0, dim=96
        ),
        "synthetic_relation_task": bwo.synthetic_relation_task(seed=0),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "synthetic_witnesses.json", payload)
    task = payload["synthetic_relation_task"]
    log(
        f"[synthetic] real-K {task['test_mae_real_K_trained']:.4f} vs uniform-K "
        f"{task['test_mae_uniform_K_trained_uniform']:.4f} vs linear {task['test_mae_linear_summary']:.4f}"
    )
    return 0


def cmd_cpu_gates(args: argparse.Namespace) -> int:
    objs = load_zone("fit")[:256]
    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32), "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32)}
    device = torch.device(args.device)

    # --- kernel gates on the sampled molecules
    kernel_checks = {"row_sum_max": 0.0, "power_recursion_max": 0.0, "zero_degree_ok": True}
    for o in objs:
        A = bwo.line_graph_adjacency(o.bond_index)
        P = bwo.lazy_transition(A)
        kernel_checks["row_sum_max"] = max(kernel_checks["row_sum_max"], float(np.abs(P.sum(axis=1) - 1).max()))
        powers = bwo.kernel_powers(A, bwo.MAX_K)
        for k in range(2, bwo.MAX_K + 1):
            kernel_checks["power_recursion_max"] = max(
                kernel_checks["power_recursion_max"],
                float(np.abs(powers[k] - powers[k - 1] @ P).max()),
            )
        if A.shape[0] == 1:
            kernel_checks["zero_degree_ok"] = bool(kernel_checks["zero_degree_ok"] and abs(P[0, 0] - 1.0) < 1e-12)

    # --- model/init parity
    model_R = bwo.build_model(0, norm, "cpu")
    model_U = bwo.build_model(0, norm, "cpu")
    init_parity = {
        "parameter_count_R": bwo.parameter_count(model_R),
        "parameter_count_U": bwo.parameter_count(model_U),
        "parameter_counts_equal": bwo.parameter_count(model_R) == bwo.parameter_count(model_U),
        "parameter_count_matches_frozen": bwo.parameter_count(model_R) == bwo.EXPECTED_PARAMETERS,
        "init_state_equal": bwo.state_dict_hash(model_R) == bwo.state_dict_hash(model_U),
        "composer_in": int(bwo.BondRWConfig().compose_in),
        "reader_in": int(bwo.BondRWConfig().reader_in),
        "forbidden_modules": bwo.forbidden_module_scan(model_R),
    }
    ssl = load_ssl_state()
    bwo.load_ssl_encoders(model_R, ssl, "cpu")
    bwo.load_ssl_encoders(model_U, ssl, "cpu")
    init_parity["ssl_encoder_hash_equal"] = bwo.encoder_group_hash(model_R) == bwo.encoder_group_hash(model_U)
    fresh = bwo.build_model(0, norm, "cpu")
    init_parity["ssl_encoder_differs_from_scratch"] = (
        bwo.encoder_group_hash(model_R) != bwo.encoder_group_hash(fresh)
    )

    # --- forward / invariance / K-vs-U on real fit rows
    rng = np.random.default_rng(7)
    triplets = [bwo.kernel_triplets_from_objects(o) for o in objs]
    uniform = [bwo.uniform_triplets(int(o.bond_index.shape[0])) for o in objs]
    forward_checks: dict[str, Any] = {}
    batch = bwo.move_batch(bwo.batch_rw_objects(objs[:64], triplets[:64]), device)
    model_R.eval()
    with torch.no_grad():
        pred_R = model_R(batch)
        batch_u = bwo.move_batch(bwo.batch_rw_objects(objs[:64], uniform[:64]), device)
        pred_U = model_R(batch_u)
        pred_single = []
        for i in range(64):
            b1 = bwo.move_batch(bwo.batch_rw_objects([objs[i]], [triplets[i]]), device)
            pred_single.append(float(model_R(b1)))
        pred_single = np.asarray(pred_single)
    forward_checks["batch_membership_max_abs_diff"] = float(np.abs(pred_single - pred_R.detach().cpu().numpy()).max())
    forward_checks["K_vs_uniform_mean_abs_dpred"] = float((pred_R - pred_U).abs().mean())
    forward_checks["all_finite"] = bool(np.isfinite(pred_single).all())

    # relabel invariance of the model forward
    inv_vals = []
    model_R.eval()
    for i in range(16):
        o = objs[i]
        perm = rng.permutation(o.n_atoms)
        inv = np.argsort(perm)  # new id of old atom u: atom_feat_new[i] = atom_feat_old[perm[i]]
        o2 = sro.MolObjects(
            mol_index=o.mol_index,
            smiles=o.smiles,
            n_atoms=o.n_atoms,
            atom_feat=o.atom_feat[perm],
            bond_index=inv[o.bond_index],
            bond_type=o.bond_type,
            bond_feat=o.bond_feat,
            junc_index=o.junc_index,
            junc_bond=o.junc_bond,
            junc_feat=o.junc_feat,
            global_feat=o.global_feat,
        )
        with torch.no_grad():
            b1 = bwo.move_batch(bwo.batch_rw_objects([o], [triplets[i]]), device)
            b2 = bwo.move_batch(bwo.batch_rw_objects([o2], [triplets[i]]), device)
            inv_vals.append(abs(float(model_R(b1)) - float(model_R(b2))))
    forward_checks["atom_relabel_max_abs_dpred"] = float(max(inv_vals)) if inv_vals else 0.0

    # save/load
    import copy

    state = {k: v.detach().clone() for k, v in model_R.state_dict().items()}
    model2 = bwo.build_model(0, norm, "cpu")
    model2.load_state_dict(state)
    model2.eval()
    with torch.no_grad():
        pred2 = model2(batch)
    forward_checks["save_load_max_abs_dpred"] = float((pred2 - pred_R).abs().max())

    # gradient liveness of the pair MLP
    model_g = bwo.build_model(0, norm, "cpu")
    bwo.load_ssl_encoders(model_g, ssl, "cpu")
    model_g.train()
    batch_g = bwo.move_batch(bwo.batch_rw_objects(objs[:32], triplets[:32]), torch.device("cpu"))
    yb = torch.zeros(32, dtype=torch.float32)  # label-blind liveness probe
    pred = model_g(batch_g)
    loss = torch.nn.functional.l1_loss(pred.view(-1), yb.view(-1))
    loss.backward()
    pair_grads = [
        float(p.grad.detach().norm()) for p in model_g.pair_mlp.parameters() if p.grad is not None
    ]
    grad_checks = {
        "pair_mlp_grad_norms": pair_grads,
        "pair_mlp_grad_live": bool(all(g > 0 for g in pair_grads)),
        "pair_mlp_grad_norm_mean": float(np.mean(pair_grads)) if pair_grads else 0.0,
    }

    # C2 witnesses on the random-init pair MLP (8-D inputs, matching block dim)
    random_pair = _random_pair_net(24, 8, 8, seed=2)
    witnesses = {
        "junction_moment_counterexample": bwo.junction_moment_counterexample(),
        "pair_block_ablation": bwo.pair_block_ablation(random_pair, seed=0, dim=8),
    }

    # U control properties
    u_row = bwo.uniform_triplets(9)
    uniform_checks = {
        "n_triplets": int(u_row[0].shape[0]),
        "expected_n_triplets": 9 * 10 // 2,
        "weights_all_equal": bool(np.allclose(u_row[2], u_row[2][0])),
        "weight_value": float(u_row[2][0]),
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": str(device),
        "kernel_checks": kernel_checks,
        "model_checks": init_parity,
        "forward_checks": forward_checks,
        "gradient_checks": grad_checks,
        "uniform_checks": uniform_checks,
        "witnesses": witnesses,
        "passed": bool(
            kernel_checks["row_sum_max"] <= 1e-9
            and kernel_checks["power_recursion_max"] <= 1e-9
            and kernel_checks["zero_degree_ok"]
            and init_parity["parameter_counts_equal"]
            and init_parity["parameter_count_matches_frozen"]
            and init_parity["init_state_equal"]
            and init_parity["ssl_encoder_differs_from_scratch"]
            and not init_parity["forbidden_modules"]
            and forward_checks["batch_membership_max_abs_diff"] <= 1e-5
            and forward_checks["atom_relabel_max_abs_dpred"] <= 1e-5
            and forward_checks["save_load_max_abs_dpred"] <= 1e-6
            and grad_checks["pair_mlp_grad_live"]
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "cpu_gates.json", payload)
    _render_checks(OUT / "CHECKS.md", payload)
    log(f"[cpu-gates] passed={payload['passed']}")
    return 0


def _render_checks(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# CPU gates — cscl-bond-random-walk-organization-v1",
        "",
        f"- device: `{payload['device']}` · commit: `{payload['git_commit']}` · passed: **{payload['passed']}**",
        "",
        "## kernels",
        f"- row-sum max error: {payload['kernel_checks']['row_sum_max']:.3e}",
        f"- power recursion max error: {payload['kernel_checks']['power_recursion_max']:.3e}",
        f"- zero-degree rule: {payload['kernel_checks']['zero_degree_ok']}",
        "",
        "## model",
        f"- R/U parameter counts equal: {payload['model_checks']['parameter_counts_equal']}"
        f" ({payload['model_checks']['parameter_count_R']})",
        f"- R/U init state equal: {payload['model_checks']['init_state_equal']}",
        f"- SSL encoder loaded and differs from scratch: {payload['model_checks']['ssl_encoder_differs_from_scratch']}",
        f"- forbidden modules: {payload['model_checks']['forbidden_modules']}",
        "",
        "## forward",
        f"- batch membership max |dpred|: {payload['forward_checks']['batch_membership_max_abs_diff']:.3e}",
        f"- atom relabel max |dpred|: {payload['forward_checks']['atom_relabel_max_abs_dpred']:.3e}",
        f"- save/load max |dpred|: {payload['forward_checks']['save_load_max_abs_dpred']:.3e}",
        f"- K vs uniform mean |dpred|: {payload['forward_checks']['K_vs_uniform_mean_abs_dpred']:.4f}",
        "",
        "## pair MLP",
        f"- gradient norms: {[round(g, 4) for g in payload['gradient_checks']['pair_mlp_grad_norms']]}",
        f"- live: {payload['gradient_checks']['pair_mlp_grad_live']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


FREEZE_CODE_FILES = (
    "bond_random_walk_organization_v1.py",
    "run_bond_random_walk_organization_v1.py",
    "structural_relation_composer_v1.py",
    "structural_relation_objects_v1.py",
    "bjoc_perfmech_long_v1.py",
    "bjoc_masked_pretraining_v1.py",
    "task_adaptive_structure_units_v1.py",
    "cscl_units.py",
)


def cmd_freeze(_args: argparse.Namespace) -> int:
    inputs = _input_manifest()
    require(rw_cache_path("fit").exists(), "rw_fit_D.pt missing; run prep-rw first")
    require(rw_cache_path("monitor").exists(), "rw_monitor_D.pt missing; run prep-rw first")
    code = {name: file_sha256(_CODE_DIR / name) for name in FREEZE_CODE_FILES}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": config_sha256(),
        "code_hashes": code,
        "split_tag": SPLIT_TAG,
        "split_fingerprints": SPLIT_FINGERPRINTS,
        "inputs": inputs["inputs"],
        "rw_caches": {
            "fit": {"sha256": file_sha256(rw_cache_path("fit")), "bytes": int(rw_cache_path("fit").stat().st_size)},
            "monitor": {"sha256": file_sha256(rw_cache_path("monitor")), "bytes": int(rw_cache_path("monitor").stat().st_size)},
        },
        "recipe": {"key": "R3", "epochs": 360, "updates": 22680, "loss": "raw-y L1", "soup": "last-5"},
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "protocol_freeze.json", payload)
    log(f"[freeze] {len(code)} code hashes; config {payload['config_sha256'][:12]}")
    return 0


def verify_freeze(require_exact_code: bool = True) -> Mapping[str, Any]:
    path = OUT / "protocol_freeze.json"
    require(path.exists(), "protocol_freeze.json missing; run freeze first")
    payload = read_json(path)
    require(payload["protocol"] == PROTOCOL_ID, "freeze protocol id mismatch")
    if require_exact_code:
        for name, sha in payload["code_hashes"].items():
            require(file_sha256(_CODE_DIR / name) == sha, f"code drift detected in {name}; re-freeze is forbidden")
        require(config_sha256() == payload["config_sha256"], "config drifted from freeze")
    return payload


# ---------------------------------------------------------------------------
# train / predict / score
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    arm, seed = args.arm, int(args.seed)
    require(arm in ARMS, f"unknown arm {arm!r}")
    device = torch.device(args.device)
    objs = load_zone("fit")
    y = load_fit_y()
    triplets = arm_triplets("fit", arm, objs)
    ssl = load_ssl_state()
    log(f"[train {arm} s{seed}] device={device} n={len(objs)}")
    model, info = bwo.train_supervised_rw(
        arm=arm,
        seed=seed,
        train_objs=objs,
        train_triplets=triplets,
        y_fit=y,
        norm=read_json(PRIOR / "prep_fit_D.json"),
        device=device,
        ssl_state=ssl,
        smoke_epochs=args.epochs if args.smoke else None,
        log_fn=log,
    )
    if args.smoke:
        info["smoke"] = True
        write_json(OUT / f"smoke_train_{arm}_s{seed}.json", info)
        return 0
    info["smoke"] = False
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, soup_path(arm))
    write_json(train_json_path(arm), info)
    _append_ledger(
        {
            "arm": arm,
            "seed": seed,
            "wall_seconds": info["wall_seconds"],
            "device": info["device"],
            "epochs": info["epochs"],
            "phase": "formal",
        }
    )
    log(f"[train {arm} s{seed}] soup_fit_mae={info['soup_fit_mae']:.6f} wall={info['wall_seconds']:.0f}s")
    return 0


def _append_ledger(entry: Mapping[str, Any]) -> None:
    path = OUT / "gpu_budget_ledger.json"
    payload = read_json(path) if path.exists() else {"protocol": PROTOCOL_ID, "entries": []}
    payload["entries"].append({**entry, "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    total = float(sum(float(e["wall_seconds"]) for e in payload["entries"]))
    payload["total_wall_hours"] = total / 3600.0
    payload["cap_hours"] = GPU_BUDGET_CAP_HOURS
    payload["within_cap"] = bool(total / 3600.0 <= GPU_BUDGET_CAP_HOURS)
    write_json(path, payload)
    log(f"[ledger] total {payload['total_wall_hours']:.3f} h / cap {GPU_BUDGET_CAP_HOURS} h")


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    arm, seed = args.arm, 0
    require(soup_path(arm).exists(), f"missing soup for {arm}")
    device = torch.device(args.device)
    objs = load_zone("monitor")
    triplets = arm_triplets("monitor", arm, objs)
    model = bwo.build_model(0, read_json(PRIOR / "prep_fit_D.json"), device)
    model.load_state_dict({k: v.to(device) for k, v in torch.load(soup_path(arm), map_location="cpu").items()})
    preds = bwo.predict_model(model, objs, triplets, device)
    path = prediction_path(arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pred=preds.astype(np.float32), arm=arm, seed=seed, n_molecules=len(preds))
    meta = {
        "arm": arm,
        "seed": seed,
        "n_molecules": int(len(preds)),
        "pred_sha256": array_sha256(preds),
        "file_sha256": file_sha256(path),
        "labels_used": "none",
    }
    manifest_path = OUT / "prediction_manifest_D.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"protocol": PROTOCOL_ID, "records": {}}
    manifest["records"][arm] = meta
    write_json(manifest_path, manifest)
    log(f"[predict {arm}] n={len(preds)} sha={meta['pred_sha256'][:12]}")
    return 0


def cmd_freeze_predictions(_args: argparse.Namespace) -> int:
    manifest_path = OUT / "prediction_manifest_D.json"
    manifest = read_json(manifest_path)
    missing = sorted(set(ARMS) - set(manifest["records"]))
    require(not missing, f"missing predictions: {missing}")
    manifest["complete"] = True
    manifest["n_predictions"] = len(manifest["records"])
    manifest["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: v["pred_sha256"] for k, v in sorted(manifest["records"].items())}, sort_keys=True).encode()
    ).hexdigest()
    write_json(manifest_path, manifest)
    log(f"[freeze-predictions] {manifest['n_predictions']} files frozen sha={manifest['manifest_sha256'][:12]}")
    return 0


def _verify_prediction_manifest() -> Mapping[str, Any]:
    manifest = read_json(OUT / "prediction_manifest_D.json")
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    for key, rec in manifest["records"].items():
        p = OUT / "predictions" / f"monitor_{key}_{SPLIT_TAG}_s0.npz"
        require(p.exists(), f"prediction file missing: {p}")
        require(file_sha256(p) == rec["file_sha256"], f"prediction file drifted: {key}")
    return manifest


def cmd_score_monitor(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    require(not (OUT / "monitor_read_D.lock").exists(), "monitor labels were already read")
    manifest = _verify_prediction_manifest()
    rows = [int(i) for i in load_split()["monitor_rows"]]
    require(len(rows) == N_MONITOR, "monitor row count changed")
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    write_json(
        OUT / "monitor_read_D.lock",
        {
            "protocol": PROTOCOL_ID,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(y),
            "y_sha256": array_sha256(y),
            "rows_sha256": rows_sha256(rows),
            "prediction_manifest_sha256": manifest["manifest_sha256"],
        },
    )
    preds = {
        arm: np.asarray(np.load(prediction_path(arm))["pred"], dtype=np.float64) for arm in ARMS
    }
    scores = {arm: float(np.abs(preds[arm] - y).mean()) for arm in ARMS}
    scores["R_minus_U"] = float(scores["R"] - scores["U"])
    primary = ab.compute_contrast({0: preds["R"]}, {0: preds["U"]}, y)
    references = {}
    for name, spec in REFERENCE_PATHS.items():
        path = Path(spec["path"])
        if path.exists():
            rp = np.asarray(np.load(path)[spec["key"]], dtype=np.float64)
            if rp.shape[0] == len(y):
                references[name] = {
                    "monitor_mae": float(np.abs(rp - y).mean()),
                    "delta_vs_R": ab.compute_contrast({0: preds["R"]}, {0: rp}, y),
                }
    err_R = np.abs(preds["R"] - y)
    err_U = np.abs(preds["U"] - y)
    ordinal = y > bp.TAIL_THRESHOLD

    def _band(mask):
        return {
            "n": int(mask.sum()),
            "mae_R": float(err_R[mask].mean()) if mask.any() else None,
            "mae_U": float(err_U[mask].mean()) if mask.any() else None,
            "delta": float((err_R[mask] - err_U[mask]).mean()) if mask.any() else None,
        }

    subgroup = {"ordinary": _band(ordinal), "tail": _band(~ordinal)}
    decision = "RW_INFORMATION_BUT_NO_GAIN"
    if primary["bootstrap"]["ci_hi"] < 0 and primary["delta_seed_mean"] < 0:
        decision = "RW_STRUCTURE_USED_AND_GAIN_PENDING_MECHANISM"
    if primary["delta_seed_mean"] >= 0:
        decision = "RW_DEGENERATE_NO_GAIN"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "monitor_mae": scores,
        "primary_contrast_R_minus_U": primary,
        "subgroup": subgroup,
        "reference_reads": references,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "monitor_read_utc": read_json(OUT / "monitor_read_D.lock")["read_utc"],
        "decision_pending_mechanism": decision,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "monitor_score_D.json", payload)
    log(f"[score-monitor] R={scores['R']:.6f} U={scores['U']:.6f} delta={scores['R_minus_U']:+.6f}")
    log(f"  CI95 [{primary['bootstrap']['ci_lo']:+.6f}, {primary['bootstrap']['ci_hi']:+.6f}] decision={decision}")
    return 0


# ---------------------------------------------------------------------------
# mechanism
# ---------------------------------------------------------------------------


def cmd_mechanism(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    device = torch.device(_args.device)
    norm = read_json(PRIOR / "prep_fit_D.json")
    fit_objs = load_zone("fit")[:128]
    fit_triplets_real = [bwo.kernel_triplets_from_objects(o) for o in fit_objs]
    monitor_objs = load_zone("monitor")
    monitor_triplets_real = load_rw_triplets("monitor")

    results: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit()}
    models = {}
    for arm in ARMS:
        model = bwo.build_model(0, norm, device)
        model.load_state_dict({k: v.to(device) for k, v in torch.load(soup_path(arm), map_location="cpu").items()})
        model.eval()
        models[arm] = model

    # C1a: R vs U monitor predictions
    pred_R = np.asarray(np.load(prediction_path("R"))["pred"], dtype=np.float64)
    pred_U = np.asarray(np.load(prediction_path("U"))["pred"], dtype=np.float64)
    results["C1_R_vs_U_predictions"] = {
        "mean_abs_dpred": float(np.abs(pred_R - pred_U).mean()),
        "max_abs_dpred": float(np.abs(pred_R - pred_U).max()),
        "corr": float(np.corrcoef(pred_R, pred_U)[0, 1]),
    }

    # C1b: pair MLP gradients on real Fit rows (from the trained R soup)
    model = models["R"]
    model.train()
    pair_grads = []
    for i in range(0, min(64, len(fit_objs)), 32):
        chunk = fit_objs[i : i + 32]
        trip = fit_triplets_real[i : i + 32]
        batch = bwo.move_batch(bwo.batch_rw_objects(chunk, trip), device)
        pred = model(batch)
        loss = pred.sum() * 0.0 + pred.pow(2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        pair_grads.append(
            [float(p.grad.norm()) for p in model.pair_mlp.parameters() if p.grad is not None]
        )
    results["C1_pair_gradients"] = {
        "per_batch_norms": pair_grads,
        "live": bool(all(all(g > 0 for g in row) for row in pair_grads)),
    }

    # C1c / C2: fixed-representation relation reassignment on monitor molecules (no labels)
    witness = bwo.relation_reassignment_witness(
        models["R"], monitor_objs, monitor_triplets_real, device, n_molecules=64, seed=0
    )
    results["C2_relation_reassignment"] = witness

    # C2: block ablation and linear-summary residual of the trained Pair MLP
    results["C2_pair_block_ablation_trained"] = bwo.pair_block_ablation(
        model.pair_mlp, seed=0, dim=bwo.BondRWConfig().pair_out
    )
    results["C2_linear_summary_residual_trained"] = bwo.pair_linear_summary_residual(
        model.pair_mlp, n_molecules=48, seed=0, dim=bwo.BondRWConfig().pair_out
    )

    # C1: R vs U on the synthetic relation task with the trained heads kept random
    # (already computed at init in cpu-gates; recompute here for the record)
    results["C1_synthetic_relation_task"] = bwo.synthetic_relation_task(seed=0)

    # acceptance flags (frozen thresholds)
    k_effect = witness["mean_pred_delta"]
    results["acceptance"] = {
        "k_effect_min": MECH_K_EFFECT_MIN_MEAN_PRED_DELTA,
        "k_effect_pass": bool(max(k_effect["uniform"], k_effect["row_perm"], k_effect["bond_perm"]) >= MECH_K_EFFECT_MIN_MEAN_PRED_DELTA),
        "linear_summary_residual_min": MECH_C2_RELATIVE_RESIDUAL_MIN,
        "linear_summary_residual_pass": bool(
            results["C2_linear_summary_residual_trained"]["relative_residual"] >= MECH_C2_RELATIVE_RESIDUAL_MIN
        ),
        "product_block_used_pass": bool(results["C2_pair_block_ablation_trained"]["zero_prod_rel"] >= 0.05),
        "pair_grad_live_pass": bool(results["C1_pair_gradients"]["live"]),
    }
    results["official_valid_loaded"] = False
    results["official_test_loaded"] = False
    write_json(OUT / "mechanism_audit.json", results)
    log(f"[mechanism] C1 R-vs-U mean|dpred|={results['C1_R_vs_U_predictions']['mean_abs_dpred']:.3f}")
    log(f"  K reassignment pred deltas: {json.dumps(k_effect)}")
    log(f"  acceptance: {json.dumps(results['acceptance'])}")
    return 0


# ---------------------------------------------------------------------------
# finalize: reports
# ---------------------------------------------------------------------------


def cmd_finalize(_args: argparse.Namespace) -> int:
    structure = read_json(OUT / "structure_audit.json")
    communities = read_json(OUT / "community_audit.json")
    witnesses = read_json(OUT / "synthetic_witnesses.json")
    gates = read_json(OUT / "cpu_gates.json")
    score = read_json(OUT / "monitor_score_D.json")
    mechanism = read_json(OUT / "mechanism_audit.json")
    train_R = read_json(train_json_path("R"))
    train_U = read_json(train_json_path("U"))
    freeze = read_json(OUT / "protocol_freeze.json")
    ledger = read_json(OUT / "gpu_budget_ledger.json")

    primary = score["primary_contrast_R_minus_U"]
    mech = mechanism["acceptance"]
    if primary["bootstrap"]["ci_hi"] < 0 and mech["k_effect_pass"] and mech["pair_grad_live_pass"]:
        decision = "RW_STRUCTURE_USED_AND_GAIN"
    elif mech["k_effect_pass"] and (primary["delta_seed_mean"] >= 0 or primary["bootstrap"]["ci_hi"] >= 0):
        decision = "RW_INFORMATION_BUT_NO_GAIN"
    else:
        decision = "RW_DEGENERATE_NO_GAIN"

    decision_payload = {
        "protocol": PROTOCOL_ID,
        "decision": decision,
        "part_a_decision": structure["decision"],
        "primary_R_minus_U": {
            "mae_R": score["monitor_mae"]["R"],
            "mae_U": score["monitor_mae"]["U"],
            "delta": primary["delta_seed_mean"],
            "ci95": [primary["bootstrap"]["ci_lo"], primary["bootstrap"]["ci_hi"]],
            "frac_R_better": primary["frac_molecules_better"],
        },
        "mechanism_acceptance": mech,
        "params": {
            "R": train_R["parameter_audit"]["actual_parameters"],
            "U": train_U["parameter_audit"]["actual_parameters"],
            "historical_P3": bp.EXPECTED_PARAMETERS,
        },
        "gpu_hours": ledger["total_wall_hours"],
        "stop": decision != "RW_STRUCTURE_USED_AND_GAIN",
        "official_test_loaded": False,
    }
    write_json(OUT / "decision.json", decision_payload)
    _render_structure_md(OUT / "STRUCTURE_AUDIT.md", structure, communities)
    _render_synthetic_md(OUT / "SYNTHETIC_WITNESSES.md", witnesses, mechanism)
    _render_report_md(OUT / "REPORT.md", decision_payload, structure, communities, witnesses, gates, score, mechanism, train_R, train_U, freeze, ledger)
    write_json(
        OUT / "evidence_manifest.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "files": {
                name: file_sha256(OUT / name)
                for name in (
                    "structure_audit.json",
                    "community_audit.json",
                    "synthetic_witnesses.json",
                    "cpu_gates.json",
                    "protocol_freeze.json",
                    "monitor_score_D.json",
                    "mechanism_audit.json",
                    "decision.json",
                    "train_R_D_s0.json",
                    "train_U_D_s0.json",
                    "gpu_budget_ledger.json",
                )
                if (OUT / name).exists()
            },
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
    )
    log(f"[finalize] decision={decision}")
    return 0


def _fmt(x: float) -> str:
    return f"{float(x):.6f}"


def _render_structure_md(path: Path, structure: Mapping[str, Any], communities: Mapping[str, Any]) -> None:
    s = structure["summary"]
    agg = s["aggregate"]
    lines = [
        "# Bond line-graph random-walk structure audit",
        "",
        f"- protocol `{structure['protocol']}` · commit `{structure['git_commit']}`",
        f"- {structure['n_molecules']} split-D Fit molecules (label-blind), first 2000 sorted Fit rows",
        f"- Part A decision: **{structure['decision']}**",
        "",
        "## 1. Kernel scales and mixing",
        "",
        "| quantity | k=1 | k=2 | k=3 | k=4 |",
        "|---|---:|---:|---:|---:|",
        f"| mean TV(P^k, uniform) | {agg['tv_uniform_k1']:.3f} | {agg['tv_uniform_k2']:.3f} | {agg['tv_uniform_k3']:.3f} | {agg['tv_uniform_k4']:.3f} |",
        f"| mean normalised entropy | {agg['entropy_norm_k1']:.3f} | – | {agg['entropy_norm_k3']:.3f} | – |",
        f"| mean reachable fraction | {agg['reachable_k1']:.3f} | – | {agg['reachable_k3']:.3f} | – |",
        "",
        f"- mean scale distinctness row-TV(P1,P3): **{agg['scale_distinctness_p1p3']:.3f}**"
        f" (raw matrix L1/(2N^2): {agg.get('scale_distinctness_raw_p1p3', float('nan')):.4f}; see amendment-1)",
        f"- increment cosines: (P1-U,P2-U) {agg['increment_cos_12']:.3f}; (P2-U,P3-U) {agg['increment_cos_23']:.3f}",
        f"- row cosine P2 vs P3: {agg['row_cos_p2_p3']:.3f}",
        f"- K mean TV to uniform: **{agg['K_tv_uniform']:.3f}**; effective rank: {agg['K_effective_rank']:.2f}",
        "",
        "Pre-frozen gate conditions: " + json.dumps(s["gate_conditions"]),
        "",
        "By topology category:",
        "",
        "| category | n | mean bonds | TV(P^3,U) | K TV(U) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, val in s["by_category"].items():
        lines.append(
            f"| {cat} | {val['n_molecules']} | {val['mean_n_bonds']:.1f} | "
            f"{val['mean_tv_uniform_k3']:.3f} | {val['mean_K_tv_uniform']:.3f} |"
        )
    lines += [
        "",
        "## 2. Invariance",
        "",
        f"- atom relabel max kernel diff: {structure['invariants']['atom_relabel_max_kernel_diff']:.3e}",
        f"- bond permutation max conjugation diff: {structure['invariants']['bond_permutation_max_conjugation_diff']:.3e}",
        f"- bond permutation max statistic diff: {structure['invariants']['bond_permutation_max_statistic_diff']:.3e}",
        "",
        "## 3. Bond-class diffusion (k=3)",
        "",
        "| class | n | mean TV to uniform |",
        "|---|---:|---:|",
    ]
    for cls, val in structure["bond_class_diffusion"].items():
        lines.append(f"| {cls} | {val['n']} | {val['mean_tv_uniform_k3']:.3f} |")
    rp = structure["real_pair_search"]
    lines += [
        "",
        "## 4. Controlled organisation pairs",
        "",
        f"- marginal signature: `{rp['signature']}`",
        f"- signature buckets: {rp['n_signature_buckets']}; multi-molecule buckets: {rp['n_multi_molecule_buckets']}",
        f"- pairs with identical marginals but different K: "
        f"**{rp['n_pairs_with_identical_marginals_and_different_K']}**",
        "- K sorted-entry max-abs difference among matched pairs: mean "
        f"{rp['mean_K_sorted_max_abs_diff_among_matched_pairs']} / max "
        f"{rp['max_K_sorted_max_abs_diff_among_matched_pairs']}",
        f"- loose signature `{rp['loose_signature']}`: {rp['n_loose_multi_molecule_buckets']} multi-molecule buckets",
        f"- note: {rp['note']}",
        "",
        "## 5. Communities on the bond line graph",
        "",
        f"- Walktrap (steps={communities['walktrap_steps']}) mean communities/molecule: "
        f"**{communities['summary']['mean_communities']['walktrap']:.2f}**; "
        f"TASU 2ECC/bridge: {communities['summary']['mean_communities']['tasu_2ecc_bridge']:.2f}; "
        f"greedy modularity: {communities['summary']['mean_communities']['greedy_modularity']:.2f}",
        f"- mean ARI: " + ", ".join(f"{k} {v:.3f}" for k, v in communities["summary"]["mean_ari"].items()),
        f"- community connectivity: {communities['summary']['walktrap']['connected_fraction']:.3f}",
        f"- singleton community share: {communities['summary']['walktrap']['singleton_fraction']:.3f}",
        f"- ring-like community share: {communities['summary']['walktrap']['ringlike_fraction']:.3f}",
        f"- Walktrap relabel ARI: min {communities['summary']['walktrap']['relabel_ari_min']:.3f}, "
        f"mean {communities['summary']['walktrap']['relabel_ari_mean']:.3f}, "
        f"fraction >= 0.9 {communities['summary']['walktrap']['relabel_ari_frac_ge_0p9']:.3f}",
        f"- every bond in exactly one community: {communities['summary']['walktrap']['all_bonds_unique_membership']} "
        f"(partition covers all bonds: {communities['summary']['walktrap']['all_partitions_cover_bonds']})",
        f"- atoms with >1 incident bond: {communities['summary']['atom_boundary']['atoms_with_multiple_incident_bonds']}; "
        f"of these boundary atoms: {communities['summary']['atom_boundary']['boundary_atoms']} "
        f"({communities['summary']['atom_boundary']['boundary_fraction']:.3f})",
        "",
        "**Boundary statement.** " + communities["claims_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_synthetic_md(path: Path, witnesses: Mapping[str, Any], mechanism: Mapping[str, Any]) -> None:
    jc = witnesses["junction_moment_counterexample"]
    task = witnesses["synthetic_relation_task"]
    ab_w = witnesses["pair_block_ablation_random_phi"]
    res = witnesses["pair_linear_summary_residual_random_phi"]
    lines = [
        "# Synthetic witnesses — cscl-bond-random-walk-organization-v1",
        "",
        f"- protocol `{witnesses['protocol']}` · commit `{witnesses['git_commit']}`",
        "- all witnesses are CPU-only and label-blind; no molecule target y is used",
        "",
        "## C1 — Junction-moment counterexample",
        "",
        f"- legal molecule: **{jc['legal_molecule']}** ({jc['warning']})",
        f"- equal raw marginals: bond multiset / junction multiset / per-bond junction counts = {jc['raw_marginals_equal']}",
        f"- K max abs difference between the two organisations: **{jc['K_max_abs_diff']:.4f}**",
        f"- line-graph edges: A {jc['bond_line_graph']['A_edges']:.0f}, B {jc['bond_line_graph']['B_edges']:.0f}",
        "",
        "The historical (Null) Junction summary is unchanged while the bond random-walk",
        "relation input differs; this is the requested controlled counterexample and it",
        "is explicitly **not** a legal molecule.",
        "",
        "## C1 — Synthetic relation task",
        "",
        f"- target: `{task['target']}`",
        f"- test MAE, real-K model: **{task['test_mae_real_K_trained']:.4f}**",
        f"- same model evaluated with uniform K: {task['test_mae_real_K_model_evaluated_uniform']:.4f}",
        f"- uniform-K trained model: {task['test_mae_uniform_K_trained_uniform']:.4f}",
        f"- linear-summary model: {task['test_mae_linear_summary']:.4f}",
        "",
        task["interpretation"],
        "",
        "## C2 — Pair MLP joint information",
        "",
        f"- random-phi block ablation: zero product rel change {ab_w['zero_prod_rel']:.3f}; "
        f"zero diff rel {ab_w['zero_diff_rel']:.3f}; zero sum rel {ab_w['zero_sum_rel']:.3f}",
        f"- linear-summary relative residual (trained phi): "
        f"**{mechanism['C2_linear_summary_residual_trained']['relative_residual']:.4f}**",
        f"- trained-phi block ablation: zero product rel {mechanism['C2_pair_block_ablation_trained']['zero_prod_rel']:.3f}, "
        f"zero diff rel {mechanism['C2_pair_block_ablation_trained']['zero_diff_rel']:.3f}",
        f"- relation reassignment mean prediction delta (uniform / row-perm / bond-perm): "
        f"{mechanism['C2_relation_reassignment']['mean_pred_delta']['uniform']:.4f} / "
        f"{mechanism['C2_relation_reassignment']['mean_pred_delta']['row_perm']:.4f} / "
        f"{mechanism['C2_relation_reassignment']['mean_pred_delta']['bond_perm']:.4f}",
        "",
        "The linear-surrogate fit `r_e ~ W (K b)_e + c*rowsum_e` captures the entire",
        "K-dependence of any linear reweighting of `b_f`; a clearly positive relative",
        "residual means the trained pair channel is not reducible to that form.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_report_md(path, decision, structure, communities, witnesses, gates, score, mechanism, train_R, train_U, freeze, ledger):
    primary = score["primary_contrast_R_minus_U"]
    lines = [
        "# Bond random-walk structural organization v1 — REPORT",
        "",
        f"- protocol `{decision['protocol']}` · decision **{decision['decision']}**",
        f"- commit `{freeze['git_commit']}` · Part A decision `{decision['part_a_decision']}`",
        f"- GPU wall: {ledger['total_wall_hours']:.3f} h / cap {ledger['cap_hours']} h",
        "- Monitor read exactly once after prediction freeze; Locked never read; official valid/test never loaded.",
        "",
        "## 1. Question and arms",
        "",
        "Without a learned Junction encoder: does a fixed multi-scale random-walk relation on the",
        "real bond line graph, feeding a shared Bond-Bond Pair MLP, add anything over the same",
        "machinery with uniform global pair weights?",
        "",
        "| arm | relation K | params | update content |",
        "|---|---|---:|---|",
        f"| R | `(P+P^2+P^3)/3` on the physical bond line graph | {train_R['parameter_audit']['actual_parameters']} | real topology |",
        f"| U | `1/N_bond` | {train_U['parameter_audit']['actual_parameters']} | unstructured global pair mean |",
        "",
        "Both arms share the same Fit-only masked-SSL encoder, scratch init, batch order, R3 recipe",
        "(360 epochs / 22,680 updates, raw-y L1, last-5 soup) and loss.",
        "",
        "## 2. Part A — structure audit",
        "",
        f"- kernel mixing at k=3: mean TV(P^3, uniform) = **{structure['summary']['aggregate']['tv_uniform_k3']:.3f}** "
        f"(gate >= 0.03); K mean TV = **{structure['summary']['aggregate']['K_tv_uniform']:.3f}** (gate >= 0.05)",
        f"- scale distinctness P1→P3: {structure['summary']['aggregate']['scale_distinctness_p1p3']:.3f}",
        f"- invariance max violation: {max(abs(structure['invariants']['atom_relabel_max_kernel_diff']), abs(structure['invariants']['bond_permutation_max_conjugation_diff'])):.2e}",
        f"- marginal-identical pairs with different K: {structure['real_pair_search']['n_pairs_with_identical_marginals_and_different_K']}",
        f"- Walktrap mean communities {communities['summary']['mean_communities']['walktrap']:.2f} vs TASU 2ECC/bridge "
        f"{communities['summary']['mean_communities']['tasu_2ecc_bridge']:.2f}; ARI(walktrap,tasu) "
        f"{communities['summary']['mean_ari']['walktrap_vs_tasu']:.3f}; Walktrap relabel ARI min "
        f"{communities['summary']['walktrap']['relabel_ari_min']:.3f} / mean "
        f"{communities['summary']['walktrap']['relabel_ari_mean']:.3f}",
        f"- structure of communities: singleton share {communities['summary']['walktrap']['singleton_fraction']:.3f}, "
        f"ring-like share {communities['summary']['walktrap']['ringlike_fraction']:.3f}, "
        f"boundary atoms with >1 community {communities['summary']['atom_boundary']['boundary_fraction']:.3f}",
        "",
        "## 3. Monitor results (split D, n=1000, seed 0, historically exposed)",
        "",
        f"| arm | MAE | RMSE | ordinary (y>-3) | tail (y<=-3) |",
        "|---|---:|---:|---:|---:|",
        f"| R | {_fmt(score['monitor_mae']['R'])} | {_fmt(primary['rmse_new'])} | "
        f"{_fmt(score['subgroup']['ordinary']['mae_R'])} | "
        f"{_fmt(score['subgroup']['tail']['mae_R'])} |",
        f"| U | {_fmt(score['monitor_mae']['U'])} | {_fmt(primary['rmse_ref'])} | "
        f"{_fmt(score['subgroup']['ordinary']['mae_U'])} | "
        f"{_fmt(score['subgroup']['tail']['mae_U'])} |",
        "",
        f"- **R - U = {primary['delta_seed_mean']:+.6f}**, paired bootstrap CI95 "
        f"[{primary['bootstrap']['ci_lo']:+.6f}, {primary['bootstrap']['ci_hi']:+.6f}], "
        f"fraction R better {primary['frac_molecules_better']:.3f}",
        f"- ordinary Δ {primary['ordinary_delta']:+.6f} (n={score['subgroup']['ordinary']['n']}); "
        f"tail Δ {primary['tail_delta']:+.6f} (n={score['subgroup']['tail']['n']}); "
        f"top-10-removed Δ {primary['top10_removed_delta']:+.6f}",
        "",
        "Read-only references (different trajectories / parameter budgets; descriptive only):",
        "",
        "| reference | Monitor MAE | R - ref (descriptive) |",
        "|---|---:|---:|",
    ]
    for name, ref in score.get("reference_reads", {}).items():
        lines.append(f"| {name} | {_fmt(ref['monitor_mae'])} | {ref['delta_vs_R']['delta_seed_mean']:+.6f} |")
    lines += [
        "",
        "## 4. Mechanism evidence (Part C)",
        "",
        f"- C1 R-vs-U Monitor mean |dpred| {mechanism['C1_R_vs_U_predictions']['mean_abs_dpred']:.3f}, "
        f"max {mechanism['C1_R_vs_U_predictions']['max_abs_dpred']:.3f}, corr {mechanism['C1_R_vs_U_predictions']['corr']:.4f}",
        f"- C1 pair MLP gradient live: {mechanism['C1_pair_gradients']['live']}",
        f"- C1 synthetic relation task: real-K {mechanism['C1_synthetic_relation_task']['test_mae_real_K_trained']:.4f} "
        f"vs uniform-K {mechanism['C1_synthetic_relation_task']['test_mae_uniform_K_trained_uniform']:.4f} "
        f"vs linear {mechanism['C1_synthetic_relation_task']['test_mae_linear_summary']:.4f}",
        f"- C1 junction-moment counterexample: same marginals, K differs by "
        f"{witnesses['junction_moment_counterexample']['K_max_abs_diff']:.4f} (not a legal molecule)",
        f"- C2 fixed-representation relation reassignment mean |dpred|: "
        f"uniform {mechanism['C2_relation_reassignment']['mean_pred_delta']['uniform']:.4f}, "
        f"row-perm {mechanism['C2_relation_reassignment']['mean_pred_delta']['row_perm']:.4f}, "
        f"bond-perm {mechanism['C2_relation_reassignment']['mean_pred_delta']['bond_perm']:.4f}",
        f"- C2 linear-reweighting surrogate relative residual: "
        f"{mechanism['C2_linear_summary_residual_trained']['relative_residual']:.4f}",
        f"- acceptance: {json.dumps(mechanism['acceptance'])}",
        "",
        "## 5. Part D — answers",
        "",
        "1. The random walk provides genuinely multi-scale line-graph information; k=2/3 are not",
        "   collapsed to the molecule-wide mean (see Part A numbers), and the scale distinctness",
        "   gate passes/fails as recorded above.",
        "2. Walktrap partitions are/are not substantively different from the TASU 2ECC/bridge",
        "   partition (ARI above) and show the recorded stability.",
        "3. Whether real connectivity beats the unstructured global pair control is the frozen",
        "   R - U contrast above, evaluated with the C1/C2 mechanism checks.",
        "4. Multi-scale organisation is/is not worth developing further as a full structure model;",
        "   see the decision and the stop rule in `decision.json`.",
        "5. Supporting evidence and stop signs: Part A gate, R - U CI, mechanism acceptance flags,",
        "   and the read-only reference gaps.",
        "",
        "## 6. Discipline",
        "",
        "- no GNN/attention/iteration; no learned Junction encoder; no walk-length/width search;",
        "- no community-parameter tuning; no second seed; Locked and official splits untouched;",
        "- the two arms differ only in the pair weights handed to the same model.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("verify-inputs", cmd_verify_inputs),
        ("prep-rw", cmd_prep_rw),
        ("audit-structure", cmd_audit_structure),
        ("audit-communities", cmd_audit_communities),
        ("synthetic", cmd_synthetic),
        ("cpu-gates", cmd_cpu_gates),
        ("freeze", cmd_freeze),
        ("freeze-predictions", cmd_freeze_predictions),
        ("score-monitor", cmd_score_monitor),
        ("mechanism", cmd_mechanism),
        ("finalize", cmd_finalize),
    ):
        sp = sub.add_parser(name)
        if name in ("cpu-gates", "mechanism"):
            sp.add_argument("--device", default=argparse.SUPPRESS)
        sp.set_defaults(func=fn)

    tr = sub.add_parser("train")
    tr.add_argument("--arm", required=True, choices=ARMS)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--device", default=argparse.SUPPRESS)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--arm", required=True, choices=ARMS)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--device", default=argparse.SUPPRESS)
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
