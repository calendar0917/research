"""cscl-bjoc-signature-conditional-value-v1 — driver (CPU only, no labels, frozen model).

Commands
--------
verify-inputs   hash and presence check of every frozen input
freeze          write protocol_freeze.json (git commit + config/code hashes)
prepare         recompute validation Junction signatures from objects_valid.pt
                and cross-check the stored sig_code mapping (label-free)
residuals       molecule-capped Arm B / Arm C residual tables (discovery only)
pseudo          16 conditional pseudo-control table replicates (discovery only)
crossfit        molecule-disjoint two-fold discovery check (diagnostic A7)
analyze         first validation comparison: A/B/C contrasts, pseudo control,
                concentration, coverage, gates and the frozen four-label decision
finalize        artifact hashes, historical-integrity proof, freeze re-check

Discipline: no prediction labels, no official test, no Locked split, no new
training, no model inference, no GPU, no post-hoc search.  Historical artifacts
are read-only.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_signature_conditional_value_v1 as bcv  # noqa: E402
import bjoc_signature_transfer_v1 as bst  # noqa: E402
import bjoc_source_ownership_v1 as own  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = bcv.PROTOCOL_ID
OUT = bcv.OUT
CONFIG_PATH = bcv.CONFIG_PATH
FREEZE = OUT / "protocol_freeze.json"
INPUT_AUDIT = OUT / "input_audit.json"
VALIDATION_SIGNATURES = OUT / "validation_signatures.npz"
VALIDATION_PREPARE = OUT / "validation_prepare.json"
DISCOVERY_JSON = OUT / "discovery_junctions.npz"
RESIDUAL_TABLES = OUT / "residual_tables.json"
PSEUDO_TABLES = OUT / "pseudo_tables.npz"
PSEUDO_META = OUT / "pseudo_tables.json"
CROSSFIT = OUT / "crossfit.json"
CROSSFIT_ROWS = OUT / "crossfit_row_metrics.npz"
ANALYSIS = OUT / "validation_analysis.json"
METRICS = OUT / "metrics.json"
DECISION = OUT / "final_decision.json"
ROW_METRICS = OUT / "row_metrics_v2.npz"
REPRODUCTION = OUT / "reproduction_check.json"
HISTORICAL_INTEGRITY = OUT / "historical_integrity.json"
FINALIZE = OUT / "finalize.json"

PRIOR_RAW = {
    0: bcv.PRIOR / "validation_raw_seed0.npz",
    1: bcv.PRIOR / "validation_raw_seed1.npz",
}
PRIOR_ROW_METRICS = bcv.PRIOR / "row_metrics.npz"
PRIOR_TABLE = bcv.PRIOR / "discovery_table.json"
PRIOR_STRUCTURE = bcv.PRIOR / "structure_overlap.json"
PRIOR_DECISION = bcv.PRIOR / "final_decision.json"
PRIOR_FINALIZE = bcv.PRIOR / "finalize.json"
PHASE0 = REPO_ROOT / "tracks/structure-contribution/results/bjoc_source_ownership_v1/phase0_results.json"
SAMPLE = REPO_ROOT / "tracks/structure-contribution/results/bjoc_source_attribution_feasibility_v1/sample_manifest.json"
OBJ_VALID = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/objects_valid.pt"
OBJ_TRAIN = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/objects_train10k.pt"
PREP_TRAIN = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/prep_train10k.json"
SOUPS = {
    0: REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1/soup_R3_seed0.pt",
    1: REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1/soup_R3_seed1.pt",
}

KS = (1, 2, 4, 8)
PRIMARY_K = 4
N_VALID = 1000


def log(msg: str) -> None:
    print(f"[bscv] {msg}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def code_paths() -> list[Path]:
    return [
        CONFIG_PATH,
        _CODE_DIR / "bjoc_signature_conditional_value_v1.py",
        Path(__file__),
        REPO_ROOT / "tracks/structure-contribution/tests/test_bjoc_signature_conditional_value_v1.py",
    ]


def code_hashes() -> dict[str, str]:
    return {str(p.relative_to(REPO_ROOT)): bcv.file_sha256(p) for p in code_paths()}


# ---------------------------------------------------------------------------
# inputs / freeze
# ---------------------------------------------------------------------------


def audit_inputs() -> dict[str, Any]:
    paths = {
        "prior_raw_seed0": PRIOR_RAW[0],
        "prior_raw_seed1": PRIOR_RAW[1],
        "prior_row_metrics": PRIOR_ROW_METRICS,
        "prior_discovery_table": PRIOR_TABLE,
        "prior_structure_overlap": PRIOR_STRUCTURE,
        "prior_final_decision": PRIOR_DECISION,
        "prior_finalize": PRIOR_FINALIZE,
        "phase0_results": PHASE0,
        "sample_manifest": SAMPLE,
        "official_valid_objects": OBJ_VALID,
        "official_train_objects": OBJ_TRAIN,
        "prep_train10k": PREP_TRAIN,
        "r3_soup0": SOUPS[0],
        "r3_soup1": SOUPS[1],
    }
    inputs: dict[str, Any] = {
        name: {
            "path": str(p.relative_to(REPO_ROOT)),
            "exists": p.exists(),
            "sha256": bcv.file_sha256(p) if p.exists() else None,
        }
        for name, p in paths.items()
    }
    inputs["all_present"] = all(inputs[name]["exists"] for name in paths)
    if PRIOR_TABLE.exists() and SAMPLE.exists():
        table = bcv.read_json(PRIOR_TABLE)
        sample = bcv.read_json(SAMPLE)
        inputs["prior_shape"] = {
            "n_candidates": int(table["n_candidates"]),
            "n_classes": int(table["n_classes"]),
            "n_diagnostic_rows": int(len(sample["diagnostic_rows"])),
            "n_reference_rows": int(len(sample["reference_rows"])),
        }
    return inputs


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    payload = audit_inputs()
    bcv.write_json(INPUT_AUDIT, payload)
    if not payload["all_present"]:
        log("verify-inputs: MISSING INPUTS")
        return 1
    log(f"verify-inputs: all present; prior_shape={payload.get('prior_shape')}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    audit = audit_inputs()
    if not audit["all_present"]:
        raise SystemExit("cannot freeze: missing inputs")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": utc_now(),
        "config_sha256": bcv.file_sha256(CONFIG_PATH),
        "code_sha256": code_hashes(),
        "input_audit": audit,
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "discipline": bcv.DISCIPLINE,
    }
    bcv.write_json(FREEZE, payload)
    log(f"freeze: written at {payload['git_commit'][:10]} ({payload['frozen_utc']})")
    return 0


def verify_freeze() -> dict[str, Any]:
    if not FREEZE.exists():
        raise SystemExit("protocol not frozen: run freeze first")
    frozen = bcv.read_json(FREEZE)
    for path, sha in frozen["code_sha256"].items():
        if bcv.file_sha256(REPO_ROOT / path) != sha:
            raise SystemExit(f"frozen code changed: {path} (amend via a recorded protocol amendment)")
    if bcv.file_sha256(CONFIG_PATH) != frozen["config_sha256"]:
        raise SystemExit("frozen config changed")
    return frozen


# ---------------------------------------------------------------------------
# raw prior reuse
# ---------------------------------------------------------------------------


def load_raw(seed: int) -> tuple[list[dict], list[dict]]:
    z = np.load(PRIOR_RAW[seed], allow_pickle=True)
    return list(z["mols_valid"]), list(z["mols_diag"])


def load_validation_key_rows() -> list[list[str]]:
    z = np.load(VALIDATION_SIGNATURES, allow_pickle=True)
    offsets = np.asarray(z["offsets"], dtype=np.int64)
    keys = [str(k) for k in z["keys"]]
    return [keys[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def load_discovery_junction_sets() -> list[bcv.JunctionSet]:
    z = np.load(DISCOVERY_JSON, allow_pickle=True)
    offsets = np.asarray(z["offsets"], dtype=np.int64)
    seeds = np.asarray(z["seed"], dtype=np.int64)
    rows = np.asarray(z["rows"], dtype=np.int64)
    keys = [str(k) for k in z["keys"]]
    degree = np.asarray(z["degree"], dtype=np.int64)
    erc = np.asarray(z["endpoint_ring_count"], dtype=np.int64)
    d = np.asarray(z["d"], dtype=np.float64)
    q = np.asarray(z["q"], dtype=np.float64)
    r = np.asarray(z["r"], dtype=np.float64)
    out: list[bcv.JunctionSet] = []
    for i in range(len(offsets) - 1):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        out.append(
            bcv.JunctionSet(
                seed=int(seeds[i]),
                row=int(rows[i]),
                keys=tuple(keys[sl]),
                degree=degree[sl].copy(),
                endpoint_ring_count=erc[sl].copy(),
                d=d[sl].copy(),
                q=q[sl].copy(),
                r=r[sl].copy(),
            )
        )
    return out


def load_pseudo_compact() -> dict[str, Any]:
    z = np.load(PSEUDO_TABLES, allow_pickle=True)
    return {
        "sig_keys": [str(k) for k in z["sig_keys"]],
        "sig_shrunk": np.asarray(z["sig_shrunk"], dtype=np.float64),
        "ctx_keys": [str(k) for k in z["ctx_keys"]],
        "ctx_shrunk": np.asarray(z["ctx_shrunk"], dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# prepare: validation signatures (label-free, structures only)
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    verify_freeze()
    prior_table = bcv.read_json(PRIOR_TABLE)
    candidate_keys = [bcv.signature_join(c["signature"]) for c in prior_table["candidates"]]
    objs = torch.load(OBJ_VALID, map_location="cpu", weights_only=False)
    mols0, _diag0 = load_raw(0)
    if len(objs) != N_VALID or len(mols0) != N_VALID:
        raise SystemExit(f"unexpected validation size: objects={len(objs)} mols={len(mols0)}")

    flat_keys: list[str] = []
    offsets = [0]
    n_j = np.zeros(N_VALID, dtype=np.int64)
    rows = np.zeros(N_VALID, dtype=np.int64)
    checks = {
        "row_order_matches": True,
        "n_j_mismatches": 0,
        "sig_code_mismatches": 0,
        "degree_mismatches": 0,
        "endpoint_ring_count_mismatches": 0,
        "signature_unknown_fraction": 0.0,
        "n_junctions": 0,
    }
    support: Counter = Counter()
    for i in range(N_VALID):
        obj = objs[i]
        mol = mols0[i]
        if int(mol["row"]) != i:
            checks["row_order_matches"] = False
        arrs = bst.junction_arrays(obj)
        n = int(obj.junc_index.shape[0])
        if n != int(mol["n_j"]):
            checks["n_j_mismatches"] += 1
        n_j[i] = n
        rows[i] = int(mol["row"])
        stored_codes = np.asarray(mol["sig_code"], dtype=np.int64)
        stored_degree = np.asarray(mol["degree"], dtype=np.int64)
        stored_erc = np.asarray(mol["endpoint_ring_count"], dtype=np.int64)
        for j in range(n):
            key = bcv.signature_join(arrs["signatures"][j])
            flat_keys.append(key)
            support[key] += 1
            code = int(stored_codes[j])
            if code >= 0 and candidate_keys[code] != key:
                checks["sig_code_mismatches"] += 1
            if int(arrs["degree"][j]) != int(stored_degree[j]):
                checks["degree_mismatches"] += 1
            if int(arrs["endpoint_ring_count"][j]) != int(stored_erc[j]):
                checks["endpoint_ring_count_mismatches"] += 1
        offsets.append(len(flat_keys))

    checks["n_junctions"] = int(len(flat_keys))
    known = int(sum(support[k] for k in candidate_keys))
    checks["signature_unknown_fraction"] = float(1.0 - known / max(len(flat_keys), 1))
    hard_ok = (
        checks["row_order_matches"]
        and checks["n_j_mismatches"] == 0
        and checks["sig_code_mismatches"] == 0
        and checks["degree_mismatches"] == 0
        and checks["endpoint_ring_count_mismatches"] == 0
    )
    vocab = sorted(support)
    np.savez_compressed(
        VALIDATION_SIGNATURES,
        rows=rows,
        n_j=n_j,
        offsets=np.asarray(offsets, dtype=np.int64),
        keys=np.asarray(flat_keys, dtype=object),
        vocab=np.asarray(vocab, dtype=object),
        vocab_counts=np.asarray([support[k] for k in vocab], dtype=np.int64),
    )
    payload = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "discipline": bcv.DISCIPLINE,
        "checks": checks,
        "hard_ok": bool(hard_ok),
        "n_vocab": int(len(vocab)),
        "n_candidate_keys_crosschecked": int(len(candidate_keys)),
        "artifact_sha256": bcv.file_sha256(VALIDATION_SIGNATURES),
        "note": (
            "signatures recomputed from objects_valid.pt (structures only); compared with the stored "
            "sig_code mapping and the stored degree / endpoint_ring_count arrays. No model inference."
        ),
    }
    bcv.write_json(VALIDATION_PREPARE, payload)
    log(f"prepare: {checks['n_junctions']} junctions, hard_ok={hard_ok}, checks={checks}")
    if not hard_ok:
        raise SystemExit("validation signature cross-check failed; aborting before any analysis")
    return 0


# ---------------------------------------------------------------------------
# discovery residual tables + pseudo control
# ---------------------------------------------------------------------------


def _discovery_inputs() -> tuple[dict[int, list[dict]], list[bcv.JunctionSet]]:
    mols_by_seed = {s: load_raw(s)[1] for s in (0, 1)}
    return mols_by_seed, bcv.junction_sets_from_diag(mols_by_seed)


def cmd_residuals(args: argparse.Namespace) -> int:
    verify_freeze()
    prior_table = bcv.read_json(PRIOR_TABLE)
    mols_by_seed, jsets = _discovery_inputs()
    tables = bcv.build_residual_tables(jsets)

    # support sanity: molecule-capped signature supports must equal the frozen table
    prior_support = {
        bcv.signature_join(c["signature"]): int(c["support_molecules"]) for c in prior_table["candidates"]
    }
    computed_support = {
        k: int(v["n_molecules"]) for k, v in tables["sig"].items() if int(v["n_molecules"]) >= bcv.SIGNATURE_FLOOR_MOLECULES
    }
    support_ok = prior_support == computed_support

    # flat discovery junction archive
    flat_keys: list[str] = []
    offsets = [0]
    rows: list[int] = []
    seeds: list[int] = []
    degree: list[int] = []
    erc: list[int] = []
    d: list[float] = []
    q: list[float] = []
    r: list[float] = []
    for js in jsets:
        rows.append(js.row)
        seeds.append(js.seed)
        for j in range(js.n_j):
            flat_keys.append(js.keys[j])
            degree.append(int(js.degree[j]))
            erc.append(int(js.endpoint_ring_count[j]))
            d.append(float(js.d[j]))
            q.append(float(js.q[j]))
            r.append(float(js.r[j]))
        offsets.append(len(flat_keys))
    np.savez_compressed(
        DISCOVERY_JSON,
        rows=np.asarray(rows, dtype=np.int64),
        seed=np.asarray(seeds, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        keys=np.asarray(flat_keys, dtype=object),
        degree=np.asarray(degree, dtype=np.int64),
        endpoint_ring_count=np.asarray(erc, dtype=np.int64),
        d=np.asarray(d, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        r=np.asarray(r, dtype=np.float64),
    )

    sig_supports = [int(v["n_molecules"]) for v in tables["sig"].values()]
    ctx_supports = [int(v["n_molecules"]) for v in tables["ctx"].values()]
    payload = dict(tables)
    payload.update(
        {
            "created_utc": utc_now(),
            "discipline": bcv.DISCIPLINE,
            "source_raw_seed0_sha256": bcv.file_sha256(PRIOR_RAW[0]),
            "source_raw_seed1_sha256": bcv.file_sha256(PRIOR_RAW[1]),
            "source_phase0_sha256": bcv.file_sha256(PHASE0),
            "n_discovery_molecules": int(len({js.row for js in jsets})),
            "n_discovery_junction_seed_units": int(len(jsets)),
            "n_discovery_junctions": int(len(flat_keys)),
            "prior_candidate_support_match": bool(support_ok),
            "signature_support_histogram": bcv.support_histogram(sig_supports),
            "context_support_histogram": bcv.support_histogram(ctx_supports),
            "signature_coverage_fraction": float(
                sum(v["n_molecules"] for v in tables["sig"].values()) / max(len(jsets), 1)
            ),
            "discovery_artifact_sha256": bcv.file_sha256(DISCOVERY_JSON),
        }
    )
    bcv.write_json(RESIDUAL_TABLES, payload)
    log(
        f"residuals: {tables['n_signature_units']} signature units "
        f"({sum(1 for v in tables['sig'].values() if v['n_molecules'] >= bcv.SIGNATURE_FLOOR_MOLECULES)} at floor), "
        f"{tables['n_context_units']} context units; prior support match={support_ok}"
    )
    if not support_ok:
        raise SystemExit("discovery support does not reproduce the frozen candidate table")
    return 0


def cmd_pseudo(args: argparse.Namespace) -> int:
    verify_freeze()
    _mols, jsets = _discovery_inputs()
    tables = bcv.pseudo_tables(jsets)
    compact = bcv.compact_tables(tables)
    np.savez_compressed(
        PSEUDO_TABLES,
        sig_keys=np.asarray(compact["sig_keys"], dtype=object),
        sig_shrunk=compact["sig_shrunk"],
        ctx_keys=np.asarray(compact["ctx_keys"], dtype=object),
        ctx_shrunk=compact["ctx_shrunk"],
    )
    gains = []
    for t in tables:
        sig_vals = np.asarray([v["R_shrunk"] for v in t["sig"].values()], dtype=np.float64)
        gains.append(float(np.mean(np.abs(sig_vals))))
    payload = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "discipline": bcv.DISCIPLINE,
        "replicates": int(bcv.PSEUDO_REPLICATES),
        "dev_bins": int(bcv.PSEUDO_DEV_BINS),
        "rng_base": int(bcv.PSEUDO_RNG_BASE),
        "rule": (
            "within each (seed, molecule) permute r inside the DEV percentile quartile bins; "
            "signature / degree / endpoint_ring_count / d_j stay with their junction"
        ),
        "n_signature_keys": int(len(compact["sig_keys"])),
        "n_context_keys": int(len(compact["ctx_keys"])),
        "mean_abs_shrunk_signature_residual_per_replicate": gains,
        "artifact_sha256": bcv.file_sha256(PSEUDO_TABLES),
    }
    bcv.write_json(PSEUDO_META, payload)
    log(f"pseudo: {bcv.PSEUDO_REPLICATES} replicates, {len(compact['sig_keys'])} sig / {len(compact['ctx_keys'])} ctx keys")
    return 0


# ---------------------------------------------------------------------------
# discovery cross-fit (molecule-disjoint two folds)
# ---------------------------------------------------------------------------


def _fold_masks(rows: Sequence[int]) -> tuple[set[int], set[int]]:
    ordered = sorted({int(r) for r in rows})
    return set(ordered[0::2]), set(ordered[1::2])


def _crossfit_evaluate(
    held: set[int],
    train_sets: Sequence[bcv.JunctionSet],
    all_sets: Sequence[bcv.JunctionSet],
    delta_lookup: Mapping[tuple[int, int], np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score one held-out fold with tables estimated on the other fold only."""
    tables = bcv.build_residual_tables(train_sets)
    by_arm_seed: dict[str, dict[int, dict[int, float]]] = {arm: {0: {}, 1: {}} for arm in ("A", "B", "C")}
    for js in all_sets:
        if js.row not in held:
            continue
        delta = delta_lookup[(js.seed, js.row)]
        sig_b, ctx_b = bcv.bonus_arrays(tables, js.keys, js.degree, js.endpoint_ring_count)
        for arm in ("A", "B", "C"):
            score = bcv.arm_score(js.d, sig_b, ctx_b, arm)
            by_arm_seed[arm][js.seed][js.row] = bcv.topk_mean_abs(delta, score, PRIMARY_K)
    rows_sorted = sorted(held)
    merged: dict[str, np.ndarray] = {}
    out: dict[str, Any] = {
        "n_molecules": int(len(held)),
        "tables": {"n_sig": tables["n_signature_units"], "n_ctx": tables["n_context_units"]},
    }
    for arm in ("A", "B", "C"):
        a0 = np.asarray([by_arm_seed[arm][0][r] for r in rows_sorted], dtype=np.float64)
        a1 = np.asarray([by_arm_seed[arm][1][r] for r in rows_sorted], dtype=np.float64)
        merged[arm] = (a0 + a1) / 2.0
        out[f"mean_{arm}_k4"] = float(merged[arm].mean())
        out[f"per_seed_mean_{arm}_k4"] = [float(a0.mean()), float(a1.mean())]
    for name, (x, y) in {
        "B_minus_A": ("B", "A"),
        "C_minus_A": ("C", "A"),
        "C_minus_B": ("C", "B"),
    }.items():
        out[name] = bcv.contrast(merged[x], merged[y])
    return out, merged


def cmd_crossfit(args: argparse.Namespace) -> int:
    verify_freeze()
    _mols, jsets = _discovery_inputs()
    mols_by_seed = {s: load_raw(s)[1] for s in (0, 1)}
    delta_lookup = {
        (s, int(m["row"])): np.abs(np.asarray(m["delta"], dtype=np.float64)) for s in (0, 1) for m in mols_by_seed[s]
    }
    fold_a, fold_b = _fold_masks([js.row for js in jsets])
    results: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "discipline": bcv.DISCIPLINE,
        "rule": "diagnostic rows sorted ascending; even index -> fold A, odd index -> fold B",
        "fold_a_rows": sorted(fold_a),
        "fold_b_rows": sorted(fold_b),
        "folds": {},
    }
    merged_by_fold: dict[str, dict[str, np.ndarray]] = {}
    for name, held in (("A", fold_a), ("B", fold_b)):
        train_sets = [js for js in jsets if js.row not in held]
        report, merged = _crossfit_evaluate(held, train_sets, jsets, delta_lookup)
        results["folds"][name] = report
        merged_by_fold[name] = merged
        log(f"crossfit fold {name}: {report['n_molecules']} molecules")
    pooled: dict[str, np.ndarray] = {
        arm: np.concatenate([merged_by_fold["A"][arm], merged_by_fold["B"][arm]]) for arm in ("A", "B", "C")
    }
    results["pooled"] = {
        "n_molecules": int(pooled["A"].size),
        "mean_A_k4": float(pooled["A"].mean()),
        "mean_B_k4": float(pooled["B"].mean()),
        "mean_C_k4": float(pooled["C"].mean()),
        "B_minus_A": bcv.contrast(pooled["B"], pooled["A"]),
        "C_minus_A": bcv.contrast(pooled["C"], pooled["A"]),
        "C_minus_B": bcv.contrast(pooled["C"], pooled["B"]),
    }
    np.savez_compressed(CROSSFIT_ROWS, **{f"{arm}_k4": pooled[arm] for arm in pooled})
    bcv.write_json(CROSSFIT, results)
    log(
        f"crossfit pooled B-A {results['pooled']['B_minus_A']['mean']:+.5f} "
        f"[{results['pooled']['B_minus_A']['ci95_lower']:+.5f},{results['pooled']['B_minus_A']['ci95_upper']:+.5f}]"
    )
    return 0


# ---------------------------------------------------------------------------
# validation evaluation
# ---------------------------------------------------------------------------


def _evaluate_molecules(
    mols: Sequence[Mapping[str, Any]],
    seed: int,
    keys_by_row: Sequence[Sequence[str]],
    tables: Mapping[str, Any],
    pseudo_compact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    n = len(mols)
    n_pseudo = int(np.asarray(pseudo_compact["sig_shrunk"]).shape[0]) if pseudo_compact else 0
    keys = [
        "A1", "A2", "A4", "A8", "B1", "B2", "B4", "B8", "C1", "C2", "C4", "C8",
        "SIG1", "SIG2", "SIG4", "SIG8", "ORACLE1", "ORACLE2", "ORACLE4", "ORACLE8", "RANDOM4",
        "pseudoB4", "pseudoC4",
        "rho_A", "rho_B", "rho_C", "rho_SIG", "ovl_A", "ovl_B", "ovl_C", "ovl_SIG",
        "known_A4", "known_B4", "known_C4", "ctxknown_C4",
        "known_junction_sig", "known_junction_ctx",
    ]
    out: dict[str, Any] = {k: np.zeros(n, dtype=np.float64) for k in keys}
    out["pseudoB4_draws"] = np.zeros((n, n_pseudo), dtype=np.float64)
    out["pseudoC4_draws"] = np.zeros((n, n_pseudo), dtype=np.float64)
    out["driver_key"] = np.empty(n, dtype=object)
    out["driver_key"][:] = None
    sig_candidates = set(tables["sig"].keys())
    ctx_units = set(tables["ctx"].keys())

    for i, mol in enumerate(mols):
        row = int(mol["row"])
        mkeys = list(keys_by_row[row])
        degree = np.asarray(mol["degree"], dtype=np.int64)
        erc = np.asarray(mol["endpoint_ring_count"], dtype=np.int64)
        d = bcv.percentile(np.asarray(mol["dev"], dtype=np.float64))
        delta = np.abs(np.asarray(mol["delta"], dtype=np.float64))
        ig = np.asarray(mol["ig_abs"], dtype=np.float64)
        tc = np.asarray(mol["tc"], dtype=np.float64)
        sig_b, ctx_b = bcv.bonus_arrays(tables, mkeys, degree, erc)
        scores = {arm: bcv.arm_score(d, sig_b, ctx_b, arm) for arm in ("A", "B", "C")}
        scores["SIG"] = tc
        scores["ORACLE"] = ig
        for arm in ("A", "B", "C", "SIG", "ORACLE"):
            for k in KS:
                out[f"{arm}{k}"][i] = bcv.topk_mean_abs(delta, scores[arm], k)
        out["RANDOM4"][i] = float(delta[np.asarray(mol["rand"], dtype=np.int64)].mean())
        for arm in ("A", "B", "C", "SIG"):
            out[f"rho_{arm}"][i] = bst.spearman(scores[arm], ig)
            out[f"ovl_{arm}"][i] = bcv.topk_oracle_overlap(scores[arm], ig, PRIMARY_K)
        # selection-level coverage
        for arm in ("A", "B", "C"):
            sel = bcv.top_k_indices(scores[arm], PRIMARY_K)
            out[f"known_{arm}4"][i] = float(np.mean([mkeys[j] in sig_candidates for j in sel]))
        selc = bcv.top_k_indices(scores["C"], PRIMARY_K)
        out["ctxknown_C4"][i] = float(
            np.mean(
                [
                    bcv.context_key(mkeys[j], int(degree[j]), int(erc[j])) in ctx_units
                    for j in selc
                ]
            )
        )
        out["known_junction_sig"][i] = float(np.mean([k in sig_candidates for k in mkeys]))
        out["known_junction_ctx"][i] = float(
            np.mean([bcv.context_key(k, int(degree[j]), int(erc[j])) in ctx_units for j, k in enumerate(mkeys)])
        )
        # driver class of the B selection (concise concentration diagnostic)
        drivers = [
            mkeys[j] for j in bcv.top_k_indices(scores["B"], PRIMARY_K) if mkeys[j] in sig_candidates
        ]
        if drivers:
            out["driver_key"][i] = max(drivers, key=lambda k: float(tables["sig"][k]["R_shrunk"]))
        # pseudo control replicates
        for p in range(n_pseudo):
            t = bcv.tables_from_compact(pseudo_compact, p)
            s_p, c_p = bcv.bonus_arrays(t, mkeys, degree, erc)
            out["pseudoB4_draws"][i, p] = bcv.topk_mean_abs(delta, bcv.arm_score(d, s_p, c_p, "B"), PRIMARY_K)
            out["pseudoC4_draws"][i, p] = bcv.topk_mean_abs(delta, bcv.arm_score(d, s_p, c_p, "C"), PRIMARY_K)
        out["pseudoB4"][i] = float(out["pseudoB4_draws"][i].mean())
        out["pseudoC4"][i] = float(out["pseudoC4_draws"][i].mean())
    return out


def _merge_seeds(ev0: Mapping[str, Any], ev1: Mapping[str, Any], key: str) -> np.ndarray:
    return (np.asarray(ev0[key], dtype=np.float64) + np.asarray(ev1[key], dtype=np.float64)) / 2.0


def _subset_report(mask: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"n_molecules": int(mask.sum())}
    if mask.any():
        out["B_minus_A"] = bcv.contrast(b[mask], a[mask])
        out["C_minus_A"] = bcv.contrast(c[mask], a[mask])
        out["C_minus_B"] = bcv.contrast(c[mask], b[mask])
    return out


def cmd_analyze(args: argparse.Namespace) -> int:
    verify_freeze()
    tables = bcv.read_json(RESIDUAL_TABLES)
    pseudo_compact = load_pseudo_compact()
    keys_by_row = load_validation_key_rows()
    mols0, _d0 = load_raw(0)
    mols1, _d1 = load_raw(1)
    if len(mols0) != N_VALID or len(mols1) != N_VALID:
        raise SystemExit("unexpected validation molecule count")
    ev0 = _evaluate_molecules(mols0, 0, keys_by_row, tables, pseudo_compact)
    ev1 = _evaluate_molecules(mols1, 1, keys_by_row, tables, pseudo_compact)
    log("analyze: per-molecule evaluation done")

    # ---- reproduction gate R: four published metrics recomputed exactly ----
    prior = np.load(PRIOR_ROW_METRICS)
    repro: dict[str, Any] = {"tolerance": 1e-9, "checks": {}, "max_abs_diff": 0.0}
    for seed, ev in ((0, ev0), (1, ev1)):
        for method in ("DEV", "SIG", "ORACLE"):
            arm = {"DEV": "A", "SIG": "SIG", "ORACLE": "ORACLE"}[method]
            got = np.asarray(ev[f"{arm}{PRIMARY_K}"], dtype=np.float64)
            want = np.asarray(prior[f"seed{seed}_{method}_k4"], dtype=np.float64)
            d = float(np.max(np.abs(got - want))) if got.size else float("nan")
            repro["checks"][f"seed{seed}_{method}_k4"] = d
            repro["max_abs_diff"] = max(repro["max_abs_diff"], d)
        got = np.asarray(ev["RANDOM4"], dtype=np.float64)
        want = np.asarray(prior[f"seed{seed}_RANDOM_k4"], dtype=np.float64)
        d = float(np.max(np.abs(got - want)))
        repro["checks"][f"seed{seed}_RANDOM_k4"] = d
        repro["max_abs_diff"] = max(repro["max_abs_diff"], d)
    for method in ("DEV", "SIG", "ORACLE"):
        got = _merge_seeds(ev0, ev1, {"DEV": "A4", "SIG": "SIG4", "ORACLE": "ORACLE4"}[method])
        want = np.asarray(prior[f"pooled_{method}_k4"], dtype=np.float64)
        d = float(np.max(np.abs(got - want)))
        repro["checks"][f"pooled_{method}_k4"] = d
        repro["max_abs_diff"] = max(repro["max_abs_diff"], d)
    got = _merge_seeds(ev0, ev1, "RANDOM4")
    want = np.asarray(prior["pooled_RANDOM_k4"], dtype=np.float64)
    d = float(np.max(np.abs(got - want)))
    repro["checks"]["pooled_RANDOM_k4"] = d
    repro["max_abs_diff"] = max(repro["max_abs_diff"], d)
    repro["hard_ok"] = bool(repro["max_abs_diff"] <= repro["tolerance"])
    bcv.write_json(REPRODUCTION, repro)
    log(f"analyze: reproduction max|diff|={repro['max_abs_diff']:.3e} hard_ok={repro['hard_ok']}")
    if not repro["hard_ok"]:
        raise SystemExit("reproduction gate failed: stored arrays do not reproduce the prior round")

    # ---- merged per-molecule arrays ----
    A = _merge_seeds(ev0, ev1, "A4")
    B = _merge_seeds(ev0, ev1, "B4")
    C = _merge_seeds(ev0, ev1, "C4")
    O = _merge_seeds(ev0, ev1, "ORACLE4")
    S = _merge_seeds(ev0, ev1, "SIG4")
    R = _merge_seeds(ev0, ev1, "RANDOM4")
    pB = _merge_seeds(ev0, ev1, "pseudoB4")
    pC = _merge_seeds(ev0, ev1, "pseudoC4")

    contrasts = {
        "B_minus_A": bcv.contrast(B, A),
        "C_minus_A": bcv.contrast(C, A),
        "C_minus_B": bcv.contrast(C, B),
        "SIG_minus_A": bcv.contrast(S, A),
        "B_minus_pseudoB": bcv.contrast(B, pB),
        "C_minus_pseudoC": bcv.contrast(C, pC),
        "context_adv_over_pseudo": bcv.contrast(C - B, pC - pB),
    }
    per_seed = {}
    for seed, ev in ((0, ev0), (1, ev1)):
        per_seed[str(seed)] = {
            "mean_A_k4": float(np.asarray(ev["A4"]).mean()),
            "mean_B_k4": float(np.asarray(ev["B4"]).mean()),
            "mean_C_k4": float(np.asarray(ev["C4"]).mean()),
            "B_minus_A": bcv.contrast(np.asarray(ev["B4"]), np.asarray(ev["A4"])),
            "C_minus_B": bcv.contrast(np.asarray(ev["C4"]), np.asarray(ev["B4"])),
        }

    # ---- subsets ----
    structure = bcv.read_json(PRIOR_STRUCTURE)
    entries = structure["valid"]
    mask_unseen = np.asarray([not bool(entries[i]["scaffold_in_discovery"]) for i in range(N_VALID)])
    mask_scaffold_not_train = np.asarray([not bool(entries[i]["scaffold_in_train"]) for i in range(N_VALID)])
    mask_no_dup = np.asarray([not bool(entries[i]["dup_train"]) for i in range(N_VALID)])
    n_junc = np.asarray([int(entries[i]["n_junctions"]) for i in range(N_VALID)], dtype=np.float64)
    q1, q2 = np.quantile(n_junc, [1 / 3, 2 / 3])
    size_masks = {
        "size_low": n_junc <= q1,
        "size_mid": (n_junc > q1) & (n_junc <= q2),
        "size_high": n_junc > q2,
    }
    subsets = {
        "all": _subset_report(np.ones(N_VALID, dtype=bool), A, B, C),
        "scaffold_unseen_discovery": _subset_report(mask_unseen, A, B, C),
        "scaffold_seen_discovery": _subset_report(~mask_unseen, A, B, C),
        "scaffold_not_in_train10k": _subset_report(mask_scaffold_not_train, A, B, C),
        "no_exact_dup_train": _subset_report(mask_no_dup, A, B, C),
        **{name: _subset_report(mask, A, B, C) for name, mask in size_masks.items()},
    }

    # ---- k sensitivity and ranking quality ----
    def _kgrid(ev, key_prefix: str) -> dict[str, float]:
        return {f"k{k}": float(np.asarray(ev[f"{key_prefix}{k}"]).mean()) for k in KS}

    k_sensitivity = {}
    for tag, ev in (("seed0", ev0), ("seed1", ev1)):
        for arm in ("A", "B", "C", "SIG", "ORACLE"):
            k_sensitivity[f"{tag}_{arm}"] = _kgrid(ev, arm)
    k_sensitivity["pooled"] = {
        arm: {f"k{k}": float(np.mean([np.asarray(ev0[f"{arm}{k}"]).mean(), np.asarray(ev1[f"{arm}{k}"]).mean()])) for k in KS}
        for arm in ("A", "B", "C", "SIG", "ORACLE")
    }
    ranking_quality = {}
    for tag, ev in (("seed0", ev0), ("seed1", ev1)):
        for arm in ("A", "B", "C", "SIG"):
            ranking_quality[f"{tag}_{arm}"] = {
                "mean_spearman_vs_abs_ig": float(np.nanmean(np.asarray(ev[f"rho_{arm}"]))),
                "mean_top4_overlap_oracle": float(np.nanmean(np.asarray(ev[f"ovl_{arm}"]))),
            }

    # ---- concentration of the B gain ----
    gains = B - A
    drivers = []
    for i in range(N_VALID):
        d0 = ev0["driver_key"][i]
        d1 = ev1["driver_key"][i]
        cand = [k for k in (d0, d1) if k is not None]
        if cand:
            drivers.append(max(cand, key=lambda k: float(tables["sig"][k]["R_shrunk"])))
        else:
            drivers.append(None)
    pos = np.maximum(gains, 0.0)
    total_pos = float(pos.sum())
    by_driver: dict[str, float] = defaultdict(float)
    for i, drv in enumerate(drivers):
        by_driver[str(drv)] += float(pos[i])
    shares = {k: (v / total_pos if total_pos > 0 else float("nan")) for k, v in by_driver.items()}
    top_driver = max(shares, key=lambda k: (shares[k] if np.isfinite(shares[k]) else -1.0)) if shares else None
    excl = np.asarray([i for i in range(N_VALID) if str(drivers[i]) != str(top_driver)], dtype=np.int64)
    concentration = {
        "total_positive_gain": total_pos,
        "n_driver_classes": int(len(by_driver)),
        "top_driver": top_driver,
        "top_driver_positive_gain_share": float(shares.get(top_driver, float("nan"))) if top_driver else float("nan"),
        "mean_gain_excluding_top_driver": float(gains[excl].mean()) if excl.size else float("nan"),
        "n_molecules_excluded": int(N_VALID - excl.size),
        "top5_driver_shares": [
            {"driver": k, "share": float(shares[k]), "positive_gain": float(by_driver[k])}
            for k in sorted(shares, key=lambda k: -shares[k])[:5]
        ],
    }

    # ---- coverage / fallback ----
    discovery_keys = set(tables["sig"].keys())
    validation_prepare = bcv.read_json(VALIDATION_PREPARE)
    z = np.load(VALIDATION_SIGNATURES, allow_pickle=True)
    vocab_list = [str(k) for k in z["vocab"]]
    vocab_counts = np.asarray(z["vocab_counts"], dtype=np.int64)
    valid_vocab_keys = set(vocab_list)
    disc_jsets = load_discovery_junction_sets()
    all_discovery_keys = {k for js in disc_jsets for k in js.keys}
    n_valid_junctions = int(np.asarray(z["keys"], dtype=object).size)
    known_j = int(sum(int(c) for k, c in zip(vocab_list, vocab_counts) if k in discovery_keys))
    coverage = {
        "n_validation_junctions": n_valid_junctions,
        "signature_units_total": int(tables["n_signature_units"]),
        "signature_units_at_floor": int(sum(1 for v in tables["sig"].values() if v["n_molecules"] >= bcv.SIGNATURE_FLOOR_MOLECULES)),
        "context_units_total": int(tables["n_context_units"]),
        "context_units_at_floor": int(sum(1 for v in tables["ctx"].values() if v["n_molecules"] >= bcv.CONTEXT_FLOOR_MOLECULES)),
        "junction_fraction_signature_table": float(known_j / max(n_valid_junctions, 1)),
        "junction_fraction_below_floor": float(
            sum(1 for k in valid_vocab_keys if k in all_discovery_keys and k not in discovery_keys)
            / max(len(valid_vocab_keys), 1)
        ),
        "junction_fraction_never_seen": float(
            sum(1 for k in valid_vocab_keys if k not in all_discovery_keys) / max(len(valid_vocab_keys), 1)
        ),
        "signature_support_histogram": tables.get("signature_support_histogram"),
        "context_support_histogram": tables.get("context_support_histogram"),
        "mean_known_fraction_selected_B4": float(np.mean(_merge_seeds(ev0, ev1, "known_B4"))),
        "mean_known_fraction_selected_C4": float(np.mean(_merge_seeds(ev0, ev1, "known_C4"))),
        "mean_ctxknown_fraction_selected_C4": float(np.mean(_merge_seeds(ev0, ev1, "ctxknown_C4"))),
        "mean_known_fraction_junctions": float(np.mean(_merge_seeds(ev0, ev1, "known_junction_sig"))),
        "mean_ctxknown_fraction_junctions": float(np.mean(_merge_seeds(ev0, ev1, "known_junction_ctx"))),
        "never_seen_vocab_size": int(len(valid_vocab_keys - all_discovery_keys)),
        "below_floor_vocab_size": int(len([k for k in valid_vocab_keys if k in all_discovery_keys and k not in discovery_keys])),
        "validation_prepare_sha256": validation_prepare["artifact_sha256"],
    }

    # ---- crossfit and pseudo draws ----
    crossfit = bcv.read_json(CROSSFIT)
    pseudo_summary: dict[str, Any] = {
        "replicates": int(bcv.PSEUDO_REPLICATES),
        "mean_pseudoB_minus_A_over_replicates": [],
        "mean_pseudoC_minus_A_over_replicates": [],
        "mean_ctx_increment_over_replicates": [],
    }
    for p in range(int(bcv.PSEUDO_REPLICATES)):
        pb = (np.asarray(ev0["pseudoB4_draws"])[:, p] + np.asarray(ev1["pseudoB4_draws"])[:, p]) / 2.0
        pc = (np.asarray(ev0["pseudoC4_draws"])[:, p] + np.asarray(ev1["pseudoC4_draws"])[:, p]) / 2.0
        pseudo_summary["mean_pseudoB_minus_A_over_replicates"].append(float((pb - A).mean()))
        pseudo_summary["mean_pseudoC_minus_A_over_replicates"].append(float((pc - A).mean()))
        pseudo_summary["mean_ctx_increment_over_replicates"].append(float(((C - B) - (pc - pb)).mean()))

    # ---- oracle gap / recovery ----
    mean_A, mean_B, mean_C, mean_O, mean_S, mean_R = (float(x.mean()) for x in (A, B, C, O, S, R))
    gap_A = mean_O - mean_A
    gap_B = mean_O - mean_B
    recovery_B = float(contrasts["B_minus_A"]["mean"] / gap_A) if gap_A > 0 else float("nan")
    recovery_CB = float(contrasts["C_minus_B"]["mean"] / gap_B) if gap_B > 0 else float("nan")
    oracle_fraction_SIG = float((mean_S - mean_R) / max(mean_O - mean_R, 1e-12))
    oracle_fraction_B = float((mean_B - mean_R) / max(mean_O - mean_R, 1e-12))

    # ---- gates ----
    g = {}
    g["A1"] = bool(contrasts["B_minus_A"]["mean"] > 0)
    g["A2"] = bool(contrasts["B_minus_A"]["ci95_lower"] > 0)
    g["A3"] = bool(
        per_seed["0"]["B_minus_A"]["mean"] >= 0 and per_seed["1"]["B_minus_A"]["mean"] >= 0
    )
    g["A4"] = bool(subsets["scaffold_unseen_discovery"]["B_minus_A"]["mean"] > 0)
    g["A5"] = bool(
        np.isfinite(concentration["mean_gain_excluding_top_driver"])
        and concentration["mean_gain_excluding_top_driver"] > 0
        and concentration["top_driver_positive_gain_share"] < bcv.CONCENTRATION_TOP1_SHARE_MAX
    )
    g["A6"] = bool(
        contrasts["B_minus_pseudoB"]["mean"] > 0 and contrasts["B_minus_pseudoB"]["ci95_lower"] > 0
    )
    g["A7"] = bool(crossfit["pooled"]["B_minus_A"]["mean"] > 0)
    g["A8"] = bool(np.isfinite(recovery_B) and recovery_B >= bcv.RECOVERY_FRACTION_MIN)
    g["B1"] = bool(contrasts["C_minus_B"]["mean"] > 0)
    g["B2"] = bool(contrasts["C_minus_B"]["ci95_lower"] > 0)
    g["B3"] = bool(
        per_seed["0"]["C_minus_B"]["mean"] >= 0 and per_seed["1"]["C_minus_B"]["mean"] >= 0
    )
    g["B4"] = bool(
        contrasts["context_adv_over_pseudo"]["mean"] > 0
        and contrasts["context_adv_over_pseudo"]["ci95_lower"] > 0
    )
    g["B5"] = bool(np.isfinite(recovery_CB) and recovery_CB >= bcv.RECOVERY_FRACTION_MIN)
    label = bcv.decide_label(g)

    # ---- persist ----
    analysis = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "discipline": bcv.DISCIPLINE,
        "primary_k": PRIMARY_K,
        "method_means_k4_pooled": {
            "A": mean_A, "B": mean_B, "C": mean_C, "SIG": mean_S, "ORACLE": mean_O, "RANDOM": mean_R,
        },
        "contrasts_pooled": contrasts,
        "per_seed": per_seed,
        "subsets": subsets,
        "k_sensitivity": k_sensitivity,
        "ranking_quality": ranking_quality,
        "concentration": concentration,
        "coverage": coverage,
        "pseudo_control": pseudo_summary,
        "crossfit": crossfit,
        "oracle_gap": {
            "mean_A": mean_A, "mean_B": mean_B, "mean_C": mean_C, "mean_ORACLE": mean_O,
            "gap_DEV_to_oracle": gap_A, "gap_B_to_oracle": gap_B,
            "recovery_fraction_B": recovery_B, "recovery_fraction_CB": recovery_CB,
            "oracle_fraction_SIG_over_random": oracle_fraction_SIG,
            "oracle_fraction_B_over_random": oracle_fraction_B,
        },
        "reproduction": repro,
        "gates": g,
        "label": label,
        "reuse_note": (
            "All per-Junction DEV / |IG| / natural-replacement responses are reused from "
            "bjoc_signature_transfer_v1 raw artifacts; reproduction gate R reproduced the published "
            "DEV/SIG/ORACLE/RANDOM per-molecule metrics to <=1e-9."
        ),
    }
    bcv.write_json(ANALYSIS, analysis)

    metrics = {
        "protocol": PROTOCOL_ID,
        "created_utc": analysis["created_utc"],
        "discipline": bcv.DISCIPLINE,
        "label": label,
        "gates": g,
        "primary_contrasts": {
            "B_minus_A": contrasts["B_minus_A"],
            "C_minus_A": contrasts["C_minus_A"],
            "C_minus_B": contrasts["C_minus_B"],
        },
        "means_k4": analysis["method_means_k4_pooled"],
        "oracle_fraction": {
            "SIG_historical": oracle_fraction_SIG,
            "B_new": oracle_fraction_B,
            "recovery_B_minus_A": recovery_B,
            "recovery_C_minus_B": recovery_CB,
        },
        "per_seed": per_seed,
        "pseudo_control": {
            "B_minus_pseudoB": contrasts["B_minus_pseudoB"],
            "C_minus_pseudoC": contrasts["C_minus_pseudoC"],
            "context_adv_over_pseudo": contrasts["context_adv_over_pseudo"],
            "replicate_gain_means": pseudo_summary,
        },
        "concentration": concentration,
        "coverage": coverage,
        "crossfit_pooled": crossfit["pooled"],
        "reproduction": {"hard_ok": repro["hard_ok"], "max_abs_diff": repro["max_abs_diff"], "tolerance": repro["tolerance"]},
        "subsets": {
            name: {
                "n_molecules": rep["n_molecules"],
                "B_minus_A": rep.get("B_minus_A"),
                "C_minus_B": rep.get("C_minus_B"),
            }
            for name, rep in subsets.items()
        },
        "ranking_quality": ranking_quality,
        "k_sensitivity": k_sensitivity,
        "interpretation_limits": (
            "This tests only predictor-internal encoded-source sensitivity of the frozen R3 model under a "
            "fixed two-bond signature, fixed context pair and fixed calibration; it is not chemical "
            "causality, not whole-molecule explanation, and a null result does not prove DEV is complete."
        ),
    }
    bcv.write_json(METRICS, metrics)
    bcv.write_json(
        DECISION,
        {
            "protocol": PROTOCOL_ID,
            "created_utc": analysis["created_utc"],
            "label": label,
            "gates": g,
            "primary_contrast": contrasts["B_minus_A"],
            "context_contrast": contrasts["C_minus_B"],
            "oracle_gap": analysis["oracle_gap"],
            "discipline": bcv.DISCIPLINE,
            "analysis_sha256": bcv.file_sha256(ANALYSIS),
            "reproduction_sha256": bcv.file_sha256(REPRODUCTION),
        },
    )
    np.savez_compressed(
        ROW_METRICS,
        **{f"seed{s}_{key}": np.asarray(ev[key], dtype=np.float64) for s, ev in ((0, ev0), (1, ev1)) for key in ev if key != "driver_key"},
        **{f"pooled_{name}": arr for name, arr in (("A4", A), ("B4", B), ("C4", C), ("SIG4", S), ("ORACLE4", O), ("RANDOM4", R), ("pseudoB4", pB), ("pseudoC4", pC))},
        driver_key_seed0=np.asarray(ev0["driver_key"], dtype=object),
        driver_key_seed1=np.asarray(ev1["driver_key"], dtype=object),
    )
    log(f"analyze: label={label}")
    log(f"analyze: gates={g}")
    log(
        f"analyze: B-A {contrasts['B_minus_A']['mean']:+.5f} "
        f"[{contrasts['B_minus_A']['ci95_lower']:+.5f},{contrasts['B_minus_A']['ci95_upper']:+.5f}], "
        f"C-B {contrasts['C_minus_B']['mean']:+.5f} "
        f"[{contrasts['C_minus_B']['ci95_lower']:+.5f},{contrasts['C_minus_B']['ci95_upper']:+.5f}], "
        f"recovery_B={recovery_B:.3f}"
    )
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> int:
    frozen = verify_freeze()
    decision = bcv.read_json(DECISION)
    artifacts = [
        "PROTOCOL.md",
        "protocol_freeze.json",
        "input_audit.json",
        "validation_signatures.npz",
        "validation_prepare.json",
        "discovery_junctions.npz",
        "residual_tables.json",
        "pseudo_tables.npz",
        "pseudo_tables.json",
        "crossfit.json",
        "crossfit_row_metrics.npz",
        "reproduction_check.json",
        "validation_analysis.json",
        "metrics.json",
        "final_decision.json",
        "row_metrics_v2.npz",
        "REPORT.md",
    ]
    hashes = {}
    missing = []
    for name in artifacts:
        p = OUT / name
        if p.exists():
            hashes[name] = bcv.file_sha256(p)
        else:
            missing.append(name)

    # historical integrity: frozen inputs unchanged and prior artifacts unchanged
    audit_now = audit_inputs()
    changed = [
        name
        for name, rec in frozen["input_audit"].items()
        if isinstance(rec, dict) and rec.get("exists") and audit_now[name]["sha256"] != rec["sha256"]
    ]
    prior = bcv.read_json(PRIOR_FINALIZE)
    prior_checks = {}
    for name, sha in prior["artifact_sha256"].items():
        p = bcv.PRIOR / name
        if p.exists():
            prior_checks[name] = bool(bcv.file_sha256(p) == sha)
    integrity = {
        "frozen_inputs_changed": changed,
        "frozen_inputs_unchanged": not changed,
        "prior_artifact_hash_checks": prior_checks,
        "prior_artifacts_unchanged": bool(prior_checks) and all(prior_checks.values()),
    }
    bcv.write_json(HISTORICAL_INTEGRITY, integrity)

    payload = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "label": decision["label"],
        "gates": decision["gates"],
        "artifact_sha256": hashes,
        "missing_artifacts": missing,
        "freeze_git_commit": frozen["git_commit"],
        "freeze_ok": True,
        "historical_integrity": integrity,
        "discipline": bcv.DISCIPLINE,
    }
    bcv.write_json(FINALIZE, payload)
    if missing:
        log(f"finalize: missing artifacts: {missing}")
        return 1
    if changed or not integrity["prior_artifacts_unchanged"]:
        log(f"finalize: integrity problem: changed={changed} prior={not integrity['prior_artifacts_unchanged']}")
        return 1
    log(f"finalize: complete; label={decision['label']}; historical artifacts unchanged")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("verify-inputs", cmd_verify_inputs),
        ("freeze", cmd_freeze),
        ("prepare", cmd_prepare),
        ("residuals", cmd_residuals),
        ("pseudo", cmd_pseudo),
        ("crossfit", cmd_crossfit),
        ("analyze", cmd_analyze),
        ("finalize", cmd_finalize),
    ):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
