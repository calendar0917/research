#!/usr/bin/env python
"""Driver for ``explicit-composition-signal-audit-v1``.

Two strictly separated stages:

Stage A (label-blind)
    verify input hashes + the frozen 4-row exclusion list, compute B0/B1/C on
    the full official ZINC-12k *train* structures, build the frozen matching
    tiers, test permutation invariance / exact C partition / reconstruction,
    write structural artifacts and ``FREEZE.json``.  No ``y`` is read.

Stage B (after the freeze is verified)
    read only official-train ``y``, measure within-bag label differences, the
    empirical median-predictor lower bound, the penalized-logP formula
    decomposition, ring-basis instability, and dominance by scaffold /
    extreme labels.  It writes ``stage_c_gate.json`` and never trains.

No official valid/test, no internal Locked labels, no model predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR.parent
REPO_ROOT = CODE_DIR.parents[4]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import ecsa_objects as obj  # noqa: E402

PROTOCOL_ID = obj.PROTOCOL_ID
PROTOCOL_PATH = RESULTS_DIR / "PROTOCOL.md"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
FREEZE_PATH = RESULTS_DIR / "FREEZE.json"

TIERS = ("M01", "M0", "M0_ring", "B1", "A1", "A3", "A4", "L_old")
#: tiers used for the Stage-B bag lower-bound table
BAG_TIERS = ("M01", "B1", "M0", "A1", "A3", "A4", "L_old")

BOOTSTRAP_SEED = 20261016
INVARIANCE_SEED = 20261016
RING_PERMUTATIONS = 16
RING_SEED = 20261016


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def environment() -> dict[str, Any]:
    import networkx
    import numpy
    import rdkit

    try:
        import pynauty

        pynauty_version = getattr(pynauty, "__version__", "unknown")
    except Exception:  # pragma: no cover
        pynauty_version = "unavailable"
    try:
        import torch

        torch_version = torch.__version__
    except Exception:  # pragma: no cover
        torch_version = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "networkx": networkx.__version__,
        "rdkit": rdkit.__version__,
        "pynauty": str(pynauty_version),
        "torch": str(torch_version),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
    }


def tier_path(tier: str) -> Path:
    return ANALYSIS_DIR / f"stage_a_tier_{tier}.json"


def structural_artifact_paths() -> dict[str, Path]:
    paths = {
        "stage_a_summary": ANALYSIS_DIR / "stage_a_summary.json",
        "stage_a_molecules": ANALYSIS_DIR / "stage_a_molecules.jsonl",
        "stage_a_ring_instability": ANALYSIS_DIR / "stage_a_ring_instability.json",
        "stage_a_legacy_nesting": ANALYSIS_DIR / "stage_a_legacy_nesting.json",
    }
    for tier in TIERS:
        paths[f"stage_a_tier_{tier}"] = tier_path(tier)
    return paths


def code_paths() -> list[Path]:
    return sorted(
        [p for p in CODE_DIR.glob("*.py")] + [p for p in (RESULTS_DIR / "tests").glob("*.py")]
    )


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------


def verify_inputs() -> dict[str, Any]:
    checks = {
        "smiles_npz": {
            "path": str(obj.ZINC_SMILES_NPZ.relative_to(REPO_ROOT)),
            "sha256": obj.sha256_file(obj.ZINC_SMILES_NPZ),
            "expected": obj.ZINC_SMILES_SHA256,
        },
        "pyg_train_pt": {
            "path": str(obj.ZINC_TRAIN_PT.relative_to(REPO_ROOT)),
            "sha256": obj.sha256_file(obj.ZINC_TRAIN_PT),
            "expected": obj.ZINC_TRAIN_PT_SHA256,
        },
        "sascorer": {
            "path": str(obj.SASCORER_SRC.relative_to(REPO_ROOT)),
            "sha256": obj.sha256_file(obj.SASCORER_SRC),
        },
        "fpscores": {
            "path": str(obj.FSCORES.relative_to(REPO_ROOT)),
            "sha256": obj.sha256_file(obj.FSCORES),
        },
    }
    for name, item in checks.items():
        if "expected" in item and item["sha256"] != item["expected"]:
            raise SystemExit(
                f"input hash mismatch for {name}: {item['sha256']} != {item['expected']}"
            )
    return checks


def ring_penalty_from_cycles(cycles: Sequence[Sequence[int]]) -> int:
    if not cycles:
        return 0
    return max(0, int(max(len(c) for c in cycles)) - 6)


def ring_instability_table(records: Sequence[obj.MoleculeRecord]) -> dict[str, Any]:
    import networkx as nx
    from rdkit import Chem
    from rdkit.Chem import rdmolops

    rng = np.random.default_rng(RING_SEED)
    table: dict[str, Any] = {}
    n_unstable = 0
    n_default_nonzero = 0
    n_min_diff = 0
    n_sssr_diff = 0
    examples: list[dict[str, Any]] = []
    for rec in records:
        mol = Chem.MolFromSmiles(rec.smiles)
        if mol is None:
            raise RuntimeError(f"RDKit parse failure at row {rec.row}")
        base = nx.Graph(rdmolops.GetAdjacencyMatrix(mol))
        default_cycles = nx.cycle_basis(base)
        default_penalty = ring_penalty_from_cycles(default_cycles)
        penalties = [default_penalty]
        identity = list(range(mol.GetNumAtoms()))
        for _ in range(int(RING_PERMUTATIONS)):
            perm = identity.copy()
            rng.shuffle(perm)
            g = nx.relabel_nodes(base, {old: int(new) for old, new in enumerate(perm)})
            penalties.append(ring_penalty_from_cycles(nx.cycle_basis(g)))
        min_cycles = nx.minimum_cycle_basis(base)
        min_penalty = ring_penalty_from_cycles(min_cycles)
        sssr = [len(r) for r in Chem.GetSymmSSSR(mol)]
        sssr_penalty = max(0, max(sssr) - 6) if sssr else 0
        unstable = (
            len(set(int(p) for p in penalties)) > 1
            or int(min_penalty) != int(default_penalty)
            or int(sssr_penalty) != int(default_penalty)
        )
        if default_penalty > 0:
            n_default_nonzero += 1
        if min_penalty != default_penalty:
            n_min_diff += 1
        if sssr_penalty != default_penalty:
            n_sssr_diff += 1
        if unstable:
            n_unstable += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "row": int(rec.row),
                        "smiles": rec.smiles,
                        "default_penalty": int(default_penalty),
                        "permutation_penalties": sorted(set(int(p) for p in penalties)),
                        "minimum_cycle_basis_penalty": int(min_penalty),
                        "symm_sssr_penalty": int(sssr_penalty),
                    }
                )
        table[str(rec.row)] = {
            "default_penalty": int(default_penalty),
            "permutation_penalties": sorted(set(int(p) for p in penalties)),
            "minimum_cycle_basis_penalty": int(min_penalty),
            "symm_sssr_penalty": int(sssr_penalty),
            "unstable": bool(unstable),
        }
    return {
        "n_molecules_checked": len(records),
        "n_permutations_per_molecule": int(RING_PERMUTATIONS),
        "seed": int(RING_SEED),
        "n_unstable": int(n_unstable),
        "fraction_unstable": float(n_unstable / len(records)) if records else 0.0,
        "n_default_penalty_nonzero": int(n_default_nonzero),
        "n_minimum_basis_differs": int(n_min_diff),
        "n_symm_sssr_differs": int(n_sssr_diff),
        "examples": examples,
        "molecules": table,
    }


def invariance_check(
    records: Sequence[obj.MoleculeRecord], n_molecules: int = 200, n_relabelings: int = 3
) -> dict[str, Any]:
    from rdkit import Chem

    rng = np.random.default_rng(INVARIANCE_SEED)
    sample = records[: min(int(n_molecules), len(records))]
    failures: list[dict[str, Any]] = []
    n_checks = 0
    for rec in sample:
        mol = Chem.MolFromSmiles(rec.smiles)
        if mol is None:
            raise RuntimeError(f"RDKit parse failure at row {rec.row}")
        Chem.RemoveStereochemistry(mol)
        n_atoms = mol.GetNumAtoms()
        for _ in range(int(n_relabelings)):
            perm = list(range(n_atoms))
            rng.shuffle(perm)
            renumbered = Chem.RenumberAtoms(mol, [int(p) for p in perm])
            sub = obj.build_mol_rec("zinc", rec.row, rec.mol_uid, rec.smiles, renumbered)
            new = obj.make_record(rec.row, rec.mol_uid, rec.smiles, renumbered, sub)
            n_checks += 1
            diffs = {
                "b0_equal": new.b0_key == rec.b0_key,
                "b1_equal": new.b1_key == rec.b1_key,
                "b1_t2_equal": new.b1_t2_key == rec.b1_t2_key,
                "b1_t3_equal": new.b1_t3_key == rec.b1_t3_key,
                "c_key_equal": new.c_key_hex == rec.c_key_hex,
                "c_smiles_equal": new.c_smiles == rec.c_smiles,
            }
            if not all(diffs.values()):
                failures.append({"row": int(rec.row), **diffs})
    return {
        "n_molecules": len(sample),
        "n_relabelings_per_molecule": int(n_relabelings),
        "n_checks": int(n_checks),
        "n_failures": len(failures),
        "failures": failures[:20],
        "seed": int(INVARIANCE_SEED),
    }


def c_partition_check(records: Sequence[obj.MoleculeRecord]) -> dict[str, Any]:
    key_to_smiles: dict[str, set[str]] = defaultdict(set)
    smiles_to_key: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        key_to_smiles[rec.c_key_hex].add(rec.c_smiles)
        smiles_to_key[rec.c_smiles].add(rec.c_key_hex)
    key_collisions = {k: sorted(v) for k, v in key_to_smiles.items() if len(v) > 1}
    key_splits = {k: sorted(v) for k, v in smiles_to_key.items() if len(v) > 1}
    return {
        "n_molecules": len(records),
        "n_C_keys": len(key_to_smiles),
        "n_C_smiles": len(smiles_to_key),
        "n_key_to_multiple_smiles": len(key_collisions),
        "n_smiles_to_multiple_keys": len(key_splits),
        "key_collision_examples": dict(list(key_collisions.items())[:5]),
        "key_split_examples": dict(list(key_splits.items())[:5]),
        "one_to_one": not key_collisions and not key_splits,
        "note": (
            "C key is the exact unit–atom incidence certificate; C smiles is an "
            "independent RDKit constitution canonicalisation.  Equality of the two "
            "partitions validates both invariance and completeness on this dataset."
        ),
    }


def legacy_nesting(records: Sequence[obj.MoleculeRecord], old_tier: Mapping[str, Any]) -> dict[str, Any]:
    by_row = {int(r.row): r for r in records}
    pairs = obj.tier_pairs(old_tier)
    n = len(pairs)
    same_b0 = 0
    same_atom = 0
    same_element = 0
    same_b1 = 0
    same_ring = 0
    for pair in pairs:
        a = by_row[int(pair["row_a"])]
        b = by_row[int(pair["row_b"])]
        same_b0 += int(a.b0_key == b.b0_key)
        same_atom += int(a.atom_class_counts == b.atom_class_counts)
        same_element += int(a.element_counts == b.element_counts)
        same_b1 += int(a.b1_key == b.b1_key)
        same_ring += int(a.n_rings == b.n_rings)
    return {
        "tier": "L_old",
        "definition": obj.TIER_DEFINITIONS["L_old"],
        "n_pairs": int(n),
        "n_pairs_same_B0": int(same_b0),
        "fraction_same_B0": float(same_b0 / n) if n else None,
        "n_pairs_same_atom_class_composition": int(same_atom),
        "fraction_same_atom_class_composition": float(same_atom / n) if n else None,
        "n_pairs_same_element_composition": int(same_element),
        "fraction_same_element_composition": float(same_element / n) if n else None,
        "n_pairs_same_B1": int(same_b1),
        "fraction_same_B1": float(same_b1 / n) if n else None,
        "n_pairs_same_ring_count": int(same_ring),
        "note": (
            "L_old is the historical bond-RW coarse marginal signature.  It does "
            "not include atom/element composition, so its C-different pairs are "
            "mostly composition-confounded.  The nesting counts quantify that."
        ),
    }


def stage_a(limit: int | None = None, n_invariance: int = 200) -> dict[str, Any]:
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log("[stage-a] verifying inputs (hashes + frozen exclusion list)")
    inputs = verify_inputs()
    pairing = obj.pairing_verification()
    if not pairing.get("reproduces_frozen_exclusion_list"):
        raise SystemExit(f"pairing verification did not reproduce the frozen exclusions: {pairing}")
    log(f"[stage-a] pairing: {pairing['n_rows_checked']} rows, mismatches={pairing['mismatch_rows']}")

    log("[stage-a] building label-blind molecule records")
    rows, smiles, records = obj.build_records(limit=limit)
    if limit is not None:
        log(f"[stage-a] SMOKE limit={limit}: {len(records)} molecules")
    n_recon_fail = sum(1 for r in records if not r.reconstruction_ok)
    if n_recon_fail:
        raise SystemExit(f"{n_recon_fail} molecules failed patch-union reconstruction")
    log(f"[stage-a] records: {len(records)}; recon failures: {n_recon_fail}")

    log("[stage-a] C-key validity: exact partition vs canonical constitution")
    ccheck = c_partition_check(records)
    log(f"[stage-a] C partition one_to_one={ccheck['one_to_one']}")
    if not ccheck["one_to_one"]:
        raise SystemExit(f"C key is not a complete/invariant constitution key: {ccheck}")

    log("[stage-a] permutation invariance")
    invariance = invariance_check(records, n_molecules=n_invariance, n_relabelings=3)
    log(f"[stage-a] invariance failures={invariance['n_failures']} / {invariance['n_checks']}")
    if invariance["n_failures"]:
        raise SystemExit(f"permutation invariance failed: {invariance['failures'][:3]}")

    log("[stage-a] building frozen matching tiers")
    tier_payloads: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        payload = obj.build_tier(records, tier)
        tier_payloads[tier] = payload
        obj.write_json(tier_path(tier), payload)
        log(
            f"  {tier}: groups={payload['n_groups']} diffC_groups={payload['n_diffC_groups']} "
            f"molecules={payload['n_molecules_in_diffC_groups']} "
            f"scaffolds={payload['n_scaffolds_in_diffC_groups']} "
            f"pairs={payload['n_pairs_diffC']} "
            f"scaffold_disjoint={payload['n_scaffold_disjoint_pairs_diffC']}"
        )

    legacy = legacy_nesting(records, tier_payloads["L_old"])
    obj.write_json(ANALYSIS_DIR / "stage_a_legacy_nesting.json", legacy)

    log("[stage-a] ring-basis instability (structure only)")
    tier_rows: set[int] = set()
    by_row = {int(r.row): r for r in records}
    for tier in TIERS:
        for pair in obj.tier_pairs(tier_payloads[tier]):
            tier_rows.add(int(pair["row_a"]))
            tier_rows.add(int(pair["row_b"]))
    ring_records = [by_row[row] for row in sorted(tier_rows) if row in by_row]
    ring = ring_instability_table(ring_records)
    obj.write_json(ANALYSIS_DIR / "stage_a_ring_instability.json", ring)
    log(f"[stage-a] ring instability: {ring['n_unstable']}/{ring['n_molecules_checked']}")

    log("[stage-a] writing per-molecule structural table")
    with (ANALYSIS_DIR / "stage_a_molecules.jsonl").open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec.structure_json(), default=obj._json_default) + "\n")

    tier_counts = {}
    for tier in TIERS:
        p = tier_payloads[tier]
        tier_counts[tier] = {
            k: p[k]
            for k in (
                "definition",
                "n_molecules",
                "n_groups",
                "n_singleton_groups",
                "n_multi_groups",
                "n_diffC_groups",
                "n_molecules_in_diffC_groups",
                "n_scaffolds_in_diffC_groups",
                "n_topologies_in_diffC_groups",
                "n_pairs_diffC",
                "n_scaffold_disjoint_pairs_diffC",
                "max_group_size",
                "max_diffC_group_size",
                "group_size_histogram",
                "pair_enumeration_capped",
            )
        }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "stage": "A_structure_only",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 3),
        "labels_loaded": False,
        "official_valid_or_test_loaded": False,
        "model_predictions_loaded": False,
        "inputs": inputs,
        "excluded_rows": list(obj.EXCLUDED),
        "pairing_verification": pairing,
        "n_molecules_analysed": len(records),
        "n_reconstruction_failures": int(n_recon_fail),
        "c_partition_check": ccheck,
        "invariance": invariance,
        "ring_instability": {
            k: ring[k]
            for k in (
                "n_molecules_checked",
                "n_permutations_per_molecule",
                "seed",
                "n_unstable",
                "fraction_unstable",
                "n_default_penalty_nonzero",
                "n_minimum_basis_differs",
                "n_symm_sssr_differs",
                "examples",
            )
        },
        "legacy_nesting": legacy,
        "tiers": tier_counts,
        "definitions": obj.TIER_DEFINITIONS,
        "environment": environment(),
    }
    obj.write_json(ANALYSIS_DIR / "stage_a_summary.json", summary)

    log("[stage-a] writing FREEZE.json")
    freeze = {
        "protocol_id": PROTOCOL_ID,
        "stage": "A_complete_frozen_before_labels",
        "created_utc": summary["created_utc"],
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "git_status_short": git_output("status", "--short"),
        "environment": summary["environment"],
        "inputs": inputs,
        "excluded_rows": list(obj.EXCLUDED),
        "pairing_verification": pairing,
        "definitions": obj.TIER_DEFINITIONS,
        "protocol_sha256": obj.sha256_file(PROTOCOL_PATH),
        "code_sha256": {
            str(p.relative_to(RESULTS_DIR)): obj.sha256_file(p) for p in code_paths()
        },
        "stage_a_artifacts_sha256": {
            name: obj.sha256_file(path) for name, path in structural_artifact_paths().items()
        },
        "tier_counts": tier_counts,
        "labels_loaded": False,
        "official_valid_or_test_loaded": False,
        "model_predictions_loaded": False,
        "freeze_rule": (
            "Stage B refuses to run unless protocol, code, inputs and every "
            "stage_a_artifact hash match this manifest.  Stage A never reads y."
        ),
    }
    obj.write_json(FREEZE_PATH, freeze)
    log(f"[stage-a] done in {summary['runtime_seconds']}s")
    return summary


# ---------------------------------------------------------------------------
# freeze verification
# ---------------------------------------------------------------------------


def verify_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.exists():
        raise SystemExit(f"missing freeze manifest: {FREEZE_PATH}")
    freeze = json.loads(FREEZE_PATH.read_text())
    mismatches: list[str] = []
    if freeze.get("labels_loaded") is not False:
        mismatches.append("freeze manifest does not declare labels_loaded=false")
    current_protocol = obj.sha256_file(PROTOCOL_PATH)
    if current_protocol != freeze["protocol_sha256"]:
        mismatches.append(f"protocol hash {current_protocol} != {freeze['protocol_sha256']}")
    current_code = {str(p.relative_to(RESULTS_DIR)): obj.sha256_file(p) for p in code_paths()}
    if current_code != freeze["code_sha256"]:
        missing = sorted(set(freeze["code_sha256"]) - set(current_code))
        extra = sorted(set(current_code) - set(freeze["code_sha256"]))
        changed = sorted(
            k for k in set(current_code) & set(freeze["code_sha256"])
            if current_code[k] != freeze["code_sha256"][k]
        )
        mismatches.append(f"code hash mismatch missing={missing} extra={extra} changed={changed}")
    for name, expected in freeze["inputs"].items():
        path = REPO_ROOT / expected["path"]
        actual = obj.sha256_file(path)
        if actual != expected["sha256"]:
            mismatches.append(f"input {name} hash {actual} != {expected['sha256']}")
    for name, path in structural_artifact_paths().items():
        expected = freeze["stage_a_artifacts_sha256"].get(name)
        if expected is None:
            mismatches.append(f"artifact {name} absent from freeze")
            continue
        actual = obj.sha256_file(path)
        if actual != expected:
            mismatches.append(f"artifact {name} hash {actual} != {expected}")
    if mismatches:
        raise SystemExit("FREEZE VERIFICATION FAILED:\n  " + "\n  ".join(mismatches))
    return {
        "verified": True,
        "n_code_files": len(current_code),
        "n_inputs": len(freeze["inputs"]),
        "n_artifacts": len(freeze["stage_a_artifacts_sha256"]),
        "freeze_created_utc": freeze["created_utc"],
        "freeze_git_commit": freeze["git_commit"],
        "labels_loaded_at_stage_a": False,
    }


# ---------------------------------------------------------------------------
# Stage B: labels
# ---------------------------------------------------------------------------


def load_official_train_y() -> tuple[np.ndarray, dict[str, Any]]:
    """Official ZINC-12k *train* y only.  Valid/test are never opened."""
    import torch

    if not obj.ZINC_TRAIN_PT.exists():
        raise SystemExit(f"missing {obj.ZINC_TRAIN_PT}")
    blob = torch.load(obj.ZINC_TRAIN_PT, map_location="cpu", weights_only=False)
    y = np.asarray(blob[0]["y"], dtype=np.float64).reshape(-1)
    if y.shape != (10000,):
        raise SystemExit(f"unexpected official-train y shape: {y.shape}")
    meta: dict[str, Any] = {
        "source": str(obj.ZINC_TRAIN_PT.relative_to(REPO_ROOT)),
        "sha256": obj.sha256_file(obj.ZINC_TRAIN_PT),
        "n": int(y.size),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=0)),
        "min": float(y.min()),
        "max": float(y.max()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    return y, meta


def crosscheck_encoded_train_y(y: np.ndarray) -> dict[str, Any]:
    """Cross-check PyG y against the frozen encoded-train cache (official train only)."""
    encoded = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
    if not encoded.exists():
        return {"performed": False, "reason": "encoded_train.pt absent"}
    try:
        import torch

        code = REPO_ROOT / "tracks/structure-contribution/code"
        if str(code) not in sys.path:
            sys.path.insert(0, str(code))
        from structural_relation_objects_v1 import MolObjects  # noqa: F401  (unpickle class)

        payload = torch.load(encoded, map_location="cpu", weights_only=False)
        y2 = np.asarray([float(payload[i].y.reshape(-1)[0]) for i in range(10000)], dtype=np.float64)
        del payload
        diff = np.abs(y - y2)
        return {
            "performed": True,
            "source": str(encoded.relative_to(REPO_ROOT)),
            "n": int(y2.size),
            "max_abs_diff": float(diff.max()),
            "n_mismatch_gt_1e-9": int((diff > 1e-9).sum()),
            "aligned": bool(np.allclose(y, y2, rtol=0.0, atol=1e-9)),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"performed": False, "reason": repr(exc)}


def load_target_formula_code():
    code = REPO_ROOT / "tracks/structure-contribution/code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import bjoc_mmp_feasibility_v1 as mmp

    return mmp


def tier_pair_table(
    tier: str,
    pairs: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    components: Mapping[int, Mapping[str, float]],
    recon: Mapping[int, float],
    ring_table: Mapping[str, Any],
    constants: Mapping[str, float],
    mmp,
    *,
    noise_floor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        ra, rb = int(pair["row_a"]), int(pair["row_b"])
        ca, cb = components[ra], components[rb]
        fd = mmp.formula_delta(ca, cb, constants)
        dy = float(y[rb] - y[ra])
        residual = float(dy - fd["d_formula"])
        ring_unstable = bool(
            ring_table.get(str(ra), {}).get("unstable", False)
            or ring_table.get(str(rb), {}).get("unstable", False)
        )
        pair_noise = abs(float(recon[ra])) + abs(float(recon[rb]))
        beyond_noise = abs(residual) > max(float(noise_floor), float(pair_noise))
        rows.append(
            {
                "tier": tier,
                "group_id": pair["group_id"],
                "row_a": ra,
                "row_b": rb,
                "mol_uid_a": int(pair["mol_uid_a"]),
                "mol_uid_b": int(pair["mol_uid_b"]),
                "scaffold_a": pair["scaffold_a"],
                "scaffold_b": pair["scaffold_b"],
                "scaffold_disjoint": bool(pair["scaffold_disjoint"]),
                "y_a": float(y[ra]),
                "y_b": float(y[rb]),
                "dy": dy,
                "d_logP_raw": float(fd["d_logP_raw"]),
                "d_SAneg_raw": float(fd["d_SAneg_raw"]),
                "d_cyclepen_raw": float(fd["d_cyclepen_raw"]),
                "d_logP_scaled": float(fd["d_logP_scaled"]),
                "d_SAneg_scaled": float(fd["d_SAneg_scaled"]),
                "d_cyclepen_scaled": float(fd["d_cyclepen_scaled"]),
                "d_formula": float(fd["d_formula"]),
                "residual": residual,
                "recon_abs_a": float(abs(recon[ra])),
                "recon_abs_b": float(abs(recon[rb])),
                "pair_recon_noise": float(pair_noise),
                "beyond_reconstruction_noise": bool(beyond_noise),
                "ring_basis_unstable": ring_unstable,
                "extreme_y": bool(abs(y[ra]) > 3.0 or abs(y[rb]) > 3.0),
                "extreme_dy": bool(abs(dy) > 5.0),
            }
        )
    if not rows:
        return [], {
            "tier": tier,
            "n_pairs": 0,
            "note": "no C-different pairs at this matching tier; no label distribution exists",
        }
    dy = np.asarray([r["dy"] for r in rows], dtype=np.float64)
    residual = np.asarray([r["residual"] for r in rows], dtype=np.float64)
    dformula = np.asarray([r["d_formula"] for r in rows], dtype=np.float64)
    clusters = [r["group_id"] for r in rows]
    dlogp = np.asarray([r["d_logP_scaled"] for r in rows], dtype=np.float64)
    dsa = np.asarray([r["d_SAneg_scaled"] for r in rows], dtype=np.float64)
    dcyc = np.asarray([r["d_cyclepen_scaled"] for r in rows], dtype=np.float64)
    corr = None
    if float(dformula.std()) > 0 and float(dy.std()) > 0:
        corr = float(np.corrcoef(dformula, dy)[0, 1])
    summary: dict[str, Any] = {
        "tier": tier,
        "n_pairs": len(rows),
        "n_groups": len(set(clusters)),
        "n_molecules": len({r["row_a"] for r in rows} | {r["row_b"] for r in rows}),
        "n_scaffolds": len({r["scaffold_a"] for r in rows} | {r["scaffold_b"] for r in rows}),
        "n_scaffold_disjoint_pairs": int(sum(r["scaffold_disjoint"] for r in rows)),
        "delta_y": obj.summarize_numeric(dy),
        "delta_y_cluster_bootstrap": obj.cluster_bootstrap_mean(
            dy, clusters, n_boot=5000, seed=BOOTSTRAP_SEED
        ),
        "formula": {
            "r2": obj.r2_score(dy, dformula),
            "corr": corr,
            "ols": obj.ols_with_intercept(dy, np.column_stack([dlogp, dsa, dcyc])),
            "delta_formula": obj.summarize_numeric(dformula),
            "component_scaled": {
                "logP": obj.summarize_numeric(dlogp),
                "SAneg": obj.summarize_numeric(dsa),
                "cyclepen": obj.summarize_numeric(dcyc),
            },
            "component_raw_abs_mean": {
                "logP": float(np.mean(np.abs([r["d_logP_raw"] for r in rows]))),
                "SAneg": float(np.mean(np.abs([r["d_SAneg_raw"] for r in rows]))),
                "cyclepen": float(np.mean(np.abs([r["d_cyclepen_raw"] for r in rows]))),
            },
        },
        "residual": obj.summarize_numeric(residual),
        "residual_cluster_bootstrap": obj.cluster_bootstrap_mean(
            np.abs(residual), clusters, n_boot=5000, seed=BOOTSTRAP_SEED + 1
        ),
        "reconstruction_noise": {
            "noise_floor": float(noise_floor),
            "n_beyond_noise": int(sum(r["beyond_reconstruction_noise"] for r in rows)),
            "fraction_beyond_noise": float(
                sum(r["beyond_reconstruction_noise"] for r in rows) / len(rows)
            ),
            "max_pair_recon_noise": float(max(r["pair_recon_noise"] for r in rows)),
        },
        "ring_basis": {
            "n_pairs_any_unstable": int(sum(r["ring_basis_unstable"] for r in rows)),
            "fraction_pairs_any_unstable": float(
                sum(r["ring_basis_unstable"] for r in rows) / len(rows)
            ),
            "residual_abs_mean_stable": obj.summarize_numeric(
                [abs(r["residual"]) for r in rows if not r["ring_basis_unstable"]]
            ),
            "residual_abs_mean_unstable": obj.summarize_numeric(
                [abs(r["residual"]) for r in rows if r["ring_basis_unstable"]]
            ),
        },
        "extreme_labels": {
            "n_pairs_extreme_y": int(sum(r["extreme_y"] for r in rows)),
            "n_pairs_extreme_dy": int(sum(r["extreme_dy"] for r in rows)),
            "sum_abs_dy_extreme_y": float(
                sum(abs(r["dy"]) for r in rows if r["extreme_y"])
            ),
            "sum_abs_dy_all": float(np.abs(dy).sum()),
            "sum_abs_residual_extreme_y": float(
                sum(abs(r["residual"]) for r in rows if r["extreme_y"])
            ),
            "sum_abs_residual_all": float(np.abs(residual).sum()),
        },
    }
    summary["dominance"] = dominance_analysis(rows)
    return rows, summary


def dominance_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_pairs": 0}
    dy = np.asarray([abs(float(r["dy"])) for r in rows], dtype=np.float64)
    res = np.asarray([abs(float(r["residual"])) for r in rows], dtype=np.float64)
    base = {"mean_abs_dy": float(dy.mean()), "mean_abs_residual": float(res.mean())}
    underpowered = len(rows) < 20

    def leave_out(field: str) -> dict[str, Any]:
        if field == "scaffold":
            values = sorted(
                {str(r["scaffold_a"]) for r in rows} | {str(r["scaffold_b"]) for r in rows}
            )
        else:
            values = sorted({str(r[field]) for r in rows})
        changes_dy: list[tuple[float, str, int]] = []
        changes_res: list[tuple[float, str, int]] = []
        for value in values:
            if field == "scaffold":
                idx = [
                    i
                    for i, r in enumerate(rows)
                    if str(r["scaffold_a"]) != value and str(r["scaffold_b"]) != value
                ]
            else:
                idx = [i for i, r in enumerate(rows) if str(r[field]) != value]
            if not idx:
                continue
            m_dy = float(dy[idx].mean())
            m_res = float(res[idx].mean())
            changes_dy.append((abs(m_dy - base["mean_abs_dy"]), value, len(idx)))
            changes_res.append((abs(m_res - base["mean_abs_residual"]), value, len(idx)))
        changes_dy.sort(reverse=True)
        changes_res.sort(reverse=True)
        return {
            "n_clusters": len(values),
            "max_abs_change_mean_abs_dy": changes_dy[0][0] if changes_dy else None,
            "max_abs_change_mean_abs_dy_cluster": changes_dy[0][1] if changes_dy else None,
            "max_abs_change_mean_abs_dy_remaining": changes_dy[0][2] if changes_dy else None,
            "max_abs_change_mean_abs_residual": changes_res[0][0] if changes_res else None,
            "max_abs_change_mean_abs_residual_cluster": changes_res[0][1] if changes_res else None,
            "max_abs_change_mean_abs_residual_remaining": changes_res[0][2] if changes_res else None,
        }

    def top_removal() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, arr in (("dy", dy), ("residual", res)):
            order = np.argsort(-arr, kind="stable")
            entry = {}
            for k in (1, max(1, int(round(0.05 * len(rows)))), max(1, int(round(0.10 * len(rows))))):
                if k >= len(rows):
                    entry[str(k)] = {"n_remaining": 0, "mean_abs": None}
                    continue
                remaining = np.delete(arr, order[:k])
                entry[str(k)] = {
                    "n_remaining": int(remaining.size),
                    "mean_abs": float(remaining.mean()),
                }
            out[name] = entry
        return out

    extreme_dy = [abs(float(r["dy"])) for r in rows if r["extreme_y"]]
    extreme_res = [abs(float(r["residual"])) for r in rows if r["extreme_y"]]
    return {
        "n_pairs": len(rows),
        "underpowered_n_lt_20": bool(underpowered),
        "base": base,
        "leave_one_group_out": leave_out("group_id"),
        "leave_one_scaffold_out": leave_out("scaffold"),
        "top_k_removal": top_removal(),
        "extreme_y": {
            "n_pairs": len(extreme_dy),
            "mean_abs_dy": float(np.mean(extreme_dy)) if extreme_dy else None,
            "mean_abs_residual": float(np.mean(extreme_res)) if extreme_res else None,
            "share_sum_abs_dy": float(np.sum(extreme_dy) / dy.sum()) if extreme_dy else 0.0,
            "share_sum_abs_residual": float(np.sum(extreme_res) / res.sum()) if extreme_res else 0.0,
        },
    }


def bag_lower_bound(tier_payload: Mapping[str, Any], y: np.ndarray, n_total: int) -> dict[str, Any]:
    def aggregate(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        n_groups = 0
        n_molecules = 0
        sum_abs = 0.0
        group_rows: list[dict[str, Any]] = []
        for group in groups:
            member_rows = [int(m["row"]) for m in group["members"]]
            vals = y[member_rows]
            med = float(np.median(vals))
            dev = np.abs(vals - med)
            n_groups += 1
            n_molecules += len(member_rows)
            sum_abs += float(dev.sum())
            group_rows.append(
                {
                    "group_id": group["group_id"],
                    "n_molecules": len(member_rows),
                    "median_y": med,
                    "mean_abs_dev_from_group_median": float(dev.mean()),
                    "sum_abs_dev": float(dev.sum()),
                    "rows": member_rows,
                }
            )
        return {
            "n_groups": int(n_groups),
            "n_molecules": int(n_molecules),
            "sum_abs_dev": float(sum_abs),
            "mean_abs_dev_from_group_median": float(sum_abs / n_molecules)
            if n_molecules
            else None,
            "whole_train_normalised": float(sum_abs / n_total) if n_total else None,
            "groups": group_rows,
        }

    diff_groups = list(tier_payload["diffC_groups"])
    duplicate_groups = list(tier_payload["sameC_duplicate_groups"])
    all_groups = diff_groups + duplicate_groups
    return {
        "tier": tier_payload["tier"],
        "definition": tier_payload["definition"],
        "n_total_molecules": int(n_total),
        "diffC_subset": aggregate(diff_groups),
        "duplicate_subset": aggregate(duplicate_groups),
        "all_multi_groups": aggregate(all_groups),
        "fraction_molecules_in_multi_groups": float(
            sum(len(g["members"]) for g in all_groups) / n_total
        )
        if n_total
        else None,
        "note": (
            "Empirical in-sample lower bound: predict each bag by the median of its "
            "own y.  Singleton bags contribute 0 by construction, so the whole-train "
            "value is small whenever the bag is a near-identity; this is not a "
            "generalisation bound and not a BJOC-P3 error bound."
        ),
    }


def duplicate_noise(tier_payload: Mapping[str, Any], y: np.ndarray) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for group in tier_payload["sameC_duplicate_groups"]:
        rows = [int(m["row"]) for m in group["members"]]
        vals = y[rows]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                deltas.append(float(vals[b] - vals[a]))
        group_rows.append(
            {
                "group_id": group["group_id"],
                "rows": rows,
                "C": group["C"],
                "y": [float(v) for v in vals],
                "max_abs_dy": float(np.max(vals) - np.min(vals)),
            }
        )
    return {
        "n_groups": len(group_rows),
        "n_molecules": sum(len(g["rows"]) for g in group_rows),
        "n_pairs": len(deltas),
        "delta_y": obj.summarize_numeric(deltas),
        "groups": group_rows,
    }


def stage_b(out_dir: Path = ANALYSIS_DIR, skip_encoded_crosscheck: bool = False) -> dict[str, Any]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log("[stage-b] verifying freeze")
    freeze_check = verify_freeze()
    log(f"[stage-b] freeze verified: {freeze_check}")

    log("[stage-b] reading official-train y only")
    y, y_meta = load_official_train_y()
    encoded_check = (
        {"performed": False, "reason": "skipped by flag"}
        if skip_encoded_crosscheck
        else crosscheck_encoded_train_y(y)
    )
    log(f"[stage-b] y: n={y.size} mean={y.mean():.4f}; encoded crosscheck={encoded_check}")

    summary_a = json.loads((ANALYSIS_DIR / "stage_a_summary.json").read_text())
    ring_table = json.loads((ANALYSIS_DIR / "stage_a_ring_instability.json").read_text())
    ring_molecules = ring_table["molecules"]
    n_total = int(summary_a["n_molecules_analysed"])

    with np.load(obj.ZINC_SMILES_NPZ, allow_pickle=True) as blob:
        all_smiles = [str(s) for s in blob["smiles"]]
    if len(all_smiles) != 10000:
        raise SystemExit("SMILES table is not 10,000 rows")

    tier_payloads = {tier: json.loads(tier_path(tier).read_text()) for tier in TIERS}
    all_pair_rows: dict[int, set[int]] = defaultdict(set)
    for tier in TIERS:
        for pair in obj.tier_pairs(tier_payloads[tier]):
            all_pair_rows[tier].add(int(pair["row_a"]))
            all_pair_rows[tier].add(int(pair["row_b"]))
    unique_pair_rows = {row for rows in all_pair_rows.values() for row in rows}
    # Formula reconstruction needs every analysable row for the noise floor.
    formula_rows = sorted(
        int(json.loads(line)["row"])
        for line in (ANALYSIS_DIR / "stage_a_molecules.jsonl").open()
    )
    unique_rows = sorted(unique_pair_rows | set(formula_rows))

    log(f"[stage-b] target components for {len(unique_rows)} unique rows")
    mmp = load_target_formula_code()
    components = mmp.target_components(all_smiles, unique_rows)
    recon_community = {
        row: float(y[row] - mmp.formula_y(components[row], obj.FORMULA_CONSTANTS_COMMUNITY))
        for row in formula_rows
    }
    recon_dataset = {
        row: float(y[row] - mmp.formula_y(components[row], obj.FORMULA_CONSTANTS_DATASET))
        for row in formula_rows
    }
    noise_floor = max(
        0.01, 2.0 * float(np.median(np.abs(list(recon_community.values()))))
    )
    noise_floor_dataset = max(
        0.01, 2.0 * float(np.median(np.abs(list(recon_dataset.values()))))
    )
    recon_summary = {}
    for label, recon in (("community", recon_community), ("dataset", recon_dataset)):
        vals = np.asarray(list(recon.values()), dtype=np.float64)
        recon_summary[label] = {
            "n": int(vals.size),
            "abs_mean": float(np.abs(vals).mean()),
            "abs_median": float(np.median(np.abs(vals))),
            "abs_p95": float(np.percentile(np.abs(vals), 95)),
            "abs_max": float(np.abs(vals).max()),
            "std": float(vals.std(ddof=0)),
            "noise_floor": float(max(0.01, 2.0 * float(np.median(np.abs(vals))))),
        }
    log(f"[stage-b] formula recon community median={recon_summary['community']['abs_median']:.5f}")

    tier_summaries: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    for tier in TIERS:
        pairs = obj.tier_pairs(tier_payloads[tier])
        rows, summary = tier_pair_table(
            tier,
            pairs,
            y,
            components,
            recon_community,
            ring_molecules,
            obj.FORMULA_CONSTANTS_COMMUNITY,
            mmp,
            noise_floor=noise_floor,
        )
        tier_summaries[tier] = summary
        all_pairs.extend(rows)
        if rows:
            log(
                f"  {tier}: pairs={len(rows)} dy_mean_abs={summary['delta_y']['mean_abs']:.4f} "
                f"resid_mean_abs={summary['residual']['mean_abs']:.4f} "
                f"R2={summary['formula']['r2']}"
            )
        else:
            log(f"  {tier}: pairs=0")

    bag_bounds = {
        tier: bag_lower_bound(tier_payloads[tier], y, n_total) for tier in BAG_TIERS
    }
    dup_noise = duplicate_noise(tier_payloads["B1"], y)

    strict = tier_summaries["M01"]
    strict_pairs = strict.get("n_pairs", 0)
    strict_res_median = (
        strict["residual"]["median_abs"] if strict_pairs and "residual" in strict else None
    )
    strict_res_ci = (
        strict["residual_cluster_bootstrap"]["ci95"]
        if strict_pairs and strict.get("residual_cluster_bootstrap")
        else None
    )
    strict_formula_r2 = strict["formula"]["r2"] if strict_pairs and "formula" in strict else None
    b1_payload = tier_payloads["B1"]

    gate = {
        "protocol_id": PROTOCOL_ID,
        "stage": "C_gate_only_no_training",
        "strict_tier": "M01",
        "strict_n_pairs": int(strict_pairs),
        "strict_n_groups": int(strict.get("n_groups", 0)),
        "strict_n_molecules": int(strict.get("n_molecules", 0)),
        "strict_n_scaffolds": int(strict.get("n_scaffolds", 0)),
        "strict_n_scaffold_disjoint_pairs": int(strict.get("n_scaffold_disjoint_pairs", 0)),
        "strict_residual_median_abs": strict_res_median,
        "strict_residual_cluster_ci95": strict_res_ci,
        "strict_formula_r2": strict_formula_r2,
        "b1_diffC_groups": int(b1_payload["n_diffC_groups"]),
        "b1_injective_on_distinct_constitutions": bool(b1_payload["n_diffC_groups"] == 0),
        "G1_strict_samples": bool(
            int(strict.get("n_groups", 0)) >= 30
            and int(strict.get("n_molecules", 0)) >= 60
            and int(strict.get("n_scaffolds", 0)) >= 20
            and int(strict.get("n_scaffold_disjoint_pairs", 0)) >= 30
        ),
        "G2_nonformula_signal": bool(
            strict_pairs > 0
            and strict_res_median is not None
            and strict_res_median >= 0.02
            and strict_formula_r2 is not None
            and strict_formula_r2 <= 0.95
            and strict_res_ci is not None
            and strict_res_ci[0] > 0.0
        ),
        "G3_fair_control": bool(b1_payload["n_diffC_groups"] > 0),
        "run_stage_c": False,
        "reason": (
            "No strict same-B0+same-B1 different-C bags exist, so G1/G2 fail.  B1 is "
            "injective on distinct constitutions (B1 different-C groups = 0), so a "
            "Control that uses B1 already contains C implicitly; a Control restricted "
            "to B0 would reintroduce the very marginal confounds this audit controls.  "
            "No fair Control/Real/Null probe can be constructed, hence G3 fails."
        ),
        "note": (
            "This is a screening gate, not a test of non-GNN composition mechanisms "
            "in general.  No training was run and no GPU was used."
        ),
    }
    gate["run_stage_c"] = bool(
        gate["G1_strict_samples"] and gate["G2_nonformula_signal"] and gate["G3_fair_control"]
    )

    payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": "B_labels_after_freeze",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 3),
        "freeze_verified": freeze_check,
        "labels_loaded": True,
        "label_source": "official ZINC(subset=True, split='train') y only",
        "official_valid_or_test_loaded": False,
        "internal_locked_labels_loaded": False,
        "model_predictions_loaded": False,
        "y": y_meta,
        "encoded_train_crosscheck": encoded_check,
        "n_molecules_analysed": n_total,
        "formula_constants": {
            "community": obj.FORMULA_CONSTANTS_COMMUNITY,
            "dataset": obj.FORMULA_CONSTANTS_DATASET,
        },
        "formula_reconstruction": recon_summary,
        "noise_floor_community": float(noise_floor),
        "noise_floor_dataset": float(noise_floor_dataset),
        "tiers": tier_summaries,
        "bag_lower_bounds": bag_bounds,
        "duplicate_label_noise_B1": dup_noise,
        "stage_c_gate": gate,
        "stage_a_summary_sha256": obj.sha256_file(ANALYSIS_DIR / "stage_a_summary.json"),
        "freeze_sha256": obj.sha256_file(FREEZE_PATH),
    }
    obj.write_json(out_dir / "stage_b_labels.json", payload)
    obj.write_json(out_dir / "stage_b_pairs.json", all_pairs)
    obj.write_json(out_dir / "stage_c_gate.json", gate)
    log(f"[stage-b] done in {payload['runtime_seconds']}s; stage_c_run={gate['run_stage_c']}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage-a", "stage-b", "all", "verify-freeze"))
    parser.add_argument("--limit", type=int, default=None, help="smoke-test molecule cap (Stage A)")
    parser.add_argument("--n-invariance", type=int, default=200)
    parser.add_argument("--skip-encoded-crosscheck", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "stage-a":
        stage_a(limit=args.limit, n_invariance=args.n_invariance)
        return 0
    if args.stage == "verify-freeze":
        print(json.dumps(verify_freeze(), indent=1))
        return 0
    if args.stage == "stage-b":
        stage_b(skip_encoded_crosscheck=args.skip_encoded_crosscheck)
        return 0
    if args.stage == "all":
        stage_a(limit=args.limit, n_invariance=args.n_invariance)
        stage_b(skip_encoded_crosscheck=args.skip_encoded_crosscheck)
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
