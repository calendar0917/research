"""E2E-DictEnv-Dictionary-Coder-Audit-v1 runner — CPU-only representation audit.

Round ``e2e_dictenv_dictionary_coder_audit_v1`` (Workstream Z, ZINC).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_dictionary_coder_audit_v1.py``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_dictionary_coder_audit_v1_preregistration.md``.

Stages
------
``preflight  stage-a  stage-d  stage-e  gate  stage-g-sanity  stage-g-train
 stage-g-audit  report  chain``

``stage-a --seed S`` audits one reused C6 dictionary (the trained ``D`` of the
C6 seed-0/1/2 soup checkpoints) with the frozen same-``D`` coder variants
(IHT-10 / IHT-30 / IHT-100 / exact OMP) on the identical official-train /
official-valid phi65 tensors.  ``stage-d``/``stage-e`` aggregate across seeds.
``gate`` applies the frozen IHT-30 training gate.  ``stage-g-*`` exist only for
the single authorised IHT-30 seed-0 run if the gate fires.

No predictor is trained unless the gate fires.  CPU only; official ZINC test is
never loaded (``official_test_loaded = false`` in every payload).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_dictionary_coder_audit_v1 as dca
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = dca.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_dictionary_coder_audit_v1"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_dictionary_coder_audit_v1_preregistration.md"

CODER_DIR = RESULTS_DIR / "coder_geometry"
DOMINANT_DIR = RESULTS_DIR / "dominant_atom"
SPECIALIZATION_DIR = RESULTS_DIR / "atom_specialization"
STABILITY_DIR = RESULTS_DIR / "cross_seed_stability"
RECOVERABILITY_DIR = RESULTS_DIR / "sparse_dense_recoverability"
TRAINING_DIR = RESULTS_DIR / "iht30_training"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit
_seed_everything = p2run._seed_everything

#: reused trained checkpoints (soup states) of the closed rounds.
CHECKPOINTS: dict[tuple[str, int], Path] = {
    ("C6", 0): TRACK_ROOT / "results/e2e_dictenv_h1_clarity_audit/matched_cpu/C6_e320_soup_state.pt",
    ("C6", 1): TRACK_ROOT / "results/e2e_dictenv_clean_mechanism_v1/stage_a_cpu_baseline/C6_seed1_soup_state.pt",
    ("C6", 2): TRACK_ROOT / "results/e2e_dictenv_clean_mechanism_v1/stage_a_cpu_baseline/C6_seed2_soup_state.pt",
    ("DENSE-TIED", 0): TRACK_ROOT / "results/e2e_dictenv_clean_mechanism_v1/stage_f_dictionary_specificity/FINAL-CLEAN-DENSE-TIED_seed0_soup_state.pt",
}
CHECKPOINT_JSON: dict[tuple[str, int], Path] = {
    ("C6", 0): TRACK_ROOT / "results/e2e_dictenv_h1_clarity_audit/matched_cpu/C6_e320.json",
    ("C6", 1): TRACK_ROOT / "results/e2e_dictenv_clean_mechanism_v1/stage_a_cpu_baseline/C6_seed1_e320.json",
    ("C6", 2): TRACK_ROOT / "results/e2e_dictenv_clean_mechanism_v1/stage_a_cpu_baseline/C6_seed2_e320.json",
    ("DENSE-TIED", 0): TRACK_ROOT / "results/e2e_dictenv_clean_mechanism_v1/stage_f_dictionary_specificity/FINAL-CLEAN-DENSE-TIED_seed0_e320.json",
}

#: single authorised new training arm (only if the frozen gate fires).
IHT30_SPEC = cm.CleanMechSpec("FINAL-CLEAN-IHT30", "C6", coding="sparse")
IHT30_STEPS = 30
IHT30_SANITY_EPOCHS = 20
IHT30_EPOCHS = 320

THREADS = 4
CONCURRENCY = 3


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CODER_DIR, DOMINANT_DIR, SPECIALIZATION_DIR, STABILITY_DIR, RECOVERABILITY_DIR, TRAINING_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# checkpoint / data access
# ---------------------------------------------------------------------------


def load_dictionary_entry(key: tuple[str, int]) -> dict[str, Any]:
    state_path = CHECKPOINTS[key]
    if not state_path.exists():
        raise FileNotFoundError(f"missing checkpoint {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "D" not in state:
        raise RuntimeError(f"checkpoint {state_path} has no dictionary parameter")
    D = np.asarray(state["D"].detach().cpu(), dtype=np.float32)
    if D.shape != (dca.PHI_DIM, dca.K_ATOMS):
        raise RuntimeError(f"unexpected dictionary shape {D.shape}")
    if not np.isfinite(D).all():
        raise RuntimeError(f"non-finite dictionary in {state_path}")
    json_path = CHECKPOINT_JSON[key]
    payload = _read_json(json_path) if json_path.exists() else {}
    return {
        "key": list(key),
        "checkpoint": str(state_path),
        "checkpoint_sha256": _sha256_file(state_path),
        "soup_valid_mae": float(payload.get("soup", {}).get("soup_valid_mae", float("nan"))),
        "dictionary": D,
    }


def load_phi(split: str) -> np.ndarray:
    path = p1run.CACHE_DIR / f"env_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing phi65 cache {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    phi = np.asarray(blob["phi"], dtype=np.float32)
    if phi.ndim != 2 or phi.shape[1] != dca.PHI_DIM:
        raise RuntimeError(f"unexpected phi shape {phi.shape}")
    return phi


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    dca.cpu_only_guard(torch.device("cpu"))
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    entries: dict[str, Any] = {}
    for key in (("C6", 0), ("C6", 1), ("C6", 2), ("DENSE-TIED", 0)):
        entry = load_dictionary_entry(key)
        entries[f"{key[0]}_seed{key[1]}"] = {
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "soup_valid_mae": entry["soup_valid_mae"],
            "dictionary_column_norm_min": float(np.linalg.norm(entry["dictionary"], axis=0).min()),
            "dictionary_column_norm_max": float(np.linalg.norm(entry["dictionary"], axis=0).max()),
        }
    train_phi = load_phi("train")
    valid_phi = load_phi("valid")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "preregistration": {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(PREREG_PATH),
        },
        "checkpoints": entries,
        "phi": {
            "train": list(train_phi.shape),
            "valid": list(valid_phi.shape),
            "cache_train": str((p1run.CACHE_DIR / "env_train.pt").relative_to(REPO_ROOT)),
            "cache_valid": str((p1run.CACHE_DIR / "env_valid.pt").relative_to(REPO_ROOT)),
        },
        "coders": list(dca.CODERS),
        "iht_steps": dict(dca.IHT_STEPS),
        "gate_thresholds": {
            "recon_factor": dca.GATE_RECON_FACTOR,
            "recon_min_abs": dca.GATE_RECON_MIN_ABS,
            "support_jaccard_gain": dca.GATE_SUPPORT_JACCARD_GAIN,
            "eff_atoms_gain": dca.GATE_EFF_ATOMS_GAIN,
            "top5_share_drop": dca.GATE_TOP5_SHARE_DROP,
            "max_rate_drop": dca.GATE_MAX_RATE_DROP,
            "top1_l1_share_drop": dca.GATE_TOP1_L1_SHARE_DROP,
            "code_cosine_gain": dca.GATE_CODE_COSINE_GAIN,
            "seeds_required": dca.GATE_SEEDS_REQUIRED,
        },
        "preregistration_snapshot": {
            "sha256": _sha256_file(PREREG_PATH),
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "frozen_at_commit": _git_commit(),
            "note": "snapshot of the preregistration hash at audit time; the file is never edited after this stage",
        },
    }
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", payload["preregistration_snapshot"])
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] 4 dictionaries, train {train_phi.shape} valid {valid_phi.shape}, "
        f"prereg sha {payload['preregistration']['sha256'][:12]}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Stage A / B / C / E for one seed
# ---------------------------------------------------------------------------


def _summary_row(
    *,
    X: np.ndarray,
    A: np.ndarray,
    Dbar: np.ndarray,
    reference: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "reconstruction": dca.reconstruction_metrics(X, A, Dbar),
        "concentration": dca.usage_concentration(dca.activation_frequency(A)),
        "coefficient_geometry": dca.coefficient_geometry(A),
    }
    if reference is not None:
        row["support_vs_omp"] = dca.support_agreement(A, reference)
        row["coefficient_vs_omp"] = dca.coefficient_agreement(A, reference)
    else:
        row["support_vs_omp"] = None
        row["coefficient_vs_omp"] = None
    return row


def stage_a(seed: int, threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out_path = CODER_DIR / f"seed{seed}" / "summary.json"
    if out_path.exists() and not force:
        print(f"[stage-a] seed {seed} cache hit", flush=True)
        return _read_json(out_path)
    started = time.perf_counter()
    dca.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(int(threads))
    entry = load_dictionary_entry(("C6", seed))
    Dbar = dca.effective_dictionary(entry["dictionary"])
    Dbar_torch = dca.effective_dictionary_torch(entry["dictionary"])
    train_phi = load_phi("train")
    valid_phi = load_phi("valid")
    splits = {"train": train_phi, "valid": valid_phi}

    codes: dict[str, dict[str, np.ndarray]] = {}
    tables: dict[str, dict[str, Any]] = {}
    frequencies: dict[str, dict[str, list[float]]] = {}
    for split, X in splits.items():
        reference = dca.codes_for("omp", X, Dbar, Dbar_torch=Dbar_torch)
        codes[split] = {"omp": reference}
        tables[split] = {}
        frequencies[split] = {}
        for coder in dca.CODERS:
            A = reference if coder == "omp" else dca.codes_for(coder, X, Dbar, Dbar_torch=Dbar_torch)
            codes[split][coder] = A
            tables[split][coder] = _summary_row(
                X=np.asarray(X, dtype=np.float64),
                A=A,
                Dbar=Dbar,
                reference=None if coder == "omp" else reference,
            )
            frequencies[split][coder] = dca.activation_frequency(A).tolist()
            print(
                f"[stage-a] seed={seed} split={split} coder={coder} "
                f"fro={tables[split][coder]['reconstruction']['recon_frobenius']:.3e} "
                f"neff={tables[split][coder]['concentration']['effective_atoms']:.3f}",
                flush=True,
            )
        del reference

    geometry = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "seed": int(seed),
        "checkpoint": {
            "path": str(CHECKPOINTS[("C6", seed)].relative_to(REPO_ROOT)),
            "sha256": entry["checkpoint_sha256"],
            "soup_valid_mae": entry["soup_valid_mae"],
        },
        "coders": list(dca.CODERS),
        "iht_steps": dict(dca.IHT_STEPS),
        "tables": tables,
        "activation_frequency": frequencies,
        "wall_clock_s": 0.0,
    }

    # ---- Stage C: fixed named structural descriptors and atom profiles ----
    Z_train = dca.structural_descriptors(train_phi)
    Z_valid = dca.structural_descriptors(valid_phi)
    specialization: dict[str, Any] = {}
    for coder in ("iht10", "omp", "iht30"):
        payload = dca.atom_specialization_rows(
            seed=seed,
            coder=coder,
            Z_train=Z_train,
            Z_valid=Z_valid,
            A_train=codes["train"][coder],
            A_valid=codes["valid"][coder],
        )
        specialization[coder] = payload
        print(
            f"[stage-a] seed={seed} specialization {coder} "
            f"median={payload['median_specialization']:.3f} "
            f"weighted={payload['usage_weighted_specialization']:.3f}",
            flush=True,
        )
    _write_json(
        SPECIALIZATION_DIR / f"seed{seed}.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "seed": int(seed),
            "coders": specialization,
        },
    )

    # ---- Stage B: dominant atom, mean direction, PCA ----
    mean_vector = dca.mean_direction(train_phi)
    pca_components, pca_ratio, _pca_mean = dca.pca_basis(train_phi, n_components=3)
    dominant: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "seed": int(seed),
        "mean_direction_norm": float(np.linalg.norm(mean_vector)),
    }
    for coder in ("iht10", "iht30", "omp"):
        dominant[coder] = dca.dominant_atom_analysis(
            Dbar=Dbar,
            X_train=train_phi,
            Z_train=Z_train,
            alpha_train=codes["train"][coder],
            frequencies=np.asarray(frequencies["train"][coder], dtype=np.float64),
            pca_components=pca_components,
            pca_ratio=pca_ratio,
            mean_vector=mean_vector,
        )
        print(
            f"[stage-a] seed={seed} dominant {coder} atom={dominant[coder]['dominant_atom']} "
            f"|cos(mu)|={dominant[coder]['dominant_atom_cosine_to_mean_direction']:.3f} "
            f"|cos(PC1)|={dominant[coder]['dominant_atom_cosine_to_pcs'][0]:.3f}",
            flush=True,
        )
    _write_json(DOMINANT_DIR / f"seed{seed}.json", dominant)

    # ---- Stage E: sparse <-> dense recoverability (same frozen D) ----
    dense_train = dca.dense_codes(train_phi, Dbar)
    dense_valid = dca.dense_codes(valid_phi, Dbar)
    dense_own_train = dense_own_valid = None
    if seed == 0 and ("DENSE-TIED", 0) in CHECKPOINTS:
        own = load_dictionary_entry(("DENSE-TIED", 0))
        own_Dbar = dca.effective_dictionary(own["dictionary"])
        dense_own_train = dca.dense_codes(train_phi, own_Dbar)
        dense_own_valid = dca.dense_codes(valid_phi, own_Dbar)
    directions: list[dict[str, Any]] = []
    pairs = [
        ("dense_tied", "iht10", dense_train, dense_valid),
        ("iht10", "dense_tied", codes["train"]["iht10"], codes["valid"]["iht10"]),
        ("dense_tied", "omp", dense_train, dense_valid),
        ("omp", "dense_tied", codes["train"]["omp"], codes["valid"]["omp"]),
        ("dense_tied", "iht30", dense_train, dense_valid),
        ("iht30", "dense_tied", codes["train"]["iht30"], codes["valid"]["iht30"]),
    ]
    for source_name, target_name, source_train, source_valid in pairs:
        target_train = dense_train if target_name == "dense_tied" else codes["train"][target_name]
        target_valid = dense_valid if target_name == "dense_tied" else codes["valid"][target_name]
        metrics = dca.fit_linear_recoverability(source_train, target_train, source_valid, target_valid)
        metrics.update(
            {
                "seed": int(seed),
                "source_representation": str(source_name),
                "target_representation": str(target_name),
                "same_frozen_dictionary": True,
            }
        )
        directions.append(metrics)
    if dense_own_train is not None:
        for source_name, target_name, source_train, source_valid in (
            ("dense_tied_own_D", "iht10", dense_own_train, dense_own_valid),
            ("iht10", "dense_tied_own_D", codes["train"]["iht10"], codes["valid"]["iht10"]),
        ):
            target_train = dense_own_train if target_name.startswith("dense_tied") else codes["train"][target_name]
            target_valid = dense_own_valid if target_name.startswith("dense_tied") else codes["valid"][target_name]
            metrics = dca.fit_linear_recoverability(source_train, target_train, source_valid, target_valid)
            metrics.update(
                {
                    "seed": int(seed),
                    "source_representation": str(source_name),
                    "target_representation": str(target_name),
                    "same_frozen_dictionary": False,
                    "note": "seed-0 trained DenseTied control dictionary, secondary context only",
                }
            )
            directions.append(metrics)
    _write_json(
        RECOVERABILITY_DIR / f"seed{seed}.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "seed": int(seed),
            "directions": directions,
        },
    )

    geometry["wall_clock_s"] = float(time.perf_counter() - started)
    _write_json(out_path, geometry)
    print(f"[stage-a] seed {seed} done in {geometry['wall_clock_s']:.0f}s", flush=True)
    return geometry


# ---------------------------------------------------------------------------
# Stage D — cross-seed dictionary stability
# ---------------------------------------------------------------------------


def stage_d() -> dict[str, Any]:
    _ensure_dirs()
    dictionaries: dict[int, np.ndarray] = {}
    usages: dict[int, np.ndarray] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {}
    for seed in dca.SEEDS:
        entry = load_dictionary_entry(("C6", seed))
        dictionaries[seed] = dca.effective_dictionary(entry["dictionary"])
        geometry = _read_json(CODER_DIR / f"seed{seed}" / "summary.json")
        usages[seed] = np.asarray(geometry["activation_frequency"]["train"]["iht10"], dtype=np.float64)
        specialization = _read_json(SPECIALIZATION_DIR / f"seed{seed}.json")
        profiles[seed] = np.asarray(
            specialization["coders"]["iht10"]["profiles"]["train_active"], dtype=np.float64
        )
    pair_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        match = dca.match_dictionaries(dictionaries[left], dictionaries[right])
        profile_cosines: list[float] = []
        usage_differences: list[float] = []
        for pair in match["pairs"]:
            atom_left = int(pair["atom_a"])
            atom_right = int(pair["atom_b"])
            profile_cosines.append(
                float(
                    dca._profile_cosine(profiles[left][atom_left], profiles[right][atom_right])
                )
            )
            usage_differences.append(float(abs(usages[left][atom_left] - usages[right][atom_right])))
            rows.append(
                {
                    "seed_a": int(left),
                    "seed_b": int(right),
                    "matched_atom_a": atom_left,
                    "matched_atom_b": atom_right,
                    "abs_cosine": float(pair["abs_cosine"]),
                    "usage_a": float(usages[left][atom_left]),
                    "usage_b": float(usages[right][atom_right]),
                    "usage_difference": float(abs(usages[left][atom_left] - usages[right][atom_right])),
                    "profile_cosine": float(profile_cosines[-1]),
                }
            )
        finite_profiles = [value for value in profile_cosines if np.isfinite(value)]
        payload = {
            "seed_a": int(left),
            "seed_b": int(right),
            "mean_matched_abs_cosine": match["matched_abs_cosine"]["mean"],
            "median_matched_abs_cosine": match["matched_abs_cosine"]["median"],
            "min_matched_abs_cosine": match["matched_abs_cosine"]["min"],
            "p10_matched_abs_cosine": match["matched_abs_cosine"]["p10"],
            "n_above_090": match["n_above_090"],
            "n_above_095": match["n_above_095"],
            "matched_profile_cosines": profile_cosines,
            "median_matched_profile_cosine": float(np.median(finite_profiles)) if finite_profiles else float("nan"),
            "mean_usage_difference": float(np.mean(usage_differences)),
            "pairs": match["pairs"],
        }
        pair_results.append(payload)
        print(
            f"[stage-d] {left}<->{right} mean|cos|={payload['mean_matched_abs_cosine']:.3f} "
            f"n90={payload['n_above_090']} profile_med={payload['median_matched_profile_cosine']:.3f}",
            flush=True,
        )
    verdict = dca.vocabulary_verdict(pair_results)
    _write_csv(
        STABILITY_DIR / "dictionary_stability.csv",
        rows,
        header=[
            "seed_a",
            "seed_b",
            "matched_atom_a",
            "matched_atom_b",
            "abs_cosine",
            "usage_a",
            "usage_b",
            "usage_difference",
            "profile_cosine",
        ],
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "pairs": pair_results,
        "verdict": verdict,
    }
    _write_json(STABILITY_DIR / "summary.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Stage E summary
# ---------------------------------------------------------------------------


def stage_e() -> dict[str, Any]:
    _ensure_dirs()
    rows: list[dict[str, Any]] = []
    for seed in dca.SEEDS:
        payload = _read_json(RECOVERABILITY_DIR / f"seed{seed}.json")
        rows.extend(payload["directions"])
    _write_csv(
        RECOVERABILITY_DIR / "recoverability.csv",
        [
            {
                "seed": row["seed"],
                "source_representation": row["source_representation"],
                "target_representation": row["target_representation"],
                "valid_r2": row["valid_r2"],
                "valid_normalized_error": row["valid_normalized_error"],
                "valid_mean_cosine": row["valid_mean_cosine"]["mean"],
                "linear_cka": row["linear_cka"],
                "design_rank": row["design_rank"],
            }
            for row in rows
        ],
        header=[
            "seed",
            "source_representation",
            "target_representation",
            "valid_r2",
            "valid_normalized_error",
            "valid_mean_cosine",
            "linear_cka",
            "design_rank",
        ],
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "rows": rows,
    }
    _write_json(RECOVERABILITY_DIR / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _gate_inputs() -> dict[int, dict[str, Any]]:
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in dca.SEEDS:
        geometry = _read_json(CODER_DIR / f"seed{seed}" / "summary.json")
        entry: dict[str, Any] = {}
        for coder in ("iht10", "iht30", "omp"):
            table = geometry["tables"]["valid"][coder]
            entry[coder] = {
                "frobenius": float(table["reconstruction"]["recon_frobenius"]),
                "support_jaccard_mean": float(table["support_vs_omp"]["jaccard"]["mean"])
                if table["support_vs_omp"] is not None
                else 1.0,
                "effective_atoms": float(table["concentration"]["effective_atoms"]),
                "top5_share": float(table["concentration"]["top5_share"]),
                "max_activation_rate": float(table["concentration"]["max_activation_rate"]),
                "top1_over_l1_mean": float(table["coefficient_geometry"]["top1_over_l1"]["mean"]),
                "code_cosine_mean": float(table["coefficient_vs_omp"]["cosine"]["mean"])
                if table["coefficient_vs_omp"] is not None
                else 1.0,
            }
        per_seed[int(seed)] = entry
    return per_seed


def stage_gate() -> dict[str, Any]:
    _ensure_dirs()
    per_seed = _gate_inputs()
    gate = dca.iht30_gate(per_seed)
    classification = dca.coder_issue_classification(per_seed)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "gate": gate,
        "classification": classification,
        "inputs": {str(seed): entry for seed, entry in per_seed.items()},
    }
    _write_json(RESULTS_DIR / "gate.json", payload)
    print(
        f"[gate] fired={gate['fired']} supported_seeds={gate['supported_seeds']} "
        f"classification={classification['fired']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Stage G — the single authorised IHT-30 seed-0 run (only if the gate fires)
# ---------------------------------------------------------------------------


class IHTStepModel(cm.CleanMechModel):
    """``CleanMechModel`` with a parameterised tied-IHT step count."""

    def __init__(
        self,
        config: Any,
        dictionary: np.ndarray,
        *,
        iht_steps: int = dca.IHT_STEPS["iht10"],
        **kwargs: Any,
    ) -> None:
        super().__init__(config, dictionary, **kwargs)
        self.iht_steps = int(iht_steps)

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        if self.coding == "sparse":
            Dbar = v0.normalized_dictionary(self.D)
            return v0.tied_iht_codes(Dbar, phi, s=dca.SPARSITY, steps=self.iht_steps)
        return super().code(phi)


def build_iht_model(
    dictionary: np.ndarray,
    seed: int,
    *,
    iht_steps: int,
    spec: cm.CleanMechSpec | None = None,
) -> IHTStepModel:
    spec = IHT30_SPEC if spec is None else spec
    torch.manual_seed(int(seed))
    return IHTStepModel(
        cm.H1_CONFIG,
        dictionary,
        node_binding=spec.node_binding,
        edge_binding=spec.edge_binding,
        coding=spec.coding,
        iht_steps=int(iht_steps),
    )


def _guard_gate_fired() -> dict[str, Any]:
    path = RESULTS_DIR / "gate.json"
    if not path.exists():
        raise RuntimeError("gate.json missing: run the audit and the gate before any training")
    payload = _read_json(path)
    if not bool(payload.get("gate", {}).get("fired")):
        raise RuntimeError("IHT-30 training gate did not fire; no training is authorised this round")
    return payload


def stage_g_sanity(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _guard_gate_fired()
    _ensure_dirs()
    out_path = TRAINING_DIR / "sanity_e20.json"
    if out_path.exists() and not force:
        print("[stage-g-sanity] cache hit", flush=True)
        return _read_json(out_path)
    started = time.perf_counter()
    dca.cpu_only_guard(torch.device("cpu"))
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")
    dictionary, dict_sha = p2run.load_dictionary("sdb32")

    def factory(dictionary_value: np.ndarray, run_seed: int) -> IHTStepModel:
        return build_iht_model(dictionary_value, run_seed, iht_steps=IHT30_STEPS)

    payload = audit.train_cpu(
        tag="IHT30-SANITY",
        mask=cm.arm_mask(IHT30_SPEC),
        epochs=IHT30_SANITY_EPOCHS,
        threads=int(threads),
        out_dir=TRAINING_DIR / "sanity",
        train_data=train,
        valid_data=valid,
        seed=0,
        save_states=True,
        log=True,
        model_factory=factory,
        arm_spec=IHT30_SPEC.as_dict(),
    )
    state_path = TRAINING_DIR / "sanity" / "IHT30-SANITY_final_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model = build_iht_model(dictionary, 0, iht_steps=IHT30_STEPS)
    model.load_state_dict({key: value.float() for key, value in state.items()})
    dictionary_finite = bool(torch.isfinite(model.D).all())
    dictionary_movement = float((model.D.detach() - torch.as_tensor(dictionary)).norm())
    loader = p1.make_env_loader(train[:256], int(p2run.BATCH_SIZE), False, 0)
    model.train()
    batch = next(iter(loader))
    prediction, aux = model(batch, mask=cm.arm_mask(IHT30_SPEC), return_aux=True)
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(cm.H1_LAMBDA) * model.reconstruction_loss(aux["phi"], aux["coord"])
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(model.D.grad.detach().norm()) if model.D.grad is not None else 0.0
    model.zero_grad(set_to_none=True)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "arm": IHT30_SPEC.as_dict(),
        "iht_steps": IHT30_STEPS,
        "checkpoint": "sanity state (20 epochs of the single formal trajectory; bit-identical prefix)",
        "finite_loss": bool(np.isfinite(payload["final_valid_mae"]) and np.isfinite(payload["best_valid_mae"])),
        "dictionary_finite": bool(dictionary_finite),
        "gradient_finite": bool(np.isfinite(gradient_norm)),
        "gradient_norm_D": float(gradient_norm),
        "dictionary_movement_frobenius": float(dictionary_movement),
        "valid_prediction_finite": bool(np.isfinite(payload["final_valid_mae"])),
        "valid_mae_e20": float(payload["final_valid_mae"]),
        "best_valid_mae_e20": float(payload["best_valid_mae"]),
        "soup_valid_mae_e20": float(payload["soup"]["soup_valid_mae"]),
        "wall_clock_s": float(payload["wall_clock_s"]),
        "round_seconds": float(time.perf_counter() - started),
    }
    result["passed"] = bool(
        result["finite_loss"]
        and result["dictionary_finite"]
        and result["gradient_finite"]
        and result["valid_prediction_finite"]
        and result["dictionary_movement_frobenius"] > 0
    )
    _write_json(out_path, result)
    print(f"[stage-g-sanity] passed={result['passed']} e20_soup={result['soup_valid_mae_e20']:.6f}", flush=True)
    return result


def stage_g_train(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _guard_gate_fired()
    _ensure_dirs()
    out_path = TRAINING_DIR / "FINAL-CLEAN-IHT30_seed0_e320.json"
    if out_path.exists() and not force:
        print("[stage-g-train] cache hit", flush=True)
        return _read_json(out_path)
    sanity = stage_g_sanity(threads=threads, force=force)
    if not bool(sanity.get("passed")):
        raise RuntimeError("epoch-20 sanity gate failed; the formal trajectory is not started")
    started = time.perf_counter()
    dca.cpu_only_guard(torch.device("cpu"))
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")

    def factory(dictionary_value: np.ndarray, run_seed: int) -> IHTStepModel:
        return build_iht_model(dictionary_value, run_seed, iht_steps=IHT30_STEPS)

    payload = audit.train_cpu(
        tag="FINAL-CLEAN-IHT30_seed0",
        mask=cm.arm_mask(IHT30_SPEC),
        epochs=IHT30_EPOCHS,
        threads=int(threads),
        out_dir=TRAINING_DIR,
        train_data=train,
        valid_data=valid,
        seed=0,
        save_states=True,
        log=True,
        model_factory=factory,
        arm_spec=IHT30_SPEC.as_dict(),
    )
    payload.update(
        {
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "arm": IHT30_SPEC.as_dict(),
            "iht_steps": IHT30_STEPS,
            "paired_reference": {
                "iht10_sparse_seed0_soup": {
                    "path": str(CHECKPOINT_JSON[("C6", 0)].relative_to(REPO_ROOT)),
                    "soup_valid_mae": load_dictionary_entry(("C6", 0))["soup_valid_mae"],
                },
                "dense_tied_seed0_soup": load_dictionary_entry(("DENSE-TIED", 0))["soup_valid_mae"],
            },
            "round_seconds": float(time.perf_counter() - started),
        }
    )
    _write_json(out_path, payload)
    print(
        f"[stage-g-train] soup={payload['soup']['soup_valid_mae']:.6f} "
        f"best={payload['best_valid_mae']:.6f}@{payload['best_epoch']}",
        flush=True,
    )
    return payload


def stage_g_audit(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _guard_gate_fired()
    _ensure_dirs()
    out_path = TRAINING_DIR / "iht30_audit.json"
    if out_path.exists() and not force:
        print("[stage-g-audit] cache hit", flush=True)
        return _read_json(out_path)
    started = time.perf_counter()
    dca.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(int(threads))
    state_path = TRAINING_DIR / "FINAL-CLEAN-IHT30_seed0_soup_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    D = np.asarray(state["D"].detach().cpu(), dtype=np.float32)
    Dbar = dca.effective_dictionary(D)
    Dbar_torch = dca.effective_dictionary_torch(D)
    train_phi = load_phi("train")
    valid_phi = load_phi("valid")
    tables: dict[str, Any] = {}
    for split, X in (("valid", valid_phi), ("train", train_phi)):
        reference = dca.codes_for("omp", X, Dbar, Dbar_torch=Dbar_torch)
        tables[split] = {}
        for coder in dca.CODERS:
            A = reference if coder == "omp" else dca.codes_for(coder, X, Dbar, Dbar_torch=Dbar_torch)
            tables[split][coder] = _summary_row(
                X=np.asarray(X, dtype=np.float64),
                A=A,
                Dbar=Dbar,
                reference=None if coder == "omp" else reference,
            )
        del reference

    # ---- frozen mechanism sanity probes on the single IHT-30 checkpoint ----
    valid = p1run.load_split("valid")
    dictionary, _dict_sha = p2run.load_dictionary("sdb32")
    probe_model = build_iht_model(dictionary, 0, iht_steps=IHT30_STEPS)
    probe_model.load_state_dict({key: value.float() for key, value in state.items()})
    probe_model.eval()
    device = audit.attach_cpu(int(threads))
    loader = p1.make_env_loader(valid, int(p2run.BATCH_SIZE), False, int(p2run.EVAL_SHUFFLE_OFFSET))
    base = audit.evaluate_mask(probe_model, loader, device, cm.arm_mask(IHT30_SPEC))
    coord_zero = audit.evaluate_mask(
        probe_model, loader, device, cm.merge_masks(cm.arm_mask(IHT30_SPEC), audit.AuditMask(coord_zero=True))
    )
    node_indep = cm.evaluate_binding_variant(
        dictionary,
        state,
        loader,
        device,
        cm.arm_mask(IHT30_SPEC),
        node_binding="indep",
        edge_binding="paired",
        baseline_predictions=base["predictions"],
    )
    edge_indep = cm.evaluate_binding_variant(
        dictionary,
        state,
        loader,
        device,
        cm.arm_mask(IHT30_SPEC),
        node_binding="paired",
        edge_binding="indep",
        baseline_predictions=base["predictions"],
    )
    probes = {
        "baseline_valid_mae": float(base["mae"]),
        "dictionary_zero": {
            "valid_mae": float(coord_zero["mae"]),
            "delta_mae": float(coord_zero["mae"] - base["mae"]),
        },
        "node_assignment_correspondence": {
            "valid_mae": float(node_indep["valid_mae"]),
            "delta_mae": float(node_indep["valid_mae"] - base["mae"]),
            "note": "frozen independence-null replacement of the paired node statistic",
        },
        "edge_assignment_correspondence": {
            "valid_mae": float(edge_indep["valid_mae"]),
            "delta_mae": float(edge_indep["valid_mae"] - base["mae"]),
            "note": "frozen independence-null replacement of the paired edge statistic",
        },
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "seed": 0,
        "iht_steps": IHT30_STEPS,
        "tables": tables,
        "probes": probes,
        "round_seconds": float(time.perf_counter() - started),
    }
    _write_json(out_path, payload)
    print(
        f"[stage-g-audit] zero={probes['dictionary_zero']['delta_mae']:+.4f} "
        f"node={probes['node_assignment_correspondence']['delta_mae']:+.4f} "
        f"edge={probes['edge_assignment_correspondence']['delta_mae']:+.4f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([row.get(key, "") for key in header])


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.{digits}f}"


def stage_report() -> dict[str, Any]:
    _ensure_dirs()
    geometry = {seed: _read_json(CODER_DIR / f"seed{seed}" / "summary.json") for seed in dca.SEEDS}
    dominant = {seed: _read_json(DOMINANT_DIR / f"seed{seed}.json") for seed in dca.SEEDS}
    specialization = {seed: _read_json(SPECIALIZATION_DIR / f"seed{seed}.json") for seed in dca.SEEDS}
    stability = _read_json(STABILITY_DIR / "summary.json")
    recoverability = _read_json(RECOVERABILITY_DIR / "summary.json")
    gate = _read_json(RESULTS_DIR / "gate.json")
    lines: list[str] = ["# e2e_dictenv_dictionary_coder_audit_v1 — generated analysis tables", ""]
    lines.append("CPU only; official test never loaded. Generated by the round runner; do not edit.")
    lines.append("")
    lines.append("## 1. Same-D coder geometry (valid split)")
    lines.append("")
    lines.append(
        "| seed | coder | recon_frobenius | recon_mean_row_sq | active | Neff | top1 | top5 | max_rate | gini | Jaccard(OMP) | exact(OMP) | cos(OMP) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for seed in dca.SEEDS:
        for coder in dca.CODERS:
            table = geometry[seed]["tables"]["valid"][coder]
            support = table["support_vs_omp"]
            coefficient = table["coefficient_vs_omp"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    seed,
                    coder,
                    _fmt(table["reconstruction"]["recon_frobenius"], 6),
                    _fmt(table["reconstruction"]["recon_mean_row_squared"], 6),
                    table["concentration"]["active_atoms"],
                    _fmt(table["concentration"]["effective_atoms"], 3),
                    _fmt(table["concentration"]["top1_share"], 3),
                    _fmt(table["concentration"]["top5_share"], 3),
                    _fmt(table["concentration"]["max_activation_rate"], 3),
                    _fmt(table["concentration"]["gini"], 3),
                    _fmt(support["jaccard"]["mean"], 3) if support else "1.000",
                    _fmt(support["exact_support_match_rate"], 3) if support else "1.000",
                    _fmt(coefficient["cosine"]["mean"], 3) if coefficient else "1.000",
                )
            )
    lines.append("")
    lines.append("## 2. Per-row coefficient geometry (valid split, mean)")
    lines.append("")
    lines.append("| seed | coder | L1 | L2 | max|a| | top1/L1 | top3/L1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for seed in dca.SEEDS:
        for coder in dca.CODERS:
            table = geometry[seed]["tables"]["valid"][coder]["coefficient_geometry"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    seed,
                    coder,
                    _fmt(table["l1"]["mean"], 3),
                    _fmt(table["l2"]["mean"], 3),
                    _fmt(table["max_abs"]["mean"], 3),
                    _fmt(table["top1_over_l1"]["mean"], 3),
                    _fmt(table["top3_over_l1"]["mean"], 3),
                )
            )
    lines.append("")
    lines.append("## 3. Dominant atom / common direction")
    lines.append("")
    lines.append("| seed | coder | atom | act_rate | |cos(atom,mu)| | |cos(atom,PC1)| | PC1 var |")
    lines.append("|---|---|---|---|---|---|---|")
    for seed in dca.SEEDS:
        for coder in ("iht10", "iht30", "omp"):
            entry = dominant[seed][coder]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    seed,
                    coder,
                    entry["dominant_atom"],
                    _fmt(entry["dominant_atom_activation_rate"], 3),
                    _fmt(entry["dominant_atom_cosine_to_mean_direction"], 3),
                    _fmt(entry["dominant_atom_cosine_to_pcs"][0], 3),
                    _fmt(entry["pca_explained_variance_ratio"][0], 3),
                )
            )
    lines.append("")
    lines.append("## 4. Atom structural specialisation (train active population)")
    lines.append("")
    lines.append("| seed | coder | median Spec | usage-weighted Spec | median train-valid profile cos |")
    lines.append("|---|---|---|---|---|")
    for seed in dca.SEEDS:
        for coder in ("iht10", "omp", "iht30"):
            payload = specialization[seed]["coders"][coder]
            cosines = [
                row["train_valid_profile_cosine"]
                for row in payload["rows"]
                if np.isfinite(row["train_valid_profile_cosine"])
            ]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    seed,
                    coder,
                    _fmt(payload["median_specialization"], 3),
                    _fmt(payload["usage_weighted_specialization"], 3),
                    _fmt(np.median(cosines) if cosines else float("nan"), 3),
                )
            )
    lines.append("")
    lines.append("## 5. Cross-seed dictionary stability")
    lines.append("")
    lines.append("| pair | mean_abs_cos | median | min | p10 | n>=0.90 | n>=0.95 | median profile cos |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pair in stability["pairs"]:
        lines.append(
            "| {}<->{} | {} | {} | {} | {} | {} | {} | {} |".format(
                pair["seed_a"],
                pair["seed_b"],
                _fmt(pair["mean_matched_abs_cosine"], 3),
                _fmt(pair["median_matched_abs_cosine"], 3),
                _fmt(pair["min_matched_abs_cosine"], 3),
                _fmt(pair["p10_matched_abs_cosine"], 3),
                pair["n_above_090"],
                pair["n_above_095"],
                _fmt(pair["median_matched_profile_cosine"], 3),
            )
        )
    lines.append("")
    lines.append(f"Vocabulary verdict: **{stability['verdict']['verdict']}**.")
    lines.append("")
    lines.append("## 6. Sparse <-> dense recoverability (valid, train-only fit)")
    lines.append("")
    lines.append("| seed | source -> target | R2 | per-dim R2 median | normalized error | mean cosine | CKA |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in recoverability["rows"]:
        lines.append(
            "| {} | {} -> {} | {} | {} | {} | {} | {} |".format(
                row["seed"],
                row["source_representation"],
                row["target_representation"],
                _fmt(row["valid_r2"], 4),
                _fmt(row["per_dimension_r2"]["median"], 4),
                _fmt(row["valid_normalized_error"], 4),
                _fmt(row["valid_mean_cosine"]["mean"], 4),
                _fmt(row["linear_cka"], 4),
            )
        )
    lines.append("")
    lines.append("## 7. Frozen IHT-30 training gate")
    lines.append("")
    lines.append(f"Gate fired: **{gate['gate']['fired']}** (supported seeds {gate['gate']['supported_seeds']}).")
    lines.append(f"Classification: {gate['classification']['fired'] or ['NONE']}.")
    lines.append("")
    for seed, evidence in sorted(gate["gate"]["evidence"].items()):
        lines.append(
            "- seed {seed}: cond1(recon)={c1} factor={f:.3f} drop={d:.2e}; cond2={c2} "
            "(support={s}, concentration={cc}, geometry={g}).".format(
                seed=seed,
                c1=evidence["condition_1_reconstruction"],
                f=evidence["recon_factor"],
                d=evidence["recon_drop"],
                c2=evidence["condition_2"],
                s=evidence["condition_2a_support"],
                cc=evidence["condition_2b_concentration"],
                g=evidence["condition_2c_geometry"],
            )
        )
    if (TRAINING_DIR / "FINAL-CLEAN-IHT30_seed0_e320.json").exists():
        training = _read_json(TRAINING_DIR / "FINAL-CLEAN-IHT30_seed0_e320.json")
        lines.append("")
        lines.append("## 8. The single IHT-30 seed-0 training run")
        lines.append("")
        lines.append(
            "| arm | valid soup | best | best epoch | params | official_test_loaded |"
        )
        lines.append("|---|---|---|---|---|---|")
        reference = _read_json(CHECKPOINT_JSON[("C6", 0)])
        dense = _read_json(CHECKPOINT_JSON[("DENSE-TIED", 0)])
        lines.append(
            f"| IHT10 sparse seed 0 [existing] | {_fmt(reference['soup']['soup_valid_mae'], 6)} | "
            f"{_fmt(reference['best_valid_mae'], 6)} | {reference['best_epoch']} | "
            f"{reference['actual_params']} | false |"
        )
        lines.append(
            f"| IHT30 sparse seed 0 [new] | {_fmt(training['soup']['soup_valid_mae'], 6)} | "
            f"{_fmt(training['best_valid_mae'], 6)} | {training['best_epoch']} | "
            f"{training['actual_params']} | false |"
        )
        lines.append(
            f"| DenseTied seed 0 [existing] | {_fmt(dense['soup']['soup_valid_mae'], 6)} | "
            f"{_fmt(dense['best_valid_mae'], 6)} | {dense['best_epoch']} | {dense['actual_params']} | false |"
        )
        if (TRAINING_DIR / "iht30_audit.json").exists():
            audit_payload = _read_json(TRAINING_DIR / "iht30_audit.json")
            lines.append("")
            lines.append(
                "IHT30 audit valid: "
                f"fro={_fmt(audit_payload['tables']['valid']['iht30']['reconstruction']['recon_frobenius'], 6)}, "
                f"Neff={_fmt(audit_payload['tables']['valid']['iht30']['concentration']['effective_atoms'], 3)}; "
                f"OMP Neff={_fmt(audit_payload['tables']['valid']['omp']['concentration']['effective_atoms'], 3)}."
            )
            lines.append("")
            lines.append(
                "Frozen probes: zero delta "
                f"{_fmt(audit_payload['probes']['dictionary_zero']['delta_mae'])}; "
                f"node indep delta {_fmt(audit_payload['probes']['node_assignment_correspondence']['delta_mae'])}; "
                f"edge indep delta {_fmt(audit_payload['probes']['edge_assignment_correspondence']['delta_mae'])}."
            )
    (RESULTS_DIR / "analysis_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "gate": gate,
        "stability_verdict": stability["verdict"],
        "recoverability": recoverability["rows"],
    }
    _write_json(RESULTS_DIR / "summary.json", summary)
    print("[report] analysis_tables.md written", flush=True)
    return summary


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def launch_stage_a(threads: int = THREADS, concurrency: int = CONCURRENCY, force: bool = False) -> None:
    queue = [seed for seed in dca.SEEDS if force or not (CODER_DIR / f"seed{seed}/summary.json").exists()]
    if not queue:
        print("[launch] all stage-a outputs present", flush=True)
        return
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(threads))
    env["MKL_NUM_THREADS"] = str(int(threads))
    env["CUDA_VISIBLE_DEVICES"] = ""
    running: list[tuple[int, subprocess.Popen]] = []
    while queue or running:
        while queue and len(running) < max(1, int(concurrency)):
            seed = queue.pop(0)
            command = [
                sys.executable,
                "-m",
                "tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_dictionary_coder_audit_v1",
                "stage-a",
                "--seed",
                str(seed),
                "--threads",
                str(int(threads)),
            ]
            if force:
                command.append("--force")
            print(f"[launch] {' '.join(command)}", flush=True)
            running.append((seed, subprocess.Popen(command, env=env)))
        time.sleep(1.0)
        for seed, process in list(running):
            if process.poll() is not None:
                running.remove((seed, process))
                if process.returncode != 0:
                    raise RuntimeError(f"stage-a seed {seed} failed with exit code {process.returncode}")


def chain(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    stage_preflight()
    launch_stage_a(threads=threads, concurrency=concurrency)
    stage_d()
    stage_e()
    gate = stage_gate()
    if bool(gate["gate"]["fired"]):
        print("[chain] gate fired -> single IHT-30 seed-0 sanity + formal run", flush=True)
        stage_g_sanity(threads=threads)
        stage_g_train(threads=threads)
        stage_g_audit(threads=threads)
    else:
        print("[chain] gate did not fire -> NO NEW TRAINING WAS SCIENTIFICALLY JUSTIFIED", flush=True)
    stage_report()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "preflight",
            "stage-a",
            "stage-d",
            "stage-e",
            "gate",
            "stage-g-sanity",
            "stage-g-train",
            "stage-g-audit",
            "report",
            "chain",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        stage_preflight()
    elif args.stage == "stage-a":
        stage_a(int(args.seed), threads=int(args.threads), force=bool(args.force))
    elif args.stage == "stage-d":
        stage_d()
    elif args.stage == "stage-e":
        stage_e()
    elif args.stage == "gate":
        stage_gate()
    elif args.stage == "stage-g-sanity":
        stage_g_sanity(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "stage-g-train":
        stage_g_train(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "stage-g-audit":
        stage_g_audit(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "report":
        stage_report()
    elif args.stage == "chain":
        chain(threads=int(args.threads), concurrency=int(args.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
