"""CSSD round runner — common-subspace-separated structural dictionary.

Round ``e2e_dictenv_common_subspace_dictionary_v1`` (Workstream Z, ZINC).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_common_subspace_dictionary_v1.py``.
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_common_subspace_dictionary_v1_preregistration.md``.

Stages
------
``preflight  stage-a  stage-b  stage-c  select  stage-train  stage-g  report
  chain``

Zero-training until the frozen q1/q2 representation gate passes.  At most one
new training trajectory (CSSD seed 0, 1 -> 40 -> 320).  CPU only; the official
ZINC test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_dictionary_coder_audit_v1 as dca
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_dictionary_coder_audit_v1 as dcarun
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = cssd.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_common_subspace_dictionary_v1_preregistration.md"

ZERO_DIR = RESULTS_DIR / "zero_training"
TRAINING_DIR = RESULTS_DIR / "training"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
STRUCTURE_DIR = RESULTS_DIR / "reusable_structure"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

THREADS = 4
CONCURRENCY = 3
TRAIN_EPOCHS = 320
GATE_EPOCH = 40
CSSD_SEED = 0

SEEDS: tuple[int, ...] = (0, 1, 2)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, ZERO_DIR, TRAINING_DIR, CHECKPOINT_DIR, STRUCTURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([row.get(key, "") for key in header])


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    cssd.cpu_only_guard(torch.device("cpu"))
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    audit_dir = TRACK_ROOT / "results/e2e_dictenv_dictionary_coder_audit_v1"
    geometry = _read_json(audit_dir / "coder_geometry/seed0/summary.json")
    table = geometry["tables"]["valid"]["iht10"]["concentration"]
    specialization = _read_json(audit_dir / "atom_specialization/seed0.json")
    weighted = specialization["coders"]["iht10"]["usage_weighted_specialization"]
    reference_checks = {
        "effective_atoms": {
            "stored": float(table["effective_atoms"]),
            "frozen": cssd.REF_NEFF_VALID,
            "ok": abs(float(table["effective_atoms"]) - cssd.REF_NEFF_VALID)
            <= cssd.REFERENCE_TOLERANCE,
        },
        "top5_share": {
            "stored": float(table["top5_share"]),
            "frozen": cssd.REF_TOP5_VALID,
            "ok": abs(float(table["top5_share"]) - cssd.REF_TOP5_VALID)
            <= cssd.REFERENCE_TOLERANCE,
        },
        "usage_weighted_spec": {
            "stored": float(weighted),
            "frozen": cssd.REF_WEIGHTED_SPEC,
            "ok": abs(float(weighted) - cssd.REF_WEIGHTED_SPEC) <= cssd.REFERENCE_TOLERANCE,
        },
    }
    if not all(entry["ok"] for entry in reference_checks.values()):
        raise RuntimeError(f"frozen reference provenance check failed: {reference_checks}")
    checkpoints: dict[str, Any] = {}
    for key in (("C6", 0), ("C6", 1), ("C6", 2), ("DENSE-TIED", 0)):
        entry = dcarun.load_dictionary_entry(key)
        checkpoints[f"{key[0]}_seed{key[1]}"] = {
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "soup_valid_mae": entry["soup_valid_mae"],
            "dictionary_shape": list(entry["dictionary"].shape),
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "preregistration": {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(PREREG_PATH),
        },
        "reference_checks": reference_checks,
        "reference_values": {
            "ref_neff_valid": cssd.REF_NEFF_VALID,
            "ref_top5_valid": cssd.REF_TOP5_VALID,
            "ref_weighted_spec": cssd.REF_WEIGHTED_SPEC,
            "ref_dc_count_gt095": cssd.REF_DC_COUNT_GT095,
            "ref_sparse_seed0_soup_mae": cssd.REF_SPARSE_SEED0_SOUP_MAE,
            "ref_dense_seed0_soup_mae": cssd.REF_DENSE_SEED0_SOUP_MAE,
        },
        "checkpoints": checkpoints,
        "gate_epoch": GATE_EPOCH,
        "epoch_budget": TRAIN_EPOCHS,
        "seed": CSSD_SEED,
    }
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", payload["preregistration"])
    _write_json(RESULTS_DIR / "preflight.json", payload)
    print(
        f"[preflight] prereg {payload['preregistration']['sha256'][:12]} "
        f"checkpoints={len(checkpoints)} references ok",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: a — feature space audit
# ---------------------------------------------------------------------------


def stage_a() -> dict[str, Any]:
    _ensure_dirs()
    cssd.cpu_only_guard(torch.device("cpu"))
    train_phi = dcarun.load_phi("train")
    valid_phi = dcarun.load_phi("valid")
    zero_columns = [
        int(index) for index in range(cssd.PHI_DIM) if np.abs(train_phi[:, index]).max() == 0.0
    ]
    named_zero = {
        "root_neighbour_shell1_column_5": bool(np.abs(train_phi[:, 5]).max() == 0.0),
        "root_walk1_column_8": bool(np.abs(train_phi[:, 8]).max() == 0.0),
    }
    # pipeline identity: data.dict_phi is exactly this cache, slice by slice.
    train_data = p1run.load_split("train", subset=8)
    blob = torch.load(
        p1run._env_cache_path("train"), map_location="cpu", weights_only=False
    )
    node_sizes = blob["node_sizes"].numpy()
    pipeline_checks: list[dict[str, Any]] = []
    offset = 0
    for index, data in enumerate(train_data):
        size = int(node_sizes[index])
        reference = blob["phi"][offset : offset + size]
        actual = data.dict_phi
        pipeline_checks.append(
            {
                "graph": int(index),
                "rows": int(size),
                "max_abs_diff": float((reference - actual).abs().max()),
                "bit_identical": bool(torch.equal(reference, actual)),
            }
        )
        offset += size
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "phi": {
            "train": {
                "shape": list(train_phi.shape),
                "dtype": str(train_phi.dtype),
                "cache": str(p1run._env_cache_path("train").relative_to(REPO_ROOT)),
                "min": float(train_phi.min()),
                "max": float(train_phi.max()),
            },
            "valid": {
                "shape": list(valid_phi.shape),
                "dtype": str(valid_phi.dtype),
                "cache": str(p1run._env_cache_path("valid").relative_to(REPO_ROOT)),
            },
            "normalization_between_cache_and_dictionary": "none",
            "note": (
                "P2Model.code consumes data.dict_phi directly; the anchor scaler "
                "only standardizes data.anchor"
            ),
        },
        "zero_columns": zero_columns,
        "named_zero_descriptors": named_zero,
        "zero_column_note": (
            "identically-zero descriptors are recorded but NOT removed; the round "
            "keeps the frozen 65-D layout"
        ),
        "pipeline_checks": pipeline_checks,
        "pipeline_bit_identical": bool(all(entry["bit_identical"] for entry in pipeline_checks)),
    }
    if not payload["pipeline_bit_identical"]:
        raise RuntimeError("dictionary input cache does not match the dataset dict_phi field")
    if not (named_zero["root_neighbour_shell1_column_5"] and named_zero["root_walk1_column_8"]):
        raise RuntimeError("named zero descriptors are not identically zero")
    _write_json(RESULTS_DIR / "stage_a.json", payload)
    print(
        f"[stage-a] train {train_phi.shape} valid {valid_phi.shape} "
        f"zero_columns={zero_columns} pipeline_ok",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: b — train-only common subspace
# ---------------------------------------------------------------------------


def stage_b() -> dict[str, Any]:
    _ensure_dirs()
    cssd.cpu_only_guard(torch.device("cpu"))
    train_phi = dcarun.load_phi("train")
    valid_phi = dcarun.load_phi("valid")
    X = np.asarray(train_phi, dtype=np.float64)
    mu = X.mean(axis=0)
    sub1 = cssd.build_common_subspace(train_phi, 1)
    sub2 = cssd.build_common_subspace(train_phi, 2)
    # PCA diagnostics in the same space (train only)
    pca_components, pca_ratio, _pca_mean = dca.pca_basis(train_phi, n_components=3)
    energies = {
        "q1_train": cssd.subspace_energy(train_phi, sub1),
        "q1_valid": cssd.subspace_energy(valid_phi, sub1),
        "q2_train": cssd.subspace_energy(train_phi, sub2),
        "q2_valid": cssd.subspace_energy(valid_phi, sub2),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "fit_split": "official train",
        "n_train": int(X.shape[0]),
        "n_valid": int(valid_phi.shape[0]),
        "q1": sub1.as_dict(),
        "q2": sub2.as_dict(),
        "mean_direction_norm": float(np.linalg.norm(mu)),
        "pca_ratio_top3": [float(value) for value in pca_ratio],
        "u1_cosine_pc1": float(abs(sub1.components[:, 0] @ pca_components[0])),
        "energies": energies,
        "orthonormality_max_error": {
            "q1": float(np.abs(sub1.components.T @ sub1.components - np.eye(1)).max()),
            "q2": float(np.abs(sub2.components.T @ sub2.components - np.eye(2)).max()),
        },
    }
    _write_json(RESULTS_DIR / "common_subspace.json", payload)
    print(
        f"[stage-b] ||mu||={payload['mean_direction_norm']:.4f} "
        f"E_common(q1, train)={energies['q1_train']['common_energy_fraction']:.4f} "
        f"E_centered_residual(q1, train)="
        f"{energies['q1_train']['centered_residual_fraction']:.4f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: c — zero-training same-D probe
# ---------------------------------------------------------------------------


def stage_c() -> dict[str, Any]:
    _ensure_dirs()
    cssd.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    train_phi = dcarun.load_phi("train")
    valid_phi = dcarun.load_phi("valid")
    X = np.asarray(train_phi, dtype=np.float64)
    mean_vector = X.mean(axis=0)
    sub1 = cssd.build_common_subspace(train_phi, 1)
    sub2 = cssd.build_common_subspace(train_phi, 2)

    probe: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    energy: dict[int, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        entry = dcarun.load_dictionary_entry(("C6", seed))
        Dbar = dca.effective_dictionary(entry["dictionary"])
        Dbar_torch = dca.effective_dictionary_torch(entry["dictionary"])
        probe[seed] = {}
        energy[seed] = {}
        for split, phi in (("train", train_phi), ("valid", valid_phi)):
            variants = cssd.descriptor_variants(phi, mean_vector, sub1, sub2)
            raw_codes_seed = dca.codes_for("iht10", variants["RAW"], Dbar, Dbar_torch=Dbar_torch)
            for variant in cssd.VARIANTS:
                target = variants[variant]
                codes = (
                    raw_codes_seed
                    if variant == "RAW"
                    else dca.codes_for("iht10", target, Dbar, Dbar_torch=Dbar_torch)
                )
                subspace = sub1 if variant in ("RAW", "MC", "Q1") else sub2
                row = cssd.probe_variant_metrics(
                    seed=seed,
                    variant=variant,
                    split=split,
                    raw=variants["RAW"],
                    target=target,
                    codes=codes,
                    reference_codes=raw_codes_seed,
                    Dbar=Dbar,
                    subspace=subspace,
                )
                row["checkpoint_soup_valid_mae"] = float(entry["soup_valid_mae"])
                probe[seed].setdefault(variant, {})[split] = row
                if variant in ("Q1", "Q2"):
                    energy[seed].setdefault(variant, {})[split] = cssd.subspace_energy(
                        phi, subspace
                    )
                else:
                    energy[seed].setdefault(variant, {})[split] = {
                        "centered_residual_fraction": float("nan")
                    }
                rows.append(row)
                print(
                    f"[stage-c] seed={seed} split={split} variant={variant} "
                    f"top5={row['top5_usage']:.3f} neff={row['effective_atoms']:.2f} "
                    f"a6={row['atom6_rate']:.3f} a24={row['atom24_rate']:.3f} "
                    f"a27={row['atom27_rate']:.3f} a23={row['atom23_rate']:.3f}",
                    flush=True,
                )
                if variant != "RAW":
                    comparison_rows.append(
                        {
                            "seed": seed,
                            "split": split,
                            "variant": variant,
                            "support_jaccard_vs_raw": row["support_jaccard_vs_raw"],
                            "code_cosine_vs_raw": row["code_cosine_vs_raw"],
                            "top5_drop_vs_raw": float(
                                probe[seed]["RAW"][split]["top5_usage"] - row["top5_usage"]
                            ),
                            "dc_rates_vs_raw": ";".join(
                                f"{atom}:{row[f'atom{atom}_rate']:.4f}"
                                for atom in cssd.DC_TRIPLET
                            ),
                        }
                    )
    _write_json(ZERO_DIR / "probe.json", {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "probe": {str(seed): payload for seed, payload in probe.items()},
        "energy": {str(seed): payload for seed, payload in energy.items()},
    })
    header = [
        "seed",
        "split",
        "variant",
        "common_dim",
        "common_energy_fraction",
        "residual_energy_fraction",
        "reconstruction_error",
        "recon_mean_row_squared",
        "active_atoms",
        "effective_atoms",
        "top1_usage",
        "top3_usage",
        "top5_usage",
        "top8_usage",
        "max_activation_rate",
        "n_atoms_gt_090",
        "n_atoms_gt_095",
        "n_atoms_gt_099",
        "atom6_rate",
        "atom23_rate",
        "atom24_rate",
        "atom27_rate",
        "usage_entropy",
        "gini",
        "support_jaccard_vs_raw",
        "code_cosine_vs_raw",
    ]
    for variant, name in (("RAW", "raw"), ("MC", "mean_center"), ("Q1", "q1"), ("Q2", "q2")):
        _write_csv(
            ZERO_DIR / f"{name}.csv",
            [row for row in rows if row["variant"] == variant],
            header,
        )
    _write_csv(
        ZERO_DIR / "comparison.csv",
        comparison_rows,
        [
            "seed",
            "split",
            "variant",
            "support_jaccard_vs_raw",
            "code_cosine_vs_raw",
            "top5_drop_vs_raw",
            "dc_rates_vs_raw",
        ],
    )
    return {
        "probe": probe,
        "energy": energy,
        "rows": rows,
        "probe_path": str((ZERO_DIR / "probe.json").relative_to(REPO_ROOT)),
    }


def stage_select() -> dict[str, Any]:
    _ensure_dirs()
    payload = _read_json(ZERO_DIR / "probe.json")
    probe = {int(seed): entry for seed, entry in payload["probe"].items()}
    energy = {int(seed): entry for seed, entry in payload["energy"].items()}
    selection = cssd.selection_rule(probe=probe, energy=energy, split="valid")
    selection_train = cssd.selection_rule(probe=probe, energy=energy, split="train")
    selection["train_view"] = selection_train
    selection["train_valid_agree"] = bool(selection["selected"] == selection_train["selected"])
    _write_json(ZERO_DIR / "selection.json", selection)
    print(
        f"[select] q1={selection['q1_passed_seeds']} q2={selection['q2_passed_seeds']} "
        f"selected={selection['selected']}",
        flush=True,
    )
    return selection


# ---------------------------------------------------------------------------
# training with the frozen epoch-40 gate
# ---------------------------------------------------------------------------


def _load_subspace(payload: Mapping[str, Any], kind: str) -> cssd.CommonSubspace:
    entry = payload[kind]
    return cssd.CommonSubspace(
        components=np.asarray(entry["components"], dtype=np.float64),
        rms=np.asarray(entry["rms"], dtype=np.float64),
        kind=str(entry["kind"]),
    )


def stage_train(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = TRAINING_DIR / "final.json"
    if final_path.exists() and not force:
        print("[train] cache hit", flush=True)
        return _read_json(final_path)
    selection = _read_json(ZERO_DIR / "selection.json")
    selected = selection.get("selected")
    if selected is None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "run": False,
            "reason": "COMMON_SUBSPACE_SEPARATION_NOT_SUPPORTED",
            "selection": selection,
        }
        _write_json(TRAINING_DIR / "NOT_RUN.json", payload)
        print("[train] selection gate failed -> NOT_RUN", flush=True)
        return payload
    subspace_payload = _read_json(RESULTS_DIR / "common_subspace.json")
    subspace = _load_subspace(subspace_payload, selected)

    train_phi = dcarun.load_phi("train")
    valid_phi = dcarun.load_phi("valid")
    Z_train = dca.structural_descriptors(train_phi)
    Z_valid = dca.structural_descriptors(valid_phi)
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    dictionary_init, _dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    device = audit.attach_cpu(int(threads))
    gradient_loader = p1.make_env_loader(train_data, int(p2run.BATCH_SIZE), False, 0)
    gradient_batch = next(iter(gradient_loader))
    initial_dictionary = torch.as_tensor(np.asarray(dictionary_init, dtype=np.float32)).clone()
    started = time.perf_counter()

    def epoch_callback(epoch: int, model: cssd.CSSDModel, optimizer: Any, curve: list[dict[str, Any]]):
        if epoch != GATE_EPOCH:
            return True, None
        started_cb = time.perf_counter()
        alpha_train = cssd.residual_codes(model, train_phi)
        alpha_valid = cssd.residual_codes(model, valid_phi)
        usage_train = cssd.usage_payload(alpha_train)
        usage_valid = cssd.usage_payload(alpha_valid)
        specialization = dca.atom_specialization_rows(
            seed=CSSD_SEED,
            coder="cssd",
            Z_train=Z_train,
            Z_valid=Z_valid,
            A_train=alpha_train,
            A_valid=alpha_valid,
        )
        frequencies = np.asarray(usage_valid["frequencies"], dtype=np.float64)
        dc_count = int((frequencies[list(cssd.DC_TRIPLET)] > 0.95).sum())
        # condition C: task gradient on a frozen train batch, evaluated in eval
        # mode so the callback cannot perturb the training RNG stream.
        model.train(False)
        batch = gradient_batch.to(device)
        prediction, aux = model(batch, mask=cssd.CSSD_MASK, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(
            cm.H1_LAMBDA
        ) * rec
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(model.D.grad.detach().norm()) if model.D.grad is not None else 0.0
        model.zero_grad(set_to_none=True)
        model.train(True)
        with torch.no_grad():
            projected = np.asarray(model.D.detach().cpu(), dtype=np.float64)
            U = subspace.components
            projected = projected - U @ (U.T @ projected)
            projected_norms = np.linalg.norm(projected, axis=0)
            Dbar_perp = dca.effective_dictionary(projected)
            u_t_d_max = float(np.abs(U.T @ Dbar_perp).max())
        movement = float((model.D.detach().cpu() - initial_dictionary).norm())
        row = curve[-1]
        catastrophic = cssd.catastrophic_check(
            train_mae=float(row["train_mae"]),
            valid_mae=float(row["valid_mae"]),
            rec_loss=float(row.get("train_rec_term", row["train_rec"])),
        )
        gate = cssd.epoch40_gate(
            dc_count_gt095=dc_count,
            top5_share=float(usage_valid["top5_share"]),
            effective_atoms=float(usage_valid["effective_atoms"]),
            usage_weighted_spec=float(specialization["usage_weighted_specialization"]),
            gradient_norm_D=float(gradient_norm),
            column_norm_min=float(projected_norms.min()),
        )
        basic_health = {
            "finite_losses": bool(
                np.isfinite(row["train_mae"])
                and np.isfinite(row["train_rec"])
                and np.isfinite(row.get("train_rec_term", row["train_rec"]))
                and np.isfinite(row["valid_mae"])
            ),
            "finite_dictionary": bool(torch.isfinite(model.D).all()),
            "finite_predictions": bool(np.isfinite(row["valid_mae"])),
            "dictionary_movement_frobenius": movement,
            "dictionary_moved": bool(movement > 0),
            "projected_column_norm_min": float(projected_norms.min()),
            "U_T_Dbar_perp_max_abs": u_t_d_max,
        }
        keep_going = bool(
            gate["passed"]
            and not catastrophic["triggered"]
            and basic_health["finite_losses"]
            and basic_health["finite_dictionary"]
            and basic_health["dictionary_moved"]
        )
        verdict = gate["verdict"]
        if catastrophic["triggered"]:
            verdict = "CSSD_CATASTROPHIC_STOP"
        elif not keep_going and gate["passed"]:
            verdict = "CSSD_TRAINING_BROKEN"
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "epoch": int(epoch),
            "selected_common_dim": str(selected),
            "basic_health": basic_health,
            "catastrophic": catastrophic,
            "usage_valid": {key: value for key, value in usage_valid.items() if key != "frequencies"},
            "usage_train": {key: value for key, value in usage_train.items() if key != "frequencies"},
            "valid_frequencies": usage_valid["frequencies"],
            "specialization": {
                "median_specialization": float(specialization["median_specialization"]),
                "usage_weighted_specialization": float(
                    specialization["usage_weighted_specialization"]
                ),
            },
            "gradient_norm_D": float(gradient_norm),
            "projected_column_norm_min": float(projected_norms.min()),
            "gate": gate,
            "verdict": verdict,
            "callback_seconds": float(time.perf_counter() - started_cb),
        }
        print(
            f"[gate@{epoch}] dc_gt095={dc_count} top5={usage_valid['top5_share']:.3f} "
            f"neff={usage_valid['effective_atoms']:.2f} "
            f"wSpec={specialization['usage_weighted_specialization']:.4f} "
            f"grad={gradient_norm:.3e} -> {verdict}",
            flush=True,
        )
        return keep_going, payload

    tag = f"CSSD-{selected.upper()}-seed{CSSD_SEED}"
    payload = cssd.train_cssd(
        tag=tag,
        epochs=TRAIN_EPOCHS,
        threads=int(threads),
        out_dir=CHECKPOINT_DIR,
        subspace=subspace,
        train_data=train_data,
        valid_data=valid_data,
        seed=CSSD_SEED,
        callback=epoch_callback,
        save_states=True,
        log=True,
    )
    payload.update(
        {
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "seed": CSSD_SEED,
            "selected_common_dim": str(selected),
            "selection": {
                "q1_passed_seeds": selection["q1_passed_seeds"],
                "q2_passed_seeds": selection["q2_passed_seeds"],
                "selected": selected,
            },
            "round_seconds": float(time.perf_counter() - started),
        }
    )
    _write_csv(
        TRAINING_DIR / "curve.csv",
        payload["curve"],
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    gate_payload = payload.get("callback_payloads", {}).get(str(GATE_EPOCH))
    if gate_payload is not None:
        _write_json(TRAINING_DIR / "epoch40_gate.json", gate_payload)
    _write_json(final_path, payload)
    if not payload["completed"]:
        _write_json(
            TRAINING_DIR / "STOPPED_AT_40.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "official_test_loaded": False,
                "reason": "epoch-40 gate did not pass",
                "verdict": (gate_payload or {}).get("verdict"),
                "epochs_run": payload["epochs_run"],
            },
        )
    print(
        f"[train] completed={payload['completed']} epochs={payload['epochs_run']} "
        f"best={payload['best_valid_mae']:.6f}@{payload['best_epoch']} "
        f"soup={payload['soup']['soup_valid_mae']:.6f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: g — reusable-structure audit
# ---------------------------------------------------------------------------


def _graph_ids(split: str) -> np.ndarray:
    blob = torch.load(p1run._env_cache_path(split), map_location="cpu", weights_only=False)
    sizes = blob["node_sizes"].numpy().astype(np.int64)
    return np.repeat(np.arange(sizes.shape[0], dtype=np.int64), sizes)


def _coverage_rows(
    *, seed: int, dictionary_label: str, codes_train: np.ndarray, codes_valid: np.ndarray,
    ids_train: np.ndarray, ids_valid: np.ndarray, n_train: int, n_valid: int,
) -> list[dict[str, Any]]:
    train_stats = cssd.graph_coverage(codes_train, ids_train, n_train)
    valid_stats = cssd.graph_coverage(codes_valid, ids_valid, n_valid)
    rows: list[dict[str, Any]] = []
    for atom in range(codes_train.shape[1]):
        rows.append(
            {
                "seed": seed,
                "dictionary": dictionary_label,
                "atom": atom,
                "coverage_train": train_stats["coverage"][atom],
                "coverage_valid": valid_stats["coverage"][atom],
                "n_active_graphs_train": int(train_stats["n_active_graphs"][atom]),
                "n_active_graphs_valid": int(valid_stats["n_active_graphs"][atom]),
                "mean_activations_per_active_graph_train": train_stats[
                    "mean_activations_per_active_graph"
                ][atom],
                "median_activations_per_active_graph_train": train_stats[
                    "median_activations_per_active_graph"
                ][atom],
                "top10_graph_share_train": train_stats["top10_graph_share"][atom],
                "mean_activations_per_active_graph_valid": valid_stats[
                    "mean_activations_per_active_graph"
                ][atom],
                "median_activations_per_active_graph_valid": valid_stats[
                    "median_activations_per_active_graph"
                ][atom],
                "top10_graph_share_valid": valid_stats["top10_graph_share"][atom],
            }
        )
    return rows


def stage_g(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out_path = STRUCTURE_DIR / "summary.json"
    final_path = TRAINING_DIR / "final.json"
    if not final_path.exists():
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "ran": False,
            "reason": "training not run (selection gate)",
        }
        _write_json(out_path, payload)
        return payload
    final = _read_json(final_path)
    if not final.get("completed"):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "ran": False,
            "reason": "training did not complete (epoch-40 gate)",
        }
        _write_json(out_path, payload)
        return payload
    if out_path.exists() and not force:
        print("[stage-g] cache hit", flush=True)
        return _read_json(out_path)
    cssd.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(int(threads))
    train_phi = dcarun.load_phi("train")
    valid_phi = dcarun.load_phi("valid")
    Z_train = dca.structural_descriptors(train_phi)
    Z_valid = dca.structural_descriptors(valid_phi)
    ids_train = _graph_ids("train")
    ids_valid = _graph_ids("valid")
    n_train = int(ids_train.max()) + 1
    n_valid = int(ids_valid.max()) + 1

    # trained CSSD soup
    state_path = CHECKPOINT_DIR / f"CSSD-{final['selected_common_dim'].upper()}-seed0_soup_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    subspace = _load_subspace(_read_json(RESULTS_DIR / "common_subspace.json"), final["selected_common_dim"])
    dictionary_init, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    model = cssd.build_cssd_model(dictionary_init, CSSD_SEED, subspace)
    model.load_state_dict({key: value.float() for key, value in state.items()})
    alpha_train = cssd.residual_codes(model, train_phi)
    alpha_valid = cssd.residual_codes(model, valid_phi)
    spec = dca.atom_specialization_rows(
        seed=CSSD_SEED, coder="cssd", Z_train=Z_train, Z_valid=Z_valid,
        A_train=alpha_train, A_valid=alpha_valid,
    )
    Dbar_perp = model.residual_dictionary().detach().double().numpy()
    coherence_cssd = cssd.atom_coherence(Dbar_perp)

    # RAW seed-0 reference
    raw_entry = dcarun.load_dictionary_entry(("C6", 0))
    Draw = raw_entry["dictionary"]
    Dbar_raw = dca.effective_dictionary(Draw)
    raw_train = cssd.raw_codes(train_phi, Draw)
    raw_valid = cssd.raw_codes(valid_phi, Draw)
    raw_spec = dca.atom_specialization_rows(
        seed=0, coder="iht10-raw", Z_train=Z_train, Z_valid=Z_valid,
        A_train=raw_train, A_valid=raw_valid,
    )
    coherence_raw = cssd.atom_coherence(Dbar_raw)

    # per-atom profiles
    usage_train = dca.activation_frequency(alpha_train)
    usage_valid = dca.activation_frequency(alpha_valid)
    cov_train = cssd.graph_coverage(alpha_train, ids_train, n_train)
    cov_valid = cssd.graph_coverage(alpha_valid, ids_valid, n_valid)
    profile_rows: list[dict[str, Any]] = []
    for atom in range(cssd.K_ATOMS):
        row = spec["rows"][atom]
        profile_rows.append(
            {
                "seed": CSSD_SEED,
                "atom": atom,
                "activation_rate_train": float(usage_train[atom]),
                "activation_rate_valid": float(usage_valid[atom]),
                "train_graph_coverage": cov_train["coverage"][atom],
                "valid_graph_coverage": cov_valid["coverage"][atom],
                "specialization_train": row["specialization_score"],
                "specialization_valid": row["specialization_score_valid"],
                "profile_cosine": row["train_valid_profile_cosine"],
                "top_response_specialization": row["top_response_specialization_score"],
                "mean_activations_per_active_graph_train": cov_train[
                    "mean_activations_per_active_graph"
                ][atom],
                "top10_graph_share_train": cov_train["top10_graph_share"][atom],
                "top_structural_features": ";".join(row["top_structural_features"]),
            }
        )
    _write_csv(
        STRUCTURE_DIR / "atom_profiles.csv",
        profile_rows,
        [
            "seed",
            "atom",
            "activation_rate_train",
            "activation_rate_valid",
            "train_graph_coverage",
            "valid_graph_coverage",
            "specialization_train",
            "specialization_valid",
            "profile_cosine",
            "top_response_specialization",
            "mean_activations_per_active_graph_train",
            "top10_graph_share_train",
            "top_structural_features",
        ],
    )
    coverage_rows = _coverage_rows(
        seed=CSSD_SEED, dictionary_label="CSSD", codes_train=alpha_train, codes_valid=alpha_valid,
        ids_train=ids_train, ids_valid=ids_valid, n_train=n_train, n_valid=n_valid,
    ) + _coverage_rows(
        seed=0, dictionary_label="RAW", codes_train=raw_train, codes_valid=raw_valid,
        ids_train=ids_train, ids_valid=ids_valid, n_train=n_train, n_valid=n_valid,
    )
    _write_csv(
        STRUCTURE_DIR / "graph_coverage.csv",
        coverage_rows,
        [
            "seed",
            "dictionary",
            "atom",
            "coverage_train",
            "coverage_valid",
            "n_active_graphs_train",
            "n_active_graphs_valid",
            "mean_activations_per_active_graph_train",
            "median_activations_per_active_graph_train",
            "top10_graph_share_train",
            "mean_activations_per_active_graph_valid",
            "median_activations_per_active_graph_valid",
            "top10_graph_share_valid",
        ],
    )

    # common coordinates
    sub_payload = _read_json(RESULTS_DIR / "common_subspace.json")
    sub_selected = _load_subspace(sub_payload, final["selected_common_dim"])
    from scipy.stats import spearmanr

    common_rows: list[dict[str, Any]] = []
    for split, phi in (("train", train_phi), ("valid", valid_phi)):
        c = np.asarray(phi, dtype=np.float64) @ sub_selected.components
        for coordinate in range(sub_selected.q):
            values = c[:, coordinate]
            common_rows.append(
                {
                    "split": split,
                    "coordinate": f"c{coordinate + 1}",
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "p10": float(np.quantile(values, 0.10)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                    "rms_train_frozen": float(sub_selected.rms[coordinate]),
                }
            )
            for index, name in enumerate(dca.STRUCTURAL_DESCRIPTOR_NAMES):
                rho, _p = spearmanr(values, Z_train[:, index] if split == "train" else Z_valid[:, index])
                common_rows.append(
                    {
                        "split": split,
                        "coordinate": f"c{coordinate + 1}",
                        "feature": name,
                        "spearman": float(rho),
                    }
                )
    _write_csv(
        STRUCTURE_DIR / "common_coordinates.csv",
        common_rows,
        ["split", "coordinate", "feature", "mean", "std", "p10", "p50", "p90", "rms_train_frozen", "spearman"],
    )

    # sparse <-> dense residual recoverability (train-only OLS)
    dense_train = np.asarray(train_phi, dtype=np.float64) @ Dbar_perp
    dense_valid = np.asarray(valid_phi, dtype=np.float64) @ Dbar_perp
    recoverability_rows: list[dict[str, Any]] = []
    for source_name, target_name, source_train, source_valid, target_train, target_valid in (
        ("dense_residual", "iht10_residual", dense_train, dense_valid, alpha_train, alpha_valid),
        ("iht10_residual", "dense_residual", alpha_train, alpha_valid, dense_train, dense_valid),
        ("raw_iht10", "cssd_iht10", raw_train, raw_valid, alpha_train, alpha_valid),
    ):
        metrics = dca.fit_linear_recoverability(source_train, target_train, source_valid, target_valid)
        metrics.update({"source_representation": source_name, "target_representation": target_name})
        recoverability_rows.append(metrics)
    _write_csv(
        STRUCTURE_DIR / "sparse_dense_recoverability.csv",
        [
            {
                "source_representation": row["source_representation"],
                "target_representation": row["target_representation"],
                "valid_r2": row["valid_r2"],
                "per_dim_r2_median": row["per_dimension_r2"]["median"],
                "valid_normalized_error": row["valid_normalized_error"],
                "valid_mean_cosine": row["valid_mean_cosine"]["mean"],
                "linear_cka": row["linear_cka"],
                "design_rank": row["design_rank"],
            }
            for row in recoverability_rows
        ],
        [
            "source_representation",
            "target_representation",
            "valid_r2",
            "per_dim_r2_median",
            "valid_normalized_error",
            "valid_mean_cosine",
            "linear_cka",
            "design_rank",
        ],
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "ran": True,
        "common_dim": final["selected_common_dim"],
        "cssd": {
            "usage_train": cssd.usage_payload(alpha_train),
            "usage_valid": cssd.usage_payload(alpha_valid),
            "usage_weighted_specialization": float(spec["usage_weighted_specialization"]),
            "median_specialization": float(spec["median_specialization"]),
            "coherence": coherence_cssd,
            "coverage_summary": {
                "coverage_train_mean": float(np.mean(cov_train["coverage"])),
                "coverage_train_median": float(np.median(cov_train["coverage"])),
                "coverage_valid_mean": float(np.mean(cov_valid["coverage"])),
                "coverage_valid_median": float(np.median(cov_valid["coverage"])),
                "top10_graph_share_mean": float(np.mean(cov_train["top10_graph_share"])),
            },
        },
        "raw_seed0": {
            "usage_train": cssd.usage_payload(raw_train),
            "usage_valid": cssd.usage_payload(raw_valid),
            "usage_weighted_specialization": float(raw_spec["usage_weighted_specialization"]),
            "median_specialization": float(raw_spec["median_specialization"]),
            "coherence": coherence_raw,
        },
        "recoverability": recoverability_rows,
        "common_coordinates": {
            "q": sub_selected.q,
            "rms": [float(value) for value in sub_selected.rms],
        },
    }
    _write_json(out_path, payload)
    print(
        f"[stage-g] cssd neff={payload['cssd']['usage_valid']['effective_atoms']:.2f} "
        f"top5={payload['cssd']['usage_valid']['top5_share']:.3f} "
        f"coverage_med={payload['cssd']['coverage_summary']['coverage_train_median']:.3f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.{digits}f}"


def _probe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (ZERO_DIR / "raw.csv", ZERO_DIR / "mean_center.csv", ZERO_DIR / "q1.csv", ZERO_DIR / "q2.csv"):
        if not path.exists():
            continue
        import csv

        with open(path, encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def stage_report() -> dict[str, Any]:
    _ensure_dirs()
    prereg = _read_json(RESULTS_DIR / "preflight.json")
    stage_a_payload = _read_json(RESULTS_DIR / "stage_a.json")
    common = _read_json(RESULTS_DIR / "common_subspace.json")
    selection = _read_json(ZERO_DIR / "selection.json")
    probe = _read_json(ZERO_DIR / "probe.json")
    final_path = TRAINING_DIR / "final.json"
    final = _read_json(final_path) if final_path.exists() else None
    structure_path = STRUCTURE_DIR / "summary.json"
    structure = _read_json(structure_path) if structure_path.exists() else None
    lines: list[str] = ["# e2e_dictenv_common_subspace_dictionary_v1 — generated analysis tables", ""]
    lines.append("CPU only; official test never loaded. Generated by the round runner.")
    lines.append("")
    lines.append("## 0. Pre-registration / preflight")
    lines.append("")
    lines.append(f"- prereg sha256 `{prereg['preregistration']['sha256']}`")
    lines.append(f"- commit `{_git_commit()}`")
    lines.append(f"- phi train `{stage_a_payload['phi']['train']['shape']}`, valid `{stage_a_payload['phi']['valid']['shape']}`")
    lines.append(f"- identically-zero descriptor columns: `{stage_a_payload['zero_columns']}` (kept, layout frozen)")
    lines.append(f"- pipeline cache == dataset `dict_phi`: `{stage_a_payload['pipeline_bit_identical']}`")
    lines.append("")
    lines.append("## 1. Common subspace (train only)")
    lines.append("")
    lines.append(f"- ||mu|| = {_fmt(common['mean_direction_norm'], 5)}; |cos(u1, PC1)| = {_fmt(common['u1_cosine_pc1'], 4)}")
    lines.append(f"- q1 RMS = {[round(v, 5) for v in common['q1']['rms']]}")
    lines.append(f"- q2 RMS = {[round(v, 5) for v in common['q2']['rms']]}")
    lines.append("")
    lines.append("| space | split | E_common/||x||^2 | E_residual/||x||^2 | E_centered_common | E_centered_residual |")
    lines.append("|---|---|---|---|---|---|")
    for key, energy in common["energies"].items():
        space, split = key.split("_")
        lines.append(
            f"| {space} | {split} | {_fmt(energy['common_energy_fraction'])} | "
            f"{_fmt(energy['residual_energy_fraction'])} | {_fmt(energy['centered_common_fraction'])} | "
            f"{_fmt(energy['centered_residual_fraction'])} |"
        )
    lines.append("")
    lines.append("## 2. Zero-training same-D probe ($D$ unchanged per seed)")
    lines.append("")
    lines.append("| seed | split | variant | recon | top1 | top3 | top5 | top8 | Neff | max | >0.95 | a6 | a24 | a27 | a23 | J(RAW) | cos(RAW) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in _probe_rows():
        lines.append(
            "| {seed} | {split} | {variant} | {recon} | {t1} | {t3} | {t5} | {t8} | {neff} | {maxr} | {gt} | {a6} | {a24} | {a27} | {a23} | {jac} | {cos} |".format(
                seed=row["seed"], split=row["split"], variant=row["variant"],
                recon=_fmt(row["reconstruction_error"], 4),
                t1=_fmt(row["top1_usage"], 3), t3=_fmt(row["top3_usage"], 3),
                t5=_fmt(row["top5_usage"], 3), t8=_fmt(row["top8_usage"], 3),
                neff=_fmt(row["effective_atoms"], 2), maxr=_fmt(row["max_activation_rate"], 3),
                gt=row["n_atoms_gt_095"], a6=_fmt(row["atom6_rate"], 3),
                a24=_fmt(row["atom24_rate"], 3), a27=_fmt(row["atom27_rate"], 3),
                a23=_fmt(row["atom23_rate"], 3),
                jac=_fmt(row["support_jaccard_vs_raw"], 3),
                cos=_fmt(row["code_cosine_vs_raw"], 3),
            )
        )
    lines.append("")
    lines.append("## 3. Frozen selection rule (valid split primary; train reported)")
    lines.append("")
    lines.append(f"- q1 passed seeds: `{selection['q1_passed_seeds']}`; q2 passed seeds: `{selection['q2_passed_seeds']}`")
    lines.append(f"- selected: **{selection['selected']}**; train/valid agree: `{selection['train_valid_agree']}`")
    lines.append("")
    lines.append("| seed | variant | A (DC broken) | B (top5 drop) | C (centered keep) | D2 (atom23) | passed |")
    lines.append("|---|---|---|---|---|---|---|")
    for seed, entry in sorted(selection["per_seed"].items()):
        for variant in ("Q1", "Q2"):
            record = entry[variant]
            c = record["condition_c"]
            d2 = record["condition_d2"]
            lines.append(
                "| {} | {} | {} ({}/{}) | {} ({}) | {} | {} | {} |".format(
                    seed, variant,
                    record["condition_a"]["satisfied"], record["condition_a"]["n_below"], 3,
                    record["condition_b"]["satisfied"], _fmt(record["condition_b"]["top5_drop"], 3),
                    "-" if c is None else f"{c['satisfied']} ({_fmt(c['centered_residual_fraction'], 3)})",
                    "-" if d2 is None else f"{d2['satisfied']} ({_fmt(d2['atom23_ratio'], 3)})",
                    record["passed"],
                )
            )
    lines.append("")
    if final is not None and final.get("run") is False:
        lines.append("## 4. Training")
        lines.append("")
        lines.append(f"- NOT RUN: `{final['reason']}`")
    elif final is not None:
        lines.append("## 4. Training (single trajectory)")
        lines.append("")
        lines.append(
            f"- arm `{final['tag']}` common_dim={final['common_dim']} "
            f"completed={final['completed']} epochs_run={final['epochs_run']}"
        )
        lines.append(
            f"- actual params {final['actual_params']} (FINAL-CLEAN 97487); "
            f"dead-column fallbacks {final['dead_column_fallbacks']}; "
            f"U^T Dbar_perp max {final.get('U_T_Dbar_perp_max_abs')}"
        )
        lines.append(
            f"- best valid {_fmt(final['best_valid_mae'], 6)} @ {final['best_epoch']}; "
            f"soup {_fmt(final['soup']['soup_valid_mae'], 6)} members {final['soup']['members']}"
        )
        gate = final.get("callback_payloads", {}).get(str(GATE_EPOCH))
        if gate:
            lines.append("")
            lines.append("### Epoch-40 gate")
            lines.append("")
            lines.append(f"- verdict **{gate['verdict']}**")
            lines.append(f"- usage(valid): Neff {_fmt(gate['usage_valid']['effective_atoms'], 2)}, top5 {_fmt(gate['usage_valid']['top5_share'], 3)}, DC>0.95 {gate['gate']['condition_a']['dc_count_gt095']}")
            lines.append(f"- weighted Spec {_fmt(gate['specialization']['usage_weighted_specialization'], 4)} (ref {_fmt(cssd.REF_WEIGHTED_SPEC, 4)})")
            lines.append(f"- gradient norm ||dL/dD|| {_fmt(gate['gradient_norm_D'], 3)}; column min {_fmt(gate['projected_column_norm_min'], 3)}")
            lines.append(f"- A: {gate['gate']['condition_a']['satisfied']}; B: {gate['gate']['condition_b']['satisfied']}; C: {gate['gate']['condition_c']['satisfied']}")
        lines.append("")
        lines.append("| epoch | train_mae | train_rec_term | train_rec_full | valid_mae | d_norm |")
        lines.append("|---|---|---|---|---|---|")
        for row in final["curve"][::10] + [final["curve"][-1]]:
            lines.append(
                f"| {row['epoch']} | {_fmt(row['train_mae'], 4)} | {_fmt(row.get('train_rec_term'), 5)} | "
                f"{_fmt(row['train_rec'], 5)} | {_fmt(row['valid_mae'], 4)} | {_fmt(row['d_norm'], 2)} |"
            )
    if structure is not None and structure.get("ran"):
        lines.append("")
        lines.append("## 5. Reusable-structure audit (CSSD soup vs RAW seed-0)")
        lines.append("")
        lines.append("| representation | active | Neff | top5 | max rate | >0.95 | weighted Spec | coherence mean | coherence max |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label, entry in (("CSSD", structure["cssd"]), ("RAW seed 0", structure["raw_seed0"])):
            usage = entry["usage_valid"]
            lines.append(
                f"| {label} | {usage['active_atoms']} | {_fmt(usage['effective_atoms'], 2)} | "
                f"{_fmt(usage['top5_share'], 3)} | {_fmt(usage['max_activation_rate'], 3)} | "
                f"{sum(1 for f in usage['frequencies'] if f > 0.95) if 'frequencies' in usage else '-'} | "
                f"{_fmt(entry['usage_weighted_specialization'], 4)} | "
                f"{_fmt(entry['coherence']['mean'], 3)} | {_fmt(entry['coherence']['max'], 3)} |"
            )
        lines.append("")
        lines.append("| direction | R2 | per-dim median R2 | normalized error | mean cosine | CKA |")
        lines.append("|---|---|---|---|---|---|")
        for row in structure["recoverability"]:
            lines.append(
                f"| {row['source_representation']} -> {row['target_representation']} | "
                f"{_fmt(row['valid_r2'], 4)} | {_fmt(row['per_dimension_r2']['median'], 4)} | "
                f"{_fmt(row['valid_normalized_error'], 4)} | {_fmt(row['valid_mean_cosine']['mean'], 4)} | "
                f"{_fmt(row['linear_cka'], 4)} |"
            )
        lines.append("")
        lines.append("RAW seed-0 reference (audit round, dense_tied vs iht10): "
                     "dense_tied->iht10 R2 0.970825, iht10->dense_tied R2 0.998845.")

        extra_path = STRUCTURE_DIR / "extra_summary.json"
        if extra_path.exists():
            extra = _read_json(extra_path)
            lines.append("")
            lines.append("## 6. Graph-level reuse (post-hoc descriptive, |alpha| mass)")
            lines.append("")
            lines.append("| split | representation | effective atoms/graph (mean / median / p10 / p90) | atoms >=5% mass | max atom share |")
            lines.append("|---|---|---|---|---|")
            for row in extra["graph_reuse"]:
                lines.append(
                    f"| {row['split']} | {row['representation']} | "
                    f"{_fmt(row['effective_atoms_mean'], 2)} / {_fmt(row['effective_atoms_median'], 2)} / "
                    f"{_fmt(row['effective_atoms_p10'], 2)} / {_fmt(row['effective_atoms_p90'], 2)} | "
                    f"{_fmt(row['atoms_ge_5pct_mean'], 2)} | {_fmt(row['max_share_mean'], 3)} |"
                )
            lines.append("")
            lines.append("Graph argmax atom (train): " + "; ".join(
                f"{key} -> " + ", ".join(
                    f"a{atom}:{count / 10000:.3f}"
                    for atom, count in sorted(extra["argmax_hist"][key].items(), key=lambda kv: -kv[1])[:4]
                )
                for key in sorted(extra["argmax_hist"])
                if key.startswith("train/")
            ))
            lines.append("")
            lines.append("## 7. Dictionary geometry (RAW atom vs common direction / CSSD atom)")
            lines.append("")
            lines.append(f"- `max |cos(CSSD atom, u1)| = {_fmt(extra['geometry_summary']['max_cssd_u1_abs_cos'], 3)}` (hard constraint)")
            lines.append(f"- RAW `|cos(atom, u1)|`: mean {_fmt(extra['geometry_summary']['raw_u1_abs_cos_mean'], 3)}, median {_fmt(extra['geometry_summary']['raw_u1_abs_cos_median'], 3)}; atoms >0.5: {extra['geometry_summary']['n_atoms_abs_cos_gt_050']} (>0.9: {extra['geometry_summary']['n_atoms_abs_cos_gt_090']})")
            lines.append(f"- mean cos(RAW atom, CSSD atom) {_fmt(extra['geometry_summary']['mean_cos_raw_cssd'], 3)}; mean cos(RAW_perp atom, CSSD atom) {_fmt(extra['geometry_summary']['mean_cos_raw_perp_cssd'], 3)}")
            lines.append("")
            lines.append("| atom | RAW \\|cos(u1)\\| | cos(RAW, CSSD) | cos(RAW_perp, CSSD) | CSSD rate | CSSD Spec |")
            lines.append("|---|---|---|---|---|---|")
            profiles = {int(float(row["atom"])): row for row in _read_csv_rows(STRUCTURE_DIR / "atom_profiles.csv")}
            for row in _read_csv_rows(STRUCTURE_DIR / "dictionary_geometry.csv"):
                atom = int(float(row["atom"]))
                profile = profiles.get(atom, {})
                lines.append(
                    f"| {atom} | {_fmt(row['raw_u1_abs_cos'], 3)} | {_fmt(row['cos_raw_cssd'], 3)} | "
                    f"{_fmt(row['cos_raw_perp_cssd'], 3)} | {_fmt(profile.get('activation_rate_valid'), 3)} | "
                    f"{_fmt(profile.get('specialization_valid'), 3)} |"
                )
            lines.append("")
            lines.append("## 8. Common coordinate c1 (valid, Spearman)")
            lines.append("")
            for row in extra["common_coordinate_top_correlations_valid"][:10]:
                lines.append(f"- `{row['feature']}` rho = {_fmt(row['spearman'], 4)}")
            lines.append("")
            lines.append("## 9. Atom-profile roll-up (CSSD soup, valid)")
            lines.append("")
            lines.append(f"- atoms with rate > 0.9: {extra['atom_profiles']['n_atoms_rate_gt_090']}; > 0.5: {extra['atom_profiles']['n_atoms_rate_gt_050']}; max rate {_fmt(extra['atom_profiles']['max_rate'], 3)} (atom {extra['atom_profiles']['top_rate_atom']})")
    (RESULTS_DIR / "analysis_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "preregistration_sha256": prereg["preregistration"]["sha256"],
        "zero_columns": stage_a_payload["zero_columns"],
        "common_subspace": {
            "q1_rms": common["q1"]["rms"],
            "q2_rms": common["q2"]["rms"],
            "u1_cosine_pc1": common["u1_cosine_pc1"],
            "energy_q1_train": common["energies"]["q1_train"],
            "energy_q2_train": common["energies"]["q2_train"],
        },
        "selection": {
            "selected": selection["selected"],
            "q1_passed_seeds": selection["q1_passed_seeds"],
            "q2_passed_seeds": selection["q2_passed_seeds"],
            "train_valid_agree": selection["train_valid_agree"],
        },
        "training": None
        if final is None
        else {
            "run": final.get("run", True),
            "completed": final.get("completed"),
            "epochs_run": final.get("epochs_run"),
            "best_valid_mae": final.get("best_valid_mae"),
            "soup_valid_mae": final.get("soup", {}).get("soup_valid_mae"),
            "best_epoch": final.get("best_epoch"),
            "soup_members": final.get("soup", {}).get("members"),
            "actual_params": final.get("actual_params"),
            "epoch40_verdict": (final.get("callback_payloads", {}).get(str(GATE_EPOCH)) or {}).get("verdict"),
        },
        "structure": None if structure is None else structure,
    }
    _write_json(RESULTS_DIR / "summary.json", summary)
    print("[report] analysis_tables.md written", flush=True)
    return summary


def chain(threads: int = THREADS) -> None:
    stage_preflight()
    stage_a()
    stage_b()
    stage_c()
    selection = stage_select()
    stage_train(threads=threads)
    stage_g(threads=threads)
    stage_report()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "preflight",
            "stage-a",
            "stage-b",
            "stage-c",
            "select",
            "stage-train",
            "stage-g",
            "report",
            "chain",
        ],
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        stage_preflight()
    elif args.stage == "stage-a":
        stage_a()
    elif args.stage == "stage-b":
        stage_b()
    elif args.stage == "stage-c":
        stage_c()
    elif args.stage == "select":
        stage_select()
    elif args.stage == "stage-train":
        stage_train(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "stage-g":
        stage_g(threads=int(args.threads), force=bool(args.force))
    elif args.stage == "report":
        stage_report()
    elif args.stage == "chain":
        chain(threads=int(args.threads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
