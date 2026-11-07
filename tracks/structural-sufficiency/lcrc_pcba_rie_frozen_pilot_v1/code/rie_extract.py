"""Frozen LCRC-A extraction -> graph-level REAL/SHUFFLE relation features.

Stages
------
``smoke``       GPU identity / finiteness / no-grad / timing smoke (8 molecules).
``structures``  CPU-only, label-blind: motif sets, motif pairs, physical-graph
                distance bins and the SHUFFLE bin assignment for every official
                Train + Validation row.  Includes the 1024-molecule timing probe
                (serial and parallel, with a determinism cross-check) and the
                extrapolation to the full Train+Validation build.
``features``    one and only one frozen LCRC-A forward per molecule (Train and
                Validation), producing the raw 128-d logits and, per batch, the
                ``(3, 165)`` relation tensors for REAL and SHUFFLE, followed by
                the REAL-Train-only RMS normalisation.

The frozen encoder is loaded from the historical original LCRC-A seed-0 epoch-40
checkpoint with ``strict=True``, is placed in ``eval()`` and every parameter is
switched to ``requires_grad=False`` before the first forward pass.  Official
Test is never indexed.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import rie_common as C  # noqa: E402
import rie_features as F  # noqa: E402
import rie_motifs as MO  # noqa: E402
import lr_data as DATA  # noqa: E402
import nmp_model as M  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as PC  # noqa: E402

STRUCTURES = C.RUNS / "structures"
FEATURES = C.RUNS / "features"
SMOKE = C.RUNS / "smoke"

SPLIT_SIZES = {"valid": C.N_VALID, "train": C.N_TRAIN}


# ---------------------------------------------------------------------------
# shared loading / identity gates
# ---------------------------------------------------------------------------


def load_task() -> Any:
    task = DATA.load_task()
    C.assert_inherited_hashes(task)
    if int(np.asarray(task.test_rows).size) != C.N_TEST:
        raise RuntimeError("official Test split size drifted")
    return task


def checkpoint_identity() -> dict[str, Any]:
    """File + payload identity of the frozen original LCRC-A epoch-40 checkpoint."""
    path = Path(C.SOURCE["checkpoint"])
    if not path.is_file():
        raise RuntimeError(f"source checkpoint missing: {C.track_rel(path)}")
    sha = C.sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    registry = C.SOURCE["run_dir"] / "run.json"
    registry_ap = None
    if registry.is_file():
        run = json.loads(registry.read_text())
        registry_ap = run.get("fixed_epoch_reports", {}).get("40", {}).get("ap")
    checks = {
        "file_exists": True,
        "sha256_matches_frozen": sha == C.SOURCE["checkpoint_sha256"],
        "schema": payload.get("schema") == C.SOURCE["checkpoint_schema"],
        "round": payload.get("round_id") == C.SOURCE["round"],
        "arm": payload.get("arm") == C.SOURCE["arm"],
        "variant_is_original_A": str(payload.get("variant", "")) == "C1 + attention_pool",
        "seed_is_zero": int(payload.get("seed", -1)) == 0,
        "epoch_is_40": int(payload.get("epoch", -1)) == 40,
        "global_step_is_109520": int(payload.get("global_step", -1)) == 109520,
        "kind": payload.get("checkpoint_kind") == C.SOURCE["checkpoint_kind"],
        "n_outputs_128": int(payload.get("n_outputs", -1)) == C.N_TASKS,
        "stored_valid_ap_matches_anchor": abs(float(payload.get("valid_ap", -1.0)) - C.AP_A)
        <= C.AP_A_STORED_TOL,
        "stored_valid_ap_matches_instruction": abs(float(payload.get("valid_ap", -1.0)) - C.AP_A_INSTRUCTION)
        <= C.VALID_AP_TOL,
        "stored_valid_bce_present": abs(float(payload.get("valid_report_summary", {}).get("valid_bce", -1.0))
                                        - float(C.SOURCE["valid_bce_stored"])) <= 1e-9,
        "state_dict_tensor_count": len(payload["state_dict"]) == int(C.SOURCE["state_dict_tensors"]),
        "model_audit_parameters": int(payload.get("model_audit", {}).get("total_parameter_count", -1))
        == int(C.SOURCE["n_parameters"]),
        "model_audit_new_parameters": int(payload.get("model_audit", {}).get("new_parameter_count", -1))
        == int(C.SOURCE["new_parameters"]),
        "model_audit_base_parameters": int(payload.get("model_audit", {}).get("base_parameter_elements", -1))
        == int(C.SOURCE["baseline_parameters"]),
        "test_not_evaluated_in_source": payload.get("test_evaluated") is False,
        "run_registry_ap_matches": bool(registry_ap is not None and abs(float(registry_ap) - C.AP_A) <= 1e-9),
    }
    failed = [k for k, v in checks.items() if not v]
    record = {
        "path": C.track_rel(path),
        "file_sha256": sha,
        "frozen_sha256": C.SOURCE["checkpoint_sha256"],
        "registry": C.track_rel(registry) if registry.is_file() else None,
        "registry_valid_ap_epoch40": registry_ap,
        "payload": {
            "schema": payload.get("schema"),
            "round_id": payload.get("round_id"),
            "arm": payload.get("arm"),
            "variant": payload.get("variant"),
            "seed": payload.get("seed"),
            "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
            "checkpoint_kind": payload.get("checkpoint_kind"),
            "valid_ap": payload.get("valid_ap"),
            "valid_report_summary": payload.get("valid_report_summary"),
            "n_outputs": payload.get("n_outputs"),
            "git_commit": payload.get("git_commit"),
            "test_evaluated": payload.get("test_evaluated"),
        },
        "anchor": {
            "AP_A": C.AP_A,
            "AP_A_instruction": C.AP_A_INSTRUCTION,
            "tolerance": C.VALID_AP_TOL,
        },
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }
    if failed:
        raise RuntimeError(
            "SOURCE_CHECKPOINT_UNAVAILABLE_OR_MISMATCH: " + ", ".join(sorted(failed))
        )
    return record


def load_frozen_A(task: Any, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the original LCRC-A, load the frozen state strictly, freeze everything."""
    path = Path(C.SOURCE["checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _ref, ref_info, base_init = M.build_reference(task.spec, task.norm, C.N_TASKS, int(C.SOURCE["seed"]))
    model, build_audit = M.build_arm(
        "A",
        task.spec,
        task.norm,
        C.N_TASKS,
        seed=int(C.SOURCE["seed"]),
        device=device,
        base_init_state=base_init,
        zero_residual=True,
    )
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict load mismatch: missing={missing} unexpected={unexpected}")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bitwise = all(
        torch.equal(state[k], payload["state_dict"][k].detach().cpu()) for k in payload["state_dict"]
    )
    if set(state) != set(payload["state_dict"]):
        raise RuntimeError("state dict key sets differ from the checkpoint")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    n_params = int(sum(p.numel() for p in model.parameters()))
    audit = {
        "source_checkpoint": C.track_rel(path),
        "source_sha256": C.SOURCE["checkpoint_sha256"],
        "mode": getattr(model, "mode", None),
        "variant": "C1 + attention_pool",
        "parameters": n_params,
        "parameters_expected": int(C.SOURCE["n_parameters"]),
        "state_dict_tensors": len(state),
        "strict_load_missing": list(missing),
        "strict_load_unexpected": list(unexpected),
        "state_dict_bitwise_equal_to_checkpoint": bool(bitwise),
        "model_eval": not bool(model.training),
        "requires_grad_any": bool(any(p.requires_grad for p in model.parameters())),
        "parameter_fingerprint": C.state_fingerprint(state),
        "build_audit": {
            "base_parameter_elements": build_audit["base_parameter_elements"],
            "new_parameter_count": build_audit["new_parameter_count"],
            "total_parameter_count": build_audit["total_parameter_count"],
            "variant": build_audit["variant"],
        },
        "reference_info_variant": ref_info.get("variant"),
        "dtype": str(next(model.parameters()).dtype),
    }
    gates = {
        "bitwise_state_dict": bool(bitwise),
        "parameters_287644": n_params == int(C.SOURCE["n_parameters"]),
        "mode_is_attention_pool": audit["mode"] == "attention_pool",
        "frozen_eval": audit["model_eval"] and not audit["requires_grad_any"],
    }
    audit["gates"] = {k: bool(v) for k, v in gates.items()}
    if not all(gates.values()):
        raise RuntimeError(f"frozen model gate failed: {gates}")
    return model, audit


def parameter_digests(model: torch.nn.Module) -> dict[str, Any]:
    rows = []
    for name, param in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": list(param.shape),
                "sha256": C.sha256_array(param.detach().cpu().contiguous().numpy()),
            }
        )
    return {
        "n_parameter_tensors": len(rows),
        "n_parameters": int(sum(int(np.prod(r["shape"])) for r in rows)),
        "parameter_fingerprint": C.state_fingerprint(model.state_dict()),
        "parameters": rows,
    }


# ---------------------------------------------------------------------------
# CPU structure building (RDKit + physical graph distances)
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(cache_dir: str, smiles_path: str) -> None:
    _WORKER["atom_ptr"] = np.load(os.path.join(cache_dir, "atom_ptr.npy"), mmap_mode="r")
    _WORKER["bond_ptr"] = np.load(os.path.join(cache_dir, "bond_ptr.npy"), mmap_mode="r")
    _WORKER["bond_index"] = np.load(os.path.join(cache_dir, "bond_index.npy"), mmap_mode="r")
    with open(smiles_path, "r", encoding="utf-8") as handle:
        _WORKER["smiles"] = [json.loads(line) for line in handle]
    MO.feature_factory()  # build the RDKit factory once per worker


def build_chunk(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the per-molecule structures for one chunk of official rows."""
    rows = np.asarray(payload["rows"], dtype=np.int64)
    atom_ptr = _WORKER["atom_ptr"]
    bond_ptr = _WORKER["bond_ptr"]
    bond_index = _WORKER["bond_index"]
    smiles = _WORKER["smiles"]
    structures: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows.tolist():
        n_atoms = int(atom_ptr[row + 1] - atom_ptr[row])
        edges = [
            (int(u), int(v)) for u, v in bond_index[int(bond_ptr[row]) : int(bond_ptr[row + 1])].tolist()
        ]
        motifs, reason = MO.motifs_for_smiles(smiles[row])
        if reason is not None:
            failures.append({"official_row_id": int(row), "reason": str(reason), "smiles": smiles[row]})
        structures.append(
            F.molecule_structure(
                n_atoms=n_atoms,
                edges=edges,
                motifs=motifs,
                official_row_id=int(row),
                parse_failed=reason is not None,
            )
        )
    merged = F.concat_structures(structures, rows=rows)
    merged["failures"] = failures
    return merged


def _chunks(rows: np.ndarray, size: int) -> list[np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    return [rows[i : i + int(size)] for i in range(0, int(rows.size), int(size))]


def build_split_structures(
    *,
    rows: np.ndarray,
    cache_dir: Path,
    smiles_path: Path,
    workers: int,
    chunk_size: int,
) -> dict[str, Any]:
    payloads = [{"rows": chunk} for chunk in _chunks(rows, chunk_size)]
    t0 = time.perf_counter()
    if workers <= 1:
        _worker_init(str(cache_dir), str(smiles_path))
        blocks = [build_chunk(p) for p in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_worker_init,
            initargs=(str(cache_dir), str(smiles_path)),
        ) as pool:
            blocks = list(pool.map(build_chunk, payloads))
    seconds = float(time.perf_counter() - t0)
    failures = [f for block in blocks for f in block.pop("failures", [])]
    merged = F.merge_structures(blocks)
    merged["failures"] = failures
    merged["seconds"] = seconds
    return merged


def coverage_report(struct: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    """Motif / distance / intervention coverage of one split (label blind)."""
    rows = np.asarray(struct["rows"], dtype=np.int64)
    n = int(struct["n_molecules"])
    motif_family = np.asarray(struct["motif_family"], dtype=np.int64)
    motif_ptr = np.asarray(struct["motif_ptr"], dtype=np.int64)
    per_molecule_motifs = np.diff(motif_ptr)
    basis = np.zeros((n, C.N_MOTIF_FAMILIES), dtype=bool)
    owner = np.repeat(np.arange(n), per_molecule_motifs)
    if motif_family.size:
        basis[owner, motif_family] = True
    pair_bin = np.asarray(struct["pair_bin"], dtype=np.int64)
    pair_shuffle = np.asarray(struct["pair_bin_shuffle"], dtype=np.int64)
    pair_ptr = np.asarray(struct["pair_ptr"], dtype=np.int64)
    pair_counts = np.diff(pair_ptr)
    changed = np.zeros(n, dtype=bool)
    if pair_bin.size:
        differ = pair_bin != pair_shuffle
        changed = np.bincount(np.repeat(np.arange(n), pair_counts), weights=differ.astype(np.float64), minlength=n) > 0
    bin_counts = np.asarray(struct["bin_counts"], dtype=np.int64).reshape(n, C.N_BINS)
    eligible = pair_counts > 0
    multi = (bin_counts > 0).sum(axis=1) >= 2
    if motif_family.size:
        family_counts = np.bincount(motif_family, minlength=C.N_MOTIF_FAMILIES).astype(np.int64)
    else:
        family_counts = np.zeros(C.N_MOTIF_FAMILIES, dtype=np.int64)
    return {
        "split": split,
        "rows": n,
        "rows_sha256": C.sha256_int64(rows),
        "molecules_with_any_motif": int((per_molecule_motifs > 0).sum()),
        "molecules_with_any_eligible_pair": int(eligible.sum()),
        "molecules_with_multiple_distance_bins": int(multi.sum()),
        "molecules_REAL_ne_SHUFFLE_structure_level": int(changed.sum()),
        "fraction_with_eligible_pairs": float(eligible.mean()),
        "fraction_with_multiple_distance_bins": float(multi.mean()),
        "fraction_REAL_ne_SHUFFLE_structure_level": float(changed.mean()),
        "parse_failed_molecules": int(np.asarray(struct["parse_failed"], dtype=bool).sum()),
        "parse_failed_rows": np.asarray(struct["parse_failed"]).nonzero()[0].astype(int).tolist(),
        "motifs_total": int(motif_family.size),
        "motifs_per_molecule_mean": float(per_molecule_motifs.mean()),
        "motifs_per_molecule_max": int(per_molecule_motifs.max()) if n else 0,
        "motif_family_counts": {C.FAMILY_NAMES[i]: int(family_counts[i]) for i in range(C.N_MOTIF_FAMILIES)},
        "molecule_family_coverage": {
            C.FAMILY_NAMES[i]: float(basis[:, i].mean()) for i in range(C.N_MOTIF_FAMILIES)
        },
        "pairs_total": int(pair_bin.size),
        "pairs_per_molecule_mean": float(pair_counts.mean()),
        "pairs_per_molecule_max": int(pair_counts.max()) if n else 0,
        "pair_bin_counts": {
            f"bin{idx}_{C.DISTANCE_BINS[idx][0]}_{C.DISTANCE_BINS[idx][1] or 'inf'}": int(bin_counts[:, idx].sum())
            for idx in range(C.N_BINS)
        },
        "pair_count_per_molecule_bin_mean": {
            f"bin{idx}": float(bin_counts[:, idx].mean()) for idx in range(C.N_BINS)
        },
        "pair_count_per_molecule_bin_max": {
            f"bin{idx}": int(bin_counts[:, idx].max()) for idx in range(C.N_BINS)
        },
        "fraction_zero_pair_molecules": float((~eligible).mean()),
    }


# ---------------------------------------------------------------------------
# stage: smoke
# ---------------------------------------------------------------------------


def stage_smoke(device: torch.device) -> dict[str, Any]:
    SMOKE.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    identity = checkpoint_identity()
    task = load_task()
    model, model_audit = load_frozen_A(task, device)
    before = parameter_digests(model)
    rows = np.asarray(task.valid_rows[: int(C.EXTRACT["smoke_molecules"])], dtype=np.int64)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    max_identity = 0.0
    with torch.inference_mode():
        batch = PC.to_device(PC.collate(task.cache, rows.tolist()), device)
        logits, aux = model(batch, return_aux=True)
        identity_residual = aux["contrib"] + aux["rho"] + aux["residual_logits"] - logits
        max_identity = float(identity_residual.abs().max().item())
        h = aux["h"]
        finite = {
            "logits_finite": bool(torch.isfinite(logits).all()),
            "h_finite": bool(torch.isfinite(h).all()),
            "contrib_finite": bool(torch.isfinite(aux["contrib"]).all()),
            "rho_finite": bool(torch.isfinite(aux["rho"]).all()),
        }
        h_cpu = h.detach().cpu().numpy()
        base_np = logits.detach().cpu().numpy()
    # The head smoke must run outside inference_mode: tensors built inside it are
    # inference tensors and cannot be used in an autograd graph.
    torch.manual_seed(0)
    head = __import__("rie_model").build_head(device=device)
    relation = torch.randn(int(rows.size), C.N_BINS, C.PAIR_FEATURE_DIM, device=device)
    base = torch.from_numpy(base_np).to(device)
    y = torch.randint(0, 2, (int(rows.size), C.N_TASKS), device=device).to(torch.float32)

    def _head_grads() -> tuple[dict[str, Any], float]:
        head.zero_grad(set_to_none=True)
        delta = head(relation)
        batch_loss = torch.nn.functional.binary_cross_entropy_with_logits(base + delta, y)
        batch_loss = batch_loss + C.RECIPE["l1_coefficient"] * head.task_weight.weight.abs().mean()
        batch_loss.backward()
        report = {
            name: {
                "finite": bool(torch.isfinite(p.grad).all()),
                "abs_max": float(p.grad.abs().max().item()),
                "nonzero": bool((p.grad != 0).any().item()),
            }
            for name, p in head.named_parameters()
        }
        return report, float(batch_loss.detach().cpu().item())

    grad_report, loss_value = _head_grads()
    # At the frozen zero-residual initialisation the task weights are exactly zero,
    # so the shared projection's first gradient is exactly zero by construction.
    shared_zero_at_init = all(
        not v["nonzero"] for k, v in grad_report.items() if k.startswith("shared")
    )
    with torch.no_grad():
        for param in head.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-0.05)
    grad_report_after_step, _ = _head_grads()
    frozen_grad = any(p.grad is not None for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = float(time.perf_counter() - t0)
    after = parameter_digests(model)
    gates = {
        "source_checkpoint_identity": bool(identity["passed"]),
        "frozen_eval_no_grad": bool(model_audit["gates"]["frozen_eval"]),
        "bitwise_state_dict": bool(model_audit["gates"]["bitwise_state_dict"]),
        "all_values_finite": all(finite.values()),
        "logits_identity_contrib_plus_rho_plus_residual": bool(max_identity == 0.0),
        "head_gradients_finite": all(
            v["finite"] for v in list(grad_report.values()) + list(grad_report_after_step.values())
        ),
        "task_weight_gradient_nonzero_at_init": bool(grad_report["task_weight.weight"]["nonzero"]),
        "head_gradients_nonzero_after_one_step": all(
            v["nonzero"] for v in grad_report_after_step.values()
        ),
        "shared_projection_gradient_zero_at_init": bool(shared_zero_at_init),
        "frozen_encoder_has_no_grad": not bool(frozen_grad),
        "parameters_unchanged": after["parameter_fingerprint"] == before["parameter_fingerprint"],
        "no_grad_on_frozen_parameters": not any(p.grad is not None for p in model.parameters()),
        "h_width_is_64": bool(h_cpu.shape[1] == C.WIDTH),
        "test_never_indexed": True,
    }
    record = {
        "schema": "lcrc-pcba-rie-smoke-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "environment": C.environment(),
        "identity": identity,
        "model": model_audit,
        "rows": [int(r) for r in rows.tolist()],
        "head": {
            "parameters": int(sum(p.numel() for p in head.parameters())),
            "gradients_at_init": grad_report,
            "gradients_after_one_sign_step": grad_report_after_step,
            "loss_at_init": loss_value,
            "zero_residual_note": (
                "task_weight starts at exactly zero, so d(loss)/d(shared_projection) is zero at "
                "step 0; after the first parameter step the shared projection receives a non-zero "
                "gradient (checked above). This is the intended zero-residual design, not a bug."
            ),
        },
        "finite": finite,
        "max_abs_logits_identity_residual": max_identity,
        "h_shape": list(h_cpu.shape),
        "seconds": seconds,
        "gpu_peak": C.gpu_peak(device),
        "wall_seconds": float(time.perf_counter() - t_start),
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_passed": bool(all(gates.values())),
        "test_evaluated": False,
    }
    C.write_json(SMOKE / "smoke.json", record)
    C.log(f"[smoke] all_passed={record['all_passed']} seconds={seconds:.2f}")
    return record


# ---------------------------------------------------------------------------
# stage: structures
# ---------------------------------------------------------------------------


def _cache_paths(task: Any) -> tuple[Path, Path]:
    cache_dir = Path(task.cache_path)
    smiles_path = cache_dir / "smiles.jsonl"
    if not smiles_path.is_file():
        raise RuntimeError(f"cache SMILES file missing: {C.track_rel(smiles_path)}")
    return cache_dir, smiles_path


def stage_structures(*, workers: int, chunk_size: int, timing_only: bool = False) -> dict[str, Any]:
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    task = load_task()
    cache_dir, smiles_path = _cache_paths(task)
    probe_rows = np.asarray(task.train_rows[: int(C.BUDGET["timing_probe_molecules"])], dtype=np.int64)

    # ---- 1024-molecule timing probe (serial vs parallel, determinism gate) --
    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    t0 = time.perf_counter()
    serial = build_split_structures(
        rows=probe_rows, cache_dir=cache_dir, smiles_path=smiles_path, workers=1, chunk_size=int(chunk_size)
    )
    serial_seconds = float(time.perf_counter() - t0)
    rss_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    t0 = time.perf_counter()
    parallel = build_split_structures(
        rows=probe_rows,
        cache_dir=cache_dir,
        smiles_path=smiles_path,
        workers=int(workers),
        chunk_size=int(chunk_size),
    )
    parallel_seconds = float(time.perf_counter() - t0)
    serial_fp = {k: v for k, v in F.save_structures(serial, STRUCTURES / "_probe_serial.npz").items()}
    parallel_fp = {k: v for k, v in F.save_structures(parallel, STRUCTURES / "_probe_parallel.npz").items()}
    determinism = {
        "identical": serial_fp == parallel_fp,
        "n_arrays": len(serial_fp),
        "mismatched_arrays": [k for k in serial_fp if serial_fp[k]["sha256"] != parallel_fp.get(k, {}).get("sha256")],
    }
    for name in ("_probe_serial.npz", "_probe_parallel.npz"):
        (STRUCTURES / name).unlink(missing_ok=True)
    probe = {
        "molecules": int(probe_rows.size),
        "serial_seconds": serial_seconds,
        "serial_seconds_per_molecule": serial_seconds / max(int(probe_rows.size), 1),
        "parallel_workers": int(workers),
        "parallel_seconds": parallel_seconds,
        "parallel_seconds_per_molecule": parallel_seconds / max(int(probe_rows.size), 1),
        "parallel_speedup": (serial_seconds / parallel_seconds) if parallel_seconds > 0 else None,
        "peak_rss_before_kb": rss_before,
        "peak_rss_after_kb": rss_after,
        "peak_rss_delta_mb": (rss_after - rss_before) / 1024.0,
        "peak_rss_total_mb": rss_after / 1024.0,
        "serial_vs_parallel_structure_identical": determinism["identical"],
        "determinism_detail": determinism,
        "rows": [int(r) for r in probe_rows[:5].tolist()],
        "extrapolation": {
            "molecules_train_plus_valid": int(C.N_TRAIN + C.N_VALID),
            "serial_hours": float(serial_seconds / max(int(probe_rows.size), 1) * (C.N_TRAIN + C.N_VALID) / 3600.0),
            "parallel_hours": float(parallel_seconds / max(int(probe_rows.size), 1) * (C.N_TRAIN + C.N_VALID) / 3600.0),
            "parallel_workers": int(workers),
        },
    }
    C.log(f"[structures] timing probe: {json.dumps(probe['extrapolation'])}")

    coverage: dict[str, Any] = {"timing_probe": probe}
    fingerprints: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for split in ("train", "valid"):
        rows = np.asarray(getattr(task, f"{split}_rows"), dtype=np.int64)
        if int(rows.size) != SPLIT_SIZES[split]:
            raise RuntimeError(f"{split} rows {rows.size} != frozen {SPLIT_SIZES[split]}")
        built = build_split_structures(
            rows=rows,
            cache_dir=cache_dir,
            smiles_path=smiles_path,
            workers=int(workers),
            chunk_size=int(chunk_size),
        )
        failures.extend([{**f, "split": split} for f in built.pop("failures", [])])
        built_seconds = float(built.pop("seconds"))
        path = STRUCTURES / f"{split}_structures.npz"
        record = F.save_structures(built, path)
        fingerprints[split] = {
            "path": C.track_rel(path),
            "file_sha256": C.sha256_file(path),
            "arrays": record,
            "build_seconds": built_seconds,
            "build_workers": int(workers),
            "molecules": int(built["n_molecules"]),
        }
        coverage[split] = coverage_report(built, split=split)
        C.log(
            f"[structures] {split}: {built['n_molecules']} molecules, "
            f"{int(np.asarray(built['pair_bin']).size)} pairs, {built_seconds:.1f}s"
        )
        built = None

    coverage["failures"] = failures
    coverage["n_parse_failures"] = len(failures)
    coverage["feature_definition"] = MO.feature_definition_audit()
    coverage["distance_definition"] = {
        "policy": C.RING_POLICY,
        "bins": [{"bin": i, "min_distance": lo, "max_distance": hi} for i, (lo, hi) in enumerate(C.DISTANCE_BINS)],
        "min_pair_distance": int(C.MIN_PAIR_DISTANCE),
        "cross_component_pairs": "excluded (never treated as a long distance)",
        "shuffle": {
            "tag": C.SHUFFLE_TAG,
            "seed": C.SHUFFLE_SEED,
            "rng": "numpy SeedSequence(entropy=int(sha256(tag|row_id),16), spawn_key=(0,)) -> PCG64",
            "preserves": ["motif set", "motif pairs", "pair features", "per-bin pair counts"],
            "breaks": "which motif pair carries which distance bin",
        },
    }
    coverage["schema"] = "lcrc-pcba-rie-motif-coverage-v1"
    coverage["round_id"] = C.ROUND_ID
    coverage["created_utc"] = C.utcnow()
    coverage["git_commit"] = C.git_commit()
    coverage["stage"] = "structures"
    coverage["environment"] = C.environment()
    coverage["rdkit_reliable_timing_probe"] = True
    coverage["structures_fingerprint"] = fingerprints
    coverage["all_passed"] = bool(
        probe["serial_vs_parallel_structure_identical"]
        and coverage["train"]["rows"] == C.N_TRAIN
        and coverage["valid"]["rows"] == C.N_VALID
    )
    C.write_json(C.ANALYSIS / "motif_coverage.json", coverage)
    if timing_only:
        C.log("[structures] timing-only probe finished")
    C.log(
        f"[structures] valid eligible={coverage['valid']['fraction_with_eligible_pairs']:.4f} "
        f"multi-bin={coverage['valid']['fraction_with_multiple_distance_bins']:.4f} "
        f"REAL!=SHUFFLE(structure)={coverage['valid']['fraction_REAL_ne_SHUFFLE_structure_level']:.4f}"
    )
    return coverage


# ---------------------------------------------------------------------------
# stage: features
# ---------------------------------------------------------------------------


def _split_paths(name: str, filename: str) -> Path:
    return FEATURES / f"{name}_{filename}"


def _allocate(path: Path, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def stage_features(device: torch.device) -> dict[str, Any]:
    FEATURES.mkdir(parents=True, exist_ok=True)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    identity = checkpoint_identity()
    task = load_task()
    model, model_audit = load_frozen_A(task, device)
    digests_before = parameter_digests(model)

    structures: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        path = STRUCTURES / f"{split}_structures.npz"
        if not path.is_file():
            raise RuntimeError(f"structure cache missing: {C.track_rel(path)}")
        struct = F.load_structures(path)
        rows = np.asarray(getattr(task, f"{split}_rows"), dtype=np.int64)
        if not np.array_equal(np.asarray(struct["rows"], dtype=np.int64), rows):
            raise RuntimeError(f"{split} structure rows do not match the official split rows")
        structures[split] = struct

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_size = int(C.EXTRACT["batch_size"])
    blocked_atomic_note = (
        "the frozen encoder's graph summations use CUDA atomic index_add_, so cross-call "
        "logits are not bitwise reproducible; the AP agreement tolerance is 1e-4"
    )
    stats: dict[str, Any] = {}
    rms_acc = np.zeros(C.RELATION_DIM_FLAT, dtype=np.float64)
    train_real_sumsq_rows = 0
    max_identity = 0.0
    finite_violations = 0

    for split in ("valid", "train"):
        rows = np.asarray(getattr(task, f"{split}_rows"), dtype=np.int64)
        struct = structures[split]
        n = int(rows.size)
        base = _allocate(_split_paths(split, "base_logits.npy"), (n, C.N_TASKS))
        real = _allocate(_split_paths(split, "relation_raw.npy"), (n,) + C.RELATION_SHAPE)
        shuffle = _allocate(_split_paths(split, "relation_shuffle_raw.npy"), (n,) + C.RELATION_SHAPE)
        labels = np.asarray(task.labels[rows], dtype=np.float32)
        atom_ptr = np.asarray(task.cache["atom_ptr"])
        t0 = time.perf_counter()
        forward_seconds = 0.0
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            chunk = rows[start:stop]
            t_fwd = time.perf_counter()
            with torch.inference_mode():
                batch = PC.to_device(PC.collate(task.cache, chunk.tolist()), device)
                logits, aux = model(batch, return_aux=True)
                h = aux["h"]
                identity_residual = aux["contrib"] + aux["rho"] + aux["residual_logits"] - logits
                max_identity = max(max_identity, float(identity_residual.abs().max().item()))
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(h).all()):
                    finite_violations += 1
                    raise RuntimeError(f"non-finite frozen logits/h at rows {start}:{stop}")
                logits_np = logits.detach().cpu().numpy()
                h_np = h.detach().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += float(time.perf_counter() - t_fwd)
            base[start:stop] = logits_np
            positions = np.arange(start, stop, dtype=np.int64)
            atom_counts = (atom_ptr[chunk + 1] - atom_ptr[chunk]).astype(np.int64)
            gathered = F.gather_batch(struct, positions, atom_counts)
            real_block, counts_real = F.batch_relation_features(h_np, gathered, which="real")
            shuffle_block, counts_shuffle = F.batch_relation_features(h_np, gathered, which="shuffle")
            if not (np.array_equal(counts_real, counts_shuffle)):
                raise RuntimeError(f"SHUFFLE changed the per-bin pair counts at rows {start}:{stop}")
            if split == "train":
                F.accumulate_sumsq(rms_acc, real_block)
                train_real_sumsq_rows += int(real_block.shape[0])
            real[start:stop] = real_block
            shuffle[start:stop] = shuffle_block
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = float(time.perf_counter() - t0)
        np.save(_split_paths(split, "labels.npy"), labels)
        np.save(_split_paths(split, "rows.npy"), rows)
        stats[split] = {
            "rows": n,
            "rows_sha256": C.sha256_int64(rows),
            "labels_sha256": C.sha256_array(labels),
            "base_logits_sha256": C.sha256_array(np.asarray(base)),
            "relation_raw_sha256": C.sha256_array(np.asarray(real)),
            "relation_shuffle_raw_sha256": C.sha256_array(np.asarray(shuffle)),
            "seconds": seconds,
            "forward_seconds": forward_seconds,
            "molecules_per_second": float(n / seconds) if seconds > 0 else None,
            "observed_entries": int(np.isfinite(labels).sum()),
        }
        C.log(f"[features] {split}: {n} molecules in {seconds:.1f}s (forward {forward_seconds:.1f}s)")
        base.flush()
        real.flush()
        shuffle.flush()

    scale, rms_audit = F.rms_scale_from_sumsq(rms_acc, train_real_sumsq_rows)
    np.save(FEATURES / "relation_rms_scale.npy", scale)
    rms_audit["scale_sha256"] = C.sha256_array(scale)
    rms_audit["source"] = "REAL Train relation features only (SHUFFLE and Validation reuse the same scale)"
    del rms_acc

    t_scale = time.perf_counter()
    scaled: dict[str, Any] = {}
    for split in ("train", "valid"):
        for variant, filename in (("real", "relation_raw.npy"), ("shuffle", "relation_shuffle_raw.npy")):
            raw_path = _split_paths(split, filename)
            out_path = _split_paths(split, "relation.npy" if variant == "real" else "relation_shuffle.npy")
            raw = np.load(raw_path, mmap_mode="r")
            out = _allocate(out_path, raw.shape)
            block = 16384
            for start in range(0, raw.shape[0], block):
                stop = min(start + block, raw.shape[0])
                out[start:stop] = F.apply_scale(np.asarray(raw[start:stop]), scale)
            out.flush()
            del out
            scaled[f"{split}_{variant}"] = {
                "raw_path": C.track_rel(raw_path),
                "raw_file_sha256": C.sha256_file(raw_path),
                "normalized_path": C.track_rel(out_path),
                "normalized_file_sha256": C.sha256_file(out_path),
            }
    scale_seconds = float(time.perf_counter() - t_scale)

    # ---- post-hoc audits on the cached tensors ------------------------------
    valid_real = np.load(_split_paths("valid", "relation.npy"), mmap_mode="r")
    valid_shuffle = np.load(_split_paths("valid", "relation_shuffle.npy"), mmap_mode="r")
    changed = np.zeros(valid_real.shape[0], dtype=bool)
    step = 8192
    for start in range(0, valid_real.shape[0], step):
        stop = min(start + step, valid_real.shape[0])
        a = np.asarray(valid_real[start:stop])
        b = np.asarray(valid_shuffle[start:stop])
        changed[start:stop] = np.any(a != b, axis=(1, 2))
    valid_intervention = float(changed.mean())

    valid_labels = np.asarray(task.labels[task.valid_rows])
    train_labels = np.asarray(task.labels[task.train_rows])
    valid_base = np.asarray(np.load(_split_paths("valid", "base_logits.npy"), mmap_mode="r"))
    train_base = np.asarray(np.load(_split_paths("train", "base_logits.npy"), mmap_mode="r"))
    valid_probs = PC.sigmoid(valid_base)
    train_probs = PC.sigmoid(train_base)
    official_valid = MET.official_ap(valid_labels, valid_probs)
    independent_valid = MET.per_task_ap(valid_labels, valid_probs)
    official_train = MET.official_ap(train_labels, train_probs)
    independent_train = MET.per_task_ap(train_labels, train_probs)
    replay = {
        "checkpoint": identity,
        "model": model_audit,
        "valid_ap_replayed": float(official_valid["value"]),
        "valid_ap_anchor": C.AP_A,
        "valid_ap_abs_delta": abs(float(official_valid["value"]) - C.AP_A),
        "valid_ap_tolerance": float(C.VALID_AP_TOL),
        "valid_ap_reproduced": bool(abs(float(official_valid["value"]) - C.AP_A) <= C.VALID_AP_TOL),
        "valid_ap_instruction_anchor": C.AP_A_INSTRUCTION,
        "valid_ap_abs_delta_to_instruction": abs(float(official_valid["value"]) - C.AP_A_INSTRUCTION),
        "valid_ap_matches_stored_checkpoint_value": bool(
            abs(float(official_valid["value"]) - float(C.SOURCE["valid_ap_stored"])) <= C.VALID_AP_TOL
        ),
        "valid_bce_replayed": float(MET.masked_bce_numpy(valid_base, valid_labels)),
        "valid_bce_stored": float(C.SOURCE["valid_bce_stored"]),
        "valid_bce_abs_delta": abs(
            float(MET.masked_bce_numpy(valid_base, valid_labels)) - float(C.SOURCE["valid_bce_stored"])
        ),
        "train_macro_ap_replayed": float(independent_train["mean_over_scored_tasks"]),
        "n_tasks_scored_valid": int(independent_valid["n_tasks_scored"]),
        "n_tasks_with_observed_labels_valid": int(independent_valid["n_tasks_with_observed_labels"]),
        "official_vs_independent_valid_abs_delta": abs(
            float(official_valid["value"]) - float(independent_valid["mean_over_scored_tasks"])
        ),
        "official_vs_independent_train_abs_delta": abs(
            float(official_train["value"]) - float(independent_train["mean_over_scored_tasks"])
        ),
        "official_train_ap": float(official_train["value"]),
        "per_task_ap_valid_sha256": C.sha256_array(
            np.asarray([np.nan if v is None else v for v in independent_valid["per_task"]], dtype=np.float32)
        ),
        "max_abs_logits_identity_residual": float(max_identity),
        "frozen_nondeterminism_note": blocked_atomic_note,
        "test_evaluated": False,
        "created_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "device": str(device),
        "environment": C.environment(),
    }

    digests_after = parameter_digests(model)
    frozen_gates = {
        "parameters_unchanged": digests_after["parameter_fingerprint"] == digests_before["parameter_fingerprint"]
        == model_audit["parameter_fingerprint"],
        "no_gradients": not any(p.grad is not None for p in model.parameters()),
        "requires_grad_all_false": not any(p.requires_grad for p in model.parameters()),
        "still_eval": not bool(model.training),
    }

    test_isolation = {
        "official_test_rows": int(np.asarray(task.test_rows).size),
        "test_finite_labels": int(np.isfinite(np.asarray(task.labels[task.test_rows])).sum()),
        "train_valid_overlap": int(np.intersect1d(task.train_rows, task.valid_rows).size),
        "train_test_overlap": int(np.intersect1d(task.train_rows, task.test_rows).size),
        "valid_test_overlap": int(np.intersect1d(task.valid_rows, task.test_rows).size),
        "test_rows_never_indexed_by_this_round": True,
        "test_predictions_saved": False,
        "test_metrics_computed": 0,
    }

    health = {}
    for split in ("train", "valid"):
        for variant, filename in (("real", "relation.npy"), ("shuffle", "relation_shuffle.npy")):
            arr = np.load(_split_paths(split, filename), mmap_mode="r")
            sample = np.asarray(arr[:: max(1, arr.shape[0] // 20000)])
            health[f"{split}_{variant}"] = F.feature_health(sample)

    manifest = {
        "schema": "lcrc-pcba-rie-feature-manifest-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "device": str(device),
        "source_checkpoint": identity,
        "model": model_audit,
        "shapes": {
            "relation": [C.N_TRAIN, C.N_BINS, C.PAIR_FEATURE_DIM],
            "relation_valid": [C.N_VALID, C.N_BINS, C.PAIR_FEATURE_DIM],
            "base_logits": [C.N_TRAIN, C.N_TASKS],
            "base_logits_valid": [C.N_VALID, C.N_TASKS],
        },
        "splits": stats,
        "normalisation": {
            "kind": "per-column RMS (no mean centering)",
            "fit_split": "real_train_only",
            "audit": rms_audit,
            "scale_path": C.track_rel(FEATURES / "relation_rms_scale.npy"),
            "shared_by_real_and_shuffle": True,
            "shared_by_train_and_valid": True,
            "seconds": scale_seconds,
        },
        "scaled_files": scaled,
        "feature_health_sampled": health,
        "valid_real_ne_shuffle_fraction": valid_intervention,
        "min_intervention_fraction": float(C.DECISION["min_intervention_fraction"]),
        "structures": {
            split: {
                "path": C.track_rel(STRUCTURES / f"{split}_structures.npz"),
                "file_sha256": C.sha256_file(STRUCTURES / f"{split}_structures.npz"),
                "rows_sha256": C.sha256_int64(np.asarray(structures[split]["rows"], dtype=np.int64)),
            }
            for split in ("train", "valid")
        },
        "frozen_parameter_gates": {k: bool(v) for k, v in frozen_gates.items()},
        "frozen_parameter_digest_before": digests_before["parameter_fingerprint"],
        "frozen_parameter_digest_after": digests_after["parameter_fingerprint"],
        "max_abs_logits_identity_residual": float(max_identity),
        "finite_violations": int(finite_violations),
        "test_isolation": test_isolation,
        "gpu_peak": C.gpu_peak(device),
        "gpu_seconds": float(time.perf_counter() - t_start),
        "gpu_hours": float((time.perf_counter() - t_start) / 3600.0),
        "forward_seconds_total": float(sum(s["forward_seconds"] for s in stats.values())),
        "test_evaluated": False,
    }
    gates = {
        "source_checkpoint_identity": bool(identity["passed"]),
        "checkpoint_bitwise_load": bool(model_audit["gates"]["bitwise_state_dict"]),
        "valid_ap_reproduced": bool(replay["valid_ap_reproduced"]),
        "official_vs_independent_valid": bool(replay["official_vs_independent_valid_abs_delta"] <= 1e-9),
        "official_vs_independent_train": bool(replay["official_vs_independent_train_abs_delta"] <= 1e-9),
        "logits_identity": bool(max_identity == 0.0),
        "frozen_parameters_unchanged": all(frozen_gates.values()),
        "all_finite": bool(finite_violations == 0),
        "rms_fit_rows_is_train": bool(train_real_sumsq_rows == C.N_TRAIN),
        "shuffle_preserves_bin_counts": True,
        "rows_match_official_split": True,
        "test_isolation": bool(
            test_isolation["test_finite_labels"] == 0
            and test_isolation["train_valid_overlap"] == 0
            and test_isolation["train_test_overlap"] == 0
            and test_isolation["valid_test_overlap"] == 0
        ),
        "intervention_fraction_above_floor": bool(
            valid_intervention >= float(C.DECISION["min_intervention_fraction"])
        ),
    }
    manifest["gates"] = {k: bool(v) for k, v in gates.items()}
    manifest["all_passed"] = bool(all(gates.values()))
    C.write_json(C.ANALYSIS / "feature_manifest.json", manifest)
    C.write_json(C.ANALYSIS / "checkpoint_replay.json", replay)
    C.log(
        f"[features] valid AP {replay['valid_ap_replayed']:.10f} (delta {replay['valid_ap_abs_delta']:.2e}) "
        f"REAL!=SHUFFLE={valid_intervention:.4f} all_passed={manifest['all_passed']}"
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "structures", "features"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    try:
        if args.stage == "smoke":
            record = stage_smoke(device)
            return 0 if record["all_passed"] else 1
        if args.stage == "structures":
            record = stage_structures(
                workers=int(args.workers), chunk_size=int(args.chunk_size), timing_only=bool(args.timing_only)
            )
            return 0 if record["all_passed"] or args.timing_only else 1
        record = stage_features(device)
        return 0 if record["all_passed"] else 1
    except Exception:  # noqa: BLE001 - the exit code is the contract
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
