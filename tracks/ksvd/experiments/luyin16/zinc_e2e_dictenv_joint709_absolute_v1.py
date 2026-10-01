"""E2E-DictEnv-Joint709-Absolute-v1 stage runner (ZINC, local CPU regime).

Implements
``tracks/ksvd/notes/e2e_dictenv_joint709_absolute_v1_preregistration.md``.

One candidate only (``JOINT-SPARSE``): the frozen K48/s12 dictionary over the
train-only scale-balanced 709-D joint input (structural residual 65 + Sem108
108 + RoleCorr correspondence 536), coordinate ``[c~ ; IHT_10(alpha_48)]``,
width 49, frozen dictionary/scaler, everything else the frozen Sem108 parent.

Stages
------
``prepare``  (train-only, ``--mode scratch``): verify_frozen -> joint_scaler ->
             joint_cache -> joint_objects -> reconstruction
``screen``   (formal, ``--mode screen``): correctness -> smoke -> train ->
             interventions -> analysis

This round has **no control arm** (no PCA48, no random/shuffled/no-dictionary
training), so it can only state an absolute valid-MAE result for this candidate.
The official ZINC **test** split is never instantiated.
"""

from __future__ import annotations

import argparse
import math
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_joint709_absolute_v1 as core
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_increment_v2 as inc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_v1 as rc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0core
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_rolecorr_increment_v2 as incrun
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_rolecorr_v1 as rcrun
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = core.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_joint709_absolute_v1"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

THREADS = 8
SEED = int(incrun.SEED)  # 0
TRAIN_EPOCHS = int(incrun.TRAIN_EPOCHS)  # 320
SMOKE_EPOCHS = 3
SMOKE_TRAIN_SUBSET = 2048
SMOKE_VALID_SUBSET = 512
TAG = f"JOINT709-{core.CANDIDATE}-seed0"
BLOCK_SHUFFLE_SEEDS = core.BLOCK_SHUFFLE_SEEDS

_write_json = incrun._write_json
_read_json = incrun._read_json
_write_csv = incrun._write_csv
_git_commit = incrun._git_commit
_sha256_array = incrun._sha256_array
_sha256_file = incrun._sha256_file
_evaluate_device = incrun._evaluate_device
_curve_summary = incrun._curve_summary

#: exact identity of the frozen reused objects (see pre-registration §2)
EXPECTED = {
    "common_subspace_sha256": incrun.EXPECTED["common_subspace_sha256"],
    "corr_cache_train": incrun.EXPECTED["corr_cache_train"],
    "corr_cache_valid": incrun.EXPECTED["corr_cache_valid"],
}

#: fixed prefixes used by the diagnostic subset checks
ORDER_CHECK_MOLECULES = 6
ORDER_CHECK_RAW_MOLECULES = 3

#: invariance tolerance: BLAS reductions can round differently for a permuted
#: row layout / a different batch offset, so relabel and batch-offset checks use
#: a tight tolerance plus an identical active support (no atom-selection flip)
INVARIANCE_TOL = 1e-5


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure(
    *, epochs: int | None = None, threads: int | None = None, device: Any = None
) -> dict[str, Any]:
    """Fix the run knobs; every scientific constant stays frozen."""
    global TRAIN_EPOCHS, THREADS
    if epochs is not None:
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive")
        TRAIN_EPOCHS = int(epochs)
    if threads is not None:
        if int(threads) <= 0:
            raise ValueError("threads must be positive")
        THREADS = int(threads)
    configured = incrun.configure(
        epochs=TRAIN_EPOCHS, threads=THREADS, device=device if device is not None else "cpu"
    )
    return {"epochs": int(TRAIN_EPOCHS), "results_dir": str(RESULTS_DIR), **configured}


def _device() -> torch.device:
    return incrun._DEVICE


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _device_payload() -> dict[str, Any]:
    return {**incrun._device_payload(), "peak_rss_bytes": _peak_rss_bytes()}


def _blocker(payload: Mapping[str, Any]) -> None:
    inc.official_test_blocker(payload)


def _soup_path() -> Path:
    return CHECKPOINT_DIR / f"{TAG}_soup_state.pt"


def _run_path(arm: str) -> Path:
    return RESULTS_DIR / f"run_{arm}.json"


def model_factory(arm: str, *, reference_state: Mapping[str, torch.Tensor] | None = None):
    """Explicit model factory for the joint-dictionary candidate (route 2)."""
    if str(arm) != core.CANDIDATE:
        raise KeyError(f"round {core.ROUND} only trains {core.CANDIDATE!r}, got {arm!r}")
    return incrun.build_arm_route2(
        arm,
        reference_state=reference_state,
        results_dir=RESULTS_DIR,
    )


# ---------------------------------------------------------------------------
# prepare stage 1: frozen-object verification
# ---------------------------------------------------------------------------


def stage_verify_frozen(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "verify_frozen.json"
    if path.exists() and not force:
        return _read_json(path)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": core.ROUND,
        "candidate": core.CANDIDATE,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
    }
    subspace_path = (
        TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
    )
    subspace_sha = _sha256_file(subspace_path)
    payload["common_subspace"] = {
        "path": str(subspace_path.relative_to(REPO_ROOT)),
        "sha256": subspace_sha,
        "matches_frozen": bool(subspace_sha == EXPECTED["common_subspace_sha256"]),
    }
    sdb_dictionary, sdb_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    payload["sdb_dictionary"] = {
        "kind": str(cm.H1_CONFIG.dict_kind),
        "shape": list(np.asarray(sdb_dictionary).shape),
        "sha256": str(sdb_sha),
    }
    corr_checks: dict[str, Any] = {}
    for split in ("train", "valid"):
        raw = rcrun.load_raw_corr(split)
        expected = EXPECTED[f"corr_cache_{split}"]
        corr_checks[split] = {
            "n_patches": int(raw["node"].shape[0]),
            "node_sha256": _sha256_array(raw["node"]),
            "edge_sha256": _sha256_array(raw["edge"]),
            "matches_frozen": bool(
                int(raw["node"].shape[0]) == int(expected["n_patches"])
                and _sha256_array(raw["node"]) == str(expected["node_sha256"])
                and _sha256_array(raw["edge"]) == str(expected["edge_sha256"])
            ),
        }
    payload["corr_caches"] = corr_checks
    env_checks: dict[str, Any] = {}
    for split in ("train", "valid"):
        blob = rcrun._env_blob(split)
        node_sizes = [int(v) for v in blob["node_sizes"].tolist()]
        env_checks[split] = {
            "n_molecules": int(len(node_sizes)),
            "n_patches": int(sum(node_sizes)),
            "n_occurrences": int(sum(int(v) for v in blob["occ_sizes"].tolist())),
            "phi_shape": list(blob["phi"].shape),
        }
    payload["env_caches"] = env_checks
    payload["test_env_cache_present"] = bool(p1run._env_cache_path("test").exists())
    payload["passed"] = bool(
        payload["common_subspace"]["matches_frozen"]
        and all(row["matches_frozen"] for row in corr_checks.values())
        and not payload["test_env_cache_present"]
        and tuple(np.asarray(sdb_dictionary).shape) == (core.PHI_DIM, inc.BASE_ATOMS)
    )
    _blocker(payload)
    _write_json(path, payload)
    print(
        f"[verify_frozen] subspace={payload['common_subspace']['matches_frozen']} "
        f"corr={[row['matches_frozen'] for row in corr_checks.values()]} "
        f"passed={payload['passed']}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError(f"frozen-object verification failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# prepare stage 2: train-only joint scaler
# ---------------------------------------------------------------------------


def stage_joint_scaler(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    scaler_path = RESULTS_DIR / "joint_standardizers.json"
    struct, sem, corr = incrun._joint_raw_blocks("train")
    if not scaler_path.exists() or force:
        scaler = inc.fit_joint_scaler(struct, sem, corr)
        payload = scaler.to_json()
        payload.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "round": core.ROUND,
                "git_commit": _git_commit(),
                "n_fit_rows": int(struct.shape[0]),
                "report": inc.joint_block_report(struct, sem, corr),
                **_device_payload(),
                "official_test_loaded": False,
            }
        )
        _blocker(payload)
        _write_json(scaler_path, payload)
        print(
            "[joint_scaler] weights="
            + ", ".join(
                f"{name}:{float(getattr(scaler, name).weight):.4f}" for name in inc.JOINT_BLOCKS
            ),
            flush=True,
        )
    scaler = incrun.load_joint_scaler(RESULTS_DIR)
    refit = inc.fit_joint_scaler(struct, sem, corr)
    match = bool(
        np.array_equal(scaler.struct.scale, refit.struct.scale)
        and np.array_equal(scaler.struct.mask, refit.struct.mask)
        and float(scaler.struct.weight) == float(refit.struct.weight)
        and np.array_equal(scaler.sem.scale, refit.sem.scale)
        and np.array_equal(scaler.sem.mask, refit.sem.mask)
        and float(scaler.sem.weight) == float(refit.sem.weight)
        and np.array_equal(scaler.corr.scale, refit.corr.scale)
        and np.array_equal(scaler.corr.mask, refit.corr.mask)
        and float(scaler.corr.weight) == float(refit.corr.weight)
    )
    scaled = inc.apply_joint_scaler(scaler, struct, sem, corr)
    energies = {
        name: float(np.mean((scaled[:, span] ** 2).sum(axis=1)))
        for name, span in inc.JOINT_BLOCK_SLICES.items()
    }
    check = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "scaler_file_sha256": _sha256_file(scaler_path),
        "train_only_refit_matches": match,
        "n_fit_rows": int(struct.shape[0]),
        "n_valid_rows_used_for_fit": 0,
        "scaled_block_mean_energy": energies,
        "energy_deviation_max": float(max(abs(v - 1.0) for v in energies.values())),
        "zero_rms_coordinates": {
            name: int(np.sum(getattr(scaler, name).scale <= inc.SCALER_FLOOR))
            for name in inc.JOINT_BLOCKS
        },
        **_device_payload(),
        "official_test_loaded": False,
    }
    _blocker(check)
    _write_json(RESULTS_DIR / "joint_scaler_check.json", check)
    del struct, sem, corr, scaled
    print(
        f"[joint_scaler] refit_match={match} energies="
        + ",".join(f"{k}:{v:.4f}" for k, v in energies.items()),
        flush=True,
    )
    if not match:
        raise RuntimeError("train-only scaler refit does not reproduce the stored scaler")
    return check


# ---------------------------------------------------------------------------
# prepare stage 3: joint caches + row-order/alignment verification
# ---------------------------------------------------------------------------


def _raw_corr_row_check(split: str, n_molecules: int) -> dict[str, Any]:
    """Re-derive the 536-D correspondence object from raw ZINC and compare."""
    raw_molecules = list(zlr._load_zinc(ZINC_ROOT, "val" if split == "valid" else "train"))[
        : int(n_molecules)
    ]
    raw = rcrun.load_raw_corr(split)
    offset = 0
    checked = 0
    max_abs = 0.0
    y_match = True
    data = p1run.load_split(split, subset=int(n_molecules))
    for index, molecule in enumerate(raw_molecules):
        graph, node_types, edge_types = zlr._data_to_graph(molecule)
        n = int(graph.n)
        if hasattr(molecule, "y") and molecule.y is not None:
            y_match = y_match and bool(
                float(np.asarray(molecule.y).reshape(-1)[0]) == float(data[index].y.reshape(-1)[0])
            )
        for root in range(n):
            blocks = rc.corr_blocks(a1.patch_blocks(graph, root, node_types, edge_types))
            node_row = raw["node"][offset + root]
            edge_row = raw["edge"][offset + root]
            max_abs = max(
                max_abs,
                float(np.abs(node_row - blocks.node.astype(np.float32)).max()),
                float(np.abs(edge_row - blocks.edge.astype(np.float32)).max()),
            )
            checked += 1
        offset += n
    return {
        "split": split,
        "molecules": int(len(raw_molecules)),
        "patches": int(checked),
        "max_abs_error_vs_raw_zinc": float(max_abs),
        "molecule_y_order_matches": bool(y_match),
        "passed": bool(max_abs <= 1e-6 and y_match),
    }


def _joint_row_check(split: str, n_molecules: int) -> dict[str, Any]:
    """Recompute the scaled 709-D rows from the split data + raw corr cache."""
    values = incrun.load_joint(split, RESULTS_DIR)
    data = p1run.load_split(split, subset=int(n_molecules))
    incrun._attach_arm_block(data, split, core.CANDIDATE, joint_values=values)
    scaler = incrun.load_joint_scaler(RESULTS_DIR)
    components = torch.as_tensor(rcrun.load_subspace().components, dtype=torch.float32)
    raw = rcrun.load_raw_corr(split)
    node_sizes = [int(v) for v in rcrun._env_blob(split)["node_sizes"].tolist()]
    offset = 0
    max_abs = 0.0
    max_attach_abs = 0.0
    for index, item in enumerate(data):
        size = node_sizes[index]
        phi = item.dict_phi
        residual = (phi - (phi @ components) @ components.t()).numpy().astype(np.float64)
        sem = item.patch_cont[:, : core.SEM_DIM].numpy().astype(np.float64)
        corr = np.concatenate(
            [raw["node"][offset : offset + size], raw["edge"][offset : offset + size]], axis=1
        ).astype(np.float64)
        recomputed = inc.apply_joint_scaler(scaler, residual, sem, corr)
        cache_slice = values[offset : offset + size]
        max_abs = max(max_abs, float(np.abs(recomputed - cache_slice).max()))
        attached = item.joint_vec.numpy()
        max_attach_abs = max(max_attach_abs, float(np.abs(attached - cache_slice).max()))
        offset += size
    return {
        "split": split,
        "molecules": int(len(data)),
        "rows": int(offset),
        "cache_vs_recomputed_max_abs": float(max_abs),
        "attach_vs_cache_max_abs": float(max_attach_abs),
        "passed": bool(max_abs <= 1e-6 and max_attach_abs <= 1e-6),
    }


def stage_joint_cache(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "cache_check_joint.json"
    if path.exists() and not force:
        return _read_json(path)
    caches: dict[str, Any] = {}
    for split in ("train", "valid"):
        caches[split] = incrun.build_joint_cache(split, force=force, results_dir=RESULTS_DIR)
    ordering = {
        split: {
            "joint_rows": _joint_row_check(split, ORDER_CHECK_MOLECULES),
            "raw_corr_rows": _raw_corr_row_check(split, ORDER_CHECK_RAW_MOLECULES),
        }
        for split in ("train", "valid")
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": core.ROUND,
        "git_commit": _git_commit(),
        "caches": caches,
        "ordering": ordering,
        "block_slices": {k: [v.start, v.stop] for k, v in inc.JOINT_BLOCK_SLICES.items()},
        "dim": int(core.JOINT_DIM),
        **_device_payload(),
        "official_test_loaded": False,
    }
    payload["passed"] = bool(
        all(int(row["dim"]) == core.JOINT_DIM for row in caches.values())
        and all(
            section["joint_rows"]["passed"] and section["raw_corr_rows"]["passed"]
            for section in ordering.values()
        )
    )
    _blocker(payload)
    _write_json(path, payload)
    print(
        f"[joint_cache] train={caches['train']['n_patches']} valid={caches['valid']['n_patches']} "
        f"ordering_passed={payload['passed']}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError(f"joint cache ordering check failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# prepare stage 4: frozen joint dictionary (K48/s12) + real reconstruction
# ---------------------------------------------------------------------------


def stage_joint_objects(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    payload = incrun.stage_joint_objects(
        force=force, results_dir=RESULTS_DIR, fit_pca=False
    )
    entry = payload["dictionary_joint"]
    if "joint_pca48" in payload:
        raise RuntimeError("this round must not build the PCA48 control object")
    if tuple(entry["shape"]) != (core.JOINT_DIM, core.JOINT_ATOMS):
        raise RuntimeError(f"unexpected joint dictionary shape {entry['shape']}")
    if int(entry["sparsity"]) != core.JOINT_SPARSITY or int(entry["iht_steps"]) != core.IHT_STEPS:
        raise RuntimeError(f"unexpected joint coding budget {entry}")
    if int(entry["ksvd_epochs"]) != core.DICT_EPOCHS or int(entry["dict_seed"]) != core.DICT_SEED:
        raise RuntimeError(f"unexpected K-SVD budget {entry}")
    dictionary = incrun.load_joint_dictionary(RESULTS_DIR)
    sha = _sha256_array(dictionary)
    if sha != str(entry["sha256_f32"]):
        raise RuntimeError("dictionary_joint.pt does not match its recorded SHA-256")
    payload["dictionary_joint"]["sha256_rechecked"] = sha
    _write_json(RESULTS_DIR / "joint_objects.json", payload)
    print(
        f"[joint_objects] shape={entry['shape']} ksvd_mse={entry['ksvd_final_fit_mse']:.6f} "
        f"seconds={entry['seconds']:.1f}",
        flush=True,
    )
    return payload


def stage_reconstruction(force: bool = False) -> dict[str, Any]:
    """Real tied-IHT(top-12) reconstruction diagnostics of the frozen dictionary."""
    _ensure_dirs()
    path = RESULTS_DIR / "reconstruction.json"
    if path.exists() and not force:
        return _read_json(path)
    dictionary = incrun.load_joint_dictionary(RESULTS_DIR)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": core.ROUND,
        "git_commit": _git_commit(),
        "dictionary_sha256": _sha256_array(dictionary),
        "dictionary_shape": list(dictionary.shape),
        "note": (
            "codes are produced by the model's actual tied-IHT top-12 / 10-iteration "
            "path; the old route-2 reconstruct() zero placeholder is not used"
        ),
        **_device_payload(),
        "official_test_loaded": False,
    }
    for split, omp_rows in (("train", 16384), ("valid", None)):
        X = incrun.load_joint(split, RESULTS_DIR)
        payload[split] = core.joint_reconstruction_diagnostic(
            dictionary, X, omp_rows=omp_rows
        )
    payload["seconds"] = float(time.perf_counter() - started)
    _blocker(payload)
    _write_json(path, payload)
    print(
        f"[reconstruction] train_rel_err={payload['train']['relative_error_tied_iht']:.6f} "
        f"valid_rel_err={payload['valid']['relative_error_tied_iht']:.6f} "
        f"({payload['seconds']:.1f}s)",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# screen stage 1: correctness gates on the new path
# ---------------------------------------------------------------------------


def _gradient_and_update_probe(
    reference_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Finite forward/backward, gradient presence, and a real optimiser step."""
    data = p1run.load_split("train", subset=256)
    values = incrun.load_joint("train", RESULTS_DIR)
    incrun._attach_arm_block(data, "train", core.CANDIDATE, joint_values=values)
    loader = p1.make_env_loader(data, 64, False, 0)
    batch = next(iter(loader)).to(_device())
    model = model_factory(core.CANDIDATE, reference_state=reference_state).to(_device())
    model.train()
    prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    rec_term = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(cm.H1_LAMBDA) * rec_term
    model.zero_grad(set_to_none=True)
    loss.backward()
    named = dict(model.named_parameters())
    reader_keys = sorted(name for name in named if name.startswith("reader."))
    probe_keys = ["W_A_S", "W_E_S", "fusion.0.weight"] + reader_keys[:1]
    grads = {
        name: (
            None if named.get(name) is None or named[name].grad is None else float(named[name].grad.norm())
        )
        for name in probe_keys
    }
    joint_rows = slice(inc.COMMON_DIM, inc.COORD_DIM)
    joint_binding_grad = (
        None if model.W_A_S.grad is None else float(model.W_A_S.grad[joint_rows].norm())
    )
    edge_joint_grad = (
        None
        if model.W_E_S.grad is None
        else float(model.W_E_S.grad.reshape(3, inc.COORD_DIM, -1)[:, joint_rows].norm())
    )
    frozen_grads_absent = bool(
        model.D.grad is None and (model.D_block is None or model.D_block.grad is None)
    )
    finite = bool(torch.isfinite(loss).item() and torch.isfinite(prediction).all().item())
    snapshot = {
        key: named[key].detach().clone()
        for key in probe_keys
        if key in named and named[key] is not None
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY)
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
    optimizer.step()
    deltas = {
        key: float((named[key].detach() - value).abs().max()) for key, value in snapshot.items()
    }
    return {
        "finite": finite,
        "loss": float(loss.detach()),
        "probe_keys": probe_keys,
        "grad_norms": grads,
        "readout_grads_nonzero": all(
            grads.get(name) is not None and float(grads[name]) > 0.0 for name in probe_keys
        ),
        "joint_binding_row_grad_norm": joint_binding_grad,
        "joint_edge_binding_row_grad_norm": edge_joint_grad,
        "joint_binding_rows_have_grad": bool(
            joint_binding_grad is not None
            and joint_binding_grad > 0.0
            and edge_joint_grad is not None
            and edge_joint_grad > 0.0
        ),
        "frozen_dictionary_grads_absent": frozen_grads_absent,
        "parameter_update_max_abs": deltas,
        "parameters_actually_updated": bool(all(v > 0.0 for v in deltas.values())),
    }


def stage_correctness(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "correctness.json"
    if path.exists() and not force:
        cached = _read_json(path)
        if str(cached.get("device")) == str(_device()):
            return cached
    for name in ("verify_frozen.json", "joint_standardizers.json", "cache_check_joint.json",
                 "joint_objects.json", "reconstruction.json"):
        if not (RESULTS_DIR / name).exists():
            raise RuntimeError(f"prepare stage artifact missing: {name}; run model.stage=prepare first")
    reference = incrun._route2_reference_state(results_dir=RESULTS_DIR)
    model = model_factory(core.CANDIDATE, reference_state=reference).to(_device())
    model.eval()

    data = p1run.load_split("valid", subset=32)
    values = incrun.load_joint("valid", RESULTS_DIR)
    incrun._attach_arm_block(data, "valid", core.CANDIDATE, joint_values=values)
    loader = p1.make_env_loader(data, 16, False, 0)
    batch = next(iter(loader)).to(_device())

    dictionary = incrun.load_joint_dictionary(RESULTS_DIR)
    dictionary_sha = _sha256_array(dictionary)
    meta = _read_json(RESULTS_DIR / "joint_objects.json")["dictionary_joint"]
    scaler = incrun.load_joint_scaler(RESULTS_DIR)

    with torch.no_grad():
        block = model.extract_block(batch)
        coord = model.code(batch.dict_phi, block)
        purity = inc.zero_block_purity(model, batch.dict_phi, block)
        l0 = (coord[:, inc.JOINT_SLICE] != 0).sum(dim=1)
        coord_nonzero_fraction = float(
            (coord[:, inc.JOINT_SLICE].abs() > 0).float().mean()
        )
        # the joint code must agree with the frozen tied-IHT diagnostic path
        Dbar = v0core.normalized_dictionary(
            torch.as_tensor(dictionary, dtype=torch.float32)
        )
        expected_codes = v0core.tied_iht_codes(
            Dbar, block, s=core.JOINT_SPARSITY, steps=core.IHT_STEPS
        )
        joint_codes_match_diagnostic = bool(torch.equal(coord[:, inc.JOINT_SLICE], expected_codes))
        # perturbation sensitivity of the coordinate and the prediction
        perturbed = batch.clone()
        perturbed.joint_vec = batch.joint_vec + 0.5 * torch.randn_like(batch.joint_vec)
        coord_perturbed = model.code(batch.dict_phi, perturbed.joint_vec)
        coord_shift = float(
            (coord[:, inc.JOINT_SLICE] - coord_perturbed[:, inc.JOINT_SLICE]).abs().max()
        )
        prediction = model(batch, mask=cm.C6_MASK)
        model.inference_zero_block = True
        try:
            prediction_zero = model(batch, mask=cm.C6_MASK)
        finally:
            model.inference_zero_block = False
        prediction_shift = float((prediction - prediction_zero).abs().max())
        # per-node relabel: permuting the node rows permutes the coordinates
        permutation = np.random.default_rng(1234).permutation(int(batch.dict_phi.shape[0]))
        perm = torch.as_tensor(permutation, dtype=torch.long)
        coord_permuted = model.code(batch.dict_phi[perm], batch.joint_vec[perm])
        relabel_max_abs = float(
            (coord_permuted[torch.argsort(perm)] - coord).abs().max()
        )
        relabel_support_identical = bool(
            torch.equal(
                (coord_permuted[torch.argsort(perm)][:, inc.JOINT_SLICE] != 0),
                (coord[:, inc.JOINT_SLICE] != 0),
            )
        )
        # batch-offset: a molecule evaluated alone vs. inside a larger batch
        single_loader = p1.make_env_loader(data[:8], 1, False, 0)
        single_pred = torch.cat(
            [
                model(b.to(_device()), mask=cm.C6_MASK).view(-1)
                for b in single_loader
            ]
        )
        batch_pred = model(
            next(iter(p1.make_env_loader(data[:8], 8, False, 0))).to(_device()),
            mask=cm.C6_MASK,
        ).view(-1)
        batch_offset_max_abs = float((single_pred - batch_pred).abs().max())
        # different batching of the same valid subset must give the same MAE
        mae_small = _evaluate_device(
            model, p1.make_env_loader(data, 16, False, 0), _device(), cm.C6_MASK
        )["mae"]
        mae_large = _evaluate_device(
            model, p1.make_env_loader(data, 32, False, 0), _device(), cm.C6_MASK
        )["mae"]
        batch_mae_max_abs = float(abs(mae_small - mae_large))

    probe = _gradient_and_update_probe(reference)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": core.ROUND,
        "candidate": core.CANDIDATE,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "coordinate": {
            "width": int(coord.shape[1]),
            "joint_slice": [int(inc.JOINT_SLICE.start), int(inc.JOINT_SLICE.stop)],
            "common_width": int(core.COMMON_DIM),
            "iht_steps": int(core.IHT_STEPS),
            "s": int(core.JOINT_SPARSITY),
            "l0_mean": float(l0.float().mean()),
            "l0_max": int(l0.max()),
            "nonzero_fraction": coord_nonzero_fraction,
        },
        "joint_input": {
            "dim": int(block.shape[1]) if block is not None else None,
            "block_slices": {k: [v.start, v.stop] for k, v in inc.JOINT_BLOCK_SLICES.items()},
            "attached_from_cache": bool(block is not None),
        },
        "scaled_block_mean_energy": {
            name: float(value)
            for name, value in _read_json(RESULTS_DIR / "joint_scaler_check.json")[
                "scaled_block_mean_energy"
            ].items()
        },
        "dictionary": {
            "shape": list(dictionary.shape),
            "sha256": dictionary_sha,
            "sha256_matches_meta": bool(dictionary_sha == str(meta["sha256_f32"])),
            "frozen": bool(not model.D_block.requires_grad and not model.D.requires_grad),
            "base_dictionary_shape": list(model.D.shape),
            "base_dictionary_frozen": bool(not model.D.requires_grad),
            "column_norms_min": float(Dbar.norm(dim=0).min()),
            "column_norms_max": float(Dbar.norm(dim=0).max()),
        },
        "scaler": {
            "fit_split": "official train",
            "file_present": bool((RESULTS_DIR / "joint_standardizers.json").exists()),
            "block_weights": {name: float(getattr(scaler, name).weight) for name in inc.JOINT_BLOCKS},
        },
        "channels": {
            "joint_codes_match_frozen_diagnostic": joint_codes_match_diagnostic,
            "coordinate_sensitive_to_joint_input": bool(coord_shift > 1e-6),
            "coordinate_shift_on_perturbation": coord_shift,
            "prediction_sensitive_to_joint_block": bool(prediction_shift > 1e-6),
            "prediction_shift_zero_block": prediction_shift,
            "zero_block_purity": purity,
        },
        "invariance": {
            "relabel_max_abs": relabel_max_abs,
            "relabel_support_identical": relabel_support_identical,
            "relabel_passed": bool(
                relabel_max_abs <= INVARIANCE_TOL and relabel_support_identical
            ),
            "batch_offset_max_abs": batch_offset_max_abs,
            "batch_offset_passed": bool(batch_offset_max_abs <= INVARIANCE_TOL),
            "batch_size_mae_delta": batch_mae_max_abs,
            "batch_size_passed": bool(batch_mae_max_abs <= INVARIANCE_TOL),
            "tolerance": INVARIANCE_TOL,
        },
        "gradient_probe": probe,
        "parameters": {
            "trainable": trainable,
            "total": total,
            "frozen_dictionary_elements": int(dictionary.size) + int(model.D.numel()),
        },
        "gates": {
            "G0_geometry": bool(
                int(coord.shape[1]) == core.COORD_DIM
                and block is not None
                and int(block.shape[1]) == core.JOINT_DIM
                and tuple(model.W_A_S.shape)[0] == core.COORD_DIM
                and tuple(model.W_E_S.shape)[0] == 3 * core.COORD_DIM
            ),
            "G1_split_provenance": bool(
                dictionary_sha == str(meta["sha256_f32"])
                and str(meta.get("fit_split", "official train")) == "official train"
            ),
            "G3_scaler_semantics": bool(
                (RESULTS_DIR / "joint_scaler_check.json").exists()
                and _read_json(RESULTS_DIR / "joint_scaler_check.json")["train_only_refit_matches"]
            ),
            "G4_freezing": bool(
                not model.D.requires_grad
                and model.D_block is not None
                and not model.D_block.requires_grad
                and dictionary_sha == str(meta["sha256_f32"])
            ),
            "G5_sparsity": bool(
                int(l0.max()) <= core.JOINT_SPARSITY and core.IHT_STEPS == 10
            ),
            "G6_channel_in_prediction_path": bool(
                joint_codes_match_diagnostic
                and coord_shift > 1e-6
                and prediction_shift > 1e-6
                and purity["other_columns_bit_identical"]
                and purity["block_columns_zero"]
            ),
            "G7_gradients_updates": bool(
                probe["finite"]
                and probe["readout_grads_nonzero"]
                and probe["joint_binding_rows_have_grad"]
                and probe["frozen_dictionary_grads_absent"]
                and probe["parameters_actually_updated"]
            ),
            "G8_relabel_batch_offset": bool(
                relabel_max_abs <= INVARIANCE_TOL
                and relabel_support_identical
                and batch_offset_max_abs <= INVARIANCE_TOL
                and batch_mae_max_abs <= INVARIANCE_TOL
            ),
            "G9_official_test_blocker": bool(
                not p1run._env_cache_path("test").exists()
                and not _read_json(RESULTS_DIR / "verify_frozen.json")[
                    "test_env_cache_present"
                ]
            ),
        },
    }
    payload["all_passed"] = bool(all(payload["gates"].values()))
    _blocker(payload)
    _write_json(path, payload)
    print(
        f"[correctness] l0_max={int(l0.max())} coord_shift={coord_shift:.3e} "
        f"pred_shift={prediction_shift:.3e} relabel={relabel_max_abs:.2e} "
        f"batch_offset={batch_offset_max_abs:.2e} passed={payload['all_passed']}",
        flush=True,
    )
    if not payload["all_passed"]:
        raise RuntimeError(f"correctness gates failed: {payload['gates']}")
    return payload


# ---------------------------------------------------------------------------
# screen stage 2: smoke (correctness / gradient / timing only)
# ---------------------------------------------------------------------------


def stage_smoke(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "smoke.json"
    if path.exists() and not force:
        cached = _read_json(path)
        if str(cached.get("device")) == str(_device()):
            return cached
    reference = incrun._route2_reference_state(results_dir=RESULTS_DIR)
    train_data = p1run.load_split("train", subset=SMOKE_TRAIN_SUBSET)
    valid_data = p1run.load_split("valid", subset=SMOKE_VALID_SUBSET)
    incrun._attach_arm_block(
        train_data, "train", core.CANDIDATE, joint_values=incrun.load_joint("train", RESULTS_DIR)
    )
    incrun._attach_arm_block(
        valid_data, "valid", core.CANDIDATE, joint_values=incrun.load_joint("valid", RESULTS_DIR)
    )
    result = incrun.train_arm_device(
        arm=core.CANDIDATE,
        train_data=train_data,
        valid_data=valid_data,
        epochs=SMOKE_EPOCHS,
        tag=f"{TAG}-smoke",
        out_dir=CHECKPOINT_DIR,
        reference_state=reference,
        save_states=False,
        resume=False,
        write_arm_artifacts=False,
        log=True,
        model_factory=model_factory,
    )
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": core.ROUND,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "epochs": SMOKE_EPOCHS,
        "train_subset": SMOKE_TRAIN_SUBSET,
        "valid_subset": SMOKE_VALID_SUBSET,
        "best_valid_mae": float(result["best_valid_mae"]),
        "final_valid_mae": float(result["final_valid_mae"]),
        "epochs_run": int(result["epochs_run"]),
        "seconds_per_epoch": float(result["seconds_per_epoch"]),
        "wall_clock_s": float(result["wall_clock_s"]),
        "trainable_params": int(result["trainable_params"]),
        "curve_head": result["curve"][:1],
        "note": "smoke only: correctness / gradients / stability / timing, not a screen result",
    }
    payload["passed"] = bool(
        math.isfinite(payload["best_valid_mae"])
        and all(math.isfinite(float(row["valid_mae"])) for row in result["curve"])
        and all(math.isfinite(float(row["train_mae"])) for row in result["curve"])
    )
    _blocker(payload)
    _write_json(path, payload)
    print(
        f"[smoke] passed={payload['passed']} best={payload['best_valid_mae']:.6f} "
        f"s/epoch={payload['seconds_per_epoch']:.2f}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError(f"smoke failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# screen stage 3: the formal 320-epoch training run
# ---------------------------------------------------------------------------


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    run_path = _run_path(core.CANDIDATE)
    if run_path.exists() and not force:
        existing = _read_json(run_path)
        if int(existing.get("epochs_run", 0)) >= int(TRAIN_EPOCHS):
            print("[train] complete cache hit", flush=True)
            return existing
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    if not correctness.get("all_passed") or not smoke.get("passed"):
        raise RuntimeError("correctness/smoke gates not passed; refusing to train")
    reference = incrun._route2_reference_state(results_dir=RESULTS_DIR)
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    incrun._attach_arm_block(
        train_data, "train", core.CANDIDATE, joint_values=incrun.load_joint("train", RESULTS_DIR)
    )
    incrun._attach_arm_block(
        valid_data, "valid", core.CANDIDATE, joint_values=incrun.load_joint("valid", RESULTS_DIR)
    )
    payload = incrun.train_arm_device(
        arm=core.CANDIDATE,
        train_data=train_data,
        valid_data=valid_data,
        epochs=int(TRAIN_EPOCHS),
        tag=TAG,
        out_dir=CHECKPOINT_DIR,
        reference_state=reference,
        save_states=True,
        resume=True,
        write_arm_artifacts=False,
        log=True,
        model_factory=model_factory,
    )
    payload.update(
        {
            "round": core.ROUND,
            "candidate": core.CANDIDATE,
            "results_dir": (
                str(RESULTS_DIR.relative_to(REPO_ROOT))
                if str(RESULTS_DIR).startswith(str(REPO_ROOT))
                else str(RESULTS_DIR)
            ),
            "joint_dim": int(core.JOINT_DIM),
            "coord_dim": int(core.COORD_DIM),
            "s": int(core.JOINT_SPARSITY),
            "iht_steps": int(core.IHT_STEPS),
            "dictionary_sha256": _read_json(RESULTS_DIR / "joint_objects.json")["dictionary_joint"][
                "sha256_f32"
            ],
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )
    _blocker(payload)
    _write_json(run_path, payload)
    _write_csv(RESULTS_DIR / f"curve_{core.CANDIDATE}.csv", payload["curve"])
    _write_json(
        RESULTS_DIR / f"soup_{core.CANDIDATE}.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "arm": core.CANDIDATE,
            "tag": TAG,
            "members": payload["soup"]["members"],
            "member_valid_mae": payload["soup"]["member_valid_mae"],
            "soup_valid_mae": payload["soup"]["soup_valid_mae"],
            "best_valid_mae": payload["best_valid_mae"],
            "best_epoch": payload["best_epoch"],
            "soup_state_sha256": payload["soup_state_sha256"],
            "best_state_sha256": payload["best_state_sha256"],
        },
    )
    return payload


# ---------------------------------------------------------------------------
# screen stage 4: frozen soup-state diagnostics (no retraining)
# ---------------------------------------------------------------------------


def _load_soup_model(reference: Mapping[str, torch.Tensor]) -> Any:
    if not _soup_path().exists():
        raise RuntimeError(f"soup checkpoint missing: {_soup_path()}")
    model = model_factory(core.CANDIDATE, reference_state=reference).to(_device())
    model.load_state_dict(torch.load(_soup_path(), map_location="cpu", weights_only=False))
    model.eval()
    return model


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "interventions.json"
    if path.exists() and not force:
        return _read_json(path)
    run = _read_json(_run_path(core.CANDIDATE))
    if str(run.get("device")) != str(_device()):
        raise RuntimeError(f"formal run on {run.get('device')}; refusing frozen probes")
    if int(run["epochs_run"]) < int(run["epoch_budget"]):
        raise RuntimeError("formal run incomplete; refusing frozen probes")
    reference = incrun._route2_reference_state(results_dir=RESULTS_DIR)
    model = _load_soup_model(reference)
    valid_data = p1run.load_split("valid")
    incrun._attach_arm_block(
        valid_data, "valid", core.CANDIDATE, joint_values=incrun.load_joint("valid", RESULTS_DIR)
    )
    valid_loader = p1.make_env_loader(
        valid_data, int(incrun.BATCH_SIZE), False, int(SEED) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    intact = _evaluate_device(model, valid_loader, _device(), mask)
    model.inference_zero_block = True
    zeroed = _evaluate_device(model, valid_loader, _device(), mask)
    model.inference_zero_block = False
    shuffle_rows: list[dict[str, Any]] = []
    for seed in BLOCK_SHUFFLE_SEEDS:
        model.inference_shuffle_block_seed = int(seed)
        mae = float(_evaluate_device(model, valid_loader, _device(), mask)["mae"])
        model.inference_shuffle_block_seed = None
        shuffle_rows.append(
            {"seed": int(seed), "mae": mae, "delta_vs_intact": mae - float(intact["mae"])}
        )
    codes = incrun.collect_codes(model, valid_loader)
    usage = rc.code_usage(codes["block"], int(core.JOINT_ATOMS))
    # soup model on the full official train split, identical evaluation protocol
    train_data = p1run.load_split("train")
    incrun._attach_arm_block(
        train_data, "train", core.CANDIDATE, joint_values=incrun.load_joint("train", RESULTS_DIR)
    )
    train_loader = p1.make_env_loader(
        train_data, int(incrun.BATCH_SIZE), False, int(SEED) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    train_eval = _evaluate_device(model, train_loader, _device(), mask)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": core.ROUND,
        "candidate": core.CANDIDATE,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "soup_valid_mae": float(intact["mae"]),
        "zero_joint_code": {
            "mae": float(zeroed["mae"]),
            "delta": float(zeroed["mae"]) - float(intact["mae"]),
        },
        "row_shuffle": shuffle_rows,
        "row_shuffle_mean_delta": float(np.mean([row["delta_vs_intact"] for row in shuffle_rows])),
        "soup_train_mae": float(train_eval["mae"]),
        "train_valid_gap": float(train_eval["mae"]) - float(intact["mae"]),
        "code_usage": usage,
        "usage_only_note": (
            "zero / row-shuffle damage only shows the channel is used; it is not a "
            "causal necessity claim and this round has no control arm"
        ),
        "n_valid_molecules": int(intact["n_molecules"]),
        "n_train_molecules": int(train_eval["n_molecules"]),
    }
    _blocker(payload)
    _write_json(path, payload)
    np.savez(
        RESULTS_DIR / "per_molecule_errors.npz",
        target=intact["targets"],
        pred=intact["predictions"],
        valid_index=np.arange(intact["targets"].size),
    )
    print(
        f"[interventions] soup_valid={payload['soup_valid_mae']:.6f} "
        f"soup_train={payload['soup_train_mae']:.6f} "
        f"zero_delta={payload['zero_joint_code']['delta']:+.6f} "
        f"shuffle_delta={payload['row_shuffle_mean_delta']:+.6f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# screen stage 5: analysis / report / decision
# ---------------------------------------------------------------------------


def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    run = _read_json(_run_path(core.CANDIDATE))
    if str(run.get("device")) != str(_device()):
        raise RuntimeError(f"formal run on {run.get('device')}; refusing analysis")
    interventions = _read_json(RESULTS_DIR / "interventions.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    objects = _read_json(RESULTS_DIR / "joint_objects.json")
    reconstruction = _read_json(RESULTS_DIR / "reconstruction.json")
    cache_check = _read_json(RESULTS_DIR / "cache_check_joint.json")
    scaler_check = _read_json(RESULTS_DIR / "joint_scaler_check.json")
    verify = _read_json(RESULTS_DIR / "verify_frozen.json")
    soup_mae = float(run["soup"]["soup_valid_mae"])
    band = core.absolute_band(soup_mae)
    verdict = f"JOINT709_{band['band'].upper()}"
    curve = _curve_summary(run["curve"])
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": core.ROUND,
        "candidate": core.CANDIDATE,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "seed": int(SEED),
        "epochs": int(run["epoch_budget"]),
        "epochs_run": int(run["epochs_run"]),
        "completed": bool(run["completed"]),
        "coordinate_dim": int(core.COORD_DIM),
        "joint_dim": int(core.JOINT_DIM),
        "soup_valid_mae": soup_mae,
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": list(run["soup"]["members"]),
        "soup_member_valid_mae": list(run["soup"]["member_valid_mae"]),
        "band": band,
        "soup_train_mae": float(interventions["soup_train_mae"]),
        "train_valid_gap": float(interventions["train_valid_gap"]),
        "zero_joint_code": interventions["zero_joint_code"],
        "row_shuffle": interventions["row_shuffle"],
        "row_shuffle_mean_delta": float(interventions["row_shuffle_mean_delta"]),
        "code_usage": interventions["code_usage"],
        "curve": curve,
        "curve_tail": [
            {
                "epoch": int(row["epoch"]),
                "train_mae": float(row["train_mae"]),
                "valid_mae": float(row["valid_mae"]),
            }
            for row in run["curve"][-20:]
        ],
        "dictionary": objects["dictionary_joint"],
        "reconstruction": {
            "train_relative_error_tied_iht": float(
                reconstruction["train"]["relative_error_tied_iht"]
            ),
            "valid_relative_error_tied_iht": float(
                reconstruction["valid"]["relative_error_tied_iht"]
            ),
            "train_blocks": reconstruction["train"]["blocks"],
            "valid_blocks": reconstruction["valid"]["blocks"],
            "omp_reference": {
                "train": reconstruction["train"].get("omp_reference"),
                "valid": reconstruction["valid"].get("omp_reference"),
            },
            "exact_l0_mean": float(reconstruction["valid"]["exact_l0_mean"]),
            "active_atoms": int(reconstruction["valid"]["active_atoms"]),
        },
        "cache_check_passed": bool(cache_check["passed"]),
        "scaler_refit_matches": bool(scaler_check["train_only_refit_matches"]),
        "frozen_objects_passed": bool(verify["passed"]),
        "correctness_all_passed": bool(correctness["all_passed"]),
        "smoke_passed": bool(smoke["passed"]),
        "parameters": correctness["parameters"],
        "wall_clock_s": float(run["wall_clock_s"]),
        "seconds_per_epoch": float(run["seconds_per_epoch"]),
        "smoke_seconds_per_epoch": float(smoke["seconds_per_epoch"]),
        "peak_gpu_memory_bytes": run.get("peak_gpu_memory_bytes"),
        "peak_rss_bytes": int(run.get("peak_rss_bytes", _peak_rss_bytes())),
        "verdict": verdict,
    }
    _blocker(payload)
    _write_json(RESULTS_DIR / "summary.json", payload)
    _write_report(payload)
    _write_decision(payload)
    print(
        f"[analysis] soup_valid={soup_mae:.9f} best={payload['best_valid_mae']:.9f}@"
        f"{payload['best_epoch']} band={band['band']} verdict={verdict}",
        flush=True,
    )
    return payload


def _write_report(summary: Mapping[str, Any]) -> None:
    band = summary["band"]
    rec = summary["reconstruction"]
    lines = [
        "# E2E-DictEnv-Joint709-Absolute-v1 — report\n\n",
        f"- protocol: `{PROTOCOL_VERSION}` (study `zinc-context-gap`, seed 0)\n",
        f"- git commit: `{summary['git_commit']}`\n",
        f"- device: `{summary['device']}` (CPU regime, explicit `runtime.device=cpu`)\n",
        "- official test loaded: `false` (never instantiated)\n",
        f"- candidate: `{summary['candidate']}` — frozen K48/s12 joint dictionary over "
        f"the train-only balanced {summary['joint_dim']}-D input; coordinate width "
        f"{summary['coordinate_dim']}\n",
        f"- verdict: **{summary['verdict']}** (band `{band['band']}`)\n\n",
        "## Absolute result (official valid 1000, Top-5 soup)\n\n",
        f"- Best valid MAE: **{summary['best_valid_mae']:.9f}** @ epoch "
        f"{summary['best_epoch']}\n",
        f"- Top-5 soup valid MAE: **{summary['soup_valid_mae']:.9f}** "
        f"(members {summary['soup_members']})\n",
        f"- Official-train soup MAE (same protocol): {summary['soup_train_mae']:.9f}; "
        f"train−valid gap {summary['train_valid_gap']:+.9f}\n",
        f"- Band: `{band['band']}` — {band['recommendation']}\n",
        f"- Thresholds: strong ≤ {band['thresholds']['strong_max']}, promising ≤ "
        f"{band['thresholds']['promising_max']}, borderline ≤ "
        f"{band['thresholds']['borderline_max']}, else stop\n\n",
        "## Training curve\n\n",
        f"- epochs run: {summary['epochs_run']} / {summary['epochs']} "
        f"(completed: {summary['completed']})\n",
        f"- valid first {summary['curve']['valid_first']:.6f} → last "
        f"{summary['curve']['valid_last']:.6f}; min {summary['curve']['valid_min']:.6f} "
        f"@ {summary['curve']['valid_min_epoch']}; last-20 mean "
        f"{summary['curve']['valid_last20_mean']:.6f}\n",
        f"- train last {summary['curve']['train_last']:.6f} (min {summary['curve']['train_min']:.6f})\n",
        f"- wall {summary['wall_clock_s']:.1f}s, {summary['seconds_per_epoch']:.2f} s/epoch\n\n",
        "## Frozen dictionary diagnostics\n\n",
        f"- K-SVD: K={summary['dictionary']['atoms']}/s={summary['dictionary']['sparsity']}, "
        f"{summary['dictionary']['ksvd_epochs']} epochs, seed {summary['dictionary']['dict_seed']}, "
        f"{summary['dictionary']['n_fit_rows']} train rows, final fit MSE "
        f"{summary['dictionary']['ksvd_final_fit_mse']:.6f}, SHA-256 "
        f"`{summary['dictionary']['sha256_f32'][:16]}…`\n",
        f"- Real joint reconstruction (tied-IHT top-12, the actual model path): "
        f"valid {rec['valid_relative_error_tied_iht']:.6f}, train "
        f"{rec['train_relative_error_tied_iht']:.6f}\n",
        "- Per-block relative error (valid): "
        + ", ".join(
            f"{name} {row['relative_error_tied_iht']:.4f}"
            for name, row in rec["valid_blocks"].items()
        )
        + "\n",
        f"- Soup code usage: {summary['code_usage']['active_atoms']}/"
        f"{summary['code_usage']['atoms']} atoms active, effective "
        f"{summary['code_usage']['effective_atoms']:.2f}, exact l0 mean "
        f"{summary['code_usage']['exact_l0_mean']:.2f}, top1 share "
        f"{summary['code_usage']['top1_share']:.3f}\n\n",
        "## Usage diagnostics (soup state, no retraining)\n\n",
        f"- zero the joint block: ΔMAE {summary['zero_joint_code']['delta']:+.6f} "
        f"({summary['zero_joint_code']['mae']:.6f})\n",
        "- within-molecule row shuffle: "
        + ", ".join(
            f"seed {row['seed']} Δ{row['delta_vs_intact']:+.6f}" for row in summary["row_shuffle"]
        )
        + "\n",
        "  These only show that the channel is used; this round has no control arm, so "
        "no increment / sparse-vs-dense / causal statement follows.\n\n",
        "## Provenance\n\n",
        f"- correctness gates all passed: {summary['correctness_all_passed']}; "
        f"smoke passed: {summary['smoke_passed']}; frozen objects verified: "
        f"{summary['frozen_objects_passed']}; cache ordering: {summary['cache_check_passed']}; "
        f"scaler train-only refit: {summary['scaler_refit_matches']}\n",
        "- split fingerprint / SHAs: control-plane manifest + `artifacts/*.json`\n",
        f"- parameters: trainable {summary['parameters']['trainable']}, "
        f"total {summary['parameters']['total']}\n",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    band = summary["band"]
    body = {
        core.BAND_STRONG: (
            "The frozen K48/s12 joint environment dictionary candidate reached the strong "
            "absolute interval. Per the pre-registered resource rule this is a strong "
            "prospect and a **separate, newly pre-registered** confirmation round is "
            "recommended; this round itself does not authorise seed 1."
        ),
        core.BAND_PROMISING: (
            "The candidate reached the promising absolute interval. It is worth one "
            "further validation round, but only under a new pre-registration; nothing "
            "is purchased automatically by this result."
        ),
        core.BAND_BORDERLINE: (
            "The candidate landed in the borderline interval: on this absolute screen the "
            "joint-dictionary route is not clearly better than the historical anchor and "
            "no further experiments are added now."
        ),
        core.BAND_STOP: (
            "The candidate did not enter a better absolute interval (> 0.1233): the frozen "
            "joint dictionary route is stopped for now. This is an absolute-performance "
            "screen only — it is not evidence about any control arm."
        ),
    }[band["band"]]
    lines = [
        "# Decision — E2E-DictEnv-Joint709-Absolute-v1\n\n",
        f"- candidate: `{summary['candidate']}` (one candidate, seed 0, 320 epochs)\n",
        f"- soup valid MAE: **{summary['soup_valid_mae']:.9f}** "
        f"(best {summary['best_valid_mae']:.9f} @ {summary['best_epoch']})\n",
        f"- band: `{band['band']}` — {band['recommendation']}\n",
        f"- official train soup MAE {summary['soup_train_mae']:.9f} "
        f"(gap {summary['train_valid_gap']:+.9f})\n\n",
        body,
        "\n\n## Not claimed by this round\n\n",
        "- No increment over any control (no control arm was run: no PCA48, no random / "
        "shuffled / no-dictionary training).\n",
        "- No sparse-dictionary-vs-PCA or joint-vs-separate causal statement.\n",
        "- No statistical-significance claim; single seed.\n",
        "- No terminal (official-test) statement.\n",
        "\n## Forbidden without a new pre-registration\n\n",
        "- seed 1, any K / s / K-SVD-epoch / scaler / width / horizon / lr change;\n",
        "- unfreezing or end-to-end fine-tuning the dictionary;\n",
        "- touching the official ZINC test split;\n",
        "- re-using these numbers as a matched comparison against historical A/B/C runs.\n",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

PREPARE_STAGES = (
    "verify_frozen",
    "joint_scaler",
    "joint_cache",
    "joint_objects",
    "reconstruction",
)
SCREEN_STAGES = ("correctness", "smoke", "train", "interventions", "analysis")


def run_stages(stage: str, only: Sequence[str] | None = None) -> dict[str, Any]:
    _ensure_dirs()
    print(
        f"[runner] commit={_git_commit()} protocol={PROTOCOL_VERSION} "
        f"stage={stage} device={_device()} threads={THREADS}",
        flush=True,
    )
    table = {
        "verify_frozen": stage_verify_frozen,
        "joint_scaler": stage_joint_scaler,
        "joint_cache": stage_joint_cache,
        "joint_objects": stage_joint_objects,
        "reconstruction": stage_reconstruction,
        "correctness": stage_correctness,
        "smoke": stage_smoke,
        "train": stage_train,
        "interventions": stage_interventions,
        "analysis": stage_analysis,
    }
    names = list(PREPARE_STAGES if stage == "prepare" else SCREEN_STAGES)
    if only:
        names = [name for name in names if name in set(only)]
    results: dict[str, Any] = {}
    for name in names:
        results[name] = table[name]()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E-DictEnv-Joint709-Absolute-v1 stage runner")
    parser.add_argument("--stage", required=True, choices=("prepare", "screen", "all"))
    parser.add_argument("--only", default=None, help="comma-separated stage subset override")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    configure(device=args.device)
    only = [part.strip() for part in args.only.split(",") if part.strip()] if args.only else None
    if args.stage in ("prepare", "all"):
        run_stages("prepare", only)
    if args.stage in ("screen", "all"):
        run_stages("screen", only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
