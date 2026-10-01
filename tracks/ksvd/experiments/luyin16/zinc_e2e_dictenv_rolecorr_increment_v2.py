"""E2E-DictEnv-RoleCorr-Increment-v2 stage runner (ZINC, device-parametric).

Implements
``tracks/ksvd/notes/e2e_dictenv_rolecorr_increment_v2_preregistration.md``.

Route 1 (``stage=route1``) appends a 16-wide frozen block to the *full* frozen
structural budget and screens, at width 49 and with a bit-identical readout
initialisation:

* ``EXTRA-STRUCT``  A: + frozen RoleCorr K16/s4 structural residual block;
* ``CORR-ADD``      B: + frozen RoleCorr scaler/dictionary correspondence block;
* ``CORR-PCA-ADD``  C: + train-fitted PCA16 of the same correspondence object.

Route 2 (``stage=route2``, conditional) replaces the whole residual coordinate
with one frozen K48/s12 dictionary over the train-only scale-balanced joint
input (structural residual 65 + Sem108 108 + correspondence 536 = 709), plus a
PCA48 dense control.

The official ZINC **test** split is never instantiated; the control plane blocks
test access for non-terminal modes and every payload re-asserts
``official_test_loaded = false``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_increment_v2 as inc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_v1 as rc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_rolecorr_v1 as rcrun
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = inc.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_rolecorr_increment_v2"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
SOURCE_DIR = rcrun.RESULTS_DIR  # RoleCorr-v1 frozen artifacts, re-used read-only

THREADS = 8
SEED = 0
TRAIN_EPOCHS = 320
SMOKE_EPOCHS = 8
SMOKE_TRAIN_SUBSET = 1024
SMOKE_VALID_SUBSET = 512
BATCH_SIZE = int(p2run.BATCH_SIZE)
SHUFFLE_SEEDS = inc.SHUFFLE_SEEDS
COORD_SHUFFLE_SEEDS = inc.COORD_SHUFFLE_SEEDS
RESUME_EVERY = 10
DEFAULT_DEVICE = "cpu"

_write_json = v0run._write_json
_read_json = v0run._read_json
_write_csv = v0run._write_csv
_git_commit = v0run._git_commit

#: frozen identity of the read-only upstream artifacts (verified at run time)
EXPECTED = {
    "common_subspace_sha256": "36636ce92836bdb8d023cc91b3f532f8d4c46a57d457514c92d90c68028f6c24",
    "dictionary_struct_sha256": (
        "57494647b90a46f0890f6e67df2d772f5c964c34c2f37f0e0b26fa7a0a660bdd"
    ),
    "dictionary_corr_sha256": (
        "8d21e24b1c67e16a48e8a8c29a9ad5ee0ac88e4dc15a2ad8798c93197b1d94da"
    ),
    "corr_cache_train": {
        "n_patches": 231664,
        "node_sha256": "6b328d7ae859f420fbd0eb08369546dee1e6c875aec6ece4afc31a5abbae287f",
        "edge_sha256": "d710705e6da09741cd557493c750ba48ec3da54736a1f3f3bfbb796fd5f3e74d",
    },
    "corr_cache_valid": {
        "n_patches": 23083,
        "node_sha256": "15da71b968bd7fa9525e16eafd6b6b2ee8543852be44b2bfedc438777ce0c53b",
        "edge_sha256": "2388c80b30b8a3bc5bc74efb66a27f0a972a8b074aa2e93d1eee3d5e04decd10",
    },
}
#: the scaler is verified by refitting it from the raw cache (no file hash needed)
EXPECTED_SCALER_FILE_SHA256 = (
    "7c731ac9e7eee6bf06bda943134c2150875c7f6b0f3d88d0d9d2fa8fa25d3e76"
)

_DEVICE = torch.device("cpu")


def configure(
    *, epochs: int | None = None, threads: int | None = None, device: Any = None
) -> dict[str, Any]:
    """Fix the run knobs; every scientific constant stays frozen."""
    global TRAIN_EPOCHS, THREADS, _DEVICE
    if epochs is not None:
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive")
        TRAIN_EPOCHS = int(epochs)
    if threads is not None:
        if int(threads) <= 0:
            raise ValueError("threads must be positive")
        THREADS = int(threads)
    _DEVICE = inc.resolve_device(device if device is not None else DEFAULT_DEVICE)
    if _DEVICE.type == "cpu":
        torch.set_num_threads(THREADS)
    return {
        "epochs": int(TRAIN_EPOCHS),
        "threads": int(THREADS),
        **inc.device_report(_DEVICE),
    }


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_payload() -> dict[str, Any]:
    return {**inc.device_report(_DEVICE), "torch_threads": int(THREADS)}


# ---------------------------------------------------------------------------
# read-only upstream artifacts (RoleCorr-v1 caches / scaler / dictionaries)
# ---------------------------------------------------------------------------


def stage_cache(force: bool = False) -> dict[str, Any]:
    """Build (or verify) the frozen 536-D correspondence caches read-only."""
    _ensure_dirs()
    checks: dict[str, Any] = {}
    for split in ("train", "valid"):
        meta = rcrun.build_corr_cache(split, force=force)
        expected = EXPECTED[f"corr_cache_{split}"]
        row = {
            "n_patches": int(meta["n_patches"]),
            "node_sha256": str(meta["node_sha256"]),
            "edge_sha256": str(meta["edge_sha256"]),
            "matches_frozen": bool(
                int(meta["n_patches"]) == int(expected["n_patches"])
                and str(meta["node_sha256"]) == str(expected["node_sha256"])
                and str(meta["edge_sha256"]) == str(expected["edge_sha256"])
            ),
        }
        checks[split] = row
        if not row["matches_frozen"]:
            raise RuntimeError(f"{split} correspondence cache does not match the frozen identity")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "source": str(SOURCE_DIR.relative_to(REPO_ROOT)),
        "cache_checks": checks,
        **_device_payload(),
        "official_test_loaded": False,
    }
    inc.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "cache_check.json", payload)
    print(f"[cache] verified {checks['train']['n_patches']}/{checks['valid']['n_patches']}", flush=True)
    return payload


def load_reused_dictionary(name: str) -> np.ndarray:
    """Load a frozen RoleCorr-v1 dictionary and verify its SHA-256."""
    meta = _read_json(SOURCE_DIR / f"dictionary_{name}.json")
    dictionary = np.asarray(rcrun.load_dictionary(name), dtype=np.float32)
    sha = _sha256_array(dictionary)
    expected = str(meta["sha256_f32"])
    if sha != expected:
        raise RuntimeError(
            f"reused dictionary {name}: sha {sha} != recorded v1 sha {expected}"
        )
    frozen = {
        "struct": EXPECTED["dictionary_struct_sha256"],
        "corr": EXPECTED["dictionary_corr_sha256"],
    }[name]
    if sha != frozen:
        raise RuntimeError(
            f"reused dictionary {name}: sha {sha} != frozen round identity {frozen}"
        )
    return dictionary


def stage_verify_reused(force: bool = False) -> dict[str, Any]:
    """Verify every re-used RoleCorr-v1 artifact (SHAs + scaler refit)."""
    _ensure_dirs()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "source_dir": str(SOURCE_DIR.relative_to(REPO_ROOT)),
        **_device_payload(),
        "official_test_loaded": False,
    }
    subspace_path = (
        TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
    )
    payload["common_subspace_sha256"] = _sha256_file(subspace_path)
    payload["common_subspace_matches_frozen"] = bool(
        payload["common_subspace_sha256"] == EXPECTED["common_subspace_sha256"]
    )
    d_struct = load_reused_dictionary("struct")
    d_corr = load_reused_dictionary("corr")
    payload["dictionary_struct"] = {
        "shape": list(d_struct.shape),
        "sha256": _sha256_array(d_struct),
    }
    payload["dictionary_corr"] = {
        "shape": list(d_corr.shape),
        "sha256": _sha256_array(d_corr),
    }
    scaler = rcrun.load_scaler()
    raw_train = rcrun.load_raw_corr("train")
    refit = rc.fit_corr_scaler(raw_train["node"], raw_train["edge"])
    scaler_ok = bool(
        np.array_equal(scaler.node.scale, refit.node.scale)
        and np.array_equal(scaler.node.mask, refit.node.mask)
        and float(scaler.node.weight) == float(refit.node.weight)
        and np.array_equal(scaler.edge.scale, refit.edge.scale)
        and np.array_equal(scaler.edge.mask, refit.edge.mask)
        and float(scaler.edge.weight) == float(refit.edge.weight)
    )
    payload["scaler_reused"] = {
        "node_weight": float(scaler.node.weight),
        "edge_weight": float(scaler.edge.weight),
        "refit_matches_reused": scaler_ok,
        "file_sha256": _sha256_file(SOURCE_DIR / "standardizers.json"),
    }
    payload["passed"] = bool(
        payload["common_subspace_matches_frozen"] and scaler_ok
    )
    inc.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "verify_reused.json", payload)
    print(
        f"[verify] subspace={payload['common_subspace_matches_frozen']} "
        f"scaler_refit={scaler_ok} passed={payload['passed']}",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError(f"re-used artifact verification failed: {payload}")
    return payload


def stage_pca16(force: bool = False) -> dict[str, Any]:
    """Train-fitted PCA16 of the RoleCorr-scaled correspondence object."""
    _ensure_dirs()
    path = RESULTS_DIR / "pca16.json"
    if path.exists() and not force:
        return _read_json(path)
    scaled = rcrun.scaled_corr("train")
    started = time.perf_counter()
    pca = rc.fit_pca16(scaled, rank=inc.EXTRA_ATOMS)
    report = rc.pca16_report(pca, scaled)
    payload = pca.to_json()
    payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "report": report,
            "n_fit_rows": int(scaled.shape[0]),
            "input": "train scaled C (frozen RoleCorr scaler)",
            "seconds": float(time.perf_counter() - started),
            **_device_payload(),
            "official_test_loaded": False,
        }
    )
    inc.official_test_blocker(payload)
    _write_json(path, payload)
    torch.save(
        {"mean": torch.as_tensor(pca.mean), "components": torch.as_tensor(pca.components)},
        RESULTS_DIR / "pca16.pt",
    )
    print(
        f"[pca16] rank={report['rank']} code_rms0={report['code_rms'][0]:.4f}", flush=True
    )
    return payload


def load_pca16() -> rc.PCA16:
    return rc.PCA16.from_json(_read_json(RESULTS_DIR / "pca16.json"))


# ---------------------------------------------------------------------------
# shared model construction
# ---------------------------------------------------------------------------


def arm_kwargs(arm: str) -> dict[str, Any]:
    if arm in (inc.ARM_EXTRA_STRUCT,):
        return {"block_dictionary": load_reused_dictionary("struct")}
    if arm == inc.ARM_CORR_ADD:
        return {"block_dictionary": load_reused_dictionary("corr")}
    if arm == inc.ARM_CORR_PCA_ADD:
        pca = load_pca16()
        return {"pca_mean": pca.mean, "pca_components": pca.components}
    raise KeyError(f"arm {arm} is not a route-1 arm")


def _readout_reference_state() -> dict[str, torch.Tensor]:
    """The shared initialisation, created exactly once from the canonical arm."""
    path = RESULTS_DIR / "readout_reference_state.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    model = inc.build_increment_model(
        arm=inc.ARM_EXTRA_STRUCT,
        dictionary=rcrun.sdb_dictionary(),
        seed=SEED,
        subspace=rcrun.load_subspace(),
        **arm_kwargs(inc.ARM_EXTRA_STRUCT),
    )
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key not in ("D", "D_block")
    }
    torch.save(state, path)
    return state


def build_arm(arm: str, *, reference_state: Mapping[str, torch.Tensor] | None = None):
    return inc.build_increment_model(
        arm=arm,
        dictionary=rcrun.sdb_dictionary(),
        seed=SEED,
        subspace=rcrun.load_subspace(),
        freeze_dictionary=True,
        reference_state=reference_state,
        **arm_kwargs(arm),
    )


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def _attach_arm_block(
    data: Sequence[Any], split: str, arm: str, *, permute_seed: int | None = None
) -> None:
    """Attach the arm-specific third-block input to a split subset (in place)."""
    if arm in (inc.ARM_CORR_ADD, inc.ARM_CORR_PCA_ADD):
        if permute_seed is None:
            rcrun.attach_corr(data, split)
        else:
            values = rcrun.scaled_corr(split)
            rng = np.random.default_rng(int(permute_seed) + 7777)
            values = values[rng.permutation(int(values.shape[0]))]
            _attach_values(data, split, values)
    elif arm in (inc.ARM_JOINT_SPARSE, inc.ARM_JOINT_PCA):
        values = load_joint(split)
        if permute_seed is not None:
            rng = np.random.default_rng(int(permute_seed) + 7777)
            values = values[rng.permutation(int(values.shape[0]))]
        _attach_values(data, split, values, name="joint_vec")
    elif arm == inc.ARM_EXTRA_STRUCT:
        return  # computed inside the model from phi
    else:
        raise KeyError(f"unknown arm {arm}")


def _attach_values(
    data: Sequence[Any], split: str, values: np.ndarray, name: str = "corr_vec"
) -> None:
    node_sizes = [int(v) for v in rcrun._env_blob(split)["node_sizes"].tolist()]
    if len(data) > len(node_sizes):
        raise RuntimeError(f"attach got {len(data)} molecules but {split} has {len(node_sizes)}")
    expected = int(sum(node_sizes[: len(data)]))
    if expected > int(values.shape[0]):
        raise RuntimeError(f"attach needs {expected} rows but cache has {values.shape[0]}")
    offset = 0
    for index, item in enumerate(data):
        size = node_sizes[index]
        setattr(
            item,
            name,
            torch.as_tensor(values[offset : offset + size], dtype=torch.float32).clone(),
        )
        offset += size
    if offset != expected:
        raise RuntimeError(f"attach consumed {offset} rows != {expected}")


# ---------------------------------------------------------------------------
# stage: correctness (route 1)
# ---------------------------------------------------------------------------


def _extension_wiring_shift(model: Any, batch: Any) -> float:
    """Probe that the 16 appended coordinate columns are wired into the bindings.

    The shared initialisation zeroes those rows (so an untrained model is by
    design insensitive to the block).  This probe temporarily fills them with a
    deterministic pattern and requires the prediction to respond to zeroing the
    block coordinate; it restores the parameters afterwards.
    """
    if model.route != 1:
        return float("nan")
    with torch.no_grad():
        saved_a = model.W_A_S.detach().clone()
        saved_e = model.W_E_S.detach().clone()
        generator = torch.Generator().manual_seed(1234)
        pattern = torch.randn(
            (inc.EXTRA_ATOMS, saved_a.shape[1]), generator=generator, dtype=saved_a.dtype
        )
        model.W_A_S[inc.EXTRA_SLICE] = 0.1 * pattern
        for block in range(3):
            rows = slice(
                block * inc.COORD_DIM + inc.EXTRA_SLICE.start,
                block * inc.COORD_DIM + inc.EXTRA_SLICE.stop,
            )
            pattern_e = torch.randn(
                (inc.EXTRA_ATOMS, saved_e.shape[1]), generator=generator, dtype=saved_e.dtype
            )
            model.W_E_S[rows] = 0.1 * pattern_e
        real = model(batch, mask=cm.C6_MASK)
        model.inference_zero_block = True
        try:
            zeroed = model(batch, mask=cm.C6_MASK)
        finally:
            model.inference_zero_block = False
        model.W_A_S.copy_(saved_a)
        model.W_E_S.copy_(saved_e)
    return float((real - zeroed).abs().max())


def stage_correctness(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "correctness.json"
    if path.exists() and not force:
        cached = _read_json(path)
        if str(cached.get("device")) == str(_DEVICE):
            return cached
    reference = _readout_reference_state()
    models = {
        arm: build_arm(arm, reference_state=reference).to(_DEVICE)
        for arm in inc.ROUTE1_ARMS
    }
    for model in models.values():
        model.eval()
    train_subset = p1run.load_split("train", subset=64)
    _attach_arm_block(train_subset, "train", inc.ARM_CORR_ADD)
    loader = p1.make_env_loader(train_subset, 32, False, 0)
    batch = next(iter(loader)).to(_DEVICE)

    state_ext = models[inc.ARM_EXTRA_STRUCT].state_dict()
    shared_keys = [
        key
        for key in state_ext
        if key not in ("D", "D_block")
        and all(
            key in models[arm].state_dict()
            and tuple(models[arm].state_dict()[key].shape) == tuple(state_ext[key].shape)
            for arm in inc.ROUTE1_ARMS
        )
    ]
    mismatch: dict[str, list[str]] = {}
    for arm in inc.ROUTE1_ARMS:
        state = models[arm].state_dict()
        mismatch[arm] = [
            key
            for key in shared_keys
            if not torch.equal(state_ext[key], state[key])
        ]
    init_identical = all(not value for value in mismatch.values())

    per_arm: dict[str, Any] = {}
    with torch.no_grad():
        for arm in inc.ROUTE1_ARMS:
            model = models[arm]
            block = model.extract_block(batch)
            coord = model.code(batch.dict_phi, block)
            base_expected = models[inc.ARM_EXTRA_STRUCT].base_codes(batch.dict_phi)
            purity = inc.zero_block_purity(model, batch.dict_phi, block)
            base_purity = inc.zero_base_purity(model, batch.dict_phi, block)
            extension_zero = bool(
                torch.equal(
                    model.W_A_S[inc.EXTRA_SLICE],
                    torch.zeros_like(model.W_A_S[inc.EXTRA_SLICE]),
                )
            )
            weight = model.W_E_S.reshape(3, inc.COORD_DIM, -1)
            extension_zero = extension_zero and bool(
                torch.equal(
                    weight[:, inc.EXTRA_SLICE], torch.zeros_like(weight[:, inc.EXTRA_SLICE])
                )
            )
            per_arm[arm] = {
                "coordinate_dim": int(coord.shape[1]),
                "alpha_base_max_l0": int((coord[:, inc.BASE_SLICE] != 0).sum(dim=1).max()),
                "alpha_block_max_l0": int((coord[:, inc.EXTRA_SLICE] != 0).sum(dim=1).max()),
                "base_coordinate_matches_extra_struct_arm": bool(
                    torch.equal(coord[:, inc.BASE_SLICE], base_expected)
                ),
                "zero_block_purity": purity,
                "zero_base_purity": base_purity,
                "binding_extension_rows_zero": extension_zero,
                "block_dictionary_frozen": bool(
                    model.D_block is None or not model.D_block.requires_grad
                ),
                "base_dictionary_frozen": bool(not model.D.requires_grad),
                "coordinate_payload": inc.coordinate_payload(model),
            }
    # block sensitivity: perturb the attached block and check the coordinate moves,
    # then probe the binding wiring with a deterministic non-zero extension pattern
    # (the shared initialisation deliberately zeroes the extension rows).
    with torch.no_grad():
        model_b = models[inc.ARM_CORR_ADD]
        coord_real = model_b.code(batch.dict_phi, batch.corr_vec)
        perturbed = batch.clone()
        perturbed.corr_vec = batch.corr_vec + 0.5 * torch.randn_like(batch.corr_vec)
        coord_perturbed = model_b.code(batch.dict_phi, perturbed.corr_vec)
        block_coordinate_shift = float(
            (coord_real[:, inc.EXTRA_SLICE] - coord_perturbed[:, inc.EXTRA_SLICE])
            .abs()
            .max()
        )
        wiring_shift = _extension_wiring_shift(model_b, batch)
        coord_free = model_b.code(batch.dict_phi, batch.corr_vec)
        rec = model_b.reconstruction_loss(batch.dict_phi, coord_free)
        rec_param_grad_absent = bool(not coord_free.requires_grad)
    trainable = {
        arm: int(sum(p.numel() for p in models[arm].parameters() if p.requires_grad))
        for arm in inc.ROUTE1_ARMS
    }
    total = {
        arm: int(sum(p.numel() for p in models[arm].parameters()))
        for arm in inc.ROUTE1_ARMS
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "coordinate_slices": inc.coordinate_slices(),
        "arms": per_arm,
        "shared_init": {
            "shared_keys": int(len(shared_keys)),
            "bit_identical": bool(init_identical),
            "mismatch": mismatch,
        },
        "block_coordinate_shift": block_coordinate_shift,
        "block_coordinate_sensitive": bool(block_coordinate_shift > 1e-6),
        "extension_wiring_shift": wiring_shift,
        "extension_wiring_sensitive": bool(wiring_shift > 1e-6),
        "reconstruction_gradient_inert": bool(rec_param_grad_absent),
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    payload["all_passed"] = bool(
        all(entry["coordinate_dim"] == inc.COORD_DIM for entry in per_arm.values())
        and all(
            entry["base_coordinate_matches_extra_struct_arm"]
            for entry in per_arm.values()
        )
        and all(entry["zero_block_purity"]["other_columns_bit_identical"] for entry in per_arm.values())
        and all(entry["zero_block_purity"]["block_columns_zero"] for entry in per_arm.values())
        and all(
            entry["zero_base_purity"]["other_slices_bit_identical"]
            for entry in per_arm.values()
        )
        and all(entry["binding_extension_rows_zero"] for entry in per_arm.values())
        and all(entry["block_dictionary_frozen"] and entry["base_dictionary_frozen"] for entry in per_arm.values())
        and init_identical
        and payload["block_coordinate_sensitive"]
        and payload["extension_wiring_sensitive"]
        and payload["reconstruction_gradient_inert"]
        and len(set(trainable.values())) == 1
    )
    inc.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        f"[correctness] coord={inc.COORD_DIM} init={init_identical} "
        f"purity={all(e['zero_block_purity']['block_columns_zero'] for e in per_arm.values())} "
        f"trainable={trainable[inc.ARM_EXTRA_STRUCT]} passed={payload['all_passed']}",
        flush=True,
    )
    if not payload["all_passed"]:
        raise RuntimeError(f"correctness gates failed: {payload}")
    return payload


# ---------------------------------------------------------------------------
# device-parametric training loop (faithful to cssd.train_cssd semantics)
# ---------------------------------------------------------------------------


def _evaluate_device(
    model: Any, loader: Any, device: torch.device, mask: Any
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch, mask=mask)
            predictions.append(prediction.view(-1).detach().cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "n_molecules": int(target.size),
        "predictions": pred,
        "targets": target,
    }


def train_arm_device(
    *,
    arm: str,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    epochs: int,
    tag: str,
    out_dir: Path,
    reference_state: Mapping[str, torch.Tensor],
    save_states: bool = True,
    resume: bool = True,
    write_arm_artifacts: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """One frozen-protocol run on ``_DEVICE`` with epoch-level resumability.

    Data order, loss, optimiser, clipping and the Top-5 soup are exactly the
    frozen ``cssd.train_cssd`` semantics; only the device is explicit.  A run
    resumed from a checkpoint restarts the loader sequence (recorded as
    ``data_order_restart``) and is not bit-identical to an uninterrupted run.
    """
    device = _DEVICE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p2run._seed_everything(int(SEED))
    model = build_arm(arm, reference_state=reference_state).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(p2run.LEARNING_RATE),
        weight_decay=float(p2run.WEIGHT_DECAY),
    )
    loader = p1.make_env_loader(
        train_data, int(BATCH_SIZE), True, int(SEED) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    eval_loader = p1.make_env_loader(
        valid_data, int(BATCH_SIZE), False, int(SEED) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    soup_k = int(p2run.SOUP_K)
    resume_path = out_dir / f"{tag}_resume.pt"
    curve: list[dict[str, Any]] = []
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    resumed_at: int | None = None
    if resume and resume_path.exists():
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        curve = list(blob["curve"])
        epoch_states = dict(blob["epoch_states"])
        best_mae = float(blob["best_mae"])
        best_epoch = int(blob["best_epoch"])
        best_state = blob["best_state"]
        torch.set_rng_state(blob["rng_cpu"])
        if device.type == "cuda" and blob.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(blob["rng_cuda"])
        resumed_at = int(blob["epoch"]) + 1
        if log:
            print(f"[{tag}] resumed from epoch {blob['epoch']}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    epoch_times: list[float] = []
    for epoch in range(int(resumed_at or 1), int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum = rec_sum = rec_term_sum = 0.0
        n_mol = n_nodes = n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            rec_term_sum += float(rec.detach())
            n_batches += 1
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float(
                (((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + 1e-12)).sum()
            )
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        train_rec_term = float(rec_term_sum / max(n_batches, 1))
        valid = _evaluate_device(model, eval_loader, device, mask)
        epoch_times.append(float(time.perf_counter() - epoch_started))
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "train_rec_term": train_rec_term,
                "valid_mae": float(valid["mae"]),
                "d_norm": float(model.D.detach().norm()),
                "seconds": float(epoch_times[-1]),
            }
        )
        epoch_states[int(epoch)] = {
            k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
        }
        keep = {
            i + 1
            for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:soup_k]
        }
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {
                k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
            }
        if log and (epoch == 1 or epoch % 10 == 0 or epoch == int(epochs)):
            print(
                f"[{tag}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec_term:.3e} "
                f"valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
        if epoch % RESUME_EVERY == 0 or epoch == int(epochs):
            torch.save(
                {
                    "epoch": int(epoch),
                    "model": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "curve": curve,
                    "epoch_states": epoch_states,
                    "best_mae": float(best_mae),
                    "best_epoch": int(best_epoch),
                    "best_state": best_state,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                },
                resume_path,
            )
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    members = sorted(
        int(i) + 1
        for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:soup_k]
    )
    soup_state = {
        k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0)
        for k in epoch_states[members[0]]
    }
    soup_model = build_arm(arm, reference_state=reference_state).to(device)
    soup_model.load_state_dict(soup_state)
    soup_valid = _evaluate_device(soup_model, eval_loader, device, mask)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "arm": str(arm),
        "route": int(inc.ROUTE_OF_ARM[arm]),
        "seed": int(SEED),
        "mask": mask.as_dict(),
        "config": cm.H1_CONFIG.as_dict(),
        "epoch_budget": int(epochs),
        "epochs_run": int(len(curve)),
        "completed": bool(len(curve) == int(epochs)),
        "resumed_at_epoch": resumed_at,
        "data_order_restart": bool(resumed_at is not None),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(p2run.LEARNING_RATE),
        "weight_decay": float(p2run.WEIGHT_DECAY),
        "grad_clip": float(p2run.GRAD_CLIP),
        "lambda_rec": float(lam),
        "soup_k": int(soup_k),
        "actual_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "curve": curve,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "final_valid_mae": float(curve[-1]["valid_mae"]),
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "soup": {
            "members": members,
            "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
            "soup_valid_mae": float(soup_valid["mae"]),
        },
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve) - (resumed_at or 1) + 1, 1)),
        "epoch_seconds": epoch_times,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "best_state_sha256": _state_sha256(best_state),
        "soup_state_sha256": _state_sha256(soup_state),
        **_device_payload(),
        "official_test_loaded": False,
    }
    if save_states:
        torch.save(best_state, out_dir / f"{tag}_raw_state.pt")
        torch.save(soup_state, out_dir / f"{tag}_soup_state.pt")
        torch.save(
            {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
            out_dir / f"{tag}_final_state.pt",
        )
    if len(curve) == int(epochs) and resume_path.exists():
        # a completed run must not be silently warm-restarted by a later launch
        resume_path.unlink()
    inc.official_test_blocker(payload)
    if not write_arm_artifacts:
        return payload
    _write_json(RESULTS_DIR / f"run_{arm}.json", payload)
    _write_csv(RESULTS_DIR / f"curve_{arm}.csv", curve)
    _write_json(
        RESULTS_DIR / f"soup_{arm}.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "arm": str(arm),
            "tag": str(tag),
            "members": members,
            "member_valid_mae": payload["soup"]["member_valid_mae"],
            "soup_valid_mae": payload["soup"]["soup_valid_mae"],
            "best_valid_mae": payload["best_valid_mae"],
            "best_epoch": payload["best_epoch"],
            "soup_state_sha256": payload["soup_state_sha256"],
            "best_state_sha256": payload["best_state_sha256"],
        },
    )
    print(
        f"[train:{arm}] best={payload['best_valid_mae']:.6f}@{payload['best_epoch']} "
        f"soup={payload['soup']['soup_valid_mae']:.6f} wall={wall:.1f}s",
        flush=True,
    )
    return payload


def _gradient_probe(arm: str, reference_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    train_data = p1run.load_split("train", subset=64)
    _attach_arm_block(train_data, "train", arm)
    loader = p1.make_env_loader(train_data, 32, False, 0)
    batch = next(iter(loader)).to(_DEVICE)
    model = build_arm(arm, reference_state=reference_state).to(_DEVICE)
    model.train()
    prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    rec_term = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(
        cm.H1_LAMBDA
    ) * rec_term
    model.zero_grad(set_to_none=True)
    loss.backward()
    named = dict(model.named_parameters())
    reader_keys = sorted(name for name in named if name.startswith("reader."))
    probe_keys = ["W_A_S", "W_E_S", "fusion.0.weight"] + reader_keys[:1]
    grads: dict[str, float | None] = {}
    for name in probe_keys:
        parameter = named.get(name)
        grads[name] = (
            None
            if parameter is None or parameter.grad is None
            else float(parameter.grad.norm())
        )
    extension_grad = (
        None
        if model.W_A_S.grad is None
        else float(model.W_A_S.grad[inc.EXTRA_SLICE].norm())
    )
    finite = bool(torch.isfinite(loss).item() and torch.isfinite(prediction).all().item())
    return {
        "finite": finite,
        "loss": float(loss.detach()),
        "probe_keys": probe_keys,
        "reader_keys": reader_keys,
        "grad_norms": grads,
        "readout_grads_nonzero": all(
            grads[name] is not None and float(grads[name]) > 0.0 for name in probe_keys
        ),
        "binding_extension_grad_norm": extension_grad,
        "binding_extension_has_grad": bool(extension_grad is not None and extension_grad > 0.0),
        "dictionary_grads_absent": bool(
            model.D.grad is None and (model.D_block is None or model.D_block.grad is None)
        ),
    }


def stage_smoke(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "smoke.json"
    if path.exists() and not force:
        cached = _read_json(path)
        if str(cached.get("device")) == str(_DEVICE):
            return cached
    reference = _readout_reference_state()
    train_data = p1run.load_split("train", subset=SMOKE_TRAIN_SUBSET)
    valid_data = p1run.load_split("valid", subset=SMOKE_VALID_SUBSET)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "epochs": SMOKE_EPOCHS,
        "train_subset": SMOKE_TRAIN_SUBSET,
        "valid_subset": SMOKE_VALID_SUBSET,
        "arms": {},
    }
    for arm in inc.ROUTE1_ARMS:
        _attach_arm_block(train_data, "train", arm)
        _attach_arm_block(valid_data, "valid", arm)
        result = train_arm_device(
            arm=arm,
            train_data=train_data,
            valid_data=valid_data,
            epochs=SMOKE_EPOCHS,
            tag=f"INC-smoke-{arm}",
            out_dir=CHECKPOINT_DIR,
            reference_state=reference,
            save_states=False,
            resume=False,
            write_arm_artifacts=False,
            log=True,
        )
        probe = _gradient_probe(arm, reference)
        payload["arms"][arm] = {
            "best_valid_mae": float(result["best_valid_mae"]),
            "epochs_run": int(result["epochs_run"]),
            "wall_clock_s": float(result["wall_clock_s"]),
            "seconds_per_epoch": float(result["seconds_per_epoch"]),
            **probe,
        }
    payload["passed"] = bool(
        all(
            entry["finite"]
            and entry["readout_grads_nonzero"]
            and entry["binding_extension_has_grad"]
            and entry["dictionary_grads_absent"]
            and math.isfinite(entry["best_valid_mae"])
            for entry in payload["arms"].values()
        )
    )
    inc.official_test_blocker(payload)
    _write_json(path, payload)
    print(f"[smoke] passed={payload['passed']}", flush=True)
    if not payload["passed"]:
        raise RuntimeError(f"smoke failed: {payload}")
    return payload


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    if all((RESULTS_DIR / f"run_{arm}.json").exists() for arm in inc.ROUTE1_ARMS) and not force:
        print("[train] cache hit", flush=True)
        return {arm: _read_json(RESULTS_DIR / f"run_{arm}.json") for arm in inc.ROUTE1_ARMS}
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    smoke = _read_json(RESULTS_DIR / "smoke.json")
    if not correctness.get("all_passed") or not smoke.get("passed"):
        raise RuntimeError("correctness/smoke gates not passed; refusing to train")
    reference = _readout_reference_state()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    outputs: dict[str, Any] = {}
    for arm in inc.ROUTE1_ARMS:
        run_path = RESULTS_DIR / f"run_{arm}.json"
        if run_path.exists() and not force:
            existing = _read_json(run_path)
            if int(existing.get("epochs_run", 0)) >= int(TRAIN_EPOCHS):
                print(f"[train:{arm}] complete cache hit", flush=True)
                outputs[arm] = existing
                continue
        _attach_arm_block(train_data, "train", arm)
        _attach_arm_block(valid_data, "valid", arm)
        outputs[arm] = train_arm_device(
            arm=arm,
            train_data=train_data,
            valid_data=valid_data,
            epochs=int(TRAIN_EPOCHS),
            tag=f"INC-{arm}-seed0",
            out_dir=CHECKPOINT_DIR,
            reference_state=reference,
            save_states=True,
            resume=True,
            log=True,
        )
    return outputs


# ---------------------------------------------------------------------------
# stage: interventions (route 1)
# ---------------------------------------------------------------------------


def _soup_model(arm: str) -> Any:
    reference = _readout_reference_state()
    model = build_arm(arm, reference_state=reference).to(_DEVICE)
    path = CHECKPOINT_DIR / f"INC-{arm}-seed0_soup_state.pt"
    if not path.exists():
        raise RuntimeError(f"soup checkpoint missing: {path}")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model


def collect_codes(model: Any, loader: Any) -> dict[str, np.ndarray]:
    base: list[np.ndarray] = []
    block: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(_DEVICE)
            if model.route == 1:
                base.append(model.base_codes(batch.dict_phi).detach().cpu().numpy())
                block_codes = model.block_codes(
                    batch.dict_phi, model.extract_block(batch)
                )
            else:
                block_codes = model.block_codes(batch.dict_phi, model.extract_block(batch))
            block.append(block_codes.detach().cpu().numpy())
    out = {"block": np.concatenate(block, axis=0)}
    out["base"] = (
        np.concatenate(base, axis=0)
        if base
        else np.zeros((0, rc.BASE_ATOMS), dtype=np.float32)
    )
    return out


def _per_molecule(
    targets: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> dict[str, Any]:
    abs_a = np.abs(targets - pred_a)
    abs_b = np.abs(targets - pred_b)
    return {
        "mean_abs_error_A": float(abs_a.mean()),
        "mean_abs_error_B": float(abs_b.mean()),
        "fraction_molecules_B_better": float(np.mean(abs_b < abs_a)),
        "mean_abs_error_delta_B_minus_A": float((abs_b - abs_a).mean()),
        "median_abs_error_delta_B_minus_A": float(np.median(abs_b - abs_a)),
        "quantiles_delta_B_minus_A": {
            f"p{int(q * 100)}": float(np.quantile(abs_b - abs_a, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "n_molecules": int(abs_a.size),
    }


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "interventions.json"
    if path.exists() and not force:
        return _read_json(path)
    runs = {arm: _read_json(RESULTS_DIR / f"run_{arm}.json") for arm in inc.ROUTE1_ARMS}
    wrong_device = {
        arm: str(run.get("device"))
        for arm, run in runs.items()
        if str(run.get("device")) != str(_DEVICE)
    }
    if wrong_device:
        raise RuntimeError(
            f"formal runs on a different device; refusing frozen probes: {wrong_device}"
        )
    incomplete = {
        arm: int(run["epochs_run"])
        for arm, run in runs.items()
        if int(run["epochs_run"]) < int(run["epoch_budget"])
    }
    if incomplete:
        raise RuntimeError(f"incomplete formal runs; refusing frozen probes: {incomplete}")
    models = {arm: _soup_model(arm) for arm in inc.ROUTE1_ARMS}
    valid_data = p1run.load_split("valid")
    for arm in inc.ROUTE1_ARMS:
        _attach_arm_block(valid_data, "valid", arm)
    loader = p1.make_env_loader(
        valid_data, int(BATCH_SIZE), False, int(SEED) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    mask = cm.C6_MASK

    base: dict[str, dict[str, Any]] = {}
    for arm in inc.ROUTE1_ARMS:
        base[arm] = _evaluate_device(models[arm], loader, _DEVICE, mask)
    zero_block: dict[str, float] = {}
    zero_base: dict[str, float] = {}
    for arm in inc.ROUTE1_ARMS:
        model = models[arm]
        model.inference_zero_block = True
        zero_block[arm] = float(_evaluate_device(model, loader, _DEVICE, mask)["mae"])
        model.inference_zero_block = False
        if model.route == 1:
            model.inference_zero_base = True
            zero_base[arm] = float(_evaluate_device(model, loader, _DEVICE, mask)["mae"])
            model.inference_zero_base = False

    shuffle_rows: list[dict[str, Any]] = []
    for seed in COORD_SHUFFLE_SEEDS:
        row: dict[str, Any] = {"seed": int(seed)}
        for arm in inc.ROUTE1_ARMS:
            model = models[arm]
            model.inference_shuffle_block_seed = int(seed)
            mae = float(_evaluate_device(model, loader, _DEVICE, mask)["mae"])
            model.inference_shuffle_block_seed = None
            row[arm] = {"mae": mae, "delta": mae - float(base[arm]["mae"])}
        shuffle_rows.append(row)

    object_shuffle: list[dict[str, Any]] = []
    for arm in (inc.ARM_CORR_ADD, inc.ARM_CORR_PCA_ADD):
        for seed in SHUFFLE_SEEDS:
            _attach_arm_block(valid_data, "valid", arm, permute_seed=int(seed))
            mae = float(_evaluate_device(models[arm], loader, _DEVICE, mask)["mae"])
            object_shuffle.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "mae": mae,
                    "delta_vs_M": mae - float(base[arm]["mae"]),
                }
            )
        _attach_arm_block(valid_data, "valid", arm)
    _attach_arm_block(valid_data, "valid", inc.ARM_CORR_ADD)

    codes = {arm: collect_codes(models[arm], loader) for arm in inc.ROUTE1_ARMS}
    usage = {
        arm: {
            "base": rc.code_usage(codes[arm]["base"], rc.BASE_ATOMS),
            "block": rc.code_usage(
                codes[arm]["block"], inc.ARM_BLOCK_ATOMS[arm]
            ),
        }
        for arm in inc.ROUTE1_ARMS
    }
    m_a = float(base[inc.ARM_EXTRA_STRUCT]["mae"])
    m_b = float(base[inc.ARM_CORR_ADD]["mae"])
    m_c = float(base[inc.ARM_CORR_PCA_ADD]["mae"])
    g_block = {arm: zero_block[arm] - float(base[arm]["mae"]) for arm in inc.ROUTE1_ARMS}
    g_shuf: dict[str, float] = {}
    for arm in inc.ROUTE1_ARMS:
        deltas = [float(row[arm]["delta"]) for row in shuffle_rows]
        g_shuf[arm] = float(np.mean(deltas))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "M_EXTRA_STRUCT": m_a,
        "M_CORR_ADD": m_b,
        "M_CORR_PCA_ADD": m_c,
        "M_block_zero": zero_block,
        "G_block_zero": g_block,
        "M_base_zero": zero_base,
        "G_base_zero": {
            arm: zero_base[arm] - float(base[arm]["mae"]) for arm in zero_base
        },
        "coord_shuffle_rows": shuffle_rows,
        "G_block_shuffle": g_shuf,
        "object_shuffle_rows": object_shuffle,
        "G_object_shuffle_CORR": float(
            np.mean(
                [row["delta_vs_M"] for row in object_shuffle if row["arm"] == inc.ARM_CORR_ADD]
            )
        ),
        "code_usage": usage,
        "per_molecule": {
            "A_vs_B": _per_molecule(
                base[inc.ARM_EXTRA_STRUCT]["targets"],
                base[inc.ARM_EXTRA_STRUCT]["predictions"],
                base[inc.ARM_CORR_ADD]["predictions"],
            ),
            "A_vs_C": _per_molecule(
                base[inc.ARM_EXTRA_STRUCT]["targets"],
                base[inc.ARM_EXTRA_STRUCT]["predictions"],
                base[inc.ARM_CORR_PCA_ADD]["predictions"],
            ),
            "C_vs_B": _per_molecule(
                base[inc.ARM_CORR_PCA_ADD]["targets"],
                base[inc.ARM_CORR_PCA_ADD]["predictions"],
                base[inc.ARM_CORR_ADD]["predictions"],
            ),
        },
    }
    inc.official_test_blocker(payload)
    _write_json(path, payload)
    np.savez(
        RESULTS_DIR / "per_molecule_errors.npz",
        target=base[inc.ARM_EXTRA_STRUCT]["targets"],
        pred_A=base[inc.ARM_EXTRA_STRUCT]["predictions"],
        pred_B=base[inc.ARM_CORR_ADD]["predictions"],
        pred_C=base[inc.ARM_CORR_PCA_ADD]["predictions"],
    )
    print(
        f"[interventions] M_A={m_a:.6f} M_B={m_b:.6f} M_C={m_c:.6f} "
        f"G_block_zero={g_block} G_block_shuffle={g_shuf}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------


def _curve_summary(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = np.asarray([float(row["valid_mae"]) for row in curve], dtype=np.float64)
    train = np.asarray([float(row["train_mae"]) for row in curve], dtype=np.float64)
    return {
        "epochs": int(valid.size),
        "valid_first": float(valid[0]),
        "valid_last": float(valid[-1]),
        "valid_min": float(valid.min()),
        "valid_min_epoch": int(valid.argmin()) + 1,
        "train_last": float(train[-1]),
        "train_min": float(train.min()),
        "valid_last20_mean": float(valid[-20:].mean()),
    }


def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    runs = {arm: _read_json(RESULTS_DIR / f"run_{arm}.json") for arm in inc.ROUTE1_ARMS}
    wrong_device = {
        arm: str(run.get("device"))
        for arm, run in runs.items()
        if str(run.get("device")) != str(_DEVICE)
    }
    if wrong_device:
        raise RuntimeError(
            f"formal runs on a different device; refusing analysis: {wrong_device}"
        )
    interventions = _read_json(RESULTS_DIR / "interventions.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    m_a = float(runs[inc.ARM_EXTRA_STRUCT]["soup"]["soup_valid_mae"])
    m_b = float(runs[inc.ARM_CORR_ADD]["soup"]["soup_valid_mae"])
    m_c = float(runs[inc.ARM_CORR_PCA_ADD]["soup"]["soup_valid_mae"])
    improvement = m_a - m_b
    gate_fired = bool(improvement >= inc.SCREEN_ABS_GATE)
    dense_advantage = bool(m_b <= m_c - inc.SPARSE_DENSE_TOLERANCE)
    g_c0 = float(interventions["G_block_zero"][inc.ARM_CORR_ADD])
    g_cshuf = float(interventions["G_block_shuffle"][inc.ARM_CORR_ADD])
    g_obj = float(interventions["G_object_shuffle_CORR"])
    block_load_bearing = bool(
        g_c0 >= inc.GATE_MECHANISM_DIRECTIONAL
        and g_cshuf >= inc.GATE_MECHANISM_DIRECTIONAL
    )
    if not gate_fired:
        verdict = inc.VERDICTS["increment_no_gain"]
    elif not block_load_bearing:
        verdict = inc.VERDICTS["increment_block_unused"]
    elif dense_advantage:
        verdict = inc.VERDICTS["increment_gain_sparse"]
    else:
        verdict = inc.VERDICTS["increment_gain_dense"]
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        **_device_payload(),
        "official_test_loaded": False,
        "seed": int(SEED),
        "epochs": int(runs[inc.ARM_EXTRA_STRUCT]["epoch_budget"]),
        "coordinate_dim": inc.COORD_DIM,
        "M_A_soup": m_a,
        "M_B_soup": m_b,
        "M_C_soup": m_c,
        "M_A_best": float(runs[inc.ARM_EXTRA_STRUCT]["best_valid_mae"]),
        "M_B_best": float(runs[inc.ARM_CORR_ADD]["best_valid_mae"]),
        "M_C_best": float(runs[inc.ARM_CORR_PCA_ADD]["best_valid_mae"]),
        "improvement_B_minus_A": float(improvement),
        "screen_abs_gate": float(inc.SCREEN_ABS_GATE),
        "screen_gate_fired": gate_fired,
        "sparse_vs_dense": {
            "B_minus_C": float(m_b - m_c),
            "sparse_advantage": dense_advantage,
            "tolerance": float(inc.SPARSE_DENSE_TOLERANCE),
        },
        "mechanism": {
            "G_block_zero_B": g_c0,
            "G_block_shuffle_B": g_cshuf,
            "G_object_shuffle_B": g_obj,
            "G_block_zero_A": float(interventions["G_block_zero"][inc.ARM_EXTRA_STRUCT]),
            "G_block_shuffle_A": float(interventions["G_block_shuffle"][inc.ARM_EXTRA_STRUCT]),
            "G_block_zero_C": float(interventions["G_block_zero"][inc.ARM_CORR_PCA_ADD]),
            "G_block_shuffle_C": float(interventions["G_block_shuffle"][inc.ARM_CORR_PCA_ADD]),
            "block_load_bearing_B": block_load_bearing,
            "directional_threshold": float(inc.GATE_MECHANISM_DIRECTIONAL),
            "clear_threshold": float(inc.GATE_MECHANISM_CLEAR),
        },
        "per_molecule": interventions["per_molecule"],
        "code_usage": interventions["code_usage"],
        "curves": {arm: _curve_summary(runs[arm]["curve"]) for arm in inc.ROUTE1_ARMS},
        "soup": {arm: dict(runs[arm]["soup"]) for arm in inc.ROUTE1_ARMS},
        "wall_clock_s": {
            arm: float(runs[arm]["wall_clock_s"]) for arm in inc.ROUTE1_ARMS
        },
        "seconds_per_epoch": {
            arm: float(runs[arm]["seconds_per_epoch"]) for arm in inc.ROUTE1_ARMS
        },
        "peak_gpu_memory_bytes": {
            arm: runs[arm].get("peak_gpu_memory_bytes") for arm in inc.ROUTE1_ARMS
        },
        "parameters": {
            "trainable": correctness["trainable_parameters"],
            "total": correctness["total_parameters"],
        },
        "correctness_all_passed": bool(correctness["all_passed"]),
        "verdict": verdict,
    }
    inc.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "summary.json", payload)
    _write_report(payload, runs)
    _write_decision(payload)
    print(
        f"[analysis] M_A={m_a:.6f} M_B={m_b:.6f} M_C={m_c:.6f} "
        f"d(B-A)={improvement:+.6f} gate={gate_fired} verdict={verdict}",
        flush=True,
    )
    return payload


def _write_report(summary: Mapping[str, Any], runs: Mapping[str, Any]) -> None:
    lines = [
        "# E2E-DictEnv-RoleCorr-Increment-v2 - report\n\n",
        f"- protocol: `{PROTOCOL_VERSION}` (study `zinc-context-gap`, seed 0)\n",
        f"- git commit: `{summary['git_commit']}`\n",
        f"- device: `{summary['device']}`\n",
        "- official test loaded: `false` (never instantiated)\n",
        f"- verdict: **{summary['verdict']}**\n\n",
        "## Primary result (official valid, Top-5 soup, width 49)\n\n",
        "| arm | third block | best valid MAE | soup valid MAE |\n",
        "|---|---|---:|---:|\n",
        f"| A `EXTRA-STRUCT` | frozen K16/s4 structural residual | "
        f"{float(runs[inc.ARM_EXTRA_STRUCT]['best_valid_mae']):.6f} | {float(summary['M_A_soup']):.6f} |\n",
        f"| B `CORR-ADD` | frozen K16/s4 correspondence (RoleCorr D_C) | "
        f"{float(runs[inc.ARM_CORR_ADD]['best_valid_mae']):.6f} | {float(summary['M_B_soup']):.6f} |\n",
        f"| C `CORR-PCA-ADD` | train-fitted PCA16 of the same object | "
        f"{float(runs[inc.ARM_CORR_PCA_ADD]['best_valid_mae']):.6f} | {float(summary['M_C_soup']):.6f} |\n",
        f"\n`M_A - M_B = {float(summary['improvement_B_minus_A']):+.6f}` "
        f"(frozen absolute gate `>= {float(summary['screen_abs_gate']):.3f}`, "
        f"fired: `{summary['screen_gate_fired']}`).\n\n",
        "## Mechanism probes (soup states)\n\n",
        f"- block zero B: G = {float(summary['mechanism']['G_block_zero_B']):+.6f}\n",
        f"- block shuffle B: G = {float(summary['mechanism']['G_block_shuffle_B']):+.6f}\n",
        f"- object shuffle B: G = {float(summary['mechanism']['G_object_shuffle_B']):+.6f}\n",
        f"- block zero/shuffle A: {float(summary['mechanism']['G_block_zero_A']):+.6f} / "
        f"{float(summary['mechanism']['G_block_shuffle_A']):+.6f}\n",
        f"- block zero/shuffle C: {float(summary['mechanism']['G_block_zero_C']):+.6f} / "
        f"{float(summary['mechanism']['G_block_shuffle_C']):+.6f}\n\n",
        "## Curves\n\n| arm | first | last | min | min epoch | last-20 mean |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for arm in inc.ROUTE1_ARMS:
        row = summary["curves"][arm]
        lines.append(
            f"| {arm} | {row['valid_first']:.6f} | {row['valid_last']:.6f} | "
            f"{row['valid_min']:.6f} | {row['valid_min_epoch']} | {row['valid_last20_mean']:.6f} |\n"
        )
    lines += [
        "\n## Paired per-molecule differences (official valid)\n\n",
    ]
    for key, row in summary["per_molecule"].items():
        lines.append(
            f"- {key}: mean delta = {row['mean_abs_error_delta_B_minus_A']:+.6f}, "
            f"median = {row['median_abs_error_delta_B_minus_A']:+.6f}, "
            f"fraction B better = {row['fraction_molecules_B_better']:.4f}\n"
        )
    lines += [
        "\n## Scope notes\n\n",
        "- All three arms keep the **full** frozen K32/s8 structural budget and share a\n",
        "  bit-identical readout initialisation; only the appended 16-wide block differs.\n",
        "- Single seed; this is a screening round, not a significance claim.\n",
        "- Route 2 (joint 709-D encoding) is only authorised after a route-1 gate miss.\n",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    verdict = str(summary["verdict"])
    if verdict == inc.VERDICTS["increment_no_gain"]:
        body = (
            "The frozen absolute gate did not fire: appending the correspondence "
            "coordinate on top of the full structural budget does not improve official-valid "
            "Top-5 soup MAE by >= 0.003 over the matched extra-structural control. Per the "
            "pre-registration route 1 stops here; no K/s/LR/horizon rescue and no seed purchase. "
            "Route 1 answers the increment question negatively under this budget."
        )
    elif verdict == inc.VERDICTS["increment_block_unused"]:
        body = (
            "The gate fired numerically but the frozen probes show the appended block is not "
            "load-bearing (zeroing / shuffling it does not materially hurt). The gain is not "
            "attributable to the new coordinate."
        )
    elif verdict == inc.VERDICTS["increment_gain_sparse"]:
        body = (
            "The gate fired, the block is load-bearing, and sparse dictionary coding of the "
            "correspondence is not worse than the dense PCA16 control: `correspondence has an "
            "increment on top of the full structural budget, sparse-specific`. Single seed; the "
            "next step is a paired-seed confirmation under a new pre-registration."
        )
    elif verdict == inc.VERDICTS["increment_gain_dense"]:
        body = (
            "The gate fired and the block is load-bearing, but the PCA16 control is at least as "
            "good as the sparse dictionary: report `correspondence has an increment, "
            "sparse-dictionary advantage not supported`. No sparse-specific claim may be made."
        )
    else:
        body = f"The round is `{verdict}`; repair the blocking condition and re-run."
    lines = [
        "# Decision - E2E-DictEnv-RoleCorr-Increment-v2 (route 1)\n\n",
        f"- verdict: **{verdict}**\n",
        f"- `M_A = {float(summary['M_A_soup']):.9f}`, `M_B = {float(summary['M_B_soup']):.9f}`, "
        f"`M_C = {float(summary['M_C_soup']):.9f}`\n\n",
        body,
        "\n\n## Forbidden without a new pre-registration\n\n",
        "- seed 1 of any arm;\n",
        "- any K / sparsity / K-SVD-epoch / scaler / PCA-rank / width / horizon change;\n",
        "- end-to-end (unfrozen-dictionary) fine-tuning of any arm;\n",
        "- touching the official ZINC test split;\n",
        "- re-using these numbers as a replacement for the historical Sem108 result.\n",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# route 2 (conditional): joint 709-D encoding
# ---------------------------------------------------------------------------

JOINT_CACHE_TRAIN = RESULTS_DIR / "cache/corr_joint_train.pt"
JOINT_CACHE_VALID = RESULTS_DIR / "cache/corr_joint_valid.pt"

_JOINT_MEMORY: dict[str, np.ndarray] = {}


def _joint_raw_blocks(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (struct residual, Sem108, corr) blocks in patch order."""
    data = p1run.load_split(split)
    components = torch.as_tensor(
        rcrun.load_subspace().components, dtype=torch.float32
    )
    struct_parts: list[np.ndarray] = []
    sem_parts: list[np.ndarray] = []
    for item in data:
        residual = item.dict_phi - (item.dict_phi @ components) @ components.t()
        struct_parts.append(residual.numpy())
        sem_parts.append(item.patch_cont[:, : inc.SEM_DIM].numpy())
    raw_corr = rcrun.load_raw_corr(split)
    struct = np.concatenate(struct_parts, axis=0).astype(np.float64)
    sem = np.concatenate(sem_parts, axis=0).astype(np.float64)
    corr = np.concatenate([raw_corr["node"], raw_corr["edge"]], axis=1).astype(np.float64)
    if not (struct.shape[0] == sem.shape[0] == corr.shape[0]):
        raise RuntimeError(
            f"{split}: joint block row mismatch {struct.shape[0]}/{sem.shape[0]}/{corr.shape[0]}"
        )
    return struct, sem, corr


def stage_joint_scaler(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "joint_standardizers.json"
    if path.exists() and not force:
        return _read_json(path)
    struct, sem, corr = _joint_raw_blocks("train")
    scaler = inc.fit_joint_scaler(struct, sem, corr)
    payload = scaler.to_json()
    payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "n_fit_rows": int(struct.shape[0]),
            "report": inc.joint_block_report(struct, sem, corr),
            **_device_payload(),
            "official_test_loaded": False,
        }
    )
    inc.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        "[joint-scaler] weights="
        + ", ".join(f"{name}:{float(getattr(scaler, name).weight):.4f}" for name in inc.JOINT_BLOCKS),
        flush=True,
    )
    return payload


def load_joint_scaler() -> inc.JointScaler:
    path = RESULTS_DIR / "joint_standardizers.json"
    if not path.exists():
        raise RuntimeError("joint_standardizers.json missing; run the `joint_scaler` stage first")
    return inc.JointScaler.from_json(_read_json(path))


def build_joint_cache(split: str, force: bool = False) -> dict[str, Any]:
    path = JOINT_CACHE_TRAIN if split == "train" else JOINT_CACHE_VALID
    meta_path = RESULTS_DIR / f"cache_meta_joint_{split}.json"
    if path.exists() and meta_path.exists() and not force:
        return _read_json(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    struct, sem, corr = _joint_raw_blocks(split)
    scaled = inc.apply_joint_scaler(load_joint_scaler(), struct, sem, corr)
    torch.save({"joint": torch.as_tensor(scaled, dtype=torch.float32)}, path)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "n_patches": int(scaled.shape[0]),
        "dim": int(scaled.shape[1]),
        "sha256_f32": _sha256_array(scaled),
        "seconds": None,
        "official_test_loaded": False,
    }
    _write_json(meta_path, report)
    print(f"[joint-cache:{split}] n={report['n_patches']}", flush=True)
    return report


def load_joint(split: str) -> np.ndarray:
    key = split
    if key in _JOINT_MEMORY:
        return _JOINT_MEMORY[key]
    path = JOINT_CACHE_TRAIN if split == "train" else JOINT_CACHE_VALID
    if not path.exists():
        raise RuntimeError(f"joint cache missing: {path}; run the `joint_cache` stage first")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    values = blob["joint"].numpy()
    _JOINT_MEMORY[key] = values
    return values


def stage_joint_objects(force: bool = False) -> dict[str, Any]:
    """Fit the frozen K48/s12 joint dictionary and the PCA48 control (train only)."""
    _ensure_dirs()
    meta_path = RESULTS_DIR / "joint_objects.json"
    if meta_path.exists() and not force:
        return _read_json(meta_path)
    scaled = load_joint("train")
    started = time.perf_counter()
    D, info = inc.fit_joint_dictionary(scaled, log=print)
    torch.save({"D": torch.as_tensor(D, dtype=torch.float32)}, RESULTS_DIR / "dictionary_joint.pt")
    pca_started = time.perf_counter()
    pca = inc.fit_joint_pca(scaled, rank=inc.JOINT_ATOMS)
    torch.save(
        {"mean": torch.as_tensor(pca.mean), "components": torch.as_tensor(pca.components)},
        RESULTS_DIR / "joint_pca48.pt",
    )
    _write_json(RESULTS_DIR / "joint_pca48.json", pca.to_json())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "dictionary_joint": {
            "shape": list(D.shape),
            "atoms": inc.JOINT_ATOMS,
            "sparsity": inc.JOINT_SPARSITY,
            "ksvd_epochs": inc.DICT_EPOCHS,
            "dict_seed": inc.DICT_SEED,
            "n_fit_rows": int(scaled.shape[0]),
            "ksvd_final_fit_mse": float(info["history"][-1]["mean_sq_err"]),
            "sha256_f32": _sha256_array(D),
            "seconds": float(time.perf_counter() - started),
        },
        "joint_pca48": {
            "shape": list(pca.components.shape),
            "seconds": float(time.perf_counter() - pca_started),
            **_device_payload(),
        },
    }
    inc.official_test_blocker(payload)
    _write_json(meta_path, payload)
    print(
        f"[joint-objects] ksvd_mse={payload['dictionary_joint']['ksvd_final_fit_mse']:.6f}",
        flush=True,
    )
    return payload


def load_joint_dictionary() -> np.ndarray:
    blob = torch.load(RESULTS_DIR / "dictionary_joint.pt", map_location="cpu", weights_only=True)
    return np.asarray(blob["D"].numpy(), dtype=np.float32)


def load_joint_pca() -> rc.PCA16:
    return rc.PCA16.from_json(_read_json(RESULTS_DIR / "joint_pca48.json"))


def arm_kwargs_route2(arm: str) -> dict[str, Any]:
    if arm == inc.ARM_JOINT_SPARSE:
        return {"block_dictionary": load_joint_dictionary()}
    if arm == inc.ARM_JOINT_PCA:
        pca = load_joint_pca()
        return {"pca_mean": pca.mean, "pca_components": pca.components}
    raise KeyError(f"arm {arm} is not a route-2 arm")


def build_arm_route2(arm: str, *, reference_state: Mapping[str, torch.Tensor] | None = None):
    return inc.build_increment_model(
        arm=arm,
        dictionary=rcrun.sdb_dictionary(),
        seed=SEED,
        subspace=rcrun.load_subspace(),
        freeze_dictionary=True,
        reference_state=reference_state,
        **arm_kwargs_route2(arm),
    )


def _route2_reference_state() -> dict[str, torch.Tensor]:
    path = RESULTS_DIR / "readout_reference_state_route2.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    model = build_arm_route2(inc.ARM_JOINT_SPARSE)
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key not in ("D", "D_block")
    }
    torch.save(state, path)
    return state


def run_route2(stages: Sequence[str]) -> dict[str, Any]:
    print("[route2] conditional joint-encoding screen", flush=True)
    results: dict[str, Any] = {}
    if "joint_scaler" in stages:
        results["joint_scaler"] = stage_joint_scaler()
    if "joint_cache" in stages:
        for split in ("train", "valid"):
            build_joint_cache(split)
    if "joint_objects" in stages:
        results["joint_objects"] = stage_joint_objects()
    return results


# ---------------------------------------------------------------------------
# stage: orchestration
# ---------------------------------------------------------------------------

ROUTE1_STAGES = (
    "cache",
    "verify",
    "pca16",
    "correctness",
    "smoke",
    "train",
    "interventions",
    "analysis",
)
ROUTE2_STAGES = ("joint_scaler", "joint_cache", "joint_objects")


def run_route1(stages: Sequence[str]) -> dict[str, Any]:
    _ensure_dirs()
    print(
        f"[runner] commit={_git_commit()} protocol={PROTOCOL_VERSION} device={_DEVICE}",
        flush=True,
    )
    results: dict[str, Any] = {}
    if "cache" in stages:
        results["cache"] = stage_cache()
    if "verify" in stages:
        results["verify"] = stage_verify_reused()
    if "pca16" in stages:
        results["pca16"] = stage_pca16()
    if "correctness" in stages:
        results["correctness"] = stage_correctness()
    if "smoke" in stages:
        results["smoke"] = stage_smoke()
    if "train" in stages:
        results["train"] = stage_train()
    if "interventions" in stages:
        results["interventions"] = stage_interventions()
    if "analysis" in stages:
        results["analysis"] = stage_analysis()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="E2E-DictEnv-RoleCorr-Increment-v2 stage runner"
    )
    parser.add_argument("--stage", required=True, choices=("route1", "route2", "all"))
    parser.add_argument("--only", default=None, help="comma-separated stage subset override")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    configure(device=args.device)
    only = [part.strip() for part in args.only.split(",") if part.strip()] if args.only else None
    if args.stage in ("route1", "all"):
        stages = list(ROUTE1_STAGES)
        if only:
            stages = [stage for stage in stages if stage in set(only)]
        run_route1(stages)
    if args.stage in ("route2", "all"):
        stages = list(ROUTE2_STAGES)
        if only:
            stages = [stage for stage in stages if stage in set(only)]
        run_route2(stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
