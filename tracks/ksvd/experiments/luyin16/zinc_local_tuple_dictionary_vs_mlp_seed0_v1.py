"""ZINC local tuple interface: is the IHT dictionary ``f_D`` worth keeping
relative to a same-parameter-count local nonlinear projection ``f_M``?

Round: ``zinc_local_tuple_dictionary_vs_mlp_seed0_v1``.

One new formal arm, seed 0, exactly 240 epochs:

* ``M`` (this round) — ``f_M(x) = SiLU(x @ A_bar.T)`` with row-L2-normalised
  ``A_bar`` (64x125, 8,000 parameters), per-tuple nonlinearity first, then the
  *real* correspondence aggregation ``e_M(v) = sum_{t,a} w_J(v,t,a) f_M(x)``;
  injected through the same zero-init ``W_loc`` (342x64) into the fusion
  first-layer preactivation.
* ``D_J`` (previous round, read-only) — the tied 10-step 8-sparse IHT
  dictionary with the same input, aggregation, injection location and shared
  initial body/bridge/W_loc.

Everything else (125-D tuple features, real ``pair_wJ``, pointer maps, root
mapping, fit-only ``phi`` scaler, fit-only ``g = y - c`` targets, body,
posterior MLP bridge, schedule and optimizer) is inherited from the previous
round byte-for-byte through the committed module.

Usage::

    uv run python -m tracks.ksvd.experiments.luyin16.\
zinc_local_tuple_dictionary_vs_mlp_seed0_v1 --phase-a
    ... --smoke --device cuda
    ... --arm M --device cuda
    ... --analyze
    ... --mechanism
    ... --replay
    ... --manifest
    ... --budget
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import (
    zinc_chemistry_dictionary_vs_mlp_seed0_v1 as zcdm,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_full_cycle_target_decomposition_v1 as zftd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_joint_dictionary_decision_v1 as zjd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

PROTOCOL_VERSION = "zinc-local-tuple-dictionary-vs-mlp-seed0-v1"
RESULT_SLUG = "zinc_local_tuple_dictionary_vs_mlp_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results" / RESULT_SLUG
PREV_DIR = prev.RESULTS_DIR
REF_DIR = zcdm.RESULTS_DIR
TUPLE_NPZ = prev.TUPLE_NPZ

# ---- frozen recipe / references (previous round, read-only) ----------------
SEED = prev.SEED
EPOCHS = prev.EPOCHS
LR = prev.LR
WEIGHT_DECAY = prev.WEIGHT_DECAY
GRAD_CLIP = prev.GRAD_CLIP
BATCH_SIZE = prev.BATCH_SIZE
SOUP_EPOCHS = prev.SOUP_EPOCHS
TRAIN_SHUFFLE_OFFSET = prev.TRAIN_SHUFFLE_OFFSET
FOLD_SEED = prev.FOLD_SEED
N_FIT = prev.N_FIT
N_DEV = prev.N_DEV
BOOT_SEED = prev.BOOT_SEED
N_BOOT = prev.N_BOOT
LOG_EPOCHS = prev.LOG_EPOCHS
REPLAY_TOL = prev.REPLAY_TOL
TARGET_NOT_READ_TOL = prev.TARGET_NOT_READ_TOL

PHI_DIM = prev.PHI_DIM
ATOM_CATEGORIES = prev.ATOM_CATEGORIES
BOND_CATEGORIES = prev.BOND_CATEGORIES
TUPLE_DIM = prev.TUPLE_DIM
K_TUPLE = prev.K_TUPLE
W_LOC_OUT = prev.W_LOC_OUT
FRAME_SEED = prev.FRAME_SEED
KAPPA_SEED = prev.KAPPA_SEED
KAPPA_SAMPLE_MAX = prev.KAPPA_SAMPLE_MAX

A_PARAMETERS = TUPLE_DIM * K_TUPLE                  # 8,000
W_LOC_PARAMETERS = W_LOC_OUT * K_TUPLE              # 21,888
EXPECTED_BODY_PARAMETERS = prev.EXPECTED_BODY_PARAMETERS  # 184,667
BRIDGE_PARAMETERS = prev.BRIDGE_PARAMETERS                # 82,944
EXPECTED_TOTAL_PARAMETERS = EXPECTED_BODY_PARAMETERS + BRIDGE_PARAMETERS + A_PARAMETERS + W_LOC_PARAMETERS  # 297,499

# ---- frozen anchors --------------------------------------------------------
FROZEN_FIT_SHA = "7bf1cfb856a79617baf83c4ba95290d08248b4633609fa24725a5c028c9096d9"
FROZEN_DEV_SHA = "a61c801073fbda237a46e4f8fe6c8e4f8f24e867e7a8b452f7aaffc86a9cceff"
FROZEN_SCHEDULE_SHA = "7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65"
FROZEN_TARGET_SHA = "e2adf5f2d8ef9fc9ca6e7ecfb948ae73e5b1178645ddc5df74d2e171f08078ac"
FROZEN_TUPLE_SHA = "4facb6ec77ff10130d49857bad461d16da0ee097495616ca4d4711de962b2c40"
FROZEN_J_INIT_SHA = "185db5ede18a940ef4888bb42db6cbb08f69dca694bb2857c8b7f88e5ce14e4b"
FROZEN_J_SOUP_SHA = "95122a3f8a9ba2a7106ad8d58b38e2e7954514df4c0b67588a9d1f63512b2482"
FROZEN_J_DLOC_SHA = "807d0247bed0c968139265c725937f99993cc75b3475cae84b7a4e16bc2d89f6"
ANCHOR_B_DEV_OVERALL_CAL = 0.10371734
ANCHOR_B_DEV_G0_CAL = 0.10187498
ANCHOR_DJ_DEV_OVERALL_CAL = 0.10190711
ANCHOR_DJ_DEV_G0_CAL = 0.10041624
ANCHOR_DJ_FIT_OVERALL_CAL = 0.02815521
ANCHOR_TOL = 1.0e-5

# ---- frozen gates ----------------------------------------------------------
GATE_DELTA = 0.003
EQUIV_BAND = 0.003
PERF_DELTA = 0.003

#: supplied expected calibrated four-grid values (previous round, read-only)
FOURGRID_EXPECTED = {
    "overall": {"JJ": 0.1019071074, "JI": 0.1019913913, "IJ": 0.1045416120, "II": 0.1046819255},
    "G0": {"JJ": 0.1004162377, "JI": 0.1005100907, "IJ": 0.1038224926, "II": 0.1039915517},
}

jsonable = prev.jsonable
write_json = prev.write_json
file_sha256 = prev.file_sha256
state_hash = prev.state_hash
seed_everything = prev.seed_everything
group_masks = prev.group_masks
_metric_table = prev._metric_table
_bootstrap_self_tests = prev._bootstrap_self_tests
_ci_inside = prev._ci_inside
_array_sha256 = prev._array_sha256


def _load_prev_predictions(arm: str) -> dict[str, np.ndarray]:
    with np.load(PREV_DIR / f"{arm}_raw_predictions.npz") as z:
        return {key: np.asarray(z[key], np.float64) for key in z.files}


def _load_ref_predictions() -> dict[str, np.ndarray]:
    with np.load(REF_DIR / "M_raw_predictions.npz") as z:
        return {key: np.asarray(z[key], np.float64) for key in z.files}


# ---------------------------------------------------------------------------
# 1. the nonlinear projection encoder M (only new trainable block)
# ---------------------------------------------------------------------------


class LocalTupleEncoderM(nn.Module):
    """``f_M(x) = SiLU(x @ A_bar.T)`` per tuple, then real-incidence pooling.

    ``A_raw`` is 64x125 (8,000 parameters); ``A_bar`` is row-L2 normalised.
    No bias, no gain, no normalisation layer, no second layer, no dropout.
    ``W_loc`` stays 342x64 zero-init and is the only injection into the fusion
    first-layer preactivation, exactly as in the dictionary arm.
    """

    def __init__(self, payload: prev.TuplePayload, a_init: torch.Tensor) -> None:
        super().__init__()
        if tuple(a_init.shape) != (K_TUPLE, TUPLE_DIM):
            raise ValueError(f"A_raw shape {tuple(a_init.shape)} != {(K_TUPLE, TUPLE_DIM)}")
        self.A_raw = nn.Parameter(a_init.detach().clone().contiguous())
        self.W_loc = nn.Parameter(torch.zeros(W_LOC_OUT, K_TUPLE, dtype=torch.float32))
        self.kappa = 1.0
        self.kappa_D = float(payload.kappa)
        self._payload = payload
        self._device_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.ablate = False
        self.mean_replace: torch.Tensor | None = None
        self.switch_independent = False
        self.last_stats: dict[str, Any] = {}

    # -- payload / feature helpers -----------------------------------------
    def _buf(self, name: str, device: torch.device) -> torch.Tensor:
        cache = self._device_cache.setdefault(str(device), {})
        if name not in cache:
            cache[name] = self._payload[name].to(device)
        return cache[name]

    def tuple_features(
        self,
        phi_raw: torch.Tensor,
        atom_v: torch.Tensor,
        atom_u: torch.Tensor,
        bond_t: torch.Tensor,
    ) -> torch.Tensor:
        return prev.tuple_features(
            phi_raw,
            atom_v,
            atom_u,
            bond_t,
            self._buf("phi_mean", phi_raw.device),
            self._buf("phi_std", phi_raw.device),
            self._payload.phi_scale,
        )

    def tuple_codes(self, x: torch.Tensor) -> torch.Tensor:
        """``SiLU(x @ A_bar.T)`` — per-tuple nonlinear code (no aggregation)."""
        a_bar = F.normalize(self.A_raw, dim=1, eps=1e-12)
        return F.silu(x @ a_bar.t())

    # -- production root-code path (per tuple first, then index_add) --------
    def root_codes(self, data: Any) -> torch.Tensor:
        phi = data.dict_phi
        atom = data.dict_atom.to(torch.long)
        device = phi.device
        n_rows = int(phi.shape[0])
        mol = data.local_mol_id.to(torch.long).reshape(-1)
        n_graphs = int(mol.numel())
        batch = data.batch.to(torch.long)
        graph_counts = torch.bincount(batch, minlength=n_graphs)
        ptr = torch.cat([graph_counts.new_zeros(1), graph_counts.cumsum(0)])
        rows = torch.arange(n_rows, device=device)
        local_pos = rows - ptr[batch]
        root_base = self._buf("root_base", device)
        global_root = root_base[mol][batch] + local_pos
        pair_ptr = self._buf("pair_ptr", device)
        start = pair_ptr[global_root]
        end = pair_ptr[global_root + 1]
        counts = end - start
        total = int(counts.sum().item())
        zeros = torch.zeros(n_rows, K_TUPLE, device=device, dtype=phi.dtype)
        if total == 0:
            out = zeros
        else:
            offsets = counts.cumsum(0) - counts
            flat = torch.repeat_interleave(start, counts) + (
                torch.arange(total, device=device) - torch.repeat_interleave(offsets, counts)
            )
            tuple_row = torch.repeat_interleave(rows, counts)
            t = self._buf("pair_t", device)[flat]
            a_u = self._buf("pair_a", device)[flat]
            key = "pair_wI" if self.switch_independent else "pair_wJ"
            w = self._buf(key, device)[flat]
            x = self.tuple_features(phi[tuple_row], atom[tuple_row], a_u, t)
            alpha = self.tuple_codes(x)
            out = zeros
            out.index_add_(0, tuple_row, alpha * w.unsqueeze(1))
        with torch.no_grad():
            nonzero = out != 0
            self.last_stats = {
                "n_roots": n_rows,
                "n_molecules": n_graphs,
                "n_tuples": total,
                "alpha_nonzero_fraction": 1.0,
                "root_code_nonzero_fraction": float(nonzero.float().mean()),
                "root_code_dead_dims": int((out.std(dim=0) < 1e-8).sum()),
                "root_code_rms": float(out.pow(2).mean().sqrt()),
                "A_norm": float(self.A_raw.norm()),
                "W_loc_norm": float(self.W_loc.norm()),
                "switch_independent": bool(self.switch_independent),
                "mean_replace_active": bool(self.mean_replace is not None),
            }
        if self.mean_replace is not None:
            has_neighbours = counts > 0
            replacement = self.mean_replace.to(device=out.device, dtype=out.dtype)
            out = torch.where(has_neighbours.unsqueeze(1), replacement.unsqueeze(0), torch.zeros_like(out))
        if self.ablate:
            return torch.zeros_like(out)
        return out

    def forward(self, data: Any) -> torch.Tensor:
        codes = self.root_codes(data)
        scaled = codes * self.kappa
        with torch.no_grad():
            contribution = F.linear(scaled, self.W_loc)
            self.last_stats = dict(self.last_stats)
            self.last_stats["e_rms_scaled"] = float(scaled.pow(2).mean().sqrt())
            self.last_stats["contribution_rms"] = float(contribution.pow(2).mean().sqrt())
            self.last_stats["W_loc_norm"] = float(self.W_loc.norm())
        return scaled


class LocalTupleFullM(prev.LocalTupleFull):
    """M_g skeleton + the M local encoder; posterior MLP bridge unchanged."""

    def __init__(self, payload: prev.TuplePayload, a_init: torch.Tensor) -> None:
        # identical construction order to ``prev.LocalTupleFull``: skeleton
        # first, then the matched posterior MLP bridge, then the local encoder.
        zbn = prev.zbn
        zbn.DeployFull.__init__(self)
        bridge = self.local_dictionary_bridge
        dictionary = bridge.D_L.detach().clone()
        values = bridge.V_L.detach().clone()
        self.local_dictionary_bridge = zw.MLPBridge(dictionary, values)
        self.local_tuple = LocalTupleEncoderM(payload, a_init)
        self.weight_key = "joint"
        if int(self.fusion[0].in_features) != 110 or int(self.fusion[0].out_features) != W_LOC_OUT:
            raise RuntimeError(
                f"unexpected fusion layout {int(self.fusion[0].in_features)} -> {int(self.fusion[0].out_features)}"
            )


def parameter_audit_mj(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "base_body_parameters": prev.base_body_parameter_count(model),
        "bridge_parameters": prev.bridge_parameter_count(model),
        "local_tuple_parameters": prev.local_parameter_count(model),
        "A_parameters": int(model.local_tuple.A_raw.numel()),
        "W_loc_parameters": int(model.local_tuple.W_loc.numel()),
    }


def build_arm_mj(payload: prev.TuplePayload | None = None, *, kappa_M: float | None = None) -> nn.Module:
    """Fresh M arm: canonical untrained J frame with ``A_raw = D_init.T``.

    ``A_raw_init`` is copied element-wise from the canonical untrained frame
    (``prev.init_d_loc().T``); ``W_loc`` is exactly zero.  No trained state is
    ever used.  The replacement consumes no global RNG draws, so the
    post-construction training RNG state is the historical one.
    """
    if payload is None:
        payload = prev.load_tuple_payload()
    model = prev.build_arm("J", payload)  # exact historical constructor
    d_init = model.local_tuple.D_loc_raw.detach().clone()
    encoder = LocalTupleEncoderM(payload, d_init.t().contiguous())
    if kappa_M is not None:
        encoder.kappa = float(kappa_M)
    model.local_tuple = encoder
    audit = parameter_audit_mj(model)
    expected = {
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "base_body_parameters": EXPECTED_BODY_PARAMETERS,
        "bridge_parameters": BRIDGE_PARAMETERS,
        "local_tuple_parameters": A_PARAMETERS + W_LOC_PARAMETERS,
        "A_parameters": A_PARAMETERS,
        "W_loc_parameters": W_LOC_PARAMETERS,
    }
    if audit != expected:
        raise RuntimeError(f"M parameter audit failed: {audit} != {expected}")
    return model


# ---------------------------------------------------------------------------
# 2. kappa_M: one-shot label-free fit-only scale matching
# ---------------------------------------------------------------------------


def compose_root_codes_m(
    a_raw: torch.Tensor,
    phi: torch.Tensor,
    atom: torch.Tensor,
    pair_ptr: torch.Tensor,
    pair_t: torch.Tensor,
    pair_a: torch.Tensor,
    pair_w: torch.Tensor,
    root_ids: torch.Tensor,
    phi_mean: torch.Tensor,
    phi_std: torch.Tensor,
    phi_scale: float,
) -> torch.Tensor:
    """``e_M(v) = sum_p w_p SiLU(x_p @ A_bar.T)`` — same operator as the model."""
    start = pair_ptr[root_ids]
    end = pair_ptr[root_ids + 1]
    counts = end - start
    total = int(counts.sum().item())
    if total == 0:
        return torch.zeros(root_ids.numel(), K_TUPLE, device=phi.device, dtype=phi.dtype)
    offsets = counts.cumsum(0) - counts
    flat = torch.repeat_interleave(start, counts) + (
        torch.arange(total, device=phi.device) - torch.repeat_interleave(offsets, counts)
    )
    tuple_row = torch.repeat_interleave(torch.arange(root_ids.numel(), device=phi.device), counts)
    t = pair_t[flat]
    a_u = pair_a[flat]
    w = pair_w[flat]
    x = prev.tuple_features(
        phi[root_ids][tuple_row], atom[root_ids][tuple_row], a_u, t, phi_mean, phi_std, phi_scale
    )
    a_bar = F.normalize(a_raw, dim=1, eps=1e-12)
    alpha = F.silu(x @ a_bar.t())
    out = torch.zeros(root_ids.numel(), K_TUPLE, device=phi.device, dtype=alpha.dtype)
    out.index_add_(0, tuple_row, alpha * w.unsqueeze(1))
    return out


def naive_root_codes_m(
    a_raw: torch.Tensor,
    payload: prev.TuplePayload,
    phi: torch.Tensor,
    atom: torch.Tensor,
    root_ids: Sequence[int],
) -> np.ndarray:
    """Direct per-pair Python reference (no gather / repeat_interleave)."""
    a_bar = F.normalize(a_raw, dim=1, eps=1e-12)
    out = np.zeros((len(root_ids), K_TUPLE), np.float64)
    for row, root in enumerate(int(r) for r in root_ids):
        start = int(payload.pair_ptr[root].item())
        end = int(payload.pair_ptr[root + 1].item())
        for p in range(start, end):
            t = torch.tensor([int(payload.pair_t[p].item())], dtype=torch.long)
            a_u = torch.tensor([int(payload.pair_a[p].item())], dtype=torch.long)
            x = prev.tuple_features(
                phi[root].view(1, -1),
                atom[root].view(1),
                a_u,
                t,
                payload.phi_mean,
                payload.phi_std,
                payload.phi_scale,
            )
            f = F.silu(x @ a_bar.t())[0]
            out[row] += float(payload.pair_wJ[p].item()) * f.double().numpy()
    return out


def compute_kappa_mj(payload: prev.TuplePayload | None = None, *, log: Any = print) -> dict[str, Any]:
    """``r_D = RMS(kappa_D * e_D_init)``, ``r_M = RMS(e_M_init)``, kappa_M = r_D/r_M."""
    if payload is None:
        payload = prev.load_tuple_payload()
    fold = prev.build_fold()
    fit_idx = np.asarray(fold["fit_idx"], np.int64)
    phi_np, atom_np, node_sizes = prev._env_phi_atom()
    root_base = payload.root_base.numpy()
    sample = payload.kappa_sample.numpy()
    fit_mol = np.zeros(10000, dtype=bool)
    fit_mol[fit_idx] = True
    root_fit = np.zeros(int(node_sizes.sum()), dtype=bool)
    for index in np.nonzero(fit_mol)[0]:
        root_fit[int(root_base[index]) : int(root_base[index + 1])] = True
    n_sample_off_fit = int((~root_fit[sample]).sum())
    if n_sample_off_fit != 0:
        raise RuntimeError(f"kappa_sample has {n_sample_off_fit} non-fit roots")
    if sample.size > KAPPA_SAMPLE_MAX:
        raise RuntimeError("kappa_sample larger than the frozen cap")

    phi_t = torch.as_tensor(phi_np, dtype=torch.float32)
    atom_t = torch.as_tensor(atom_np, dtype=torch.long)
    ptr_t = payload.pair_ptr
    t_t = payload.pair_t
    a_t = payload.pair_a
    j_t = payload.pair_wJ
    root_ids = torch.as_tensor(sample, dtype=torch.long)
    d_init = prev.init_d_loc()
    a_init = d_init.t().contiguous()
    with torch.no_grad():
        e_d = prev.compose_root_codes(
            d_init, phi_t, atom_t, ptr_t, t_t, a_t, j_t, root_ids,
            payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
        e_m = compose_root_codes_m(
            a_init, phi_t, atom_t, ptr_t, t_t, a_t, j_t, root_ids,
            payload.phi_mean, payload.phi_std, payload.phi_scale,
        )
    kappa_D = float(payload.kappa)
    e_d_rms = float(e_d.pow(2).mean().sqrt())
    e_m_rms = float(e_m.pow(2).mean().sqrt())
    r_D = kappa_D * e_d_rms
    r_M = e_m_rms
    kappa_M = r_D / r_M if r_M > 0.0 else float("nan")
    if not math.isfinite(kappa_M) or kappa_M <= 0.0 or kappa_M > 1e9:
        raise RuntimeError(f"MECHANISM_INIT_BLOCKED: kappa_M={kappa_M}")
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": "kappa_M = RMS(kappa_D * e_D_init) / RMS(e_M_init) over the frozen fit-only kappa_sample",
        "kappa_D": kappa_D,
        "r_D": r_D,
        "r_M": r_M,
        "e_D_init_rms": e_d_rms,
        "e_M_init_rms": e_m_rms,
        "kappa_M": kappa_M,
        "n_roots": int(sample.size),
        "sample_sha256": _array_sha256(sample, np.int64),
        "sample_head": sample[:8].tolist(),
        "n_nonfit_roots": n_sample_off_fit,
        "tuple_npz_sha256": file_sha256(TUPLE_NPZ),
        "labels_used": False,
        "dev_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(RESULTS_DIR / "kappa_M.json", report)
    log(f"[kappa] kappa_D={kappa_D:.6f} r_D={r_D:.6f} r_M={r_M:.6f} kappa_M={kappa_M:.6f}")
    return report


def load_kappa_m() -> dict[str, Any]:
    path = RESULTS_DIR / "kappa_M.json"
    if not path.exists():
        raise FileNotFoundError("kappa_M.json missing; run --phase-a first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 3. training the single formal M arm (previous recipe, unchanged)
# ---------------------------------------------------------------------------


def train_arm_mj(
    *,
    device: torch.device,
    out_dir: Path,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    kappa_M: float | None = None,
    log: Any = print,
) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = prev.load_tuple_payload()
    if kappa_M is None:
        kappa_M = float(load_kappa_m()["kappa_M"])
    fold, prep_meta, fit_data, dev_data, tgt = prev.load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    schedule, schedule_hash = prev.build_schedule(len(fit_data), int(epochs), SEED + TRAIN_SHUFFLE_OFFSET)
    if int(epochs) == EPOCHS and schedule_hash != FROZEN_SCHEDULE_SHA:
        raise RuntimeError(f"schedule hash {schedule_hash} != frozen {FROZEN_SCHEDULE_SHA}")

    seed_everything(SEED)
    model = build_arm_mj(payload, kappa_M=kappa_M).to(device)
    audit = parameter_audit_mj(model)
    init_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    init_a = model.local_tuple.A_raw.detach().cpu().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    soup_epochs = set(int(e) for e in SOUP_EPOCHS)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probe_log: list[dict[str, Any]] = []
    steps_done = 0
    stream = hashlib.sha256()
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0, 0, 0.0, 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = schedule[epoch - 1][start : start + BATCH_SIZE]
            index_list = [int(i) for i in indices.tolist()]
            stream.update(np.asarray(index_list, np.int64).tobytes())
            batch = zftd.make_batch(fit_data, index_list, target_fit, device)
            prediction = model(batch, mask=prev.cm.C6_MASK)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                entry = {"epoch": int(epoch), "step_in_epoch": 1, "clip_total_norm": total_norm}
                entry["W_loc_grad_norm"] = (
                    float(model.local_tuple.W_loc.grad.norm()) if model.local_tuple.W_loc.grad is not None else None
                )
                entry["A_grad_norm"] = (
                    float(model.local_tuple.A_raw.grad.norm()) if model.local_tuple.A_raw.grad is not None else None
                )
                entry["W_loc_norm"] = float(model.local_tuple.W_loc.detach().norm())
                entry["A_norm"] = float(model.local_tuple.A_raw.detach().norm())
                entry["health"] = {k: v for k, v in model.local_tuple.last_stats.items()}
                probe_log.append(entry)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        curve.append(
            {
                "epoch": int(epoch),
                "train_task_mae": float(task_sum / max(n_mol, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if epoch in soup_epochs:
            soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[M:local] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"gnorm={curve[-1]['grad_norm']:.3g} clip={curve[-1]['clip_fraction']:.3f} "
                f"{curve[-1]['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    if not soup and max_steps is not None:
        soup[int(epoch)] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    members = sorted(soup)
    soup_state = {key: torch.stack([soup[e][key].float() for e in members]).mean(0) for key in soup[members[0]]}
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    a_drift = float((last_state["local_tuple.A_raw"] - init_a).norm() / init_a.norm())
    wloc_norm = float(last_state["local_tuple.W_loc"].norm())

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": "M",
        "weight_key": "joint",
        "supervision_target": "g = y - c (frozen fit-only decomposition)",
        "seed": SEED,
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "steps_expected": int(epochs) * math.ceil(len(fit_data) / BATCH_SIZE),
        "stopped_reason": stopped_reason,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
        "soup_epochs": members if max_steps is None else [],
        "n_fit": int(len(fit_data)), "n_dev": int(len(dev_data)),
        "kappa_M": float(kappa_M),
        "fold": {"seed": FOLD_SEED, "fit_idx_sha256": fold["fit_idx_sha256"], "dev_idx_sha256": fold["dev_idx_sha256"]},
        "target": {
            "source": str(zcdm.TARGETS_NPZ), "sha256": file_sha256(zcdm.TARGETS_NPZ),
            "definition": "g = y - c; constants fitted on 8000 fit rows only",
            "target_sha256": hashlib.sha256(np.ascontiguousarray(g, np.float64).tobytes()).hexdigest(),
        },
        "tuple_npz": {"path": str(TUPLE_NPZ), "sha256": file_sha256(TUPLE_NPZ)},
        "parameter_audit": audit,
        "init_state_hash": state_hash(init_state),
        "raw_soup_state_hash": state_hash(soup_state),
        "last_state_hash": state_hash(last_state),
        "init_A_sha256": _array_sha256(init_a.numpy()),
        "soup_A_sha256": _array_sha256(soup_state["local_tuple.A_raw"].numpy()),
        "soup_W_loc_sha256": _array_sha256(soup_state["local_tuple.W_loc"].numpy()),
        "soup_W_loc_norm": float(soup_state["local_tuple.W_loc"].norm()),
        "last_W_loc_norm": wloc_norm,
        "A_relative_drift_last": a_drift,
        "schedule_sha256": schedule_hash,
        "data_stream_sha256": stream.hexdigest(),
        "curve": curve,
        "probe_log": probe_log,
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if max_steps is not None:
        result["smoke_last_state_hash"] = state_hash(last_state)
        return result

    torch.save(init_state, out_dir / "M_init_state.pt")
    torch.save(last_state, out_dir / "M_last_state.pt")
    torch.save(soup_state, out_dir / "M_raw_soup_state.pt")

    raw_predictions: dict[str, np.ndarray] = {}
    for state_name, state in (("init", init_state), ("last", last_state), ("raw_soup", soup_state)):
        replay = build_arm_mj(payload, kappa_M=kappa_M)
        replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in state.items()}, strict=True)
        replay = replay.to(device)
        raw_predictions[f"{state_name}_fit"] = prev.evaluate_state(replay, fit_data, g_fit, device)
        raw_predictions[f"{state_name}_dev"] = prev.evaluate_state(replay, dev_data, g_dev, device)
    np.savez_compressed(
        out_dir / "M_raw_predictions.npz",
        **{key: value.astype(np.float32) for key, value in raw_predictions.items()},
    )

    b_soup = float(np.median(g_fit - raw_predictions["raw_soup_fit"]))
    b_init = float(np.median(g_fit - raw_predictions["init_fit"]))
    b_last = float(np.median(g_fit - raw_predictions["last_fit"]))
    result["calibration_b"] = {"init": b_init, "last": b_last, "raw_soup": b_soup}

    replay = build_arm_mj(payload, kappa_M=kappa_M)
    replay.load_state_dict({key: value.to(torch.device("cpu")) for key, value in soup_state.items()}, strict=True)
    batch_data = fit_data[:128]
    gpu_pred = prev.evaluate_state(replay.to(device), batch_data, g_fit[:128], device)
    cpu_pred = prev.evaluate_state(
        replay.to(torch.device("cpu")), batch_data, g_fit[:128], torch.device("cpu")
    )
    max_diff = float(np.max(np.abs(gpu_pred - cpu_pred))) if device.type != "cpu" else 0.0
    result["replay_max_abs_diff"] = max_diff
    if device.type != "cpu" and max_diff > REPLAY_TOL:
        raise RuntimeError(f"CPU/GPU replay maxdiff {max_diff} > {REPLAY_TOL}")
    write_json(out_dir / "M_meta.json", result)
    write_json(out_dir / "M_curve.json", curve)
    log(f"[M:local] done steps={steps_done} wall={result['wall_clock_s']:.1f}s b={b_soup:.6f} replay={max_diff:.2e}")
    return result


# ---------------------------------------------------------------------------
# 4. phase A: anchors, init/RNG identity, kappa
# ---------------------------------------------------------------------------


def phase_a(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    payload = prev.load_tuple_payload()
    fold, prep_meta, fit_data, dev_data, tgt = prev.load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    k_dev = np.asarray(tgt["k"], np.int64)[fold["dev_idx"]]
    g0_dev = k_dev == 0

    if file_sha256(TUPLE_NPZ) != FROZEN_TUPLE_SHA:
        raise RuntimeError("tuple index hash mismatch")
    if not (
        fold["fit_idx_sha256"] == FROZEN_FIT_SHA and fold["dev_idx_sha256"] == FROZEN_DEV_SHA
    ):
        raise RuntimeError("fold hash mismatch")
    if file_sha256(zcdm.TARGETS_NPZ) != FROZEN_TARGET_SHA:
        raise RuntimeError("target hash mismatch")

    # -- anchors B and D_J, recomputed from stored raw fit predictions ------
    b_pred = _load_ref_predictions()
    b_bias = float(np.median(g_fit - b_pred["raw_soup_fit"]))
    b_anchor = {
        "arm": "B = M_g raw soup (historical, read-only)",
        "bias_recomputed": b_bias,
        "fit_overall_cal_mae": float(np.mean(np.abs(g_fit - (b_pred["raw_soup_fit"] + b_bias)))),
        "dev_overall_cal_mae": float(np.mean(np.abs(g_dev - (b_pred["raw_soup_dev"] + b_bias)))),
        "dev_G0_cal_mae": float(np.mean(np.abs((g_dev - (b_pred["raw_soup_dev"] + b_bias))[g0_dev]))),
    }
    b_anchor["dev_overall_delta"] = abs(b_anchor["dev_overall_cal_mae"] - ANCHOR_B_DEV_OVERALL_CAL)
    b_anchor["dev_G0_delta"] = abs(b_anchor["dev_G0_cal_mae"] - ANCHOR_B_DEV_G0_CAL)
    b_anchor["anchor_ok"] = bool(max(b_anchor["dev_overall_delta"], b_anchor["dev_G0_delta"]) <= ANCHOR_TOL)

    dj_pred = _load_prev_predictions("J")
    dj_bias = float(np.median(g_fit - dj_pred["raw_soup_fit"]))
    dj_anchor = {
        "arm": "D_J = previous-round J raw soup (read-only)",
        "bias_recomputed": dj_bias,
        "bias_from_meta": float(
            json.loads((PREV_DIR / "J_meta.json").read_text())["calibration_b"]["raw_soup"]
        ),
        "fit_overall_cal_mae": float(np.mean(np.abs(g_fit - (dj_pred["raw_soup_fit"] + dj_bias)))),
        "dev_overall_cal_mae": float(np.mean(np.abs(g_dev - (dj_pred["raw_soup_dev"] + dj_bias)))),
        "dev_G0_cal_mae": float(np.mean(np.abs((g_dev - (dj_pred["raw_soup_dev"] + dj_bias))[g0_dev]))),
    }
    dj_anchor["fit_overall_delta"] = abs(dj_anchor["fit_overall_cal_mae"] - ANCHOR_DJ_FIT_OVERALL_CAL)
    dj_anchor["dev_overall_delta"] = abs(dj_anchor["dev_overall_cal_mae"] - ANCHOR_DJ_DEV_OVERALL_CAL)
    dj_anchor["dev_G0_delta"] = abs(dj_anchor["dev_G0_cal_mae"] - ANCHOR_DJ_DEV_G0_CAL)
    dj_anchor["anchor_ok"] = bool(
        max(dj_anchor["fit_overall_delta"], dj_anchor["dev_overall_delta"], dj_anchor["dev_G0_delta"]) <= ANCHOR_TOL
    )
    write_json(out_dir / "historical_anchor_checks.json", {"B": b_anchor, "D_J": dj_anchor})

    # -- init identity and RNG/schedule witness -----------------------------
    seed_everything(SEED)
    model_m = build_arm_mj(payload, kappa_M=1.0)
    rng_after_m = torch.get_rng_state().clone()
    seed_everything(SEED)
    model_j = prev.build_arm("J", payload)
    rng_after_j = torch.get_rng_state().clone()
    model_ref = prev.build_reference_m()
    model_m.eval(); model_j.eval(); model_ref.eval()

    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    batch = zftd.make_batch(fit_data[:128], list(range(128)), target_fit, torch.device("cpu"))
    with torch.no_grad():
        out_m = model_m(batch, mask=prev.cm.C6_MASK).view(-1).numpy()
        out_j = model_j(batch, mask=prev.cm.C6_MASK).view(-1).numpy()
        out_ref = model_ref(batch, mask=prev.cm.C6_MASK).view(-1).numpy()

    j_init = torch.load(PREV_DIR / "J_init_state.pt", map_location="cpu", weights_only=False)
    j_sd = {key: value.detach().clone() for key, value in model_j.state_dict().items()}
    m_sd = {key: value.detach().clone() for key, value in model_m.state_dict().items()}
    shared_keys = sorted(set(j_init) - {"local_tuple.D_loc_raw", "local_tuple.W_loc"})
    shared_mismatch = {k: float((j_init[k] - j_sd[k]).abs().max()) for k in shared_keys}
    shared_mismatch_m = {k: float((j_init[k] - m_sd[k]).abs().max()) for k in shared_keys}
    a_init = m_sd["local_tuple.A_raw"]
    d_init = j_sd["local_tuple.D_loc_raw"]
    identity = {
        "j_init_state_hash_recomputed": state_hash({k: v for k, v in j_sd.items()}),
        "j_init_state_hash_stored": FROZEN_J_INIT_SHA,
        "j_init_hash_ok": bool(state_hash({k: v for k, v in j_sd.items()}) == FROZEN_J_INIT_SHA),
        "j_init_vs_stored_max_abs_max": max(shared_mismatch.values()),
        "m_shared_vs_stored_j_init_max_abs_max": max(shared_mismatch_m.values()),
        "m_shared_keys": len(shared_keys),
        "A_equals_D_init_T_max_abs": float((a_init - d_init.t()).abs().max()),
        "W_loc_all_zero": bool(float(m_sd["local_tuple.W_loc"].abs().sum()) == 0.0),
        "parameter_audit": parameter_audit_mj(model_m),
        "max_abs_diff_M_vs_J_init_eval": float(np.max(np.abs(out_m - out_j))),
        "max_abs_diff_M_vs_fresh_ref_eval": float(np.max(np.abs(out_m - out_ref))),
        "init_function_equal": bool(
            np.max(np.abs(out_m - out_j)) <= 1e-5 and np.max(np.abs(out_m - out_ref)) <= 1e-5
        ),
        "rng_state_equal_after_construction": bool(torch.equal(rng_after_m, rng_after_j)),
        "kappa_M_placeholder": 1.0,
        "note": "W_loc == 0 at init, so kappa_M is irrelevant for the initial function",
    }
    write_json(out_dir / "init_identity.json", identity)

    batch_grp = zftd.make_batch(fit_data, list(range(8)), target_fit, torch.device("cpu"))
    with torch.no_grad():
        grouped = model_m(batch_grp, mask=prev.cm.C6_MASK).view(-1).numpy()
        singles = []
        for i in range(8):
            b1 = zftd.make_batch(fit_data, [i], target_fit, torch.device("cpu"))
            singles.append(float(model_m(b1, mask=prev.cm.C6_MASK).view(-1)[0]))
        single = np.asarray(singles)
    offsets = {"max_abs_diff": float(np.max(np.abs(grouped - single))), "ok": bool(np.max(np.abs(grouped - single)) < 1e-5)}

    # -- production operator check: model root_codes == compose_root_codes_m --
    phi_np, atom_np, node_sizes = prev._env_phi_atom()
    phi_t = torch.as_tensor(phi_np, dtype=torch.float32)
    atom_t = torch.as_tensor(atom_np, dtype=torch.long)
    root_ids = []
    for i in range(8):
        mol = int(fit_data[i].local_mol_id.item())
        n = int(fit_data[i].dict_phi.shape[0])
        root_ids.extend(range(int(payload.root_base[mol]), int(payload.root_base[mol]) + n))
    root_ids_t = torch.as_tensor(root_ids, dtype=torch.long)
    with torch.no_grad():
        codes_batch = model_m.local_tuple.root_codes(batch_grp)
        codes_direct = compose_root_codes_m(
            model_m.local_tuple.A_raw, phi_t, atom_t, payload.pair_ptr, payload.pair_t,
            payload.pair_a, payload.pair_wJ, root_ids_t, payload.phi_mean, payload.phi_std,
            payload.phi_scale,
        )
    operator = {
        "model_vs_compose_max_abs": float((codes_batch - codes_direct).abs().max()),
        "n_roots_checked": len(root_ids),
        "endpoint_offsets": offsets,
    }
    naive_ids = root_ids[:16]
    with torch.no_grad():
        naive = naive_root_codes_m(
            model_m.local_tuple.A_raw, payload, phi_t, atom_t,
            [int(r) for r in payload.kappa_sample[:16].tolist()],
        )
        vec = compose_root_codes_m(
            model_m.local_tuple.A_raw, phi_t, atom_t, payload.pair_ptr, payload.pair_t,
            payload.pair_a, payload.pair_wJ,
            torch.as_tensor(payload.kappa_sample[:16], dtype=torch.long),
            payload.phi_mean, payload.phi_std, payload.phi_scale,
        ).numpy()
    operator["naive_loop_vs_compose_max_abs"] = float(np.max(np.abs(naive - vec)))
    operator["naive_sample_roots"] = payload.kappa_sample[:16].tolist()
    operator["ok"] = bool(
        operator["model_vs_compose_max_abs"] < 1e-5
        and operator["naive_loop_vs_compose_max_abs"] < 1e-5
        and offsets["ok"]
    )
    write_json(out_dir / "operator_path_checks.json", operator)

    # -- kappa_M ------------------------------------------------------------
    kappa_report = compute_kappa_mj(payload, log=log)

    # -- environment + input manifest ---------------------------------------
    env_checks = prev.verify_env_cache(payload)
    env_checks["kappa_D"] = float(payload.kappa)
    env_checks["pair_count"] = int(payload.pair_ptr[-1])
    write_json(out_dir / "tuple_environment_checks.json", env_checks)

    sources = {
        "encoded_train": zjd.TRACK_ROOT / "results/zinc_static_dictionary_pair/cache/encoded_train.pt",
        "env_train": prev.ENV_TRAIN_CACHE,
        "fold_objects": zw.PREP_BLOB,
        "raw_train_processed": prev.ZINC_TRAIN_PROCESSED,
        "targets_npz": zcdm.TARGETS_NPZ,
        "tuple_npz": TUPLE_NPZ,
        "prev_J_raw_predictions": PREV_DIR / "J_raw_predictions.npz",
        "prev_J_meta": PREV_DIR / "J_meta.json",
        "prev_J_init_state": PREV_DIR / "J_init_state.pt",
        "prev_J_raw_soup_state": PREV_DIR / "J_raw_soup_state.pt",
        "prev_mechanism_health": PREV_DIR / "mechanism_health.json",
        "ref_M_raw_predictions": REF_DIR / "M_raw_predictions.npz",
        "ref_M_meta": REF_DIR / "M_meta.json",
    }
    manifest = {
        "files": {
            name: {
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() else None,
                "bytes": int(path.stat().st_size) if path.exists() else None,
            }
            for name, path in sources.items()
        },
        "missing": [name for name, path in sources.items() if not path.exists()],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "raw_graph_file": "data/ZINC/subset/processed/train.pt (official train only)",
    }
    write_json(out_dir / "input_manifest.json", manifest)

    out = {
        "historical_anchor": {"B": b_anchor, "D_J": dj_anchor},
        "init_identity": identity,
        "operator_path_checks": operator,
        "kappa_M": kappa_report,
        "env_checks": env_checks,
        "manifest_missing": manifest["missing"],
        "seconds": float(time.perf_counter() - t0),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "phase_a.json", out)
    log(f"[phase-a] done in {out['seconds']:.1f}s")
    return out


# ---------------------------------------------------------------------------
# 5. fit-only smoke (checks + <=10-step tiny train, discarded)
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = prev.load_tuple_payload()
    kappa_M = float(load_kappa_m()["kappa_M"])
    fold, prep_meta, fit_data, dev_data, tgt = prev.load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit = g[fold["fit_idx"]]
    target_fit = torch.as_tensor(g_fit, dtype=torch.float32)
    checks: dict[str, Any] = {"kappa_M": kappa_M}

    model = build_arm_mj(payload, kappa_M=kappa_M).to(device)
    model.eval()
    batch = zftd.make_batch(fit_data, list(range(32)), target_fit, device)
    with torch.no_grad():
        pred = model(batch, mask=prev.cm.C6_MASK)
    checks["forward_finite"] = bool(torch.isfinite(pred).all().item())

    # target never read / label permutation
    with torch.no_grad():
        pred_perm = model(
            zftd.make_batch(fit_data, list(range(32)), target_fit.flip(0).clone(), device),
            mask=prev.cm.C6_MASK,
        )
    checks["label_permutation_maxdiff"] = float((pred - pred_perm).abs().max())
    checks["target_not_read_ok"] = bool(checks["label_permutation_maxdiff"] <= TARGET_NOT_READ_TOL)

    # dropout-stream identity vs the historical J frame (same construction RNG)
    seed_everything(SEED)
    model_j = prev.build_arm("J", payload).to(device)
    seed_everything(SEED)
    model_m = build_arm_mj(payload, kappa_M=kappa_M).to(device)
    model_j.train(); model_m.train()
    seed_everything(SEED)
    model_j(batch, mask=prev.cm.C6_MASK)
    rng_j = torch.get_rng_state().clone()
    seed_everything(SEED)
    model_m(batch, mask=prev.cm.C6_MASK)
    rng_m = torch.get_rng_state().clone()
    checks["dropout_stream_identical"] = bool(torch.equal(rng_j, rng_m))
    model_j.eval(); model_m.eval()

    # task-gradient flow: W_loc first step nonzero, A only after W_loc path exists
    seed_everything(SEED)
    model2 = build_arm_mj(payload, kappa_M=kappa_M).to(device)
    model2.train()
    b = zftd.make_batch(fit_data, list(range(32)), target_fit, device)
    loss = F.l1_loss(model2(b, mask=prev.cm.C6_MASK).view(-1), b.y.view(-1))
    model2.zero_grad()
    loss.backward()
    checks["w_loc_grad_first_step"] = (
        float(model2.local_tuple.W_loc.grad.norm()) if model2.local_tuple.W_loc.grad is not None else 0.0
    )
    checks["a_grad_first_step"] = (
        float(model2.local_tuple.A_raw.grad.norm()) if model2.local_tuple.A_raw.grad is not None else 0.0
    )
    checks["w_loc_grad_nonzero"] = bool(checks["w_loc_grad_first_step"] > 0.0)
    generator = torch.Generator(device=device).manual_seed(123)
    with torch.no_grad():
        model2.local_tuple.W_loc.normal_(0.0, 1e-3, generator=generator)
    loss = F.l1_loss(model2(b, mask=prev.cm.C6_MASK).view(-1), b.y.view(-1))
    model2.zero_grad()
    loss.backward()
    checks["a_grad_after_w_loc_path"] = (
        float(model2.local_tuple.A_raw.grad.norm()) if model2.local_tuple.A_raw.grad is not None else 0.0
    )
    checks["a_path_established"] = bool(checks["a_grad_after_w_loc_path"] > 0.0)

    # SiLU is really applied elementwise per tuple (not a linear code)
    with torch.no_grad():
        x = model2.local_tuple.tuple_features(
            torch.randn(7, PHI_DIM, generator=torch.Generator().manual_seed(7)),
            torch.randint(0, ATOM_CATEGORIES, (7,), generator=torch.Generator().manual_seed(8)),
            torch.randint(0, ATOM_CATEGORIES, (7,), generator=torch.Generator().manual_seed(9)),
            torch.randint(0, BOND_CATEGORIES, (7,), generator=torch.Generator().manual_seed(10)),
        )
        code = model2.local_tuple.tuple_codes(x)
        a_bar = F.normalize(model2.local_tuple.A_raw, dim=1, eps=1e-12)
        ref = F.silu(x @ a_bar.t())
        lin = x @ a_bar.t()
    checks["tuple_code_matches_silu_max_abs"] = float((code - ref).abs().max())
    checks["tuple_code_is_nonlinear"] = bool(float((code - lin).abs().max()) > 1e-3)

    # tiny fit-only train (discarded)
    smoked = train_arm_mj(
        device=device, out_dir=out_dir / "smoke", epochs=1, max_steps=3, kappa_M=kappa_M, log=log
    )
    checks["tiny_train"] = {
        "steps": smoked["steps_done"],
        "loss_curve": [e["train_task_mae"] for e in smoked["curve"]],
        "schedule_sha256": smoked["schedule_sha256"],
        "stopped_reason": smoked["stopped_reason"],
    }
    checks["all_ok"] = bool(
        checks["forward_finite"]
        and checks["target_not_read_ok"]
        and checks["dropout_stream_identical"]
        and checks["w_loc_grad_nonzero"]
        and checks["a_path_established"]
        and checks["tuple_code_matches_silu_max_abs"] < 1e-6
        and checks["tuple_code_is_nonlinear"]
    )
    write_json(out_dir / "smoke_checks.json", checks)
    if not checks["all_ok"]:
        raise RuntimeError(f"smoke checks failed: {checks}")
    log(f"[smoke] all_ok={checks['all_ok']}")
    return checks


# ---------------------------------------------------------------------------
# 6. analysis: calibration, paired bootstrap, gates
# ---------------------------------------------------------------------------


def _mae(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(target - pred)))


def _paired_gain_bootstrap(
    errs: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray | None,
    pairs: Sequence[tuple[str, str]],
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Paired per-row bootstrap with one shared resample index set per draw.

    ``gain(ref, arm) = mean|err_ref| - mean|err_arm|`` (positive = arm better).
    """
    abs_errs = {key: np.abs(np.asarray(value, np.float64)) for key, value in errs.items()}
    if mask is not None:
        abs_errs = {key: value[mask] for key, value in abs_errs.items()}
    size = next(iter(abs_errs.values())).size
    if any(value.size != size for value in abs_errs.values()):
        raise RuntimeError("paired arrays must share length")
    rng = np.random.default_rng(int(seed))
    samples = {pair: np.empty(int(n_boot), np.float64) for pair in pairs}
    for b in range(int(n_boot)):
        idx = rng.choice(size, size=size, replace=True)
        means = {key: float(value[idx].mean()) for key, value in abs_errs.items()}
        for ref, arm in pairs:
            samples[(ref, arm)][b] = means[ref] - means[arm]
    out: dict[str, Any] = {
        "n": int(size), "seed": int(seed), "n_boot": int(n_boot), "shared_indices": True,
        "gain_definition": "mean|err_ref| - mean|err_arm|; positive => arm better",
    }
    for pair in pairs:
        ref, arm = pair
        point = float(abs_errs[ref].mean() - abs_errs[arm].mean())
        out[f"{ref}->{arm}"] = {
            "point": point,
            "ci95": [float(np.percentile(samples[pair], 2.5)), float(np.percentile(samples[pair], 97.5))],
        }
    return out


def _gain_self_tests(err_dj: np.ndarray, err_m: np.ndarray) -> dict[str, Any]:
    """Estimator witnesses in the previous round's diff form (centred on
    ``|err_D_J| - |err_M_J|``): identical predictions -> 0, swapped -> mirror,
    constant shift -> exact + bounded move."""
    return prev._bootstrap_self_tests(err_dj, err_m)


def analyze(*, out_dir: Path, device: torch.device | None = None, log: Any = print) -> dict[str, Any]:
    fold, prep_meta, fit_data, dev_data, tgt = prev.load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    k = np.asarray(tgt["k"], np.int64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    k_dev = k[fold["dev_idx"]]
    g0 = k_dev == 0
    # disjoint k groups only (the overlapping k<=-2 view is descriptive and
    # would break the contribution-sum identity)
    groups = {"k=0": k_dev == 0, "k=-1": k_dev == -1, "k=-2": k_dev == -2, "k<=-3": k_dev <= -3}

    m_meta = json.loads((out_dir / "M_meta.json").read_text())
    dj_meta = json.loads((PREV_DIR / "J_meta.json").read_text())
    b_meta = json.loads((REF_DIR / "M_meta.json").read_text())
    if int(m_meta["steps_done"]) != int(m_meta["steps_expected"]) or m_meta["stopped_reason"] != "completed":
        raise RuntimeError("M arm incomplete; classification INCOMPLETE")
    m_pred = _load_prev_predictions_from(out_dir, "M")
    dj_pred = _load_prev_predictions("J")
    b_pred = _load_ref_predictions()

    def cal(meta: Mapping[str, Any], pred: Mapping[str, np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
        bias = float(np.median(g_fit - pred["raw_soup_fit"]))
        return bias, pred["raw_soup_fit"] + bias, pred["raw_soup_dev"] + bias

    arms: dict[str, dict[str, Any]] = {}
    for name, meta, pred in (("B", b_meta, b_pred), ("D_J", dj_meta, dj_pred), ("M_J", m_meta, m_pred)):
        bias, fit_cal, dev_cal = cal(meta, pred)
        raw_fit, raw_dev = pred["raw_soup_fit"], pred["raw_soup_dev"]
        entry = {
            "calibration_b": bias,
            "fit_overall_raw": _mae(g_fit, raw_fit),
            "fit_overall_cal": _mae(g_fit, fit_cal),
            "dev_overall_raw": _mae(g_dev, raw_dev),
            "dev_overall_cal": _mae(g_dev, dev_cal),
            "dev_G0_raw": _mae(g_dev[g0], raw_dev[g0]),
            "dev_G0_cal": _mae(g_dev[g0], dev_cal[g0]),
            "groups": {},
        }
        err = g_dev - dev_cal
        for gname, mask in groups.items():
            n = int(mask.sum())
            entry["groups"][gname] = {
                "n": n,
                "mae": _mae(g_dev[mask], dev_cal[mask]) if n else None,
                "contribution": float(np.abs(err[mask]).sum() / k_dev.size) if n else 0.0,
            }
        entry["group_contribution_sum_check"] = bool(
            abs(sum(v["contribution"] for v in entry["groups"].values()) - entry["dev_overall_cal"]) < 1e-12
            or abs(sum(v["contribution"] for v in entry["groups"].values()) - entry["dev_overall_cal"]) < 1e-9
        )
        arms[name] = entry

    err = {
        "B": g_dev - (b_pred["raw_soup_dev"] + float(np.median(g_fit - b_pred["raw_soup_fit"]))),
        "D_J": g_dev - (dj_pred["raw_soup_dev"] + float(np.median(g_fit - dj_pred["raw_soup_fit"]))),
        "M_J": g_dev - (m_pred["raw_soup_dev"] + float(np.median(g_fit - m_pred["raw_soup_fit"]))),
    }
    err_raw = {
        "B": g_dev - b_pred["raw_soup_dev"],
        "D_J": g_dev - dj_pred["raw_soup_dev"],
        "M_J": g_dev - m_pred["raw_soup_dev"],
    }

    pairs = [("D_J", "M_J"), ("B", "M_J"), ("B", "D_J")]
    boot = {
        "overall_cal": _paired_gain_bootstrap(err, mask=None, pairs=pairs),
        "G0_cal": _paired_gain_bootstrap(err, mask=g0, pairs=pairs),
        "overall_raw": _paired_gain_bootstrap(err_raw, mask=None, pairs=pairs),
        "G0_raw": _paired_gain_bootstrap(err_raw, mask=g0, pairs=pairs),
    }
    witnesses = _gain_self_tests(err["D_J"], err["M_J"])

    # -- frozen classification (M_J vs D_J) ---------------------------------
    g0c = boot["G0_cal"]["D_J->M_J"]
    ovc = boot["overall_cal"]["D_J->M_J"]
    g0r = boot["G0_raw"]["D_J->M_J"]
    ovr = boot["overall_raw"]["D_J->M_J"]
    mechanism_failed = False
    m_init = torch.load(out_dir / "M_init_state.pt", map_location="cpu", weights_only=False)
    m_soup = torch.load(out_dir / "M_raw_soup_state.pt", map_location="cpu", weights_only=False)

    def _state_delta(state: Mapping[str, torch.Tensor], key: str) -> float:
        return float((state[key] - m_init[key]).abs().max())

    mechanism_failed = bool(
        _state_delta(m_soup, "local_tuple.A_raw") == 0.0
        or float(m_soup["local_tuple.W_loc"].abs().max()) == 0.0
        or int(m_meta["probe_log"][-1]["health"].get("root_code_dead_dims", 0)) >= K_TUPLE
    )
    cond_mlp = (
        g0c["point"] >= GATE_DELTA and g0c["ci95"][0] > 0.0 and g0r["point"] > 0.0 and ovc["point"] >= -0.001
    )
    cond_dict = (
        g0c["point"] <= -GATE_DELTA and g0c["ci95"][1] < 0.0 and g0r["point"] < 0.0 and ovc["point"] <= 0.001
    )
    cond_equiv = bool(
        _ci_inside(g0c["ci95"], -EQUIV_BAND, EQUIV_BAND)
        and _ci_inside(ovc["ci95"], -EQUIV_BAND, EQUIV_BAND)
        and abs(g0r["point"]) <= EQUIV_BAND
        and abs(ovr["point"]) <= EQUIV_BAND
    )
    # mechanism probes are produced in --mechanism (after training); the
    # training-time flag above covers the frozen MECHANISM_FAILED conditions.
    if mechanism_failed:
        classification = "MLP_MECHANISM_FAILED"
    elif cond_mlp:
        classification = "MLP_LOCAL_SUPPORT"
    elif cond_dict:
        classification = "DICT_LOCAL_SUPPORT"
    elif cond_equiv:
        classification = "LOCAL_EQUIVALENCE"
    else:
        classification = "INCONCLUSIVE"

    perf_conditions = {
        "G0_cal_gain_ge_0.003": bool(boot["G0_cal"]["B->M_J"]["point"] >= PERF_DELTA),
        "overall_cal_gain_ge_0.003": bool(boot["overall_cal"]["B->M_J"]["point"] >= PERF_DELTA),
        "G0_cal_ci_low_gt_0": bool(boot["G0_cal"]["B->M_J"]["ci95"][0] > 0.0),
        "G0_raw_gain_gt_0": bool(boot["G0_raw"]["B->M_J"]["point"] > 0.0),
        "overall_raw_gain_gt_0": bool(boot["overall_raw"]["B->M_J"]["point"] > 0.0),
    }
    performance_signal = all(perf_conditions.values())

    # -- pre-fixed sensitivity: drop the unique largest combined-error row ---
    combined = (np.abs(err["D_J"]) + np.abs(err["M_J"])) / 2.0
    drop_row = int(np.argmax(combined))
    keep = np.ones(k_dev.size, dtype=bool)
    keep[drop_row] = False
    sensitivity = {
        "rule": "drop the unique dev row with max mean(|err_D_J|, |err_M_J|) over calibrated errors",
        "row_index": drop_row,
        "gid": int(np.asarray(tgt["gid"])[fold["dev_idx"]][drop_row]),
        "in_G0": bool(g0[drop_row]),
        "G0_cal_gain_after_drop": float(
            np.abs(err["D_J"][g0 & keep]).mean() - np.abs(err["M_J"][g0 & keep]).mean()
        ),
        "overall_cal_gain_after_drop": float(
            (np.abs(err["D_J"][keep]).mean()) - (np.abs(err["M_J"][keep]).mean())
        ),
    }

    # -- four-grid decomposition (previous round, read-only) ----------------
    health = json.loads((PREV_DIR / "mechanism_health.json").read_text())
    inter = health["interventions"]
    fourgrid: dict[str, Any] = {"source": str(PREV_DIR / "mechanism_health.json"), "tables": {}}
    for metric in ("overall", "G0"):
        suffix = "" if metric == "overall" else "G0_"
        vals = {
            "JJ": inter["J"]["dev_operator_switched"][f"{suffix}mae_orig"],
            "JI": inter["J"]["dev_operator_switched"][f"{suffix}mae_new"],
            "IJ": inter["I"]["dev_operator_switched"][f"{suffix}mae_new"],
            "II": inter["I"]["dev_operator_switched"][f"{suffix}mae_orig"],
        }
        expected = FOURGRID_EXPECTED[metric]
        deltas = {key: abs(float(vals[key]) - expected[key]) for key in vals}
        G = vals["II"] - vals["JJ"]
        O = ((vals["JI"] - vals["JJ"]) + (vals["II"] - vals["IJ"])) / 2.0
        W = ((vals["IJ"] - vals["JJ"]) + (vals["II"] - vals["JI"])) / 2.0
        fourgrid["tables"][metric] = {
            "values": {key: float(vals[key]) for key in vals},
            "expected_match_max_delta": max(deltas.values()),
            "expected_match_ok": bool(max(deltas.values()) <= 1e-5),
            "G": float(G), "O": float(O), "W": float(W),
            "G_equals_O_plus_W": bool(abs(G - (O + W)) < 1e-12),
            "W_fraction_of_G": float(W / G) if G != 0 else None,
            "note": "descriptive decomposition of the frozen J/I intervention grid; not a unique causal split",
        }

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "arms": arms,
        "bootstrap": boot,
        "bootstrap_witnesses": witnesses,
        "classification": {
            "category": classification,
            "frozen_conditions": {
                "MLP_LOCAL_SUPPORT": cond_mlp,
                "DICT_LOCAL_SUPPORT": cond_dict,
                "LOCAL_EQUIVALENCE": cond_equiv,
                "MLP_MECHANISM_FAILED": mechanism_failed,
            },
            "gain_MJ": {
                "G0_cal": {"point": g0c["point"], "ci95": g0c["ci95"], "positive_means_M_better": True},
                "overall_cal": {"point": ovc["point"], "ci95": ovc["ci95"]},
                "G0_raw": {"point": g0r["point"], "ci95": g0r["ci95"]},
                "overall_raw": {"point": ovr["point"], "ci95": ovr["ci95"]},
            },
        },
        "performance_signal_vs_B": {
            "pass": performance_signal,
            "conditions": perf_conditions,
            "gains": {
                "G0_cal": boot["G0_cal"]["B->M_J"],
                "overall_cal": boot["overall_cal"]["B->M_J"],
                "G0_raw": boot["G0_raw"]["B->M_J"],
                "overall_raw": boot["overall_raw"]["B->M_J"],
            },
        },
        "sensitivity": sensitivity,
        "four_grid": fourgrid,
        "raw_vs_cal_ranking": {
            "G0": {
                "raw": {name: arms[name]["dev_G0_raw"] for name in arms},
                "cal": {name: arms[name]["dev_G0_cal"] for name in arms},
            },
            "overall": {
                "raw": {name: arms[name]["dev_overall_raw"] for name in arms},
                "cal": {name: arms[name]["dev_overall_cal"] for name in arms},
            },
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "analysis.json", report)
    write_json(out_dir / "gains.json", {"gains": {k: v for k, v in boot.items()}, "performance_vs_B": report["performance_signal_vs_B"]})
    write_json(out_dir / "bootstrap.json", {"bootstrap": boot, "witnesses": witnesses})
    write_json(out_dir / "gate.json", report["classification"])

    # csv tables
    with (out_dir / "main_table.csv").open("w") as handle:
        handle.write("arm,fit_overall_cal,dev_overall_raw,dev_overall_cal,dev_G0_raw,dev_G0_cal,bias\n")
        for name in ("B", "D_J", "M_J"):
            a = arms[name]
            handle.write(
                f"{name},{a['fit_overall_cal']:.8f},{a['dev_overall_raw']:.8f},{a['dev_overall_cal']:.8f},"
                f"{a['dev_G0_raw']:.8f},{a['dev_G0_cal']:.8f},{a['calibration_b']:.8f}\n"
            )
    with (out_dir / "group_table.csv").open("w") as handle:
        handle.write("arm,group,n,mae_cal,contribution\n")
        for name in ("B", "D_J", "M_J"):
            for gname, value in arms[name]["groups"].items():
                mae = "" if value["mae"] is None else f"{value['mae']:.8f}"
                handle.write(f"{name},{gname},{value['n']},{mae},{value['contribution']:.8f}\n")
    dev_idx = fold["dev_idx"]
    with (out_dir / "per_graph_dev.csv").open("w") as handle:
        handle.write("dev_pos,gid,k,g,raw_B,cal_B,raw_D_J,cal_D_J,raw_M_J,cal_M_J\n")
        b_bias = arms["B"]["calibration_b"]
        d_bias = arms["D_J"]["calibration_b"]
        m_bias = arms["M_J"]["calibration_b"]
        for i in range(k_dev.size):
            gid = int(np.asarray(tgt["gid"])[dev_idx][i])
            handle.write(
                f"{i},{gid},{int(k_dev[i])},{g_dev[i]:.8f},"
                f"{b_pred['raw_soup_dev'][i]:.8f},{b_pred['raw_soup_dev'][i] + b_bias:.8f},"
                f"{dj_pred['raw_soup_dev'][i]:.8f},{dj_pred['raw_soup_dev'][i] + d_bias:.8f},"
                f"{m_pred['raw_soup_dev'][i]:.8f},{m_pred['raw_soup_dev'][i] + m_bias:.8f}\n"
            )
    fit_idx = fold["fit_idx"]
    with (out_dir / "per_graph_fit.csv").open("w") as handle:
        handle.write("fit_pos,gid,raw_D_J,raw_M_J,raw_B\n")
        for i in range(g_fit.size):
            gid = int(np.asarray(tgt["gid"])[fit_idx][i])
            handle.write(
                f"{i},{gid},{dj_pred['raw_soup_fit'][i]:.8f},{m_pred['raw_soup_fit'][i]:.8f},"
                f"{b_pred['raw_soup_fit'][i]:.8f}\n"
            )
    log(f"[analyze] classification={classification} perf_pass={performance_signal}")
    return report


def _load_prev_predictions_from(dirpath: Path, arm: str) -> dict[str, np.ndarray]:
    with np.load(dirpath / f"{arm}_raw_predictions.npz") as z:
        return {key: np.asarray(z[key], np.float64) for key in z.files}


# ---------------------------------------------------------------------------
# 7. mechanism: operator switch + fit-mean replacement (fit/dev, no retraining)
# ---------------------------------------------------------------------------


def _unscaled_root_codes(model: nn.Module, data_list: Sequence[Any], device: torch.device) -> np.ndarray:
    """Per-fit-root unscaled codes ``e(v)`` (the model multiplies by kappa)."""
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        target = torch.zeros(len(data_list), dtype=torch.float32)
        for start in range(0, len(data_list), 128):
            idx = list(range(start, min(start + 128, len(data_list))))
            batch = zftd.make_batch(data_list, idx, target, device)
            codes = model.local_tuple.root_codes(batch)
            out.append(codes.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _neighbour_rows(encoder: nn.Module, data: Any) -> torch.Tensor:
    """Boolean per concatenated row: the root has at least one incident tuple."""
    phi = data.dict_phi
    device = phi.device
    mol = data.local_mol_id.to(torch.long).reshape(-1)
    n_graphs = int(mol.numel())
    batch = data.batch.to(torch.long)
    graph_counts = torch.bincount(batch, minlength=n_graphs)
    ptr = torch.cat([graph_counts.new_zeros(1), graph_counts.cumsum(0)])
    rows = torch.arange(int(phi.shape[0]), device=device)
    local_pos = rows - ptr[batch]
    root_base = encoder._buf("root_base", device)
    global_root = root_base[mol][batch] + local_pos
    pair_ptr = encoder._buf("pair_ptr", device)
    return pair_ptr[global_root + 1] > pair_ptr[global_root]


def _patch_mean_replace(model: nn.Module, mean_unscaled: torch.Tensor) -> None:
    """Replace the local code by a constant for roots with neighbours.

    Works uniformly for the IHT encoder (previous round) and the M encoder:
    both return the *unscaled* code from ``root_codes`` and multiply by
    ``kappa`` in ``forward``, so the mean-preserving value is inserted before
    that scaling.  Roots without neighbours (d=0) stay exactly zero.
    """
    import types

    encoder = model.local_tuple
    original = encoder.root_codes
    mean_local = mean_unscaled.detach().clone()

    def patched(self: nn.Module, data: Any) -> torch.Tensor:
        out = original(data)
        has = _neighbour_rows(self, data)
        value = mean_local.to(device=out.device, dtype=out.dtype)
        return torch.where(has.unsqueeze(1), value.unsqueeze(0), torch.zeros_like(out))

    encoder.root_codes = types.MethodType(patched, encoder)


def _intervention_entry(
    model: nn.Module,
    fit_data: Sequence[Any],
    dev_data: Sequence[Any],
    g_fit: np.ndarray,
    g_dev: np.ndarray,
    k_dev: np.ndarray,
    bias: float,
    *,
    kind: str,
    device: torch.device,
    native_fit: np.ndarray | None = None,
    native_dev: np.ndarray | None = None,
) -> dict[str, Any]:
    model.eval()
    if native_fit is None:
        native_fit = prev.evaluate_state(model, fit_data, g_fit, device)
    if native_dev is None:
        native_dev = prev.evaluate_state(model, dev_data, g_dev, device)
    extra: dict[str, Any] = {}
    if kind == "operator_switch":
        model.local_tuple.switch_independent = True
        if hasattr(model.local_tuple, "weight_key"):
            model.local_tuple.weight_key = "independent"
    elif kind == "mean_replace":
        unscaled = _unscaled_root_codes(model, fit_data, device)
        mu = unscaled.mean(axis=0)
        extra["mean_scaled_rms"] = float(np.sqrt(np.mean((model.local_tuple.kappa * mu) ** 2)))
        extra["mean_unscaled_rms"] = float(np.sqrt(np.mean(mu**2)))
        _patch_mean_replace(model, torch.as_tensor(mu, dtype=torch.float32))
    else:
        raise ValueError(kind)
    pred_fit = prev.evaluate_state(model, fit_data, g_fit, device)
    pred_dev = prev.evaluate_state(model, dev_data, g_dev, device)
    # restore
    model.local_tuple.switch_independent = False
    if kind == "operator_switch" and hasattr(model.local_tuple, "weight_key"):
        model.local_tuple.weight_key = "joint"
    if kind == "mean_replace":
        del model.local_tuple.root_codes
    g0 = k_dev == 0
    entry: dict[str, Any] = dict(extra)
    for part, target, orig, pred, kk in (
        ("fit", g_fit, native_fit, pred_fit, np.ones(g_fit.shape[0], dtype=bool)),
        ("dev", g_dev, native_dev, pred_dev, g0),
    ):
        delta = pred - orig
        entry[part] = {
            "prediction_delta_mean_abs": float(np.mean(np.abs(delta))),
            "prediction_delta_p95_abs": float(np.percentile(np.abs(delta), 95)),
            "prediction_delta_max_abs": float(np.max(np.abs(delta))),
            "signed_mean": float(delta.mean()),
            "mae_orig_raw": _mae(target, orig),
            "mae_new_raw": _mae(target, pred),
            "mae_change_raw": float(_mae(target, pred) - _mae(target, orig)),
            "mae_orig_cal": _mae(target, orig + bias),
            "mae_new_cal": _mae(target, pred + bias),
            "mae_change_cal": float(_mae(target, pred + bias) - _mae(target, orig + bias)),
            "G0_mae_orig_raw": float(np.mean(np.abs((target - orig)[kk]))),
            "G0_mae_new_raw": float(np.mean(np.abs((target - pred)[kk]))),
            "G0_mae_orig_cal": float(np.mean(np.abs((target - (orig + bias))[kk]))),
            "G0_mae_new_cal": float(np.mean(np.abs((target - (pred + bias))[kk]))),
        }
    return entry


def collect_mechanism(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    payload = prev.load_tuple_payload()
    kappa_M = float(load_kappa_m()["kappa_M"])
    fold, prep_meta, fit_data, dev_data, tgt = prev.load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    k_dev = np.asarray(tgt["k"], np.int64)[fold["dev_idx"]]
    m_meta = json.loads((out_dir / "M_meta.json").read_text())
    dj_meta = json.loads((PREV_DIR / "J_meta.json").read_text())
    dj_pred = _load_prev_predictions("J")
    m_pred = _load_prev_predictions_from(out_dir, "M")
    report: dict[str, Any] = {"official_valid_loaded": False, "official_test_loaded": False}

    # ---------------- D_J: states, health, mean replacement -----------------
    dj_init = torch.load(PREV_DIR / "J_init_state.pt", map_location="cpu", weights_only=False)
    dj_soup = torch.load(PREV_DIR / "J_raw_soup_state.pt", map_location="cpu", weights_only=False)
    dj_last = torch.load(PREV_DIR / "J_last_state.pt", map_location="cpu", weights_only=False)
    dj_model = prev.build_arm("J", payload)
    dj_model.load_state_dict({k: v.to(device) for k, v in dj_soup.items()}, strict=True)
    dj_model = dj_model.to(device)
    dj_bias = float(np.median(g_fit - dj_pred["raw_soup_fit"]))
    dj_entry = {
        "kappa": float(payload.kappa),
        "A_or_D_relative_drift_soup": float(
            (dj_soup["local_tuple.D_loc_raw"] - dj_init["local_tuple.D_loc_raw"]).norm()
            / dj_init["local_tuple.D_loc_raw"].norm()
        ),
        "W_loc_norm_soup": float(dj_soup["local_tuple.W_loc"].norm()),
        "probe_log": dj_meta["probe_log"],
        "native_fit_cal": _mae(g_fit, dj_pred["raw_soup_fit"] + dj_bias),
        "native_dev_cal": _mae(g_dev, dj_pred["raw_soup_dev"] + dj_bias),
        "operator_switch_to_I": _intervention_entry(
            dj_model, fit_data, dev_data, g_fit, g_dev, k_dev, dj_bias,
            kind="operator_switch", device=device,
            native_fit=dj_pred["raw_soup_fit"], native_dev=dj_pred["raw_soup_dev"],
        ),
        "mean_replace": _intervention_entry(
            dj_model, fit_data, dev_data, g_fit, g_dev, k_dev, dj_bias,
            kind="mean_replace", device=device,
            native_fit=dj_pred["raw_soup_fit"], native_dev=dj_pred["raw_soup_dev"],
        ),
    }
    prev_health = json.loads((PREV_DIR / "mechanism_health.json").read_text())
    dj_entry["operator_switch_from_previous_round"] = {
        "source": str(PREV_DIR / "mechanism_health.json"),
        "fit_operator_switched": prev_health["interventions"]["J"]["fit_operator_switched"],
        "dev_operator_switched": prev_health["interventions"]["J"]["dev_operator_switched"],
    }
    report["D_J"] = dj_entry

    # ---------------- M_J: states, health, operator switch, mean replace ----
    m_init = torch.load(out_dir / "M_init_state.pt", map_location="cpu", weights_only=False)
    m_soup = torch.load(out_dir / "M_raw_soup_state.pt", map_location="cpu", weights_only=False)
    m_last = torch.load(out_dir / "M_last_state.pt", map_location="cpu", weights_only=False)
    m_model = build_arm_mj(payload, kappa_M=kappa_M)
    m_model.load_state_dict({k: v.to(device) for k, v in m_soup.items()}, strict=True)
    m_model = m_model.to(device)
    m_bias = float(np.median(g_fit - m_pred["raw_soup_fit"]))
    # root-code health on fit (unscaled codes; kappa recorded separately)
    codes = _unscaled_root_codes(m_model, fit_data, device) * float(m_model.local_tuple.kappa)
    codes_centered = codes - codes.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(codes_centered, compute_uv=False)
    energy = singular**2
    effective_rank = float((energy.sum() ** 2) / (np.sum(energy**2) + 1e-30))
    m_entry = {
        "kappa": kappa_M,
        "A_relative_drift_soup": float(
            (m_soup["local_tuple.A_raw"] - m_init["local_tuple.A_raw"]).norm()
            / m_init["local_tuple.A_raw"].norm()
        ),
        "A_relative_drift_last": float(
            (m_last["local_tuple.A_raw"] - m_init["local_tuple.A_raw"]).norm()
            / m_init["local_tuple.A_raw"].norm()
        ),
        "W_loc_norm_soup": float(m_soup["local_tuple.W_loc"].norm()),
        "probe_log": m_meta["probe_log"],
        "fit_root_code_rms_scaled": float(np.sqrt(np.mean(codes**2))),
        "fit_root_code_dead_dims": int((codes.std(axis=0) < 1e-8).sum()),
        "fit_root_code_effective_rank": effective_rank,
        "fit_root_code_singular_values_head": singular[:8].tolist(),
        "native_fit_cal": _mae(g_fit, m_pred["raw_soup_fit"] + m_bias),
        "native_dev_cal": _mae(g_dev, m_pred["raw_soup_dev"] + m_bias),
        "operator_switch_to_I": _intervention_entry(
            m_model, fit_data, dev_data, g_fit, g_dev, k_dev, m_bias,
            kind="operator_switch", device=device,
            native_fit=m_pred["raw_soup_fit"], native_dev=m_pred["raw_soup_dev"],
        ),
        "mean_replace": _intervention_entry(
            m_model, fit_data, dev_data, g_fit, g_dev, k_dev, m_bias,
            kind="mean_replace", device=device,
            native_fit=m_pred["raw_soup_fit"], native_dev=m_pred["raw_soup_dev"],
        ),
    }
    # zero-ablation at soup (same interface as the previous round)
    m_model.eval()
    with torch.no_grad():
        base_fit = prev.evaluate_state(m_model, fit_data, g_fit, device)
        base_dev = prev.evaluate_state(m_model, dev_data, g_dev, device)
        m_model.local_tuple.ablate = True
        ab_fit = prev.evaluate_state(m_model, fit_data, g_fit, device)
        ab_dev = prev.evaluate_state(m_model, dev_data, g_dev, device)
        m_model.local_tuple.ablate = False
    m_entry["zero_ablation"] = {
        "fit_prediction_delta_mean_abs": float(np.mean(np.abs(ab_fit - base_fit))),
        "fit_mae_change_raw": float(_mae(g_fit, ab_fit) - _mae(g_fit, base_fit)),
        "fit_mae_change_cal": float(_mae(g_fit, ab_fit + m_bias) - _mae(g_fit, base_fit + m_bias)),
        "dev_prediction_delta_mean_abs": float(np.mean(np.abs(ab_dev - base_dev))),
        "dev_mae_change_raw": float(_mae(g_dev, ab_dev) - _mae(g_dev, base_dev)),
        "dev_mae_change_cal": float(_mae(g_dev, ab_dev + m_bias) - _mae(g_dev, base_dev + m_bias)),
    }
    report["M_J"] = m_entry
    write_json(out_dir / "mechanism_health.json", report)
    log("[mechanism] done")
    return report


# ---------------------------------------------------------------------------
# 8. replay, manifest, budget
# ---------------------------------------------------------------------------


def collect_replay(*, out_dir: Path, device: torch.device, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    payload = prev.load_tuple_payload()
    kappa_M = float(load_kappa_m()["kappa_M"])
    fold, prep_meta, fit_data, dev_data, tgt = prev.load_prepared()
    g = np.asarray(tgt["g"], np.float64)
    g_fit, g_dev = g[fold["fit_idx"]], g[fold["dev_idx"]]
    stored = _load_prev_predictions_from(out_dir, "M")
    soup = torch.load(out_dir / "M_raw_soup_state.pt", map_location="cpu", weights_only=False)
    checks: dict[str, Any] = {}
    for part, data, target, key in (
        ("fit", fit_data[:256], g_fit[:256], "raw_soup_fit"),
        ("dev", dev_data[:256], g_dev[:256], "raw_soup_dev"),
    ):
        model = build_arm_mj(payload, kappa_M=kappa_M)
        model.load_state_dict({k: v.to(device) for k, v in soup.items()}, strict=True)
        pred = prev.evaluate_state(model.to(device), data, target, device)
        checks[f"{part}_gpu_max_abs"] = float(np.max(np.abs(pred - stored[key][:256])))
        model_cpu = build_arm_mj(payload, kappa_M=kappa_M)
        model_cpu.load_state_dict(soup, strict=True)
        pred_cpu = prev.evaluate_state(model_cpu, data, target, torch.device("cpu"))
        checks[f"{part}_cpu_max_abs"] = float(np.max(np.abs(pred_cpu - stored[key][:256])))
        checks[f"{part}_cpu_gpu_max_abs"] = float(np.max(np.abs(pred_cpu - pred))) if device.type != "cpu" else 0.0
    checks["ok"] = bool(
        all(v <= REPLAY_TOL for k, v in checks.items() if k != "ok")
    )
    checks["tolerance"] = REPLAY_TOL
    write_json(out_dir / "replay_checks.json", checks)
    log(f"[replay] ok={checks['ok']}")
    if not checks["ok"]:
        raise RuntimeError(f"replay failed: {checks}")
    return checks


def make_manifest(*, out_dir: Path) -> dict[str, Any]:
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".csv", ".pt", ".npz", ".png"})
    manifest = {
        "files": {
            str(path.relative_to(out_dir)): {
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for path in files
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def make_budget(*, out_dir: Path, round_start: str, round_end: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    meta = {}
    for name in ("M_meta.json",):
        path = out_dir / name
        if path.exists():
            meta[name] = json.loads(path.read_text())
    budget = {
        "round_start": round_start,
        "round_end": round_end,
        "formal_arms": 1,
        "arm": "M_J",
        "seed": SEED,
        "epochs": EPOCHS,
        "steps": int(meta.get("M_meta.json", {}).get("steps_done", 0)),
        "training_seconds": float(meta.get("M_meta.json", {}).get("wall_clock_s", 0.0)),
        "eta": "not included",
        "cpu_threads_cap": 8,
    }
    if extra:
        budget.update(jsonable(extra))
    write_json(out_dir / "budget.json", budget)
    return budget


# ---------------------------------------------------------------------------
# 9. entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC local tuple dictionary vs same-capacity MLP projection")
    parser.add_argument("--phase-a", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=("M",))
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--round-start", default="")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    device = zw.resolve_device(args.device)
    if args.phase_a:
        print(json.dumps(jsonable(phase_a(out_dir=out_dir)), indent=2)[:3000])
        return 0
    if args.smoke:
        run_smoke(device=device, out_dir=out_dir)
        return 0
    if args.arm:
        train_arm_mj(device=device, out_dir=out_dir, epochs=int(args.epochs), max_steps=args.max_steps)
        return 0
    if args.analyze:
        analyze(out_dir=out_dir, device=device)
        return 0
    if args.mechanism:
        collect_mechanism(out_dir=out_dir, device=device)
        return 0
    if args.replay:
        collect_replay(out_dir=out_dir, device=device)
        return 0
    if args.manifest:
        make_manifest(out_dir=out_dir)
        return 0
    if args.budget:
        import datetime

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
        make_budget(out_dir=out_dir, round_start=args.round_start or now, round_end=now)
        return 0
    parser.error("choose --phase-a, --smoke, --arm M, --analyze, --mechanism, --replay, --manifest or --budget")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
