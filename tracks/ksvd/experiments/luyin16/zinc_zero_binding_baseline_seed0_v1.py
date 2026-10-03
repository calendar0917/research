"""ZINC zero-binding baseline, single seed (seed 0).

Two consecutive questions on top of the frozen ``zinc-structure-semantic-
factorial-seed0-v1`` (source branch ``task/zinc-structure-semantic-factorial-
seed0-v1``, deployed commit ``118e2481362f``):

A. **Inference simplification.**  Trained ``S_M`` (sparse tied-IHT coding,
   independent-pairing aggregation) has node/edge binding weights at the
   denormal floor.  Are the two slot tensors *constant* over the whole 8000
   fit + 2000 dev stream, and is the final prediction therefore exactly
   reproducible without D/U/coding, the binding projections and the slot
   encoders?  If yes, export an algebraically equivalent reduced model
   (fusion input ``[Sem108; size2]``, 110-D) and check
   ``max|pred_raw_diff| <= 1e-5``.

B. **N0 trajectory.**  One *new* formal 240-epoch trajectory, identical to
   ``S_M`` in every recipe detail, except that both slot tensors are fixed to
   zero from the very first optimizer step:
   ``node_slots = node_slots * 0.0; edge_slots = edge_slots * 0.0`` after the
   aggregation and before the encoders.  The multiply (not ``zeros_like`` /
   ``detach``) keeps the original autograd graph, so upstream parameters get
   exact *zero* task gradients while ``D`` keeps its reconstruction gradient
   and the optimizer still sees all 408,651 trainable parameters.

The structural dictionary ``D``/``U`` (phi65 -> common coordinate + sparse
residual code) is the object whose *binding path* is tested here.  The task
dictionary ``D_L/V_L``, Sem108, size2, topology25 and the static relations all
remain in both the reduced deployment model and N0: removing the explicit
binding is not "dictionary-free".

Train-only: ``encoded_train.pt`` + ``env_train.pt``.  The official validation
split is never instantiated; the official test split is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_structure_semantic_factorial_seed0_v1 as zsf

PROTOCOL_VERSION = "zinc-zero-binding-baseline-seed0-v1"

#: source branch HEAD (git rev-parse task/zinc-structure-semantic-factorial-seed0-v1)
SOURCE_COMMIT = "4083ee38d8785c5a44d1f9628a5baeed568d4c34"
#: commit that actually trained/exported the four source arms
SOURCE_DEPLOYED_COMMIT = "118e2481362f"
SOURCE_BRANCH = "task/zinc-structure-semantic-factorial-seed0-v1"

SOURCE_RESULTS = zjd.TRACK_ROOT / "results/zinc_structure_semantic_factorial_seed0_v1"
RESULTS_DIR = zjd.TRACK_ROOT / "results/zinc_zero_binding_baseline_seed0_v1"

SEED = 0
SCALE_SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
TRAIN_SHUFFLE_OFFSET = 101
HEALTH_EPOCHS = (1, 40, 120, 240)
HEALTH_SAMPLE = 256
N0_FUSION_IN = 110
#: theoretical fusion first-layer weight saving: 342 * (446 - 110)
FUSION_FIRST_LAYER_SAVING = 342 * (446 - N0_FUSION_IN)

COMPRESS_TOL = 1.0e-5
SOURCE_ARM_BIAS = {"S_J": -0.021329760551452637, "S_M": -0.0004400014877319336}
BOOT_SEED = 20261003
N_BOOT = 1000
DELTA = 0.003
GROUP_KEYS = ("k0", "k-1", "kle-2")
GROUP_NAMES = {"k0": "k=0", "k-1": "k=-1", "kle-2": "k<=-2"}


# ---------------------------------------------------------------------------
# small helpers (same conventions as the source module)
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def group_of(kv: int) -> str:
    if int(kv) == 0:
        return "k=0"
    if int(kv) == -1:
        return "k=-1"
    return "k<=-2"


def group_mask(k_dev: np.ndarray, key: str) -> np.ndarray:
    if key == "k0":
        return k_dev == 0
    if key == "k-1":
        return k_dev == -1
    return k_dev <= -2


def schedule_hash(n: int, epochs: int, offset: int) -> dict[str, str]:
    """Deterministic hash of the full per-epoch index schedule."""
    gen = torch.Generator().manual_seed(int(SEED) + int(offset))
    whole = hashlib.sha256()
    first = None
    for epoch in range(int(epochs)):
        batches = zftd.epoch_batches(int(n), BATCH_SIZE, gen, True)
        for indices in batches:
            arr = np.asarray(indices, np.int64)
            whole.update(arr.tobytes())
            if first is None:
                first = hashlib.sha256(arr.tobytes()).hexdigest()
    return {"full_schedule_sha256": whole.hexdigest(), "first_batch_sha256": first}


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_source_state(arm: str, device: torch.device) -> dict[str, torch.Tensor]:
    path = SOURCE_RESULTS / f"{arm}_raw_soup_state.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    return {k: v.to(device) for k, v in state.items()}


# ---------------------------------------------------------------------------
# N0: S_M with the two slot tensors fixed to zero from the first step
# ---------------------------------------------------------------------------


class ZeroBindingFull(zsf.FactorialFull):
    """``FactorialFull`` (sparse + indep = ``S_M``) with the N0 switch.

    The switch replaces, in the independent-pairing environment, the two slot
    tensors by ``slot * 0.0`` *after* the bucket aggregation and *before* the
    slot encoders.  Multiplication (not ``zeros_like``/``detach``) is the
    intervention: the upstream binding parameters still receive a (zero)
    gradient tensor, so Adam's coupled L2 and the global grad-norm clip see
    the original parameter set.
    """

    def __init__(self, *args: Any, zero_binding_enabled: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.zero_binding_enabled = bool(zero_binding_enabled)
        self._probe: dict[str, torch.Tensor] | None = None

    # -- intervention ---------------------------------------------------------

    def _indep_environment_from_parts(
        self,
        coord: torch.Tensor,
        data: Any,
        interface: torch.Tensor,
        *,
        occ_coord_node: torch.Tensor | None = None,
        bond_u: torch.Tensor | None = None,
        bond_v: torch.Tensor | None = None,
        node_binding_zero: bool = False,
        edge_binding_zero: bool = False,
        mask: Any = None,
    ) -> torch.Tensor:
        if not self.zero_binding_enabled:
            return super()._indep_environment_from_parts(
                coord,
                data,
                interface,
                occ_coord_node=occ_coord_node,
                bond_u=bond_u,
                bond_v=bond_v,
                node_binding_zero=node_binding_zero,
                edge_binding_zero=edge_binding_zero,
                mask=mask,
            )
        # --- body copied from FactorialFull._indep_environment_from_parts with
        # --- exactly two added lines (marked N0 INTERVENTION).  Keep in sync.
        n = int(coord.shape[0])
        q = F.one_hot(data.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
        if occ_coord_node is None:
            occ_coord_node = data.env_occ_node
        occ_coord_node = occ_coord_node.to(coord.device)
        c = coord[occ_coord_node]
        qc = q[data.env_occ_node.to(coord.device)]
        a = c @ self.W_A_S
        b = qc @ self.W_A_C
        node_index = data.env_occ_root.to(coord.device) * int(p2.N_SHELLS) + data.env_occ_shell.to(
            coord.device
        )
        node_flat = zsf.indep_bucket(a, b, node_index, n * int(p2.N_SHELLS), int(p2.D_A))
        if node_binding_zero:
            node_flat = torch.zeros_like(node_flat)
        node_slots = node_flat.view(n, int(p2.N_SHELLS), int(p2.D_A))

        d_e = int(self.config.d_e)
        if bond_u is None:
            bond_u = data.env_bond_u
        if bond_v is None:
            bond_v = data.env_bond_v
        bond_u = bond_u.to(coord.device)
        bond_v = bond_v.to(coord.device)
        cu = coord[bond_u]
        cv = coord[bond_v]
        g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
        if edge_binding_zero:
            g = torch.zeros_like(g)
        s = g @ self.W_E_S
        bcat = F.one_hot(data.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
        ce = bcat @ self.W_E_C
        edge_index = data.env_bond_root.to(coord.device) * int(p2.SHELLPAIR_CLASSES) + (
            data.env_bond_shellpair.to(coord.device)
        )
        edge_flat = zsf.indep_bucket(s, ce, edge_index, n * int(p2.SHELLPAIR_CLASSES), int(d_e))
        edge_slots = edge_flat.view(n, int(p2.SHELLPAIR_CLASSES), int(d_e))

        # N0 INTERVENTION: fix both slot tensors, keeping the autograd graph.
        node_slots = node_slots * 0.0
        edge_slots = edge_slots * 0.0

        node_out = self.node_encoder(node_slots)
        edge_out = self.edge_encoder(edge_slots)
        if self._probe is not None:
            self._probe.update(
                node_flat_prezero=node_flat.detach(),
                edge_flat_prezero=edge_flat.detach(),
                node_slots_postzero=node_slots.detach(),
                edge_slots_postzero=edge_slots.detach(),
                node_out=node_out.detach(),
                edge_out=edge_out.detach(),
            )
        if self.parent_interface:
            raise RuntimeError("parent_interface is not supported by the factorial model")
        fused = torch.cat([interface, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
        return self.fusion(fused)


def build_zero_model(
    blob: Mapping[str, Any],
    seed: int = SEED,
    *,
    zero_binding_enabled: bool = True,
    device: torch.device | None = None,
) -> ZeroBindingFull:
    """Same RNG stream as ``zsf.build_factorial_model(..., 'sparse', 'indep')``."""
    torch.manual_seed(int(seed))
    model = ZeroBindingFull(
        cm.H1_CONFIG,
        np.asarray(blob["D_fit"], np.float32),
        subspace=zjd.fold_subspace(blob),
        spec=sc.FULL,
        coding_mode="sparse",
        binding_mode="indep",
        scale_seed=SCALE_SEED,
        zero_binding_enabled=bool(zero_binding_enabled),
    )
    audit_ = sc.scale_parameter_audit(model)
    if int(audit_["actual_parameters"]) != zsf.FULL_PARAMETERS or not audit_["parameter_exact"]:
        raise RuntimeError(f"N0 parameter audit failed: {audit_}")
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# deployment: S_M / N0 with the structural slot path removed
# ---------------------------------------------------------------------------


class DeployFull(zsf.FactorialFull):
    """Reduced prediction model: no ``D``/``U``/coding, no binding projections,
    no slot encoders; the fusion first layer takes ``[Sem108; size2]`` (110-D).

    Constructed from the source class only to inherit the exact task-path
    modules and ``AuditModel.forward``; the structural path is deleted in
    ``__init__`` and the reduced fusion is loaded from
    :func:`reduction_from_state`.  ``code`` returns a zero placeholder (the
    prediction path must never read phi65 again) and ``environments_masked``
    jumps straight to ``fusion(interface)``.
    """

    def __init__(self) -> None:
        rng = np.random.RandomState(0)
        dummy = rng.randn(int(cssd.PHI_DIM), int(cssd.K_ATOMS)).astype(np.float32) * 0.1
        u = np.ones((int(cssd.PHI_DIM), 1), np.float32)
        u /= np.linalg.norm(u)
        subspace = cssd.CommonSubspace(components=u, rms=np.ones(1, np.float32), kind="q1")
        torch.manual_seed(int(SEED))
        zsf.FactorialFull.__init__(
            self,
            cm.H1_CONFIG,
            dummy,
            subspace=subspace,
            spec=sc.FULL,
            coding_mode="sparse",
            binding_mode="indep",
            scale_seed=SCALE_SEED,
        )
        self.zero_binding_enabled = True
        self._probe = None
        # delete the structural / slot path
        for name in ("D", "W_A_S", "W_A_C", "W_E_S", "W_E_C"):
            delattr(self, name)
        del self.node_encoder
        del self.edge_encoder
        for buffer_name in ("U", "common_rms", "kappa"):
            delattr(self, buffer_name)
        old_fusion = self.fusion
        self.fusion = nn.Sequential(nn.Linear(N0_FUSION_IN, int(old_fusion[0].out_features)), old_fusion[1], old_fusion[2])
        self.anchor_encoder = None
        self.deploy_model = True
        self.deploy_fusion_in = N0_FUSION_IN

    # -- removed inputs -------------------------------------------------------

    def code(self, phi: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (int(phi.shape[0]), int(self.common_dim) + int(cssd.K_ATOMS)),
            dtype=phi.dtype,
            device=phi.device,
        )

    def _environment_from_parts(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("deploy model has no structural slot path")

    def environments_masked(
        self,
        coord: torch.Tensor,
        data: Any,
        mask: Any,
        fill: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        interface = sem.SEM108Model.semantic_interface(self, coord, data, mask, fill)
        return self.local_dictionary_bridge(self.fusion(interface))


REMOVED_STATE_KEYS = {
    "D",
    "W_A_S",
    "W_A_C",
    "W_E_S",
    "W_E_C",
    "U",
    "common_rms",
    "kappa",
}
REMOVED_STATE_PREFIXES = ("node_encoder.", "edge_encoder.")


def _keep_state_key(key: str) -> bool:
    return key not in REMOVED_STATE_KEYS and not key.startswith(REMOVED_STATE_PREFIXES)


def reduction_from_state(model: zsf.FactorialFull) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    """Exact algebraic reduction of a loaded S_M/N0 soup state.

    ``W_reduced = W_sem`` (the first 110 fusion columns) and
    ``b_reduced = b_fusion + W_slot @ c_slot`` with
    ``c_slot = [node_encoder(0); edge_encoder(0)]``.  The removed modules are
    dropped from the checkpoint, so the returned dict is the deployment state.
    """
    dev = next(model.parameters()).device
    with torch.no_grad():
        node0 = model.node_encoder(
            torch.zeros(1, int(p2.N_SHELLS), int(p2.D_A), dtype=torch.float32, device=dev)
        )
        edge0 = model.edge_encoder(
            torch.zeros(1, int(p2.SHELLPAIR_CLASSES), int(model.config.d_e), dtype=torch.float32, device=dev)
        )
        c_slot = torch.cat([node0.reshape(-1), edge0.reshape(-1)]).clone()
        old = model.fusion
        if not isinstance(old, nn.Sequential) or not isinstance(old[0], nn.Linear):
            raise RuntimeError("unexpected fusion layout")
        out_features = int(old[0].out_features)
        reduced_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if _keep_state_key(k)}
        reduced_state["fusion.0.weight"] = old[0].weight[:, :N0_FUSION_IN].detach().cpu().clone()
        reduced_state["fusion.0.bias"] = (old[0].bias + old[0].weight[:, N0_FUSION_IN:] @ c_slot).detach().cpu().clone()
        full_params = int(sum(p.numel() for p in model.parameters()))
        kept_modules = sorted(set(k.split(".")[0] for k in reduced_state if not k.startswith("fusion.0")))
    meta = {
        "c_slot": c_slot.cpu(),
        "c_slot_dim": int(c_slot.numel()),
        "full_trainable_parameters": full_params,
        "fusion_weight_saving": int(out_features * (446 - N0_FUSION_IN)),
        "removed_state_keys": sorted(REMOVED_STATE_KEYS),
        "kept_modules": kept_modules,
    }
    return c_slot.cpu(), reduced_state, meta


def build_deploy_model(state: Mapping[str, torch.Tensor] | None = None, *, device: torch.device | None = None) -> DeployFull:
    """Standalone loader: reduced architecture + optional state dict."""
    model = DeployFull()
    if state is not None:
        model.load_state_dict({k: v.to(torch.device("cpu")) for k, v in state.items()}, strict=True)
    if device is not None:
        model = model.to(device)
    return model


def build_reduced_from_soup(model: zsf.FactorialFull, *, device: torch.device | None = None) -> tuple[DeployFull, dict[str, Any]]:
    """Reduced deployment model built from the loaded full soup state."""
    c_slot, reduced_state, meta = reduction_from_state(model)
    deploy = build_deploy_model(reduced_state, device=device)
    meta["c_slot"] = c_slot
    meta["deploy_parameters"] = int(sum(p.numel() for p in deploy.parameters()))
    meta["removed_parameters"] = int(meta["full_trainable_parameters"] - meta["deploy_parameters"])
    return deploy, meta


# ---------------------------------------------------------------------------
# Phase A — constants, substitution, equivalence
# ---------------------------------------------------------------------------


def _load_source_fit_dev():
    blob = zsf.load_prep_blob()
    decomp = zsf.load_target()
    train_data = zsf.load_train_only()
    zftd.apply_prep_train_only(train_data, blob)
    fit_data, dev_data = zsf.build_fit_dev(train_data, blob["fit_idx"], blob["dev_idx"])
    fit_idx = np.asarray(blob["fit_idx"], np.int64)
    dev_idx = np.asarray(blob["dev_idx"], np.int64)
    y = np.asarray(decomp["y"], np.float64)
    return blob, decomp, train_data, fit_data, dev_data, fit_idx, dev_idx, y


def phaseA_check(*, device: torch.device = torch.device("cpu"), log: Any = print) -> dict[str, Any]:
    """Stream all 8000 fit + 2000 dev rows through the trained S_M soup.

    Records the *production* slot tensors (captured with forward hooks on the
    slot encoders), the raw M aggregation, encoder outputs, and S_M raw
    predictions.  Then substitutes coord / perturbs D on a fixed batch.
    """
    t0 = time.perf_counter()
    torch.set_num_threads(8)
    blob, decomp, _train_data, fit_data, dev_data, fit_idx, dev_idx, y = _load_source_fit_dev()
    y_fit = torch.as_tensor(y[fit_idx], dtype=torch.float32)
    y_dev = torch.as_tensor(y[dev_idx], dtype=torch.float32)

    model = zsf.build_factorial_model(blob, zsf.SEED, "sparse", "indep").to(device)
    state = load_source_state("S_M", device)
    model.load_state_dict(state)
    model.eval()
    got = zsf.state_hash(state)
    expected = "2def4cb64c1350a60348fb6fc7ce88a07bffe563367ac7e9666ee6f7dff5c5b8"
    if got != expected:
        raise RuntimeError(f"S_M raw soup state hash mismatch: {got} != {expected}")

    captured: dict[str, list[torch.Tensor]] = {"node_slots": [], "edge_slots": [], "node_out": [], "edge_out": []}

    def _capture_node(module, inputs, output):
        captured["node_slots"].append(inputs[0].detach().cpu())
        captured["node_out"].append(output.detach().cpu())

    def _capture_edge(module, inputs, output):
        captured["edge_slots"].append(inputs[0].detach().cpu())
        captured["edge_out"].append(output.detach().cpu())

    handles = [
        model.node_encoder.register_forward_hook(_capture_node),
        model.edge_encoder.register_forward_hook(_capture_edge),
    ]

    stats: dict[str, dict[str, float]] = {}
    raw_pred: dict[str, np.ndarray] = {}
    for split, data, target in (("fit", fit_data, y_fit), ("dev", dev_data, y_dev)):
        preds = []
        node_absmax = edge_absmax = 0.0
        node_sq = edge_sq = 0.0
        node_elems = edge_elems = 0
        node_rowvar = edge_rowvar = 0.0
        nodeout_rowvar = edgeout_rowvar = 0.0
        for i in range(0, len(data), BATCH_SIZE):
            indices = list(range(i, min(i + BATCH_SIZE, len(data))))
            batch = zsf.make_batch(data, indices, target, torch.device("cpu"))
            with torch.no_grad():
                pred = model(batch, mask=cm.C6_MASK)
            preds.append(pred.view(-1).detach())
            ns = captured["node_slots"][-1]
            es = captured["edge_slots"][-1]
            no = captured["node_out"][-1]
            eo = captured["edge_out"][-1]
            node_absmax = max(node_absmax, float(ns.abs().max()))
            edge_absmax = max(edge_absmax, float(es.abs().max()))
            node_sq += float(ns.pow(2).sum())
            edge_sq += float(es.pow(2).sum())
            node_elems += int(ns.numel())
            edge_elems += int(es.numel())
            # per-row variation (deviation of every row from the first row)
            node_rowvar = max(node_rowvar, float((ns - ns[0:1]).abs().max()))
            edge_rowvar = max(edge_rowvar, float((es - es[0:1]).abs().max()))
            nodeout_rowvar = max(nodeout_rowvar, float((no - no[0:1]).abs().max()))
            edgeout_rowvar = max(edgeout_rowvar, float((eo - eo[0:1]).abs().max()))
        raw_pred[split] = torch.cat(preds).numpy().astype(np.float64)
        stats[split] = {
            "node_slot_absmax": node_absmax,
            "edge_slot_absmax": edge_absmax,
            "node_slot_rms": float(math.sqrt(node_sq / max(node_elems, 1))),
            "edge_slot_rms": float(math.sqrt(edge_sq / max(edge_elems, 1))),
            "node_slot_rowvar": node_rowvar,
            "edge_slot_rowvar": edge_rowvar,
            "node_encoder_out_rowvar": nodeout_rowvar,
            "edge_encoder_out_rowvar": edgeout_rowvar,
        }
        for key in captured:
            captured[key].clear()
    for handle in handles:
        handle.remove()

    # replay vs the released S_M predictions (same rows, fit then dev)
    released = np.load(SOURCE_RESULTS / "S_M_predictions.npz")
    released_fit = released["fit_raw"].astype(np.float64)
    pred_csv = np.genfromtxt(SOURCE_RESULTS / "predictions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    csv_dev = np.asarray([float(v) for v in pred_csv["S_M_raw"][-len(dev_idx):]], dtype=np.float64)
    csv_fit = np.asarray([float(v) for v in pred_csv["S_M_raw"][: len(fit_idx)]], dtype=np.float64)

    # --- code-substitution / D-intervention on a fixed batch -----------------
    sub_batch = zsf.make_batch(fit_data, list(range(64)), y_fit, torch.device("cpu"))
    orig_code = model.code

    def _run_with(code_fn):
        model.code = code_fn
        try:
            with torch.no_grad():
                return model(sub_batch, mask=cm.C6_MASK).view(-1).detach().clone()
        finally:
            model.code = orig_code

    base = _run_with(orig_code)
    zeros = _run_with(lambda phi: torch.zeros_like(orig_code(phi)))
    const = _run_with(lambda phi: torch.full_like(orig_code(phi), 0.7))
    with torch.no_grad():
        model.D.add_(0.01)
    perturbed = _run_with(orig_code)
    with torch.no_grad():
        model.D.sub_(0.01)
    restore = _run_with(orig_code)

    precheck = {
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "source_arm": "S_M",
        "source_soup_state_sha256": got,
        "source_soup_state_sha256_expected": expected,
        "n_fit": int(len(fit_idx)),
        "n_dev": int(len(dev_idx)),
        "slot_constants": stats,
        "replay_vs_released_fit_raw_max_abs": float(np.max(np.abs(raw_pred["fit"] - released_fit))),
        "replay_vs_old_predictions_csv": {
            "fit_max_abs": float(np.max(np.abs(raw_pred["fit"] - csv_fit))),
            "dev_max_abs": float(np.max(np.abs(raw_pred["dev"] - csv_dev))),
        },
        "code_substitution": {
            "real_vs_zero_coord_max_abs": float((base - zeros).abs().max()),
            "real_vs_const_coord_max_abs": float((base - const).abs().max()),
            "real_vs_perturbed_D_max_abs": float((base - perturbed).abs().max()),
            "real_vs_restored_max_abs": float((base - restore).abs().max()),
            "tolerance": COMPRESS_TOL,
        },
        "seconds": float(time.perf_counter() - t0),
    }
    precheck["pass"] = bool(
        stats["fit"]["node_slot_absmax"] == 0.0
        and stats["fit"]["edge_slot_absmax"] == 0.0
        and stats["dev"]["node_slot_absmax"] == 0.0
        and stats["dev"]["edge_slot_absmax"] == 0.0
        and stats["fit"]["node_encoder_out_rowvar"] <= COMPRESS_TOL
        and stats["fit"]["edge_encoder_out_rowvar"] <= COMPRESS_TOL
        and stats["dev"]["node_encoder_out_rowvar"] <= COMPRESS_TOL
        and stats["dev"]["edge_encoder_out_rowvar"] <= COMPRESS_TOL
        and precheck["code_substitution"]["real_vs_zero_coord_max_abs"] <= COMPRESS_TOL
        and precheck["code_substitution"]["real_vs_const_coord_max_abs"] <= COMPRESS_TOL
        and precheck["code_substitution"]["real_vs_perturbed_D_max_abs"] <= COMPRESS_TOL
        and precheck["code_substitution"]["real_vs_restored_max_abs"] <= COMPRESS_TOL
    )
    write_json(RESULTS_DIR / "phaseA_precheck.json", precheck)
    np.savez_compressed(RESULTS_DIR / "phaseA_sm_replay.npz", fit_raw=raw_pred["fit"], dev_raw=raw_pred["dev"])
    log(f"[phaseA_check] pass={precheck['pass']} seconds={precheck['seconds']:.1f}")
    return precheck


def phaseA_export(*, device: torch.device = torch.device("cpu"), log: Any = print) -> dict[str, Any]:
    """Build the reduced deployment model for the trained S_M soup and verify
    full fit+dev equivalence, label independence and graph-order invariance."""
    t0 = time.perf_counter()
    torch.set_num_threads(8)
    blob, _decomp, _train_data, fit_data, dev_data, fit_idx, dev_idx, y = _load_source_fit_dev()
    y_fit = torch.as_tensor(y[fit_idx], dtype=torch.float32)
    y_dev = torch.as_tensor(y[dev_idx], dtype=torch.float32)

    model = zsf.build_factorial_model(blob, zsf.SEED, "sparse", "indep").to(device)
    model.load_state_dict(load_source_state("S_M", device))
    model.eval()
    deploy, shrink = build_reduced_from_soup(model, device=device)
    deploy.eval()
    model = deploy

    replay = np.load(RESULTS_DIR / "phaseA_sm_replay.npz")
    report: dict[str, Any] = {"source_arm": "S_M", "shrink": {k: v for k, v in shrink.items() if k != "c_slot"}}
    diffs = {}
    for split, data, target, key in (
        ("fit", fit_data, y_fit, "fit_raw"),
        ("dev", dev_data, y_dev, "dev_raw"),
    ):
        pred, _ = zsf.predict(model, data, target, device)
        ref = replay[key]
        diffs[split] = float(np.max(np.abs(pred - ref)))
        yv = y[fit_idx if split == "fit" else dev_idx]
        report[f"{split}_mae_deploy"] = float(np.mean(np.abs(pred - yv)))
        report[f"{split}_mae_source"] = float(np.mean(np.abs(ref - yv)))
        report[f"{split}_mae_diff"] = report[f"{split}_mae_deploy"] - report[f"{split}_mae_source"]
    report["equivalence"] = diffs
    report["tolerance"] = COMPRESS_TOL
    report["pass"] = bool(max(diffs.values()) <= COMPRESS_TOL)

    # label independence + graph order on a fixed batch
    sub = zsf.make_batch(fit_data, list(range(64)), y_fit, torch.device("cpu"))
    with torch.no_grad():
        p1_ = model(sub, mask=cm.C6_MASK).view(-1).clone()
        sub2 = sub.clone()
        sub2.y = torch.randn_like(sub2.y)
        p2_ = model(sub2, mask=cm.C6_MASK).view(-1).clone()
        ab = model(zsf.make_batch(fit_data, [0, 1], y_fit, torch.device("cpu")), mask=cm.C6_MASK).view(-1)
        ba = model(zsf.make_batch(fit_data, [1, 0], y_fit, torch.device("cpu")), mask=cm.C6_MASK).view(-1)
    report["label_independence_max_abs"] = float((p1_ - p2_).abs().max())
    report["graph_order_map_diff"] = float(max(abs(float(ab[0]) - float(ba[1])), abs(float(ab[1]) - float(ba[0]))))
    report["pass"] = bool(
        report["pass"] and report["label_independence_max_abs"] <= COMPRESS_TOL and report["graph_order_map_diff"] <= COMPRESS_TOL
    )

    # independent loader check
    torch.save(model.state_dict(), RESULTS_DIR / "S_M_deploy_state.pt")
    fresh = build_deploy_model(model.state_dict(), device=device)
    fresh.eval()
    report["deploy_parameters"] = int(sum(p.numel() for p in fresh.parameters()))
    with torch.no_grad():
        pf = fresh(sub, mask=cm.C6_MASK).view(-1)
    report["independent_load_max_abs"] = float((pf - p1_[: pf.numel()]).abs().max())
    report["independent_load_keys_equal"] = bool(set(fresh.state_dict().keys()) == set(model.state_dict().keys()))
    report["pass"] = bool(report["pass"] and report["independent_load_max_abs"] <= COMPRESS_TOL)

    report["deploy_parameters"] = int(sum(p.numel() for p in fresh.parameters()))
    report["seconds"] = float(time.perf_counter() - t0)
    write_json(RESULTS_DIR / "phaseA_export.json", report)
    log(f"[phaseA_export] pass={report['pass']} diffs={diffs} seconds={report['seconds']:.1f}")
    return report


# ---------------------------------------------------------------------------
# health + training
# ---------------------------------------------------------------------------


def n0_health_snapshot(model: ZeroBindingFull, batch: Any) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    model._probe = {}
    with torch.no_grad():
        coord = model.code(batch.dict_phi)
        pred, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        p = dict(model._probe)
        node_pre = p["node_flat_prezero"]
        edge_pre = p["edge_flat_prezero"]
        ns = p["node_slots_postzero"]
        es = p["edge_slots_postzero"]
        no = p["node_out"]
        eo = p["edge_out"]
        payload = {
            "n_rows": int(coord.shape[0]),
            "coord_absmax": float(coord.abs().max()),
            "coord_std": float(coord.std()),
            "node_prezero_absmax": float(node_pre.abs().max()),
            "node_prezero_rms": float(node_pre.pow(2).mean().sqrt()),
            "edge_prezero_absmax": float(edge_pre.abs().max()),
            "edge_prezero_rms": float(edge_pre.pow(2).mean().sqrt()),
            "node_slots_absmax": float(ns.abs().max()),
            "edge_slots_absmax": float(es.abs().max()),
            "node_encoder_out_rowvar": float((no - no[0:1]).abs().max()),
            "edge_encoder_out_rowvar": float((eo - eo[0:1]).abs().max()),
            "node_encoder_out_absmax": float(no.abs().max()),
            "edge_encoder_out_absmax": float(eo.abs().max()),
            "reconstruction_loss": float(rec),
            "pred_absmax": float(pred.abs().max()),
        }
        payload["finite"] = bool(
            bool(torch.isfinite(node_pre).all())
            and bool(torch.isfinite(edge_pre).all())
            and math.isfinite(payload["coord_absmax"])
        )
    model._probe = None
    model.train(was_training)
    return payload


def train_n0(
    *,
    device: torch.device,
    out_dir: Path = RESULTS_DIR,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """The single new 240-epoch N0 trajectory (or a <=4-step smoke)."""
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    blob, decomp, _train_data, fit_data, _dev_data, fit_idx, _dev_idx, y = _load_source_fit_dev()
    y_fit_np = np.asarray(y, np.float64)[fit_idx]
    target_fit = torch.as_tensor(y_fit_np, dtype=torch.float32)

    seed_everything(SEED)
    model = build_zero_model(blob, SEED, zero_binding_enabled=True, device=device)
    kappa_info = zsf.compute_kappa(model, fit_data)
    with torch.no_grad():
        model.kappa.fill_(float(kappa_info["kappa"]))
    init_hash = zsf.trainable_parameter_hash(model)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(int(SEED) + TRAIN_SHUFFLE_OFFSET)
    soup_epochs = set(range(max(1, int(epochs) - 4), int(epochs) + 1))
    health_batch = zsf.make_batch(
        fit_data,
        list(range(min(HEALTH_SAMPLE, len(fit_data)))),
        torch.as_tensor(y_fit_np[:HEALTH_SAMPLE], dtype=torch.float32),
        device,
    )
    curve: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    soup: dict[int, dict[str, torch.Tensor]] = {}
    stream = hashlib.sha256()
    steps_done = 0
    started = time.perf_counter()
    stopped_reason = "completed"

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps, gnorm_sum, clip_hits = 0.0, 0.0, 0, 0, 0.0, 0
        for indices in zftd.epoch_batches(len(fit_data), BATCH_SIZE, train_gen, True):
            stream.update(np.asarray(indices, np.int64).tobytes())
            batch = zsf.make_batch(fit_data, indices, target_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + H1_LAMBDA * rec
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
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
                "train_rec": float(rec_sum / max(n_steps, 1)),
                "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                "clip_fraction": float(clip_hits / max(n_steps, 1)),
                "seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if int(epoch) in HEALTH_EPOCHS:
            health[str(epoch)] = n0_health_snapshot(model, health_batch)
        if int(epoch) in soup_epochs:
            soup[int(epoch)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(
                f"[N0] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"rec={curve[-1]['train_rec']:.4g} gnorm={curve[-1]['grad_norm']:.3g} "
                f"clip={curve[-1]['clip_fraction']:.3f} {curve[-1]['seconds']:.1f}s"
            )
        if max_steps is not None and steps_done >= int(max_steps):
            break

    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": "N0",
        "seed": SEED,
        "coding_mode": "sparse",
        "binding_mode": "indep_zero_slots",
        "epochs": int(epochs),
        "steps_done": int(steps_done),
        "stopped_reason": stopped_reason,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "h1_lambda": H1_LAMBDA,
        "trainable_parameters": zsf.trainable_parameter_count(model),
        "init_trainable_hash": init_hash,
        "kappa": kappa_info,
        "curve": curve,
        "health": health,
        "data_stream_sha256": stream.hexdigest(),
        "wall_clock_s": float(time.perf_counter() - started),
        "device": str(device),
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }

    if max_steps is not None:
        result["last_trainable_hash"] = zsf.trainable_parameter_hash(model)
        if log:
            log(f"[N0] smoke steps={steps_done} stream={stream.hexdigest()[:16]}")
        return result

    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(soup_state, out_dir / "N0_raw_soup_state.pt")
    torch.save(last_state, out_dir / "N0_last_state.pt")
    torch.save(init_state, out_dir / "N0_init_state.pt")

    replay = build_zero_model(blob, SEED, zero_binding_enabled=True, device=device)
    replay.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
    replay.eval()
    fit_raw, _ = zsf.predict(replay, fit_data, torch.as_tensor(y_fit_np, dtype=torch.float32), device)
    b = float(np.median(y_fit_np - fit_raw))
    result.update(
        {
            "soup_members": members,
            "soup_state_sha256": zsf.state_hash(soup_state),
            "last_state_sha256": zsf.state_hash(last_state),
            "init_state_sha256": zsf.state_hash(init_state),
            "calibration": {"b": b, "folded_into_state": False},
            "fit_raw_mae_y": float(np.mean(np.abs(fit_raw - y_fit_np))),
            "fit_cal_mae_y": float(np.mean(np.abs(fit_raw + b - y_fit_np))),
        }
    )
    np.savez_compressed(out_dir / "N0_predictions.npz", fit_raw=fit_raw.astype(np.float32), fit_idx=fit_idx, b=np.float64(b))
    write_json(out_dir / "N0.json", result)
    log(
        f"[N0] DONE fit_raw={result['fit_raw_mae_y']:.6f} fit_cal={result['fit_cal_mae_y']:.6f} "
        f"b={b:.6f} wall={result['wall_clock_s']:.0f}s"
    )
    return result


# ---------------------------------------------------------------------------
# smoke (<= 4 optimizer steps, all state discarded)
# ---------------------------------------------------------------------------


def run_smoke(*, device: torch.device, out_dir: Path = RESULTS_DIR, max_steps: int = 4, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    blob, decomp, _train_data, fit_data, _dev_data, fit_idx, _dev_idx, y = _load_source_fit_dev()
    y_fit = np.asarray(y, np.float64)[fit_idx]
    checks: dict[str, Any] = {"official_test_loaded": False, "official_valid_loaded": False}

    # 1. initial state identical to the source S_M factory and to the released init
    seed_everything(SEED)
    source = zsf.build_factorial_model(blob, zsf.SEED, "sparse", "indep").to(device)
    source_kappa = zsf.compute_kappa(source, fit_data)
    with torch.no_grad():
        source.kappa.fill_(float(source_kappa["kappa"]))
    source_state = {k: v.detach().cpu() for k, v in source.state_dict().items()}
    seed_everything(SEED)
    n0 = build_zero_model(blob, SEED, zero_binding_enabled=True, device=device)
    kappa_info = zsf.compute_kappa(n0, fit_data)
    with torch.no_grad():
        n0.kappa.fill_(float(kappa_info["kappa"]))
    n0_state = {k: v.detach().cpu() for k, v in n0.state_dict().items()}
    mismatch = sorted(set(source_state) ^ set(n0_state))
    value_mismatch = [
        k for k in sorted(set(source_state) & set(n0_state)) if not torch.equal(source_state[k], n0_state[k])
    ]
    saved_init = torch.load(SOURCE_RESULTS / "S_M_init_state.pt", map_location="cpu", weights_only=False)
    saved_mismatch = [
        k for k in sorted(set(saved_init) & set(n0_state)) if not torch.equal(saved_init[k], n0_state[k])
    ]
    checks["init_identity"] = {
        "source_factory_vs_n0_key_mismatch": mismatch,
        "source_factory_vs_n0_value_mismatch": value_mismatch,
        "released_S_M_init_value_mismatch": saved_mismatch,
        "released_S_M_init_missing_keys": sorted(set(saved_init) ^ set(n0_state)),
        "kappa": kappa_info,
        "pass": bool(not mismatch and not value_mismatch and not saved_mismatch),
    }

    # 2. switch off reproduces the S_M forward; switch on zeroes both slots
    batch = zsf.make_batch(fit_data, list(range(32)), torch.as_tensor(y_fit[:32], dtype=torch.float32), device)
    with torch.no_grad():
        source.eval()
        n0.eval()
        n0.zero_binding_enabled = False
        p_off, _ = n0(batch, mask=cm.C6_MASK, return_aux=True)
        p_src, _ = source(batch, mask=cm.C6_MASK, return_aux=True)
        n0.zero_binding_enabled = True
        n0._probe = {}
        p_on, _ = n0(batch, mask=cm.C6_MASK, return_aux=True)
        probe = dict(n0._probe)
        n0._probe = None
    checks["switch"] = {
        "off_vs_source_max_abs": float((p_off - p_src).abs().max()),
        "on_node_slots_absmax": float(probe["node_slots_postzero"].abs().max()),
        "on_edge_slots_absmax": float(probe["edge_slots_postzero"].abs().max()),
        "prezero_finite": bool(torch.isfinite(probe["node_flat_prezero"]).all() and torch.isfinite(probe["edge_flat_prezero"]).all()),
        "pass": bool(
            float((p_off - p_src).abs().max()) <= COMPRESS_TOL
            and float(probe["node_slots_postzero"].abs().max()) == 0.0
            and float(probe["edge_slots_postzero"].abs().max()) == 0.0
        ),
    }

    # 3. label shuffle does not change the forward
    with torch.no_grad():
        b2 = batch.clone()
        b2.y = torch.randn_like(b2.y)
        p_shuf, _ = n0(b2, mask=cm.C6_MASK, return_aux=True)
    checks["label_independence"] = {
        "max_abs": float((p_on - p_shuf).abs().max()),
        "tolerance": 1.0e-5,
        "pass": bool(float((p_on - p_shuf).abs().max()) <= 1e-5),
    }

    # 4. one backward: task gradient identically zero on the binding params,
    #    reconstruction gradient alive on D, downstream task grads alive
    n0.train()
    b16 = zsf.make_batch(fit_data, list(range(16)), torch.as_tensor(y_fit[:16], dtype=torch.float32), device)
    pred, aux = n0(b16, mask=cm.C6_MASK, return_aux=True)
    task = F.l1_loss(pred.view(-1), b16.y.view(-1))
    rec = n0.reconstruction_loss(aux["phi"], aux["coord"])
    n0.zero_grad(set_to_none=True)
    (task + H1_LAMBDA * rec).backward()
    grads = {name: None if p.grad is None else p.grad.detach().clone() for name, p in n0.named_parameters()}
    zero_exact = {
        name: bool(g is not None and float(g.abs().max()) == 0.0)
        for name in ("W_A_S", "W_A_C", "W_E_S", "W_E_C")
        for g in [grads[name]]
    }
    n_with_grad = int(sum(p.numel() for p in n0.parameters() if p.grad is not None))
    n_one = int(sum(p.numel() for p in n0.parameters() if p.grad is not None and float(p.grad.abs().max()) > 0.0))
    reader_names = [f"reader.{name}" for name, _ in n0.reader.named_parameters()]
    bridge_names = [f"local_dictionary_bridge.{name}" for name, _ in n0.local_dictionary_bridge.named_parameters()]
    checks["gradients"] = {
        "binding_grads_exactly_zero": zero_exact,
        "D_grad_absmax": float(grads["D"].abs().max()) if grads["D"] is not None else None,
        "node_encoder0_bias_grad_absmax": float(grads["node_encoder.0.bias"].abs().max()),
        "fusion0_weight_grad_absmax": float(grads["fusion.0.weight"].abs().max()),
        "bridge_first_param": bridge_names[0],
        "bridge_first_param_grad_absmax": float(grads[bridge_names[0]].abs().max()),
        "reader_first_param": reader_names[0],
        "reader_grad_absmax": float(grads[reader_names[0]].abs().max()),
        "parameters_with_grad": n_with_grad,
        "parameters_with_nonzero_grad": n_one,
        "total_trainable_parameters": zsf.trainable_parameter_count(n0),
        "pass": bool(
            all(zero_exact.values())
            and grads["D"] is not None
            and float(grads["D"].abs().max()) > 0
            and n_with_grad == zsf.FULL_PARAMETERS
        ),
    }
    n0.zero_grad(set_to_none=True)

    # 5. the smoke itself: <= max_steps real optimizer steps, state discarded
    smoke = train_n0(device=device, out_dir=out_dir, epochs=1, max_steps=int(max_steps), log=None)
    checks["smoke_steps"] = {
        "steps_done": smoke["steps_done"],
        "data_stream_sha256": smoke["data_stream_sha256"],
        "last_trainable_hash": smoke["last_trainable_hash"],
        "pass": bool(smoke["steps_done"] == int(max_steps) and smoke["last_trainable_hash"] != smoke["init_trainable_hash"]),
    }
    # clip-range witness on the first smoke step: norm returned before clipping
    checks["clip"] = {
        "grad_clip": GRAD_CLIP,
        "first_epoch_mean_grad_norm": float(smoke["curve"][0]["grad_norm"]),
        "first_epoch_clip_fraction": float(smoke["curve"][0]["clip_fraction"]),
        "note": "clip_grad_norm_ over model.parameters() before optimizer.step(), the source order",
    }
    checks["mechanism_ok"] = bool(
        checks["init_identity"]["pass"]
        and checks["switch"]["pass"]
        and checks["label_independence"]["pass"]
        and checks["gradients"]["pass"]
        and checks["smoke_steps"]["pass"]
    )
    write_json(out_dir / "smoke" / "smoke.json", checks)
    log(f"[smoke] mechanism_ok={checks['mechanism_ok']}")
    return checks


# ---------------------------------------------------------------------------
# phase 0 — frozen protocol before any fitting
# ---------------------------------------------------------------------------


def phase0(*, log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    blob, decomp, _train_data, fit_data, _dev_data, fit_idx, dev_idx, y = _load_source_fit_dev()
    k = np.asarray(decomp["k"], np.int64)
    fit_hash = hashlib.sha256(fit_idx.tobytes()).hexdigest()
    dev_hash = hashlib.sha256(dev_idx.tobytes()).hexdigest()
    counts = {
        "k0": int((k[dev_idx] == 0).sum()),
        "k-1": int((k[dev_idx] == -1).sum()),
        "kle-2": int((k[dev_idx] <= -2).sum()),
    }
    seed_everything(SEED)
    n0 = build_zero_model(blob, SEED, zero_binding_enabled=True)
    kappa_info = zsf.compute_kappa(n0, fit_data)
    sched = schedule_hash(len(fit_data), EPOCHS, TRAIN_SHUFFLE_OFFSET)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "source_branch": SOURCE_BRANCH,
        "source_branch_head": SOURCE_COMMIT,
        "source_deployed_commit": SOURCE_DEPLOYED_COMMIT,
        "question": "does fixing both binding slots to zero from the first training step preserve S_M G0 generalisation?",
        "arms": ["S_J", "S_M", "N0"],
        "new_formal_trajectories": ["N0_seed0"],
        "seed": SEED,
        "scale_seed": SCALE_SEED,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE,
        "h1_lambda": H1_LAMBDA,
        "soup_epochs": list(SOUP_EPOCHS),
        "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
        "full_parameters": zsf.FULL_PARAMETERS,
        "fusion_input_reduced": N0_FUSION_IN,
        "fusion_first_layer_weight_saving": FUSION_FIRST_LAYER_SAVING,
        "kappa": kappa_info,
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "delta": DELTA},
        "group_names": GROUP_NAMES,
        "primary_endpoint": "dev G0 (k=0) cal MAE",
        "secondary_endpoint": "dev overall cal MAE",
        "split": {
            "n_fit": int(len(fit_idx)),
            "n_dev": int(len(dev_idx)),
            "fit_idx_sha256": fit_hash,
            "dev_idx_sha256": dev_hash,
            "expected_fit_idx_sha256": "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea",
            "expected_dev_idx_sha256": "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376",
            "dev_group_counts": counts,
            "dev_group_counts_expected": {"k0": 1926, "k-1": 65, "kle-2": 9},
            "pass": bool(
                fit_hash == "165e87ef4398ba8ef57411c2f118c4cca4ea74007f2485611b84f0c09bbdd9ea"
                and dev_hash == "fb8b78063e7c8a5c759bfa7d553738068cee1738a2d67b210e5d9dc492331376"
                and counts == {"k0": 1926, "k-1": 65, "kle-2": 9}
            ),
        },
        "schedule": sched,
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    if not protocol["split"]["pass"]:
        raise RuntimeError("frozen split mismatch")
    write_json(RESULTS_DIR / "protocol.json", protocol)
    log(f"[phase0] split_ok=True kappa={kappa_info['kappa']:.8f} schedule={sched['full_schedule_sha256'][:16]}")
    return protocol


# ---------------------------------------------------------------------------
# analysis (local, after the remote trajectory is pulled)
# ---------------------------------------------------------------------------


def analyze(*, device: torch.device = torch.device("cpu"), log: Any = print) -> dict[str, Any]:
    torch.set_num_threads(8)
    blob, decomp, _train_data, fit_data, dev_data, fit_idx, dev_idx, y = _load_source_fit_dev()
    k = np.asarray(decomp["k"], np.int64)
    gid = np.asarray(decomp["gid"], np.int64)
    y_fit = y[fit_idx]
    y_dev = y[dev_idx]
    k_dev = k[dev_idx]
    counts = {"k0": int((k_dev == 0).sum()), "k-1": int((k_dev == -1).sum()), "kle-2": int((k_dev <= -2).sum())}
    if counts != {"k0": 1926, "k-1": 65, "kle-2": 9}:
        raise RuntimeError(f"dev group counts changed: {counts}")
    dev_pos = {key: np.where(group_mask(k_dev, key))[0] for key in GROUP_KEYS}
    g0 = dev_pos["k0"]

    # --- N0 fit prediction released by the remote run -------------------------
    n0_pred = np.load(RESULTS_DIR / "N0_predictions.npz")
    n0_fit_raw = n0_pred["fit_raw"].astype(np.float64)
    n0_json = json.loads((RESULTS_DIR / "N0.json").read_text())
    b_n0 = float(n0_json["calibration"]["b"])
    b_n0_recomputed = float(np.median(y_fit - n0_fit_raw))
    n0_gpu_fit_raw = n0_fit_raw.copy()

    # --- N0 raw soup: local replay of fit+dev --------------------------------
    n0_model = build_zero_model(blob, SEED, zero_binding_enabled=True, device=device)
    n0_state = torch.load(RESULTS_DIR / "N0_raw_soup_state.pt", map_location=device, weights_only=False)
    if zsf.state_hash(n0_state) != n0_json["soup_state_sha256"]:
        raise RuntimeError("N0 raw soup state hash mismatch vs N0.json")
    n0_model.load_state_dict({key: value.to(device) for key, value in n0_state.items()})
    n0_model.eval()
    n0_fit_local, _ = zsf.predict(n0_model, fit_data, torch.as_tensor(y_fit, dtype=torch.float32), device)
    n0_dev_raw, _ = zsf.predict(n0_model, dev_data, torch.as_tensor(y_dev, dtype=torch.float32), device)

    # --- S_J / S_M raw replay (dev) and released predictions ------------------
    arm_raw_dev: dict[str, np.ndarray] = {}
    arm_raw_fit: dict[str, np.ndarray] = {}
    replay_check: dict[str, Any] = {}
    pred_csv = np.genfromtxt(SOURCE_RESULTS / "predictions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    for arm in ("S_J", "S_M"):
        m = zsf.build_factorial_model(blob, zsf.SEED, zsf.CODING_OF[arm], zsf.BINDING_OF[arm]).to(device)
        m.load_state_dict(load_source_state(arm, device))
        m.eval()
        raw_dev, _ = zsf.predict(m, dev_data, torch.as_tensor(y_dev, dtype=torch.float32), device)
        arm_raw_dev[arm] = raw_dev
        arm_raw_fit[arm] = np.asarray([float(v) for v in pred_csv[f"{arm}_raw"][: len(fit_idx)]], np.float64)
        csv_dev = np.asarray([float(v) for v in pred_csv[f"{arm}_raw"][-len(dev_idx):]], np.float64)
        replay_check[arm] = {
            "dev_replay_vs_predictions_csv_max_abs": float(np.max(np.abs(raw_dev - csv_dev))),
            "released_soup_sha256": zsf.state_hash({k2: v for k2, v in load_source_state(arm, device).items()}),
            "bias_used": SOURCE_ARM_BIAS[arm],
        }
    bias = {"S_J": SOURCE_ARM_BIAS["S_J"], "S_M": SOURCE_ARM_BIAS["S_M"], "N0": b_n0}
    raw_dev = {"S_J": arm_raw_dev["S_J"], "S_M": arm_raw_dev["S_M"], "N0": n0_dev_raw}
    raw_fit = {"S_J": arm_raw_fit["S_J"], "S_M": arm_raw_fit["S_M"], "N0": n0_fit_local}
    cal_dev = {arm: raw_dev[arm] + bias[arm] for arm in raw_dev}
    cal_fit = {arm: raw_fit[arm] + bias[arm] for arm in raw_fit}

    def mae(values: np.ndarray) -> float:
        return float(np.mean(np.abs(values)))

    def gains_cal(pred_map: dict[str, np.ndarray], mask=None) -> dict[str, float]:
        l = {}
        for arm in pred_map:
            err = pred_map[arm] - y_dev
            l[arm] = float(np.mean(np.abs(err[mask]))) if mask is not None else float(np.mean(np.abs(err)))
        return {
            "gain_N0_vs_SM": l["S_M"] - l["N0"],
            "gain_N0_vs_SJ": l["S_J"] - l["N0"],
            "gain_SM_vs_SJ": l["S_J"] - l["S_M"],
            "levels": l,
        }

    main = {
        "fit_cal_mae": {arm: mae(cal_fit[arm] - y_fit) for arm in raw_fit},
        "fit_raw_mae": {arm: mae(raw_fit[arm] - y_fit) for arm in raw_fit},
        "dev_cal_mae": {arm: mae(cal_dev[arm] - y_dev) for arm in raw_dev},
        "dev_raw_mae": {arm: mae(raw_dev[arm] - y_dev) for arm in raw_dev},
        "dev_cal_mae_g0": {arm: float(np.mean(np.abs((cal_dev[arm] - y_dev)[g0]))) for arm in raw_dev},
        "dev_raw_mae_g0": {arm: float(np.mean(np.abs((raw_dev[arm] - y_dev)[g0]))) for arm in raw_dev},
        "bias": bias,
        "fit_dev_gap_cal": {arm: mae(cal_dev[arm] - y_dev) - mae(cal_fit[arm] - y_fit) for arm in raw_dev},
        "g0_cal": gains_cal(cal_dev, g0),
        "g0_raw": gains_cal(raw_dev, g0),
        "overall_cal": gains_cal(cal_dev),
        "overall_raw": gains_cal(raw_dev),
    }

    # group table
    group_rows = []
    for key in GROUP_KEYS:
        mask = dev_pos[key]
        n = int(mask.size)
        for arm in ("S_J", "S_M", "N0"):
            err_cal = cal_dev[arm][mask] - y_dev[mask]
            err_raw = raw_dev[arm][mask] - y_dev[mask]
            group_rows.append(
                {
                    "group": GROUP_NAMES[key],
                    "n": n,
                    "arm": arm,
                    "cal_mae": float(np.mean(np.abs(err_cal))),
                    "raw_mae": float(np.mean(np.abs(err_raw))),
                    "cal_residual_y_minus_pred": float(np.mean(y_dev[mask] - cal_dev[arm][mask])),
                    "cal_contribution": float(np.abs(err_cal).sum() / len(y_dev)),
                    "raw_contribution": float(np.abs(err_raw).sum() / len(y_dev)),
                }
            )
    contribution_identity = {}
    for arm in ("S_J", "S_M", "N0"):
        total = float(sum(r["cal_contribution"] for r in group_rows if r["arm"] == arm))
        contribution_identity[arm] = {
            "contribution_sum": total,
            "overall_mae": main["dev_cal_mae"][arm],
            "abs_diff": abs(total - main["dev_cal_mae"][arm]),
        }

    # --- paired bootstrap (identical indices across arms and raw/cal) --------
    rng = np.random.default_rng(BOOT_SEED)
    samplers = []
    for _ in range(N_BOOT):
        idx_g0 = g0[rng.integers(0, len(g0), size=len(g0))]
        parts = [dev_pos[key][rng.integers(0, len(dev_pos[key]), size=len(dev_pos[key]))] for key in GROUP_KEYS]
        samplers.append((idx_g0, np.concatenate(parts)))

    def boot_mae(pred, idx):
        return float(np.mean(np.abs(pred[idx] - y_dev[idx])))

    boot = {name: {"g0": [], "overall": []} for name in ("gain_N0_vs_SM", "gain_N0_vs_SJ", "gain_SM_vs_SJ")}
    boot_raw = {name: {"g0": [], "overall": []} for name in ("gain_N0_vs_SM", "gain_N0_vs_SJ", "gain_SM_vs_SJ")}
    for idx_g0, idx_ov in samplers:
        for scope, idx in (("g0", idx_g0), ("overall", idx_ov)):
            lev = {arm: boot_mae(cal_dev[arm], idx) for arm in ("S_J", "S_M", "N0")}
            raws = {arm: boot_mae(raw_dev[arm], idx) for arm in ("S_J", "S_M", "N0")}
            boot["gain_N0_vs_SM"][scope].append(lev["S_M"] - lev["N0"])
            boot["gain_N0_vs_SJ"][scope].append(lev["S_J"] - lev["N0"])
            boot["gain_SM_vs_SJ"][scope].append(lev["S_J"] - lev["S_M"])
            boot_raw["gain_N0_vs_SM"][scope].append(raws["S_M"] - raws["N0"])
            boot_raw["gain_N0_vs_SJ"][scope].append(raws["S_J"] - raws["N0"])
            boot_raw["gain_SM_vs_SJ"][scope].append(raws["S_J"] - raws["S_M"])

    points = {
        "gain_N0_vs_SM": {
            "g0_cal": main["g0_cal"]["gain_N0_vs_SM"],
            "overall_cal": main["overall_cal"]["gain_N0_vs_SM"],
            "g0_raw": main["g0_raw"]["gain_N0_vs_SM"],
            "overall_raw": main["overall_raw"]["gain_N0_vs_SM"],
        },
        "gain_N0_vs_SJ": {
            "g0_cal": main["g0_cal"]["gain_N0_vs_SJ"],
            "overall_cal": main["overall_cal"]["gain_N0_vs_SJ"],
            "g0_raw": main["g0_raw"]["gain_N0_vs_SJ"],
            "overall_raw": main["overall_raw"]["gain_N0_vs_SJ"],
        },
        "gain_SM_vs_SJ": {
            "g0_cal": main["g0_cal"]["gain_SM_vs_SJ"],
            "overall_cal": main["overall_cal"]["gain_SM_vs_SJ"],
            "g0_raw": main["g0_raw"]["gain_SM_vs_SJ"],
            "overall_raw": main["overall_raw"]["gain_SM_vs_SJ"],
        },
    }
    boot_summary: dict[str, Any] = {}
    for name in ("gain_N0_vs_SM", "gain_N0_vs_SJ", "gain_SM_vs_SJ"):
        boot_summary[name] = {}
        for scope_key in ("g0", "overall"):
            for kind, samples in (("cal", boot), ("raw", boot_raw)):
                arr = np.asarray(samples[name][scope_key], np.float64)
                lo, hi = np.quantile(arr, [0.025, 0.975])
                boot_summary[name][f"{scope_key}_{kind}"] = {
                    "point": points[name][f"{scope_key}_{kind}"],
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "n_boot": N_BOOT,
                    "seed": BOOT_SEED,
                }
    # preregistered interpretation row for the primary contrast
    primary = boot_summary["gain_N0_vs_SM"]["g0_cal"]
    overall_worsening = main["overall_cal"]["gain_N0_vs_SM"] * -1.0
    if primary["point"] >= DELTA and primary["ci_lo"] > 0 and overall_worsening <= 0.001:
        primary_verdict = "positive_purchase_signal: N0 is a new single-seed performance candidate"
    elif abs(primary["point"]) <= 0.001 and primary["ci_lo"] >= -DELTA and primary["ci_hi"] <= DELTA and overall_worsening <= 0.001:
        primary_verdict = "close_to_S_M: early binding path not necessary beyond practical threshold (this seed/dev)"
    elif primary["point"] <= -DELTA and primary["ci_hi"] < 0:
        primary_verdict = "negative_purchase_signal: early binding path carries a training-time role; not attributable to dictionary transfer alone"
    elif (primary["point"] > 0) != (points["gain_N0_vs_SM"]["g0_raw"] > 0):
        primary_verdict = "calibration_sensitive_or_inconclusive: raw/cal directions differ"
    else:
        primary_verdict = "inconclusive"

    # bootstrap witnesses
    base = cal_dev["S_J"]
    identical = float(np.mean(np.abs(base - y_dev)) - np.mean(np.abs(base - y_dev)))
    gain_sj_sm = float(np.mean(np.abs(cal_dev["S_M"] - y_dev)) - np.mean(np.abs(cal_dev["S_J"] - y_dev)))
    gain_sm_sj = -gain_sj_sm
    shifted = float(np.mean(np.abs(base + 0.5 - y_dev)) - np.mean(np.abs(base - y_dev)))
    witnesses = {
        "identical_predictions_gain_zero": identical,
        "identical_is_zero": bool(identical == 0.0),
        "swap_sign_flip": {"gain_SJ_to_SM": gain_sj_sm, "gain_SM_to_SJ": gain_sm_sj, "sign_flipped": bool(abs(gain_sj_sm + gain_sm_sj) < 1e-12)},
        "constant_shift_bound": {"shift": 0.5, "shifted_gain": shifted, "within_bound": bool(abs(shifted) <= 0.5 + 1e-12)},
    }

    # --- N0 deployment compression (same algebra as Phase A) -----------------
    n0_model.eval()
    n0_deploy, n0_shrink = build_reduced_from_soup(n0_model, device=device)
    n0_deploy.eval()
    n0_dev_deploy, _ = zsf.predict(n0_deploy, dev_data, torch.as_tensor(y_dev, dtype=torch.float32), device)
    deploy_diff = float(np.max(np.abs(n0_dev_deploy - n0_dev_raw)))
    torch.save(n0_deploy.state_dict(), RESULTS_DIR / "N0_deploy_state.pt")
    n0_deploy_meta = {
        **{kk: vv for kk, vv in n0_shrink.items() if kk != "c_slot"},
        "dev_max_abs_diff_vs_N0": deploy_diff,
        "tolerance": COMPRESS_TOL,
        "pass": bool(deploy_diff <= COMPRESS_TOL),
        "independent_load_keys": sorted(build_deploy_model(n0_deploy.state_dict()).state_dict().keys()),
    }
    write_json(RESULTS_DIR / "N0_deploy_meta.json", n0_deploy_meta)

    rows = []
    for split, idx, yv, kv, raw_map, cal_map in (
        ("fit", fit_idx, y[fit_idx], k[fit_idx], raw_fit, cal_fit),
        ("dev", dev_idx, y_dev, k_dev, raw_dev, cal_dev),
    ):
        for pos in range(len(idx)):
            record = {
                "split": split,
                "global_index": int(idx[pos]),
                "canonical_group_id": int(gid[idx[pos]]),
                "y": float(yv[pos]),
                "k": int(kv[pos]),
                "group": group_of(int(kv[pos])),
            }
            for arm in ("S_J", "S_M", "N0"):
                record[f"{arm}_raw"] = float(raw_map[arm][pos])
                record[f"{arm}_cal"] = float(cal_map[arm][pos])
            rows.append(record)
    import pandas as pd

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "predictions.csv", index=False)
    pd.DataFrame(group_rows).to_csv(RESULTS_DIR / "group_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": arm,
                "g0_cal_mae": main["dev_cal_mae_g0"][arm],
                "overall_cal_mae": main["dev_cal_mae"][arm],
                "overall_raw_mae": main["dev_raw_mae"][arm],
                "fit_cal_mae": main["fit_cal_mae"][arm],
                "fit_raw_mae": main["fit_raw_mae"][arm],
                "bias": bias[arm],
                "fit_dev_gap_cal": main["fit_dev_gap_cal"][arm],
            }
            for arm in ("S_J", "S_M", "N0")
        ]
    ).to_csv(RESULTS_DIR / "main_table.csv", index=False)

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "dev_group_counts": counts,
        "main": main,
        "bootstrap": boot_summary,
        "bootstrap_witness": witnesses,
        "contribution_identity": contribution_identity,
        "n0_replay": {
            "source_fit_raw_vs_local_fit_raw_max_abs": float(np.max(np.abs(n0_gpu_fit_raw - n0_fit_local))),
            "b_released": b_n0,
            "b_recomputed": b_n0_recomputed,
            "b_abs_diff": abs(b_n0 - b_n0_recomputed),
        },
        "replay_check_vs_old": replay_check,
        "primary_verdict": primary_verdict,
        "primary_overall_worsening": overall_worsening,
        "n0_deploy": n0_deploy_meta,
    }
    write_json(RESULTS_DIR / "summary.json", summary)
    log(json.dumps(jsonable({"main": main, "bootstrap": boot_summary, "verdict": primary_verdict}), indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZINC zero-binding baseline seed0 v1")
    parser.add_argument("--mode", required=True, choices=("phase0", "phaseA_check", "phaseA_export", "smoke", "train", "analyze"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)

    if args.mode == "phase0":
        phase0()
        return 0
    if args.mode == "phaseA_check":
        checks = phaseA_check(device=zsf.resolve_device(args.device))
        return 0 if checks["pass"] else 1
    if args.mode == "phaseA_export":
        report = phaseA_export(device=zsf.resolve_device(args.device))
        return 0 if report["pass"] else 1
    if args.mode == "smoke":
        checks = run_smoke(device=zsf.resolve_device(args.device), out_dir=out_dir)
        return 0 if checks["mechanism_ok"] else 1
    if args.mode == "train":
        train_n0(device=zsf.resolve_device(args.device), out_dir=out_dir, epochs=int(args.epochs))
        return 0
    if args.mode == "analyze":
        analyze(device=zsf.resolve_device(args.device))
        return 0
    return 2


H1_LAMBDA = float(cm.H1_LAMBDA)

if __name__ == "__main__":
    raise SystemExit(main())
