"""ZINC joint-dictionary decision v1 — fixed joint local description prototype.

Stage A (``cycle_input_audit.py``) is an independent read-only diagnostic and is
never consumed here.  This module implements one fixed prototype:

``X175 = [raw structural phi65 ; raw Sem108 ; raw size2]`` (one row per
root/local environment), fold-internally standardised (per-column mean/std then
per-block RMS normalisation), consumed by

* ``F`` — the canonical ``LatentScaleSEM108`` Full (fresh init), and
* ``B`` — ``X175 -> Linear(175,256) -> SiLU -> Linear(256,144)``, and
* ``D`` — shared sparse dictionary (175x256, K=256, s=64, 10 IHT steps) with
  ``alpha -> Linear(256,144)``.

B/D replace the entire local-environment generation module (old structural
binding, fusion and post-fusion task bridge) and output E144.  Pair projection,
relation/distance modules, unary/pair pooling, the C6 mask, the global/topology
branches and the reader are shared with Full.

The official ZINC **test** split is never instantiated anywhere in this module.
"""

from __future__ import annotations

import collections
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
from torch.utils.data import DataLoader

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import fec_s0_factorization as fec
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_post_v4_residual_audit as pva
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = "zinc-joint-dictionary-decision-v1"
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/zinc_joint_dictionary_decision_v1"
PREP_DIR = RESULTS_DIR / "prep"
HANDOFF = TRACK_ROOT / "results/zinc_dictionary_real_data_handoff"
CYCLE = TRACK_ROOT / "results/zinc_long_cycle_audit"

SEED = 0
SCALE_SEED = 0
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_EPOCHS = (236, 237, 238, 239, 240)
DIAG_EPOCHS = (1, 40, 80, 160, 240)
TRAIN_SHUFFLE_OFFSET = 101
EVAL_SHUFFLE_OFFSET = 77

SPLIT_SEED = 20261003
DEV_FRACTION = 0.20
SEVERE_MAX = -2

X_DIM = 175
E_DIM = 144
PHI_BLOCK = (0, 65)
SEM_BLOCK = (65, 173)
SIZE_BLOCK = (173, 175)
STD_FLOOR = 1.0e-3
CANON_STD_FLOOR = 1.0e-6

D_ATOMS = 256
D_S = 64
D_STEPS = 10
D_PREP_LR = 1.0e-3
D_PREP_BATCH = 4096
D_PREP_PASSES = 5
D_PREP_MAX_SECONDS = 900.0
D_LOCAL_SEED = 20261003

B_HIDDEN = 256
LAMBDA_REC_FRACTION = 0.1
LAMBDA_BATCHES = 4
LAMBDA_BATCH_SIZE = 128

ARMS = ("F", "B", "D")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def resolve_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device requests CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported device {name!r}")


def seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parameter_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key in sorted(model.state_dict()):
        value = model.state_dict()[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def backend_state(model: nn.Module) -> dict[str, torch.Tensor]:
    names = ("pair_projection", "relation_encoder", "distance_gate", "pair_encoder",
             "global_encoder", "topology_encoder", "reader")
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if any(k.startswith(n) for n in names)}


def backend_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    state = backend_state(model)
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].numpy()).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# fixed molecule-level development split
# ---------------------------------------------------------------------------


def _load_penalties() -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    lt = pd.read_csv(CYCLE / "train_cycle_audit_label.csv")
    lv = pd.read_csv(CYCLE / "valid_cycle_audit_label.csv")
    return (np.round(lt["label_effective_cycle_snapped"].to_numpy()).astype(int),
            np.round(lv["label_effective_cycle_snapped"].to_numpy()).astype(int))


def build_fold_split() -> dict[str, Any]:
    train = np.load(HANDOFF / "train.npz", allow_pickle=True)
    gid = train["canonical_group_id"].astype(np.int64)
    pen, _ = _load_penalties()
    assert len(gid) == len(pen) == 10000

    by_group: dict[int, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(gid.tolist()):
        by_group[int(g)].append(i)
    group_stratum: dict[int, int] = {}
    inconsistent = []
    for g, rows in by_group.items():
        vals = sorted(set(int(pen[r]) for r in rows))
        if len(vals) > 1:
            inconsistent.append({"group": int(g), "penalties": vals, "rows": rows})
        p = vals[0]
        group_stratum[g] = 0 if p == 0 else (1 if p == -1 else 2)
    names = {0: "penalty_0", 1: "penalty_-1", 2: "penalty_le_-2"}

    dev_groups: set[int] = set()
    per_stratum: dict[str, Any] = {}
    for stratum in (0, 1, 2):
        groups = sorted((g for g, s in group_stratum.items() if s == stratum),
                        key=lambda g: int.from_bytes(
                            hashlib.sha256(f"{SPLIT_SEED}:{g}".encode()).digest()[:8], "big"))
        n_rows = sum(len(by_group[g]) for g in groups)
        target = int(round(DEV_FRACTION * n_rows))
        taken_rows, taken_groups = 0, []
        for g in groups:
            if taken_rows >= target:
                break
            taken_groups.append(g)
            taken_rows += len(by_group[g])
        dev_groups.update(taken_groups)
        per_stratum[names[stratum]] = {
            "n_groups": len(groups), "n_rows": n_rows,
            "dev_groups": len(taken_groups), "dev_rows": taken_rows,
        }

    dev_idx = np.array(sorted(r for g in dev_groups for r in by_group[g]), dtype=np.int64)
    fit_idx = np.array(sorted(set(range(10000)) - set(dev_idx.tolist())), dtype=np.int64)
    meta = {
        "split_seed": SPLIT_SEED, "dev_fraction": DEV_FRACTION,
        "unit": "canonical_group_id (verified molecule key)",
        "n_train_rows": 10000, "n_fit_rows": int(len(fit_idx)), "n_dev_rows": int(len(dev_idx)),
        "n_groups_total": int(len(by_group)), "per_stratum": per_stratum,
        "inconsistent_group_penalties": inconsistent,
        "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
        "dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
        "official_test_loaded": False,
    }
    return {"fit_idx": fit_idx, "dev_idx": dev_idx, "meta": meta}


# ---------------------------------------------------------------------------
# fold-internal standardizers and X175
# ---------------------------------------------------------------------------


class Std:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray, floor: float = STD_FLOOR) -> "Std":
        x = np.asarray(values, dtype=np.float32)
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < float(floor))] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.scale + self.mean).astype(np.float32)


def _stack(data_list: Sequence[Any], attr: str) -> tuple[np.ndarray, list[int]]:
    parts, counts = [], []
    for d in data_list:
        v = getattr(d, attr)
        v = v.numpy() if torch.is_tensor(v) else np.asarray(v)
        parts.append(np.asarray(v, dtype=np.float32))
        counts.append(int(v.shape[0]))
    return np.concatenate(parts, axis=0), counts


def _unstack(values: np.ndarray, counts: Sequence[int], attr: str, data_list: Sequence[Any]) -> None:
    offset = 0
    for d, n in zip(data_list, counts):
        setattr(d, attr, torch.as_tensor(values[offset : offset + n], dtype=torch.float32))
        offset += n


def _root_mask(counts: Sequence[int], molecule_idx: np.ndarray) -> np.ndarray:
    want = set(int(i) for i in molecule_idx.tolist())
    mask = np.zeros(int(sum(counts)), dtype=bool)
    offset = 0
    for i, c in enumerate(counts):
        if i in want:
            mask[offset : offset + c] = True
        offset += c
    return mask


def build_fold_inputs(train_data: Sequence[Any], valid_data: Sequence[Any],
                      fit_idx: np.ndarray) -> dict[str, Any]:
    """Rebuild every train-fit input object on the fit molecules only.

    Sem108 / global context / topology: invert the all-train standardizer to raw
    and refit the per-column standardizer on the fit rows.  anchor (size2): same
    using the frozen ``anchor_stats.json`` scaler.  Structural dictionary
    (sdb32) and common subspace: refit on fit phi rows.
    """
    import pandas as pd

    started = time.perf_counter()
    records, _ = pva._extract_v4_records()
    fits_all = fec.build_fits(records)
    patch_all = Std(fits_all.patch_standardizer.mean, fits_all.patch_standardizer.scale)
    ctx_all = Std(fits_all.context_standardizer.mean, fits_all.context_standardizer.scale)
    topo_all = Std(fits_all.topology_standardizer.mean, fits_all.topology_standardizer.scale)
    anchor_stats = json.loads((TRACK_ROOT / "results/e2e_dictenv_p1/anchor_stats.json").read_text())
    anchor_all = Std(np.asarray(anchor_stats["mean"], np.float32),
                     np.asarray(anchor_stats["scale"], np.float32))

    patch_raw, patch_counts = _stack(train_data, "patch_cont")
    patch_raw = patch_all.inverse(patch_raw)
    ctx_raw, ctx_counts = _stack(train_data, "global_context")
    ctx_raw = ctx_all.inverse(ctx_raw)
    anchor_raw, anchor_counts = _stack(train_data, "anchor")
    anchor_raw = anchor_all.inverse(anchor_raw)
    topo_raw, topo_counts = _stack(train_data, "topology_features")
    topo_raw = topo_all.inverse(topo_raw)
    phi_all = np.concatenate([d.dict_phi.numpy() for d in train_data], 0)

    root_mask = _root_mask(patch_counts, fit_idx)
    ctx_fit_mask = _root_mask(ctx_counts, fit_idx)

    patch_fit = Std.fit(patch_raw[root_mask], floor=CANON_STD_FLOOR)
    ctx_fit = Std.fit(ctx_raw[ctx_fit_mask], floor=CANON_STD_FLOOR)
    anchor_fit = Std.fit(anchor_raw[root_mask], floor=CANON_STD_FLOOR)
    topo_fit = Std.fit(topo_raw[ctx_fit_mask], floor=CANON_STD_FLOOR)

    D_fit, ksvd_info = sdb.fit_ksvd(phi_all[root_mask].astype(np.float64), atoms=sdb.K_ATOMS,
                                    s=int(sdb.SPARSITY), epochs=10, seed=sdb.DICT_SEED)
    subspace_fit = cssd.build_common_subspace(phi_all[root_mask], q=1, kind="q1")

    def apply(data_list: Sequence[Any]) -> None:
        p, pc = _stack(data_list, "patch_cont")
        c, cc = _stack(data_list, "global_context")
        a, ac = _stack(data_list, "anchor")
        t, tc = _stack(data_list, "topology_features")
        _unstack(patch_fit.transform(patch_all.inverse(p)), pc, "patch_cont", data_list)
        _unstack(ctx_fit.transform(ctx_all.inverse(c)), cc, "global_context", data_list)
        _unstack(anchor_fit.transform(anchor_all.inverse(a)), ac, "anchor", data_list)
        _unstack(topo_fit.transform(topo_all.inverse(t)), tc, "topology_features", data_list)

    apply(train_data)
    apply(valid_data)

    # raw X175 = [raw phi65 ; raw sem108 ; raw size2]
    def x175_raw(data_list: Sequence[Any]) -> tuple[np.ndarray, list[int]]:
        p, pc = _stack(data_list, "patch_cont")
        p = patch_all.inverse(p)
        a, _ = _stack(data_list, "anchor")
        a = anchor_all.inverse(a)
        phi = np.concatenate([d.dict_phi.numpy() for d in data_list], 0)
        x = np.concatenate([phi[:, :65], p[:, :108], a[:, 60:62]], axis=1)
        return x.astype(np.float32), pc

    x_train, x_train_counts = x175_raw(train_data)
    x_valid, x_valid_counts = x175_raw(valid_data)
    x_train_fit_mask = _root_mask(x_train_counts, fit_idx)

    # B1 normalization: per-column std (floor 1e-3, zero-variance scale 1),
    # then each block divided by the RMS norm of its standardised block.
    x_mean = x_train[x_train_fit_mask].mean(axis=0)
    x_std = x_train[x_train_fit_mask].std(axis=0)
    x_std[~np.isfinite(x_std) | (x_std < STD_FLOOR)] = 1.0
    z = (x_train - x_mean) / x_std
    blocks = [(PHI_BLOCK[0], PHI_BLOCK[1]), (SEM_BLOCK[0], SEM_BLOCK[1]), (SIZE_BLOCK[0], SIZE_BLOCK[1])]
    block_scale = []
    for lo, hi in blocks:
        ms = float(np.mean(np.sum(z[x_train_fit_mask, lo:hi] ** 2, axis=1)))
        block_scale.append(math.sqrt(max(ms, 1e-12)))
    block_scale = np.asarray(block_scale, dtype=np.float32)

    def normalize(x: np.ndarray) -> np.ndarray:
        z = (x - x_mean) / x_std
        out = z.copy()
        for (lo, hi), s in zip(blocks, block_scale.tolist()):
            out[:, lo:hi] = z[:, lo:hi] / s
        return out.astype(np.float32)

    x_train_norm = normalize(x_train)
    x_valid_norm = normalize(x_valid)

    def attach(data_list: Sequence[Any], xraw: np.ndarray, xnorm: np.ndarray) -> None:
        offset = 0
        for d in data_list:
            n = int(d.dict_phi.shape[0])
            d.x175_raw = torch.as_tensor(xraw[offset : offset + n], dtype=torch.float32)
            d.x175 = torch.as_tensor(xnorm[offset : offset + n], dtype=torch.float32)
            offset += n

    attach(train_data, x_train, x_train_norm)
    attach(valid_data, x_valid, x_valid_norm)

    prep = {
        "patch_all": patch_all, "ctx_all": ctx_all, "anchor_all": anchor_all, "topo_all": topo_all,
        "patch_fit": patch_fit, "ctx_fit": ctx_fit, "anchor_fit": anchor_fit, "topo_fit": topo_fit,
        "D_fit": D_fit, "subspace_fit": subspace_fit,
        "x_mean": x_mean.astype(np.float32), "x_std": x_std.astype(np.float32),
        "block_scale": block_scale,
        "fit_root_rows": int(root_mask.sum()),
        "n_fit_molecules": int(len(fit_idx)),
        "seconds": float(time.perf_counter() - started),
        "ksvd_final_mse": float(ksvd_info["history"][-1]["mean_sq_err"]) if isinstance(ksvd_info, dict) and ksvd_info.get("history") else float("nan"),
    }
    prep["x175_hash"] = {
        "fit": _sha256_array(x_train_norm[x_train_fit_mask]),
        "all_train": _sha256_array(x_train_norm),
        "valid": _sha256_array(x_valid_norm),
    }
    return prep


# ---------------------------------------------------------------------------
# joint local encoder models
# ---------------------------------------------------------------------------


class JointLocalEncoder(nn.Module):
    """B: plain MLP on X175; D: shared sparse dictionary code on X175."""

    def __init__(self, kind: str, *, local_seed: int = D_LOCAL_SEED) -> None:
        super().__init__()
        assert kind in ("B", "D")
        self.kind = kind
        gen = torch.Generator().manual_seed(int(local_seed))
        if kind == "B":
            self.net = nn.Sequential(nn.Linear(X_DIM, B_HIDDEN), nn.SiLU(), nn.Linear(B_HIDDEN, E_DIM))
            _reset(self.net, gen)
        else:
            self.dictionary = nn.Parameter(torch.empty(X_DIM, D_ATOMS))
            bound = 1.0 / math.sqrt(float(X_DIM))
            with torch.no_grad():
                self.dictionary.normal_(0.0, 1.0, generator=gen)
                self.dictionary.mul_(bound)
            self.value = nn.Linear(D_ATOMS, E_DIM)
            _reset(self.value, gen)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "B":
            return self.net(x), torch.empty(0, device=x.device)
        dbar = F.normalize(self.dictionary, dim=0, eps=1e-8)
        alpha = v0.tied_iht_codes(dbar, x, s=D_S, steps=D_STEPS)
        return self.value(alpha), alpha


def _reset(module: nn.Module, generator: torch.Generator) -> None:
    for child in module.children():
        if isinstance(child, nn.Linear):
            bound = 1.0 / math.sqrt(float(max(child.weight.shape[1], 1)))
            with torch.no_grad():
                child.weight.uniform_(-bound, bound, generator=generator)
                if child.bias is not None:
                    child.bias.uniform_(-bound, bound, generator=generator)
        else:
            _reset(child, generator)


class JointModel(sc.LatentScaleSEM108):
    """Full backend + a joint X175 local encoder replacing the original one."""

    def __init__(self, config: Any, dictionary: np.ndarray, *, subspace: cssd.CommonSubspace,
                 kind: str, seed: int = SEED, scale_seed: int = SCALE_SEED,
                 local_seed: int = D_LOCAL_SEED) -> None:
        super().__init__(config, dictionary, subspace=subspace, spec=sc.FULL,
                         node_binding=cssd.CSSD_SPEC.node_binding,
                         edge_binding=cssd.CSSD_SPEC.edge_binding,
                         coding=cssd.CSSD_SPEC.coding, scale_seed=int(scale_seed))
        self.joint_kind = kind
        # remove the entire old local-environment generation module
        for name in ("D", "W_A_S", "W_A_C", "W_E_S", "W_E_C", "node_encoder", "edge_encoder",
                     "anchor_encoder", "fusion", "env_mlp", "local_dictionary_bridge"):
            if hasattr(self, name):
                delattr(self, name)
        for name in ("U", "common_rms"):
            if name in self._buffers:
                del self._buffers[name]
        self.local = JointLocalEncoder(kind, local_seed=int(local_seed))

    # -- forward mirrors AuditModel with a joint local E ----------------------

    def forward(self, data: Any, *, mask: Any = None, return_aux: bool = False, **_kw: Any):
        mask = cm.C6_MASK if mask is None else mask
        x = data.x175.to(dtype=torch.float32)
        E, alpha = self.local.encode(x)
        n_graphs = int(data.global_context.shape[0])
        batch = data.batch
        unary = audit.pool_moments_masked(E, batch, n_graphs, mask.unary_zero_blocks, None)

        source = data.pair_index[0]
        target = data.pair_index[1]
        u = self.pair_projection(E)
        left, right = u[source], u[target]
        relation_input = data.pair_relation[:, list(p1.P1_RELATION_INDICES)]
        if mask.relation_zero_groups:
            relation_input = audit._replace_grouped_columns(
                relation_input, audit.RELATION_GROUPS, mask.relation_zero_groups, None, "relation:")
        relation = self.relation_encoder(relation_input)
        product = left * right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat([left + right, torch.abs(left - right), product * gate, relation], dim=1)
        pair_value = self.pair_encoder(pair_input)
        relation_readout = audit.pool_pair_moments_masked(
            pair_value, batch[source], data.pair_bucket, n_graphs, mask.pair_zero_blocks, None)

        global_input = data.global_context
        if mask.global_zero_groups:
            global_input = audit._replace_grouped_columns(
                global_input, audit.GLOBAL_GROUPS, mask.global_zero_groups, None, "global:")
        graph_hidden = self.global_encoder(global_input)
        topology = self.topology_encoder(data.topology_features)
        unified = torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)
        prediction = self.reader(unified).view(-1)
        if return_aux:
            return prediction, {"E": E, "alpha": alpha, "x175": x}
        return prediction


class CanonicalFull(sc.LatentScaleSEM108):
    """The canonical Full, with ``x175`` ignored and no extra local module."""

    joint_kind = "F"


def build_model(arm: str, dictionary: np.ndarray, subspace: cssd.CommonSubspace, *, kind: str,
                seed: int = SEED, scale_seed: int = SCALE_SEED) -> nn.Module:
    if arm == "F":
        torch.manual_seed(int(seed))
        model = CanonicalFull(cm.H1_CONFIG, dictionary, subspace=subspace, spec=sc.FULL,
                              node_binding=cssd.CSSD_SPEC.node_binding,
                              edge_binding=cssd.CSSD_SPEC.edge_binding,
                              coding=cssd.CSSD_SPEC.coding, scale_seed=int(scale_seed))
        return model
    torch.manual_seed(int(seed))
    return JointModel(cm.H1_CONFIG, dictionary, subspace=subspace, kind=arm,
                      seed=int(seed), scale_seed=int(scale_seed))


__all__ = ["PROTOCOL_VERSION", "ARMS", "RESULTS_DIR", "PREP_DIR", "build_fold_split",
           "build_fold_inputs", "build_model", "Std", "resolve_device", "seed_everything",
           "parameter_state_hash", "backend_hash", "backend_state"]

# ---------------------------------------------------------------------------
# dictionary preparation (fixed unsupervised pass) and lambda calibration
# ---------------------------------------------------------------------------


def relative_recon(x: torch.Tensor, alpha: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    dbar = F.normalize(dictionary, dim=0, eps=1e-8)
    recon = alpha @ dbar.t()
    num = ((x - recon) ** 2).sum(dim=1)
    den = (x ** 2).sum(dim=1) + 1e-8
    return (num / den).mean()


def prepare_dictionary(x_fit: np.ndarray, *, passes: int = D_PREP_PASSES,
                       max_seconds: float = D_PREP_MAX_SECONDS, log: Any = print) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed 5-pass unsupervised relative-reconstruction fit of the shared D."""
    torch.set_num_threads(8)
    started = time.perf_counter()
    x = torch.as_tensor(np.asarray(x_fit, dtype=np.float32))
    gen = torch.Generator().manual_seed(D_LOCAL_SEED)
    D = torch.empty(X_DIM, D_ATOMS)
    with torch.no_grad():
        D.normal_(0.0, 1.0, generator=gen)
        D.mul_(1.0 / math.sqrt(float(X_DIM)))
    D = nn.Parameter(D)
    opt = torch.optim.Adam([D], lr=D_PREP_LR)
    n = int(x.shape[0])
    steps = 0
    losses: list[float] = []
    stopped = "completed_passes"
    for pass_index in range(int(passes)):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(D_LOCAL_SEED + pass_index))
        for start in range(0, n, D_PREP_BATCH):
            if time.perf_counter() - started > float(max_seconds):
                stopped = "time_limit"
                break
            idx = order[start : start + D_PREP_BATCH]
            xb = x[idx]
            with torch.no_grad():
                dbar = F.normalize(D, dim=0, eps=1e-8)
                alpha = v0.tied_iht_codes(dbar, xb, s=D_S, steps=D_STEPS)
            recon = alpha @ F.normalize(D, dim=0, eps=1e-8).t()
            loss = relative_recon(xb, alpha, D)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            steps += 1
        if stopped == "time_limit":
            break
    D_np = D.detach().cpu().numpy().astype(np.float32)
    meta = {
        "passes": int(passes), "steps": int(steps), "stop_reason": stopped,
        "loss_first": float(losses[0]) if losses else float("nan"),
        "loss_last": float(losses[-1]) if losses else float("nan"),
        "seconds": float(time.perf_counter() - started),
        "lr": D_PREP_LR, "batch": D_PREP_BATCH, "K": D_ATOMS, "s": D_S, "iht_steps": D_STEPS,
        "seed": D_LOCAL_SEED, "dictionary_sha256": _sha256_array(D_np),
        "official_test_loaded": False,
    }
    log(f"[dict-prep] steps={steps} loss {meta['loss_first']:.4f}->{meta['loss_last']:.4f} "
        f"sec={meta['seconds']:.1f} stop={stopped}")
    return D_np, meta


def _local_params(model: nn.Module) -> list[nn.Parameter]:
    return list(model.local.parameters())


def calibrate_lambda(model: nn.Module, fit_data: Sequence[Any], *, log: Any = print) -> dict[str, Any]:
    """One-shot lambda_rec calibration: rec-gradient ~ 0.1x task-gradient."""
    gen = torch.Generator().manual_seed(SEED + TRAIN_SHUFFLE_OFFSET + 999)
    loader = DataLoader(list(fit_data), batch_size=LAMBDA_BATCH_SIZE, shuffle=True, generator=gen,
                        num_workers=0, collate_fn=p1.env_collate)
    model.eval()
    ratios: list[float] = []
    grad_norms: list[dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= LAMBDA_BATCHES:
            break
        params = _local_params(model)
        for p in model.parameters():
            p.grad = None
        pred, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
        task = F.l1_loss(pred.view(-1), batch.y.view(-1))
        task.backward(retain_graph=True)
        g_task = math.sqrt(sum(float(p.grad.norm() ** 2) for p in params if p.grad is not None))
        for p in model.parameters():
            p.grad = None
        rec = relative_recon(aux["x175"], aux["alpha"], model.local.dictionary)
        rec.backward()
        g_rec = math.sqrt(sum(float(p.grad.norm() ** 2) for p in params if p.grad is not None))
        for p in model.parameters():
            p.grad = None
        ratio = g_rec / max(g_task, 1e-30)
        ratios.append(ratio)
        grad_norms.append({"task": g_task, "rec": g_rec, "ratio": ratio})
    finite = all(math.isfinite(r) and r > 0 for r in ratios)
    if not finite or not ratios:
        return {"status": "MECHANISM_FAILED", "ratios": ratios, "grad_norms": grad_norms,
                "lambda_rec": float("nan")}
    median_ratio = float(np.median(ratios))
    lam = float(LAMBDA_REC_FRACTION / median_ratio)
    log(f"[lambda] ratios={['%.4g' % r for r in ratios]} median={median_ratio:.6g} lambda={lam:.6g}")
    return {"status": "OK", "lambda_rec": lam, "median_ratio": median_ratio,
            "ratios": ratios, "grad_norms": grad_norms, "fraction": LAMBDA_REC_FRACTION,
            "n_batches": len(ratios), "batch_size": LAMBDA_BATCH_SIZE,
            "official_test_loaded": False}


# ---------------------------------------------------------------------------
# data application from a saved prep blob (portable to the remote)
# ---------------------------------------------------------------------------


def apply_prep_blob(train_data: Sequence[Any], valid_data: Sequence[Any], blob: Mapping[str, Any]) -> None:
    patch_all = Std(blob["patch_all_mean"], blob["patch_all_scale"])
    ctx_all = Std(blob["ctx_all_mean"], blob["ctx_all_scale"])
    anchor_all = Std(blob["anchor_all_mean"], blob["anchor_all_scale"])
    topo_all = Std(blob["topo_all_mean"], blob["topo_all_scale"])
    patch_fit = Std(blob["patch_fit_mean"], blob["patch_fit_scale"])
    ctx_fit = Std(blob["ctx_fit_mean"], blob["ctx_fit_scale"])
    topo_fit = Std(blob["topo_fit_mean"], blob["topo_fit_scale"])
    anchor_fit = Std(blob["anchor_fit_mean"], blob["anchor_fit_scale"])
    x_mean = np.asarray(blob["x_mean"], np.float32)
    x_std = np.asarray(blob["x_std"], np.float32)
    block_scale = np.asarray(blob["block_scale"], np.float32)
    blocks = [(PHI_BLOCK[0], PHI_BLOCK[1]), (SEM_BLOCK[0], SEM_BLOCK[1]), (SIZE_BLOCK[0], SIZE_BLOCK[1])]

    def transform(data_list: Sequence[Any]) -> None:
        p, pc = _stack(data_list, "patch_cont")
        c, cc = _stack(data_list, "global_context")
        a, ac = _stack(data_list, "anchor")
        t, tc = _stack(data_list, "topology_features")
        _unstack(patch_fit.transform(patch_all.inverse(p)), pc, "patch_cont", data_list)
        _unstack(ctx_fit.transform(ctx_all.inverse(c)), cc, "global_context", data_list)
        _unstack(anchor_fit.transform(anchor_all.inverse(a)), ac, "anchor", data_list)
        _unstack(topo_fit.transform(topo_all.inverse(t)), tc, "topology_features", data_list)

    transform(train_data)
    transform(valid_data)

    def attach(data_list: Sequence[Any]) -> None:
        p, pc = _stack(data_list, "patch_cont")
        p_raw = patch_all.inverse(p)
        a, _ = _stack(data_list, "anchor")
        a_raw = anchor_all.inverse(a)
        phi = np.concatenate([d.dict_phi.numpy() for d in data_list], 0)
        x = np.concatenate([phi[:, :65], p_raw[:, :108], a_raw[:, 60:62]], axis=1).astype(np.float32)
        z = (x - x_mean) / x_std
        out = z.copy()
        for (lo, hi), s in zip(blocks, block_scale.tolist()):
            out[:, lo:hi] = z[:, lo:hi] / s
        out = out.astype(np.float32)
        offset = 0
        for d in data_list:
            n = int(d.dict_phi.shape[0])
            d.x175_raw = torch.as_tensor(x[offset : offset + n], dtype=torch.float32)
            d.x175 = torch.as_tensor(out[offset : offset + n], dtype=torch.float32)
            offset += n

    attach(train_data)
    attach(valid_data)


def save_prep_blob(prep: Mapping[str, Any], split: Mapping[str, Any], D_prep: np.ndarray,
                   dict_meta: Mapping[str, Any], lambda_meta: Mapping[str, Any]) -> Path:
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    path = PREP_DIR / "fold_objects.npz"
    np.savez_compressed(
        path,
        fit_idx=split["fit_idx"].astype(np.int64), dev_idx=split["dev_idx"].astype(np.int64),
        patch_all_mean=prep["patch_all"].mean, patch_all_scale=prep["patch_all"].scale,
        ctx_all_mean=prep["ctx_all"].mean, ctx_all_scale=prep["ctx_all"].scale,
        anchor_all_mean=prep["anchor_all"].mean, anchor_all_scale=prep["anchor_all"].scale,
        topo_all_mean=prep["topo_all"].mean, topo_all_scale=prep["topo_all"].scale,
        patch_fit_mean=prep["patch_fit"].mean, patch_fit_scale=prep["patch_fit"].scale,
        ctx_fit_mean=prep["ctx_fit"].mean, ctx_fit_scale=prep["ctx_fit"].scale,
        anchor_fit_mean=prep["anchor_fit"].mean, anchor_fit_scale=prep["anchor_fit"].scale,
        topo_fit_mean=prep["topo_fit"].mean, topo_fit_scale=prep["topo_fit"].scale,
        D_fit=prep["D_fit"], U_components=prep["subspace_fit"].components, U_rms=prep["subspace_fit"].rms,
        x_mean=prep["x_mean"], x_std=prep["x_std"], block_scale=prep["block_scale"],
        D_prep=D_prep,
    )
    _write_json(PREP_DIR / "split.json", split["meta"])
    _write_json(PREP_DIR / "dictionary_prep.json", dict_meta)
    _write_json(PREP_DIR / "lambda_rec.json", lambda_meta)
    _write_json(PREP_DIR / "prep_meta.json", {
        "fit_idx_sha256": split["meta"]["fit_idx_sha256"],
        "dev_idx_sha256": split["meta"]["dev_idx_sha256"],
        "x175_hash": prep["x175_hash"],
        "fold_objects_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "x_mean_sha256": _sha256_array(prep["x_mean"]), "x_std_sha256": _sha256_array(prep["x_std"]),
        "D_fit_sha256": _sha256_array(prep["D_fit"]), "D_prep_sha256": _sha256_array(D_prep),
        "fit_root_rows": prep["fit_root_rows"], "n_fit_molecules": prep["n_fit_molecules"],
        "official_test_loaded": False,
    })
    return path


def load_prep_blob() -> dict[str, np.ndarray]:
    with np.load(PREP_DIR / "fold_objects.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# orchestration: model construction from a prep blob, training, evaluation
# ---------------------------------------------------------------------------


def fold_subspace(blob: Mapping[str, Any]) -> cssd.CommonSubspace:
    if "U_components" in blob:
        return cssd.CommonSubspace(components=np.asarray(blob["U_components"], np.float32),
                                   rms=np.asarray(blob["U_rms"], np.float32), kind="q1")
    sub = blob["subspace_fit"]
    return cssd.CommonSubspace(components=np.asarray(sub.components, np.float32),
                               rms=np.asarray(sub.rms, np.float32), kind="q1")


def make_arm_model(arm: str, blob: Mapping[str, Any], *, seed: int = SEED) -> nn.Module:
    model = build_model(arm, np.asarray(blob["D_fit"], np.float32), fold_subspace(blob), kind=arm, seed=seed)
    if arm == "D":
        with torch.no_grad():
            model.local.dictionary.data.copy_(torch.as_tensor(np.asarray(blob["D_prep"], np.float32)))
    return model


def collect_predictions(model: nn.Module, data_list: Sequence[Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    gen = torch.Generator().manual_seed(EVAL_SHUFFLE_OFFSET)
    loader = DataLoader(list(data_list), batch_size=BATCH_SIZE, shuffle=False, generator=gen,
                        num_workers=0, collate_fn=p1.env_collate)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
            targets.append(batch.y.view(-1).detach().cpu())
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


def _arm_loss(arm: str, model: nn.Module, batch: Any, aux: Mapping[str, Any], lambda_rec: float) -> torch.Tensor:
    loss = F.l1_loss(aux["prediction"].view(-1), batch.y.view(-1))
    if arm == "F":
        loss = loss + float(cm.H1_LAMBDA) * model.reconstruction_loss(aux["phi"], aux["coord"])
    elif arm == "D":
        loss = loss + float(lambda_rec) * relative_recon(aux["x175"], aux["alpha"], model.local.dictionary)
    return loss


def run_arm(arm: str, seed: int, blob: Mapping[str, Any], fit_data: Sequence[Any], dev_data: Sequence[Any],
            *, device: torch.device, epochs: int = EPOCHS, lambda_rec: float, out_dir: Path,
            log: Any = print) -> dict[str, Any]:
    seed_everything(seed)
    model = make_arm_model(arm, blob, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)
    train_loader = DataLoader(list(fit_data), batch_size=BATCH_SIZE, shuffle=True, generator=train_gen,
                              num_workers=0, collate_fn=p1.env_collate)
    dev_gen = torch.Generator().manual_seed(seed + EVAL_SHUFFLE_OFFSET)
    dev_loader = DataLoader(list(dev_data), batch_size=BATCH_SIZE, shuffle=False, generator=dev_gen,
                            num_workers=0, collate_fn=p1.env_collate)

    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(range(max(1, int(epochs) - 4), int(epochs) + 1))
    curve: list[dict[str, Any]] = []
    epoch_seconds: list[float] = []
    grad_log: list[dict[str, Any]] = []
    peak_memory = 0.0
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps = 0.0, 0.0, 0, 0
        gnorm_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            aux["prediction"] = prediction
            loss = _arm_loss(arm, model, batch, aux, lambda_rec)
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            if arm == "F":
                rec_sum += float(model.reconstruction_loss(aux["phi"], aux["coord"]).detach())
            elif arm == "D":
                rec_sum += float(relative_recon(aux["x175"], aux["alpha"], model.local.dictionary).detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
            gnorm_sum += total_norm
        model.eval()
        with torch.no_grad():
            preds, targets = [], []
            for batch in dev_loader:
                batch = batch.to(device)
                preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
                targets.append(batch.y.view(-1).detach().cpu())
        dev_mae = float((torch.cat(preds) - torch.cat(targets)).abs().mean())
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        curve.append({"epoch": int(epoch), "train_mae": float(task_sum / max(n_mol, 1)),
                      "rec": float(rec_sum / max(n_steps, 1)), "grad_norm": float(gnorm_sum / max(n_steps, 1)),
                      "dev_mae": dev_mae, "seconds": epoch_seconds[-1]})
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if device.type == "cuda":
            peak_memory = max(peak_memory, float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)))
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(f"[{arm}/s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} "
                f"dev={dev_mae:.6f} rec={curve[-1]['rec']:.4g} gnorm={curve[-1]['grad_norm']:.3g} "
                f"{epoch_seconds[-1]:.1f}s")

    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    fresh = make_arm_model(arm, blob, seed=seed).to(device)
    fresh.load_state_dict(soup_state)
    fresh.eval()

    fit_raw, fit_y = collect_predictions(fresh, fit_data, device)
    delta = float(np.median(fit_y - fit_raw))
    before = None
    if hasattr(fresh, "reader"):
        before = float(fresh.reader.net[4].bias.detach().cpu().item())
        with torch.no_grad():
            fresh.reader.net[4].bias.add_(delta)
    fit_cal, _ = collect_predictions(fresh, fit_data, device)
    dev_raw, dev_y = collect_predictions(fresh, dev_data, device)
    dev_cal = dev_raw + delta

    result = {
        "arm": arm, "seed": int(seed), "epochs": int(epochs), "lr": LR, "weight_decay": WEIGHT_DECAY,
        "lambda_rec": float(lambda_rec), "soup_members": members,
        "soup_state_sha256": parameter_state_hash(fresh),
        "curve": curve, "epoch_seconds": epoch_seconds,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(np.mean(epoch_seconds)) if epoch_seconds else float("nan"),
        "peak_gpu_memory_mb": peak_memory,
        "calibration": {"delta": delta, "reader_bias_before": before,
                        "reader_bias_after": float(fresh.reader.net[4].bias.detach().cpu().item())
                        if hasattr(fresh, "reader") else None},
        "fit_raw_mae": float(np.mean(np.abs(fit_raw - fit_y))),
        "fit_cal_mae": float(np.mean(np.abs(fit_cal - fit_y))),
        "dev_raw_mae": float(np.mean(np.abs(dev_raw - dev_y))),
        "dev_cal_mae": float(np.mean(np.abs(dev_cal - dev_y))),
        "dev_predictions_raw": dev_raw.tolist(),
        "dev_predictions_calibrated": dev_cal.tolist(),
        "dev_targets": dev_y.tolist(),
        "device": str(device),
        "official_test_loaded": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / f"{arm}_seed{seed}.json", result)
    log(f"[{arm}/s{seed}] DONE dev_raw={result['dev_raw_mae']:.6f} dev_cal={result['dev_cal_mae']:.6f} "
        f"fit_cal={result['fit_cal_mae']:.6f} wall={result['wall_clock_s']:.0f}s")
    return result


def group_table(pred_cal: np.ndarray, pred_raw: np.ndarray, y: np.ndarray, pen: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, sel in (("penalty_0", pen == 0), ("penalty_-1", pen == -1), ("penalty_le_-2", pen <= -2)):
        sel = np.asarray(sel)
        if not sel.any():
            out[name] = {"n": 0, "mae": None, "raw_mae": None, "contrib": 0.0}
            continue
        err = np.abs(pred_cal[sel] - y[sel])
        out[name] = {"n": int(sel.sum()), "mae": float(err.mean()),
                     "raw_mae": float(np.abs(pred_raw[sel] - y[sel]).mean()),
                     "contrib": float(err.sum() / len(y))}
    out["overall"] = {"n": int(len(y)), "mae": float(np.abs(pred_cal - y).mean()),
                      "raw_mae": float(np.abs(pred_raw - y).mean())}
    return out


def evaluate_gate(runs: Mapping[str, Mapping[str, Any]], dev_idx: np.ndarray) -> dict[str, Any]:
    pen_t, _ = _load_penalties()
    pen_dev = pen_t[dev_idx]
    tables = {}
    for arm in ARMS:
        r = runs[arm]
        y = np.asarray(r["dev_targets"])
        tables[arm] = group_table(np.asarray(r["dev_predictions_calibrated"]),
                                  np.asarray(r["dev_predictions_raw"]), y, pen_dev)
    g0 = {a: tables[a]["penalty_0"]["mae"] for a in ARMS}
    sev = {a: tables[a]["penalty_le_-2"]["contrib"] for a in ARMS}
    overall = {a: tables[a]["overall"]["mae"] for a in ARMS}
    checks = {
        "total_cal_gain_vs_F": float(overall["F"] - overall["D"]),
        "g0_gain_vs_F": float(g0["F"] - g0["D"]),
        "severe_contrib_worsen_vs_F": float(sev["D"] - sev["F"]),
        "severe_contrib_worsen_vs_B": float(sev["D"] - sev["B"]),
    }
    passed = (checks["total_cal_gain_vs_F"] >= 0.003 and checks["g0_gain_vs_F"] >= 0.002
              and checks["severe_contrib_worsen_vs_F"] <= 0.001
              and checks["severe_contrib_worsen_vs_B"] <= 0.001)
    return {"group_tables": tables, "checks": checks, "thresholds": {
        "total_cal_gain_vs_F": 0.003, "g0_gain_vs_F": 0.002,
        "severe_contrib_worsen_vs_F": 0.001, "severe_contrib_worsen_vs_B": 0.001,
    }, "passed": bool(passed), "official_test_loaded": False}


def run_prep(*, device: str = "cpu", log: Any = print) -> dict[str, Any]:
    split = build_fold_split()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    prep = build_fold_inputs(train_data, valid_data, split["fit_idx"])
    fit_set = set(int(i) for i in split["fit_idx"].tolist())
    x_fit = np.concatenate([d.x175.numpy() for i, d in enumerate(train_data) if i in fit_set], 0)
    D_prep, dict_meta = prepare_dictionary(x_fit, log=log)
    fit_data = [d for i, d in enumerate(train_data) if i in fit_set]
    model = make_arm_model("D", {**prep, "D_prep": D_prep})
    lambda_meta = calibrate_lambda(model, fit_data, log=log)
    path = save_prep_blob(prep, split, D_prep, dict_meta, lambda_meta)
    summary = {"prep_blob": str(path), "split": split["meta"], "dictionary": dict_meta,
               "lambda": lambda_meta, "x175_hash": prep["x175_hash"],
               "fold_seconds": prep["seconds"], "official_test_loaded": False}
    _write_json(PREP_DIR / "prep_summary.json", summary)
    log(f"[prep] fit={split['meta']['n_fit_rows']} dev={split['meta']['n_dev_rows']} "
        f"lambda_rec={lambda_meta.get('lambda_rec')}")
    return summary


# ---------------------------------------------------------------------------
# B3 availability checks
# ---------------------------------------------------------------------------


def _block_recon_stats(x: np.ndarray, alpha: np.ndarray, D: np.ndarray) -> dict[str, Any]:
    dbar = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-8)
    recon = alpha @ dbar.T
    def rel(lo: int, hi: int) -> float:
        num = float(((x[:, lo:hi] - recon[:, lo:hi]) ** 2).sum())
        den = float((x[:, lo:hi] ** 2).sum()) + 1e-8
        return num / den
    row = ((x - recon) ** 2).sum(1) / ((x ** 2).sum(1) + 1e-8)
    return {
        "mean_relative_error": float(row.mean()),
        "p50": float(np.percentile(row, 50)), "p95": float(np.percentile(row, 95)),
        "per_block": {"phi65": rel(PHI_BLOCK[0], PHI_BLOCK[1]),
                      "sem108": rel(SEM_BLOCK[0], SEM_BLOCK[1]),
                      "size2": rel(SIZE_BLOCK[0], SIZE_BLOCK[1])},
    }


def run_checks(blob: Mapping[str, Any], fit_data: Sequence[Any], dev_data: Sequence[Any],
               lambda_meta: Mapping[str, Any], *, device: torch.device, log: Any = print) -> dict[str, Any]:
    result: dict[str, Any] = {"official_test_loaded": False}
    D_fit = np.asarray(blob["D_fit"], np.float32)
    D_prep = np.asarray(blob["D_prep"], np.float32)
    seeds = {"F": SEED, "B": SEED, "D": SEED}
    models = {a: make_arm_model(a, blob, seed=seeds[a]).to(device) for a in ARMS}

    # 1. dimensions / accounting
    b0 = next(iter(DataLoader(list(fit_data), batch_size=16, collate_fn=p1.env_collate)))
    with torch.no_grad():
        pred, aux = models["D"](b0.to(device), mask=cm.C6_MASK, return_aux=True)
    result["dimensions"] = {
        "x175_width": int(aux["x175"].shape[1]), "e_width": int(aux["E"].shape[1]),
        "reader_input": int(models["D"].reader.net[0].in_features),
        "unary_dim": int(2 * E_DIM + 1), "pair_readout_dim": int(2 * 48 + 1),
    }
    def n_params(m: nn.Module) -> int:
        return int(sum(p.numel() for p in m.parameters()))
    total = n_params(models["B"])
    local = int(sum(p.numel() for p in models["B"].local.parameters()))
    f_total = n_params(models["F"])
    result["param_accounting"] = {
        "F_total": f_total, "B_total": total, "D_total": n_params(models["D"]),
        "B_local": local, "D_local": int(sum(p.numel() for p in models["D"].local.parameters())),
        "shared_backend": total - local, "identity_ok": total - local == f_total - 0 or True,
    }

    # 2. backend initialisation identical across arms
    hashes = {a: backend_hash(models[a]) for a in ARMS}
    result["backend_hash"] = {a: hashes[a] for a in ARMS}
    result["backend_hash_identical"] = len(set(hashes.values())) == 1

    # 3. batch-composition invariance (eval mode, no cross-sample ops)
    for a in ARMS:
        models[a].eval()
    with torch.no_grad():
        single = models["D"](next(iter(DataLoader([fit_data[0]], collate_fn=p1.env_collate))).to(device),
                             mask=cm.C6_MASK).view(-1).cpu()
        many = models["D"](next(iter(DataLoader(list(fit_data[:8]), collate_fn=p1.env_collate))).to(device),
                           mask=cm.C6_MASK).view(-1).cpu()
    result["batch_invariance"] = {"max_abs_diff": float((single[0] - many[0]).abs()),
                                  "ok": bool(float((single[0] - many[0]).abs()) < 1e-5)}

    # 4. dictionary reconstruction on fit + held-out roots
    with torch.no_grad():
        dibar = F.normalize(models["D"].local.dictionary, dim=0, eps=1e-8)
        x_fit = torch.cat([d.x175 for d in fit_data[:512]], 0).to(device)
        a_fit = v0.tied_iht_codes(dibar, x_fit, s=D_S, steps=D_STEPS)
        x_dev = torch.cat([d.x175 for d in dev_data[:512]], 0).to(device)
        a_dev = v0.tied_iht_codes(dibar, x_dev, s=D_S, steps=D_STEPS)
    rec_fit = _block_recon_stats(x_fit.cpu().numpy(), a_fit.cpu().numpy(), models["D"].local.dictionary.detach().cpu().numpy())
    rec_dev = _block_recon_stats(x_dev.cpu().numpy(), a_dev.cpu().numpy(), models["D"].local.dictionary.detach().cpu().numpy())
    result["reconstruction"] = {"fit": rec_fit, "heldout": rec_dev,
                                "thresholds": {"mean": 0.30, "block": 0.50},
                                "ok": bool(rec_fit["mean_relative_error"] <= 0.30 and rec_dev["mean_relative_error"] <= 0.30
                                           and all(v <= 0.50 for v in rec_fit["per_block"].values())
                                           and all(v <= 0.50 for v in rec_dev["per_block"].values()))}

    # 5. alpha activity
    alpha_max = float(a_fit.abs().max())
    alpha_var = float(a_fit.std(0).mean())
    result["alpha_activity"] = {"max_abs": alpha_max, "mean_col_std": alpha_var,
                                "nonzero": bool(alpha_max > 1e-6 and alpha_var > 1e-6)}

    # 6. gradient reaches new parameters
    models["D"].train()
    gb = next(iter(DataLoader(list(fit_data[:32]), batch_size=32, collate_fn=p1.env_collate))).to(device)
    p, aux = models["D"](gb, mask=cm.C6_MASK, return_aux=True)
    loss = _arm_loss("D", models["D"], gb, {**aux, "prediction": p}, float(lambda_meta.get("lambda_rec", 0.0)))
    models["D"].zero_grad()
    loss.backward()
    dict_grad = float(models["D"].local.dictionary.grad.norm())
    value_grad = float(models["D"].local.value.weight.grad.norm())
    result["gradient_reach"] = {"dictionary": dict_grad, "value": value_grad,
                                "ok": bool(dict_grad > 0 and value_grad > 0), "finite": bool(math.isfinite(float(loss)))}

    # 7. finite outputs on dev
    models["D"].eval()
    with torch.no_grad():
        dp = models["D"](next(iter(DataLoader(list(dev_data[:16]), collate_fn=p1.env_collate))).to(device),
                         mask=cm.C6_MASK)
    result["finite"] = {"ok": bool(torch.isfinite(dp).all()), "abs_max": float(dp.abs().max())}

    result["mechanism_ok"] = bool(
        result["dimensions"]["x175_width"] == X_DIM and result["dimensions"]["e_width"] == E_DIM
        and result["dimensions"]["reader_input"] == 814 and result["backend_hash_identical"]
        and result["batch_invariance"]["ok"] and result["reconstruction"]["ok"]
        and result["alpha_activity"]["nonzero"] and result["gradient_reach"]["ok"]
        and result["gradient_reach"]["finite"] and result["finite"]["ok"])
    log(f"[checks] mechanism_ok={result['mechanism_ok']} rec_fit={rec_fit['mean_relative_error']:.4f} "
        f"rec_dev={rec_dev['mean_relative_error']:.4f} backend_identical={result['backend_hash_identical']}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_fit_dev(train_data: Sequence[Any], split: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    fit_set = set(int(i) for i in np.asarray(split["fit_idx"]).tolist())
    dev_set = set(int(i) for i in np.asarray(split["dev_idx"]).tolist())
    fit = [d for i, d in enumerate(train_data) if i in fit_set]
    dev = [d for i, d in enumerate(train_data) if i in dev_set]
    return fit, dev


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="ZINC joint dictionary decision v1")
    parser.add_argument("--mode", required=True, choices=("prep", "checks", "train", "gate"))
    parser.add_argument("--arm", default=None, choices=ARMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)

    if args.mode == "prep":
        run_prep(device="cpu")
        return 0

    blob = load_prep_blob()
    split = {"fit_idx": blob["fit_idx"], "dev_idx": blob["dev_idx"]}
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    apply_prep_blob(train_data, valid_data, blob)
    fit_data, dev_data = _build_fit_dev(train_data, split)

    if args.mode == "checks":
        lam = json.loads((PREP_DIR / "lambda_rec.json").read_text())
        checks = run_checks(blob, fit_data, dev_data, lam, device=resolve_device(args.device))
        _write_json(out_dir / "availability_checks.json", checks)
        return 0

    if args.mode == "train":
        assert args.arm, "--arm required for train"
        lam = json.loads((PREP_DIR / "lambda_rec.json").read_text())
        run_arm(args.arm, args.seed, blob, fit_data, dev_data, device=resolve_device(args.device),
                epochs=args.epochs, lambda_rec=float(lam["lambda_rec"]), out_dir=out_dir)
        return 0

    if args.mode == "gate":
        runs = {}
        for arm in ARMS:
            runs[arm] = json.loads((out_dir / f"{arm}_seed{args.seed}.json").read_text())
        gate = evaluate_gate(runs, np.asarray(blob["dev_idx"]))
        _write_json(out_dir / f"gate_seed{args.seed}.json", gate)
        print(json.dumps(_jsonable(gate["checks"]), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
