"""Source-specificity judgement + Junction-to-Bond ownership models (v1).

Protocol: ``configs/bjoc_source_ownership_v1.yaml`` (frozen before execution).
Track: ``tracks/structure-contribution``.  Namespace: ``bjoc_source_ownership_v1``.

This module defines the *only* model change of the round: a functional
ownership-grouping switch plus one independent global content pathway.
No GNN/attention/iteration, no width search, no new feature family.

Arms
----
``N`` (no ownership)   ``W_e = n_e * mean_G(w)``, ``V_e = n_e * mean_G(w^2)``
``O`` (true ownership) ``W_e = sum_{j in inc(e)} w_j``, ``V_e = sum_j w_j^2``
``S`` (sham ownership) count-matched deterministic draws from the molecule's
                       own canonical junction pool; per-bond counts and vector
                       format identical to ``O``, true incidence destroyed.

All arms share one class and state-dict layout.  ``build_ownership_from_init``
starts from the historical scratch BJOC-Null init, copies every base module and
widens the reader first layer with zero columns for the new global block, so at
init every arm is exactly the frozen P3 function.

Phase-0 helpers (junction signature, rankings, matched controls) live here too,
so the driver and the tests share one implementation.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-source-ownership-v1"

MODE_N = "N"
MODE_O = "O"
MODE_S = "S"
MODES = (MODE_N, MODE_O, MODE_S)
MODE_NAMES = {
    MODE_N: "NO_OWNERSHIP_NULL_AGGREGATE",
    MODE_O: "TRUE_JUNCTION_TO_BOND_OWNERSHIP",
    MODE_S: "COUNT_MATCHED_SHUFFLED_OWNERSHIP",
}
ARM_MODES = {"N": MODE_N, "O": MODE_O, "S": MODE_S}

#: global content pathway widths (frozen)
GLOBAL_CONTENT_IN = 2 * 96 + 1
GLOBAL_CONTENT_HIDDEN = 96
GLOBAL_CONTENT_OUT = 96

#: deterministic key coefficients (frozen; used only for label-invariant ordering)
JUNC_KEY_SEED = 0x5EED0117
BOND_KEY_SEED = 0x5EED0119
MIX_MUL_A = 6364136223846793005
MIX_MUL_B = 1442695040888963407
MIX_MASK = (1 << 62) - 1

#: composer layout of the frozen BJOC model
_COMPOSE_B = 96
_COMPOSE_W_OFF = 96
_COMPOSE_V_OFF = 192
_COMPOSE_Q_IDX = 288
_COMPOSE_RAW_OFF = 289

#: reader layout with the appended global block
READER_LOCAL_BLOCKS = {
    "z_bond": (0, 96),
    "sum_atom": (96, 192),
    "sum_atom_squared": (192, 288),
    "global_topology": (288, 321),
    "global_content": (321, 417),
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class OwnershipConfig(src.BJOCConfig):
    """Frozen widths + one appended global content block (96 reader dims)."""

    @property
    def reader_in(self) -> int:
        return int(src.BJOCConfig.reader_in.fget(self)) + GLOBAL_CONTENT_OUT

    def inventory(self) -> dict[str, int]:
        out = dict(src.BJOCConfig.inventory(self))

        def lin(i: int, o: int) -> int:
            return i * o + o

        gc = lin(GLOBAL_CONTENT_IN, GLOBAL_CONTENT_HIDDEN) + lin(GLOBAL_CONTENT_HIDDEN, GLOBAL_CONTENT_OUT)
        out["global_content"] = gc
        out["total"] = out["total"] + gc
        return out


# ---------------------------------------------------------------------------
# label-invariant molecule / junction / bond keys
# ---------------------------------------------------------------------------


def _sorted_rows_bytes(x: np.ndarray) -> bytes:
    a = np.ascontiguousarray(np.atleast_2d(x), dtype=np.float32)
    order = np.lexsort(a.T[::-1])
    return np.ascontiguousarray(a[order]).tobytes()


def molecule_key(obj: sro.MolObjects) -> int:
    """Deterministic label-invariant 63-bit key of one molecule object."""
    h = hashlib.sha256()
    h.update(_sorted_rows_bytes(obj.atom_feat))
    h.update(_sorted_rows_bytes(obj.bond_feat))
    h.update(_sorted_rows_bytes(obj.junc_feat))
    h.update(np.ascontiguousarray(obj.global_feat, dtype=np.float32).tobytes())
    h.update(np.asarray([obj.n_atoms, obj.bond_feat.shape[0], obj.junc_feat.shape[0]], dtype=np.int64).tobytes())
    return int.from_bytes(h.digest()[:8], "big") & ((1 << 63) - 1)


def molecule_keys(objs: Sequence[sro.MolObjects]) -> np.ndarray:
    return np.asarray([molecule_key(o) for o in objs], dtype=np.int64)


def attach_mol_keys(batch: Mapping[str, torch.Tensor], keys: Sequence[int] | np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
    out = dict(batch)
    out["mol_key"] = torch.as_tensor(np.asarray(keys, dtype=np.int64))
    return out


def batch_objects_with_keys(objs: Sequence[sro.MolObjects], keys: Sequence[int] | np.ndarray) -> dict[str, torch.Tensor]:
    return attach_mol_keys(sro.batch_objects(objs), keys)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class BJOCSourceOwnership(src.BJOC):
    """One-shot BJOC with a functional ownership grouping switch + global block."""

    def __init__(
        self,
        config: OwnershipConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        mode: str = MODE_N,
    ) -> None:
        super().__init__(config, global_mean, global_scale, organization_mode=src.NULL)
        if mode not in MODES:
            raise ValueError(f"unknown ownership mode {mode!r}")
        self.group_mode = mode
        self.global_content = src._MLP(GLOBAL_CONTENT_IN, GLOBAL_CONTENT_HIDDEN, GLOBAL_CONTENT_OUT, 1, config.dropout)
        gen = torch.Generator().manual_seed(JUNC_KEY_SEED)
        self.register_buffer("junc_key_coef", torch.randn(96 + sro.JUNC_FEAT_DIM, generator=gen, dtype=torch.float64), persistent=False)
        gen = torch.Generator().manual_seed(BOND_KEY_SEED)
        self.register_buffer(
            "bond_key_coef",
            torch.randn(sro.BOND_FEAT_DIM + 2 * sro.ATOM_FEAT_DIM + 1, generator=gen, dtype=torch.float64),
            persistent=False,
        )

    # -- label-invariant scalar keys ----------------------------------------
    def _junction_key(self, junc_feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([w.to(torch.float64), junc_feat.to(torch.float64)], dim=1)
        return (x * self.junc_key_coef.to(x.device)).sum(-1)

    def _bond_key(
        self,
        bond_feat: torch.Tensor,
        au: torch.Tensor,
        av: torch.Tensor,
        n_e: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                bond_feat.to(torch.float64),
                (au + av).to(torch.float64),
                (au - av).abs().to(torch.float64),
                n_e.unsqueeze(1).to(torch.float64),
            ],
            dim=1,
        )
        return (x * self.bond_key_coef.to(x.device)).sum(-1)

    def bond_descriptor(self, batch: Mapping[str, torch.Tensor], n_e: torch.Tensor) -> torch.Tensor:
        """Concatenated label-invariant per-bond descriptor used for the sham key."""
        idx = batch["bond_index"]
        au = batch["atom_feat"][idx[:, 0]]
        av = batch["atom_feat"][idx[:, 1]]
        return torch.cat([batch["bond_feat"], au + av, (au - av).abs(), n_e.unsqueeze(1)], dim=1)

    # -- grouping ------------------------------------------------------------
    def group_to_bonds_ownership(
        self,
        w: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == MODE_N:
            return src.BJOC.group_to_bonds(self, w, batch, src.NULL)
        if mode == MODE_O:
            return src.BJOC.group_to_bonds(self, w, batch, src.ORGANIZED)
        if mode == MODE_S:
            return self._group_shuffled(w, batch)
        raise ValueError(f"unknown ownership mode {mode!r}")

    def _group_shuffled(
        self,
        w: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Count-matched deterministic draw from the molecule's own junction pool.

        For every bond ``e`` with ``n_e`` incident junctions, draw ``n_e``
        junction vectors (with replacement) from the molecule's junction pool
        ordered by a label-invariant key.  Per-bond counts and the vector format
        are identical to the true-ownership arm; the incidence is destroyed.
        """
        junc_bond = batch["junc_bond"]
        bond_batch = batch["bond_batch"]
        junc_batch = batch["junc_batch"]
        n_bonds = int(batch["bond_feat"].shape[0])
        n_graphs = int(batch["n_graphs"])
        device = w.device
        dtype = w.dtype

        n_e = torch.zeros(n_bonds, device=device, dtype=torch.float32)
        n_e.index_add_(0, junc_bond.reshape(-1), torch.ones_like(junc_bond.reshape(-1), dtype=torch.float32))

        mol_key = batch.get("mol_key")
        if mol_key is None:
            mol_key = torch.zeros(n_graphs, dtype=torch.int64, device=device)
        mol_key = mol_key.to(device=device, dtype=torch.int64)

        jkey = self._junction_key(batch["junc_feat"], w)
        idx = batch["bond_index"]
        au = batch["atom_feat"][idx[:, 0]]
        av = batch["atom_feat"][idx[:, 1]]
        bkey = self._bond_key(batch["bond_feat"], au, av, n_e).view(torch.int64)

        W = torch.zeros((n_bonds, w.shape[1]), device=device, dtype=dtype)
        V = torch.zeros_like(W)

        # Canonical pool order per molecule: stable sort by key, then stable sort
        # by molecule -> lexicographic (molecule, key) with original index as the
        # final tie-break, matching the per-graph `argsort(jkey[jm], stable=True)`.
        order = torch.argsort(jkey, stable=True)
        order = order[torch.argsort(junc_batch[order], stable=True)]
        wcan_all = w[order]
        nj_per_mol = torch.bincount(junc_batch, minlength=n_graphs)
        starts = torch.cumsum(nj_per_mol, 0) - nj_per_mol

        counts_long = n_e.to(torch.int64)
        e_idx = torch.repeat_interleave(torch.arange(n_bonds, device=device), counts_long)
        if int(e_idx.numel()) == 0:
            return W, V, n_e
        k_idx = torch.arange(e_idx.numel(), device=device, dtype=torch.int64) - torch.repeat_interleave(
            torch.cumsum(counts_long, 0) - counts_long, counts_long
        )
        h = (mol_key[bond_batch[e_idx]] * MIX_MUL_A) ^ (bkey[e_idx] * MIX_MUL_B) ^ (k_idx * 2654435761)
        h = torch.bitwise_and(h, MIX_MASK)
        r = torch.remainder(h, nj_per_mol[bond_batch[e_idx]]).to(torch.int64)
        drawn = wcan_all[starts[bond_batch[e_idx]] + r]
        W.index_add_(0, e_idx, drawn)
        V.index_add_(0, e_idx, drawn * drawn)
        return W, V, n_e

    # -- forward -------------------------------------------------------------
    def graph_moments(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_graphs = int(batch["n_graphs"])
        junc_batch = batch["junc_batch"]
        w_sum = torch.zeros((n_graphs, w.shape[1]), device=w.device, dtype=w.dtype)
        w_sum.index_add_(0, junc_batch, w)
        w2_sum = torch.zeros_like(w_sum)
        w2_sum.index_add_(0, junc_batch, w * w)
        counts = torch.bincount(junc_batch, minlength=n_graphs).to(w.dtype).clamp(min=1.0).unsqueeze(1)
        return w_sum / counts, w2_sum / counts, counts

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        mode: str | None = None,
        zero_local: bool = False,
        p_override: torch.Tensor | None = None,
        zero_bonds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.group_mode if mode is None else mode
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        W, V, n_b = self.group_to_bonds_ownership(w, batch, mode)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]],
            dim=1,
        )
        c = self.group_composer(compose_in)
        if zero_bonds is not None:
            c = c.clone()
            c[zero_bonds] = 0.0
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        if zero_local:
            z = torch.zeros_like(z)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        mu, nu, n_junc = self.graph_moments(w, batch)
        gc_in = torch.cat([mu, nu, torch.log1p(n_junc)], dim=1)
        p = self.global_content(gc_in)
        if p_override is not None:
            p = p_override.to(device=p.device, dtype=p.dtype)
        reader_in = torch.cat([z, a_sum, a_sq, g, p], dim=1)
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "mu": mu,
            "nu": nu,
            "n_junc": n_junc,
            "p_global": p,
            "reader_in": reader_in,
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        return_features: bool = False,
        mode: str | None = None,
        **audit: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(batch, organization_mode, mode=mode, **audit)
        pred = self.reader(feats["reader_in"]).view(-1)
        if return_features:
            return pred, feats
        return pred


# ---------------------------------------------------------------------------
# construction from a shared init
# ---------------------------------------------------------------------------

OWNERSHIP_EXPECTED_PARAMETERS = int(OwnershipConfig().inventory()["total"])


def build_ownership_from_init(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    mode: str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, BJOCSourceOwnership, dict[str, Any]]:
    """Build (base scratch P3 init, ownership model) from one shared init.

    Every base parameter equals the historical scratch BJOC-Null init for the
    seed; the reader first layer is widened with exact zero columns for the
    global content block, so at init ``model == base`` as a black-box function.
    """
    if mode not in MODES:
        raise SystemExit(f"unknown ownership mode {mode!r}")
    base = bp.build_bjoc_null(seed, norm, device)
    scratch_full = src.state_dict_hash(base)
    scratch_group = mp.group_state_hash(base)
    scratch_encoder = bp.encoder_group_hash(base)

    cfg = OwnershipConfig()
    torch.manual_seed(int(seed) + 7919)
    model = BJOCSourceOwnership(
        cfg,
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        mode=mode,
    )
    model.atom_encoder.load_state_dict(base.atom_encoder.state_dict())
    model.bond_encoder.load_state_dict(base.bond_encoder.state_dict())
    model.junction_encoder.load_state_dict(base.junction_encoder.state_dict())
    model.group_composer.load_state_dict(base.group_composer.state_dict())
    _expand_reader_first_layer(base.reader, model.reader)
    model.to(device)
    if ssl_state is not None:
        mp.load_encoder_state(model, ssl_state, device)

    audit = parameter_audit_ownership(cfg, model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != OWNERSHIP_EXPECTED_PARAMETERS:
        raise SystemExit(f"ownership parameter audit failed: {audit}")
    encoder_used = bp.encoder_group_hash(model)
    info = {
        "mode": mode,
        "mode_name": MODE_NAMES[mode],
        "base_scratch_init_full_sha256": scratch_full,
        "base_scratch_init_group_state_sha256": scratch_group,
        "base_scratch_init_encoder_sha256": scratch_encoder,
        "base_encoder_used_sha256": encoder_used,
        "ownership_parameters": OWNERSHIP_EXPECTED_PARAMETERS,
        "added_parameters_vs_p3": OWNERSHIP_EXPECTED_PARAMETERS - bp.EXPECTED_PARAMETERS,
        "reader_first_layer_zero_columns": True,
    }
    return base, model, info


def _expand_reader_first_layer(base_reader: src._MLP, new_reader: src._MLP) -> None:
    """Copy a 321-dim reader into the 417-dim reader with zero new columns."""
    old = base_reader.lins[0]
    new = new_reader.lins[0]
    assert new.weight.shape[1] == old.weight.shape[1] + GLOBAL_CONTENT_OUT
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.weight.shape[1]].copy_(old.weight.detach())
        new.bias.copy_(old.bias.detach())
        for i in range(1, len(base_reader.lins)):
            new_reader.lins[i].load_state_dict(base_reader.lins[i].state_dict())


def parameter_audit_ownership(config: OwnershipConfig, model: nn.Module) -> dict[str, Any]:
    expected = config.inventory()
    actual = int(sum(p.numel() for p in model.parameters()))
    return {
        "expected": expected,
        "actual_parameters": actual,
        "parameter_exact": actual == expected["total"],
    }


def model_state_hash(model: nn.Module) -> str:
    """Full parameter hash incl. the new modules (buffers with persistent=False skipped)."""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# prediction helpers
# ---------------------------------------------------------------------------


def _mv(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_model_ownership(
    model: BJOCSourceOwnership,
    objs: Sequence[sro.MolObjects],
    keys: Sequence[int] | np.ndarray,
    device: torch.device | str,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), batch_size):
            chunk = list(objs[i : i + batch_size])
            batch = _mv(batch_objects_with_keys(chunk, keys[i : i + batch_size]), torch.device(device))
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# supervised training loop (line-for-line R3 copy, model builder parameterised)
# ---------------------------------------------------------------------------


def train_supervised_ownership(
    *,
    arm: str,
    seed: int,
    recipe_key: str,
    train_objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: torch.device,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    smoke_epochs: int | None = None,
    log_fn: Callable[[str], None] = log,
) -> dict[str, Any]:
    mode = ARM_MODES[arm]
    if recipe_key not in bp.RECIPES:
        raise SystemExit(f"unknown recipe {recipe_key!r}")
    recipe = bp.RECIPES[recipe_key]
    if ssl_state is None:
        raise SystemExit("the ownership round trains only from the reused masked-SSL init")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")

    base, model, info = build_ownership_from_init(seed, norm, device, mode, ssl_state)
    init_full_sha = info["base_scratch_init_full_sha256"]
    init_group_sha = info["base_scratch_init_group_state_sha256"]
    init_encoder_sha = info["base_scratch_init_encoder_sha256"]
    encoder_used_sha = info["base_encoder_used_sha256"]
    audit = parameter_audit_ownership(OwnershipConfig(), model)

    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    keys = molecule_keys(train_objs)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"]))
    elif recipe["schedule"] != "constant":
        raise SystemExit(f"unknown schedule {recipe['schedule']!r}")

    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    import time as _time

    started = _time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = _time.perf_counter()
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = _mv(batch_objects_with_keys([train_objs[i] for i in indices], keys[indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), bp.GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > bp.GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
                lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        total_mol += n_mol
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": _time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = _time.perf_counter() - started

    members, soup = _last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_model_ownership(model, train_objs, keys, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "mode": mode,
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {bp.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_state": soup_state,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "lr0": float(recipe["lr"]),
            "lr_min": None if recipe["lr_min"] is None else float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "init_full_sha256": init_full_sha,
        "init_group_state_sha256": init_group_sha,
        "init_encoder_sha256": init_encoder_sha,
        "encoder_used_sha256": encoder_used_sha,
        "mode_info": info,
        "smoke": bool(smoke_epochs is not None),
        "full_recipe_epochs": int(recipe["epochs"]),
        "parameter_audit": audit,
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "full_hash_independent_recomputation": bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
            "full_matches": order_hash.hexdigest() == bp.batch_order_hash(seed, n, bp.BATCH_SIZE, epochs),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "lr_curve": curve,
        "wall_seconds": wall,
        "molecules_seen": int(total_mol),
        "device": str(device),
        "torch": torch.__version__,
        "labels_used": "fit rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    assert 0 < len(states) <= bp.SOUP_LAST, len(states)
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0].keys()}
    return members, soup


# ---------------------------------------------------------------------------
# Phase-0 primitives (frozen-model source specificity)
# ---------------------------------------------------------------------------

ATOM_TYPE_SLOTS = sro.ATOM_CATEGORIES


def atom_type(obj: sro.MolObjects, atom: int) -> int:
    return int(np.argmax(obj.atom_feat[atom, :ATOM_TYPE_SLOTS]))


def junction_signature(obj: sro.MolObjects, j: int) -> tuple:
    """Deterministic label-free chemical signature of one Junction."""
    i, c, k = (int(v) for v in obj.junc_index[j])
    bt = [int(obj.bond_type[b]) for b in obj.junc_bond[j]]
    n_ring_bonds = int(round(float(obj.junc_feat[j, 3])))
    center_in_ring = int(round(float(obj.atom_feat[c, ATOM_TYPE_SLOTS + sro.ATOM_DEGREE_SLOTS])))
    return (
        atom_type(obj, c),
        tuple(sorted((atom_type(obj, i), atom_type(obj, k)))),
        tuple(sorted(bt)),
        center_in_ring,
        n_ring_bonds,
    )


def center_degree(obj: sro.MolObjects, j: int) -> int:
    return int(round(float(obj.junc_feat[j, 0])))


def signature_corpus_counts(objs: Sequence[sro.MolObjects]) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for obj in objs:
        for j in range(obj.junc_index.shape[0]):
            s = junction_signature(obj, j)
            counts[s] = counts.get(s, 0) + 1
    return counts


def dev_ranking(w: torch.Tensor) -> torch.Tensor:
    """Encoder-space deviation from the molecule mean."""
    return (w - w.mean(0, keepdim=True)).norm(dim=1)


def rarity_ranking(obj: sro.MolObjects, counts: Mapping[tuple, int], total: int) -> np.ndarray:
    out = np.zeros(obj.junc_index.shape[0], dtype=np.float64)
    for j in range(obj.junc_index.shape[0]):
        p = counts.get(junction_signature(obj, j), 0) / max(int(total), 1)
        out[j] = -math.log2(max(p, 1e-12))
    return out


def matched_controls(
    obj: sro.MolObjects,
    j: int,
    dev: np.ndarray,
    signature: tuple,
    degree: int,
    in_ring: int,
    exclude: set[int],
    n_controls: int,
    rng: np.random.Generator,
    allow_dev_fallback: bool = True,
) -> list[int]:
    """Same-molecule control junctions matched on structure and dev-decile."""
    n_j = int(obj.junc_index.shape[0])
    lo = np.floor(dev * 10.0).clip(0, 9)
    same: list[int] = []
    same_no_dev: list[int] = []
    for jp in range(n_j):
        if jp == j or jp in exclude:
            continue
        if junction_signature(obj, jp) != signature:
            continue
        if center_degree(obj, jp) != degree:
            continue
        if junction_signature(obj, jp)[3] != in_ring:
            continue
        same_no_dev.append(jp)
        if lo[jp] == lo[j]:
            same.append(jp)
    pool = same if same else (same_no_dev if allow_dev_fallback else [])
    if not pool:
        return []
    if len(pool) <= n_controls:
        return sorted(pool)
    return sorted(int(v) for v in rng.choice(np.asarray(pool), size=n_controls, replace=False))


# ---------------------------------------------------------------------------
# mechanism audit helpers (CPU, trained models)
# ---------------------------------------------------------------------------


def per_bond_attribution(pack: Mapping[str, Any]) -> np.ndarray:
    return pack["c"].detach().norm(dim=1).numpy()


def local_pathway_delta(
    model: BJOCSourceOwnership,
    batch: Mapping[str, torch.Tensor],
    *,
    zero_local: bool = True,
) -> np.ndarray:
    with torch.no_grad():
        natural = model(batch)
        modified = model(batch, zero_local=zero_local)
    return (modified - natural).numpy()


def clone_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def forbidden_module_scan(model: nn.Module) -> list[str]:
    hits = []
    for name, module in model.named_modules():
        for token in src.FORBIDDEN_TOKENS:
            if token.lower() in type(module).__name__.lower() or token.lower() in name.lower():
                hits.append(f"{name}:{type(module).__name__}")
    return hits
