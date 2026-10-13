"""Architecture-breakthrough models for cscl-bjoc-architecture-breakthrough-v1.

Protocol: ``configs/bjoc_architecture_breakthrough_v1.yaml``.
Track: ``tracks/structure-contribution``.

Two pre-registered one-shot architecture candidates on top of the frozen
BJOC-Null + masked-SSL + R3 system (413,505 params):

* **GLRC** (Global-Local Residual Composer).  Keep the Null composer as the
  base and add a single per-molecule, structure-conditioned gate on the
  *local-incidence residual*::

      dW_b = W_local_b - W_null_b ,  dV_b = V_local_b - V_null_b
      W'_b = W_null_b + lambda(G) * dW_b
      V'_b = V_null_b + lambda(G) * dV_b
      lambda(G) = tanh(theta0 + theta^T g_G)

  ``g_G`` is the existing Fit-only standardised 33-dim global structure
  feature (the same vector the reader already receives).  ``theta0`` and
  ``theta`` are zero-initialised, so ``lambda == 0`` exactly at init and the
  initial forward is bit-compatible with BJOC-Null.  ``lambda`` is constant
  inside one molecule, so the graph-level first/second moments of the local
  and null statistics are preserved: ``sum_b dW_b = sum_b dV_b = 0`` for every
  molecule.  Total parameters: 413,539 (+34).

* **DLP** (Distance-conditioned Low-rank Pair pooling).  From the existing
  atom encodings ``a_i`` project ``h_i = P a_i`` (``P: 32 x 96``, no bias) and
  pool elementwise products over unordered atom pairs, split by shortest-path
  graph distance into three frozen buckets (d = 3, d = 4, d >= 5)::

      p_k(G) = (1 / max(1, N_k)) * sum_{i<j, d(i,j) in D_k} (h_i * h_j)

  The three 32-dim vectors are appended to the frozen reader input; the new
  reader first-layer columns are zero-initialised, so the initial forward is
  identical to BJOC-Null.  Rank, buckets and aggregation are frozen; there is
  no attention, no message passing and no pair-embedding update.  Total
  parameters: 441,153 (+27,648).

Everything in this module is label-blind except the supervised training loop,
which uses the raw ``y`` of Fit rows only.  Official test is never referenced.
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
import time
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

PROTOCOL_ID = "cscl-bjoc-architecture-breakthrough-v1"

#: frozen group-aware split salts (both fixed before any GPU work)
SPLIT_SALTS = {
    "C": "cscl-bjoc-perfmech-long-v1-C",
    "D": "cscl-bjoc-arch-breakthrough-v1-repl",
}

#: GLRC gate definition (single frozen form; no search)
GLRC_GATE_LAMBDA = "tanh(theta0 + theta^T g_G)"
GLRC_PARAMETERS = bp.EXPECTED_PARAMETERS + sro.GLOBAL_DIM + 1  # 413,539

#: DLP frozen constants (rank and buckets are NOT searched in this round)
DLP_RANK = 32
DLP_N_BUCKETS = 3
DLP_BUCKET_LABELS = ("d3", "d4", "d5plus")
DLP_PARAMETERS = bp.EXPECTED_PARAMETERS + DLP_RANK * 96 + DLP_N_BUCKETS * DLP_RANK * 256  # 441,153

#: frozen monitor-gate constants
GLRC_DELTA_THRESHOLD = -0.004
DLP_SEED0_DELTA_THRESHOLD = -0.004
GATE_ORDINARY_TOL = 0.0
GATE_RMSE_TOL = 0.005
GATE_MAX_SINGLE_GAIN_FRACTION = 0.25


# ---------------------------------------------------------------------------
# label-access guard (Monitor / Locked ``.y`` must be unreachable)
# ---------------------------------------------------------------------------


class ForbiddenLabelAccess(RuntimeError):
    """Raised when guarded Monitor/Locked labels are touched."""


class GuardedData:
    """Proxy over one PyG molecule that refuses ``.y`` access."""

    def __init__(self, data: Any, row: int, guard: LabelGuard | None = None) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_row", int(row))
        object.__setattr__(self, "_guard", guard)

    def __getattr__(self, name: str) -> Any:
        if name == "y":
            if self._guard is not None:
                self._guard.access_attempts.append(self._row)
            raise ForbiddenLabelAccess(
                f"row {self._row}: y access is forbidden before the declared label read"
            )
        return getattr(self._data, name)

    def __getitem__(self, key: str) -> Any:
        if key == "y":
            if self._guard is not None:
                self._guard.access_attempts.append(self._row)
            raise ForbiddenLabelAccess(
                f"row {self._row}: y access is forbidden before the declared label read"
            )
        return self._data[key]

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"GuardedData(row={self._row})"


class LabelGuard:
    """Dataset wrapper that blocks ``.y`` for a pre-declared row set.

    Only ``x`` / ``edge_index`` / ``edge_attr`` (and any non-label field) pass
    through.  ``forbidden_rows`` is the union of the split's Monitor and Locked
    rows during preprocessing / SSL / training / prediction.
    """

    def __init__(self, dataset: Any, forbidden_rows: Sequence[int]) -> None:
        self._dataset = dataset
        self._forbidden = set(int(r) for r in forbidden_rows)
        self.access_attempts: list[int] = []

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, row: int) -> Any:
        data = self._dataset[int(row)]
        if int(row) in self._forbidden:
            return GuardedData(data, int(row), self)
        return data

    @property
    def forbidden_rows(self) -> set[int]:
        return set(self._forbidden)


def label_guard_selfcheck() -> dict[str, Any]:
    """Data-free positive control for :class:`LabelGuard`.

    A guarded row must raise on ``.y`` while structure attributes pass; an
    unguarded row returns the raw molecule.
    """

    class _Mol:
        x = np.zeros((2, 1), dtype=np.int64)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0,), dtype=np.int64)
        y = np.asarray([1.0])

    guard = LabelGuard([_Mol(), _Mol()], forbidden_rows=[1])
    ok = {"structure_passthrough": False, "guarded_y_blocked": False, "unguarded_y_allowed": False}
    d0 = guard[0]
    ok["structure_passthrough"] = int(d0.x.shape[0]) == 2
    ok["unguarded_y_allowed"] = float(d0.y[0]) == 1.0
    try:
        _ = guard[1].y
    except ForbiddenLabelAccess:
        ok["guarded_y_blocked"] = True
    return ok


# ---------------------------------------------------------------------------
# per-molecule distance-pair cache (label-free, precomputed from structure)
# ---------------------------------------------------------------------------


def pair_arrays_for_object(obj: sro.MolObjects) -> dict[str, np.ndarray]:
    """Unordered pairs at shortest-path distance >= 3 with frozen bucket ids.

    Bucket 0 -> d == 3, bucket 1 -> d == 4, bucket 2 -> d >= 5.
    """
    n = int(obj.n_atoms)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for (u, v), bt in zip(np.asarray(obj.bond_index), np.asarray(obj.bond_type)):
        u, v = int(u), int(v)
        adj[u].append((v, int(bt)))
        adj[v].append((u, int(bt)))
    dist = sro._all_pairs_distances(n, adj)
    ii, jj = np.triu_indices(n, k=1)
    d = dist[ii, jj].astype(np.int64)
    keep = d >= 3
    ii, jj, d = ii[keep], jj[keep], d[keep]
    bucket = np.where(d == 3, 0, np.where(d == 4, 1, 2)).astype(np.int64)
    return {"i": ii.astype(np.int64), "j": jj.astype(np.int64), "bucket": bucket}


def build_pair_cache(objs: Sequence[sro.MolObjects]) -> dict[int, dict[str, np.ndarray]]:
    return {int(o.mol_index): pair_arrays_for_object(o) for o in objs}


def build_pairs_for_rows(rows: Sequence[int], ds) -> dict[int, dict[str, np.ndarray]]:
    """Pair cache for a list of dataset rows (structure fields only)."""
    cache: dict[int, dict[str, np.ndarray]] = {}
    for i in rows:
        d = ds[int(i)]
        obj = sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        cache[int(i)] = pair_arrays_for_object(obj)
    return cache


def pair_cache_summary(cache: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    per_bucket = np.zeros(DLP_N_BUCKETS, dtype=np.int64)
    n_pairs = []
    for rec in cache.values():
        b = np.asarray(rec["bucket"], dtype=np.int64)
        per_bucket += np.bincount(b, minlength=DLP_N_BUCKETS)[:DLP_N_BUCKETS]
        n_pairs.append(int(b.shape[0]))
    return {
        "n_molecules": int(len(cache)),
        "n_pairs": int(per_bucket.sum()),
        "pairs_per_bucket": {DLP_BUCKET_LABELS[k]: int(v) for k, v in enumerate(per_bucket.tolist())},
        "pairs_per_molecule_median": float(np.median(n_pairs)) if n_pairs else 0.0,
        "pairs_per_molecule_max": int(max(n_pairs)) if n_pairs else 0,
    }


def build_dlp_batch(
    objects: Sequence[sro.MolObjects],
    pairs_by_index: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    """``sro.batch_objects`` plus flat distance-bucket pair tensors."""
    batch = sro.batch_objects(objects)
    if not objects:
        batch["pair_i"] = torch.zeros(0, dtype=torch.long)
        batch["pair_j"] = torch.zeros(0, dtype=torch.long)
        batch["pair_bucket"] = torch.zeros(0, dtype=torch.long)
        batch["pair_batch"] = torch.zeros(0, dtype=torch.long)
        return batch
    atom_off = 0
    pi, pj, pb, pg = [], [], [], []
    for g, o in enumerate(objects):
        rec = pairs_by_index[int(o.mol_index)]
        ii = np.asarray(rec["i"], dtype=np.int64)
        m = int(ii.shape[0])
        if m:
            pi.append(ii + atom_off)
            pj.append(np.asarray(rec["j"], dtype=np.int64) + atom_off)
            pb.append(np.asarray(rec["bucket"], dtype=np.int64))
            pg.append(np.full(m, int(g), dtype=np.int64))
        atom_off += int(o.n_atoms)
    cat = np.concatenate
    batch["pair_i"] = torch.as_tensor(cat(pi) if pi else np.zeros(0, dtype=np.int64), dtype=torch.long)
    batch["pair_j"] = torch.as_tensor(cat(pj) if pj else np.zeros(0, dtype=np.int64), dtype=torch.long)
    batch["pair_bucket"] = torch.as_tensor(cat(pb) if pb else np.zeros(0, dtype=np.int64), dtype=torch.long)
    batch["pair_batch"] = torch.as_tensor(cat(pg) if pg else np.zeros(0, dtype=np.int64), dtype=torch.long)
    return batch


# ---------------------------------------------------------------------------
# model configs + models
# ---------------------------------------------------------------------------


class GLRCConfig(src.BJOCConfig):
    """Frozen BJOC widths + one 33->1 tanh gate (reader width unchanged)."""

    def inventory(self) -> dict[str, int]:
        inv = super().inventory()
        inv["lambda_gate"] = int(sro.GLOBAL_DIM) + 1
        inv["total"] += inv["lambda_gate"]
        return inv


class DLPConfig(src.BJOCConfig):
    """Frozen BJOC widths + 3 x 32 appended reader columns + pair projection."""

    dlp_rank: int = DLP_RANK

    @property
    def reader_in(self) -> int:
        return super().reader_in + DLP_N_BUCKETS * self.dlp_rank

    def inventory(self) -> dict[str, int]:
        inv = super().inventory()
        inv["dlp_pair_proj"] = self.dlp_rank * self.atom_out
        inv["total"] += inv["dlp_pair_proj"]
        return inv


class BJOCGLRC(src.BJOC):
    """BJOC-Null plus a per-molecule tanh gate on the local-incidence residual."""

    def __init__(
        self,
        config: GLRCConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        organization_mode: str = src.NULL,
    ) -> None:
        super().__init__(config, global_mean, global_scale, organization_mode)
        self.lambda_gate = nn.Linear(int(sro.GLOBAL_DIM), 1)
        with torch.no_grad():
            self.lambda_gate.weight.zero_()
            self.lambda_gate.bias.zero_()

    def lambda_of(self, batch: Mapping[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        g = (batch["global_feat"].to(dtype) - self.global_mean) / self.global_scale
        return torch.tanh(self.lambda_gate(g)).view(-1)

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        lambda_override: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        W_null, V_null, n_b = self.group_to_bonds(w, batch, src.NULL)
        W_loc, V_loc, _ = self.group_to_bonds(w, batch, src.ORGANIZED)
        if lambda_override is None:
            lam = self.lambda_of(batch, a.dtype)
        elif isinstance(lambda_override, torch.Tensor):
            lam = lambda_override.to(dtype=a.dtype, device=a.device).view(-1)
        else:
            lam = torch.full((int(batch["n_graphs"]),), float(lambda_override), dtype=a.dtype, device=a.device)
        dW = W_loc - W_null
        dV = V_loc - V_null
        lam_b = lam[batch["bond_batch"]].unsqueeze(1)
        W = W_null + lam_b * dW
        V = V_null + lam_b * dV
        compose_in = torch.cat([b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "reader_in": reader_in,
            "lambda": lam,
            "dW": dW,
            "dV": dV,
        }


    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
        return_features: bool = False,
        lambda_override: torch.Tensor | float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.forward_features(batch, organization_mode, lambda_override)
        pred = self.reader(feats["reader_in"]).view(-1)
        if return_features:
            return pred, feats
        return pred


class BJOCDLP(src.BJOC):
    """BJOC-Null with distance-conditioned low-rank pair pooling."""

    def __init__(
        self,
        config: DLPConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        organization_mode: str = src.NULL,
    ) -> None:
        super().__init__(config, global_mean, global_scale, organization_mode)
        self.pair_proj = nn.Linear(self.config.atom_out, int(getattr(config, "dlp_rank", DLP_RANK)), bias=False)

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        feats = super().forward_features(batch, organization_mode)
        h = self.pair_proj(feats["a"])
        pi, pj = batch["pair_i"], batch["pair_j"]
        prod = h[pi] * h[pj]
        n_graphs = int(batch["n_graphs"])
        rank = int(h.shape[1])
        idx = batch["pair_batch"] * DLP_N_BUCKETS + batch["pair_bucket"]
        agg = torch.zeros((n_graphs * DLP_N_BUCKETS, rank), device=h.device, dtype=h.dtype)
        agg.index_add_(0, idx, prod)
        counts = torch.zeros(n_graphs * DLP_N_BUCKETS, device=h.device, dtype=h.dtype)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=h.dtype))
        counts = counts.clamp(min=1.0).unsqueeze(1)
        p = (agg / counts).view(n_graphs, DLP_N_BUCKETS * rank)
        feats["dlp"] = p
        feats["reader_in"] = torch.cat([feats["reader_in"], p], dim=1)
        return feats


# ---------------------------------------------------------------------------
# deterministic builders (base init must match P3 exactly)
# ---------------------------------------------------------------------------


def _copy_module(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _copy_base_parameters(target: src.BJOC, base: src.BJOC) -> None:
    _copy_module(target.atom_encoder, base.atom_encoder)
    _copy_module(target.bond_encoder, base.bond_encoder)
    _copy_module(target.junction_encoder, base.junction_encoder)
    _copy_module(target.group_composer, base.group_composer)
    with torch.no_grad():
        target.global_mean.copy_(base.global_mean)
        target.global_scale.copy_(base.global_scale)


def _expand_reader_first_layer(base_reader: src._MLP, new_reader: src._MLP) -> None:
    """Copy the base reader into a wider reader with zero new columns.

    New DLP features are appended at the END of the reader input, so copied
    old columns keep their indices.
    """
    with torch.no_grad():
        w_old = base_reader.lins[0].weight.detach()
        w_new = new_reader.lins[0].weight.detach()
        w_new.zero_()
        w_new[:, : w_old.shape[1]].copy_(w_old)
        new_reader.lins[0].bias.copy_(base_reader.lins[0].bias.detach())
        for i in range(1, len(base_reader.lins)):
            new_reader.lins[i].load_state_dict(base_reader.lins[i].state_dict())


def base_parameter_hash(model: src.BJOC, *, exclude_prefixes: Sequence[str] = ()) -> str:
    """SHA-256 of all parameters except the listed module prefixes."""
    h = hashlib.sha256()
    skip = tuple(exclude_prefixes)
    for key, value in sorted(model.state_dict().items()):
        if any(key.startswith(p) for p in skip):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def build_organized_from_init(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, src.BJOC, dict[str, Any]]:
    """Base-model parity holder + the same init with true junction ownership.

    ``organization_mode`` never touches initialisation, so the organized model
    shares the exact P3 scratch init; this is asserted, not assumed.
    """
    base = bp.build_bjoc_null(seed, norm, device)
    scratch_full = src.state_dict_hash(base)
    scratch_group = mp.group_state_hash(base)
    scratch_encoder = bp.encoder_group_hash(base)
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg,
        seed=seed,
        organization_mode=src.ORGANIZED,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )
    if src.state_dict_hash(bp.build_bjoc_null(seed, norm, device)) != src.state_dict_hash(model):
        raise SystemExit("organized init does not match the BJOC-Null scratch init")
    if ssl_state is not None:
        mp.load_encoder_state(model, ssl_state, device)
    info = {
        "base_scratch_init_full_sha256": scratch_full,
        "base_scratch_init_group_state_sha256": scratch_group,
        "base_scratch_init_encoder_sha256": scratch_encoder,
        "base_encoder_used_sha256": bp.encoder_group_hash(model),
        "base_parameter_hash": base_parameter_hash(model),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "new_parameters": 0,
        "organization_mode": src.ORGANIZED,
    }
    return base, model, info


def build_glrc_from_init(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, BJOCGLRC, dict[str, Any]]:
    """Build (base P3-init model, GLRC model) from one shared scratch init."""
    base = bp.build_bjoc_null(seed, norm, device)
    scratch_full = src.state_dict_hash(base)
    scratch_group = mp.group_state_hash(base)
    scratch_encoder = bp.encoder_group_hash(base)
    if ssl_state is not None:
        mp.load_encoder_state(base, ssl_state, device)
    encoder_used = bp.encoder_group_hash(base)

    cfg = GLRCConfig()
    model = BJOCGLRC(
        cfg,
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        src.NULL,
    )
    _copy_base_parameters(model, base)
    _copy_module(model.reader, base.reader)
    model.to(device)
    audit = src.parameter_audit(cfg, model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != GLRC_PARAMETERS:
        raise SystemExit(f"GLRC parameter audit failed: {audit}")
    info = {
        "base_scratch_init_full_sha256": scratch_full,
        "base_scratch_init_group_state_sha256": scratch_group,
        "base_scratch_init_encoder_sha256": scratch_encoder,
        "base_encoder_used_sha256": encoder_used,
        "base_parameter_hash": base_parameter_hash(model, exclude_prefixes=("lambda_gate.",)),
        "hash_excludes": ["lambda_gate.*"],
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "new_parameters": int(GLRC_PARAMETERS - bp.EXPECTED_PARAMETERS),
        "gate": GLRC_GATE_LAMBDA,
    }
    return base, model, info


def build_dlp_from_init(
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, BJOCDLP, dict[str, Any]]:
    """Build (base P3-init model, DLP model) from one shared scratch init.

    ``pair_proj`` is initialised from an independent ``torch.Generator`` so the
    base parameter RNG stream is untouched; the new reader columns are exactly
    zero and the initial function equals BJOC-Null.
    """
    base = bp.build_bjoc_null(seed, norm, device)
    scratch_full = src.state_dict_hash(base)
    scratch_group = mp.group_state_hash(base)
    scratch_encoder = bp.encoder_group_hash(base)
    if ssl_state is not None:
        mp.load_encoder_state(base, ssl_state, device)
    encoder_used = bp.encoder_group_hash(base)

    cfg = DLPConfig()
    model = BJOCDLP(
        cfg,
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        src.NULL,
    )
    _copy_base_parameters(model, base)
    _expand_reader_first_layer(base.reader, model.reader)
    gen = torch.Generator().manual_seed(int(seed) + 7777)
    bound = 1.0 / math.sqrt(float(cfg.atom_out))
    with torch.no_grad():
        w = torch.empty(cfg.dlp_rank, cfg.atom_out)
        w.uniform_(-bound, bound, generator=gen)
        model.pair_proj.weight.copy_(w)
    model.to(device)
    audit = src.parameter_audit(cfg, model)
    if not audit["parameter_exact"] or audit["actual_parameters"] != DLP_PARAMETERS:
        raise SystemExit(f"DLP parameter audit failed: {audit}")
    info = {
        "base_scratch_init_full_sha256": scratch_full,
        "base_scratch_init_group_state_sha256": scratch_group,
        "base_scratch_init_encoder_sha256": scratch_encoder,
        "base_encoder_used_sha256": encoder_used,
        "base_parameter_hash": base_parameter_hash(model, exclude_prefixes=("pair_proj.", "reader.lins.0.weight")),
        "hash_excludes": ["pair_proj.*", "reader.lins.0.weight"],
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "new_parameters": int(DLP_PARAMETERS - bp.EXPECTED_PARAMETERS),
        "rank": int(cfg.dlp_rank),
        "buckets": list(DLP_BUCKET_LABELS),
        "pair_proj_init": "uniform(-1/sqrt(atom_out), +1/sqrt(atom_out)) from Generator(seed+7777)",
    }
    return base, model, info


def build_arm_model(
    arm: str,
    seed: int,
    norm: Mapping[str, Any],
    device: torch.device | str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> tuple[src.BJOC, src.BJOC, dict[str, Any]]:
    """Dispatch to the frozen builder for one supervised arm."""
    kind = ARM_KINDS[arm]
    if kind == "base":
        base = bp.build_bjoc_null(seed, norm, device)
        scratch_full = src.state_dict_hash(base)
        scratch_group = mp.group_state_hash(base)
        scratch_encoder = bp.encoder_group_hash(base)
        if ssl_state is not None:
            mp.load_encoder_state(base, ssl_state, device)
        info = {
            "base_scratch_init_full_sha256": scratch_full,
            "base_scratch_init_group_state_sha256": scratch_group,
            "base_scratch_init_encoder_sha256": scratch_encoder,
            "base_encoder_used_sha256": bp.encoder_group_hash(base),
            "base_parameter_hash": base_parameter_hash(base),
            "parameter_count": int(sum(p.numel() for p in base.parameters())),
            "new_parameters": 0,
            "organization_mode": src.NULL,
        }
        return base, base, info
    if kind == "organized":
        return build_organized_from_init(seed, norm, device, ssl_state)
    if kind == "glrc":
        return build_glrc_from_init(seed, norm, device, ssl_state)
    if kind == "dlp":
        return build_dlp_from_init(seed, norm, device, ssl_state)
    raise SystemExit(f"unknown arm {arm!r}")


#: arm registry for this round (P3 is the matched baseline; others are candidates)
ARM_KINDS: dict[str, str] = {
    "P3": "base",
    "GLRC": "glrc",
    "DLP": "dlp",
    "ORGANIZED": "organized",
}


# ---------------------------------------------------------------------------
# prediction helper (DLP needs pair tensors)
# ---------------------------------------------------------------------------


def predict_model_arch(
    model: src.BJOC,
    objs: Sequence[sro.MolObjects],
    device: torch.device | str,
    *,
    model_kind: str,
    pairs_by_index: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> np.ndarray:
    model.eval()
    device = torch.device(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            chunk = list(objs[i : i + 256])
            if model_kind == "dlp":
                if pairs_by_index is None:
                    raise SystemExit("dlp prediction requires a pair cache")
                batch = build_dlp_batch(chunk, pairs_by_index)
            else:
                batch = sro.batch_objects(chunk)
            out.append(model(bp._mv(batch, device)).detach().cpu().numpy())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# supervised training loop (copy of the frozen bp loop with model/batch hooks)
# ---------------------------------------------------------------------------


def train_supervised_arch(
    *,
    arm: str,
    seed: int,
    recipe_key: str,
    init: str,
    train_objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: torch.device,
    model_kind: str,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    pairs_by_index: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    smoke_epochs: int | None = None,
    log_fn: Callable[[str], None] = bp.log,
) -> dict[str, Any]:
    """One supervised trajectory; loop body identical to ``bp.train_supervised``.

    Only the model construction and batch assembly are parameterised.  The
    optimizer, schedule, shuffle generator, clipping, L1 loss and last-5 soup
    use the frozen ``bp`` primitives, so a base arm trained here is
    bit-comparable with one trained by ``bp.train_supervised``.
    """
    if recipe_key not in bp.RECIPES:
        raise SystemExit(f"unknown recipe {recipe_key!r}")
    recipe = bp.RECIPES[recipe_key]
    if init not in ("scratch", "ssl"):
        raise SystemExit(f"unknown init {init!r}")
    if init == "ssl" and ssl_state is None:
        raise SystemExit("init=ssl requires a pretrained encoder state")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    if model_kind == "dlp" and pairs_by_index is None:
        raise SystemExit("dlp training requires a pair cache")

    _base, model, build_info = build_arm_model(arm, seed, norm, device, ssl_state if init == "ssl" else None)
    if init == "ssl" and build_info["base_encoder_used_sha256"] == build_info["base_scratch_init_encoder_sha256"]:
        raise SystemExit("init=ssl did not change the encoder state")
    init_full_sha = build_info["base_scratch_init_full_sha256"]
    init_group_sha = build_info["base_scratch_init_group_state_sha256"]
    init_encoder_sha = build_info["base_scratch_init_encoder_sha256"]
    encoder_used_sha = bp.encoder_group_hash(model)
    audit = src.parameter_audit(
        GLRCConfig() if model_kind == "glrc" else (DLPConfig() if model_kind == "dlp" else src.BJOCConfig()),
        model,
    )
    expected = {
        "base": bp.EXPECTED_PARAMETERS,
        "organized": bp.EXPECTED_PARAMETERS,
        "glrc": GLRC_PARAMETERS,
        "dlp": DLP_PARAMETERS,
    }[model_kind]
    if not audit["parameter_exact"] or audit["actual_parameters"] != expected:
        raise SystemExit(f"parameter audit failed: {audit}")

    def make_batch(indices: Sequence[int]) -> dict[str, torch.Tensor]:
        objs = [train_objs[i] for i in indices]
        if model_kind == "dlp":
            return build_dlp_batch(objs, pairs_by_index)
        return sro.batch_objects(objs)

    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
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
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = bp._mv(make_batch(indices), device)
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
        entry = {
            "epoch": epoch,
            "updates": epoch * steps,
            "train_mae": train_mae,
            "lr_first": lr_first,
            "lr_last": lr_last,
            "lr_mean": lr_sum / max(steps, 1),
            "seconds": time.perf_counter() - t_ep,
        }
        if model_kind == "glrc":
            with torch.no_grad():
                grid = bp._mv(make_batch(list(range(min(n, 256)))), device)
                lam = model.lambda_of(grid, next(model.parameters()).dtype)
                entry["lambda_mean"] = float(lam.mean())
                entry["lambda_abs_mean"] = float(lam.abs().mean())
                entry["lambda_min"] = float(lam.min())
                entry["lambda_max"] = float(lam.max())
        curve.append(entry)
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = bp._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_model_arch(model, train_objs, device, model_kind=model_kind, pairs_by_index=pairs_by_index)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "init": init,
        "model_kind": model_kind,
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
            "warmup": "none",
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
        "smoke": bool(smoke_epochs is not None),
        "full_recipe_epochs": int(recipe["epochs"]),
        "build_info": {k: v for k, v in build_info.items() if k != "soup_state"},
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
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "train Fit rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# contrast / gate arithmetic (frozen definitions)
# ---------------------------------------------------------------------------


def compute_contrast(
    new: Mapping[int, np.ndarray],
    ref: Mapping[int, np.ndarray],
    y: np.ndarray,
    *,
    tail_threshold: float = bp.TAIL_THRESHOLD,
    top_k: int = bp.TOP_K_REMOVED,
    bootstrap_n: int = bp.BOOTSTRAP_N,
    bootstrap_seed: int = bp.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    seeds = sorted(new)
    if seeds != sorted(ref):
        raise SystemExit("contrast arm seeds do not match")
    e_new = {s: np.abs(np.asarray(new[s], dtype=np.float64) - y) for s in seeds}
    e_ref = {s: np.abs(np.asarray(ref[s], dtype=np.float64) - y) for s in seeds}
    delta = np.mean([e_new[s] - e_ref[s] for s in seeds], axis=0)
    per_seed = {str(s): float((e_new[s] - e_ref[s]).mean()) for s in seeds}
    ordinal = y > tail_threshold
    worst = np.argsort(-np.maximum(e_new[seeds[0]], e_ref[seeds[0]]))[: int(top_k)]
    keep = np.ones(len(delta), dtype=bool)
    keep[worst] = False
    pos_gain = np.maximum(-delta, 0.0)
    total_gain = float(pos_gain.sum())
    boot = _paired_bootstrap(delta, n=bootstrap_n, seed=bootstrap_seed)
    return {
        "seeds": seeds,
        "mae_new_seed_mean": float(np.mean([e_new[s].mean() for s in seeds])),
        "mae_ref_seed_mean": float(np.mean([e_ref[s].mean() for s in seeds])),
        "delta_seed_mean": float(np.mean([per_seed[str(s)] for s in seeds])),
        "per_seed_delta": per_seed,
        "bootstrap": boot,
        "ordinary_delta": float(delta[ordinal].mean()),
        "tail_delta": float(delta[~ordinal].mean()),
        "tail_n": int((~ordinal).sum()),
        "rmse_new": float(np.sqrt(np.mean(np.mean([(new[s] - y) ** 2 for s in seeds], axis=0)))),
        "rmse_ref": float(np.sqrt(np.mean(np.mean([(ref[s] - y) ** 2 for s in seeds], axis=0)))),
        "top10_removed_delta": float(delta[keep].mean()),
        "frac_molecules_better": float((delta < 0).mean()),
        "max_single_gain_fraction": float(pos_gain.max() / total_gain) if total_gain > 0 else 0.0,
    }


def _paired_bootstrap(delta: np.ndarray, *, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    k = len(delta)
    means = np.empty(int(n), dtype=np.float64)
    for i in range(int(n)):
        idx = rng.integers(0, k, k)
        means[i] = delta[idx].mean()
    return {
        "mean": float(delta.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "frac_better": float((delta < 0).mean()),
        "n_resamples": int(n),
        "seed": int(seed),
        "unit": "molecule",
    }


def monitor_gate(
    contrast: Mapping[str, Any],
    *,
    two_seed: bool,
    delta_threshold: float = GLRC_DELTA_THRESHOLD,
    ordinary_tol: float = GATE_ORDINARY_TOL,
    rmse_tol: float = GATE_RMSE_TOL,
    max_single_share: float = GATE_MAX_SINGLE_GAIN_FRACTION,
) -> dict[str, Any]:
    """Frozen promotion gate; identical shape for GLRC (2 seeds) and DLP (seed 0)."""
    per_seed_ok = all(v < 0 for v in contrast["per_seed_delta"].values()) if two_seed else True
    conditions = {
        "seed_mean_delta_le_threshold": bool(contrast["delta_seed_mean"] <= delta_threshold),
        "per_seed_all_negative": bool(per_seed_ok),
        "bootstrap_ci_upper_negative": bool(contrast["bootstrap"]["ci_hi"] < 0),
        "ordinary_not_worse": bool(contrast["ordinary_delta"] <= ordinary_tol),
        "top10_removed_negative": bool(contrast["top10_removed_delta"] < 0),
        "rmse_within_tolerance": bool(contrast["rmse_new"] - contrast["rmse_ref"] <= rmse_tol),
        "no_single_row_dominance": bool(contrast["max_single_gain_fraction"] <= max_single_share),
    }
    return {
        "delta_threshold": float(delta_threshold),
        "two_seed": bool(two_seed),
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }


# ---------------------------------------------------------------------------
# CPU information-gate helpers
# ---------------------------------------------------------------------------


def pair_feature_partition_witness() -> dict[str, Any]:
    """Same h-multiset, different distance partition -> different DLP features.

    Six fixed 2-D vectors ``h_i`` pooled over two different distance partitions
    (all-6-cycle partition vs a partition where the same products are
    redistributed).  Sum(h) and sum(h^2) are identical by construction, so any
    function of the first/second atom moments alone cannot distinguish the two
    pair features.
    """
    h = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.5, -1.0], [-0.5, -1.0]],
        dtype=np.float64,
    )
    partition_a = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)  # two pairs per bucket
    partition_b = np.asarray([2, 2, 0, 0, 1, 1], dtype=np.int64)  # same counts, different assignment
    pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (3, 4), (1, 5)]

    def pooled(partition: np.ndarray) -> np.ndarray:
        out = np.zeros((3, h.shape[1]), dtype=np.float64)
        counts = np.zeros(3, dtype=np.float64)
        for (i, j), k in zip(pairs, partition):
            out[k] += h[i] * h[j]
            counts[k] += 1
        return (out / np.maximum(counts, 1.0).reshape(-1, 1)).reshape(-1)

    pa, pb = pooled(partition_a), pooled(partition_b)
    return {
        "sum_h_identical": bool(np.allclose(h.sum(0), h.sum(0))),
        "sqsum_h_identical": bool(np.allclose((h**2).sum(0), (h**2).sum(0))),
        "pair_feature_distance_a": pa.tolist(),
        "pair_feature_distance_b": pb.tolist(),
        "max_abs_pair_feature_diff": float(np.abs(pa - pb).max()),
        "first_second_moments_equal": bool(np.array_equal(h.sum(0), h.sum(0)) and np.array_equal((h**2).sum(0), (h**2).sum(0))),
        "different_pair_features": bool(np.abs(pa - pb).max() > 1e-12),
    }


def linear_recoverability(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int = 0,
    lam: float = 1e-2,
    train_frac: float = 0.8,
) -> dict[str, Any]:
    """Ridge probe: how well can ``targets`` be linearly recovered from ``features``."""
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(features))
    ntr = int(train_frac * len(features))
    tr, te = perm[:ntr], perm[ntr:]
    mu, sd = features[tr].mean(0), features[tr].std(0) + 1e-9
    A = np.concatenate([np.ones((ntr, 1)), (features[tr] - mu) / sd], 1)
    B = np.concatenate([np.ones((len(te), 1)), (features[te] - mu) / sd], 1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ targets[tr])
    p_tr, p_te = A @ W, B @ W
    r2_tr = 1 - ((targets[tr] - p_tr) ** 2).sum(0) / np.maximum(((targets[tr] - targets[tr].mean(0)) ** 2).sum(0), 1e-12)
    r2_te = 1 - ((targets[te] - p_te) ** 2).sum(0) / np.maximum(((targets[te] - targets[te].mean(0)) ** 2).sum(0), 1e-12)
    centered_resid = float(
        np.sqrt(((targets[te] - p_te) ** 2).sum()) / max(np.sqrt(((targets[te] - targets[te].mean(0)) ** 2).sum()), 1e-12)
    )
    return {
        "train_r2_median": float(np.median(r2_tr)),
        "test_r2_median": float(np.median(r2_te)),
        "test_r2_mean": float(np.mean(r2_te)),
        "centered_relative_residual": centered_resid,
    }


def forbidden_module_scan_arch(model: nn.Module) -> list[str]:
    found = src.forbidden_module_scan(model)
    for name, module in model.named_modules():
        cls = type(module).__name__
        for token in ("Transformer", "Attention", "MessagePassing", "LSTM", "GRU", "RNN"):
            if token in cls:
                found.append(f"{name}:{cls}")
    return sorted(set(found))


# ---------------------------------------------------------------------------
# serialization helper
# ---------------------------------------------------------------------------


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()
