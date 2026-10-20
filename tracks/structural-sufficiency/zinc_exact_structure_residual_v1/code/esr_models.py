"""Exact rooted-patch type residual — model layer.

Two arms share one interface:

    h_v = a_v^P3 + a[t*(v)],      a[t] = P @ beta[t]

* ``P`` is ``atom_out x rank`` with orthonormal, sign-fixed, non-zero columns.
* ``beta`` is ``K x rank`` initialised to zero, so at step 0 the arm is
  *bit-identical* to the frozen P3 model.
* the residual is inserted into the atom embedding before the (unchanged)
  bond / junction / composer / reader stack; every original P3 parameter is
  retained and trained with the original R3 recipe.

``E`` (independent) uses ``R_E = mean_t ||a[t]||^2 / (depth(t) * Sh2)``.
``H`` (parent-shared) uses ``R_H = mean_t ||a[t] - a[parent(t)]||^2 / Sh2``
with ``a[parent(r1)] = 0`` (the r0 correction is fixed to zero).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, str(REPO), str(REPO / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ARMS = ("B", "E", "H")
DEFAULT_RANK = 4
RESIDUAL_SEED_OFFSET = 20_260


class ESRModelError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ESRModelError(message)


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def orthonormal_columns(atom_out: int, rank: int, seed: int) -> torch.Tensor:
    """Deterministic non-zero orthonormal columns, sign-fixed for reproducibility."""
    gen = torch.Generator().manual_seed(int(seed) + RESIDUAL_SEED_OFFSET)
    raw = torch.randn(int(atom_out), int(rank), generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw)
    for j in range(int(rank)):
        pivot = int(q[:, j].abs().argmax())
        if float(q[pivot, j]) < 0.0:
            q[:, j] = -q[:, j]
    return q.to(torch.float32)


class AtomTypeResidual(nn.Module):
    """Shared low-rank per-type correction ``a[t] = P @ beta[t]``."""

    def __init__(
        self,
        n_types: int,
        depth: np.ndarray,
        parent: np.ndarray,
        *,
        atom_out: int = 96,
        rank: int = DEFAULT_RANK,
        seed: int = 0,
        arm: str = "E",
        sh2: float = 1.0,
    ):
        super().__init__()
        if arm not in ("E", "H"):
            raise ValueError(f"unknown residual arm {arm!r}")
        require(n_types > 0, "empty residual vocabulary")
        require(depth.shape == (n_types,) and parent.shape == (n_types,), "type metadata shape mismatch")
        require(float(sh2) > 0.0, "Sh2 must be a positive lower bound")
        self.arm = arm
        self.atom_out = int(atom_out)
        self.rank = int(rank)
        self.sh2 = float(sh2)
        self.P = nn.Parameter(orthonormal_columns(self.atom_out, self.rank, int(seed)))
        self.beta = nn.Parameter(torch.zeros((int(n_types), self.rank), dtype=torch.float32))
        self.register_buffer("depth", torch.as_tensor(np.asarray(depth, dtype=np.int64)), persistent=True)
        self.register_buffer("parent", torch.as_tensor(np.asarray(parent, dtype=np.int64)), persistent=True)

    # -- forward ---------------------------------------------------------
    def forward(self, atom_type: torch.Tensor) -> torch.Tensor:
        """[n_atoms] type ids -> [n_atoms, atom_out]; -1 means zero correction."""
        if atom_type.numel() == 0:
            return self.beta.new_zeros((0, self.atom_out))
        idx = atom_type.clamp(min=0)
        coeff = self.beta.index_select(0, idx)  # [n_atoms, rank]
        out = coeff @ self.P.t()  # [n_atoms, atom_out]
        keep = (atom_type >= 0).unsqueeze(1)
        return out * keep

    # -- prior -----------------------------------------------------------
    def correction_matrix(self) -> torch.Tensor:
        """[atom_out, K] matrix of every type correction (used by the prior)."""
        return self.P @ self.beta.t()

    def prior(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (R_arm, per-type contributions). Full vocabulary, every step."""
        a = self.correction_matrix()  # [d, K]
        if self.arm == "E":
            per = (a * a).sum(dim=0) / (self.depth.to(a.dtype) * self.sh2)
        elif self.arm == "H":
            diff = a.clone()
            has_parent = self.parent >= 0
            if bool(has_parent.any()):
                parent = self.parent.clamp(min=0)
                diff[:, has_parent] = a[:, has_parent] - a[:, parent[has_parent]]
            per = (diff * diff).sum(dim=0) / self.sh2
        else:  # pragma: no cover
            raise AssertionError(self.arm)
        return per.mean(), per

    def correction_norms(self) -> torch.Tensor:
        a = self.correction_matrix()
        return a.norm(dim=0)

    def depth_prior_scale(self) -> torch.Tensor:
        return self.depth.to(torch.float32)


class ResidualBJOC(src.BJOC):
    """P3 with an exact-structure atom residual; no other architectural change."""

    def __init__(self, config: src.BJOCConfig, global_mean, global_scale, residual: AtomTypeResidual):
        super().__init__(config, global_mean=global_mean, global_scale=global_scale, organization_mode=src.NULL)
        self.atom_residual = residual

    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a = super().encode_atoms(batch)
        types = batch.get("atom_type")
        if types is not None:
            a = a + self.atom_residual(types)
        return a


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_p3(seed: int, norm: Mapping[str, Any], device) -> src.BJOC:
    model = f10k._make_downstream_model(int(seed), norm, device)
    require(sum(p.numel() for p in model.parameters()) == 413_505, "P3 parameter audit failed")
    return model


def build_residual_arm(
    arm: str,
    seed: int,
    norm: Mapping[str, Any],
    vocab: Mapping[str, Any],
    sh2: float,
    device,
    *,
    rank: int = DEFAULT_RANK,
) -> ResidualBJOC:
    """Build E/H from the same canonical P3 initialisation as B."""
    require(arm in ("E", "H"), "residual builder needs arm E or H")
    canonical = build_p3(seed, norm, device)
    residual = AtomTypeResidual(
        int(vocab["n_types"]),
        np.asarray(vocab["depth"]),
        np.asarray(vocab["parent"]),
        atom_out=int(canonical.config.atom_out),
        rank=int(rank),
        seed=int(seed),
        arm=arm,
        sh2=float(sh2),
    )
    model = ResidualBJOC(
        canonical.config,
        global_mean=canonical.global_mean,
        global_scale=canonical.global_scale,
        residual=residual,
    )
    # overwrite every P3 parameter/buffer with the canonical P3 starting point
    model.load_state_dict(canonical.state_dict(), strict=False)
    model.to(device)
    return model


def group_hash(model: nn.Module) -> str:
    return mp.group_state_hash(model)


def shared_module_hash(model: nn.Module) -> str:
    combined: dict[str, torch.Tensor] = {}
    for name in ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer", "reader"):
        for k, v in getattr(model, name).state_dict().items():
            combined[f"{name}:{k}"] = v
    return f10k.state_group_sha256(combined)


def residual_parameter_count(model: nn.Module) -> int:
    module = getattr(model, "atom_residual", None)
    if module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters()))


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    total = int(sum(p.numel() for p in model.parameters()))
    return {
        "total_parameters": total,
        "p3_parameters": total - residual_parameter_count(model),
        "residual_parameters": residual_parameter_count(model),
        "within_cap": bool(total <= 500_000),
    }


def residual_state_hash(model: nn.Module) -> str:
    module = getattr(model, "atom_residual", None)
    if module is None:
        return ""
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_arm(
    arm: str,
    seed: int,
    norm: Mapping[str, Any],
    vocab: Mapping[str, Any] | None,
    sh2: float,
    device,
    *,
    rank: int = DEFAULT_RANK,
) -> nn.Module:
    if arm == "B":
        return build_p3(seed, norm, device)
    require(vocab is not None, "residual arm requires a vocabulary")
    return build_residual_arm(arm, seed, norm, vocab, sh2, device, rank=rank)


def load_encoder(model: nn.Module, seed: int, encoder_path: Path, device) -> None:
    state = torch.load(Path(encoder_path), map_location="cpu", weights_only=False)
    mp.load_encoder_state(model, state, device)


# ---------------------------------------------------------------------------
# batching with per-atom type ids
# ---------------------------------------------------------------------------


def batch_with_types(
    objs,
    rows,
    assigned,
    device,
):
    """Batch the frozen objects and add the per-atom deepest-retained type id."""
    batch = sro.batch_objects([objs[int(i)] for i in rows])
    if len(rows) == 0:
        types = np.zeros((0,), dtype=np.int64)
    else:
        types = np.concatenate([np.asarray(assigned[int(i)]["t_star"], dtype=np.int64) for i in rows], axis=0)
    require(int(types.size) == int(batch["atom_feat"].shape[0]), "atom_type / atom_feat length mismatch")
    batch["atom_type"] = torch.as_tensor(types, dtype=torch.long)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model: nn.Module, objs, assigned, device, *, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), int(batch_size)):
            rows = list(range(i, min(i + int(batch_size), len(objs))))
            batch = batch_with_types(objs, rows, assigned, device)
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0,), dtype=np.float32)
