"""RMC — rooted moment conditioning: one independent predictor.

Computation graph (per forward)
-------------------------------
::

    a_i, b_e, w_j = P3Encoders(batch)                    # unchanged P3 order/magnitudes
    e_i = MLP_290([a_i, mean_e b_e, mean_{root(i)} w_j,
                   log1p deg_i, log1p n_root_junctions_i])          # 96
    q_i = 4 tanh(Linear_96->48(e_i)).reshape(4,12) / sqrt(12)       # 4 heads
    head 0/1 -> view 0, head 2/3 -> view 1; each head queries all 8 scales
    F_i, Z_i = readout(cached mu/C2/C3 of the rooted walk measure)  # exact quadratic kernel
    h_i = MLP_556([e_i, std F_i (384), log Z_i (32), std log m_i (8),
                   std global33, std log1p deg_i, log1p n_atoms, log1p n_bonds])   # 96
    t_i = MLP_96->128->1(h_i)
    y_hat = graph_bias + sum_i t_i / train_mean_n_atoms

Non-negotiable boundaries implemented here:

* no access to another atom's trainable hidden state inside the readout, no
  neighbour-state recursion, no trainable structure operator;
* the rooted measure / moments are a fixed cache (dependence on ``y``: none);
* the frozen memory atom encoder is *not* a submodule (it is used once, at
  prep time) but its 26,336 parameters are reported as deployment cost;
* M and G share the construction recipe and differ only by the ``use_c3``
  flag, never by an RNG draw.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_SS = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_SS)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as AC  # noqa: E402
import ad_data as AD  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402

import rmc_common as C  # noqa: E402
import rmc_data as D  # noqa: E402
import rmc_moments as RM  # noqa: E402

#: head -> (view, channel block)
HEAD_VIEW = (0, 0, 1, 1)


# ---------------------------------------------------------------------------
# process-level split moment cache (loaded once per process, never a buffer)
# ---------------------------------------------------------------------------

_PROC_CACHE: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def split_cache(split: str, device) -> dict[str, torch.Tensor]:
    key = (str(split), str(device))
    if key not in _PROC_CACHE:
        path = C.moments_path(str(split))
        if not path.exists():
            raise SystemExit(f"missing moment cache {path}; run prep")
        raw = D.load_moments(path)
        moved = {k: v.to(device) for k, v in raw.items()}
        del raw
        _PROC_CACHE[key] = moved
    return _PROC_CACHE[key]


def clear_split_cache() -> None:
    _PROC_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class RMCModel(nn.Module):
    """Rooted moment conditioning with a fixed structure cache."""

    def __init__(
        self,
        norm: Mapping[str, Any],
        *,
        arm: str,
        seed: int,
        use_c3: bool | None = None,
        global_measure: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if arm not in ("M", "G", "U"):
            raise ValueError("RMCModel is built for M/G/U")
        self.arm = arm
        self.use_c3 = bool(use_c3) if use_c3 is not None else (arm == "M")
        self.global_measure = bool(global_measure)
        if arm == "G" and self.use_c3:
            raise ValueError("G must run with C3 = 0")
        if arm == "U" and use_c3 is None:
            raise ValueError("U needs an explicit C3 rule from the selected arm A")

        # encoders are constructed first, exactly like P3NullModel, under the
        # supervised seed; the fork keeps the global RNG untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.encoders = M1.P3Encoders(dropout=dropout)
            self.e_mlp = M1.MLP(96 + 96 + 96 + 2, 128, 96, 1, dropout)
            self.q_proj = nn.Linear(96, C.N_HEAD * C.PROJ_DIM)
            self.h_mlp = M1.MLP(C.READOUT_IN, 224, 96, 1, dropout)
            self.t_mlp = M1.MLP(96, 128, 1, 1, dropout)
            self.graph_bias = nn.Parameter(torch.zeros(()))

        # fixed train-only constants (buffers, not parameters)
        self.register_buffer("R", torch.as_tensor(norm["R"], dtype=torch.float32))
        center = torch.as_tensor(norm["mu_mean"], dtype=torch.float32)  # (8, 2, 12)
        scale = torch.as_tensor(norm["mu_scale"], dtype=torch.float32)
        self.register_buffer("F_center", center.permute(1, 0, 2)[list(HEAD_VIEW)].clone())  # (4, 8, 12)
        self.register_buffer("F_scale", scale.permute(1, 0, 2)[list(HEAD_VIEW)].clone())
        self.register_buffer("logm_mean", torch.as_tensor(norm["logm_mean"], dtype=torch.float32))
        self.register_buffer("logm_scale", torch.as_tensor(norm["logm_scale"], dtype=torch.float32))
        self.register_buffer("deg_mean", torch.as_tensor(float(norm["deg_mean"]), dtype=torch.float32))
        self.register_buffer("deg_scale", torch.as_tensor(float(norm["deg_scale"]), dtype=torch.float32))
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer(
            "train_mean_n_atoms", torch.as_tensor(float(norm["train_mean_n_atoms"]), dtype=torch.float32)
        )

    # -- query / readout ---------------------------------------------------
    def rooted_readout(self, e: torch.Tensor, batch: Mapping[str, Any]):
        q = 4.0 * torch.tanh(self.q_proj(e)).reshape(-1, C.N_HEAD, C.PROJ_DIM) / math.sqrt(C.PROJ_DIM)
        cache = split_cache(str(batch["rmc_split"]), e.device)
        if self.global_measure:
            idx = batch["rmc_mol_index"]
            mu = cache["uniform_mu"][idx].unsqueeze(1).expand(-1, C.N_SCALE, -1, -1)
            c2 = cache["uniform_c2"][idx].unsqueeze(1).expand(-1, C.N_SCALE, -1, -1)
            c3 = cache["uniform_c3"][idx].unsqueeze(1).expand(-1, C.N_SCALE, -1, -1)
        else:
            idx = batch["rmc_moment_index"]
            mu = cache["mu"][idx]
            c2 = cache["c2"][idx]
            c3 = cache["c3"][idx]
        logm = cache["logm"][batch["rmc_moment_index"]]
        F0, Z0 = RM.readout_batched(mu[:, :, 0, :], c2[:, :, 0, :], c3[:, :, 0, :],
                                    q[:, 0:2, :], use_c3=self.use_c3)
        F1, Z1 = RM.readout_batched(mu[:, :, 1, :], c2[:, :, 1, :], c3[:, :, 1, :],
                                    q[:, 2:4, :], use_c3=self.use_c3)
        F = torch.cat([F0, F1], dim=2)  # (N, 8, 4, 12)
        Z = torch.cat([Z0, Z1], dim=2)  # (N, 8, 4)
        F = F.permute(0, 2, 1, 3)  # (N, 4, 8, 12)  head-major
        Z = Z.permute(0, 2, 1)  # (N, 4, 8)
        F_std = (F - self.F_center.unsqueeze(0)) / self.F_scale.unsqueeze(0)
        logZ = torch.log(Z)
        return (
            F_std.reshape(F.shape[0], C.N_HEAD * C.N_SCALE * C.PROJ_DIM),
            logZ.reshape(F.shape[0], C.N_HEAD * C.N_SCALE),
            logm,
            q,
        )

    # -- forward -----------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, return_stats: bool = False):
        a, b0, w0 = self.encoders(batch)
        n_atoms = int(a.shape[0])
        bi = batch["bond_index"]
        dev, dtype = a.device, a.dtype

        deg = batch["rmc_deg"].to(dtype)
        sum_b = a.new_zeros((n_atoms, b0.shape[1]))
        sum_b.index_add_(0, torch.cat([bi[:, 0], bi[:, 1]]), torch.cat([b0, b0], dim=0))
        mean_b = sum_b / deg.clamp_min(1.0).unsqueeze(1)

        ji = batch["junc_index"]
        njunc = batch["rmc_njunc"].to(dtype)
        sum_w = a.new_zeros((n_atoms, w0.shape[1]))
        sum_w.index_add_(0, ji[:, 1], w0)
        mean_w = sum_w / njunc.clamp_min(1.0).unsqueeze(1)

        e_in = torch.cat([a, mean_b, mean_w, torch.log1p(deg).unsqueeze(1),
                          torch.log1p(njunc).unsqueeze(1)], dim=1)
        e = self.e_mlp(e_in)
        F_flat, logZ_flat, logm, q = self.rooted_readout(e, batch)

        logm_std = (logm - self.logm_mean.unsqueeze(0)) / self.logm_scale.unsqueeze(0)
        deg_std = (torch.log1p(deg) - self.deg_mean) / self.deg_scale
        g33 = (batch["global"] - self.global_mean) / self.global_scale
        g33_atom = g33[batch["atom_batch"]]
        n_atoms_col = torch.log1p(batch["rmc_natoms"].to(dtype)).unsqueeze(1)
        n_bonds_col = torch.log1p(batch["rmc_nbonds"].to(dtype)).unsqueeze(1)

        readout_in = torch.cat(
            [e, F_flat, logZ_flat, logm_std, g33_atom, deg_std.unsqueeze(1), n_atoms_col, n_bonds_col],
            dim=1,
        )
        h = self.h_mlp(readout_in)
        t = self.t_mlp(h)
        n_graphs = int(batch["n_graphs"])
        tot = t.new_zeros((n_graphs,))
        tot.index_add_(0, batch["atom_batch"], t.reshape(-1))
        pred = self.graph_bias + tot / self.train_mean_n_atoms
        if return_stats:
            stats = {
                "F_abs_mean": F_flat.abs().mean().detach(),
                "logZ_mean": logZ_flat.mean().detach(),
                "query_norm_mean": q.norm(dim=-1).mean().detach(),
                "e_norm_mean": e.norm(dim=1).mean().detach(),
                "t_abs_mean": t.abs().mean().detach(),
            }
            return pred, stats
        return pred


# ---------------------------------------------------------------------------
# SSL loading / builders / hashing
# ---------------------------------------------------------------------------


def load_ssl_encoders(model: nn.Module, ssl_path: Path) -> dict[str, Any]:
    state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    enc = model.encoders
    loaded = {}
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        if name not in state:
            raise SystemExit(f"SSL checkpoint {ssl_path} has no {name}")
        getattr(enc, name).load_state_dict(state[name], strict=True)
        loaded[name] = C.sha256_state_dict(state[name])
    return {
        "source": C.rel(ssl_path),
        "sha256": C.sha256_file(ssl_path),
        "encoder_tensor_hashes": loaded,
        "note": "train-only masked atom/bond pretext; same file for every arm and seed",
    }


def _ssl_path_from_config() -> Path:
    cfg = C.load_config()
    path = C.REPO / cfg["ssl"]["path"]
    if not path.exists():
        raise SystemExit(f"missing SSL checkpoint: {path}")
    got = C.sha256_file(path)
    if got != cfg["ssl"]["sha256"]:
        raise SystemExit(f"SSL checkpoint hash mismatch: {got} != {cfg['ssl']['sha256']}")
    return path


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    out = {name: int(sum(p.numel() for p in mod.parameters())) for name, mod in model.named_children()}
    out["total"] = int(sum(p.numel() for p in model.parameters()))
    return out


def build_arm(
    arm: str,
    norm: Mapping[str, Any],
    seed: int,
    device,
    *,
    ssl_path: Path | None = None,
    u_c3: bool | None = None,
):
    """Build one arm.  P is the verbatim P3NullModel; M/G/U are RMC bodies."""
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    ssl_path = _ssl_path_from_config() if ssl_path is None else Path(ssl_path)
    if arm == "P":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            model = A2.P3NullModel(norm)
    else:
        model = RMCModel(
            norm,
            arm=arm,
            seed=int(seed),
            use_c3=u_c3 if arm == "U" else (arm == "M"),
            global_measure=(arm == "U"),
        )
    ssl_info = load_ssl_encoders(model, ssl_path)
    model = model.to(device)
    params = int(sum(p.numel() for p in model.parameters()))
    info = {
        "arm": arm,
        "seed": int(seed),
        "ssl": ssl_info,
        "parameters": params,
        "parameters_online": params,
        "parameters_memory_encoder": D.memory_encoder_parameters(),
        "parameters_deployment": params + D.memory_encoder_parameters(),
        "use_c3": bool(getattr(model, "use_c3", False)),
        "global_measure": bool(getattr(model, "global_measure", False)),
        "parameter_breakdown": parameter_breakdown(model),
        "state_sha256": C.sha256_state_dict(model.state_dict()),
        "init_state_sha256": C.sha256_state_dict(model.state_dict()),
    }
    return model, info


def load_checkpoint(model: nn.Module, path: Path, device) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state_dict"] if "state_dict" in ck else ck.get("state", ck)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    return ck


def state_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck.get("state", ck)
