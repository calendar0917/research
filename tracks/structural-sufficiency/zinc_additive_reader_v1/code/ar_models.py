"""A / D readers over the unchanged DCC-L local structural computation.

Both models keep the exact historical front end:

* ``P3Encoders`` (atom / bond / junction, train-only masked-SSL initialisation),
* the P3-Null junction-to-bond grouping used by ``group_composer``,
* the organised junction-to-bond ownership used by ``local_composer``,
* the original deterministic structural inputs and Fit-only standardisation.

Only the final reader is replaced.  For every physical bond ``b`` the existing
local composer output ``c_loc_b`` is passed through one shared local network

    h_b = phi(c_loc_b) in R^32,    S = sum_b h_b,

and the existing global state is summarised as

    G = [sum_b c_glob_b, sum_v a_v, sum_v a_v^2, standardised global33] in R^321.

A:  y = w^T S + psi(G)          (additive fusion; psi never sees S / sum c_loc)
D:  y = R([S, G])              (parameter-matched ordinary dense reader)

There is no recursion, attention, extra structural statistic, auxiliary
supervision or per-bond / per-pattern identifier.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ar_common as C  # noqa: E402

for _p in (str(C.V1), str(C.DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import dcc_models as DCC  # noqa: E402

WIDTH = 96
GLOBAL_DIM = D.GLOBAL_DIM          # 33
BOND_FEAT_DIM = D.BOND_FEAT_DIM    # 7
G_DIM = 3 * WIDTH + GLOBAL_DIM     # 321
PHI_HIDDEN = 64
PHI_OUT = 32
READER_HIDDEN = 256


class LocalTrunk(nn.Module):
    """Historical DCC-L front end, identical in A and D."""

    def __init__(self, norm: Mapping[str, object], *, phi_hidden: int = PHI_HIDDEN,
                 phi_out: int = PHI_OUT, dropout: float = 0.0):
        super().__init__()
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.group_composer = M1.MLP(
            WIDTH + 2 * WIDTH + 1 + BOND_FEAT_DIM, 160, WIDTH, 1, dropout
        )
        self.local_composer = M1.MLP(
            WIDTH + 2 * WIDTH + 1 + BOND_FEAT_DIM, 80, WIDTH, 1, dropout
        )
        self.phi = M1.MLP(WIDTH, int(phi_hidden), int(phi_out), n_hidden=1, dropout=dropout)
        self.phi_out = int(phi_out)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        # The historical DCC-L L arm uses organised (true junction-to-bond) ownership.
        self.organized_local = True

    # -- exact historical grouping code paths --------------------------------
    def group_to_bonds(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        n_graphs = int(batch["n_graphs"])
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w)
        w2_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w * w)
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        mean_w = w_sum / cnt.unsqueeze(1)
        mean_w2 = w2_sum / cnt.unsqueeze(1)
        bond_batch = batch["bond_batch"]
        return n_b.unsqueeze(1) * mean_w[bond_batch], n_b.unsqueeze(1) * mean_w2[bond_batch], n_b

    def local_grouping(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor], *,
                       organized: bool = True):
        if not organized:
            return self.group_to_bonds(w, batch)
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        W = torch.zeros((int(batch["bond_feat"].shape[0]), w.shape[1]),
                        device=w.device, dtype=w.dtype)
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V, n_b

    # -- production local computation ---------------------------------------
    def local_components(self, batch: Mapping[str, torch.Tensor],
                         *, h_override: torch.Tensor | None = None) -> dict:
        """Return every existing state, plus phi(c_loc), h and S.

        ``h_override`` is only used by the CPU additivity test; production
        training and evaluation never pass it.
        """
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = self.local_grouping(w, batch, organized=bool(self.organized_local))
        c_loc = self.local_composer(
            torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        h = self.phi(c_loc) if h_override is None else h_override
        B = int(batch["n_graphs"])
        z = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, batch["bond_batch"], c_glob)
        S = h.new_zeros((B, h.shape[1])).index_add_(0, batch["bond_batch"], h)
        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        G = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {
            "a": a, "b": b, "w": w,
            "Wg": Wg, "Vg": Vg, "Wl": Wl, "Vl": Vl,
            "n_b": n_b, "c_glob": c_glob, "c_loc": c_loc,
            "h": h, "S": S, "z": z, "a_sum": a_sum, "a_sq": a_sq, "g": g,
            "G": G,
        }


class AdditiveReaderModel(LocalTrunk):
    """A: y = w^T S + psi(G)."""

    def __init__(self, norm, *, phi_hidden: int = PHI_HIDDEN, phi_out: int = PHI_OUT,
                 reader_hidden: int = READER_HIDDEN, dropout: float = 0.0):
        super().__init__(norm, phi_hidden=phi_hidden, phi_out=phi_out, dropout=dropout)
        self.w = nn.Linear(int(phi_out), 1, bias=False)
        self.psi = M1.MLP(G_DIM, int(reader_hidden), 1, n_hidden=2, dropout=dropout)

    def forward(self, batch, *, h_override: torch.Tensor | None = None) -> torch.Tensor:
        c = self.local_components(batch, h_override=h_override)
        return self.additive_terms(c).view(-1)

    def additive_terms(self, c: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.w(c["S"]).view(-1) + self.psi(c["G"]).view(-1)

    def local_term(self, c: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.w(c["S"]).view(-1)

    def global_term(self, c: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.psi(c["G"]).view(-1)


class DenseReaderModel(LocalTrunk):
    """D: y = R([S, G]) with matched overall parameter count."""

    def __init__(self, norm, *, phi_hidden: int = PHI_HIDDEN, phi_out: int = PHI_OUT,
                 reader_hidden: int = READER_HIDDEN, dropout: float = 0.0):
        super().__init__(norm, phi_hidden=phi_hidden, phi_out=phi_out, dropout=dropout)
        self.reader = M1.MLP(G_DIM + int(phi_out), int(reader_hidden), 1, n_hidden=2,
                             dropout=dropout)

    def forward(self, batch, *, h_override: torch.Tensor | None = None) -> torch.Tensor:
        c = self.local_components(batch, h_override=h_override)
        return self.reader(torch.cat([c["S"], c["G"]], dim=1)).view(-1)


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------
def _reset_linears(module: nn.Module) -> None:
    """Reset every Linear in registration order under the ambient RNG state."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()


def _trunk_state_from_history(seed: int, norm: Mapping[str, object]):
    """Build the historical shared state (SSL encoders + P3/ DCC composers)."""
    dual, base_info = DCC.build_initial_state(norm, seed)
    state = dual.state_dict()
    shared = {
        k: v.detach().clone()
        for k, v in state.items()
        if k.startswith(("encoders.", "group_composer.", "local_composer."))
        or k in ("global_mean", "global_scale")
    }
    return shared, dual, base_info


def _shared_phi_state(seed: int, phi_hidden: int, phi_out: int) -> dict:
    torch.manual_seed(int(seed) + 20011)
    tmp = M1.MLP(WIDTH, int(phi_hidden), int(phi_out), n_hidden=1, dropout=0.0)
    return {f"phi.{k}": v.detach().clone() for k, v in tmp.state_dict().items()}


def _reader_state(model: nn.Module, prefixes: tuple[str, ...]) -> dict:
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k.startswith(prefixes)
    }


def build_pair(norm: Mapping[str, object], seed: int, *, cfg: Mapping | None = None) -> tuple:
    """Build A and D from the same historical trunk and the same phi state."""
    cfg = C.load_config() if cfg is None else cfg
    mcfg = cfg["model"]
    hidden = int(mcfg["phi"]["hidden"])
    out = int(mcfg["phi"]["out"])
    rh = int(mcfg["additive_reader"]["psi"]["hidden"])

    shared, dual, base_info = _trunk_state_from_history(int(seed), norm)
    phi_state = _shared_phi_state(int(seed), hidden, out)

    # Readers are initialised under explicit, arm-specific streams; only the
    # legacy trunk and phi are shared.  No supervised DCC-L checkpoint is read.
    torch.manual_seed(int(seed) + 20021)
    model_a = AdditiveReaderModel(norm, phi_hidden=hidden, phi_out=out, reader_hidden=rh)
    torch.manual_seed(int(seed) + 20021)
    _reset_linears(model_a.w)
    _reset_linears(model_a.psi)

    torch.manual_seed(int(seed) + 20022)
    model_d = DenseReaderModel(norm, phi_hidden=hidden, phi_out=out, reader_hidden=rh)
    torch.manual_seed(int(seed) + 20022)
    _reset_linears(model_d.reader)

    for model in (model_a, model_d):
        missing, unexpected = model.load_state_dict({**shared, **phi_state}, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected shared keys: {unexpected}")
        if any(not k.startswith(("w.", "psi.", "reader.")) and k not in ("global_mean", "global_scale")
               for k in missing):
            raise RuntimeError(f"unexpected missing shared keys: {missing}")

    # exact shared-state identity between the arms
    a_state, d_state = model_a.state_dict(), model_d.state_dict()
    for key in sorted(set(shared) | set(phi_state)):
        if not torch.equal(a_state[key], d_state[key]):
            raise RuntimeError(f"A/D shared tensor differs at init: {key}")

    pa, pd = C.parameter_count(model_a), C.parameter_count(model_d)
    rel = abs(pa - pd) / max(pa, pd)
    if max(pa, pd) > 500_000:
        raise RuntimeError(f"parameter cap exceeded: A={pa} D={pd}")
    if rel > 0.02:
        raise RuntimeError(f"A/D parameter mismatch {rel:.4%} > 2%")

    def breakdown(model):
        return {
            "encoders": C.parameter_count(model.encoders),
            "group_composer": C.parameter_count(model.group_composer),
            "local_composer": C.parameter_count(model.local_composer),
            "phi": C.parameter_count(model.phi),
            "w": C.parameter_count(model.w) if hasattr(model, "w") else 0,
            "psi": C.parameter_count(model.psi) if hasattr(model, "psi") else 0,
            "reader": C.parameter_count(model.reader) if hasattr(model, "reader") else 0,
            "total": C.parameter_count(model),
            "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
            "parameter_tensors": sum(1 for p in model.parameters()),
        }

    info = {
        "protocol_id": cfg["protocol_id"],
        "seed": int(seed),
        "ssl": base_info.get("ssl"),
        "historical_p3_reference_state_sha256": base_info.get("p3_reference_state_sha256"),
        "historical_dual_initial_state_sha256": base_info.get("initial_state_sha256"),
        "historical_dual_parameters": base_info.get("parameters"),
        "shared_trunk_phi_state_sha256": C.state_dict_hash({**shared, **phi_state}),
        "trunk_state_sha256": C.state_dict_hash(shared),
        "phi_state_sha256": C.state_dict_hash(phi_state),
        "A_reader_state_sha256": C.state_dict_hash(_reader_state(model_a, ("w.", "psi."))),
        "D_reader_state_sha256": C.state_dict_hash(_reader_state(model_d, ("reader.",))),
        "A_initial_state_sha256": C.state_hash(model_a),
        "D_initial_state_sha256": C.state_hash(model_d),
        "parameters_A": pa,
        "parameters_D": pd,
        "parameters_relative_difference": float(rel),
        "parameters_breakdown_A": breakdown(model_a),
        "parameters_breakdown_D": breakdown(model_d),
        "cfg_sha256": C.sha256_file(C.CONFIG),
        "git_commit": C.git_commit(),
        "init_streams": {
            "historical_reference": int(seed),
            "historical_local_composer": int(seed) + 90731,
            "phi": int(seed) + 20011,
            "A_w_psi": int(seed) + 20021,
            "D_reader": int(seed) + 20022,
        },
        "supervised_dccl_checkpoint_loaded": False,
    }
    return model_a, model_d, info


def build_arm(arm: str, norm: Mapping[str, object], seed: int, device) -> tuple:
    if arm not in ("A", "D"):
        raise ValueError(f"unknown arm {arm!r}")
    a, d, info = build_pair(norm, seed)
    model = a if arm == "A" else d
    model.to(device)
    out = dict(info)
    out["arm"] = arm
    out["candidate"] = f"AdditiveReader_{arm}"
    return model, out


def load_trunk_from_checkpoint(model: nn.Module, state: Mapping[str, torch.Tensor],
                               *, strict: bool = False) -> None:
    """Load a full checkpoint state (used only for frozen inference replay)."""
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    return missing
