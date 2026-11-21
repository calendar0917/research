"""R0 candidate family: shared rooted-object primitive + explicit cut function.

Four arms share the primitive encoder ``phi`` and, except for ``GLOBAL``, the
fragment-state encoder ``E``:

* ``FACTOR`` -- low-rank shared response state, ``K`` fixed at 32
* ``DIRECT`` -- matched unconstrained cut head (ranking control)
* ``ADD``    -- matched additive control without environment interaction
* ``GLOBAL`` -- independent whole-molecule reference over the same primitive

Nothing is loaded from earlier rounds.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import trc_common as C


# ---------------------------------------------------------------------------
# deterministic per-module init (identical across arms for shared modules)
# ---------------------------------------------------------------------------


def module_seed(name: str, seed: int) -> int:
    tag = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    return int((seed * 1_000_003 + tag) % (2 ** 31 - 1))


_LEAF = (nn.Linear, nn.Embedding, nn.LayerNorm)


def _init_linear(lin: nn.Linear, gen: torch.Generator) -> None:
    nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5), generator=gen)
    if lin.bias is not None:
        bound = 1.0 / math.sqrt(lin.in_features) if lin.in_features > 0 else 0.0
        nn.init.uniform_(lin.bias, -bound, bound, generator=gen)


def deterministic_init(root: nn.Module, seed: int, prefix: str = "") -> None:
    """Re-initialise every module from a name-derived generator."""
    for name, param in root._parameters.items():
        if param is None:
            continue
        gen = torch.Generator().manual_seed(module_seed(f"{prefix}::{name}", seed))
        if param.dim() >= 2:
            nn.init.normal_(param, 0.0, 0.02, generator=gen)
        else:
            nn.init.zeros_(param)
    for name, module in root._modules.items():
        if module is None:
            continue
        path = f"{prefix}.{name}" if prefix else name
        gen = torch.Generator().manual_seed(module_seed(path, seed))
        if isinstance(module, nn.Linear):
            _init_linear(module, gen)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 1.0, generator=gen)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if not isinstance(module, _LEAF):
            deterministic_init(module, seed, path)


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().to("cpu").numpy()).tobytes())
    return digest.hexdigest()


def module_hashes(model: nn.Module, prefixes: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    named = dict(model.named_parameters())
    for prefix in prefixes:
        sel = {k: v for k, v in named.items() if k == prefix or k.startswith(prefix + ".")}
        if sel:
            out[prefix] = state_hash(sel)
    return out


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(int(sizes[i]), int(sizes[i + 1])))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PhiEncoder(nn.Module):
    """Rooted 1-4 primitive: every frame -> phi -> mean over frames."""

    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 dims: Mapping[str, int]) -> None:
        super().__init__()
        cat = int(dims["cat_dim"])
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), cat) for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), cat) for d in bond_dims])
        self.atom_mlp = _mlp([len(atom_dims) * cat, dims["atom_hidden"], dims["atom_out"]])
        self.bond_mlp = _mlp([len(bond_dims) * cat, dims["bond_hidden"], dims["bond_out"]])
        self.n_atom_fields = len(atom_dims)
        self.n_bond_fields = len(bond_dims)
        self.frame_mlp = nn.Sequential(
            nn.Linear(int(dims["frame_input"]), int(dims["frame_hidden"])),
            nn.SiLU(),
            nn.Linear(int(dims["frame_hidden"]), int(dims["frame_out"])),
            nn.SiLU(),
            nn.LayerNorm(int(dims["frame_out"])),
        )

    def forward(self, atom_vals: torch.Tensor, bond_vals: torch.Tensor,
                nmask: torch.Tensor, emask: torch.Tensor,
                size_oh: torch.Tensor) -> torch.Tensor:
        n = atom_vals.shape[0]
        a = torch.cat([emb(atom_vals[:, :, i]) for i, emb in enumerate(self.atom_emb)], dim=-1)
        a = self.atom_mlp(a) * nmask[:, :, None]
        b = torch.cat([emb(bond_vals[:, :, i]) for i, emb in enumerate(self.bond_emb)], dim=-1)
        b = self.bond_mlp(b) * emask[:, :, None]
        x = torch.cat([a.reshape(n, -1), b.reshape(n, -1), nmask, emask, emask, size_oh], dim=-1)
        return self.frame_mlp(x)


class FragmentEncoder(nn.Module):
    def __init__(self, dims: Mapping[str, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dims["fragment_input"]), int(dims["fragment_hidden"])),
            nn.SiLU(),
            nn.Linear(int(dims["fragment_hidden"]), int(dims["fragment_out"])),
            nn.SiLU(),
            nn.LayerNorm(int(dims["fragment_out"])),
            nn.Dropout(float(dims["fragment_dropout"])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class ResponseModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], atom_dims: Sequence[int],
                 bond_dims: Sequence[int], arm: str) -> None:
        super().__init__()
        if arm not in C.ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm = arm
        self.cfg = cfg
        dims = cfg["dims"]
        self.hidden = int(dims["hidden"])
        self.rank = int(dims["state_rank"])
        self.cut_dim = int(cfg["cuts"]["null_cut_dim"])
        self.phi = PhiEncoder(atom_dims, bond_dims, dims)
        self.needs_fragments = arm != "GLOBAL"
        self.needs_context = arm in ("FACTOR", "DIRECT", "ADD")
        if self.needs_fragments:
            self.enc = FragmentEncoder(dims)
        if self.needs_context:
            self.null_cut = nn.Parameter(torch.zeros(self.cut_dim))
        if arm == "FACTOR":
            self.a = _mlp([self.hidden, 64, 1])
            self.q = _mlp([self.hidden + self.cut_dim, 64, self.rank])
            self.r = _mlp([self.hidden + self.cut_dim, 64, self.rank])
            self.b = _mlp([self.cut_dim, 1])
            self.beta = nn.Parameter(torch.zeros(1))
            self.h_empty = nn.Parameter(torch.zeros(self.hidden))
        elif arm == "DIRECT":
            self.head = _mlp([2 * self.hidden + self.cut_dim, 96, 48, 1])
            self.beta = nn.Parameter(torch.zeros(1))
            self.h_empty = nn.Parameter(torch.zeros(self.hidden))
        elif arm == "ADD":
            self.a_add = _mlp([self.hidden, 128, 96, 1])
            self.b = _mlp([self.cut_dim, 1])
            self.beta = nn.Parameter(torch.zeros(1))
        elif arm == "GLOBAL":
            nb = int(cfg["heads"]["GLOBAL"]["buckets"][-1])
            self.global_head = nn.Sequential(
                nn.Linear(nb * 65 + 3, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Dropout(float(cfg["recipe"]["dropout_global"])),
                nn.Linear(128, 1),
            )
        deterministic_init(self, int(cfg["recipe"]["seed"]))

    # -- graph buffers ------------------------------------------------------
    def load_pack(self, pack: Mapping[str, Any]) -> None:
        def reg_long(name: str) -> None:
            self.register_buffer(name, torch.as_tensor(
                np.ascontiguousarray(pack[name])).long(), persistent=False)

        def reg_float(name: str) -> None:
            self.register_buffer(name, torch.as_tensor(
                np.ascontiguousarray(pack[name])).float(), persistent=False)

        for name in ("frag_ptr", "mol_ptr", "frag_frame", "mol_frame", "parse_S", "parse_C",
                     "frag_bucket", "mol_bucket", "parse_cut_bond_row", "frame_k",
                     "frag_counts", "frag_natoms", "frag_nbonds", "frag_ncomp",
                     "mol_counts", "mol_meta", "atom_table", "bond_table",
                     "frame_atom", "frame_bond"):
            reg_long(name)
        for name in ("frag_weight", "mol_weight"):
            reg_float(name)

        atom_table = np.asarray(pack["atom_table"])
        bond_table = np.asarray(pack["bond_table"])
        frame_atom = np.asarray(pack["frame_atom"])
        frame_bond = np.asarray(pack["frame_bond"])
        frame_k = np.asarray(pack["frame_k"]).astype(np.int64)
        n_frames, slots = frame_atom.shape
        atom_vals = atom_table[frame_atom]                       # [F, 4, 9]
        bond_vals = np.zeros((n_frames, frame_bond.shape[1], bond_table.shape[1]), np.int64)
        real = frame_bond >= 0
        bond_vals[real] = bond_table[frame_bond[real]]
        nmask = (np.arange(slots)[None, :] < frame_k[:, None]).astype(np.float32)
        emask = real.astype(np.float32)
        size_oh = np.zeros((n_frames, slots), np.float32)
        size_oh[np.arange(n_frames), frame_k - 1] = 1.0
        self.register_buffer("frame_atom_vals", torch.as_tensor(atom_vals).long(), persistent=False)
        self.register_buffer("frame_bond_vals", torch.as_tensor(bond_vals).long(), persistent=False)
        self.register_buffer("frame_nmask", torch.as_tensor(nmask), persistent=False)
        self.register_buffer("frame_emask", torch.as_tensor(emask), persistent=False)
        self.register_buffer("frame_size_oh", torch.as_tensor(size_oh), persistent=False)
        self.register_buffer(
            "mol_parse_ptr",
            torch.as_tensor(np.ascontiguousarray(pack["mol_parse_ptr"])).long(),
            persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- primitive frames ---------------------------------------------------
    def frame_phi(self) -> torch.Tensor:
        return self.phi(self.frame_atom_vals, self.frame_bond_vals, self.frame_nmask,
                        self.frame_emask, self.frame_size_oh)

    # -- fragment features --------------------------------------------------
    def _expand(self, ids: torch.Tensor, ptr_name: str):
        ptr = getattr(self, ptr_name)
        counts = ptr[ids + 1] - ptr[ids]
        total = int(counts.sum().item())
        starts = torch.cumsum(counts, 0) - counts
        ar = torch.arange(total, device=ids.device)
        local = ar - torch.repeat_interleave(starts, counts)
        gidx = torch.repeat_interleave(ptr[ids], counts) + local
        owner = torch.repeat_interleave(torch.arange(ids.numel(), device=ids.device), counts)
        return gidx, owner, counts

    def fragment_features(self, ids: torch.Tensor, phi: torch.Tensor,
                          erase_distance: bool = False) -> torch.Tensor:
        gidx, owner, _counts = self._expand(ids, "frag_ptr")
        bucket = self.frag_bucket[gidx]
        frame = self.frag_frame[gidx]
        weight = self.frag_weight[gidx]
        n = ids.numel()
        slot = owner * C.N_BUCKETS + bucket
        sum_d = torch.zeros((n * C.N_BUCKETS, phi.shape[1]),
                            device=phi.device, dtype=phi.dtype)
        sum_d.index_add_(0, slot, phi[frame] * weight[:, None])
        sum_d = sum_d.view(n, C.N_BUCKETS, -1)
        natoms = self.frag_natoms[ids].float()
        nbonds = self.frag_nbonds[ids].float()
        ncomp = self.frag_ncomp[ids].float()
        counts = self.frag_counts[ids].float()
        if erase_distance:
            total = sum_d.sum(dim=1, keepdim=True)
            n_total = counts.sum(dim=1, keepdim=True)
            mean = total.expand(-1, C.N_BUCKETS, -1) / C.N_BUCKETS
            density = total / (C.N_BUCKETS * natoms.view(-1, 1, 1).clamp(min=1.0))
            logcount = torch.log1p(n_total / C.N_BUCKETS).expand(-1, C.N_BUCKETS)
        else:
            nd = counts[:, :, None]
            mean = sum_d / nd.clamp(min=1.0)
            density = sum_d / natoms.view(-1, 1, 1).clamp(min=1.0)
            logcount = torch.log1p(nd.squeeze(-1))
        feat = torch.cat([mean, density, logcount[:, :, None]], dim=-1).reshape(n, -1)
        tail = torch.stack([torch.log1p(natoms), torch.log1p(nbonds),
                            torch.log1p(ncomp), torch.ones_like(natoms)], dim=-1)
        return torch.cat([feat, tail], dim=-1)

    def encode_fragments(self, ids: torch.Tensor, phi: torch.Tensor | None = None,
                         erase_distance: bool = False) -> torch.Tensor:
        phi = self.frame_phi() if phi is None else phi
        feat = self.fragment_features(ids, phi, erase_distance=erase_distance)
        return self.enc(feat)

    # -- cut head -----------------------------------------------------------
    def _cut_embedding(self, bond_row: torch.Tensor) -> torch.Tensor:
        n = bond_row.shape[0]
        out = self.null_cut[None, :].repeat(n, 1)
        real = bond_row >= 0
        if bool(real.any()):
            vals = self.bond_table[bond_row[real].clamp(min=0)]
            emb = torch.cat([e(vals[:, i]) for i, e in enumerate(self.phi.bond_emb)], dim=-1)
            out = out.index_copy(0, torch.nonzero(real).squeeze(-1), self.phi.bond_mlp(emb))
        return out

    def cut_scores(self, h_c: torch.Tensor | None, h_s: torch.Tensor,
                   e: torch.Tensor, has_c: torch.Tensor | None = None) -> torch.Tensor:
        if self.arm == "FACTOR":
            zc = torch.cat([h_c, e], dim=-1)
            zs = torch.cat([h_s, e], dim=-1)
            q = torch.tanh(self.q(zc))
            r = torch.tanh(self.r(zs))
            a_c = self.a(h_c) * has_c[:, None].float() if has_c is not None else self.a(h_c)
            return (self.beta + a_c + self.a(h_s) + self.b(e)
                    + (q * r).sum(-1, keepdim=True) / math.sqrt(self.rank)).squeeze(-1)
        if self.arm == "DIRECT":
            return (self.beta + self.head(torch.cat([h_c, h_s, e], dim=-1))).squeeze(-1)
        if self.arm == "ADD":
            a_c = self.a_add(h_c) * has_c[:, None].float() if has_c is not None else self.a_add(h_c)
            return (self.beta + a_c + self.a_add(h_s) + self.b(e)).squeeze(-1)
        raise RuntimeError("cut_scores is undefined for GLOBAL")

    # -- molecule forward ---------------------------------------------------
    def _parse_index(self, rows: torch.Tensor):
        ptr = self.mol_parse_ptr
        counts = ptr[rows + 1] - ptr[rows]
        total = int(counts.sum().item())
        starts = torch.cumsum(counts, 0) - counts
        pidx = (torch.repeat_interleave(ptr[rows], counts)
                + torch.arange(total, device=rows.device)
                - torch.repeat_interleave(starts, counts))
        return pidx, counts

    def molecule_cuts(self, rows: torch.Tensor, phi: torch.Tensor | None = None,
                      erase_distance: bool = False) -> dict[str, torch.Tensor]:
        pidx, counts = self._parse_index(rows)
        frag_s = self.parse_S[pidx]
        frag_c = self.parse_C[pidx]
        has_c = frag_c >= 0
        need = torch.cat([frag_s, frag_c[has_c]])
        uniq, inv = torch.unique(need, sorted=True, return_inverse=True)
        h = self.encode_fragments(uniq, phi, erase_distance=erase_distance)
        n_parse = int(frag_s.numel())
        h_s = h[inv[:n_parse]]
        h_c = (self.h_empty[None, :].repeat(n_parse, 1) if self.arm in ("FACTOR", "DIRECT")
               else torch.zeros(n_parse, self.hidden, device=rows.device))
        if bool(has_c.any()):
            h_c = h_c.index_copy(0, torch.nonzero(has_c).squeeze(-1), h[inv[n_parse:]])
        e = self._cut_embedding(self.parse_cut_bond_row[pidx])
        f = self.cut_scores(h_c, h_s, e, has_c)
        owner = torch.repeat_interleave(torch.arange(rows.numel(), device=rows.device), counts)
        f_bar = torch.zeros(rows.numel(), device=rows.device).index_add_(0, owner, f)
        f_bar = f_bar / counts.float().clamp(min=1.0)
        return {"f_cut": f, "f_bar": f_bar, "owner": owner, "counts": counts}

    def global_scores(self, rows: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        phi = self.frame_phi() if phi is None else phi
        gidx, owner, _counts = self._expand(rows, "mol_ptr")
        bucket = self.mol_bucket[gidx]
        frame = self.mol_frame[gidx]
        weight = self.mol_weight[gidx]
        n = rows.numel()
        nb = int(self.cfg["heads"]["GLOBAL"]["buckets"][-1])
        slot = owner * nb + bucket
        sum_d = torch.zeros((n * nb, phi.shape[1]), device=phi.device, dtype=phi.dtype)
        sum_d.index_add_(0, slot, phi[frame] * weight[:, None])
        sum_d = sum_d.view(n, nb, -1)
        cnt = self.mol_counts[rows].float()[:, :, None]
        natoms = self.mol_meta[rows][:, 0].float().view(-1, 1, 1)
        mean = sum_d / cnt.clamp(min=1.0)
        share = sum_d / natoms.clamp(min=1.0)
        logc = torch.log1p(cnt.squeeze(-1))
        feat = torch.cat([mean, share, logc[:, :, None]], dim=-1).reshape(n, -1)
        meta = self.mol_meta[rows].float()
        tail = torch.stack([torch.log1p(meta[:, 0]), torch.log1p(meta[:, 1]),
                            torch.log1p(meta[:, 2])], dim=-1)
        return self.global_head(torch.cat([feat, tail], dim=-1)).squeeze(-1)

    def molecule_forward(self, rows: torch.Tensor, erase_distance: bool = False) -> dict[str, Any]:
        if self.arm == "GLOBAL":
            f_bar = self.global_scores(rows)
            return {"f_bar": f_bar, "f_cut": None, "owner": None, "counts": None}
        phi = self.frame_phi()
        return self.molecule_cuts(rows, phi, erase_distance=erase_distance)

    # -- aux pair delta -----------------------------------------------------
    def pair_cut_delta(self, pairs: Mapping[str, torch.Tensor],
                       phi: torch.Tensor | None = None) -> torch.Tensor:
        ctx = pairs["C"]
        sa = pairs["Sa"]
        sb = pairs["Sb"]
        uniq, inv = torch.unique(torch.cat([ctx, sa, sb]), sorted=True, return_inverse=True)
        h = self.encode_fragments(uniq, phi)
        n = int(ctx.numel())
        h_c = h[inv[:n]]
        h_a = h[inv[n:2 * n]]
        h_b = h[inv[2 * n:]]
        e = self._cut_embedding(pairs["bond"])
        return self.cut_scores(h_c, h_b, e) - self.cut_scores(h_c, h_a, e)

    # -- algebra / ledger ---------------------------------------------------
    def parameter_ledger(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, p in self.named_parameters():
            top = name.split(".")[0]
            out[top] = out.get(top, 0) + int(p.numel())
        out["total"] = int(sum(p.numel() for p in self.parameters()))
        out["trainable"] = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return out


# ---------------------------------------------------------------------------
# training-set category support
# ---------------------------------------------------------------------------


def category_support(raw: Mapping[str, Any], train_rows: Sequence[int],
                     atom_dims: Sequence[int], bond_dims: Sequence[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-field category masks over the *current* training molecules."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    atom_ok = [np.zeros(int(d), bool) for d in atom_dims]
    bond_ok = [np.zeros(int(d), bool) for d in bond_dims]
    for row in np.asarray(train_rows).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        for i in range(atom_feat.shape[1]):
            vals = atom_feat[a0:a1, i].astype(np.int64)
            atom_ok[i][np.unique(vals)] = True
        for i in range(bond_feat.shape[1]):
            vals = bond_feat[b0:b1, i].astype(np.int64)
            bond_ok[i][np.unique(vals)] = True
    return atom_ok, bond_ok


def apply_atom_bond_support(model: ResponseModel, atom_support: Sequence[np.ndarray],
                            bond_support: Sequence[np.ndarray]) -> dict[str, int]:
    zeroed: dict[str, int] = {}
    with torch.no_grad():
        for i, emb in enumerate(model.phi.atom_emb):
            mask = torch.as_tensor(np.asarray(atom_support[i], bool), device=emb.weight.device)
            if mask.numel() != emb.weight.shape[0]:
                raise ValueError("atom support width mismatch")
            zeroed[f"atom_field_{i}"] = int((~mask).sum().item())
            emb.weight[~mask] = 0.0
        for i, emb in enumerate(model.phi.bond_emb):
            mask = torch.as_tensor(np.asarray(bond_support[i], bool), device=emb.weight.device)
            if mask.numel() != emb.weight.shape[0]:
                raise ValueError("bond support width mismatch")
            zeroed[f"bond_field_{i}"] = int((~mask).sum().item())
            emb.weight[~mask] = 0.0
    return zeroed
