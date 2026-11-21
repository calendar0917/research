"""The candidate network: shared word functions + real-binding composition.

There is **no message passing** anywhere.  Every quantity is computed from
explicit slot-aligned local frames:

* ``AtomMLP`` / ``BondMLP`` encode the raw 9 / 3 attribute columns of one atom
  or one physical bond.
* ``WordMLP`` reads a padded 4-slot / 6-edge local frame -> ``H_U^g`` (64).
* a shared ``Q``/``K`` and a relation MLP ``lambda(r_UV^{g,h})`` (32) combine
  two words through the *actual* variable binding (same-atom matrix ``E``,
  physical cross-bond matrix ``C`` and the cross-bond attributes).
* ``z_(S,v)`` averages over covers and frames, then a sentence LayerNorm+SiLU
  and a size-wise sum/mean pooling feed the residual join ``J``.

Modes: ``prefix`` (word bag only), ``full``, ``bag_cont`` (sentence context
forced to zero), ``early`` (attribute permutation pooled before binding) and
``flat`` (one MLP on the concatenated large-local object).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import swf_common as C
import swf_data as D

N_TEMPLATES = D.N_TEMPLATES
BAG_FEATURE = N_TEMPLATES * (64 + 64 + 1) + 2   # 2066
SENTENCE_CONTEXT = 2 * 2 * 32 + 2               # 130


def _mlp_checked(dims: Sequence[int]) -> nn.Sequential:
    if len(dims) != 3:
        raise ValueError("expected a 3-layer chain")
    return nn.Sequential(nn.Linear(dims[0], dims[1]), nn.SiLU(), nn.Linear(dims[1], dims[2]))


class StructuralWordModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], atom_dims: Sequence[int], bond_dims: Sequence[int]) -> None:
        super().__init__()
        d = cfg["dims"]
        rec = cfg["recipe"]
        self.cfg = cfg
        self.n_atom_cols = len(atom_dims)
        self.n_bond_cols = len(bond_dims)
        self.atom_emb = nn.ModuleList(
            [nn.Embedding(int(x), int(d["atom_feature_dim"])) for x in atom_dims]
        )
        self.atom_mlp = _mlp_checked(
            [len(atom_dims) * int(d["atom_feature_dim"]), int(d["atom_hidden"]), int(d["atom_out"])]
        )
        self.bond_emb = nn.ModuleList(
            [nn.Embedding(int(x), int(d["bond_feature_dim"])) for x in bond_dims]
        )
        self.bond_mlp = _mlp_checked(
            [len(bond_dims) * int(d["bond_feature_dim"]), int(d["bond_hidden"]), int(d["bond_out"])]
        )
        self.tmpl_emb = nn.Embedding(N_TEMPLATES, int(d["template_emb"]))
        self.word_mlp = _mlp_checked(
            [int(d["word_input"]), int(d["word_hidden"]), int(d["word_out"])]
        )
        self.word_ln = nn.LayerNorm(int(d["word_out"]))
        self.bag_mlp = nn.Sequential(
            nn.Linear(BAG_FEATURE, int(d["bag_hidden1"])),
            nn.SiLU(),
            nn.Dropout(float(d["bag_dropout"])),
            nn.Linear(int(d["bag_hidden1"]), int(d["bag_hidden2"])),
            nn.SiLU(),
            nn.Dropout(float(d["bag_dropout"])),
        )
        self.bag_out = nn.Linear(int(d["bag_hidden2"]), 1)
        self.q = nn.Linear(int(d["word_out"]), int(d["coord_dim"]))
        self.k = nn.Linear(int(d["word_out"]), int(d["coord_dim"]))
        self.rel_mlp = _mlp_checked([int(d["rel_input"]), int(d["rel_hidden"]), int(d["rel_out"])])
        self.sent_ln = nn.LayerNorm(int(d["sentence_out"]))
        self.flat_mlp = _mlp_checked([int(d["flat_input"]), int(d["flat_hidden"]), int(d["flat_out"])])
        self.join = nn.Sequential(
            nn.Linear(int(d["join_input"]), int(d["join_hidden"])),
            nn.SiLU(),
            nn.Dropout(float(d["join_dropout"])),
            nn.Linear(int(d["join_hidden"]), 1),
        )
        nn.init.zeros_(self.join[-1].weight)
        nn.init.zeros_(self.join[-1].bias)
        self.register_buffer("tmpl_active", torch.ones(N_TEMPLATES, dtype=torch.float32))
        self.register_buffer("mu_T", torch.ones(N_TEMPLATES, dtype=torch.float32))
        self.register_buffer("nu_k", torch.ones(2, dtype=torch.float32))
        self.config_recipe = rec

    # ------------------------------------------------------------------ views
    def atom_vectors(self, atom_code: torch.Tensor) -> torch.Tensor:
        parts = [emb(atom_code[:, j]) for j, emb in enumerate(self.atom_emb)]
        return self.atom_mlp(torch.cat(parts, dim=1))

    def bond_vectors(self, bond_code: torch.Tensor) -> torch.Tensor:
        parts = [emb(bond_code[:, j]) for j, emb in enumerate(self.bond_emb)]
        return self.bond_mlp(torch.cat(parts, dim=1))

    def word_views(self, b: Mapping[str, torch.Tensor]):
        A = self.atom_vectors(b["atom_code"])                      # [CA, 32]
        E = self.bond_vectors(b["bond_code"])                       # [CB, 16]
        a = A[b["win_atom_idx"]] * b["win_node_mask"].unsqueeze(-1)  # [Uw, 4, 32]
        e = E[b["win_bond_idx"]] * b["win_edge_mask"].unsqueeze(-1)  # [Uw, 6, 16]
        x = torch.cat(
            [
                a.flatten(1),
                e.flatten(1),
                b["win_node_mask"],
                b["win_edge_mask"],
                b["win_adj"],
                self.tmpl_emb(b["win_tmpl"]),
            ],
            dim=1,
        )
        if x.shape[1] != int(self.cfg["dims"]["word_input"]):
            raise AssertionError(f"word input dim {x.shape[1]} != 256")
        H = self.word_ln(self.word_mlp(x))
        return E, x, H

    def word_average(self, b: Mapping[str, torch.Tensor], H: torch.Tensor) -> torch.Tensor:
        Hf = H[b["frame_win"]]
        sums = H.new_zeros((int(b["n_words"]), H.shape[1])).index_add_(
            0, b["frame_word"], Hf
        )
        return sums / b["word_nframes"].unsqueeze(-1)

    def bag_readout(self, b: Mapping[str, torch.Tensor], H: torch.Tensor):
        bar = self.word_average(b, H)
        B = int(b["n_molecules"])
        flat = (b["word_mol"] * N_TEMPLATES + b["word_tmpl"]).long()
        acc = bar.new_zeros((B * N_TEMPLATES, bar.shape[1])).index_add_(0, flat, bar)
        cnt = bar.new_zeros(B * N_TEMPLATES).index_add_(
            0, flat, torch.ones(b["word_tmpl"].shape[0], dtype=bar.dtype, device=bar.device)
        )
        acc = acc.view(B, N_TEMPLATES, bar.shape[1])
        cnt = cnt.view(B, N_TEMPLATES)
        u = acc / self.mu_T.view(1, N_TEMPLATES, 1)
        v = acc / cnt.clamp_min(1.0).unsqueeze(-1)
        logn = torch.log1p(cnt).unsqueeze(-1)
        act = self.tmpl_active.view(1, N_TEMPLATES, 1)
        u = u * act
        v = v * act
        logn = logn * act
        feat = torch.cat(
            [
                torch.cat([u, v, logn], dim=2).flatten(1),
                torch.log1p(b["n_atoms"]).unsqueeze(-1),
                torch.log1p(b["n_bonds"]).unsqueeze(-1),
            ],
            dim=1,
        )
        if feat.shape[1] != BAG_FEATURE:
            raise AssertionError(f"bag feature dim {feat.shape[1]} != {BAG_FEATURE}")
        g = self.bag_mlp(feat)
        y_bag = self.bag_out(g).squeeze(-1)
        return g, y_bag, bar

    def relation_inputs(self, b: Mapping[str, torch.Tensor], E: torch.Tensor) -> torch.Tensor:
        cross = E[b["rel_bond_idx"]] * b["rel_c"].unsqueeze(-1)         # [Rr, 16, 16]
        r = torch.cat(
            [
                b["rel_e"],
                b["rel_c"],
                cross.flatten(1),
                b["rel_mask_u"],
                b["rel_mask_v"],
                self.tmpl_emb(b["rel_tmpl_u"]),
                self.tmpl_emb(b["rel_tmpl_v"]),
            ],
            dim=1,
        )
        if r.shape[1] != int(self.cfg["dims"]["rel_input"]):
            raise AssertionError(f"relation input dim {r.shape[1]} != 328")
        return r

    # ------------------------------------------------------------------- core
    def sentence_z(
        self,
        b: Mapping[str, torch.Tensor],
        E: torch.Tensor,
        H: torch.Tensor,
        winput: torch.Tensor,
        bar: torch.Tensor,
        mode: str,
        *,
        flat_chunk: int = 25000,
    ) -> torch.Tensor:
        nsb = int(b["n_sentences"])
        Z = int(self.cfg["dims"]["sentence_out"])
        if mode == "bag_cont":
            return H.new_zeros((nsb, Z))
        if mode == "flat":
            return self._sentence_z_flat(b, E, winput, flat_chunk)
        if mode in ("full", "early"):
            if mode == "full":
                Hf = H[b["frame_win"]]
                qh = self.q(Hf)
                kh = self.k(Hf)
                ql = qh[b["row_left"]]
                kr = kh[b["row_right"]]
                qr = qh[b["row_right"]]
                kl = kh[b["row_left"]]
            else:
                qb = self.q(bar)
                kb = self.k(bar)
                ql = qb[b["row_word_u"]]
                kr = kb[b["row_word_v"]]
                qr = qb[b["row_word_v"]]
                kl = kb[b["row_word_u"]]
            lam = torch.tanh(self.rel_mlp(self.relation_inputs(b, E)))
            fwd = ql * kr * lam[b["row_rel_uv"]]
            rev = qr * kl * lam[b["row_rel_vu"]]
            contrib = 0.5 * (fwd + rev) * b["row_weight"].unsqueeze(-1)
            return H.new_zeros((nsb, Z)).index_add_(0, b["row_sent"], contrib)
        raise ValueError(f"unknown mode {mode}")

    def _sentence_z_flat(
        self,
        b: Mapping[str, torch.Tensor],
        E: torch.Tensor,
        winput: torch.Tensor,
        chunk: int,
    ) -> torch.Tensor:
        nsb = int(b["n_sentences"])
        Z = int(self.cfg["dims"]["flat_out"])
        nrow = int(b["row_left"].shape[0])
        rel = self.relation_inputs(b, E)
        z = winput.new_zeros((nsb, Z))
        wl = winput[b["row_win_left"]]
        wr = winput[b["row_win_right"]]
        for start in range(0, nrow, chunk):
            stop = min(nrow, start + chunk)
            fwd = self.flat_mlp(
                torch.cat([wl[start:stop], wr[start:stop], rel[b["row_rel_uv"][start:stop]]], dim=1)
            )
            rev = self.flat_mlp(
                torch.cat([wr[start:stop], wl[start:stop], rel[b["row_rel_vu"][start:stop]]], dim=1)
            )
            contrib = 0.5 * (fwd + rev) * b["row_weight"][start:stop].unsqueeze(-1)
            z = z.index_add(0, b["row_sent"][start:stop], contrib)
        return z

    def pool_sentences(self, b: Mapping[str, torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        B = int(b["n_molecules"])
        n = b["sent_n"]
        outs = []
        counts = []
        for ki, k in enumerate((5, 6)):
            mask = n == k
            idx = b["sent_mol"][mask]
            hk = h[mask]
            m = h.new_zeros(B).index_add_(
                0, idx, torch.ones(hk.shape[0], dtype=h.dtype, device=h.device)
            )
            s = h.new_zeros((B, h.shape[1])).index_add_(0, idx, hk)
            nu = self.nu_k[ki]
            outs.append(s / nu)
            outs.append(s / m.clamp_min(1.0).unsqueeze(-1))
            counts.append(torch.log1p(m).unsqueeze(-1))
        ctx = torch.cat(outs + counts, dim=1)
        if ctx.shape[1] != SENTENCE_CONTEXT:
            raise AssertionError(f"sentence context dim {ctx.shape[1]} != {SENTENCE_CONTEXT}")
        return ctx

    # --------------------------------------------------------------- forward
    def forward(
        self,
        b: Mapping[str, torch.Tensor],
        mode: str = "full",
        *,
        return_parts: bool = False,
    ):
        E, winput, H = self.word_views(b)
        g, y_bag, bar = self.bag_readout(b, H)
        if mode == "prefix":
            if return_parts:
                return y_bag, {"g": g, "y_bag": y_bag}
            return y_bag
        B = int(b["n_molecules"])
        if mode == "bag_cont":
            c = H.new_zeros((B, SENTENCE_CONTEXT))
        else:
            z = self.sentence_z(b, E, H, winput, bar, mode)
            h = F.silu(self.sent_ln(z))
            c = self.pool_sentences(b, h)
        delta = self.join(torch.cat([g, c], dim=1)).squeeze(-1)
        y = y_bag + delta
        if return_parts:
            return y, {"g": g, "y_bag": y_bag, "context": c, "delta": delta}
        return y

    # ---------------------------------------------------------- introspection
    def prefix_parameter_names(self) -> set[str]:
        names = set()
        for prefix in ("atom_emb", "atom_mlp", "bond_emb", "bond_mlp", "tmpl_emb", "word_mlp", "word_ln", "bag_mlp", "bag_out"):
            for name, _ in self.named_parameters():
                if name.startswith(prefix):
                    names.add(name)
        return names

    def parameter_ledger(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            head = name.split(".")[0]
            groups[head] = groups.get(head, 0) + int(p.numel())
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "by_module": groups,
            "trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }


def official_feature_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != C.ATOM_COLUMNS or len(bond) != C.BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature definition")
    return atom, bond


def build_model(cfg: Mapping[str, Any], *, seed: int | None = None) -> StructuralWordModel:
    atom_dims, bond_dims = official_feature_dims()
    torch.manual_seed(int(cfg["seed"] if seed is None else seed))
    model = StructuralWordModel(cfg, atom_dims, bond_dims)
    return model


def apply_train_support(model: StructuralWordModel, stats: Mapping[str, Any]) -> dict[str, Any]:
    """Zero the embedding rows of categories / templates never seen in Train."""
    atom_dims, bond_dims = official_feature_dims()
    report: dict[str, Any] = {"atom": [], "bond": [], "template_zero": []}
    with torch.no_grad():
        for j, emb in enumerate(model.atom_emb):
            seen = set(int(v) for v in stats["atom_support"][j])
            missing = [v for v in range(int(atom_dims[j])) if v not in seen]
            for v in missing:
                emb.weight[v].zero_()
            report["atom"].append({"column": j, "n_categories": int(atom_dims[j]), "n_seen_train": len(seen), "zeroed": missing})
        for j, emb in enumerate(model.bond_emb):
            seen = set(int(v) for v in stats["bond_support"][j])
            missing = [v for v in range(int(bond_dims[j])) if v not in seen]
            for v in missing:
                emb.weight[v].zero_()
            report["bond"].append({"column": j, "n_categories": int(bond_dims[j]), "n_seen_train": len(seen), "zeroed": missing})
        active = torch.as_tensor(stats["template_active"], dtype=torch.float32)
        inactive = [t for t in range(N_TEMPLATES) if float(active[t]) == 0.0]
        for t in inactive:
            model.tmpl_emb.weight[t].zero_()
        model.tmpl_active.copy_(active)
        if inactive:
            report["template_zero"] = inactive
    report["mu_T"] = [float(x) for x in stats["mu_T"]]
    report["nu_k"] = [float(x) for x in stats["nu_k"]]
    return report


def bind_constants(model: StructuralWordModel, stats: Mapping[str, Any]) -> None:
    with torch.no_grad():
        model.mu_T.copy_(torch.as_tensor(stats["mu_T"], dtype=torch.float32))
        model.nu_k.copy_(torch.as_tensor(stats["nu_k"], dtype=torch.float32))
        model.tmpl_active.copy_(torch.as_tensor(stats["template_active"], dtype=torch.float32))


def load_prefix_into(model: StructuralWordModel, state: Mapping[str, Any]) -> list[str]:
    """Copy only the word-bag prefix submodules from ``state``."""
    own = model.state_dict()
    names = model.prefix_parameter_names()
    incoming = {k: v for k, v in state.items() if k in names}
    missing = names - set(incoming)
    if missing:
        raise KeyError(f"prefix checkpoint lacks {sorted(missing)[:5]}")
    own.update(incoming)
    model.load_state_dict(own)
    return sorted(incoming)
