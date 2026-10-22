"""B_object_tokens: one prediction network over atom / rooted-branch / cycle tokens.

Token types (fixed ids): 0 = A (atom), 1 = S3, 2 = S4, 3 = F4, 4 = C (cycle).

Every real object is a token with its own hidden state:

    h_A(v) = Linear96->96(a_v)
    h_F(o) = MLP228->96->96([a_root, mean_{u in members} a_u,
                             phi_branch, log|members|, family_onehot3])
    h_C(o) = MLP129->96->96([mean_{u in ring} a_u, phi_ring, log L])

plus one learned 5 x 96 type embedding.  ``a_v`` is the frozen-P3 *atom*
encoder (train-only masked-SSL initialisation, then fine-tuned) applied to the
raw108 atom descriptor.  Nothing from P3's bond / junction / composer / reader
stack is inherited.

The trunk is four pre-LN Transformer blocks (width 96, 4 heads, FFN 192,
SiLU, dropout 0.05).  Attention is full within a molecule: every real token
pair may interact, only padding and other molecules are masked.  Five static
relation tables enter as an additive per-head logit bias; they are shared by
all four layers.  There is no relation-conditioned value tensor, no local
adjacency mask and no repeated hidden-state update along object incidence.

The readout keeps magnitude: for the atom / branch / cycle groups it uses
``sum / fixed train mean count`` and ``mean`` (empty group -> exact zero), then
global33 and three log1p counts; the reader is 612 -> 96 -> 48 -> 1.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ot_data as O  # noqa: E402

WIDTH = 96
HEADS = 4
FF = 192
DROPOUT = 0.05
ATOM_CLASSES = 28
D_ATOM = 8
D_PAIR = 4
PHI_HIDDEN = 64
PHI_OUT = 32
N_REL_TYPE = 25      # ordered 5 x 5 token-type pairs
N_REL_OVERLAP = 9    # |S_i cap S_j| = 0..8
N_REL_DIST = 11      # 0..8 exact, 9 = finite >= 9, 10 = disconnected
N_REL_BOND = 5       # 0 = not atom-atom, 1 = atom-atom no bond, 2..4 = bond 1..3
N_REL_ROOT = 3       # 0 = a side has no root, 1 = different roots, 2 = same root
N_GROUPS = 3
READER_IN = 6 * WIDTH + D.GLOBAL_DIM + N_GROUPS  # 612


class TransformerBlock(nn.Module):
    """Pre-LN block: LN -> QKV -> full attention(+static bias) -> res -> LN -> FFN -> res."""

    def __init__(self, d: int = WIDTH, heads: int = HEADS, ff: int = FF, dropout: float = DROPOUT):
        super().__init__()
        if d % heads:
            raise ValueError("width must be divisible by heads")
        self.d = int(d)
        self.heads = int(heads)
        self.hd = self.d // self.heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d),
            nn.Dropout(dropout),
        )
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        ctx = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, self.d)
        x = x + self.out(ctx)
        x = x * valid.unsqueeze(-1)
        x = x + self.ff(self.ln2(x))
        return x * valid.unsqueeze(-1)


class ObjectTokenModel(nn.Module):
    def __init__(self, norm: Mapping[str, object], *, group_mean: Mapping[str, float] | None = None):
        super().__init__()
        # small P3 atom encoder only (raw108 -> 128 -> 96); no bond/junction stack
        self.atom_encoder = M1.MLP(D.ATOM_FEAT_DIM, 128, WIDTH, 1, 0.0)
        # shared raw content embeddings (one table each, registered once)
        self.atom_emb = nn.Embedding(ATOM_CLASSES, D_ATOM)
        self.pair_emb = nn.Embedding(4, D_PAIR)
        # content encoders
        branch_in = O.MAX_NODES * D_ATOM + O.N_PAIRS * D_PAIR + O.MAX_NODES + O.N_PAIRS + len(O.FAMILIES)
        self.branch_phi = M1.MLP(branch_in, PHI_HIDDEN, PHI_OUT, 1, 0.0)          # 98 -> 64 -> 32
        ring_in = O.MAX_RING * D_ATOM + O.MAX_RING * D_PAIR + O.MAX_RING + O.MAX_RING + 6
        self.ring_phi = M1.MLP(ring_in, PHI_HIDDEN, PHI_OUT, 1, 0.0)              # 118 -> 64 -> 32
        # token projections
        self.proj_atom = nn.Linear(WIDTH, WIDTH)
        self.proj_branch = M1.MLP(WIDTH + WIDTH + PHI_OUT + 1 + len(O.FAMILIES),
                                  WIDTH, WIDTH, 1, 0.0)                            # 228 -> 96 -> 96
        self.proj_cycle = M1.MLP(WIDTH + PHI_OUT + 1, WIDTH, WIDTH, 1, 0.0)        # 129 -> 96 -> 96
        self.type_emb = nn.Embedding(O.N_TOKEN_TYPES, WIDTH)
        # static relation tables, shared by all four blocks
        self.rel_type = nn.Parameter(torch.zeros(N_REL_TYPE, HEADS))
        self.rel_overlap = nn.Parameter(torch.zeros(N_REL_OVERLAP, HEADS))
        self.rel_dist = nn.Parameter(torch.zeros(N_REL_DIST, HEADS))
        self.rel_bond = nn.Parameter(torch.zeros(N_REL_BOND, HEADS))
        self.rel_root = nn.Parameter(torch.zeros(N_REL_ROOT, HEADS))
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(4)])
        self.reader = nn.Sequential(
            nn.Linear(READER_IN, 96),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(48, 1),
        )
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        mean = dict(group_mean or {"atom": 1.0, "branch": 1.0, "cycle": 1.0})
        gm = torch.as_tensor(
            [max(float(mean["atom"]), 1.0), max(float(mean["branch"]), 1.0),
             max(float(mean["cycle"]), 1.0)],
            dtype=torch.float32,
        )
        self.register_buffer("group_mean", gm)
        self.register_buffer("pair_j", torch.as_tensor(O.PAIR_J, dtype=torch.long))

    # -- content encoders ---------------------------------------------------
    def branch_content(self, batch) -> torch.Tensor:
        atom = batch["br_pat_atom"]
        pair = batch["br_pat_pair"]
        length = batch["br_pat_len"]
        P = int(atom.shape[0])
        dev = atom.device
        node_mask = (torch.arange(O.MAX_NODES, device=dev).unsqueeze(0) < length.unsqueeze(1))
        pair_mask = (self.pair_j.unsqueeze(0) < length.unsqueeze(1))
        nm = node_mask.to(torch.float32)
        pm = pair_mask.to(torch.float32)
        fam = F.one_hot(batch["br_pat_family"], num_classes=len(O.FAMILIES)).to(torch.float32)
        x = torch.cat([
            (self.atom_emb(atom) * nm.unsqueeze(-1)).reshape(P, -1),
            (self.pair_emb(pair) * pm.unsqueeze(-1)).reshape(P, -1),
            nm, pm, fam,
        ], dim=1)
        return self.branch_phi(x)

    def ring_content(self, batch) -> torch.Tensor:
        atom = batch["ring_pat_atom"]
        bond = batch["ring_pat_bond"]
        length = batch["ring_pat_len"]
        P = int(atom.shape[0])
        dev = atom.device
        pos = torch.arange(O.MAX_RING, device=dev).unsqueeze(0)
        node_mask = (pos < length.unsqueeze(1)).to(torch.float32)
        edge_mask = (pos < length.unsqueeze(1)).to(torch.float32)
        length_oh = F.one_hot(length - 3, num_classes=6).to(torch.float32)
        x = torch.cat([
            (self.atom_emb(atom) * node_mask.unsqueeze(-1)).reshape(P, -1),
            (self.pair_emb(bond) * edge_mask.unsqueeze(-1)).reshape(P, -1),
            node_mask, edge_mask, length_oh,
        ], dim=1)
        return self.ring_phi(x)

    # -- static relation bias ----------------------------------------------
    def relation_categories(self, batch, B: int, T: int) -> dict[str, torch.Tensor]:
        """Integer relation category per token pair, computed from static inputs.

        Returned keys: ``type`` (5x5 ordered token types), ``overlap``
        (|S_i cap S_j|), ``dist`` (min atom distance), ``bond`` (atom-atom
        direct bond) and ``root``.  The model only uses these through the
        additive per-head tables; exposing them makes the semantics checkable
        against an independent implementation.
        """
        K = O.MAX_MEMBERS
        dev = batch["tok_type_pad"].device
        tt = batch["tok_type_pad"].view(B, T)
        root = batch["tok_root_pad"].view(B, T)
        mem = batch["tok_members_pad"].view(B, T, K)
        mmask = batch["tok_member_mask"].view(B, T, K)
        dist = batch["atom_dist"]
        bond = batch["atom_bond"]
        big = 10 ** 4

        pair_idx = tt.unsqueeze(2) * O.N_TOKEN_TYPES + tt.unsqueeze(1)

        # |S_i cap S_j| via the atom-incidence matrix (exact for <=8 members)
        nmax = int(batch["Nmax"])
        inc = torch.zeros((B, T, nmax + 1), dtype=torch.float32, device=dev)
        idx = torch.where(mmask, mem, torch.full_like(mem, nmax))
        inc.scatter_(2, idx, torch.ones_like(idx, dtype=torch.float32))
        inc = inc[:, :, :nmax]
        overlap = (inc @ inc.transpose(1, 2)).round().long()

        # d2[b, t, n] = min over real members v of token t of dist(v, n)
        d2 = torch.full((B, T, batch["Nmax"]), big, dtype=torch.long, device=dev)
        for k in range(K):
            g = dist.gather(1, mem[:, :, k].unsqueeze(-1).expand(-1, -1, batch["Nmax"]))
            d2 = torch.where(mmask[:, :, k].unsqueeze(-1), torch.minimum(d2, g), d2)
        d2t = d2.transpose(1, 2).contiguous()  # [B, N, T]
        mind = torch.full((B, T, T), big, dtype=torch.long, device=dev)
        for k in range(K):
            # index[i, j] = member k of token i (depends on the query row i)
            idx = mem[:, :, k].unsqueeze(2).expand(-1, -1, T)
            g = d2t.gather(1, idx)
            mind = torch.where(mmask[:, :, k].unsqueeze(2), torch.minimum(mind, g), mind)
        dcat = torch.where(mind >= big, torch.full_like(mind, 10),
                           torch.where(mind >= 9, torch.full_like(mind, 9), mind))

        atom_only = tt == 0
        atom_id = torch.where(atom_only, mem[:, :, 0], torch.zeros_like(mem[:, :, 0]))
        nmax = int(batch["Nmax"])
        bmat = bond.gather(1, atom_id.unsqueeze(2).expand(-1, -1, nmax)).gather(
            2, atom_id.unsqueeze(1).expand(-1, T, -1))
        both = atom_only.unsqueeze(2) & atom_only.unsqueeze(1)
        bcat = torch.where(
            both,
            torch.where(bmat > 0, bmat + 1, torch.ones_like(bmat)),
            torch.zeros_like(bmat),
        )

        has = root >= 0
        both_root = has.unsqueeze(2) & has.unsqueeze(1)
        same = (root.unsqueeze(2) == root.unsqueeze(1)).long()
        rcat = torch.where(both_root, same + 1, torch.zeros_like(same))
        return {"type": pair_idx, "overlap": overlap, "dist": dcat,
                "bond": bcat, "root": rcat}

    def relation_bias(self, batch, B: int, T: int) -> torch.Tensor:
        cat = self.relation_categories(batch, B, T)
        bias = (
            self.rel_type[cat["type"]]
            + self.rel_overlap[cat["overlap"]]
            + self.rel_dist[cat["dist"]]
            + self.rel_bond[cat["bond"]]
            + self.rel_root[cat["root"]]
        )
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, T, T]
        valid_keys = batch["tok_valid"].view(B, T)
        return bias.masked_fill(~valid_keys[:, None, None, :], -1e9)

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        B = int(batch["n_graphs"])
        T = int(batch["Tmax"])
        dev = batch["atom_feat"].device
        a = self.atom_encoder(batch["atom_feat"])

        h = a.new_zeros((B * T, WIDTH))
        h[batch["atom_tok_pos"]] = self.proj_atom(a)

        if int(batch["branch_tok_pos"].numel()):
            root_a = a[batch["branch_root"]]
            bmem = batch["branch_members"]
            bmask = torch.arange(O.MAX_BRANCH_MEMBERS, device=dev).unsqueeze(0) < \
                batch["branch_nmembers"].unsqueeze(1)
            mem_a = (a[bmem] * bmask.unsqueeze(-1).to(a.dtype)).sum(dim=1)
            mean_a = mem_a / batch["branch_nmembers"].to(a.dtype).clamp(min=1).unsqueeze(1)
            phi = self.branch_content(batch)[batch["branch_pat"]]
            fam = F.one_hot(batch["branch_family"], num_classes=len(O.FAMILIES)).to(a.dtype)
            x = torch.cat([
                root_a, mean_a, phi,
                torch.log(batch["branch_nmembers"].to(a.dtype)).unsqueeze(1),
                fam,
            ], dim=1)
            h[batch["branch_tok_pos"]] = self.proj_branch(x)

        if int(batch["ring_tok_pos"].numel()):
            rmem = batch["ring_members"]
            rlen = batch["ring_len"]
            rmask = torch.arange(O.MAX_MEMBERS, device=dev).unsqueeze(0) < rlen.unsqueeze(1)
            mem_a = (a[rmem] * rmask.unsqueeze(-1).to(a.dtype)).sum(dim=1)
            mean_a = mem_a / rlen.to(a.dtype).clamp(min=1).unsqueeze(1)
            phi = self.ring_content(batch)[batch["ring_pat"]]
            x = torch.cat([
                mean_a, phi, torch.log(rlen.to(a.dtype)).unsqueeze(1),
            ], dim=1)
            h[batch["ring_tok_pos"]] = self.proj_cycle(x)

        valid = batch["tok_valid"]
        h = h + self.type_emb(batch["tok_type_pad"])
        h = h * valid.unsqueeze(-1).to(h.dtype)

        bias = self.relation_bias(batch, B, T)
        x = h.view(B, T, WIDTH)
        valid2 = valid.view(B, T)
        for block in self.blocks:
            x = block(x, bias, valid2)

        type3 = batch["tok_type_pad"].view(B, T)
        group = torch.where(type3 == 0, torch.zeros_like(type3),
                            torch.where(type3 == O.TYPE_C,
                                        torch.full_like(type3, 2), torch.ones_like(type3)))
        parts = []
        counts = []
        for g in range(N_GROUPS):
            m = ((group == g) & valid2).to(x.dtype)
            s = (x * m.unsqueeze(-1)).sum(dim=1)
            cnt = m.sum(dim=1)
            parts.append(s / self.group_mean[g])
            parts.append(s / cnt.clamp(min=1.0).unsqueeze(1))
            counts.append(torch.log1p(cnt).unsqueeze(1))
        gfeat = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat(parts + [gfeat] + counts, dim=1)
        return self.reader(z).view(-1)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

SSL_ATOM_SHA256 = "006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97"


def build_model(norm: Mapping[str, object], *, group_mean: Mapping[str, float] | None = None,
                device="cpu", seed: int = 0) -> tuple[ObjectTokenModel, dict]:
    torch.manual_seed(int(seed))
    model = ObjectTokenModel(norm, group_mean=group_mean)
    path = C.P3_SSL_S0
    if not path.exists():
        raise FileNotFoundError(
            f"missing train-only SSL atom encoder {path}; refusing to silently "
            "fall back to a random atom encoder"
        )
    sha = C.sha256_file(path)
    if sha != SSL_ATOM_SHA256:
        raise RuntimeError(f"SSL atom encoder sha256 {sha} != {SSL_ATOM_SHA256}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.atom_encoder.load_state_dict({k: v for k, v in state["atom_encoder"].items()})
    model = model.to(device)
    total = C.parameter_count(model)
    info = {
        "candidate": "B_object_tokens",
        "seed": int(seed),
        "parameters": total,
        "parameters_breakdown": parameter_breakdown(model),
        "parameter_cap": 500_000,
        "ssl_atom_encoder": {
            "source": str(path.relative_to(C.REPO)),
            "sha256": sha,
            "loaded_keys": sorted(state["atom_encoder"].keys()),
            "note": "atom encoder only; P3 bond/junction/composer/reader are not inherited",
        },
        "all_require_grad": bool(all(p.requires_grad for p in model.parameters())),
    }
    if total > 500_000:
        raise RuntimeError(f"{total} parameters > 500k")
    return model, info


def parameter_breakdown(model: ObjectTokenModel) -> dict[str, int]:
    def n(mod):
        return int(sum(p.numel() for p in mod.parameters()))

    out = {
        "atom_encoder": n(model.atom_encoder),
        "shared_raw_embeddings": n(model.atom_emb) + n(model.pair_emb),
        "branch_content_encoder": n(model.branch_phi),
        "ring_content_encoder": n(model.ring_phi),
        "token_projections": n(model.proj_atom) + n(model.proj_branch) + n(model.proj_cycle),
        "type_embedding": n(model.type_emb),
        "transformer_blocks": n(model.blocks),
        "relation_tables": int(sum(model.rel_type.numel() + model.rel_overlap.numel()
                                   + model.rel_dist.numel() + model.rel_bond.numel()
                                   + model.rel_root.numel() for _ in (0,))),
        "reader": n(model.reader),
    }
    out["total"] = int(sum(p.numel() for p in model.parameters()))
    return out
