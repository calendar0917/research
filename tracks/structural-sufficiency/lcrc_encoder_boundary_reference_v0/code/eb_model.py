"""B: the real repository GINE encoder feeding A's frozen ``post_local``.

Encoded stack (reused from ``tracks/structural-query-learning/code/sq_gine.py``,
only the depth and the final projection differ):

* node init = sum of 9 per-column atom embeddings(128) + degree embedding + min-ring
  embedding; bond init = sum of 3 per-column bond embeddings(128);
* ``GINE_LAYERS`` ``GINEConv`` blocks (MLP 128->256->128, SiLU, ``train_eps=True``)
  each followed by BatchNorm, SiLU, dropout 0.10 and a residual;
* ``Linear(128 -> 64)`` to the shared read-out width.

The graph-level head is not re-implemented: ``forward`` calls
``bs_model.BSModel.post_local`` on a module that *owns* the reference head modules
(``f``, ``rho``, ``readout`` and the two global buffers) taken from
``bs_model.build_model(scale=1, seed=0)`` -- the same constructor and seed that
produced A.  ``post_local`` therefore reads the exact same module instance and
initialisation; only the per-atom representation differs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch
import torch.nn as nn

import eb_common as E
import cs_common as C

C.setup_paths()

import bs_model as BM  # noqa: E402


class GineEncoder(nn.Module):
    def __init__(self, n_categories: Mapping[str, int]):
        super().__init__()
        h = E.GINE_HIDDEN
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(int(n_categories[f"atom_{j}"]), h) for j in range(9)]
        )
        self.degree_embedding = nn.Embedding(int(n_categories["degree"]), h)
        self.ring_embedding = nn.Embedding(int(n_categories["ring"]), h)
        self.bond_embeddings = nn.ModuleList(
            [nn.Embedding(int(n_categories[f"bond_{j}"]), h) for j in range(3)]
        )
        from torch_geometric.nn import GINEConv

        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(h, E.GINE_MLP_HIDDEN),
                        nn.SiLU(),
                        nn.Linear(E.GINE_MLP_HIDDEN, h),
                    ),
                    train_eps=True,
                )
                for _ in range(E.GINE_LAYERS)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(E.GINE_LAYERS)])
        self.drop = nn.Dropout(E.GINE_DROPOUT)
        self.out_proj = nn.Linear(h, E.OUT_WIDTH)

    def encode_nodes(self, x_cat, deg, ring):
        out = self.atom_embeddings[0](x_cat[:, 0])
        for j in range(1, 9):
            out = out + self.atom_embeddings[j](x_cat[:, j])
        return out + self.degree_embedding(deg) + self.ring_embedding(ring)

    def encode_edges(self, edge_cat):
        out = self.bond_embeddings[0](edge_cat[:, 0])
        for j in range(1, 3):
            out = out + self.bond_embeddings[j](edge_cat[:, j])
        return out

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        h = self.encode_nodes(batch["x_cat"], batch["deg"], batch["ring"])
        edge_attr = self.encode_edges(batch["edge_cat"])
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            update = conv(h, batch["edge_index"], edge_attr)
            h = norm(update)
            h = torch.nn.functional.silu(h)
            h = self.drop(h)
            h = h + residual
        return self.out_proj(h)


class EncoderBoundaryModel(nn.Module):
    """GINE encoder + A's ``post_local`` (same modules, same initialisation)."""

    def __init__(self, encoder: GineEncoder, reference: BM.BSModel):
        super().__init__()
        self.encoder = encoder
        # take the reference head modules as they are (identical init to A)
        self.readout = reference.readout
        self.f = reference.f
        self.rho = reference.rho
        self.n_classes = reference.n_classes
        self.register_buffer("global_mean", reference.global_mean.detach().clone())
        self.register_buffer("global_scale", reference.global_scale.detach().clone())

    def encode_h(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.encoder(batch)

    def forward(self, batch: Mapping[str, Any]):
        h = self.encode_h(batch)
        # literal reuse of the inherited graph-level read-out (no formula copy)
        return BM.BSModel.post_local(self, h, batch)


def parameter_groups(model: nn.Module) -> dict[str, int]:
    groups = {"encoder": 0, "head": 0}
    for name, p in model.named_parameters():
        groups["head" if (name.startswith(E.POST_LOCAL_PREFIXES) or name in E.POST_LOCAL_BUFFERS) else "encoder"] += int(p.numel())
    groups["total"] = sum(p.numel() for p in model.parameters())
    return groups


def build_model(spec: Any, norm: Mapping[str, Any], n_categories: Mapping[str, int], *, seed: int = 0, device="cpu"):
    ref, ref_audit = BM.build_model(
        C.CFG_KEY, spec, norm, C.N_OUT, seed=int(seed), device="cpu", verify_parameters=False
    )
    if int(ref_audit["total_parameters"]) != 259831:
        raise RuntimeError(f"reference BSModel parameters {ref_audit['total_parameters']} != 259831")
    head_keys = E.head_keys(ref.state_dict())
    head_fp = E.state_fingerprint({k: ref.state_dict()[k] for k in head_keys})

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        encoder = GineEncoder(n_categories)
    model = EncoderBoundaryModel(encoder, ref)
    del ref
    model.to(torch.device(device))

    groups = parameter_groups(model)
    ratio = groups["total"] / float(E.A_TOTAL_PARAMS)
    if not (E.PARAM_LOW <= groups["total"] <= E.PARAM_HIGH):
        raise RuntimeError(
            f"B parameters {groups['total']} outside A +/-15% [{E.PARAM_LOW}, {E.PARAM_HIGH}]"
        )
    encoder_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
    audit = {
        "model": "B_GINE_ENCODER",
        "gine_hidden": E.GINE_HIDDEN,
        "gine_mlp_hidden": E.GINE_MLP_HIDDEN,
        "gine_layers": E.GINE_LAYERS,
        "gine_dropout": E.GINE_DROPOUT,
        "out_width": E.OUT_WIDTH,
        "post_local": "BSModel.post_local, literal call",
        "head_keys": head_keys,
        "head_state_sha256": head_fp,
        "encoder_state_sha256": E.state_fingerprint(encoder_state),
        "parameter_groups": groups,
        "a_total_parameters": E.A_TOTAL_PARAMS,
        "param_ratio_b_over_a": ratio,
        "param_budget": [E.PARAM_LOW, E.PARAM_HIGH],
        "param_budget_ok": True,
        "reference_total_parameters": int(ref_audit["total_parameters"]),
        "reference_head_parameters": int(
            sum(p.numel() for n, p in model.named_parameters() if n.startswith(("f.", "rho.", "readout.")))
        ),
        "categories": {k: int(v) for k, v in n_categories.items()},
        "init_rule": (
            "head: BM.build_model(scale=1, seed=0) reference state, byte-identical to A; "
            "encoder: torch.manual_seed(0) inside fork_rng then default PyTorch init. "
            "Different encoders cannot be parameter-matched; no paired-init claim."
        ),
    }
    return model, audit


def head_state_hash_from_checkpoint(path) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    return E.state_fingerprint({k: state[k] for k in E.head_keys(state)})


def encoder_parameter_filter(name: str) -> bool:
    return not (name.startswith(E.POST_LOCAL_PREFIXES) or name in E.POST_LOCAL_BUFFERS)
