"""Dense (padded) GRIT backbone for zinc-performance-search-v1, arm C.

The model is a faithful port of the vendored upstream ``GritTransformer``
(`zinc_grit_reference_preparation_v1/upstream/GRIT`, commit 6c988ea): typed
atom/bond embeddings, RRWP I,P,...,P^20 node/pair encodings, full-attention
GritTransformer layers with edge-state updates, degree scaler, BatchNorm over
real nodes/pairs, and the ``san_graph`` sum-pool + 3-layer MLP head.  The only
change is the *execution*: every graph is padded to a dense (B, N, N) layout so
that the per-layer work is batched tensor algebra instead of scatter/index_add.

This is an implementation adaptation of an existing method, not a new method.

Official test is never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_ZPP = _SS / "zinc_perf_protocol_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_ZPP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
CACHE = EXP / "cache"
ROUND_ID = "zinc-performance-search-v1"

# Native upstream recipe (configs/zinc-GRIT-RRWP.yaml) for the declared C0 run.
NATIVE = dict(
    layers=10, n_heads=8, dim_hidden=64, ksteps=21,
    dropout=0.0, attn_dropout=0.2, layer_norm=False, batch_norm=True,
    update_e=True, edge_enhance=True, O_e=True, norm_e=True, deg_scaler=True,
    clamp=5.0, act="relu", graph_pooling="sum",
    optimizer="adamw", base_lr=1e-3, min_lr=1e-6, weight_decay=1e-5,
    warmup_epochs=50, max_epochs=2000, batch_size=32, clip_grad_norm=1.0,
    loss="l1",
)

_PARAM_KEYS = None


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _rrwp_from_edge_index(edge_index: torch.Tensor, n: int, ksteps: int = 21):
    """I, P, P^2, ..., P^(ksteps-1) with P = D^{-1} A (same as upstream)."""
    adj = torch.zeros((n, n), dtype=torch.float64)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(dim=1)
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    p = adj * deg_inv.view(-1, 1)
    out = [torch.eye(n, dtype=torch.float64), p]
    cur = p
    for _ in range(ksteps - 2):
        cur = cur @ p
        out.append(cur)
    pe = torch.stack(out, dim=-1)
    return pe, deg


def _load_reference(split: str, ksteps: int):
    """Official graphs; train has a frozen RRWP cache, valid only the raw split."""
    from zpp_common import CACHE as ZPP_CACHE
    base = ZPP_CACHE / f"zinc_subset_{split}_norrwp.pt"
    ref = torch.load(base, map_location="cpu", weights_only=False)
    rrwp_cache = ZPP_CACHE / f"zinc_subset_{split}_rrwp{ksteps}.pt"
    cached = torch.load(rrwp_cache, map_location="cpu", weights_only=False) if rrwp_cache.exists() else None
    return ref, cached


def prepare(split: str, *, ksteps: int = 21, cache_dir: Path = CACHE) -> list[dict]:
    """Per-graph dense arrays; RRWP is verified against the frozen upstream cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"dense_{split}_rrwp{ksteps}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    ref, cached = _load_reference(split, ksteps)
    graphs = []
    max_diff = 0.0
    for i, d in enumerate(ref):
        n = int(d.num_nodes)
        pe, deg = _rrwp_from_edge_index(d.edge_index.long(), n, ksteps)
        if cached is not None:
            src = cached[i]
            max_diff = max(max_diff, float((pe - src.rrwp.double()).abs().max()))
            de = float((deg - src.deg.double()).abs().max())
            if de > 0:
                raise RuntimeError(f"{split}[{i}] degree mismatch {de}")
        bond = torch.zeros((n, n), dtype=torch.long)
        ei = d.edge_index.long()
        bond[ei[0], ei[1]] = d.edge_attr.long().view(-1).clamp(0, 3)
        graphs.append(
            dict(
                x=d.x.long().view(-1).clamp(0, 20),
                bond=bond,
                rrwp=pe.float(),
                deg=deg.float(),
                log_deg=torch.log(deg.float() + 1.0),
                y=float(d.y.view(-1)[0]),
                n=n,
            )
        )
    if cached is not None and max_diff > 1e-5:
        raise RuntimeError(f"{split}: dense RRWP disagrees with frozen cache by {max_diff}")
    torch.save(graphs, path)
    print(f"[dense] prepared {split}: {len(graphs)} graphs, rrwp max|diff vs cache|="
          f"{max_diff:.3e} (cache={cached is not None})", flush=True)
    return graphs


def collate(graphs: list[dict], rows, device="cpu") -> dict:
    rows = [int(r) for r in rows]
    B = len(rows)
    N = max(graphs[r]["n"] for r in rows)
    K = graphs[rows[0]]["rrwp"].shape[-1]
    rrwp = np.zeros((B, N, N, K), dtype=np.float32)
    bond = np.zeros((B, N, N), dtype=np.int64)
    x = np.zeros((B, N), dtype=np.int64)
    deg = np.zeros((B, N), dtype=np.float32)
    log_deg = np.zeros((B, N), dtype=np.float32)
    mask = np.zeros((B, N), dtype=bool)
    y = np.zeros((B,), dtype=np.float32)
    for b, r in enumerate(rows):
        g = graphs[r]
        n = g["n"]
        rrwp[b, :n, :n] = g["rrwp"].numpy()
        bond[b, :n, :n] = g["bond"].numpy()
        x[b, :n] = g["x"].numpy()
        deg[b, :n] = g["deg"].numpy()
        log_deg[b, :n] = g["log_deg"].numpy()
        mask[b, :n] = True
        y[b] = g["y"]
    out = {
        "x": torch.from_numpy(x),
        "bond": torch.from_numpy(bond),
        "rrwp": torch.from_numpy(rrwp),
        "mask": torch.from_numpy(mask),
        "log_deg": torch.from_numpy(log_deg),
        "deg": torch.from_numpy(deg),
        "y": torch.from_numpy(y),
    }
    if device is not None:
        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    return out


def batch_pred(model, batch):
    return model(batch["x"], batch["bond"], batch["rrwp"], batch["mask"], batch["log_deg"])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class DenseGritAttention(nn.Module):
    def __init__(self, dim, heads, attn_dropout=0.0, clamp=5.0,
                 edge_enhance=True, act="relu"):
        super().__init__()
        self.out_dim = dim // heads
        self.num_heads = heads
        self.dim = dim
        self.dropout = nn.Dropout(attn_dropout)
        self.clamp = clamp
        self.edge_enhance = edge_enhance
        self.act = {"relu": F.relu, "gelu": F.gelu}[act]

        self.Q = nn.Linear(dim, dim, bias=True)
        self.K = nn.Linear(dim, dim, bias=False)
        self.E = nn.Linear(dim, dim * 2, bias=True)
        self.V = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)
        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1))
        nn.init.xavier_normal_(self.Aw)
        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim))
            nn.init.xavier_normal_(self.VeRow)

    def forward(self, x, e, pair_mask, _log_deg=None):
        B, N, dim = x.shape
        H, d = self.num_heads, self.out_dim
        Q = self.Q(x).view(B, N, H, d)
        K = self.K(x).view(B, N, H, d)
        V = self.V(x).view(B, N, H, d)
        E = self.E(e).view(B, N, N, H, 2, d)
        E_w, E_b = E[..., 0, :], E[..., 1, :]

        score = (K.unsqueeze(1) + Q.unsqueeze(2)) * E_w
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = score + E_b
        score = self.act(score)
        e_t = score
        wE = score.reshape(B, N, N, dim)

        logits = torch.einsum("bijhd,dhc->bijhc", score, self.Aw)
        if self.clamp is not None:
            logits = torch.clamp(logits, min=-self.clamp, max=self.clamp)
        logits = logits.masked_fill(~pair_mask[:, :, :, None, None], -1e9)
        attn = torch.softmax(logits, dim=2)
        attn = self.dropout(attn)

        wV = torch.einsum("bijh,bijhd->bihd", attn.squeeze(-1), V.unsqueeze(1).expand(-1, N, -1, -1, -1))
        if self.edge_enhance:
            rowV = torch.einsum("bijhd,bijh->bihd", e_t, attn.squeeze(-1))
            rowV = torch.einsum("bihd,dhc->bihc", rowV, self.VeRow)
            wV = wV + rowV
        return wV.reshape(B, N, dim), wE


class DenseGritLayer(nn.Module):
    def __init__(self, dim=64, heads=8, dropout=0.0, attn_dropout=0.2,
                 layer_norm=False, batch_norm=True, residual=True, act="relu",
                 norm_e=True, O_e=True, deg_scaler=True, clamp=5.0,
                 edge_enhance=True, bn_momentum=0.1):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.deg_scaler = deg_scaler
        self.act = {"relu": F.relu, "gelu": F.gelu}[act]
        self.attention = DenseGritAttention(dim, heads, attn_dropout, clamp,
                                            edge_enhance, act)
        self.O_h = nn.Linear(dim, dim)
        self.O_e = nn.Linear(dim, dim) if O_e else nn.Identity()
        if deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, dim, 2))
            nn.init.xavier_normal_(self.deg_coef)
        if batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(dim, track_running_stats=True, eps=1e-5,
                                                momentum=bn_momentum)
            self.batch_norm1_e = (nn.BatchNorm1d(dim, track_running_stats=True, eps=1e-5,
                                                 momentum=bn_momentum)
                                  if norm_e else nn.Identity())
        if layer_norm:
            self.layer_norm1_h = nn.LayerNorm(dim)
            self.layer_norm1_e = nn.LayerNorm(dim) if norm_e else nn.Identity()
        self.FFN_h_layer1 = nn.Linear(dim, dim * 2)
        self.FFN_h_layer2 = nn.Linear(dim * 2, dim)
        if batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(dim, track_running_stats=True, eps=1e-5,
                                                momentum=bn_momentum)
        if layer_norm:
            self.layer_norm2_h = nn.LayerNorm(dim)

    @staticmethod
    def _bn_masked(bn, values, flat_mask):
        shape = values.shape
        sel = bn(values.reshape(-1, shape[-1])[flat_mask])
        out = torch.zeros_like(values.reshape(-1, shape[-1]))
        out[flat_mask] = sel
        return out.view(shape)

    def forward(self, x, e, mask, pair_mask, log_deg):
        B, N, dim = x.shape
        h = x
        h_in1 = h
        e_in1 = e

        h_attn, e_attn = self.attention(h, e, pair_mask, log_deg)
        h = F.dropout(h_attn, self.dropout, training=self.training)
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg.unsqueeze(-1)], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)
        h = self.O_h(h)
        e = F.dropout(e_attn, self.dropout, training=self.training)
        e = self.O_e(e)
        if self.residual:
            h = h_in1 + h
            e = e + e_in1
        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self._bn_masked(self.batch_norm1_h, h, mask.reshape(-1))
            e = self._bn_masked(self.batch_norm1_e, e, pair_mask.reshape(-1))

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        if self.residual:
            h = h_in2 + h
        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self._bn_masked(self.batch_norm2_h, h, mask.reshape(-1))
        return h, e


class DenseGrit(nn.Module):
    """Same parameter names/shapes as the upstream GritTransformer."""

    def __init__(self, cfg: dict = None, num_atom_types=21, num_bond_types=4):
        super().__init__()
        cfg = dict(NATIVE if cfg is None else cfg)
        self.cfg = cfg
        dim = int(cfg["dim_hidden"])
        heads = int(cfg["n_heads"])
        k = int(cfg["ksteps"])

        class _NodeEnc(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Embedding(num_atom_types, dim)

        class _EdgeEnc(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Embedding(num_bond_types, dim)

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_encoder = _NodeEnc()
                self.edge_encoder = _EdgeEnc()

        class _FC(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(k, dim, bias=False)
                nn.init.xavier_uniform_(self.fc.weight)

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.FC_layers = nn.ModuleList([
                    nn.Linear(dim, dim // 2, bias=True),
                    nn.Linear(dim // 2, dim // 4, bias=True),
                    nn.Linear(dim // 4, 1, bias=True),
                ])

        self.encoder = _Encoder()
        self.rrwp_abs_encoder = _FC()
        self.rrwp_rel_encoder = _FC()
        # upstream registers an (unused) zero-padding buffer on this encoder
        self.rrwp_rel_encoder.register_buffer("padding", torch.zeros(1, dim))
        self.layers = nn.ModuleList([
            DenseGritLayer(
                dim=dim, heads=heads, dropout=float(cfg["dropout"]),
                attn_dropout=float(cfg["attn_dropout"]),
                layer_norm=bool(cfg["layer_norm"]), batch_norm=bool(cfg["batch_norm"]),
                act=cfg["act"], norm_e=bool(cfg["norm_e"]), O_e=bool(cfg["O_e"]),
                deg_scaler=bool(cfg["deg_scaler"]), clamp=float(cfg["clamp"]),
                edge_enhance=bool(cfg["edge_enhance"]),
            )
            for _ in range(int(cfg["layers"]))
        ])
        self.post_mp = _Head()

    def forward(self, x_type, bond_type, rrwp, mask, log_deg, return_embedding=False):
        B, N = x_type.shape
        pair_mask = mask[:, :, None] & mask[:, None, :]
        x = self.encoder.node_encoder.encoder(x_type)
        abs_pe = torch.diagonal(rrwp, dim1=1, dim2=2).transpose(1, 2)
        x = x + self.rrwp_abs_encoder.fc(abs_pe)
        e = self.rrwp_rel_encoder.fc(rrwp)
        # upstream adds the bond embedding only on real edges and a zero
        # vector on self-loops / non-edges (never the NONE embedding row)
        is_bond = (bond_type > 0) & pair_mask
        e = e + self.encoder.edge_encoder.encoder(bond_type) * is_bond.unsqueeze(-1)
        x = x * mask.unsqueeze(-1)
        e = e * pair_mask.unsqueeze(-1)

        for layer in self.layers:
            x, e = layer(x, e, mask, pair_mask, log_deg)

        graph = (x * mask.unsqueeze(-1)).sum(dim=1)
        if return_embedding:
            return graph
        h = graph
        for i in range(2):
            h = self.post_mp.FC_layers[i](h)
            h = F.relu(h)
        h = self.post_mp.FC_layers[2](h)
        return h


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def _sha_state(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


@torch.no_grad()
def predict(model, graphs, rows, device, batch_size=256):
    model.eval()
    preds = []
    with C.isolated_rng():
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            batch = collate(graphs, chunk, device)
            preds.append(batch_pred(model, batch).detach().float().cpu().view(-1).numpy())
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)


def evaluate(model, graphs, rows, y, device, batch_size=256):
    pred = predict(model, graphs, rows, device, batch_size)
    return pred, float(np.abs(pred.astype(np.float64) - y[rows]).mean())


def _lr_at(step, total, warmup, base, min_lr):
    return C.cosine_with_warmup_lr(step, total, warmup, base, min_lr)


def run_dir(tag: str, seed: int, variant: str) -> Path:
    return RUNS / f"{tag}{variant}_s{int(seed)}"


def train(*, variant: str = "C0", seed: int = 0, device="cuda:0",
          epochs: int | None = None, batch_size: int | None = None,
          tag: str | None = None, force: bool = False, reason: str | None = None,
          smoke_epochs: int | None = None) -> dict:
    cfg = dict(NATIVE)
    ep = int(epochs or cfg["max_epochs"])
    bs = int(batch_size or cfg["batch_size"])
    out = run_dir(tag or "gritd_", seed, variant)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        print(f"[gritd] already complete: {out}")
        return C.read_json(out / "train.json")

    train_g = prepare("train")
    valid_g = prepare("valid")
    y_train = np.asarray([g["y"] for g in train_g], dtype=np.float64)
    y_valid = np.asarray([g["y"] for g in valid_g], dtype=np.float64)
    steps_per_epoch = math.ceil(len(train_g) / bs)
    total_steps = steps_per_epoch * ep
    warmup_steps = int(cfg["warmup_epochs"]) * steps_per_epoch

    torch.manual_seed(int(seed))
    model = DenseGrit(cfg).to(device)
    n_params = param_count(model)
    if n_params > 500_000:
        raise RuntimeError(f"parameter cap exceeded: {n_params}")
    init_hash = _sha_state(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["base_lr"]),
                            weight_decay=float(cfg["weight_decay"]),
                            betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator().manual_seed(int(seed) * 1000 + 7)
    curve, best = [], {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    snap_rows = np.arange(1000)
    step = 0
    started = time.perf_counter()
    declaration = {
        "round": ROUND_ID, "arm": "C", "variant": variant, "seed": int(seed),
        "config": cfg, "epochs": ep, "batch_size": bs,
        "steps_per_epoch": steps_per_epoch, "updates": total_steps,
        "warmup_steps": warmup_steps, "reason": reason,
        "parameter_count": n_params, "initial_state_sha256": init_hash,
        "implementation": "dense padded port of vendored upstream GRIT "
                          "(commit 6c988ea600a606fbb49a2246c64a2d37396b3ab5)",
        "official_test_loaded": False,
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_json(out / "search.json", declaration)

    for epoch in range(1, ep + 1):
        model.train()
        order = torch.randperm(len(train_g), generator=gen).tolist()
        t_ep = time.perf_counter()
        total_err, n_mol, last_lr = 0.0, 0, float(cfg["base_lr"])
        for s in range(0, len(order), bs):
            rows = order[s:s + bs]
            batch = collate(train_g, rows, device)
            pred = batch_pred(model, batch).view(-1)
            target = batch["y"].view(-1)
            loss = F.l1_loss(pred, target)
            step += 1
            last_lr = _lr_at(step, total_steps, warmup_steps,
                             float(cfg["base_lr"]), float(cfg["min_lr"]))
            for g_ in opt.param_groups:
                g_["lr"] = last_lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip_grad_norm"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            opt.step()
            total_err += float((pred.detach() - target).abs().sum())
            n_mol += int(pred.numel())
        online = total_err / max(n_mol, 1)
        _, valid_mae = evaluate(model, valid_g, np.arange(len(valid_g)), y_valid, device)
        sec = time.perf_counter() - t_ep
        curve.append({"epoch": epoch, "online_train_mae": online,
                      "valid_mae": float(valid_mae), "lr": last_lr,
                      "seconds": sec, "steps": steps_per_epoch})
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = C.cpu_state(model)
            torch.save({"state_dict": best_state, "epoch": int(epoch),
                        "valid_mae": float(valid_mae), "variant": variant,
                        "seed": int(seed)}, out / "best.pt")
        print(f"[gritd {variant} s{seed}] ep={epoch:04d} valid={valid_mae:.6f} "
              f"online={online:.6f} lr={last_lr:.2e} best={best['valid_mae']:.6f}@"
              f"{best['epoch']} {sec:.2f}s", flush=True)
        if smoke_epochs is not None and epoch >= smoke_epochs:
            break

    wall = time.perf_counter() - started
    torch.save({"state_dict": C.cpu_state(model), "epoch": ep, "best": best,
                "curve": curve, "step": step}, out / "final.pt")
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    best_model = DenseGrit(cfg).to(device)
    if best_state is not None:
        C.load_state(best_model, best_state, device)
        pred_best, best_valid = evaluate(best_model, valid_g, np.arange(len(valid_g)),
                                        y_valid, device)
        _, best_train = evaluate(best_model, train_g, np.arange(len(train_g)), y_train,
                                 device, batch_size=512)
        np.savez(pred_dir / "valid_best.npz", row=np.arange(len(valid_g)), pred=pred_best)
        _, final_valid = evaluate(model, valid_g, np.arange(len(valid_g)), y_valid, device)
        _, final_train = evaluate(model, train_g, np.arange(len(train_g)), y_train,
                                  device, batch_size=512)
    else:
        best_valid = best_train = final_valid = final_train = float("nan")
    info = {
        "round": ROUND_ID, "candidate": "C", "variant": variant, "seed": int(seed),
        "git_commit": C.git_commit(), "environment": C.environment(),
        "data": {"source": "official PyG ZINC subset=True", "n_train": len(train_g),
                 "n_valid": len(valid_g), "official_test_loaded": False,
                 "train_label_sha256": C.sha256_array(y_train),
                 "valid_label_sha256": C.sha256_array(y_valid)},
        "parameter_count": n_params,
        "parameter_audit": {"parameters": n_params,
                            "optimizer_parameters": C.optimizer_parameter_count(opt)},
        "model_info": {"implementation": declaration["implementation"],
                       "initial_state_sha256": init_hash,
                       "native_recipe_source": "upstream configs/zinc-GRIT-RRWP.yaml",
                       "deviations": ["padded dense execution",
                                      "per-step cosine (upstream graphgym is per-epoch)",
                                      "batch_size / epochs as declared in search.json"]},
        "recipe": {**cfg, "epochs": ep, "batch_size": bs,
                   "steps_per_epoch": steps_per_epoch, "updates": total_steps,
                   "warmup_steps": warmup_steps, "loss": "raw-y L1",
                   "valid_every_epoch": True},
        "best_valid": best,
        "fixed_objects": {"best": {"valid_mae": float(best_valid),
                                   "train_mae_eval_mode": float(best_train),
                                   "epoch": best["epoch"]},
                          "final": {"valid_mae": float(final_valid),
                                    "train_mae_eval_mode": float(final_train),
                                    "epoch": ep}},
        "curve": curve, "wall_seconds": wall,
        "gpu_hours": float(wall) / 3600.0, "smoke_override": smoke_epochs is not None,
        "device": str(device), "search_declaration": declaration,
    }
    C.write_json(out / "train.json", info)
    keys = ["epoch", "online_train_mae", "valid_mae", "lr", "seconds", "steps"]
    with open(out / "curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow({k: row[k] for k in keys})
    print(f"[gritd {variant} s{seed}] done best={best_valid:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s", flush=True)
    return info


# ---------------------------------------------------------------------------
# parity against the sparse reference
# ---------------------------------------------------------------------------
def parity(device="cuda:0", batch_rows=None, seed=0) -> dict:
    """Same weights, dropout off: dense forward/gradient vs upstream sparse."""
    from zpp_common import ensure_grit_runtime, REF_BUNDLE
    ensure_grit_runtime()
    import grit  # noqa: F401
    from torch_geometric.graphgym.config import cfg as gcfg, set_cfg
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.data import Batch

    set_cfg(gcfg)
    gcfg.set_new_allowed(True)
    gcfg.merge_from_file(str(REF_BUNDLE / "configs" / "zinc-GRIT-RRWP.yaml"))
    gcfg.accelerator = "cpu"
    ref = create_model().model
    dense = DenseGrit(NATIVE)
    missing = dense.load_state_dict(ref.state_dict(), strict=True)
    del missing

    ref = ref.to(device).float()
    dense = dense.to(device).float()
    ref.train(False)
    dense.train(False)

    from zpp_common import CACHE as ZPP_CACHE
    base = torch.load(ZPP_CACHE / "zinc_subset_valid_norrwp.pt", map_location="cpu",
                      weights_only=False)
    rows = batch_rows or list(range(8))
    from grit.transform.rrwp import add_full_rrwp
    data = [add_full_rrwp(base[i].clone(), walk_length=21, add_identity=True) for i in rows]
    def make_batch():
        b = Batch.from_data_list([d.clone() for d in data]).to(device)
        b.split = "valid"
        assert b.x.dtype == torch.long
        return b

    pyg_batch = make_batch()
    with torch.no_grad():
        p_ref, _ = ref(pyg_batch)
    graphs = prepare("valid")
    xb = collate(graphs, rows, device)
    with torch.no_grad():
        p_dense = batch_pred(dense, xb)
    diff = float((p_ref.view(-1) - p_dense.view(-1)).abs().max())

    # gradient parity on the same prediction sum
    ref.zero_grad(set_to_none=True)
    dense.zero_grad(set_to_none=True)
    pr = ref(make_batch())[0].sum()
    pr.backward()
    dense.zero_grad(set_to_none=True)
    pd = batch_pred(dense, xb).sum()
    pd.backward()
    grads = {}
    max_grad_diff = 0.0
    for k, p in ref.named_parameters():
        q = dict(dense.named_parameters())[k]
        gd = float((p.grad - q.grad).abs().max()) if p.grad is not None else float("nan")
        gs = float(p.grad.abs().max()) if p.grad is not None else float("nan")
        grads[k] = {"max_abs_diff": gd, "ref_max_abs_grad": gs}
        if gs and math.isfinite(gd):
            max_grad_diff = max(max_grad_diff, gd)
    result = {
        "rows": rows, "n_graphs": len(rows),
        "param_count_dense": param_count(dense),
        "param_count_ref": sum(p.numel() for p in ref.parameters()),
        "forward_max_abs_diff": diff,
        "grad_max_abs_diff": max_grad_diff,
        "per_param_grad": grads,
        "dropout": "off", "bn_mode": "eval",
    }
    # BN running-stat parity in train mode (dropout still respected; both 0 here)
    ref.train(True)
    dense.train(True)
    with torch.no_grad():
        ref(make_batch())
        batch_pred(dense, xb)
    bn = {}
    rstats = dict(ref.named_buffers())
    for k, v in dense.named_buffers():
        if k in rstats:
            bn[k] = float((v.float() - rstats[k].float()).abs().max())
    result["bn_buffer_max_abs_diff"] = max(bn.values()) if bn else None
    result["bn_buffers"] = bn
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "dense_parity.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_param_grad"}, indent=1))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "parity", "prepare"])
    p.add_argument("--variant", default="C0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default=None)
    p.add_argument("--smoke-epochs", type=int, default=None)
    args = p.parse_args(argv)
    if args.command == "prepare":
        prepare("train")
        prepare("valid")
    elif args.command == "parity":
        parity(args.device)
    else:
        train(variant=args.variant, seed=args.seed, device=args.device,
              epochs=args.epochs, batch_size=args.batch_size, tag=args.tag,
              force=args.force, reason=args.reason, smoke_epochs=args.smoke_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
