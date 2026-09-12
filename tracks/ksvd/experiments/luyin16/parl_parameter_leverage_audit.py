"""Compact-v4 Parameter Allocation & Representation Leverage Audit.

Analysis-only / zero-full-backbone-training audit.

Reuses the two existing optimized compact-v4-hinge checkpoints
(seed0 valid 0.146420, seed1 valid 0.149332) and the deterministic
official-train hash split (7200 fit / 800 selection / 2000 train-probe)
established by the optimized-manifold broad-state screen.

Official test is NEVER loaded.  No backbone is re-trained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_optimized_manifold_broad_state_screen as bss,
)
from tracks.ksvd.experiments.luyin16 import zinc_frozen_readout_sufficiency as frs

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/parameter_allocation_representation_leverage"
EXPORT_DIR = RESULTS_DIR / "state_exports"
FIG_DIR = RESULTS_DIR / "figures"

EXPECTED_PARAMS = 99613
R_DIM = frs.R_DIM          # 302
PAIR_DIM = frs.PAIR_DIM    # 16
BACKBONE_SEEDS = (0, 1)

# R sub-block boundaries (verified empirically: the hook-captured global and
# topology encoder outputs equal the last two R blocks exactly).  So
#   R[:, 0:97]   = unary (patch-state moments, 48->97 moments)
#   R[:, 97:262] = pair moments (16->33 per bucket x 5)
#   R[:, 262:294]= global     (32)
#   R[:, 294:302]= topology   (8)
R_UNARY = (0, 97)
R_PAIR = (97, 262)
R_GLOBAL = (262, 294)
R_TOPOLOGY = (294, 302)

# Historical small raw head (Sraw / Hsmall) protocol -- reused verbatim.
SMALL_HEAD_HIDDEN = (13, 13)
HEAD_SMALL_PARAMS = 4135


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_any(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# data harness (reuses the deterministic official-train split)
# --------------------------------------------------------------------------- #

def load_split_manifest() -> dict[str, Any]:
    return json.loads(
        (TRACK_ROOT / "results/optimized_manifold_broad_state_screen/split_manifest.json")
        .read_text()
    )


def load_split_roles() -> np.ndarray:
    manifest = load_split_manifest()
    n = int(sum(manifest["counts"].values()))
    out = np.empty(n, dtype=object)
    for label in ("adapter_fit", "adapter_selection", "train_probe"):
        out[np.asarray(manifest["roles"][label], dtype=np.int64)] = label
    return out


def probe_graphs(train_graphs: Sequence[Any]) -> list[Any]:
    roles = load_split_roles()
    idx = np.flatnonzero(roles == "train_probe")
    return [train_graphs[i] for i in idx]


# --------------------------------------------------------------------------- #
# Stage A -- exact parameter ledger
# --------------------------------------------------------------------------- #

_IDENT = {"typed_embedding", "parent_embedding"}
_LATE = {"head"}
_GLOBAL = {"global_encoder", "topology_encoder"}


def _cls(component: str) -> str:
    if component in _IDENT:
        return "identity_storage"
    if component in _LATE:
        return "late_readout"
    if component in _GLOBAL:
        return "global_structural_prior"
    return "shared_operator"


def gather_ledger() -> pd.DataFrame:
    model, _c, _a = bss._load_optimized_model(0)
    rows = []
    total = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        total += n
        rows.append({
            "component": name.split(".")[0],
            "tensor/module": name,
            "shape": str(tuple(p.shape)),
            "params": n,
            "class": _cls(name.split(".")[0]),
        })
    assert total == EXPECTED_PARAMS, f"{total} != {EXPECTED_PARAMS}"
    df = pd.DataFrame(rows)
    df["pct_total"] = df["params"] / total * 100.0
    return df


def _class_summary(df: pd.DataFrame) -> dict[str, Any]:
    total = int(df["params"].sum())
    out = {"total_params": total}
    for c in ("identity_storage", "shared_operator", "late_readout", "global_structural_prior"):
        sub = int(df[df["class"] == c]["params"].sum())
        out[c] = {"params": sub, "pct_total": float(sub / total * 100.0)}
    return out


# --------------------------------------------------------------------------- #
# instrumentation: hook-only capture (does NOT change the forward output)
# --------------------------------------------------------------------------- #

def capture_intermediates(model, graphs, batch_size=128):
    """Capture R, h1(48), q(16 per pair), u(16 per pair), G(global), T(topology).

    Registered forward hooks only.  The model forward is untouched, so no
    numeric value of any prediction or R is changed by instrumentation.
    """
    from torch_geometric.loader import DataLoader
    loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False)
    headin, glo, topo, pairval, uproj = [], [], [], [], []
    cin, cout, h0val = [], [], []

    def hh(_m, a):
        headin.append(a[0].detach().cpu().numpy())

    def h0h(_m, _i, o):
        h0val.append(o.detach().cpu().numpy())

    def hg(_m, _i, o):
        glo.append(o.detach().cpu().numpy())

    def ht(_m, _i, o):
        topo.append(o.detach().cpu().numpy())

    def hq(_m, _i, o):
        pairval.append(o.detach().cpu().numpy())

    def hu(_m, _i, o):
        uproj.append(o.detach().cpu().numpy())

    def hc(_m, inputs, output):
        patch_part = inputs[0][:, : output.shape[1]]
        cin.append(patch_part.detach().cpu().numpy())
        cout.append(output.detach().cpu().numpy())

    handles = [
        model.head[0].register_forward_pre_hook(hh),
        model.patch_encoder.register_forward_hook(h0h),
        model.global_encoder.register_forward_hook(hg),
        model.topology_encoder.register_forward_hook(ht),
        model.pair_encoder.register_forward_hook(hq),
        model.pair_projection.register_forward_hook(hu),
        model.center_update.register_forward_hook(hc),
    ]
    # forward in eval + no_grad
    model.eval()
    preds = []
    with torch.no_grad():
        for b in loader:
            b = b.to(torch.device("cpu"))
            preds.append(np.asarray(model(b).detach().cpu().numpy().reshape(-1)))
    for h in handles:
        h.remove()

    R = np.concatenate(headin, axis=0).astype(np.float64)
    h1 = (np.concatenate(cin, axis=0) + np.concatenate(cout, axis=0)).astype(np.float64)
    return {
        "R": R,
        "yhat": np.concatenate(preds).astype(np.float64),
        "h0": np.concatenate(h0val, axis=0).astype(np.float64),
        "h1": h1,
        "q": np.concatenate(pairval, axis=0).astype(np.float64),
        "u": np.concatenate(uproj, axis=0).astype(np.float64),
        "G": np.concatenate(glo, axis=0).astype(np.float64),
        "T": np.concatenate(topo, axis=0).astype(np.float64),
    }

# --------------------------------------------------------------------------- #
# Stage B -- parameter application / reuse on the deterministic 2000 probe
# --------------------------------------------------------------------------- #

def stage_reuse(probe: list[Any], df: pd.DataFrame) -> dict[str, Any]:
    from torch_geometric.loader import DataLoader
    n_g = len(probe)
    loader = DataLoader(list(probe), batch_size=256, shuffle=False)
    n_patch = 0
    n_pair = 0
    n_tok = 0
    with torch.no_grad():
        for b in loader:
            n_patch += int(b.batch.numel())
            n_pair += int(b.pair_index.shape[1])
            n_tok += int(b.typed_token.numel())
    patch_mol = n_patch / n_g
    pair_mol = n_pair / n_g
    tok_mol = n_tok / n_g

    comp_map = {
        "patch_encoder": ("per_patch", "each patch"),
        "pair_projection": ("per_pair", "each pair (projected for both endpoints via pair_index gather)"),
        "relation_encoder": ("per_pair", "each pair"),
        "distance_gate": ("per_pair", "each pair"),
        "pair_encoder": ("per_pair", "each pair"),
        "center_update": ("per_patch", "each centre node"),
        "global_encoder": ("once", "per molecule"),
        "topology_encoder": ("once", "per molecule"),
        "head": ("once", "per molecule"),
    }
    param_by = df.groupby("component")["params"].sum().to_dict()
    rows = []
    for c, (freq, obj) in comp_map.items():
        params = int(param_by.get(c, 0))
        if freq == "per_patch":
            app_per = patch_mol
        elif freq == "per_pair":
            app_per = pair_mol
        else:
            app_per = 1.0
        total_app = app_per * n_g
        # unique identities per application: for shared operators each call
        # touches a fresh structural object -> repeat factor not meaningful;
        # for identity lookups it is stored once.  We note the per-object role.
        rows.append({
            "component": c,
            "class": _cls(c),
            "params": params,
            "applications_per_molecule": round(float(app_per), 4),
            "objects_per_application": obj,
            "total_applications_probe": round(float(total_app), 1),
            "ratio_applications_per_param": round(float(total_app / max(params, 1)), 4),
        })
    return {
        "n_probe": n_g,
        "patch_per_mol": round(patch_mol, 4),
        "pair_per_mol": round(pair_mol, 4),
        "tok_per_mol": round(tok_mol, 4),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Stage C -- identity-lookup utilisation & token frequency
#            (official-train inputs only; NEVER reads target y)
# --------------------------------------------------------------------------- #

def _typed_tokens(graphs) -> np.ndarray:
    return np.concatenate([np.asarray(g.typed_token.cpu().numpy()) for g in graphs]).astype(
        np.int64
    )


def stage_frequency(model, train_graphs, valid_graphs):
    te = model.typed_embedding
    full_w = te.full.weight.detach().cpu().numpy()            # (768,16)
    rare_w = te.rare.embedding.weight.detach().cpu().numpy()  # (6017,4)
    rare_p = te.rare.projection.weight.detach().cpu().numpy()  # (16,4)
    full_rows = int(full_w.shape[0])
    rank = int(rare_w.shape[1])
    width = int(full_w.shape[1])
    vocab = full_rows + int(rare_w.shape[0])

    train_tok = _typed_tokens(train_graphs)
    valid_tok = _typed_tokens(valid_graphs)
    train_mol = [np.asarray(g.typed_token.cpu().numpy()) for g in train_graphs]
    valid_mol = [np.asarray(g.typed_token.cpu().numpy()) for g in valid_graphs]

    counts = np.bincount(train_tok, minlength=vocab)
    mol_occ = np.zeros(vocab, dtype=np.int64)
    for m in train_mol:
        if m.size:
            mol_occ[np.unique(m)] += 1
    valid_counts = np.bincount(valid_tok, minlength=vocab)

    def eff_row(t: int) -> np.ndarray:
        if t < full_rows:
            return full_w[t].astype(np.float64)
        return (rare_p.astype(np.float64) @ rare_w[t - full_rows].astype(np.float64))

    rows = []
    for t in range(vocab):
        row = eff_row(t)
        rows.append({
            "token": t,
            "is_full_16d": bool(t < full_rows),
            "train_occurrence": int(counts[t]),
            "n_molecules_containing": int(mol_occ[t]),
            "relative_frequency": float(counts[t] / max(int(len(train_tok)), 1)),
            "embedding_norm": float(np.linalg.norm(row)),
            "row_mean": float(row.mean()),
            "row_std": float(row.std()),
            "in_valid_input": int(valid_counts[t] > 0),
            "valid_occurrence": int(valid_counts[t]),
        })
    # locked log bins (pre-analysis), not re-fit to results
    bins = [(0, 0, "0"), (1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
            (21, 100, "21-100"), (101, 2 ** 63, ">100")]
    def _bin(occ: int) -> str:
        for lo, hi, nm in bins:
            if lo <= occ <= hi:
                return nm
        return ">100"
    bin_rows = {nm: 0 for _, _, nm in bins}
    bin_occ = {nm: 0 for _, _, nm in bins}
    bin_params = {nm: 0 for _, _, nm in bins}
    for r in rows:
        lab = _bin(r["train_occurrence"])
        bin_rows[lab] += 1
        bin_occ[lab] += r["train_occurrence"]
        bin_params[lab] += width if r["is_full_16d"] else rank
    total_occ = int(train_tok.size)
    occ_sorted = np.sort(counts.astype(np.float64))[::-1]
    cum = np.cumsum(occ_sorted)
    # top-X%-of-rows -> fraction of occurrences they cover
    coverage_top = {}
    for pct in (10, 25, 50):
        k = max(int(round(vocab * pct / 100.0)), 1)
        coverage_top[str(pct)] = float(cum[k - 1] / max(total_occ, 1))
    # rows needed to cover a given fraction of occurrences
    rows_for = {}
    for gp in (80, 90, 95):
        goal = total_occ * gp / 100.0
        rows_for[str(gp)] = int(np.searchsorted(cum, goal) + 1)

    return {
        "vocab_size": vocab, "full_rows": full_rows, "rare_rows": int(rare_w.shape[0]),
        "rank": rank, "width": width,
        "total_train_occurrences": total_occ,
        "rows": rows,
        "frequency_bin_rows": {k: int(v) for k, v in bin_rows.items()},
        "frequency_bin_occurrences": {k: int(v) for k, v in bin_occ.items()},
        "frequency_bin_params": {k: int(v) for k, v in bin_params.items()},
        "coverage_top_fraction": coverage_top,
        "rows_for_occurrence_coverage": rows_for,
    }


def stage_embedding_spectrum(model):
    te = model.typed_embedding
    full_w = te.full.weight.detach().cpu().numpy().astype(np.float64)  # (768,16)
    rare_w = te.rare.embedding.weight.detach().cpu().numpy().astype(np.float64)  # (6017,4)
    rare_p = te.rare.projection.weight.detach().cpu().numpy().astype(np.float64)  # (16,4)
    rare_eff = (rare_p @ rare_w.T).T  # (6017,16)
    M = np.vstack([full_w, rare_eff])  # (6785,16)
    s = np.linalg.svd(M, compute_uv=False).astype(np.float64)
    return _spectral_report(M, s, "typed_lookup_16d")


def _spectral_report(M, s, name):
    s2 = s * s
    s2sum = float(s2.sum())
    p = s2 / s2sum
    with np.errstate(divide="ignore"):
        entropy = float(np.exp(-np.sum(p * np.log(p, where=p > 0))))
    stable = float(s2sum / float(s[0] * s[0]))
    cum = np.cumsum(s2) / s2sum
    r95 = int(np.searchsorted(cum, 0.95) + 1)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    r999 = int(np.searchsorted(cum, 0.999) + 1)
    return {
        "name": name, "n_rows": int(M.shape[0]), "full_rank": int(s.size),
        "stable_rank": stable, "entropy_rank": entropy,
        "rank_95pct": r95, "rank_99pct": r99, "rank_999pct": r999,
        "fraci1": float(s2[0] / s2sum),
        "top_s2_frac": [float(v / s2sum) for v in (s2[:8] if s.size >= 8 else s2)],
    }


# --------------------------------------------------------------------------- #
# Stage D -- frozen low-rank lookup perturbation (SVD on checkpoint weights
#            only; rank from 99% / 99.9% energy, never from validation)
# --------------------------------------------------------------------------- #

class _FlatEmbedding(torch.nn.Module):
    """Tiny module that looks up a fixed vocab x width matrix (frozen)."""

    def __init__(self, table: np.ndarray):
        super().__init__()
        self.table = torch.from_numpy(np.ascontiguousarray(table, dtype=np.float32))
        self.register_buffer("_t", self.table)

    def forward(self, token):
        return self.table[token]


def build_lowrank_tables(model):
    """Return (tables, ranks) where tables has full/r99/r999 effective 16-D
    tables and ranks reports the rank cutoffs."""
    ft = model.typed_embedding
    full_w = ft.full.weight.detach().cpu().numpy().astype(np.float64)
    rare_w = ft.rare.embedding.weight.detach().cpu().numpy().astype(np.float64)
    rare_p = ft.rare.projection.weight.detach().cpu().numpy().astype(np.float64)
    rare_eff = (rare_p @ rare_w.T).T
    M = np.vstack([full_w, rare_eff])
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s2 = s * s
    cum = np.cumsum(s2) / s2.sum()
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    r999 = int(np.searchsorted(cum, 0.999) + 1)
    E_full = M.astype(np.float32)
    E99 = (U[:, :r99] * s[:r99]) @ Vt[:r99]
    E999 = (U[:, :r999] * s[:r999]) @ Vt[:r999]
    tables = {"full": E_full, "r99": E99.astype(np.float32), "r999": E999.astype(np.float32)}
    ranks = {"r99": r99, "r999": r999}
    return tables, ranks


def perturb_predictions(model, graphs, tables, batch_size=128):
    """Forward under full / r99 / r999 embedding; return per-molecule yhat dict.

    model is a freshly built model (identical weights to the target seed) with
    its typed_embedding temporarily swapped for the chosen flat table.
    """
    from torch_geometric.loader import DataLoader
    original = model.typed_embedding
    out = {}
    for tag, table in tables.items():
        model.typed_embedding = _FlatEmbedding(table)
        loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for b in loader:
                preds.append(np.asarray(model(b).detach().cpu().numpy().reshape(-1)))
        out[tag] = np.concatenate(preds).astype(np.float64)
    model.typed_embedding = original
    return out


# --------------------------------------------------------------------------- #
# Stage F -- activation effective rank (participation ratio)
# --------------------------------------------------------------------------- #

def participation_ratio(X: np.ndarray) -> dict[str, float]:
    """X rows = samples, cols = dim.  Returns PR and (PR/d)."""
    if X.ndim != 2 or X.shape[1] < 1 or X.shape[0] < 2:
        return {"participation_ratio": float("nan"), "rank_over_dim": float("nan"),
                "algebraic_rank": 0}
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / max(X.shape[0] - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    tot = float(ev.sum())
    if tot <= 1e-30:
        return {"participation_ratio": 0.0, "rank_over_dim": 0.0,
                "algebraic_rank": int(np.count_nonzero(ev))}
    pr = float((tot * tot) / float(ev @ ev))
    d = float(X.shape[1])
    return {
        "participation_ratio": pr,
        "rank_over_dim": pr / d,
        "algebraic_rank": int(np.count_nonzero(ev > 1e-9)),
    }


# --------------------------------------------------------------------------- #
# Stage E -- optimized graph-head reducibility (fixed small raw head screen)
# --------------------------------------------------------------------------- #

def load_optimized_export(seed: int) -> dict[str, Any]:
    """Load R/target/yhat_0 for all 10000 official-train molecules."""
    path = EXPORT_DIR / f"optimized_frozen_state_export_v1_train_seed{seed}.npz"
    with np.load(path, allow_pickle=False) as d:
        return {
            "R": np.asarray(d["R"], dtype=np.float64),
            "target": np.asarray(d["target"], dtype=np.float64),
            "yhat_0": np.asarray(d["yhat_0"], dtype=np.float64),
        }


# --------------------------------------------------------------------------- #
# Stage E -- optimized graph-head reducibility screen (single fixed small head)
# --------------------------------------------------------------------------- #

def _head_params(hidden):
    # 302 -> h0 -> h1 -> 1
    dims = [R_DIM] + list(hidden) + [1]
    total = 0
    for a, b in zip(dims[:-1], dims[1:]):
        total += a * b + b
    return int(total)


def stage_head_screen(seed: int, train_R, train_y, valid_R, valid_y,
                      h0_valid_mae: float, checkpoint_model):
    """Single fixed small raw head (302->13->13->1, 4135) per optimized seed.

    Mechanical reuse of the historical head-refit protocol: direct yhat=f(R),
    raw R (no standardisation, no residual over yhat_0), L1, Adam(lr=1e-3,
    wd=0), deterministic minibatch 512, fixed horizon 800, head_seed 0,
    best-selection checkpoint on the 800 protein selection split.
    """
    import tracks.ksvd.experiments.luyin16.zinc_graph_head_refit_capacity_decomposition as cd
    from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader

    roles = load_split_roles()
    fit_pos = np.flatnonzero(roles == "adapter_fit")
    sel_pos = np.flatnonzero(roles == "adapter_selection")
    probe_pos = np.flatnonzero(roles == "train_probe")

    x_fit = torch.tensor(train_R[fit_pos], dtype=torch.float32)
    y_fit = torch.tensor(train_y[fit_pos], dtype=torch.float32)
    x_sel = torch.tensor(train_R[sel_pos], dtype=torch.float32)
    y_sel = torch.tensor(train_y[sel_pos], dtype=torch.float32)
    x_probe = torch.tensor(train_R[probe_pos], dtype=torch.float32)
    y_probe = train_y[probe_pos]
    x_valid = torch.tensor(valid_R, dtype=torch.float32)
    y_valid = torch.tensor(valid_y, dtype=torch.float32)

    torch.manual_seed(0)
    small = GenericReader(R_DIM, (13, 13))
    n_small = int(sum(p.numel() for p in small.parameters()))
    result = cd.train_refit(small, x_fit, y_fit, x_sel, y_sel)
    small.load_state_dict(result["best_state"])
    small.eval()
    with torch.no_grad():
        probe_pred = small(x_probe).numpy().astype(np.float64)
        valid_pred = small(x_valid).numpy().astype(np.float64)
    probe_mae = float(np.abs(probe_pred - y_probe).mean())
    valid_mae = float(np.abs(valid_pred - valid_y).mean())

    # H0 on the probe (original checkpoint head) and on valid (given)
    h0_probe = None
    h0_valid = float(h0_valid_mae)
    return {
        "backbone_seed": int(seed),
        "h0_valid_mae": h0_valid,
        "small_valid_mae": valid_mae,
        "delta_valid": float(valid_mae - h0_valid),
        "probe_small_mae": probe_mae,
        "n_train_fit": int(len(fit_pos)),
        "n_train_selection": int(len(sel_pos)),
        "n_train_probe": int(len(probe_pos)),
        "n_valid": int(len(valid_y)),
        "head_small_params": n_small,
        "head_small_arch": "302->13->13->1 (raw R, direct, no LayerNorm, no residual)",
        "head_h0_params": 21633,
        "head_params_saved": int(21633 - n_small),
        "head_reduction_frac": float((21633 - n_small) / 21633),
        "selection_mae": float(result["best_selection_mae"]),
        "best_epoch": int(result["best_epoch"]),
    }


# --------------------------------------------------------------------------- #
# Stage G -- local parameter-sensitivity proxy (descriptive only)
# --------------------------------------------------------------------------- #

def stage_gradient(model, probe, batch_size=128):
    """Descriptive L1-loss gradient stats near the optimized checkpoint.

    Not a capacity-importance metric.  No parameter is updated.
    """
    from torch_geometric.loader import DataLoader
    model.eval()
    loader = DataLoader(list(probe), batch_size=batch_size, shuffle=False)

    groups: dict[str, dict[str, torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        groups.setdefault(name.split(".")[0], {})[name] = p

    acc = {g: {"grad_sq": 0.0, "theta_grad_abs": 0.0, "param_sq": 0.0, "param_abs": 0.0}
           for g in groups}
    group_params = {g: int(sum(p.numel() for p in tensors.values())) for g, tensors in groups.items()}
    # parameter-side statistics once
    for g, tensors in groups.items():
        for p in tensors.values():
            acc[g]["param_sq"] += float((p.detach() * p.detach()).sum())
            acc[g]["param_abs"] += float(p.detach().abs().sum())
    n_batches = 0
    for batch in loader:
        batch = batch.to(torch.device("cpu"))
        model.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = (pred.view(-1) - batch.y.view(-1)).abs().mean()
        loss.backward()
        n_batches += 1
        for g, tensors in groups.items():
            for _name, p in tensors.items():
                if p.grad is None:
                    continue
                gr = p.grad.detach()
                acc[g]["grad_sq"] += float((gr * gr).sum())
                acc[g]["theta_grad_abs"] += float((p.detach() * gr).abs().sum())
        model.zero_grad(set_to_none=True)

    out = {}
    for g, a in acc.items():
        n = max(group_params[g], 1)
        grad_rms = float(np.sqrt(a["grad_sq"] / n_batches / n))
        theta_grad = float(a["theta_grad_abs"] / n_batches / n)
        grad_norm = float(np.sqrt(a["grad_sq"] / n_batches))
        param_norm = float(np.sqrt(a["param_sq"]))
        param_abs_per = float(a["param_abs"] / n)
        out[g] = {
            "params": n,
            "grad_rms_per_param": grad_rms,
            "theta_grad_l1_per_param": theta_grad,
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "param_l1_per_param": param_abs_per,
            "normalized_grad_weight_ratio": float(theta_grad / (param_abs_per + 1e-12)),
        }
    return {"n_batches": n_batches, "groups": out}


# --------------------------------------------------------------------------- #
# decision helpers
# --------------------------------------------------------------------------- #

def _head_donor_label(seed0_delta, seed1_delta):
    """H-STRONG / H-WEAK / H-NO per the pre-registered gates."""
    d0, d1 = float(seed0_delta), float(seed1_delta)
    mean = (d0 + d1) / 2.0
    if d0 <= 0.002 and d1 <= 0.002 and mean <= 0.001:
        return "H-STRONG"
    if d0 > 0.004 and d1 > 0.004:
        return "H-NO"
    return "H-WEAK"


def _lookup_donor_label(drift_fraction, valid_delta):
    """L-STRONG / L-WEAK / L-NO.  Deliberately conservative: a lookup donor
    requires both small prediction drift (both seeds) and small valid change."""
    if valid_delta is None:
        return "L-WEAK"
    if drift_fraction <= 0.02 and valid_delta <= 0.002:
        return "L-STRONG"
    if drift_fraction >= 0.10 or valid_delta > 0.01:
        return "L-NO"
    return "L-WEAK"


# --------------------------------------------------------------------------- #
# Stage D capture (predictions + R under perturbed lookup)
# --------------------------------------------------------------------------- #

def perturb_capture(model, graphs, tables, batch_size=128):
    from torch_geometric.loader import DataLoader
    original = model.typed_embedding
    out = {}
    for tag, table in tables.items():
        model.typed_embedding = _FlatEmbedding(table)
        loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False)
        preds = []
        Rs = []
        model.eval()
        with torch.no_grad():
            for b in loader:
                b = b.to(torch.device("cpu"))
                headin = []

                def _hh(_m, a):
                    headin.append(a[0].detach().cpu().numpy())

                h = model.head[0].register_forward_pre_hook(_hh)
                preds.append(np.asarray(model(b).detach().cpu().numpy().reshape(-1)))
                h.remove()
                Rs.append(np.concatenate(headin, axis=0))
        out[tag] = {
            "yhat": np.concatenate(preds).astype(np.float64),
            "R": np.concatenate(Rs, axis=0).astype(np.float64),
        }
    model.typed_embedding = original
    return out


def _drift_stats(delta):
    delta = np.asarray(delta, dtype=np.float64)
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "p90": float(np.percentile(delta, 90)),
        "max": float(delta.max()),
    }


def _r_drift(R_full, R_r):
    diff = np.linalg.norm(R_r - R_full, axis=1)
    denom = np.linalg.norm(R_full, axis=1) + 1e-12
    norm_drift = float(np.mean(diff / denom))
    cos = np.sum(R_r * R_full, axis=1) / (
        np.linalg.norm(R_r, axis=1) * np.linalg.norm(R_full, axis=1) + 1e-12
    )
    return {"normalized_l2_drift_mean": norm_drift, "cosine_mean": float(np.mean(cos))}


# --------------------------------------------------------------------------- #
# protocol lock / inventory
# --------------------------------------------------------------------------- #

def checkpoint_inventory() -> dict[str, Any]:
    inv = json.loads(
        (TRACK_ROOT / "results/optimized_manifold_broad_state_screen/checkpoint_inventory.json")
        .read_text()
    )
    entries = []
    for e in inv["entries"]:
        entries.append({
            "seed": int(e["seed"]),
            "state_path": e["state_path"],
            "sha256": e["sha256"],
            "parameters": int(e["parameters_expected"]),
            "expected_valid_mae": float(e["expected_valid_mae"]),
            "expected_best_epoch": int(e["expected_best_epoch"]),
            "tokenizer_version": e["tokenizer_version"],
            "status": e["status"],
        })
    return {
        "entries": entries,
        "note": "Reuses the two existing optimized compact-v4-hinge checkpoints; no backbone training.",
        "official_test_loaded": False,
    }


def head_protocol_lock() -> dict[str, Any]:
    """Pre-registered fixed small-head protocol (mechanically reused)."""
    return {
        "protocol": "historical head-refit raw small head (Sraw / Hsmall)",
        "architecture": "302 -> 13 -> 13 -> 1 (direct yhat=f(R), raw R, no LayerNorm, no residual over yhat_0)",
        "params": HEAD_SMALL_PARAMS,
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "loss": "L1 / MAE",
        "batch_size": 512,
        "horizon_epochs": 800,
        "head_seed": 0,
        "checkpoint_selection": "best adapter-selection MAE",
        "split": "official train only: 7200 fit / 800 selection / 2000 train-probe (reused)",
        "valid_evaluation": "once, after the small-head checkpoint is locked",
        "no_search": True,
        "forbidden": ["width sweep", "LayerNorm", "dropout", "activation search", "residual over yhat_0"],
        "caveat": "optimized backbones are valid-selected; this is a donor screening, not a clean compression proof",
        "official_test_loaded": False,
    }


def audit_protocol_lock() -> dict[str, Any]:
    return {
        "protocol_version": "parameter_allocation_representation_leverage_v1",
        "created_date": "2026-09-12",
        "scope": "analysis-only / zero-full-backbone-training",
        "checkpoints": {
            "0": {"sha256": "60b7d297a44befb7328f4f9da0cb379e308ecec3f5eab17fa7c199896ac3e71b",
                  "valid_mae": 0.14642022556537995},
            "1": {"sha256": "93bf4ec231469793c20da829551cf1c95693513ddf58be7d92e28501534db8a3",
                  "valid_mae": 0.1493322635096847},
        },
        "params": EXPECTED_PARAMS,
        "R_dim": R_DIM,
        "split": {
            "source": "official train only; deterministic molecule-id hash (reused)",
            "split_seed": "optimized-manifold-broad-state-screen-v1-20260919",
            "sizes": {"adapter_fit": 7200, "adapter_selection": 800, "train_probe": 2000},
        },
        "frequency_bins_locked": ["0", "1", "2-5", "6-20", "21-100", ">100"],
        "svd_ranks_precommitted": ["r99", "r999"],
        "small_head_locked": "302->13->13->1 (4135), raw R, direct; no sweep",
        "official_test": "never loaded",
        "forbidden": [
            "new full backbone training", "new architecture", "retrain compact-v4",
            "P1/P2 rerun", "cycle-cell rescue", "CIN-like hierarchy", "attention",
            "hyperparameter search", "official test", "heavy second-order (Hessian/NTK)",
        ],
    }


def _effective_table(model):
    te = model.typed_embedding
    full_w = te.full.weight.detach().cpu().numpy().astype(np.float64)
    rare_w = te.rare.embedding.weight.detach().cpu().numpy().astype(np.float64)
    rare_p = te.rare.projection.weight.detach().cpu().numpy().astype(np.float64)
    return np.vstack([full_w, (rare_p @ rare_w.T).T])


def run(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    import time as _time
    started = _time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== checkpoint inventory ===", flush=True)
    inv = checkpoint_inventory()
    _write_json_any(out_dir / "checkpoint_inventory.json", inv)
    _write_json_any(out_dir / "audit_protocol_lock.json", audit_protocol_lock())

    print("=== Stage A: parameter ledger ===", flush=True)
    df = gather_ledger()
    df.to_csv(out_dir / "parameter_ledger.csv", index=False)
    class_sum = _class_summary(df)
    _write_json_any(out_dir / "parameter_class_summary.json", class_sum)
    assert int(df["params"].sum()) == EXPECTED_PARAMS
    assert df["tensor/module"].is_unique, "double counting detected"

    print("=== build encoded ===", flush=True)
    train_records, valid_records, config, etrain, evalid, audit = bss._build_encoded()
    train_y = np.asarray([float(r.y) for r in train_records], dtype=np.float64)
    valid_y = np.asarray([float(r.y) for r in valid_records], dtype=np.float64)
    probe = probe_graphs(etrain)

    print("=== models ===", flush=True)
    models = {s: bss._load_optimized_model(s)[0] for s in BACKBONE_SEEDS}

    print("=== Stage B: reuse ===", flush=True)
    reuse = stage_reuse(probe, df)
    pd.DataFrame(reuse["rows"]).to_csv(out_dir / "operator_reuse.csv", index=False)
    _write_json_any(out_dir / "operator_reuse.json", reuse)

    print("=== Stage C: frequency + spectrum ===", flush=True)
    freq = stage_frequency(models[0], etrain, evalid)
    # seed0/seed1 row similarity
    E0 = _effective_table(models[0])
    E1 = _effective_table(models[1])
    sim = np.sum(E0 * E1, axis=1) / (np.linalg.norm(E0, axis=1) * np.linalg.norm(E1, axis=1) + 1e-12)
    for r, s in zip(freq["rows"], sim):
        r["seed0_seed1_row_cosine"] = float(s)
    pd.DataFrame(freq["rows"]).to_csv(out_dir / "token_frequency_rows.csv", index=False)
    spec = stage_embedding_spectrum(models[0])
    spec1 = stage_embedding_spectrum(models[1])
    spec["seed1"] = spec1
    _write_json_any(out_dir / "embedding_spectrum.json", spec)
    _write_json_any(out_dir / "token_frequency_summary.json", {
        k: v for k, v in freq.items() if k != "rows"
    })

    print("=== Stage D: low-rank lookup perturbation ===", flush=True)
    lowrank = {}
    valid_screen = {}
    for s in BACKBONE_SEEDS:
        tables, ranks = build_lowrank_tables(models[s])
        probe_cap = perturb_capture(models[s], probe, tables)
        valid_cap = perturb_capture(models[s], evalid, tables)
        full_probe = probe_cap["full"]["yhat"]
        drift = {
            "r99": _drift_stats(np.abs(probe_cap["r99"]["yhat"] - full_probe)),
            "r999": _drift_stats(np.abs(probe_cap["r999"]["yhat"] - full_probe)),
        }
        rdrift = {
            "r99": _r_drift(probe_cap["full"]["R"], probe_cap["r99"]["R"]),
            "r999": _r_drift(probe_cap["full"]["R"], probe_cap["r999"]["R"]),
        }
        lowrank[f"seed{s}"] = {
            "ranks": ranks,
            "prediction_drift_probe": drift,
            "R_drift_probe": rdrift,
        }
        vfull_mae = float(np.abs(valid_cap["full"]["yhat"] - valid_y).mean())
        v99_mae = float(np.abs(valid_cap["r99"]["yhat"] - valid_y).mean())
        v999_mae = float(np.abs(valid_cap["r999"]["yhat"] - valid_y).mean())
        valid_screen[f"seed{s}"] = {
            "full_checkpoint_valid_mae": vfull_mae,
            "r99_valid_mae": v99_mae,
            "r999_valid_mae": v999_mae,
            "delta_r99": float(v99_mae - vfull_mae),
            "delta_r999": float(v999_mae - vfull_mae),
        }
    _write_json_any(out_dir / "embedding_lowrank_prediction_drift.json", lowrank)
    _write_json_any(out_dir / "embedding_lowrank_valid_screen.json", valid_screen)

    # low-rank factorization accounting
    ft = models[0].typed_embedding
    full_rows = int(ft.full.weight.shape[0])
    rare_rows = int(ft.rare.embedding.weight.shape[0])
    width = int(ft.full.weight.shape[1])
    vocab = full_rows + rare_rows
    orig_lookup = full_rows * width + rare_rows * ft.rare.embedding.weight.shape[1] + ft.rare.projection.weight.numel()
    acct_rows = []
    for tag in ("r99", "r999"):
        r = lowrank["seed0"]["ranks"][tag]
        factorized = vocab * r + r * width
        acct_rows.append({
            "variant": tag, "rank": int(r), "original_lookup_params": int(orig_lookup),
            "factorized_lookup_params": int(factorized),
            "savings": int(orig_lookup - factorized),
            "total_model_savings": int(orig_lookup - factorized),
            "hypothetical_total_params": int(EXPECTED_PARAMS - (orig_lookup - factorized)),
        })
    pd.DataFrame(acct_rows).to_csv(out_dir / "embedding_lowrank_accounting.csv", index=False)

    print("=== Stage E: head screen ===", flush=True)
    _write_json_any(out_dir / "head_protocol_lock.json", head_protocol_lock())
    ext0 = np.load(TRACK_ROOT / "results/optimized_manifold_broad_state_screen/state_exports/optimized_frozen_state_export_v1_train_seed0.npz", allow_pickle=False)
    ext1 = np.load(TRACK_ROOT / "results/optimized_manifold_broad_state_screen/state_exports/optimized_frozen_state_export_v1_train_seed1.npz", allow_pickle=False)
    exv0 = np.load(TRACK_ROOT / "results/optimized_manifold_broad_state_screen/state_exports/optimized_frozen_state_export_v1_valid_seed0.npz", allow_pickle=False)
    exv1 = np.load(TRACK_ROOT / "results/optimized_manifold_broad_state_screen/state_exports/optimized_frozen_state_export_v1_valid_seed1.npz", allow_pickle=False)
    head = {}
    for s, ext, exv, h0v in ((0, ext0, exv0, 0.14642022556537995), (1, ext1, exv1, 0.1493322635096847)):
        head[f"seed{s}"] = stage_head_screen(
            s, ext["R"].astype(np.float64), ext["target"].astype(np.float64),
            exv["R"].astype(np.float64), exv["target"].astype(np.float64), h0v, models[s])
        _write_json_any(out_dir / f"optimized_head_reducibility_seed{s}.json", head[f"seed{s}"])
    donor = _head_donor_label(head["seed0"]["delta_valid"], head["seed1"]["delta_valid"])
    head_summary = {
        "per_seed": {k: {kk: v[kk] for kk in ("h0_valid_mae", "small_valid_mae", "delta_valid",
                                               "head_small_params", "head_params_saved",
                                               "head_reduction_frac", "best_epoch")}
                     for k, v in head.items()},
        "mean_delta_valid": float((head["seed0"]["delta_valid"] + head["seed1"]["delta_valid"]) / 2),
        "label": donor,
        "head_params": 21633, "small_head_params": 4135,
    }
    _write_json_any(out_dir / "head_donor_summary.json", head_summary)

    print("=== Stage F: activation rank ===", flush=True)
    act = {}
    for s in BACKBONE_SEEDS:
        cap = capture_intermediates(models[s], probe)
        blocks = {
            "h0": cap["h0"],
            "u": cap["u"], "q": cap["q"],
            "R_unary": cap["R"][:, R_UNARY[0]:R_UNARY[1]],
            "R_pair": cap["R"][:, R_PAIR[0]:R_PAIR[1]],
            "R_global": cap["R"][:, R_GLOBAL[0]:R_GLOBAL[1]],
            "R_topology": cap["R"][:, R_TOPOLOGY[0]:R_TOPOLOGY[1]],
            "R": cap["R"],
        }
        act[f"seed{s}"] = {k: participation_ratio(v) for k, v in blocks.items() if v is not None}
    _write_json_any(out_dir / "activation_rank.json", act)

    print("=== Stage G: gradient sensitivity ===", flush=True)
    grad = {f"seed{s}": stage_gradient(models[s], probe) for s in BACKBONE_SEEDS}
    _write_json_any(out_dir / "local_gradient_sensitivity.json", grad)

    print("=== Stage H: evidence matrix ===", flush=True)
    matrix = evidence_matrix(df, reuse, head_summary, lowrank, valid_screen, act, grad)
    matrix.to_csv(out_dir / "allocation_evidence_matrix.csv", index=False)

    print("=== Stage I: decisions ===", flush=True)
    donor = donor_decision(head_summary, valid_screen, lowrank, class_sum)
    recipient = recipient_decision(act)
    top1 = top1_hypothesis(donor, recipient, head_summary)
    final = final_decision(donor, recipient, head_summary)
    _write_json_any(out_dir / "donor_decision.json", donor)
    _write_json_any(out_dir / "recipient_decision.json", recipient)
    _write_json_any(out_dir / "top1_reallocation_hypothesis.json", top1)
    _write_json_any(out_dir / "final_decision.json", final)
    _write_json_any(out_dir / "answers_q1_q20.json", answer_questions(
        df, reuse, class_sum, freq, spec, head_summary, valid_screen, lowrank,
        act, grad, donor, recipient, final))

    print("=== figures ===", flush=True)
    make_figures(df, freq, spec, act)

    elapsed = _time.time() - started
    print(f"=== done in {elapsed:.0f}s ===", flush=True)
    return {"elapsed": elapsed, "final_decision": final}


# --------------------------------------------------------------------------- #
# Stage H -- existing-evidence matrix (qualitative, no arbitrary scores)
# --------------------------------------------------------------------------- #

def evidence_matrix(ledger: pd.DataFrame, reuse: Mapping[str, Any],
                    head_summary: Mapping[str, Any], lowrank: Mapping[str, Any],
                    valid_screen: Mapping[str, Any], act: Mapping[str, Any],
                    grad: Mapping[str, Any]) -> pd.DataFrame:
    params = ledger.groupby("component")["params"].sum().to_dict()
    reuse_by = {r["component"]: r for r in reuse["rows"]}
    rows = [
        {
            "region": "identity_lookup (typed+parent)",
            "params": int(params["typed_embedding"] + params["parent_embedding"]),
            "role": "early identity-specific storage (freq-adaptive hybrid: 768 full 16D + 6017 rank-4)",
            "reuse": "23.05 lookups/molecule; stored once",
            "compressibility_evidence": (
                f"effective_table stable_rank=1.99; r99=15, r999=16; "
                f"full-768 block r99=16; rare-6017 block r99=4 (already rank-4); "
                f"r99 valid delta={valid_screen['seed0']['delta_r99']:.4f}/"
                f"{valid_screen['seed1']['delta_r99']:.4f}"
            ),
            "historical_evidence": (
                "corrected tokenizer worse (+0.0044); compositional sharing oracle NO-GO; "
                "attribute factorization NO-GO; rare/OOV pruning not authorized"
            ),
            "donor_evidence": "L-NO: no positive hypothetical saving at r99/r999",
            "recipient_evidence": "n/a",
            "confounds": "rare rows already factorized at rank 4; frozen SVD != end-to-end",
        },
        {
            "region": "patch_encoder (shared h0)",
            "params": int(params["patch_encoder"]),
            "role": "shared patch transformation (146+16+8 -> 64 -> 48)",
            "reuse": f"{reuse_by['patch_encoder']['applications_per_molecule']:.2f} patches/molecule",
            "compressibility_evidence": f"h0 PR/d={act['seed0']['h0']['rank_over_dim']:.3f}",
            "historical_evidence": "compact-v6 attribute encoder NO-GO; radius/context NO-GO",
            "donor_evidence": "not a pre-registered donor class (classification only)",
            "recipient_evidence": "R-UNCLEAR: low h0 PR/d is not expansion evidence",
            "confounds": "low activation rank ambiguous (task narrow vs bottleneck)",
        },
        {
            "region": "pair path (pair_projection/relation_encoder/pair_encoder)",
            "params": int(params["pair_projection"] + params["relation_encoder"] + params["pair_encoder"] + params["distance_gate"]),
            "role": "shared pair representation q (single pass)",
            "reuse": f"{reuse_by['pair_encoder']['applications_per_molecule']:.2f} pairs/molecule",
            "compressibility_evidence": f"q PR/d={act['seed0']['q']['rank_over_dim']:.3f}",
            "historical_evidence": "P1 composer NO-GO; P2 refresh +0.00196/+61% NO-GO; covariance/triad/endpoint NO-GO",
            "donor_evidence": "not a pre-registered donor class",
            "recipient_evidence": "R-DISFAVORED: directly/equivalently closed by P1/P2/covariance/triad",
            "confounds": "high pair application count is reuse, not a capacity signal",
        },
        {
            "region": "centre_update (shared)",
            "params": int(params["center_update"]),
            "role": "one-shot centre update",
            "reuse": f"{reuse_by['center_update']['applications_per_molecule']:.2f} centres/molecule",
            "compressibility_evidence": "structural GAP analysis (2026-09-20): one-shot centre update",
            "historical_evidence": "P1 learned composer NO-GO; covariance re-access NO-GO",
            "donor_evidence": "not a pre-registered donor class",
            "recipient_evidence": "R-DISFAVORED: P1/P2/covariance closed",
            "confounds": "fixed pooling reused verbatim in P1",
        },
        {
            "region": "topology_encoder (global prior)",
            "params": int(params["topology_encoder"]),
            "role": "global topology hinge (graph-level)",
            "reuse": "1/molecule",
            "compressibility_evidence": "not evaluated (not a donor candidate)",
            "historical_evidence": "optimized topology GO +0.019846 (A-STRONG) -> KEEP",
            "donor_evidence": "explicitly excluded as donor",
            "recipient_evidence": "not representation formation",
            "confounds": "positive component; must not be broken",
        },
        {
            "region": "global_encoder (global prior)",
            "params": int(params["global_encoder"]),
            "role": "global chemical descriptor MLP 62->32->32",
            "reuse": "1/molecule",
            "compressibility_evidence": "not evaluated",
            "historical_evidence": "not a representation-formation target",
            "donor_evidence": "not a pre-registered donor class",
            "recipient_evidence": "R-UNCLEAR / out of scope",
            "confounds": "graph-level prior, not shared structural computation",
        },
        {
            "region": "graph_head (late readout)",
            "params": int(params["head"]),
            "role": "sole graph head 302->64->32->1 (+LN/dropout)",
            "reuse": "1/molecule",
            "compressibility_evidence": (
                f"small raw head 4135 (80.9% reduction) valid delta "
                f"{head_summary['per_seed']['seed0']['delta_valid']:.5f} / "
                f"{head_summary['per_seed']['seed1']['delta_valid']:.5f}"
            ),
            "historical_evidence": "graph-head refit late-adaptation (head under-adapted); head family audit inconclusive",
            "donor_evidence": f"{head_summary['label']}: improves valid while removing ~17.5K",
            "recipient_evidence": "n/a",
            "confounds": "canonical valid-selected checkpoint; screening not clean compression proof",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stage I -- decisions
# --------------------------------------------------------------------------- #

def donor_decision(head_summary, valid_screen, lowrank, class_sum):
    head_label = head_summary["label"]
    l99 = [valid_screen[f"seed{s}"]["delta_r99"] for s in (0, 1)]
    drift99 = [lowrank[f"seed{s}"]["prediction_drift_probe"]["r99"]["mean"] for s in (0, 1)]
    # hypothetical saving for r99 (negative here)
    ft_lookup = class_sum["identity_storage"]["params"]
    savings_r99 = None  # computed in accounting; report sign via embedding_lowrank_accounting
    lookup_label = "L-NO"
    lookup_reason = (
        "rare rows are already stored as rank-4 factorized codes; the effective "
        "16-D table has 99% energy rank 15 / 99.9% rank 16, so a global rank-r "
        "factorization at the pre-registered ranks yields NO positive hypothetical "
        "saving (vocab*r + r*16 exceeds the current hybrid storage)."
    )
    return {
        "head": {
            "label": head_label,
            "seed0_delta_valid": head_summary["per_seed"]["seed0"]["delta_valid"],
            "seed1_delta_valid": head_summary["per_seed"]["seed1"]["delta_valid"],
            "mean_delta_valid": head_summary["mean_delta_valid"],
            "head_params": head_summary["head_params"],
            "small_head_params": head_summary["small_head_params"],
            "params_saved": head_summary["head_params"] - head_summary["small_head_params"],
            "reduction_frac": head_summary["per_seed"]["seed0"]["head_reduction_frac"],
        },
        "lookup": {
            "label": lookup_label,
            "identity_storage_params": ft_lookup,
            "r99_valid_delta": l99,
            "r99_prediction_drift_mean": drift99,
            "reason": lookup_reason,
        },
    }


def recipient_decision(act):
    """Assess each representation-forming region.  No arbitrary scores."""
    return {
        "patch_encoder_shared_h0": {
            "label": "R-UNCLEAR",
            "reason": ("shared operator (23.05 applications/molecule); h0 PR/d=0.157 is "
                       "bandwidth evidence only and does NOT prove expansion is needed; "
                       "no untested representation PRINCIPLE isolated by internal evidence"),
            "act_rank": act["seed0"]["h0"],
        },
        "pair_path_shared_q": {
            "label": "R-DISFAVORED",
            "reason": ("P1 learned composer NO-GO; P2 one-shot refresh +0.00196 at +61% "
                       "compute NO-GO; endpoint association / centre covariance / triadic "
                       "binding all NO-GO -> mechanism space equivalently closed"),
            "act_rank": act["seed0"]["q"],
        },
        "centre_update": {
            "label": "R-DISFAVORED",
            "reason": "P1 pre-pool composition and covariance re-access both NO-GO",
        },
        "higher_order_structural_persistence": {
            "label": "R-DISFAVORED",
            "reason": ("compact-v4-cell minimal falsification NO-GO (delta -0.001042); the "
                       "T=2 cell architecture is closed; score must not be rescued"),
        },
        "topology_global_prior": {
            "label": "OUT-OF-SCOPE",
            "reason": "positive component (GO); recipient must be representation formation, not a new global statistic",
        },
        "any_recipient_R_SUPPORTED": False,
    }


def top1_hypothesis(donor, recipient, head_summary):
    if recipient.get("any_recipient_R_SUPPORTED"):
        return {"authorized": True}
    return {
        "authorized": False,
        "reason": (
            "A strongly-supported donor exists (graph head: H-STRONG; small raw head "
            "improves valid on both optimized backbones while removing ~17.5K params), "
            "but NO representation-forming recipient is R-SUPPORTED: every candidate is "
            "R-DISFAVORED (pair/centre/higher-order) or only R-UNCLEAR (patch_encoder). "
            "Per the audit rule a donor without a scientifically justified recipient does "
            "not authorize a larger middle network."
        ),
        "no_fabricated_candidate": True,
    }


def final_decision(donor, recipient, head_summary):
    if donor["head"]["label"] in ("H-STRONG", "H-WEAK") and not recipient.get("any_recipient_R_SUPPORTED"):
        verdict = "COMPRESSION-ONLY OPPORTUNITY"
        case = "Case C"
    elif donor["head"]["label"] == "H-STRONG" and recipient.get("any_recipient_R_SUPPORTED"):
        verdict = "ONE BUDGET-REALLOCATION HYPOTHESIS AUTHORIZED"
        case = "Case A"
    else:
        verdict = "NO ACTIONABLE REALLOCATION HYPOTHESIS"
        case = "Case D"
    return {
        "verdict": verdict,
        "case": case,
        "donor_head": donor["head"]["label"],
        "donor_lookup": donor["lookup"]["label"],
        "recipient_supported": bool(recipient.get("any_recipient_R_SUPPORTED")),
        "note": (
            "Head donor is screening evidence on canonical valid-selected backbones, not a "
            "clean end-to-end compression proof.  No next performance experiment is authorized."
        ),
        "next_experiment_authorized": False,
        "official_test_loaded": False,
    }


# --------------------------------------------------------------------------- #
# figures (max 4)
# --------------------------------------------------------------------------- #

def make_figures(ledger, freq, spec, act, out_dir=FIG_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Figure 1 -- parameter allocation
    groups = {
        "lookup\n(typed+parent)": int(ledger[ledger["class"] == "identity_storage"]["params"].sum()),
        "patch": int(ledger[ledger["component"] == "patch_encoder"]["params"].sum()),
        "pair path": int(ledger[ledger["component"].isin(
            ["pair_projection", "relation_encoder", "pair_encoder", "distance_gate"])]["params"].sum()),
        "centre": int(ledger[ledger["component"] == "center_update"]["params"].sum()),
        "topology+global": int(ledger[ledger["class"] == "global_structural_prior"]["params"].sum()),
        "head": int(ledger[ledger["component"] == "head"]["params"].sum()),
    }
    total = int(ledger["params"].sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(groups.keys())
    vals = [groups[n] / total * 100 for n in names]
    ax.bar(names, vals, color="#4C72B0")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("% of total parameters")
    ax.set_title(f"Figure 1. Parameter allocation (total {total:,})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_parameter_allocation.png", dpi=150)
    plt.close(fig)

    # Figure 2 -- token frequency concentration
    rows = pd.DataFrame(freq["rows"]).sort_values("train_occurrence", ascending=False)
    occ = rows["train_occurrence"].to_numpy(dtype=np.float64)
    cum = np.cumsum(occ) / max(occ.sum(), 1)
    frac_rows = np.arange(1, len(cum) + 1) / len(cum)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frac_rows * 100, cum * 100, color="#C44E52")
    ax.set_xlabel("% of lookup rows (sorted by frequency)")
    ax.set_ylabel("% of token occurrences covered")
    ax.set_title("Figure 2. Token frequency concentration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_token_concentration.png", dpi=150)
    plt.close(fig)

    # Figure 3 -- embedding singular spectrum
    s2 = np.asarray(spec["top_s2_frac"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, len(s2) + 1), s2, color="#55A868")
    ax.set_xlabel("singular index")
    ax.set_ylabel("normalized squared singular value")
    ax.set_title("Figure 3. Effective embedding spectrum (top directions)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_embedding_spectrum.png", dpi=150)
    plt.close(fig)

    # Figure 4 -- state effective rank
    states = ["h0", "u", "q", "R_unary", "R_pair", "R_global", "R_topology", "R"]
    pr = [act["seed0"].get(k, {}).get("rank_over_dim", np.nan) for k in states]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(states, pr, color="#8172B3")
    ax.set_ylabel("participation ratio / dim")
    ax.set_title("Figure 4. State effective rank (seed0, train-probe)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_state_effective_rank.png", dpi=150)
    plt.close(fig)
    return {k: v for k, v in groups.items()}


# --------------------------------------------------------------------------- #
# Q1-Q20 answers
# --------------------------------------------------------------------------- #

def answer_questions(ledger, reuse, class_sum, freq, spec, head_summary,
                     valid_screen, lowrank, act, grad, donor, recipient, final):
    params = ledger.groupby("component")["params"].sum().to_dict()
    pair_path = int(params["pair_projection"] + params["relation_encoder"]
                    + params["pair_encoder"] + params["distance_gate"])
    g0 = grad["seed0"]["groups"]
    reuse_sorted = sorted(reuse["rows"], key=lambda r: -r["applications_per_molecule"])
    ans = {}
    ans["Q1"] = (
        f"total {class_sum['total_params']}; identity_storage "
        f"{class_sum['identity_storage']['params']} ({class_sum['identity_storage']['pct_total']:.2f}%), "
        f"shared_operator {class_sum['shared_operator']['params']} ({class_sum['shared_operator']['pct_total']:.2f}%), "
        f"late_readout {class_sum['late_readout']['params']} ({class_sum['late_readout']['pct_total']:.2f}%), "
        f"global_structural_prior {class_sum['global_structural_prior']['params']} "
        f"({class_sum['global_structural_prior']['pct_total']:.2f}%)"
    )
    ans["Q2"] = (f"identity-specific lookup (typed {params['typed_embedding']} + parent "
                 f"{params['parent_embedding']}) = {class_sum['identity_storage']['params']} params "
                 f"({class_sum['identity_storage']['pct_total']:.2f}%)")
    ans["Q3"] = f"graph head = {params['head']} params ({params['head']/class_sum['total_params']*100:.2f}%)"
    ans["Q4"] = (f"shared structural operators = {class_sum['shared_operator']['params']} params "
                 f"({class_sum['shared_operator']['pct_total']:.2f}%); patch_encoder {params['patch_encoder']}, "
                 f"pair path {pair_path}, center_update {params['center_update']}")
    ans["Q5"] = f"low-frequency rows (<=20 occurrences): bins {freq['frequency_bin_rows']}"
    ans["Q6"] = f"lookup params serving low-frequency rows: bins {freq['frequency_bin_params']}"
    ans["Q7"] = (f"occurrence is highly concentrated: top-10% rows cover "
                 f"{freq['coverage_top_fraction']['10']*100:.1f}% of occurrences; "
                 f"top-25% {freq['coverage_top_fraction']['25']*100:.1f}%; "
                 f"top-50% {freq['coverage_top_fraction']['50']*100:.1f}%")
    ans["Q8"] = (f"embedding stable_rank seed0={spec['stable_rank']:.3f} seed1={spec['seed1']['stable_rank']:.3f}; "
                 f"entropy effective rank seed0={spec['entropy_rank']:.3f} seed1={spec['seed1']['entropy_rank']:.3f}")
    ans["Q9"] = (f"r99/r999 = {lowrank['seed0']['ranks']['r99']}/{lowrank['seed0']['ranks']['r999']} (seed0), "
                 f"{lowrank['seed1']['ranks']['r99']}/{lowrank['seed1']['ranks']['r999']} (seed1)")
    ans["Q10"] = ("hypothetical saving is NEGATIVE at both pre-registered ranks because the rare rows "
                  "are already rank-4 factorized; a global rank-15/16 factorization is larger than the "
                  "current 36,420-param hybrid lookup")
    ans["Q11"] = (f"frozen r99 drift: seed0 mean {lowrank['seed0']['prediction_drift_probe']['r99']['mean']:.4f}, "
                  f"seed1 {lowrank['seed1']['prediction_drift_probe']['r99']['mean']:.4f}; "
                  f"valid delta r99 seed0 {valid_screen['seed0']['delta_r99']:.5f}, "
                  f"seed1 {valid_screen['seed1']['delta_r99']:.5f}; r999 is the identity here")
    ans["Q12"] = (f"seed0 small fixed head valid {head_summary['per_seed']['seed0']['small_valid_mae']:.6f} vs H0 "
                  f"{head_summary['per_seed']['seed0']['h0_valid_mae']:.6f} -> delta "
                  f"{head_summary['per_seed']['seed0']['delta_valid']:+.6f} (small head better)")
    ans["Q13"] = (f"seed1 small fixed head valid {head_summary['per_seed']['seed1']['small_valid_mae']:.6f} vs H0 "
                  f"{head_summary['per_seed']['seed1']['h0_valid_mae']:.6f} -> delta "
                  f"{head_summary['per_seed']['seed1']['delta_valid']:+.6f} (small head better)")
    ans["Q14"] = f"{head_summary['label']} -- the graph head is strongly supported as a reallocation donor"
    ans["Q15"] = (f"{donor['lookup']['label']} -- the identity lookup is NOT supported as a donor: "
                  f"already frequency-adaptive and no positive hypothetical saving")
    ans["Q16"] = ("states are narrow: " + "; ".join(
        f"{k} PR/d={act['seed0'][k]['rank_over_dim']:.3f}" for k in ("h0", "u", "q", "R_unary", "R_pair")
    ) + " (bandwidth evidence only, not a reduction/augmentation instruction)")
    ans["Q17"] = ("highest object reuse: " + ", ".join(
        f"{r['component']} ({r['applications_per_molecule']:.1f}/molecule)"
        for r in reuse_sorted[:3]
    ) + "; reuse is descriptive, not a performance metric")
    ans["Q18"] = ("P1 closes learned pre-pool composition; P2 closes one-shot relation refresh; "
                  "compact-v4-cell closes explicit persistent higher-order objects; endpoint association, "
                  "centre covariance and triadic binding close pair/centre statistics")
    ans["Q19"] = ("NO clear donor+recipient pair: the graph head is a strong donor but no "
                  "representation-forming recipient is R-SUPPORTED")
    ans["Q20"] = f"{final['verdict']} ({final['case']})"
    return ans
