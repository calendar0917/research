"""Runner: I-CRATE-v0 incidence-structured dictionary transformer on ZINC.

Implements ``tracks/ksvd/notes/icrate_v0_preregistration.md``: persistent atom /
bond tokens on the incidence graph, 4 structural layers of Incidence-MSSA +
overcomplete nonnegative sparse dictionary coding with residual synthesis, a
size-sensitive graph seed, one-way whole-graph Cross-MSSA, a global sparse code
``alpha_G`` and a ``96 -> 64 -> 1`` head.  Canonical ZINC training protocol with
a fixed Top-5 checkpoint soup.

Official ZINC **test** is never loaded.  Data access is train+valid only
(``context.test_access`` from the control plane).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import icrate  # noqa: E402

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_icrate.yaml"

# canonical protocol (mirrors zinc_wg_icsc.OPTIMIZED_PROTOCOL)
OPTIMIZED_PROTOCOL: dict[str, Any] = {
    "optimizer": "Adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "max_epochs": 240,
    "patience": 40,
    "gradient_clip_norm": 5.0,
    "train_shuffle_seed_offset": 91011,
}

# Matched A100 Top-5 soup references (valid-only orientation, zinc-context-gap).
REFERENCE_SOUP_VALID = {"B-Full": 0.119818, "A0": 0.124704}

# Pre-registered gates (preregistration §10).
GATE_CASE_A = 0.20
GATE_CASE_B = 0.13

SPARSITY_EPS = 1.0e-8


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_icrate",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description="I-CRATE-v0 incidence-structured white-box dictionary transformer (ZINC)",
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# fingerprints / data
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _load_zinc_root(root: Path, split: str):
    from torch_geometric.datasets import ZINC

    return ZINC(root=str(root), subset=True, split=split)


def load_samples(root: Path, split: str, limit: int | None = None) -> list[icrate.Molecule]:
    if split == "test":
        raise RuntimeError("I-CRATE never loads the official test split")
    dataset = _load_zinc_root(root, split)
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    return [icrate.molecule_from_pyg(dataset[i]) for i in range(n)]


def make_batches(samples: Sequence[icrate.Molecule], batch_size: int) -> list[icrate.TokenBatch]:
    return [
        icrate.collate(list(samples[start : start + batch_size]))
        for start in range(0, len(samples), batch_size)
    ]


def is_atom_mask(batch: icrate.TokenBatch) -> torch.Tensor:
    idx = torch.arange(batch.T, device=batch.valid.device)
    return (idx.unsqueeze(0) < batch.n_atoms.unsqueeze(1)) & batch.valid


def is_bond_mask(batch: icrate.TokenBatch) -> torch.Tensor:
    nmax = batch.atom_idx.shape[1]
    idx = torch.arange(batch.T, device=batch.valid.device)
    return (idx.unsqueeze(0) >= nmax) & (idx.unsqueeze(0) < (nmax + batch.m_bonds).unsqueeze(1)) & batch.valid


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict(model: icrate.ICrateV0, batches, device) -> tuple[np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            moved = batch.to(device)
            prediction, _ = model(moved)
            targets.append(moved.y.cpu().numpy())
            preds.append(prediction.cpu().numpy())
    return np.concatenate(targets).astype(np.float64), np.concatenate(preds).astype(np.float64)


def mae(targets: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - preds)))


def evaluate_state(model, state, batches, device) -> float:
    if state is not None:
        model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    targets, preds = predict(model, batches, device)
    return mae(targets, preds)


def soup_state(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = list(states[0].keys())
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([s[key].float() for s in states], dim=0)
        out[key] = stacked.mean(dim=0).to(states[0][key].dtype)
    return out


# ---------------------------------------------------------------------------
# mechanism audit
# ---------------------------------------------------------------------------


def _token_odl_stats(coeff_rows: list[torch.Tensor]) -> dict[str, float]:
    """Per-layer token ODL sparsity / utilisation / concentration.

    ``coeff_rows[li]`` is ``[nv, M]``: the coefficients of every valid token
    (pooled across batches, so token counts never need to match).
    """
    out: dict[str, float] = {}
    n_layers = len(coeff_rows)
    per_layer_active = []
    per_layer_top1 = []
    per_layer_top5 = []
    per_layer_entropy = []
    per_layer_util = []
    per_layer_dead = []
    for li in range(n_layers):
        av = coeff_rows[li]  # [nv, M]
        nv = av.shape[0]
        nz = av > SPARSITY_EPS
        out[f"L{li}_nonzero_frac"] = float(nz.to(torch.float32).mean()) if nv else float("nan")
        active_count = nz.sum(dim=1).to(torch.float32)  # [nv]
        per_layer_active.append(active_count)
        mass = av.sum(dim=1)
        top1 = av.max(dim=1).values / (mass + 1e-12)
        per_layer_top1.append(top1)
        k = min(5, av.shape[1])
        top5 = av.topk(k, dim=1).values.sum(dim=1) / (mass + 1e-12)
        per_layer_top5.append(top5)
        p = av / (mass.unsqueeze(1) + 1e-12)
        ent = -(p * torch.log(p + 1e-12)).sum(dim=1)
        per_layer_entropy.append(ent)
        util = nz.to(torch.float32).mean(dim=0)  # [M] activation frequency
        per_layer_util.append(util.cpu())
        per_layer_dead.append(int((util == 0).sum()))

    def _cat(xs):
        return torch.cat(xs) if xs else torch.tensor([])

    all_active = _cat(per_layer_active)
    out["active_atoms_per_token_mean"] = float(all_active.mean()) if all_active.numel() else float("nan")
    out["active_atoms_per_token_median"] = float(all_active.median()) if all_active.numel() else float("nan")
    out["top1_mass_mean"] = float(_cat(per_layer_top1).mean())
    out["top5_mass_mean"] = float(_cat(per_layer_top5).mean())
    out["coefficient_entropy_mean"] = float(_cat(per_layer_entropy).mean())
    util_stack = torch.stack(per_layer_util)  # [L, M]
    out["dictionary_utilisation_mean"] = float(util_stack.mean())
    out["dictionary_utilisation_median"] = float(util_stack.median())
    for li in range(n_layers):
        out[f"L{li}_active_atoms_per_token_mean"] = float(per_layer_active[li].mean())
        out[f"L{li}_dictionary_utilisation_mean"] = float(per_layer_util[li].mean())
        out[f"L{li}_dead_atoms"] = per_layer_dead[li]
    out["dead_atoms_total"] = int(sum(per_layer_dead))
    return out


def _global_alpha_stats(alpha: torch.Tensor) -> dict[str, Any]:
    nz = alpha > SPARSITY_EPS
    n_graphs, m = alpha.shape
    util = nz.to(torch.float32).mean(dim=0)
    return {
        "nonzero_frac": float(nz.to(torch.float32).mean()),
        "active_atoms_per_graph_mean": float(nz.sum(dim=1).to(torch.float32).mean()),
        "active_atoms_per_graph_median": float(nz.sum(dim=1).to(torch.float32).median()),
        "dead_atoms": int((util == 0).sum()),
        "activation_frequency_mean": float(util.mean()),
        "activation_frequency_median": float(util.median()),
        "amplitude_mean": float(alpha.mean()),
        "amplitude_max": float(alpha.max()),
        "amplitude_p95": float(alpha.quantile(0.95)),
        "n_graphs": int(n_graphs),
    }


def _dictionary_audit(dictionaries: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mat in dictionaries.items():
        m = mat.detach().to(torch.float32)
        sv = torch.linalg.svdvals(m)
        eff_rank = float((sv.sum() ** 2) / (sv.pow(2).sum() + 1e-12))
        cols = m / (m.norm(dim=0, keepdim=True) + 1e-8)
        gram = cols.t() @ cols
        k = gram.shape[0]
        off = gram - torch.eye(k, dtype=gram.dtype, device=gram.device)
        coherence = off.abs()
        out[name] = {
            "effective_rank": eff_rank,
            "rank_ratio": eff_rank / float(m.shape[1]),
            "coherence_mean": float(coherence.sum() / max(k * (k - 1), 1)),
            "coherence_max": float(coherence.max()) if k > 1 else 0.0,
            "near_collinear_frac": float((coherence > 0.99).to(torch.float32).sum() / max(k * (k - 1), 1)),
            "singular_values": [float(v) for v in sv[:16]],
        }
    return out


def _structural_vitality(delta_rows: list[torch.Tensor], z_rows: list[torch.Tensor]) -> dict[str, float]:
    """``r_mssa = ||Delta_MSSA||_F / ||Z||_F`` per layer over pooled valid tokens."""
    out: dict[str, float] = {}
    for li in range(len(delta_rows)):
        d = delta_rows[li].pow(2).sum()
        z = z_rows[li].pow(2).sum()
        out[f"L{li}_mssa_ratio"] = float(torch.sqrt(d) / (torch.sqrt(z) + 1e-8))
        out[f"L{li}_mssa_fro"] = float(torch.sqrt(d))
    return out


def _attention_stats(model, batch, device) -> dict[str, float]:
    moved = batch.to(device)
    atom_m = is_atom_mask(moved)
    bond_m = is_bond_mask(moved)
    with torch.no_grad():
        _, info = model(moved, record_attn=True)
    out: dict[str, float] = {}
    entropies: list[float] = []
    maxws: list[float] = []
    for li, attn in enumerate(info.attn):
        p = attn.clamp_min(1e-12)
        ent = -(attn * torch.log(p)).sum(dim=-1)  # [B, T]
        has = attn.sum(dim=-1) > 0  # legal-key rows
        for name, m in (("atom", atom_m), ("bond", bond_m)):
            sel = has & m
            if sel.any():
                out[f"L{li}_{name}_attn_entropy_mean"] = float(ent[sel].mean())
                out[f"L{li}_{name}_attn_max_mean"] = float(attn[sel].max(dim=-1).values.mean())
        entropies.append(float(ent[has].mean()) if has.any() else float("nan"))
        maxws.append(float(attn[has].max(dim=-1).values.mean()) if has.any() else float("nan"))
    out["attn_entropy_mean"] = float(np.mean(entropies))
    out["attn_max_mean"] = float(np.mean(maxws))
    return out


def _incidence_rewire(mol: icrate.Molecule, generator: torch.Generator) -> icrate.Molecule:
    m = mol.m
    if m < 2:
        return mol
    pairs = list(zip(mol.src.tolist(), mol.dst.tolist()))
    perm = torch.randperm(m, generator=generator).tolist()
    new_src = [pairs[perm[e]][0] for e in range(m)]
    new_dst = [pairs[perm[e]][1] for e in range(m)]
    return icrate.Molecule(
        atom=mol.atom,
        src=torch.tensor(new_src, dtype=torch.long),
        dst=torch.tensor(new_dst, dtype=torch.long),
        bond=mol.bond,
        y=mol.y,
    )


def incidence_intervention(
    model: icrate.ICrateV0,
    samples: Sequence[icrate.Molecule],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Within-graph endpoint rewiring; returns Δα, Δy and layer-wise ΔZ."""
    generator = torch.Generator().manual_seed(int(seed))
    da: list[float] = []
    dy: list[float] = []
    ly: list[list[float]] = [[] for _ in range(icrate.L_LAYERS + 1)]
    with torch.no_grad():
        for mol in samples:
            rew = _incidence_rewire(mol, generator)
            mol_d = icrate.Molecule(mol.atom.to(device), mol.src.to(device), mol.dst.to(device), mol.bond.to(device), mol.y)
            rew_d = icrate.Molecule(rew.atom.to(device), rew.src.to(device), rew.dst.to(device), rew.bond.to(device), rew.y)
            pred_o, info_o = model.forward_reference(mol_d)
            pred_r, info_r = model.forward_reference(rew_d)
            denom = float(info_o.alpha.norm()) + 1e-8
            da.append(float((info_o.alpha - info_r.alpha).norm()) / denom)
            dy.append(float((pred_o - pred_r).abs()))
            for li in range(icrate.L_LAYERS + 1):
                zo = info_o.z_layers[li]
                zr = info_r.z_layers[li]
                ly[li].append(float((zo - zr).norm()) / (float(zo.norm()) + 1e-8))
    arr = np.asarray(da, dtype=np.float64)
    out: dict[str, Any] = {
        "n_graphs": len(samples),
        "alpha_relative_change_mean": float(arr.mean()),
        "alpha_relative_change_median": float(np.median(arr)),
        "alpha_relative_change_p10": float(np.quantile(arr, 0.10)),
        "alpha_relative_change_p90": float(np.quantile(arr, 0.90)),
        "prediction_abs_change_mean": float(np.mean(dy)),
    }
    for li in range(icrate.L_LAYERS + 1):
        vals = np.asarray(ly[li], dtype=np.float64)
        out[f"L{li}_z_relative_change_mean"] = float(vals.mean())
        out[f"L{li}_z_relative_change_median"] = float(np.median(vals))
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _epoch_support(model, batch, device) -> dict[str, float]:
    moved = batch.to(device)
    with torch.no_grad():
        _, info = model(moved)
    valid = moved.valid
    out: dict[str, float] = {}
    for li, a in enumerate(info.coeffs):
        av = a[valid]
        out[f"L{li}_nonzero"] = float((av > SPARSITY_EPS).to(torch.float32).mean())
    out["alpha_nonzero"] = float((info.alpha > SPARSITY_EPS).to(torch.float32).mean())
    delta_rows = [info.delta_mssa[li][valid].cpu() for li in range(icrate.L_LAYERS)]
    z_rows = [info.z_layers[li][valid].cpu() for li in range(icrate.L_LAYERS)]
    out.update(_structural_vitality(delta_rows, z_rows))
    return out


def train(
    model: icrate.ICrateV0,
    train_samples: Sequence[icrate.Molecule],
    valid_samples: Sequence[icrate.Molecule],
    *,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
    mech_batch: icrate.TokenBatch,
) -> dict[str, Any]:
    model_cfg = dict(config.get("model", {}))
    batch_size = int(model_cfg.get("batch_size", OPTIMIZED_PROTOCOL["batch_size"]))
    epochs = int(model_cfg.get("epochs", OPTIMIZED_PROTOCOL["max_epochs"]))
    patience = int(model_cfg.get("patience", OPTIMIZED_PROTOCOL["patience"]))
    lr = float(model_cfg.get("learning_rate", OPTIMIZED_PROTOCOL["learning_rate"]))
    wd = float(model_cfg.get("weight_decay", OPTIMIZED_PROTOCOL["weight_decay"]))
    clip = float(model_cfg.get("gradient_clip_norm", OPTIMIZED_PROTOCOL["gradient_clip_norm"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    valid_batches = make_batches(valid_samples, batch_size)
    generator = torch.Generator().manual_seed(seed + OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"])
    n_train = len(train_samples)

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    top_states: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).tolist()
        epoch_task = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            indices = order[start : start + batch_size]
            batch = icrate.collate([train_samples[i] for i in indices]).to(device)
            prediction, _ = model(batch)
            task = F.l1_loss(prediction, batch.y)
            optimizer.zero_grad()
            task.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_task += float(task.detach()) * len(indices)
            seen += len(indices)
        train_task = epoch_task / max(seen, 1)

        valid_mae = evaluate_state(model, None, valid_batches, device)
        snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_states.append((valid_mae, epoch, snapshot))
        top_states = sorted(top_states, key=lambda item: (item[0], item[1]))[:5]

        support = _epoch_support(model, mech_batch, device)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_task_mae": float(train_task),
            "valid_mae": float(valid_mae),
            "best_valid_mae": float(min(best_mae, valid_mae)),
            **support,
        }
        history.append(row)
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = snapshot
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[icrate seed{seed}] epoch={epoch:03d} train={train_task:.5f} valid={valid_mae:.6f} "
                f"best={best_mae:.6f}@{best_epoch} L0nz={support.get('L0_nonzero', float('nan')):.3f} "
                f"a_nz={support.get('alpha_nonzero', float('nan')):.3f} "
                f"mssa0={support.get('L0_mssa_ratio', float('nan')):.3f}",
                flush=True,
            )
        if stale >= patience:
            print(f"[icrate seed{seed}] early_stop epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_valid_mae = evaluate_state(model, best_state, valid_batches, device)
    soup = soup_state([state for _m, _e, state in top_states])
    soup_valid_mae = evaluate_state(model, soup, valid_batches, device)
    model.load_state_dict(soup)

    return {
        "model": model,
        "best_state": best_state,
        "soup_state": soup,
        "best_epoch": int(best_epoch),
        "best_valid_mae": float(best_valid_mae),
        "soup_valid_mae": float(soup_valid_mae),
        "top5_epochs": [int(e) for _m, e, _s in top_states],
        "top5_valid_mae": [float(m) for m, _e, _s in top_states],
        "epochs_run": len(history),
        "history": history,
        "wall_clock_seconds": float(time.perf_counter() - started),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }


# ---------------------------------------------------------------------------
# final full audit
# ---------------------------------------------------------------------------


def full_audit(
    model,
    valid_batches: Sequence[icrate.TokenBatch],
    device: torch.device,
    seed: int,
    train_batch: icrate.TokenBatch,
) -> dict[str, Any]:
    n_layers = icrate.L_LAYERS
    has_odl = hasattr(model, "frozen_dictionaries")
    coeffs_all: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    deltas_all: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    zinputs_all: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    alpha_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in valid_batches:
            moved = batch.to(device)
            _, info = model(moved)
            vb = moved.valid
            for li in range(n_layers):
                deltas_all[li].append(info.delta_mssa[li][vb].cpu())  # [nv, d]
                zinputs_all[li].append(info.z_layers[li][vb].cpu())  # [nv, d]
                if has_odl:
                    coeffs_all[li].append(info.coeffs[li][vb].cpu())  # [nv, M]
            alpha_all.append(info.alpha.cpu())
    deltas = [torch.cat(d) for d in deltas_all]
    zinputs = [torch.cat(z) for z in zinputs_all]
    alpha = torch.cat(alpha_all)

    token_stats = _token_odl_stats([torch.cat(c) for c in coeffs_all]) if has_odl else {}
    global_stats = _global_alpha_stats(alpha)
    vitality = _structural_vitality(deltas, zinputs)
    dict_audit = _dictionary_audit(model.frozen_dictionaries()) if has_odl else {}
    attn_stats = _attention_stats(model, train_batch, device)
    return {
        "variant": "icrate" if has_odl else "ffn_control",
        "token_odl": token_stats,
        "global_alpha": global_stats,
        "structural_vitality": vitality,
        "dictionary_audit": dict_audit,
        "attention": attn_stats,
    }


def _gradient_vitality(model, batch, device) -> dict[str, float]:
    moved = batch.to(device)
    model.zero_grad(set_to_none=True)
    prediction, _ = model(moved)
    loss = F.l1_loss(prediction, moved.y)
    loss.backward()
    out: dict[str, float] = {}
    def _norm(p):
        return float(p.grad.norm()) if p.grad is not None else float("nan")
    out["E_V"] = _norm(model.e_v.weight)
    out["E_E"] = _norm(model.e_e.weight)
    for li, layer in enumerate(model.layers):
        out[f"U^{li}"] = _norm(layer.u)
        if hasattr(layer, "d_a"):
            out[f"D_a^{li}"] = _norm(layer.d_a)
            out[f"D_s^{li}"] = _norm(layer.d_s)
        else:
            out[f"w1^{li}"] = _norm(layer.w1.weight)
            out[f"w2^{li}"] = _norm(layer.w2.weight)
    out["U^G"] = _norm(model.u_g)
    if hasattr(model, "d_g"):
        out["D_G"] = _norm(model.d_g)
    else:
        out["g_proj"] = _norm(model.g_proj.weight)
    out["head"] = _norm(model.head[0].weight)
    model.zero_grad(set_to_none=True)
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def _write_plots(artifact_dir: Path, history, audit) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return paths
    artifact_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, [row["train_task_mae"] for row in history], label="train task MAE")
    ax[0].plot(epochs, [row["valid_mae"] for row in history], label="valid MAE")
    ax[0].set_ylabel("MAE")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    for li in range(icrate.L_LAYERS):
        ax[1].plot(epochs, [row.get(f"L{li}_nonzero", np.nan) for row in history], label=f"L{li} nz")
    ax[1].plot(epochs, [row.get("alpha_nonzero", np.nan) for row in history], label="alpha nz", ls="--")
    ax[1].set_ylabel("nonzero fraction")
    ax[1].set_xlabel("epoch")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    p = artifact_dir / "training_curve.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p.name)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for li in range(icrate.L_LAYERS):
        ax[0].plot(epochs, [row.get(f"L{li}_mssa_ratio", np.nan) for row in history], label=f"L{li}")
    ax[0].set_ylabel("r_mssa = ||ΔMSSA||/||Z||")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    spectra = {k: v["singular_values"] for k, v in audit["dictionary_audit"].items()}
    for name, sv in spectra.items():
        ax[1].plot(range(len(sv)), sv, marker="o", ms=2, label=name)
    ax[1].set_ylabel("singular value")
    ax[1].set_xlabel("index")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    p = artifact_dir / "vitality_spectra.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p.name)
    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run(config: dict[str, Any], context: RunContext) -> RunResult:
    model_cfg = dict(config.get("model", {}))
    data_root = resolve_path(config["data"]["root"])
    seed = int(config.get("seed", 0))
    device = torch.device(str(model_cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("icrate requested CUDA but CUDA is unavailable")
    threads = int(config.get("runtime", {}).get("torch_threads", 4))
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if context.test_access == "granted":
        raise RuntimeError("I-CRATE is a feasibility round; it must never run with test access granted")

    limit_train = config.get("data", {}).get("limit_train")
    limit_valid = config.get("data", {}).get("limit_valid")
    print(f"[icrate] loading ZINC train/valid (limit_train={limit_train}, limit_valid={limit_valid})", flush=True)
    train_samples = load_samples(data_root, "train", limit_train)
    valid_samples = load_samples(data_root, "val", limit_valid)
    print(f"[icrate] train={len(train_samples)} valid={len(valid_samples)}", flush=True)

    model = icrate.ICrateV0(
        seed=seed,
        init_std=float(model_cfg.get("init_std", 1.0)),
        no_incidence=bool(model_cfg.get("no_incidence", False)),
    ).to(device) if str(model_cfg.get("variant", "icrate")) != "ffn" else icrate.ICrateFFNControl(
        seed=seed,
        init_std=float(model_cfg.get("init_std", 1.0)),
        no_incidence=bool(model_cfg.get("no_incidence", False)),
    ).to(device)
    variant = str(model_cfg.get("variant", "icrate"))
    parameters = model.parameter_breakdown()
    print(f"[icrate] params={parameters}", flush=True)

    mech_batch = icrate.collate(train_samples[:128])
    trained = train(
        model,
        train_samples,
        valid_samples,
        config=config,
        seed=seed,
        device=device,
        mech_batch=mech_batch,
    )

    valid_batches = make_batches(valid_samples, int(model_cfg.get("batch_size", 128)))
    print("[icrate] running full mechanism audit ...", flush=True)
    audit = full_audit(model, valid_batches, device, seed, mech_batch)
    grad_vitality = _gradient_vitality(model, icrate.collate(train_samples[:128]).to(device), device)

    n_interv = int(config.get("intervention", {}).get("n_graphs", 500))
    interv_samples = valid_samples[:n_interv]
    print(f"[icrate] incidence intervention on {len(interv_samples)} valid graphs ...", flush=True)
    intervention = incidence_intervention(model, interv_samples, device, seed=seed + 777)

    plots = _write_plots(context.artifact_dir, trained["history"], audit)
    torch.save(trained["soup_state"], context.artifact_dir / "soup_state.pt")

    soup_valid = float(trained["soup_valid_mae"])
    if soup_valid > GATE_CASE_A:
        gate = "A"
    elif soup_valid > GATE_CASE_B:
        gate = "B"
    else:
        gate = "C"

    results: dict[str, Any] = {
        "candidate": "icrate-v0",
        "variant": variant,
        "seed": int(seed),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else None
        ),
        "parameters": parameters,
        "config": {k: v for k, v in config.items() if k != "output"},
        "train_graphs": len(train_samples),
        "valid_graphs": len(valid_samples),
        "official_test_loaded": False,
        "test_access": context.test_access,
        "best_valid_mae": trained["best_valid_mae"],
        "soup_valid_mae": soup_valid,
        "best_epoch": trained["best_epoch"],
        "top5_epochs": trained["top5_epochs"],
        "top5_valid_mae": trained["top5_valid_mae"],
        "epochs_run": trained["epochs_run"],
        "wall_clock_seconds": trained["wall_clock_seconds"],
        "history": trained["history"],
        "audit": audit,
        "gradient_vitality": grad_vitality,
        "incidence_intervention": intervention,
        "reference_soup_valid": REFERENCE_SOUP_VALID,
        "pre_registered_gate": gate,
        "artifacts": plots,
    }
    (context.artifact_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    metrics = {
        "valid_mae": soup_valid,
        "valid_soup_mae": soup_valid,
        "valid_best_mae": float(trained["best_valid_mae"]),
        "best_epoch": int(trained["best_epoch"]),
        "epochs_run": int(trained["epochs_run"]),
        "parameters": int(parameters["total"]),
        "runtime_seconds": float(trained["wall_clock_seconds"]),
        "split_sizes": {"train": len(train_samples), "valid": len(valid_samples), "test": None},
        "test_access": context.test_access,
        "pre_registered_gate": gate,
        "alpha_nonzero_frac": audit["global_alpha"].get("nonzero_frac"),
        "alpha_dead_atoms": audit["global_alpha"].get("dead_atoms"),
        "L0_nonzero_frac": audit["token_odl"].get("L0_nonzero_frac") if audit["token_odl"] else None,
        "L0_mssa_ratio": audit["structural_vitality"].get("L0_mssa_ratio"),
        "incidence_alpha_relative_change": float(intervention["alpha_relative_change_mean"]),
        "incidence_pred_abs_change": float(intervention["prediction_abs_change_mean"]),
    }
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=["artifacts/results.json", "artifacts/soup_state.pt", *[f"artifacts/{n}" for n in plots]],
    )
