"""Runner: WG-ICSC-v0 whole-graph incidence-coupled sparse coding on ZINC.

Implements the pre-registration
``tracks/ksvd/notes/wg_icsc_v0_preregistration.md``: raw atom/bond one-hot
inputs, shared relational dictionaries, an unrolled whole-graph block
proximal-gradient solver, a 192-D row-L2 graph code, a two-layer head, and the
canonical ZINC training protocol with a fixed Top-5 checkpoint soup.

The official ZINC **test** split is never loaded.  Data access is train+valid
only, obeying ``context.test_access`` from the control plane.
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

# The scientific model module lives in the repository tree (not in the installed
# wheel); make the repo importable regardless of the invocation cwd, exactly
# like the legacy runner does for its experiment module.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import icsc  # noqa: E402

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_wg_icsc.yaml"

# canonical protocol (mirrors zinc_compact_v4_smallhead_e2e.OPTIMIZED_PROTOCOL)
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

# x1 calibration grid; the whole grid may be scaled by 3 once.
CALIBRATION_GRID = [
    (0.01, 0.02),
    (0.01, 0.05),
    (0.01, 0.10),
    (0.03, 0.02),
    (0.03, 0.05),
    (0.03, 0.10),
    (0.10, 0.02),
    (0.10, 0.05),
    (0.10, 0.10),
]
TARGET_WHOLE = (0.15, 0.30)
TARGET_ELEMENT_IN_ACTIVE = (0.20, 0.50)
COLLAPSE_DEAD_FRACTION = 0.60


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_wg_icsc",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description="WG-ICSC-v0 whole-graph incidence-coupled sparse coding (ZINC)",
        run_module=sys.modules[__name__],
    )


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _load_zinc_root(root: Path, split: str):
    from torch_geometric.datasets import ZINC

    return ZINC(root=str(root), subset=True, split=split)


def load_samples(root: Path, split: str, limit: int | None = None) -> list[icsc.GraphSample]:
    """Official PyG ZINC split -> list of :class:`GraphSample` (never test here)."""
    if split == "test":
        raise RuntimeError("WG-ICSC never loads the official test split")
    dataset = _load_zinc_root(root, split)
    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    return [icsc.graph_sample_from_pyg(dataset[i]) for i in range(n)]


def make_batches(samples: Sequence[icsc.GraphSample], batch_size: int) -> list[icsc.Batch]:
    return [
        icsc.collate(list(samples[start : start + batch_size]))
        for start in range(0, len(samples), batch_size)
    ]


# ---------------------------------------------------------------------------
# calibration (train-only, unlabelled, once)
# ---------------------------------------------------------------------------


def _interval_distance(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


def select_calibration(rows: Sequence[Mapping[str, Any]], k_total: int) -> dict[str, Any]:
    """Apply the frozen selection rule to the calibration table."""
    dead_limit = COLLAPSE_DEAD_FRACTION * k_total
    eligible = [r for r in rows if (r["dead_node_atoms"] + r["dead_edge_atoms"]) <= dead_limit]
    pool = eligible or list(rows)

    def score(row: Mapping[str, Any]) -> float:
        return _interval_distance(
            float(row["whole_active_row_frac_mean"]), *TARGET_WHOLE
        ) + 0.5 * _interval_distance(
            float(row["element_nonzero_frac_in_active_rows"]), *TARGET_ELEMENT_IN_ACTIVE
        )

    ranked = sorted(
        pool, key=lambda r: (score(r), r["dead_node_atoms"] + r["dead_edge_atoms"])
    )
    chosen = ranked[0]
    return {
        "chosen": chosen,
        "score": score(chosen),
        "eligible_rows": len(eligible),
        "dead_limit": dead_limit,
        "rule": (
            "discard rows with dead atoms > 0.60*(K_V+K_E); then minimise "
            "dist(whole_active,[0.15,0.30]) + 0.5*dist(element_in_active,[0.20,0.50]); "
            "ties by fewer dead atoms"
        ),
    }


def calibrate(
    model: icsc.WGICSC,
    train_samples: Sequence[icsc.GraphSample],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cal_cfg = dict(config.get("calibration", {}))
    n_graphs = int(cal_cfg.get("n_graphs", 256))
    max_expansions = int(cal_cfg.get("max_expansions", 1))
    subset = list(train_samples[:n_graphs])
    device = next(model.parameters()).device
    batch = icsc.collate(subset).to(device)
    d_v, d_e = model.normalized_dictionaries()

    table: list[dict[str, Any]] = []
    all_tables: list[dict[str, Any]] = []
    scale = 1.0
    expansion = 0
    while True:
        table = []
        for lambda1, lambda_g in CALIBRATION_GRID:
            with torch.no_grad():
                out = icsc.solve_batched(
                    d_v,
                    d_e,
                    model.w,
                    batch,
                    gamma=model.gamma,
                    rho=model.rho,
                    lambda1=lambda1 * scale,
                    lambda_g=lambda_g * scale,
                    t_steps=model.t_steps,
                )
            metrics = icsc.support_metrics(out, batch)
            table.append(
                {
                    "lambda1": float(lambda1 * scale),
                    "lambda_g": float(lambda_g * scale),
                    "grid_scale": float(scale),
                    "whole_active_row_frac_mean": metrics["whole_active_row_frac_mean"],
                    "node_active_row_frac_mean": metrics["node_active_row_frac_mean"],
                    "edge_active_row_frac_mean": metrics["edge_active_row_frac_mean"],
                    "element_nonzero_frac": metrics["element_nonzero_frac"],
                    "element_nonzero_frac_in_active_rows": metrics[
                        "element_nonzero_frac_in_active_rows"
                    ],
                    "dead_node_atoms": metrics["dead_node_atoms"],
                    "dead_edge_atoms": metrics["dead_edge_atoms"],
                }
            )
        whole = [row["whole_active_row_frac_mean"] for row in table]
        all_tables.append({"grid_scale": float(scale), "rows": table})
        all_dense = all(value > TARGET_WHOLE[1] for value in whole)
        all_collapse = all(value < 0.05 for value in whole)
        if (all_dense or all_collapse) and expansion < max_expansions:
            scale *= 3.0
            expansion += 1
            print(
                f"[wg-icsc] calibration x1 all_dense={all_dense} all_collapse={all_collapse}; "
                f"expanding grid x{scale:g}",
                flush=True,
            )
            continue
        break

    selection = select_calibration(table, icsc.K_V + icsc.K_E)
    chosen = selection["chosen"]
    result = {
        "n_graphs": len(subset),
        "grid_scale": float(chosen["grid_scale"]),
        "expansions": int(expansion),
        "table": table,
        "all_tables": all_tables,
        "selection": selection,
        "frozen_lambda1": float(chosen["lambda1"]),
        "frozen_lambda_g": float(chosen["lambda_g"]),
    }
    print(
        f"[wg-icsc] calibration frozen lambda1={result['frozen_lambda1']} "
        f"lambda_g={result['frozen_lambda_g']} whole={chosen['whole_active_row_frac_mean']:.3f} "
        f"el_in_active={chosen['element_nonzero_frac_in_active_rows']:.3f}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict(model: icsc.WGICSC, batches: Sequence[icsc.Batch], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            moved = batch.to(device)
            prediction, _ = model(moved)
            targets.append(moved.y.cpu().numpy())
            preds.append(prediction.cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float64),
        np.concatenate(preds).astype(np.float64),
    )


def mae(targets: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - preds)))


def evaluate_state(
    model: icsc.WGICSC,
    state: Mapping[str, torch.Tensor] | None,
    batches: Sequence[icsc.Batch],
    device: torch.device,
) -> float:
    if state is not None:
        model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    targets, preds = predict(model, batches, device)
    return mae(targets, preds)


def soup_state(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = list(states[0].keys())
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([state[key].float() for state in states], dim=0)
        out[key] = stacked.mean(dim=0).to(states[0][key].dtype)
    return out


# ---------------------------------------------------------------------------
# mechanism metrics
# ---------------------------------------------------------------------------


def _support_summary(model: icsc.WGICSC, batch: icsc.Batch, device: torch.device) -> dict[str, float]:
    moved = batch.to(device)
    with torch.no_grad():
        out = model.solve(moved)
    metrics = icsc.support_metrics(out, moved)
    d_v, d_e = model.normalized_dictionaries()
    metrics.update(icsc.dictionary_stats(num_graphs=moved.n_graphs, d_v=d_v, d_e=d_e, w=model.w))
    return metrics


def _activation_utilisation(
    model: icsc.WGICSC, batches: Sequence[icsc.Batch], device: torch.device
) -> dict[str, Any]:
    """Per-atom activation frequency / amplitude over a set of batches."""
    node_freq = torch.zeros(icsc.K_V)
    edge_freq = torch.zeros(icsc.K_E)
    node_amp = torch.zeros(icsc.K_V)
    edge_amp = torch.zeros(icsc.K_E)
    n_graphs = 0
    with torch.no_grad():
        for batch in batches:
            moved = batch.to(device)
            out = model.solve(moved)
            node_freq += (out.per_graph["node_sq"] > icsc.SPARSITY_EPS**2).float().sum(dim=1).cpu()
            edge_freq += (out.per_graph["edge_sq"] > icsc.SPARSITY_EPS**2).float().sum(dim=1).cpu()
            node_amp += torch.sqrt(out.per_graph["node_sq"] + icsc.NORM_EPS).sum(dim=1).cpu()
            edge_amp += torch.sqrt(out.per_graph["edge_sq"] + icsc.NORM_EPS).sum(dim=1).cpu()
            n_graphs += moved.n_graphs
    denom = max(n_graphs, 1)
    return {
        "n_graphs": int(n_graphs),
        "node_activation_frequency_mean": float(node_freq.mean() / denom),
        "edge_activation_frequency_mean": float(edge_freq.mean() / denom),
        "node_activation_frequency_median": float(node_freq.median() / denom),
        "edge_activation_frequency_median": float(edge_freq.median() / denom),
        "node_dead_atoms": int((node_freq == 0).sum()),
        "edge_dead_atoms": int((edge_freq == 0).sum()),
        "node_mean_amplitude": float(node_amp.mean() / denom),
        "edge_mean_amplitude": float(edge_amp.mean() / denom),
    }


def _solver_dynamics(model: icsc.WGICSC, batch: icsc.Batch, device: torch.device, t_steps: int) -> dict[str, Any]:
    moved = batch.to(device)
    with torch.no_grad():
        out = model.solve(moved, t_steps=t_steps, record=True)
    return {
        "t_steps": int(t_steps),
        "steps": out.step_stats,
        "grad_ratio": out.grad_ratio_stats,
    }


def _relative_change(model: icsc.WGICSC, batch: icsc.Batch, device: torch.device, t_a: int, t_b: int) -> float:
    moved = batch.to(device)
    with torch.no_grad():
        alpha_a = model.solve(moved, t_steps=t_a).alpha
        alpha_b = model.solve(moved, t_steps=t_b).alpha
    denom = float(alpha_a.norm()) + 1e-8
    return float((alpha_a - alpha_b).norm()) / denom


def _assignment_permutation_intervention(
    model: icsc.WGICSC,
    batches: Sequence[icsc.Batch],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Evaluation-only: reassign bond objects to different endpoint pairs.

    The permutation is applied **within each graph** so the bond objects stay
    attached to their own molecule; only the bond<->endpoint assignment changes.
    """
    generator = torch.Generator().manual_seed(int(seed))
    ratios: list[float] = []
    pred_changes: list[float] = []
    with torch.no_grad():
        for batch in batches:
            moved = batch.to(device)
            base = model.solve(moved)
            base_pred = model.head(base.alpha).view(-1)
            perm = torch.arange(moved.M, device=device)
            for g in range(moved.n_graphs):
                idx = (moved.edge_graph == g).nonzero(as_tuple=True)[0]
                if idx.numel() > 1:
                    local = torch.randperm(idx.numel(), generator=generator).to(device)
                    perm[idx] = idx[local]
            shuffled = icsc.Batch(
                xv=moved.xv,
                xe=moved.xe,
                src=moved.src[perm],
                dst=moved.dst[perm],
                node_graph=moved.node_graph,
                edge_graph=moved.edge_graph,
                n_nodes=moved.n_nodes,
                m_edges=moved.m_edges,
                y=moved.y,
                n_graphs=moved.n_graphs,
            )
            alt = model.solve(shuffled)
            alt_pred = model.head(alt.alpha).view(-1)
            for g in range(moved.n_graphs):
                denom = float(base.alpha[g].norm()) + 1e-8
                ratios.append(float((base.alpha[g] - alt.alpha[g]).norm()) / denom)
            pred_changes.append(float((base_pred - alt_pred).abs().mean()))
    return {
        "alpha_relative_change_mean": float(np.mean(ratios)),
        "alpha_relative_change_median": float(np.median(ratios)),
        "alpha_relative_change_p10": float(np.quantile(ratios, 0.10)),
        "alpha_relative_change_p90": float(np.quantile(ratios, 0.90)),
        "prediction_abs_change_mean": float(np.mean(pred_changes)),
    }


def _gradient_vitality(dynamics: Mapping[str, Any]) -> dict[str, float]:
    ratios = []
    for row in dynamics.get("grad_ratio", []):
        attr = float(row.get("g_attr", 0.0))
        comp = float(row.get("g_comp", 0.0))
        ratios.append(comp / (attr + 1e-12))
    if not ratios:
        return {"composition_to_attribute_grad_ratio_mean": float("nan")}
    ratios_np = np.asarray(ratios, dtype=np.float64)
    return {
        "composition_to_attribute_grad_ratio_mean": float(np.mean(ratios_np)),
        "composition_to_attribute_grad_ratio_median": float(np.median(ratios_np)),
        "composition_to_attribute_grad_ratio_p10": float(np.quantile(ratios_np, 0.10)),
        "composition_to_attribute_grad_ratio_p90": float(np.quantile(ratios_np, 0.90)),
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    model: icsc.WGICSC,
    train_samples: Sequence[icsc.GraphSample],
    valid_samples: Sequence[icsc.GraphSample],
    *,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
    mech_batches: Sequence[icsc.Batch],
) -> dict[str, Any]:
    model_cfg = dict(config.get("model", {}))
    batch_size = int(model_cfg.get("batch_size", OPTIMIZED_PROTOCOL["batch_size"]))
    epochs = int(model_cfg.get("epochs", OPTIMIZED_PROTOCOL["max_epochs"]))
    patience = int(model_cfg.get("patience", OPTIMIZED_PROTOCOL["patience"]))
    lr = float(model_cfg.get("learning_rate", OPTIMIZED_PROTOCOL["learning_rate"]))
    wd = float(model_cfg.get("weight_decay", OPTIMIZED_PROTOCOL["weight_decay"]))
    clip = float(model_cfg.get("gradient_clip_norm", OPTIMIZED_PROTOCOL["gradient_clip_norm"]))
    mu = float(model_cfg.get("mu", icsc.MU))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    valid_batches = make_batches(valid_samples, batch_size)
    generator = torch.Generator().manual_seed(seed + OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"])
    n_train = len(train_samples)

    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    top_states: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).tolist()
        epoch_task = 0.0
        epoch_fit = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            indices = order[start : start + batch_size]
            batch = icsc.collate([train_samples[i] for i in indices]).to(device)
            prediction, output = model(batch)
            task = F.l1_loss(prediction, batch.y)
            fit = model.fit_loss(batch, output)
            loss = task + mu * fit
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_task += float(task.detach()) * len(indices)
            epoch_fit += float(fit.detach()) * len(indices)
            seen += len(indices)
        train_task = epoch_task / max(seen, 1)
        train_fit = epoch_fit / max(seen, 1)

        valid_mae = evaluate_state(model, None, valid_batches, device)
        snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_states.append((valid_mae, epoch, snapshot))
        top_states = sorted(top_states, key=lambda item: (item[0], item[1]))[:5]

        support = _support_summary(model, mech_batches[0], device)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_task_mae": float(train_task),
            "train_fit": float(train_fit),
            "valid_mae": float(valid_mae),
            "best_valid_mae": float(min(best_mae, valid_mae)),
            "whole_active_row_frac": float(support["whole_active_row_frac_mean"]),
            "node_active_row_frac": float(support["node_active_row_frac_mean"]),
            "edge_active_row_frac": float(support["edge_active_row_frac_mean"]),
            "element_nonzero_frac": float(support["element_nonzero_frac"]),
            "element_in_active_frac": float(support["element_nonzero_frac_in_active_rows"]),
            "dead_node_atoms": float(support["dead_node_atoms"]),
            "dead_edge_atoms": float(support["dead_edge_atoms"]),
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
                f"[wg-icsc seed{seed}] epoch={epoch:03d} task={train_task:.5f} fit={train_fit:.5f} "
                f"valid={valid_mae:.6f} best={best_mae:.6f}@{best_epoch} "
                f"whole_active={support['whole_active_row_frac_mean']:.3f}",
                flush=True,
            )
        if stale >= patience:
            print(f"[wg-icsc seed{seed}] early_stop epoch={epoch} best={best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_valid_mae = evaluate_state(model, best_state, valid_batches, device)
    soup = soup_state([state for _mae_value, _ep, state in top_states])
    soup_valid_mae = evaluate_state(model, soup, valid_batches, device)
    model.load_state_dict(soup)

    return {
        "model": model,
        "best_state": best_state,
        "soup_state": soup,
        "best_epoch": int(best_epoch),
        "best_valid_mae": float(best_valid_mae),
        "soup_valid_mae": float(soup_valid_mae),
        "top5_epochs": [int(ep) for _m, ep, _s in top_states],
        "top5_valid_mae": [float(m) for m, _ep, _s in top_states],
        "epochs_run": len(history),
        "history": history,
        "wall_clock_seconds": float(time.perf_counter() - started),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def _write_plots(artifact_dir: Path, history: Sequence[Mapping[str, Any]], dynamics: Mapping[str, Any], calibration: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is best-effort
        return paths

    artifact_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(epochs, [row["train_task_mae"] for row in history], label="train task MAE")
    ax[0].plot(epochs, [row["valid_mae"] for row in history], label="valid MAE")
    ax[0].set_ylabel("MAE")
    ax[0].legend()
    ax[1].plot(epochs, [row["whole_active_row_frac"] for row in history], label="whole active")
    ax[1].plot(epochs, [row["node_active_row_frac"] for row in history], label="node active")
    ax[1].plot(epochs, [row["edge_active_row_frac"] for row in history], label="edge active")
    ax[1].set_ylabel("active row fraction")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    p = artifact_dir / "training_curve.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p.name)

    steps = [row["step"] for row in dynamics.get("steps", [])]
    if steps:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(steps, [row["energy"] for row in dynamics["steps"]], marker="o")
        ax[0].set_ylabel("lower energy")
        ax[1].plot(steps, [row["node_recon"] for row in dynamics["steps"]], marker="o", label="node recon")
        ax[1].plot(steps, [row["edge_recon"] for row in dynamics["steps"]], marker="o", label="edge recon")
        ax[1].plot(steps, [row["composition"] for row in dynamics["steps"]], marker="o", label="composition")
        ax[1].set_ylabel("per-step term")
        ax[1].set_xlabel("solver step")
        ax[1].legend()
        fig.tight_layout()
        p = artifact_dir / "solver_dynamics.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p.name)

    table = calibration.get("table", [])
    if table:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [f"{row['lambda1']:.2f}" for row in table]
        ys = [f"{row['lambda_g']:.2f}" for row in table]
        values = [row["whole_active_row_frac_mean"] for row in table]
        scatter = ax.scatter(range(len(table)), values, c=values, cmap="viridis")
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels([f"l1={x}\nlg={y}" for x, y in zip(xs, ys)], fontsize=6)
        ax.axhspan(TARGET_WHOLE[0], TARGET_WHOLE[1], color="green", alpha=0.15)
        ax.set_ylabel("whole active row fraction")
        fig.colorbar(scatter, ax=ax)
        fig.tight_layout()
        p = artifact_dir / "calibration.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p.name)
    return paths


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def run(config: dict[str, Any], context: RunContext) -> RunResult:
    model_cfg = dict(config.get("model", {}))
    data_root = resolve_path(config["data"]["root"])
    seed = int(config.get("seed", 0))
    device = torch.device(str(model_cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("wg-icsc requested CUDA but CUDA is unavailable")
    threads = int(config.get("runtime", {}).get("torch_threads", 4))
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if context.test_access == "granted":
        raise RuntimeError(
            "WG-ICSC is a feasibility round; it must never run with test access granted"
        )

    limit_train = config.get("data", {}).get("limit_train")
    limit_valid = config.get("data", {}).get("limit_valid")
    print(f"[wg-icsc] loading ZINC train/valid (limit_train={limit_train}, limit_valid={limit_valid})", flush=True)
    train_samples = load_samples(data_root, "train", limit_train)
    valid_samples = load_samples(data_root, "val", limit_valid)
    print(f"[wg-icsc] train={len(train_samples)} valid={len(valid_samples)}", flush=True)

    model = icsc.WGICSC(
        seed=seed,
        gamma=float(model_cfg.get("gamma", icsc.GAMMA)),
        rho=float(model_cfg.get("rho", icsc.RHO)),
        lambda1=0.0,
        lambda_g=0.0,
        t_steps=int(model_cfg.get("t_steps", icsc.T_STEPS)),
        init_std=float(model_cfg.get("init_std", 0.5)),
    ).to(device)
    parameters = model.parameter_groups()

    calibration = calibrate(model, train_samples, config)
    lambda1 = float(model_cfg.get("lambda1") if model_cfg.get("lambda1") is not None else calibration["frozen_lambda1"])
    lambda_g = float(model_cfg.get("lambda_g") if model_cfg.get("lambda_g") is not None else calibration["frozen_lambda_g"])
    model.lambda1 = lambda1
    model.lambda_g = lambda_g
    print(
        f"[wg-icsc] using lambda1={lambda1} lambda_g={lambda_g} gamma={model.gamma} mu={model_cfg.get('mu', icsc.MU)}",
        flush=True,
    )

    mech_batches = make_batches(train_samples[:128], 128)
    trained = train(
        model,
        train_samples,
        valid_samples,
        config=config,
        seed=seed,
        device=device,
        mech_batches=mech_batches,
    )

    valid_batches = make_batches(valid_samples, int(model_cfg.get("batch_size", 128)))
    dynamics = _solver_dynamics(model, mech_batches[0], device, model.t_steps)
    vitality = _gradient_vitality(dynamics)
    intervention = _assignment_permutation_intervention(
        model, valid_batches, device, seed=seed + 777
    )
    t16 = {
        "alpha_relative_change_T8_to_T16": _relative_change(model, mech_batches[0], device, 8, 16),
        "dynamics_T16": _solver_dynamics(model, mech_batches[0], device, 16),
    }
    utilisation = _activation_utilisation(model, valid_batches, device)
    d_v, d_e = model.normalized_dictionaries()
    dict_stats = icsc.dictionary_stats(
        num_graphs=len(valid_samples), d_v=d_v, d_e=d_e, w=model.w
    )

    plots = _write_plots(context.artifact_dir, trained["history"], dynamics, calibration)
    torch.save(trained["soup_state"], context.artifact_dir / "soup_state.pt")

    results: dict[str, Any] = {
        "candidate": "wg-icsc-v0",
        "seed": int(seed),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else None
        ),
        "parameters": parameters,
        "config": {k: v for k, v in config.items() if k != "output"},
        "lambda1": lambda1,
        "lambda_g": lambda_g,
        "calibration": calibration,
        "train_graphs": len(train_samples),
        "valid_graphs": len(valid_samples),
        "official_test_loaded": False,
        "test_access": context.test_access,
        "best_valid_mae": trained["best_valid_mae"],
        "soup_valid_mae": trained["soup_valid_mae"],
        "best_epoch": trained["best_epoch"],
        "top5_epochs": trained["top5_epochs"],
        "top5_valid_mae": trained["top5_valid_mae"],
        "epochs_run": trained["epochs_run"],
        "wall_clock_seconds": trained["wall_clock_seconds"],
        "history": trained["history"],
        "solver_dynamics": dynamics,
        "gradient_vitality": vitality,
        "assignment_intervention": intervention,
        "T16_check": t16,
        "utilisation": utilisation,
        "dictionary_stats": dict_stats,
        "artifacts": plots,
    }
    (context.artifact_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    final_support = icsc.support_metrics(
        model.solve(mech_batches[0].to(device)), mech_batches[0].to(device)
    )
    metrics = {
        "valid_mae": float(trained["soup_valid_mae"]),
        "valid_soup_mae": float(trained["soup_valid_mae"]),
        "valid_best_mae": float(trained["best_valid_mae"]),
        "best_epoch": int(trained["best_epoch"]),
        "epochs_run": int(trained["epochs_run"]),
        "parameters": int(parameters["total"]),
        "lambda1": float(lambda1),
        "lambda_g": float(lambda_g),
        "runtime_seconds": float(trained["wall_clock_seconds"]),
        "split_sizes": {"train": len(train_samples), "valid": len(valid_samples), "test": None},
        "test_access": context.test_access,
        "whole_active_row_frac": float(final_support["whole_active_row_frac_mean"]),
        "element_in_active_frac": float(final_support["element_nonzero_frac_in_active_rows"]),
        "composition_grad_ratio": float(
            vitality["composition_to_attribute_grad_ratio_mean"]
        ),
    }
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=[
            "artifacts/results.json",
            "artifacts/soup_state.pt",
            *[f"artifacts/{name}" for name in plots],
        ],
    )
