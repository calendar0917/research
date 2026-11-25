"""Mechanism diagnostics and the three frozen interventions (PROTOCOL.md section 10).

For the best-Valid checkpoints of the Gate arms (C/D):

* Gate output mean / RMS / std, per-relation modulation differences,
* ``Q_i`` RMS and the state change it projects to through ``fc2``,
* heterogeneous-relation node and molecule coverage,
* the fraction of (node, relation) groups where ``P_i,r`` actually differs from
  ``P_null_i,r``,

and, for D, three frozen inference interventions with every parameter and the
whole original FIXED1 body unchanged:

  I0  Q_i = 0
  I1  P_i,r -> P_null_i,r  (mixed grouping; environment and Gate outputs unchanged)
  I2  delta_i,r -> delta_bar_r  (Train relation-occurrence weighted Gate mean)

The interventions answer "does the trained model use this computation", which is
a different question from "does training with it generalise better".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cs_common as C

C.setup_paths()

import torch  # noqa: E402

import cs_data as D  # noqa: E402
import cs_model as M  # noqa: E402


def load_best(dataset: str, arm: str, *, device):
    path = C.RUNS_DIR / f"{dataset.replace('ogbg-', '')}_{arm}_seed{C.SEED}" / "checkpoint_best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bundle = D.load_bundle(dataset)
    model, _audit = M.build_csc(bundle.spec, bundle.norm, arm=arm, seed=C.SEED, device=device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, bundle, payload, path


def _metrics(rep: Mapping[str, Any]) -> dict[str, float]:
    return {"rmse_raw": float(rep["rmse_raw"]), "mae_raw": float(rep["mae_raw"]), "n_rows": int(rep["n_rows"])}


def _pred_change(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "abs_max": float(np.abs(diff).max()),
        "sign_change_fraction": float(np.mean(np.sign(a) != np.sign(b))),
    }


def gate_coverage(model, bundle, device, *, split: str = "valid") -> dict[str, Any]:
    """Node/molecule coverage of heterogeneous relation neighbourhoods."""
    order, _ids, _y = D.split_order(bundle, split)
    bs = int(C.RECIPE["eval_batch_size"])
    n_atoms = 0
    n_hetero_atoms = 0
    n_graphs = 0
    n_hetero_graphs = 0
    n_groups = 0
    n_group_gap = 0
    gap_values: list[np.ndarray] = []
    delta_values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, order.size, bs):
            chunk = order[start:start + bs]
            batch = D.make_batch(bundle, split, chunk, device)
            states = model.forward_states(batch)
            n_r = states["n_r"].cpu().numpy()
            present = n_r > 0
            hetero = present.sum(axis=1) >= 2
            n_atoms += int(present.shape[0])
            n_hetero_atoms += int(hetero.sum())
            atom_batch = batch["atom_batch"].cpu().numpy()
            per_graph = np.zeros(int(batch["n_graphs"]), dtype=bool)
            np.logical_or.at(per_graph, atom_batch, hetero)
            n_graphs += int(per_graph.size)
            n_hetero_graphs += int(per_graph.sum())
            delta = states["delta"].cpu().numpy()
            P = states["P"].cpu().numpy()
            B = states["B"].cpu().numpy()
            denom = np.maximum(n_r.sum(axis=1, keepdims=True), 1.0)
            P_null = (n_r / denom)[:, :, None] * B[:, None, :]
            gap = np.abs(P - P_null).max(axis=-1)
            n_groups += int(present.sum())
            n_group_gap += int((gap[present] > 1e-6).sum())
            gap_values.append(gap[present])
            delta_values.append(delta[present])
    gaps = np.concatenate(gap_values) if gap_values else np.zeros(0)
    deltas = np.concatenate(delta_values) if delta_values else np.zeros(0)
    return {
        "split": split,
        "n_atoms": int(n_atoms),
        "n_hetero_atoms": int(n_hetero_atoms),
        "hetero_atom_fraction": float(n_hetero_atoms / n_atoms) if n_atoms else 0.0,
        "n_molecules": int(n_graphs),
        "n_hetero_molecules": int(n_hetero_graphs),
        "hetero_molecule_fraction": float(n_hetero_graphs / n_graphs) if n_graphs else 0.0,
        "n_relation_groups_present": int(n_groups),
        "n_groups_with_real_vs_mixed_gap": int(n_group_gap),
        "grouping_gap_fraction": float(n_group_gap / n_groups) if n_groups else 0.0,
        "grouping_gap_abs_mean": float(gaps.mean()) if gaps.size else 0.0,
        "grouping_gap_abs_p95": float(np.percentile(gaps, 95)) if gaps.size else 0.0,
        "delta_abs_mean": float(np.abs(deltas).mean()) if deltas.size else 0.0,
        "delta_rms": float(np.sqrt(np.mean(deltas**2))) if deltas.size else 0.0,
        "delta_abs_max": float(np.abs(deltas).max()) if deltas.size else 0.0,
    }


def diagnostics(dataset: str, *, device: str = "cpu") -> dict[str, Any]:
    device = torch.device(device)
    out: dict[str, Any] = {"dataset": dataset, "created_utc": C.utcnow(), "arms": {}}
    for arm in C.STAT_ARMS:
        try:
            model, bundle, payload, path = load_best(dataset, arm, device=device)
        except FileNotFoundError as exc:
            out["arms"][arm] = {"error": f"missing checkpoint: {exc}"}
            C.log(f"mechanism: {dataset} {arm} skipped ({exc})")
            continue
        diag_rows = D.split_order(bundle, "valid")[0][:256]
        diag_batch = D.make_batch(bundle, "valid", diag_rows, device)
        stats = M.gate_statistics(model, diag_batch)
        bar = M.train_relation_delta_bar(model, bundle, device)
        entry = {
            "checkpoint": C.track_rel(path),
            "checkpoint_sha256": C.sha256_file(path),
            "best_epoch": int(payload.get("epoch", -1)),
            "valid_rmse_raw_at_selection": float(payload.get("valid_rmse_raw", float("nan"))),
            "gate_statistics_diagnostic_batch": stats,
            "coverage": {split: gate_coverage(model, bundle, device, split=split) for split in ("valid", "test")},
            "delta_bar_train": {
                "per_relation": {
                    name: {
                        "delta_bar_abs_mean": float(bar["delta_bar"][r].abs().mean().item()),
                        "delta_bar_rms": float(bar["delta_bar"][r].pow(2).mean().sqrt().item()),
                        "n_edge_occurrences": float(bar["counts"][r].item()),
                    }
                    for r, name in enumerate(C.RELATION_NAMES)
                },
                "method": bar["method"],
            },
        }
        if arm == "D":
            entry["interventions"] = interventions(model, bundle, device, bar["delta_bar"])
        out["arms"][arm] = entry
    return out


def interventions(model, bundle, device, delta_bar: torch.Tensor) -> dict[str, Any]:
    """I0 / I1 / I2 on the same D checkpoint, complete Valid/Test scoring."""
    out: dict[str, Any] = {}
    base = {split: D.evaluate(bundle, model, split, device) for split in ("valid", "test")}
    variants = {
        "I0_no_correction": {"zero_correction": True},
        "I1_mixed_grouping": {"content": "mixed"},
        "I2_frozen_delta_bar": {"delta_mode": "bar", "delta_bar": delta_bar},
    }
    for name, kwargs in variants.items():
        entry: dict[str, Any] = {"kwargs": {k: (None if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}}
        for split in ("valid", "test"):
            preds = []
            bs = int(C.RECIPE["eval_batch_size"])
            order, _ids, y_raw = D.split_order(bundle, split)
            model.eval()
            with torch.no_grad():
                for start in range(0, order.size, bs):
                    chunk = order[start:start + bs]
                    batch = D.make_batch(bundle, split, chunk, device)
                    preds.append(model(batch, **kwargs).detach().cpu().numpy().reshape(-1))
            pred_std = np.concatenate(preds) if preds else np.zeros(0)
            pred_raw = pred_std * bundle.scaler["std"] + bundle.scaler["mean"]
            entry[split] = {
                "rmse_raw": C.rmse(pred_raw, y_raw),
                "mae_raw": C.mae(pred_raw, y_raw),
                "delta_vs_full": _pred_change(pred_raw, base[split]["pred_raw"]),
                "pred_raw_sha256": C.sha256_array(pred_raw.astype(np.float32)),
            }
        out[name] = entry
    out["baseline"] = {
        split: {"rmse_raw": base[split]["rmse_raw"], "mae_raw": base[split]["mae_raw"], **_metrics(base[split])}
        for split in ("valid", "test")
    }
    out["note"] = (
        "frozen-weight interventions: a drop shows the trained model depends on the computation. "
        "It is not evidence that training with the computation generalises better."
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "mechanism_diagnostics.json"))
    args = parser.parse_args(argv)
    report = {"round_id": C.ROUND_ID, "created_utc": C.utcnow(), "datasets": {}}
    for dataset in C.DATASETS:
        try:
            report["datasets"][dataset] = diagnostics(dataset, device=args.device)
        except FileNotFoundError as exc:
            report["datasets"][dataset] = {"error": f"missing checkpoint: {exc}"}
            C.log(f"mechanism: {dataset} skipped ({exc})")
    C.write_json(Path(args.out), report)
    C.log(f"mechanism diagnostics written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
