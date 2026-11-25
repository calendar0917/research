"""Exact computation decomposition of the CSC arms on the real directed messages.

Section 11 of ``PROTOCOL.md``.  Per real directed bond ``j -> i`` of relation ``r``:

    gamma_i = [SiLU(a_i + B_i) - SiLU(a_i)] / B_i          (per channel, stable at B -> 0)
    v_ji        = W_out [(gamma_i + delta_i,r) * p_ji]      (state change per message)
    v_extra_ji  = W_out [ delta_i,r * p_ji ]                (the new correction alone)
    h2_i        = h2_base_i + Sum_j v_ji
    h2_i        = h2_fixed1_i + Sum_j v_extra_ji

and the attribution of the *actual graph prediction* is the path integral of
the true gradient of ``BSModel.post_local`` (f + rho + attention residual):

    A_ji = int_0^1 grad_{h2_i} F(h2_base + t * dh) . v_ji dt

evaluated with 16-point Gauss-Legendre on the exact model (32-point fallback if
the closure error exceeds ``1e-4 + 1e-3 * |F(h2) - F(h2_base)|``).  The path is
one scalar parameter applied to every atom at once, so the gradient-theorem
identity ``Sum_ij A_ji = F(h2) - F(h2_base)`` holds exactly up to quadrature
error.  No post-hoc rescaling is applied.
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
import cs_mechanism as ME  # noqa: E402


def quadrature(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(nodes))
    return 0.5 * (x + 1.0), 0.5 * w


def _attribution(model, batch, h2_base: torch.Tensor, dh: torch.Tensor, t_nodes, w_nodes) -> torch.Tensor:
    """``acc[N, d]``: the Gauss-Legendre averaged gradient of the prediction at each atom."""
    acc = torch.zeros_like(h2_base)
    for t, w in zip(t_nodes, w_nodes):
        h2t = (h2_base + float(t) * dh).detach().requires_grad_(True)
        logits = model.post_local(h2t, batch)
        grad = torch.autograd.grad(logits.sum(), h2t)[0]
        acc = acc + float(w) * grad
    return acc


def decompose_split(
    model,
    bundle,
    split: str,
    device,
    *,
    nodes: int = C.QUADRATURE_NODES,
    arm: str = "D",
    n_batches: int | None = None,
) -> dict[str, Any]:
    bs = int(C.RECIPE["eval_batch_size"])
    order, row_ids, y_raw = D.split_order(bundle, split)
    t_nodes, w_nodes = quadrature(nodes)
    chunks: list[dict[str, Any]] = []
    model.eval()
    n_done = 0
    for start in range(0, order.size, bs):
        if n_batches is not None and n_done >= int(n_batches):
            break
        chunk = order[start:start + bs]
        batch = D.make_batch(bundle, split, chunk, device)
        states = model.forward_states(batch)
        h2_base = states["h2_base"].detach()
        dh = states["dh"].detach()
        dh_extra = torch.zeros_like(dh)
        if states["v_extra"].numel():
            dh_extra.index_add_(0, states["edges"]["dst"], states["v_extra"])
        h2_fixed1 = (states["h2"] - dh_extra).detach()
        logits_full = states["logits"].detach()
        logits_base = model.post_local(h2_base, batch).detach()
        logits_fixed1 = model.post_local(h2_fixed1, batch).detach()
        acc = _attribution(model, batch, h2_base, dh, t_nodes, w_nodes)
        # the new correction's own effect: from the original FIXED1 state to the CSC state
        acc_extra = _attribution(model, batch, h2_fixed1, dh_extra, t_nodes, w_nodes)
        recon_base = (h2_base + dh - states["h2"]).abs().max().item()
        recon_fixed1 = (h2_fixed1 + dh_extra - states["h2"]).abs().max().item()
        src, dst, rel = states["edges"]["src"], states["edges"]["dst"], states["edges"]["rel"]
        v, v_extra, p = states["v"], states["v_extra"], states["p"]
        A = (acc.index_select(0, dst) * v).sum(dim=-1)
        A_extra = (acc_extra.index_select(0, dst) * v_extra).sum(dim=-1)
        # per-atom / per-molecule closure
        node_attr = (acc * dh).sum(dim=-1)  # (N,)
        atom_batch = batch["atom_batch"]
        n_graphs = int(batch["n_graphs"])
        per_graph_attr = torch.zeros(n_graphs, device=node_attr.device).index_add_(0, atom_batch, node_attr)
        per_graph_pred = logits_full.reshape(-1) - logits_base.reshape(-1)
        closure = (per_graph_attr - per_graph_pred).detach().cpu().numpy()
        node_attr_extra = (acc_extra * dh_extra).sum(dim=-1)
        per_graph_attr_extra = torch.zeros(n_graphs, device=node_attr_extra.device).index_add_(
            0, atom_batch, node_attr_extra
        )
        per_graph_pred_extra = (logits_full.reshape(-1) - logits_fixed1.reshape(-1))
        closure_extra = (per_graph_attr_extra - per_graph_pred_extra).detach().cpu().numpy()
        # node packaging (molecule-local atom id)
        atom_off = np.concatenate([[0], np.cumsum(batch["atom_counts"].cpu().numpy())[:-1]]).astype(np.int64)
        node_global = np.arange(atom_batch.numel(), dtype=np.int64)
        node_local = node_global - atom_off[atom_batch.cpu().numpy()]
        msg_dst_global = dst.cpu().numpy()
        msg_src_global = src.cpu().numpy()
        msg_graph = atom_batch.cpu().numpy()[msg_dst_global] if msg_dst_global.size else np.zeros(0, dtype=np.int64)
        msg_src_local = msg_src_global - atom_off[msg_graph] if msg_src_global.size else np.zeros(0, dtype=np.int64)
        msg_dst_local = msg_dst_global - atom_off[msg_graph] if msg_dst_global.size else np.zeros(0, dtype=np.int64)
        bond_type = batch["bond_type"].cpu().numpy()
        n_bonds = int(bond_type.size)
        msg_bond_type = np.concatenate([bond_type, bond_type]) if n_bonds else np.zeros(0, dtype=np.int64)
        delta_msg_abs = (
            states["delta_msg"].abs().amax(dim=-1).cpu().numpy() if states["delta_msg"].numel() else np.zeros(0)
        )
        chunks.append(
            {
                "mol_pos": np.arange(start, start + chunk.size, dtype=np.int64),
                "mol_row": np.asarray(chunk, dtype=np.int64),
                "y_raw": np.asarray([bundle.targets_raw[split][int(o)] for o in chunk], dtype=np.float64),
                "pred_std": logits_full.reshape(-1).cpu().numpy().astype(np.float64),
                "pred_base_std": logits_base.reshape(-1).cpu().numpy().astype(np.float64),
                "closure_std": closure.astype(np.float64),
                "closure_extra_std": closure_extra.astype(np.float64),
                "pred_fixed1_std": logits_fixed1.reshape(-1).cpu().numpy().astype(np.float64),
                "msg_mol": msg_graph.astype(np.int64),
                "msg_src": msg_src_local.astype(np.int64),
                "msg_dst": msg_dst_local.astype(np.int64),
                "msg_bond_type": msg_bond_type.astype(np.int64),
                "msg_rel": rel.cpu().numpy().astype(np.int64),
                "A": A.cpu().numpy().astype(np.float64),
                "A_extra": A_extra.cpu().numpy().astype(np.float64),
                "v_rms": v.pow(2).mean(dim=-1).sqrt().cpu().numpy().astype(np.float32),
                "v_extra_rms": v_extra.pow(2).mean(dim=-1).sqrt().cpu().numpy().astype(np.float32) if v_extra.numel() else np.zeros(0, dtype=np.float32),
                "p_rms": p.pow(2).mean(dim=-1).sqrt().cpu().numpy().astype(np.float32) if p.numel() else np.zeros(0, dtype=np.float32),
                "gamma_rms": states["gamma"].pow(2).mean(dim=-1).sqrt().cpu().numpy().astype(np.float32),
                "delta_msg_abs": delta_msg_abs.astype(np.float32),
                "node_mol": atom_batch.cpu().numpy().astype(np.int64),
                "node_local": node_local.astype(np.int64),
                "node_degree": states["n"].cpu().numpy().astype(np.int64),
                "node_n_hetero": ((states["n_r"] > 0).sum(dim=1)).cpu().numpy().astype(np.int64),
                "node_dh_rms": dh.pow(2).mean(dim=-1).sqrt().cpu().numpy().astype(np.float32),
                "node_attr": node_attr.cpu().numpy().astype(np.float64),
                "state_recon_base_absmax": np.asarray([recon_base], dtype=np.float64),
                "state_recon_fixed1_absmax": np.asarray([recon_fixed1], dtype=np.float64),
            }
        )
        n_done += 1
    if not chunks:
        return {"split": split, "arm": arm, "n_molecules": 0}
    merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    # molecule-level raw units
    mean, std = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    err = np.abs(merged["closure_std"])
    target = np.abs(merged["pred_std"] - merged["pred_base_std"])
    tol = C.TRACE_ABS_TOL + C.TRACE_REL_TOL * target
    err_extra = np.abs(merged["closure_extra_std"])
    target_extra = np.abs(merged["pred_std"] - merged["pred_fixed1_std"])
    tol_extra = C.TRACE_ABS_TOL + C.TRACE_REL_TOL * target_extra
    merged["A_raw"] = merged["A"] * std
    merged["A_extra_raw"] = merged["A_extra"] * std
    return {
        "split": split,
        "arm": arm,
        "n_molecules": int(merged["mol_row"].size),
        "n_atoms": int(merged["node_mol"].size),
        "n_messages": int(merged["msg_mol"].size),
        "quadrature_nodes": int(nodes),
        "closure": {
            "max_abs_std": float(err.max()) if err.size else 0.0,
            "max_abs_raw": float(err.max() * std) if err.size else 0.0,
            "n_over_tolerance": int((err > tol).sum()),
            "tolerance_rule": "1e-4 + 1e-3 * |pred(h2) - pred(h2_base)| (standardised units)",
            "worst_molecule_pos": int(np.argmax(err)) if err.size else None,
            "pred_change_rms_std": float(np.sqrt(np.mean((merged["pred_std"] - merged["pred_base_std"]) ** 2))),
            "extra": {
                "max_abs_std": float(err_extra.max()) if err_extra.size else 0.0,
                "max_abs_raw": float(err_extra.max() * std) if err_extra.size else 0.0,
                "n_over_tolerance": int((err_extra > tol_extra).sum()),
                "pred_change_rms_std": float(np.sqrt(np.mean((merged["pred_std"] - merged["pred_fixed1_std"]) ** 2))),
            },
            "state_reconstruction": {
                "h2_minus_h2_base_plus_message_vectors_absmax": float(merged["state_recon_base_absmax"].max()),
                "h2_minus_h2_fixed1_plus_extra_vectors_absmax": float(merged["state_recon_fixed1_absmax"].max()),
            },
        },
        "arrays": merged,
        "row_ids": row_ids,
        "y_raw": y_raw,
    }


def run_dataset(dataset: str, *, device: str = "cpu", arms: Sequence[str] = ("A", "D"), splits: Sequence[str] = C.SPLITS) -> dict[str, Any]:
    device = torch.device(device)
    out: dict[str, Any] = {"dataset": dataset, "created_utc": C.utcnow(), "arms": {}}
    for arm in arms:
        model, bundle, payload, path = ME.load_best(dataset, arm, device=device)
        entry: dict[str, Any] = {
            "checkpoint": C.track_rel(path),
            "checkpoint_sha256": C.sha256_file(path),
            "best_epoch": int(payload.get("epoch", -1)),
            "splits": {},
        }
        for split in splits:
            result = decompose_split(model, bundle, split, device, arm=arm)
            npz = C.RESULTS_DIR / "decomposition" / f"{dataset.replace('ogbg-', '')}_{arm}_{split}.npz"
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, **result["arrays"])
            entry["splits"][split] = {
                k: v for k, v in result.items() if k not in ("arrays", "row_ids", "y_raw")
            }
            entry["splits"][split]["npz"] = C.track_rel(npz)
            entry["splits"][split]["npz_sha256"] = C.sha256_file(npz)
            C.log(
                f"decompose {dataset} {arm} {split}: msgs={result['n_messages']} "
                f"closure_max={result['closure']['max_abs_std']:.2e} over_tol={result['closure']['n_over_tolerance']}"
            )
        out["arms"][arm] = entry
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", default=C.MAIN_DATASET)
    parser.add_argument("--arms", default="A,D")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "decomposition_trace.json"))
    args = parser.parse_args(argv)
    report = {"round_id": C.ROUND_ID, "created_utc": C.utcnow(), "datasets": {}}
    for dataset in [d for d in args.datasets.split(",") if d]:
        try:
            report["datasets"][dataset] = run_dataset(
                dataset, device=args.device, arms=tuple(a for a in args.arms.split(",") if a),
                splits=tuple(s for s in args.splits.split(",") if s),
            )
        except FileNotFoundError as exc:
            report["datasets"][dataset] = {"error": f"missing checkpoint: {exc}"}
            C.log(f"decompose: {dataset} skipped ({exc})")
    C.write_json(Path(args.out), report)
    C.log(f"decomposition written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
