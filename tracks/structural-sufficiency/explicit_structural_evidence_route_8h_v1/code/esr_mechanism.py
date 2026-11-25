"""Phase E/F: exact prediction accounting, interventions, structure rules, cases.

Everything runs on *frozen* checkpoints (``checkpoint_best.pt``) and never
updates a weight.  The diagnostic forward path asserts that it reproduces
``model.forward`` exactly, so an accounting identity is only reported when the
same computation that produced the prediction is the one being decomposed.

    python esr_mechanism.py --runs runs --analysis analysis --device cuda

Writes ``decomposition_trace.json``, ``mechanism_diagnostics.json``,
``structure_rules.json`` and ``explanation_cases.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import esr_common as C
import esr_data as D
import esr_model as M

#: frozen atom vocabulary of the inherited caches (feature_spec._LIPO_ATOM_NAMES)
ATOM_NAMES = (
    "B", "Br", "C", "C_aromatic", "Cl", "F", "I", "N", "N_aromatic", "O",
    "O_aromatic", "P", "S", "S_aromatic", "Si", "UNK",
)
RECON_TOL = 2e-5
COHORT_PER_BAND = 8
TRACE_MOLECULES = 6
MAX_CASES = 8
PCBA_SUBSET = 4096


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row_primary(view: D.View, metrics: Mapping[str, Any]) -> float:
    return float(view.primary(metrics))


def _view_run(run_dir: Path) -> tuple[dict[str, Any], D.View, torch.nn.Module]:
    run = json.loads((run_dir / "run.json").read_text())
    view = D.load_view(run["dataset"], max_dist=int(run.get("max_dist", 2)), workers=4)
    view = D.view_with_pairs(
        view,
        relation_mode=str(run.get("relation_mode", "real")),
        max_dist=int(run.get("max_dist", 2)),
        workers=4,
    )
    model = M.build_model(
        arm=run["arm"],
        view=view,
        seed=int(run["seed"]),
        device="cpu",
        hidden_width=(run.get("model_info") or {}).get("hidden_width"),
    )[0]
    payload = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return run, view, model


def _split_rows(view: D.View, split: str, limit: int | None = None) -> np.ndarray:
    n = int(view.rows[split].size)
    idx = np.arange(n, dtype=np.int64)
    if limit is not None and n > limit:
        rng = np.random.default_rng(20261125)
        idx = np.sort(rng.choice(n, size=limit, replace=False))
    return idx


def _collate_rows(view: D.View, split: str, positions: Sequence[int]) -> dict[str, Any]:
    cache_rows = [int(view.rows[split][p]) for p in positions]
    return view.collate(split, cache_rows)


# ---------------------------------------------------------------------------
# exact accounting
# ---------------------------------------------------------------------------


def _bias_sum(*linears: Any) -> torch.Tensor:
    """Sum of the readout biases as a ``[1, T]`` row (zeros when bias=False)."""
    total = None
    for lin in linears:
        if lin is None:
            continue
        term = lin.bias.unsqueeze(0) if lin.bias is not None else lin.weight.new_zeros((1, lin.weight.shape[0]))
        total = term if total is None else total + term
    assert total is not None
    return total


def accounting(model: M.SRENNet, batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per-part contributions whose sum is the logit vector (checked by caller)."""
    with torch.no_grad():
        h0, u = model.encode_tokens(batch)
        c = model.global_linear(batch["global_ext"])
        atom_seg, atom_counts = batch["atom_batch"], batch["atom_counts"]
        pair_seg, pair_counts = batch["pair_batch"], batch["pair_counts"]
        out: dict[str, torch.Tensor] = {"global": c}
        if model.variant in (C.ADD, C.EVID):
            v = model.relation_tokens(u, batch)
            a_scores, b_scores = model.q_u(u), model.q_v(v)
            if model.variant == C.ADD:
                out["local"] = M.segment_sum(a_scores, atom_seg, int(atom_counts.numel())) / torch.sqrt(
                    atom_counts.clamp(min=1).to(a_scores.dtype)
                ).unsqueeze(1)
                out["relation"] = M.segment_sum(b_scores, pair_seg, int(pair_counts.numel())) / torch.sqrt(
                    pair_counts.clamp(min=1).to(b_scores.dtype)
                ).unsqueeze(1)
                out["token_local"] = a_scores / torch.sqrt(atom_counts.clamp(min=1).to(a_scores.dtype)).repeat_interleave(
                    atom_counts
                ).unsqueeze(1)
                out["token_relation"] = b_scores / torch.sqrt(pair_counts.clamp(min=1).to(b_scores.dtype)).repeat_interleave(
                    pair_counts
                ).unsqueeze(1)
            else:
                du = M.evidence_decompose(u, atom_seg, atom_counts, model.theta_u)
                dv = M.evidence_decompose(v, pair_seg, pair_counts, model.theta_v)
                for part in ("mean", "pos", "neg"):
                    out[f"local_{part}"] = du[part]
                    out[f"relation_{part}"] = dv[part]
                out["local"] = du["mean"] + du["pos"] - du["neg"]
                out["relation"] = dv["mean"] + dv["pos"] - dv["neg"]
                out["theta_local"] = torch.softmax(model.theta_u, dim=-1).mean(dim=0)
                out["theta_relation"] = torch.softmax(model.theta_v, dim=-1).mean(dim=0)
        elif model.variant in (C.BANK, C.UNARY):
            v = model.self_tokens(u) if model.variant == C.UNARY else model.relation_tokens(u, batch)
            u_bank = M.evidence_aggregate(u, atom_seg, atom_counts, model.theta_u)
            v_bank = M.evidence_aggregate(
                v,
                atom_seg if model.variant == C.UNARY else pair_seg,
                atom_counts if model.variant == C.UNARY else pair_counts,
                model.theta_v,
            )
            # q(u_bank) = u_bank @ W^T + b  ->  per-channel term u_bank[:, k] * W[t, k]
            out["channel_local"] = u_bank.unsqueeze(1) * model.q_u.weight.unsqueeze(0)   # [n, T, K]
            out["channel_relation"] = v_bank.unsqueeze(1) * model.q_v.weight.unsqueeze(0)
            out["readout_bias"] = _bias_sum(model.q_u, model.q_v)
            out["local"] = model.q_u(u_bank)
            out["relation"] = model.q_v(v_bank)
        elif model.variant == C.HIDDEN:
            out["note"] = torch.zeros(1)
        elif model.variant == C.COMPOUND:
            v = model.relation_tokens(u, batch)
            pu, pv = batch["pair_u"], batch["pair_v"]
            n_atoms = int(u.shape[0])
            deg = u.new_zeros(n_atoms)
            acc = u.new_zeros((n_atoms, model.K))
            if pu.numel():
                deg = deg.index_add_(0, pu, torch.ones_like(pu, dtype=u.dtype)).index_add_(
                    0, pv, torch.ones_like(pu, dtype=u.dtype)
                )
                acc = acc.index_add_(0, pu, v).index_add_(0, pv, v)
            c_i = acc / torch.sqrt(deg.clamp(min=1.0)).unsqueeze(1)
            w = model.compound_norm(u + model.compound(u, c_i, torch.log1p(deg)))
            w_bank = M.evidence_aggregate(w, atom_seg, atom_counts, model.theta_w)
            out["channel_relation"] = w_bank.unsqueeze(1) * model.q_w.weight.unsqueeze(0)
            out["readout_bias"] = _bias_sum(model.q_w)
            out["relation"] = model.q_w(w_bank)
            out["local"] = torch.zeros_like(out["relation"])
        elif model.variant == C.FLEX:
            v = model.relation_tokens(u, batch)
            du = M.evidence_decompose(u, atom_seg, atom_counts, model.theta_u)
            dv = M.evidence_decompose(v, pair_seg, pair_counts, model.theta_v)
            for stream, dec in (("u", du), ("v", dv)):
                for part in ("mean", "pos", "neg"):
                    out[f"{stream}_{part}"] = model.flex[f"{stream}_{part}"](dec[part])
            out["local"] = out["u_mean"] + out["u_pos"] + out["u_neg"]
            out["relation"] = out["v_mean"] + out["v_pos"] + out["v_neg"]
        else:
            raise KeyError(model.variant)
    return {k: v.detach().cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}


def _reconstruct(model: M.SRENNet, parts: Mapping[str, np.ndarray]) -> np.ndarray:
    if model.variant in (C.ADD, C.EVID):
        return parts["global"] + parts["local"] + parts["relation"]
    if model.variant in (C.BANK, C.UNARY):
        return (
            parts["global"]
            + parts["channel_local"].sum(axis=2)
            + parts["channel_relation"].sum(axis=2)
            + parts["readout_bias"]
        )
    if model.variant == C.COMPOUND:
        return parts["global"] + parts["channel_relation"].sum(axis=2) + parts["readout_bias"]
    if model.variant == C.FLEX:
        return parts["global"] + parts["local"] + parts["relation"]
    raise KeyError(model.variant)


# ---------------------------------------------------------------------------
# interventions
# ---------------------------------------------------------------------------


def _metrics_under_ablation(
    model: M.SRENNet,
    view: D.View,
    device: torch.device,
    *,
    split: str,
    limit: int | None,
    ablations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    positions = _split_rows(view, split, limit)
    base_metrics, _ = D.evaluate(model, view, split, device)
    out: dict[str, Any] = {
        "n_molecules": int(positions.size),
        "split": split,
        "full_primary": _row_primary(view, base_metrics),
        "ablations": {},
    }
    for name, ablate in ablations.items():
        logits = []
        model.eval()
        for start in range(0, positions.size, 256):
            chunk = positions[start : start + 256]
            batch = view.to_device(_collate_rows(view, split, chunk), device)
            with torch.no_grad():
                logits.append(model(batch, ablate).detach().cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        subset = view.y[split]
        if limit is not None:
            subset = subset[positions]
        metrics = view.metrics(split, logits) if limit is None else _subset_metrics(view, subset, logits)
        out["ablations"][name] = {
            "primary": _row_primary(view, metrics),
            "delta": _row_primary(view, metrics) - out["full_primary"],
            "metrics": {k: v for k, v in metrics.items() if k in ("rmse", "mae", "ap", "macro_roc_auc", "masked_bce")},
        }
    return out


def _subset_metrics(view: D.View, y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    from mb2_metrics import task_metrics

    if view.task == "regression":
        return task_metrics("regression", logits.reshape(-1), y.reshape(-1), view.target_norm)
    if view.task == "multitask":
        return task_metrics("multitask", logits, y, None, view.task_names)
    return D.ap_report(y, logits)


# ---------------------------------------------------------------------------
# motif / relation signatures
# ---------------------------------------------------------------------------


def _wl_signatures(cache: Mapping[str, Any], positions: Sequence[int]) -> list[dict[str, Any]]:
    """WL-2 style rooted signatures built only from the frozen cache arrays."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    out = []
    for p in positions:
        a0, a1 = int(atom_ptr[p]), int(atom_ptr[p + 1])
        b0, b1 = int(bond_ptr[p]), int(bond_ptr[p + 1])
        n = a1 - a0
        types = atom_type[a0:a1]
        bonds = bond_index[b0:b1]
        bt = bond_type[b0:b1]
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for k in range(bonds.shape[0]):
            i, j = int(bonds[k, 0]), int(bonds[k, 1])
            adj[i].append((j, int(bt[k])))
            adj[j].append((i, int(bt[k])))
        wl1 = [hashlib.sha1(f"{int(types[i])}|{len(adj[i])}".encode()).hexdigest()[:12] for i in range(n)]
        wl2 = [
            hashlib.sha1(
                (wl1[i] + "|" + "".join(sorted(f"{wl1[j]}:{b}" for j, b in adj[i]))).encode()
            ).hexdigest()[:16]
            for i in range(n)
        ]
        ring_proxy = hashlib.sha1(
            ("|".join(sorted(wl2)) + f"|{n}").encode()
        ).hexdigest()[:16]
        out.append(
            {
                "position": int(p),
                "n_atoms": int(n),
                "wl1": wl1,
                "wl2": wl2,
                "element": [ATOM_NAMES[int(t)] if int(t) < len(ATOM_NAMES) else "UNK" for t in types],
                "degree": [len(a) for a in adj],
                "scaffold_proxy": ring_proxy,
                "wl2_multiset": sorted(wl2),
            }
        )
    return out


def _relation_bucket(rel: np.ndarray) -> str:
    dist = int(np.argmax(rel[:5]))
    common = float(np.expm1(rel[5]))
    return f"d{dist}_common{min(int(round(common)), 3)}"


# ---------------------------------------------------------------------------
# per-dataset analysis
# ---------------------------------------------------------------------------


def analyse_run(run_dir: Path, device: torch.device, *, analysis: Path) -> dict[str, Any]:
    run, view, model = _view_run(run_dir)
    model.to(device)
    dataset, arm = run["dataset"], run["arm"]
    limit = PCBA_SUBSET if dataset == C.PCBA else None
    metrics, logits = D.evaluate(model, view, "valid", device)
    positions = _split_rows(view, "valid", limit)
    y = view.y["valid"] if limit is None else view.y["valid"][positions]
    if limit is not None:
        logits = logits[positions]
    primary = _row_primary(view, metrics)
    entry: dict[str, Any] = {
        "dataset": dataset,
        "arm": arm,
        "run": run_dir.name,
        "parameters": int(run.get("parameters") or 0),
        "valid_primary": primary,
        "cohort": {"n_molecules": int(positions.size), "limit": limit},
    }
    if arm == C.BASE:
        entry["accounting"] = {"skipped": "BASE has no explicit object set; accounting is undefined by construction"}
        entry["accounting_exact"] = None
        entry["interventions"] = {"skipped": "BASE is the reference; stream ablations are defined on the explicit arms"}
        return entry
    # ---- exact accounting on a small cohort -----------------------------
    if dataset in (C.LIPO, C.ESOL):
        err = np.abs(logits.reshape(-1) * float(view.target_norm["scale"]) + float(view.target_norm["mean"]) - y.reshape(-1))
    else:
        err = np.abs(y - np.round(1.0 / (1.0 + np.exp(-logits))))
    order = np.argsort(err)
    picks = (
        list(order[-COHORT_PER_BAND:])
        + list(order[len(order) // 2 - COHORT_PER_BAND // 2 : len(order) // 2 + COHORT_PER_BAND // 2])
        + list(order[:COHORT_PER_BAND])
    )
    picks = sorted({int(p) for p in picks})
    trace = {"rows": [], "max_abs_reconstruction_error": 0.0, "tolerance": RECON_TOL}
    rows_for_trace = picks[:TRACE_MOLECULES]
    for pos in rows_for_trace:
        batch = view.to_device(_collate_rows(view, "valid", [pos]), device)
        with torch.no_grad():
            direct = model(batch).detach().cpu().numpy()
        if model.variant == C.HIDDEN:
            trace["rows"].append({"position": pos, "note": "no exact accounting by construction (nonlinear readout)"})
            continue
        parts = accounting(model, batch)
        rebuilt = _reconstruct(model, parts)
        trace["max_abs_reconstruction_error"] = max(
            trace["max_abs_reconstruction_error"], float(np.abs(rebuilt - direct).max())
        )
        row = {
            "position": int(pos),
            "logit_reconstructed": float(rebuilt.reshape(-1)[0]) if rebuilt.size == 1 else None,
            "global_term": parts["global"].reshape(-1).tolist(),
            "local_part": parts["local"].reshape(-1).tolist(),
            "relation_part": parts["relation"].reshape(-1).tolist(),
            "share_local": float(np.abs(parts["local"]).sum() / max(np.abs(parts["local"]).sum() + np.abs(parts["relation"]).sum(), 1e-12)),
        }
        if model.variant == C.EVID:
            row["streams"] = {
                part: {
                    "local": parts[f"local_{part}"].reshape(-1).tolist(),
                    "relation": parts[f"relation_{part}"].reshape(-1).tolist(),
                }
                for part in ("mean", "pos", "neg")
            }
            row["theta"] = {
                "local": parts["theta_local"].tolist(),
                "relation": parts["theta_relation"].tolist(),
                "note": "softmax(theta) averaged over tasks; LME is not a positive contribution",
            }
        if "token_local" in parts:
            tl = parts["token_local"]
            tr = parts["token_relation"]
            row["top_local_tokens"] = (
                np.argsort(-np.abs(tl).max(axis=1))[:5].tolist() if tl.size else []
            )
            row["top_relation_tokens"] = (
                np.argsort(-np.abs(tr).max(axis=1))[:5].tolist() if tr.size else []
            )
            row["token_local_abs_share"] = float(
                np.abs(tl).sum() / max(np.abs(tl).sum() + np.abs(tr).sum(), 1e-12)
            )
        trace["rows"].append(row)
    entry["accounting"] = trace
    entry["accounting_exact"] = bool(trace["max_abs_reconstruction_error"] <= RECON_TOL) and model.variant != C.HIDDEN

    # ---- interventions ---------------------------------------------------
    ablations = {
        "zero_relation_tokens": {"zero_v": True},
        "zero_local_tokens": {"zero_u": True},
        "zero_h0": {"zero_h0": True},
        "zero_global_descriptors": {"zero_global": True},
    }
    entry["interventions"] = _metrics_under_ablation(
        model, view, device, split="valid", limit=limit, ablations=ablations
    )
    # object-level zeroing: drop each atom's local token and its incident pairs
    if model.variant in (C.ADD, C.EVID, C.BANK, C.HIDDEN, C.COMPOUND, C.FLEX) and positions.size:
        sample = positions[: min(64, positions.size)]
        batch = view.to_device(_collate_rows(view, "valid", sample), device)
        atom_counts = batch["atom_counts"].detach().cpu().numpy()
        pair_counts = batch["pair_counts"].detach().cpu().numpy()
        atom_off = np.concatenate([[0], np.cumsum(atom_counts)])
        pair_off = np.concatenate([[0], np.cumsum(pair_counts)])
        deltas = []
        with torch.no_grad():
            base = model(batch).detach().cpu().numpy()
            for g in range(min(8, sample.size)):
                if atom_counts[g] == 0:
                    continue
                per_atom = []
                for i in range(atom_counts[g]):
                    atom_keep = torch.ones(int(atom_counts.sum()), device=device)
                    atom_keep[atom_off[g] + i] = 0.0
                    pair_keep = torch.ones(int(pair_counts.sum()), device=device)
                    pu = batch["pair_u"].detach().cpu().numpy()
                    if pu.size:
                        sel = (pu >= atom_off[g] + i) & (pu < atom_off[g] + i + 1)
                        rel = np.where(sel)[0]
                        if rel.size:
                            pair_keep[rel] = 0.0
                    out = model(batch, {"atom_keep": atom_keep, "pair_keep": pair_keep}).detach().cpu().numpy()
                    per_atom.append(float(np.abs(out[g] - base[g]).max()))
                deltas.append({"molecule": int(sample[g]), "n_atoms": int(atom_counts[g]), "per_atom_max_abs_delta": per_atom})
        entry["object_ablation"] = {
            "n_molecules": len(deltas),
            "mean_max_abs_delta": float(np.mean([max(d["per_atom_max_abs_delta"]) for d in deltas])) if deltas else None,
            "examples": deltas[:4],
        }
    return entry


# ---------------------------------------------------------------------------
# structure rules + cases
# ---------------------------------------------------------------------------


def structure_rules(run_dir: Path, device: torch.device, *, max_molecules: int | None) -> dict[str, Any]:
    run, view, model = _view_run(run_dir)
    model.to(device)
    dataset, arm = run["dataset"], run["arm"]
    if arm == C.HIDDEN:
        return {"dataset": dataset, "arm": arm, "skipped": "no exact per-token accounting for HIDDEN"}
    positions = _split_rows(view, "valid", max_molecules)
    motifs: dict[str, dict[str, Any]] = {}
    relations: dict[str, dict[str, Any]] = {}
    model.eval()
    for start in range(0, positions.size, 128):
        chunk = positions[start : start + 128]
        batch = view.to_device(_collate_rows(view, "valid", chunk), device)
        if model.variant not in (C.ADD, C.EVID):
            break
        parts = accounting(model, batch)
        tl, tr = parts.get("token_local"), parts.get("token_relation")
        if tl is None:
            return {"dataset": dataset, "arm": arm, "skipped": "no per-token contributions for this arm"}
        atom_batch = batch["atom_batch"].detach().cpu().numpy()
        pair_batch = batch["pair_batch"].detach().cpu().numpy()
        pair_r = batch["pair_r"].detach().cpu().numpy()
        atom_counts = batch["atom_counts"].detach().cpu().numpy()
        atom_off = np.concatenate([[0], np.cumsum(atom_counts)])
        local_sig = _wl_signatures(
            view.caches["valid"],
            [int(view.rows["valid"][p]) for p in chunk],
        )
        rows_local = np.asarray([int(view.rows["valid"][p]) for p in chunk], dtype=np.int64)
        wl2_all = []
        for sig in local_sig:
            wl2_all.append(sig["wl2"])
        # per-molecule aggregation of token contributions onto the graph
        for gi, pos in enumerate(chunk):
            sel_atoms = np.where(atom_batch == gi)[0]
            if sel_atoms.size == 0:
                continue
            signed = tl[sel_atoms].sum(axis=1)
            rel_signed = np.zeros(sel_atoms.size)
            sel_pairs = np.where(pair_batch == gi)[0]
            if sel_pairs.size:
                pu = batch["pair_u"].detach().cpu().numpy()[sel_pairs] - atom_off[gi]
                pv = batch["pair_v"].detach().cpu().numpy()[sel_pairs] - atom_off[gi]
                vals = tr[sel_pairs].sum(axis=1)
                np.add.at(rel_signed, pu, vals)
                np.add.at(rel_signed, pv, vals)
            total = signed + rel_signed
            k = int(np.argmax(np.abs(total)))
            sig = wl2_all[gi][k]
            rec = motifs.setdefault(
                sig,
                {
                    "wl2": sig,
                    "element": local_sig[gi]["element"][k],
                    "degree": int(local_sig[gi]["degree"][k]),
                    "n_molecules": set(),
                    "scaffolds": set(),
                    "signed_sum": 0.0,
                    "n_positive": 0,
                    "n_negative": 0,
                    "examples": [],
                },
            )
            rec["n_molecules"].add(int(pos))
            rec["scaffolds"].add(local_sig[gi]["scaffold_proxy"])
            rec["signed_sum"] += float(total[k])
            rec["n_positive"] += int(total[k] > 0)
            rec["n_negative"] += int(total[k] < 0)
            if len(rec["examples"]) < 3:
                rec["examples"].append({"valid_position": int(pos), "atom_index": int(k), "contribution": float(total[k])})
            if sel_pairs.size:
                j = int(np.argmax(np.abs(tr[sel_pairs].sum(axis=1))))
                bucket = _relation_bucket(pair_r[sel_pairs][j])
                rrec = relations.setdefault(
                    bucket,
                    {"bucket": bucket, "n_molecules": set(), "signed_sum": 0.0, "n_positive": 0, "n_negative": 0},
                )
                rrec["n_molecules"].add(int(pos))
                rrec["signed_sum"] += float(tr[sel_pairs][j].sum())
                rrec["n_positive"] += int(tr[sel_pairs][j].sum() > 0)
                rrec["n_negative"] += int(tr[sel_pairs][j].sum() < 0)
    # Train support for the same signatures (cross-molecule reuse)
    train_sig = _wl_signatures(
        view.caches["train"], [int(x) for x in np.asarray(view.rows["train"])[: min(20000, view.rows["train"].size)]]
    )
    train_counts: dict[str, int] = {}
    for sig in train_sig:
        for s in set(sig["wl2"]):
            train_counts[s] = train_counts.get(s, 0) + 1
    rules = []
    for rec in motifs.values():
        n_mol, n_scaf = len(rec["n_molecules"]), len(rec["scaffolds"])
        rules.append(
            {
                "wl2": rec["wl2"],
                "element": rec["element"],
                "degree": rec["degree"],
                "n_molecules": n_mol,
                "n_scaffolds": n_scaf,
                "train_support": int(train_counts.get(rec["wl2"], 0)),
                "reused_across_molecules_in_train": bool(train_counts.get(rec["wl2"], 0) >= 2),
                "signed_sum": rec["signed_sum"],
                "mean_contribution": rec["signed_sum"] / max(n_mol, 1),
                "n_positive": rec["n_positive"],
                "n_negative": rec["n_negative"],
                "sign_consistency": max(rec["n_positive"], rec["n_negative"]) / max(n_mol, 1),
                "examples": rec["examples"],
            }
        )
    rules = [r for r in rules if r["n_molecules"] >= 20 and r["n_scaffolds"] >= 5]
    rules.sort(key=lambda r: -abs(r["mean_contribution"]))
    rel_rows = [
        {
            "bucket": r["bucket"],
            "n_molecules": len(r["n_molecules"]),
            "signed_sum": r["signed_sum"],
            "mean_contribution": r["signed_sum"] / max(len(r["n_molecules"]), 1),
            "n_positive": r["n_positive"],
            "n_negative": r["n_negative"],
        }
        for r in relations.values()
        if len(r["n_molecules"]) >= 20
    ]
    rel_rows.sort(key=lambda r: -abs(r["mean_contribution"]))
    return {
        "dataset": dataset,
        "arm": arm,
        "n_valid_molecules_analysed": int(positions.size),
        "support_rule": "kept when >=20 valid molecules and >=5 scaffold proxies (frozen before computation)",
        "rooted_structure_rules": rules[:20],
        "relation_rules": rel_rows[:20],
        "n_signatures_total": len(motifs),
        "n_signatures_passing_support": len(rules),
        "vocabulary": "atom names follow the inherited cache feature spec (_LIPO_ATOM_NAMES)",
        "limitations": [
            "scaffold proxy = hash of the sorted WL-2 multiset + atom count, not a Bemis-Murcko scaffold",
            "contributions are the frozen checkpoint's own per-token terms; no causal claim",
            "WL-2 hashes are not human-readable drug-chemistry motifs on their own; cases show SMILES",
        ],
    }


def explanation_cases(run_dir: Path, device: torch.device, *, n_cases: int = MAX_CASES) -> dict[str, Any]:
    run, view, model = _view_run(run_dir)
    model.to(device)
    dataset, arm = run["dataset"], run["arm"]
    metrics, logits = D.evaluate(model, view, "valid", device)
    if view.task == "regression":
        pred = logits.reshape(-1) * float(view.target_norm["scale"]) + float(view.target_norm["mean"])
        err = np.abs(pred - view.y["valid"].reshape(-1))
        truth = view.y["valid"].reshape(-1)
    else:
        prob = 1.0 / (1.0 + np.exp(-logits))
        err = np.abs(view.y["valid"] - prob).mean(axis=1)
        pred = prob
        truth = view.y["valid"]
    order = np.argsort(-err)
    picks = list(order[: n_cases // 2]) + list(order[len(order) // 2 : len(order) // 2 + n_cases // 4]) + list(order[-max(1, n_cases // 4) :])
    cases = []
    for pos in picks[:n_cases]:
        batch = view.to_device(_collate_rows(view, "valid", [int(pos)]), device)
        row: dict[str, Any] = {
            "valid_position": int(pos),
            "official_row_id": int(view.ids["valid"][pos]),
            "prediction": pred[pos].tolist() if np.ndim(pred[pos]) else float(pred[pos]),
            "truth": truth[pos].tolist() if np.ndim(truth[pos]) else (None if not np.isfinite(truth[pos]) else float(truth[pos])),
            "abs_error": float(np.mean(err[pos])) if np.ndim(err[pos]) else float(err[pos]),
        }
        sig = _wl_signatures(view.caches["valid"], [int(view.rows["valid"][pos])])[0]
        row["atoms"] = [
            {"index": i, "element": sig["element"][i], "degree": sig["degree"][i]} for i in range(min(sig["n_atoms"], 40))
        ]
        if model.variant != C.HIDDEN:
            parts = accounting(model, batch)
            rebuilt = _reconstruct(model, parts)
            with torch.no_grad():
                direct = model(batch).detach().cpu().numpy()
            row["reconstruction_error"] = float(np.abs(rebuilt - direct).max())
            row["global_term"] = parts["global"].reshape(-1).tolist()[:5]
            row["local_part"] = parts["local"].reshape(-1).tolist()[:5]
            row["relation_part"] = parts["relation"].reshape(-1).tolist()[:5]
            if "token_local" in parts:
                tl, tr = parts["token_local"], parts["token_relation"]
                k = int(np.argmax(np.abs(tl).max(axis=1))) if tl.size else 0
                j = int(np.argmax(np.abs(tr).max(axis=1))) if tr.size else 0
                row["top_local_token"] = {
                    "atom_index": k,
                    "element": sig["element"][k] if k < len(sig["element"]) else None,
                    "degree": sig["degree"][k] if k < len(sig["degree"]) else None,
                    "contribution_first5": tl[k].tolist()[:5],
                }
                if tr.size:
                    pu = batch["pair_u"].detach().cpu().numpy()[j]
                    pv = batch["pair_v"].detach().cpu().numpy()[j]
                    rel = batch["pair_r"].detach().cpu().numpy()[j]
                    row["top_relation_token"] = {
                        "bucket": _relation_bucket(rel),
                        "common_neighbours": int(round(float(np.expm1(rel[5])))),
                        "contribution_first5": tr[j].tolist()[:5],
                    }
        cases.append(row)
    return {
        "dataset": dataset,
        "arm": arm,
        "selection_rule": "half largest |error|, a quarter around the median |error|, a quarter smallest |error|",
        "n_cases": len(cases),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _pick_runs(runs_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for run_json in sorted(runs_root.glob("*/run.json")):
        payload = json.loads(run_json.read_text())
        if payload.get("smoke") or payload.get("status", "completed") != "completed":
            continue
        groups.setdefault(payload["dataset"], []).append(run_json.parent)
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase E/F mechanism diagnostics")
    ap.add_argument("--runs", default=str(C.RUNS))
    ap.add_argument("--analysis", default=str(C.ANALYSIS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases", type=int, default=MAX_CASES)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    analysis = Path(args.analysis)
    groups = _pick_runs(Path(args.runs))
    diagnostics: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for dataset, run_dirs in groups.items():
        scored = []
        for run_dir in run_dirs:
            run = json.loads((run_dir / "run.json").read_text())
            scored.append((float((run.get("best_valid_metrics") or {}).get("ap", 0.0)) if dataset == C.PCBA else 0.0, run["arm"], run_dir))
        arms = {run_json["arm"]: run_dir for run_json, run_dir in ((json.loads((d / "run.json").read_text()), d) for d in run_dirs)}
        order = ["BASE", C.ADD, C.EVID, C.BANK, C.COMPOUND, C.UNARY, C.HIDDEN, C.FLEX]
        wanted = [a for a in order if a in arms] + [a for a in arms if a not in order]
        if dataset == C.PCBA:
            best = max((a for a in arms if a not in ("BASE", "HIDDEN")), key=lambda a: _prefix_best(arms[a]), default=None)
        else:
            best = max((a for a in arms if a != "BASE"), key=lambda a: _valid_primary(arms[a], dataset), default=None)
        if best and best not in wanted:
            wanted.append(best)
        for arm in wanted:
            run_dir = arms[arm]
            C.log(f"mechanism: {dataset}/{arm}")
            entry = analyse_run(run_dir, device, analysis=analysis)
            diagnostics.append(entry)
            if arm in (best, C.ADD, C.EVID):
                rules.append(structure_rules(run_dir, device, max_molecules=PCBA_SUBSET if dataset == C.PCBA else None))
                cases.append(explanation_cases(run_dir, device, n_cases=args.cases))
    C.write_json(analysis / "decomposition_trace.json", {"recon_tolerance": RECON_TOL, "traces": [d.get("accounting", {}) | {"dataset": d["dataset"], "arm": d["arm"], "exact": d.get("accounting_exact")} for d in diagnostics]})
    C.write_json(analysis / "mechanism_diagnostics.json", {"entries": diagnostics})
    C.write_json(analysis / "structure_rules.json", {"datasets": rules})
    C.write_json(analysis / "explanation_cases.json", {"datasets": cases})
    return 0


def _valid_primary(run_dir: Path, dataset: str) -> float:
    run = json.loads((run_dir / "run.json").read_text())
    metrics = run.get("best_valid_metrics") or {}
    key = "rmse" if dataset in (C.LIPO, C.ESOL) else "macro_roc_auc"
    value = metrics.get(key)
    if value is None:
        return float("inf") if key == "rmse" else float("-inf")
    return -float(value) if key == "rmse" else float(value)


def _prefix_best(run_dir: Path) -> float:
    curve = run_dir / "curve.csv"
    if not curve.exists():
        return float("-inf")
    import csv as _csv

    with open(curve, newline="", encoding="utf-8") as fh:
        rows = [r for r in _csv.DictReader(fh) if float(r["epoch"]) <= 25]
    return max((float(r["valid_ap"]) for r in rows), default=float("-inf"))


if __name__ == "__main__":
    raise SystemExit(main())
