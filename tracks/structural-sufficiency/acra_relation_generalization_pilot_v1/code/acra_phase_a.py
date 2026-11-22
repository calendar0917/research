"""Phase A: exact relation algebra, old-checkpoint provenance, capability check.

No new model is trained here.  The parent ACRA_REL checkpoint is loaded into
the ``ACRA_REL_REF`` architecture (same state-dict keys by construction), and

* the old direct pair sum is compared with the exact moment + histogram
  decomposition on real molecules and random small synthetic graphs;
* the old checkpoint's E_rel is split per molecule and related to size and
  compositional-novelty descriptors;
* moment / histogram are removed at test time (frozen weights) to observe the
  prediction and Valid-error change;
* the category-to-pair assignment is permuted while the category histogram and
  every edge representation are held fixed.  Original ACRA_REL must be exactly
  invariant; this is an expressivity property, not chemical evidence.

The official Valid labels are used only for the frozen-weight diagnostic in
this phase (they are already part of the parent round's recorded result).  No
new model is evaluated on Valid here, and Test labels are never read.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_cache as K
import acra_data as D
import acra_model as M
import acra_split as S

PARENT = C.ROUND_DIR.parent / "attachment_conditioned_response_algebra_pilot_v1"
PARENT_CHECKPOINT = PARENT / "results" / "ACRA_REL_S0" / "state_best.pt"
DESCRIPTOR_NAMES = ("n_bonds", "n_atoms", "cycle_rank", "unseen_combo_fraction")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return C.pearson(np.asarray(x, np.float64), np.asarray(y, np.float64))


def direct_relation(model: M.ACRAModel, batch: Mapping[str, Any],
                    out: Mapping[str, Any], categories: np.ndarray | None = None,
                    gamma: np.ndarray | None = None) -> Any:
    """The literal old pair loop, used as an independent check of the algebra."""
    import torch

    if int(batch["n_pairs"]) == 0:
        return torch.zeros(int(batch["n_molecules"]), device=model.beta.device)
    qe, ke = out["rel_qe"], out["rel_ke"]
    pe, pf = batch["pair_e"], batch["pair_f"]
    dots = 0.5 * ((qe[pe] * ke[pf]).sum(-1) + (qe[pf] * ke[pe]).sum(-1))
    cat = (out["rel_categories"] if categories is None
           else torch.as_tensor(categories, dtype=torch.long, device=qe.device))
    pair_value = dots / math.sqrt(model.rel_rank)
    if hasattr(model, "rel_bias"):
        pair_value = pair_value + model.rel_bias.weight.squeeze(-1)[cat]
    if gamma is not None:
        pair_value = pair_value + (torch.as_tensor(gamma, dtype=qe.dtype,
                                                   device=qe.device)[cat] - 1.0) * (
            dots / math.sqrt(model.rel_rank))
    total = torch.zeros(int(batch["n_molecules"]), device=qe.device).index_add_(
        0, batch["pair_mol"], pair_value)
    gate = torch.sigmoid(model.gate_raw)
    return gate * total / batch["n_bonds"].float().clamp(min=1.0)


def descriptors(pack: Mapping[str, Any], train_rows) -> dict[str, np.ndarray]:
    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    combo = np.asarray(pack["ev_combo"], np.int64)
    ev_mol = np.repeat(np.arange(int(pack["n_molecules"])), np.diff(ev_ptr))
    mol_train = np.zeros(int(pack["n_molecules"]), bool)
    mol_train[np.asarray(train_rows, np.int64)] = True
    train_counts = np.bincount(combo[mol_train[ev_mol]],
                               minlength=int(np.asarray(pack["combo_s"]).size))
    unseen = np.zeros(int(pack["n_molecules"]), np.float64)
    for m in range(int(pack["n_molecules"])):
        ids = combo[ev_ptr[m]:ev_ptr[m + 1]]
        if ids.size:
            unseen[m] = float(np.mean(train_counts[ids] == 0))
    return {
        "n_bonds": np.asarray(pack["mol_n_bonds"], np.float64),
        "n_atoms": np.asarray(pack["mol_n_atoms"], np.float64),
        "cycle_rank": np.asarray(pack["mol_cycle_rank"], np.float64),
        "unseen_combo_fraction": unseen,
    }


def _load_old(cfg: Mapping[str, Any], pack: Mapping[str, Any], dims) -> tuple[Any, dict[str, Any]]:
    import torch

    ckpt = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    model = M.ACRAModel(cfg, dims[0], dims[1], "ACRA_REL_REF")
    model.load_pack(pack)
    model.load_state_dict(ckpt["state"], strict=True)
    model.eval()
    return model, ckpt


def _synthetic_pack(cfg: Mapping[str, Any], raw: Mapping[str, Any], n_graphs: int = 12,
                    seed: int = 0) -> dict[str, Any]:
    import networkx as nx

    PD = C.PD
    rng = np.random.default_rng(int(seed))
    train = np.asarray(raw["train_idx"], np.int64)
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"], np.uint8)
    bond_feat = np.asarray(raw["bond_feat"], np.uint8)
    mols = []
    for g in range(int(n_graphs)):
        n = int(rng.integers(6, 11))
        graph = nx.gnp_random_graph(n, 0.30, seed=int(rng.integers(1 << 30)))
        while not nx.is_connected(graph) or graph.number_of_edges() < 2:
            graph = nx.gnp_random_graph(n, 0.30, seed=int(rng.integers(1 << 30)))
        bond_list = sorted((min(int(a), int(b)), max(int(a), int(b)))
                           for a, b in graph.edges())
        rows = [int(rng.choice(train)) for _ in range(n)]
        atom_rows = []
        for row in rows:
            a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
            atom_rows.append(atom_feat[int(rng.integers(a0, a1))])
        bond_rows = []
        for _ in bond_list:
            row = int(rng.choice(train))
            b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
            bond_rows.append(bond_feat[int(rng.integers(b0, b1))])
        mols.append(PD.Molecule(
            row_id=g, n_atoms=n, atom_feat=np.stack(atom_rows),
            bond_u=np.asarray([a for a, _ in bond_list], np.int32),
            bond_v=np.asarray([b for _, b in bond_list], np.int32),
            bond_feat=np.stack(bond_rows),
        ))
    return K.pack_from_molecules(cfg, mols, [f"synth{g}" for g in range(len(mols))],
                                 PD.ColorSpace.from_raw(raw), "synthetic", list(range(len(mols))),
                                 log=lambda _m: None)


def _rel_metrics(pred_std: np.ndarray, y_std: np.ndarray) -> dict[str, float]:
    return {
        "rmse_std": float(np.sqrt(np.mean((pred_std - y_std) ** 2))),
        "mae_std": float(np.mean(np.abs(pred_std - y_std))),
    }


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    import torch

    pack = K.ensure_pack(cfg, log=log)
    raw = D.load_raw(cfg)
    dims = C.official_feature_dims()
    split_payload = S.load(cfg)
    model, ckpt = _load_old(cfg, pack, dims)
    source = D.BatchSource(pack, "cpu")
    official = D.official_splits(raw)
    rows_real = np.sort(np.concatenate([official["train"], official["valid"]]))
    labels = np.asarray(raw["labels"], np.float64)
    scaler = ckpt["scaler"]
    y_std_all = (labels - float(scaler["mean"])) / float(scaler["std"])
    desc = descriptors(pack, official["train"])

    # ---- 1. exact algebra on real molecules ---------------------------------
    direct_err = []
    decomp_err = []
    count_err = []
    out_all: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, rows_real.size, 128):
            rows = rows_real[start:start + 128]
            batch = source.make(rows)
            out = model.forward_batch(batch)
            direct = direct_relation(model, batch, out)
            direct_err.append((direct - out["e_rel"]).abs().max().item())
            decomp_err.append((out["e_rel_moment"] + out["e_rel_histogram"]
                               - out["e_rel"]).abs().max().item())
            m = batch["n_bonds"].long()
            count_err.append(float((out["relation_counts"].sum(1)
                                    - m * (m - 1) // 2).abs().max().item()))
            for key in ("yhat_std", "e_atom", "e_bond", "e_rel", "e_rel_moment",
                        "e_rel_histogram", "moment", "histogram", "beta"):
                out_all.setdefault(key, []).append(out[key].detach().double().cpu().numpy())
    out_real = {k: np.concatenate(v) for k, v in out_all.items()}
    order = np.argsort(rows_real)
    rows_sorted = rows_real[order]
    for k in out_real:
        out_real[k] = out_real[k][order]
    real_block = {
        "n_molecules": int(rows_real.size),
        "n_bond_pairs": int(np.sum(
            desc["n_bonds"][rows_sorted] * (desc["n_bonds"][rows_sorted] - 1) / 2)),
        "max_abs_error_direct_minus_decomposition": float(np.max(direct_err)),
        "max_abs_error_moment_plus_histogram_minus_e_rel": float(np.max(decomp_err)),
        "max_abs_error_relation_count_sum": float(np.max(count_err)),
        "rows_used": "official Train + Valid only (Test never touched)",
    }

    # ---- 2. random small graphs ---------------------------------------------
    syn_pack = _synthetic_pack(cfg, raw, n_graphs=12, seed=0)
    syn_model, _ = _load_old(cfg, syn_pack, dims)
    syn_source = D.BatchSource(syn_pack, "cpu")
    syn_direct, syn_decomp = [], []
    with torch.no_grad():
        batch = syn_source.make(list(range(int(syn_pack["n_molecules"]))))
        out = syn_model.forward_batch(batch)
        syn_direct.append(float((direct_relation(syn_model, batch, out) - out["e_rel"]).abs().max()))
        syn_decomp.append(float((out["e_rel_moment"] + out["e_rel_histogram"]
                                 - out["e_rel"]).abs().max()))
    synthetic_block = {
        "n_graphs": int(syn_pack["n_molecules"]),
        "generator": "networkx connected G(n,p=0.30), n in [6,10], real Train atom/bond attribute rows",
        "max_abs_error_direct_minus_decomposition": float(max(syn_direct)),
        "max_abs_error_moment_plus_histogram_minus_e_rel": float(max(syn_decomp)),
    }

    # ---- 3. per-molecule decomposition and scale ---------------------------
    d = {k: v[rows_sorted] for k, v in desc.items()}
    e_rel = out_real["e_rel"]
    e_mom = out_real["e_rel_moment"]
    e_hist = out_real["e_rel_histogram"]
    moment = out_real["moment"]
    histogram = out_real["histogram"]
    m = d["n_bonds"]
    correlations: dict[str, dict[str, float]] = {}
    for term_name, term in (("abs_moment", np.abs(moment)),
                            ("abs_histogram", np.abs(histogram)),
                            ("abs_e_rel", np.abs(e_rel)),
                            ("e_rel", e_rel)):
        correlations[term_name] = {
            "vs_n_bonds": _pearson(term, m),
            "vs_n_atoms": _pearson(term, d["n_atoms"]),
            "vs_cycle_rank": _pearson(term, d["cycle_rank"]),
            "vs_unseen_combo_fraction": _pearson(term, d["unseen_combo_fraction"]),
        }
    # bond-count quartiles (explicit scale-effect diagnostic)
    edges = np.quantile(m, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartiles = []
    for i in range(4):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (m >= lo) & (m <= hi) if i == 3 else (m >= lo) & (m < hi)
        if not mask.any():
            continue
        quartiles.append({
            "quartile": i + 1,
            "n_bonds_range": [lo, hi],
            "n": int(mask.sum()),
            "mean_moment": float(np.mean(moment[mask])),
            "mean_histogram": float(np.mean(histogram[mask])),
            "mean_e_rel": float(np.mean(e_rel[mask])),
            "mean_abs_moment": float(np.mean(np.abs(moment[mask]))),
            "mean_abs_histogram": float(np.mean(np.abs(histogram[mask]))),
            "mean_abs_e_rel": float(np.mean(np.abs(e_rel[mask]))),
            "mean_abs_moment_per_bond": float(np.mean(np.abs(moment[mask]) / np.maximum(1, m[mask]))),
            "mean_abs_histogram_per_bond": float(np.mean(np.abs(histogram[mask]) / np.maximum(1, m[mask]))),
            "mean_abs_e_rel_per_bond": float(np.mean(np.abs(e_rel[mask]) / np.maximum(1, m[mask]))),
        })
    # exact identity on every molecule (counts already checked in batch)
    factorization = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "checkpoint": str(PARENT_CHECKPOINT.relative_to(C.REPO_ROOT)),
        "parent_best_epoch": int(ckpt["epoch"]),
        "formula": {
            "direct": "E_rel = gate * sum_{e<f} [0.5(q_e.k_f + q_f.k_e)/sqrt(rank) + bias[bucket(e,f)]] / max(1,m)",
            "moment": "0.5 * (Q.K - sum_e q_e.k_e) / sqrt(rank), Q=sum_e q_e, K=sum_e k_e",
            "histogram": "sum_bucket count[bucket] * bias[bucket]",
            "identity": "E_rel = gate * (moment + histogram) / max(1,m)",
        },
        "real_molecules": real_block,
        "synthetic_small_graphs": synthetic_block,
        "per_molecule_identity_max_abs_error": float(np.max(np.abs(
            e_mom + e_hist - e_rel))),
        "decomposition_means": {
            "mean_abs_moment": float(np.mean(np.abs(moment))),
            "mean_abs_histogram": float(np.mean(np.abs(histogram))),
            "mean_abs_e_rel": float(np.mean(np.abs(e_rel))),
            "mean_moment": float(np.mean(moment)),
            "mean_histogram": float(np.mean(histogram)),
            "mean_e_rel": float(np.mean(e_rel)),
        },
        "correlations": correlations,
        "bond_count_quartiles": quartiles,
        "test_labels_read": False,
    }
    C.write_json(C.analysis_dir() / "relation_factorization.json", factorization)
    log(f"wrote relation_factorization.json: real max error "
        f"{real_block['max_abs_error_direct_minus_decomposition']:.3e}")

    # ---- 4. provenance + frozen ablations ---------------------------------
    valid = np.asarray(official["valid"], np.int64)
    valid_pos = np.searchsorted(rows_sorted, valid)
    valid_out = {k: v[valid_pos] for k, v in out_real.items()}
    y_valid = y_std_all[valid]
    base_std = valid_out["yhat_std"]
    variants = {
        "base": base_std,
        "without_moment": base_std - valid_out["e_rel_moment"],
        "without_histogram": base_std - valid_out["e_rel_histogram"],
        "without_both": base_std - valid_out["e_rel"],
    }
    frozen = {}
    for name, pred in variants.items():
        frozen[name] = {
            **_rel_metrics(pred, y_valid),
            "rmse_label_units": float(np.sqrt(np.mean(
                (pred * scaler["std"] + scaler["mean"] - labels[valid]) ** 2))),
            "mae_label_units": float(np.mean(np.abs(
                pred * scaler["std"] + scaler["mean"] - labels[valid]))),
        }
        if name != "base":
            delta = pred - base_std
            frozen[name]["prediction_change"] = {
                "mean_abs": float(np.mean(np.abs(delta))),
                "max_abs": float(np.max(np.abs(delta))),
                "pearson_with_base": _pearson(delta, base_std),
            }
    gate = float(torch.sigmoid(model.gate_raw).detach().cpu())
    provenance = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "checkpoint": str(PARENT_CHECKPOINT.relative_to(C.REPO_ROOT)),
        "best_epoch": int(ckpt["epoch"]),
        "gate_sigmoid": gate,
        "definitions": {
            "moment": "global low-rank second moment of the symmetric bond-pair kernel",
            "histogram": "sum over the 8 (distance bucket, shared-ring) categories of count * bias",
            "frozen_ablation": "same checkpoint; one algebraic term set to zero at test time; "
                               "not a retrained causal effect",
        },
        "official_valid": {
            "n": int(valid.size),
            "labels_used": "frozen-weight diagnostic only; parent-round label split",
            "frozen_weight_variants": frozen,
        },
        "train_sub_dev": {
            "n_train_sub": int(split_payload["n_train_sub"]),
            "n_dev": int(split_payload["n_dev"]),
            "note": "the old full-Train checkpoint is not a Dev baseline; it has seen Dev labels",
        },
        "term_shares_on_train_plus_valid": {
            "mean_abs_moment": float(np.mean(np.abs(moment))),
            "mean_abs_histogram": float(np.mean(np.abs(histogram))),
            "mean_abs_e_rel": float(np.mean(np.abs(e_rel))),
            "moment_fraction_of_abs_e_rel_components": float(
                np.mean(np.abs(moment)) / max(1e-12, np.mean(np.abs(moment))
                                              + np.mean(np.abs(histogram)))),
        },
        "test_labels_read": False,
    }
    C.write_json(C.analysis_dir() / "relation_term_provenance.json", provenance)
    log(f"wrote relation_term_provenance.json: gate={gate:.4f}")

    # ---- 5. fixed-histogram category permutation ---------------------------
    rng = np.random.default_rng(0)
    invariance = []
    with torch.no_grad():
        for row in rows_sorted[:80].tolist():
            batch = source.make([int(row)])
            out = model.forward_batch(batch)
            cat = out["rel_categories"].cpu().numpy()
            if int(np.unique(cat).size) < 2 or int(batch["n_pairs"]) < 4:
                continue
            perm = rng.permutation(cat)
            base = direct_relation(model, batch, out)
            changed = direct_relation(model, batch, out, categories=perm)
            invariance.append(float((base - changed).abs().max()))
            if len(invariance) >= 20:
                break
    factor_perm = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "definition": "edge representations, pair identities and the joint relation-category "
                      "histogram are fixed; only the assignment of categories to concrete "
                      "bond pairs is permuted",
        "n_cases": len(invariance),
        "max_abs_delta_e_rel": float(max(invariance)) if invariance else 0.0,
        "mean_abs_delta_e_rel": float(np.mean(invariance)) if invariance else 0.0,
        "interpretation": "original ACRA_REL is invariant by construction; capability only",
        "test_labels_read": False,
    }
    C.write_json(C.analysis_dir() / "relation_permutation_invariance.json", factor_perm)
    return {"factorization": factorization, "provenance": provenance,
            "permutation": factor_perm}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())