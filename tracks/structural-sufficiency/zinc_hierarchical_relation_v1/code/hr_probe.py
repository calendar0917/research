"""Post-hoc inference probes on the frozen valid split (no training, no test).

Loads the selected best checkpoint and measures whether the claimed mechanisms
are actually used: support gates, patch-edge/junction sensitivity, and (for B)
real-vs-null relation assignment.  Analysis only; does not change any gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import hr_common as C
import hr_data as D
import hr_models as M
from hr_runner import load_y


def _pred(model, entries, batch_size=256, relation_mode=None):
    model.eval()
    out = []
    stats = []
    gates = []
    with torch.no_grad():
        for i in range(0, len(entries), batch_size):
            b = D.move(D.collate(entries[i : i + batch_size]), "cpu")
            out.append(model(b, relation_mode=relation_mode).numpy())
            stats.append({k: float(v) for k, v in model._last_stats.items()})
            if b["p_support"].numel():
                st = model.support_trunk(b["p_support"])
                gates.append(torch.cat([torch.sigmoid(model.support_node(st)),
                                        torch.sigmoid(model.support_edge(st))], dim=1).numpy())
    pred = np.concatenate(out)
    gate = np.concatenate(gates) if gates else np.zeros((0, 2))
    return pred, stats, gate


def probe(arm: str) -> dict:
    cfg = C.load_config()
    run = C.RUNS_DIR / f"HR_{arm}_s0"
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model, info = M.build_model(arm=arm, ssl_path=C.REPO / cfg["ssl"]["path"],
                                seed=int(cfg["recipe"]["seed"]), cfg=cfg["model"], device="cpu")
    model.load_state_dict(ckpt["state"])
    model.eval()
    entries = D.load_cache(C.CACHE_DIR / "hr_val.pt")
    y = load_y("val")
    out = {"arm": arm, "protocol_id": C.PROTOCOL_ID, "parameters": info["parameters"],
           "checkpoint_epoch": int(ckpt["epoch"]), "checkpoint_valid_mae": float(ckpt["valid_mae"]),
           "official_test_loaded": False}
    p_real, stats, gate = _pred(model, entries)
    out["valid_mae_real"] = C.mae(p_real, y)
    if gate.size:
        out["support_gate"] = {
            "n": int(gate.shape[0]),
            "node_mean": float(gate[:, 0].mean()), "node_std": float(gate[:, 0].std()),
            "edge_mean": float(gate[:, 1].mean()), "edge_std": float(gate[:, 1].std()),
            "node_q05": float(np.quantile(gate[:, 0], 0.05)), "node_q95": float(np.quantile(gate[:, 0], 0.95)),
            "edge_q05": float(np.quantile(gate[:, 1], 0.05)), "edge_q95": float(np.quantile(gate[:, 1], 0.95)),
        }
    if stats:
        keys = stats[0].keys()
        out["forward_stats"] = {k: float(np.mean([s[k] for s in stats])) for k in keys}
    # perturbation probes
    base = D.load_cache(C.CACHE_DIR / "hr_val.pt")
    def perturb(field, value):
        ents = []
        for e in base:
            e = dict(e)
            if field == "p_support":
                e[field] = np.zeros_like(e[field])
            elif field == "pedge_type":
                e[field] = np.zeros_like(e[field])
            elif field == "junc_feat":
                e[field] = np.zeros_like(e[field])
            elif field == "global_feat":
                e[field] = np.zeros_like(e[field])
            ents.append(e)
        pred, _, _ = _pred(model, ents)
        return {"valid_mae": C.mae(pred, y), "max_abs_delta": float(np.max(np.abs(pred - p_real)))}
    out["support_zero"] = perturb("p_support", None)
    out["patch_edge_type_zero"] = perturb("pedge_type", None)
    out["junction_feat_zero"] = perturb("junc_feat", None)
    out["global_feat_zero"] = perturb("global_feat", None)
    # support is patch-specific: shuffle rows across the whole valid set
    rng = np.random.default_rng(123)
    ents = []
    for e in base:
        e = dict(e)
        n = e["n_bonds"]
        e["p_support"] = e["p_support"][rng.permutation(n)].copy()
        ents.append(e)
    pred, _, _ = _pred(model, ents)
    out["support_shuffled"] = {"valid_mae": C.mae(pred, y), "max_abs_delta": float(np.max(np.abs(pred - p_real)))}
    # force the support gate open / half-open (diagnostic, not a rescue)
    import torch.nn as nn
    class Const(nn.Module):
        def __init__(self, dim, value):
            super().__init__(); self.dim = dim; self.value = value
        def forward(self, x):
            return torch.full((x.shape[0], self.dim), self.value, dtype=x.dtype, device=x.device)
    for name, value in (("gate_open", 10.0), ("gate_half", 0.0)):
        saved_n, saved_e = model.support_node, model.support_edge
        model.support_node = Const(saved_n.out_features, value)
        model.support_edge = Const(saved_e.out_features, value)
        pred, _, _ = _pred(model, entries)
        out[name] = {"valid_mae": C.mae(pred, y), "max_abs_delta": float(np.max(np.abs(pred - p_real)))}
        model.support_node, model.support_edge = saved_n, saved_e
    # gate means by the frozen radius2-unseen grouping
    groups = C.load_valid_groups()
    unseen = groups["radius2_unseen"]
    patch_unseen = np.concatenate([np.full(e["n_bonds"], bool(unseen[i])) for i, e in enumerate(base)])
    if gate.size and len(patch_unseen) == gate.shape[0]:
        out["support_gate_by_group"] = {
            "unseen_node_mean": float(gate[patch_unseen, 0].mean()),
            "covered_node_mean": float(gate[~patch_unseen, 0].mean()),
            "unseen_edge_mean": float(gate[patch_unseen, 1].mean()),
            "covered_edge_mean": float(gate[~patch_unseen, 1].mean()),
        }
    if arm in ("B", "C"):
        p_null, stats_null, _ = _pred(model, entries, relation_mode="null")
        out["valid_mae_null_relations"] = C.mae(p_null, y)
        out["real_vs_null"] = {
            "mae_real": C.mae(p_real, y), "mae_null": C.mae(p_null, y),
            "mean_abs_pred_delta": float(np.mean(np.abs(p_real - p_null))),
            "max_abs_pred_delta": float(np.max(np.abs(p_real - p_null))),
            "corr": float(np.corrcoef(p_real, p_null)[0, 1]),
        }
    return out


def main():
    result = {}
    for arm in ("A", "B"):
        result[arm] = probe(arm)
        C.write_json(C.ANALYSIS_DIR / f"probe_{arm}_s0.json", result[arm])
        print(json.dumps(result[arm], indent=2))


if __name__ == "__main__":
    main()
