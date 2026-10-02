"""Stage-A audit for zinc_overnight_bottleneck_v1 (read-only, CPU).

Replays the latest product control soup and measures the node-chain magnitudes
on the canonical fresh initialisation.  Never touches official test.

Run:
    uv run python -m tracks.ksvd.results.zinc_overnight_bottleneck_v1.audit_stageA
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1lib
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_node_binding_residual_pair_v1 as rp
from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

OUT = REPO_ROOT / "tracks/ksvd/results/zinc_overnight_bottleneck_v1"
PAIR = REPO_ROOT / "tracks/ksvd/results/zinc_node_binding_residual_pair_v1"
N_GRAPHS = 1024
BATCH = 128


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return float(obj.detach().cpu().reshape(-1)[0]) if obj.numel() == 1 else obj.detach().cpu().tolist()
    return obj


def replay_control() -> dict:
    """Reload the latest product-control soup and re-measure raw/cal valid MAE."""
    device = uprun.resolve_device("cpu")
    torch.set_num_threads(8)
    dictionary = uprun._dictionary_tensor()
    subspace = uprun._load_parent_subspace()
    model = rp.build_pair_model("control", dictionary, subspace)
    state = torch.load(PAIR / "control/soup_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    train = uprun.load_split("control", "train")
    valid = uprun.load_split("control", "valid")
    raw_v, y_v = rp._collect(model, valid, device)
    raw_t, y_t = rp._collect(model, train, device)
    bias = float(np.median(y_t - raw_t))
    cal_v = raw_v + bias
    return {
        "raw_valid_mae": float(np.mean(np.abs(raw_v - y_v))),
        "calibrated_valid_mae": float(np.mean(np.abs(cal_v - y_v))),
        "raw_train_mae": float(np.mean(np.abs(raw_t - y_t))),
        "train_fitted_bias": bias,
        "n_valid": int(len(y_v)),
        "official_test_loaded": False,
    }


def magnitude_audit() -> dict:
    """Canonical fresh init magnitudes along the real forward."""
    device = torch.device("cpu")
    dictionary = uprun._dictionary_tensor()
    subspace = uprun._load_parent_subspace()
    model = rp.build_pair_model("control", dictionary, subspace).to(device).eval()
    init_hash = rp.parameter_state_hash(model)
    train = uprun.load_split("control", "train")
    rng = np.random.default_rng(0)
    selected = np.sort(rng.choice(len(train), size=N_GRAPHS, replace=False))
    subset = [train[int(i)] for i in selected]
    loader = p1lib.make_env_loader(subset, BATCH, False, rp.SEED + int(uprun.EVAL_SHUFFLE_OFFSET))

    node_enc = model.node_encoder
    fusion = model.fusion
    W1 = node_enc[0].weight.detach()  # [64, 96]
    b1 = node_enc[0].bias.detach()
    W2 = node_enc[2].weight.detach()  # [48, 64]
    b2 = node_enc[2].bias.detach()
    F1w = fusion[0].weight.detach()  # [342, 446]
    F1b = fusion[0].bias.detach()
    F2w = fusion[2].weight.detach()

    acc: dict[str, list[float]] = {}
    def add(k, v):
        acc.setdefault(k, []).append(float(v))

    node_offset = 0
    edge_offset = 0
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            coord = model.code(batch.dict_phi)
            q = torch.nn.functional.one_hot(batch.dict_atom, num_classes=int(p2.ATOM_CATEGORIES)).to(coord.dtype)
            occ = batch.env_occ_node.to(device)
            c = coord[occ]
            qc = q[occ]
            s = c @ model.W_A_S
            a = qc @ model.W_A_C
            u0 = (s * a) / math.sqrt(float(p2.D_A))
            n = int(batch.dict_phi.shape[0])
            flat = torch.zeros((n * int(p2.N_SHELLS), int(p2.D_A)), device=device, dtype=u0.dtype)
            idx = batch.env_occ_root.to(device) * int(p2.N_SHELLS) + batch.env_occ_shell.to(device)
            flat.index_add_(0, idx, u0)
            node_slots = flat.view(n, int(p2.N_SHELLS), int(p2.D_A))

            # edge slots (for scale comparison)
            d_e = int(model.config.d_e)
            bond_u = batch.env_bond_u.to(device)
            bond_v = batch.env_bond_v.to(device)
            cu, cv = coord[bond_u], coord[bond_v]
            g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
            b = torch.nn.functional.one_hot(batch.env_bond_type, num_classes=int(p2.BOND_CATEGORIES)).to(coord.dtype)
            ue = (g @ model.W_E_S) * (b @ model.W_E_C) / math.sqrt(float(d_e))
            flat_e = torch.zeros((n * int(p2.SHELLPAIR_CLASSES), d_e), device=device, dtype=ue.dtype)
            flat_e.index_add_(
                0,
                batch.env_bond_root.to(device) * int(p2.SHELLPAIR_CLASSES) + batch.env_bond_shellpair.to(device),
                ue,
            )
            edge_slots = flat_e.view(n, int(p2.SHELLPAIR_CLASSES), d_e)

            interface = model.semantic_interface(coord, batch, None, None)
            node_out = node_enc(node_slots)
            edge_out = model.edge_encoder(edge_slots)
            fused = torch.cat([interface, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1)
            pre1 = node_slots.reshape(-1, int(p2.D_A)) @ W1.t() + b1  # [n*3, 64]
            node_out_flat = node_out.reshape(-1, int(p2.ENV_DIM))
            fused_pre = fused @ F1w.t() + F1b  # [n, 342]

            rms = lambda t: float(torch.sqrt((t.reshape(-1).to(torch.float64) ** 2).mean()).item())

            add("rms_s", rms(s)); add("rms_a", rms(a)); add("rms_u0", rms(u0))
            add("rms_node_slots", rms(node_slots))
            add("rms_edge_slots", rms(edge_slots))
            add("rms_edge_out", rms(edge_out))
            add("rms_interface", rms(interface))
            add("rms_node_out", rms(node_out))
            add("rms_fusion_in", rms(fused))
            add("rms_fusion_pre1", rms(fused_pre))
            # per-shell aggregated slot RMS (3 shells)
            for sh in range(int(p2.N_SHELLS)):
                add(f"rms_slot_shell{sh}", rms(node_slots[:, sh, :]))
            # node encoder first layer: signal vs bias
            add("node_enc_W1_norm", float(W1.norm().item()))
            add("node_enc_b1_rms", rms(b1))
            add("node_enc_pre1_bias_rms", rms(pre1.detach() - (node_slots.reshape(-1, int(p2.D_A)) @ W1.t())))
            add("node_enc_pre1_signal_rms", rms(node_slots.reshape(-1, int(p2.D_A)) @ W1.t()))
            add("node_enc_pre1_rms", rms(pre1))
            add("node_out_col_std_mean", float(node_out_flat.std(dim=0).mean().item()))
            add("node_out_col_std_max", float(node_out_flat.std(dim=0).max().item()))
            # fusion first layer columns corresponding to node part (offset 110..110+144)
            Fnode = F1w[:, 110:254]
            add("fusion_first_node_colnorm_mean", float(Fnode.norm(dim=1).mean().item()))
            node_mean = node_out.reshape(n, -1).mean(dim=0).detach()
            add("fusion_first_aug_bias_absmax", float((F1b + Fnode @ node_mean).abs().max().item()))
            add("fusion_first_bias_absmax", float(F1b.abs().max().item()))
            add("fusion_first_bias_neg_frac", float((F1b < 0).float().mean().item()))
            add("fusion_first_pre1_colstd_mean", float(fused_pre.std(dim=0).mean().item()))
            n_seen += n

    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in acc.items()}
    out["n_graphs"] = int(n_seen)
    out["n_occurrences"] = None
    out["init_state_sha256"] = init_hash
    out["node_encoder_W1_shape"] = list(W1.shape)
    out["node_encoder_W2_shape"] = list(W2.shape)
    out["fusion_W1_shape"] = list(F1w.shape)
    out["node_encoder_b2_absmax"] = float(b2.abs().max().item())
    # fixed-batch slot RMS (as in the pair protocol) on the first 128 train graphs
    diag = next(iter(p1lib.make_env_loader(train[:128], 128, False, rp.SEED + int(uprun.EVAL_SHUFFLE_OFFSET))))
    model.zero_grad(set_to_none=True)
    return out


def main() -> None:
    started = time.perf_counter()
    payload = {"official_test_loaded": False}
    payload["replay_control"] = replay_control()
    payload["magnitudes"] = magnitude_audit()
    payload["seconds"] = float(time.perf_counter() - started)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_stageA.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))
    print(f"[audit] seconds={payload['seconds']:.1f}")


if __name__ == "__main__":
    main()