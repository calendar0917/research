"""Nonlinear head probe on the frozen Full ZINC soup.

Protocol: tracks/ksvd/protocols/zinc-full-nonlinear-probe-v1.yaml

Three configs, two seeds, frozen head configuration:
  A = [R]           nonlinear readout change candidate
  B = [R, z]        node-joint information conditional on R
  C = [R, q(R)]     same-width control (fixed-seed linear random projection of R)

No backbone update.  Official ZINC test never instantiated.  Reuses the frozen
handoff R cache (identity verified against the published soup prediction).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
HANDOFF = REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff"
AUDIT = REPO / "tracks/ksvd/results/zinc_full_bottleneck_audit_v1"

PARENT_CAL_VALID = 0.11506585458567133
PARENT_RAW_VALID = 0.1191540920053958
GATE = 0.003
SEEDS = (0, 1)
MAX_EPOCHS = 150
TARGET_FIT_SECONDS = 80.0
HARD_FIT_SECONDS = 90.0
BATCH = 256
LR = 1e-3
WD = 1e-4
CLIP = 5.0
QR_SEED = 20261002  # fixed, label-free
THREADS = 8


# --------------------------------------------------------------------------
# feature preparation
# --------------------------------------------------------------------------
def drop_constant_and_standardize(Xtr, Xva, tag, log):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    keep = sd > 1e-8
    n_drop = int((~keep).sum())
    Xtr = (Xtr[:, keep] - mu[keep]) / sd[keep]
    Xva = (Xva[:, keep] - mu[keep]) / sd[keep]
    log[tag] = {"in_dim": int(keep.size), "kept": int(keep.sum()),
                "dropped_constant": n_drop}
    return Xtr.astype(np.float32), Xva.astype(np.float32)


def build_features(log):
    tr = np.load(HANDOFF / "train_reader.npz")
    va = np.load(HANDOFF / "valid_reader.npz")
    Rtr, Rva = tr["R"].astype(np.float64), va["R"].astype(np.float64)
    ytr, yva = tr["y"].astype(np.float64), va["y"].astype(np.float64)
    ptr, pva = tr["p_base"].astype(np.float64), va["p_base"].astype(np.float64)
    ids_v = va["ids"].astype(np.int64)

    nj = np.load(AUDIT / "probe_features.npz")
    ztr, zva = nj["NJt"].astype(np.float64), nj["NJv"].astype(np.float64)
    assert nj["yt"].shape[0] == ytr.shape[0] and nj["yv"].shape[0] == yva.shape[0]
    ztr_raw, zva_raw = ztr, zva

    # identity: replay reader from exported weights, compare to p_base
    rd = np.load(HANDOFF / "reader.npz")
    def replay(Rm):
        h1 = np.maximum(Rm @ rd["W1"].T.astype(np.float64) + rd["b1"], 0.0)
        h2 = np.maximum(h1 @ rd["W2"].T.astype(np.float64) + rd["b2"], 0.0)
        return (h2 @ rd["W3"].T.astype(np.float64) + rd["b3"]).ravel()
    r_tr, r_va = replay(Rtr), replay(Rva)
    ident = {
        "reader_replay_train_max_abs_diff": float(np.max(np.abs(r_tr - ptr))),
        "reader_replay_valid_max_abs_diff": float(np.max(np.abs(r_va - pva))),
        "reader_replay_valid_raw_mae": float(np.mean(np.abs(r_va - yva))),
        "published_raw_valid_mae": PARENT_RAW_VALID,
        "p_base_sha": "17f5fcc3",
    }

    Rtr_s, Rva_s = drop_constant_and_standardize(Rtr, Rva, "R", log)
    ztr_s, zva_s = drop_constant_and_standardize(ztr_raw, zva_raw, "z", log)
    z_dim = ztr_s.shape[1]

    # q(R): fixed-seed linear random projection of the standardized R, matched to z width
    rng = np.random.default_rng(QR_SEED)
    P = rng.normal(size=(Rtr_s.shape[1], z_dim)) / np.sqrt(Rtr_s.shape[1])
    Qtr, Qva = Rtr_s @ P, Rva_s @ P
    Qtr_s, Qva_s = drop_constant_and_standardize(Qtr, Qva, "qR", log)
    log["qR"]["projection_seed"] = QR_SEED
    log["z"]["semantics"] = ("graph-level sum over occurrences of outer(coord_33, "
                             "onehot(atom_type_28)); no shell weighting; reused from "
                             "zinc_full_bottleneck_audit_v1 P2 (probe_features.npz NJ)")

    feats = {
        "Rtr": Rtr_s, "Rva": Rva_s,
        "ztr": ztr_s, "zva": zva_s,
        "Qtr": Qtr_s, "Qva": Qva_s,
        "ytr": ytr, "yva": yva, "ptr": ptr, "pva": pva, "ids_v": ids_v,
    }
    return feats, ident


# --------------------------------------------------------------------------
# shared-init head
# --------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, in_dim, W1, b1, W2, b2, W3, b3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(W1[:, :in_dim]); self.fc1.bias.copy_(b1)
            self.fc2.weight.copy_(W2); self.fc2.bias.copy_(b2)
            self.fc3.weight.copy_(W3); self.fc3.bias.copy_(b3)

    def forward(self, x):
        return self.fc3(F.silu(self.fc2(F.silu(self.fc1(x))))).squeeze(-1)


def make_shared_inits(seed, max_width):
    g = torch.Generator().manual_seed(seed)
    def draw(*shape):
        return (torch.randn(*shape, generator=g) * (1.0 / np.sqrt(shape[1])))
    W1 = draw(64, max_width)
    b1 = torch.zeros(64)
    W2 = draw(32, 64); b2 = torch.zeros(32)
    W3 = draw(1, 32); b3 = torch.zeros(1)
    return W1, b1, W2, b2, W3, b3


def train_head(Xtr, ytr, Xva, yva, in_dim, inits, epochs, seed):
    torch.manual_seed(seed)
    model = Head(in_dim, *inits)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if n.endswith("bias") else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LR)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr).float()
    Xv = torch.from_numpy(Xva)
    g = torch.Generator().manual_seed(seed + 12345)
    n = Xt.shape[0]
    last = None
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        model.train()
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.l1_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        last = float(loss.detach())
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        ptr = model(Xt).numpy().astype(np.float64)
        pva = model(Xv).numpy().astype(np.float64)
    train_loss_note = last
    return ptr, pva, elapsed, train_loss_note


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def group_delta(cand_cal, base_cal, y, ids, decile_mask):
    d = np.abs(base_cal - y) - np.abs(cand_cal - y)  # >0 candidate better
    i172 = int(np.argmax(np.abs(base_cal - y)))
    out = {
        "gain_all": float(d.mean()),
        "gain_ex_id172": float((d.sum() - d[i172]) / (len(d) - 1)),
        "gain_id172": float(d[i172]),
        "gain_lowest_train_decile": float(d[decile_mask].mean()) if decile_mask.sum() else None,
        "n_lowest_train_decile": int(decile_mask.sum()),
        "id172_valid_index": i172,
        "groups_gain_id_mod5": [float(d[(ids % 5) == g].mean()) for g in range(5)],
        "groups_improved_id_mod5": int(sum(d[(ids % 5) == g].mean() > 0 for g in range(5))),
    }
    return out


def tail_recheck(ptr, ytr, pva, yva, ids_v):
    b = float(np.median(ytr - ptr))
    cal_tr, cal_va = ptr + b, pva + b
    i172 = int(np.argmax(np.abs(cal_va - yva)))
    ex = np.ones(len(yva), bool); ex[i172] = False

    def seg(y, pred, m=None):
        if m is None:
            m = np.ones(len(y), bool)
        yy, pp = y[m], pred[m]
        if len(yy) == 0:
            return {"n": 0}
        slope = float(np.polyfit(yy, pp, 1)[0]) if len(yy) >= 3 else None
        se = pp - yy  # signed error = pred - y
        return {"n": int(len(yy)), "mae": mae(pp, yy),
                "signed_mean": float(se.mean()), "signed_median": float(np.median(se)),
                "ols_slope_pred_on_y": slope}

    res = {
        "calibration_bias_train_median": b,
        "parent_cal_valid_mae": mae(cal_va, yva),
        "parent_cal_train_mae": mae(cal_tr, ytr),
        "y_lt_minus3": {
            "train": seg(ytr, cal_tr, ytr < -3.0),
            "valid": seg(yva, cal_va, yva < -3.0),
            "valid_ex_id172": seg(yva, cal_va, (yva < -3.0) & ex),
        },
    }
    # binned by fixed train-tail target quantiles
    tm = ytr < -3.0
    edges = np.quantile(ytr[tm], [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for k in range(4):
        lo, hi = edges[k], edges[k + 1]
        mm = (ytr >= lo) & (ytr <= hi) if k == 3 else (ytr >= lo) & (ytr < hi)
        vm = (yva >= lo) & (yva <= hi) if k == 3 else (yva >= lo) & (yva < hi)
        bins.append({
            "bin": k, "lo": float(lo), "hi": float(hi),
            "train": seg(ytr, cal_tr, mm),
            "valid": seg(yva, cal_va, vm),
            "valid_ex_id172": seg(yva, cal_va, vm & ex),
        })
    res["train_tail_quantile_bins"] = bins

    # error budget
    dec = float(np.quantile(ytr, 0.1))
    dm = yva <= dec
    abs_err = np.abs(cal_va - yva)
    res["error_budget"] = {
        "lowest_train_decile_threshold": dec,
        "n_valid_in_decile": int(dm.sum()),
        "decile_valid_mae": mae(cal_va[dm], yva[dm]) if dm.sum() else None,
        "decile_share_of_abs_error_sum": float(abs_err[dm].sum() / abs_err.sum()),
        "decile_contribution_to_overall_mae": float(abs_err[dm].sum() / len(yva)),
        "overall_valid_mae": mae(cal_va, yva),
        "residual_mae_if_decile_perfect": float((abs_err.sum() - abs_err[dm].sum()) / len(yva)),
        "central_90_valid_mae": mae(cal_va[~dm], yva[~dm]),
    }
    tail_mask = yva < float(np.quantile(ytr, 0.05))
    res["tail_p5_budget"] = {
        "train_p5_threshold": float(np.quantile(ytr, 0.05)),
        "n_valid_tail": int(tail_mask.sum()),
        "tail_valid_mae": mae(cal_va[tail_mask], yva[tail_mask]) if tail_mask.sum() else None,
        "tail_share_of_abs_error_sum": float(abs_err[tail_mask].sum() / abs_err.sum()),
        "tail_contribution_to_overall_mae": float(abs_err[tail_mask].sum() / len(yva)),
        "tail_ex_id172_mae": mae(cal_va[tail_mask & ex], yva[tail_mask & ex]) if (tail_mask & ex).sum() else None,
        "n_tail_ex_id172": int((tail_mask & ex).sum()),
    }
    return res, b


def node_member_check():
    ck = REPO / "tracks/ksvd/results/e2e_dictenv_scale_v1/checkpoints"
    files = {"init_raw": ck / "SCALE-FULL-seed0_raw_state.pt",
             "final": ck / "SCALE-FULL-seed0_final_state.pt",
             "soup": ck / "SCALE-FULL-seed0_soup_state.pt"}
    keys = ("W_A_S", "W_A_C", "node_encoder.0.weight")
    out = {"note": "only init/final/soup checkpoints are on disk; individual soup "
                   "members [268,276,289,299,317] were not saved", "files": {}}
    for name, path in files.items():
        if not path.exists():
            out["files"][name] = "missing"
            continue
        st = torch.load(path, map_location="cpu", weights_only=False)
        rec = {}
        for k in keys:
            if k in st:
                w = st[k].float()
                rec[k] = {"absmax": float(w.abs().max()), "norm": float(w.norm()),
                          "n_zero": int((w == 0).sum()), "numel": int(w.numel())}
        out["files"][name] = {"present": True, "n_params": len(st), "weights": rec}
    return out


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    torch.set_num_threads(THREADS)
    log, results = {}, {}
    feats, ident = build_features(log)

    ytr, yva = feats["ytr"], feats["yva"]
    ids_v = feats["ids_v"]
    Rtr, Rva = feats["Rtr"], feats["Rva"]
    ztr, zva = feats["ztr"], feats["zva"]
    Qtr, Qva = feats["Qtr"], feats["Qva"]
    b_parent = float(np.median(ytr - feats["ptr"]))
    dec = float(np.quantile(ytr, 0.1))
    decile_mask = yva <= dec
    parent_cal = feats["pva"] + b_parent

    blocks = {
        "A": (Rtr, Rva),
        "B": (np.concatenate([Rtr, ztr], 1), np.concatenate([Rva, zva], 1)),
        "C": (np.concatenate([Rtr, Qtr], 1), np.concatenate([Rva, Qva], 1)),
    }
    max_width = blocks["B"][0].shape[1]

    # ---- speed test on widest config B seed 0 (no valid inspection) ----
    inits0 = make_shared_inits(0, max_width)
    t0 = time.time()
    _, _, _, _ = train_head(blocks["B"][0], ytr, blocks["B"][1], yva,
                            blocks["B"][0].shape[1], inits0, epochs=3, seed=0)
    sec_per_epoch = (time.time() - t0) / 3.0
    epochs = int(min(MAX_EPOCHS, max(1, np.floor(TARGET_FIT_SECONDS / sec_per_epoch))))
    results["frozen"] = {
        "speed_test_seconds_per_epoch": sec_per_epoch,
        "frozen_epochs": epochs,
        "frozen_before_valid_inspection": True,
        "max_epochs_cap": MAX_EPOCHS,
        "hard_fit_cap_seconds": HARD_FIT_SECONDS,
    }

    # ---- fits ----
    valid_pairs = {"graph_id": ids_v.tolist(), "y": yva.tolist(),
                   "parent_calibrated": parent_cal.tolist()}
    runs = {}
    for cfg in ("A", "B", "C"):
        Xtr, Xva = blocks[cfg]
        in_dim = Xtr.shape[1]
        for seed in SEEDS:
            inits = make_shared_inits(seed, max_width)
            ptr_hat, pva_hat, elapsed, lastloss = train_head(
                Xtr, ytr, Xva, yva, in_dim, inits, epochs, seed)
            b = float(np.median(ytr - ptr_hat))
            cal_tr, cal_va = ptr_hat + b, pva_hat + b
            rec = {
                "raw_train_mae": mae(ptr_hat, ytr),
                "raw_valid_mae": mae(pva_hat, yva),
                "calibrated_train_mae": mae(cal_tr, ytr),
                "calibrated_valid_mae": mae(cal_va, yva),
                "bias_train_median": b,
                "final_batch_train_l1": lastloss,
                "seconds": elapsed,
                "in_dim": in_dim,
                "epochs": epochs,
                "finite": bool(np.all(np.isfinite(ptr_hat)) and np.all(np.isfinite(pva_hat))),
                "gain_vs_parent_cal_valid": PARENT_CAL_VALID - mae(cal_va, yva),
                **group_delta(cal_va, parent_cal, yva, ids_v, decile_mask),
            }
            runs[f"{cfg}_seed{seed}"] = rec
            valid_pairs[f"{cfg}_seed{seed}_raw"] = pva_hat.tolist()
            valid_pairs[f"{cfg}_seed{seed}_calibrated"] = cal_va.tolist()
            print(f"[probe] {cfg} seed{seed} cal_valid={rec['calibrated_valid_mae']:.6f} "
                  f"gain={rec['gain_vs_parent_cal_valid']:+.6f} ({elapsed:.1f}s)", flush=True)
    results["runs"] = runs

    # two-seed aggregates
    agg = {}
    for cfg in ("A", "B", "C"):
        g = [runs[f"{cfg}_seed{s}"]["gain_vs_parent_cal_valid"] for s in SEEDS]
        m = [runs[f"{cfg}_seed{s}"]["calibrated_valid_mae"] for s in SEEDS]
        agg[cfg] = {"mean_cal_valid": float(np.mean(m)), "mean_gain_vs_parent": float(np.mean(g)),
                    "gains": g, "both_positive": bool(all(x > 0 for x in g))}
    for pair, base_name, cand_name in (("B_minus_A", "A", "B"), ("B_minus_C", "C", "B")):
        d = [runs[f"{base_name}_seed{s}"]["calibrated_valid_mae"]
             - runs[f"{cand_name}_seed{s}"]["calibrated_valid_mae"] for s in SEEDS]
        agg[pair] = {"gains": d, "mean_gain": float(np.mean(d)),
                     "both_same_sign_positive": bool(all(x > 0 for x in d))}
    results["two_seed_aggregates"] = agg

    # ---- tail recheck (parent prediction) ----
    tail, b = tail_recheck(feats["ptr"], ytr, feats["pva"], yva, ids_v)
    results["tail_recheck"] = tail
    results["identity_checks"] = ident
    results["feature_log"] = log
    results["constants"] = {
        "parent_cal_valid_mae": PARENT_CAL_VALID, "parent_raw_valid_mae": PARENT_RAW_VALID,
        "parent_bias_train_median": b_parent, "gate_gain": GATE,
        "seeds": list(SEEDS), "official_test_loaded": False, "no_backbone_training": True,
    }

    # ---- node member check ----
    results["node_member_check"] = node_member_check()
    results["wall_clock_seconds"] = time.time() - t_start

    (OUT / "nonlinear_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # paired valid CSV
    import csv
    cols = ["graph_id", "y", "parent_calibrated"]
    for cfg in ("A", "B", "C"):
        for s in SEEDS:
            cols += [f"{cfg}_seed{s}_raw", f"{cfg}_seed{s}_calibrated"]
    with open(OUT / "valid_paired.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for i in range(len(ids_v)):
            w.writerow([valid_pairs[c][i] for c in cols])
    print(json.dumps({"aggregates": agg, "frozen": results["frozen"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()