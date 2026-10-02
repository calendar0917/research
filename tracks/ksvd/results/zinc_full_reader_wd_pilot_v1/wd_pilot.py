"""Reader-weight-decay paired pilot stages (registered, no test access).

Reuses the frozen warm protocol in
``tracks/ksvd/experiments/luyin16/zinc_upstream_portfolio_v1.py`` verbatim
except for one line: the Adam optimiser is built with an explicit two-group
parameter split (non-reader / reader) so that the reader group's coupled L2
weight decay can be multiplied.  With ``reader_wd_mult == 1`` this is
mathematically identical to the original single-group Adam protocol.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_upstream_portfolio_v1 as upcfg
from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as up

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
HANDOFF = REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff"
CYCLE = REPO / "tracks/ksvd/results/zinc_long_cycle_audit"
CACHE_DIR = OUT / "cache"


def _reader_param_names(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if name.startswith("reader.")]


def build_param_groups(model: torch.nn.Module, reader_wd_mult: float):
    """Explicit {non-reader, reader} groups; every parameter appears exactly once."""
    names = dict(model.named_parameters())
    reader = [(n, p) for n, p in names.items() if n.startswith("reader.")]
    other = [(n, p) for n, p in names.items() if not n.startswith("reader.")]
    assert len(reader) + len(other) == len(names)
    assert {n for n, _ in reader} | {n for n, _ in other} == set(names)
    groups = [
        {"params": [p for _, p in other], "weight_decay": up.WEIGHT_DECAY},
        {"params": [p for _, p in reader], "weight_decay": up.WEIGHT_DECAY * float(reader_wd_mult)},
    ]
    return groups, [n for n, _ in reader], [n for n, _ in other]


def train_arm_wd(arm, model, train_data, valid_data, *, epochs, lr, device, soup_mode,
                 reader_wd_mult, log=True):
    up._seed_everything(up.SEED)
    model.to(device)
    groups, reader_names, other_names = build_param_groups(model, reader_wd_mult)
    optimizer = torch.optim.Adam(groups, lr=float(lr))
    loader = p1run.p1.make_env_loader(train_data, up.BATCH_SIZE, True,
                                      up.SEED + up.TRAIN_SHUFFLE_OFFSET)
    eval_loader = p1run.p1.make_env_loader(valid_data, up.BATCH_SIZE, False,
                                           up.SEED + up.EVAL_SHUFFLE_OFFSET)
    mask = cm.C6_MASK
    lam = float(cm.H1_LAMBDA)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    epoch_seconds: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum = 0.0
        n_molecules = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), up.GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_molecules += int(batch.y.numel())
        model.eval()
        with torch.no_grad():
            preds, tgts = [], []
            for batch in eval_loader:
                batch = batch.to(device)
                preds.append(model(batch, mask=mask).view(-1).detach().cpu())
                tgts.append(batch.y.view(-1).detach().cpu())
        prediction = torch.cat(preds); target = torch.cat(tgts)
        valid_mae = float((prediction - target).abs().mean())
        train_mae = float(task_sum / max(n_molecules, 1))
        epoch_seconds.append(float(time.perf_counter() - epoch_started))
        curve.append({"epoch": int(epoch), "train_mae": train_mae, "valid_mae": valid_mae,
                      "seconds": epoch_seconds[-1]})
        up._keep_soup(soup, epoch, model.state_dict(), curve, soup_mode)
        if log and (epoch == 1 or epoch % 5 == 0 or epoch == int(epochs)):
            print(f"[{arm}] epoch={epoch:03d} train={train_mae:.6f} valid={valid_mae:.6f} "
                  f"{epoch_seconds[-1]:.1f}s", flush=True)
    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0)
                  for k in soup[members[0]]}
    fresh = up.make_model("control", up._dictionary_tensor(), up._load_parent_subspace(),
                          r3_scales=(1.0, 1.0, 1.0), drop_p=upcfg.DROP_P)
    fresh.to(device)
    fresh.load_state_dict(soup_state)
    raw_predictions, targets = up.collect_predictions(fresh, valid_data, device, mask)
    calibration = up.train_median_bias(fresh, train_data, device, mask)
    calibrated_predictions, _ = up.collect_predictions(fresh, valid_data, device, mask)
    # reader norms at init / last / soup
    def rnorm(state):
        return {n: float(state[n].float().norm()) for n in reader_names if n in state}
    return {
        "arm": arm, "epochs": int(epochs), "lr": float(lr), "soup_mode": soup_mode,
        "reader_wd_mult": float(reader_wd_mult),
        "reader_param_names": reader_names, "n_reader_params": len(reader_names),
        "n_other_params": len(other_names),
        "members": members,
        "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
        "curve": curve,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(sum(epoch_seconds) / max(len(epoch_seconds), 1)),
        "raw_valid_mae": float(np.mean(np.abs(raw_predictions - targets))),
        "calibrated_valid_mae": float(np.mean(np.abs(calibrated_predictions - targets))),
        "calibration": calibration,
        "reader_norms_init": rnorm(model.state_dict()),
        "reader_norms_soup": rnorm(soup_state),
        "soup_state": soup_state,
        "valid_predictions_raw": raw_predictions.tolist(),
        "valid_predictions_calibrated": calibrated_predictions.tolist(),
        "valid_targets": targets.tolist(),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# four-forward reference decomposition (corrects the prior strong attribution)
# ---------------------------------------------------------------------------
def four_forwards() -> dict[str, Any]:
    v = np.load(HANDOFF / "valid.npz", allow_pickle=True)
    t = np.load(HANDOFF / "train.npz", allow_pickle=True)
    rd = {k: np.load(HANDOFF / "reader.npz")[k].astype(np.float64)
          for k in np.load(HANDOFF / "reader.npz").files}
    TOPO = slice(806, 814)

    def reader(R):
        h1 = np.maximum(R @ rd["W1"].T + rd["b1"], 0.0)
        h2 = np.maximum(h1 @ rd["W2"].T + rd["b2"], 0.0)
        return (h2 @ rd["W3"].T + rd["b3"]).ravel()

    t_mean = t["topo_model_input"].astype(np.float64).mean(0)
    T0 = np.maximum(t_mean @ rd["topology_W1"].T + rd["topology_b1"], 0.0) @ rd["topology_W2"].T + rd["topology_b2"]
    i_t = int(np.where(t["ids"] == 3776)[0][0]); i_v = 172
    Rt = t["R"][i_t].astype(np.float64); Rv = v["R"][i_v].astype(np.float64)
    Rt0, Rv0 = Rt.copy(), Rv.copy(); Rt0[TOPO] = T0; Rv0[TOPO] = T0
    p_t, p_v = float(reader(Rt)[0]), float(reader(Rv)[0])
    p_t0, p_v0 = float(reader(Rt0)[0]), float(reader(Rv0)[0])
    D_native, D_ref = p_v - p_t, p_v0 - p_t0
    d_v, d_t = p_v - p_v0, p_t - p_t0
    return {
        "note": "T0 = parent_topology_encoder(mean_train(topo_model_input)); NOT mean(topology8)",
        "train3776_topology25_equals_valid0172": bool(np.array_equal(
            t["topo_model_input"][i_t], v["topo_model_input"][i_v])),
        "p_t": p_t, "p_v": p_v, "p_t0": p_t0, "p_v0": p_v0,
        "D_native": D_native, "D_ref": D_ref, "delta_v": d_v, "delta_t": d_t,
        "identity_check_D_native_minus_(D_ref+delta_v-delta_t)": D_native - (D_ref + d_v - d_t),
        "T0_norm": float(np.linalg.norm(T0)),
        "train3776_y": float(t["y"][i_t]), "valid0172_y": float(v["y"][i_v]),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def timing_run(device, torch_threads: int) -> dict[str, Any]:
    torch.set_num_threads(int(torch_threads))
    train_data = up.load_split("control", "train")
    valid_data = up.load_split("control", "valid")
    _seed = up.SEED
    up._seed_everything(_seed)
    model = up.make_model("control", up._dictionary_tensor(), up._load_parent_subspace(),
                          r3_scales=(1.0, 1.0, 1.0), drop_p=upcfg.DROP_P)
    upcfg.load_parent_soup(model, "control", up.load_parent_soup_state())
    res = train_arm_wd("control", model, train_data, valid_data, epochs=1, lr=up.PILOT_LR,
                       device=device, soup_mode="last5", reader_wd_mult=1.0, log=False)
    return {"seconds_per_epoch_control_epoch1": res["seconds_per_epoch"],
            "train_epoch_seconds": res["curve"][0]["seconds"],
            "official_test_loaded": False}


def pair_run(device, h: int, reader_wd_mult: float) -> dict[str, Any]:
    train_data = up.load_split("control", "train")
    valid_data = up.load_split("control", "valid")
    soup_state = up.load_parent_soup_state()
    results = {}
    for arm, mult in (("control", 1.0), ("candidate", float(reader_wd_mult))):
        up._seed_everything(up.SEED)
        model = up.make_model("control", up._dictionary_tensor(), up._load_parent_subspace(),
                              r3_scales=(1.0, 1.0, 1.0), drop_p=upcfg.DROP_P)
        loaded = upcfg.load_parent_soup(model, "control", soup_state)
        res = train_arm_wd(arm, model, train_data, valid_data, epochs=h, lr=up.PILOT_LR,
                           device=device, soup_mode="last5", reader_wd_mult=mult)
        res["parent_load"] = loaded
        torch.save(res.pop("soup_state"), OUT / f"pilot_{arm}_soup_state.pt")
        results[arm] = res
        print(f"[pair:{arm}] mult={mult} raw={res['raw_valid_mae']:.6f} "
              f"cal={res['calibrated_valid_mae']:.6f} wall={res['wall_clock_s']:.0f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# evaluation / gate
# ---------------------------------------------------------------------------
def evaluate(results: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd
    v = np.load(HANDOFF / "valid.npz", allow_pickle=True)
    y = v["y"]
    lab = pd.read_csv(CYCLE / "valid_cycle_audit_label.csv")
    lc = lab["label_effective_cycle_snapped"].to_numpy()
    ids = v["ids"]
    g172 = ids == 172
    g1 = (lc < 0) & ~g172
    g0 = (lc == 0) & ~g172
    masks = {"G0": g0, "G1": g1, "G172": g172}

    def table(cal):
        abs_err = np.abs(cal - y)
        out = {}
        for name, m in masks.items():
            out[name] = {"n": int(m.sum()), "mae": float(abs_err[m].mean()),
                         "contribution": float(abs_err[m].sum() / len(y)),
                         "signed_mean": float((cal[m] - y[m]).mean())}
        out["overall"] = {"n": int(len(y)), "mae": float(abs_err.mean())}
        return out

    parent_raw = v["p_base"].astype(np.float64)
    b_parent = float(np.median(np.load(HANDOFF / "train_reader.npz")["y"] - np.load(HANDOFF / "train_reader.npz")["p_base"]))
    tables = {"parent": table(parent_raw + b_parent)}
    for arm in ("control", "candidate"):
        tables[arm] = table(np.asarray(results[arm]["valid_predictions_calibrated"]))
    c, k = results["control"], results["candidate"]
    cc = np.asarray(c["valid_predictions_calibrated"]); kk = np.asarray(k["valid_predictions_calibrated"])
    gain = float(np.abs(cc - y).mean() - np.abs(kk - y).mean())
    ex = np.ones(len(y), bool); ex[172] = False
    gain_ex = float(np.abs(cc - y)[ex].mean() - np.abs(kk - y)[ex].mean())
    gate = {
        "gain_total": gain, "gain_without_id172": gain_ex,
        "g0_contribution_worsening": tables["candidate"]["G0"]["contribution"] - tables["control"]["G0"]["contribution"],
        "raw_gain": float(c["raw_valid_mae"] - k["raw_valid_mae"]),
        "bias_control": float(c["calibration"]["delta"]), "bias_candidate": float(k["calibration"]["delta"]),
    }
    gate["PASS"] = bool(gate["gain_total"] >= 0.003 and gate["gain_without_id172"] > 0
                        and gate["g0_contribution_worsening"] <= 0.001)
    return {"tables": tables, "gate": gate}


def write_valid_csv(results: Mapping[str, Any]) -> None:
    import csv
    v = np.load(HANDOFF / "valid.npz", allow_pickle=True)
    y = v["y"]; ids = v["ids"]
    c = results["control"]; k = results["candidate"]
    with open(OUT / "valid_paired_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["graph_id", "y", "control_raw", "control_calibrated",
                    "candidate_raw", "candidate_calibrated"])
        for i in range(len(y)):
            w.writerow([int(ids[i]), y[i], c["valid_predictions_raw"][i],
                        c["valid_predictions_calibrated"][i],
                        k["valid_predictions_raw"][i], k["valid_predictions_calibrated"][i]])