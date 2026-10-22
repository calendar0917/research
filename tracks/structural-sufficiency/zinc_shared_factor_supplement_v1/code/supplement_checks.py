"""Targeted checks for the seed-1 supplement (CPU only, no training).

Four checks, each answering one thing the PROTOCOL promised to verify:

1. ``init``      — seed 1 reproduces the *frozen* init/content hashes and the
                   pre-declared P3-carrier hash (D/T/N identical within the seed);
2. ``order``     — T seed 1 and N seed 1 share the same batch order for all 600
                   epochs, and seed 1 uses a different stream than seed 0;
3. ``branch``    — the frozen N checkpoint reproduces its stored valid MAE when
                   evaluated through the N factorisation, and (informational) what the
                   *same weights* give through the T factorisation: if the two agree
                   the learned state has become vacuous and the contrast is empty;
4. ``budget``    — both arms are 451,777 parameters with a zero-initialised W_s,
                   79 steps/epoch and 47,400 updates, inside the 1 h/arm cap;
5. ``collapse``  — how far apart T and N actually are at the frozen checkpoints
                   (log-block gap and its constant/pattern-varying split), the q entropy,
                   the within-channel spread of K, and how much weight the trained
                   readout puts on each content block;
6. ``ablation``   — post-hoc, on the frozen checkpoints: what happens to valid MAE when
                   the structural (log) block is zeroed or shuffled across molecules,
                   and when the q_root block is zeroed (scale reference).

Writes ``analysis/supplement_checks.json``. Nothing is fitted, no official test.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
EXP = _HERE.parent
_SF = EXP.parent / "zinc_shared_factor_v1"
for _p in (str(_SF / "code"), str(_SF / "code" / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sf_core as SC  # noqa: E402  (also puts the shared code dirs on sys.path)
import ad_common as C  # noqa: E402
import ad_data  # noqa: E402

SF_R = 4  # R_STATE, for the entropy reference only
RUNS = _SF / "runs"
ANALYSIS = EXP / "analysis"
SUPP_HASHES = EXP / "configs" / "supplement_hashes.json"
TOL = 1e-9


def check_init(cfg: dict) -> tuple[bool, dict]:
    norm = ad_data.load_norm()
    declared_carrier = C.read_json(SUPP_HASHES)["carrier_initial_state_sha256"]
    detail: dict = {}
    ok = True
    for seed in (0, 1):
        detail[str(seed)] = {}
        for arm in ("D", "T", "N"):
            model, info = SC.build(arm, norm, torch.device("cpu"), seed=seed)
            h = SC.init_hashes(model)
            exp = cfg["model"]["arms"][arm]
            same = {
                "params": int(info["parameters"]) == int(exp["parameters_expected"]),
                "init": h["init_state_sha256"] == exp["init_state_sha256"][str(seed)],
                "content": h["content_state_sha256"] == exp["content_state_sha256"][str(seed)],
                "carrier": h["public_state_sha256"] == declared_carrier[str(seed)],
                "w_s_zero": h["w_s_zero"],
            }
            ok &= all(same.values())
            detail[str(seed)][arm] = {"params": int(info["parameters"]),
                                      "carrier_sha256": h["public_state_sha256"],
                                      "matches_frozen": same}
        carriers = {detail[str(seed)][a]["carrier_sha256"] for a in ("D", "T", "N")}
        detail[str(seed)]["carrier_identical_across_arms"] = len(carriers) == 1
        ok &= len(carriers) == 1
    detail["seed1_carrier_differs_from_seed0"] = (detail["1"]["T"]["carrier_sha256"]
                                                  != detail["0"]["T"]["carrier_sha256"])
    return ok, detail


def check_order() -> tuple[bool, dict]:
    t1 = C.read_json(RUNS / "SF_T_s1" / "order_hashes.json")
    n1 = C.read_json(RUNS / "SF_N_s1" / "order_hashes.json")
    t0 = C.read_json(RUNS / "SF_T_s0" / "order_hashes.json")
    epochs_equal = t1["epochs"] == n1["epochs"]
    detail = {
        "T_s1_generator_seed": t1["generator_seed"],
        "N_s1_generator_seed": n1["generator_seed"],
        "T_s0_generator_seed": t0["generator_seed"],
        "n_epochs": len(t1["epochs"]),
        "epoch_hashes_equal_T1_N1": bool(epochs_equal),
        "n_epochs_differing": int(sum(1 for a, b in zip(t1["epochs"], n1["epochs"]) if a != b)),
        "seed1_stream_differs_from_seed0": t1["epochs"] != t0["epochs"],
        "same_n_molecules": t1["n_molecules"] == n1["n_molecules"] == t0["n_molecules"] == 10000,
    }
    return epochs_equal and detail["same_n_molecules"], detail


def check_branch(cfg: dict) -> tuple[bool, dict]:
    """Same N weights, evaluated as N and as T: must match the frozen N score, then move."""
    norm = ad_data.load_norm()
    device = torch.device("cpu")
    valid_cache, _, valid_rows = SC.load_split("valid")
    ck = torch.load(RUNS / "SF_N_s1" / "best.pt", map_location="cpu", weights_only=False)
    freeze = C.read_json(_SF / "analysis" / "SF_N_s1" / "freeze.json")
    bs = int(cfg["schedule"]["valid_batch_size"])
    out: dict = {}
    preds = {}
    for mode in ("N", "T"):
        model, _ = SC.build(mode, norm, device, seed=1)
        C.load_state(model, ck["state_dict"], device)
        preds[mode], mae = SC.evaluate(model, valid_cache, valid_rows, device, bs)
        out[mode] = {"valid_mae": float(mae)}
    out["stored_valid_mae"] = float(ck["valid_mae"])
    out["frozen_best_valid_mae"] = float(freeze["best_valid_mae"])
    out["N_reload_matches_stored"] = abs(out["N"]["valid_mae"] - float(ck["valid_mae"])) < 1e-6
    out["N_reload_matches_frozen"] = abs(out["N"]["valid_mae"]
                                         - float(freeze["best_valid_mae"])) < 1e-6
    out["delta_N_vs_T_same_weights"] = float(out["T"]["valid_mae"] - out["N"]["valid_mae"])
    out["prediction_max_abs_diff"] = float(np.abs(preds["N"] - preds["T"]).max())
    out["modes_differ_materially"] = bool(abs(out["delta_N_vs_T_same_weights"]) > 1e-3)
    out["informational_only"] = True
    ok = out["N_reload_matches_stored"] and out["N_reload_matches_frozen"]
    return ok, out


def check_budget(cfg: dict) -> tuple[bool, dict]:
    out: dict = {}
    ok = True
    for arm in ("T", "N"):
        info = C.read_json(RUNS / f"SF_{arm}_s1" / "train.json")
        steps = math.ceil(int(cfg["data"]["train"]) / int(cfg["schedule"]["batch_size"]))
        detail = {
            "params": int(info["parameter_count"]),
            "steps_per_epoch": int(info["plan"]["steps_per_epoch"]),
            "updates": int(info["plan"]["total_updates"]),
            "epochs": int(info["plan"]["epochs"]),
            "w_s_zero_at_init": bool(info["gate"]["w_s_zero"]),
            "paired_T_init_identical": bool(info["gate"].get("paired_T_init", {})
                                            .get("identical", False)),
            "gpu_hours": float(info["gpu_hours"]),
            "wall_seconds": float(info["wall_seconds"]),
        }
        good = (detail["params"] == 451777 and detail["steps_per_epoch"] == steps == 79
                and detail["updates"] == 47400 and detail["epochs"] == 600
                and detail["w_s_zero_at_init"] and detail["wall_seconds"] < 3600.0)
        if arm == "N":
            good &= detail["paired_T_init_identical"]
        detail["pass"] = bool(good)
        ok &= good
        out[arm] = detail
    out["total_gpu_hours"] = sum(out[a]["gpu_hours"] for a in ("T", "N"))
    return ok, out


def check_collapse(cfg: dict) -> tuple[bool, dict]:
    """Is the learned atomic state still *expressive*, i.e. does T differ from N?

    Two ways the shared/independent contrast can become vacuous: ``q`` collapses to
    one-hot (then the sum over the non-root state has a single term) or ``K`` becomes
    constant across states (then the edge factor is state-free).  Measured on the
    frozen valid pattern table with the *same* checkpoint evaluated both ways.
    """
    norm = ad_data.load_norm()
    device = torch.device("cpu")
    table = SC.pattern_table("valid")
    out: dict = {}
    ok = True
    for arm, seed in (("T", 0), ("T", 1), ("N", 1)):
        ck = torch.load(RUNS / f"SF_{arm}_s{seed}" / "best.pt", map_location="cpu",
                        weights_only=False)
        rec: dict = {"epoch": int(ck["epoch"]), "valid_mae": float(ck["valid_mae"])}
        blocks = {}
        for mode in ("T", "N"):
            model, _ = SC.build(mode, norm, device, seed=seed)
            C.load_state(model, ck["state_dict"], device)
            with torch.no_grad():
                x = model.branch.content_input(table).double().numpy()
                st = model.branch.state
                q = st.q_log().exp().double().numpy()
                kk = st.log_k().exp().double().numpy()
            blocks[mode] = x[:, 64:128]
            if mode == "N":
                ent = -(q * np.log(q + 1e-30)).sum(-1)
                within = kk.reshape(kk.shape[0], kk.shape[1], -1).std(axis=2)
                rec.update({
                    "q_entropy_mean": float(ent.mean()),
                    "q_entropy_max_possible": float(np.log(SF_R)),
                    "q_max_prob_mean": float(q.max(-1).mean()),
                    "k_within_channel_std_mean": float(within.mean()),
                    "k_within_channel_std_max": float(within.max()),
                    "k_min": float(kk.min()), "k_max": float(kk.max()),
                })
        w1 = ck["state_dict"]["branch.l1.weight"].double().numpy()   # [64, 133]
        per_block = {
            "q_root": (slice(0, 64), x[:, :64]),
            "logT_or_logN": (slice(64, 128), blocks["N"]),
            "family_and_counts": (slice(128, 133), x[:, 128:133]),
        }
        usage = {}
        for name, (sl, feat) in per_block.items():
            w = float(np.linalg.norm(w1[:, sl]))
            usage[name] = {"weight_l2": w,
                           "feature_std_mean": float(feat.std(axis=0).mean()),
                           "typical_preactivation": float(w * feat.std(axis=0).mean())}
        rec["readout_usage"] = usage
        d = np.abs(blocks["T"] - blocks["N"])
        rec["log_block_max_abs_gap_T_vs_N"] = float(d.max())
        rec["log_block_mean_abs_gap_T_vs_N"] = float(d.mean())
        rec["log_block_std_across_patterns"] = float(blocks["N"].std(axis=0).mean())
        rec["T_and_N_vacuous"] = bool(d.max() < 1e-3)
        # how much of the T-vs-N gap is a per-channel constant (absorbed by the
        # first linear layer's bias) versus pattern-varying information?
        delta = blocks["T"] - blocks["N"]                      # [PL, 64]
        const = delta.mean(axis=0)                             # [64]
        resid = delta - const[None, :]
        rec["gap_decomposition"] = {
            "mean_abs_constant": float(np.abs(const).mean()),
            "mean_abs_pattern_varying": float(np.abs(resid).mean()),
            "pattern_varying_share": float(np.abs(resid).mean()
                                           / max(np.abs(delta).mean(), 1e-12)),
            "pattern_varying_over_log_block_std": float(
                resid.std(axis=0).mean() / max(blocks["N"].std(axis=0).mean(), 1e-12)),
            "log_block_std_vs_qroot_block_std": float(
                blocks["N"].std(axis=0).mean() / max(x[:, :64].std(axis=0).mean(), 1e-12)),
        }
        out[f"{arm}_s{seed}"] = rec
    out["verdict"] = ("T and N are numerically the same function at "
                      + ", ".join(k for k, v in out.items()
                                  if isinstance(v, dict) and v.get("T_and_N_vacuous"))
                      if any(isinstance(v, dict) and v.get("T_and_N_vacuous")
                             for v in out.values()) else "T and N stay distinguishable")
    return ok, out


def check_ablation(cfg: dict) -> tuple[bool, dict]:
    """Post-hoc use of the frozen content blocks (no training, no re-selection).

    The pre-registered comparison asks whether sharing helps; this check asks the
    prior question of whether the *structural* block is read at all: zero it, shuffle
    it across patterns (marginal kept, correspondence destroyed), and zero the q_root
    block for scale.  Diagnostics only — nothing here is used to pick a checkpoint.
    """
    norm = ad_data.load_norm()
    device = torch.device("cpu")
    valid_cache, _, valid_rows = SC.load_split("valid")
    bs = int(cfg["schedule"]["valid_batch_size"])
    rng = np.random.default_rng(20261022)
    out: dict = {}
    for arm, seed in (("T", 0), ("T", 1), ("N", 1)):
        ck = torch.load(RUNS / f"SF_{arm}_s{seed}" / "best.pt", map_location="cpu",
                        weights_only=False)
        model, _ = SC.build(arm, norm, device, seed=seed)
        C.load_state(model, ck["state_dict"], device)
        orig = model.branch.content_input
        _, base = SC.evaluate(model, valid_cache, valid_rows, device, bs)
        rec: dict = {"baseline": float(base)}
        cases = {
            "zero_log_block": lambda b: b * 0.0,
            "shuffle_log_block": lambda b: b[torch.as_tensor(rng.permutation(b.shape[0]))],
            "zero_qroot_block": None,          # handled separately (different slice)
        }
        for name, fn in cases.items():
            def patched(table, _fn=fn, _name=name):
                x = orig(table)
                y = x.clone()
                if _name == "zero_qroot_block":
                    y[:, :64] = 0.0
                else:
                    y[:, 64:128] = _fn(x[:, 64:128])
                return y
            model.branch.content_input = patched
            _, mae = SC.evaluate(model, valid_cache, valid_rows, device, bs)
            rec[name] = {"valid_mae": float(mae), "delta_vs_baseline": float(mae - base)}
            model.branch.content_input = orig
        out[f"{arm}_s{seed}"] = rec
    out["note"] = ("informational: the frozen checkpoints are reused unchanged; "
                   "deltas are diagnostics, not selection criteria")
    return True, out


def main() -> int:
    cfg = C.read_json(_SF / "configs" / "sf_atomic_v1.json")
    checks = {}
    ok = True
    for name, fn in (("init", lambda: check_init(cfg)), ("order", check_order),
                     ("branch", lambda: check_branch(cfg)), ("budget", lambda: check_budget(cfg)),
                     ("collapse", lambda: check_collapse(cfg)),
                     ("ablation", lambda: check_ablation(cfg))):
        good, detail = fn()
        checks[name] = {**detail, "pass": bool(good)}
        ok &= good
        print(f"[supplement check {name}] {'PASS' if good else 'FAIL'}")
    doc = {"protocol": "zinc-shared-factor-supplement-v1",
           "config_sha256": C.sha256_file(_SF / "configs" / "sf_atomic_v1.json"),
           "supplement_hashes_sha256": C.sha256_file(SUPP_HASHES),
           "checks": checks, "n_pass": sum(1 for v in checks.values() if v["pass"]),
           "n_fail": sum(1 for v in checks.values() if not v["pass"]),
           "pass": bool(ok), "created": C.utcnow()}
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "supplement_checks.json", doc)
    print(f"[supplement-checks] {doc['n_pass']}/{len(checks)} pass -> "
          f"{ANALYSIS / 'supplement_checks.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
