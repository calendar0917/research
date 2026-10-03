"""ZINC overnight tail branch (CPU): severity-balanced cycle head, seed 0.

Companion of ``zinc_overnight_interface_and_tail_seed0_v1``.  The frozen 8k
chemistry ``h_raw`` (``O_seed0`` g-predictor) and the *actual* frozen Full
``topology25`` input are reused; no Full model is retrained.  Two heads are
trained from the same seed-0 initialisation and the same data schedule:

* ``q_U`` — the original unweighted mean ``L1(c)``;
* ``q_B`` — severity-group balanced weighted mean ``L1(c)`` with fixed weights
  ``w_g = N_fit / (G * n_g)`` over the fit-side ``k`` groups
  (``k=0``, ``k=-1``, ``k=-2``, ``k<=-3``); the sample weights average to 1.

Inference is ``P = h_raw + q(T25) + b_P``; ``b_P`` is one fit-median
calibration per head.  ``c`` is only a training label / score, never an
inference input.  Exact-equal-input conflict analysis on ``T25`` is train-only
and non-training.

Official-valid/test are never loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

PROTOCOL_VERSION = "zinc-overnight-interface-and-tail-seed0-v1-cpu"

TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_overnight_interface_and_tail_seed0_v1"
FROZEN_CYCLE = TRACK_ROOT / "results/zinc_frozen_chemistry_learned_cycle_v1"
FROZEN_O = TRACK_ROOT / "results/zinc_full_cycle_target_decomposition_v1"
CONFIRMATION = TRACK_ROOT / "results/zinc_full_decomposition_valid_test_confirmation_v1"

SEED = 0
EPOCHS = 300
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TOPOLOGY_IN = 25
HIDDEN = (64, 32)
HEAD_PARAMETERS = 3777
TRAIN_GEN_BASE = 20261003
BOOT_SEED = 20261003
N_BOOT = 1000
DELTA = 0.003
G0_TOL = 0.001


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def _sha_arr(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def median(values: np.ndarray) -> float:
    return float(np.median(np.asarray(values, np.float64)))


def build_head(seed: int, bias_value: float) -> nn.Module:
    torch.manual_seed(int(seed))
    head = nn.Sequential(
        nn.Linear(TOPOLOGY_IN, HIDDEN[0]),
        nn.SiLU(),
        nn.Linear(HIDDEN[0], HIDDEN[1]),
        nn.SiLU(),
        nn.Linear(HIDDEN[1], 1),
    )
    n_params = int(sum(p.numel() for p in head.parameters()))
    if n_params != HEAD_PARAMETERS:
        raise RuntimeError(f"cycle head parameter audit failed: {n_params}")
    nn.init.zeros_(head[4].weight)
    with torch.no_grad():
        head[4].bias.copy_(torch.tensor(float(bias_value)))
    return head


def head_forward(head: nn.Module, T: torch.Tensor) -> torch.Tensor:
    return head(T).view(-1)


def severity_group(k: int) -> str:
    if int(k) == 0:
        return "k=0"
    if int(k) == -1:
        return "k=-1"
    if int(k) == -2:
        return "k=-2"
    return "k<=-3"


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "k=0": k == 0,
        "k=-1": k == -1,
        "k=-2": k == -2,
        "k<=-3": k <= -3,
        "k<=-2": k <= -2,
    }


def make_weights(k_fit: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    groups = {"k=0": k_fit == 0, "k=-1": k_fit == -1, "k=-2": k_fit == -2, "k<=-3": k_fit <= -3}
    present = {name: int(mask.sum()) for name, mask in groups.items() if int(mask.sum()) > 0}
    n_fit = int(k_fit.shape[0])
    g = len(present)
    weights = np.zeros(n_fit, np.float64)
    for name, n_g in present.items():
        weights[groups[name]] = n_fit / (g * n_g)
    effective = float(weights.sum() ** 2 / np.square(weights).sum())
    return weights, {
        "group_counts": present,
        "n_nonempty_groups": int(g),
        "max_weight": float(weights.max()),
        "mean_weight": float(weights.mean()),
        "effective_sample_size": effective,
        "note": "weights fixed on the fit side; mean weight == 1",
    }


def train_head(
    T_fit: np.ndarray,
    c_fit: np.ndarray,
    weights: np.ndarray,
    *,
    bias_value: float,
    seed: int,
    epochs: int = EPOCHS,
    log: Any = print,
) -> dict[str, Any]:
    n = int(T_fit.shape[0])
    head = build_head(seed, bias_value)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    init_hash = _state_hash(head)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(int(TRAIN_GEN_BASE) + int(seed))
    T_t = torch.as_tensor(T_fit, dtype=torch.float32)
    c_t = torch.as_tensor(c_fit, dtype=torch.float32)
    w_t = torch.as_tensor(weights, dtype=torch.float32)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        abs_sum, wsum, grad_norm = 0.0, 0.0, 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            prediction = head_forward(head, T_t[idx])
            err = (prediction - c_t[idx]).abs()
            if w_t is None:
                loss = err.mean()
            else:
                loss = (w_t[idx] * err).sum() / w_t[idx].sum().clamp_min(1.0e-12)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            abs_sum += float(err.sum())
            wsum += float(idx.numel())
            steps += 1
        if epoch >= int(epochs) - SOUP_LAST + 1:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append(
            {
                "epoch": int(epoch),
                "train_l1": abs_sum / max(wsum, 1.0),
                "grad_norm": grad_norm,
                "seconds": float(time.perf_counter() - started),
            }
        )
    soup_mean = {key: torch.stack([soup[e][key] for e in sorted(soup)], dim=0).mean(dim=0) for key in soup[max(soup)]}
    head.load_state_dict(soup_mean)
    head.eval()
    with torch.no_grad():
        q_fit = head_forward(head, T_t).numpy().astype(np.float64)
        q_all = head_forward(head, torch.as_tensor(T_fit, dtype=torch.float32)).numpy().astype(np.float64)
    return {
        "head": head,
        "init_state": init_state,
        "soup_mean": soup_mean,
        "init_hash": init_hash,
        "soup_hash": _state_hash(head),
        "curve": curve,
        "steps": steps,
        "soup_members": sorted(soup),
        "q_fit": q_fit,
        "q_all": q_all,
        "seconds": float(time.perf_counter() - started),
    }


def metric_table(residual: np.ndarray, k: np.ndarray, name: str) -> dict[str, Any]:
    masks = group_masks(k)
    out: dict[str, Any] = {"n": int(k.shape[0]), "mae": float(np.mean(np.abs(residual)))}
    for key, mask in masks.items():
        out[key] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(residual[mask]))) if int(mask.sum()) else None,
            "contribution": float(np.abs(residual[mask]).sum() / k.shape[0]),
        }
    return out


def paired_bootstrap_gain(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    strat_masks: Sequence[np.ndarray],
    *,
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """gain = mean|a| - mean|b| (positive = b better); shared resample indices."""
    rng = np.random.default_rng(int(seed))
    point = float(np.abs(residual_a).mean() - np.abs(residual_b).mean())
    gains = np.empty(int(n_boot), np.float64)
    strat_idx = [np.where(m)[0] for m in strat_masks]
    for b in range(int(n_boot)):
        idx = np.concatenate([rng.choice(rows, size=rows.size, replace=True) for rows in strat_idx])
        gains[b] = np.abs(residual_a[idx]).mean() - np.abs(residual_b[idx]).mean()
    return {
        "point": point,
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "n_boot": int(n_boot),
        "seed": int(seed),
        "shared_indices": True,
    }


def exact_conflict_analysis(T: np.ndarray, c: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    """Exact-equal ``T25`` row classes: c conflicts and the class-median L1 bound."""
    # exact byte-equality classes (bitwise, no float tolerance)
    T = np.ascontiguousarray(T, np.float32)
    order = np.lexsort(T.T)
    Ts = T[order]
    cs = c[order]
    ks = k[order]
    new_class = np.ones(Ts.shape[0], dtype=bool)
    if Ts.shape[0] > 1:
        new_class[1:] = np.any(Ts[1:] != Ts[:-1], axis=1)
    class_id = np.cumsum(new_class) - 1
    n_classes = int(class_id[-1]) + 1
    counts = np.bincount(class_id)
    size = counts[class_id]
    multi = size >= 2
    per_row_irreducible = np.zeros(Ts.shape[0], np.float64)
    conflict_classes = 0
    multi_classes = 0
    for cid in np.unique(class_id[multi]):
        rows = class_id == cid
        vals = cs[rows]
        med = float(np.median(vals))
        per_row_irreducible[rows] = np.abs(vals - med)
        if np.unique(vals).size > 1:
            conflict_classes += 1
        multi_classes += 1
    irreducible = float(per_row_irreducible.sum() / Ts.shape[0])
    groups = group_masks(ks)
    out: dict[str, Any] = {
        "n_rows": int(Ts.shape[0]),
        "n_exact_classes": int(n_classes),
        "n_multi_classes": int(multi_classes),
        "n_conflict_classes": int(conflict_classes),
        "conflict_row_fraction": float(multi.mean()),
        "irreducible_l1_per_row": irreducible,
        "irreducible_total_l1": float(per_row_irreducible.sum()),
        "per_group": {},
    }
    for name, mask in groups.items():
        rows = mask
        n = int(rows.sum())
        out["per_group"][name] = {
            "n": n,
            "irreducible_l1_per_row": float(per_row_irreducible[rows].sum() / n) if n else None,
            "irreducible_total_l1": float(per_row_irreducible[rows].sum()),
        }
    # k<=-3 rows explicitly
    return out


def analyse(
    *,
    T_fit: np.ndarray,
    c_fit: np.ndarray,
    k_fit: np.ndarray,
    h_fit: np.ndarray,
    y_fit: np.ndarray,
    T_dev: np.ndarray,
    c_dev: np.ndarray,
    k_dev: np.ndarray,
    h_dev: np.ndarray,
    y_dev: np.ndarray,
    results: Mapping[str, Mapping[str, Any]],
    log: Any = print,
) -> dict[str, Any]:
    out: dict[str, Any] = {"arms": {}}
    b_values: dict[str, float] = {}
    for name, res in results.items():
        q_fit = np.asarray(res["q_fit"], np.float64)
        q_dev = np.asarray(res["q_dev"], np.float64)
        p_raw_fit = h_fit + q_fit
        p_raw_dev = h_dev + q_dev
        b = float(np.median(y_fit - p_raw_fit))
        b_values[name] = b
        p_cal_fit = p_raw_fit + b
        p_cal_dev = p_raw_dev + b
        out["arms"][name] = {
            "b_P": b,
            "q_fit_summary": {
                "mean": float(q_fit.mean()),
                "std": float(q_fit.std()),
                "min": float(q_fit.min()),
                "max": float(q_fit.max()),
                "slope_vs_c": float(np.polyfit(c_fit, q_fit, 1)[0]) if np.std(c_fit) > 0 else None,
            },
            "q_dev_summary": {
                "mean": float(q_dev.mean()),
                "std": float(q_dev.std()),
                "min": float(q_dev.min()),
                "max": float(q_dev.max()),
                "slope_vs_c": float(np.polyfit(c_dev, q_dev, 1)[0]) if np.std(c_dev) > 0 else None,
            },
            "q_mae_c_fit": metric_table(q_fit - c_fit, k_fit, "fit"),
            "q_mae_c_dev": metric_table(q_dev - c_dev, k_dev, "dev"),
            "P_raw_fit": metric_table(y_fit - p_raw_fit, k_fit, "fit"),
            "P_cal_fit": metric_table(y_fit - p_cal_fit, k_fit, "fit"),
            "P_raw_dev": metric_table(y_dev - p_raw_dev, k_dev, "dev"),
            "P_cal_dev": metric_table(y_dev - p_cal_dev, k_dev, "dev"),
            "mapping": "raw = h+q; cal = h+q+b_P",
        }
    strata = [k_dev == 0, k_dev == -1, k_dev <= -2]
    if "P_U" in results and "P_B" in results:
        pu_raw = h_dev + np.asarray(results["P_U"]["q_dev"], np.float64)
        pb_raw = h_dev + np.asarray(results["P_B"]["q_dev"], np.float64)
        pu_cal = pu_raw + b_values["P_U"]
        pb_cal = pb_raw + b_values["P_B"]
        out["tail_gate"] = {
            "gain_B_vs_U_overall_cal": paired_bootstrap_gain(y_dev - pu_cal, y_dev - pb_cal, strata),
            "gain_B_vs_U_overall_raw": paired_bootstrap_gain(y_dev - pu_raw, y_dev - pb_raw, strata),
            "gain_B_vs_U_g0_cal": paired_bootstrap_gain(
                (y_dev - pu_cal)[k_dev == 0], (y_dev - pb_cal)[k_dev == 0], [np.ones(int((k_dev == 0).sum()))]
            ),
            "point_g0_cal_worsening": float(
                np.abs((y_dev - pb_cal)[k_dev == 0]).mean() - np.abs((y_dev - pu_cal)[k_dev == 0]).mean()
            ),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description="ZINC overnight tail CPU branch")
    parser.add_argument("--stage", default="8k", choices=("8k", "10k"))
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if args.stage == "8k":
        t25 = np.load(FROZEN_CYCLE / "T25_all.npz", allow_pickle=True)
        T = np.asarray(t25["T"], np.float32)
        o = np.load(FROZEN_O / "O_seed0_predictions.npz", allow_pickle=True)
        fit_idx = np.asarray(o["fit_idx"], np.int64)
        dev_idx = np.asarray(o["dev_idx"], np.int64)
        y_fit, c_fit, h_fit = (np.asarray(o["fit_y"], np.float64), np.asarray(o["fit_c"], np.float64), np.asarray(o["fit_raw"], np.float64))
        y_dev, c_dev, h_dev = (np.asarray(o["dev_y"], np.float64), np.asarray(o["dev_c"], np.float64), np.asarray(o["dev_raw"], np.float64))
        with np.load(FROZEN_O / "target_decomposition.npz", allow_pickle=False) as z:
            k_all = np.asarray(z["k"], np.int64)
        k_fit, k_dev = k_all[fit_idx], k_all[dev_idx]
        T_fit, T_dev = T[fit_idx], T[dev_idx]
        provenance = {
            "T25_source": str((FROZEN_CYCLE / "T25_all.npz").relative_to(TRACK_ROOT)),
            "T25_sha256": _sha_arr(T),
            "h_raw_source": str((FROZEN_O / "O_seed0_predictions.npz").relative_to(TRACK_ROOT)),
            "fit_idx_sha256": hashlib.sha256(fit_idx.tobytes()).hexdigest(),
            "dev_idx_sha256": hashlib.sha256(dev_idx.tobytes()).hexdigest(),
            "n_fit": int(len(fit_idx)),
            "n_dev": int(len(dev_idx)),
        }
    else:
        t25 = np.load(CONFIRMATION / "T25_all.npz", allow_pickle=True)
        T = np.asarray(t25["T"], np.float32)
        h = np.load(CONFIRMATION / "H_seed0_fit_pred.npz", allow_pickle=True)
        y_all = np.asarray(h["y"], np.float64)
        c_all = np.asarray(h["c"], np.float64)
        k_all = np.asarray(h["k"], np.int64)
        h_all = np.asarray(h["fit_raw"], np.float64)
        if T.shape[0] != 10000 or y_all.shape[0] != 10000:
            raise RuntimeError("10k prep/T25 mismatch")
        T_fit, T_dev = T, T
        y_fit, y_dev = y_all, y_all
        c_fit, c_dev = c_all, c_all
        k_fit, k_dev = k_all, k_all
        h_fit, h_dev = h_all, h_all
        provenance = {
            "T25_source": str((CONFIRMATION / "T25_all.npz").relative_to(TRACK_ROOT)),
            "T25_sha256": _sha_arr(T),
            "h_raw_source": str((CONFIRMATION / "H_seed0_fit_pred.npz").relative_to(TRACK_ROOT)),
            "n_fit": 10000,
            "n_dev": None,
            "note": "all-10k training; no internal dev holdout",
        }

    conflicts = exact_conflict_analysis(T_fit, c_fit, k_fit)
    write_json(out_dir / f"cpu_conflict_{args.stage}.json", conflicts)
    print(f"[conflict] classes={conflicts['n_exact_classes']} conflict={conflicts['n_conflict_classes']} "
          f"irreducible/row={conflicts['irreducible_l1_per_row']:.6f}")

    bias_value = median(c_fit)
    weights, weight_meta = make_weights(k_fit)
    print(f"[weights] {weight_meta}")

    results: dict[str, dict[str, Any]] = {}
    for name, use_weights in (("P_U", False), ("P_B", True)):
        w = weights if use_weights else np.ones_like(weights)
        res = train_head(T_fit, c_fit, w, bias_value=bias_value, seed=SEED, log=print)
        with torch.no_grad():
            q_dev = head_forward(res["head"], torch.as_tensor(T_dev, dtype=torch.float32)).numpy().astype(np.float64)
        torch.save(res["soup_mean"], out_dir / f"cpu_{name}_head_soup_state.pt")
        torch.save(res["init_state"], out_dir / f"cpu_{name}_head_init_state.pt")
        results[name] = {
            "q_fit": res["q_fit"],
            "q_dev": q_dev,
            "init_hash": res["init_hash"],
            "soup_hash": res["soup_hash"],
            "soup_members": res["soup_members"],
            "steps": res["steps"],
            "seconds": res["seconds"],
            "weights": bool(use_weights),
        }
        np.savez_compressed(
            out_dir / f"cpu_{name}_predictions.npz",
            q_fit=res["q_fit"].astype(np.float32),
            q_dev=q_dev.astype(np.float32),
        )
        write_json(
            out_dir / f"cpu_{name}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "stage": args.stage,
                "arm": name,
                "seed": SEED,
                "epochs": EPOCHS,
                "soup_members": res["soup_members"],
                "init_hash": res["init_hash"],
                "soup_hash": res["soup_hash"],
                "bias_value_median_fit_c": bias_value,
                "weight_meta": weight_meta if use_weights else None,
                "curve_tail": res["curve"][-5:],
                "curve_first": res["curve"][:2],
                "steps": res["steps"],
                "seconds": res["seconds"],
                "provenance": provenance,
                "official_valid_loaded": False,
                "official_test_loaded": False,
            },
        )
        print(f"[{name}] done in {res['seconds']:.1f}s soup={res['soup_hash'][:12]}")

    analysis = analyse(
        T_fit=T_fit, c_fit=c_fit, k_fit=k_fit, h_fit=h_fit, y_fit=y_fit,
        T_dev=T_dev, c_dev=c_dev, k_dev=k_dev, h_dev=h_dev, y_dev=y_dev,
        results=results,
    )
    analysis["protocol_version"] = PROTOCOL_VERSION
    analysis["stage"] = args.stage
    analysis["provenance"] = provenance
    analysis["weight_meta"] = weight_meta
    analysis["conflict"] = conflicts
    analysis["official_valid_loaded"] = False
    analysis["official_test_loaded"] = False
    analysis["seconds"] = float(time.perf_counter() - t0)
    write_json(out_dir / f"cpu_analysis_{args.stage}.json", analysis)
    if "tail_gate" in analysis and args.stage == "8k":
        gate = analysis["tail_gate"]
        gain_cal = gate["gain_B_vs_U_overall_cal"]
        gain_raw = gate["gain_B_vs_U_overall_raw"]
        g0_worse = gate["point_g0_cal_worsening"]
        passed = bool(
            gain_cal["point"] >= DELTA
            and gain_cal["ci95"][0] > 0
            and gain_raw["point"] > 0
            and g0_worse <= G0_TOL
        )
        gate["gate_passed"] = passed
        gate["rule"] = {
            "dev_overall_cal_gain_ge": DELTA,
            "paired_ci_lower_gt": 0.0,
            "dev_overall_raw_gain_gt": 0.0,
            "g0_cal_worsening_le": G0_TOL,
        }
        write_json(out_dir / f"cpu_analysis_{args.stage}.json", analysis)
        print(f"[gate] passed={passed} gain_cal={gain_cal['point']:.6f} CI={gain_cal['ci95']}")
    print(f"[done] {args.stage} total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
