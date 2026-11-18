"""Full-batch float64 FISTA for the grouped sparse additive arms.

The frozen objective (PROTOCOL.md section 3.1) is

    F(w, b) = ||X w + b - z||^2 / (2 n_train) + lambda * sum_r ||w_r||_2

with two columns per rule group and an unpenalised bias.  The bias is removed
analytically (centre the Train columns and the standardised labels), so the
solver only ever touches ``w``.  ``L`` is carried across epochs and doubled on
majorisation failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sq_pair_common as C
import sq_pair_rules as R


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------


@dataclass
class AdditiveSystem:
    arm: str
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    y_raw: dict[str, np.ndarray]
    z_train: np.ndarray
    scaler: dict[str, float]
    groups: list[slice]
    column_names: list[str]
    scales: np.ndarray
    matrix_hashes: dict[str, str]
    raw_train: np.ndarray
    raw_valid: np.ndarray
    raw_test: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def n_groups(self) -> int:
        return len(self.groups)


def _column_names_for(base: Sequence[str], ext: Sequence[str], arm: str) -> list[str]:
    if arm == "BASE":
        return list(base)
    if arm in ("NULL", "REAL"):
        return list(base) + [f"{arm}::{name}" for name in ext]
    raise KeyError(arm)


def load_system(prep_dir: Path, arm: str) -> AdditiveSystem:
    vocab = C.read_json(Path(prep_dir) / "rules_vocabulary.json")
    column_order = C.read_json(Path(prep_dir) / "column_order.json")
    scales = C.read_npz(Path(prep_dir) / "scales.npz")
    scaler = vocab["target_scaler"]
    common_scale = np.asarray(scales["common_scale"], dtype=np.float64)
    ext_scale = np.asarray(
        scales["real_scale"] if arm == "REAL" else scales["null_scale"], dtype=np.float64
    )
    x: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    y_raw: dict[str, np.ndarray] = {}
    row_ids: dict[str, np.ndarray] = {}
    matrix_hashes: dict[str, str] = {}
    for split in C.SPLITS:
        path = Path(prep_dir) / "matrices" / f"{split}.npz"
        payload = C.read_npz(path)
        matrix_hashes[split] = C.sha256_file(path)
        base = payload["base_raw"].astype(np.float64) / common_scale[None, :]
        if arm == "BASE":
            matrix = base
            raw_matrix = payload["base_raw"].astype(np.float64)
            names = _column_names_for(column_order["common"], [], arm)
        else:
            key = "real_raw" if arm == "REAL" else "null_raw"
            ext = payload[key].astype(np.float64) / ext_scale[None, :]
            matrix = np.concatenate([base, ext], axis=1)
            raw_matrix = np.concatenate(
                [payload["base_raw"].astype(np.float64), payload[key].astype(np.float64)], axis=1
            )
            names = _column_names_for(
                column_order["common"],
                column_order["real_extension"] if arm == "REAL" else column_order["null_extension"],
                arm,
            )
        if matrix.shape[1] != len(names):
            raise ValueError(f"{arm}/{split}: matrix width {matrix.shape[1]} != {len(names)} names")
        x[split] = np.ascontiguousarray(matrix, dtype=np.float64)
        raw[split] = np.ascontiguousarray(raw_matrix, dtype=np.float64)
        y_raw[split] = np.asarray(payload["y_raw"], dtype=np.float64)
        row_ids[split] = np.asarray(payload["row_id"], dtype=np.int64)
    if not np.array_equal(y_raw["train"], np.asarray(vocab_targets(prep_dir, "train"))):
        raise ValueError("Train labels disagree between matrix and vocabulary payload")
    z_train = (y_raw["train"] - float(scaler["mean"])) / float(scaler["std"])
    n_groups = int(x["train"].shape[1] // 2)
    return AdditiveSystem(
        arm=arm,
        x_train=x["train"],
        x_valid=x["valid"],
        x_test=x["test"],
        y_raw=y_raw,
        z_train=z_train,
        scaler=dict(scaler),
        groups=C.group_slices(n_groups),
        column_names=names,
        scales=np.concatenate([common_scale, ext_scale]) if arm != "BASE" else common_scale,
        matrix_hashes=matrix_hashes,
        raw_train=raw["train"],
        raw_valid=raw["valid"],
        raw_test=raw["test"],
    )


def vocab_targets(prep_dir: Path, split: str) -> np.ndarray:
    path = Path(prep_dir) / "matrices" / f"{split}.npz"
    return np.asarray(C.read_npz(path)["y_raw"], dtype=np.float64)


# ---------------------------------------------------------------------------
# core operators
# ---------------------------------------------------------------------------


def group_norms(w: np.ndarray, groups: Sequence[slice]) -> np.ndarray:
    return np.asarray([float(np.linalg.norm(w[g])) for g in groups], dtype=np.float64)


def group_soft_threshold(v: np.ndarray, groups: Sequence[slice], threshold: float) -> np.ndarray:
    out = np.zeros_like(v)
    for g in groups:
        norm = float(np.linalg.norm(v[g]))
        if norm <= 0.0 or threshold <= 0.0:
            out[g] = v[g]
        else:
            factor = max(0.0, 1.0 - threshold / norm)
            out[g] = factor * v[g]
    return out


def proximal_step(q: np.ndarray, grad: np.ndarray, groups: Sequence[slice], lam: float, L: float) -> np.ndarray:
    v = q - grad / float(L)
    return group_soft_threshold(v, groups, lam / float(L))


def smooth_loss(xc: np.ndarray, zc: np.ndarray, w: np.ndarray, n: int) -> float:
    residual = xc @ w - zc
    return float(residual @ residual / (2.0 * int(n)))


def lambda_max_groups(xc: np.ndarray, zc: np.ndarray, n: int, groups: Sequence[slice]) -> tuple[float, list[float]]:
    grad = -(xc.T @ zc) / float(n)
    norms = [float(np.linalg.norm(grad[g])) for g in groups]
    return (max(norms) if norms else 0.0), norms


def compute_bias(mean_x: np.ndarray, mean_z: float, w: np.ndarray, groups: Sequence[slice]) -> float:
    value = float(mean_z)
    for g in groups:
        if np.any(w[g] != 0.0):
            value -= float(mean_x[g] @ w[g])
    return float(value)


def select_minus32_groups(
    w: np.ndarray,
    groups: Sequence[slice],
    x_train: np.ndarray,
    *,
    budget: int = 32,
    tie_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Rank active groups by the Train standard deviation of their output."""
    order = list(range(len(groups))) if tie_order is None else list(tie_order)
    active = [int(gidx) for gidx in order if float(np.linalg.norm(w[groups[gidx]])) > C.ZERO_WEIGHT_EPS]
    scored = []
    for gidx in active:
        values = x_train[:, groups[gidx]] @ w[groups[gidx]]
        scored.append(
            {
                "group_index": int(gidx),
                "train_output_std": float(np.std(values, ddof=0)),
            }
        )
    scored.sort(key=lambda row: (-row["train_output_std"], row["group_index"]))
    kept = scored[: int(budget)]
    kept_indices = [row["group_index"] for row in kept]
    return {
        "active_groups": active,
        "scored": scored,
        "kept": kept_indices,
        "n_active": len(active),
        "n_kept": len(kept_indices),
        "skipped_over_budget": [row["group_index"] for row in scored[int(budget):]],
    }


def predict_standardized(x: np.ndarray, w: np.ndarray, bias: float) -> np.ndarray:
    return x @ w + float(bias)


def to_raw(pred_std: np.ndarray, scaler: Mapping[str, float]) -> np.ndarray:
    return np.asarray(pred_std, dtype=np.float64) * float(scaler["std"]) + float(scaler["mean"])


def metrics(pred_raw: np.ndarray, target_raw: np.ndarray) -> dict[str, float]:
    return {
        "rmse_raw": C.rmse(pred_raw, target_raw),
        "mae_raw": C.mae(pred_raw, target_raw),
        "n": int(np.asarray(target_raw).size),
    }


def prediction_payload(
    row_id: np.ndarray, y_raw: np.ndarray, pred_raw: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "row_id": np.asarray(row_id, dtype=np.int64),
        "y_raw": np.asarray(y_raw, dtype=np.float64),
        "pred_raw": np.asarray(pred_raw, dtype=np.float64),
    }


def save_predictions(path: Path, data: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {}
    for split in C.SPLITS:
        for key, value in data[split].items():
            payload[f"{split}_{key}"] = value
    C.save_npz(path, payload)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def fit_additive(
    *,
    prep_dir: Path,
    arm: str,
    out_dir: Path,
    lambda_max_real: float | None = None,
    epochs: int = C.EPOCHS,
    resume: bool = True,
    ledger=None,
) -> dict[str, Any]:
    out = C.ensure_dir(out_dir)
    system = load_system(prep_dir, arm)
    mean_x = system.x_train.mean(axis=0)
    mean_z = float(system.z_train.mean())
    zc = system.z_train - mean_z
    xc = system.x_train - mean_x[None, :]
    lam_max, grad_norms = lambda_max_groups(xc, zc, system.n_train, system.groups)
    if lambda_max_real is None:
        if arm != "REAL":
            raise ValueError("lambda_max_REAL must be supplied for BASE/NULL")
        lambda_max_real = lam_max
    lam = float(C.LAMBDA_FRACTION * lambda_max_real)

    signature = C.sha256_json(
        {
            "round_id": C.ROUND_ID,
            "arm": arm,
            "epochs": int(epochs),
            "lambda": lam,
            "lambda_max_real": float(lambda_max_real),
            "lambda_max_arm": float(lam_max),
            "matrix_hashes": system.matrix_hashes,
            "scales": C.sha256_array(system.scales),
            "columns": system.column_names,
            "scaler": system.scaler,
            "code": C.code_hashes(),
            "protocol": C.protocol_hash(),
        }
    )
    status_path = out / "status.json"
    if resume and status_path.is_file():
        try:
            existing = C.read_json(status_path)
        except Exception:  # noqa: BLE001
            existing = {}
        if existing.get("state") == "completed" and existing.get("signature") == signature:
            existing["resumed"] = True
            return existing

    t0 = time.perf_counter()
    resume_path = out / "resume_state.npz"
    curves_path = out / "curves.json"
    best_path = out / "best.json"
    w = np.zeros(system.n_columns, dtype=np.float64)
    q = w.copy()
    t = 1.0
    L = 1.0
    start_epoch = 1
    curves: list[dict[str, Any]] = []
    best = {
        "FULL": {"valid_rmse_raw": float("inf"), "epoch": None},
        "-32": {"valid_rmse_raw": float("inf"), "epoch": None},
    }
    if resume and resume_path.is_file() and curves_path.is_file() and best_path.is_file():
        try:
            state = C.read_npz(resume_path)
            saved_sig = str(state.get("signature", np.asarray([""]))[0]) if "signature" in state else ""
            if saved_sig == signature:
                w = np.asarray(state["w"], dtype=np.float64)
                q = np.asarray(state["q"], dtype=np.float64)
                t = float(np.asarray(state["t"]).reshape(-1)[0])
                L = float(np.asarray(state["L"]).reshape(-1)[0])
                start_epoch = int(np.asarray(state["epoch"]).reshape(-1)[0]) + 1
                curves = C.read_json(curves_path)
                best = C.read_json(best_path)
        except Exception:  # noqa: BLE001 - stale resume state is discarded, not trusted
            w = np.zeros(system.n_columns, dtype=np.float64)
            q = w.copy()
            t, L, start_epoch = 1.0, 1.0, 1
            curves, best = [], {
                "FULL": {"valid_rmse_raw": float("inf"), "epoch": None},
                "-32": {"valid_rmse_raw": float("inf"), "epoch": None},
            }

    x_splits = {"train": system.x_train, "valid": system.x_valid, "test": system.x_test}
    row_ids = {split: np.asarray(C.read_npz(Path(prep_dir) / "matrices" / f"{split}.npz")["row_id"], dtype=np.int64) for split in C.SPLITS}

    def evaluate_state(w_vec: np.ndarray, kept: Sequence[int] | None) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
        groups = [system.groups[g] for g in kept] if kept is not None else list(system.groups)
        if kept is None:
            bias_std = compute_bias(mean_x, mean_z, w_vec, system.groups)
        else:
            bias_std = float(mean_z) - float(
                np.concatenate([mean_x[g] for g in groups]) @ np.concatenate([w_vec[g] for g in groups])
            )
        preds: dict[str, dict[str, np.ndarray]] = {}
        scores: dict[str, dict[str, float]] = {}
        for split in C.SPLITS:
            x = x_splits[split]
            if kept is None:
                pred_std = predict_standardized(x, w_vec, bias_std)
            else:
                pred_std = np.asarray(bias_std) + sum((x[:, g] @ w_vec[g] for g in groups), start=np.zeros(x.shape[0]))
            pred_raw = to_raw(pred_std, system.scaler)
            preds[split] = prediction_payload(row_ids[split], system.y_raw[split], pred_raw)
            scores[split] = metrics(pred_raw, system.y_raw[split])
        return preds, scores

    def save_best(tag: str, preds: Mapping[str, Mapping[str, np.ndarray]]) -> None:
        save_predictions(out / f"best_{tag}_predictions.npz", preds)

    for epoch in range(start_epoch, int(epochs) + 1):
        t_epoch = time.perf_counter()
        grad = xc.T @ (xc @ q - zc) / float(system.n_train)
        backtrack = 0
        while True:
            w_new = proximal_step(q, grad, system.groups, lam, L)
            f_new = smooth_loss(xc, zc, w_new, system.n_train)
            diff = w_new - q
            f_q = smooth_loss(xc, zc, q, system.n_train)
            upper = f_q + float(grad @ diff) + 0.5 * float(L) * float(diff @ diff) + 1e-12
            if f_new <= upper:
                break
            L *= 2.0
            backtrack += 1
            if backtrack > 80:
                raise RuntimeError("FISTA backtracking failed to find a finite majoriser")
        norms = group_norms(w_new, system.groups)
        penalty = float(lam * norms.sum())
        objective = float(f_new + penalty)
        if not np.isfinite(objective) or not np.isfinite(f_new) or not np.isfinite(penalty):
            raise RuntimeError(f"{arm}: non-finite objective at epoch {epoch}")
        full_preds, full_scores = evaluate_state(w_new, None)
        selection = select_minus32_groups(w_new, system.groups, system.x_train)
        kept = selection["kept"]
        minus_preds, minus_scores = evaluate_state(w_new, kept)
        row = {
            "epoch": int(epoch),
            "smooth_loss": float(f_new),
            "penalty": penalty,
            "objective": objective,
            "L": float(L),
            "n_backtrack": int(backtrack),
            "n_active_groups": int(selection["n_active"]),
            "n_groups_total": int(system.n_groups),
            "n_kept_-32": int(selection["n_kept"]),
        }
        for split in C.SPLITS:
            for readout, scores in (("FULL", full_scores), ("-32", minus_scores)):
                row[f"{readout}::{split}_rmse_raw"] = scores[split]["rmse_raw"]
                row[f"{readout}::{split}_mae_raw"] = scores[split]["mae_raw"]
        row["seconds"] = time.perf_counter() - t_epoch
        curves.append(row)

        if full_scores["valid"]["rmse_raw"] < float(best["FULL"]["valid_rmse_raw"]):
            best["FULL"] = {
                "valid_rmse_raw": float(full_scores["valid"]["rmse_raw"]),
                "epoch": int(epoch),
                "scores": full_scores,
                "n_active_groups": int(selection["n_active"]),
            }
            save_best("FULL", full_preds)
        if minus_scores["valid"]["rmse_raw"] < float(best["-32"]["valid_rmse_raw"]):
            best["-32"] = {
                "valid_rmse_raw": float(minus_scores["valid"]["rmse_raw"]),
                "epoch": int(epoch),
                "scores": minus_scores,
                "kept_groups": kept,
                "n_active_groups": int(selection["n_active"]),
            }
            save_best("-32", minus_preds)
            save_predictions(out / "real32_best_epoch_FULL_predictions.npz", full_preds) if arm == "REAL" else None
            C.save_npz(
                out / "best_-32_weights.npz",
                {
                    "w": w_new,
                    "kept_groups": np.asarray(kept, dtype=np.int64),
                    "bias_std": np.asarray(
                        [compute_bias(mean_x, mean_z, w_new, [system.groups[g] for g in kept])]
                    ),
                    "epoch": np.asarray([epoch], dtype=np.int64),
                },
            )
            C.write_json(out / "best_-32_selection.json", selection)

        # FISTA extrapolation
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        q = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        C.write_json(curves_path, curves)
        C.write_json(best_path, best)
        C.save_npz(
            resume_path,
            {
                "w": w,
                "q": q,
                "t": np.asarray([t]),
                "L": np.asarray([L]),
                "epoch": np.asarray([epoch]),
                "signature": np.asarray([signature]),
            },
        )
        if ledger is not None and (epoch % 20 == 0 or epoch == epochs):
            ledger.record(
                f"{arm}_progress",
                epoch,
                epoch=int(epoch),
                objective=objective,
                valid_full=row["FULL::valid_rmse_raw"],
                valid_minus32=row["-32::valid_rmse_raw"],
            )

    final_full, final_full_scores = evaluate_state(w, None)
    selection = select_minus32_groups(w, system.groups, system.x_train)
    final_minus, final_minus_scores = evaluate_state(w, selection["kept"])
    save_predictions(out / "epoch180_FULL_predictions.npz", final_full)
    save_predictions(out / "epoch180_-32_predictions.npz", final_minus)
    C.write_json(out / "curves.json", curves)
    C.write_json(out / "best.json", best)
    summary = {
        "state": "completed",
        "signature": signature,
        "arm": arm,
        "epochs": int(epochs),
        "n_train": int(system.n_train),
        "n_columns": int(system.n_columns),
        "n_groups": int(system.n_groups),
        "n_common_groups": int(C.read_json(Path(prep_dir) / "column_order.json")["n_common_groups"]),
        "lambda": lam,
        "lambda_max_real": float(lambda_max_real),
        "lambda_max_arm": float(lam_max),
        "lambda_fraction": C.LAMBDA_FRACTION,
        "best": best,
        "epoch180": {
            "FULL": final_full_scores,
            "-32": final_minus_scores,
            "n_active_groups": int(selection["n_active"]),
            "n_kept_-32": int(selection["n_kept"]),
        },
        "curve_last": curves[-1],
        "train_seconds": float(sum(row["seconds"] for row in curves)),
        "fit_seconds": time.perf_counter() - t0,
        "matrix_hashes": system.matrix_hashes,
        "scales_sha256": C.sha256_array(system.scales),
        "columns": system.column_names,
        "environment": C.environment(),
        "run_provenance": C.run_provenance(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_json(status_path, summary)
    _write_curve_csv(out / "curve.csv", curves)
    return summary


def _write_curve_csv(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "epoch",
        "smooth_loss",
        "penalty",
        "objective",
        "L",
        "n_backtrack",
        "n_active_groups",
        "n_groups_total",
        "n_kept_-32",
    ]
    for readout in C.READOUTS:
        for split in C.SPLITS:
            columns.append(f"{readout}::{split}_rmse_raw")
            columns.append(f"{readout}::{split}_mae_raw")
    columns.append("seconds")
    C.write_csv(path, curves, columns=columns)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_proximal_closed_form() -> dict[str, Any]:
    groups = [slice(0, 2), slice(2, 4)]
    v = np.asarray([3.0, 4.0, 0.0, 0.0])
    out = group_soft_threshold(v, groups, 1.0)  # L = 1, lambda = 1
    expected = np.asarray([2.4, 3.2, 0.0, 0.0])
    return {
        "input": v.tolist(),
        "lambda_over_L": 1.0,
        "output": out.tolist(),
        "expected": expected.tolist(),
        "zero_group_stays_zero": bool(np.all(out[2:4] == 0.0)),
        "max_abs_error": float(np.abs(out - expected).max()),
        "pass": bool(np.allclose(out, expected, atol=1e-15)),
    }


def check_bias_and_budget() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n, n_groups = 12, 6
    x = rng.normal(size=(n, 2 * n_groups))
    groups = C.group_slices(n_groups)
    w = np.zeros(2 * n_groups)
    scale = np.asarray([0.2, 0.5, 0.9, 0.1, 0.4, 0.0, 0.7, 0.3, 0.8, 0.6, 0.0, 0.0])
    for gidx in range(n_groups):
        w[groups[gidx]] = scale[2 * gidx : 2 * gidx + 2]
    z = rng.normal(size=n)
    mean_x = x.mean(axis=0)
    mean_z = float(z.mean())
    full_bias = compute_bias(mean_x, mean_z, w, groups)
    full_pred = x @ w + full_bias
    if not np.allclose(full_pred, mean_z + (x - mean_x) @ w, atol=1e-12):
        raise ValueError("full bias recomputation disagrees")
    selection = select_minus32_groups(w, groups, x, budget=4)
    kept = selection["kept"]
    minus_bias = float(mean_z) - float(
        np.concatenate([mean_x[groups[g]] for g in kept]) @ np.concatenate([w[groups[g]] for g in kept])
    )
    minus_pred = np.asarray(minus_bias) + sum(
        (x[:, groups[g]] @ w[groups[g]] for g in kept), start=np.zeros(n)
    )
    rebuilt = np.asarray(minus_bias) + np.column_stack([x[:, groups[g]] for g in kept]) @ np.concatenate(
        [w[groups[g]] for g in kept]
    )
    expected_order = sorted(
        range(n_groups),
        key=lambda g: (-float(np.std(x[:, groups[g]] @ w[groups[g]], ddof=0)), g),
    )[:4]
    return {
        "full_bias": full_bias,
        "full_bias_matches_centered": bool(np.allclose(full_pred, mean_z + (x - mean_x) @ w, atol=1e-12)),
        "kept_groups": kept,
        "expected_kept_groups": expected_order,
        "minus_bias": minus_bias,
        "minus_bias_matches_rebuild": bool(np.allclose(minus_pred, rebuilt, atol=1e-12)),
        "selected_order_matches_std": bool(kept == expected_order),
        "pass": bool(
            np.allclose(full_pred, mean_z + (x - mean_x) @ w, atol=1e-12)
            and np.allclose(minus_pred, rebuilt, atol=1e-12)
            and kept == expected_order
        ),
    }
