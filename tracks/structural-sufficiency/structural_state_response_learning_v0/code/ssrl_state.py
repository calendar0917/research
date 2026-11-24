"""SSRL v0 state-response account on the best RECT checkpoint and epoch 0.

No interpreter is trained.  The frozen W3 graph is forward-run on a fixed-hash
sample of real official-Valid replacement pairs, its exact whole-graph state
``U(G) = sum_v phi(h_v)`` is assembled, and three identities are checked in
float64:

* mean pooling: ``m_new - m_old = (delta_s - m_old*delta_n)/(n_old+delta_n)``
  for both ``mean(h)`` and ``mean(h^2)``;
* attention pooling: ``delta_z = (delta_N - z*delta_Z)/(Z+delta_Z)`` for every
  head, with one fixed per-checkpoint shift ``c_k`` for all molecules;
* prediction decomposition:
  ``delta_F = delta_sum_f + delta_rho + delta_attention`` in eV.

The JSON contains three complete instances (two real replacement pairs -- the
smallest and largest absolute replacement error in the fixed sample -- and one
strict same-edit rectangle), for each checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssrl_common as C


ARMS = ("ABS", "PAIR", "RECT")


def _torch():
    import torch

    return torch


def _load_model(checkpoint: Path, device: str):
    torch = _torch()
    import ew_model as EM
    import feature_spec as FS

    spec = FS.load_spec(C.ROUND_DIR / "configs" / "pcqm_spec.json")
    norm = C.load_feature_norm()
    label_norm = C.load_label_norm()
    model, audit = EM.build_model(spec, norm, scale=3, seed=0, device=device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, payload, spec, norm, label_norm, audit


def _forward(model, cache, rows: np.ndarray, device: str):
    torch = _torch()
    import ew_collate as EC

    rows = np.asarray(rows, np.int64)
    batch = EC.collate_v2(cache, rows.tolist())
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        logits, aux = model(batch, return_aux=True)
        h = aux["h"]
        scores = torch.cat([head(h) for head in model.readout.heads], dim=1)
    return {
        "pred_z": logits[:, 0].detach().cpu().numpy().astype(np.float64),
        "h": h.detach().cpu().numpy().astype(np.float64),
        "scores": scores.detach().cpu().numpy().astype(np.float64),
        "atom_start": batch["atom_off"].detach().cpu().numpy().astype(np.int64),
        "global": np.asarray(cache["global"])[rows].astype(np.float64),
        "rows": rows,
    }


def _selection(d) -> dict[str, Any]:
    main = np.asarray(d["pair_main"], bool)
    idx = np.flatnonzero(main)
    key = np.asarray(d["pair_key"], np.int64)[idx]
    ctx = np.asarray(d["pair_ctx"], np.int64)[idx]
    pi = np.asarray(d["pair_pos_i"], np.int64)[idx]
    pj = np.asarray(d["pair_pos_j"], np.int64)[idx]
    h = np.asarray([
        C.hash64("state-pair", int(k), int(c), int(i), int(j))
        for k, c, i, j in zip(key.tolist(), ctx.tolist(), pi.tolist(), pj.tolist())
    ], np.uint64)
    # one fixed-hash observation per pair type
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    ends = np.r_[starts[1:], len(key_s)]
    chosen = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        group = order[a:b]
        chosen.append(int(group[np.argmin(h[group])]))
    chosen = np.asarray(chosen, np.int64)
    pair_hash_order = np.asarray(
        sorted(range(len(chosen)),
               key=lambda j: C.hash64("state-pair-type", int(key[chosen[j]]))),
        np.int64,
    )
    n_pair_types_available = int(len(starts))
    chosen = chosen[pair_hash_order][:128]
    return {
        "main_index": idx[chosen],
        "pair_key": key[chosen],
        "ctx": ctx[chosen],
        "pos_i": pi[chosen],
        "pos_j": pj[chosen],
        "hash": h[chosen],
        "n_pair_types_available": n_pair_types_available,
        "selection": (
            "one real evaluable Valid observation per canonical pair type; "
            "pair types and observations ordered by a fixed label-free hash; "
            "first <=128 pair types kept"
        ),
    }


def _strict_rect(d) -> dict[str, Any]:
    strict = np.asarray(d["strict_rect_index"], np.int64)
    if strict.size == 0:
        raise RuntimeError("no strict same-edit rectangles")
    rect = {k[5:]: np.asarray(v) for k, v in d.items() if k.startswith("rect_")}
    order = np.argsort(np.asarray(rect["hash"], np.uint64)[strict], kind="stable")
    i = int(strict[order[0]])
    return {
        "rect_index": i,
        "row1": int(rect["row1"][i]),
        "row2": int(rect["row2"][i]),
        "ctx1": int(rect["ctx1"][i]),
        "ctx2": int(rect["ctx2"][i]),
        "pair_key": int(rect["pair_key"][i]),
        "pos1_i": int(rect["pos1_i"][i]),
        "pos1_j": int(rect["pos1_j"][i]),
        "pos2_i": int(rect["pos2_i"][i]),
        "pos2_j": int(rect["pos2_j"][i]),
        "delta_true_1": float(rect["delta_true_1"][i]),
        "delta_true_2": float(rect["delta_true_2"][i]),
        "D_true": float(rect["delta_true_2"][i] - rect["delta_true_1"][i]),
    }


def _state_for(sel, strict, d, forward, machine):
    """Whole-graph states for all molecules used by the sample and strict rect."""
    cell_cache = np.asarray(d["cell_cache_row"], np.int64)
    rect = {k[5:]: np.asarray(v) for k, v in d.items() if k.startswith("rect_")}
    rows = set()
    for j in range(len(sel["pair_key"])):
        rows.add(int(cell_cache[sel["pos_i"][j]]))
        rows.add(int(cell_cache[sel["pos_j"][j]]))
    for key in ("pos1_i", "pos1_j", "pos2_i", "pos2_j"):
        rows.add(int(cell_cache[rect[key][strict["rect_index"]]]))
    rows = np.asarray(sorted(rows), np.int64)
    fw = forward(rows)
    machine.shift = np.max(fw["scores"], axis=0)
    U = machine.graph_state(fw["h"], fw["scores"], fw["atom_start"])
    row_to_i = {int(r): i for i, r in enumerate(rows.tolist())}
    return {
        "rows": rows,
        "row_to_i": row_to_i,
        "U": U,
        "g": fw["global"],
        "pred_z": fw["pred_z"],
        "pred_eV": fw["pred_z"] * machine.label_std_eV + machine.label_mean_eV,
        "shift": np.asarray(machine.shift, np.float64),
        "n_atoms": int(fw["h"].shape[0]),
    }


def _pair_account(sel, strict, d, state, machine, pred_eV):
    """Vectorised formula checks and the three example instances."""
    cell_cache = np.asarray(d["cell_cache_row"], np.int64)
    pair_true = np.asarray(d["pair_delta"], np.float64)
    sign = np.asarray(d["pair_sign"], np.float64)
    A = np.asarray([state["row_to_i"][int(cell_cache[i])] for i in sel["pos_i"]], np.int64)
    B = np.asarray([state["row_to_i"][int(cell_cache[j])] for j in sel["pos_j"]], np.int64)
    U_A, U_B = state["U"][A], state["U"][B]
    g_A, g_B = state["g"][A], state["g"][B]
    pA = machine.state_parts(U_A)
    pB = machine.state_parts(U_B)
    nA, nB = pA["n"], pB["n"]
    dn = nB - nA
    mean_h_A = pA["sum_h"] / np.maximum(nA, 1.0)[:, None]
    mean_h_B = pB["sum_h"] / np.maximum(nB, 1.0)[:, None]
    mean_h2_A = pA["sum_h2"] / np.maximum(nA, 1.0)[:, None]
    mean_h2_B = pB["sum_h2"] / np.maximum(nB, 1.0)[:, None]
    ds_h = pB["sum_h"] - pA["sum_h"]
    ds_h2 = pB["sum_h2"] - pA["sum_h2"]
    pred_dh = mean_h_A + (ds_h - mean_h_A * dn[:, None]) / (nA + dn)[:, None]
    pred_dh2 = mean_h2_A + (ds_h2 - mean_h2_A * dn[:, None]) / (nA + dn)[:, None]
    mean_h_err = float(np.max(np.abs(pred_dh - mean_h_B)))
    mean_h2_err = float(np.max(np.abs(pred_dh2 - mean_h2_B)))

    # attention pooling, one formula per head
    zA = [pA["P"][k] / pA["Z"][:, k:k + 1] for k in range(machine.H)]
    zB = [pB["P"][k] / pB["Z"][:, k:k + 1] for k in range(machine.H)]
    attn_err = 0.0
    for k in range(machine.H):
        dN = pB["P"][k] - pA["P"][k]
        dZ = pB["Z"][:, k] - pA["Z"][:, k]
        pred = zA[k] + (dN - zA[k] * dZ[:, None]) / (pA["Z"][:, k:k + 1] + dZ[:, None])
        attn_err = max(attn_err, float(np.max(np.abs(pred - zB[k]))))

    # prediction decomposition
    rA = machine.readout(U_A, g_A)
    rB = machine.readout(U_B, g_B)
    d_contrib = (rB["contrib"] - rA["contrib"]) * machine.label_std_eV
    d_rho = (rB["rho"] - rA["rho"]) * machine.label_std_eV
    d_attn = (rB["residual"] - rA["residual"]) * machine.label_std_eV
    dF = d_contrib + d_rho + d_attn
    dF_err = float(np.max(np.abs(dF - (rB["pred_eV"] - rA["pred_eV"]))))
    true_delta = pair_true[sel["main_index"]]
    pair_sign = sign[sel["main_index"]]
    model_delta = pair_sign * (pred_eV[B] - pred_eV[A])
    state_delta = pair_sign * (rB["pred_eV"] - rA["pred_eV"])
    model_state_err = float(np.max(np.abs(model_delta - state_delta)))
    d_contrib = pair_sign * d_contrib
    d_rho = pair_sign * d_rho
    d_attn = pair_sign * d_attn
    dF = pair_sign * dF

    # examples: smallest / largest absolute replacement error in the fixed sample
    err = np.abs(model_delta - true_delta)
    def example(j: int, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "pair_key": int(sel["pair_key"][j]),
            "A_rank": int(sel["pair_key"][j] // 64),
            "B_rank": int(sel["pair_key"][j] % 64),
            "ctx": int(sel["ctx"][j]),
            "cache_row_A": int(state["rows"][A[j]]),
            "cache_row_B": int(state["rows"][B[j]]),
            "delta_true_eV": float(true_delta[j]),
            "delta_pred_eV": float(model_delta[j]),
            "delta_state_readout_eV": float(state_delta[j]),
            "abs_error_eV": float(err[j]),
            "success_at_0.10_eV": bool(err[j] < C.DIRECTION_THRESHOLD_EV),
            "delta_n": int(dn[j]),
            "delta_U_l2": float(np.linalg.norm(U_B[j] - U_A[j])),
            "delta_U_max_abs": float(np.max(np.abs(U_B[j] - U_A[j]))),
            "delta_g_max_abs": float(np.max(np.abs(g_B[j] - g_A[j]))),
            "delta_sum_f_eV": float(d_contrib[j]),
            "delta_rho_eV": float(d_rho[j]),
            "delta_attention_eV": float(d_attn[j]),
            "delta_F_eV": float(dF[j]),
            "state_readout_reproduction_error_eV": float(abs(model_delta[j] - state_delta[j])),
        }
    lo = int(np.argmin(err))
    hi = int(np.argmax(err))
    examples = [example(lo, "real_replacement_smallest_error"),
                example(hi, "real_replacement_largest_error")]

    # strict same-edit rectangle: two contexts, four molecules
    rect = {k[5:]: np.asarray(v) for k, v in d.items() if k.startswith("rect_")}
    i = strict["rect_index"]
    def row_state(pos):
        return state["row_to_i"][int(cell_cache[pos])]
    i1a, i1b = row_state(rect["pos1_i"][i]), row_state(rect["pos1_j"][i])
    i2a, i2b = row_state(rect["pos2_i"][i]), row_state(rect["pos2_j"][i])
    s1 = machine.readout(state["U"][[i1a]], state["g"][[i1a]])
    s1b = machine.readout(state["U"][[i1b]], state["g"][[i1b]])
    s2 = machine.readout(state["U"][[i2a]], state["g"][[i2a]])
    s2b = machine.readout(state["U"][[i2b]], state["g"][[i2b]])
    sign1 = float(np.asarray(d["pair_sign"], np.float64)[strict["row1"]])
    sign2 = float(np.asarray(d["pair_sign"], np.float64)[strict["row2"]])
    def dec(a, b, sgn):
        return {
            "delta_sum_f_eV": float(sgn * (b["contrib"][0] - a["contrib"][0]) * machine.label_std_eV),
            "delta_rho_eV": float(sgn * (b["rho"][0] - a["rho"][0]) * machine.label_std_eV),
            "delta_attention_eV": float(sgn * (b["residual"][0] - a["residual"][0]) * machine.label_std_eV),
            "delta_F_eV": float(sgn * (b["pred_eV"][0] - a["pred_eV"][0])),
        }
    dec1 = dec(s1, s1b, sign1)
    dec2 = dec(s2, s2b, sign2)
    D_pred = dec2["delta_F_eV"] - dec1["delta_F_eV"]
    D_true = strict["D_true"]
    examples.append({
        "kind": "strict_same_edit_rectangle",
        "rect_index": int(i),
        "pair_key": int(strict["pair_key"]),
        "ctx1": int(strict["ctx1"]),
        "ctx2": int(strict["ctx2"]),
        "cache_row_C1_A": int(state["rows"][i1a]),
        "cache_row_C1_B": int(state["rows"][i1b]),
        "cache_row_C2_A": int(state["rows"][i2a]),
        "cache_row_C2_B": int(state["rows"][i2b]),
        "delta_true_C1_eV": float(strict["delta_true_1"]),
        "delta_true_C2_eV": float(strict["delta_true_2"]),
        "D_true_eV": float(D_true),
        "D_pred_eV": float(D_pred),
        "D_abs_error_eV": float(abs(D_pred - D_true)),
        "success_at_0.10_eV": bool(abs(D_pred - D_true) < C.DIRECTION_THRESHOLD_EV),
        "context1": dec1,
        "context2": dec2,
        "D_decomposition_eV": {
            "delta_sum_f": dec2["delta_sum_f_eV"] - dec1["delta_sum_f_eV"],
            "delta_rho": dec2["delta_rho_eV"] - dec1["delta_rho_eV"],
            "delta_attention": dec2["delta_attention_eV"] - dec1["delta_attention_eV"],
            "sum": ((dec2["delta_sum_f_eV"] - dec1["delta_sum_f_eV"])
                    + (dec2["delta_rho_eV"] - dec1["delta_rho_eV"])
                    + (dec2["delta_attention_eV"] - dec1["delta_attention_eV"])),
        },
        "strict_same_edit_screen": True,
        "note": "same canonical replacement and identical edit records; different initial states",
    })
    return {
        "n_pairs": int(len(A)),
        "formula_checks": {
            "mean_h_max_abs_error": mean_h_err,
            "mean_h2_max_abs_error": mean_h2_err,
            "mean_pooling_passes_1e-10": bool(mean_h_err < 1e-10 and mean_h2_err < 1e-10),
            "attention_max_abs_error": attn_err,
            "attention_pooling_passes_1e-10": bool(attn_err < 1e-10),
            "delta_F_max_abs_error_eV": dF_err,
            "delta_F_passes_1e-10_eV": bool(dF_err < 1e-10),
            "model_vs_state_readout_max_abs_error_eV": model_state_err,
            "model_vs_state_readout_passes_1e-4_eV": bool(model_state_err < 1e-4),
        },
        "examples": examples,
        "delta_U_l2_median": float(np.median(np.linalg.norm(U_B - U_A, axis=1))),
        "delta_g_max_abs_median": float(np.median(np.max(np.abs(g_B - g_A), axis=1))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "state_response_examples.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch-checkpoint", default=str(C.LCW_CHECKPOINT))
    parser.add_argument("--rect-checkpoint",
                        default=str(C.RUNS_DIR / "PCQM_SSRL_RECT" / "checkpoint_best_valid.pt"))
    args = parser.parse_args(argv)
    import ssa_state as SS
    import pq_data as PQD

    d = {k: v for k, v in np.load(C.DATA_DIR / "valid_eval_directory.npz").items()}
    spec = C.load_spec()
    norm = C.load_feature_norm()
    label_norm = C.load_label_norm()
    cache = PQD.load_cache(C.VALID_CACHE)
    if str(cache["meta"]["feature_spec_hash"]) != spec.hash():
        raise RuntimeError("Valid cache spec hash does not match the frozen spec")

    sel = _selection(d)
    strict = _strict_rect(d)
    checkpoints = {
        "epoch_0": Path(args.epoch_checkpoint),
        "RECT_best": Path(args.rect_checkpoint),
    }
    if not Path(checkpoints["RECT_best"]).exists():
        raise RuntimeError("RECT best checkpoint is missing; run the formal RECT arm first")
    out: dict[str, Any] = {}
    for tag, path in checkpoints.items():
        path = Path(path)
        print(f"[{tag}] loading {path}", flush=True)
        model, payload, spec_i, norm_i, label_i, audit = _load_model(path, args.device)
        machine = SS.StateMachine(model, spec_i, norm_i, label_i)
        forward = lambda rows: _forward(model, cache, rows, args.device)
        state = _state_for(sel, strict, d, forward, machine)
        account = _pair_account(sel, strict, d, state, machine, state["pred_eV"])
        out[tag] = {
            "checkpoint": C.track_rel(path),
            "checkpoint_sha256": C.sha256_file(path),
            "epoch": int(payload.get("epoch", -1)),
            "valid_mae_eV": float(payload.get("valid_mae_eV", float("nan"))),
            "arm": payload.get("arm"),
            "parameters": int(audit["total_parameters"]),
            "attention_shift_c_k": [float(x) for x in state["shift"].tolist()],
            "n_molecules_forwarded": int(len(state["rows"])),
            "n_atoms": int(state["n_atoms"]),
            "account": account,
        }
        print(f"[{tag}] formulas: mean {account['formula_checks']['mean_h_max_abs_error']:.3g}, "
              f"mean2 {account['formula_checks']['mean_h2_max_abs_error']:.3g}, "
              f"attn {account['formula_checks']['attention_max_abs_error']:.3g}, "
              f"dF {account['formula_checks']['delta_F_max_abs_error_eV']:.3g} eV",
              flush=True)

    payload = {
        "schema": "ssrl-v0-state-response-examples",
        "round": "structural-state-response-learning-v0",
        "created_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "selection": {
            "n_pairs": int(len(sel["pair_key"])),
            "n_pair_types_available": int(sel["n_pair_types_available"]),
            "rule": sel["selection"],
            "strict_rectangles_available": int(np.asarray(d["strict_rect_index"]).size),
            "strict_rectangle_rule": "first strict same-edit rectangle in fixed hash order",
            "examples": 3,
        },
        "pair_keys": [int(x) for x in sel["pair_key"].tolist()],
        "pair_contexts": [int(x) for x in sel["ctx"].tolist()],
        "strict_rect_index": int(strict["rect_index"]),
        "checkpoints": out,
        "formula": {
            "phi": "[1, f(h_v), h_v, h_v^2, exp(score_k-c_k), exp(score_k-c_k)*h_v for k=1..4]",
            "U": "sum_v phi(h_v)",
            "mean_pooling": "m_new - m_old = (delta_s - m_old*delta_n)/(n_old+delta_n)",
            "attention_pooling": "delta_z = (delta_N - z*delta_Z)/(Z+delta_Z)",
            "prediction_decomposition": "delta_F = delta_sum_f + delta_rho + delta_attention (eV)",
            "shift": "one fixed per-head constant per checkpoint, identical for every molecule",
        },
        "no_interpreter_trained": True,
        "test_labels_read": False,
    }
    C.write_json(Path(args.out), payload)
    C.write_csv(C.ANALYSIS_DIR / "state_pair_selection.csv", [
        {"pair_key": int(k), "context": int(c), "hash": int(h)}
        for k, c, h in zip(sel["pair_key"], sel["ctx"], sel["hash"])
    ], columns=["pair_key", "context", "hash"])
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
