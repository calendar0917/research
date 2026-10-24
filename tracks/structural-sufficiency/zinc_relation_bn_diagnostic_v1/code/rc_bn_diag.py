"""Fixed-checkpoint BatchNorm inference-state diagnostic (D/S relation probe).

Four evaluation states on the **same frozen weights** (no optimizer, no
backward, no parameter change):

* ``O``  original eval: the model exactly as shipped, ``model.eval()``.
* ``B``  batch-stat BN, dropout off (train-only diagnostic): every BN uses the
         current batch's statistics through an explicit
         ``F.batch_norm(x, None, None, w, b, training=True, momentum=0.0)`` in a
         forward hook, so no running buffer is ever written; every other module
         stays in eval.
* ``T``  B plus the *original* attention dropout (train-only, gated: only run
         when B leaves most of the online-vs-fixed gap unexplained).
* ``C``  eval with BN running statistics re-estimated layer by layer on train
         only (see ``calibrate``), everything else identical to O.

The official test split is never constructed.  valid is only ever *evaluated*
(eval mode, no statistics collected from it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SS = _EXP.parent
_REPO = _SS.parents[1]
for p in (
    _SS / "zinc_relation_composition_probe_v1" / "code",
    _SS / "zinc_arch_discovery_v1" / "code",
    _SS / "zinc_dual_channel_composer_v1" / "code",
):
    sys.path.insert(0, str(p))

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import rc_data as RD  # noqa: E402
import rc_models as RM  # noqa: E402

ANALYSIS = _EXP / "analysis"
SRC_RUNS = _SS / "zinc_relation_composition_probe_v1" / "runs"
CKPTS = {"best": "best.pt", "final": "final.pt"}
SUBSET_SEED = 20261024
SUBSET_SEED_2 = 20261025
SUBSET_N = 2048
DIAG_BATCH = 128
REPLAY_BATCH = 256
#: T is only run when B closes less than this share of (online - O) train gap
T_GATE_SHARE = 0.5


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------
def tensor_hash(items) -> str:
    h = hashlib.sha256()
    for name, t in items:
        h.update(name.encode())
        h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def param_hash(model: nn.Module) -> str:
    return tensor_hash(sorted(model.named_parameters()))


def bn_hash(model: nn.Module) -> str:
    keep = [(n, b) for n, b in sorted(model.named_buffers())
            if "running_" in n or "num_batches_tracked" in n]
    return tensor_hash(keep)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sha256(state) -> str:
    return tensor_hash(sorted(state.items()))


# ---------------------------------------------------------------------------
# Chan / Welford streaming moments (float64)
# ---------------------------------------------------------------------------
class Moments:
    """Streaming per-channel count / mean / M2 with pairwise Chan merging."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float64)
        if x.ndim != 2:
            raise ValueError(f"expected [N, C] BN input, got {tuple(x.shape)}")
        n_b = int(x.shape[0])
        if n_b == 0:
            return
        mean_b = x.mean(dim=0)
        m2_b = ((x - mean_b) ** 2).sum(dim=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = n_b, mean_b, m2_b
            return
        n = self.n + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + m2_b + delta * delta * (self.n * n_b / n)
        self.n = n

    def result(self, unbiased: bool = True):
        if self.n < 2:
            raise RuntimeError(f"n={self.n} < 2, refusing to write a variance")
        var = self.m2 / (self.n - 1 if unbiased else self.n)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(var).all():
            raise RuntimeError("non-finite moments")
        return self.n, self.mean, var


def _moment_selftest() -> dict:
    """Streaming moments must equal direct moments of the concatenation."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for n_batches in (2, 5):
        for n_b in (3, 64, 257):
            parts = [rng.normal(loc=0.3 * i, scale=0.7 + 0.2 * i,
                                size=(n_b, 6)).astype(np.float64)
                     for i in range(n_batches)]
            acc = Moments()
            for p in parts:
                acc.update(torch.from_numpy(p))
            n, mean, var = acc.result()
            cat = np.concatenate(parts, 0)
            worst = max(worst,
                        float(np.abs(mean.numpy() - cat.mean(0)).max()),
                        float(np.abs(var.numpy() - cat.var(0, ddof=1)).max()))
    return {"ok": worst < 1e-10, "max_abs_diff": worst,
            "note": "streaming Chan merge vs direct moments of the concatenation, "
                    "different batch means and sizes, float64"}


# ---------------------------------------------------------------------------
# evaluation states
# ---------------------------------------------------------------------------
def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def bn_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm1d)]


def bn_call_order(model, collator, cache, rows, device, *, batch_size=DIAG_BATCH,
                  hooks=None) -> list[dict]:
    """BN call order and entry counts on one fixed pass (eval mode, no writes)."""
    order: list[str] = []
    counts: dict[str, int] = {}
    shapes: dict[str, tuple] = {}
    mods = dict(bn_modules(model))

    def make_hook(name):
        def hook(module, args):
            if name not in counts:
                order.append(name)
                shapes[name] = tuple(args[0].shape)
            counts[name] = counts.get(name, 0) + int(args[0].shape[0])
        return hook

    handles = [mods[n].register_forward_pre_hook(make_hook(n)) for n in mods]
    try:
        model.eval()
        with torch.no_grad(), C.isolated_rng():
            for s in range(0, len(rows), batch_size):
                batch = _to_device(collator(cache, rows[s:s + batch_size], "RC", valid=True), device)
                model(batch)
    finally:
        for hd in handles:
            hd.remove()
    repeated = len(order) != len(counts)
    return [{"name": n, "calls": counts[n], "first_input_shape": list(shapes[n])}
            for n in order], repeated


def eval_predict(model, collator, cache, rows, device, *, batch_size=DIAG_BATCH) -> np.ndarray:
    """Plain eval-mode forward (no hooks, no batch stats)."""
    model.eval()
    out = []
    with torch.no_grad(), C.isolated_rng():
        for s in range(0, len(rows), batch_size):
            batch = _to_device(collator(cache, rows[s:s + batch_size], "RC", valid=True), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def enable_batch_stats(model) -> list:
    """B state: BN uses the batch statistics, writes no buffers."""
    handles = []
    for _n, m in bn_modules(model):

        def hook(module, args, output, _m=m):
            x = args[0]
            return F.batch_norm(x, None, None, _m.weight, _m.bias,
                                training=True, momentum=0.0, eps=_m.eps)

        handles.append(m.register_forward_hook(hook))
    return handles


def eval_batch_stats(model, collator, cache, rows, device, *,
                     batch_size=DIAG_BATCH, attention_dropout=False,
                     seed=20261024) -> np.ndarray:
    before = bn_hash(model)
    handles = enable_batch_stats(model)
    dropouts = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Dropout)
                and getattr(m, "p", 0.0) > 0]
    model.eval()
    if attention_dropout:
        # model.eval() first, then re-enable ONLY the attention dropout modules
        for _n, m in dropouts:
            m.train()
    out = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            batch = _to_device(collator(cache, rows[s:s + batch_size], "RC", valid=True), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    for hd in handles:
        hd.remove()
    for _n, m in dropouts:
        m.eval()
    after = bn_hash(model)
    if before != after:
        raise RuntimeError("batch-stat diagnostics wrote BN running buffers")
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# C: sequential train-only calibration of every BN's running statistics
# ---------------------------------------------------------------------------
def set_running(model, name: str, mean: torch.Tensor, var: torch.Tensor) -> None:
    bn = dict(bn_modules(model))[name]
    bn.running_mean.copy_(mean.to(bn.running_mean.dtype))
    bn.running_var.copy_(var.to(bn.running_var.dtype))


def calibrate(model, base_state, order, collator, train_cache, rows, device, *,
              batch_size=DIAG_BATCH) -> tuple[dict, dict]:
    """Estimate each BN's running statistics from train, in true forward order."""
    mods = dict(bn_modules(model))
    calib: dict[str, dict] = {}
    log = {}
    for k, name in enumerate(order):
        model.load_state_dict(base_state)
        for prev in order[:k]:
            set_running(model, prev, calib[prev]["mean"], calib[prev]["var"])
        model.eval()
        acc = Moments()
        handle = mods[name].register_forward_pre_hook(lambda mo, args: acc.update(args[0]))
        try:
            with torch.no_grad(), C.isolated_rng():
                for s in range(0, len(rows), batch_size):
                    batch = _to_device(collator(train_cache, rows[s:s + batch_size], "RC", valid=True), device)
                    model(batch)
        finally:
            handle.remove()
        n, mean, var = acc.result(unbiased=True)
        calib[name] = {"mean": mean, "var": var, "n": int(n)}
        bn = mods[name]
        log[name] = {
            "index": k,
            "n_entries": int(n),
            "n_channels": int(bn.running_mean.numel()),
            "eps": float(bn.eps),
            "old_mean_l2": float(torch.linalg.vector_norm(bn.running_mean.to(torch.float64))),
            "new_mean_l2": float(torch.linalg.vector_norm(mean)),
            "mean_abs_shift_max": float((mean - bn.running_mean.to(torch.float64)).abs().max()),
            "mean_abs_shift_mean": float((mean - bn.running_mean.to(torch.float64)).abs().mean()),
            "var_ratio_median": float(torch.median(var / bn.running_var.to(torch.float64).clamp_min(1e-12))),
            "var_ratio_min": float((var / bn.running_var.to(torch.float64).clamp_min(1e-12)).min()),
            "var_ratio_max": float((var / bn.running_var.to(torch.float64).clamp_min(1e-12)).max()),
        }
    # final: everything calibrated
    model.load_state_dict(base_state)
    for name in order:
        set_running(model, name, calib[name]["mean"], calib[name]["var"])
    return calib, log


def verify_calibration(model, order, calib, collator, train_cache, rows, device, *,
                       batch_size=DIAG_BATCH) -> dict:
    """Re-measure every BN's global moments on the finished C and compare."""
    mods = dict(bn_modules(model))
    accs = {n: Moments() for n in order}
    handles = [mods[n].register_forward_pre_hook(
        (lambda nm: (lambda mo, args: accs[nm].update(args[0])))(n)) for n in order]
    model.eval()
    try:
        with torch.no_grad(), C.isolated_rng():
            for s in range(0, len(rows), batch_size):
                batch = _to_device(collator(train_cache, rows[s:s + batch_size], "RC", valid=True), device)
                model(batch)
    finally:
        for hd in handles:
            hd.remove()
    worst_mean, worst_var, rows_out = 0.0, 0.0, []
    for n in order:
        got_n, mean, var = accs[n].result()
        bn = mods[n]
        stored_mean = bn.running_mean.to(torch.float64)
        stored_var = bn.running_var.to(torch.float64)
        d_mean = float((mean - stored_mean).abs().max())
        scale = torch.sqrt(stored_var.clamp_min(bn.eps))
        d_norm = float(((mean - stored_mean).abs() / scale).max())
        rel_var = float(((var - stored_var).abs() / stored_var.clamp_min(bn.eps)).max())
        worst_mean = max(worst_mean, d_norm)
        worst_var = max(worst_var, rel_var)
        rows_out.append({
            "name": n, "n_recount": int(got_n), "n_calibrated": int(calib[n]["n"]),
            "max_abs_mean_diff": d_mean,
            "max_abs_mean_diff_in_sigma": d_norm,
            "max_rel_var_diff": rel_var,
        })
    return {"layers": rows_out,
            "worst_abs_mean_diff_in_sigma": worst_mean,
            "worst_rel_var_diff": worst_var,
            "counts_match": all(r["n_recount"] == r["n_calibrated"] for r in rows_out),
            "passed": worst_mean < 5e-2 and worst_var < 5e-2}


# ---------------------------------------------------------------------------
def run_arm(arm: str, device: str, *, subset_n: int = SUBSET_N) -> dict:
    t0 = time.time()
    train_rel = RD.load_split("train")
    collator = RD.RCCollator({"train": train_rel, "valid": RD.load_split("valid")})
    stats = train_rel["stats"]
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    norm = D.load_norm()
    n_train = len(train_cache["labels"])
    n_valid = len(valid_cache["labels"])
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    y_valid = np.asarray(valid_cache["labels"], dtype=np.float64)

    # fixed diagnostic subset of train; the second permutation holds the rows
    # fixed and only changes the batch composition (batch-context sensitivity)
    rng = np.random.default_rng(SUBSET_SEED)
    sub1 = rng.permutation(n_train)[:subset_n]
    sub2 = np.random.default_rng(SUBSET_SEED_2).permutation(sub1)

    result = {
        "round": "zinc-relation-bn-diagnostic-v1",
        "arm": arm,
        "device": device,
        "official_test_loaded": False,
        "valid_used_for": "evaluation only (no statistics, no selection)",
        "subset": {"n": int(subset_n), "seed": SUBSET_SEED, "second_seed": SUBSET_SEED_2,
                   "batch": DIAG_BATCH, "rows_sha256": C.sha256_array(sub1.astype(np.int64)),
                   "rows_second_sha256": C.sha256_array(sub2.astype(np.int64))},
        "splits": {"train": int(n_train), "valid": int(n_valid)},
        "checkpoints": {},
        "moment_selftest": _moment_selftest(),
    }

    for tag, fname in CKPTS.items():
        path = SRC_RUNS / f"RC_{arm}_s0" / fname
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        src_state = {k: v.detach().clone() for k, v in ckpt["state_dict"].items()}
        model, _ = RM.build_model("RC", norm, 0, device, arm=arm, stats=stats)
        C.load_state(model, src_state, device)
        model.eval()
        entry = {
            "source_file": str(path.relative_to(_REPO)),
            "file_sha256": file_sha256(path),
            "epoch": int(ckpt.get("epoch", -1)),
            "state_sha256": state_sha256(src_state),
            "param_hash": param_hash(model),
            "bn_buffer_hash": bn_hash(model),
            "tensors": int(len(src_state)),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
        o_bn_before = bn_hash(model)

        # ---- O: original eval, batch 256 (matches the source evaluation)
        t = time.time()
        p_tr = eval_predict(model, collator, train_cache, np.arange(n_train), device,
                            batch_size=REPLAY_BATCH)
        p_va = eval_predict(model, collator, valid_cache, np.arange(n_valid), device,
                            batch_size=REPLAY_BATCH)
        entry["O"] = {"train_mae": float(np.abs(p_tr - y_train).mean()),
                      "valid_mae": float(np.abs(p_va - y_valid).mean()),
                      "seconds": time.time() - t}
        # ---- O/B on the shared subset, same rows and batches
        t = time.time()
        o_sub = eval_predict(model, collator, train_cache, sub1, device, batch_size=DIAG_BATCH)
        entry["O"]["subset_mae"] = float(np.abs(o_sub - y_train[sub1]).mean())
        b_sub = eval_batch_stats(model, collator, train_cache, sub1, device)
        b_sub2 = eval_batch_stats(model, collator, train_cache, sub2, device)
        b_full = eval_batch_stats(model, collator, train_cache, np.arange(n_train), device)
        # align the second permutation back to sub1's order for a paired diff
        # b_sub2 is ordered like sub2; map it back onto sub1's positions
        pos = {int(r): i for i, r in enumerate(sub2)}
        b_sub2_aligned = np.asarray([b_sub2[pos[int(r)]] for r in sub1])
        entry["B"] = {
            "subset_mae": float(np.abs(b_sub - y_train[sub1]).mean()),
            "subset_mae_second_permutation": float(np.abs(b_sub2 - y_train[sub2]).mean()),
            "train_mae": float(np.abs(b_full - y_train).mean()),
            "seconds": time.time() - t,
            "paired_subset": {
                "mean_abs_diff_pred": float(np.abs(b_sub - o_sub).mean()),
                "median_abs_diff_pred": float(np.median(np.abs(b_sub - o_sub))),
                "share_O_better": float((np.abs(o_sub - y_train[sub1])
                                         < np.abs(b_sub - y_train[sub1])).mean()),
            },
        }
        # batch-context sensitivity: same 2048 molecules, different batched order
        entry["B"]["batch_context_sensitivity"] = {
            "subset_mae_second_permutation": float(np.abs(b_sub2 - y_train[sub2]).mean()),
            "mean_abs_diff_matched_rows": float(np.abs(b_sub2_aligned - b_sub).mean()),
            "median_abs_diff_matched_rows": float(np.median(np.abs(b_sub2_aligned - b_sub))),
            "note": "identical molecule set, only the batch composition/order changes",
        }
        # ---- T (gated)
        online = float(C.read_json(SRC_RUNS / f"RC_{arm}_s0" / "train.json")
                       ["curve"][int(ckpt["epoch"]) - 1]["online_train_mae"])
        entry["online_train_mae_at_epoch"] = online
        gap = online - entry["O"]["train_mae"]
        closed = (online - entry["B"]["train_mae"]) / gap if abs(gap) > 1e-12 else float("nan")
        entry["B"]["share_of_online_gap_closed"] = float(closed)
        if not (closed >= T_GATE_SHARE):
            t = time.time()
            t_sub = eval_batch_stats(model, collator, train_cache, sub1, device,
                                     attention_dropout=True, seed=SUBSET_SEED)
            entry["T"] = {
                "subset_mae": float(np.abs(t_sub - y_train[sub1]).mean()),
                "mean_abs_diff_vs_B": float(np.abs(t_sub - b_sub).mean()),
                "seconds": time.time() - t,
                "note": "batch-stat BN + original attention dropout 0.2, fixed RNG",
            }
        else:
            entry["T"] = {"skipped": True,
                          "reason": f"B already closes {closed:.1%} of the online-vs-O train gap"}

        # ---- C: sequential train-only calibration, then eval
        t = time.time()
        order_info, repeated = bn_call_order(model, collator, train_cache, np.arange(n_train), device)
        order = [o["name"] for o in order_info]
        entry["bn_modules"] = order_info
        entry["bn_call_order_repeated"] = bool(repeated)
        nb_ = np.asarray(train_cache["n_bonds_graph"], dtype=np.int64)
        expected_tokens = int(nb_.sum())
        expected_pairs = int((nb_ * nb_).sum())
        counts = {o["name"]: o["calls"] for o in order_info}
        entry["bn_count_check"] = {
            "expected_token_entries": expected_tokens,
            "expected_pair_entries": expected_pairs,
            "observed": counts,
            "token_like": sorted(n for n, c in counts.items() if c == expected_tokens),
            "pair_like": sorted(n for n, c in counts.items() if c == expected_pairs),
            "unexpected": sorted(n for n, c in counts.items()
                                 if c not in (expected_tokens, expected_pairs)),
            "passed": all(c in (expected_tokens, expected_pairs) for c in counts.values()),
        }
        calib, log = calibrate(model, src_state, order, collator, train_cache,
                               np.arange(n_train), device)
        entry["bn_layers"] = log
        entry["param_hash_after_calibration"] = param_hash(model)
        entry["params_identical"] = entry["param_hash_after_calibration"] == entry["param_hash"]
        c_bn_after = bn_hash(model)
        entry["derived_bn_buffer_hash"] = c_bn_after
        entry["bn_buffers_changed"] = c_bn_after != o_bn_before
        # only BN running statistics may differ
        c_state = C.cpu_state(model)
        diffs = {}
        for k, v in src_state.items():
            w = c_state[k]
            if not torch.equal(v.to(w.dtype), w):
                diffs[k] = "running_stat" if ("running_mean" in k or "running_var" in k) else "OTHER"
        entry["state_diff_keys"] = sorted(diffs)
        entry["state_diff_only_running_stats"] = all(v == "running_stat" for v in diffs.values())
        entry["verify"] = verify_calibration(model, order, calib, collator, train_cache,
                                            np.arange(n_train), device)
        c_tr = eval_predict(model, collator, train_cache, np.arange(n_train), device,
                            batch_size=REPLAY_BATCH)
        c_va = eval_predict(model, collator, valid_cache, np.arange(n_valid), device,
                            batch_size=REPLAY_BATCH)
        entry["C"] = {"train_mae": float(np.abs(c_tr - y_train).mean()),
                      "valid_mae": float(np.abs(c_va - y_valid).mean()),
                      "subset_mae": None, "seconds": time.time() - t}
        c_sub = eval_predict(model, collator, train_cache, sub1, device, batch_size=DIAG_BATCH)
        entry["C"]["subset_mae"] = float(np.abs(c_sub - y_train[sub1]).mean())
        # re-batching invariance of C (eval-mode statistics do not depend on batching)
        rebatch = eval_predict(model, collator, train_cache, sub1[:256], device, batch_size=64)
        rebatch_ref = eval_predict(model, collator, train_cache, sub1[:256], device, batch_size=256)
        entry["C"]["rebatch_max_abs_diff"] = float(np.abs(rebatch - rebatch_ref).max())
        entry["C"]["bn_hash_after_eval"] = bn_hash(model)
        entry["C"]["bn_hash_stable_after_valid"] = entry["C"]["bn_hash_after_eval"] == c_bn_after
        # O must be untouched throughout: re-load the shipped state and hash it
        model.load_state_dict(src_state)
        model.eval()
        entry["O"]["param_hash_after_all"] = param_hash(model)
        entry["O"]["bn_hash_after_all"] = bn_hash(model)
        entry["O"]["unchanged_by_diagnostics"] = (
            entry["O"]["param_hash_after_all"] == entry["param_hash"]
            and entry["O"]["bn_hash_after_all"] == entry["bn_buffer_hash"])

        entry["summary"] = {
            "O_train": entry["O"]["train_mae"], "B_train": entry["B"]["train_mae"],
            "C_train": entry["C"]["train_mae"],
            "O_valid": entry["O"]["valid_mae"], "C_valid": entry["C"]["valid_mae"],
            "C_minus_O_train": entry["C"]["train_mae"] - entry["O"]["train_mae"],
            "C_minus_O_valid": entry["C"]["valid_mae"] - entry["O"]["valid_mae"],
            "B_minus_O_train_subset": entry["B"]["subset_mae"] - entry["O"]["subset_mae"],
            "C_minus_O_train_subset": entry["C"]["subset_mae"] - entry["O"]["subset_mae"],
        }
        result["checkpoints"][tag] = entry
        # keep the raw predictions for the merged CSVs
        np.savez(ANALYSIS / f"raw_{arm}_{tag}.npz", train_o=p_tr, train_b=b_full, train_c=c_tr,
                 valid_o=p_va, valid_c=c_va)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    result["wall_seconds"] = time.time() - t0
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["D", "S"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--subset", type=int, default=SUBSET_N)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    res = run_arm(args.arm, args.device, subset_n=args.subset)
    out = Path(args.out) if args.out else ANALYSIS / f"diag_{args.arm}.json"
    C.write_json(out, res)
    print(json.dumps({t: e["summary"] for t, e in res["checkpoints"].items()}, indent=2))
    print(f"[diag {args.arm}] wall {res['wall_seconds']:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
