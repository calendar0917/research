"""Load real trained states: checkpoint replay, corrected gradient kernel,
and native pure-lookup export.

Only the two semantic generator arms are analysed (``FS_N100_lipo_R`` and
``FP_N100_lipo_R``), at their own best-Val-epoch checkpoint.  Nothing is
retrained and no permutation is re-drawn.

* **Replay**: rebuild the exact training-time model (spec / norm / generator
  seed 17 / pi400 / sigma80), ``strict=True`` load the stored ``state_dict``,
  and check the full Valid/Test forward against the stored ``pred_*_best``
  (raw ``max|Δ| <= 1e-5``).
* **Gradient kernel**: ``E(c) = model.base.local.path_emb(c)`` (the *actual*
  table, including the fixed affine buffers and, for FP, the composite-code
  permutation), ``s(c)=q^T E(c)``, ``J_c = d s / d theta`` over the generator
  parameters that actually influence this bank, cosine ``K_ab`` grouped by the
  original mid/b1/b2 overlap.  Reported for init and trained-best, plus the
  change.  A free-lookup ``LS`` kernel must have zero cross-code coupling; that
  is used as an implementation check on "lookup tables do not share".
* **Native export**: materialise the six tables from the replayed best state,
  build the original DISSECT ``C1`` and ``strict=True`` load; full Valid/Test
  raw ``max|Δ| <= 1e-5`` against the generator, and recomputed RMSE/MAE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import rp_common as RP
import rp_reader as RR


# ---------------------------------------------------------------------------
# model rebuild helpers
# ---------------------------------------------------------------------------


def _view():
    import lp_data as LPD

    return LPD.build_view()


def _checkpoint(arm: str) -> Path:
    p = RP.CKPT_DIR / arm / "checkpoint_best.pt"
    if not p.is_file():
        # fall back to a pulled run dir
        alt = RP.OLD_RESULTS / {"FS_R": "FS_N100_lipo_R", "FP_R": "FP_N100_lipo_R"}[arm] / "checkpoint_best.pt"
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"missing checkpoint for {arm}; pull it via `rr pull`")
    return p


def _build_generator(view, family: str):
    import se_model as M
    import se_tables as T

    perms = T.build_permutations()
    model = M.build_semantic_model(view.spec, view.train_norm, family, permutations=perms)
    return model, perms


def _load_best(view, family: str, arm: str):
    model, perms = _build_generator(view, family)
    ck = torch.load(_checkpoint(arm), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model, ck, perms


def _buffer_hash(view, family: str) -> dict[str, str]:
    import se_common as SEC
    import se_model as M
    import se_tables as T

    model, perms = _build_generator(view, family)
    gen = model.generator
    e0 = M._e0_tables(view.spec, view.train_norm)
    buffers = T.build_buffers(gen, e0)
    h = RP.sha256_bytes(
        b"".join(np.ascontiguousarray(buffers[b][k], dtype=np.float64).tobytes()
                 for b in sorted(buffers) for k in ("mu_f0", "sd_f0", "mu_e0", "sd_e0"))
    )
    perm_h = RP.sha256_bytes(
        np.ascontiguousarray(perms["path"]).tobytes() + np.ascontiguousarray(perms["binary"]).tobytes()
    )
    return {"buffer_hash": h, "permutation_hash": perm_h}


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def replay(view, family: str, arm: str) -> dict[str, Any]:
    import lp_common as LPC
    import lp_data as LPD

    LPC.setup_determinism(strict=True)
    model, ck, perms = _load_best(view, family, arm)
    out: dict[str, Any] = {
        "arm": arm,
        "family": family,
        "checkpoint_sha256": RP.sha256_file(_checkpoint(arm)),
        "checkpoint_family": ck.get("family"),
        "checkpoint_best_epoch": ck.get("best_epoch"),
        "checkpoint_state_hash": ck.get("state_hash"),
        "loaded_state_hash": RP.state_hash(ck["state_dict"]),
        **_buffer_hash(view, family),
        "splits": {},
    }
    for split in RP.SPLITS:
        preds = LPD.predict_cache(model, view.caches[split], torch.device("cpu"))
        raw = view.to_raw(preds)
        p = RP.OLD_RESULTS / {"FS_R": "FS_N100_lipo_R", "FP_R": "FP_N100_lipo_R"}[arm] / f"pred_{split}_best.npz"
        ref_std = None
        ref_raw = None
        if p.is_file():
            d = np.load(p, allow_pickle=False)
            ref_std = np.asarray(d["standardised"], float)
            ref_raw = np.asarray(d["raw"], float)
        maxd_std = float(np.max(np.abs(preds - ref_std))) if ref_std is not None else None
        maxd_raw = float(np.max(np.abs(raw - ref_raw))) if ref_raw is not None else None
        y = view.y[split]
        out["splits"][split] = {
            "n": int(y.size),
            "rmse": float(np.sqrt(np.mean((raw - y) ** 2))),
            "mae": float(np.mean(np.abs(raw - y))),
            "max_abs_delta_standardised": maxd_std,
            "max_abs_delta_raw": maxd_raw,
            "identity_ok": bool(maxd_raw is not None and maxd_raw <= RP.REPRO_TOL),
        }
    out["all_identity_ok"] = all(out["splits"][s]["identity_ok"] for s in ("valid", "test"))
    out["note"] = "CPU forward vs the original GPU forward; max|Δ|raw is the cross-device replay gap"
    return out


# ---------------------------------------------------------------------------
# corrected gradient kernel
# ---------------------------------------------------------------------------


def choose_codes(view, max_codes: int = 64) -> list[int]:
    """Top-32 / bottom-32 Train document-frequency path codes (ties by code)."""
    df = RR.document_frequency(view.caches["train"], name="train")
    df_path = df["df_path"]
    codes = sorted(df_path)
    by_df = sorted(codes, key=lambda c: (int(df_path[c]), int(c)))
    k = max_codes // 2
    low = by_df[:k]
    high = by_df[-k:]
    chosen = sorted(set(low) | set(high))
    if len(chosen) > max_codes:
        chosen = chosen[:max_codes]
    return chosen


def _comps(c: int) -> tuple[int, int, int]:
    return (c // 25, (c % 25) // 5, c % 5)


def gradient_kernel(view, family: str, *, arm: str | None, max_codes: int = 64, seed: int = RP.KERNEL_SEED,
                    device: str = "cpu", init: bool = False) -> dict[str, Any]:
    """Cosine kernel of the per-code path_emb row gradient w.r.t. the actual params."""
    import se_tables as T

    codes = choose_codes(view, max_codes)
    if init:
        model, perms = _build_generator(view, family)
        model.to(device).eval()
        arm_label = "init"
    else:
        model, ck, perms = _load_best(view, family, arm)
        model.to(device).eval()
        arm_label = arm
    q = np.random.default_rng(int(seed)).integers(0, 2, size=RP.RANK).astype(np.float64) * 2 - 1
    q_t = torch.as_tensor(q / np.sqrt(RP.RANK), dtype=torch.float32, device=device)

    if family in ("FS", "FP"):
        params = list(model.generator.parameters())
        table = model.base.local.path_emb
    else:  # free lookup check (LS / LP)
        params = list(model.base.local.path_emb.parameters())
        table = model.base.local.path_emb

    raw_grads: dict[int, list[Any]] = {}
    for c in codes:
        model.zero_grad(set_to_none=True)
        e = table(torch.as_tensor([int(c)], device=device))
        s = (e[0] * q_t).sum()
        s.backward()
        raw_grads[c] = [p.grad.detach().clone() if p.grad is not None else None for p in params]

    active = [
        i for i in range(len(params))
        if any(g[i] is not None and float(g[i].abs().sum()) > 0.0 for g in raw_grads.values())
    ]
    J: dict[int, np.ndarray] = {}
    for c in codes:
        vec = []
        for i in active:
            g = raw_grads[c][i]
            vec.append(g.reshape(-1).cpu().numpy() if g is not None else np.zeros(params[i].numel(), np.float32))
        J[c] = np.concatenate(vec) if vec else np.zeros(0, np.float32)
    norms = {c: float(np.linalg.norm(J[c])) for c in codes}

    groups: dict[int, list[float]] = {0: [], 1: [], 2: []}
    diag = []
    off_by_pair: list[dict[str, Any]] = []
    for i, a in enumerate(codes):
        if norms[a] > 0:
            diag.append(float(np.dot(J[a], J[a]) / (norms[a] * norms[a])))
        for b in codes[i + 1:]:
            if norms[a] > 0 and norms[b] > 0:
                k = float(np.dot(J[a], J[b]) / (norms[a] * norms[b]))
            else:
                k = 0.0
            ov = sum(1 for x, y in zip(_comps(a), _comps(b)) if x == y)
            groups[ov].append(k)
            off_by_pair.append({"a": int(a), "b": int(b), "overlap": ov, "k": k})

    def _summ(v: list[float]):
        if not v:
            return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
        arr = np.asarray(v)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    return {
        "arm": arm_label,
        "family": family,
        "init": bool(init),
        "n_codes": len(codes),
        "chosen_codes": codes,
        "df_of_chosen": _df_of_chosen(view, codes),
        "active_params": active,
        "n_active_params": len(active),
        "dim_J": int(sum(params[i].numel() for i in active)),
        "direction_seed": int(seed),
        "diag_mean": float(np.mean(diag)) if diag else None,
        "kernel_by_overlap": {str(k): _summ(v) for k, v in groups.items()},
        "pairs": off_by_pair,
        "note": (
            "positive K = the per-code table-row directions are coupled through shared generator "
            "parameters; a mechanism/implementation check, not a task-benefit or chemical claim"
        ),
    }


def _df_of_chosen(view, codes) -> dict[int, int]:
    df = RR.document_frequency(view.caches["train"], name="train")["df_path"]
    return {int(c): int(df.get(int(c), 0)) for c in codes}


def ls_zero_coupling(view) -> dict[str, Any]:
    """Implementation check: a free lookup table has zero cross-code coupling."""
    import lp_common as LPC
    import se_model as M
    import se_tables as T

    LPC.setup_determinism(strict=True)
    perms = T.build_permutations()
    codes = choose_codes(view, 64)
    model = M.build_semantic_model(view.spec, view.train_norm, "LS", permutations=perms)
    model.eval()
    table = model.base.local.path_emb
    param = table.table.weight
    q = np.random.default_rng(RP.KERNEL_SEED).integers(0, 2, size=RP.RANK).astype(np.float64) * 2 - 1
    q_t = torch.as_tensor(q / np.sqrt(RP.RANK), dtype=torch.float32)
    J: dict[int, np.ndarray] = {}
    for c in codes:
        model.zero_grad(set_to_none=True)
        s = (table(torch.as_tensor([int(c)]))[0] * q_t).sum()
        s.backward()
        J[c] = param.grad.detach().reshape(-1).cpu().numpy().copy()
    max_off = 0.0
    n_pairs = 0
    for i, a in enumerate(codes):
        na = np.linalg.norm(J[a])
        for b in codes[i + 1:]:
            nb = np.linalg.norm(J[b])
            if na > 0 and nb > 0:
                max_off = max(max_off, abs(float(np.dot(J[a], J[b]) / (na * nb))))
            n_pairs += 1
    return {
        "family": "LS",
        "n_codes": len(codes),
        "n_pairs": n_pairs,
        "max_abs_offdiag_cosine": max_off,
        "zero_coupling_ok": bool(max_off <= RP.KERNEL_TOL),
        "note": (
            "free lookup: code c only touches its own embedding row, so cross-code coupling must be 0; "
            "this checks the implementation, it does not mix in the shared downstream layers"
        ),
    }


# ---------------------------------------------------------------------------
# native pure-lookup export
# ---------------------------------------------------------------------------


def native_export(view, family: str, arm: str) -> dict[str, Any]:
    import lp_common as LPC
    import lp_data as LPD
    import dissect_model as DM

    LPC.setup_determinism(strict=True)
    model, ck, perms = _load_best(view, family, arm)
    tables = model.materialise_tables(permute=model.permute)
    native: dict[str, torch.Tensor] = {}
    for k, v in ck["state_dict"].items():
        if k.startswith("base."):
            native[k[len("base."):]] = v
    for bank, mat in tables.items():
        native[f"local.{bank}.weight"] = mat.detach().cpu()
    c1 = DM.DissectLCRC(view.spec, view.train_norm, "C1")
    c1.load_state_dict(native, strict=True)
    c1.eval()
    out: dict[str, Any] = {
        "exported_from": arm,
        "family": family,
        "permute": bool(model.permute),
        "n_params": int(sum(p.numel() for p in c1.parameters())),
        "splits": {},
    }
    for split in RP.SPLITS:
        pg = LPD.predict_cache(model, view.caches[split], torch.device("cpu"))
        pl = LPD.predict_cache(c1, view.caches[split], torch.device("cpu"))
        raw_g = view.to_raw(pg)
        raw_l = view.to_raw(pl)
        y = view.y[split]
        out["splits"][split] = {
            "n": int(y.size),
            "max_abs_delta_generator_native_raw": float(np.max(np.abs(raw_g - raw_l))),
            "generator_rmse": float(np.sqrt(np.mean((raw_g - y) ** 2))),
            "generator_mae": float(np.mean(np.abs(raw_g - y))),
            "native_rmse": float(np.sqrt(np.mean((raw_l - y) ** 2))),
            "native_mae": float(np.mean(np.abs(raw_l - y))),
        }
    out["equivalence_ok"] = all(
        out["splits"][s]["max_abs_delta_generator_native_raw"] <= RP.REPRO_TOL for s in ("valid", "test")
    )
    ckpt_path = RP.DATA / "checkpoints" / arm / "checkpoint_best_native_lookup.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in native.items()},
            "family": "C1_native_lookup",
            "exported_from": arm,
            "checkpoint": "best_native_lookup",
            "equivalence_ok": out["equivalence_ok"],
            "source_state_hash": ck.get("state_hash"),
            "note": "native C1 strict-loadable lookup; NOT the old checkpoint_best_lookup.pt",
        },
        ckpt_path,
    )
    out["native_checkpoint"] = RP.track_rel(ckpt_path)
    out["native_checkpoint_sha256"] = RP.sha256_file(ckpt_path)
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run(view=None) -> dict[str, Any]:
    import lp_data as LPD

    view = view or LPD.build_view()

    replay_payload: dict[str, Any] = {}
    kernel_payload: dict[str, Any] = {}
    export_payload: dict[str, Any] = {}
    for family, arm in (("FS", "FS_R"), ("FP", "FP_R")):
        RP.log(f"replay {family}")
        replay_payload[arm] = replay(view, family, arm)
        RP.log(f"kernel {family} init")
        k_init = gradient_kernel(view, family, arm=None, init=True)
        RP.log(f"kernel {family} best")
        k_best = gradient_kernel(view, family, arm=arm, init=False)
        kernel_payload[arm] = {"init": k_init, "best": k_best, "change": _kernel_change(k_init, k_best)}
        RP.log(f"native export {family}")
        export_payload[arm] = native_export(view, family, arm)
    kernel_payload["ls_check"] = ls_zero_coupling(view)

    rp_payload = {
        "schema": "lcrc-semantic-analysis-repair-checkpoint-replay-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "git_commit": RP.git_commit(),
        "device": "cpu",
        "replay": replay_payload,
        "identity_note": "raw max|Δ| <= 1e-5 against the stored best predictions",
    }
    RP.write_json(RP.ANALYSIS / "checkpoint_replay.json", rp_payload)
    kp = {
        "schema": "lcrc-semantic-analysis-repair-kernel-corrected-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "old_status": "INIT_RAW_DIAGNOSTIC (predecessor built a fresh model and used bank_raw without the affine)",
        **kernel_payload,
    }
    RP.write_json(RP.ANALYSIS / "kernel_init_best_corrected.json", kp)
    ep = {
        "schema": "lcrc-semantic-analysis-repair-export-equivalence-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        **export_payload,
    }
    RP.write_json(RP.ANALYSIS / "export_equivalence.json", ep)
    return {
        "replay": {k: v["all_identity_ok"] for k, v in replay_payload.items()},
        "export": {k: v["equivalence_ok"] for k, v in export_payload.items()},
        "ls_zero_coupling": kernel_payload["ls_check"]["zero_coupling_ok"],
        "artifacts": [
            RP.track_rel(RP.ANALYSIS / "checkpoint_replay.json"),
            RP.track_rel(RP.ANALYSIS / "kernel_init_best_corrected.json"),
            RP.track_rel(RP.ANALYSIS / "export_equivalence.json"),
        ],
    }


def _kernel_change(k_init: Mapping[str, Any], k_best: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ov in ("0", "1", "2"):
        a = k_init["kernel_by_overlap"][ov]["mean"]
        b = k_best["kernel_by_overlap"][ov]["mean"]
        out[ov] = {"init": a, "best": b, "delta": (b - a) if (a is not None and b is not None) else None}
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
