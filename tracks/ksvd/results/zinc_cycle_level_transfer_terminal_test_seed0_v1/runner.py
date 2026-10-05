"""zinc_cycle_level_transfer_terminal_test_seed0_v1 — runner.

Phases (strict order):
  identity      verify frozen sources replay (train/valid, published metrics)
  folds         T25 class grouping + label-blind 3-fold assignment + vocabulary
  train_folds   6 trajectories (R/D x 3 folds), OOF predictions, purchase gate
  dfull         (gate PASS only) train D_full on all 10000 rows, freeze C package
  freeze        write terminal_eval_manifest (roster frozen before any test read)
  terminal      one unconditional valid+test evaluation of the whole frozen roster
  analyze       consolidate analysis.json

CPU only, <= 8 threads, FP32. No body training. Test is read only in `terminal`,
after `freeze`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(REPO))

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_component_supervision_fulltrain_confirmation_seed0_v1 as src,
)
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp  # noqa: E402
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run  # noqa: E402
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as ztraining  # noqa: E402
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd  # noqa: E402
from tracks.ksvd.experiments.luyin16 import zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev  # noqa: E402

PROTOCOL_VERSION = "zinc-cycle-level-transfer-terminal-test-seed0-v1"
OUT = Path(__file__).resolve().parent
SRC = REPO / "tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1"
PROTO = REPO / "tracks/ksvd/results/zinc_cycle_prototype_transfer_cpu_v1"

# ---- frozen recipe constants -------------------------------------------------
N_FOLDS = 3
HEAD_EPOCHS = 300
BATCH = 128
LR = 1.0e-3
WD = 1.0e-5
GRAD_CLIP = 5.0
SOUP_EPOCHS = (296, 297, 298, 299, 300)
TRAIN_GEN_BASE = 20261003
BUILD_SEED = 0
HIDDEN = (64, 32)
TOPOLOGY_IN = 25
R_PARAMS = 3777
D_PARAMS_EXTRA_PER_LEVEL = 33  # 32 weights + 1 bias per extra level
REPLAY_TOL = 1.0e-5
IDENT_TOL = 1.0e-9

# gate
GATE_OVERALL = 0.003
GATE_KM2 = 0.25
GATE_K0_WORSEN = 0.001
GATE_KM1_WORSEN = 0.05
CI_SEED = 20261011
N_CI = 1000
BOOT_SEED = 20261012
N_BOOT = 1000


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_arr(value: np.ndarray, dtype=np.float32) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64))))


def access_log(split: str, meta: Mapping[str, Any]) -> None:
    path = OUT / "heldout_access.json"
    data = json.loads(path.read_text()) if path.exists() else {"events": []}
    data["events"].append({"split": split, "accessed_at_utc": now_utc(), **dict(meta)})
    data["protocol_version"] = PROTOCOL_VERSION
    write_json(path, data)


# ---------------------------------------------------------------------------
# frozen source loading
# ---------------------------------------------------------------------------


def load_constants() -> dict[str, float]:
    with np.load(SRC / "full_train_targets.npz", allow_pickle=False) as z:
        return {
            str(n): float(v) for n, v in zip(z["constant_names"].tolist(), z["constants"].tolist())
        }


def load_train_arrays() -> dict[str, np.ndarray]:
    with np.load(SRC / "full_train_targets.npz", allow_pickle=False) as z:
        return {k: z[k] for k in ("y", "c", "g", "k", "gid")}


def load_q_head() -> nn.Module:
    with np.load(SRC / "full_train_targets.npz", allow_pickle=False) as z:
        bias_value = float(np.median(z["c"]))
    head = src.build_q_head(0, bias_value)
    head.load_state_dict(torch.load(SRC / "Q_raw_soup_state.pt", map_location="cpu", weights_only=True), strict=True)
    head.eval()
    return head


def q_forward_np(head: nn.Module, T: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return src.q_forward(head, torch.as_tensor(np.asarray(T, np.float32))).double().numpy()


def load_body(arm: str) -> nn.Module:
    payload_arrays = np.load(SRC / "full_train_payload.npz", allow_pickle=False)
    payload = prev.TuplePayload(payload_arrays)
    kappa = float(payload_arrays["kappa"].reshape(-1)[0])
    model = src.build_component_model(arm, payload, kappa)
    state = torch.load(SRC / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.to(torch.device("cpu")) for k, v in state.items()}, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_train_data() -> list[Any]:
    prep_meta = src._read_npz(SRC / "full_train_prep.npz")
    return src.build_fulltrain_data(prep_meta)


def load_valid_data() -> list[Any]:
    valid, meta = src.load_valid_data(SRC)
    return valid, meta


def load_test_data() -> tuple[list[Any], dict[str, Any]]:
    """Official ZINC subset test (PyG), frozen input path; split explicit."""
    train_records, _valid_records = ztraining.load_train_valid_records()
    test_records = ztraining.extract_test_records()
    config = ztraining.base_config()
    _train_enc, test_data, _audit = ztraining.build_encoded(train_records, test_records, config)
    test_data = list(test_data)
    raw_test = list(ztraining._load_zinc(ztraining.ZINC_ROOT, "test"))
    if len(test_data) != len(raw_test) or len(test_data) != 1000:
        raise RuntimeError(f"official-test length mismatch: {len(test_data)} vs {len(raw_test)}")
    env_meta = p1run._attach_test_env(test_data, raw_test)
    # frozen full-train prep application (same code path as the source valid loader)
    prep_meta = src._read_npz(SRC / "full_train_prep.npz")
    blob = np.load(src.PREP_BLOB, allow_pickle=False)
    patch_all = zjd.Std(blob["patch_all_mean"], blob["patch_all_scale"])
    ctx_all = zjd.Std(blob["ctx_all_mean"], blob["ctx_all_scale"])
    anchor_all = zjd.Std(blob["anchor_all_mean"], blob["anchor_all_scale"])
    topo_all = zjd.Std(blob["topo_all_mean"], blob["topo_all_scale"])
    patch_fit = zjd.Std(prep_meta["patch_fit_mean"], prep_meta["patch_fit_scale"])
    ctx_fit = zjd.Std(prep_meta["ctx_fit_mean"], prep_meta["ctx_fit_scale"])
    anchor_fit = zjd.Std(prep_meta["anchor_fit_mean"], prep_meta["anchor_fit_scale"])
    topo_fit = zjd.Std(prep_meta["topo_fit_mean"], prep_meta["topo_fit_scale"])
    p, pc = zjd._stack(test_data, "patch_cont")
    c, cc = zjd._stack(test_data, "global_context")
    a, ac = zjd._stack(test_data, "anchor")
    t, tc = zjd._stack(test_data, "topology_features")
    zjd._unstack(patch_fit.transform(patch_all.inverse(p)), pc, "patch_cont", test_data)
    zjd._unstack(ctx_fit.transform(ctx_all.inverse(c)), cc, "global_context", test_data)
    zjd._unstack(anchor_fit.transform(anchor_all.inverse(a)), ac, "anchor", test_data)
    zjd._unstack(topo_fit.transform(topo_all.inverse(t)), tc, "topology_features", test_data)
    import torch as _torch

    for index, data in enumerate(test_data):
        data.local_mol_id = _torch.tensor([int(index)], dtype=_torch.long)
    meta = {
        "n_rows": 1000,
        "source": "official ZINC subset test (PyG) via zinc_compact_v4_training_sufficiency loader",
        "raw_split": "test",
        "prep": "frozen full-train prep (10000-row standardizers, no test refit)",
        "builder": "zinc_full_decomposition_valid_test_confirmation_v1.load_test_data path (engineering reuse)",
        "official_test_loaded": True,
        **env_meta,
    }
    return test_data, meta


def build_test_labels() -> dict[str, np.ndarray]:
    """Official test y + component labels with the frozen train constants."""
    import pandas as pd

    df = pd.read_csv(REPO / "tracks/ksvd/results/zinc_long_cycle_audit/test_cycle_audit_label.csv")
    df = df.sort_values("subset_index").reset_index(drop=True)
    if not np.array_equal(df["subset_index"].to_numpy(np.int64), np.arange(len(df))):
        raise RuntimeError("test label csv rows are not positional")
    y = df["target"].to_numpy(np.float64)
    k = np.round(df["label_effective_cycle_snapped"].to_numpy(np.float64)).astype(np.int64)
    const = load_constants()
    c = (k.astype(np.float64) - const["mu_cycle"]) / const["sigma_cycle"]
    g = y - c
    props = np.load(REPO / "tracks/ksvd/results/zinc_long_cycle_audit/cache/gvae_full_properties.npz")
    lines = df["smi_line"].to_numpy(np.int64)
    logP = np.asarray(props["logP"], np.float64)[lines]
    ell = (logP - src.MU_LOGP) / const["sigma_logP"]
    s = g - ell
    sa = np.asarray(props["SA"], np.float64)[lines]
    s_SA = (sa - const["mu_SA"]) / const["sigma_SA"]
    checks = {
        "n_rows": int(len(y)),
        "y_equals_g_plus_c_max_abs": float(np.max(np.abs(y - (g + c)))),
        "g_equals_ell_plus_s_max_abs": float(np.max(np.abs(g - (ell + s)))),
        "k_counts": {str(int(v)): int((k == v).sum()) for v in sorted(set(k.tolist()))},
        "constants_from": "frozen full-train constants (component_supervision_fulltrain_confirmation_seed0_v1)",
        "note_csv_c_column_unused": "csv label_cycle_component uses the old 8k refine constants; scoring-side c is rebuilt from k with the frozen full-train constants",
    }
    return {
        "y": y, "g": g, "c": c, "k": k, "ell": ell, "s": s, "s_SA": s_SA,
        "ids": np.array([f"test:{i:04d}" for i in range(len(y))], dtype=object),
        "checks": checks,
    }


def build_valid_labels() -> dict[str, np.ndarray]:
    diag = src.load_valid_diagnostics(SRC)
    return {k: np.asarray(diag[k]) for k in ("y", "g", "c", "k", "ell", "s", "s_SA")}


# ---------------------------------------------------------------------------
# phase: identity
# ---------------------------------------------------------------------------


def phase_identity(log=print) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.set_num_threads(8)
    arrs = load_train_arrays()
    const = load_constants()
    out: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "constants": const}

    # float64 identities on train targets
    out["train_identity"] = {
        "y_equals_g_plus_c_max_abs": float(np.max(np.abs(arrs["y"] - (arrs["g"] + arrs["c"])))),
        "c_from_k_rule_max_abs": float(np.max(np.abs(arrs["c"] - ((arrs["k"].astype(np.float64) - const["mu_cycle"]) / const["sigma_cycle"])))),
        "k_all_integer": bool(np.all(arrs["k"] == np.round(arrs["k"]))),
    }

    # replay frozen Q on train T25 vs cache
    train_data = load_train_data()
    T_train = src.topology_matrix(train_data)
    out["T25_train_sha256"] = sha_arr(T_train)
    q_head = load_q_head()
    q_train_replay = q_forward_np(q_head, T_train)
    with np.load(SRC / "Q_train_predictions.npz", allow_pickle=False) as z:
        q_train_cache = z["q_raw"].astype(np.float64)
    out["q_train_replay_max_abs"] = float(np.max(np.abs(q_train_replay - q_train_cache)))

    # body replay on train: COMP raw_soup_train vs cache (spot check via evaluate)
    comp = load_body("COMP")
    sums, _comps = src.evaluate_state_components(comp, train_data[:512], torch.device("cpu"))
    with np.load(SRC / "COMP_raw_predictions.npz", allow_pickle=False) as z:
        comp_cache = z["raw_soup_train"].astype(np.float64)
    out["comp_train_head512_replay_max_abs"] = float(np.max(np.abs(sums - comp_cache[:512])))
    cal = json.loads((SRC / "calibration.json").read_text())

    # valid replay from the frozen models in float64 (published metrics)
    valid_data, _vmeta = load_valid_data()
    T_valid = src.topology_matrix(valid_data)
    q_valid = q_forward_np(q_head, T_valid)
    sum_body = load_body("SUM")
    comp_body = load_body("COMP")
    SUM_h_v, _ = src.evaluate_state_components(sum_body, valid_data, torch.device("cpu"))
    COMP_h_v, _ = src.evaluate_state_components(comp_body, valid_data, torch.device("cpu"))
    b_y_comp = float(cal["per_arm"]["COMP"]["b_y"])
    b_y_sum = float(cal["per_arm"]["SUM"]["b_y"])
    b_g_comp = float(cal["per_arm"]["COMP"]["b_g"])
    diag = src.load_valid_diagnostics(SRC)
    y_v64 = np.asarray(diag["y"], np.float64)
    g_v64 = np.asarray(diag["g"], np.float64)
    c_v64 = np.asarray(diag["c"], np.float64)
    B_y_cal = COMP_h_v + q_valid + b_y_comp
    SUM_y_cal = SUM_h_v + q_valid + b_y_sum
    B_g_cal = COMP_h_v + b_g_comp
    # H = prototype route on the same float64 h/q
    pkg = torch.load(PROTO / "H_model_package.pt", map_location="cpu", weights_only=False)
    key_map = {bytes.fromhex(kk): int(ss) for kk, ss in zip(pkg["key_order"], pkg["key_slot"])}
    proto_val = np.asarray(pkg["proto_val"], np.float64)
    consistent = np.asarray(pkg["consistent"], bool)
    q_H = q_valid.copy()
    route_H = np.empty(len(T_valid), dtype=object)
    for i in range(len(T_valid)):
        slot = key_map.get(key_bytes(T_valid[i]))
        if slot is None:
            route_H[i] = "UNSEEN_FALLBACK"
        elif not consistent[slot]:
            route_H[i] = "TRAIN_CONFLICT_FALLBACK"
        else:
            route_H[i] = "CONSISTENT_HIT"
            q_H[i] = float(proto_val[slot])
    H_y_cal = COMP_h_v + q_H + float(pkg["b_H"])
    out["valid_replay"] = {
        "B_y_cal_mae": mae(B_y_cal, y_v64),
        "B_y_cal_mae_expected": 0.11740618350630393,
        "B_g_cal_mae": mae(B_g_cal, g_v64),
        "B_g_cal_mae_expected": 0.08930046045603672,
        "SUM_y_cal_mae": mae(SUM_y_cal, y_v64),
        "SUM_y_cal_mae_expected": 0.12265322754724184,
        "H_y_cal_mae": mae(H_y_cal, y_v64),
        "H_y_cal_mae_expected": 0.09669437497661369,
        "H_route_counts": {r: int((route_H == r).sum()) for r in set(route_H.tolist())},
        "biases": {"b_y_COMP": b_y_comp, "b_y_SUM": b_y_sum, "b_g_COMP": b_g_comp, "b_H": float(pkg["b_H"])},
        "note": "recomputed float64 from frozen bodies + Q replay (prototype-round method)",
    }
    out["valid_replay"]["B_y_cal_match"] = bool(
        abs(out["valid_replay"]["B_y_cal_mae"] - 0.11740618350630393) <= IDENT_TOL
    )
    out["valid_replay"]["B_g_match"] = bool(
        abs(out["valid_replay"]["B_g_cal_mae"] - 0.08930046045603672) <= IDENT_TOL
    )
    out["valid_replay"]["SUM_match"] = bool(
        abs(out["valid_replay"]["SUM_y_cal_mae"] - 0.12265322754724184) <= IDENT_TOL
    )
    out["valid_replay"]["H_match"] = bool(
        abs(out["valid_replay"]["H_y_cal_mae"] - 0.09669437497661369) <= IDENT_TOL
    )
    out["valid_replay"]["c_from_k_rule_max_abs"] = float(np.max(np.abs(
        c_v64 - (np.asarray(diag["k"], np.int64).astype(np.float64) - const["mu_cycle"]) / const["sigma_cycle"]
    )))
    # float32 cache agreement at prediction tolerance 1e-5
    with np.load(SRC / "valid_frozen_predictions.npz", allow_pickle=False) as z:
        vp = {k: z[k] for k in z.files}
    out["valid_cache_prediction_tolerance"] = {
        "COMP_h_max_abs": float(np.max(np.abs(COMP_h_v - vp["COMP_h_raw"].astype(np.float64)))),
        "SUM_h_max_abs": float(np.max(np.abs(SUM_h_v - vp["SUM_h_raw"].astype(np.float64)))),
        "q_max_abs": float(np.max(np.abs(q_valid - vp["q_raw"].astype(np.float64)))),
    }

    # prototype package hashes
    out["source_file_sha256"] = {
        "targets": file_sha256(SRC / "full_train_targets.npz"),
        "prep": file_sha256(SRC / "full_train_prep.npz"),
        "COMP_soup": file_sha256(SRC / "COMP_raw_soup_state.pt"),
        "SUM_soup": file_sha256(SRC / "SUM_raw_soup_state.pt"),
        "Q_soup": file_sha256(SRC / "Q_raw_soup_state.pt"),
        "valid_frozen_predictions": file_sha256(SRC / "valid_frozen_predictions.npz"),
        "prototype_table": file_sha256(PROTO / "prototype_table.npz"),
        "H_model_package": file_sha256(PROTO / "H_model_package.pt"),
    }
    ok = (
        out["train_identity"]["y_equals_g_plus_c_max_abs"] <= 1e-9
        and out["q_train_replay_max_abs"] <= REPLAY_TOL
        and out["comp_train_head512_replay_max_abs"] <= REPLAY_TOL
        and out["valid_replay"]["B_y_cal_match"]
        and out["valid_replay"]["B_g_match"]
        and out["valid_replay"]["SUM_match"]
        and out["valid_replay"]["H_match"]
    )
    out["all_source_identity_ok"] = bool(ok)
    out["seconds"] = float(time.perf_counter() - t0)
    write_json(OUT / "source_identity.json", out)
    log(f"[identity] all_ok={ok} ({out['seconds']:.1f}s)")
    if not ok:
        raise RuntimeError("source identity failed; refusing to continue")
    return out


# ---------------------------------------------------------------------------
# phase: folds
# ---------------------------------------------------------------------------


def key_bytes(row: np.ndarray) -> bytes:
    row = np.ascontiguousarray(row, np.float32)
    if np.isnan(row).any() or np.isinf(row).any():
        raise RuntimeError("T25 key contains NaN/Inf")
    neg = np.signbit(row) & (row == 0.0)
    if neg.any():
        row = row.copy()
        row[neg] = 0.0
    return row.tobytes()


def phase_folds(log=print) -> dict[str, Any]:
    t0 = time.perf_counter()
    arrs = load_train_arrays()
    k_train = np.asarray(arrs["k"], np.int64)
    c_train = np.asarray(arrs["c"], np.float64)

    # T25 classes from the prototype round's cache (already verified identity)
    with np.load(PROTO / "T25_cache.npz", allow_pickle=False) as z:
        T_train = z["T_train"]
        class_id = z["class_id"]
    n = len(T_train)
    classes = sorted(set(int(v) for v in class_id.tolist()))

    # class support + canonical key bytes
    class_rows: dict[int, np.ndarray] = {}
    class_key: dict[int, bytes] = {}
    class_support: dict[int, int] = {}
    for cid in classes:
        rows = np.nonzero(class_id == cid)[0]
        class_rows[cid] = rows
        class_support[cid] = int(len(rows))
        class_key[cid] = key_bytes(T_train[rows[0]])

    # label-blind assignment: support desc, tie by canonical key bytes asc,
    # greedy to the lightest fold, tie to lowest fold index
    order = sorted(classes, key=lambda cid: (-class_support[cid], class_key[cid]))
    fold_rows = [0] * N_FOLDS
    class_fold: dict[int, int] = {}
    for cid in order:
        j = min(range(N_FOLDS), key=lambda f: (fold_rows[f], f))
        class_fold[cid] = j
        fold_rows[j] += class_support[cid]
    row_fold = np.empty(n, np.int64)
    for cid in classes:
        row_fold[class_rows[cid]] = class_fold[cid]

    # disjointness witness
    disjoint_ok = True
    for j in range(N_FOLDS):
        held = set(class_id[row_fold == j].tolist())
        fit = set(class_id[row_fold != j].tolist())
        if held & fit:
            disjoint_ok = False

    # vocabulary
    vocab = sorted(set(int(v) for v in k_train.tolist()))
    const = load_constants()
    c_levels = np.array([(kk - const["mu_cycle"]) / const["sigma_cycle"] for kk in vocab], np.float64)

    fold_info = []
    for j in range(N_FOLDS):
        fit_mask = row_fold != j
        held_mask = row_fold == j
        k_fit = k_train[fit_mask]
        fit_levels = sorted(set(int(v) for v in k_fit.tolist()))
        fold_info.append({
            "fold": j,
            "n_fit": int(fit_mask.sum()),
            "n_held": int(held_mask.sum()),
            "n_fit_classes": int(len(set(class_id[fit_mask].tolist()))),
            "n_held_classes": int(len(set(class_id[held_mask].tolist()))),
            "fit_levels": fit_levels,
            "missing_fit_levels": [v for v in vocab if v not in fit_levels],
            "held_k_counts": {str(v): int((k_train[held_mask] == v).sum()) for v in vocab},
            "fit_k_counts": {str(v): int((k_fit == v).sum()) for v in vocab},
        })

    out = {
        "protocol_version": PROTOCOL_VERSION,
        "n_train_rows": n,
        "n_classes": len(classes),
        "T25_train_sha256": sha_arr(T_train),
        "assignment_rule": "support desc, tie canonical key bytes asc, greedy lightest fold, tie lowest index; label-blind",
        "fold_row_counts": fold_rows,
        "held_fit_class_disjoint": disjoint_ok,
        "folds": fold_info,
        "vocab": vocab,
        "K": len(vocab),
        "c_levels": c_levels,
        "class_fold_sha256": hashlib.sha256(
            b"".join(f"{cid}:{class_fold[cid]}".encode() for cid in classes)
        ).hexdigest(),
        "row_fold_sha256": sha_arr(row_fold, np.int64),
        "seconds": float(time.perf_counter() - t0),
    }
    np.savez_compressed(
        OUT / "T25_group_folds.npz",
        T_train=T_train,
        class_id=class_id,
        row_fold=row_fold,
        k_train=k_train,
        c_train=c_train,
        y_train=np.asarray(arrs["y"], np.float64),
        g_train=np.asarray(arrs["g"], np.float64),
        gid_train=np.asarray(arrs["gid"], np.int64),
        vocab=np.array(vocab, np.int64),
        c_levels=c_levels,
    )
    write_json(OUT / "folds.json", out)
    log(f"[folds] row counts={fold_rows} disjoint={disjoint_ok} vocab={vocab} ({out['seconds']:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------


def build_head(kind: str, out_dim: int, bias_value: float) -> nn.Module:
    """Shared hidden stack built by the Q seed-0 untrained construction rule.

    kind R: out_dim=1, last weight 0, last bias=bias_value.
    kind D: out_dim=K, last weight 0, last bias 0.
    Hidden init identical item-by-item between R and D (same construction RNG).
    """
    torch.manual_seed(BUILD_SEED)
    head = nn.Sequential(
        nn.Linear(TOPOLOGY_IN, HIDDEN[0]),
        nn.SiLU(),
        nn.Linear(HIDDEN[0], HIDDEN[1]),
        nn.SiLU(),
        nn.Linear(HIDDEN[1], out_dim),
    )
    nn.init.zeros_(head[4].weight)
    with torch.no_grad():
        if kind == "R":
            head[4].bias.copy_(torch.tensor(float(bias_value)))
        else:
            head[4].bias.zero_()
    return head


def hidden_of_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Hidden (non-output-layer) entries of a head state dict."""
    return {k: v.detach().clone() for k, v in state.items() if not k.startswith("4.")}


def train_one_head(
    kind: str,
    out_dim: int,
    T_fit: np.ndarray,
    target_fit: np.ndarray,
    bias_value: float,
    gen_seed: int,
) -> dict[str, Any]:
    """kind R: out_dim=1, float c target, L1. kind D: out_dim=K, int class target, CE."""
    n = int(T_fit.shape[0])
    head = build_head(kind, out_dim, bias_value)
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
    generator = torch.Generator().manual_seed(int(gen_seed))
    T_t = torch.as_tensor(T_fit, dtype=torch.float32)
    if kind == "R":
        tgt_t = torch.as_tensor(np.asarray(target_fit, np.float64), dtype=torch.float32)
    else:
        tgt_t = torch.as_tensor(np.asarray(target_fit, np.int64), dtype=torch.long)
    soup: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        order = torch.randperm(n, generator=generator)
        abs_sum = 0.0
        grad_norm = 0.0
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            logits = head(T_t[idx])
            if kind == "R":
                pred = logits.view(-1)
                loss = (pred - tgt_t[idx]).abs().mean()
                abs_sum += float((pred - tgt_t[idx]).abs().sum())
            else:
                loss = F.cross_entropy(logits, tgt_t[idx])
                abs_sum += float(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            steps += 1
        if epoch in SOUP_EPOCHS:
            soup[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curve.append({"epoch": epoch, "train_loss": abs_sum / n, "grad_norm": grad_norm,
                      "seconds": float(time.perf_counter() - started)})
    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    head.load_state_dict(soup_state)
    head.eval()
    return {
        "head": head, "init_state": init_state, "soup_state": soup_state,
        "init_hash": state_hash(init_state), "soup_hash": state_hash(soup_state),
        "hidden_init": hidden_of_state(init_state), "curve": curve, "steps": steps,
        "soup_members": members, "seconds": float(time.perf_counter() - started),
        "gen_seed": int(gen_seed),
    }


def decode_levels(logits: np.ndarray, c_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median-level decode: softmax; levels sorted by c_level asc; first cum >= 0.5."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    order = np.argsort(c_levels, kind="stable")  # ascending c_level
    sorted_probs = probs[:, order]
    cum = np.cumsum(sorted_probs, axis=1)
    hit = (cum >= 0.5).argmax(axis=1)  # first index with cum >= 0.5
    level_idx = order[hit]
    return c_levels[level_idx], probs


def phase_train_folds(log=print) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.set_num_threads(8)
    with np.load(OUT / "T25_group_folds.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    T_train, row_fold = d["T_train"], d["row_fold"]
    k_train, c_train = d["k_train"], d["c_train"]
    vocab, c_levels = d["vocab"], d["c_levels"]
    K = len(vocab)
    k_to_idx = {int(v): i for i, v in enumerate(vocab.tolist())}
    k_idx_train = np.array([k_to_idx[int(v)] for v in k_train.tolist()], np.int64)

    results: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "folds": {}}
    oof_R = np.empty(len(T_train), np.float64)
    oof_D = np.empty(len(T_train), np.float64)
    oof_D_idx = np.empty(len(T_train), np.int64)
    oof_D_probs = np.zeros((len(T_train), K), np.float64)

    for j in range(N_FOLDS):
        fit_mask = row_fold != j
        held_mask = row_fold == j
        T_fit = T_train[fit_mask]
        c_fit = c_train[fit_mask]
        idx_fit = k_idx_train[fit_mask]
        bias_value = float(np.median(c_fit))
        gen_seed = TRAIN_GEN_BASE + j

        # R: continuous regression, L1
        tR = time.perf_counter()
        rres = train_one_head("R", 1, T_fit, c_fit, bias_value, gen_seed)
        with torch.no_grad():
            q_R_held = rres["head"](torch.as_tensor(T_train[held_mask], dtype=torch.float32)).view(-1).double().numpy()
            q_R_fit = rres["head"](torch.as_tensor(T_fit, dtype=torch.float32)).view(-1).double().numpy()
        # D: discrete levels, CE
        tD = time.perf_counter()
        dres = train_one_head("D", K, T_fit, idx_fit, 0.0, gen_seed)
        with torch.no_grad():
            logits_held = dres["head"](torch.as_tensor(T_train[held_mask], dtype=torch.float32)).double().numpy()
            logits_fit = dres["head"](torch.as_tensor(T_fit, dtype=torch.float32)).double().numpy()
        q_D_held, probs_held = decode_levels(logits_held, c_levels)
        q_D_fit, probs_fit = decode_levels(logits_fit, c_levels)
        pred_idx_held = probs_held.argmax(1)
        pred_idx_fit = probs_fit.argmax(1)

        oof_R[held_mask] = q_R_held
        oof_D[held_mask] = q_D_held
        oof_D_idx[held_mask] = pred_idx_held
        oof_D_probs[held_mask] = probs_held

        # hidden init identity witness R vs D
        hid_equal = all(torch.equal(rres["hidden_init"][k2], dres["hidden_init"][k2]) for k2 in rres["hidden_init"])

        # fold metrics (fit + held)
        fold_metrics = {
            "fold": j,
            "gen_seed": gen_seed,
            "n_fit": int(fit_mask.sum()), "n_held": int(held_mask.sum()),
            "R_init_hash": rres["init_hash"], "R_soup_hash": rres["soup_hash"],
            "D_init_hash": dres["init_hash"], "D_soup_hash": dres["soup_hash"],
            "R_hidden_init_equals_D_hidden_init": bool(hid_equal),
            "R_params": R_PARAMS, "D_params": R_PARAMS + D_PARAMS_EXTRA_PER_LEVEL * (K - 1),
            "R_bias_value_median_c_fit": bias_value,
            "R_fit_c_mae": mae(q_R_fit, c_fit),
            "D_fit_c_mae": mae(q_D_fit, c_fit),
            "R_OOF_c_mae": mae(q_R_held, c_train[held_mask]),
            "D_OOF_c_mae": mae(q_D_held, c_train[held_mask]),
            "D_OOF_k_accuracy": float(np.mean(pred_idx_held == k_idx_train[held_mask])),
            "D_fit_k_accuracy": float(np.mean(pred_idx_fit == idx_fit)),
            "seconds_R": float(time.perf_counter() - tR),
            "seconds_D": float(time.perf_counter() - tD),
            "R_curve_last": rres["curve"][-1], "D_curve_last": dres["curve"][-1],
        }
        # per-k held errors
        per_k = {}
        for kk in vocab.tolist():
            m = k_train[held_mask] == kk
            if m.sum() == 0:
                per_k[str(kk)] = {"n": 0}
                continue
            ch = c_train[held_mask]
            per_k[str(kk)] = {
                "n": int(m.sum()),
                "R_c_mae": mae(q_R_held[m], ch[m]),
                "D_c_mae": mae(q_D_held[m], ch[m]),
                "D_k_acc": float(np.mean(pred_idx_held[m] == k_idx_train[held_mask][m])),
            }
        fold_metrics["held_per_k"] = per_k
        # confusion matrix held (rows true, cols pred)
        conf = np.zeros((K, K), np.int64)
        for a, b in zip(k_idx_train[held_mask].tolist(), pred_idx_held.tolist()):
            conf[a, b] += 1
        fold_metrics["held_confusion"] = conf.tolist()
        results["folds"][str(j)] = fold_metrics
        torch.save(rres["init_state"], OUT / f"R_fold{j}_init_state.pt")
        torch.save(rres["soup_state"], OUT / f"R_fold{j}_soup_state.pt")
        torch.save(dres["init_state"], OUT / f"D_fold{j}_init_state.pt")
        torch.save(dres["soup_state"], OUT / f"D_fold{j}_soup_state.pt")
        write_json(OUT / f"R_fold{j}_curve.json", rres["curve"])
        write_json(OUT / f"D_fold{j}_curve.json", dres["curve"])
        log(f"[fold {j}] R OOF c MAE={fold_metrics['R_OOF_c_mae']:.6f} D OOF c MAE={fold_metrics['D_OOF_c_mae']:.6f} "
            f"kAcc={fold_metrics['D_OOF_k_accuracy']:.4f} hidden_init_equal={hid_equal}")

    np.savez_compressed(
        OUT / "OOF_predictions.npz",
        oof_R=oof_R, oof_D=oof_D, oof_D_idx=oof_D_idx, oof_D_probs=oof_D_probs,
        row_fold=row_fold, k_train=k_train, c_train=c_train, vocab=vocab, c_levels=c_levels,
    )

    # ---- pooled OOF purchase gate (computed once, after all six trajectories)
    gain_overall = mae(oof_R, c_train) - mae(oof_D, c_train)
    per_k_gain = {}
    for kk in vocab.tolist():
        m = k_train == kk
        per_k_gain[str(kk)] = {
            "n": int(m.sum()),
            "R_mae": mae(oof_R[m], c_train[m]),
            "D_mae": mae(oof_D[m], c_train[m]),
            "gain": mae(oof_R[m], c_train[m]) - mae(oof_D[m], c_train[m]),
        }
    cond = {
        "1_overall_gain_ge_0.003": bool(gain_overall >= GATE_OVERALL),
        "2_km2_gain_ge_0.25": bool(per_k_gain["-2"]["gain"] >= GATE_KM2) if "-2" in per_k_gain and per_k_gain["-2"]["n"] > 0 else None,
        "3_k0_worsening_le_0.001": bool(-per_k_gain["0"]["gain"] <= GATE_K0_WORSEN) if "0" in per_k_gain else None,
        "4_km1_worsening_le_0.05": bool(-per_k_gain["-1"]["gain"] <= GATE_KM1_WORSEN) if "-1" in per_k_gain and per_k_gain["-1"]["n"] > 0 else None,
    }
    estimable = all(v is not None and v is not False for v in cond.values()) and all(v is not None for v in cond.values())
    gate_pass = bool(all(v is True for v in cond.values()))

    # descriptive class-level CI (shared draws across R/D)
    with np.load(PROTO / "T25_cache.npz", allow_pickle=False) as z:
        class_id = z["class_id"]
    classes = np.unique(class_id)
    class_rows = {int(cid): np.nonzero(class_id == cid)[0] for cid in classes}
    rng = np.random.default_rng(CI_SEED)
    draws = rng.integers(0, len(classes), size=(N_CI, len(classes)))
    err_R = np.abs(oof_R - c_train)
    err_D = np.abs(oof_D - c_train)
    gains_ci = np.empty(N_CI)
    for b in range(N_CI):
        rows = np.concatenate([class_rows[int(classes[c])] for c in draws[b]])
        gains_ci[b] = float(err_R[rows].mean() - err_D[rows].mean())
    ci = {
        "method": "class-level resampling of complete T25 classes; all rows kept; same draws shared R/D",
        "n_boot": N_CI, "seed": CI_SEED,
        "overall_gain_point": gain_overall,
        "overall_gain_ci95": [float(np.percentile(gains_ci, 2.5)), float(np.percentile(gains_ci, 97.5))],
    }

    gate = {
        "protocol_version": PROTOCOL_VERSION,
        "pooled_OOF": {
            "R_overall_c_mae": mae(oof_R, c_train),
            "D_overall_c_mae": mae(oof_D, c_train),
            "overall_gain": gain_overall,
            "per_k": per_k_gain,
        },
        "conditions": cond,
        "D_FULL_PURCHASE": gate_pass,
        "markers": None,
        "descriptive_ci": ci,
        "seconds": float(time.perf_counter() - t0),
    }
    # SINGLE_ROW-style dominance diagnostics on the OOF gain (row-level, descriptive)
    row_gain = err_R - err_D
    pos = row_gain[row_gain > 0]
    neg = row_gain[row_gain < 0]
    imax = int(np.argmax(row_gain))
    gate["markers"] = {
        "max_positive_row": int(imax),
        "max_positive_row_k": int(k_train[imax]),
        "max_positive_row_gain": float(row_gain[imax]),
        "positive_sum": float(pos.sum()) if pos.size else 0.0,
        "negative_sum": float(neg.sum()) if neg.size else 0.0,
        "max_positive_share": float(row_gain[imax] / pos.sum()) if pos.size and pos.sum() > 0 else None,
    }
    write_json(OUT / "gate.json", gate)
    log(f"[gate] overall gain={gain_overall:.6f} conds={cond} PASS={gate_pass} "
        f"CI={ci['overall_gain_ci95']} ({gate['seconds']:.1f}s)")
    results["gate"] = gate
    return results


# ---------------------------------------------------------------------------
# phase: fold_metrics (deterministic reload recovery; no retraining)
# ---------------------------------------------------------------------------


def phase_fold_metrics(log=print) -> dict[str, Any]:
    """Rebuild per-fold R/D metrics + the init-identity witness from SAVED states.

    Recovery for the bookkeeping bug where `hidden_init` was captured post-
    training; the trained weights, recipe, OOF predictions and gate scores are
    unchanged and re-verified here from the saved artifacts (deterministic
    reload, not a retrain).
    """
    t0 = time.perf_counter()
    torch.set_num_threads(8)
    with np.load(OUT / "T25_group_folds.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    T_train, row_fold = d["T_train"], d["row_fold"]
    k_train, c_train = d["k_train"], d["c_train"]
    vocab, c_levels = d["vocab"], d["c_levels"]
    K = len(vocab)
    k_to_idx = {int(v): i for i, v in enumerate(vocab.tolist())}
    k_idx_train = np.array([k_to_idx[int(v)] for v in k_train.tolist()], np.int64)
    with np.load(OUT / "OOF_predictions.npz", allow_pickle=False) as z:
        oof = {k: z[k] for k in z.files}

    # Q's untrained hidden init (the seed-0 construction rule reference)
    q_init = torch.load(SRC / "Q_init_state.pt", map_location="cpu", weights_only=True)
    q_hidden = hidden_of_state(q_init)

    summary: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "folds": {}}
    oof_R_check = np.empty(len(T_train), np.float64)
    oof_D_check = np.empty(len(T_train), np.float64)
    for j in range(N_FOLDS):
        fit_mask = row_fold != j
        held_mask = row_fold == j
        T_fit, c_fit = T_train[fit_mask], c_train[fit_mask]
        idx_fit = k_idx_train[fit_mask]
        bias_value = float(np.median(c_fit))
        gen_seed = TRAIN_GEN_BASE + j
        r_init = torch.load(OUT / f"R_fold{j}_init_state.pt", map_location="cpu", weights_only=True)
        r_soup = torch.load(OUT / f"R_fold{j}_soup_state.pt", map_location="cpu", weights_only=True)
        d_init = torch.load(OUT / f"D_fold{j}_init_state.pt", map_location="cpu", weights_only=True)
        d_soup = torch.load(OUT / f"D_fold{j}_soup_state.pt", map_location="cpu", weights_only=True)
        r_hid, d_hid = hidden_of_state(r_init), hidden_of_state(d_init)
        hid_equal = all(torch.equal(r_hid[k2], d_hid[k2]) for k2 in r_hid)
        hid_equal_q = all(torch.equal(r_hid[k2], q_hidden[k2]) for k2 in r_hid)
        # reload soup heads and recompute fit/held predictions
        r_head = build_head("R", 1, bias_value)
        r_head.load_state_dict(r_soup, strict=True)
        r_head.eval()
        d_head = build_head("D", K, 0.0)
        d_head.load_state_dict(d_soup, strict=True)
        d_head.eval()
        with torch.no_grad():
            q_R_fit = r_head(torch.as_tensor(T_fit, dtype=torch.float32)).view(-1).double().numpy()
            q_R_held = r_head(torch.as_tensor(T_train[held_mask], dtype=torch.float32)).view(-1).double().numpy()
            logits_fit = d_head(torch.as_tensor(T_fit, dtype=torch.float32)).double().numpy()
            logits_held = d_head(torch.as_tensor(T_train[held_mask], dtype=torch.float32)).double().numpy()
        q_D_fit, probs_fit = decode_levels(logits_fit, c_levels)
        q_D_held, probs_held = decode_levels(logits_held, c_levels)
        pred_idx_held = probs_held.argmax(1)
        pred_idx_fit = probs_fit.argmax(1)
        oof_R_check[held_mask] = q_R_held
        oof_D_check[held_mask] = q_D_held
        per_k = {}
        for kk in vocab.tolist():
            m = k_train[held_mask] == kk
            entry = {"n": int(m.sum())}
            if m.sum():
                ch = c_train[held_mask]
                entry.update({
                    "R_c_mae": mae(q_R_held[m], ch[m]),
                    "D_c_mae": mae(q_D_held[m], ch[m]),
                    "D_k_acc": float(np.mean(pred_idx_held[m] == k_idx_train[held_mask][m])),
                })
            per_k[str(kk)] = entry
        conf = np.zeros((K, K), np.int64)
        for a, b in zip(k_idx_train[held_mask].tolist(), pred_idx_held.tolist()):
            conf[a, b] += 1
        summary["folds"][str(j)] = {
            "fold": j, "gen_seed": gen_seed,
            "n_fit": int(fit_mask.sum()), "n_held": int(held_mask.sum()),
            "R_init_hash": state_hash(r_init), "R_soup_hash": state_hash(r_soup),
            "D_init_hash": state_hash(d_init), "D_soup_hash": state_hash(d_soup),
            "R_hidden_init_equals_D_hidden_init": bool(hid_equal),
            "R_hidden_init_equals_Q_untrained_hidden_init": bool(hid_equal_q),
            "R_params": R_PARAMS, "D_params": R_PARAMS + D_PARAMS_EXTRA_PER_LEVEL * (K - 1),
            "R_bias_value_median_c_fit": bias_value,
            "R_fit_c_mae": mae(q_R_fit, c_fit),
            "D_fit_c_mae": mae(q_D_fit, c_fit),
            "R_OOF_c_mae": mae(q_R_held, c_train[held_mask]),
            "D_OOF_c_mae": mae(q_D_held, c_train[held_mask]),
            "D_OOF_k_accuracy": float(np.mean(pred_idx_held == k_idx_train[held_mask])),
            "D_fit_k_accuracy": float(np.mean(pred_idx_fit == idx_fit)),
            "held_per_k": per_k,
            "held_confusion": conf.tolist(),
        }
        log(f"[fold_metrics {j}] hidden_init_equal(R,D)={hid_equal} equal_Q_untrained={hid_equal_q}")
    # OOF cross-check vs the saved OOF arrays (must be identical: deterministic reload)
    check = {
        "oof_R_reload_vs_saved_max_abs": float(np.max(np.abs(oof_R_check - oof["oof_R"]))),
        "oof_D_reload_vs_saved_max_abs": float(np.max(np.abs(oof_D_check - oof["oof_D"]))),
    }
    summary["oof_reload_check"] = check
    summary["oof_reload_identical"] = bool(check["oof_R_reload_vs_saved_max_abs"] == 0.0 and check["oof_D_reload_vs_saved_max_abs"] == 0.0)
    summary["seconds"] = float(time.perf_counter() - t0)
    write_json(OUT / "folds_summary.json", summary)
    log(f"[fold_metrics] reload identical={summary['oof_reload_identical']} ({summary['seconds']:.1f}s)")
    return summary


# ---------------------------------------------------------------------------
# phase: dfull (purchased only)
# ---------------------------------------------------------------------------


def phase_dfull(log=print) -> dict[str, Any]:
    torch.set_num_threads(8)
    gate = json.loads((OUT / "gate.json").read_text())
    if not gate["D_FULL_PURCHASE"]:
        log("[dfull] gate FAIL -> D_full NOT trained; C NOT built")
        return {"D_full_trained": False, "reason": "purchase gate FAIL"}
    t0 = time.perf_counter()
    with np.load(OUT / "T25_group_folds.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    T_train, k_train, c_train = d["T_train"], d["k_train"], d["c_train"]
    vocab, c_levels = d["vocab"], d["c_levels"]
    K = len(vocab)
    k_to_idx = {int(v): i for i, v in enumerate(vocab.tolist())}
    k_idx_train = np.array([k_to_idx[int(v)] for v in k_train.tolist()], np.int64)

    dres = train_one_head("D", K, T_train, k_idx_train, 0.0, TRAIN_GEN_BASE)
    with torch.no_grad():
        logits = dres["head"](torch.as_tensor(T_train, dtype=torch.float32)).double().numpy()
    q_D_train, probs_train = decode_levels(logits, c_levels)
    out = {
        "protocol_version": PROTOCOL_VERSION,
        "D_full_trained": True,
        "gen_seed": TRAIN_GEN_BASE,
        "steps": dres["steps"],
        "epochs": HEAD_EPOCHS,
        "soup_members": dres["soup_members"],
        "init_hash": dres["init_hash"],
        "soup_hash": dres["soup_hash"],
        "parameters": R_PARAMS + D_PARAMS_EXTRA_PER_LEVEL * (K - 1),
        "train_c_mae_decoded": mae(q_D_train, c_train),
        "train_k_accuracy": float(np.mean(probs_train.argmax(1) == k_idx_train)),
        "seconds": dres["seconds"],
    }
    torch.save(dres["init_state"], OUT / "D_full_init_state.pt")
    torch.save(dres["soup_state"], OUT / "D_full_soup_state.pt")
    write_json(OUT / "D_full_meta.json", {k2: v for k2, v in out.items() if k2 != "protocol_version"})
    write_json(OUT / "D_full_curve.json", dres["curve"])
    np.savez_compressed(OUT / "D_full_train_predictions.npz",
                        q_decoded=q_D_train, probs=probs_train, k_idx=k_idx_train, c=c_train)
    log(f"[dfull] trained steps={dres['steps']} train_c_mae={out['train_c_mae_decoded']:.6f} "
        f"kAcc={out['train_k_accuracy']:.4f} ({out['seconds']:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# C wrapper (purchased only)
# ---------------------------------------------------------------------------


class CWrapper:
    """Frozen C: COMP h_raw + prototype table routing + (unseen) D_full decode + b_H."""

    def __init__(self) -> None:
        pkg = torch.load(PROTO / "H_model_package.pt", map_location="cpu", weights_only=False)
        self.b_H = float(pkg["b_H"])
        self.key_map = {bytes.fromhex(k): int(s) for k, s in zip(pkg["key_order"], pkg["key_slot"])}
        self.proto_val = np.asarray(pkg["proto_val"], np.float64)
        self.consistent = np.asarray(pkg["consistent"], bool)
        self.q_head = load_q_head()
        with np.load(OUT / "T25_group_folds.npz", allow_pickle=False) as z:
            self.vocab = z["vocab"]
            self.c_levels = z["c_levels"]
        self.K = len(self.vocab)
        self.d_full = build_head("D", self.K, 0.0)
        self.d_full.load_state_dict(torch.load(OUT / "D_full_soup_state.pt", map_location="cpu", weights_only=True), strict=True)
        self.d_full.eval()

    def route_of(self, row: np.ndarray) -> tuple[str, float | None]:
        slot = self.key_map.get(key_bytes(row))
        if slot is None:
            return "UNSEEN_FALLBACK", None
        if not self.consistent[slot]:
            return "TRAIN_CONFLICT_FALLBACK", None
        return "CONSISTENT_HIT", float(self.proto_val[slot])

    def q_frozen(self, T: np.ndarray) -> np.ndarray:
        return q_forward_np(self.q_head, T)

    def q_dfull(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            logits = self.d_full(torch.as_tensor(np.asarray(T, np.float32))).double().numpy()
        q_dec, probs = decode_levels(logits, self.c_levels)
        return q_dec, probs, logits

    def q_system(self, T: np.ndarray, system: str):
        """system in {B (pure frozen Q), H (prototype), C (prototype+D_full unseen)}."""
        T = np.asarray(T, np.float32)
        q = self.q_frozen(T)
        if system == "B":
            return q, np.array(["Q"] * len(T), dtype=object)
        route = np.empty(len(T), dtype=object)
        for i in range(len(T)):
            r, v = self.route_of(T[i])
            route[i] = r
            if v is not None:
                q[i] = v
        if system == "H":
            return q, route
        if system == "C":
            unseen = route == "UNSEEN_FALLBACK"
            if unseen.any():
                q_d, probs, _ = self.q_dfull(T[unseen])
                q[unseen] = q_d
            return q, route
        raise ValueError(system)


def phase_freeze(log=print) -> dict[str, Any]:
    """Write terminal_eval_manifest: roster frozen before any test prediction/metric."""
    gate = json.loads((OUT / "gate.json").read_text())
    purchased = bool(gate["D_FULL_PURCHASE"])
    cal = json.loads((SRC / "calibration.json").read_text())
    roster = {
        "SUM_Q": {"body": "source SUM_raw_soup_state.pt", "cycle": "source Q_raw_soup_state.pt",
                  "bias": float(cal["per_arm"]["SUM"]["b_y"])},
        "B": {"body": "source COMP_raw_soup_state.pt", "cycle": "source Q_raw_soup_state.pt",
              "bias": float(cal["per_arm"]["COMP"]["b_y"])},
        "H": {"body": "source COMP_raw_soup_state.pt",
              "cycle": "prototype_table.npz CONSISTENT_HIT median_c + frozen Q fallback",
              "bias": -0.011446799464432368},
    }
    if purchased:
        roster["C"] = {"body": "source COMP_raw_soup_state.pt",
                       "cycle": "prototype + conflict Q + unseen D_full decode",
                       "bias": -0.011446799464432368}
    manifest = {
        "manifest_version": "terminal-eval-manifest-v1",
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at_utc": now_utc(),
        "execution_commit_source": "947d2837c4936ef1276fcbbf0e9189e7d44ace23",
        "prototype_round_commit": "5a510d9",
        "purchase_gate_passed": purchased,
        "roster": roster,
        "roster_decided_by": "train-only purchase gate only; never by this round's valid scores",
        "artifacts_sha256": {
            "SUM_soup": file_sha256(SRC / "SUM_raw_soup_state.pt"),
            "COMP_soup": file_sha256(SRC / "COMP_raw_soup_state.pt"),
            "Q_soup": file_sha256(SRC / "Q_raw_soup_state.pt"),
            "prototype_table": file_sha256(PROTO / "prototype_table.npz"),
            "full_train_targets": file_sha256(SRC / "full_train_targets.npz"),
            "full_train_prep": file_sha256(SRC / "full_train_prep.npz"),
            "full_train_payload": file_sha256(SRC / "full_train_payload.npz"),
            "calibration": file_sha256(SRC / "calibration.json"),
            "T25_group_folds": file_sha256(OUT / "T25_group_folds.npz"),
            "gate": file_sha256(OUT / "gate.json"),
        },
        "decoder": "softmax; levels ascending by c_level; first cumulative >= 0.5 -> that c_level (median rule)",
        "input_builder": {
            "train": "zftd.load_train_only + build_fulltrain_data (frozen full-train prep)",
            "valid": "encoded_valid.pt + env_valid.pt + frozen full-train prep (source loader)",
            "test": "official ZINC subset test (PyG) via zinc_compact_v4_training_sufficiency loader + _attach_test_env + frozen full-train prep (zinc_full_decomposition_valid_test_confirmation_v1.load_test_data engineering path)",
        },
        "test_label_mapping": {
            "source": "zinc_long_cycle_audit/test_cycle_audit_label.csv + cache/gvae_full_properties.npz",
            "rule": "k=round(label_effective_cycle_snapped); c=(k-mu_cycle)/sigma_cycle frozen full-train constants; g=y-c; ell=(logP-MU_LOGP)/sigma_logP; s=g-ell; csv label_cycle_component column unused (old 8k constants)",
        },
        "valid_status": "exposed/replayed (source + prototype rounds already used it)",
        "test_status": "this round reads it after this manifest; project has used test before (disclosed)",
        "metric_definitions": {
            "primary": "MAE (y units) raw/cal per system per split",
            "gain": "reference MAE - candidate MAE, positive = improvement",
            "paired_bootstrap": f"{N_BOOT} draws, seed {BOOT_SEED}, shared row indices",
        },
    }
    if purchased:
        manifest["artifacts_sha256"]["D_full_soup"] = file_sha256(OUT / "D_full_soup_state.pt")
    write_json(OUT / "terminal_eval_manifest.json", manifest)
    log(f"[freeze] roster={list(roster)} purchased={purchased}")
    return manifest


# ---------------------------------------------------------------------------
# phase: terminal
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(ref_pred: np.ndarray, cand_pred: np.ndarray, y: np.ndarray,
                         n: int, seed: int = BOOT_SEED) -> dict[str, Any]:
    """Paired row bootstrap of gain = MAE(ref) - MAE(cand); positive = cand improves."""
    y = np.asarray(y, np.float64)
    abs_ref = np.abs(np.asarray(ref_pred, np.float64) - y)
    abs_cand = np.abs(np.asarray(cand_pred, np.float64) - y)
    rng = np.random.default_rng(seed)
    gains = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        gains[b] = float(abs_ref[idx].mean() - abs_cand[idx].mean())
    return {
        "point": float(abs_ref.mean() - abs_cand.mean()),
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "n_boot": N_BOOT, "seed": seed,
    }


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {"k=0": k == 0, "k=-1": k == -1, "k=-2": k == -2, "k<=-3": k <= -3}


def eval_split(split: str, log=print) -> dict[str, Any]:
    """Evaluate the whole frozen roster on one split. test is read post-freeze."""
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    if split == "valid":
        data, meta = load_valid_data()
        labels = build_valid_labels()
        labels["ids"] = np.array([f"valid:{i:04d}" for i in range(len(data))], dtype=object)
        T = src.topology_matrix(data)
    elif split == "test":
        data, meta = load_test_data()
        labels = build_test_labels()
        T = src.topology_matrix(data)
        access_log("test", {"phase": "terminal", "n_rows": 1000, "post_manifest_freeze": True,
                            "source": meta["source"]})
    else:
        raise ValueError(split)
    n = len(data)
    cal = json.loads((SRC / "calibration.json").read_text())
    b_y_sum = float(cal["per_arm"]["SUM"]["b_y"])
    b_y_comp = float(cal["per_arm"]["COMP"]["b_y"])
    b_g_comp = float(cal["per_arm"]["COMP"]["b_g"])
    b_g_sum = float(cal["per_arm"]["SUM"]["b_g"])
    b_H = -0.011446799464432368
    y = np.asarray(labels["y"], np.float64)
    g = np.asarray(labels["g"], np.float64)
    c = np.asarray(labels["c"], np.float64)
    k = np.asarray(labels["k"], np.int64)

    gate = json.loads((OUT / "gate.json").read_text())
    purchased = bool(gate["D_FULL_PURCHASE"])

    # bodies (h_raw = g-part prediction); replayed in float64 from frozen states on both
    # splits (the valid float32 cache is cross-checked at 1e-5 prediction tolerance)
    sum_body = load_body("SUM")
    comp_body = load_body("COMP")
    SUM_h, _ = src.evaluate_state_components(sum_body, data, torch.device("cpu"))
    COMP_h, _ = src.evaluate_state_components(comp_body, data, torch.device("cpu"))
    q_cache = None
    if split == "valid":
        with np.load(SRC / "valid_frozen_predictions.npz", allow_pickle=False) as z:
            vp = {kk: z[kk] for kk in z.files}
        q_cache = vp["q_raw"].astype(np.float64)

    wrapper = CWrapper()
    # replay witness: wrapper's frozen Q on this split T vs cache (valid) / fresh (test)
    q_B = wrapper.q_frozen(T)
    q_cache_match = float(np.max(np.abs(q_B - q_cache))) if q_cache is not None else None

    q_H, route_H = wrapper.q_system(T, "H")
    systems: dict[str, dict[str, Any]] = {}

    # SUM_Q
    systems["SUM_Q"] = {
        "h": SUM_h, "q": q_B, "bias": b_y_sum,
        "raw": SUM_h + q_B, "route": np.array(["Q"] * n, dtype=object),
    }
    # B
    systems["B"] = {
        "h": COMP_h, "q": q_B, "bias": b_y_comp,
        "raw": COMP_h + q_B, "route": np.array(["Q"] * n, dtype=object),
    }
    # H
    systems["H"] = {"h": COMP_h, "q": q_H, "bias": b_H, "raw": COMP_h + q_H, "route": route_H}
    # C
    if purchased:
        q_C, route_C = wrapper.q_system(T, "C")
        systems["C"] = {"h": COMP_h, "q": q_C, "bias": b_H, "raw": COMP_h + q_C, "route": route_C}

    out: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION, "split": split, "n": n, "meta": meta,
        "labels_checks": labels.get("checks", {}),
        "q_cache_replay_max_abs": q_cache_match,
        "y_equals_g_plus_c_max_abs": float(np.max(np.abs(y - (g + c)))),
    }
    per_row = {"y": y, "g": g, "c": c, "k": k,
               "ids": labels["ids"], "SUM_h_raw": SUM_h, "COMP_h_raw": COMP_h, "q_raw": q_B}

    main_rows = []
    for name, s in systems.items():
        cal_pred = s["raw"] + s["bias"]
        s["cal"] = cal_pred
        per_row[f"{name}_y_raw"] = s["raw"]
        per_row[f"{name}_y_cal"] = cal_pred
        per_row[f"{name}_q"] = s["q"]
        per_row[f"{name}_route"] = s["route"]
        # g endpoint: h + b_g (diagnostic); identical across B/H/C by construction
        if name == "SUM_Q":
            g_cal = s["h"] + b_g_sum
        else:
            g_cal = s["h"] + b_g_comp
        per_row[f"{name}_g_cal"] = g_cal
        main_rows.append({
            "system": name, "n": n, "bias": s["bias"],
            "y_raw_mae": mae(s["raw"], y), "y_cal_mae": mae(cal_pred, y),
            "g_cal_mae": mae(g_cal, g), "y_cal_below_0.09": bool(mae(cal_pred, y) < 0.09),
        })
        # group table
        gm = {}
        for gname, mask in group_masks(k).items():
            if mask.sum() == 0:
                gm[gname] = {"n": 0}
                continue
            gm[gname] = {
                "n": int(mask.sum()),
                "y_cal_mae": mae(cal_pred[mask], y[mask]),
                "y_cal_contrib": float(np.abs(cal_pred[mask] - y[mask]).sum() / n),
                "q_c_mae": mae(s["q"][mask], c[mask]),
            }
        s["groups"] = gm
        s["y_cal_err"] = cal_pred - y
        s["y_raw_err"] = s["raw"] - y
        s["e_g"] = s["h"] - g
        s["e_c"] = s["q"] - c

    out["main_table"] = main_rows
    out["group_tables"] = {name: s["groups"] for name, s in systems.items()}

    # routing tables (H and C)
    for name in ("H", "C"):
        if name not in systems:
            continue
        s = systems[name]
        rt = {}
        for rname in ("CONSISTENT_HIT", "TRAIN_CONFLICT_FALLBACK", "UNSEEN_FALLBACK"):
            m = s["route"] == rname
            if m.sum() == 0:
                rt[rname] = {"n": 0}
                continue
            entry = {
                "n": int(m.sum()),
                "q_c_mae": mae(s["q"][m], c[m]),
                "y_cal_mae": mae(s["cal"][m], y[m]),
                "y_cal_contrib": float(np.abs(s["cal"][m] - y[m]).sum() / n),
            }
            if name == "C" and rname != "UNSEEN_FALLBACK":
                # delta pred vs H on seen routes must be ~0
                d = float(np.max(np.abs(s["q"][m] - systems["H"]["q"][m])))
                entry["delta_pred_vs_H_max_abs"] = d
            rt[rname] = entry
        out.setdefault("routing_tables", {})[name] = rt

    # unseen-branch detail for C
    if purchased and "C" in systems:
        s = systems["C"]
        m = s["route"] == "UNSEEN_FALLBACK"
        if m.any():
            q_d, probs, logits = wrapper.q_dfull(T[m])
            pred_idx = probs.argmax(1)
            out["C_unseen_detail"] = {
                "n": int(m.sum()),
                "true_k": k[m].tolist(),
                "pred_level_k": [int(wrapper.vocab[i]) for i in pred_idx.tolist()],
                "decoded_c": q_d.tolist(),
                "true_c": c[m].tolist(),
                "c_err": np.abs(q_d - c[m]).tolist(),
                "y_cal_err": np.abs(s["cal"][m] - y[m]).tolist(),
                "ids": [str(x) for x in labels["ids"][m].tolist()],
                "row_indices": np.nonzero(m)[0].tolist(),
            }

    # fixed comparisons with paired CI; gain = MAE(reference) - MAE(candidate),
    # positive = candidate improves over the reference (B over SUM_Q, H over B, C over H)
    comps = {}
    pairs = [("SUM_Q", "B"), ("B", "H")]
    if purchased:
        pairs.append(("H", "C"))
    for ref, cand in pairs:
        comps[f"{cand}_minus_{ref}"] = {
            "reference": ref, "candidate": cand,
            "gain_cal": paired_bootstrap_ci(systems[ref]["cal"], systems[cand]["cal"], y, n),
            "gain_raw": paired_bootstrap_ci(systems[ref]["raw"], systems[cand]["raw"], y, n),
        }
    out["comparisons"] = comps

    # bootstrap witnesses (deterministic statistics checks)
    w_ident = paired_bootstrap_ci(systems["B"]["cal"], systems["B"]["cal"], y, n)
    w_swap = paired_bootstrap_ci(systems["B"]["cal"], systems["H"]["cal"], y, n)
    w_swap_mirror = paired_bootstrap_ci(systems["H"]["cal"], systems["B"]["cal"], y, n)
    shifted = systems["H"]["cal"] + 0.25
    w_shift = paired_bootstrap_ci(systems["B"]["cal"], shifted, y, n)
    out["bootstrap_witnesses"] = {
        "identical_point": w_ident["point"],
        "identical_is_zero": bool(w_ident["point"] == 0.0),
        "swap_point": w_swap["point"],
        "swap_mirror_ok": bool(abs(w_swap["point"] + w_swap_mirror["point"]) <= 1e-12),
        "constant_shift_delta": 0.25,
        "constant_shift_gain_changes_by": float(w_shift["point"] - comps["H_minus_B"]["gain_cal"]["point"] + 0.25) if "H_minus_B" in comps else None,
        "constant_shift_inside_bound": bool(w_shift["point"] < 0.0),
    }

    # cancellation diagnostics for B and H (+C)
    for name in ("B", "H", "C"):
        if name not in systems:
            continue
        s = systems[name]
        e_g, e_c = s["e_g"], s["e_c"]
        out.setdefault("cancellation", {})[name] = {
            "e_g_mae": mae(e_g, np.zeros(n)),
            "e_c_mae": mae(e_c, np.zeros(n)),
            "opposite_sign_rate": float(np.mean(np.sign(e_g) != np.sign(e_c))),
            "triangle_gap": float(np.mean(np.abs(e_g) + np.abs(e_c) - np.abs(e_g + e_c))),
            "sum_abs_components": float((np.abs(e_g) + np.abs(e_c)).mean()),
            "true_abs_sum": float(np.abs(e_g + e_c).mean()),
            "bias": s["bias"],
        }

    # top-10 tail: reference B fixed top-10 cal errors + which route + improvements
    berr = np.abs(systems["B"]["cal"] - y)
    top10 = np.argsort(-berr)[:10]
    tail = []
    for i in top10.tolist():
        row = {
            "id": str(labels["ids"][i]), "idx": i, "k": int(k[i]),
            "B_y_cal_abs_err": float(berr[i]),
            "route_H": str(systems["H"]["route"][i]),
            "H_y_cal_abs_err": float(abs(systems["H"]["cal"][i] - y[i])),
        }
        if purchased and "C" in systems:
            row["C_y_cal_abs_err"] = float(abs(systems["C"]["cal"][i] - y[i]))
        tail.append(row)
    out["B_top10_cal_errors"] = tail

    # valid motivating rows 0935/0214/0249 explained post-freeze (valid only)
    if split == "valid":
        mot = {}
        for rid in ("valid:0935", "valid:0214", "valid:0249"):
            i = int(rid.split(":")[1])
            entry = {
                "k": int(k[i]), "c": float(c[i]), "y": float(y[i]),
                "COMP_h_raw": float(COMP_h[i]), "q_B": float(q_B[i]),
                "route_H": str(systems["H"]["route"][i]),
                "q_H": float(systems["H"]["q"][i]),
                "B_y_cal_err": float(systems["B"]["cal"][i] - y[i]),
                "H_y_cal_err": float(systems["H"]["cal"][i] - y[i]),
            }
            if purchased and "C" in systems:
                entry["q_C"] = float(systems["C"]["q"][i])
                entry["C_y_cal_err"] = float(systems["C"]["cal"][i] - y[i])
            mot[rid] = entry
        out["motivating_rows_post_freeze"] = mot

    # single-row dominance + markers for C on valid
    if split == "valid" and purchased and "C" in systems:
        sH, sC = systems["H"], systems["C"]
        gain_rows = np.abs(sH["cal"] - y) - np.abs(sC["cal"] - y)
        pos = gain_rows[gain_rows > 0]
        imax = int(np.argmax(gain_rows)) if gain_rows.size else -1
        worst_H = int(np.argmax(np.abs(sH["cal"] - y)))
        gain_excl_worst = float(
            np.mean(np.delete(np.abs(sH["cal"] - y), worst_H)) - np.mean(np.delete(np.abs(sC["cal"] - y), worst_H))
        )
        overall_cal_gain = mae(sH["cal"], y) - mae(sC["cal"], y)
        overall_raw_gain = mae(sH["raw"], y) - mae(sC["raw"], y)
        g0_mask = k == 0
        g0_worsen = (mae(sC["cal"][g0_mask], y[g0_mask]) - mae(sH["cal"][g0_mask], y[g0_mask])) if g0_mask.any() else None
        ci = comps["C_minus_H"]["gain_cal"]
        point_pass = bool(overall_cal_gain >= 0.003 and overall_raw_gain > 0 and (g0_worsen is not None and g0_worsen <= 0.001))
        single_row = bool(
            (pos.size and imax >= 0 and gain_rows[imax] >= 0.5 * pos.sum()) or (gain_excl_worst <= 0)
        )
        out["C_valid_markers"] = {
            "POINT_EFFECT_PASS": point_pass,
            "overall_cal_gain": overall_cal_gain,
            "overall_raw_gain": overall_raw_gain,
            "G0_cal_worsening": g0_worsen,
            "CI_SUPPORTED": bool(ci["ci95"][0] > 0),
            "cal_ci95": ci["ci95"],
            "SINGLE_ROW_DOMINATED": single_row,
            "max_positive_row": imax,
            "max_positive_row_id": str(labels["ids"][imax]) if imax >= 0 else None,
            "max_positive_row_k": int(k[imax]) if imax >= 0 else None,
            "max_positive_share": float(gain_rows[imax] / pos.sum()) if pos.size and pos.sum() > 0 else None,
            "gain_excluding_H_worst_row": gain_excl_worst,
            "TARGETED_REPAIR_ONLY": bool(point_pass and not (ci["ci95"][0] > 0)),
            "y_cal_below_0.09": bool(mae(sC["cal"], y) < 0.09),
            "y_cal_mae": mae(sC["cal"], y),
        }

    np.savez_compressed(OUT / f"{split}_row_predictions.npz", **per_row)
    out["seconds"] = float(time.perf_counter() - t0)
    write_json(OUT / f"{split}_evaluation.json", out)
    log(f"[{split}] n={n} " + " ".join(
        f"{r['system']}:y_cal={r['y_cal_mae']:.6f}" for r in main_rows) + f" ({out['seconds']:.1f}s)")
    return out


def phase_terminal(log=print) -> dict[str, Any]:
    manifest = json.loads((OUT / "terminal_eval_manifest.json").read_text())
    if not manifest.get("frozen_at_utc"):
        raise RuntimeError("terminal_eval_manifest missing; refusing to read test")
    log("[terminal] valid (exposed/replay) first, then test (post-freeze read)")
    valid_eval = eval_split("valid", log)
    test_eval = eval_split("test", log)
    # actual valid-test gaps per system
    gaps = {}
    for rv, rt in zip(valid_eval["main_table"], test_eval["main_table"]):
        gaps[rv["system"]] = {
            "valid_y_cal": rv["y_cal_mae"], "test_y_cal": rt["y_cal_mae"],
            "gap_test_minus_valid": rt["y_cal_mae"] - rv["y_cal_mae"],
            "valid_y_raw": rv["y_raw_mae"], "test_y_raw": rt["y_raw_mae"],
            "gap_raw": rt["y_raw_mae"] - rv["y_raw_mae"],
        }
    out = {"protocol_version": PROTOCOL_VERSION, "valid_test_gaps": gaps}
    write_json(OUT / "valid_test_gaps.json", out)
    return out


# ---------------------------------------------------------------------------
# phase: analyze
# ---------------------------------------------------------------------------


def phase_analyze(log=print) -> dict[str, Any]:
    files = ("source_identity.json", "folds.json", "gate.json", "D_full_meta.json",
             "terminal_eval_manifest.json", "valid_evaluation.json", "test_evaluation.json",
             "valid_test_gaps.json", "heldout_access.json")
    out = {"protocol_version": PROTOCOL_VERSION, "at_utc": now_utc()}
    for f in files:
        p = OUT / f
        if p.exists():
            out[f.replace(".json", "")] = json.loads(p.read_text())
    write_json(OUT / "analysis.json", out)
    log("[analyze] consolidated analysis.json")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=PROTOCOL_VERSION)
    parser.add_argument("--phase", required=True,
                        choices=("identity", "folds", "train_folds", "fold_metrics", "dfull", "freeze", "terminal", "analyze"))
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "identity":
        phase_identity()
    elif args.phase == "folds":
        phase_folds()
    elif args.phase == "train_folds":
        phase_train_folds()
    elif args.phase == "fold_metrics":
        phase_fold_metrics()
    elif args.phase == "dfull":
        phase_dfull()
    elif args.phase == "freeze":
        phase_freeze()
    elif args.phase == "terminal":
        phase_terminal()
    elif args.phase == "analyze":
        phase_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
