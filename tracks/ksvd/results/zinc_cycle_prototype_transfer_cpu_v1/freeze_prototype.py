"""zinc_cycle_prototype_transfer_cpu_v1 — freeze phase.

Builds, from train only:
  * the exact T25 float32 equivalence classes of the frozen 10000-row train,
  * the train-consistent class prototype table (median c_train per class),
  * the frozen key codec and the consistent/conflict member tables,
  * the train-only b_H for candidate H.

Saves a frozen deploy package (COMP + Q + prototype table + key codec + b_H) and a
frozen_eval_manifest. NO valid prediction is produced here. NO optimizer/backward.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

SRC = REPO / "tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1"
RUNNER = REPO / "tracks/ksvd/experiments/luyin16/zinc_component_supervision_fulltrain_confirmation_seed0_v1.py"
OUT = Path(__file__).resolve().parent

PROTOCOL_VERSION = "zinc-cycle-prototype-transfer-cpu-v1"
KEY_DIM = 25
KEY_DTYPE = np.float32


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_arr32(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, np.float32).tobytes()).hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in names})


# ---------------------------------------------------------------------------
# key codec — exact float32 equivalence class, -0 normalised to +0
# ---------------------------------------------------------------------------


def normalize_keys(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != KEY_DIM:
        raise RuntimeError(f"key shape mismatch: {X.shape}")
    X = np.ascontiguousarray(X, KEY_DTYPE)
    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("key contains NaN/Inf")
    # normalise -0 to +0 deterministically
    neg_zero = np.signbit(X) & (X == 0.0)
    if neg_zero.any():
        X = X.copy()
        X[neg_zero] = 0.0
    return X


def key_bytes(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, KEY_DTYPE).tobytes()


def exact_classes(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact float32 equality classes using lexsort; returns (class_id, counts)."""
    X = normalize_keys(X)
    order = np.lexsort(X.T)
    Xs = X[order]
    new_class = np.ones(Xs.shape[0], dtype=bool)
    if Xs.shape[0] > 1:
        new_class[1:] = np.any(Xs[1:] != Xs[:-1], axis=1)
    class_id_sorted = np.cumsum(new_class) - 1
    class_id = np.empty(Xs.shape[0], np.int64)
    class_id[order] = class_id_sorted
    counts = np.bincount(class_id)
    return class_id, counts


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "k=0": k == 0,
        "k=-1": k == -1,
        "k=-2": k == -2,
        "k<=-3": k <= -3,
        "k<=-2": k <= -2,
    }


def group_name(kv: int) -> str:
    if int(kv) == 0:
        return "k=0"
    if int(kv) == -1:
        return "k=-1"
    if int(kv) == -2:
        return "k=-2"
    return "k<=-3"


def stable_id_train(row: int) -> str:
    return f"train:{int(row):04d}"


# ---------------------------------------------------------------------------
# build prototype table
# ---------------------------------------------------------------------------


def build_train_table(
    runner: Any,
) -> dict[str, Any]:
    torch.set_num_threads(8)
    targets = np.load(SRC / "full_train_targets.npz", allow_pickle=False)
    y_train = targets["y"].astype(np.float64)
    c_train = targets["c"].astype(np.float64)
    k_train = targets["k"].astype(np.int64)
    gid_train = targets["gid"].astype(np.int64)

    # integer / finiteness verification of the frozen k
    k_checks = {
        "k_all_finite": bool(np.isfinite(k_train).all()),
        "k_all_integer_valued": bool(np.array_equal(k_train.astype(np.float64), np.round(k_train.astype(np.float64)))),
        "k_min": int(k_train.min()),
        "k_max": int(k_train.max()),
        "gid_is_arange": bool(np.array_equal(gid_train, np.arange(gid_train.size))),
    }
    if not (k_checks["k_all_finite"] and k_checks["k_all_integer_valued"]):
        raise RuntimeError("frozen k failed integer/finiteness verification")

    prep_meta = np.load(SRC / "full_train_prep.npz", allow_pickle=False)
    prep_meta_dict = {key: prep_meta[key] for key in prep_meta.files}
    train_data = runner.build_fulltrain_data(prep_meta_dict)
    T = runner.topology_matrix(train_data)
    T = normalize_keys(T)

    class_id, counts = exact_classes(T)
    n, n_classes = int(T.shape[0]), int(counts.size)

    # constant-level c identity check: c must equal (k - mu_cycle)/sigma_cycle
    constants = {str(nm): float(v) for nm, v in zip(targets["constant_names"].tolist(), targets["constants"].tolist())}
    c_rule = (k_train.astype(np.float64) - constants["mu_cycle"]) / constants["sigma_cycle"]
    c_rule_max_abs = float(np.max(np.abs(c_train - c_rule)))

    # per-class statistics
    class_median = np.empty(n_classes, np.float64)
    class_min = np.empty(n_classes, np.float64)
    class_max = np.empty(n_classes, np.float64)
    class_span = np.empty(n_classes, np.float64)
    class_unique_k = np.empty(n_classes, np.int64)
    class_k_min = np.empty(n_classes, np.int64)
    class_k_max = np.empty(n_classes, np.int64)
    class_consistent = np.empty(n_classes, bool)
    class_key_sha = [""] * n_classes
    class_key_vec = np.empty((n_classes, KEY_DIM), KEY_DTYPE)
    class_members: list[list[int]] = [[] for _ in range(n_classes)]

    for cid in range(n_classes):
        rows = np.where(class_id == cid)[0]
        vals = c_train[rows]
        class_median[cid] = float(np.median(vals))
        class_min[cid] = float(vals.min())
        class_max[cid] = float(vals.max())
        class_span[cid] = float(vals.max() - vals.min())
        ks = k_train[rows]
        class_unique_k[cid] = int(np.unique(ks).size)
        class_k_min[cid] = int(ks.min())
        class_k_max[cid] = int(ks.max())
        class_consistent[cid] = bool(class_unique_k[cid] == 1)
        class_key_vec[cid] = T[rows[0]]
        class_key_sha[cid] = hashlib.sha256(key_bytes(T[rows[0]])).hexdigest()
        class_members[cid] = rows.tolist()

    # collision check: number of unique byte keys must equal class count
    all_keys = set(key_bytes(T[i]) for i in range(n))
    collision_free = bool(len(all_keys) == n_classes)
    if not collision_free:
        raise RuntimeError("full-vector key collision detected")

    # conflict classes (k not unanimous) and their k composition
    conflict_cids = np.where(~class_consistent)[0]
    consistent_cids = np.where(class_consistent)[0]

    # class-consistent c span check: span must be at the c=f(k) precision
    max_span_consistent = float(class_span[consistent_cids].max()) if consistent_cids.size else 0.0
    span_ok = bool(max_span_consistent <= max(1e-12, c_rule_max_abs * 100))

    # train in-class best constant-L1 floor (per-class median)
    class_l1 = np.abs(c_train - class_median[class_id])
    train_floor = float(class_l1.mean())
    train_floor_by_group = {name: (float(class_l1[mask].mean()) if mask.any() else None) for name, mask in group_masks(k_train).items()}

    return {
        "T": T,
        "y_train": y_train,
        "c_train": c_train,
        "k_train": k_train,
        "gid_train": gid_train,
        "class_id": class_id,
        "counts": counts,
        "class_median": class_median,
        "class_min": class_min,
        "class_max": class_max,
        "class_span": class_span,
        "class_unique_k": class_unique_k,
        "class_k_min": class_k_min,
        "class_k_max": class_k_max,
        "class_consistent": class_consistent,
        "class_key_sha": class_key_sha,
        "class_key_vec": class_key_vec,
        "class_members": class_members,
        "n": n,
        "n_classes": n_classes,
        "conflict_cids": conflict_cids,
        "consistent_cids": consistent_cids,
        "c_rule_max_abs": c_rule_max_abs,
        "span_ok": span_ok,
        "max_span_consistent": max_span_consistent,
        "collision_free": collision_free,
        "train_floor": train_floor,
        "train_floor_by_group": train_floor_by_group,
        "k_checks": k_checks,
        "constants": constants,
    }


# ---------------------------------------------------------------------------
# prototype model (frozen, deployable)
# ---------------------------------------------------------------------------


class PrototypeTable:
    """Deterministic train-only T25 exact-class prototype lookup."""

    def __init__(self, key_map: dict[bytes, int], proto_val: np.ndarray, consistent: np.ndarray, class_ids: list[int]):
        self.key_map = key_map              # bytes -> prototype slot
        self.proto_val = proto_val          # (m,) float64 prototype values
        self.consistent = consistent        # (m,) bool
        self.class_ids = class_ids          # slot -> train class id

    def route_and_value(self, row: np.ndarray) -> tuple[str, float | None]:
        slot = self.key_map.get(key_bytes(row))
        if slot is None:
            return "UNSEEN_FALLBACK", None
        if not self.consistent[slot]:
            return "TRAIN_CONFLICT_FALLBACK", None
        return "CONSISTENT_HIT", float(self.proto_val[slot])

    @property
    def n_slots(self) -> int:
        return int(self.proto_val.size)

    @property
    def n_consistent_slots(self) -> int:
        return int(self.consistent.sum())


def build_prototype(table: dict[str, Any]) -> PrototypeTable:
    """Only consistent classes get a prototype slot; conflict classes are held for routing."""
    key_map: dict[bytes, int] = {}
    proto_val: list[float] = []
    consistent: list[bool] = []
    class_ids: list[int] = []
    for cid in range(table["n_classes"]):
        row = table["class_key_vec"][cid]
        key_map[key_bytes(row)] = len(proto_val)
        proto_val.append(float(table["class_median"][cid]))
        consistent.append(bool(table["class_consistent"][cid]))
        class_ids.append(int(cid))
    return PrototypeTable(
        key_map=key_map,
        proto_val=np.asarray(proto_val, np.float64),
        consistent=np.asarray(consistent, bool),
        class_ids=class_ids,
    )


def h_forward(
    h_raw: np.ndarray,
    q_B_raw: np.ndarray,
    q_frozen_raw: np.ndarray,
    proto: PrototypeTable,
    T: np.ndarray,
    b_H: float,
    want_diag: bool = False,
):
    """Candidate H forward.

    h_raw        : frozen COMP body output           (n,)
    q_B_raw      : unused here, kept for signature clarity
    q_frozen_raw : frozen Q output on T              (n,)
    proto        : train prototype table
    T            : current T25 inputs                (n, 25)
    b_H          : train-only candidate bias
    """
    n = int(T.shape[0])
    q_H = np.array(q_frozen_raw, np.float64, copy=True)
    route = np.empty(n, dtype=object)
    for i in range(n):
        r, v = proto.route_and_value(T[i])
        route[i] = r
        if v is not None:
            q_H[i] = v
    y_raw = h_raw + q_H
    y_cal = y_raw + b_H
    if not want_diag:
        return y_cal
    return {"q_H": q_H, "y_raw": y_raw, "y_cal": y_cal, "route": route}


def main() -> int:
    torch.set_num_threads(8)
    runner = load_module(RUNNER, "proto_runner_freeze")

    table = build_train_table(runner)
    proto = build_prototype(table)

    # ---- build H on train (no valid read) ----
    q_train_cache = np.load(SRC / "Q_train_predictions.npz", allow_pickle=False)["q_raw"].astype(np.float64)
    comp_train = np.load(SRC / "COMP_raw_predictions.npz", allow_pickle=False)["raw_soup_train"].astype(np.float64)

    T = table["T"]
    q_H_train = np.array(proto.proto_val[0:0], np.float64)  # placeholder, replaced below
    q_H_train = np.empty(table["n"], np.float64)
    route_train = np.empty(table["n"], dtype=object)
    for i in range(table["n"]):
        r, v = proto.route_and_value(T[i])
        route_train[i] = r
        q_H_train[i] = q_train_cache[i] if v is None else v

    b_H = float(np.median(table["y_train"] - comp_train - q_H_train))

    # ---- routing counts on train by group ----
    route_train_counts: dict[str, Any] = {}
    for r in ("CONSISTENT_HIT", "TRAIN_CONFLICT_FALLBACK", "UNSEEN_FALLBACK"):
        mask = route_train == r
        route_train_counts[r] = {
            "n": int(mask.sum()),
            "by_group": {name: int((mask & gm).sum()) for name, gm in group_masks(table["k_train"]).items() if name != "k<=-2"},
        }

    # ---- train q error comparison ----
    q_err_B = float(np.mean(np.abs(q_train_cache - table["c_train"])))
    q_err_H = float(np.mean(np.abs(q_H_train - table["c_train"])))
    y_raw_B = comp_train + q_train_cache
    y_cal_B = y_raw_B + (-0.011630002409219742)
    y_raw_H = comp_train + q_H_train
    y_cal_H = y_raw_H + b_H
    train_metrics = {
        "q_mae_B_vs_c": q_err_B,
        "q_mae_H_vs_c": q_err_H,
        "y_raw_mae_B": float(np.mean(np.abs(table["y_train"] - y_raw_B))),
        "y_cal_mae_B": float(np.mean(np.abs(table["y_train"] - y_cal_B))),
        "y_raw_mae_H": float(np.mean(np.abs(table["y_train"] - y_raw_H))),
        "y_cal_mae_H": float(np.mean(np.abs(table["y_train"] - y_cal_H))),
        "b_H": b_H,
        "b_y_COMP": -0.011630002409219742,
        "delta_b": b_H - (-0.011630002409219742),
    }

    # ---- leave-one-out coverage diagnostic (no new model) ----
    loo_lost_support = 0
    loo_still_consistent = 0
    loo_now_conflict = 0
    for i in range(table["n"]):
        cid = int(table["class_id"][i])
        members = table["class_members"][cid]
        remaining = [r for r in members if r != i]
        if not remaining:
            loo_lost_support += 1
            continue
        ks = table["k_train"][np.asarray(remaining)]
        if np.unique(ks).size == 1:
            loo_still_consistent += 1
        else:
            loo_now_conflict += 1
    loo = {
        "n_train_rows": table["n"],
        "lost_support_singletons": int(loo_lost_support),
        "still_consistent": int(loo_still_consistent),
        "now_conflict": int(loo_now_conflict),
        "note": "diagnostic only; does not replace the formal b_H/prediction rule",
    }

    # ---- train class tables ----
    class_rows = []
    for cid in range(table["n_classes"]):
        members = table["class_members"][cid]
        class_rows.append({
            "class_id": cid,
            "n_support": int(table["counts"][cid]),
            "k_unique": int(table["class_unique_k"][cid]),
            "k_min": int(table["class_k_min"][cid]),
            "k_max": int(table["class_k_max"][cid]),
            "is_consistent": bool(table["class_consistent"][cid]),
            "median_c": float(table["class_median"][cid]),
            "min_c": float(table["class_min"][cid]),
            "max_c": float(table["class_max"][cid]),
            "c_span": float(table["class_span"][cid]),
            "k_group": group_name(int(table["class_k_min"][cid])),
            "key_sha256": table["class_key_sha"][cid],
            "member_row_ids": ";".join(stable_id_train(r) for r in members),
        })
    write_csv(OUT / "train_table.csv", class_rows)
    write_json(OUT / "train_table.json", {
        "n_train_rows": table["n"],
        "n_exact_classes": table["n_classes"],
        "n_consistent_classes": int(table["class_consistent"].sum()),
        "n_conflict_classes": int((~table["class_consistent"]).sum()),
        "n_singleton_classes": int((table["counts"] == 1).sum()),
        "n_multi_classes": int((table["counts"] >= 2).sum()),
        "conflict_class_ids": [int(x) for x in table["conflict_cids"]],
        "consistent_class_ids": [int(x) for x in table["consistent_cids"]],
        "collision_free": table["collision_free"],
        "c_rule_max_abs": table["c_rule_max_abs"],
        "max_span_consistent": table["max_span_consistent"],
        "span_ok": table["span_ok"],
        "k_checks": table["k_checks"],
        "train_class_constant_l1_floor": table["train_floor"],
        "train_class_constant_l1_floor_by_group": table["train_floor_by_group"],
        "class_table": class_rows,
    })

    # ---- member table (all train rows) ----
    member_rows = []
    for i in range(table["n"]):
        cid = int(table["class_id"][i])
        member_rows.append({
            "train_row": i,
            "stable_id": stable_id_train(i),
            "class_id": cid,
            "is_consistent_class": bool(table["class_consistent"][cid]),
            "class_n_support": int(table["counts"][cid]),
            "class_k_unique": int(table["class_unique_k"][cid]),
            "k": int(table["k_train"][i]),
            "group": group_name(int(table["k_train"][i])),
            "c": float(table["c_train"][i]),
            "y": float(table["y_train"][i]),
            "class_median_c": float(table["class_median"][cid]),
            "q_B_raw": float(q_train_cache[i]),
            "q_H_raw": float(q_H_train[i]),
            "route": str(route_train[i]),
            "q_err_B": float(q_train_cache[i] - table["c_train"][i]),
            "q_err_H": float(q_H_train[i] - table["c_train"][i]),
            "y_err_B_cal": float(table["y_train"][i] - y_cal_B[i]),
            "y_err_H_cal": float(table["y_train"][i] - y_cal_H[i]),
        })
    write_csv(OUT / "per_row_train.csv", member_rows)

    # ---- conflict class detail ----
    conflict_rows = []
    for cid in table["conflict_cids"].tolist():
        members = table["class_members"][cid]
        conflict_rows.append({
            "class_id": int(cid),
            "n_support": int(table["counts"][cid]),
            "k_values": ";".join(str(int(x)) for x in np.unique(table["k_train"][np.asarray(members)])),
            "member_ids": ";".join(stable_id_train(r) for r in members),
            "member_ks": ";".join(str(int(table["k_train"][r])) for r in members),
            "member_cs": ";".join(f"{float(table['c_train'][r]):.10f}" for r in members),
            "q_B_errors": ";".join(f"{float(q_train_cache[r] - table['c_train'][r]):.10f}" for r in members),
            "key_sha256": table["class_key_sha"][cid],
        })
    write_csv(OUT / "train_conflict_classes.csv", conflict_rows)

    write_json(OUT / "train_bias.json", {
        "b_H": b_H,
        "b_y_COMP": -0.011630002409219742,
        "delta_b": b_H - (-0.011630002409219742),
        "rule": "median(y_train - h_raw_train - q_H_raw_train)",
        "uses_valid": False,
    })
    write_json(OUT / "train_metrics.json", train_metrics)
    write_json(OUT / "loo_coverage.json", loo)
    write_json(OUT / "train_routing.json", route_train_counts)

    # ---- frozen deploy package (single file, reloadable) ----
    package = {
        "protocol_version": PROTOCOL_VERSION,
        "key_dim": KEY_DIM,
        "key_dtype": "float32",
        "key_encoding": "contiguous little-endian float32, -0 normalised to +0, NaN/Inf forbidden",
        "key_order": [key.hex() for key in sorted(proto.key_map.keys())],
        "key_slot": [int(proto.key_map[key]) for key in sorted(proto.key_map.keys())],
        "proto_val": proto.proto_val.astype(np.float64),
        "consistent": proto.consistent,
        "class_ids": np.asarray(proto.class_ids, np.int64),
        "class_median": table["class_median"],
        "class_counts": table["counts"],
        "n_classes": int(table["n_classes"]),
        "b_H": float(b_H),
        "b_y_COMP": -0.011630002409219742,
        "inference_rule": (
            "q_H = train_table[key].median_c if key in train_table and k_is_unanimous "
            "else frozen_Q(T25); y = h_raw + q_H + b_H"
        ),
        "source_dir": str(SRC.relative_to(REPO)),
        "execution_commit": "947d2837c4936ef1276fcbbf0e9189e7d44ace23",
    }
    torch.save(package, OUT / "H_model_package.pt")

    # also save a pure-numpy table for standalone reload
    np.savez_compressed(
        OUT / "prototype_table.npz",
        keys=np.asarray([k for k in sorted(proto.key_map.keys())], dtype="S100"),
        key_slot=np.asarray([proto.key_map[k] for k in sorted(proto.key_map.keys())], np.int64),
        proto_val=proto.proto_val.astype(np.float64),
        consistent=proto.consistent,
        class_ids=np.asarray(proto.class_ids, np.int64),
        class_median=table["class_median"],
        class_counts=table["counts"],
        key_dim=np.asarray([KEY_DIM], np.int64),
        b_H=np.asarray([b_H], np.float64),
    )
    np.savez_compressed(
        OUT / "T25_cache.npz",
        T_train=table["T"].astype(np.float32),
        c_train=table["c_train"],
        k_train=table["k_train"],
        y_train=table["y_train"],
        class_id=table["class_id"],
        gid_train=table["gid_train"],
        T25_train_sha256=np.asarray([sha_arr32(table["T"])]),
    )

    # ---- frozen eval manifest ----
    manifest = {
        "manifest_version": "frozen-eval-manifest-v1",
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "execution_commit": "947d2837c4936ef1276fcbbf0e9189e7d44ace23",
        "artifacts": {
            "source_dir": str(SRC.relative_to(REPO)),
            "runner_source": {"path": str(RUNNER.relative_to(REPO)), "sha256": sha256_file(RUNNER)},
            "targets": {"path": str((SRC / "full_train_targets.npz").relative_to(REPO)), "sha256": sha256_file(SRC / "full_train_targets.npz")},
            "prep": {"path": str((SRC / "full_train_prep.npz").relative_to(REPO)), "sha256": sha256_file(SRC / "full_train_prep.npz")},
            "COMP_soup": {"path": str((SRC / "COMP_raw_soup_state.pt").relative_to(REPO)), "sha256": sha256_file(SRC / "COMP_raw_soup_state.pt")},
            "Q_soup": {"path": str((SRC / "Q_raw_soup_state.pt").relative_to(REPO)), "sha256": sha256_file(SRC / "Q_raw_soup_state.pt")},
            "prototype_table": {"path": "prototype_table.npz", "sha256": sha256_file(OUT / "prototype_table.npz")},
            "H_model_package": {"path": "H_model_package.pt", "sha256": sha256_file(OUT / "H_model_package.pt")},
        },
        "key": {
            "dim": KEY_DIM,
            "dtype": "float32",
            "encoding": "contiguous little-endian float32, -0 normalised to +0, NaN/Inf forbidden",
            "collision_free": bool(table["collision_free"]),
            "n_classes": int(table["n_classes"]),
        },
        "prototype": {
            "n_slots": int(proto.n_slots),
            "n_consistent_slots": int(proto.n_consistent_slots),
            "n_conflict_slots": int(proto.n_slots - proto.n_consistent_slots),
            "rule": "median(c_train) per train class; conflict classes held for fallback routing",
        },
        "bias": {"b_H": float(b_H), "b_y_COMP": -0.011630002409219742},
        "prediction_rule": "if input_key in train_table and k_is_unanimous: q_H = median_c else q_H = frozen_Q(T25); y = h_raw + q_H + b_H",
        "official_valid_loaded_in_freeze": False,
        "official_test_loaded": False,
        "freeze_before_heldout": True,
    }
    write_json(OUT / "frozen_eval_manifest.json", manifest)

    # ---- budget ----
    write_json(OUT / "budget.json", {
        "optimizer": False,
        "backward": False,
        "training_steps": 0,
        "gpu_hours": 0.0,
        "remote_jobs": 0,
        "cpu_threads_max": 8,
        "official_valid_read_this_phase": False,
        "official_test_read": False,
    })

    print(f"[freeze] n_classes={table['n_classes']} consistent={int(table['class_consistent'].sum())} "
          f"conflict={int((~table['class_consistent']).sum())} singletons={int((table['counts']==1).sum())}")
    print(f"[freeze] b_H={b_H:.12f} b_y_COMP={-0.011630002409219742:.12f} delta={b_H-(-0.011630002409219742):.12f}")
    print(f"[freeze] train q_mae_B={q_err_B:.8f} q_mae_H={q_err_H:.8f}")
    print(f"[freeze] train y_cal_mae_B={train_metrics['y_cal_mae_B']:.8f} y_cal_mae_H={train_metrics['y_cal_mae_H']:.8f}")
    print(f"[freeze] train routing: {route_train_counts}")
    print("[freeze] H package + prototype table + manifest written; no valid prediction produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
