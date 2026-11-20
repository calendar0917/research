"""Shared frozen context for the response-dictionary explanation check.

Read-only reuse of the two source rounds:

* ``response_dictionary_small_s0_v1`` — frozen Lipo RESP best-Val state,
  fragment cache, inventory, dictionary machinery;
* ``response_dictionary_lipo_followup_s0_v1`` — the K=16 dictionary rule and
  the saved stress-test predictions used for the replay identity check.

No training, no optimiser update, no new data.  Everything below is deterministic
linear algebra on the frozen per-fragment cache plus (for the projection stage)
SciPy SLSQP on the unchanged convex panel objective.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
FU_DIR = ROUND_DIR.parent / "response_dictionary_lipo_followup_s0_v1"
FU_CODE = FU_DIR / "code"
S0_DIR = ROUND_DIR.parent / "response_dictionary_small_s0_v1"
S0_CODE = S0_DIR / "code"
S0_RESULTS = S0_DIR / "results"
FU_RESULTS = FU_DIR / "results"

for _path in (FU_CODE, S0_CODE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fu_common as FC  # noqa: E402  (also puts S0 code on sys.path)
import fu_dict_k16 as F16  # noqa: E402
import rd_common as RC  # noqa: E402
import rd_dictionary as RD  # noqa: E402
import rd_dict_run as RDR  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402

CONFIG_PATH = ROUND_DIR / "configs" / "explanation_check_s0.json"
PROTOCOL_PATH = ROUND_DIR / "PROTOCOL.md"
DICT_NAMES = ("D_RESPONSE", "D_LATENT", "D_FREQ")
SPLITS = ("train", "valid", "test")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def protocol_hash() -> str:
    payload = {
        "config": load_config(),
        "protocol_sha256": RC.sha256_file(PROTOCOL_PATH),
        "parent_config_sha256": RC.sha256_file(S0_DIR / "configs" / "rd_small_s0.json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare(device: str = "cpu") -> dict[str, Any]:
    """Load the frozen model, per-fragment cache, inventory and K=16 dictionaries."""
    import torch

    config = load_config()
    recipe = FC.load_parent_config()
    follow = FC.load_config()
    dataset = RT.build_dataset(config["dataset"], config["task"])
    ckpt = S0_RESULTS / f"{config['task']}_{config['source_state']['arm']}" / "best.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = RM.build_model(dataset.dims, RM.ModelConfig(
        hidden=int(recipe["hidden"]), latent=int(recipe["latent"]),
        n_layers=int(recipe["n_layers"]), dropout=float(recipe["dropout"]), mode="RESP"), state)
    model.eval()
    frozen = RD.build_frozen(model, dataset, device=device)
    inventory = RD.build_inventory(RC.fragment_cache(config["dataset"]))
    kmap = F16.k_map(inventory, int(follow["dictionary_stress"]["k_compressed_classes"]),
                     int(recipe["dictionary"]["K"]))
    dicts16 = F16.build_dictionaries(frozen, inventory, kmap,
                                     int(recipe["dictionary"]["panel_size"]))
    compressed = [tau for tau in inventory.class_words
                  if len(inventory.class_words[tau]) > int(recipe["dictionary"]["K"])]
    compressed = sorted(compressed, key=lambda t: -len(inventory.class_words[t]))
    if len(compressed) != 3:
        raise AssertionError(f"expected 3 compressed classes, got {len(compressed)}")
    if sorted(RD.tau_label(t) for t in compressed) != sorted(config["compressed_tau"]):
        raise AssertionError("compressed classes differ from the frozen protocol list")
    std = float(dataset.scaler["std"])
    mean = float(dataset.scaler["mean"])
    return {
        "config": config, "recipe": recipe, "follow": follow,
        "dataset": dataset, "model": model, "frozen": frozen,
        "inventory": inventory, "dicts16": dicts16, "kmap": kmap,
        "compressed": compressed, "std": std, "mean": mean,
        "checkpoint": ckpt,
        "checkpoint_sha256": RC.sha256_file(ckpt),
    }


# ---------------------------------------------------------------------------
# replay / identity
# ---------------------------------------------------------------------------


def replay_checks(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Native F and the three K=16 dictionaries against the saved follow-up files."""
    config = ctx["config"]
    frozen, dataset, inventory = ctx["frozen"], ctx["dataset"], ctx["inventory"]
    dicts16, compressed, kmap = ctx["dicts16"], ctx["compressed"], ctx["kmap"]
    std, mean = ctx["std"], ctx["mean"]
    out: dict[str, Any] = {"checks": {}}

    parent_summary = RC.read_json(S0_RESULTS / f"{config['task']}_RESP" / "dictionary"
                                  / "dictionary_summary.json")
    out["checkpoint_sha256"] = ctx["checkpoint_sha256"]
    out["parent_checkpoint_sha256"] = parent_summary["checkpoint_sha256"]
    if out["checkpoint_sha256"] != out["parent_checkpoint_sha256"]:
        raise AssertionError("checkpoint identity differs from the parent dictionary stage")

    # native RESP F(G) vs the saved parent predictions
    saved_parent = np.load(S0_RESULTS / f"{config['task']}_RESP" / "predictions_best.npz")
    native: dict[str, Any] = {}
    for split_name in SPLITS:
        split = dataset.splits[split_name]
        pred, _, _ = RD.split_scores(frozen, frozen.per_split[split_name], len(split.row_id))
        raw = pred * std + mean
        saved = np.asarray(saved_parent[f"{split_name}_pred_raw"])
        diff = float(np.max(np.abs(raw - saved))) if saved.size else 0.0
        native[split_name] = {
            "max_abs_diff_raw_vs_saved": diff,
            "rmse_raw": RC.rmse(raw, split.label_raw),
            "saved_rmse_raw": RC.rmse(saved, np.asarray(saved_parent[f"{split_name}_target_raw"])),
        }
        if diff > 1e-5:
            raise AssertionError(f"native RESP replay differs on {split_name}: {diff}")
    out["checks"]["native_resp"] = native

    # K=16 word count / candidate count
    n_words = int(sum(len(v) for v in dicts16["D_RESPONSE"]["prototypes"].values()))
    n_cand = int(sum(len(v) for v in inventory.class_words.values()))
    out["n_words"] = n_words
    out["n_candidates"] = n_cand
    out["compressed_classes"] = [RD.tau_label(t) for t in compressed]
    out["compressed_candidates"] = [int(len(inventory.class_words[t])) for t in compressed]
    expected = config["expected"]
    if n_words != int(expected["n_words"]) or n_cand != int(expected["n_candidates"]):
        raise AssertionError(f"word/candidate count changed: {n_words}/{n_cand}")

    # K=16 selections: nested first 16 of the parent K=64 for compressed classes,
    # identical elsewhere; panels and D0 unchanged.
    dicts64 = RDR._build_dictionaries(frozen, inventory, ctx["recipe"]["dictionary"])
    nested, exact = {}, {}
    for name in DICT_NAMES:
        nested_ok = exact_ok = True
        for tau in inventory.class_words:
            p16, p64 = dicts16[name]["prototypes"][tau], dicts64[name]["prototypes"][tau]
            if tau in compressed:
                nested_ok &= (p16 == p64[:16])
            else:
                exact_ok &= (p16 == p64)
        nested[name], exact[name] = bool(nested_ok), bool(exact_ok)
    panels_ok = all(dicts16[name]["panels"][tau] == dicts64[name]["panels"][tau]
                    for name in DICT_NAMES for tau in inventory.class_words)
    d0_ok = all(dicts16[name]["d0"][tau] == dicts64[name]["d0"][tau]
                for name in DICT_NAMES for tau in inventory.class_words)
    out["checks"]["k16_nested_prefix"] = nested
    out["checks"]["other_classes_identical"] = exact
    out["checks"]["panels_identical"] = bool(panels_ok)
    out["checks"]["d0_identical"] = bool(d0_ok)
    out["kmap"] = {RD.tau_label(t): int(k) for t, k in kmap.items()}
    out["n_classes_exact_storage"] = int(sum(
        1 for tau in inventory.class_words if int(kmap[tau]) >= len(inventory.class_words[tau])))
    if not all(nested.values()) or not all(exact.values()) or not panels_ok or not d0_ok:
        raise AssertionError("K=16 dictionary identity check failed")
    if out["n_classes_exact_storage"] != int(expected["n_classes_exact_storage"]):
        raise AssertionError("exact-storage class count changed")

    return out


def check_fw_predictions(ctx: Mapping[str, Any], predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The FW64 recomputation must reproduce the saved stress predictions."""
    saved = np.load(FU_RESULTS / f"{ctx['config']['task']}_RESP" / "dictionary_stress_k16"
                    / "stress_predictions.npz")
    report: dict[str, Any] = {}
    for name in DICT_NAMES:
        for split in SPLITS:
            key = f"{name}_{split}_pred_raw"
            diff = float(np.max(np.abs(predictions[key] - np.asarray(saved[key]))))
            report[f"{name}|{split}"] = diff
            if diff > 1e-9:
                raise AssertionError(f"FW64 replay differs from saved stress predictions: {key} {diff}")
    report["max_abs_diff"] = max(report.values())
    return report


# ---------------------------------------------------------------------------
# words -> scores / predictions
# ---------------------------------------------------------------------------


def word_interfaces(ctx: Mapping[str, Any], alpha: Mapping[int, np.ndarray],
                    support: Mapping[int, bool], name: str):
    frozen, inventory = ctx["frozen"], ctx["inventory"]
    return RDR._word_interfaces(frozen, inventory, ctx["dicts16"][name], alpha, support)


def view_scores(ctx: Mapping[str, Any], a_words: np.ndarray, q_words: np.ndarray,
                s: np.ndarray, c: np.ndarray) -> np.ndarray:
    frozen = ctx["frozen"]
    if s.size == 0:
        return np.zeros(0)
    scale = frozen.q.shape[1] ** 0.5
    t = (q_words[s] * frozen.k[c] * frozen.w[None, :]).sum(axis=1) / scale
    return frozen.b[c] + a_words[s] + t


def predict_all(ctx: Mapping[str, Any], a_words: np.ndarray, q_words: np.ndarray,
                include_views: bool = False) -> dict[str, Any]:
    """Full F(G) for every split, in raw units, plus optional per-view scores."""
    frozen, dataset = ctx["frozen"], ctx["dataset"]
    std, mean = ctx["std"], ctx["mean"]
    out: dict[str, Any] = {"pred_raw": {}, "row_ids": {}, "target": {}}
    if include_views:
        out["view_score_raw"] = {}
    for split_name in SPLITS:
        split = dataset.splits[split_name]
        payload = frozen.per_split[split_name]
        pred_std = RD.predict_with_words(frozen, payload, len(split.row_id),
                                        a_words=a_words, q_words=q_words)
        out["pred_raw"][split_name] = pred_std * std + mean
        out["row_ids"][split_name] = np.asarray(split.row_id, dtype=np.int64)
        out["target"][split_name] = np.asarray(split.label_raw, dtype=np.float64)
        if include_views:
            s = np.asarray(payload["view_s_frag"], dtype=np.int64)
            c = np.asarray(payload["view_c_frag"], dtype=np.int64)
            out["view_score_raw"][split_name] = view_scores(ctx, a_words, q_words, s, c) * std + mean
    return out


def resp_predictions(ctx: Mapping[str, Any]) -> dict[str, Any]:
    frozen, dataset = ctx["frozen"], ctx["dataset"]
    std, mean = ctx["std"], ctx["mean"]
    out = {"pred_raw": {}, "row_ids": {}, "target": {}, "view_score_raw": {}}
    for split_name in SPLITS:
        split = dataset.splits[split_name]
        payload = frozen.per_split[split_name]
        pred, _, _ = RD.split_scores(frozen, payload, len(split.row_id))
        out["pred_raw"][split_name] = pred * std + mean
        out["row_ids"][split_name] = np.asarray(split.row_id, dtype=np.int64)
        out["target"][split_name] = np.asarray(split.label_raw, dtype=np.float64)
        s = np.asarray(payload["view_s_frag"], dtype=np.int64)
        c = np.asarray(payload["view_c_frag"], dtype=np.int64)
        out["view_score_raw"][split_name] = view_scores(
            ctx, frozen.a, frozen.q, s, c) * std + mean
    return out


def raw_lookup(pred: Mapping[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for split_name in SPLITS:
        for row, value in zip(np.asarray(pred["row_ids"][split_name]).tolist(),
                              np.asarray(pred["pred_raw"][split_name]).tolist()):
            out[int(row)] = float(value)
    return out


def metric_row(pred: Mapping[str, Any], resp: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split_name in SPLITS:
        p = np.asarray(pred["pred_raw"][split_name])
        r = np.asarray(resp["pred_raw"][split_name])
        y = np.asarray(pred["target"][split_name])
        rows.append({
            "split": split_name,
            "n_molecules": int(p.size),
            "rmse_raw": RC.rmse(p, y),
            "mae_raw": RC.mae(p, y),
            "resp_rmse_raw": RC.rmse(r, y),
            "resp_mae_raw": RC.mae(r, y),
            "delta_rmse_raw": RC.rmse(p, y) - RC.rmse(r, y),
            "fidelity_rmse_raw": float(np.sqrt(np.mean((p - r) ** 2))),
            "fidelity_mae_raw": float(np.mean(np.abs(p - r))),
            "max_abs_fidelity_raw": float(np.max(np.abs(p - r))),
        })
    return rows


def sha256_array(array: np.ndarray) -> str:
    return RC.sha256_array(np.ascontiguousarray(array))
