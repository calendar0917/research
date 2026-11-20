"""Frozen-RESP dictionary stress test at fixed K=16 (Lipo, S0 best-Val RESP).

Read-only on the parent round.  The script

1. replays the parent dictionary stage (native RESP scores and the three K=64
   dictionaries) against the saved parent predictions and records the identity
   check;
2. rebuilds the three dictionaries with the **only** changed rule: the three
   ``tau`` classes with more than 64 Train candidates drop from 64 to 16
   prototypes, the other 13 keep exact storage;
3. reports full Train/Val/Test RMSE/MAE, prediction fidelity
   ``sqrt(mean((F_D - F_RESP)^2))``, the structure-defined subset of molecules
   that use one of the three compressed classes, per-view seen/unseen
   fragments, alpha/simplex diagnostics and worst samples;
4. reports the response-dimension diagnostics of the three compressed classes
   on the frozen panel;
5. runs the frozen-model intervention "replace ``k(C)`` by the Train panel mean
   ``k`` of its class" with ``a``, ``b``, ``q``, ``w`` and all fallback paths
   untouched.

No training, no GNN re-run per environment x candidate pair, no K sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import fu_common as FC  # noqa: E402
import rd_common as RC  # noqa: E402
import rd_dictionary as RD  # noqa: E402
import rd_dict_run as RDR  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402

TASK = "lipo"
DATASET = "ogbg-mollipo"
DICT_NAMES = ("D_RESPONSE", "D_LATENT", "D_FREQ")
SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# dictionaries with a per-class K map
# ---------------------------------------------------------------------------


def build_dictionaries(frozen, inventory, class_k: Mapping[tuple, int], panel_size: int):
    """Parent ``_build_dictionaries`` logic with an explicit K per tau class."""
    out: dict[str, dict] = {}
    for name in DICT_NAMES:
        out[name] = {
            "prototypes": {}, "panels": {}, "d0": {}, "n_candidates": {},
            "word_set": {}, "fw_steps": {}, "class_specs": {},
        }
    classes = sorted(inventory.class_words, key=lambda t: RD.tau_label(t))
    for tau in classes:
        words = inventory.class_words[tau]
        d0 = RD.reference_fragment(inventory, tau)
        envs = RD.environment_panel(inventory, tau, panel_size)
        if not envs:
            continue
        k = int(class_k[tau])
        cand = [int(f) for f in words]
        d0_local = cand.index(int(d0))
        r_cand = RD.response_matrix(frozen, cand, d0, envs)
        latent = np.concatenate([frozen.a[cand][:, None], frozen.q[cand]], axis=1)
        latent_norm = np.linalg.norm(latent, axis=1, keepdims=True)
        latent_unit = latent / np.where(latent_norm > 0, latent_norm, 1.0)
        tie_keys = [inventory.keys[int(f)] for f in cand]
        selections = {
            "D_RESPONSE": RD.farthest_first(r_cand, d0_local, k, "l2", keys=tie_keys),
            "D_LATENT": RD.farthest_first(latent_unit, d0_local, k, "cos", keys=tie_keys),
            "D_FREQ": RD.frequency_selection(inventory, cand, d0_local, k),
        }
        for name, local_indices in selections.items():
            protos = [int(cand[i]) for i in local_indices]
            if d0 not in protos:
                raise AssertionError("D0 missing from a dictionary")
            if len(set(protos)) != len(protos):
                raise AssertionError("duplicate prototype")
            out[name]["prototypes"][tau] = protos
            out[name]["panels"][tau] = envs
            out[name]["d0"][tau] = int(d0)
            out[name]["n_candidates"][tau] = int(len(cand))
            out[name]["word_set"][tau] = set(protos)
            out[name]["class_specs"][tau] = {
                "word_mol_freq": {int(f): int(inventory.word_mol_freq.get(f, 0)) for f in protos},
                "panel_freq": [int(inventory.env_mol_freq.get(int(e), 0)) for e in envs],
            }
    for name in out:
        flat: dict[int, bool] = {}
        for tau, protos in out[name]["prototypes"].items():
            for f in protos:
                flat[int(f)] = True
        out[name]["word_set"] = flat
    return out


def k_map(inventory, k_compressed: int, k_cap: int = 64) -> dict[tuple, int]:
    out = {}
    for tau, words in inventory.class_words.items():
        n = len(words)
        out[tau] = int(k_compressed) if n > int(k_cap) else min(int(k_cap), n)
    return out


# ---------------------------------------------------------------------------
# predictors that keep per-view tensors (needed for seen/unseen fidelity)
# ---------------------------------------------------------------------------


def dictionary_views(frozen, split_payload: Mapping[str, np.ndarray], a_words, q_words):
    s = np.asarray(split_payload["view_s_frag"], dtype=np.int64)
    c = np.asarray(split_payload["view_c_frag"], dtype=np.int64)
    scale = frozen.q.shape[1] ** 0.5
    t_new = (q_words[s] * frozen.k[c] * frozen.w[None, :]).sum(axis=1) / scale
    t_old = (frozen.q[s] * frozen.k[c] * frozen.w[None, :]).sum(axis=1) / scale
    s_new = frozen.b[c] + a_words[s] + t_new
    s_old = frozen.b[c] + frozen.a[s] + t_old
    return {"s": s, "c": c, "new": s_new, "old": s_old}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0-results", default=str(FC.S0_RESULTS))
    parser.add_argument("--out", default=str(FC.run_dir(TASK, "RESP") / "dictionary_stress_k16"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    import torch

    recipe = FC.load_parent_config()
    follow = FC.load_config()
    stress_cfg = follow["dictionary_stress"]
    s0 = Path(args.s0_results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    dataset = RT.build_dataset(DATASET, TASK)
    ckpt = s0 / "lipo_RESP" / "best.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = RM.build_model(dataset.dims, RM.ModelConfig(
        hidden=int(recipe["hidden"]), latent=int(recipe["latent"]),
        n_layers=int(recipe["n_layers"]), dropout=float(recipe["dropout"]), mode="RESP"), state)
    model.eval()

    frozen = RD.build_frozen(model, dataset, device=args.device)
    inventory = RD.build_inventory(RC.fragment_cache(DATASET))
    std = float(dataset.scaler["std"])
    mean = float(dataset.scaler["mean"])
    n_cand_total = int(sum(len(v) for v in inventory.class_words.values()))

    # -- A. replay parent stage ------------------------------------------------
    replay = _replay_parent(frozen, dataset, inventory, recipe, s0, std, mean, ckpt)

    # -- B. K map and the only change -----------------------------------------
    kmap = k_map(inventory, int(stress_cfg["k_compressed_classes"]),
                 int(recipe["dictionary"]["K"]))
    compressed = [tau for tau, k in kmap.items()
                  if len(inventory.class_words[tau]) > int(recipe["dictionary"]["K"])]
    compressed = sorted(compressed, key=lambda t: -len(inventory.class_words[t]))
    if len(compressed) != 3:
        raise AssertionError(f"expected 3 compressed classes, got {len(compressed)}")
    dicts16 = build_dictionaries(frozen, inventory, kmap, int(recipe["dictionary"]["panel_size"]))
    identity = _identity_checks(dicts16, replay["dicts64"], inventory, compressed)

    words16 = int(sum(len(v) for v in dicts16["D_RESPONSE"]["prototypes"].values()))
    expected_words = int(stress_cfg["expected_words"])
    if words16 != expected_words:
        raise AssertionError(f"K=16 word count {words16} != {expected_words}")
    if n_cand_total != int(stress_cfg["expected_candidates"]):
        raise AssertionError(f"candidate count {n_cand_total} changed")

    # -- C. alpha + predictions ------------------------------------------------
    alpha, support = RDR._alpha_all(frozen, inventory, dicts16, recipe["dictionary"])
    alpha_info = _alpha_diagnostics(frozen, inventory, dicts16, alpha, support, recipe)

    # -- D. metrics -----------------------------------------------------------
    rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}
    for name in DICT_NAMES:
        a_words, q_words = RDR._word_interfaces(frozen, inventory, dicts16[name], alpha[name], support[name])
        for split_name in SPLITS:
            split = dataset.splits[split_name]
            payload = frozen.per_split[split_name]
            n_mol = len(split.row_id)
            n_views = int(np.asarray(payload["view_s_frag"]).size)
            view = dictionary_views(frozen, payload, a_words, q_words)
            pred_std = molecule_mean(payload, view["new"], n_mol, frozen)
            resp_std = molecule_mean(payload, view["old"], n_mol, frozen)
            reference_pred = RD.predict_with_words(frozen, payload, n_mol,
                                                   a_words=a_words, q_words=q_words)
            cross_check = float(np.max(np.abs(pred_std - reference_pred)))
            if cross_check > 1e-10:
                raise AssertionError(f"{name} {split_name}: per-view and parent prediction paths differ")
            reference_resp, _, _ = RD.split_scores(frozen, payload, n_mol)
            if float(np.max(np.abs(resp_std - reference_resp))) > 1e-10:
                raise AssertionError("native RESP path differs from split_scores")
            pred_raw = pred_std * std + mean
            resp_raw = resp_std * std + mean
            target = split.label_raw
            support_view = np.asarray([bool(support[name].get(int(f), False))
                                       for f in np.asarray(payload["view_s_frag"])], dtype=bool) \
                if n_views else np.zeros(0, dtype=bool)
            exact_view = np.asarray([bool(dicts16[name]["word_set"].get(int(f), False))
                                     for f in np.asarray(payload["view_s_frag"])], dtype=bool) \
                if n_views else np.zeros(0, dtype=bool)
            fb = np.asarray(payload["fallback_frag"]) >= 0
            taus_view = _view_taus(inventory, payload)
            in_subset = _molecule_subset(payload, taus_view, set(compressed), n_mol)
            rows.append({
                "dictionary": name, "split": split_name, "n_molecules": n_mol,
                "n_views": n_views,
                "rmse_raw": RC.rmse(pred_raw, target),
                "mae_raw": RC.mae(pred_raw, target),
                "resp_rmse_raw": RC.rmse(resp_raw, target),
                "resp_mae_raw": RC.mae(resp_raw, target),
                "delta_rmse_raw": RC.rmse(pred_raw, target) - RC.rmse(resp_raw, target),
                "fidelity_rmse_raw": float(np.sqrt(np.mean((pred_raw - resp_raw) ** 2))),
                "supported_view_fraction": float(support_view.mean()) if n_views else 0.0,
                "exact_word_view_fraction": float(exact_view.mean()) if n_views else 0.0,
                "fallback_molecule_fraction": float(fb.mean()),
                "n_subset_molecules": int(in_subset.sum()),
            })
            if in_subset.any():
                subset_rows.append({
                    "dictionary": name, "split": split_name,
                    "n_molecules": int(in_subset.sum()),
                    "n_views": int(np.isin(
                        np.asarray(payload["view_mol_local"]), np.flatnonzero(in_subset)).sum()),
                    "rmse_raw": RC.rmse(pred_raw[in_subset], target[in_subset]),
                    "mae_raw": RC.mae(pred_raw[in_subset], target[in_subset]),
                    "resp_rmse_raw": RC.rmse(resp_raw[in_subset], target[in_subset]),
                    "resp_mae_raw": RC.mae(resp_raw[in_subset], target[in_subset]),
                    "fidelity_rmse_raw": float(np.sqrt(np.mean(
                        (pred_raw[in_subset] - resp_raw[in_subset]) ** 2))),
                })
            # per-class fidelity (by actual view frequency in this split)
            if n_views:
                cls_diff = (view["new"] - view["old"]) * std
                lab = np.asarray([RD.tau_label(t) for t in taus_view], dtype=object)
                for label in sorted(set(lab.tolist())):
                    mask = lab == label
                    class_rows.append({
                        "dictionary": name, "split": split_name, "tau": str(label),
                        "n_views": int(mask.sum()),
                        "fidelity_rmse_raw": float(np.sqrt(np.mean(cls_diff[mask] ** 2))),
                        "fidelity_mae_raw": float(np.mean(np.abs(cls_diff[mask]))),
                        "share_of_split_squared_fidelity_change": float(
                            np.sum(cls_diff[mask] ** 2) / max(1e-12, np.sum(cls_diff ** 2))),
                    })
            # per-view seen/unseen fragment fidelity
            seen = np.asarray([int(f) in inventory.word_mol_freq
                               for f in np.asarray(payload["view_s_frag"])], dtype=bool) if n_views else np.zeros(0, dtype=bool)
            for label, mask in (("train_seen", seen), ("train_unseen", ~seen)):
                if n_views and mask.any():
                    diff = view["new"][mask] - view["old"][mask]
                    view_rows.append({
                        "dictionary": name, "split": split_name, "group": label,
                        "n_views": int(mask.sum()),
                        "n_unique_word_fragments": int(len(set(np.asarray(payload["view_s_frag"])[mask].tolist()))),
                        "fidelity_rmse_raw": float(np.sqrt(np.mean(diff ** 2)) * std),
                        "fidelity_mae_raw": float(np.mean(np.abs(diff)) * std),
                        "supported_view_fraction": float(support_view[mask].mean()),
                    })
            preds[f"{name}_{split_name}_pred_raw"] = pred_raw
            preds[f"{name}_{split_name}_resp_pred_raw"] = resp_raw
            preds[f"{name}_{split_name}_row_ids"] = split.row_id
            preds[f"{name}_{split_name}_target_raw"] = target
            # worst fidelity samples
            order = np.argsort(-np.abs(pred_raw - resp_raw))[:5]
            for rank, i in enumerate(order.tolist()):
                worst.append({
                    "dictionary": name, "split": split_name, "rank": rank + 1,
                    "row_id": int(split.row_id[i]),
                    "y_raw": float(target[i]),
                    "pred_raw": float(pred_raw[i]),
                    "resp_pred_raw": float(resp_raw[i]),
                    "delta_pred_raw": float(pred_raw[i] - resp_raw[i]),
                    "abs_delta_pred_raw": float(abs(pred_raw[i] - resp_raw[i])),
                })
    np.savez_compressed(out / "stress_predictions.npz", **preds)

    # -- E. counts / coverage by actual view frequency ------------------------
    counts, view_freq = _counts_and_coverage(inventory, dicts16, frozen, dataset, compressed)

    # -- F. response dimensions ----------------------------------------------
    response = _response_dimensions(frozen, inventory, compressed)

    # -- G. k(C) -> panel mean intervention ----------------------------------
    intervention = _panel_mean_intervention(frozen, inventory, dataset, std, mean)

    # -- H. write -------------------------------------------------------------
    RC.write_csv(out / "stress_table.csv", rows)
    RC.write_csv(out / "stress_subset_table.csv", subset_rows)
    RC.write_csv(out / "stress_view_groups.csv", view_rows)
    RC.write_csv(out / "stress_class_fidelity.csv", class_rows)
    RC.write_csv(out / "parent_k64_table.csv", replay["table"])
    RC.write_csv(out / "stress_counts.csv", counts)
    RC.write_csv(out / "stress_view_frequency.csv", view_freq)
    RC.write_csv(out / "stress_worst_samples.csv", worst)
    RC.write_json(out / "replay.json", replay["report"])
    RC.write_json(out / "identity_checks.json", identity)
    RC.write_json(out / "alpha_diagnostics.json", alpha_info)
    RC.write_json(out / "response_dimensions.json", response)
    RC.write_json(out / "panel_intervention.json", intervention)

    table = {(r["dictionary"], r["split"]): r for r in rows}
    decision = _signal_decision(table, std, follow["reference_gain"], alpha_info["summary"])
    summary = {
        "task": TASK, "dataset": DATASET,
        "parent_round": follow["parent_round"],
        "checkpoint": str(ckpt), "checkpoint_sha256": RC.sha256_file(ckpt),
        "parent_best_epoch": 63,
        "k_rule": stress_cfg["k_rule"],
        "compressed_classes": [RD.tau_label(t) for t in compressed],
        "compressed_candidates": [int(len(inventory.class_words[t])) for t in compressed],
        "n_words_total": words16, "n_candidate_words_total": n_cand_total,
        "n_classes": int(len(inventory.class_words)),
        "n_classes_exact_storage": int(sum(1 for tau in inventory.class_words
                                           if kmap[tau] >= len(inventory.class_words[tau]))),
        "table": rows, "subset_table": subset_rows, "view_groups": view_rows,
        "replay": replay["report"], "identity_checks": identity,
        "alpha": alpha_info["summary"], "response_dimensions": response,
        "panel_intervention": {k: v for k, v in intervention.items() if k != "per_split_raw"},
        "decision": decision,
        "seconds": time.perf_counter() - t_start,
        "protocol_hash": FC.protocol_hash(),
        "commit": RC.git_commit(),
        "test_use": follow["test_use"],
    }
    RC.write_json(out / "stress_summary.json", summary)
    print(json.dumps(RC.jsonable({k: summary[k] for k in (
        "compressed_classes", "compressed_candidates", "n_words_total",
        "n_classes_exact_storage", "decision", "seconds")}), indent=2))
    for row in rows:
        print(f"{row['dictionary']:11s} {row['split']:5s} rmse {row['rmse_raw']:.4f} "
              f"(resp {row['resp_rmse_raw']:.4f}, d {row['delta_rmse_raw']:+.4f}) "
              f"fid {row['fidelity_rmse_raw']:.4f} support {row['supported_view_fraction']:.3f}")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def molecule_mean(payload, view_scores, n_mol, frozen):
    mol = np.asarray(payload["view_mol_local"], dtype=np.int64)
    sums = np.bincount(mol, weights=view_scores, minlength=n_mol) if view_scores.size else np.zeros(n_mol)
    counts = np.bincount(mol, minlength=n_mol).astype(np.float64)
    has = counts > 0
    out = np.zeros(n_mol)
    out[has] = sums[has] / counts[has]
    fb_frag = np.asarray(payload["fallback_frag"], dtype=np.int64)
    fb = fb_frag >= 0
    if bool(fb.any()):
        out[fb] = frozen.fallback[fb_frag[fb]]
    return out


def _view_taus(inventory, payload) -> list:
    """Python list of tau tuples, one per view (tuples must not become an array)."""
    s = np.asarray(payload["view_s_frag"], dtype=np.int64)
    return [inventory.tau[int(f)] for f in s] if s.size else []


def _molecule_subset(payload, taus_view, compressed: set, n_mol: int) -> np.ndarray:
    mol = np.asarray(payload["view_mol_local"], dtype=np.int64)
    flag = np.asarray([bool(t in compressed) for t in taus_view], dtype=bool) if len(taus_view) else np.zeros(0, dtype=bool)
    out = np.zeros(n_mol, dtype=bool)
    if flag.size:
        out[np.unique(mol[flag])] = True
    return out


def _replay_parent(frozen, dataset, inventory, recipe, s0: Path, std, mean, ckpt) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {"checkpoint": str(ckpt), "checks": {}}
    saved_dc_summary = RC.read_json(s0 / "lipo_RESP" / "dictionary" / "dictionary_summary.json")
    report["checkpoint_sha256"] = RC.sha256_file(ckpt)
    report["parent_checkpoint_sha256"] = saved_dc_summary["checkpoint_sha256"]
    if report["checkpoint_sha256"] != report["parent_checkpoint_sha256"]:
        raise AssertionError("checkpoint identity differs from the parent dictionary stage")

    # native RESP F(G) vs the saved parent predictions
    native = {}
    stored = np.load(s0 / "lipo_RESP" / "predictions_best.npz")
    for split_name in SPLITS:
        split = dataset.splits[split_name]
        pred, _, _ = RD.split_scores(frozen, frozen.per_split[split_name], len(split.row_id))
        raw = pred * std + mean
        saved = np.asarray(stored[f"{split_name}_pred_raw"])
        native[split_name] = {
            "max_abs_diff_raw": float(np.max(np.abs(raw - saved))),
            "cache_rmse_raw": RC.rmse(raw, split.label_raw),
            "saved_rmse_raw": RC.rmse(saved, np.asarray(stored[f"{split_name}_target_raw"])),
        }
        if native[split_name]["max_abs_diff_raw"] > 1e-5:
            raise AssertionError(f"native replay differs on {split_name}")
    report["checks"]["native_resp"] = native

    # the three K=64 dictionaries through the parent code
    dicts64 = RDR._build_dictionaries(frozen, inventory, recipe["dictionary"])
    alpha64, support64 = RDR._alpha_all(frozen, inventory, dicts64, recipe["dictionary"])
    saved_dict = np.load(s0 / "lipo_RESP" / "dictionary" / "dictionary_predictions.npz")
    k64: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    for name in DICT_NAMES:
        a_words, q_words = RDR._word_interfaces(frozen, inventory, dicts64[name], alpha64[name], support64[name])
        per_split = {}
        for split_name in SPLITS:
            split = dataset.splits[split_name]
            payload = frozen.per_split[split_name]
            # parent prediction path (identical function, identical inputs)
            pred_d = RD.predict_with_words(frozen, payload, len(split.row_id),
                                           a_words=a_words, q_words=q_words)
            raw = pred_d * std + mean
            resp_raw = RD.split_scores(frozen, payload, len(split.row_id))[0] * std + mean
            saved = np.asarray(saved_dict[f"{name}_{split_name}_pred_raw"])
            diff = float(np.max(np.abs(raw - saved)))
            table.append({
                "dictionary": name, "split": split_name, "k_mode": "parent_K64",
                "n_words": int(sum(len(v) for v in dicts64[name]["prototypes"].values())),
                "n_candidates": int(sum(dicts64[name]["n_candidates"].values())),
                "rmse_raw": RC.rmse(raw, split.label_raw),
                "mae_raw": RC.mae(raw, split.label_raw),
                "resp_rmse_raw": RC.rmse(resp_raw, split.label_raw),
                "resp_mae_raw": RC.mae(resp_raw, split.label_raw),
                "delta_rmse_raw": RC.rmse(raw, split.label_raw) - RC.rmse(resp_raw, split.label_raw),
                "fidelity_rmse_raw": float(np.sqrt(np.mean((raw - resp_raw) ** 2))),
                "max_abs_diff_vs_saved_raw": diff,
            })
            per_split[split_name] = {
                "max_abs_diff_raw": diff,
                "rmse_raw": RC.rmse(raw, split.label_raw),
                "saved_rmse_raw": RC.rmse(saved, np.asarray(saved_dict[f"{name}_{split_name}_target_raw"])),
                "mae_raw": RC.mae(raw, split.label_raw),
            }
            if diff > 1e-6:
                raise AssertionError(f"{name} {split_name} replay differs by {diff}")
        k64[name] = per_split
    report["checks"]["dictionaries_k64"] = k64
    report["n_words_k64"] = {name: int(sum(len(v) for v in dicts64[name]["prototypes"].values()))
                             for name in DICT_NAMES}
    report["state"] = "REPLAYED"
    report["note"] = ("Replay uses the frozen checkpoint and the parent dictionary code; "
                      "differences are float64 round-off only.")
    return {"report": report, "dicts64": dicts64, "table": table}


def _identity_checks(dicts16, dicts64, inventory, compressed) -> dict[str, Any]:
    out: dict[str, Any] = {"compressed_classes": [RD.tau_label(t) for t in compressed]}
    nested = {}
    exact = {}
    for name in DICT_NAMES:
        nested_ok = True
        exact_ok = True
        for tau in inventory.class_words:
            p16 = dicts16[name]["prototypes"][tau]
            p64 = dicts64[name]["prototypes"][tau]
            if tau in compressed:
                nested_ok &= (p16 == p64[:16])
            else:
                exact_ok &= (p16 == p64)
        nested[name] = bool(nested_ok)
        exact[name] = bool(exact_ok)
    out["k16_is_prefix_of_k64_for_compressed_classes"] = nested
    out["other_classes_identical_to_parent"] = exact
    out["panels_identical_to_parent"] = all(
        dicts16[name]["panels"][tau] == dicts64[name]["panels"][tau]
        for name in DICT_NAMES for tau in inventory.class_words)
    out["d0_identical_to_parent"] = all(
        dicts16[name]["d0"][tau] == dicts64[name]["d0"][tau]
        for name in DICT_NAMES for tau in inventory.class_words)
    if not all(nested.values()) or not all(exact.values()):
        raise AssertionError("K=16 dictionaries are not the nested parent selections")
    if not out["panels_identical_to_parent"] or not out["d0_identical_to_parent"]:
        raise AssertionError("panel or D0 changed")
    out["ok"] = True
    return out


def _alpha_diagnostics(frozen, inventory, dicts, alpha, support, recipe) -> dict[str, Any]:
    per: dict[str, Any] = {}
    violations: list[float] = []
    steps: list[int] = []
    ratios: dict[str, list[float]] = {}
    for name, spec in dicts.items():
        per[name] = {}
        for tau, protos in spec["prototypes"].items():
            frags = [int(f) for f in inventory.word_fragments if inventory.tau.get(int(f)) == tau]
            coef = np.stack([alpha[name][f] for f in frags]) if frags else np.zeros((0, len(protos)))
            stats = spec["fw_steps"].get(tau, {})
            if coef.size:
                dev = float(max(abs(coef.sum(axis=1) - 1.0).max(), max(0.0, -coef.min())))
            else:
                dev = 0.0
            violations.append(dev)
            steps.append(int(stats.get("steps_used", 0)))
            before = float(stats.get("mean_panel_residual_before", 0.0))
            after = float(stats.get("mean_panel_residual_after", 0.0))
            ratios.setdefault(name, []).append(after / before if before > 0 else 0.0)
            per[name][RD.tau_label(tau)] = {
                "n_candidates": int(len(inventory.class_words[tau])),
                "n_prototypes": int(len(protos)),
                "n_targets": int(len(frags)),
                "fw_steps_used": int(stats.get("steps_used", 0)),
                "canonical_steps": int(stats.get("canonical_steps", recipe["dictionary"]["frank_wolfe_steps"])),
                "mean_panel_residual_before": before,
                "mean_panel_residual_after": after,
                "residual_ratio_after_over_before": after / before if before > 0 else 0.0,
                "max_simplex_violation": dev,
                "mean_sparsity_atoms_per_target": float((coef > 1e-9).sum(axis=1).mean()) if coef.size else 0.0,
            }
    summary = {
        "max_simplex_violation": float(max(violations)) if violations else 0.0,
        "n_classes_at_64_steps": int(sum(1 for s in steps if s >= 64)),
        "n_class_entries": len(steps),
        "residual_ratio_range": {name: [float(min(v)), float(max(v))] for name, v in ratios.items()} if ratios else {},
        "convergence_note": ("64 Frank-Wolfe steps are the frozen budget; a class left at 64 steps is "
                             "reported as limited, not re-run with more iterations."),
        "n_unsupported_fragments": int(sum(1 for f, ok in support["D_RESPONSE"].items() if not ok)),
    }
    return {"summary": summary, "per_class": per}


def _counts_and_coverage(inventory, dicts, frozen, dataset, compressed) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts, view_freq = [], []
    for tau, words in sorted(inventory.class_words.items(), key=lambda kv: RD.tau_label(kv[0])):
        n_cand = len(words)
        spec = dicts["D_RESPONSE"]
        view_counts = {}
        for split_name in SPLITS:
            taus = _view_taus(inventory, frozen.per_split[split_name])
            view_counts[split_name] = int(sum(1 for t in taus if t == tau))
        total_views = sum(view_counts.values())
        counts.append({
            "tau": RD.tau_label(tau),
            "n_candidates": n_cand,
            "n_words": len(spec["prototypes"][tau]),
            "k": int(len(spec["prototypes"][tau])),
            "compressed": bool(tau in compressed),
            "exact_storage": bool(len(spec["prototypes"][tau]) >= n_cand),
            "panel_size": len(spec["panels"][tau]),
            "candidate_share_of_all_words": n_cand / max(1, sum(len(v) for v in inventory.class_words.values())),
            "n_train_views": view_counts["train"],
            "n_valid_views": view_counts["valid"],
            "n_test_views": view_counts["test"],
            "n_views_total": total_views,
            "view_share_of_all_views": total_views / max(1, sum(
                int(np.asarray(frozen.per_split[s]["view_s_frag"]).size) for s in SPLITS)),
        })
        view_freq.append({
            "tau": RD.tau_label(tau),
            "compressed": bool(tau in compressed),
            "n_candidates": n_cand,
            "n_words": len(spec["prototypes"][tau]),
            "train_views": view_counts["train"],
            "valid_views": view_counts["valid"],
            "test_views": view_counts["test"],
            "prototypes_per_candidate": len(spec["prototypes"][tau]) / max(1, n_cand),
        })
    return counts, view_freq


def _response_dimensions(frozen, inventory, compressed) -> dict[str, Any]:
    out: dict[str, Any] = {"classes": {}, "note": (
        "Response-dimension statistics of the frozen reference (D0) and the 16-environment Train panel; "
        "they describe this reference/panel, not the structural complexity of the data.")}
    for tau in compressed:
        cand = [int(f) for f in inventory.class_words[tau]]
        d0 = RD.reference_fragment(inventory, tau)
        envs = RD.environment_panel(inventory, tau, 16)
        R = RD.response_matrix(frozen, cand, d0, envs)
        f = np.asarray([inventory.word_mol_freq.get(int(x), 0) for x in cand], dtype=np.float64)
        m = R.mean(axis=1)
        num = float(np.sum(f * np.sum((R - m[:, None]) ** 2, axis=1)))
        den = float(np.sum(f * np.sum(R ** 2, axis=1))) + 1e-12
        w = np.sqrt(f)[:, None]
        sw_r = np.linalg.svd(w * R, compute_uv=False)
        sw_d = np.linalg.svd(w * (R - m[:, None]), compute_uv=False)
        out["classes"][RD.tau_label(tau)] = {
            "n_candidates": int(len(cand)),
            "n_panel": int(len(envs)),
            "d0_fragment_id": int(d0),
            "d0_train_molecule_frequency": int(inventory.word_mol_freq.get(int(d0), 0)),
            "rho_within_to_total": num / den,
            "m_S_mean_abs": float(np.mean(np.abs(m))),
            "singular_values_weighted_R": [float(x) for x in sw_r],
            "singular_values_weighted_row_demeaned_R": [float(x) for x in sw_d],
            "rank95_weighted_R": _rank95(sw_r),
            "rank95_weighted_row_demeaned_R": _rank95(sw_d),
            "energy_top1_weighted_R": float(sw_r[0] ** 2 / np.sum(sw_r ** 2)) if sw_r.sum() > 0 else 0.0,
        }
    return out


def _rank95(sv: np.ndarray) -> int:
    if sv.size == 0 or np.sum(sv ** 2) <= 0:
        return 0
    cumulative = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return int(np.searchsorted(cumulative, 0.95) + 1)


def _panel_mean_intervention(frozen, inventory, dataset, std, mean) -> dict[str, Any]:
    """Replace per-view ``k(C)`` by the Train panel mean ``k`` of its class."""
    kbar = {}
    missing = []
    for tau in inventory.class_words:
        envs = RD.environment_panel(inventory, tau, 16)
        if not envs:
            missing.append(RD.tau_label(tau))
            continue
        kbar[tau] = frozen.k[np.asarray(envs, dtype=np.int64)].mean(axis=0)
    out: dict[str, Any] = {"classes_with_panel": len(kbar), "classes_without_panel": len(missing),
                           "classes_without_panel_labels": missing,
                           "per_split": {}, "per_split_raw": {}}
    for split_name in SPLITS:
        split = dataset.splits[split_name]
        payload = frozen.per_split[split_name]
        n_mol = len(split.row_id)
        s = np.asarray(payload["view_s_frag"], dtype=np.int64)
        c = np.asarray(payload["view_c_frag"], dtype=np.int64)
        taus = _view_taus(inventory, payload)
        has = np.asarray([bool(t in kbar) for t in taus], dtype=bool) if len(taus) else np.zeros(0, dtype=bool)
        k_eff = frozen.k[c].copy()
        if len(taus) and bool(has.any()):
            k_eff[has] = np.stack([kbar[t] for t in taus if t in kbar])
        scale = frozen.q.shape[1] ** 0.5
        t_new = (frozen.q[s] * k_eff * frozen.w[None, :]).sum(axis=1) / scale
        t_old = (frozen.q[s] * frozen.k[c] * frozen.w[None, :]).sum(axis=1) / scale
        new_std = molecule_mean(payload, frozen.b[c] + frozen.a[s] + t_new, n_mol, frozen)
        old_std = molecule_mean(payload, frozen.b[c] + frozen.a[s] + t_old, n_mol, frozen)
        new_raw = new_std * std + mean
        old_raw = old_std * std + mean
        out["per_split"][split_name] = {
            "n_views": int(s.size),
            "n_views_with_panel": int(has.sum()) if len(has) else 0,
            "n_views_without_panel": int((~has).sum()) if len(has) else 0,
            "rmse_raw_resp": RC.rmse(old_raw, split.label_raw),
            "rmse_raw_panel_mean_k": RC.rmse(new_raw, split.label_raw),
            "delta_rmse_raw": RC.rmse(new_raw, split.label_raw) - RC.rmse(old_raw, split.label_raw),
            "mae_raw_resp": RC.mae(old_raw, split.label_raw),
            "mae_raw_panel_mean_k": RC.mae(new_raw, split.label_raw),
            "prediction_change_rmse_raw": RC.rmse(new_raw, old_raw),
            "max_abs_prediction_change_raw": float(np.max(np.abs(new_raw - old_raw))),
        }
        out["per_split_raw"][split_name] = {"new": new_raw, "old": old_raw,
                                            "row_ids": split.row_id,
                                            "target": split.label_raw}
    out["note"] = ("Frozen-model intervention on the best-Val RESP state. a(S), b(C), q(S), w and every "
                   "fallback path are unchanged; only the environment-dependent factor k(C) inside the "
                   "fusion term is replaced by its class panel mean. This checks whether the trained "
                   "function uses environment variation at all; it is not a generalisation test.")
    return out


def _signal_decision(table, std, ref, alpha_summary=None) -> dict[str, Any]:
    fid = {split: {name: table[(name, split)]["fidelity_rmse_raw"] for name in DICT_NAMES}
           for split in ("valid", "test")}
    better = {split: min(fid[split]["D_LATENT"], fid[split]["D_FREQ"]) for split in ("valid", "test")}
    reduction = {split: 1.0 - fid[split]["D_RESPONSE"] / better[split] for split in ("valid", "test")}
    delta_val = {name: table[(name, "valid")]["delta_rmse_raw"] for name in DICT_NAMES}
    delta_test = {name: table[(name, "test")]["delta_rmse_raw"] for name in DICT_NAMES}
    allowed_increase = float(ref["property_rmse_delta_max_train_std_fraction"]) * std
    passes = {
        "fidelity_20pct_val_and_test": bool(min(reduction.values()) >= float(ref["fidelity_reduction_min"])),
        "property_val_delta_within": bool(delta_val["D_RESPONSE"] <= allowed_increase),
    }
    label = ("RESPONSE_SELECTOR_ADVANTAGE_AT_THIS_RATE" if all(passes.values())
             else "NO_RESPONSE_SELECTOR_ADVANTAGE_AT_THIS_RATE")
    return {
        "fidelity_rmse": fid, "best_baseline_fidelity": better,
        "relative_reduction_vs_better_baseline": reduction,
        "property_delta_val": delta_val, "property_delta_test": delta_test,
        "property_delta_allowed_increase": allowed_increase,
        "property_delta_threshold_negative_side_unused": -allowed_increase,
        "checks": passes, "label": label,
        "caveat_labels": ["PROJECTION_STEP_LIMITED"]
        if alpha_summary and int(alpha_summary.get("n_classes_at_64_steps", 0)) else [],
        "note": ("20% fidelity reduction is this round's investment reference only; a small difference is "
                 "reported as is and never called equivalence."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
