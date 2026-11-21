"""Aggregation, the decision rule, structure statistics and the conditional analyses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import swf_common as C

BRANCH_ARMS = ("FULL", "BAG_CONT", "EARLY_POOL", "FLAT_CONTEXT")


def _run_dirs() -> dict[str, Path]:
    root = C.results_root()
    if not root.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if (path / "run.json").is_file():
            out[path.name] = path
    return out


def load_runs() -> dict[str, dict[str, Any]]:
    return {name: C.read_json(path / "run.json") for name, path in _run_dirs().items()}


def _baselines() -> dict[str, Any]:
    path = C.analysis_dir() / "baselines.json"
    return C.read_json(path) if path.is_file() else {}


def _predictions(run: str, kind: str = "best") -> dict[str, np.ndarray] | None:
    path = C.results_root() / run / f"predictions_{kind}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as handle:
        return {k: handle[k] for k in handle.files}


def _row(model: str, run: Mapping[str, Any] | None, *, kind: str, parent_epoch=None, extra=None) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model, "kind": kind}
    if run is None:
        row.update({"available": False})
        return row
    best = run.get("best") or {}
    final = run.get("final") or {}
    row.update({
        "available": True,
        "run_id": run.get("run_id"),
        "parent_prefix_epoch": parent_epoch if parent_epoch is not None else run.get("parent_prefix_epoch", 0),
        "best_epoch": run.get("best_epoch"),
        "epochs_run": run.get("stopped_at") or run.get("epochs"),
        "train_rmse_best": best.get("train_rmse"),
        "valid_rmse_best": best.get("valid_rmse"),
        "test_rmse_best": best.get("test_rmse"),
        "train_mae_best": best.get("train_mae"),
        "valid_mae_best": best.get("valid_mae"),
        "test_mae_best": best.get("test_mae"),
        "train_rmse_final": final.get("train_rmse"),
        "valid_rmse_final": final.get("valid_rmse"),
        "test_rmse_final": final.get("test_rmse"),
        "wall_seconds": run.get("wall_seconds_total", run.get("wall_seconds")),
        "gpu_hours": run.get("gpu_hours"),
        "parameters": (run.get("parameters") or {}).get("total") if isinstance(run.get("parameters"), dict) else run.get("n_parameters"),
        "device": run.get("device"),
    })
    if extra:
        row.update(extra)
    return row


def main_table() -> list[dict[str, Any]]:
    runs = load_runs()
    bl = _baselines()
    rows: list[dict[str, Any]] = []
    prefix = runs.get("lipo_WORD_PREFIX_s0")
    rows.append(_row("WORD_PREFIX", prefix, kind="neural"))
    rows.append(_row("GINE_RAW_REF", runs.get("lipo_GINE_RAW_REF"), kind="neural_reference"))
    for arm in BRANCH_ARMS:
        run = runs.get(f"lipo_{arm}_s0")
        parent = (prefix or {}).get("best_epoch")
        rows.append(_row(arm, run, kind="neural", parent_epoch=parent))
    for key, name in (("count_ridge", "COUNT_RIDGE"), ("fp_krr", "FP_KRR")):
        payload = bl.get(key)
        if not payload or not payload.get("selected"):
            rows.append({"model": name, "kind": "cpu_baseline", "available": False})
            continue
        sel = payload["selected"]
        rows.append({
            "model": name,
            "kind": "cpu_baseline",
            "available": True,
            "best_alpha": payload.get("selected_alpha"),
            "train_rmse_best": sel["train"]["rmse"],
            "valid_rmse_best": sel["valid"]["rmse"],
            "test_rmse_best": sel["test"]["rmse"],
            "train_mae_best": sel["train"]["mae"],
            "valid_mae_best": sel["valid"]["mae"],
            "test_mae_best": sel["test"]["mae"],
            "parameters": payload.get("n_keys_kept") or payload.get("n_fit"),
        })
    return rows


def _best_valid(rows: Sequence[Mapping[str, Any]], model: str) -> float | None:
    for row in rows:
        if row.get("model") == model and row.get("available") and row.get("valid_rmse_best") is not None:
            return float(row["valid_rmse_best"])
    return None


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cfg = C.load_config()
    th = cfg["thresholds"]
    e_g = _best_valid(rows, "GINE_RAW_REF")
    e_f = _best_valid(rows, "FULL")
    e_b = _best_valid(rows, "BAG_CONT")
    e_c = _best_valid(rows, "COUNT_RIDGE")
    e_p = _best_valid(rows, "FP_KRR")
    e_early = _best_valid(rows, "EARLY_POOL")
    e_flat = _best_valid(rows, "FLAT_CONTEXT")
    simple_candidates = [v for v in (e_c, e_p) if v is not None]
    e_simple = min(simple_candidates) if simple_candidates else None

    payload: dict[str, Any] = {
        "e_gine": e_g, "e_full": e_f, "e_bag": e_b,
        "e_count_ridge": e_c, "e_fp_krr": e_p, "e_simple": e_simple,
        "e_early_pool": e_early, "e_flat_context": e_flat,
        "gap_full_vs_gine": (e_f - e_g) if (e_f is not None and e_g is not None) else None,
        "gain_over_bag": (e_b - e_f) if (e_f is not None and e_b is not None) else None,
        "full_minus_simple": (e_f - e_simple) if (e_f is not None and e_simple is not None) else None,
    }
    if e_g is None or e_c is None:
        payload["status"] = "REFERENCE_OR_IMPLEMENTATION_INCOMPLETE"
        payload["reason"] = "healthy GINE reference and COUNT_RIDGE are both required before any verdict"
        return payload
    if e_f is None or e_b is None:
        payload["status"] = "CORE_BRANCHES_INCOMPLETE"
        payload["reason"] = "FULL and BAG_CONT share the common horizon; both must finish"
        return payload
    gap = payload["gap_full_vs_gine"]
    gain = payload["gain_over_bag"]
    margin = payload["full_minus_simple"]
    if gap > float(th["stop_gap"]) or (margin is not None and margin > float(th["stop_simple_margin"])) or gain < 0.0:
        payload["status"] = "STOP_AT_PILOT"
        if gain < 0.0 and gap <= float(th["stop_gap"]):
            payload["reason"] = "BAG_CONT beats FULL: the real-binding composition has no benefit at this pilot"
        elif gap > float(th["stop_gap"]):
            payload["reason"] = "FULL is more than 0.10 RMSE behind the same-round GINE reference"
        else:
            payload["reason"] = "FULL is worse than the best simple baseline by more than 0.03"
        return payload
    if gap <= float(th["conditional_gap"]) and gain >= float(th["conditional_gain_bag"]) and (
        margin is None or margin <= float(th["conditional_simple_margin"])
    ):
        payload["status"] = "RUN_EARLY_POOL_AND_FLAT_CONTEXT"
        return payload
    if gap <= float(th["flat_gap"]) and gain >= float(th["flat_gain_bag"]) and (
        margin is None or margin <= float(th["flat_simple_margin"])
    ):
        payload["status"] = "RUN_FLAT_CONTEXT_ONLY"
        return payload
    payload["status"] = "END_THIS_ROUND"
    payload["reason"] = "no combination benefit large enough to justify the mechanism controls"
    return payload


# ---------------------------------------------------------------------------
# structure statistics (section 12.3)
# ---------------------------------------------------------------------------


def structure_stats() -> dict[str, Any]:
    import swf_cache as K
    import swf_data as D

    cache = K.Cache(need_sentences=True)
    aut = cache.aut
    n_words = int(cache.word_tmpl.size)
    n_frames = int(cache.frame_win.size)
    word_of_frame = np.repeat(np.arange(n_words, dtype=np.int64), np.diff(cache.word_frame_ptr))
    pair = np.stack([word_of_frame, cache.frame_win.astype(np.int64)], axis=1)
    uniq_pair = np.unique(pair, axis=0)
    distinct_wins = np.bincount(uniq_pair[:, 0], minlength=n_words)
    frames_per_word = np.diff(cache.word_frame_ptr)
    attr_varies_word = distinct_wins > 1

    n_cov = int(cache.cover_sent.size)
    n_rows = int(cache.row_rel_uv.size)
    cu = cache.cover_u.astype(np.int64)
    cv = cache.cover_v.astype(np.int64)
    au = aut[cu]
    av = aut[cv]
    nontrivial_aut = (au > 1) | (av > 1)
    attr_varies_cover = attr_varies_word[cu] | attr_varies_word[cv]

    cover_of_row = np.repeat(np.arange(n_cov, dtype=np.int64), cache.cover_nrows)
    key = cover_of_row * (2 ** 32) + cache.row_rel_uv.astype(np.int64)
    uniq_key = np.unique(key)
    cov_of_uniq = uniq_key // (2 ** 32)
    distinct_rel = np.bincount(cov_of_uniq, minlength=n_cov)
    rel_varies_cover = distinct_rel > 1

    mol_cover_ptr = cache.mol_cover_ptr
    mol_of_cover = np.repeat(
        np.arange(cache.n_mol, dtype=np.int64), np.diff(mol_cover_ptr)
    )

    def mols_with(mask: np.ndarray) -> int:
        return int(np.unique(mol_of_cover[mask]).size)

    cosets = int(n_cov)
    out = {
        "n_molecules": int(cache.n_mol),
        "n_words": n_words,
        "n_frames": n_frames,
        "n_sentences": int(cache.sent_n.size),
        "n_covers": cosets,
        "n_frame_pair_rows": n_rows,
        "nontrivial_root_automorphism": {
            "covers": int(nontrivial_aut.sum()),
            "cover_fraction": float(nontrivial_aut.mean()) if cosets else 0.0,
            "molecules": mols_with(nontrivial_aut),
            "words": int((aut > 1).sum()),
        },
        "attributes_differ_across_frames": {
            "covers": int(attr_varies_cover.sum()),
            "cover_fraction": float(attr_varies_cover.mean()) if cosets else 0.0,
            "molecules": mols_with(attr_varies_cover),
            "words": int(attr_varies_word.sum()),
        },
        "relation_input_differs_across_frames": {
            "covers": int(rel_varies_cover.sum()),
            "cover_fraction": float(rel_varies_cover.mean()) if cosets else 0.0,
            "molecules": mols_with(rel_varies_cover),
            "molecules_among_multi_frame_covers": mols_with(rel_varies_cover & attr_varies_cover),
        },
        "note": (
            "descriptive reach only: these fractions are not an upper bound on any gain, "
            "and a near-zero intervention does not identify the two operators."
        ),
        "templates": [D.TEMPLATES[t].as_dict() for t in range(D.N_TEMPLATES)],
    }
    return out


# ---------------------------------------------------------------------------
# paired bootstrap and the fixed-weight intervention (section 12.1-12.2)
# ---------------------------------------------------------------------------


def _sq_errors(pred: Mapping[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray]:
    return pred[f"{split}_row_id"], (pred[f"{split}_pred"] - pred[f"{split}_label"]) ** 2


def paired_bootstrap(*, n_boot: int = 1000, seed: int = 20261121) -> dict[str, Any]:
    runs = load_runs()
    reference = "lipo_FULL_s0"
    preds = {name: _predictions(name) for name in runs}
    preds = {k: v for k, v in preds.items() if v is not None}
    if reference not in preds:
        return {"available": False, "reason": "FULL predictions missing"}
    out: dict[str, Any] = {"available": True, "reference": reference, "n_boot": n_boot, "seed": seed, "splits": {}}
    rng = np.random.default_rng(seed)
    for split in ("valid", "test"):
        ids_ref, sq_ref = _sq_errors(preds[reference], split)
        n = ids_ref.size
        idx_draws = rng.integers(0, n, size=(n_boot, n))
        base_rmse = np.sqrt(sq_ref.mean())
        for other, payload in preds.items():
            if other == reference:
                continue
            ids_o, sq_o = _sq_errors(payload, split)
            if not np.array_equal(ids_ref, ids_o):
                out["splits"].setdefault(split, {})[other] = {"comparable": False}
                continue
            diffs = np.empty(n_boot)
            for b in range(n_boot):
                idx = idx_draws[b]
                diffs[b] = math.sqrt(sq_o[idx].mean()) - math.sqrt(sq_ref[idx].mean())
            out["splits"].setdefault(split, {})[other] = {
                "comparable": True,
                "rmse_full": float(base_rmse),
                "rmse_other": float(np.sqrt(sq_o.mean())),
                "delta_median": float(np.median(diffs)),
                "delta_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "note": "sample-composition sensitivity only; excludes training-seed and model-selection uncertainty",
            }
    return out


def intervention(*, device: str = "cpu") -> dict[str, Any]:
    """Swap the sentence operator of the trained FULL weights to EARLY_POOL once."""
    import torch

    import swf_cache as K
    import swf_model as M
    import swf_train as T

    cfg = C.load_config()
    run_dir = C.results_root() / "lipo_FULL_s0"
    state = run_dir / "state_best.pt"
    if not state.is_file():
        return {"available": False, "reason": "FULL state_best.pt missing"}
    cache = K.Cache(need_sentences=True)
    model = M.build_model(cfg)
    M.apply_train_support(model, cache.stats)
    M.bind_constants(model, cache.stats)
    model.load_state_dict(torch.load(state, map_location="cpu"))
    model = model.to(device)
    out: dict[str, Any] = {"available": True, "device": device, "state": str(state),
                           "state_sha256": C.sha256_file(state), "splits": {}}
    for split in ("valid", "test"):
        rows = cache.split_rows[split]
        p_full = T.predict(model, cache, rows, "full", device, int(cfg["recipe"]["eval_batch"]))
        p_early = T.predict(model, cache, rows, "early", device, int(cfg["recipe"]["eval_batch"]))
        raw_full = T.predict_raw(p_full, cache.scaler)
        raw_early = T.predict_raw(p_early, cache.scaler)
        truth = cache.labels[rows]
        out["splits"][split] = {
            "rmse_full": C.rmse(raw_full, truth),
            "rmse_early_operator": C.rmse(raw_early, truth),
            "rmse_delta": C.rmse(raw_early, truth) - C.rmse(raw_full, truth),
            "prediction_rms_change": float(np.sqrt(np.mean((raw_full - raw_early) ** 2))),
            "note": "dependence check under fixed weights; retraining EARLY_POOL is what tests benefit",
        }
    return out


# ---------------------------------------------------------------------------
# structural cases (section 12.4)
# ---------------------------------------------------------------------------


def structure_cases(*, device: str = "cpu", n_cases: int = 4) -> dict[str, Any]:
    import torch

    import swf_cache as K
    import swf_data as D
    import swf_model as M
    import swf_train as T

    cfg = C.load_config()
    full_pred = _predictions("lipo_FULL_s0")
    bag_pred = _predictions("lipo_BAG_CONT_s0")
    if full_pred is None or bag_pred is None:
        return {"available": False, "reason": "FULL/BAG_CONT predictions missing"}
    split = "valid"
    ids = full_pred[f"{split}_row_id"]
    sq_full = (full_pred[f"{split}_pred"] - full_pred[f"{split}_label"]) ** 2
    sq_bag = (bag_pred[f"{split}_pred"] - bag_pred[f"{split}_label"]) ** 2
    improvement = sq_bag - sq_full
    order = np.argsort(improvement)
    picks = [int(order[-1]), int(order[0])]
    hashed = sorted(range(ids.size), key=lambda i: (int(ids[i]) * 2654435761) % (2 ** 32))
    picks.extend(hashed[:2])
    picks = picks[:n_cases]

    cache = K.Cache(need_sentences=True)
    raw = D.load_raw_cache(cfg)
    model = M.build_model(cfg)
    M.apply_train_support(model, cache.stats)
    M.bind_constants(model, cache.stats)
    state = C.results_root() / "lipo_FULL_s0" / "state_best.pt"
    model.load_state_dict(torch.load(state, map_location="cpu"))
    model = model.to(device)

    valid_rows = cache.split_rows[split]
    entries = []
    for pos in picks:
        row = int(valid_rows[pos])
        mol = D.raw_molecule(raw, row)
        st = D.enumerate_molecule(mol, D.CodeTables())
        pack = cache.pack([row], sentences=True)
        b = T.to_tensors(pack, device, torch.float32)
        with torch.no_grad():
            model.eval()
            _ = model(b, "full")
        entries.append({
            "row_id": int(ids[pos]),
            "selection": "largest full-vs-bag squared-error improvement" if pos == picks[0]
            else ("largest degradation" if pos == picks[1] else "fixed row_id hash draw"),
            "smiles_atoms": int(mol.n_atoms),
            "sq_error_full": float(sq_full[pos]),
            "sq_error_bag_cont": float(sq_bag[pos]),
            "pred_full": float(full_pred[f"{split}_pred"][pos]),
            "pred_bag_cont": float(bag_pred[f"{split}_pred"][pos]),
            "label": float(full_pred[f"{split}_label"][pos]),
            "n_words": int(st.word_root.size),
            "n_frames": int(st.frame_slots.shape[0]),
            "n_sentences": int(st.sent_root.size),
            "n_covers": int(st.cover_sent.size),
            "n_frame_pair_rows": int(st.n_rows),
            "sample_sentences": [
                {
                    "root": int(st.sent_root[s]),
                    "atom_set": st.sent_atoms[s].tolist(),
                    "size": int(st.sent_n[s]),
                    "n_covers": int(st.sent_ncovers[s]),
                    "covers": [
                        {
                            "U": st.word_atoms[int(st.cover_u[c])].tolist(),
                            "V": st.word_atoms[int(st.cover_v[c])].tolist(),
                            "template_U": int(st.word_tmpl[int(st.cover_u[c])]),
                            "template_V": int(st.word_tmpl[int(st.cover_v[c])]),
                        }
                        for c in range(int(st.cover_sent.size))
                        if int(st.cover_sent[c]) == s
                    ],
                }
                for s in range(min(2, int(st.sent_root.size)))
            ],
            "atom_features": mol.atom_feat.tolist(),
            "bonds": [
                {"u": int(u), "v": int(v), "feat": f.tolist()}
                for u, v, f in zip(mol.bond_u, mol.bond_v, mol.bond_feat)
            ],
        })
    return {"available": True, "split": split, "cases": entries,
            "selection_rule": "1 best improvement, 1 worst degradation, 2 fixed row_id hash draws (no cherry-picking)"}


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def summarize(*, write: bool = True) -> dict[str, Any]:
    rows = main_table()
    status = decide(rows)
    summary: dict[str, Any] = {
        "protocol_id": C.load_config()["protocol_id"],
        "commit": C.git_commit(),
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "main_table": rows,
        "runs": {name: str(path) for name, path in _run_dirs().items()},
        "ledger": _ledger_summary(),
    }
    out: dict[str, Any] = {"status": status, "rows": rows}
    if write:
        C.write_csv(C.analysis_dir() / "main_table.csv", rows)
        C.write_json(C.analysis_dir() / "summary.json", summary)
        stats = None
        try:
            stats = structure_stats()
        except Exception as exc:  # noqa: BLE001
            stats = {"available": False, "error": repr(exc)}
        C.write_json(C.analysis_dir() / "structure_stats.json", stats)
        out["structure_stats"] = {"available": isinstance(stats, dict) and "n_covers" in stats}
        if status.get("status") in ("RUN_EARLY_POOL_AND_FLAT_CONTEXT", "RUN_FLAT_CONTEXT_ONLY") or _has_conditionals():
            conditional: dict[str, Any] = {}
            for name, fn in (("paired_bootstrap", paired_bootstrap),
                             ("intervention", intervention),
                             ("structure_cases", structure_cases)):
                try:
                    conditional[name] = fn()
                except Exception as exc:  # noqa: BLE001
                    conditional[name] = {"available": False, "error": repr(exc)}
            C.write_json(C.analysis_dir() / "conditional.json", conditional)
            out["conditional"] = sorted(conditional)
    return out


def _has_conditionals() -> bool:
    root = C.results_root()
    return any((root / f"lipo_{arm}_s0" / "run.json").is_file() for arm in ("EARLY_POOL", "FLAT_CONTEXT"))


def _ledger_summary() -> dict[str, Any]:
    path = C.analysis_dir() / "resource_ledger.jsonl"
    entries = C.read_jsonl(path)
    by_run: dict[str, float] = {}
    by_arm: dict[str, float] = {}
    for e in entries:
        if e.get("stage") == "train":
            by_run[e.get("run_id", "?")] = by_run.get(e.get("run_id", "?"), 0.0) + float(e["seconds"])
            by_arm[e.get("arm", "?")] = by_arm.get(e.get("arm", "?"), 0.0) + float(e["seconds"])
    return {
        "entries": len(entries),
        "wall_seconds_total": float(sum(float(e["seconds"]) for e in entries)),
        "train_wall_seconds_by_run": by_run,
        "train_wall_seconds_by_arm": by_arm,
        "gpu_hours_total": float(sum(by_run.values()) / 3600.0),
    }
