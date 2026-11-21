"""Round bookkeeping: structure freeze record, run manifest, frozen interventions, cases."""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import wp_common as C
import wp_data as D

PLACEHOLDER = "unavailable"


def _config_sha() -> str:
    return C.sha256_json(C.load_config())


def code_hashes() -> dict[str, str]:
    out = {}
    for path in sorted((C.ROUND_DIR / "code").glob("*.py")):
        out[path.name] = C.sha256_file(path)
    for path in sorted((C.ROUND_DIR / "code").glob("*.sh")):
        out[path.name] = C.sha256_file(path)
    for path in sorted((C.ROUND_DIR / "configs").glob("*.json")):
        out[path.name] = C.sha256_file(path)
    return out


def input_hashes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for alias in ("lipo", "esol"):
        name = {"lipo": "ogbg-mollipo", "esol": "ogbg-molesol"}[alias]
        record = C.read_json(C.derived_record_path(name))
        derived = C.ROUND_DIR / "data" / f"{name}_portbind_portbind-v2.npz"
        old = C.old_word_path(name)
        out[name] = {
            "inherited_word_cache": str(old),
            "inherited_word_cache_sha256": record.get("inherited_word_cache_sha256"),
            "inherited_word_record": str(C.old_record_path(name)),
            "derived_file": str(derived),
            "derived_file_sha256": C.sha256_file(derived) if derived.is_file() else None,
            "derived_content_sha256": record.get("derived_sha256"),
            "split_sizes": record.get("split_sizes"),
            "labels_used": record.get("labels_used"),
        }
    cache = Path(C.__dict__.get("RAW_CACHE", "")) if False else None
    del cache
    return out


# ---------------------------------------------------------------------------
# structure freeze
# ---------------------------------------------------------------------------


def write_freeze(args) -> None:
    """analysis/structure_freeze.json: inherited identity, new hashes, support."""
    dataset = {"lipo": "ogbg-mollipo", "esol": "ogbg-molesol"}.get(args.dataset, args.dataset)
    record = C.read_json(C.derived_record_path(dataset))
    payload = {
        "dataset": dataset,
        "commit": C.git_commit(),
        "config_sha256": _config_sha(),
        "inherited": {
            "reference_commit": C.REFERENCE_COMMIT,
            "word_cache_sha256": record.get("inherited_word_cache_sha256"),
            "word_record": record.get("inherited_word_record"),
            "template_table": record.get("inherited_template_table"),
            "read_only": True,
        },
        "derived": {
            "path": record.get("derived_path"),
            "file_sha256": record.get("derived_file_sha256"),
            "content_sha256": record.get("derived_sha256"),
            "cache_version": record.get("cache_version"),
            "definitions": record.get("definitions"),
        },
        "support": record.get("support"),
        "split_sizes": record.get("split_sizes"),
        "labels_used": record.get("labels_used"),
        "ports": _port_definition(),
    }
    support_path = C.analysis_dir() / "support.json"
    if support_path.is_file():
        payload["round_support"] = C.read_json(support_path)
    C.write_json(C.analysis_dir() / "structure_freeze.json", C.jsonable(payload))
    print(json.dumps({"written": str(C.analysis_dir() / "structure_freeze.json"),
                      "derived_content_sha256": payload["derived"]["content_sha256"]}, indent=2))


def _port_definition() -> dict[str, Any]:
    return {
        "site": "every atom of every word instance",
        "compatible_group": "sites of the same word instance with identical original nine-column "
                            "atom rows (element, degree, charge, radical, hybridisation, aromatic, "
                            "hydrogens, chirality flags)",
        "port_state": "the word-side state p_a of that site, built from the internal word graph "
                      "and the double-port relation R of the same word",
        "binding": "the cross-word message uses the state of the site it is actually attached to",
        "mean_control": "the cross-word message uses the mean state over the compatible group of "
                        "that site (identical local content, identical adjacency, identical R)",
        "coverage": "groups of size one make the intervention a no-op, so effective coverage is "
                    "reported as the fraction of incidences whose own group has size >= 2",
    }


# ---------------------------------------------------------------------------
# run manifest
# ---------------------------------------------------------------------------


def _memory_mb(pid: int | None = None) -> float | None:
    try:
        if pid is None:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def write_manifest(args) -> None:
    import wp_analyze as A

    cfg = C.load_config()
    dataset = "ogbg-mollipo"
    runs = A.collect_runs(dataset)
    provenance = C.read_json(C.ROUND_DIR / "analysis" / "execution_regime.json")
    entries = []
    total_seconds = 0.0
    for run in sorted(runs, key=lambda r: (r["arm"], r["seed"])):
        run_dir = Path(run["_path"]).parent
        artifacts = {}
        for name in ("state_best.pt", "state_final.pt", "preds_best.npz", "preds_final.npz",
                     "preds_test.npz"):
            path = run_dir / name
            if path.is_file():
                artifacts[name] = {"path": str(path.relative_to(C.ROUND_DIR)),
                                   "sha256": C.sha256_file(path),
                                   "bytes": path.stat().st_size}
        curves = run.get("curves") or []
        seconds = float(run.get("wall_seconds") or 0.0)
        total_seconds += seconds
        entries.append({
            "run_id": run["run_id"], "arm": run["arm"], "seed": run["seed"],
            "status": run["status"], "spec": run.get("spec"),
            "command": f"bash code/wp_launch.sh {run['arm']} {run['seed']}",
            "launch_commit": run.get("commit"), "config_sha256": run.get("config_sha256"),
            "dataset": run.get("dataset"), "device": run.get("device_name"),
            "torch": run.get("torch"), "epochs": len(curves),
            "wall_seconds": seconds, "seconds_per_epoch": run.get("seconds_per_epoch"),
            "n_parameters": run.get("n_parameters"),
            "resumed_from_epoch": run.get("resumed_from_epoch"),
            "max_rss_mb": run.get("max_rss_mb"), "cpu_seconds": run.get("cpu_seconds"),
            "gpu_model": run.get("gpu_model"), "gpu_hours": run.get("gpu_hours"),
            "final_state_written": run.get("final_state_written"),
            "best": run.get("best"), "final": run.get("final"), "last20": run.get("last20"),
            "artifacts": artifacts,
        })
    manifest = {
        "round": "word_sentence_port_binding_context_v2",
        "t0_utc": provenance.get("t0_utc"), "t0_monotonic": provenance.get("t0_monotonic"),
        "hard_stop_utc": provenance.get("budget", {}).get("hard_stop_utc"),
        "stop_launching_after_utc": provenance.get("budget", {}).get("stop_launching_after_utc"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_regime": provenance.get("regime"),
        "regime_deviation": provenance.get("deviation"),
        "host": {"platform": platform.platform(), "processor": platform.processor(),
                 "cpu_count": None, "memory_mb": _memory_mb()},
        "config_sha256": _config_sha(),
        "commit": C.git_commit(),
        "code_hashes": code_hashes(),
        "inputs": input_hashes(),
        "budget": {
            "gpu_hours_used": 0.0,
            "cpu_worker_hours": round(total_seconds / 3600.0, 3),
            "units_launched": len(entries),
            "units_complete": sum(1 for e in entries if e["status"] == "COMPLETE"),
            "units_partial": sum(1 for e in entries if e["status"] == "PARTIAL"),
        },
        "remote_jobs": [],
        "failures": _failures(),
        "runs": entries,
    }
    try:
        import os
        manifest["host"]["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    C.write_json(C.ROUND_DIR / "run_manifest.json", C.jsonable(manifest))
    print(json.dumps({"runs": len(entries), "complete": manifest["budget"]["units_complete"],
                      "cpu_worker_hours": manifest["budget"]["cpu_worker_hours"]}, indent=2))


def _failures() -> list[dict[str, Any]]:
    path = C.ROUND_DIR / "analysis" / "failures.json"
    return C.read_json(path) if path.is_file() else []


# ---------------------------------------------------------------------------
# frozen interventions on the selected word checkpoint
# ---------------------------------------------------------------------------


def _load_word_checkpoint(arm: str, seed: int, data, hidden: int):
    import torch

    import wp_model as M
    import wp_train as T

    short = "mollipo" if data["record"].get("dataset", "").endswith("mollipo") else "molesol"
    run_dir = C.results_root() / f"{short}_{arm}_s{seed}"
    payload = torch.load(run_dir / "state_best.pt", map_location="cpu", weights_only=False)
    tmpl = {k[len("tmpl/"):]: v for k, v in data["words"].items() if k.startswith("tmpl/")}
    tmpl["dist"] = data["derived"]["tmpl/dist"]
    spec = M.ARMS[arm]
    model = T.build_word_model(spec, tmpl, seed, "cpu", hidden=hidden)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, spec, run_dir


def write_interventions(args) -> None:
    """Swap the port/context operator on frozen weights and measure the response."""
    import torch

    import wp_analyze as A

    cfg = C.load_config()
    dataset = args.dataset if args.dataset.startswith("ogbg-") else A.dataset_name(args.dataset)
    data = D.load_dataset(dataset)
    scaler = C.target_scaler(data["raw"])
    valid = D.make_split(data, "valid")
    summary_path = C.analysis_dir() / "summary.json"
    summary = C.read_json(summary_path) if summary_path.is_file() else {}
    w = (summary.get("candidates") or {}).get("W") or args.arm
    seed = int(args.seed)
    model, spec, run_dir = _load_word_checkpoint(w, seed, data, cfg["model"]["hidden"])
    y = valid.labels
    out: dict[str, Any] = {"arm": w, "seed": seed, "checkpoint": str(run_dir / "state_best.pt"),
                           "frozen_weights": True, "conditions": {}}

    def predict(port: str, context: str) -> dict[str, np.ndarray]:
        model.spec = type(spec)(**{**spec.__dict__, "port": port, "context": context})
        return A_word_predict(model, valid, scaler)

    base = predict(spec.port, spec.context)
    for port in ("bind", "mean"):
        for context in ("rel", "self"):
            pred = predict(port, context)
            rmse = float(np.sqrt(np.mean((pred["pred"] - y) ** 2)))
            out["conditions"][f"{port}_{context}"] = {
                "valid_rmse_raw": rmse,
                "mean_abs_delta_vs_base": float(np.mean(np.abs(pred["pred"] - base["pred"]))),
                "max_abs_delta_vs_base": float(np.max(np.abs(pred["pred"] - base["pred"]))),
                "pred_std": float(np.std(pred["pred"])),
            }
            pred["_pred"] = pred["pred"]
    # per-molecule response to the port swap, against its non-equivalent incidence share
    bind = predict("bind", spec.context)
    mean = predict("mean", spec.context)
    delta = np.abs(bind["pred"] - mean["pred"])
    non_equiv = A_split_incidence_share(data, "valid")
    order = {int(row): i for i, row in enumerate(valid.row_ids)}
    share = np.asarray([non_equiv.get(int(row), 0.0) for row in valid.row_ids])
    if share.std() > 0:
        corr = float(np.corrcoef(delta, share)[0, 1])
    else:
        corr = None
    out["port_swap"] = {
        "mean_abs_delta": float(np.mean(delta)), "max_abs_delta": float(np.max(delta)),
        "spearman_vs_non_equivalent_share": _spearman(delta, share),
        "pearson_vs_non_equivalent_share": corr,
        "n_molecules_with_delta_gt_1e-6": int(np.sum(delta > 1e-6)),
        "n_molecules": int(len(delta)),
    }
    out["port_state_geometry"] = _port_state_geometry(model, valid, data)
    del order
    tag = getattr(args, "tag", "") or f"{w.lower()}_s{seed}"
    out["tag"] = tag
    C.write_json(C.analysis_dir() / f"interventions_{tag}.json", C.jsonable(out))
    print(json.dumps(C.jsonable(out), indent=2))


def A_word_predict(model, split, scaler) -> dict[str, np.ndarray]:
    """Raw-scale predictions for one split."""
    import wp_train as T

    res = T.evaluate_word(model, split, "cpu", scaler)
    pred = res["pred_std"] * float(scaler["std"]) + float(scaler["mean"])
    return {"pred": pred, "row": res["row_ids"]}


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def A_split_incidence_share(data, split: str) -> dict[int, float]:
    """Share of non-equivalent incidences per molecule (cached from support features)."""
    import wp_analyze as A

    cache_path = C.analysis_dir() / f"incidence_share_{split}.json"
    if cache_path.is_file():
        payload = C.read_json(cache_path)
        return {int(k): float(v) for k, v in payload.items()}
    features = A._split_port_features(data, split)
    arr = features["arr"]
    site_group, group_count = arr["site_group"], arr["group_count"]
    site_ptr, inc_ptr = features["site_ptr"], features["inc_ptr"]
    word_of_inc = np.repeat(np.arange(features["n_words"]), np.diff(inc_ptr))
    own_site = site_ptr[word_of_inc] + features["inc_own"]
    groups = site_group[own_site]
    sizes = group_count[groups]
    distinct = np.zeros(len(sizes), dtype=bool)
    for idx in np.nonzero(sizes >= 2)[0]:
        members = np.nonzero(site_group == groups[idx])[0]
        distinct[idx] = len({features["site_key"][m] for m in members}) > 1
    mol = features["word_of_graph"][word_of_inc]
    out: dict[int, float] = {}
    counts = np.bincount(mol, minlength=len(features["row_ids"]))
    hits = np.bincount(mol[distinct], minlength=len(features["row_ids"]))
    for position, row in enumerate(features["row_ids"]):
        out[int(row)] = float(hits[position] / counts[position]) if counts[position] else 0.0
    C.write_json(cache_path, C.jsonable({str(k): v for k, v in out.items()}))
    return out


def _port_state_geometry(model, split, data) -> dict[str, Any]:
    """Distance between the site state and its compatible-group mean (PROTOCOL 12.3)."""
    import torch

    arr = split.arrays
    idx = np.arange(len(split))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(idx), 512):
            batch = D.to_device(D.pack_batch(split, idx[start:start + 512]), "cpu")
            out = model(batch)
            chosen = out["chosen"].numpy()
            raw = out["c"].numpy()
            groups = batch["site_group"].numpy()
            sizes = batch["group_count"].numpy()[groups]
            chunks.append((raw - chosen, sizes))
    diff = np.concatenate([c[0] for c in chunks])
    sizes = np.concatenate([c[1] for c in chunks])
    norms = np.linalg.norm(diff, axis=1)
    affected = sizes >= 2
    return {
        "n_sites": int(len(norms)),
        "n_sites_group_ge2": int(np.sum(affected)),
        "mean_norm_group_ge2": float(norms[affected].mean()) if affected.any() else None,
        "p90_norm_group_ge2": float(np.percentile(norms[affected], 90)) if affected.any() else None,
        "mean_norm_group_eq1": float(norms[~affected].mean()) if (~affected).any() else None,
        "max_norm": float(norms.max()) if len(norms) else None,
        "state_dim": int(diff.shape[1]),
        "note": "chosen - c is the mean-pooling displacement the MEAN control removes; it is "
                "zero by construction for single-site groups",
    }


# ---------------------------------------------------------------------------
# cases (PROTOCOL 13)
# ---------------------------------------------------------------------------


def write_cases(args) -> None:
    import torch

    import wp_analyze as A

    cfg = C.load_config()
    data = D.load_dataset("ogbg-mollipo")
    valid = D.make_split(data, "valid")
    summary = C.read_json(C.analysis_dir() / "summary.json")
    w = getattr(args, "arm_a", "") or summary["candidates"]["W"]
    core = getattr(args, "arm_b", "") or (summary.get("core_pair") or {}).get("C")
    seed = int(getattr(args, "seed", 0) or 0)
    model_w, spec_w, _ = _load_word_checkpoint(w, seed, data, cfg["model"]["hidden"])
    scaler = C.target_scaler(data["raw"])
    pred_w = A_word_predict(model_w, valid, scaler)["pred"]
    cases: dict[str, Any] = {"pair": [w, core], "seed": seed, "cases": [],
                             "pair_source": "explicit" if getattr(args, "arm_a", "") else "summary",
                             "arms": {"a": w, "b": core},
                             "label_rule": "improved = arm_b closer to the label than arm_a"}
    if core:
        model_c, _, _ = _load_word_checkpoint(core, seed, data, cfg["model"]["hidden"])
        pred_c = A_word_predict(model_c, valid, scaler)["pred"]
    else:
        pred_c = pred_w
    y = valid.labels
    err_a = np.abs(pred_w - y)
    err_b = np.abs(pred_c - y)
    improvement = err_a - err_b        # positive when the second arm is closer
    share = A_split_incidence_share(data, "valid")
    rows = np.asarray(valid.row_ids)
    shares = np.asarray([share.get(int(r), 0.0) for r in rows])
    # two structural cases fixed before reading Valid labels: first row of each split
    # is not allowed, so use the highest non-equivalent share with a group >= 3 word,
    # which is a structure-only criterion.
    order_structural = np.lexsort((rows, -shares))
    structural = [int(rows[i]) for i in order_structural[:2]]
    improved = [int(rows[i]) for i in np.lexsort((rows, -improvement))
                if int(rows[i]) not in structural][:3]
    worsened = [int(rows[i]) for i in np.lexsort((rows, improvement))
                if int(rows[i]) not in structural and int(rows[i]) not in improved][:3]
    for row in structural + improved + worsened:
        pos = int(np.nonzero(rows == row)[0][0])
        kind = ("structural_support" if row in structural
                else "improved" if row in improved else "worsened")
        cases["cases"].append({
            "row_id": row, "kind": kind,
            "smiles": _smiles(data, row),
            "n_atoms": int(valid.graph_n_atoms[pos]),
            "n_words": int(np.diff(valid.arrays["graph_word_ptr"])[pos]),
            "non_equivalent_incidence_share": float(shares[pos]),
            "label": float(y[pos]),
            "pred_a": float(pred_w[pos]), "pred_b": float(pred_c[pos]),
            "abs_err_a": float(err_a[pos]), "abs_err_b": float(err_b[pos]),
            "improvement_b_over_a": float(improvement[pos]),
            "words": _word_list(valid, pos),
        })
    cases["note"] = ("the first two cases are chosen from structure only (highest share of "
                     "non-equivalent incidences, ties by row id); the remaining six are post hoc "
                     "explanations of the largest paired error changes")
    C.write_json(C.analysis_dir() / "cases.json", C.jsonable(cases))
    print(json.dumps(C.jsonable({"pair": cases["pair"], "rows": [c["row_id"] for c in cases["cases"]],
                                "kinds": [c["kind"] for c in cases["cases"]]}), indent=2))


def _smiles(data, row: int) -> str:
    text = data["raw"]["smiles"][int(row)]
    return text.decode() if isinstance(text, bytes) else str(text)


def _word_list(split, position: int) -> list[dict[str, Any]]:
    arr = split.arrays
    gw = arr["graph_word_ptr"]
    words = np.arange(int(gw[position]), int(gw[position + 1]))
    out = []
    for word in words:
        sp = arr["site_ptr"]
        ip = arr["inc_ptr"]
        atoms = np.arange(int(sp[word]), int(sp[word + 1]))
        out.append({
            "template": int(arr["word_tmpl"][word]),
            "n_atoms": int(len(atoms)),
            "n_ports": int(ip[word + 1] - ip[word]),
            "n_ibonds": int(arr["word_n_ibonds"][word]),
            "site_groups": [int(arr["site_group"][a]) for a in atoms],
            "group_sizes": [int(arr["group_count"][arr["site_group"][a]]) for a in atoms],
        })
    return out
