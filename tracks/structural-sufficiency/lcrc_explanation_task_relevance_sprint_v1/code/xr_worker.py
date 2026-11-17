"""One GPU job: Main A (real lipophilicity) or Main B (WC_LOGP).

Every endpoint is processed independently.  Discovery sees only the frozen
molecule graph and one model's gradient field; the paired endpoint, real label,
edit region and oracle contribution vector are attached afterwards.  Records
and the complete candidate log are flushed incrementally.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import xr_common as C
import xr_assets as A
import xr_discover as D
import xr_models as M


def endpoint_key(rec: Mapping[str, Any]) -> str:
    # A molecule can occur in several pairs, so row/split/side is not unique.
    return f"{rec['pair_id']}:{rec['side']}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _grad_stats(grad: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grad, dtype=np.float64).reshape(-1)
    return {
        "n_bonds": int(g.size),
        "sum": float(g.sum()) if g.size else 0.0,
        "sum_abs": float(np.abs(g).sum()) if g.size else 0.0,
        "max_abs": float(np.abs(g).max()) if g.size else 0.0,
        "mean_abs": float(np.abs(g).mean()) if g.size else 0.0,
    }


def _search_summary(res: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "score_name": res["score_name"],
        "topk": res["topk"],
        "n_unique_evaluated": int(res["n_unique_evaluated"]),
        "score_pool_size": int(res["score_pool_size"]),
        "levels": res["levels"],
        "rejected_overlap": len(res["rejected_overlap"]),
    }


def _control_summary(ctrl: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in ctrl.items() if k not in ("controls", "matched", "secondary")}
    key = "controls" if "controls" in ctrl else ("matched" if "matched" in ctrl else "secondary")
    out["n_records"] = len(ctrl.get(key, []))
    out["bonds"] = [rec["bonds"] for rec in ctrl.get(key, [])]
    return out


def _top1(res: Mapping[str, Any]) -> tuple[tuple[int, ...] | None, float | None]:
    if not res["topk"]:
        return None, None
    rec = res["topk"][0]
    return tuple(int(x) for x in rec["bonds"]), float(rec["score"])


def _clean_lcrc_gp(gp: dict[str, Any]) -> dict[str, Any]:
    gp = dict(gp)
    gp.pop("batch", None)
    return gp


def process_endpoint(
    endpoint: Mapping[str, Any],
    *,
    b_base: Any,
    b_wc: Any,
    models: Mapping[str, Any],
    cache: Mapping[str, Any],
    occ: Any,
    mode: str,
    candidate_log: D.CandidateLog,
) -> dict[str, Any]:
    mods = C.source_modules()
    AG = mods["as_gate"]
    row = int(endpoint["cache_row"])
    split = str(endpoint["split"])
    mol = AG.SingleMolecule(cache, row, b_base.spec)
    if mol.n_bonds != int(endpoint["n_bonds"]):
        raise RuntimeError(f"{endpoint_key(endpoint)}: cache bond count {mol.n_bonds} != catalogue {endpoint['n_bonds']}")
    if mol.n_atoms != int(endpoint["n_atoms"]):
        raise RuntimeError(f"{endpoint_key(endpoint)}: cache atom count {mol.n_atoms} != catalogue {endpoint['n_atoms']}")

    b_model = b_wc if mode == "oracle" else b_base
    t0 = time.time()

    # -- LCRC ---------------------------------------------------------------
    gp_l = _clean_lcrc_gp(M.lcrc_gradient_pass(models["LCRC"], cache, occ, row, b_model))
    grad_l = np.asarray(gp_l["grad_transformed"][0], dtype=np.float64)
    contrib_l = np.abs(np.asarray(gp_l["contrib_atom_std"], dtype=np.float64).reshape(-1))
    res_l = D.discover_from_gradient(mol, grad_l, score_name="A_discovery")
    top_l, score_l = _top1(res_l)
    random_l = D.same_size_random_controls(mol, top_l or (), seed=[C.CONTROL_SEED, int(endpoint["row_id"]), 1])
    s2_l = D.s2_matched_random_controls(mol, top_l or (), seed=[C.CONTROL_SEED, int(endpoint["row_id"]), 4])
    candidate_log.write(endpoint_key(endpoint), "LCRC", res_l)
    res_l.pop("log", None)

    # -- GINE ---------------------------------------------------------------
    gine_batch = M.gine_batch_for_row(b_base, split, row, device=models["device"])
    gp_g = M.gine_gradient_pass(models["GINE"], gine_batch, b_model)
    grad_g = np.asarray(gp_g["grad_transformed"][0], dtype=np.float64)
    res_g = D.discover_from_gradient(mol, grad_g, score_name="A_discovery")
    top_g, score_g = _top1(res_g)
    random_g = D.same_size_random_controls(mol, top_g or (), seed=[C.CONTROL_SEED, int(endpoint["row_id"]), 2])
    candidate_log.write(endpoint_key(endpoint), "GINE", res_g)
    res_g.pop("log", None)

    # -- local heuristic on the LCRC atom output ----------------------------
    res_local = D.discover_local_heuristic(mol, contrib_l)
    top_local, score_local = _top1(res_local)
    random_local = D.same_size_random_controls(mol, top_local or (), seed=[C.CONTROL_SEED, int(endpoint["row_id"]), 3])

    # -- external localisation (structure only) ----------------------------
    edit = {
        "edit_atoms": endpoint["edit_atoms"],
        "edit_bonds": endpoint["edit_bonds"],
        "attach_atom": endpoint["attach_atom"],
        "attach_bonds": endpoint["attach_bonds"],
    }
    localisation: dict[str, Any] = {}
    if top_l is not None:
        localisation["LCRC"] = D.localisation_for_candidate(mol, top_l, edit, random_l["controls"])
        localisation["LCRC_s2_matched"] = D.localisation_for_candidate(mol, top_l, edit, s2_l["matched"])
    if top_g is not None:
        localisation["GINE"] = D.localisation_for_candidate(mol, top_g, edit, random_g["controls"])
    if top_local is not None:
        localisation["LOCAL_LCRC"] = D.localisation_for_candidate(mol, top_local, edit, random_local["controls"])

    rec: dict[str, Any] = {
        "endpoint": endpoint_key(endpoint),
        "mode": mode,
        "meta": {
            "pair_index": int(endpoint["pair_index"]),
            "pair_id": str(endpoint["pair_id"]),
            "side": str(endpoint["side"]),
            "row_id": int(endpoint["row_id"]),
            "split": split,
            "cache_row": row,
            "component": int(endpoint["component"]),
            "split_pair": str(endpoint["split_pair"]),
            "train_train": bool(endpoint["train_train"]),
            "in_H": bool(endpoint["in_H"]),
            "evaluable": bool(endpoint["evaluable"]),
            "smiles": str(endpoint["smiles"]),
            "n_atoms": int(mol.n_atoms),
            "n_bonds": int(mol.n_bonds),
        },
        "pred": {
            "LCRC": {"pred_raw": float(gp_l["pred_raw"][0]), "pred_std": float(gp_l["pred_std"][0])},
            "GINE": {"pred_raw": float(gp_g["pred_raw"][0]), "pred_std": float(gp_g["pred_std"][0])},
        },
        "gradient_stats": {"LCRC": _grad_stats(grad_l), "GINE": _grad_stats(grad_g)},
        "search": {
            "LCRC": _search_summary(res_l),
            "GINE": _search_summary(res_g),
            "LOCAL_LCRC": _search_summary(res_local),
        },
        "top1": {
            "LCRC": None if top_l is None else {"bonds": list(top_l), "score": float(score_l), "k": len(top_l)},
            "GINE": None if top_g is None else {"bonds": list(top_g), "score": float(score_g), "k": len(top_g)},
            "LOCAL_LCRC": None if top_local is None else {"bonds": list(top_local), "score": float(score_local), "k": len(top_local)},
        },
        "random": {
            "LCRC": _control_summary(random_l),
            "GINE": _control_summary(random_g),
            "LOCAL_LCRC": _control_summary(random_local),
            "LCRC_s2_matched": _control_summary(s2_l),
        },
        "localisation": localisation,
        "edit": edit,
        "runtime_s": None,
    }

    if mode == "oracle":
        delta = np.asarray(endpoint["oracle_delta"], dtype=np.float64)
        if delta.size != mol.n_atoms:
            raise RuntimeError(f"{endpoint_key(endpoint)}: oracle delta size {delta.size} != atoms {mol.n_atoms}")
        capture: dict[str, Any] = {}
        if top_l is not None:
            capture["LCRC"] = D.oracle_capture_for_candidate(delta, localisation["LCRC"]["top1"]["candidate_atoms"], random_l["controls"])
            capture["LCRC_s2_matched"] = D.oracle_capture_for_candidate(delta, localisation["LCRC_s2_matched"]["top1"]["candidate_atoms"], s2_l["matched"])
        if top_g is not None:
            capture["GINE"] = D.oracle_capture_for_candidate(delta, localisation["GINE"]["top1"]["candidate_atoms"], random_g["controls"])
        if top_local is not None:
            capture["LOCAL_LCRC"] = D.oracle_capture_for_candidate(delta, localisation["LOCAL_LCRC"]["top1"]["candidate_atoms"], random_local["controls"])
        rec["oracle"] = capture
        rec["oracle_denominator"] = float(np.abs(delta).sum())
    rec["runtime_s"] = float(time.time() - t0)
    return _jsonable(rec)


def run(job: str, *, out_dir: Path | str, device: str = "cuda", max_endpoints: int | None = None, start: int = 0) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.install_source_paths()
    b_base = M.bundle(C.DATASET)
    b_wc = None
    if job == "oracle":
        mods = C.source_modules()
        b_wc = mods["gr_data"].oracle_bundle(C.DATASET, "WC_LOGP")
        model_names = {"LCRC": "L_WC_LOGP", "GINE": "T_WC_LOGP"}
        mode = "oracle"
    elif job == "measured":
        model_names = {"LCRC": "L_lipophilicity", "GINE": "G0_lipophilicity"}
        mode = "measured"
    else:
        raise KeyError(job)
    models = {"device": torch.device(device) if device != "cpu" else "cpu"}
    for role, name in model_names.items():
        model, payload = M.load_model(name, b_base, device=models["device"])
        models[role] = model
        print(f"[worker:{job}] loaded {role}={name} epoch={payload.get('epoch')}", flush=True)

    endpoints = C.read_jsonl_gz(C.PAIR_ENDPOINTS)
    endpoints = endpoints[start:] if start else endpoints
    if max_endpoints is not None:
        endpoints = endpoints[: int(max_endpoints)]
    endpoint_path = out_dir / "endpoints.jsonl"
    log_path = out_dir / "candidate_logs.jsonl.gz"
    n_done = 0
    occ_cache: dict[str, Any] = {}
    t_start = time.time()
    with open(endpoint_path, "w", encoding="utf-8") as out_handle, D.CandidateLog(log_path) as candidate_log:
        for i, endpoint in enumerate(endpoints):
            split = str(endpoint["split"])
            if split not in occ_cache:
                mods = C.source_modules()
                AD = mods["as_data"]
                occ_cache[split] = AD.build_split_occ(b_base.caches[split], b_base.spec, b_base.dataset, split)
            cache = b_base.caches[split]
            occ = occ_cache[split]
            try:
                rec = process_endpoint(
                    endpoint,
                    b_base=b_base,
                    b_wc=b_wc,
                    models=models,
                    cache=cache,
                    occ=occ,
                    mode=mode,
                    candidate_log=candidate_log,
                )
                rec["status"] = "OK"
            except Exception as exc:  # noqa: BLE001 - one bad endpoint must not kill the job
                rec = {
                    "endpoint": endpoint_key(endpoint),
                    "mode": mode,
                    "meta": {k: endpoint[k] for k in ("pair_index", "pair_id", "side", "row_id", "split", "cache_row", "component", "split_pair", "train_train", "in_H", "evaluable", "smiles")},
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[worker:{job}] ERROR {rec['endpoint']}: {rec['error']}", flush=True)
            out_handle.write(json.dumps(rec, sort_keys=True) + "\n")
            out_handle.flush()
            n_done += 1
            if i % 20 == 0 or i + 1 == len(endpoints):
                print(f"[worker:{job}] {i+1}/{len(endpoints)} {rec['endpoint']} status={rec.get('status')} t={time.time()-t_start:.1f}s", flush=True)
            if models["device"] != "cpu" and (i + 1) % 50 == 0:
                import torch
                torch.cuda.empty_cache()
                gc.collect()
    import torch
    if models["device"] != "cpu":
        torch.cuda.empty_cache()
    summary = {
        "job": job,
        "mode": mode,
        "n_endpoints": n_done,
        "n_ok": None,
        "out_dir": str(out_dir),
        "endpoints_sha256": C.sha256_file(endpoint_path),
        "candidate_log": candidate_log.close() if False else {"path": str(log_path), "sha256": C.sha256_file(log_path) if log_path.is_file() else None},
        "runtime_s": float(time.time() - t_start),
        "device": str(models["device"]),
        "model_names": model_names,
    }
    # count statuses without loading the whole file twice
    n_ok = n_err = 0
    with open(endpoint_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                if json.loads(line).get("status") == "OK":
                    n_ok += 1
                else:
                    n_err += 1
    summary["n_ok"] = n_ok
    summary["n_error"] = n_err
    C.write_json(out_dir / "worker_summary.json", summary)
    print(f"[worker:{job}] done ok={n_ok} err={n_err} runtime={summary['runtime_s']:.1f}s", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["measured", "oracle"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-endpoints", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    out = run(args.job, out_dir=args.out_dir, device=args.device, max_endpoints=args.max_endpoints, start=args.start)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
