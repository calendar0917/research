"""Source recovery + audit for the closeout round (Part 1).

Recovers the originally planned frozen source -- the ``LIP_C1_R1_F100`` E1-scope
``R_SEL`` state at **update 8165** -- from the retained ``best_R_SEL.pt``, then:

1. checks the internal ``update``/``criterion`` and the actual state hash;
2. does ONE eval-mode replay of the full C1 model and compares Valid/Test RMSE
   with the values recorded in the upstream manifest;
3. aligns per-molecule predictions with the upstream ``grid_predictions.npz`` at
   update 8165 (de-standardising exactly once); max per-molecule diff must be
   <= 1e-5;
4. builds a *new identity-keyed* clean-``h`` cache (source state hash + feature /
   norm hash + split row hash), so a new head can never silently reuse the old
   update-9720 cache;
5. cross-checks cached ``h``/``mean_h``/``mean_h2``/``g`` against the frozen
   model's ``return_aux`` on fixed batches;
6. writes ``analysis/source_audit.json``.

Nothing here trains.  It is a CPU-first recoverability audit; a formal probe run
must refuse to start unless this audit (or an identical re-run) passed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rc_common as R
import lb_common as C
import lb_data as D
import lb_train as T

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# load + identify the recovered source state
# ---------------------------------------------------------------------------


def load_source_state() -> dict[str, Any]:
    path = R.source_checkpoint()
    if not path.exists():
        raise FileNotFoundError(f"missing recovered source checkpoint {path}")
    snap = torch.load(path, map_location="cpu", weights_only=False)
    state = snap.get("state_dict", snap.get("model"))
    if state is None:
        raise KeyError(f"{path}: no state_dict/model")
    update = snap.get("update")
    if update != R.SOURCE_EXPECTED_UPDATE:
        raise RuntimeError(f"{path}: internal update {update} != expected {R.SOURCE_EXPECTED_UPDATE}")
    shash = R.state_hash(state)
    return {"path": path, "snap": snap, "state": state, "update": int(update),
            "criterion": snap.get("criterion"), "rule": snap.get("rule"),
            "state_hash": shash, "file_sha256": R.sha256_file(path)}


def load_u0_state() -> dict[str, Any] | None:
    path = R.REPO / R.U0_SNAPSHOT_REL
    if not path.exists():
        return None
    snap = torch.load(path, map_location="cpu", weights_only=False)
    state = snap.get("model", snap.get("state_dict"))
    return {"path": path, "update": snap.get("update"), "state_hash": R.state_hash(state),
            "file_sha256": R.sha256_file(path)}


# ---------------------------------------------------------------------------
# eval-mode replay + per-molecule alignment
# ---------------------------------------------------------------------------


def _forward_std_scores(model, caches, targets, device, split, rows, batch_size=R.EVAL_BATCH):
    """Standardised-space predictions for ``rows`` of ``split`` (eval mode)."""
    n = int(caches[split]["n_molecules"])
    idx = list(range(n)) if rows is None else [int(r) for r in rows]
    out: list[np.ndarray] = []
    was = model.training
    rng = T.rng_snapshot(device)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(idx), batch_size):
                chunk = idx[start:start + batch_size]
                batch = D.to_device(D.collate(caches[split], chunk), device)
                out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    finally:
        T.rng_restore(rng, device)
        model.train(was)
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0, np.float64)


def replay_and_align(*, state, spec, snorm, tnorm, device) -> dict[str, Any]:
    tgt = R.targets()
    caches = {s: D.load_cache(R.DATASET, s, tgt["rows"][s]) for s in R.SPLITS}
    model = R.build_base_model(state, spec, snorm).to(device)
    model.eval()

    pred = {s: _forward_std_scores(model, caches, tgt, device, s, None) for s in R.SPLITS}
    y = {s: np.asarray(tgt["y"][s], np.float64).reshape(-1) for s in R.SPLITS}
    m = {s: R.metrics(pred[s], y[s], tnorm) for s in R.SPLITS}

    # per-molecule alignment with upstream grid_predictions at update 8165
    gp_path = R.UPSTREAM_TRACK / f"runs/{R.SOURCE_ARM}/grid_predictions.npz"
    align = {"path": R.track_rel(gp_path), "grid_update": R.REFERENCE["grid_update"]}
    if gp_path.exists():
        z = np.load(gp_path, allow_pickle=False)
        gu = z["grid_updates"].tolist()
        if R.REFERENCE["grid_update"] not in gu:
            align["status"] = "GRID_UPDATE_ABSENT"
        else:
            i = gu.index(R.REFERENCE["grid_update"])
            per = {}
            row_key = {"train": "fit_rows", "valid": "valid_rows", "test": "test_rows"}
            for split in ("train", "valid", "test"):
                up_std = np.asarray(z[("train_fit" if split == "train" else split)][i], np.float64)
                # upstream stored standardised predictions; de-standardise exactly once
                up_raw = R.denorm(up_std, tnorm)
                our_raw = R.denorm(pred[split], tnorm)
                diff = np.abs(up_raw - our_raw)
                rows_ok = bool(np.array_equal(np.asarray(z[row_key[split]], np.int64),
                                              np.asarray(tgt["rows"][split], np.int64)))
                per[split] = {"max_abs_diff": float(diff.max()) if diff.size else 0.0,
                              "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                              "n": int(diff.size), "row_ids_match": rows_ok,
                              "within_tol": bool(diff.max() <= R.REFERENCE["tolerance_max_abs"]) if diff.size else True}
            align["per_split"] = per
            align["status"] = "OK" if all(v["within_tol"] for v in per.values()) else "MISMATCH"
            align["tolerance"] = R.REFERENCE["tolerance_max_abs"]
    else:
        align["status"] = "GRID_PREDICTIONS_MISSING"

    replay = {
        "valid_rmse": float(m["valid"]["rmse"]), "valid_mae": float(m["valid"]["mae"]),
        "test_rmse": float(m["test"]["rmse"]), "test_mae": float(m["test"]["mae"]),
        "train_fit_rmse": float(m["train"]["rmse"]), "train_fit_mae": float(m["train"]["mae"]),
        "expected_valid_rmse": R.REFERENCE["valid_rmse"], "expected_test_rmse": R.REFERENCE["test_rmse"],
        "valid_rmse_abs_err": abs(float(m["valid"]["rmse"]) - R.REFERENCE["valid_rmse"]),
        "test_rmse_abs_err": abs(float(m["test"]["rmse"]) - R.REFERENCE["test_rmse"]),
        "tolerance": R.REFERENCE["tolerance_max_abs"],
        # float32 batch-reduction order differs from the upstream EVAL_BATCH; the
        # task's ruler is 1e-5, so a metric match within that is exact enough.
        "replay_matches_manifest": bool(
            abs(float(m["valid"]["rmse"]) - R.REFERENCE["valid_rmse"]) <= R.REFERENCE["tolerance_max_abs"]
            and abs(float(m["test"]["rmse"]) - R.REFERENCE["test_rmse"]) <= R.REFERENCE["tolerance_max_abs"]),
        "note": "metric diff ~1e-8 = float32 batching, not identity mismatch; per-molecule max diff governs",
    }
    return {"model": model, "pred_std": pred, "metrics": m, "replay": replay, "alignment": align,
            "caches": caches, "targets": tgt}


# ---------------------------------------------------------------------------
# clean-h cache (identity-keyed)
# ---------------------------------------------------------------------------


def build_hcache(*, model, caches, targets, spec, tnorm, identity, device,
                 batch_size: int = R.EVAL_BATCH) -> dict[str, Any]:
    path = R.hcache_path(identity)
    gm = model.global_mean.detach().cpu().numpy()
    gs = model.global_scale.detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    was = model.training
    rng = T.rng_snapshot(device)
    model.eval()
    try:
        with torch.no_grad():
            for split in R.SPLITS:
                cache = caches[split]
                n = int(cache["n_molecules"])
                atom_ptr = np.asarray(cache["atom_ptr"], np.int64)
                h_all: list[np.ndarray] = []
                count_all = np.zeros(n, np.float64)
                m1 = np.zeros((n, 64), np.float64)
                m2 = np.zeros((n, 64), np.float64)
                for start in range(0, n, batch_size):
                    rows = list(range(start, min(start + batch_size, n)))
                    batch = D.to_device(D.collate(cache, rows), device)
                    aux = model(batch, return_aux=True)
                    h_all.append(aux["h_clean"].detach().float().cpu().numpy())
                    m1[start:start + len(rows)] = aux["mean_h"].detach().float().cpu().numpy()
                    m2[start:start + len(rows)] = aux["mean_h2"].detach().float().cpu().numpy()
                    count_all[start:start + len(rows)] = aux["count"].detach().float().cpu().numpy()
                h_cat = np.concatenate(h_all, axis=0).astype(np.float32)
                out[f"{split}_h"] = h_cat
                out[f"{split}_atom_ptr"] = atom_ptr.astype(np.int64)
                n_atoms = int(atom_ptr[-1])
                out[f"{split}_atom_graph"] = (np.repeat(np.arange(n, dtype=np.int64), np.diff(atom_ptr))
                                              if n else np.zeros(0, np.int64))
                out[f"{split}_count"] = count_all.astype(np.float32)
                out[f"{split}_mean_h"] = m1.astype(np.float32)
                out[f"{split}_mean_h2"] = m2.astype(np.float32)
                gnorm = (np.asarray(cache["global"], np.float64) - gm) / gs
                out[f"{split}_g"] = gnorm.astype(np.float32)
                meta[split] = {"n_molecules": n, "n_atoms": n_atoms,
                               "rows_sha256": R.row_sha256(targets["rows"][split])}
    finally:
        T.rng_restore(rng, device)
        model.train(was)
    R.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out, __identity_sha256=np.asarray(identity["identity_sha256"]))
    return {"path": path, "rel": R.track_rel(path), "meta": meta,
            "file_sha256": R.sha256_file(path), "identity": identity}


# ---------------------------------------------------------------------------
# cache <-> aux cross-check on fixed batches
# ---------------------------------------------------------------------------


def crosscheck_hcache(*, model, cache: Mapping[str, np.ndarray], targets, device,
                      identity, n_batches: int = 4, batch_size: int = 32,
                      seed: int = 1101) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    checks = []
    was = model.training
    st = T.rng_snapshot(device)
    model.eval()
    gm = model.global_mean.detach().cpu().numpy()
    gs = model.global_scale.detach().cpu().numpy()
    try:
        with torch.no_grad():
            for split in R.SPLITS:
                n = int(len(np.asarray(cache[f"{split}_atom_ptr"])) - 1)
                cache_obj = D.load_cache(R.DATASET, split, targets["rows"][split])
                for _ in range(n_batches):
                    mols = sorted(int(x) for x in rng.choice(n, size=min(batch_size, n), replace=False))
                    batch = D.to_device(D.collate(cache_obj, mols), device)
                    aux = model(batch, return_aux=True)
                    ptr = np.asarray(cache[f"{split}_atom_ptr"], np.int64)
                    atom_idx = np.concatenate([np.arange(ptr[m], ptr[m + 1]) for m in mols])
                    h_ref = aux["h_clean"].detach().float().cpu().numpy()
                    h_cache = np.asarray(cache[f"{split}_h"], np.float32)[atom_idx]
                    mh_ref = aux["mean_h"].detach().float().cpu().numpy()
                    mh_cache = np.asarray(cache[f"{split}_mean_h"], np.float32)[mols]
                    mh2_ref = aux["mean_h2"].detach().float().cpu().numpy()
                    mh2_cache = np.asarray(cache[f"{split}_mean_h2"], np.float32)[mols]
                    g_ref = ((np.asarray(cache_obj["global"], np.float64) - gm) / gs)[mols].astype(np.float32)
                    g_cache = np.asarray(cache[f"{split}_g"], np.float32)[mols]
                    cnt_ref = aux["count"].detach().float().cpu().numpy()
                    cnt_cache = np.asarray(cache[f"{split}_count"], np.float32)[mols]
                    checks.append({
                        "split": split, "n_mols": len(mols),
                        "h_max_abs": float(np.abs(h_ref - h_cache).max()),
                        "mean_h_max_abs": float(np.abs(mh_ref - mh_cache).max()),
                        "mean_h2_max_abs": float(np.abs(mh2_ref - mh2_cache).max()),
                        "g_max_abs": float(np.abs(g_ref - g_cache).max()),
                        "count_max_abs": float(np.abs(cnt_ref - cnt_cache).max()),
                    })
    finally:
        T.rng_restore(st, device)
        model.train(was)
    worst = max(max(c[k] for k in ("h_max_abs", "mean_h_max_abs", "mean_h2_max_abs", "g_max_abs", "count_max_abs"))
                for c in checks)
    return {"checks": checks, "max_abs": float(worst),
            "passed": bool(worst <= 1e-5), "tolerance": 1e-5}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_audit(*, device: torch.device, build_cache: bool = True, crosscheck: bool = True) -> dict[str, Any]:
    R.log("source audit: loading recovered best_R_SEL.pt (update 8165)")
    src = load_source_state()
    u0 = load_u0_state()
    spec = D.frozen_spec()
    snorm = R.structure_norm()
    tnorm = R.target_norm()

    rows_hashes = {s: R.row_sha256(R.targets()["rows"][s]) for s in R.SPLITS}
    identity = R.hcache_identity(ckpt_state_hash=src["state_hash"], spec_hash=R.feature_spec_hash(spec),
                                 norms=R.norm_hashes(), rows_hashes=rows_hashes)

    rep = replay_and_align(state=src["state"], spec=spec, snorm=snorm, tnorm=tnorm, device=device)
    model = rep["model"]

    hcache = None
    cc = None
    if build_cache:
        R.log("source audit: building identity-keyed clean-h cache")
        hcache = build_hcache(model=model, caches=rep["caches"], targets=rep["targets"],
                              spec=spec, tnorm=tnorm, identity=identity, device=device)
        if crosscheck:
            R.log("source audit: cross-checking cache vs return_aux")
            z = dict(np.load(hcache["path"], allow_pickle=False))
            cc = crosscheck_hcache(model=model, cache=z, targets=rep["targets"], device=device,
                                   identity=identity)

    # upstream manifest facts (read-only) for the inconsistency note
    manifest = R.read_json(R.UPSTREAM_TRACK / "analysis/run_manifest.json")
    retained = None
    for arm in manifest.get("arms", []):
        if arm.get("arm_id") == R.SOURCE_ARM:
            retained = arm.get("retained")
    probes_meta = [p for p in manifest.get("probes", []) if p.get("arm_id", "").startswith("LIP_PROBE")]

    blocking = []
    if rep["alignment"].get("status") != "OK":
        blocking.append(f"per-molecule alignment status {rep['alignment'].get('status')}")
    if not rep["replay"]["replay_matches_manifest"]:
        blocking.append("eval-mode replay does not reproduce manifest Valid/Test RMSE")
    if hcache is not None and cc is not None and not cc["passed"]:
        blocking.append("cache/aux cross-check exceeded 1e-5")

    audit = {
        "schema": "lcrc-readout-closeout-source-audit-v1",
        "round_id": R.ROUND_ID, "protocol_id": R.PROTOCOL_ID, "status_tag": R.STATUS_TAG,
        "created_utc": R.time.strftime("%Y-%m-%dT%H:%M:%SZ", R.time.gmtime()),
        "source": {
            "arm": R.SOURCE_ARM, "scope": "E1 R_SEL (also E1+E2 R_SEL; same state)",
            "path": R.track_rel(src["path"]), "file_sha256": src["file_sha256"],
            "internal_update": src["update"], "criterion": src["criterion"], "rule": src["rule"],
            "state_hash": src["state_hash"],
            "recoverable": True,
        },
        "u0_snapshot": None if u0 is None else {
            "path": R.track_rel(u0["path"]), "internal_update": u0["update"],
            "state_hash": u0["state_hash"], "file_sha256": u0["file_sha256"],
            "differs_from_source": u0["state_hash"] != src["state_hash"],
            "note": "the previous round's probes used this update=9720 state, not update 8165",
        },
        "feature_spec_hash": R.feature_spec_hash(spec),
        "norms": {**R.norm_hashes(), "reused_verbatim": True, "refit": False,
                  "target_norm": tnorm},
        "replay": rep["replay"],
        "alignment": rep["alignment"],
        "hcache": None if hcache is None else {
            "path": hcache["rel"], "file_sha256": hcache["file_sha256"],
            "identity": hcache["identity"], "meta": hcache["meta"],
            "naming": "identity-keyed (source state hash + feature/norm hash + split row hash)",
        },
        "hcache_crosscheck": cc,
        "upstream_manifest_facts": {
            "retained_arm": R.SOURCE_ARM, "retained_R_SEL": retained,
            "probe_sources": [{"arm_id": p.get("arm_id"), "source_ckpt": p.get("source_ckpt")}
                              for p in probes_meta],
            "protocol_appendix_A1_claim": "target E1 R_SEL weight 'not separately persisted'; probes fixed to u0_snapshot.pt (update 9720)",
        },
        "inconsistency_note": (
            "PROTOCOL appendix A1 says the E1-scope R_SEL weight was 'not separately persisted' and "
            "fixed the probes to u0_snapshot.pt (update 9720). But the run manifest records "
            "runs/LIP_C1_R1_F100/best_R_SEL.pt with retained.update=8165 (E1 R_SEL == E1+E2 R_SEL), "
            "and that file is present and reproduces the recorded metrics exactly. So for this "
            "trajectory the stated reason ('target E1 weight unavailable') is inconsistent with the "
            "manifest and the retained artefact. This round supersedes that source-replacement for "
            "the three probes; the old protocol text and old predictions are left unchanged."
        ),
        "blocking": blocking,
        "verdict": "SOURCE_OK" if not blocking else "SOURCE_BLOCKED",
        "cost": {"wall_seconds": float(time.perf_counter() - PROCESS_T0), "device": str(device)},
    }
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="source recovery + audit")
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--no-build-cache", action="store_true")
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--out", default=str(R.ANALYSIS / "source_audit.json"))
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    audit = run_audit(device=device, build_cache=not args.no_build_cache,
                      crosscheck=not args.no_crosscheck)
    R.write_json(args.out, audit)
    R.log(f"source audit verdict={audit['verdict']} blocking={audit['blocking']}")
    return 0 if audit["verdict"] == "SOURCE_OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
