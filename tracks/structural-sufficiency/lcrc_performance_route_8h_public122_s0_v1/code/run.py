"""Entry point: prepare / smoke / train / dmnn-parity / baseline / analyze / report.

Every stage is re-runnable and idempotent; a finished run directory is never
silently overwritten unless ``--force`` is given.  The training host consumes
only prepared artefacts, so ``prepare`` (RDKit / CPU) and ``train`` (GPU) are
separate stages.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import lcpr_common as C
import lcpr_data as D


def _prepare(datasets: Sequence[str], *, with_dmpnn: bool, force: bool = False) -> dict[str, Any]:
    C.CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"schema": "lcrc-performance-route-prepare-v1", "created_utc": C.utcnow(),
                           "datasets": {}, "feature_spec_hash": C.FEATURE_SPEC_HASH}
    for ds in datasets:
        entry: dict[str, Any] = {}
        C.log(f"[prepare] {ds}: verifying published split / cache")
        entry["verify"] = D.verify_dataset(ds)
        C.log(f"[prepare] {ds}: support table (Train structures only)")
        entry["support"] = D.build_support(ds) if (force or not D.support_path(ds).exists()) else {
            "reused": True, "path": C.track_rel(D.support_path(ds))}
        C.log(f"[prepare] {ds}: permutation table (seed {C.SHUFFLE_SEED})")
        perms = D.load_perms(ds, C.LONG_EPOCHS)
        entry["perms"] = {"shape": list(perms.shape), "seed": int(C.SHUFFLE_SEED),
                          "sha256": C.sha256_bytes(perms.tobytes())}
        entry["targets"] = {k: v for k, v in D.targets(ds, include_test=False).items()
                            if k in ("task", "primary", "units", "split_method", "target_norm")}
        C.log(f"[prepare] {ds}: Train-only structure normalisation")
        import lcpr_train as T

        norm = T._train_structure_norm(D.load_cache(ds, "train"), D.frozen_spec())
        entry["train_norm"] = {"keys": sorted(norm), "fingerprint": C.DD.PREP_DATA.norm_fingerprint(
            {**norm, "valid_or_test_used": False, "source": f"{ds}_train_structures_only"})}
        out["datasets"][ds] = entry
    if with_dmpnn:
        import lcpr_dmpnn as DMN

        out["dmpnn_featurizer"] = DMN.featurizer_shapes()
        out["dmpnn_parity"] = DMN.parity_check()
        out["dmpnn_features"] = {}
        for ds in datasets:
            C.log(f"[prepare] {ds}: chemprop v2.2.1 native graph features")
            out["dmpnn_features"][ds] = DMN.build_features(ds, force=force)
    C.write_json(C.ANALYSIS / "prepare.json", out)
    C.log("[prepare] done")
    return out


def _prepare_aux(datasets: Sequence[str], *, force: bool = False) -> dict[str, Any]:
    out = {}
    for ds in datasets:
        if D.aux_target_path(ds).exists() and not force:
            out[ds] = {"reused": True, "path": C.track_rel(D.aux_target_path(ds))}
            continue
        C.log(f"[prepare] {ds}: WC_LOGP / MolWt auxiliary targets (Train-only shuffle)")
        out[ds] = D.build_aux_targets(ds)
    C.write_json(C.ANALYSIS / "prepare_aux.json", out)
    return out


def _cache_audit(datasets: Sequence[str]) -> dict[str, Any]:
    """chem8 alignment + support/aux summary across datasets."""
    out: dict[str, Any] = {"created_utc": C.utcnow(), "datasets": {}}
    for ds in datasets:
        entry: dict[str, Any] = {}
        cache = D.load_cache(ds, "train")
        try:
            atoms = D.chem8_atoms(ds, "train", cache)
            entry["chem8_train_atoms"] = list(atoms.shape)
            entry["chem8_channels"] = list(D.CHEM8_CHANNELS)
            entry["chem8_matches_cache_atom_count"] = bool(atoms.shape[0] == int(np.asarray(cache["atom_ptr"])[-1]))
        except Exception as exc:  # pragma: no cover - reported, not hidden
            entry["chem8_error"] = f"{type(exc).__name__}: {exc}"
        if D.support_path(ds).exists():
            sup = D.load_support(ds)
            entry["support"] = {"kappa": sup["kappa"], "a_bar": sup["a_bar"],
                                "n_occupied_keys": int(len(sup["counts"])),
                                "path_count_median_train": float(np.median(sup["path_count"]["train"]))
                                if len(sup["path_count"]["train"]) else None}
        if D.aux_target_path(ds).exists():
            aux = D.load_aux_targets(ds)
            entry["aux"] = {"wc_logp_norm": aux["wc_logp_norm"], "molwt_norm": aux["molwt_norm"],
                            "shuffle_seed": int(C.AUX_SHUFFLE_SEED)}
        out["datasets"][ds] = entry
    C.write_json(C.ANALYSIS / "cache_audit.json", out)
    return out


def _c1_replay_gate(device: str) -> dict[str, Any]:
    """Replay the historical C1 best checkpoint and compare to the frozen record."""
    ref_ckpt = Path(
        "tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/runs/A_s0/best.pt"
    )
    path = C.REPO / ref_ckpt
    out: dict[str, Any] = {"checkpoint": str(ref_ckpt), "exists": path.exists()}
    if not path.exists():
        C.write_json(C.ANALYSIS / "replay_gate.json", out)
        return out
    import lcpr_train as T

    dev = torch.device(device)
    spec = D.frozen_spec()
    tgt = D.targets("Lipophilicity", include_test=True)
    caches = {s: D.load_cache("Lipophilicity", s, tgt["rows"][s]) for s in C.SPLITS}
    train_norm = T._train_structure_norm(caches["train"], spec)
    model = _build_c1_for_replay(spec, train_norm)
    state = torch.load(path, map_location="cpu", weights_only=False)
    own = model.state_dict()
    incoming = {k: v for k, v in state["state_dict"].items() if k in own}
    missing = sorted(set(own) - set(incoming))
    unexpected = sorted(set(state["state_dict"]) - set(own))
    model.load_state_dict(incoming, strict=False)
    model.to(dev)
    preds = T.evaluate_all(model, caches, dev)
    tnorm = tgt["target_norm"]
    metrics = {s: T.split_metrics("regression", preds[s], tgt["y"][s], tnorm) for s in C.SPLITS}
    frozen_test = 0.57616744
    out.update({
        "missing_keys_from_round_local_buffers": missing,
        "unexpected_keys_in_checkpoint": unexpected,
        "best_epoch": int(state.get("best_epoch", -1)),
        "frozen_reference": {"test_rmse": frozen_test,
                             "source": "lcrc_lipo_edit_supervision_public122_s0_v1 REPORT/analysis/results.csv"},
        "metrics": metrics,
        "abs_diff_test_rmse": abs(float(metrics["test"]["rmse"]) - frozen_test),
        "tolerance": C.REPRO_TOL_PRED,
        "passed": bool(abs(float(metrics["test"]["rmse"]) - frozen_test) <= C.REPRO_TOL_PRED),
    })
    C.write_json(C.ANALYSIS / "replay_gate.json", out)
    C.log(f"[replay] test_rmse={metrics['test']['rmse']:.8f} diff={out['abs_diff_test_rmse']:.2e}")
    return out


def _build_c1_for_replay(spec, train_norm):
    import lcpr_model as M

    model = M.RouteLCRC(spec, train_norm, "c1", dict(C.C1_CFG), variant="C1", extra_atom_cols=0)
    return model


def _smoke(datasets: Sequence[str], candidates: Sequence[str], device: str) -> dict[str, Any]:
    import lcpr_train as T

    dev = torch.device(device)
    out: dict[str, Any] = {"created_utc": C.utcnow(), "runs": {}}
    for ds in datasets:
        for cand in candidates:
            out_dir = C.RUNS / "smoke" / f"{cand}__{ds}"
            payload = T.train_candidate(candidate=cand, dataset=ds, device=dev, out_dir=out_dir,
                                        recipe_name="R0" if C.CANDIDATES[cand]["recipe"] == "R*" else None,
                                        epochs_override=2, smoke=True)
            out["runs"][f"{cand}__{ds}"] = {
                "train_json": C.track_rel(out_dir / "train.json"),
                "metrics": payload["metrics"],
                "params": payload["model"]["parameters"],
                "shared_init_bitwise_equal": payload["model"]["shared_init_bitwise_equal"],
                "curve_rows": payload["recipe"]["epochs_run"],
            }
    C.write_json(C.ANALYSIS / "smoke.json", out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LCRC performance-route round")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--datasets", default="Lipophilicity,ESOL,BACE")
    p.add_argument("--dmpnn", action="store_true")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("prepare-aux")
    p.add_argument("--datasets", default="Lipophilicity,ESOL,BACE")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("audit")
    p.add_argument("--datasets", default="Lipophilicity,ESOL,BACE")

    p = sub.add_parser("replay-gate")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("smoke")
    p.add_argument("--datasets", default="Lipophilicity")
    p.add_argument("--candidates", default="C1_R0")
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("train")
    p.add_argument("--candidate", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--recipe", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=None)

    p = sub.add_parser("dmnn-parity")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("analyze")
    p.add_argument("--stage", default="all")

    p = sub.add_parser("report")
    p.add_argument("--stage", default="all")

    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        _prepare([s for s in args.datasets.split(",") if s], with_dmpnn=args.dmpnn, force=args.force)
        return 0
    if args.cmd == "prepare-aux":
        _prepare_aux([s for s in args.datasets.split(",") if s], force=args.force)
        return 0
    if args.cmd == "audit":
        _cache_audit([s for s in args.datasets.split(",") if s])
        return 0
    if args.cmd == "replay-gate":
        _c1_replay_gate(args.device)
        return 0
    if args.cmd == "smoke":
        _smoke([s for s in args.datasets.split(",") if s],
               [s for s in args.candidates.split(",") if s], args.device)
        return 0
    if args.cmd == "dmnn-parity":
        import lcpr_dmpnn as DMN

        r = DMN.parity_check()
        C.write_json(C.ANALYSIS / "dmpnn_parity.json", r)
        print(json.dumps(r, indent=2))
        return 0 if r["passed"] else 1
    if args.cmd == "train":
        import lcpr_train as T

        dev = torch.device(args.device)
        out = Path(args.out_dir) if args.out_dir else C.run_dir(args.candidate, args.dataset)
        T.train_candidate(candidate=args.candidate, dataset=args.dataset, device=dev, out_dir=out,
                          recipe_name=args.recipe, epochs_override=args.epochs)
        return 0
    if args.cmd == "analyze":
        import lcpr_analysis as A

        A.run(stage=args.stage)
        return 0
    if args.cmd == "report":
        import lcpr_report as R

        R.run(stage=args.stage)
        return 0
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
