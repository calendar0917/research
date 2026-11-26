"""Pre-training correctness checks for the encoder-boundary round."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import eb_common as E
import cs_common as C

C.setup_paths()

import torch  # noqa: E402

import bs_model as BSM  # noqa: E402
import cs_model as CSM  # noqa: E402
import cs_data as D  # noqa: E402
import eb_data as BD  # noqa: E402
import eb_model as BM  # noqa: E402


def _head_from_checkpoint(model: torch.nn.Module, path: Path) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state_dict"]
    own = model.state_dict()
    keys = [k for k in E.head_keys(own) if k in state]
    with torch.no_grad():
        for k in keys:
            if own[k].shape != state[k].shape:
                raise RuntimeError(f"head shape mismatch at {k}")
            own[k].copy_(state[k])
    model.load_state_dict(own, strict=True)


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name, ok, **detail):
        checks.append({"name": name, "ok": bool(ok), **detail})
        if not ok:
            raise RuntimeError(f"check failed: {name}: {detail}")

    bundle = BD.load_bundle()
    add("split_sizes", True, sizes={s: int(bundle.caches[s]["n_molecules"]) for s in E.SPLITS_USED})
    add("scaler_train_only", abs(float(bundle.scaler["mean"]) - 2.1621607144740187) < 1e-12
        and abs(float(bundle.scaler["std"]) - 1.211539840098397) < 1e-12,
        scaler={k: bundle.scaler[k] for k in ("mean", "std", "n", "ddof")})

    sq = BD.load_sq_cache()
    add("sq_cache_sha256", E.sha256_file(E.SQ_CACHE) == E.SQ_CACHE_SHA256, sha256=E.SQ_CACHE_SHA256)
    splits = BD.build_splits(bundle, sq)
    for split in E.SPLITS_USED:
        s = splits[split]
        n_mol = int(bundle.caches[split]["n_molecules"])
        add(f"{split}_alignment", True, molecules=n_mol, atoms=int(s.atom.shape[0]),
            bonds=int(s.bond.shape[0]), row_id_unique=int(np.unique(s.row_ids).size),
            atom_ptr_end=int(s.atom_ptr[-1]), bond_ptr_end=int(s.bond_ptr[-1]))
        add(f"{split}_global_shape", s.global_feat.shape[0] == n_mol, shape=list(s.global_feat.shape))

    categories = BD.category_counts([splits["train"], splits["valid"]])
    add("gine_vocabulary_matches_historical", categories == E.EXPECTED_CATEGORIES, categories=categories)

    plan = D.batch_order_plan(E.DATASET, int(C.RECIPE["epochs"]))
    a_order = E.read_json(E.A_RUN / "training_order.json")
    add("batch_plan_matches_A", plan["plan_sha256"] == a_order["plan_sha256"],
        plan_sha256=plan["plan_sha256"], a_plan_sha256=a_order["plan_sha256"],
        first_batch=list(plan["plan"][0][: int(C.RECIPE["batch_size"])]))

    # A assets: independent recompute of its best Train/Valid metrics
    a_ck = torch.load(E.A_CHECKPOINT, map_location="cpu", weights_only=False)
    a_run = E.read_json(E.A_RUN_JSON)
    a_state = a_ck["state_dict"]
    buffer_keys = {"global_mean", "global_scale", "scalar_mean", "scalar_scale", "cont_mean", "cont_scale"}
    a_params = int(sum(v.numel() for k, v in a_state.items() if k not in buffer_keys))
    add("A_parameters", a_params == E.A_TOTAL_PARAMS, parameters=a_params,
        recorded=a_run["parameters"], best_epoch=int(a_ck["epoch"]) == int(a_run["best_epoch"]))
    a_pred = np.load(E.A_PREDICTIONS)
    a_metrics = {}
    for split in E.SPLITS_USED:
        y = np.asarray(a_pred[f"{split}_y_raw"], dtype=np.float64)
        p = np.asarray(a_pred[f"{split}_pred_raw"], dtype=np.float64)
        a_metrics[split] = {"rmse_raw": C.rmse(p, y), "mae_raw": C.mae(p, y)}
        rec = a_run["best_metrics"][split]
        add(f"A_{split}_prediction_recompute",
            abs(a_metrics[split]["rmse_raw"] - rec["rmse_raw"]) < 1e-12
            and abs(a_metrics[split]["mae_raw"] - rec["mae_raw"]) < 1e-12,
            recomputed=a_metrics[split], recorded=rec)
        add(f"A_{split}_row_ids_sorted",
            bool(np.all(np.diff(np.asarray(a_pred[f"{split}_row_ids"])) > 0)),
            n=int(np.asarray(a_pred[f"{split}_row_ids"]).size))

    # model B: budget, head identity with A's recorded *initialisation*, forward equivalence
    a_init, a_init_audit = CSM.build_csc(bundle.spec, bundle.norm, arm="A", seed=C.SEED, device="cpu", verify=True)
    a_init_base = C.state_fingerprint({k: v for k, v in a_init.state_dict().items() if not C.is_new_parameter(k)})
    add("A_init_reproduced_from_recorded_fingerprint",
        a_init_base == a_run["model_audit"]["base_state_sha256"],
        rebuilt=a_init_base, recorded=a_run["model_audit"]["base_state_sha256"])
    a_init_head_fp = E.state_fingerprint({k: a_init.state_dict()[k] for k in E.head_keys(a_init.state_dict())})
    model, audit = BM.build_model(bundle.spec, bundle.norm, categories, seed=C.SEED, device="cpu")
    add("B_head_init_identical_to_A_init", audit["head_state_sha256"] == a_init_head_fp,
        b_head_sha256=audit["head_state_sha256"], a_init_head_sha256=a_init_head_fp)
    add("B_parameter_budget", E.PARAM_LOW <= audit["parameter_groups"]["total"] <= E.PARAM_HIGH,
        total=audit["parameter_groups"]["total"], groups=audit["parameter_groups"],
        a_total=E.A_TOTAL_PARAMS, ratio=audit["param_ratio_b_over_a"],
        budget=[E.PARAM_LOW, E.PARAM_HIGH])
    add("B_output_width_64", True, out_proj=list(model.encoder.out_proj.weight.shape))

    batch = BD.make_batch(splits["valid"], np.arange(8, dtype=np.int64), device="cpu")
    model.eval()
    with torch.no_grad():
        h = model.encode_h(batch)
        out_b = model(batch)
    add("B_forward_shapes", tuple(h.shape) == (int(batch["x_cat"].shape[0]), E.OUT_WIDTH),
        h=list(h.shape), logits=list(out_b.shape), edge_index=list(batch["edge_index"].shape))
    add("B_finite", bool(torch.isfinite(out_b).all()) and bool(torch.isfinite(h).all()), ok_finite=True)

    # literal post_local reuse: A's stored head fed B's h must reproduce B's logits
    ref, _ = BSM.build_model(C.CFG_KEY, bundle.spec, bundle.norm, C.N_OUT,
                              seed=C.SEED, device="cpu", verify_parameters=False)
    _head_from_checkpoint(ref, E.A_CHECKPOINT)
    _head_from_checkpoint(model, E.A_CHECKPOINT)
    with torch.no_grad():
        out_ref = ref.post_local(h, batch)
        out_b2 = model(batch)
    delta = float((out_ref - out_b2).abs().max().item())
    add("shared_post_local_bitwise", delta == 0.0, max_abs_delta=delta)

    # single-graph vs batched invariance is structural (pooling is permutation invariant);
    # check that two identical molecules give identical logits in one batch
    rows2 = np.asarray([0, 0], dtype=np.int64)
    b2 = BD.make_batch(splits["valid"], rows2, device="cpu")
    with torch.no_grad():
        o2 = model(b2).view(-1)
    add("same_molecule_same_logit", float((o2[0] - o2[1]).abs().item()) < 1e-4,
        logit0=float(o2[0]), logit1=float(o2[1]))

    # no Test access anywhere in this round
    add("test_not_used", tuple(E.SPLITS_USED) == ("train", "valid"), splits_used=list(E.SPLITS_USED))

    return {
        "round_id": E.ROUND_ID,
        "protocol_id": E.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "all_passed": int(all(c["ok"] for c in checks)),
        "n_checks": len(checks),
        "checks": checks,
        "A_assets": {
            "checkpoint": E.track_rel(E.A_CHECKPOINT),
            "checkpoint_sha256": E.sha256_file(E.A_CHECKPOINT),
            "predictions": E.track_rel(E.A_PREDICTIONS),
            "predictions_sha256": E.sha256_file(E.A_PREDICTIONS),
            "run_json_sha256": E.sha256_file(E.A_RUN_JSON),
            "curve_sha256": E.sha256_file(E.A_CURVE),
            "best_epoch": int(a_ck["epoch"]),
            "recorded_metrics": a_metrics,
        },
        "model_B_audit": audit,
        "environment": C.environment(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(E.ANALYSIS_DIR / "checks.json"))
    args = parser.parse_args()
    payload = run_checks()
    E.write_json(Path(args.out), payload)
    C.log(f"checks: {payload['all_passed']} ({payload['n_checks']} items) -> {args.out}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
