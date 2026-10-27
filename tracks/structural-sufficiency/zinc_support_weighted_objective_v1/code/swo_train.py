"""Run one arm of the support-weighted-objective pilot.

    python swo_train.py --arm E --seed 0 --device cuda:0    # uniform raw-y L1
    python swo_train.py --arm W --seed 0 --device cuda:0    # frozen support weights

Both arms are the *unchanged* DCC-L / B-R0 body (494,193 parameters, forward
untouched, imported verbatim from ``zinc_dual_channel_composer_v1``), the same
frozen R0 recipe (read from ``zinc_performance_search_v1``), the same seed, the
same SSL encoder, the same initial state and therefore the same batch order.  The
only difference is the per-training-molecule weight vector (``None`` for E).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import swo_common as S
import swo_weights


def _env() -> dict:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_names": ([torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]
                      if torch.cuda.is_available() else []),
        "driver": None,
    }
    try:
        import subprocess
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception:
        pass
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=("E", "W"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    import ad_common as C
    import ad_data as D
    import swo_train_core as T

    arm = args.arm
    seed = int(args.seed)
    tag = args.tag or (f"{arm}_s{seed}" + ("_smoke" if args.smoke else ""))
    out = S.RUNS / tag
    out.mkdir(parents=True, exist_ok=True)

    recipe = S.r0_recipe()
    if args.smoke:
        recipe = dict(recipe, epochs=2)
    if args.epochs is not None:
        recipe = dict(recipe, epochs=int(args.epochs))

    weights = None
    wmeta = None
    cfg = S.read_json(S.CONFIG)
    frozen = cfg["weights_frozen"]
    if arm == "W":
        # Recompute the frozen function from TRAIN STRUCTURE on this machine and
        # check it against the sha256 declared in configs/swo_v1.json before any
        # training step. No weight file is shipped, so the function itself is
        # what is frozen, not an artifact.
        built = swo_weights.build()
        w = built["weights"]
        got = S.sha256_array(w)
        if got != frozen["sha256"]:
            raise SystemExit(f"weight function drifted: {got} != {frozen['sha256']}")
        st = built["stats"]
        checks = {
            "n": st["n_train"] == frozen["n"],
            "vocab": st["vocab_size"] == frozen["vocab_size"],
            "ess": abs(st["effective_sample_size"] - frozen["effective_sample_size"]) < 1e-6,
            "clip": abs(st["clip_fraction"] - frozen["clip_fraction"]) < 1e-9,
            "min": abs(st["weights"]["min"] - frozen["w_min"]) < 1e-9,
            "max": abs(st["weights"]["max"] - frozen["w_max"]) < 1e-9,
            "mean": abs(st["weights"]["mean"] - 1.0) < 1e-12,
            "label_free": st["labels_used"] is False,
            "train_only": st["valid_or_test_used"] is False,
        }
        if not all(checks.values()):
            raise SystemExit(f"frozen weight statistics mismatch: {checks}")
        weights = w
        wmeta = {"weight_sha256": got, "recomputed_on_host": True,
                 "effective_sample_size": st["effective_sample_size"],
                 "effective_sample_size_fraction": st["effective_sample_size_fraction"],
                 "weights_min": st["weights"]["min"], "weights_max": st["weights"]["max"],
                 "clip_fraction": st["clip_fraction"],
                 "frac_gt_1": st["weights"]["frac_gt_1"],
                 "corr_weight_n_atoms": st["corr_weight_n_atoms"],
                 "frozen_function": frozen["function"], "checks": checks}

    cand, builder, collate = S.dcc_candidate()
    device = torch.device(args.device)

    # initial state + SSL identity, hashed the same way the owning rounds do
    model, minfo = builder(cand, D.load_norm(), seed, device)
    init = {"initial_state_sha256": C.state_hash(model),
            "parameters": C.parameter_count(model),
            "ssl": (minfo.get("ssl") or {}).get("source"),
            "ssl_sha256": (minfo.get("ssl") or {}).get("sha256")}
    if init["parameters"] > S.PARAM_CAP:
        raise SystemExit(f"{init['parameters']} > cap")
    del model

    declaration = {
        "round": S.ROUND_ID, "arm": arm, "seed": seed, "tag": tag,
        "candidate": cand, "recipe": recipe, "optimizer_changes": None,
        "architecture_changes": None, "model": "DCC-L / B-R0 (verbatim import)",
        "weights": wmeta, "init": init, "env": _env(),
        "declared_at": S.now(), "official_test_loaded": False,
        "hypothesis": ("weighting structurally rare training molecules up teaches "
                       "compositional coverage, improving valid low_support/any_unseen"),
    }
    S.write_json(out / "swo.json", declaration)

    info = T.train(
        cand, seed, device,
        epochs=int(recipe["epochs"]),
        recipe_override=recipe,
        model_builder=builder,
        collate_fn=collate,
        out_dir=out,
        loss="l1",
        sample_weight=weights,
        force=args.force,
    )
    # stamp the round's provenance into the run record (train.json is written by
    # the trainer; we only append, never overwrite its own fields)
    info["round"] = S.ROUND_ID
    info["arm"] = arm
    info["sample_weighting"] = wmeta
    info["init"] = init
    info["env"] = declaration["env"]
    info["declared_at"] = declaration["declared_at"]
    S.write_json(out / "train.json", info)
    decl = S.read_json(out / "swo.json")
    decl["finished_at"] = S.now()
    best = info.get("best_valid") or info.get("best") or {}
    decl["best_valid"] = best
    S.write_json(out / "swo.json", decl)
    print(f"[{S.ROUND_ID} {tag}] best={best.get('valid_mae'):.6f} "
          f"@{best.get('epoch')} gpu_h={info.get('gpu_hours')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
