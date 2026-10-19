#!/usr/bin/env python
"""Standalone CPU/GPU functional verification for the frozen GRIT reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import reference_common as rc

rc.setup_import_paths()

from torch_geometric import seed_everything  # noqa: E402
from torch_geometric.graphgym.model_builder import create_model  # noqa: E402
from torch_geometric.graphgym.utils.comp_budget import params_count  # noqa: E402

import reference_checks as checks  # noqa: E402


def _base_dataset(loader):
    ds = loader.dataset
    return getattr(ds, "dataset", ds)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--max-train-graphs", type=int, default=64)
    p.add_argument("--out", type=Path,
                   default=rc.BUNDLE / "functional_checks.json")
    args = p.parse_args(argv)

    guard = {"pass": False}
    try:
        rc.load_reference_config(seed=args.seed, device=args.device,
                                 splits="test", timing=False)
    except ValueError as e:
        guard = {"pass": True, "error": str(e)}
    else:
        guard = {"pass": False, "error": "test split was accepted"}

    cfg = rc.load_reference_config(
        seed=args.seed,
        device=args.device,
        splits="train",
        timing=False,
        max_train_graphs=args.max_train_graphs,
        functional_probe=False,
        out_dir=rc.BUNDLE / "runs",
        run_name="verify",
    )
    seed_everything(cfg.seed)
    loaders = rc.make_loaders()
    model = create_model()
    dataset = _base_dataset(loaders[0])
    data0 = dataset[0].clone()

    probe = {
        "params": checks.parameter_check(model),
        "rrwp": checks.rrwp_check(data0),
        "participation": checks.participation_check(model, data0),
        "forward_backward": checks.forward_backward_check(model, data0),
        "renumbering": checks.renumbering_check(model, dataset, count=5),
    }
    probe["all_pass"] = all(bool(v.get("pass")) for v in probe.values()
                            if isinstance(v, dict) and "pass" in v)

    base = _base_dataset(loaders[0])
    keys = sorted(base.data.keys())
    no_test_attr = not any(k.startswith("test") for k in keys)
    no_test_loader = len(loaders) == 1 and int(cfg.share.num_splits) == 1
    config_guard = {
        "test_split_guard": guard,
        "loader_count": len(loaders),
        "num_splits": int(cfg.share.num_splits),
        "no_test_loader": no_test_loader,
        "dataset_keys": keys,
        "no_test_attribute": no_test_attr,
        "pass": bool(guard["pass"] and no_test_loader and no_test_attr),
    }

    payload = {
        "schema": "zinc_grit_reference_functional_checks_v1",
        "status": "ok",
        "upstream_commit": rc.UPSTREAM_COMMIT,
        "config_sha256": rc.sha256_file(rc.DEFAULT_CONFIG),
        "patch_sha256": rc.sha256_file(rc.PATCH),
        "device": args.device,
        "seed": args.seed,
        "train_graphs_loaded": len(dataset),
        "params_total": int(params_count(model)),
        "checks": probe,
        "test_disable_checks": config_guard,
        "pass": bool(probe["all_pass"] and config_guard["pass"]),
        "notes": [
            "Only the official ZINC train split is loaded.",
            "Renumbering error is max |pred - pred(permuted graph)| in float32.",
            "The forward/backward check runs on a deep-copied model so the "
            "reference model's BN state is untouched.",
        ],
    }
    rc.write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
