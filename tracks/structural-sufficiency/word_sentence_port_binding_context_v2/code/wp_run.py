"""CLI for the port-binding / sentence-context round (``wp`` = word-port)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import wp_common as C
import wp_data as D

DATASET_ALIASES = {"lipo": "ogbg-mollipo", "mollipo": "ogbg-mollipo",
                   "esol": "ogbg-molesol", "molesol": "ogbg-molesol"}


def dataset_name(alias: str) -> str:
    return DATASET_ALIASES.get(alias, alias)


def cmd_prepare(args) -> None:
    for alias in args.dataset:
        name = dataset_name(alias)
        t0 = time.perf_counter()
        arrays, freeze = D.build_derived(name)
        freeze["timing_seconds"] = time.perf_counter() - t0
        freeze["commit"] = C.git_commit()
        freeze["protocol_config_sha256"] = C.sha256_json(C.load_config())
        path = C.derived_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        freeze["derived_file_sha256"] = C.sha256_file(path)
        if args.write_record:
            C.write_json(C.derived_record_path(name), C.jsonable(freeze))
        print(json.dumps({"dataset": name, "derived_sha256": freeze["derived_sha256"],
                          "file": str(path), "seconds": freeze["timing_seconds"],
                          "support": freeze["support"]}, indent=2))


def _load(dataset: str):
    return D.load_dataset(dataset)


def _max_rss_mb() -> float | None:
    import resource

    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:  # pragma: no cover
        return None


def _cpu_seconds() -> float | None:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def cmd_train(args) -> None:
    import torch

    import wp_model as M
    import wp_train as T

    cfg = C.load_config()
    name = dataset_name(args.dataset)
    data = _load(name)
    raw = data["raw"]
    scaler = C.target_scaler(raw)
    spec = resolve_spec(args)
    device = args.device
    recipe = T.Recipe(epochs=args.epochs, batch=args.batch, seed=args.seed,
                      recipe_id=args.recipe, weight_decay=args.weight_decay)
    wd = cfg["recipe"].get("r_g_weight_decay", 1e-3)
    if args.recipe == "RG":
        recipe.weight_decay = float(wd)
    deadline = None
    if args.deadline:
        deadline = float(args.deadline)
    tmpl = {k[len("tmpl/"):]: v for k, v in data["words"].items() if k.startswith("tmpl/")}
    tmpl["dist"] = data["derived"]["tmpl/dist"]
    timing = None
    if args.timing:
        warm, timed = (int(x) for x in args.timing.split("+"))
        timing = (warm, timed)
        recipe.epochs = warm + timed
    if args.arm in ("GINE_REF", "GINE_RAW"):
        splits = T.gine_data_lists(raw, scaler)
        model = T.build_gine_model(raw, args.seed, device, raw_features=args.arm == "GINE_RAW")
        kind = "gine"
    else:
        splits = {split: D.make_split(data, split) for split in ("train", "valid")}
        if args.partition == "random":
            splits, tmpl = build_random_partition(data, splits, name, cfg)
        model = T.build_word_model(spec, tmpl, args.seed, device, hidden=cfg["model"]["hidden"])
        kind = "word"
    n_params = int(sum(p.numel() for p in model.parameters()))
    run_id = args.run_id or f"{name.replace('ogbg-', '')}_{args.arm}_s{args.seed}" + \
        ("_random" if args.partition != "frozen" else "") + \
        (f"_{args.recipe}" if args.recipe != "R0" else "")
    init_hash = None
    if kind == "word":
        init_hash = M.init_by_name(model, 1000 + args.seed)  # identical to build-time init
    print(json.dumps({"run_id": run_id, "arm": args.arm, "seed": args.seed,
                      "params": n_params, "epochs": recipe.epochs,
                      "spec": spec.as_dict(), "timing": timing,
                      "device": device, "torch": torch.__version__}), flush=True)
    started = time.time()
    result = T.train(model=model, kind=kind, splits=splits, scaler=scaler, recipe=recipe,
                     device=device, run_id=run_id, deadline=deadline, resume=args.resume,
                     timing=timing)
    run = {
        "run_id": run_id, "dataset": name, "arm": args.arm, "seed": args.seed,
        "spec": spec.as_dict(), "recipe": recipe.as_dict(), "kind": kind,
        "partition": args.partition, "n_parameters": n_params,
        "started_unix": started, "wall_seconds": result["wall_seconds"],
        "status": "PARTIAL" if result.get("partial") else "COMPLETE",
        "resumed_from_epoch": result.get("resumed_from_epoch"),
        "commit": C.git_commit(), "config_sha256": C.sha256_json(cfg),
        "inherited_word_cache_sha256": data["record"].get("cache_sha256"),
        "derived_sha256": data["derived_record"].get("derived_sha256"),
        "order_seed": T.ORDER_SEED_BASE + args.seed,
        "scaler": scaler, "init_hash": init_hash,
        "device_name": _device_name(device),
        "torch": torch.__version__,
        "max_rss_mb": _max_rss_mb(),
        "cpu_seconds": _cpu_seconds(),
        "timing": result.get("timing"),
        "curves": result["curves"],
        "epoch_seconds_including_eval": result.get("epoch_seconds_including_eval"),
        "first_train_batch_seconds": result["batch_seconds"][:3] if result.get("batch_seconds") else None,
        "predicted_epochs": recipe.epochs,
    }
    for key in ("best", "final", "last20"):
        if key in result:
            run[key] = C.jsonable(_strip(result[key]))
    if "predictions_best" in result:
        best = result["predictions_best"]
        run["best_train_rmse_raw"] = best["train"]["rmse_raw"]
        run["best_valid_rmse_raw"] = best["valid"]["rmse_raw"]
    import torch as _t

    if torch.cuda.is_available():
        run["cuda_device_name"] = torch.cuda.get_device_name(0)
        run["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    T.write_run(run_id, run)
    print(json.dumps({"run_id": run_id, "status": run["status"],
                      "best": run.get("best"), "final": run.get("final"),
                      "wall_seconds": run["wall_seconds"]}, indent=2), flush=True)


def _strip(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in payload.items():
        if isinstance(value, Mapping) and "pred_raw" in value:
            out[key] = {"rmse_raw": value["rmse_raw"], "mae_raw": value["mae_raw"]}
        else:
            out[key] = value
    return out


def _device_name(device: str) -> str:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def resolve_spec(args):
    import wp_model as M

    base = M.ARMS.get(args.arm)
    port = args.port or (base.port if base else "bind")
    context = args.context or (base.context if base else "rel")
    instance = args.instance or (base.instance if base else "graph")
    readout = args.readout or (base.readout if base else "vector")
    bag = bool(args.bag) or (base.bag if base else False)
    return M.ModelSpec(port=port, context=context, instance=instance, readout=readout, bag=bag)


def build_random_partition(data: Mapping[str, Any], splits: Mapping[str, Any], dataset: str,
                           cfg: Mapping[str, Any]):
    """D3: fixed-salt random connected partition (see PROTOCOL §5 D3)."""
    import wp_random_part as R

    rebuilt, template = R.rebuild_splits(data, splits, cfg)
    path = C.analysis_dir() / f"random_partition_{dataset.replace('ogbg-', '')}.json"
    C.write_json(path, C.jsonable(R.summary(rebuilt, template)))
    return rebuilt, template


def cmd_smoke(args) -> None:
    """3-epoch smoke: reload ``state_best`` and ``state_final`` independently."""
    import torch

    import wp_model as M
    import wp_train as T

    name = dataset_name(args.dataset)
    data = _load(name)
    raw = data["raw"]
    scaler = C.target_scaler(raw)
    spec = resolve_spec(args)
    device = args.device
    tmpl = {k[len("tmpl/"):]: v for k, v in data["words"].items() if k.startswith("tmpl/")}
    tmpl["dist"] = data["derived"]["tmpl/dist"]
    model = T.build_word_model(spec, tmpl, 0, device, hidden=64)
    splits = {split: D.make_split(data, split) for split in ("train", "valid")}
    recipe = T.Recipe(epochs=3, seed=0)
    run_id = args.run_id or f"smoke_{args.arm.lower()}"
    res = T.train(model=model, kind="word", splits=splits, scaler=scaler, recipe=recipe,
                  device=device, run_id=run_id)
    out_dir = T.run_dir(run_id)
    report: dict[str, Any] = {"run_id": run_id, "best_epoch": res["best"]["epoch"],
                              "valid_curves": [c["valid_rmse_raw"] for c in res["curves"]]}
    saved = {tag: np.load(out_dir / f"preds_{tag}.npz") for tag in ("best", "final")}
    for tag in ("best", "final"):
        m = T.build_word_model(spec, tmpl, 0, device, hidden=64)
        payload = torch.load(out_dir / f"state_{tag}.pt", map_location=device, weights_only=False)
        m.load_state_dict(payload["model"])
        m.eval()
        block = {}
        for split in ("train", "valid"):
            out = T.evaluate_word(m, splits[split], device, scaler)
            metrics = T.metric_block(out, scaler)
            block[split] = {
                "rmse_raw": metrics["rmse_raw"],
                "rows_are_split_rows": bool(np.array_equal(np.sort(metrics["row_ids"]),
                                                           np.sort(splits[split].row_ids))),
                "max_abs_diff_vs_saved": float(np.max(np.abs(
                    saved[tag][f"{split}_pred"] - metrics["pred_raw"]))),
                "saved_rows_by_rowid": bool(np.array_equal(
                    np.sort(saved[tag][f"{split}_row"]), np.sort(splits[split].row_ids))),
            }
        report[tag] = block
    report["best_epoch_is_not_final"] = bool(res["best"]["epoch"] != recipe.epochs)
    report["ok"] = bool(all(v["max_abs_diff_vs_saved"] <= 1e-5 and v["rows_are_split_rows"]
                            for tag in ("best", "final") for v in report[tag].values()))
    print(json.dumps(report, indent=2), flush=True)
    C.write_json(C.analysis_dir() / "smoke.json", C.jsonable(report))


def cmd_timing(args) -> None:
    import torch

    import wp_model as M
    import wp_train as T

    cfg = C.load_config()
    name = dataset_name(args.dataset)
    data = _load(name)
    raw = data["raw"]
    scaler = C.target_scaler(raw)
    device = args.device
    tmpl = {k[len("tmpl/"):]: v for k, v in data["words"].items() if k.startswith("tmpl/")}
    tmpl["dist"] = data["derived"]["tmpl/dist"]
    warm, timed = (int(x) for x in cfg["recipe"]["timing"].split("+"))
    out: dict[str, Any] = {"dataset": name, "warm_epochs": warm, "timed_epochs": timed,
                           "unit": "s/epoch", "device_name": _device_name(device)}
    for arm in args.arms:
        args.arm = arm
        spec = resolve_spec(args)
        if arm in ("GINE_REF", "GINE_RAW"):
            splits = T.gine_data_lists(raw, scaler)
            model = T.build_gine_model(raw, 0, device, raw_features=arm == "GINE_RAW")
            kind = "gine"
        else:
            splits = {split: D.make_split(data, split) for split in ("train", "valid")}
            model = T.build_word_model(spec, tmpl, 0, device, hidden=cfg["model"]["hidden"])
            kind = "word"
        recipe = T.Recipe(epochs=warm + timed, seed=0)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        res = T.train(model=model, kind=kind, splits=splits, scaler=scaler, recipe=recipe,
                      device=device, run_id=f"timing_{arm.lower()}", timing=(warm, timed))
        entry = dict(res["timing"])
        entry["params"] = res["n_parameters"]
        if torch.cuda.is_available():
            entry["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        out[arm] = entry
        print(json.dumps({arm: entry}, indent=2), flush=True)
    t_epoch = max(v["median_train_step_seconds"] for k, v in out.items()
                  if isinstance(v, dict) and "median_train_step_seconds" in v)
    out["t_epoch_max_median"] = t_epoch
    out["projection"] = {
        "E240_minutes": 240 * t_epoch * 1.25 / 60.0,
        "E160_minutes": 160 * t_epoch * 1.25 / 60.0,
        "budget_minutes": 25.0,
    }
    epochs = 240 if out["projection"]["E240_minutes"] <= 25.0 else (
        160 if out["projection"]["E160_minutes"] <= 25.0 else 160)
    out["frozen_epochs"] = epochs
    out["meets_budget"] = out["projection"]["E240_minutes" if epochs == 240 else "E160_minutes"] <= 25.0
    out["commit"] = C.git_commit()
    C.write_json(C.analysis_dir() / f"timing_{name.replace('ogbg-', '')}.json", C.jsonable(out))
    print(json.dumps({"t_epoch_max_median": t_epoch, "frozen_epochs": epochs,
                      "projection": out["projection"]}, indent=2))


def cmd_checks(args) -> None:
    import wp_probes as P

    P.run_engineering_checks(args)


def cmd_probe(args) -> None:
    import wp_probes as P

    P.run_probes(args)


def cmd_analyze(args) -> None:
    import wp_analyze as A

    A.write_summary(args)


def cmd_support(args) -> None:
    import wp_analyze as A

    A.write_support(args)


def cmd_finalist(args) -> None:
    import wp_analyze as A

    A.write_finalist(args)


def cmd_terminal(args) -> None:
    import wp_analyze as A

    A.run_terminal(args)


def cmd_freeze(args) -> None:
    import wp_report as R

    R.write_freeze(args)


def cmd_manifest(args) -> None:
    import wp_report as R

    R.write_manifest(args)


def cmd_intervene(args) -> None:
    import wp_report as R

    R.write_interventions(args)


def cmd_test_support(args) -> None:
    import wp_analyze as A

    A.write_test_support(args)


def cmd_cases(args) -> None:
    import wp_report as R

    R.write_cases(args)


def _configure_threads() -> None:
    """Optional thread cap so two local workers do not oversubscribe the CPU."""
    import os

    value = os.environ.get("WP_THREADS", "").strip()
    if value:
        import torch

        torch.set_num_threads(int(value))
        torch.set_num_interop_threads(1) if torch.get_num_interop_threads() > 1 else None


def main(argv: list[str] | None = None) -> int:
    _configure_threads()
    ap = argparse.ArgumentParser(prog="wp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", nargs="+", default=["lipo"])
    p.add_argument("--write-record", action="store_true")
    p.set_defaults(func=cmd_prepare)

    def add_train_args(parser, train=True):
        parser.add_argument("--dataset", default="lipo")
        parser.add_argument("--arm", default="BIND_REL")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--port")
        parser.add_argument("--context")
        parser.add_argument("--instance")
        parser.add_argument("--readout")
        parser.add_argument("--bag", action="store_true")
        parser.add_argument("--partition", default="frozen", choices=["frozen", "random"])
        if train:
            parser.add_argument("--epochs", type=int, default=240)
            parser.add_argument("--batch", type=int, default=128)
            parser.add_argument("--recipe", default="R0")
            parser.add_argument("--weight-decay", type=float, default=1e-5)
            parser.add_argument("--deadline")
            parser.add_argument("--resume", action="store_true")
            parser.add_argument("--timing")
            parser.add_argument("--run-id")

    p = sub.add_parser("train")
    add_train_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("smoke")
    add_train_args(p, train=False)
    p.add_argument("--run-id")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("timing")
    add_train_args(p, train=False)
    p.add_argument("--arms", nargs="+", default=["MEAN_SELF", "BIND_REL", "GINE_REF"])
    p.set_defaults(func=cmd_timing)

    p = sub.add_parser("checks")
    p.add_argument("--dataset", default="lipo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=32)
    p.set_defaults(func=cmd_checks)

    p = sub.add_parser("probe")
    p.add_argument("--dataset", default="lipo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--which", nargs="+", default=["a1", "a2"])
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("analyze")
    p.add_argument("--dataset", default="lipo")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("support")
    p.add_argument("--dataset", default="lipo")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_support)

    p = sub.add_parser("finalist")
    p.add_argument("--frozen-dir", default="")
    p.add_argument("--include", default="")
    p.set_defaults(func=cmd_finalist)

    p = sub.add_parser("terminal")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_terminal)

    p = sub.add_parser("test-support")
    p.add_argument("--dataset", default="lipo")
    p.set_defaults(func=cmd_test_support)

    p = sub.add_parser("freeze")
    p.add_argument("--dataset", default="lipo")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("manifest")
    p.add_argument("--dataset", default="lipo")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("intervene")
    p.add_argument("--tag", default="")
    p.add_argument("--dataset", default="lipo")
    p.add_argument("--arm", default="BIND_REL")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_intervene)

    p = sub.add_parser("cases")
    p.add_argument("--dataset", default="lipo")
    p.add_argument("--arm-a", default="")
    p.add_argument("--arm-b", default="")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_cases)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
