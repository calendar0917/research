"""Single CLI entry point for the word/sentence port route.

Sub-commands
------------
``prepare``  learn the Train-only word vocabulary, replay it on Valid, write the
             frozen structure cache and its freeze record.
``timing``   short full-Train timing runs used to freeze the shared horizon E.
``train``    one arm x seed run of the unified recipe.
``probe``    Stage-A controlled examples.
``terminal`` the single terminal development-Test scoring of frozen finalists.
``analyze``  summary tables from finished runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws_common as C  # noqa: E402
import ws_train as T  # noqa: E402
import ws_words as W  # noqa: E402


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


class MergeLearner:
    def __init__(self, molecules, train_graphs, table, min_df: int, max_rounds: int = 512):
        self.molecules = molecules
        self.train_graphs = list(train_graphs)
        self.table = table
        self.min_df = int(min_df)
        self.max_rounds = int(max_rounds)
        self.cache: dict = {}
        self.states = {g: W.initial_state(molecules[g], table) for g in self.train_graphs}
        self.history: list[int] = []
        self.log: list[dict] = []
        self.rounds = 0

    def run(self, cap: int) -> None:
        while self.rounds < self.max_rounds and len(set(self.history)) < cap:
            df: dict[int, int] = {}
            instances: dict[int, list] = {}
            for g in self.train_graphs:
                mol = self.molecules[g]
                state = self.states[g]
                atom_piece = state.atom_piece()
                seen: set[int] = set()
                for bond_idx in range(len(mol.bond_u)):
                    if atom_piece[int(mol.bond_u[bond_idx])] == atom_piece[int(mol.bond_v[bond_idx])]:
                        continue
                    found = W._instance_atoms(mol, state, atom_piece, bond_idx, self.table, self.cache)
                    if found is None:
                        continue
                    tid = found[0]
                    if tid not in seen:
                        seen.add(tid)
                        df[tid] = df.get(tid, 0) + 1
                    instances.setdefault(tid, []).append(g)
            candidates = [(t, d) for t, d in df.items() if d >= self.min_df]
            candidates.sort(key=lambda item: (-item[1], self.table.templates[item[0]].n,
                                              self.table.templates[item[0]].canon))
            applied = False
            for tid, d in candidates:
                n_merged = 0
                for g in sorted(set(instances[tid])):
                    n_merged += W.apply_template(self.molecules[g], self.states[g], tid,
                                                 self.table, self.cache)
                if n_merged == 0:
                    continue
                self.history.append(tid)
                self.rounds += 1
                self.log.append({
                    "round": self.rounds, "template_id": tid,
                    "canonical": self.table.templates[tid].canon, "df": d,
                    "n_atoms": self.table.templates[tid].n,
                    "n_merged_molecules": int(len(set(instances[tid]))),
                    "n_merged_instances": int(n_merged),
                    "n_multi_words_after": len(self.table.multi),
                })
                applied = True
                break
            if not applied:
                break


def _intern_monatomic(molecules, graphs, table: W.TemplateTable) -> None:
    empty = np.zeros(0, dtype=np.int64)
    for g in graphs:
        mol = molecules[g]
        for a in range(mol.n_atoms):
            table.intern(mol.atom_feat[a][W.WORDS_ATOM_COLS][None, :], empty, empty, empty)


def _packed_fields(raw: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    import sq_data as SD

    packed = SD.packed_split(raw, split)
    return {
        "atom_feat": np.asarray(packed.atom_feat, dtype=np.int16),
        "degree": np.asarray(packed.degree, dtype=np.int16),
        "min_ring": np.asarray(packed.min_ring, dtype=np.int16),
        "in_cycle": np.asarray(packed.in_cycle, dtype=bool),
        "node_graph": np.asarray(packed.node_graph, dtype=np.int32),
        "bond_u": np.asarray(packed.bond_u, dtype=np.int32),
        "bond_v": np.asarray(packed.bond_v, dtype=np.int32),
        "bond_feat": np.asarray(packed.bond_feat, dtype=np.int16),
        "graph_n_atoms": np.asarray(packed.graph_n_atoms, dtype=np.int32),
        "graph_n_bonds": np.asarray(packed.graph_n_bonds, dtype=np.int32),
        "graph_n_components": np.asarray(packed.graph_n_components, dtype=np.int32),
        "graph_cycle_rank": np.asarray(packed.graph_cycle_rank, dtype=np.int32),
        "row_ids": np.asarray(packed.row_ids, dtype=np.int64),
        "labels": np.asarray(packed.labels, dtype=np.float32),
    }


def cmd_prepare(args) -> None:
    cfg = C.load_config()
    dataset = cfg["datasets"][args.dataset]
    t_start = time.perf_counter()
    raw = C.dataset_cache(dataset)
    atom_names = ("atomic_num", "chirality", "degree", "formal_charge", "num_hs",
                  "num_radical_electrons", "hybridization", "is_aromatic", "is_in_ring")
    import sq_data as SD

    columns = C.check_word_columns(SD.ATOM_COLUMN_NAMES, SD.BOND_COLUMN_NAMES)
    molecules = W.load_molecules(raw)
    t_mol = time.perf_counter() - t_start
    train_graphs = [int(i) for i in raw["train_idx"]]
    valid_graphs = [int(i) for i in raw["valid_idx"]]
    test_graphs = [int(i) for i in raw["test_idx"]]
    table = W.TemplateTable()
    _intern_monatomic(molecules, range(len(molecules)), table)
    n_mono = len(table)
    t_mono = time.perf_counter() - t_start
    mc = cfg["merge"]
    min_df = max(int(mc["min_df_abs"]), int(np.ceil(mc["min_df_frac"] * len(train_graphs))))
    learner = MergeLearner(molecules, train_graphs, table, min_df, int(mc["max_rounds"]))
    learner.run(int(mc["base_cap"]))
    t_base = time.perf_counter() - t_start
    states_train = dict(learner.states)
    stats_base = W.support_stats(molecules, train_graphs, states_train, table)
    expand = (stats_base["multi_atom_atom_coverage"] < float(mc["expand_coverage_floor"])
              or stats_base["n_dual_site_layout_supported_instances"] < int(mc["expand_dual_site_support_floor"]))
    expansion_record = {
        "checked_at_multi_words": len(set(learner.history)),
        "coverage_floor": float(mc["expand_coverage_floor"]),
        "coverage_at_check": stats_base["multi_atom_atom_coverage"],
        "dual_site_support_floor": int(mc["expand_dual_site_support_floor"]),
        "dual_site_support_at_check": stats_base["n_dual_site_layout_supported_instances"],
        "expanded_to": int(mc["expand_cap"]) if expand else None,
        "triggered": bool(expand),
        "decided_before_any_valid_metric": True,
    }
    if expand:
        learner.run(int(mc["expand_cap"]))
        states_train = dict(learner.states)
        t_expand = time.perf_counter() - t_start
    else:
        t_expand = t_base
    raw_history = list(learner.history)
    raw_log = list(learner.log)
    states_valid = W.replay_history(molecules, valid_graphs, table, raw_history)
    char_train = W.char_states(molecules, train_graphs, table)
    char_valid = W.char_states(molecules, valid_graphs, table)
    t_replay = time.perf_counter() - t_start
    # compact the table to the templates that actually occur as words
    used = set(range(n_mono)) | set(raw_history)
    for st in list(states_train.values()) + list(states_valid.values()) + \
            list(char_train.values()) + list(char_valid.values()):
        used |= set(st.tmpl)
    order = sorted(used, key=lambda t: (table.templates[t].n, table.templates[t].canon))
    compact = W.TemplateTable()
    mapping = {old: compact.add_clone(table.templates[old]) for old in order}
    for st in list(states_train.values()) + list(states_valid.values()) + \
            list(char_train.values()) + list(char_valid.values()):
        st.tmpl = [mapping[t] for t in st.tmpl]
    history = [mapping[t] for t in raw_history]
    log = [{**row, "template_id": mapping[row["template_id"]]} for row in raw_log]
    table = compact
    states_train = dict(states_train)
    stats_train = W.support_stats(molecules, train_graphs, states_train, table)
    partition_check = {
        "train": W.verify_partition(molecules, train_graphs, states_train),
        "valid": W.verify_partition(molecules, valid_graphs, states_valid),
        "char_train": W.verify_partition(molecules, train_graphs, char_train),
    }
    support_flag = "LAYOUT_SUPPORT_OK"
    if stats_train["n_dual_site_layout_supported_instances"] < int(mc["expand_dual_site_support_floor"]):
        support_flag = "LAYOUT_SUPPORT_WEAK"
    dims = C.official_dims()

    from ws_model import template_arrays  # local import: needs torch-free arrays only

    tarr = template_arrays(table, dims, int(dims["bond"][0]))
    arrays: dict[str, np.ndarray] = {}
    for key, value in tarr.items():
        arrays[f"tmpl/{key}"] = np.asarray(value)
    for name, split, graphs, states in (
        ("word", "train", train_graphs, states_train),
        ("word", "valid", valid_graphs, states_valid),
        ("char", "train", train_graphs, char_train),
        ("char", "valid", valid_graphs, char_valid),
    ):
        payload = W.build_split(molecules, graphs, states, table)
        for key, value in payload.items():
            arrays[f"{name}/{split}/{key}"] = value
    base_parts = {}
    for split, graphs in (("train", train_graphs), ("valid", valid_graphs), ("test", test_graphs)):
        fields = _packed_fields(raw, split)
        for key, value in fields.items():
            arrays[f"base/{split}/{key}"] = value
            base_parts.setdefault(key, split)
    for split in ("train", "valid", "test"):
        arrays[f"row_ids/{split}"] = arrays[f"base/{split}/row_ids"]
        arrays[f"labels/{split}"] = arrays[f"base/{split}/labels"]
    t_build = time.perf_counter() - t_start
    path = T.cache_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    digest = C.sha256_file(path)
    content = T.content_sha256(arrays)
    freeze = {
        "dataset": dataset,
        "cache_version": T.CACHE_VERSION,
        "cache_path": str(path),
        "cache_sha256": content,
        "cache_file_sha256": digest,
        "commit": C.git_commit(),
        "protocol_config_sha256": C.sha256_json(cfg),
        "word_columns": columns,
        "official_dims": dims,
        "template_table": {
            "n_templates": len(table),
            "n_monatomic_base_types": int(n_mono),
            "n_multi_atom_words": len(set(history)),
            "templates": [
                {"tid": t.tid, "canon": t.canon, "key": t.key, "n": t.n,
                 "labels": t.labels.tolist(), "bu": t.bu.tolist(), "bv": t.bv.tolist(),
                 "btype": t.btype.tolist()} for t in table.templates
            ],
            "candidate_templates_dropped": int(len(order) - (n_mono + len(set(history)))),
        },
        "merge": {"min_df": min_df, "rounds": len(history), "history": [int(h) for h in history],
                  "log": log, "max_rounds": int(mc["max_rounds"])},
        "expansion": expansion_record,
        "support": {"train": stats_train, "at_base_cap": stats_base, "flag": support_flag},
        "partition_check": partition_check,
        "split_sizes": {s: int(len(g)) for s, g in (("train", train_graphs), ("valid", valid_graphs),
                                                    ("test", test_graphs))},
        "timing_seconds": {"molecules_and_ranks": t_mol, "monatomic_intern": t_mono - t_mol,
                           "merge_to_base_cap": t_base - t_mono, "expansion": t_expand - t_base,
                           "support_and_replay": t_replay - t_expand,
                           "tensor_build": t_build - t_replay, "total": time.perf_counter() - t_start},
        "labels_used_for_vocabulary": False,
    }
    C.write_json(T.freeze_path(dataset), freeze)
    if args.write_record:
        C.write_json(T.record_path(dataset), freeze)
    print(json.dumps({"dataset": dataset, "cache": str(path), "content_sha256": content,
                      "file_sha256": digest,
                      "multi_words": len(set(history)), "rounds": len(history),
                      "support_flag": support_flag,
                      "coverage": stats_train["multi_atom_atom_coverage"],
                      "dual_support": stats_train["n_dual_site_layout_supported_instances"],
                      "seconds": freeze["timing_seconds"]["total"]}, indent=2))


# ---------------------------------------------------------------------------
# model spec resolution
# ---------------------------------------------------------------------------

BASE_ARMS = {
    "full_id": ("id", "full"),
    "full_code": ("code", "full"),
    "full_wgine": ("wgine", "full"),
    "erased_id": ("id", "erased"),
    "erased_code": ("code", "erased"),
    "erased_wgine": ("wgine", "erased"),
    "bag": ("code", "bag"),
    "char_full": ("code", "full"),
    "gine_ref": ("gine_ref", "full"),
}


def arm_parts(arm: str):
    """Parse ``full_wgine_attrpair`` style names (optional ``_r1`` recipe suffix)."""
    base, recipe_tag = arm, "r0"
    if base.endswith("_r1"):
        recipe_tag, base = "r1", base[:-3]
    extra, para = "none", "none"
    if base.endswith("_para_rel"):
        para, base = "rel", base[:-9]
    elif base.endswith("_para_cap"):
        para, base = "cap", base[:-9]
    if base.endswith("_attrpair"):
        extra, base = "attrpair", base[:-9]
    elif base.endswith("_rootmean"):
        extra, base = "rootmean", base[:-9]
    if base not in BASE_ARMS:
        raise KeyError(f"unknown arm {arm!r}")
    enc, mode = BASE_ARMS[base]
    return (enc, mode, extra, para), recipe_tag, base


def build_spec(arm: str):
    import ws_model as M

    (enc, mode, extra, para), recipe_tag, _ = arm_parts(arm)
    partition = "char" if arm.startswith("char") else "word"
    return M.ModelSpec(encoder=enc, mode=mode, extra=extra, para=para), recipe_tag, partition


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_timing(args) -> None:
    cfg = C.load_config()
    dataset = cfg["datasets"][args.dataset]
    payload, freeze = T.load_prepared(dataset)
    raw = C.dataset_cache(dataset)
    scaler = C.target_scaler(raw)
    device = args.device
    tr = cfg["recipe"]
    results = {}
    for arm in args.arms:
        spec, recipe_tag, partition = build_spec(arm)
        model = T.make_model(spec, payload, seed=0, device=device)
        recipe = T.Recipe(epochs=int(tr["timing_warm"]) + int(tr["timing_epochs"]))
        if spec.encoder == "gine_ref":
            gine = T.build_gine_splits(raw, scaler, recipe, seed=0)
            out = T.train_gine(model=model, gine=gine, scaler=scaler, recipe=recipe,
                               timing=(int(tr["timing_warm"]), int(tr["timing_epochs"])),
                               log_prefix=arm)
        else:
            splits = {s: T.split_data(payload, partition, s) for s in ("train", "valid")}
            extra = spec.extra == "attrpair"
            hop = spec.para != "none"
            out = T.train_word(model=model, splits=splits, scaler=scaler, recipe=recipe, seed=0,
                               device=device, extra=extra, hop=hop,
                               timing=(int(tr["timing_warm"]), int(tr["timing_epochs"])),
                               log_prefix=arm)
        results[arm] = out["timing"] | {"n_parameters": out["n_parameters"], "seconds": out["seconds"]}
        print(json.dumps({arm: results[arm]}, indent=2), flush=True)
    slowest = max(r["mean_train_step_seconds"] for r in results.values())
    slowest_arm = max(results, key=lambda k: results[k]["mean_train_step_seconds"])
    epoch_seconds = slowest * (1.0 + float(tr["eval_allowance"]))
    horizons = []
    chosen = None
    for epochs in (int(tr["epochs_default"]), *[int(e) for e in tr["epochs_alt"]]):
        minutes = epoch_seconds * epochs * float(tr["timing_safety"]) / 60.0
        horizons.append({"epochs": epochs, "projected_minutes": minutes})
        if chosen is None and minutes <= float(tr["epoch_budget_minutes"]):
            chosen = epochs
    if chosen is None:
        chosen = int(tr["epochs_alt"][-1])
    payload_out = {"dataset": dataset, "per_arm": results, "slowest_arm": slowest_arm,
                   "slowest_epoch_seconds": slowest,
                   "epoch_seconds_with_eval_allowance": epoch_seconds,
                   "eval_allowance": float(tr["eval_allowance"]),
                   "timing_safety": float(tr["timing_safety"]),
                   "projections": horizons, "chosen_epochs": chosen,
                   "epochs_alt": [int(e) for e in tr["epochs_alt"]],
                   "commit": C.git_commit(), "environment": C.environment_snapshot()}
    C.write_json(C.results_root() / f"timing_{args.dataset}.json", payload_out)
    print(json.dumps({"chosen_epochs": chosen, "slowest_epoch_seconds": slowest,
                      "slowest_arm": slowest_arm,
                      "projections": horizons}, indent=2))


def cmd_train(args) -> None:
    cfg = C.load_config()
    dataset = cfg["datasets"][args.dataset]
    payload, freeze = T.load_prepared(dataset)
    raw = C.dataset_cache(dataset)
    scaler = C.target_scaler(raw)
    spec, recipe_tag, partition = build_spec(args.arm)
    epochs = int(args.epochs) if args.epochs else int(cfg["recipe"]["epochs_default"])
    tr = cfg["recipe"]
    recipe = T.Recipe(epochs=epochs)
    if recipe_tag == "r1":
        recipe.peak_lr = float(tr["r1_peak_lr"])
        recipe.weight_decay = float(tr["r1_weight_decay"])
    run_id = args.run_id or T.allocate_run_id(args.dataset, args.arm, args.seed)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    model = T.make_model(spec, payload, seed=args.seed, device=args.device)
    init_hash = T.state_hash(model)
    body = T.body_hash(model)
    n_params = int(sum(p.numel() for p in model.parameters()))
    extra = spec.extra == "attrpair"
    hop = spec.para != "none"
    if spec.encoder == "gine_ref":
        gine = T.build_gine_splits(raw, scaler, recipe, args.seed)
        out = T.train_gine(model=model, gine=gine, scaler=scaler, recipe=recipe,
                           log_prefix=run_id)
    else:
        splits = {s: T.split_data(payload, partition, s, with_hop=hop)
                  for s in ("train", "valid")}
        out = T.train_word(model=model, splits=splits, scaler=scaler, recipe=recipe,
                           seed=args.seed, device=args.device, extra=extra, hop=hop,
                           log_prefix=run_id)
    outdir = T.run_dir(run_id)
    outdir.mkdir(parents=True, exist_ok=True)
    import torch

    final_state = out.pop("final_state", None)
    torch.save({"state": model.state_dict()}, outdir / "state_best.pt")
    if final_state is not None:
        torch.save({"state": final_state}, outdir / "state_final.pt")
    preds = out.pop("predictions_best")
    for split, values in preds.items():
        np.savez_compressed(outdir / f"preds_best_{split}.npz",
                            row_ids=values["row_ids"], pred_raw=values["pred_raw"],
                            target_raw=values["target_raw"])
    final = out.pop("predictions_final")
    for split, values in final.items():
        np.savez_compressed(outdir / f"preds_final_{split}.npz",
                            row_ids=values["row_ids"], pred_raw=values["pred_raw"],
                            target_raw=values["target_raw"])
    record = {
        "run_id": run_id, "arm": args.arm, "dataset": args.dataset, "seed": int(args.seed),
        "partition": partition, "spec": spec.__dict__, "recipe_tag": recipe_tag,
        "recipe": recipe.as_dict(), "n_parameters": n_params,
        "init_state_hash": init_hash, "shared_body_hash": body,
        "commit": C.git_commit(), "cache_sha256": payload["cache_sha256"],
        "freeze_sha256": C.sha256_json(freeze),
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": C.environment_snapshot(),
        "status": "OK", "split": "official", "scaler": scaler,
        "test_used_for_selection": False,
        **{k: v for k, v in out.items() if k not in ("curves",)},
        "n_curve_rows": len(out["curves"]),
    }
    C.write_json(outdir / "run.json", record)
    C.write_json(outdir / "curves.json", {"curves": out["curves"]})
    print(json.dumps({k: record[k] for k in ("run_id", "best", "final", "last20",
                                             "seconds", "n_parameters")}, indent=2))


def cmd_probe(args) -> None:
    import ws_probes as P

    cfg = C.load_config()
    dataset = cfg["datasets"][args.dataset]
    payload, freeze = T.load_prepared(dataset)
    raw = C.dataset_cache(dataset)
    scaler = C.target_scaler(raw)
    dims = C.official_dims()
    out: dict[str, Any] = {"dataset": dataset, "commit": C.git_commit(),
                           "device": args.device,
                           "environment": C.environment_snapshot()}
    t0 = time.perf_counter()
    out["fidelity"] = P.probe_fidelity(payload, freeze)
    out["fidelity_seconds"] = time.perf_counter() - t0
    if not args.quick:
        t1 = time.perf_counter()
        out["relabel_batch"] = P.probe_relabel_and_batch(payload, freeze, device=args.device)
        out["relabel_seconds"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        out["content"] = P.probe_content(dims, device=args.device)
        out["content_seconds"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        out["interface"] = P.probe_interface(payload, dims, device=args.device)
        out["interface_seconds"] = time.perf_counter() - t3
        t4 = time.perf_counter()
        out["health"] = P.probe_health(payload, scaler, device=args.device)
        out["health_seconds"] = time.perf_counter() - t4
    out["total_seconds"] = time.perf_counter() - t0
    out["gates"] = {
        "data_fidelity": bool(out["fidelity"]["ok"]),
        "relabel_batch": bool(args.quick or (out["relabel_batch"]["batch_invariant"]
                                            and out["relabel_batch"]["word_identity_relabel_invariant"]
                                            and out["relabel_batch"]["prediction_relabel_invariant"])),
        "content_id": bool(args.quick or out["content"]["id_fit_success"]),
        "content_code": bool(args.quick or out["content"]["code_fit_success"]),
        "interface_full": bool(args.quick or out["interface"]["full_passes"]),
        "interface_erased_identical": bool(args.quick or out["interface"]["erased_groups_identical"]),
        "training_health": bool(args.quick or (out["health"]["loss_decreased"]
                                              and out["health"]["all_gradients_finite"])),
    }
    C.write_json(C.results_root() / f"probes_{args.dataset}.json", out)
    print(json.dumps({"gates": out["gates"], "seconds": round(out["total_seconds"], 1),
                      "fidelity_ok": out["fidelity"]["ok"],
                      "content": out.get("content", {}).get("wgine_distinguishes"),
                      "interface_full": out.get("interface", {}).get("full_passes"),
                      "interface_erased_identical": out.get("interface", {}).get("erased_groups_identical"),
                      "health": out.get("health", {})}, indent=2, default=str))


def _terminal(args):
    import ws_analyze as A

    A.cmd_terminal(args)


def _summary(args):
    import ws_analyze as A

    A.cmd_summary(args)


def _manifest(args):
    import ws_analyze as A

    A.cmd_manifest(args)


def _measure(args):
    import ws_analyze as A

    A.cmd_measure(args)


def _finalist(args):
    import ws_analyze as A

    A.cmd_finalist(args)


def _inspect(args):
    import ws_analyze as A

    A.cmd_inspect(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ws_run")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default="mollipo")
    p.add_argument("--write-record", action="store_true")
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("timing")
    p.add_argument("--dataset", default="mollipo")
    p.add_argument("--arms", nargs="+", default=["full_code", "full_wgine", "gine_ref"])
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_timing)
    p = sub.add_parser("probe")
    p.add_argument("--dataset", default="mollipo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_probe)
    p = sub.add_parser("train")
    p.add_argument("--dataset", default="mollipo")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("terminal")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=_terminal)
    p = sub.add_parser("summary")
    p.set_defaults(func=_summary)
    p = sub.add_parser("manifest")
    p.add_argument("--jobs-json", default=None)
    p.set_defaults(func=_manifest)
    p = sub.add_parser("finalist")
    p.add_argument("--dataset", default="mollipo")
    p.add_argument("--terminal-runs", nargs="+", required=True)
    p.set_defaults(func=_finalist)
    p = sub.add_parser("measure")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=_measure)
    p = sub.add_parser("inspect")
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", default="mollipo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cases", action="store_true")
    p.set_defaults(func=_inspect)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
