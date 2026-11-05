"""Static/runtime audits for the no-MP sprint.

Produces:
  analysis/architecture_audit.json
  analysis/information_access_audit.json
  analysis/initialization_parity.json

GPU-dependent runtime evidence is merged from ``analysis/preflight_gpu*.json``
when present.  Everything else uses only frozen data identities and the CPU
model builder.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import nmp_common as C  # noqa: E402
import nmp_data as D  # noqa: E402
import nmp_model as M  # noqa: E402
import pc_model as BM  # noqa: E402

FORBIDDEN_NEW_MODULE_NAMES = {
    "bond_index",
    "bond_feat",
    "bond_batch",
    "bond_type",
    "nbr_atom",
    "nbr_ptr",
    "path_ptr",
    "path_end",
    "edge_index",
    "edge_attr",
    "adjacency",
    "adj",
}
NEW_CLASSES = ("StaticFFNBlock", "GraphContextBlock", "AttentionScorer", "AttentionPoolReadout")


def _source_segments() -> dict[str, str]:
    path = C.CODE_DIR / "nmp_model.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    segments: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in NEW_CLASSES:
            segments[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if isinstance(node, ast.ClassDef) and node.name == "NMPModel":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in ("propagate", "_graph_mean", "_atom_count"):
                    segments[f"NMPModel.{sub.name}"] = "\n".join(
                        lines[sub.lineno - 1 : sub.end_lineno]
                    )
    return segments


def _loaded_names(source: str) -> set[str]:
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def static_information_access() -> dict[str, Any]:
    segments = _source_segments()
    rows = {}
    for name, source in segments.items():
        loaded = _loaded_names(source)
        hits = sorted(loaded & FORBIDDEN_NEW_MODULE_NAMES)
        rows[name] = {
            "forbidden_tokens": hits,
            "passed": not hits,
            "reads": sorted(loaded & {"h", "x", "atom_batch", "n_graphs", "context", "scores"}),
        }
    # Explicit structural checks.
    text = "\n".join(segments.values())
    structure = {
        "S_no_other_atom_or_bond_access": not (set(segments["StaticFFNBlock"].split()) & {"atom_batch", "bond_index", "bond_feat"}),
        "G_uses_atom_batch_and_n_graphs_only": all(
            token in segments["NMPModel.propagate"] for token in ("atom_batch", "n_graphs")
        ),
        "A_per_molecule_segment_softmax": "atom_batch" in segments["AttentionPoolReadout"]
        and "scatter_reduce_" in segments["AttentionPoolReadout"],
        "no_full_atom_self_attention": "softmax(" not in text or "segment_softmax" in text,
        "no_atom_state_update_in_A": "h = h +" not in segments["AttentionPoolReadout"],
    }
    return {"segments": rows, "structural_checks": structure,
            "all_static_passed": bool(all(v["passed"] for v in rows.values()) and all(structure.values()))}


def build_architecture(task: Any) -> dict[str, Any]:
    ref0, info0, base0 = M.build_reference(task.spec, task.norm, C.N_TASKS, 0)
    rows = {}
    for arm in C.ARM_ORDER:
        model, audit = M.build_arm(
            arm, task.spec, task.norm, C.N_TASKS, seed=0, device="cpu",
            base_init_state=base0, zero_residual=True,
        )
        modules = {}
        for name, module in model.named_modules():
            if name.startswith(("blocks.", "readout.")):
                modules[name] = int(sum(p.numel() for p in module.parameters(recurse=False)))
        rows[arm] = {
            "mode": model.mode,
            "total_parameters": int(audit["total_parameter_count"]),
            "base_parameters": int(audit["base_parameter_elements"]),
            "new_parameters": int(audit["new_parameter_count"]),
            "new_parameter_tensors": int(audit["new_parameter_tensors"]),
            "new_parameter_names": list(audit["new_parameter_names"]),
            "own_parameter_breakdown": modules,
            "zero_residual_at_init": bool(audit["zero_residual_at_init"]),
            "relative_new_parameter_gap_to_V": float(
                abs(audit["new_parameter_count"] - C.HISTORICAL["V"]["new_parameters"])
                / C.HISTORICAL["V"]["new_parameters"]
            ),
        }
    return {
        "schema": "lcrc-pcba-no-mp-architecture-audit-v1",
        "round_id": C.ROUND_ID,
        "c1": {
            "parameters": int(info0["parameters"]),
            "state_tensors": len(base0),
            "variant": info0.get("variant"),
            "local_input_dim": int(info0.get("local_input_dim", -1)),
        },
        "historical_V": {
            "parameters": C.HISTORICAL["V"]["parameters"],
            "new_parameters": C.HISTORICAL["V"]["new_parameters"],
            "role": C.HISTORICAL["V"]["role"],
        },
        "expected_parameters": dict(C.EXPECTED_PARAMETERS),
        "arms": rows,
        "parameter_gap_tolerance": C.NEW_PARAMETER_TOLERANCE,
        "all_parameter_checks_passed": bool(
            all(v["relative_new_parameter_gap_to_V"] <= C.NEW_PARAMETER_TOLERANCE for v in rows.values())
            and all(v["total_parameters"] == C.EXPECTED_PARAMETERS[k] for k, v in rows.items())
        ),
        "flops_source": "torch.utils.flop_counter.FlopCounterMode on a fixed 4-molecule batch",
        "test_evaluated": False,
    }


def load_preflights() -> list[dict[str, Any]]:
    out = []
    for path in sorted(C.ANALYSIS.glob("preflight_gpu*.json")):
        try:
            out.append({"path": C.track_rel(path), **json.load(open(path, encoding="utf-8"))})
        except Exception:  # pragma: no cover - defensive
            continue
    return out


def merge_runtime(preflights: Iterable[dict[str, Any]]) -> dict[str, Any]:
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in C.ARM_ORDER}
    for pf in preflights:
        for arm, row in (pf.get("per_arm") or {}).items():
            per_arm.setdefault(arm, []).append({"preflight": pf.get("path"), **row})
    merged = {}
    for arm, rows in per_arm.items():
        if not rows:
            merged[arm] = {"available": False, "passed": False}
            continue
        checks = {}
        for row in rows:
            for name, check in (row.get("checks") or {}).items():
                checks.setdefault(name, []).append({"preflight": row.get("preflight"), **check})
        merged[arm] = {
            "available": True,
            "preflights": [r.get("preflight") for r in rows],
            "checks": checks,
            "all_passed": bool(all(r.get("all_passed") for r in rows)),
            "speed": rows[-1].get("speed"),
            "flops": rows[-1].get("flops"),
            "parameters": rows[-1].get("parameters"),
            "new_parameters": rows[-1].get("new_parameters"),
        }
    return merged


def build_information_access(preflights: list[dict[str, Any]]) -> dict[str, Any]:
    static = static_information_access()
    runtime = merge_runtime(preflights)
    expected_runtime = {
        "identity_eval",
        "bond_perturbation_invariance",
        "graph_isolation",
        "permutation_invariance",
        "gradient_flow",
        "parameters",
        "base_parity",
    }
    runtime_ok = {}
    for arm, row in runtime.items():
        names = set((row.get("checks") or {}).keys())
        runtime_ok[arm] = bool(
            row.get("all_passed")
            and expected_runtime.issubset(names)
            and (arm != "A" or "attention_softmax" in names)
        )
    return {
        "schema": "lcrc-pcba-no-mp-information-access-audit-v1",
        "round_id": C.ROUND_ID,
        "policy": {
            "added_modules_may_read": ["h", "atom_batch", "n_graphs"],
            "added_modules_may_not_read": sorted(FORBIDDEN_NEW_MODULE_NAMES),
            "S": "per-atom transform only; no other atom, no bond, no graph context",
            "A": "four graph-level attention pools; h is never updated; no atom-atom attention",
            "G": "per-molecule mean context broadcast; no bond messages",
        },
        "static": static,
        "runtime": runtime,
        "runtime_passed": runtime_ok,
        "preflight_files": [p.get("path") for p in preflights],
        "all_passed": bool(static["all_static_passed"] and all(runtime_ok.values())),
        "test_evaluated": False,
    }


def build_initialization(task: Any) -> dict[str, Any]:
    rows = {}
    for seed in (0, 1):
        ref, info, base = M.build_reference(task.spec, task.norm, C.N_TASKS, seed)
        plan = D.make_plan40(task, seed=seed)
        prefix20 = [p["row_order_sha256"] for p in plan[:20]]
        expected = C.HISTORICAL_PARITY[seed]
        rows[str(seed)] = {
            "base_init_fingerprint": C.state_fingerprint(base),
            "expected_base_init_fingerprint": expected["base_init_fingerprint"],
            "base_init_matches_historical_phase3": bool(
                C.state_fingerprint(base) == expected["base_init_fingerprint"]
            ),
            "global_plan_sha256": plan[0]["global_plan_sha256"],
            "expected_global_plan_sha256": expected["global_plan_sha256"],
            "plan_matches_historical_phase3": bool(
                plan[0]["global_plan_sha256"] == expected["global_plan_sha256"]
            ),
            "prefix20_sha256": C.sha256_bytes("|".join(prefix20).encode()),
            "consumed40_prefix_sha256": C.sha256_bytes(
                "|".join(p["row_order_sha256"] for p in plan).encode()
            ),
            "expected_consumed40_prefix_sha256": expected["consumed40_prefix_sha256"],
            "prefix40_matches_historical": bool(
                C.sha256_bytes("|".join(p["row_order_sha256"] for p in plan).encode())
                == expected["consumed40_prefix_sha256"]
            ),
            "n_train_rows": int(task.train_rows.size),
            "n_valid_rows": int(task.valid_rows.size),
            "n_test_rows": int(task.test_rows.size),
            "test_labels_all_nan": bool(not np.isfinite(task.labels[task.test_rows]).any()),
            "feature_spec_hash": task.spec.hash(),
            "norm_fingerprint": task.norm["norm_fingerprint"],
            "split_partition_sha256": task.split_partition_sha256,
            "label_artifact_sha256": task.label_artifact_sha256,
        }
    old_run = json.load(open(
        C.SG_SPRINT / "runs/phase3_seed0_C/run.json", encoding="utf-8"
    )) if (C.SG_SPRINT / "runs/phase3_seed0_C/run.json").exists() else None
    old_recipe = None if old_run is None else old_run.get("recipe", {})
    new_recipe = C.recipe_for_arm("S", "D")
    keys = (
        "optimizer", "adam_betas", "adam_eps", "weight_decay", "learning_rate",
        "warmup_epochs", "warmup_start_lr", "end_lr", "dropout", "batch_size",
        "gradient_clip", "precision", "epochs", "early_stopping", "loss",
    )
    recipe_cmp = {
        key: {
            "previous_phase3": None if old_recipe is None else old_recipe.get(key),
            "this_round": new_recipe.get(key),
            "equal": None
            if old_recipe is None
            else (
                list(old_recipe.get(key)) == list(new_recipe.get(key))
                if isinstance(old_recipe.get(key), (list, tuple))
                and isinstance(new_recipe.get(key), (list, tuple))
                else old_recipe.get(key) == new_recipe.get(key)
            ),
        }
        for key in keys
    }
    old_lr = None if old_recipe is None else (old_recipe.get("lr_schedule") or {}).get("phase3")
    new_lr = C.lr_audit()
    lr_numeric_equal = None
    if old_lr is not None:
        lr_numeric_equal = all(
            abs(float(old_lr[key]) - float(new_lr[key])) <= 1e-15
            for key in ("step0", "warmup_last", "post_warmup_first", "step40e")
        )
    return {
        "schema": "lcrc-pcba-no-mp-initialization-parity-v1",
        "round_id": C.ROUND_ID,
        "source_phase3_run": None if old_run is None else C.track_rel(
            C.SG_SPRINT / "runs/phase3_seed0_C/run.json"
        ),
        "seeds": rows,
        "lr_audit": C.lr_audit(),
        "previous_lr_audit": None if old_recipe is None else old_recipe.get("lr_schedule"),
        "lr_numeric_equal": lr_numeric_equal,
        "recipe_comparison": recipe_cmp,
        "recipe_equivalent_to_previous_phase3": bool(
            old_recipe is not None
            and all(v["equal"] is True for v in recipe_cmp.values())
            and lr_numeric_equal is True
        ),
        "historical_references": C.HISTORICAL,
        "all_passed": bool(
            all(v["base_init_matches_historical_phase3"] for v in rows.values())
            and all(v["plan_matches_historical_phase3"] for v in rows.values())
            and all(v["prefix40_matches_historical"] for v in rows.values())
            and all(v["test_labels_all_nan"] for v in rows.values())
            and old_recipe is not None
            and all(v["equal"] is True for v in recipe_cmp.values())
            and lr_numeric_equal is True
        ),
        "test_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-preflight", action="store_true")
    args = parser.parse_args(argv)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    task = D.load_task()
    preflights = load_preflights()
    architecture = build_architecture(task)
    info = build_information_access(preflights)
    init = build_initialization(task)
    arch_path = C.ANALYSIS / "architecture_audit.json"
    info_path = C.ANALYSIS / "information_access_audit.json"
    init_path = C.ANALYSIS / "initialization_parity.json"
    C.write_json(arch_path, architecture)
    C.write_json(info_path, info)
    C.write_json(init_path, init)
    C.log(f"wrote {C.track_rel(arch_path)} {C.track_rel(info_path)} {C.track_rel(init_path)}")
    ok = bool(architecture["all_parameter_checks_passed"] and init["all_passed"])
    if args.require_preflight:
        ok = ok and info["all_passed"]
    if not ok:
        C.log("AUDIT GATES FAILED")
        return 1
    C.log("audit gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
