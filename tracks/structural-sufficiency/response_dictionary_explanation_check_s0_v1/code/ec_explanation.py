"""Explanation acceptance for the frozen K=16 response dictionaries (Lipo).

Read-only on the 823-pair public122 edit catalogue.  The object of the analysis
is the **full** frozen prediction change of two real molecules, not a single
fragment score:

    delta_full = F_D(G_B) - F_D(G_A)
    delta_local = s_D(C,S_B) - s_D(C,S_A)
    remainder   = delta_full - delta_local

with a common context ``C`` selected only by exact structure (same canonical
fragment identity and cut-bond attributes in the existing fixed <=4 views;
largest C, tie smallest key).  No pair is selected by model behaviour and no
new split or approximate core is invented.

The word account, for the actual ``S_A`` reference and ``alpha(S_B)`` of a given
dictionary, is exact:

    contribution_k = alpha_k(S_B) [ s_D(C,D_k) - s_D(C,S_A) ]
    sum_k contribution_k = delta_local
    delta_full = sum_k contribution_k + remainder

Closure is checked numerically; it is not evidence of causal correctness.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import ec_common as EC  # noqa: E402
import ec_structures as ES  # noqa: E402
import rd_common as RC  # noqa: E402
import rd_dictionary as RD  # noqa: E402
import rd_dict_run as RDR  # noqa: E402
import fu_edits as FE  # noqa: E402

SPLIT_ORDER = ("train", "valid", "test")


def _cell_name(sa: str, sb: str) -> str:
    ordered = sorted((str(sa), str(sb)), key=lambda s: SPLIT_ORDER.index(s))
    return "-".join(s.capitalize() for s in ordered)


def _component_map(pairs) -> dict[int, int]:
    stub = [{"row_lo": min(p["a"], p["b"]), "row_hi": max(p["a"], p["b"])} for p in pairs]
    return FE._components(stub)


def load_pairs(ctx: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from rdkit import Chem

    payload = RC.dataset_cache(EC.load_config()["dataset"])
    labels = np.asarray(payload["labels"], dtype=np.float64).reshape(-1)
    smiles = [str(s) for s in payload["smiles"]]
    official = np.empty(len(smiles), dtype=object)
    for split in SPLIT_ORDER:
        official[np.asarray(payload[f"{split}_idx"], dtype=np.int64)] = split
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    if len(set(canon)) != len(canon):
        raise SystemExit("canonical SMILES are not unique across the official rows")
    row_of = {c: i for i, c in enumerate(canon)}

    path = EC.REPO_ROOT / EC.load_config()["explanation"]["pairs_csv"]
    source = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    problems: list[str] = []
    stored_row_mismatch = 0
    stored_split_mismatch = 0
    pairs: list[dict[str, Any]] = []
    seen: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    self_pairs = 0
    for r in source:
        a = row_of.get(Chem.MolToSmiles(Chem.MolFromSmiles(str(r["smiles_a"]))))
        b = row_of.get(Chem.MolToSmiles(Chem.MolFromSmiles(str(r["smiles_b"]))))
        if a is None or b is None:
            problems.append(f"{r['pair_id']}: canonical SMILES not in the official rows")
            continue
        if (int(r["row_a"]), int(r["row_b"])) != (a, b):
            stored_row_mismatch += 1
        if (r["split_a"], r["split_b"]) != (official[a], official[b]):
            stored_split_mismatch += 1
        if a == b:
            self_pairs += 1
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            duplicates += 1
            continue
        entry = {
            "pair_id": str(r["pair_id"]),
            "a": int(a), "b": int(b),
            "row_lo": int(key[0]), "row_hi": int(key[1]),
            "split_a": str(official[a]), "split_b": str(official[b]),
            "cell": _cell_name(official[a], official[b]),
            "y_a": float(labels[a]), "y_b": float(labels[b]),
            "delta_y": float(labels[b] - labels[a]),
            "class_id": str(r.get("class_id", "")),
            "core_smiles": str(r.get("core_smiles", "")),
            "core_heavy": int(r.get("core_heavy", 0) or 0),
            "attach_a": str(r.get("attach_a", "")),
            "attach_b": str(r.get("attach_b", "")),
            "smiles_a": str(r["smiles_a"]), "smiles_b": str(r["smiles_b"]),
            "official_smiles_a": smiles[a], "official_smiles_b": smiles[b],
        }
        seen[key] = entry
        pairs.append(entry)

    comp = _component_map(pairs)
    for p in pairs:
        p["component_id"] = int(comp[p["row_lo"]])
    # deterministic order by pair_id for coverage/cases
    pairs.sort(key=lambda p: p["pair_id"])

    cells = []
    for cell in ("Train-Train", "Train-Valid", "Train-Test"):
        sub = [p for p in pairs if p["cell"] == cell]
        endpoints = set()
        for p in sub:
            endpoints.update((p["a"], p["b"]))
        non_train = [e for e in endpoints if official[e] != "train"]
        cells.append({
            "cell": cell,
            "n_pairs": len(sub),
            "n_endpoints": len(endpoints),
            "n_components": len({p["component_id"] for p in sub}),
            "n_non_train_endpoints": len(non_train),
            "non_train_endpoints_by_split": {
                s: int(sum(1 for e in non_train if official[e] == s))
                for s in ("valid", "test")
            },
        })
    # no pair with both endpoints non-Train should exist
    expected = {"Train-Train": 791, "Train-Valid": 5, "Train-Test": 27}
    counts = {c["cell"]: c["n_pairs"] for c in cells}
    report = {
        "catalogue": str(path.relative_to(EC.REPO_ROOT)),
        "catalogue_sha256": RC.sha256_file(path),
        "source_rows": len(source),
        "mapped_pairs": len(pairs),
        "unmapped": len(problems),
        "problems": problems[:10],
        "stored_row_columns_agree_with_canonical_mapping": stored_row_mismatch == 0,
        "rows_whose_stored_split_column_disagrees_with_official": stored_split_mismatch,
        "duplicate_pairs": duplicates,
        "self_pairs": self_pairs,
        "n_endpoints": len({p["a"] for p in pairs} | {p["b"] for p in pairs}),
        "n_components": len({p["component_id"] for p in pairs}),
        "cells": cells,
        "cell_counts_reproduced": counts == expected,
        "expected_cell_counts": expected,
        "both_non_train_pairs": int(sum(1 for p in pairs
                                        if p["split_a"] != "train" and p["split_b"] != "train")),
        "orientation": "catalogue row_a -> row_b; delta = B - A",
        "note": ("The catalogue's split_a/split_b columns are not used; the official split is "
                 "re-derived from the mapped OGB row id. 823 pairs / 704 endpoints / 231 "
                 "components is the follow-up support, reused read-only."),
    }
    # compare with the follow-up support report if present
    fu_path = EC.FU_DIR / "analysis" / "edit_support.json"
    if fu_path.is_file():
        fu = RC.read_json(fu_path)
        report["followup_cells"] = fu.get("cells", [])
        report["followup_pair_count"] = fu.get("mapping", {}).get("mapped_pairs")
        report["followup_component_count"] = fu.get("mapping", {}).get("n_components_all_pairs")
    return pairs, report


def _views_by_row(ctx: Mapping[str, Any], payload) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for split in SPLIT_ORDER:
        s = ctx["dataset"].splits[split]
        vs = np.asarray(payload[f"{split}_view_s"], dtype=np.int64)
        vc = np.asarray(payload[f"{split}_view_c"], dtype=np.int64)
        vb = np.asarray(payload[f"{split}_view_bond_feat"], dtype=np.int64).reshape(-1, 3)
        mol = np.asarray(s.mol_ptr, dtype=np.int64)
        for local, row in enumerate(np.asarray(s.row_id, dtype=np.int64).tolist()):
            rows = []
            for v in range(int(mol[local]), int(mol[local + 1])):
                rows.append({"split": split, "view": int(v),
                             "s": int(vs[v]), "c": int(vc[v]),
                             "bond": tuple(int(x) for x in vb[v])})
            out[int(row)] = rows
    return out


def select_common_context(pairs: Sequence[dict[str, Any]], views: Mapping[int, list],
                          n_atoms: np.ndarray, keys: Sequence[str],
                          support: Mapping[int, bool],
                          word_mol_freq: Mapping[int, int]) -> None:
    for p in pairs:
        va = views.get(p["a"], [])
        vb = views.get(p["b"], [])
        cands = []
        for x in va:
            for y in vb:
                if x["c"] == y["c"] and x["bond"] == y["bond"]:
                    cands.append((int(n_atoms[x["c"]]), keys[x["c"]], x, y))
        if not cands:
            p["context_status"] = "NO_SHARED_SELECTED_CONTEXT"
            continue
        best = min(cands, key=lambda z: (-z[0], z[1]))
        p["C_atoms"], p["C_key"], p["view_a"], p["view_b"] = best[0], best[1], best[2], best[3]
        p["C"] = int(best[2]["c"])
        p["S_A"] = int(best[2]["s"])
        p["S_B"] = int(best[3]["s"])
        p["bond"] = tuple(best[2]["bond"])
        p["S_A_is_train_word"] = bool(p["S_A"] in word_mol_freq)
        p["S_B_is_train_word"] = bool(p["S_B"] in word_mol_freq)
        p["context_status"] = ("OK" if support.get(p["S_A"], False)
                               and support.get(p["S_B"], False)
                               else "SHARED_CONTEXT_UNSUPPORTED")
    # `same_tau` must be answered by tau identity, not support
    return


def run(ctx: Mapping[str, Any], proj: Mapping[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    config = ctx["config"]
    fragment_payload = RC.fragment_cache(config["dataset"])
    frozen, inventory = ctx["frozen"], ctx["inventory"]
    keys = inventory.keys
    n_atoms = np.asarray(fragment_payload["frag_n_atoms"], dtype=np.int64)

    pairs, mapping = load_pairs(ctx)
    views = _views_by_row(ctx, fragment_payload)
    support = proj["supports"]["D_RESPONSE"]
    select_common_context(pairs, views, n_atoms, keys, support,
                          ctx["inventory"].word_mol_freq)
    for p in pairs:
        if p.get("context_status") == "OK":
            p["same_tau"] = bool(inventory.tau[p["S_A"]] == inventory.tau[p["S_B"]])
            p["tau_A"] = inventory.tau[p["S_A"]]
            p["tau_B"] = inventory.tau[p["S_B"]]

    # ---- coverage -----------------------------------------------------------
    fallback = {
        int(row): len(views.get(int(row), [])) == 0
        for p in pairs for row in (p["a"], p["b"])
    }
    cells = ("Train-Train", "Train-Valid", "Train-Test")
    coverage_rows = []
    for cell in cells + ("ALL",):
        sub = pairs if cell == "ALL" else [p for p in pairs if p["cell"] == cell]
        interp = [p for p in sub if p.get("context_status") == "OK"]
        no_shared = [p for p in sub if p.get("context_status") == "NO_SHARED_SELECTED_CONTEXT"]
        both_fb = sum(1 for p in no_shared
                      if fallback[p["a"]] and fallback[p["b"]])
        one_fb = sum(1 for p in no_shared
                     if fallback[p["a"]] != fallback[p["b"]])
        both_viewed = len(no_shared) - both_fb - one_fb
        coverage_rows.append({
            "cell": cell,
            "n_pairs": len(sub),
            "n_components_all": len({p["component_id"] for p in sub}),
            "n_interpretable_pairs": len(interp),
            "n_interpretable_components": len({p["component_id"] for p in interp}),
            "fraction_interpretable": len(interp) / max(1, len(sub)),
            "n_no_shared_selected_context": len(no_shared),
            "n_no_shared_one_fallback_endpoint": int(one_fb),
            "n_no_shared_both_fallback_endpoints": int(both_fb),
            "n_no_shared_both_viewed": int(both_viewed),
            "n_shared_context_unsupported": int(sum(
                1 for p in sub if p.get("context_status") == "SHARED_CONTEXT_UNSUPPORTED")),
            "n_non_train_endpoints": len({p["a"] for p in sub if p["split_a"] != "train"}
                                          | {p["b"] for p in sub if p["split_b"] != "train"}),
            "n_non_train_pairs": int(sum(1 for p in sub
                                         if p["split_a"] != "train" or p["split_b"] != "train")),
        })
    # by endpoint split (A and B separately)
    endpoint_rows = []
    for side in ("a", "b"):
        for split in SPLIT_ORDER:
            sub = [p for p in pairs if p[f"split_{side}"] == split]
            if not sub:
                continue
            endpoint_rows.append({
                "side": side.upper(),
                "split": split,
                "n_pairs": len(sub),
                "n_interpretable_pairs": int(sum(1 for p in sub
                                                 if p.get("context_status") == "OK")),
                "n_components_all": len({p["component_id"] for p in sub}),
            })
    RC.write_csv(out / "explanation_coverage.csv", coverage_rows)
    RC.write_csv(out / "explanation_coverage_by_endpoint.csv", endpoint_rows)
    # list uncovered pairs (no model numbers; identity only)
    uncovered = [{
        "pair_id": p["pair_id"], "cell": p["cell"],
        "split_a": p["split_a"], "split_b": p["split_b"],
        "component_id": p["component_id"],
        "status": p.get("context_status", "UNKNOWN"),
        "class_id": p["class_id"],
    } for p in pairs if p.get("context_status") != "OK"]
    RC.write_csv(out / "explanation_uncovered_pairs.csv", uncovered)

    interp = [p for p in pairs if p.get("context_status") == "OK"]

    # ---- full predictions and local readouts --------------------------------
    versions = (("fw64", "FW64"), ("cert", "HP_CERTIFIED"), ("all", "HP_SLSQP_ALL"))
    lookups: dict[tuple[str, str], dict[int, float]] = {}
    resp_lookup = EC.raw_lookup(proj["resp"])
    for name in EC.DICT_NAMES:
        for version, _ in versions:
            lookups[(name, version)] = EC.raw_lookup(proj["predictions"][f"{name}|{version}"])
    residuals: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    s_ids = np.asarray([p["S_A"] for p in interp], dtype=np.int64)
    c_ids = np.asarray([p["C"] for p in interp], dtype=np.int64)
    sb_ids = np.asarray([p["S_B"] for p in interp], dtype=np.int64)
    s_a_resp = EC.view_scores(ctx, frozen.a, frozen.q, s_ids, c_ids) * ctx["std"] + ctx["mean"]
    s_b_resp = EC.view_scores(ctx, frozen.a, frozen.q, sb_ids, c_ids) * ctx["std"] + ctx["mean"]

    for name in EC.DICT_NAMES:
        spec = ctx["dicts16"][name]
        for version, label in versions:
            a_words, q_words = RDR._word_interfaces(
                frozen, inventory, spec, proj["alpha_versions"][name][version],
                proj["supports"][name])
            s_a = EC.view_scores(ctx, a_words, q_words, s_ids, c_ids) * ctx["std"] + ctx["mean"]
            s_b = EC.view_scores(ctx, a_words, q_words, sb_ids, c_ids) * ctx["std"] + ctx["mean"]
            for i, p in enumerate(interp):
                delta_local = float(s_b[i] - s_a[i])
                delta_full = float(lookups[(name, version)][p["b"]]
                                   - lookups[(name, version)][p["a"]])
                delta_resp = float(resp_lookup[p["b"]] - resp_lookup[p["a"]])
                delta_y = float(p["delta_y"])
                remainder = float(delta_full - delta_local)
                residuals.append({
                    "pair_id": p["pair_id"], "cell": p["cell"],
                    "component_id": p["component_id"],
                    "dictionary": name, "version": label,
                    "row_a": p["a"], "row_b": p["b"],
                    "split_a": p["split_a"], "split_b": p["split_b"],
                    "class_id": p["class_id"],
                    "C_fragment_id": p["C"], "C_atoms": int(p["C_atoms"]),
                    "C_key_prefix": p["C_key"][:16],
                    "S_A_fragment_id": p["S_A"], "S_B_fragment_id": p["S_B"],
                    "S_A_atoms": int(n_atoms[p["S_A"]]), "S_B_atoms": int(n_atoms[p["S_B"]]),
                    "S_A_train_word": bool(p["S_A_is_train_word"]),
                    "S_B_train_word": bool(p["S_B_is_train_word"]),
                    "same_tau": bool(p["same_tau"]),
                    "tau_A": RD.tau_label(inventory.tau[p["S_A"]]),
                    "tau_B": RD.tau_label(inventory.tau[p["S_B"]]),
                    "cut_bond": "/".join(str(x) for x in p["bond"]),
                    "delta_y": delta_y,
                    "delta_full": delta_full,
                    "delta_local": delta_local,
                    "remainder": remainder,
                    "delta_resp": delta_resp,
                    "delta_local_resp": float(s_b_resp[i] - s_a_resp[i]),
                    "remainder_resp": float(delta_resp - (s_b_resp[i] - s_a_resp[i])),
                    "error_full": float(delta_full - delta_y),
                    "error_local": float(delta_local - delta_y),
                    "error_resp": float(delta_resp - delta_y),
                    "error_full_model": float(delta_full - delta_resp),
                    "error_local_model": float(delta_local - (s_b_resp[i] - s_a_resp[i])),
                    "zero_change_abs_error": float(abs(delta_y)),
                })
                # exact word account for S_B
                tau_b = inventory.tau[p["S_B"]]
                protos = np.asarray(spec["prototypes"][tau_b], dtype=np.int64)
                alpha = np.asarray(proj["alpha_versions"][name][version][p["S_B"]],
                                   dtype=np.float64)
                s_proto = EC.view_scores(ctx, frozen.a, frozen.q, protos,
                                         np.full(protos.size, p["C"], dtype=np.int64))
                s_proto = s_proto * ctx["std"] + ctx["mean"]
                contrib = alpha * (s_proto - s_a[i])
                closure = float(np.sum(contrib) - delta_local)
                for k, fid in enumerate(protos.tolist()):
                    if abs(float(alpha[k])) <= 1e-12:
                        continue
                    accounting.append({
                        "pair_id": p["pair_id"], "cell": p["cell"],
                        "component_id": p["component_id"],
                        "dictionary": name, "version": label,
                        "tau_B": RD.tau_label(tau_b),
                        "prototype_rank": int(k),
                        "prototype_id": int(fid),
                        "prototype_key_prefix": str(keys[fid])[:16],
                        "prototype_atoms": int(n_atoms[fid]),
                        "prototype_train_freq": int(inventory.word_mol_freq.get(int(fid), 0)),
                        "is_d0": bool(int(fid) == int(spec["d0"][tau_b])),
                        "alpha": float(alpha[k]),
                        "s_D_C_prototype_raw": float(s_proto[k]),
                        "s_D_C_S_A_raw": float(s_a[i]),
                        "contribution_raw": float(contrib[k]),
                        "abs_contribution_raw": float(abs(contrib[k])),
                    })
                closures.append({
                    "pair_id": p["pair_id"], "cell": p["cell"],
                    "component_id": p["component_id"],
                    "dictionary": name, "version": label,
                    "n_terms": int(np.sum(np.abs(alpha) > 1e-12)),
                    "sum_contributions": float(np.sum(contrib)),
                    "delta_local": float(delta_local),
                    "closure_error": closure,
                })

    # ---- residual metrics ---------------------------------------------------
    metric_rows: list[dict[str, Any]] = []
    for name in EC.DICT_NAMES:
        for _, label in versions:
            for cell in cells + ("ALL",):
                sub = [r for r in residuals
                       if r["dictionary"] == name and r["version"] == label
                       and (cell == "ALL" or r["cell"] == cell)]
                if not sub:
                    continue
                met = _residual_metrics(sub, ctx["std"])
                metric_rows.append({"dictionary": name, "version": label,
                                    "cell": cell, **met})
    # ---- closure summary ----------------------------------------------------
    closure_max = max((abs(r["closure_error"]) for r in closures), default=0.0)
    closure_summary = {
        "n_equations": len(closures),
        "max_abs_closure_error": float(closure_max),
        "label": "WORD_ACCOUNT_CLOSES" if closure_max <= 1e-9 else "WORD_ACCOUNT_CLOSURE_FAILED",
        "note": ("sum_k contribution_k = delta_local by linearity of a(S), q(S), b(C) and the "
                 "bilinear term; this is an algebraic closure check, not a claim of explanatory "
                 "correctness."),
    }

    # ---- identity checks against the follow-up replay -----------------------
    ident = _identity_checks(residuals, mapping, pairs)

    # ---- write --------------------------------------------------------------
    RC.write_csv(out / "explanation_residuals.csv", residuals)
    RC.write_csv(out / "explanation_metrics.csv", metric_rows)
    RC.write_csv(out / "word_accounting.csv", accounting)
    RC.write_csv(out / "word_accounting_closure.csv", closures)
    RC.write_json(out / "explanation_mapping.json", mapping)
    RC.write_json(out / "explanation_checks.json", {
        "identity": ident,
        "closure": closure_summary,
        "coverage": coverage_rows,
        "support_note": ("Train-Train pairs are descriptions of the frozen function. The only "
                         "held-out supports are the 27 Train-Test and 5 Train-Valid pairs of the "
                         "follow-up, here restricted to the pairs with a shared selected context."),
    })
    summary = _explanation_summary(residuals, metric_rows, coverage_rows, closure_summary,
                                  pairs, interp, ctx)
    RC.write_json(out / "explanation_summary.json", summary)

    # ---- cases --------------------------------------------------------------
    cases = _select_cases(interp)
    case_data = _case_payload(ctx, proj, fragment_payload, cases, residuals, accounting,
                              closures, metric_rows, n_atoms, keys)
    _write_cases(out, case_data, ctx, summary, mapping)
    RC.write_csv(out / "cases.csv", [c["row"] for c in case_data])
    return {
        "pairs": pairs, "mapping": mapping, "coverage": coverage_rows,
        "residuals": residuals, "metrics": metric_rows, "accounting": accounting,
        "closures": closures, "closure_summary": closure_summary,
        "identity": ident, "summary": summary, "cases": case_data,
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"rmse": 0.0, "mae": 0.0, "median_abs": 0.0, "p90_abs": 0.0,
                "median": 0.0, "mean": 0.0, "max_abs": 0.0}
    a = np.abs(v)
    return {
        "rmse": float(np.sqrt(np.mean(v * v))),
        "mae": float(np.mean(a)),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "max_abs": float(np.max(a)),
    }


def _component_equal(values: np.ndarray, comps: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    comps = np.asarray(comps, dtype=np.int64)
    if vals.size == 0:
        return 0.0
    return float(np.mean([np.mean(np.abs(vals[comps == c]))
                          for c in sorted(set(comps.tolist()))]))


def _residual_metrics(sub, train_std: float) -> dict[str, Any]:
    comp = np.asarray([r["component_id"] for r in sub], dtype=np.int64)
    df = np.asarray([r["delta_full"] for r in sub])
    dl = np.asarray([r["delta_local"] for r in sub])
    rem = np.asarray([r["remainder"] for r in sub])
    dy = np.asarray([r["delta_y"] for r in sub])
    ef = np.asarray([r["error_full"] for r in sub])
    el = np.asarray([r["error_local"] for r in sub])
    er = np.asarray([r["error_resp"] for r in sub])
    efm = np.asarray([r["error_full_model"] for r in sub])
    elm = np.asarray([r["error_local_model"] for r in sub])
    rem_resp = np.asarray([r["remainder_resp"] for r in sub])
    zero = np.asarray([r["zero_change_abs_error"] for r in sub])
    sdf, sdl, srem = _stats(df), _stats(dl), _stats(rem)
    sef, sel, ser = _stats(ef), _stats(el), _stats(er)
    sefm, selm, srem_resp = _stats(efm), _stats(elm), _stats(rem_resp)
    return {
        "n_pairs": int(len(sub)),
        "n_components": int(len(set(comp.tolist()))),
        "low_support": bool(len(set(comp.tolist())) < 20),
        "delta_full_rmse": sdf["rmse"],
        "delta_full_mae": sdf["mae"],
        "delta_full_median_abs": sdf["median_abs"],
        "delta_full_p90_abs": sdf["p90_abs"],
        "delta_local_rmse": sdl["rmse"],
        "delta_local_mae": sdl["mae"],
        "delta_local_median_abs": sdl["median_abs"],
        "delta_local_p90_abs": sdl["p90_abs"],
        "remainder_rmse": srem["rmse"],
        "remainder_mae": srem["mae"],
        "remainder_median_abs": srem["median_abs"],
        "remainder_p90_abs": srem["p90_abs"],
        "remainder_max_abs": srem["max_abs"],
        "remainder_component_equal_mae": _component_equal(rem, comp),
        "error_full_rmse": sef["rmse"],
        "error_full_mae": sef["mae"],
        "error_full_component_equal_mae": _component_equal(ef, comp),
        "error_local_rmse": sel["rmse"],
        "error_local_mae": sel["mae"],
        "error_resp_rmse": ser["rmse"],
        "error_resp_mae": ser["mae"],
        "error_full_model_rmse": sefm["rmse"],
        "error_full_model_mae": sefm["mae"],
        "error_local_model_rmse": selm["rmse"],
        "error_local_model_mae": selm["mae"],
        "remainder_resp_rmse": srem_resp["rmse"],
        "remainder_resp_mae": srem_resp["mae"],
        "remainder_resp_median_abs": srem_resp["median_abs"],
        "remainder_resp_p90_abs": srem_resp["p90_abs"],
        "remainder_minus_resp_rmse": _stats(rem - rem_resp)["rmse"],
        "remainder_minus_resp_mae": _stats(rem - rem_resp)["mae"],
        "delta_y_mae": float(np.mean(np.abs(dy))),
        "delta_y_median_abs": float(np.median(np.abs(dy))),
        "zero_change_mae": float(np.mean(zero)),
        "train_label_std": float(train_std),
        "remainder_rmse_over_train_std": srem["rmse"] / train_std if train_std else 0.0,
        "remainder_median_abs_over_train_std": srem["median_abs"] / train_std if train_std else 0.0,
        "remainder_rmse_over_delta_full_rmse": (srem["rmse"] / sdf["rmse"]
                                                if sdf["rmse"] > 0 else 0.0),
        "remainder_median_abs_over_delta_full_median_abs": (
            srem["median_abs"] / sdf["median_abs"] if sdf["median_abs"] > 0 else 0.0),
    }


def _explanation_summary(residuals, metric_rows, coverage_rows, closure_summary,
                         pairs, interp, ctx) -> dict[str, Any]:
    table = {(r["dictionary"], r["version"], r["cell"]): r for r in metric_rows}
    cells = ("Train-Train", "Train-Valid", "Train-Test", "ALL")
    key_cells = {cell: table[("D_RESPONSE", "HP_CERTIFIED", cell)] for cell in cells}
    return {
        "n_pairs_total": len(pairs),
        "n_pairs_interpretable": len(interp),
        "fraction_interpretable": len(interp) / max(1, len(pairs)),
        "coverage": coverage_rows,
        "closure": closure_summary,
        "D_RESPONSE_HP_CERTIFIED": key_cells,
        "all_metrics": metric_rows,
        "labels": {
            "WORD_ACCOUNT_CLOSES": closure_summary["label"] == "WORD_ACCOUNT_CLOSES",
            "WORD_ACCOUNT_LOCAL_ONLY": True,
            "HELDOUT_EDIT_SUPPORT_LOW": True,
            "TASK_LAW_NOT_ESTABLISHED": True,
        },
        "notes": [
            "remainder is the part of the full F change not represented by the one selected "
            "common-context view; it includes other views and any change in the view set.",
            "Train-Train pairs describe the frozen function only. Train-Test/Train-Valid numbers "
            "are LOW_SUPPORT (7 Test molecules / 5 Train-Test components; 4 Valid molecules / "
            "3 Train-Valid components among the interpretable pairs).",
            "No relative ratio is used where delta_full is near zero; absolute errors are reported.",
            "Closure of the word account is algebraic and does not establish a chemical causal law.",
        ],
        "test_use": ctx["config"]["test_use"],
    }


# ---------------------------------------------------------------------------
# identity checks against the follow-up replay
# ---------------------------------------------------------------------------


def _identity_checks(residuals, mapping, pairs) -> dict[str, Any]:
    out: dict[str, Any] = {"cell_counts_reproduced": mapping["cell_counts_reproduced"]}
    # compare |error_resp| per pair with the follow-up's saved full-F RESP replay
    path = EC.FU_DIR / "analysis" / "edit_pairs.csv"
    if not path.is_file():
        out["followup_edit_pairs_present"] = False
        return out
    fu = {str(r["pair_id"]): r for r in csv.DictReader(path.open(encoding="utf-8"))}
    ours = {}
    for r in residuals:
        if r["dictionary"] == "D_RESPONSE" and r["version"] == "FW64":
            ours[r["pair_id"]] = r
    diffs = []
    missing = []
    for pid, r in ours.items():
        if pid not in fu:
            missing.append(pid)
            continue
        # follow-up orientation is row_lo -> row_hi; absolute errors are invariant
        diffs.append(abs(abs(float(fu[pid]["error_RESP"])) - abs(float(r["error_resp"]))))
    out.update({
        "followup_edit_pairs_present": True,
        "n_compared": len(diffs),
        "n_missing": len(missing),
        "max_abs_diff_of_abs_error_RESP": float(max(diffs)) if diffs else 0.0,
        "mean_abs_diff_of_abs_error_RESP": float(np.mean(diffs)) if diffs else 0.0,
        "note": ("The follow-up stores the native RESP delta on row_lo -> row_hi; this round "
                 "orients A=row_a, B=row_b. Only absolute errors are compared. Any mismatch is "
                 "reported, not forced to agree."),
    })
    return out


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def _select_cases(interp) -> list[dict[str, Any]]:
    non_train = [p for p in interp if p["split_a"] != "train" or p["split_b"] != "train"]
    train = [p for p in interp if p["split_a"] == "train" and p["split_b"] == "train"]
    selected = sorted(non_train, key=lambda p: p["pair_id"])[:6]
    selected += sorted(train, key=lambda p: p["pair_id"])[:6]
    if len(selected) < 12:
        chosen = {p["pair_id"] for p in selected}
        for p in sorted(train, key=lambda p: p["pair_id"]):
            if p["pair_id"] not in chosen:
                selected.append(p)
                if len(selected) >= 12:
                    break
    return selected[:12]


def _case_payload(ctx, proj, fragment_payload, cases, residuals, accounting, closures,
                  metric_rows, n_atoms, keys) -> list[dict[str, Any]]:
    by_key = {(r["pair_id"], r["dictionary"], r["version"]): r for r in residuals}
    acc = defaultdict(list)
    for r in accounting:
        if r["dictionary"] == "D_RESPONSE" and r["version"] == "HP_CERTIFIED":
            acc[r["pair_id"]].append(r)
    acc_fw = defaultdict(list)
    for r in accounting:
        if r["dictionary"] == "D_RESPONSE" and r["version"] == "FW64":
            acc_fw[r["pair_id"]].append(r)
    cl = {(r["pair_id"], r["dictionary"], r["version"]): r for r in closures}
    out = []
    smiles_cache: dict[int, str] = {}

    def fsmi(fid):
        fid = int(fid)
        if fid not in smiles_cache:
            smiles_cache[fid] = ES.fragment_smiles(fragment_payload, fid)
        return smiles_cache[fid]

    for idx, p in enumerate(cases, start=1):
        res = {v: by_key[(p["pair_id"], "D_RESPONSE", v)]
               for v in ("FW64", "HP_CERTIFIED", "HP_SLSQP_ALL")}
        others = {n: by_key[(p["pair_id"], n, "HP_CERTIFIED")] for n in EC.DICT_NAMES}
        alpha_fw = {r["prototype_id"]: r for r in acc_fw[p["pair_id"]]}
        alpha_hp = {r["prototype_id"]: r for r in acc[p["pair_id"]]}
        proto_ids = set(alpha_fw) | set(alpha_hp)
        top = sorted(proto_ids, key=lambda k: -max(
            abs(float(alpha_fw.get(k, {}).get("contribution_raw", 0.0))),
            abs(float(alpha_hp.get(k, {}).get("contribution_raw", 0.0)))))[:4]
        d0 = int(ctx["dicts16"]["D_RESPONSE"]["d0"][p["tau_B"]])
        if d0 not in top:
            top.append(d0)
        terms = []
        for fid in top:
            rec_h = alpha_hp.get(fid)
            rec_f = alpha_fw.get(fid)
            if rec_h is not None or rec_f is not None:
                s_proto = float((rec_h or rec_f)["s_D_C_prototype_raw"])
            else:
                s_proto = float(EC.view_scores(
                    ctx, ctx["frozen"].a, ctx["frozen"].q,
                    np.asarray([fid], dtype=np.int64),
                    np.asarray([p["C"]], dtype=np.int64))[0]
                    * ctx["std"] + ctx["mean"])
            terms.append({
                "prototype_id": int(fid),
                "prototype_key_prefix": str(keys[fid])[:16],
                "prototype_atoms": int(n_atoms[fid]),
                "prototype_smiles": fsmi(fid),
                "port": ES.port_info(fragment_payload, fid),
                "train_freq": int(ctx["inventory"].word_mol_freq.get(int(fid), 0)),
                "is_d0": bool(int(fid) == d0),
                "s_D_C_prototype_raw": s_proto,
                "alpha_FW64": float(rec_f["alpha"]) if rec_f else 0.0,
                "contrib_FW64": float(rec_f["contribution_raw"]) if rec_f else 0.0,
                "alpha_HP": float(rec_h["alpha"]) if rec_h else 0.0,
                "contrib_HP": float(rec_h["contribution_raw"]) if rec_h else 0.0,
            })
        # max alpha change for S_B across all of its prototypes
        alpha_change = 0.0
        alpha_change_all = 0.0
        all_protos = ctx["dicts16"]["D_RESPONSE"]["prototypes"][p["tau_B"]]
        alpha_all = proj["alpha_versions"]["D_RESPONSE"]["all"]
        for k, fid in enumerate(all_protos):
            a_f = float(alpha_fw.get(int(fid), {}).get("alpha", 0.0))
            a_h = float(alpha_hp.get(int(fid), {}).get("alpha", 0.0))
            a_a = float(alpha_all.get(int(p["S_B"]), np.zeros(len(all_protos)))[k])
            alpha_change = max(alpha_change, abs(a_f - a_h))
            alpha_change_all = max(alpha_change_all, abs(a_f - a_a))
        row = {
            "case": idx, "pair_id": p["pair_id"], "cell": p["cell"],
            "component_id": p["component_id"],
            "split_a": p["split_a"], "split_b": p["split_b"],
            "class_id": p["class_id"],
            "row_a": p["a"], "row_b": p["b"],
            "smiles_a": p["smiles_a"], "smiles_b": p["smiles_b"],
            "y_a": p["y_a"], "y_b": p["y_b"], "delta_y": p["delta_y"],
            "C_fragment_id": p["C"], "C_atoms": int(p["C_atoms"]),
            "C_key_prefix": p["C_key"][:16], "C_smiles": fsmi(p["C"]),
            "cut_bond": "/".join(str(x) for x in p["bond"]),
            "S_A_fragment_id": p["S_A"], "S_A_atoms": int(n_atoms[p["S_A"]]),
            "S_A_smiles": fsmi(p["S_A"]), "S_A_train_word": bool(p["S_A_is_train_word"]),
            "S_A_is_prototype": bool(p["S_A"] in ctx["dicts16"]["D_RESPONSE"]["prototypes"][p["tau_A"]]),
            "S_B_fragment_id": p["S_B"], "S_B_atoms": int(n_atoms[p["S_B"]]),
            "S_B_smiles": fsmi(p["S_B"]), "S_B_train_word": bool(p["S_B_is_train_word"]),
            "S_B_is_prototype": bool(p["S_B"] in ctx["dicts16"]["D_RESPONSE"]["prototypes"][p["tau_B"]]),
            "same_tau": bool(p["same_tau"]),
            "tau_A": RD.tau_label(p["tau_A"]), "tau_B": RD.tau_label(p["tau_B"]),
            "core_smiles": p["core_smiles"], "core_heavy": p["core_heavy"],
            "attach_a": p["attach_a"], "attach_b": p["attach_b"],
            "delta_full_FW64": res["FW64"]["delta_full"],
            "delta_local_FW64": res["FW64"]["delta_local"],
            "remainder_FW64": res["FW64"]["remainder"],
            "delta_full_HP_CERTIFIED": res["HP_CERTIFIED"]["delta_full"],
            "delta_local_HP_CERTIFIED": res["HP_CERTIFIED"]["delta_local"],
            "remainder_HP_CERTIFIED": res["HP_CERTIFIED"]["remainder"],
            "delta_full_HP_SLSQP_ALL": res["HP_SLSQP_ALL"]["delta_full"],
            "delta_local_HP_SLSQP_ALL": res["HP_SLSQP_ALL"]["delta_local"],
            "remainder_HP_SLSQP_ALL": res["HP_SLSQP_ALL"]["remainder"],
            "delta_resp": res["FW64"]["delta_resp"],
            "delta_local_resp": res["FW64"]["delta_local_resp"],
            "remainder_resp": res["FW64"]["remainder_resp"],
            "error_full_model_FW64": res["FW64"]["error_full_model"],
            "error_full_model_HP_CERTIFIED": res["HP_CERTIFIED"]["error_full_model"],
            "error_local_model_FW64": res["FW64"]["error_local_model"],
            "error_local_model_HP_CERTIFIED": res["HP_CERTIFIED"]["error_local_model"],
            "error_full_FW64": res["FW64"]["error_full"],
            "error_full_HP_CERTIFIED": res["HP_CERTIFIED"]["error_full"],
            "error_full_HP_SLSQP_ALL": res["HP_SLSQP_ALL"]["error_full"],
            "error_local_FW64": res["FW64"]["error_local"],
            "error_local_HP_CERTIFIED": res["HP_CERTIFIED"]["error_local"],
            "error_local_HP_SLSQP_ALL": res["HP_SLSQP_ALL"]["error_local"],
            "error_resp": res["FW64"]["error_resp"],
            "alpha_max_change_S_B": float(alpha_change),
            "alpha_max_change_S_B_HP_SLSQP_ALL": float(alpha_change_all),
            "delta_local_change_HP_minus_FW": float(res["HP_CERTIFIED"]["delta_local"]
                                                    - res["FW64"]["delta_local"]),
            "delta_full_change_HP_minus_FW": float(res["HP_CERTIFIED"]["delta_full"]
                                                   - res["FW64"]["delta_full"]),
            "delta_local_change_SLSQP_ALL_minus_FW": float(
                res["HP_SLSQP_ALL"]["delta_local"] - res["FW64"]["delta_local"]),
            "delta_full_change_SLSQP_ALL_minus_FW": float(
                res["HP_SLSQP_ALL"]["delta_full"] - res["FW64"]["delta_full"]),
            "closure_error_HP": float(cl[(p["pair_id"], "D_RESPONSE", "HP_CERTIFIED")]["closure_error"]),
            "closure_error_FW": float(cl[(p["pair_id"], "D_RESPONSE", "FW64")]["closure_error"]),
            "delta_full_D_LATENT_HP": others["D_LATENT"]["delta_full"],
            "remainder_D_LATENT_HP": others["D_LATENT"]["remainder"],
            "delta_full_D_FREQ_HP": others["D_FREQ"]["delta_full"],
            "remainder_D_FREQ_HP": others["D_FREQ"]["remainder"],
        }
        out.append({"row": row, "terms": terms, "pair": p, "res": res})
    return out


def _fmt(x, nd=4):
    return f"{float(x):+.{nd}f}"


def _write_cases(out: Path, cases, ctx, summary, mapping) -> None:
    lines = []
    lines.append("# Explanation check — auditable cases (max 12)\n")
    lines.append("Selection rule (fixed before results): the first 6 interpretable pairs with at "
                 "least one non-Train endpoint, then the first 6 interpretable Train–Train pairs, "
                 "ordered by catalogue `pair_id`. Nothing is selected by effect size.\n")
    lines.append("`C` is the shared complement side of the selected existing view; `S_A`/`S_B` "
                 "are the small sides. `*` marks the cut bond. The word account uses "
                 "`D_RESPONSE` and the actual `S_A` as reference. `HP_CERTIFIED` is the primary "
                 "high-precision version; `HP_SLSQP_ALL` is the uncertified sensitivity that also "
                 "replaces rows where SLSQP missed the 1e-8 gap certificate.\n")
    all_cov = [p for p in summary["coverage"] if p["cell"] == "ALL"][0]
    lines.append(f"Interpretable pairs: **{all_cov['n_interpretable_pairs']}/823** "
                 f"({100.0*summary['fraction_interpretable']:.1f}%). "
                 f"Word-account closure max |error| = {summary['closure']['max_abs_closure_error']:.2e}. "
                 f"Mapping: {mapping['mapped_pairs']} pairs, {mapping['n_components']} components; "
                 f"cell counts reproduced = {mapping['cell_counts_reproduced']}.\n")
    lines.append("Train–Train cases describe the frozen function; they are not new-structure "
                 "generalisation. Held-out cases are low-support: the whole interpretable set has "
                 "5 Test molecules / 5 Train–Test components and 3 Valid molecules / 3 Train–Valid "
                 "components (the full 823-pair support is 7 Test / 7 components and 4 Valid / 4 "
                 "components; the shared-C restriction removes 2 Test and 1 Valid molecule). "
                 "Cases 1–6 use only 2 distinct Test molecules (rows 37 and 758) / 2 components, "
                 "so they are not 6 independent held-out experiments. Cases 7–12 use component "
                 "28 (5 cases) and component 53 (1 case); they are also clustered, not 6 "
                 "independent Train examples.\n")
    lines.append("## Summary table\n")
    lines.append("| # | pair | cell | component | Δy | Δlocal FW | Δfull FW | rem FW | "
                 "Δlocal HP | Δfull HP | rem HP |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        r = c["row"]
        lines.append(f"| {r['case']} | `{r['pair_id']}` | {r['cell']} | {r['component_id']} | "
                     f"{_fmt(r['delta_y'])} | {_fmt(r['delta_local_FW64'])} | "
                     f"{_fmt(r['delta_full_FW64'])} | {_fmt(r['remainder_FW64'])} | "
                     f"{_fmt(r['delta_local_HP_CERTIFIED'])} | "
                     f"{_fmt(r['delta_full_HP_CERTIFIED'])} | {_fmt(r['remainder_HP_CERTIFIED'])} |")
    lines.append("")
    for c in cases:
        r = c["row"]
        lines.append(f"## Case {r['case']} — `{r['pair_id']}` ({r['cell']}, component {r['component_id']})\n")
        lines.append(f"**Molecules.** `A` ({r['split_a']}, row {r['row_a']}): `{r['smiles_a']}`  ")
        lines.append(f"`B` ({r['split_b']}, row {r['row_b']}): `{r['smiles_b']}`  ")
        lines.append(f"Catalogue edit: `{r['attach_a']}` → `{r['attach_b']}` on core "
                     f"`{r['core_smiles']}` ({r['core_heavy']} heavy atoms); class `{r['class_id']}`.\n")
        lines.append(f"**Common context.** `C` = fragment {r['C_fragment_id']} "
                     f"({r['C_atoms']} atoms, key `{r['C_key_prefix']}…`), cut bond "
                     f"`{r['cut_bond']}`. `S_A` = {r['S_A_fragment_id']} "
                     f"(`{r['S_A_smiles']}`, {r['S_A_atoms']} atoms, Train word={r['S_A_train_word']}, "
                     f"prototype={r['S_A_is_prototype']}); `S_B` = {r['S_B_fragment_id']} "
                     f"(`{r['S_B_smiles']}`, {r['S_B_atoms']} atoms, Train word={r['S_B_train_word']}, "
                     f"prototype={r['S_B_is_prototype']}); same τ = {r['same_tau']}.\n")
        lines.append(f"**Word account (`D_RESPONSE`).** Main prototypes of τ(S_B) "
                     f"`{r['tau_B']}`; `s_D(C,·)` is the frozen readout, `contribution_k = "
                     f"α_k(S_B)[s_D(C,D_k)-s_D(C,S_A)]`.\n")
        lines.append("| prototype | structure | port | α FW64 | contrib FW64 | α HP | contrib HP |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for t in c["terms"]:
            tag = " (D0)" if t["is_d0"] else ""
            lines.append(f"| {t['prototype_id']}{tag} | `{t['prototype_smiles']}` | "
                         f"{t['port']['port_element']} | {t['alpha_FW64']:.4f} | "
                         f"{_fmt(t['contrib_FW64'])} | {t['alpha_HP']:.4f} | "
                         f"{_fmt(t['contrib_HP'])} |")
        lines.append("")
        lines.append(f"**Numbers.** Δy = {_fmt(r['delta_y'])}; "
                     f"Δresp (native RESP full F) = {_fmt(r['delta_resp'])}; "
                     f"error of native Δresp = {_fmt(r['error_resp'])}; "
                     f"native one-view Δlocal = {_fmt(r['delta_local_resp'])}; "
                     f"native remainder = {_fmt(r['remainder_resp'])}. "
                     f"Dictionary vs native model: Δfull error = "
                     f"{_fmt(r['error_full_model_HP_CERTIFIED'])} (HP) / "
                     f"{_fmt(r['error_full_model_FW64'])} (FW); Δlocal error = "
                     f"{_fmt(r['error_local_model_HP_CERTIFIED'])} (HP) / "
                     f"{_fmt(r['error_local_model_FW64'])} (FW).\n")
        lines.append("| version | Δlocal | remainder | Δfull | error Δfull vs Δy | error Δlocal vs Δy |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for label in ("FW64", "HP_CERTIFIED", "HP_SLSQP_ALL"):
            lines.append(f"| {label} | {_fmt(r[f'delta_local_{label}'])} | "
                         f"{_fmt(r[f'remainder_{label}'])} | {_fmt(r[f'delta_full_{label}'])} | "
                         f"{_fmt(r[f'error_full_{label}'])} | "
                         f"{_fmt(r[f'error_local_{label}'])} |")
        lines.append("")
        if r["S_B_is_prototype"]:
            hp_note = ("`S_B` is a stored prototype, so `alpha(S_B)` is one-hot by definition; "
                       "no high-precision version can change this word account. ")
        else:
            hp_note = ("`S_B` is a projected (non-prototype) fragment. ")
        lines.append(f"High precision vs FW64: {hp_note}max |Δα(S_B)| = "
                     f"{r['alpha_max_change_S_B']:.3e} (HP_CERTIFIED) / "
                     f"{r['alpha_max_change_S_B_HP_SLSQP_ALL']:.3e} (HP_SLSQP_ALL); "
                     f"ΔΔlocal = {_fmt(r['delta_local_change_HP_minus_FW'])} (HP) / "
                     f"{_fmt(r['delta_local_change_SLSQP_ALL_minus_FW'])} (all); "
                     f"ΔΔfull = {_fmt(r['delta_full_change_HP_minus_FW'])} (HP) / "
                     f"{_fmt(r['delta_full_change_SLSQP_ALL_minus_FW'])} (all); "
                     f"closure errors FW/HP = {r['closure_error_FW']:.2e}/"
                     f"{r['closure_error_HP']:.2e}. "
                     f"`HP_CERTIFIED` keeps FW64 on rows that miss the 1e-8 certificate; "
                     f"`HP_SLSQP_ALL` is the uncertified all-rows sensitivity. "
                     f"Other selectors (HP): D_LATENT Δfull = {_fmt(r['delta_full_D_LATENT_HP'])}, "
                     f"remainder = {_fmt(r['remainder_D_LATENT_HP'])}; "
                     f"D_FREQ Δfull = {_fmt(r['delta_full_D_FREQ_HP'])}, "
                     f"remainder = {_fmt(r['remainder_D_FREQ_HP'])}.\n")
        if r["split_a"] == "train" and r["split_b"] == "train":
            lines.append("**Scope.** Train–Train: describes the frozen function; not held-out "
                         "evidence and not a new-structure generalisation claim.\n")
        else:
            lines.append("**Scope.** Has a non-Train endpoint: held-out support is LOW "
                         "(see the report); this is not a task-law validation.\n")
    (out / "CASES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
