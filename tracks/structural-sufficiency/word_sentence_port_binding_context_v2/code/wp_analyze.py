"""Analysis, support statistics, finalist freezing and terminal Test scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import wp_common as C
import wp_data as D

MAIN_ARMS = ["MEAN_SELF", "BIND_SELF", "MEAN_REL", "BIND_REL"]
TIRE_ORDER = {"MEAN_SELF": 0, "MEAN_REL": 1, "BIND_SELF": 2, "BIND_REL": 3}


def dataset_name(alias: str) -> str:
    return {"lipo": "ogbg-mollipo", "esol": "ogbg-molesol"}.get(alias, alias)


# ---------------------------------------------------------------------------
# run collection
# ---------------------------------------------------------------------------


def collect_runs(dataset: str = "ogbg-mollipo") -> list[dict[str, Any]]:
    root = C.results_root()
    runs = []
    for path in sorted(root.glob("*/run.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("dataset") != dataset:
            continue
        if payload.get("status") not in ("COMPLETE", "PARTIAL"):
            continue
        payload["_path"] = str(path)
        runs.append(payload)
    return runs


def arm_runs(runs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    out: dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in runs:
        arm = str(run["arm"])
        if str(run.get("partition", "frozen")) != "frozen":
            arm = f"{arm}_{run['partition']}"
        key = (arm, int(run["seed"]))
        prev = out.get(key)
        if prev is None or (prev.get("status") == "PARTIAL" and run.get("status") == "COMPLETE"):
            out[key] = run
    return out


def threshold(dataset: str, name: str, cfg: Mapping[str, Any]) -> float:
    sigma = float(cfg["thresholds"]["train_sigma"][dataset])
    return float(cfg["thresholds"][name]) * sigma


def main_table(runs: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in sorted(runs, key=lambda r: (r["arm"], r["seed"])):
        best = run.get("best") or {}
        final = run.get("final") or {}
        last20 = run.get("last20") or {}
        curves = run.get("curves") or []
        epoch_seconds = [c.get("train_step_seconds") for c in curves if c.get("train_step_seconds")]
        rows.append({
            "arm": run["arm"], "seed": run["seed"], "status": run["status"],
            "partition": run.get("partition", "frozen"),
            "spec": json.dumps(run.get("spec"), sort_keys=True),
            "n_parameters": run.get("n_parameters"),
            "best_epoch": best.get("epoch"),
            "best_valid_rmse": best.get("valid_rmse_raw"),
            "best_train_rmse": best.get("train_rmse_raw"),
            "final_valid_rmse": final.get("valid_rmse_raw"),
            "final_train_rmse": final.get("train_rmse_raw"),
            "last20_train_rmse": last20.get("mean_last20_train_rmse_raw"),
            "last20_valid_rmse": last20.get("mean_last20_valid_rmse_raw"),
            "s_per_epoch_mean": float(np.mean(epoch_seconds)) if epoch_seconds else None,
            "s_per_epoch_max": float(np.max(epoch_seconds)) if epoch_seconds else None,
            "epochs_run": len(curves),
            "wall_seconds": run.get("wall_seconds"),
            "commit": run.get("commit"),
            "device": run.get("device_name"),
            "resumed_from_epoch": run.get("resumed_from_epoch"),
        })
    return rows


# ---------------------------------------------------------------------------
# condition differences
# ---------------------------------------------------------------------------


def condition_differences(index: Mapping[tuple[str, int], Mapping[str, Any]],
                          cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    out: dict[str, Any] = {"per_seed": {}, "thresholds": {}}
    for name in ("delta_mechanism", "delta_pipeline", "tie"):
        out["thresholds"][name] = threshold(dataset, name + "_sigma", cfg) if name != "tie" else \
            float(cfg["thresholds"]["tie_sigma"]) * float(cfg["thresholds"]["train_sigma"][dataset])
    deltas = {}
    for seed in (0, 1, 2):
        vals = {}
        for arm in MAIN_ARMS:
            run = index.get((arm, seed))
            if run is None or run.get("status") != "COMPLETE":
                continue
            vals[arm] = float(run["best"]["valid_rmse_raw"])
        if len(vals) < 4:
            out["per_seed"][str(seed)] = {"complete": False, "valid": vals}
            continue
        rec = {
            "complete": True, "valid": vals,
            "g_bind_self": vals["MEAN_SELF"] - vals["BIND_SELF"],
            "g_bind_rel": vals["MEAN_REL"] - vals["BIND_REL"],
            "g_ctx_mean": vals["MEAN_SELF"] - vals["MEAN_REL"],
            "g_ctx_bind": vals["BIND_SELF"] - vals["BIND_REL"],
        }
        rec["I"] = rec["g_ctx_bind"] - rec["g_ctx_mean"]
        out["per_seed"][str(seed)] = rec
        deltas[str(seed)] = rec
    names = ("g_bind_self", "g_bind_rel", "g_ctx_mean", "g_ctx_bind", "I")
    agg = {}
    for name in names:
        values = [deltas[s][name] for s in deltas]
        both_positive = bool(len(values) >= 2 and all(v > 0 for v in values[:2]))
        both_negative = bool(len(values) >= 2 and all(v < 0 for v in values[:2]))
        agg[name] = {
            "per_seed": {s: deltas[s][name] for s in deltas},
            "mean": float(np.mean(values)) if values else None,
            "both_seeds_positive": both_positive,
            "both_seeds_negative": both_negative,
            "reproducible_positive": bool(both_positive
                                         and np.mean(values) >= out["thresholds"]["delta_mechanism"]),
        }
    out["aggregate"] = agg
    out["mechanism_signal"] = any(v["reproducible_positive"] for k, v in agg.items() if k != "I")
    return out


def select_candidates(index: Mapping[tuple[str, int], Mapping[str, Any]], cfg: Mapping[str, Any],
                      dataset: str) -> dict[str, Any]:
    def mean_valid(arm: str) -> tuple[float | None, list[int]]:
        seeds = [s for s in (0, 1) if (arm, s) in index and index[(arm, s)]["status"] == "COMPLETE"]
        if len(seeds) < 2:
            return None, seeds
        return float(np.mean([index[(arm, s)]["best"]["valid_rmse_raw"] for s in seeds])), seeds

    def tie_key(arm: str) -> tuple:
        spec = next((index[(arm, s)].get("spec") or {} for s in (0, 1) if (arm, s) in index), {})
        return (
            0 if spec.get("port") == "mean" else 1,
            0 if spec.get("context") == "self" else 1,
            0 if not spec.get("bag") else 1,
            TIRE_ORDER.get(arm, 99),
            arm,
        )

    w_scores = {arm: mean_valid(arm) for arm in MAIN_ARMS}
    candidates = [(v[0], tie_key(arm), arm) for arm, v in w_scores.items() if v[0] is not None]
    w = min(candidates)[2] if candidates else None
    f_scores = {arm: mean_valid(arm) for arm in sorted({a for a, _ in index})}
    f_candidates = [(v[0], tie_key(arm), arm) for arm, v in f_scores.items() if v[0] is not None]
    f = min(f_candidates)[2] if f_candidates else None
    tie = float(cfg["thresholds"]["tie_sigma"]) * float(cfg["thresholds"]["train_sigma"][dataset])
    w_values = {arm: v[0] for arm, v in w_scores.items() if v[0] is not None}
    if w is not None:
        near = [arm for arm, value in w_values.items() if value - w_values[w] <= tie]
        w = min(near, key=tie_key)
    return {"W": w, "F": f, "mean_valid": w_values,
            "all_mean_valid": {arm: v[0] for arm, v in f_scores.items() if v[0] is not None},
            "seeds": {arm: (seeds := [s for s in (0, 1) if (arm, s) in index]) for arm in f_scores},
            "tie": tie}


def pick_core_pair(candidates: Mapping[str, Any], index, cfg) -> dict[str, Any]:
    """PROTOCOL §6: the core comparison C for the chosen W."""
    w = candidates["W"]
    means = candidates["mean_valid"]
    sign_ok = {}
    for arm in ("MEAN_SELF", "BIND_SELF", "MEAN_REL", "BIND_REL"):
        if arm not in means:
            continue
        diffs = [means[w] - means[arm]] if w in means else []
        sign_ok[arm] = bool(diffs and diffs[0] > 0)
    if w == "BIND_REL":
        choices = [("MEAN_REL", -1), ("BIND_SELF", -1)]
        scored = []
        for arm, _ in choices:
            if arm in means:
                gap = means[arm] - means[w]
                scored.append((gap, 0 if arm == "BIND_SELF" else 1, arm))
        if not scored:
            return {"C": None, "reason": "no comparable arm"}
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = scored[0]
        positive_both = all((index[(arm, s)]["best"]["valid_rmse_raw"]
                             - index[(w, s)]["best"]["valid_rmse_raw"]) > 0
                            for s in (0, 1)
                            if (arm, s) in index and (w, s) in index)
        return {"C": best[2], "gap": float(best[0]), "same_seed_positive_both": bool(positive_both),
                "rule": "max mean gap with both seeds positive; tie -> BIND_SELF"}
    if w == "MEAN_REL":
        return {"C": "MEAN_SELF", "rule": "W=MEAN_REL -> MEAN_SELF"}
    if w == "BIND_SELF":
        return {"C": "MEAN_SELF", "rule": "W=BIND_SELF -> MEAN_SELF"}
    return {"C": None, "rule": "W=MEAN_SELF or no positive core pair -> C=null"}


# ---------------------------------------------------------------------------
# scaffold bootstrap
# ---------------------------------------------------------------------------


def scaffold_groups(raw: Mapping[str, Any], row_ids: Sequence[int]) -> tuple[dict[int, int], dict[str, Any]]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    smiles = raw["smiles"]
    groups, info = {}, {"n_failed": 0, "n_scaffolds": 0}
    for row in row_ids:
        text = smiles[int(row)]
        if isinstance(text, bytes):
            text = text.decode()
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            info["n_failed"] += 1
            groups[int(row)] = -1 - int(row)
            continue
        groups[int(row)] = hash(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    info["n_scaffolds"] = len({v for v in groups.values() if v >= 0})
    info["level"] = "scaffold" if info["n_failed"] == 0 else "molecule_fallback_for_failures"
    return groups, info


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def write_summary(args) -> None:
    dataset = dataset_name(getattr(args, "dataset", "lipo"))
    cfg = C.load_config()
    runs = collect_runs(dataset)
    index = arm_runs(runs)
    rows = main_table(runs, cfg)
    diffs = condition_differences(index, cfg, dataset)
    candidates = select_candidates(index, cfg, dataset)
    core = pick_core_pair(candidates, index, cfg)
    summary = {
        "dataset": dataset, "commit": C.git_commit(), "config_sha256": C.sha256_json(cfg),
        "n_runs": len(runs),
        "runs": rows,
        "condition_differences": diffs,
        "candidates": candidates,
        "core_pair": core,
        "bootstrap": bootstrap_block(dataset, index, cfg),
        "historical": {
            "full_wgine_valid": cfg["thresholds"]["historical_full_wgine_valid"],
            "full_wgine_test": cfg["thresholds"]["historical_full_wgine_test"],
            "gine_valid": cfg["thresholds"]["historical_gine_valid"],
            "gine_test": cfg["thresholds"]["historical_gine_test"],
        },
    }
    suffix = "" if dataset == "ogbg-mollipo" else "_" + dataset.replace("ogbg-mol", "")
    C.write_json(C.analysis_dir() / f"summary{suffix}.json", C.jsonable(summary))
    columns = ["arm", "seed", "status", "partition", "best_epoch", "best_valid_rmse", "best_train_rmse",
               "final_valid_rmse", "final_train_rmse", "last20_train_rmse", "last20_valid_rmse",
               "n_parameters", "s_per_epoch_mean", "epochs_run", "wall_seconds", "device", "spec"]
    C.write_csv(C.analysis_dir() / f"main_table{suffix}.csv", rows, columns)
    print(json.dumps({"runs": len(runs), "W": candidates["W"], "F": candidates["F"],
                      "core_pair": core, "mechanism_signal": diffs["mechanism_signal"],
                      "aggregate": {k: {"mean": round(v["mean"], 5) if v["mean"] is not None else None,
                                        "rep": v["reproducible_positive"]}
                                    for k, v in diffs["aggregate"].items()}}, indent=2))


def contrast_specs() -> list[tuple[str, list[tuple[str, float]], list[tuple[str, float]]]]:
    """(name, coefficients over arm RMSEs, coefficients for the interaction part)."""
    return [
        ("MEAN_SELF-BIND_SELF", [("MEAN_SELF", 1.0), ("BIND_SELF", -1.0)], []),
        ("MEAN_REL-BIND_REL", [("MEAN_REL", 1.0), ("BIND_REL", -1.0)], []),
        ("MEAN_SELF-MEAN_REL", [("MEAN_SELF", 1.0), ("MEAN_REL", -1.0)], []),
        ("BIND_SELF-BIND_REL", [("BIND_SELF", 1.0), ("BIND_REL", -1.0)], []),
        ("I_ctx_bind_minus_ctx_mean",
         [("BIND_SELF", 1.0), ("BIND_REL", -1.0), ("MEAN_SELF", -1.0), ("MEAN_REL", 1.0)], []),
        ("MEAN_SELF-SET_MEAN_SELF", [("MEAN_SELF", 1.0), ("SET_MEAN_SELF", -1.0)], []),
        ("BIND_SELF-BIND_SELF_SCALAR", [("BIND_SELF", 1.0), ("BIND_SELF_SCALAR", -1.0)], []),
        ("BIND_REL-GINE_REF", [("BIND_REL", 1.0), ("GINE_REF", -1.0)], []),
        ("MEAN_SELF-GINE_REF", [("MEAN_SELF", 1.0), ("GINE_REF", -1.0)], []),
        ("BIND_REL-BAG_CAP", [("BIND_REL", 1.0), ("BAG_CAP", -1.0)], []),
        ("BIND_REL-GLOBAL_CTX_BIND", [("BIND_REL", 1.0), ("GLOBAL_CTX_BIND", -1.0)], []),
        ("BIND_REL-RANDOM_PART", [("BIND_REL", 1.0), ("BIND_REL_random", -1.0)], []),
        ("BIND_REL-GINE_RAW", [("BIND_REL", 1.0), ("GINE_RAW", -1.0)], []),
    ]


def _predictions(dataset: str) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray, np.ndarray,
                                        dict[str, Any]]:
    """Valid predictions per (arm, seed), the Valid targets and the row ids."""
    import wp_report as R

    raw = C.dataset_cache(dataset)
    row_ids = np.asarray(raw["valid_idx"], dtype=np.int64)
    order = np.argsort(row_ids)
    row_ids = row_ids[order]
    preds: dict[tuple[str, int], np.ndarray] = {}
    for run in collect_runs(dataset):
        if run.get("status") != "COMPLETE":
            continue
        path = Path(run["_path"]).parent / "preds_best.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        lookup = {int(r): float(p) for r, p in zip(data["valid_row"], data["valid_pred"])}
        if not all(int(r) in lookup for r in row_ids):
            continue
        arm = str(run["arm"])
        if str(run.get("partition", "frozen")) != "frozen":
            arm = f"{arm}_{run['partition']}"
        preds[(arm, int(run["seed"]))] = np.asarray([lookup[int(r)] for r in row_ids])
    y = np.asarray([float(raw["labels"][int(r)]) for r in row_ids])
    del R
    return preds, y, row_ids, {}


def bootstrap_block(dataset: str, index, cfg) -> dict[str, Any]:
    """Paired scaffold bootstrap over the fixed Valid predictions (all contrasts at once)."""
    preds, y, row_ids, _ = _predictions(dataset)
    if not preds:
        return {"available": False, "reason": "no saved predictions"}
    groups, info = scaffold_groups(C.dataset_cache(dataset), row_ids)
    by_group: dict[int, list[int]] = {}
    for position, row in enumerate(row_ids):
        by_group.setdefault(int(groups[int(row)]), []).append(position)
    keys = sorted(by_group)
    specs = contrast_specs()
    usable = {}
    for name, coefs, _ in specs:
        arms = [arm for arm, _ in coefs]
        seeds = [s for s in (0, 1) if all((arm, s) in preds for arm in arms)]
        if seeds:
            usable[name] = seeds
    n_boot = 4000
    draws: dict[str, list[float]] = {}
    point: dict[str, float] = {}
    for name in usable:
        coefs = dict(specs[[s[0] for s in specs].index(name)][1])
        seeds = usable[name]
        values = []
        for seed in seeds:
            values.append(sum(coef * float(np.sqrt(np.mean((preds[(arm, seed)] - y) ** 2)))
                              for arm, coef in coefs.items()))
        point[name] = float(np.mean(values))
        # a dedicated generator per contrast keeps every interval stable when the
        # contrast list grows (same rows resampled, same stream, same count)
        rng = np.random.default_rng(1703 + (abs(hash(name)) % 10007))
        reps = []
        for _ in range(n_boot):
            chosen = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([by_group[k] for k in chosen])
            per_seed = [sum(coef * float(np.sqrt(np.mean((preds[(arm, seed)][idx] - y[idx]) ** 2)))
                            for arm, coef in coefs.items()) for seed in seeds]
            reps.append(float(np.mean(per_seed)))
        draws[name] = reps
    result: dict[str, Any] = {
        "available": True, "n_scaffolds": info["n_scaffolds"], "n_rows": int(len(y)),
        "n_boot": n_boot, "parse": info, "contrasts": {},
        "definition": "paired scaffold bootstrap; each replicate resamples scaffolds with "
                      "replacement and recomputes every contrast on the same rows; the point "
                      "estimate reproduces the reported best-Valid RMSE differences",
    }
    for name in usable:
        values = np.asarray(draws[name])
        result["contrasts"][name] = {
            "seeds": usable[name], "point": point[name],
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "frac_positive": float(np.mean(values > 0)),
        }
    return result


# ---------------------------------------------------------------------------
# support statistics (PROTOCOL §7)
# ---------------------------------------------------------------------------


def _rooted_key(labels: Sequence[Sequence[int]], bonds: Mapping[tuple[int, int], Sequence[int]],
                root: int) -> str:
    """Exact rooted canonical key: minimise the serialised word over root-fixing orders."""
    from itertools import permutations

    n = len(labels)
    others = [i for i in range(n) if i != root]
    best: str | None = None
    for perm in permutations(others):
        order = [root] + list(perm)
        back = {old: new for new, old in enumerate(order)}
        node_part = ";".join(",".join(str(int(v)) for v in labels[old]) for old in order)
        mapped = []
        for (i, j), code in bonds.items():
            a, b = back[i], back[j]
            mapped.append((min(a, b), max(a, b), code))
        mapped.sort(key=lambda item: (item[0], item[1]))
        edge_part = ";".join(f"{a}-{b}:" + ",".join(str(int(v)) for v in code)
                             for a, b, code in mapped)
        key = f"{n}|{node_part}|{edge_part}"
        if best is None or key < best:
            best = key
    return best or ""


def write_support(args) -> None:
    dataset = dataset_name(getattr(args, "dataset", "lipo"))
    data = D.load_dataset(dataset)
    train_features = _split_port_features(data, "train")
    seen = _seen_sets(train_features)
    report: dict[str, Any] = {
        "dataset": dataset, "commit": C.git_commit(),
        "inherited_word_cache_sha256": data["record"].get("cache_sha256"),
        "derived_sha256": data["derived_record"].get("derived_sha256"),
        "seen_content": {"n_strict": len(seen["strict"]), "n_local": len(seen["local"]),
                         "n_arrangements": len(seen["arrangement"])},
        "splits": {},
    }
    for split in ("train", "valid"):
        features = train_features if split == "train" else _split_port_features(data, split)
        report["splits"][split] = _support_split(data, split, features, seen)
    report["definition"] = {
        "compatible_group": "sites of one word with identical original nine-column atom rows",
        "effected_incidence": "incidence whose own site belongs to a compatible group of size >= 2",
        "non_equivalent": "that group also contains a different exact rooted internal key",
        "strict_content": "centre type, both exact rooted keys and both port contexts",
        "local_content": "centre type and both port contexts (no rooted identity)",
        "arrangement": "local content plus the internal port distance and the port positions",
        "seen_local_new_arrangement": (
            "local content seen in Train while the exact relative arrangement is unseen: the "
            "case word templates cannot cover by memorising seen combinations"),
        "note": "diagnostic only; no gate uses these counts",
    }
    C.write_json(C.analysis_dir() / "support.json", C.jsonable(report))
    print(json.dumps(C.jsonable({s: v for s, v in report["splits"].items()}), indent=2))


def _split_port_features(data: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Exact rooted keys, port contexts and internal distances for every site."""
    words, derived = data["words"], data["derived"]
    arr = D.merge_split_arrays(words, derived, split)
    base = {k: np.asarray(words[f"base/{split}/{k}"]) for k in
            ("atom_feat", "bond_u", "bond_v", "bond_feat")}
    atom_feat = base["atom_feat"].astype(np.int64)
    bond_feat = base["bond_feat"].astype(np.int64)
    bond_u, bond_v = base["bond_u"].astype(np.int64), base["bond_v"].astype(np.int64)
    graph_atom_ptr = arr["graph_atom_ptr"]
    graph_word_ptr = arr["graph_word_ptr"]
    word_of_graph = np.repeat(np.arange(len(graph_word_ptr) - 1), np.diff(graph_word_ptr))
    site_ptr, site_atom = arr["site_ptr"], arr["site_atom"]
    word_ptr_bond, word_bonds = arr["word_ptr_bond"], arr["word_bonds"]
    word_tmpl, inc_ptr = arr["word_tmpl"], arr["inc_ptr"]
    inc_own_site, inc_other_site = arr["inc_own_site"], arr["inc_other_site"]
    n_words = int(word_tmpl.shape[0])
    n_sites = int(site_ptr[-1])
    site_key: list[str] = [""] * n_sites
    site_ctx: list[tuple] = [()] * n_sites
    site_dist: list[tuple] = [()] * n_sites
    site_word = np.repeat(np.arange(n_words, dtype=np.int64), np.diff(site_ptr))
    site_position = np.concatenate([np.arange(n, dtype=np.int64)
                                    for n in np.diff(site_ptr)]) if n_sites else \
        np.zeros(0, dtype=np.int64)
    atom_site = np.full(int(graph_atom_ptr[-1]), -1, dtype=np.int64)
    atom_site[site_atom] = np.arange(n_sites, dtype=np.int64)
    cache: dict[tuple, tuple[str, tuple, tuple]] = {}
    for w in range(n_words):
        atoms = [int(a) for a in site_atom[site_ptr[w]:site_ptr[w + 1]]]
        position = {atom: index for index, atom in enumerate(atoms)}
        bidx = word_bonds[word_ptr_bond[w]:word_ptr_bond[w + 1]]
        local_bonds: dict[tuple[int, int], tuple[int, ...]] = {}
        for bid in bidx:
            lo, hi = position[int(bond_u[bid])], position[int(bond_v[bid])]
            local_bonds[(min(lo, hi), max(lo, hi))] = tuple(int(v) for v in bond_feat[bid])
        labels = tuple(tuple(int(v) for v in atom_feat[a]) for a in atoms)
        signature = (int(word_tmpl[w]), labels,
                     tuple(sorted(local_bonds.items())))
        hit = cache.get(signature)
        if hit is None:
            keys, ctxs, dists = _word_port_signature(labels, local_bonds)
            hit = (keys, ctxs, dists)
            cache[signature] = hit
        for p in range(len(atoms)):
            site_key[site_ptr[w] + p] = hit[0][p]
            site_ctx[site_ptr[w] + p] = hit[1][p]
            site_dist[site_ptr[w] + p] = hit[2][p]
        del atoms, labels
    return {
        "split": split, "arr": arr, "graph_word_ptr": graph_word_ptr,
        "word_of_graph": word_of_graph, "row_ids": np.asarray(words[f"row_ids/{split}"]),
        "labels": np.asarray(words[f"labels/{split}"], dtype=np.float64),
        "site_key": site_key, "site_ctx": site_ctx, "site_dist": site_dist,
        "site_word": site_word, "site_position": site_position, "atom_site": atom_site,
        "inc_own_atom": arr["inc_own_atom"], "inc_other_atom": arr["inc_other_atom"],
        "n_words": n_words, "n_sites": n_sites,
        "inc_ptr": inc_ptr, "inc_own": inc_own_site, "inc_other": inc_other_site,
        "word_tmpl": word_tmpl, "site_ptr": site_ptr,
    }


def _word_port_signature(labels, local_bonds):
    n = len(labels)
    keys = [_rooted_key(labels, local_bonds, root=p) for p in range(n)]
    neighbours = {p: [] for p in range(n)}
    for (lo, hi), bond in local_bonds.items():
        neighbours[lo].append((hi, bond))
        neighbours[hi].append((lo, bond))
    ctxs = []
    for p in range(n):
        parts = sorted(((tuple(labels[o]), tuple(bond)) for o, bond in neighbours[p]))
        ctxs.append(tuple(parts))
    dists = []
    for source in range(n):
        dist = [-1] * n
        dist[source] = 0
        queue = [source]
        while queue:
            cur = queue.pop(0)
            for other, _ in neighbours[cur]:
                if dist[other] < 0:
                    dist[other] = dist[cur] + 1
                    queue.append(other)
        dists.append(tuple(dist))
    return tuple(keys), tuple(ctxs), tuple(dists)


def _job_sites(features, slot: int) -> tuple[int, int]:
    return (int(features["atom_site"][int(features["inc_own_atom"][slot])]),
            int(features["atom_site"][int(features["inc_other_atom"][slot])]))


def _job_distance(features, site_own: int, site_other: int) -> int:
    if int(features["site_word"][site_own]) != int(features["site_word"][site_other]):
        return -1
    return int(features["site_dist"][site_own][int(features["site_position"][site_other])])


def _job_key(features, slot: int) -> tuple:
    """Strict content: centre type, both rooted keys and both port contexts."""
    site_own, site_other = _job_sites(features, slot)
    return (int(features["word_tmpl"][int(features["site_word"][site_own])]),
            features["site_key"][site_own], features["site_key"][site_other],
            features["site_ctx"][site_own], features["site_ctx"][site_other])


def _job_local(features, slot: int) -> tuple:
    """Local word content only: centre type plus both port contexts."""
    site_own, site_other = _job_sites(features, slot)
    return (int(features["word_tmpl"][int(features["site_word"][site_own])]),
            features["site_ctx"][site_own], features["site_ctx"][site_other])


def _job_arrangement(features, slot: int) -> tuple:
    """Local content plus the relative arrangement of the two ports."""
    site_own, site_other = _job_sites(features, slot)
    own, other = int(features["inc_own"][slot]), int(features["inc_other"][slot])
    return (_job_local(features, slot),
            _job_distance(features, site_own, site_other), min(own, other), max(own, other))


def _iter_jobs(features):
    inc_ptr = features["inc_ptr"]
    for word in range(features["n_words"]):
        for slot in range(int(inc_ptr[word]), int(inc_ptr[word + 1])):
            yield word, slot


def _seen_sets(features) -> dict[str, set]:
    strict, local, arrangement = set(), set(), set()
    for _, slot in _iter_jobs(features):
        strict.add(_job_key(features, slot))
        local.add(_job_local(features, slot))
        arrangement.add(_job_arrangement(features, slot))
    return {"strict": strict, "local": local, "arrangement": arrangement}


def _support_split(data, split, features, seen) -> dict[str, Any]:
    arr = features["arr"]
    site_group, group_count = arr["site_group"], arr["group_count"]
    site_ptr, word_tmpl = features["site_ptr"], features["word_tmpl"]
    inc_ptr = features["inc_ptr"]
    n_words = features["n_words"]
    word_of_inc = np.repeat(np.arange(n_words), np.diff(inc_ptr))
    own_site = site_ptr[word_of_inc] + features["inc_own"]
    groups_of_inc = site_group[own_site]
    sizes = group_count[groups_of_inc]
    distinct = np.zeros(len(sizes), dtype=bool)
    site_keys = features["site_key"]
    for idx in np.nonzero(sizes >= 2)[0]:
        members = np.nonzero(site_group == groups_of_inc[idx])[0]
        distinct[idx] = len({site_keys[m] for m in members}) > 1
    word_of_graph = features["word_of_graph"]
    incidence_mol = word_of_graph[word_of_inc]
    affected_words = np.zeros(n_words, dtype=bool)
    if len(distinct):
        affected_words[np.unique(word_of_inc[distinct])] = True
    affected_mols = np.unique(word_of_graph[affected_words])
    row_ids, labels = features["row_ids"], features["labels"]
    unaffected = np.setdiff1d(np.arange(len(row_ids)), affected_mols)
    # seam statistics: content seen in Train but the exact job layout unseen
    unseen_content = unseen_arrangement = 0
    for _, slot in _iter_jobs(features):
        if _job_key(features, slot) not in seen["strict"]:
            unseen_content += 1
        if _job_local(features, slot) in seen["local"] and \
                _job_arrangement(features, slot) not in seen["arrangement"]:
            unseen_arrangement += 1
    return {
        "n_graphs": int(len(features["graph_word_ptr"]) - 1),
        "n_words": n_words,
        "n_sites": features["n_sites"],
        "n_incidences": int(len(sizes)),
        "n_incidences_group_ge2": int(np.sum(sizes >= 2)),
        "frac_incidences_group_ge2": float(np.mean(sizes >= 2)) if len(sizes) else 0.0,
        "n_incidences_non_equivalent": int(np.sum(distinct)),
        "frac_incidences_non_equivalent": float(np.mean(distinct)) if len(distinct) else 0.0,
        "n_words_affected": int(np.sum(affected_words)),
        "n_words_affected_with_branch": int(np.sum(affected_words & (np.diff(inc_ptr) >= 3))),
        "n_molecules_affected": int(len(affected_mols)),
        "frac_molecules_affected": float(len(affected_mols) / max(1, len(row_ids))),
        "n_word_types_affected": int(len(np.unique(word_tmpl[affected_words])))
        if affected_words.any() else 0,
        "n_scaffolds_affected": int(len({_scaffold_key(data, int(row_ids[int(m)]))
                                         for m in affected_mols})),
        "word_size_histogram": {str(int(k)): int(v) for k, v in
                                zip(*np.unique(np.diff(site_ptr), return_counts=True))},
        "port_count_histogram": {str(int(k)): int(v) for k, v in
                                 zip(*np.unique(np.diff(inc_ptr), return_counts=True))},
        "y_std_affected": float(np.std(labels[affected_mols])) if len(affected_mols) else None,
        "y_std_unaffected": float(np.std(labels[unaffected])) if len(unaffected) else None,
        "n_incidences_strict_content_unseen": int(unseen_content),
        "frac_incidences_strict_content_unseen": float(unseen_content / max(1, len(sizes))),
        "n_incidences_seen_local_new_arrangement": int(unseen_arrangement),
        "frac_incidences_seen_local_new_arrangement": float(unseen_arrangement / max(1, len(sizes))),
        "n_incidences_equivalent_in_group": int(np.sum((sizes >= 2) & ~distinct)),
    }


def _scaffold_key(data: Mapping[str, Any], row: int) -> int:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    text = data["raw"]["smiles"][int(row)]
    if isinstance(text, bytes):
        text = text.decode()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return -int(row)
    return hash(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))


# ---------------------------------------------------------------------------
# finalist + terminal
# ---------------------------------------------------------------------------


def write_finalist(args) -> None:
    cfg = C.load_config()
    dataset = "ogbg-mollipo"
    runs = collect_runs(dataset)
    index = arm_runs(runs)
    candidates = select_candidates(index, cfg, dataset)
    core = pick_core_pair(candidates, index, cfg)
    w, f = candidates["W"], candidates["F"]
    forced = [a for a in (getattr(args, "include", "") or "").split(",") if a]
    order: list[str] = []
    for arm in [w, core.get("C"), "GINE_REF", f] + forced:
        if arm and arm in candidates["all_mean_valid"] and arm not in order:
            order.append(arm)
    for arm in ("SET_MEAN_SELF", "BIND_SELF_SCALAR", "GINE_RAW", "BAG_CAP",
                "GLOBAL_CTX_BIND", "BIND_REL_random"):
        if len(order) >= 6:
            break
        if arm in candidates["all_mean_valid"] and arm not in order:
            order.append(arm)
    order = order[:6]
    architecture_ids = []
    for arm in order:
        seeds = [s for s in (0, 1, 2) if (arm, s) in index and index[(arm, s)]["status"] == "COMPLETE"]
        entry = {"arm": arm, "seeds": seeds, "runs": {}}
        for seed in seeds:
            run = index[(arm, seed)]
            run_dir = Path(run["_path"]).parent
            checkpoint = run_dir / "state_best.pt"
            entry["runs"][str(seed)] = {
                "run_id": run["run_id"], "commit": run.get("commit"),
                "spec": run.get("spec"), "partition": run.get("partition", "frozen"),
                "best_epoch": run["best"]["epoch"],
                "best_valid_rmse": run["best"]["valid_rmse_raw"],
                "checkpoint": str(checkpoint.relative_to(C.ROUND_DIR)),
                "checkpoint_sha256": C.sha256_file(checkpoint) if checkpoint.is_file() else None,
            }
        architecture_ids.append(entry)
    record = C.read_json(C.derived_record_path(dataset))
    manifest = {
        "frozen_before_terminal_test": True,
        "dataset": dataset,
        "commit": C.git_commit(),
        "config_sha256": C.sha256_json(cfg),
        "inherited_word_cache_sha256": record.get("inherited_word_cache_sha256"),
        "derived_sha256": record.get("derived_sha256"),
        "test_policy": "TERMINAL_DEVELOPMENT_TEST",
        "selection_basis": {
            "W_rule": "minimum mean best-Valid RMSE over the four matrix arms, seeds 0/1",
            "F_rule": "minimum mean best-Valid RMSE over every completed configuration, seeds 0/1",
            "core_pair_rule": "PROTOCOL section 6 / D4",
            "candidates": candidates,
            "core_pair": core,
            "forced_inclusions": forced,
            "forced_inclusion_reason": (
                "PROTOCOL section 4 makes the BIND x REL interaction the primary analysis, so both "
                "cells of that interaction are scored on Test even though neither is W or C"),
            "cap_note": "at most six architectures; the readout bridge lost the last slot to the "
                        "instance-geometry bridge and the interaction pair",
        },
        "architectures": architecture_ids,
        "no_test_used_for_selection": True,
    }
    C.write_json(C.ROUND_DIR / "finalist_manifest.json", C.jsonable(manifest))
    print(json.dumps({"finalist": order, "W": w, "F": f, "C": core.get("C"),
                      "mean_valid": {k: round(v, 5) for k, v in candidates["all_mean_valid"].items()}},
                     indent=2))


def build_test_split(dataset: str, derived_record: Mapping[str, Any]):
    """Replay the frozen merge history on the Test molecules (never used before)."""
    import ws_words as W

    words, record = D.load_old_words(dataset)
    raw = C.dataset_cache(dataset)
    table = D.table_from_record(record)
    history = [int(h) for h in record["merge"]["history"]]
    test_idx = [int(i) for i in raw["test_idx"]]
    node_ptr = np.asarray(raw["node_ptr"])
    bond_ptr = np.asarray(raw["bond_ptr"])
    atoms, bonds, offsets = [], [], []
    n_atoms = n_bonds = 0
    node_out = [0]
    bond_out = [0]
    atom_parts, bu_parts, bv_parts, bf_parts = [], [], [], []
    for g in test_idx:
        a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        atom_parts.append(np.asarray(raw["atom_feat"])[a0:a1])
        bu_parts.append(np.asarray(raw["bond_u"])[b0:b1])
        bv_parts.append(np.asarray(raw["bond_v"])[b0:b1])
        bf_parts.append(np.asarray(raw["bond_feat"])[b0:b1])
        n_atoms += a1 - a0
        n_bonds += b1 - b0
        node_out.append(n_atoms)
        bond_out.append(n_bonds)
    subset = {
        "node_ptr": np.asarray(node_out, dtype=np.int64),
        "bond_ptr": np.asarray(bond_out, dtype=np.int64),
        "atom_feat": np.concatenate(atom_parts, axis=0),
        "bond_u": np.concatenate(bu_parts),
        "bond_v": np.concatenate(bv_parts),
        "bond_feat": np.concatenate(bf_parts, axis=0),
    }
    molecules = W.load_molecules(subset)
    graphs = list(range(len(molecules)))
    states = W.replay_history(molecules, graphs, table, history)
    payload = W.build_split(molecules, graphs, states, table)
    base = {k: np.asarray(words[f"base/test/{k}"]) for k in
            ("atom_feat", "bond_u", "bond_v", "bond_feat", "graph_n_atoms", "graph_n_bonds")}
    if not np.array_equal(base["atom_feat"].astype(np.int64), subset["atom_feat"].astype(np.int64)):
        raise ValueError("test atom features do not match the frozen base arrays")
    arrays = {f"word/test/{k}": v for k, v in payload.items()}
    arrays.update({f"base/test/{k}": v for k, v in base.items()})
    arrays["row_ids/test"] = np.asarray(words["row_ids/test"], dtype=np.int64)
    arrays["labels/test"] = np.asarray(words["labels/test"], dtype=np.float64)
    dist = np.asarray(D.load_derived(dataset)["tmpl/dist"])
    split = D.split_from_arrays(arrays, dist, "test")
    prefixed = dict(arrays)
    for key, value in split.arrays.items():
        if key.startswith("word/") or key.startswith("base/"):
            continue
        prefixed[f"derived/test/{key}"] = value
    return split, table, prefixed


def write_test_support(args) -> None:
    """Support statistics for the frozen Test split (never used before scoring)."""
    dataset = "ogbg-mollipo"
    data = D.load_dataset(dataset)
    split, table, prefixed = build_test_split(dataset, data["derived_record"])
    words = dict(data["words"])
    words.update(prefixed)
    derived = dict(data["derived"])
    derived.update({k: v for k, v in prefixed.items() if k.startswith("derived/")})
    fake = {"words": words, "derived": derived, "raw": data["raw"], "record": data["record"]}
    seen = _seen_sets(_split_port_features(data, "train"))
    features = _split_port_features(fake, "test")
    stats = _support_split(fake, "test", features, seen)
    path = C.analysis_dir() / "support.json"
    payload = C.read_json(path) if path.is_file() else {"splits": {}, "seen_content": {}}
    payload["splits"]["test"] = stats
    payload["test_note"] = ("computed through build_test_split, which replays the frozen merge "
                            "history on Test molecules; the atom feature block is asserted equal "
                            "to the inherited base/test arrays before use")
    C.write_json(path, C.jsonable(payload))
    print(json.dumps(C.jsonable(stats), indent=2))


def run_terminal(args) -> None:
    import torch

    import wp_model as M
    import wp_train as T

    cfg = C.load_config()
    dataset = "ogbg-mollipo"
    manifest = C.read_json(C.ROUND_DIR / "finalist_manifest.json")
    if not manifest.get("frozen_before_terminal_test"):
        raise RuntimeError("finalist manifest is not frozen")
    data = D.load_dataset(dataset)
    scaler = C.target_scaler(data["raw"])
    test_split, table, _ = build_test_split(dataset, data["derived_record"])
    train_split = D.make_split(data, "train")
    raw = data["raw"]
    tmpl = {k[len("tmpl/"):]: v for k, v in data["words"].items() if k.startswith("tmpl/")}
    tmpl["dist"] = data["derived"]["tmpl/dist"]
    results: dict[str, Any] = {"dataset": dataset, "policy": "TERMINAL_DEVELOPMENT_TEST",
                               "commit": C.git_commit(), "scores": {}}
    for architecture in manifest["architectures"]:
        arm = architecture["arm"]
        for seed_text, info in architecture["runs"].items():
            checkpoint = C.ROUND_DIR / info["checkpoint"]
            payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
            if arm in ("GINE_REF", "GINE_RAW"):
                model = T.build_gine_model(raw, int(seed_text), args.device,
                                           raw_features=arm == "GINE_RAW")
                model.load_state_dict(payload["model"])
            else:
                spec = M.ModelSpec(**info["spec"])
                model = T.build_word_model(spec, tmpl, int(seed_text), args.device,
                                           hidden=cfg["model"]["hidden"])
                model.load_state_dict(payload["model"])
            model.eval()
            if arm in ("GINE_REF", "GINE_RAW"):
                res = T.evaluate_gine(model, T.gine_test_list(raw, scaler), args.device)
            else:
                res = T.evaluate_word(model, test_split, args.device, scaler)
            metrics = T.metric_block(res, scaler)
            results["scores"][f"{arm}_s{seed_text}"] = {
                "test_rmse_raw": metrics["rmse_raw"], "test_mae_raw": metrics["mae_raw"],
                "n_rows": int(len(metrics["pred_raw"])),
                "best_epoch": info["best_epoch"], "best_valid_rmse": info["best_valid_rmse"],
                "checkpoint_sha256": info["checkpoint_sha256"],
            }
            np.savez(Path(checkpoint).parent / "preds_test.npz",
                     row=metrics["row_ids"], pred=metrics["pred_raw"])
            C.write_json(C.analysis_dir() / "terminal_test.json", C.jsonable(results))
            print(json.dumps({f"{arm}_s{seed_text}": results["scores"][f"{arm}_s{seed_text}"]}),
                  flush=True)
    C.write_json(C.analysis_dir() / "terminal_test.json", C.jsonable(results))
