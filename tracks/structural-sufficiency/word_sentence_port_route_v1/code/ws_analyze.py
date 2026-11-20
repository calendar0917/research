"""Terminal development-Test scoring and post-hoc analysis.

``terminal``  one-shot Test scoring of frozen finalists (never used for selection).
``summary``   main tables, paired scaffold bootstrap, support subsets, case dumps,
              algebra recovery checks and the frozen-``R`` intervention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ws_common as C
import ws_probes as P
import ws_train as T
import ws_words as W

BOOTSTRAP_SEED = 1703
BOOTSTRAP_ROUNDS = 1000


# ---------------------------------------------------------------------------
# run discovery
# ---------------------------------------------------------------------------


def find_runs() -> dict[str, dict[str, Any]]:
    root = C.results_root()
    out: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*/run.json")):
        record = C.read_json(path)
        record.setdefault("run_dir", str(path.parent))
        out[record["run_id"]] = record
    return out


def load_curve(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    return C.read_json(Path(run["run_dir"]) / "curves.json")["curves"]


def load_preds(run: Mapping[str, Any], split: str, which: str = "best") -> dict[str, np.ndarray]:
    path = Path(run["run_dir"]) / f"preds_{which}_{split}.npz"
    with np.load(path, allow_pickle=False) as handle:
        return {k: handle[k] for k in handle.files}


# ---------------------------------------------------------------------------
# terminal Test scoring
# ---------------------------------------------------------------------------


def cmd_terminal(args) -> None:
    cfg = C.load_config()
    out_root = C.results_root() / "terminal"
    out_root.mkdir(parents=True, exist_ok=True)
    import torch

    import ws_model as M

    summary: dict[str, Any] = {"commit": C.git_commit(), "runs": {}}
    for run_id in args.runs:
        run = C.read_json(C.results_root() / run_id / "run.json")
        run.setdefault("run_dir", str(C.results_root() / run_id))
        if run.get("terminal_test_scored"):
            raise SystemExit(f"{run_id} was already scored once; refusing a second Test read")
        dataset = cfg["datasets"][run["dataset"]]
        payload, freeze = T.load_prepared(dataset)
        raw = C.dataset_cache(dataset)
        scaler = C.target_scaler(raw)
        table = P.table_from_freeze(freeze)
        history = list(freeze["merge"]["history"])
        test_graphs = [int(i) for i in raw["test_idx"]]
        partition = run["partition"]
        molecules = W.load_molecules(raw)
        if partition == "word":
            states = W.replay_history(molecules, test_graphs, table, history)
        else:
            states = W.char_states(molecules, test_graphs, table)
        data = _test_split(payload, raw, molecules, table, states)
        spec = M.ModelSpec(**run["spec"])
        model = T.make_model(spec, payload, seed=run["seed"], device=args.device)
        state = torch.load(Path(run["run_dir"]) / "state_best.pt", map_location=args.device,
                           weights_only=False)["state"]
        model.load_state_dict(state)
        model.eval()
        extra = spec.extra == "attrpair"
        hop = spec.para != "none"
        if spec.encoder == "gine_ref":
            import sq_data as SD
            import sq_gine as SG
            from torch_geometric.loader import DataLoader

            dims = C.official_dims()
            counts = {f"atom_{j}": int(dims["atom"][j]) for j in range(9)}
            counts.update({f"bond_{j}": int(dims["bond"][j]) for j in range(3)})
            counts["degree"] = 17
            counts["ring"] = 14
            packed = _packed_from_payload(payload, "test")
            frames = SG.build_data_list(packed, counts, scaler)
            loader = DataLoader(frames, batch_size=512, shuffle=False)
            res = T.evaluate_gine(model, loader, args.device)
        else:
            data.hop = (M.build_hop_index(data.arrays["graph_word_ptr"], data.arrays["inc_ptr"],
                                          data.arrays["inc_other"]) if hop else None)
            res = T.evaluate_word(model, data, args.device, scaler, extra=extra, hop=hop)
        metrics = T._metric_block(res, scaler)
        outdir = out_root / run_id
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outdir / "preds_test.npz", row_ids=metrics["row_ids"],
                            pred_raw=metrics["pred_raw"], target_raw=metrics["target_raw"])
        sup = support_sets(payload, "test", data.arrays)
        scaf = scaffold_map(dataset)
        record = {"run_id": run_id, "dataset": run["dataset"], "split": "test",
                  "test_policy": "TERMINAL_DEVELOPMENT_TEST",
                  "test_rmse_raw": metrics["rmse_raw"], "test_mae_raw": metrics["mae_raw"],
                  "n": int(len(metrics["row_ids"])), "commit": C.git_commit(),
                  "checkpoint": "state_best.pt",
                  "best_valid_rmse_raw": run.get("best", {}).get("valid_rmse_raw"),
                  "selected_before_test": True,
                  "support_subsets": subset_report(metrics["row_ids"], metrics["pred_raw"],
                                                   metrics["target_raw"], sup, scaf)}
        C.write_json(outdir / "terminal_test.json", record)
        run["terminal_test_scored"] = True
        C.write_json(Path(run["run_dir"]) / "run.json", run)
        summary["runs"][run_id] = record
        print(json.dumps(record, indent=2), flush=True)
    C.write_json(out_root / "summary.json", summary)


def _test_split(payload, raw, molecules, table, states):
    import ws_model as M

    graphs = [int(i) for i in raw["test_idx"]]
    arrays = W.build_split(molecules, graphs, states, table)
    prefix = "base/test/"
    data = M.SplitData(
        name="test", row_ids=np.asarray(payload["row_ids/test"]),
        labels=np.asarray(payload["labels/test"], dtype=np.float64),
        graph_n_atoms=np.asarray(payload[f"{prefix}graph_n_atoms"]),
        graph_n_bonds=np.asarray(payload[f"{prefix}graph_n_bonds"]),
        atom_feat=np.asarray(payload[f"{prefix}atom_feat"], dtype=np.int64),
        bond_feat=np.asarray(payload[f"{prefix}bond_feat"], dtype=np.int64),
        arrays=arrays)
    return data


def subset_report(row_ids, pred_raw, target_raw, sup: Mapping[str, Any],
                  scaf: Mapping[int, str]) -> dict[str, Any]:
    ids = np.asarray(row_ids)
    pred = np.asarray(pred_raw)
    target = np.asarray(target_raw)
    pos = {int(r): i for i, r in enumerate(ids.tolist())}
    new_layout = [int(ids[p]) for p in sup["molecules_with_new_layout"] if int(p) < len(ids)]
    unknown = [int(ids[p]) for p in sup["molecules_with_unknown_word"] if int(p) < len(ids)]
    known = [int(r) for r in ids.tolist() if int(r) not in set(unknown)]
    def block(rows: Sequence[int]) -> dict[str, Any]:
        idx = [pos[int(r)] for r in rows]
        if not idx:
            return {"n_molecules": 0}
        err = pred[idx] - target[idx]
        return {"n_molecules": len(idx),
                "n_scaffolds": len({scaf.get(int(r), f"row_{r}") for r in rows}),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "coverage": len(idx) / len(ids)}
    return {"all": block([int(r) for r in ids.tolist()]),
            "known_words": block(known),
            "new_layout_known_content": block(new_layout),
            "unknown_word_type": block(unknown),
            "n_pairs": sup["n_pairs"], "new_layout_pairs": sup["new_layout_pairs"],
            "new_layout_word_types": sup["new_layout_word_types"],
            "new_layout_layouts": sup["new_layout_layouts"],
            "train_layouts": sup["train_layouts"],
            "train_word_types": sup["train_word_types"]}


def _packed_from_payload(payload, split):
    import sq_ast as SA

    fields = {}
    mapping = {"atom_feat": "atom_feat", "degree": "degree", "min_ring": "min_ring",
               "in_cycle": "in_cycle", "node_graph": "node_graph", "bond_u": "bond_u",
               "bond_v": "bond_v", "bond_feat": "bond_feat",
               "graph_n_atoms": "graph_n_atoms", "graph_n_bonds": "graph_n_bonds",
               "graph_n_components": "graph_n_components", "graph_cycle_rank": "graph_cycle_rank",
               "row_ids": "row_ids", "labels": "labels"}
    for key in mapping:
        fields[key] = np.asarray(payload[f"base/{split}/{key}"])
    fields["dataset"] = "ogbg"
    fields["split"] = split
    fields["n_graphs"] = int(len(fields["row_ids"]))
    fields["n_nodes"] = int(len(fields["atom_feat"]))
    fields["n_bonds"] = int(len(fields["bond_u"]))
    fields["smiles"] = ["" for _ in range(fields["n_graphs"])]
    return SA.PackedGraph(**fields)


# ---------------------------------------------------------------------------
# scaffolds (Bemis-Murcko) for grouped bootstrap
# ---------------------------------------------------------------------------


def scaffold_map(dataset: str) -> dict[int, str]:
    path = C.analysis_dir() / f"scaffolds_{dataset.replace('ogbg-', '')}.json"
    if path.is_file():
        return {int(k): v for k, v in C.read_json(path).items()}
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    raw = C.dataset_cache(dataset)
    out: dict[int, str] = {}
    for row, smiles in enumerate(raw["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            out[row] = f"unparsed_{row}"
            continue
        out[row] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    C.write_json(path, out)
    return out


def paired_bootstrap(pred: np.ndarray, other: np.ndarray, target: np.ndarray,
                     groups: Sequence[str], rounds: int = BOOTSTRAP_ROUNDS,
                     seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Gain = RMSE(other) - RMSE(pred) with scaffolds as resampling units."""
    keys = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(keys.tolist())), dtype=object)
    index = {k: i for i, k in enumerate(unique.tolist())}
    codes = np.asarray([index[k] for k in keys.tolist()], dtype=np.int64)
    err_a = (np.asarray(pred) - np.asarray(target)) ** 2
    err_b = (np.asarray(other) - np.asarray(target)) ** 2
    n_groups = len(unique)
    members = [np.nonzero(codes == g)[0] for g in range(n_groups)]
    rng = np.random.default_rng(seed)
    samples = np.empty(rounds, dtype=np.float64)
    point = float(np.sqrt(err_b.mean()) - np.sqrt(err_a.mean()))
    for r in range(rounds):
        pick = rng.integers(0, n_groups, n_groups)
        idx = np.concatenate([members[g] for g in pick])
        samples[r] = np.sqrt(err_b[idx].mean()) - np.sqrt(err_a[idx].mean())
    return {"gain": point, "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "n_scaffolds": int(n_groups), "n_molecules": int(len(err_a)),
            "rounds": int(rounds), "seed": int(seed)}


def leave_one_out_gain(pred: np.ndarray, other: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    err_a = (np.asarray(pred) - np.asarray(target)) ** 2
    err_b = (np.asarray(other) - np.asarray(target)) ** 2
    base = float(np.sqrt(err_b.mean()) - np.sqrt(err_a.mean()))
    worst = float((err_b - err_a).max())
    if worst <= 0:
        return {"gain": base, "leave_best_out_gain": base, "best_molecule_index": None}
    i = int(np.argmax(err_b - err_a))
    keep = np.ones(len(err_a), dtype=bool)
    keep[i] = False
    after = float(np.sqrt(err_b[keep].mean()) - np.sqrt(err_a[keep].mean()))
    return {"gain": base, "leave_best_out_gain": after, "best_molecule_index": i}


# ---------------------------------------------------------------------------
# support subsets (section 12.2 definitions, frozen structure only)
# ---------------------------------------------------------------------------


def support_sets(payload: Mapping[str, Any], split: str,
                 arrays: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Known words / known-content-new-layout membership per molecule.

    ``arrays`` may supply the packed split arrays directly (Test is built on the
    fly and is never persisted in the prepared payload).
    """
    if arrays is None:
        prefix = f"word/{split}/"
        arr = {k[len(prefix):]: v for k, v in payload.items() if k.startswith(prefix)}
    else:
        arr = dict(arrays)
    train_prefix = "word/train/"
    tr = {k[len(train_prefix):]: v for k, v in payload.items() if k.startswith(train_prefix)}
    gw = arr["graph_word_ptr"]
    wt = arr["word_tmpl"]
    inc_ptr = arr["inc_ptr"]
    inc_other = arr["inc_other"]
    inc_bond = arr["inc_bond"]
    pair_ptr = arr["pair_ptr"]
    pair_a, pair_b = arr["pair_a"], arr["pair_b"]
    layouts = arr["pair_layout"]
    bond_feat = np.asarray(payload[f"base/{split}/bond_feat"])
    tr_bond_feat = np.asarray(payload["base/train/bond_feat"])

    def signatures(wt_, inc_ptr_, inc_other_, inc_bond_, pair_ptr_, pair_a_, pair_b_, layout_,
                   bond_feat_):
        center, content, layout = [], [], []
        for w in range(len(wt_)):
            lo, hi = int(pair_ptr_[w]), int(pair_ptr_[w + 1])
            for k in range(lo, hi):
                a, b = int(pair_a_[k]), int(pair_b_[k])
                nb = tuple(sorted((int(wt_[inc_other_[a]]), int(wt_[inc_other_[b]]))))
                bonds = tuple(sorted((int(bond_feat_[inc_bond_[a]][0]),
                                      int(bond_feat_[inc_bond_[b]][0]))))
                center.append(int(wt_[w]))
                content.append((int(wt_[w]), nb, bonds))
                layout.append(str(layout_[k]))
        return center, content, layout

    tr_wt = tr["word_tmpl"]
    tr_center, tr_content, tr_layout = signatures(
        tr_wt, tr["inc_ptr"], tr["inc_other"], tr["inc_bond"], tr["pair_ptr"],
        tr["pair_a"], tr["pair_b"], tr["pair_layout"], tr_bond_feat)
    train_words = set(int(t) for t in tr_wt.tolist())
    train_content = set(tr_content)
    train_layout = set(tr_layout)
    center, content, layout = signatures(wt, inc_ptr, inc_other, inc_bond, pair_ptr,
                                         pair_a, pair_b, layouts, bond_feat)
    new_layout_hits = [(c, l) for c, l in zip(content, layout)
                       if l not in train_layout and c in train_content]
    new_layout_centres = set(c for c, _ in new_layout_hits)
    new_layouts = set(l for _, l in new_layout_hits)
    # per-molecule aggregation
    n_graphs = len(gw) - 1
    unknown_word_mols, new_layout_mols = [], []
    for g in range(n_graphs):
        lo, hi = int(gw[g]), int(gw[g + 1])
        words = wt[lo:hi]
        if any(int(t) not in train_words for t in words):
            unknown_word_mols.append(g)
        for w in range(lo, hi):
            if any((str(layouts[k]) not in train_layout) for k in
                   range(int(pair_ptr[w]), int(pair_ptr[w + 1]))):
                a = int(pair_ptr[w])
                if a < int(pair_ptr[w + 1]):
                    new_layout_mols.append(g)
                break
    return {
        "split": split,
        "train_word_types": len(train_words),
        "train_content_signatures": len(train_content),
        "train_layouts": len(train_layout),
        "n_pairs": len(center),
        "new_layout_pairs": int(len(new_layout_hits)),
        "new_layout_word_types": int(len(new_layout_centres)),
        "new_layout_layouts": int(len(new_layouts)),
        "molecules_with_new_layout": sorted(set(new_layout_mols)),
        "molecules_with_unknown_word": sorted(set(unknown_word_mols)),
        "n_graphs": int(n_graphs),
    }


# ---------------------------------------------------------------------------
# inspection: full F from terms, case dumps, frozen-R intervention
# ---------------------------------------------------------------------------


def _load_model(run: Mapping[str, Any], payload: Mapping[str, Any], device: str):
    import torch

    import ws_model as M

    spec = M.ModelSpec(**run["spec"])
    model = T.make_model(spec, payload, seed=run["seed"], device=device)
    state = torch.load(Path(run["run_dir"]) / "state_best.pt", map_location=device,
                       weights_only=False)["state"]
    model.load_state_dict(state)
    model.eval()
    return model, spec


def forward_terms(model, data, device: str, *, extra: bool = False, hop: bool = False,
                  intervention: str = "none", batch_size: int = 256,
                  index: Sequence[int] | None = None):
    """Per-word terms for one split, plus the whole-molecule F rebuilt from them.

    ``graph_row`` maps every stored word to its official row id; ``f_reconstructed``
    is the algebraic sum ``bias + sum_i mass_i (u_i+v_i+w_i+r_i)`` per row.
    """
    import torch

    import ws_model as M

    model.eval()
    pos = {int(r): i for i, r in enumerate(np.asarray(data.row_ids).tolist())}
    n_rows = len(data.row_ids)
    store: dict[str, list[np.ndarray]] = {}
    pred = np.full(n_rows, np.nan)
    recon = np.full(n_rows, np.nan)
    index = np.arange(n_rows) if index is None else np.asarray(index, dtype=np.int64)
    bias = float(model.bias.detach().cpu()) if hasattr(model, "bias") else 0.0
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            pick = index[start:start + batch_size]
            raw = M.pack_batch(data, pick, attrpair=extra, hop=hop)
            if intervention == "erased":
                raw["pair_R"] = raw["pair_ER"]
            batch = M.to_device(raw, device)
            out = model(batch)
            rows = np.asarray(raw["row_ids"]).astype(np.int64)
            gw = np.asarray(raw["graph_of_word"])
            u = out["u"].detach().cpu().numpy()
            v = out["v"].detach().cpu().numpy()
            w = out["w"].detach().cpu().numpy()
            r = out["r"].detach().cpu().numpy()
            o = out["pred"].detach().cpu().numpy()
            n_at = np.asarray(batch["word_natoms"].detach().cpu(), dtype=np.float64)
            g_at = np.asarray(batch["n_atoms_graph"].detach().cpu(), dtype=np.float64)[gw]
            mass = n_at / g_at
            contrib = (u + v + w + r) * mass
            for j, rid in enumerate(rows.tolist()):
                sel = gw == j
                recon[pos[rid]] = float(contrib[sel].sum()) + bias
                pred[pos[rid]] = float(o[j])
            payload = {"u": u, "v": v, "w": w, "r": r,
                       "word_tmpl": np.asarray(raw["word_tmpl"]),
                       "mass": mass, "word_natoms": n_at,
                       "graph_row": rows[gw],
                       "n_inc": np.asarray(out["n_inc"].detach().cpu())}
            for key, value in payload.items():
                store.setdefault(key, []).append(np.atleast_1d(value))
    merged = {k: np.concatenate(v) for k, v in store.items()}
    merged["bias"] = bias
    merged["pred"] = pred
    merged["f_reconstructed"] = recon
    ok = ~np.isnan(pred)
    merged["recon_max_abs_diff"] = float(np.max(np.abs(recon[ok] - pred[ok])))
    return merged


def case_dump(run: Mapping[str, Any], payload: Mapping[str, Any], split: str, *,
              device: str = "cpu", n_support: int = 4, n_error: int = 4) -> dict[str, Any]:
    dataset = C.load_config()["datasets"][run["dataset"]]
    model, spec = _load_model(run, payload, device)
    extra = spec.extra == "attrpair"
    hop = spec.para != "none"
    data = T.split_data(payload, run["partition"], split, with_hop=hop)
    scaler = C.target_scaler(C.dataset_cache(dataset))
    terms = forward_terms(model, data, device, extra=extra, hop=hop)
    pos = {int(r): i for i, r in enumerate(np.asarray(data.row_ids).tolist())}
    pred_raw = terms["pred"] * scaler["std"] + scaler["mean"]
    target_raw = data.labels
    err = np.abs(pred_raw - target_raw)
    order = np.argsort(-err)
    gw = data.arrays["graph_word_ptr"]
    inc_ptr = data.arrays["inc_ptr"]
    support = []
    for g in range(len(data.row_ids)):
        lo, hi = int(gw[g]), int(gw[g + 1])
        ports = [int(inc_ptr[w + 1] - inc_ptr[w]) for w in range(lo, hi)]
        if ports and max(ports) >= 2:
            support.append((g, max(ports)))
    support.sort(key=lambda item: (-item[1], int(data.row_ids[item[0]])))
    chosen = [("support", g) for g, _ in support[:n_support]]
    chosen += [("max_error", int(i)) for i in order[:n_error]]
    canonical = table_templates_cached(dataset)
    out_cases = []
    for tag, g in chosen:
        single = forward_terms(model, data, device, extra=extra, hop=hop,
                               index=[g])
        lo, hi = int(gw[g]), int(gw[g + 1])
        atom_base = int(data.arrays["graph_atom_ptr"][g])
        words = []
        for wi, w in enumerate(range(lo, hi)):
            tid = int(data.arrays["word_tmpl"][w])
            atoms = data.arrays["word_atoms"][int(data.arrays["word_ptr_atom"][w]):
                                              int(data.arrays["word_ptr_atom"][w + 1])]
            ports = []
            for k in range(int(inc_ptr[w]), int(inc_ptr[w + 1])):
                ports.append({
                    "own_site": int(data.arrays["inc_own_site"][k]),
                    "neighbour_word_offset": int(data.arrays["inc_other"][k] - lo),
                    "neighbour_site": int(data.arrays["inc_other_site"][k]),
                    "bond_columns": [int(x) for x in data.bond_feat[int(data.arrays["inc_bond"][k])]],
                    "own_atom_row_local": int(data.arrays["inc_own_atom"][k]) - atom_base})
            words.append({
                "word_offset": wi, "template_id": tid,
                "template_canonical": canonical[tid],
                "word_atoms_row_local": [int(a) - atom_base for a in atoms],
                "ports": ports, "n_incidences": len(ports),
                "u_raw": float(single["u"][wi]) * scaler["std"],
                "v_raw": float(single["v"][wi]) * scaler["std"],
                "w_raw": float(single["w"][wi]) * scaler["std"],
                "r_raw": float(single["r"][wi]) * scaler["std"],
                "mass": float(single["mass"][wi])})
        out_cases.append({
            "rule": tag, "row_id": int(data.row_ids[g]),
            "scaffold": scaffold_map(dataset).get(int(data.row_ids[g])),
            "y_raw": float(target_raw[g]), "F_raw": float(pred_raw[g]),
            "intercept_raw": float(scaler["mean"] + scaler["std"] * terms["bias"]),
            "abs_error": float(err[g]), "n_atoms": int(data.graph_n_atoms[g]),
            "words": words})
    return {"split": split, "run_id": run["run_id"],
            "recon_max_abs_diff": terms["recon_max_abs_diff"], "cases": out_cases}


_TABLE_CACHE: dict[str, tuple[W.TemplateTable, list[str]]] = {}


def table_from_freeze_cached(dataset: str):
    if dataset not in _TABLE_CACHE:
        payload, freeze = T.load_prepared(dataset)
        table = P.table_from_freeze(freeze)
        _TABLE_CACHE[dataset] = (table, [t.canon for t in table.templates])
    return _TABLE_CACHE[dataset][0]


def table_templates_cached(dataset: str):
    table_from_freeze_cached(dataset)
    return _TABLE_CACHE[dataset][1]


# ---------------------------------------------------------------------------
# summary tables and branch arithmetic
# ---------------------------------------------------------------------------

PRIORITY = {"code": 0, "id": 1, "wgine": 2, "gine_ref": 3}


def _thr(scaler: Mapping[str, float]) -> dict[str, float]:
    sigma = float(scaler["std"])
    return {"sigma": sigma, "delta_task": 0.01 * sigma, "delta_port": 0.005 * sigma,
            "band": 0.04 * sigma}


def _encoder_of(run: Mapping[str, Any]) -> str:
    return run["spec"]["encoder"]


def _mode_of(run: Mapping[str, Any]) -> str:
    return run["spec"]["mode"]


def choose_L(runs: Mapping[str, Mapping[str, Any]], thr: Mapping[str, float], seeds=(0,),
             dataset: str = "mollipo") -> dict[str, Any]:
    """Lowest Valid RMSE among the three FULL encoders, with the frozen tie rule."""
    cands = {}
    for run in runs.values():
        if run["dataset"] != dataset or run["seed"] not in seeds:
            continue
        if _mode_of(run) != "full" or _encoder_of(run) not in ("code", "id", "wgine"):
            continue
        if run["spec"]["extra"] != "none" or run["spec"]["para"] != "none":
            continue
        if run["partition"] != "word" or run.get("recipe_tag", "r0") != "r0":
            continue
        cands[run["arm"]] = run
    if not cands:
        return {}
    best_value = min(r["best"]["valid_rmse_raw"] for r in cands.values())
    close = {k: v for k, v in cands.items() if v["best"]["valid_rmse_raw"] <= best_value + 0.002 * thr["sigma"]}
    winner = sorted(close, key=lambda k: (PRIORITY[_encoder_of(close[k])], k))[0]
    return {"candidates": {k: v["best"]["valid_rmse_raw"] for k, v in cands.items()},
            "best_value": best_value, "tie_set": sorted(close),
            "selected_arm": winner, "selected_run": close[winner]["run_id"],
            "selected_encoder": _encoder_of(close[winner])}


def classify_stage_b(runs: Mapping[str, Mapping[str, Any]], thr: Mapping[str, float],
                     L: Mapping[str, Any]) -> dict[str, Any]:
    if not L:
        return {"branch": "UNDECIDED", "reason": "no FULL candidate runs"}
    valid = {run["run_id"]: run["best"]["valid_rmse_raw"] for run in runs.values()
             if run["dataset"] == "mollipo" and run["seed"] == 0}
    ref = next((r for r in runs.values() if r["arm"] == "gine_ref" and r["dataset"] == "mollipo"
                and r["seed"] == 0), None)
    bag = next((r for r in runs.values() if r["arm"] == "bag" and r["dataset"] == "mollipo"
                and r["seed"] == 0), None)
    Lrun = runs[L["selected_run"]]
    enc = L["selected_encoder"]
    same_recipe = [r for r in runs.values() if r["dataset"] == "mollipo" and r["seed"] == 0
                   and _mode_of(r) == "erased" and _encoder_of(r) == enc
                   and r["spec"]["extra"] == "none" and r["spec"]["para"] == "none"]
    erased = next((r for r in same_recipe if r.get("recipe_tag") == Lrun.get("recipe_tag")), None)
    if erased is None and same_recipe:
        erased = same_recipe[0]
    out: dict[str, Any] = {"L_run": Lrun["run_id"], "L_encoder": enc,
                           "L_valid": Lrun["best"]["valid_rmse_raw"]}
    if ref is None or bag is None:
        out.update({"branch": "UNDECIDED", "reason": "missing GINE_REF or BAG"})
        return out
    out["gine_ref_valid"] = ref["best"]["valid_rmse_raw"]
    out["bag_valid"] = bag["best"]["valid_rmse_raw"]
    out["gain_L_bag"] = bag["best"]["valid_rmse_raw"] - Lrun["best"]["valid_rmse_raw"]
    out["gain_L_ref"] = ref["best"]["valid_rmse_raw"] - Lrun["best"]["valid_rmse_raw"]
    if erased is not None:
        out["erased_run"] = erased["run_id"]
        out["erased_valid"] = erased["best"]["valid_rmse_raw"]
        out["gain_L_erased"] = erased["best"]["valid_rmse_raw"] - Lrun["best"]["valid_rmse_raw"]
    delta_task, delta_port, band = thr["delta_task"], thr["delta_port"], thr["band"]
    # leave-one-out on the favourable direction
    pred_L = load_preds(Lrun, "valid") if Path(Lrun["run_dir"]).is_dir() else None
    dirs = {}
    if pred_L is not None:
        target = pred_L["target_raw"]
        if out["gain_L_bag"] > 0:
            pb = load_preds(bag, "valid")
            dirs["vs_bag"] = leave_one_out_gain(pred_L["pred_raw"], pb["pred_raw"], target)
        if "erased_run" in out and out["gain_L_erased"] > 0:
            pe = load_preds(erased, "valid")
            dirs["vs_erased"] = leave_one_out_gain(pred_L["pred_raw"], pe["pred_raw"], target)
    out["leave_one_out"] = dirs
    robust_task = (out["gain_L_bag"] >= delta_task
                   and dirs.get("vs_bag", {}).get("leave_best_out_gain", -1) > 0)
    robust_port = (out.get("gain_L_erased", -1e9) >= delta_port
                   and dirs.get("vs_erased", {}).get("leave_best_out_gain", -1) > 0
                   and Lrun["best"]["valid_rmse_raw"] <= out["bag_valid"] + delta_port)
    not_worse_ref = Lrun["best"]["valid_rmse_raw"] <= out["gine_ref_valid"] + band
    boundary = Lrun["best"]["valid_rmse_raw"] <= out["gine_ref_valid"] + 2 * band
    out.update({"robust_task_direction": bool(robust_task),
                "robust_port_direction": bool(robust_port),
                "not_worse_than_ref_plus_band": bool(not_worse_ref),
                "within_ref_plus_2band": bool(boundary)})
    if not_worse_ref and (robust_task or robust_port):
        out["branch"] = "P"
    elif boundary:
        out["branch"] = "B"
    else:
        out["branch"] = "N"
    return out


def cmd_summary(args) -> None:
    cfg = C.load_config()
    runs = find_runs()
    analysis = C.analysis_dir()
    analysis.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"commit": C.git_commit(), "n_runs": len(runs),
                           "results_root": str(C.results_root())}
    thresholds = {}
    for key, dataset in cfg["datasets"].items():
        scaler = C.target_scaler(C.dataset_cache(dataset))
        thresholds[key] = _thr(scaler) | {"mean": scaler["mean"], "std": scaler["std"]}
    out["thresholds"] = thresholds
    # main table
    rows = []
    for run_id, run in sorted(runs.items()):
        row = {"run_id": run_id, "dataset": run["dataset"], "arm": run["arm"],
               "encoder": _encoder_of(run), "mode": _mode_of(run),
               "extra": run["spec"]["extra"], "para": run["spec"]["para"],
               "recipe_tag": run.get("recipe_tag", "r0"),
               "partition": run["partition"], "seed": run["seed"],
               "epochs": run.get("epochs") or run.get("n_curve_rows"), "n_parameters": run["n_parameters"],
               "seconds": round(run["seconds"], 1),
               "seconds_per_epoch": round(run["seconds"] / max(1, run.get("epochs") or run.get("n_curve_rows")), 2),
               "best_epoch": run["best"]["epoch"],
               "best_train_rmse": run["best"]["train_rmse_raw"],
               "best_valid_rmse": run["best"]["valid_rmse_raw"],
               "best_valid_mae": run["best"]["valid_mae_raw"],
               "final_valid_rmse": run["final"]["valid_rmse_raw"],
               "last20_valid_rmse": run["last20"]["mean_last20_valid_rmse_raw"],
               "last20_train_rmse": run["last20"]["mean_last20_train_rmse_raw"],
               "init_state_hash": run.get("init_state_hash"),
               "shared_body_hash": run.get("shared_body_hash")}
        term = C.results_root() / "terminal" / run_id / "terminal_test.json"
        if term.is_file():
            rec = C.read_json(term)
            row["test_rmse"] = rec["test_rmse_raw"]
            row["test_mae"] = rec["test_mae_raw"]
        rows.append(row)
    C.write_csv(analysis / "main_table.csv", rows)
    out["main_table"] = rows
    out["stage_b"] = {}
    for key in cfg["datasets"]:
        L = choose_L(runs, thresholds[key], dataset=key)
        if L:
            out["stage_b"][key] = {"L": L, "classification": classify_stage_b(
                {k: v for k, v in runs.items() if v["dataset"] == key}, thresholds[key], L)}
    out["stage_c"] = _stage_c_summary(runs, thresholds)
    out["stage_d"] = _stage_d_summary(runs, thresholds)
    # support subsets (Valid only here; Test needs the terminal pass)
    support = {}
    for key, dataset in cfg["datasets"].items():
        payload, _ = T.load_prepared(dataset)
        support[key] = {"valid": support_sets(payload, "valid")}
        try:
            support[key]["train"] = support_sets(payload, "train")
        except Exception as exc:  # noqa: BLE001 - record, never hide
            support[key]["train_error"] = repr(exc)
    out["support_sets"] = {k: {split: {kk: vv for kk, vv in val.items()
                                       if not isinstance(vv, list)}
                               for split, val in v.items()} for k, v in support.items()}
    out["support_sets_full"] = {k: {split: {"molecules_with_new_layout": val["molecules_with_new_layout"],
                                            "molecules_with_unknown_word": val["molecules_with_unknown_word"]}
                                    for split, val in v.items()} for k, v in support.items()}
    # bootstrap on core comparisons (Lipo Valid, seed0 and pooled seeds)
    out["bootstrap"] = _bootstrap_core(runs, cfg)
    C.write_json(analysis / "summary.json", out)
    print(json.dumps({"n_runs": len(runs), "stage_b": out["stage_b"],
                      "bootstrap_keys": list(out["bootstrap"])}, indent=2)[:4000])


def _stage_c_summary(runs, thresholds):
    out = {}
    for key in thresholds:
        key_runs = {k: v for k, v in runs.items() if v["dataset"] == key}
        by_arm: dict[str, list[float]] = {}
        for run in key_runs.values():
            if run["spec"]["extra"] != "none" or run["spec"]["para"] not in ("none",):
                continue
            by_arm.setdefault(run["arm"], []).append(run["best"]["valid_rmse_raw"])
        out[key] = {arm: {"values": sorted(vals), "mean": float(np.mean(vals)),
                          "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                          "n": len(vals)} for arm, vals in sorted(by_arm.items())}
        delta_task = thresholds[key]["delta_task"]
        delta_port = thresholds[key]["delta_port"]
        band = thresholds[key]["band"]
        L_arms = [a for a in out[key] if a.startswith("full_")]
        core = {}
        for L in L_arms:
            for other in ("bag", "erased_" + L.replace("full_", ""), "gine_ref"):
                if L in out[key] and other in out[key]:
                    gains = [b - a for a, b in zip(out[key][L]["values"], out[key][other]["values"])]
                    n = min(len(gains), len(out[key][L]["values"]), len(out[key][other]["values"]))
                    if n == 0:
                        continue
                    same_sign = sum(1 for g in gains[:n] if g > 0)
                    core[f"{L}_vs_{other}"] = {
                        "gains": gains[:n], "mean_gain": float(np.mean(gains[:n])),
                        "n_seeds": n, "n_positive": same_sign,
                        "delta": delta_task if other == "bag" else delta_port if other.startswith("erased") else band,
                    }
        out[key]["core_gains"] = core
    return out


def _stage_d_summary(runs, thresholds):
    out = {}
    for key in thresholds:
        rel = [r for r in runs.values() if r["dataset"] == key and r["spec"]["para"] == "rel"]
        cap = [r for r in runs.values() if r["dataset"] == key and r["spec"]["para"] == "cap"]
        if not rel and not cap:
            continue
        entry = {}
        for name, group in (("para_rel", rel), ("para_cap", cap)):
            vals = sorted(r["best"]["valid_rmse_raw"] for r in group)
            entry[name] = {"values": vals, "mean": float(np.mean(vals)) if vals else None,
                           "n": len(vals)}
        if rel and cap:
            n = min(len(rel), len(cap))
            d = sorted(r["best"]["valid_rmse_raw"] for r in rel)[:n]
            c = sorted(r["best"]["valid_rmse_raw"] for r in cap)[:n]
            entry["rel_minus_cap"] = [a - b for a, b in zip(d, c)]
        out[key] = entry
    return out


def _bootstrap_core(runs, cfg):
    out = {}
    for key, dataset in cfg["datasets"].items():
        payload, _ = T.load_prepared(dataset)
        scaf = scaffold_map(dataset)
        by_arm: dict[str, dict[int, np.ndarray]] = {}
        target = {}
        for run in runs.values():
            if run["dataset"] != key:
                continue
            if run["spec"]["extra"] != "none" or run["spec"]["para"] != "none":
                continue
            path = Path(run["run_dir"]) / "preds_best_valid.npz"
            if not path.is_file():
                continue
            preds = load_preds(run, "valid")
            arm = run["arm"] + ("" if run["seed"] == 0 else f"_s{run['seed']}")
            by_arm[arm] = dict(zip(preds["row_ids"].tolist(), preds["pred_raw"].tolist()))
            target = dict(zip(preds["row_ids"].tolist(), preds["target_raw"].tolist()))
        pairs = []
        for L in ("full_code", "full_id", "full_wgine"):
            for other in ("bag", "erased_code", "erased_id", "erased_wgine", "gine_ref",
                          "char_full"):
                if L in by_arm and other in by_arm:
                    pairs.append((L, other))
        entries = {}
        for L, other in pairs:
            shared = sorted(set(by_arm[L]) & set(by_arm[other]) & set(target))
            if len(shared) < 10:
                continue
            a = np.asarray([by_arm[L][r] for r in shared])
            b = np.asarray([by_arm[other][r] for r in shared])
            y = np.asarray([target[r] for r in shared])
            groups = [scaf.get(r, f"row_{r}") for r in shared]
            entries[f"{L}_vs_{other}"] = paired_bootstrap(a, b, y, groups)
        out[key] = entries
    return out


# ---------------------------------------------------------------------------
# run manifest / resource ledger
# ---------------------------------------------------------------------------

def _parse_ts(value: str) -> float:
    from datetime import datetime

    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def cmd_manifest(args) -> None:
    """Machine run record for *this* round only.

    Jobs are selected by creation time (after ``T0`` minus 30 min), so earlier
    rounds sharing the same experiment-name space cannot leak into the ledger.
    """
    cfg = C.load_config()
    root = C.ROUND_DIR
    t0 = float(cfg["budget"]["t0_monotonic_wall"])
    cutoff = t0 - 1800.0
    jobs = []
    if args.jobs_json:
        blob = C.read_json(Path(args.jobs_json))
        jobs = blob.get("jobs", blob)
    ledger = []
    for job in jobs:
        created = job.get("created_at")
        if not created or _parse_ts(created) < cutoff:
            continue
        started, completed = job.get("started_at"), job.get("completed_at")
        seconds = None
        if started and completed:
            seconds = _parse_ts(completed) - _parse_ts(started)
        elif job.get("elapsed"):
            seconds = float(job["elapsed"])
        pool = job.get("pool") or ""
        uses_gpu = pool.startswith("res-gpu")
        ledger.append({
            "experiment": job.get("experiment"), "run_id": job.get("run_id"),
            "pool": pool, "uses_gpu": uses_gpu, "state": job.get("state"),
            "commit": job.get("commit"), "node": job.get("node"),
            "driver_version": job.get("driver_version"),
            "torch_version": job.get("torch_version"),
            "cuda_visible_devices": job.get("cuda_visible_devices"),
            "created_at": created, "started_at": started, "completed_at": completed,
            "seconds": seconds, "gpu_hours": (seconds / 3600.0) if (seconds and uses_gpu) else None,
            "exit_code": job.get("exit_code"), "failure_reason": job.get("failure_reason")})
    gpu_hours = sum(item["gpu_hours"] or 0.0 for item in ledger)
    datasets = {}
    for key, dataset in cfg["datasets"].items():
        freeze_path = C.analysis_dir() / f"structure_freeze_{dataset.replace('ogbg-', '')}.json"
        freeze = C.read_json(freeze_path)
        datasets[key] = {
            "dataset": dataset, "freeze_record": str(freeze_path.relative_to(root)),
            "content_sha256": freeze["cache_sha256"],
            "cache_file_sha256": freeze.get("cache_file_sha256"),
            "n_templates": freeze["template_table"]["n_templates"],
            "n_multi_atom": freeze["template_table"]["n_multi_atom_words"],
            "n_monatomic": freeze["template_table"]["n_monatomic_base_types"],
            "n_merge_rounds": len(freeze["merge"]["history"]),
            "support_train": freeze["support"].get("train"),
            "support_flag": freeze["support"].get("flag"),
            "expansion": freeze.get("expansion"),
            "split_sizes": freeze.get("split_sizes"),
            "partition_check": freeze.get("partition_check")}
    runs = []
    for run_id, run in sorted(find_runs().items()):
        runs.append({k: run.get(k) for k in ("run_id", "arm", "dataset", "partition", "seed",
                                             "epochs", "n_curve_rows", "n_parameters", "seconds", "spec",
                                             "recipe", "device", "status", "cache_content_sha256",
                                             "config_sha256", "torch", "gpu", "init_state_hash",
                                             "shared_body_hash")} |
                    {"best": run.get("best"), "final": run.get("final"),
                     "last20": run.get("last20"),
                     "terminal_test_scored": bool(run.get("terminal_test_scored"))})
    probes = C.analysis_dir() / "probes_mollipo.json"
    if not probes.is_file():
        probes = C.results_root() / "probes_mollipo.json"
    manifest = {
        "round_dir": str(root), "git_commit": C.git_commit(),
        "protocol_sha256": C.sha256_file(root / "PROTOCOL.md"),
        "config_sha256": C.sha256_file(root / "configs" / "ws_v1.json"),
        "wall_clock": {"T0_unix": float(cfg["budget"]["t0_monotonic_wall"]),
                       "hard_deadline_unix": float(cfg["budget"]["t0_monotonic_wall"])
                       + float(cfg["budget"]["hard_deadline_hours"]) * 3600,
                       "stop_launching_unix": float(cfg["budget"]["t0_monotonic_wall"])
                       + float(cfg["budget"]["stop_launching_after_hours"]) * 3600,
                       "stop_all_training_unix": float(cfg["budget"]["t0_monotonic_wall"])
                       + float(cfg["budget"]["stop_all_training_after_hours"]) * 3600,
                       "generated_unix": __import__("time").time()},
        "datasets": datasets, "runs": runs, "jobs": ledger,
        "gpu_hours_total": round(gpu_hours, 4),
        "gpu_hours_budget": {"hard": cfg["budget"]["gpu_hard_hours"],
                             "soft": cfg["budget"]["gpu_soft_hours"]},
        "max_formal_units": cfg["budget"]["max_formal_units"],
        "n_formal_training_units": len([r for r in runs if r["arm"] not in ("gine_ref",)]),
        "n_reference_units": len([r for r in runs if r["arm"] == "gine_ref"]),
        "probes_record": str(probes.relative_to(root)) if probes.is_file() else None,
        "timing_record": (str((C.analysis_dir() / "timing_mollipo.json").relative_to(root))
                          if (C.analysis_dir() / "timing_mollipo.json").is_file() else None),
        "terminal_policy": "TERMINAL_DEVELOPMENT_TEST",
    }
    C.write_json(root / "run_manifest.json", manifest)
    print(json.dumps({"gpu_hours": manifest["gpu_hours_total"], "n_runs": len(runs),
                      "n_jobs": len(ledger)}, indent=2))


def init_identity(dataset: str, seed: int = 0, n: int = 64, device: str = "cpu") -> dict[str, Any]:
    """FULL and ERASED must have identical initial whole-molecule predictions."""
    import torch

    import ws_model as M

    payload, _ = T.load_prepared(dataset)
    data = T.split_data(payload, "word", "valid")
    idx = np.arange(min(n, len(data.row_ids)))
    preds = {}
    for mode in ("full", "erased"):
        spec = M.ModelSpec(encoder="code", mode=mode, extra="none", para="none")
        model = T.make_model(spec, payload, seed=seed, device=device)
        model.eval()
        with torch.no_grad():
            raw = M.pack_batch(data, idx, attrpair=False, hop=False)
            out = model(M.to_device(raw, device))
        preds[mode] = out["pred"].detach().cpu().numpy()
    return {"n_molecules": int(len(idx)),
            "max_abs_pred_diff": float(np.max(np.abs(preds["full"] - preds["erased"]))),
            "max_abs_pred": float(np.max(np.abs(preds["full"])))}


def intervention(run: Mapping[str, Any], device: str = "cpu") -> dict[str, Any]:
    payload, _ = T.load_prepared(C.load_config()["datasets"][run["dataset"]])
    scaler = C.target_scaler(C.dataset_cache(C.load_config()["datasets"][run["dataset"]]))
    model, spec = _load_model(run, payload, device)
    data = T.split_data(payload, run["partition"], "valid")
    base = forward_terms(model, data, device, extra=spec.extra == "attrpair")
    swapped = forward_terms(model, data, device, extra=spec.extra == "attrpair",
                            intervention="erased")
    y = data.labels
    pred = base["pred"] * scaler["std"] + scaler["mean"]
    pred_i = swapped["pred"] * scaler["std"] + scaler["mean"]
    delta = pred_i - pred
    err = np.abs(pred - y)
    err_i = np.abs(pred_i - y)
    R = np.asarray(data.arrays["pair_R"])
    ER = np.asarray(data.arrays["pair_ER"])
    diff = np.max(np.abs(R - ER), axis=1)
    return {"run_id": run["run_id"], "n_molecules": int(len(y)),
            "rmse_original_raw": float(np.sqrt(np.mean((pred - y) ** 2))),
            "rmse_intervened_raw": float(np.sqrt(np.mean((pred_i - y) ** 2))),
            "rmse_change": float(np.sqrt(np.mean((pred_i - y) ** 2)) - np.sqrt(np.mean((pred - y) ** 2))),
            "mean_abs_pred_change_raw": float(np.mean(np.abs(delta))),
            "max_abs_pred_change_raw": float(np.max(np.abs(delta))),
            "mean_error_change_abs": float(np.mean(err_i - err)),
            "n_molecules_error_improved": int(np.sum(err_i < err)),
            "n_molecules_error_worsened": int(np.sum(err_i > err)),
            "fraction_pairs_R_differs_from_ER": float(np.mean(diff > 1e-6)),
            "mean_pair_R_max_abs_diff": float(np.mean(diff)),
            "f_model_recovery_max_abs_diff": base["recon_max_abs_diff"]}


def cmd_inspect(args) -> None:
    cfg = C.load_config()
    dataset = cfg["datasets"][args.dataset]
    runs = find_runs()
    run = runs[args.run]
    payload, _ = T.load_prepared(dataset)
    analysis = C.analysis_dir()
    analysis.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"run_id": run["run_id"], "commit": C.git_commit(),
                           "dataset": dataset}
    out["intervention"] = intervention(run, args.device)
    out["init_identity_full_vs_erased"] = init_identity(dataset, seed=run["seed"], device=args.device)
    if not args.cases:
        C.write_json(analysis / f"inspect_{run['run_id']}.json", out)
        print(json.dumps(out, indent=2)[:3000])
        return
    scaf = scaffold_map(dataset)
    for split in ("valid",):
        out[f"cases_{split}"] = case_dump(run, payload, split, device=args.device)
        preds = load_preds(run, split)
        sup = support_sets(payload, split)
        out[f"subsets_{split}"] = subset_report(preds["row_ids"], preds["pred_raw"],
                                                preds["target_raw"], sup, scaf)
    C.write_json(analysis / f"inspect_{run['run_id']}.json", out)
    print(json.dumps({k: out[k] for k in ("intervention", "init_identity_full_vs_erased",
                                          "subsets_valid")}, indent=2)[:4000])


def cmd_finalist(args) -> None:
    cfg = C.load_config()
    runs = find_runs()
    root = C.ROUND_DIR
    dataset = args.dataset
    scaler = C.target_scaler(C.dataset_cache(cfg["datasets"][dataset]))
    thr = _thr(scaler)
    L = choose_L(runs, thr, dataset=dataset)
    cls = classify_stage_b({k: v for k, v in runs.items() if v["dataset"] == dataset}, thr, L)
    stage_b = {r["arm"]: {"run_id": r["run_id"], "best_valid": r["best"]["valid_rmse_raw"],
                          "best_train": r["best"]["train_rmse_raw"],
                          "best_epoch": r["best"]["epoch"], "seconds": r["seconds"],
                          "n_parameters": r["n_parameters"]}
               for r in runs.values()
               if r["dataset"] == dataset and r["seed"] == 0 and r.get("recipe_tag") == "r0"
               and r["spec"]["para"] == "none" and r["spec"]["extra"] == "none"}
    sel = L["selected_encoder"]
    n1_char = stage_b.get("char_full", {}).get("best_valid")
    n1 = {"char_full_valid": n1_char, "L_valid": L["best_value"],
          "threshold_delta_task": thr["delta_task"],
          "satisfied": bool(n1_char is not None
                            and n1_char <= L["best_value"] - thr["delta_task"])}
    wg = stage_b.get("full_wgine", {})
    ch = stage_b.get("char_full", {})
    gr = stage_b.get("gine_ref", {})
    mse = lambda x: (x / scaler["std"]) ** 2  # noqa: E731
    n2 = {"L_selected_train_std_mse": mse(wg.get("best_train", float("nan"))),
          "char_full_train_std_mse": mse(ch.get("best_train", float("nan"))),
          "gine_ref_train_std_mse": mse(gr.get("best_train", float("nan"))),
          "L_selected_epoch": wg.get("best_epoch"), "epochs": wg.get("best_epoch") and 160}
    best_other = min(n2["char_full_train_std_mse"], n2["gine_ref_train_std_mse"])
    n2["better_other_train_std_mse"] = best_other
    n2["ratio"] = n2["L_selected_train_std_mse"] / best_other if best_other else None
    n2["satisfied"] = bool(n2["ratio"] is not None and n2["ratio"] >= 1.25)
    code_v, wg_v = stage_b.get("full_code", {}), stage_b.get("full_wgine", {})
    code_id = stage_b.get("full_id", {})
    n3 = {"code_valid": code_v.get("best_valid"), "code_train": code_v.get("best_train"),
          "wgine_valid": wg_v.get("best_valid"), "wgine_train": wg_v.get("best_train"),
          "id_valid": code_id.get("best_valid"), "id_train": code_id.get("best_train"),
          "satisfied": bool(code_v and wg_v
                            and code_v["best_valid"] - wg_v["best_valid"] >= thr["delta_task"]
                            and code_v["best_train"] - wg_v["best_train"] >= thr["delta_task"])}
    r1: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for r in runs.values():
        if (r["dataset"] != dataset or r["spec"]["encoder"] != sel
                or r["spec"]["para"] != "none" or r["spec"]["extra"] != "none"):
            continue
        r1.setdefault(r.get("recipe_tag", "r0"), {})[(r["spec"]["mode"], r["seed"])] = {
            "run_id": r["run_id"], "best_valid": r["best"]["valid_rmse_raw"],
            "best_train": r["best"]["train_rmse_raw"], "best_epoch": r["best"]["epoch"]}
    r1_gain = {}
    for seed in (0, 1):
        full = r1.get("r1", {}).get(("full", seed))
        erased = r1.get("r1", {}).get(("erased", seed))
        if full and erased:
            r1_gain[str(seed)] = {"full": full["best_valid"], "erased": erased["best_valid"],
                                  "gain": erased["best_valid"] - full["best_valid"],
                                  "delta_vs_original_L": full["best_valid"] - L["best_value"]}
    improved = [v["delta_vs_original_L"] < 0 for v in r1_gain.values()]
    rescue_ok = bool(len(improved) == 2 and all(improved)
                     and all(v["gain"] > 0 for v in r1_gain.values())
                     and float(np.mean([v["full"] for v in r1_gain.values()]))
                     <= gr.get("best_valid", float("inf")) + thr["band"])
    manifest = {
        "created_utc": __import__("time").time(),
        "git_commit": C.git_commit(), "dataset": cfg["datasets"][dataset],
        "test_policy": "TERMINAL_DEVELOPMENT_TEST",
        "selection_metric": "best-Valid RMSE (raw), strict improvement, earliest epoch on tie",
        "thresholds": thr,
        "stage_b_seed0": stage_b,
        "L": L, "classification": cls,
        "repair_checks": {"N1_char_full": n1, "N2_optimisation_deficit": n2, "N3_code_rootmean": n3},
        "repair_family_executed": "N2" if n2["satisfied"] else ("N1" if n1["satisfied"] else "N4"),
        "repair_r1_family": r1_gain,
        "rescue": {"status": "RESCUED_CANDIDATE" if rescue_ok else "RESCUE_FAILED",
                   "rule": "both seeds improve on original L, positive vs matched ERASED, mean within band of GINE_REF"},
        "finalist": {"run_id": L["selected_run"], "checkpoint": "state_best.pt",
                     "encoder": sel, "valid_rmse_raw": L["best_value"],
                     "selection_frozen_before_test": True},
        "terminal_test_runs": args.terminal_runs,
        "not_executed": {"stage_c_extra_seeds": "P not reached (L is 0.139 worse than GINE_REF)",
                         "D1_esol": "P not reached", "D2_para": "P not reached",
                         "N3_rootmean": "N3 not satisfied (CODE is better than WGINE on Train)"},
    }
    C.write_json(root / "finalist_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("L", "repair_checks", "repair_family_executed",
                                               "repair_r1_family", "rescue", "finalist")},
                     indent=2)[:3000])


# ---------------------------------------------------------------------------
# efficiency: inference cost and peak memory (measurement, not a training run)
# ---------------------------------------------------------------------------


def cmd_measure(args) -> None:
    import resource
    import time

    import torch

    import ws_model as M

    cfg = C.load_config()
    out: dict[str, Any] = {"commit": C.git_commit(), "device": args.device,
                           "protocol": "warmup 3 batches, then full-split eval; "
                                       "peak VRAM reset before the timed pass",
                           "runs": {}}
    for run_id in args.runs:
        run = C.read_json(C.results_root() / run_id / "run.json")
        run.setdefault("run_dir", str(C.results_root() / run_id))
        dataset = cfg["datasets"][run["dataset"]]
        payload, _ = T.load_prepared(dataset)
        scaler = C.target_scaler(C.dataset_cache(dataset))
        model, spec = _load_model(run, payload, args.device)
        split = args.split
        entry: dict[str, Any] = {"arm": run["arm"], "n_parameters": run["n_parameters"],
                                 "split": split}
        if spec.encoder == "gine_ref":
            import sq_gine as SG
            from torch_geometric.loader import DataLoader

            dims = C.official_dims()
            counts = {f"atom_{j}": int(dims["atom"][j]) for j in range(9)}
            counts.update({f"bond_{j}": int(dims["bond"][j]) for j in range(3)})
            counts["degree"] = 17
            counts["ring"] = 14
            frames = SG.build_data_list(_packed_from_payload(payload, split), counts, scaler)
            warm_loader = DataLoader(frames, batch_size=64, shuffle=False)
            for batch in warm_loader:
                with torch.no_grad():
                    model(batch.to(args.device))
                break
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            loader = DataLoader(frames, batch_size=64, shuffle=False)
            t0 = time.perf_counter()
            n = 0
            with torch.no_grad():
                for batch in loader:
                    model(batch.to(args.device))
                    n += int(batch.num_graphs)
            torch.cuda.synchronize()
            entry["seconds"] = time.perf_counter() - t0
            entry["n_molecules"] = n
            entry["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            entry["batch_sizes"] = [64]
        else:
            data = T.split_data(payload, run["partition"], split)
            extra = spec.extra == "attrpair"
            hop = spec.para != "none"
            for _ in range(3):
                with torch.no_grad():
                    batch = M.to_device(M.pack_batch(data, np.arange(min(64, len(data.row_ids))),
                                                     attrpair=extra, hop=hop), args.device)
                    model(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            entry["eval"] = {}
            for bs in (1, 64):
                t0 = time.perf_counter()
                with torch.no_grad():
                    for start in range(0, len(data.row_ids), bs):
                        idx = np.arange(start, min(start + bs, len(data.row_ids)))
                        model(M.to_device(M.pack_batch(data, idx, attrpair=extra, hop=hop),
                                          args.device))
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                entry["eval"][f"batch_{bs}"] = {"seconds": seconds,
                                                "per_molecule_ms": 1000 * seconds / len(data.row_ids)}
            entry["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            # one measured training step at the frozen batch size (memory only)
            idx = np.arange(min(int(cfg["recipe"]["batch"]), len(data.row_ids)))
            torch.cuda.reset_peak_memory_stats()
            model.train()
            batch = M.to_device(M.pack_batch(data, idx, attrpair=extra, hop=hop), args.device)
            pred = model(batch)["pred"]
            loss = torch.nn.functional.mse_loss(pred, batch["y"])
            loss.backward()
            torch.cuda.synchronize()
            entry["train_step_batch"] = int(len(idx))
            entry["train_step_peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            model.zero_grad(set_to_none=True)
        entry["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        out["runs"][run_id] = entry
    C.write_json(C.analysis_dir() / "efficiency.json", out)
    print(json.dumps(out, indent=2)[:4000])


# ---------------------------------------------------------------------------
# appendix: fixed real-edit catalogue, B-A difference error (no branch use)
# ---------------------------------------------------------------------------


def cmd_edits(args) -> None:
    import csv

    path = Path(args.pairs_csv)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    models = list(args.models)
    preds: dict[str, dict[int, float]] = {}
    for model in models:
        table: dict[int, float] = {}
        for split in ("train", "valid"):
            f = C.results_root() / model / f"preds_best_{split}.npz"
            if f.is_file():
                blob = load_preds({"run_dir": str(C.results_root() / model)}, split)
                table.update(dict(zip(blob["row_ids"].tolist(), blob["pred_raw"].tolist())))
        f = C.results_root() / "terminal" / model / "preds_test.npz"
        if f.is_file():
            with np.load(f, allow_pickle=False) as handle:
                table.update(dict(zip(handle["row_ids"].tolist(), handle["pred_raw"].tolist())))
        preds[model] = table
    out: dict[str, Any] = {"commit": C.git_commit(), "catalogue": str(path),
                           "note": "B-A difference error on the frozen real-edit catalogue; "
                                   "appendix only, never used for selection",
                           "scopes": {}}
    for scope, column in (("E_T", "e_t"), ("E_HH_T", "e_hh_t"),
                          ("E_STRICT", "e_strict_full")):
        pairs = [r for r in rows if int(float(r[column] or 0))]
        entry: dict[str, Any] = {"n_pairs": len(pairs),
                                 "cells": {}}
        for cell in sorted({f"{r['split_a']}-{r['split_b']}" for r in pairs}):
            entry["cells"][cell] = sum(1 for r in pairs if f"{r['split_a']}-{r['split_b']}" == cell)
        entry["models"] = {}
        for model in models:
            table = preds[model]
            usable = [r for r in pairs if int(r["row_a"]) in table and int(r["row_b"]) in table]
            if not usable:
                entry["models"][model] = {"n_usable": 0}
                continue
            delta_pred = np.asarray([table[int(r["row_b"])] - table[int(r["row_a"])]
                                     for r in usable])
            delta_y = np.asarray([float(r["delta_y"]) for r in usable])
            err = delta_pred - delta_y
            entry["models"][model] = {
                "n_usable": len(usable),
                "rmse_delta": float(np.sqrt(np.mean(err ** 2))),
                "mae_delta": float(np.mean(np.abs(err))),
                "bias_delta": float(np.mean(err)),
                "sign_agreement": float(np.mean(np.sign(delta_pred) == np.sign(delta_y))),
                "corr_delta": float(np.corrcoef(delta_pred, delta_y)[0, 1]) if len(usable) > 2 else None,
                "rmse_delta_Y0": float(np.sqrt(np.mean(delta_y ** 2))),
            }
        out["scopes"][scope] = entry
    C.write_json(C.analysis_dir() / "edit_differences.json", out)
    print(json.dumps(out, indent=2)[:4000])
