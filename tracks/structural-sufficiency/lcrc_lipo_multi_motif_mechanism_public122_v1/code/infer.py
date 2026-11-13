"""Stage B: frozen-checkpoint inference over every rewired graph (zero training).

Two frozen parameter states, three tasks:

``scopes_trained``
    TRAINED (``A_s0``) predictions for every graph under ``FULL`` and
    ``G_FIXED`` (local inputs from the rebuilt graph, raw global42 restored to
    the molecule's own O value).
``scopes_init``
    INIT (verified training-start weights) predictions for every graph under
    ``G_FIXED`` plus the O graph.
``internal_trained``
    TRAINED only: the eight-factor internal-path experiment.  ``local_input``
    (the tensor entering ``LocalMLP``) is split into three frozen groups

    ``G1`` all first-order aggregation blocks (``b_sum`` and the typed-neighbour
         table; C1 has no ``b_mean`` -- the slice map is derived from the model,
         never hard-coded),
    ``G2`` every S2 aggregation block (type / end / joint / closure / content),
    ``G3`` the nine atom-scalar columns,

    with the AF51 encoding block always taken from the **O** view (AF51 is
    atom-wise identical by construction).  For every real T / K0 / C graph the
    seven non-empty subsets ``S`` of ``{1,2,3}`` are injected through a read-only
    ``forward_pre_hook`` on ``LocalMLP``; weights, LayerNorm, readout, pooling and
    the global block are untouched and dropout is disabled (``eval``).  ``S``
    empty must reproduce the O prediction and ``S = 123`` must reproduce the real
    ``G_FIXED`` prediction of that graph (checked to 1e-5).

Inference only: ``eval()`` + ``torch.inference_mode``, frozen weights, frozen
Train-only normalisation, zero optimiser, zero backward, zero new seed.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import families as FAM
import graphlib as G
import feats as F
import rewire as R

SUBSETS: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (2,),
    (3,),
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 2, 3),
)


# ---------------------------------------------------------------------------
# frozen-input gate
# ---------------------------------------------------------------------------


def verify_freeze(analysis_dir: Path) -> dict[str, Any]:
    freeze = C.read_json(analysis_dir / "freeze.json")
    observed: dict[str, str] = {}
    missing: list[str] = []
    for name, want in (freeze.get("files") or {}).items():
        path = analysis_dir / name
        got = C.sha256_file(path) if path.exists() else "missing"
        observed[name] = got
        if got != want:
            missing.append(name)
    return {"pass": not missing, "mismatched": missing, "observed": observed, "frozen": freeze.get("files") or {}}


# ---------------------------------------------------------------------------
# local-input group map (derived from the model, never hard-coded)
# ---------------------------------------------------------------------------


def local_group_slices(model) -> dict[str, Any]:
    """Column slices of AF51 / G1 / G2 / G3 inside the C1 ``local_input`` tensor."""
    local = model.local
    enabled: list[str] = ["a", "b_sum"]
    if local.b_mean:
        enabled.append("b_mean")
    if local.s1_joint:
        enabled.append("s1_joint")
    for flag, name in (
        (local.s2_type, "s2_type"),
        (local.s2_end, "s2_end"),
        (local.s2_joint, "s2_joint"),
        (local.s2_closure, "s2_closure"),
        (local.s2_content, "s2_content"),
    ):
        if flag:
            enabled.append(name)
    enabled.append("atom_scalar")
    field_map: dict[str, list[int]] = {}
    pos = 0
    n_scalars = len(local.scalar_cols)
    for name in enabled:
        lo, hi = C.DC.LOCAL_LAYOUT[name]
        width = (hi - lo) if name != "atom_scalar" else n_scalars
        field_map[name] = list(range(pos, pos + width))
        pos += width
    if pos != len(local.local_cols) or pos != model.local.mlp.lins[0].in_features:
        raise RuntimeError(
            f"local-input group map width {pos} != frozen local width {len(local.local_cols)} "
            f"/ MLP in_features {model.local.mlp.lins[0].in_features}"
        )
    def span(names: Sequence[str]) -> list[int]:
        cols: list[int] = []
        for n in names:
            cols.extend(field_map[n])
        return cols

    g1 = span([n for n in ("b_sum", "b_mean", "s1_joint") if n in field_map])
    g2 = span([n for n in ("s2_type", "s2_end", "s2_joint", "s2_closure", "s2_content") if n in field_map])
    g3 = span(["atom_scalar"])
    af = span(["a"])
    groups = {"1": g1, "2": g2, "3": g3}
    covered = sorted(af + g1 + g2 + g3)
    if covered != list(range(pos)):
        raise RuntimeError("local-input groups do not partition the frozen layout")
    return {
        "field_map": {k: [int(v[0]), int(v[-1]) + 1] for k, v in field_map.items()},
        "groups": {k: [int(v[0]), int(v[-1]) + 1] for k, v in groups.items() if v},
        "af_columns": [int(af[0]), int(af[-1]) + 1],
        "n_columns": int(pos),
        "n_scalars": int(n_scalars),
        "group_members": {
            "1": [n for n in ("b_sum", "b_mean", "s1_joint") if n in field_map],
            "2": [n for n in ("s2_type", "s2_end", "s2_joint", "s2_closure", "s2_content") if n in field_map],
            "3": ["atom_scalar"],
        },
    }


class _LocalInputOverride:
    """Read-only ``forward_pre_hook`` swapping the LocalMLP input tensor."""

    def __init__(self, module):
        self.value: torch.Tensor | None = None
        self.calls = 0
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if self.value is None:
            return None
        self.calls += 1
        return (self.value,) + tuple(args[1:])

    def close(self) -> None:
        self._handle.remove()


def mixed_local_input(x_o: torch.Tensor, x_v: torch.Tensor, groups: Mapping[str, list[int]], subset: Sequence[int]) -> torch.Tensor:
    """AF51 and every non-selected group from O; selected groups from the view."""
    out = x_o.clone()
    for gid in subset:
        lo, hi = groups[str(gid)]
        out[:, lo:hi] = x_v[:, lo:hi]
    return out


# ---------------------------------------------------------------------------
# prediction helpers
# ---------------------------------------------------------------------------


def predict_batch(model, cache: Mapping[str, Any], idx: Sequence[int], *, device, o_global: Mapping[int, np.ndarray] | None, scopes: Sequence[str], return_local: bool = False):
    batch = C.DD.PREP_DATA.collate(cache, [int(i) for i in idx])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out: dict[str, np.ndarray] = {}
    local: torch.Tensor | None = None
    with torch.inference_mode():
        need_local = bool(return_local)
        if "FULL" in scopes or need_local:
            y, aux = model(batch, return_aux=True)
            if "FULL" in scopes:
                out["FULL"] = y.detach().float().cpu().numpy().reshape(-1)
            if need_local:
                local = aux["local_input"].detach().float().cpu()
        if "G_FIXED" in scopes:
            g = batch["global"].clone()
            for j, i in enumerate(idx):
                rid = int(np.asarray(cache["rows"])[int(i)])
                g[j] = torch.from_numpy(np.asarray(o_global[rid], dtype=np.float32).reshape(-1)).to(device)
            batch_fixed = dict(batch)
            batch_fixed["global"] = g
            out["G_FIXED"] = model(batch_fixed).detach().float().cpu().numpy().reshape(-1)
    return out, local, batch


# ---------------------------------------------------------------------------
# structural re-verification on the exact tensors fed to the model
# ---------------------------------------------------------------------------


def verify_graphs(
    *,
    graph_rows: Sequence[Mapping[str, str]],
    split: str,
    cache: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    train_min: np.ndarray,
    train_max: np.ndarray,
) -> dict[str, Any]:
    idx_by_key = {str(k): i for i, k in enumerate(cache["view_ids"])}
    atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
    bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
    glob = np.asarray(cache["global"], dtype=np.float64)
    o_index = {
        int(r["row_id"]): idx_by_key[str(r["graph_id"])] for r in graph_rows if str(r["kind"]) == "O"
    }
    o_atoms = {int(r["row_id"]): int(atom_counts[idx_by_key[str(r["graph_id"])]]) for r in graph_rows if str(r["kind"]) == "O"}
    o_bonds = {int(r["row_id"]): int(bond_counts[idx_by_key[str(r["graph_id"])]]) for r in graph_rows if str(r["kind"]) == "O"}
    failures: list[dict[str, Any]] = []
    af51_failures: list[dict[str, Any]] = []
    count_failures: list[dict[str, Any]] = []
    comp_bad: list[str] = []
    in_range: dict[str, bool] = {}
    max_af51 = 0.0
    max_scalar = 0.0
    n_checked = 0
    for r in graph_rows:
        if str(r["split"]) != split:
            continue
        gid = str(r["graph_id"])
        i = idx_by_key[gid]
        rid = int(r["row_id"])
        ok_range, _bad = _range_verdict(glob[i], train_min, train_max)
        in_range[gid] = bool(ok_range)
        if int(np.asarray(cache["n_components"])[i]) != 1:
            comp_bad.append(gid)
        if str(r["kind"]) == "O":
            continue
        n_checked += 1
        remove = _parse_bonds(str(r["remove"]))
        add = _parse_bonds(str(r["add"]))
        g0 = G.summarise_raw(raw_rows[rid], rid)
        base = F.row_from_artifact({**dict(raw_rows[rid]), "mol_index": rid})
        view_row, _info = R.edit_row(base, remove, add)
        insts = FAM_instances_for_removed(
            g0, remove, add, kind=str(r["kind"]), families=[f for f in str(r.get("families", "")).split(",") if f]
        )
        if not insts:
            failures.append({"graph_id": gid, "reason": "instance_not_found"})
            continue
        chk = R.verify_view_structure(g0, view_row, remove=remove, add=add, kind=str(r["kind"]), inst=insts)
        if not chk["pass"]:
            failures.append({"graph_id": gid, "kind": r["kind"], "failed": [k for k, v in chk.items() if v is False]})
        dif = F.af51_difference(cache, o_index[rid], i) if o_index else {}
        if dif:
            max_af51 = max(max_af51, float(dif["max_abs_atom_feat_diff"]))
            max_scalar = max(max_scalar, float(dif["max_abs_atom_scalar_diff"]))
            if not dif["atom_feat_equal"] or not dif["atom_type_equal"]:
                af51_failures.append({"graph_id": gid, **dif})
        if int(atom_counts[i]) != o_atoms[rid] or int(bond_counts[i]) != o_bonds[rid]:
            count_failures.append({"graph_id": gid, "atoms": int(atom_counts[i]), "bonds": int(bond_counts[i])})
    return {
        "n_graphs_checked": int(n_checked),
        "structural_failures": failures[:20],
        "n_structural_failures": len(failures),
        "af51_failures": af51_failures[:10],
        "n_af51_failures": len(af51_failures),
        "af51_max_abs_atom_feat_diff": max_af51,
        "atom_scalar_max_abs_diff": max_scalar,
        "input_count_failures": count_failures,
        "n_components_not_one": comp_bad,
        "n_graphs_out_of_train_range": int(sum(1 for v in in_range.values() if not v)),
        "pass": not failures and not af51_failures and not count_failures and not comp_bad,
        "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
    }


def FAM_instances_for_removed(
    g0: G.RawGraph,
    remove: Sequence[tuple[int, int]],
    add: Sequence[tuple[int, int]] = (),
    kind: str = "T",
    families: Sequence[str] = (),
):
    """Recover the unique instance that produced a rewired graph (exact bond-set match).

    ``T`` removes ``center-target``, ``K0`` removes ``center-control``; the pair
    is re-built from the catalogue and must reproduce the removed **and** added
    bond sets exactly.  Exactly one instance must match, otherwise the graph
    cannot be re-verified and the check fails instead of guessing.
    """
    rem = _bond_set(remove)
    addset = _bond_set(add)
    fams = [f for f in (families or C.FAMILIES) if f in C.FAMILIES]
    found: list[Any] = []
    for fam in fams:
        for inst in FAM.main_instances(g0, fam):
            anchor = (inst.center, inst.target) if kind == "T" else (inst.center, inst.control)
            if _bond_set([anchor]) - rem:
                continue
            for donor in R.donor_bonds(g0, inst):
                pair = R.build_pair(g0, inst, donor)
                got = pair.t_remove if kind == "T" else pair.k_remove
                got_add = pair.t_add if kind == "T" else pair.k_add
                if _bond_set(got) == rem and _bond_set(got_add) == addset:
                    found.append(inst)
    # Several controls of the *same* target instance give the same T graph; the
    # T-side verification does not depend on which control was used.
    uniq = {(i.family, i.center, i.target): i for i in found}
    if len(uniq) == 1:
        return next(iter(uniq.values()))
    return None


def _bond_set(bonds: Sequence[Sequence[int]]) -> set[frozenset[int]]:
    return {frozenset((int(a), int(b))) for a, b in bonds}


def _range_verdict(vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[bool, list[int]]:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    bad = [int(i) for i in np.flatnonzero((v < lo - C.RANGE_TOL) | (v > hi + C.RANGE_TOL))]
    return (not bad), bad


def _parse_bonds(code: str) -> list[tuple[int, int]]:
    code = str(code).strip()
    if not code:
        return []
    out = []
    for chunk in code.split("."):
        a, b = chunk.split("-")
        out.append((int(a), int(b)))
    return out


# ---------------------------------------------------------------------------
# frozen baseline reproduction
# ---------------------------------------------------------------------------


def baseline_check(
    preds: Mapping[int, float], *, target_norm: Mapping[str, float], previous: Mapping[str, Any]
) -> dict[str, Any]:
    """O-view reproduction against the executed previous round (row-id keyed)."""
    y = previous["y"]
    split = previous["split"]
    o_pred = previous["o_pred"]
    per_split: dict[str, Any] = {}
    for name in ("valid", "test"):
        rows, raw, ref = [], [], []
        for rid_i, value in sorted(preds.items()):
            if split.get(int(rid_i)) != name:
                continue
            rows.append(int(rid_i))
            raw.append(float(target_norm["mean"] + target_norm["scale"] * float(value)))
            ref.append(float(o_pred[int(rid_i)]))
        if not rows:
            continue
        raw_arr = np.asarray(raw)
        yy = np.asarray([float(y[r]) for r in rows])
        ref_arr = np.asarray(ref)
        rmse = float(np.sqrt(np.mean((raw_arr - yy) ** 2)))
        mae = float(np.mean(np.abs(raw_arr - yy)))
        ref_rmse = {"valid": C.REFERENCE["valid_rmse"], "test": C.REFERENCE["test_rmse"]}[name]
        per_split[name] = {
            "n": len(rows),
            "rmse_raw": rmse,
            "reference_rmse": ref_rmse,
            "rmse_diff": rmse - ref_rmse,
            "mae_raw": mae,
            "mae_diff": (mae - C.REFERENCE["test_mae"]) if name == "test" else None,
            "max_abs_pred_diff_vs_previous_round": float(np.abs(raw_arr - ref_arr).max()),
            "rows_sha256": C.row_sha256(rows),
        }
    ok = True
    for name, d in per_split.items():
        ok = ok and abs(d["rmse_diff"]) <= C.REPRO_TOL
        ok = ok and d["max_abs_pred_diff_vs_previous_round"] <= C.REPRO_TOL
        if d["mae_diff"] is not None:
            ok = ok and abs(d["mae_diff"]) <= C.REPRO_TOL
    return {"per_split": per_split, "tolerance": C.REPRO_TOL, "pass": bool(ok)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tasks", default="scopes_trained")
    ap.add_argument("--splits", default="valid,test")
    args = ap.parse_args(argv)
    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    analysis_dir = Path(args.analysis_dir)
    C.log(f"infer: device={args.device} tasks={tasks} splits={splits}")

    gate_freeze = verify_freeze(analysis_dir)
    if not gate_freeze["pass"]:
        raise SystemExit(f"FROZEN_INPUT_MISMATCH: {gate_freeze['mismatched']}")
    manifest = C.read_json(analysis_dir / "inputs_manifest.json")
    train_min = np.asarray(manifest["train_global_range"]["min"], dtype=np.float64)
    train_max = np.asarray(manifest["train_global_range"]["max"], dtype=np.float64)
    graph_rows = read_csv(analysis_dir / "graph_manifest.csv")
    raw_rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    previous = C.load_previous_inputs()
    spec = C.DD.frozen_spec()
    ck = C.load_checkpoint()
    tgt = C.target_norm(ck)

    t_start = time.perf_counter()
    checks: dict[str, Any] = {
        "freeze": gate_freeze,
        "round_id": C.ROUND_ID,
        "tasks": tasks,
        "splits": splits,
        "pass": True,
        "views": {},
        "internal": {},
    }
    runtime: dict[str, Any] = {"round_id": C.ROUND_ID, "stage": "infer", "tasks": tasks, "timings_seconds": {}}

    # ---- trained model (shared by scopes_trained and internal_trained) --
    need_trained = any(t.startswith("scopes_trained") or t.startswith("internal_trained") for t in tasks)
    if need_trained:
        model, spec = C.build_frozen_model(ck, device=args.device)
        state_before = C.DM.state_fingerprint(model.state_dict())
        norm_before = C.DD.PREP_DATA.norm_fingerprint(C.frozen_norm(ck))
        C.log(f"TRAINED model ready: {C.VARIANT} params={C.EXPECTED_PARAMS} state={C.STATE_HASH[:12]}")
        group_map = local_group_slices(model)
        checks["local_input_groups"] = group_map
        override = _LocalInputOverride(model.local.mlp)
    else:
        model = None
        state_before = norm_before = None
        group_map = None
        override = None

    if "scopes_trained" in tasks:
        rows_out: list[dict[str, Any]] = []
        for split in splits:
            t0 = time.perf_counter()
            entries, kept = build_split_cache(analysis_dir, graph_rows, split, raw_rows, spec)
            cache = F.build_cache(entries, spec, split=f"multi_motif_views_{split}")
            idx_by = {gid: i for i, (gid, _row) in enumerate(entries)}
            o_global = {
                int(np.asarray(cache["rows"])[i]): np.asarray(cache["global"][i], dtype=np.float32)
                for i, (gid, _row) in enumerate(entries)
                if gid in set(r["graph_id"] for r in kept if r["kind"] == "O")
            }
            order = list(range(len(entries)))
            preds: dict[str, dict[str, np.ndarray]] = {k: {} for k in ("FULL", "G_FIXED")}
            n_done = 0
            for start in range(0, len(order), args.batch_size):
                idx = order[start : start + args.batch_size]
                out, _local, _batch = predict_batch(
                    model, cache, idx, device=args.device, o_global=o_global, scopes=("FULL", "G_FIXED")
                )
                for scope, arr in out.items():
                    preds[scope][start] = arr
                n_done += len(idx)
            full = np.concatenate([preds["FULL"][s] for s in sorted(preds["FULL"])]) if preds["FULL"] else np.zeros(0)
            fixed = np.concatenate([preds["G_FIXED"][s] for s in sorted(preds["G_FIXED"])]) if preds["G_FIXED"] else np.zeros(0)
            atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
            bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
            meta = {str(r["graph_id"]): r for r in kept}
            if len(kept) != len(entries):
                raise SystemExit("GRAPH_CACHE_LENGTH_MISMATCH")
            for i, (gid, _row) in enumerate(entries):
                r = meta[gid]
                base = {
                    "state": "trained",
                    "split": split,
                    "graph_id": gid,
                    "row_id": int(r["row_id"]),
                    "kind": r["kind"],
                    "families": r["families"],
                    "remove": r["remove"],
                    "add": r["add"],
                    "scope": "FULL",
                    "pred_standardised": float(full[i]),
                    "pred_raw": float(tgt["mean"] + tgt["scale"] * float(full[i])),
                    "n_atoms": int(atom_counts[i]),
                    "n_bonds": int(bond_counts[i]),
                    "n_components": int(np.asarray(cache["n_components"])[i]),
                }
                rows_out.append(base)
                if r["kind"] != "O":
                    rows_out.append({**base, "scope": "G_FIXED", "pred_standardised": float(fixed[i]),
                                     "pred_raw": float(tgt["mean"] + tgt["scale"] * float(fixed[i]))})
            o_mask = np.asarray([r["kind"] == "O" for r in kept], dtype=bool)
            o_max = float(np.abs(full[o_mask] - fixed[o_mask]).max()) if o_mask.any() else 0.0
            blk = verify_graphs(
                graph_rows=kept, split=split, cache=cache, raw_rows=raw_rows, train_min=train_min, train_max=train_max
            )
            blk["o_view_max_abs_diff_full_vs_gfixed"] = o_max
            blk["o_view_gfixed_equals_full_within_tol"] = bool(o_max <= C.REPRO_TOL)
            blk["pass"] = bool(blk["pass"] and o_max <= C.REPRO_TOL)
            checks["views"][split] = blk
            checks["pass"] = bool(checks["pass"] and blk["pass"])
            runtime["timings_seconds"][f"scopes_trained_{split}"] = float(time.perf_counter() - t0)
            C.log(f"scopes_trained {split}: {len(entries)} graphs, o_max={o_max:.2e}, pass={blk['pass']}")
        write_csv(
            analysis_dir / "per_graph_predictions.csv",
            rows_out,
            ["state", "split", "graph_id", "row_id", "kind", "families", "remove", "add", "scope",
             "pred_standardised", "pred_raw", "n_atoms", "n_bonds", "n_components"],
        )
        o_full = {int(r["row_id"]): r["pred_standardised"] for r in rows_out if r["kind"] == "O" and r["scope"] == "FULL"}
        base = baseline_check(o_full, target_norm=tgt, previous=previous)
        checks["baseline"] = base
        checks["pass"] = bool(checks["pass"] and base["pass"])
        C.log(f"baseline reproduction pass={base['pass']}")

    # ---- INIT state (G_FIXED only) -------------------------------------
    if "scopes_init" in tasks:
        init_model, spec_i, init_info = C.build_init_model(device=args.device)
        init_before = C.DM.state_fingerprint(init_model.state_dict())
        rows_out = []
        for split in splits:
            t0 = time.perf_counter()
            entries, kept = build_split_cache(analysis_dir, graph_rows, split, raw_rows, spec_i)
            cache = F.build_cache(entries, spec_i, split=f"multi_motif_views_{split}")
            o_global = {
                int(np.asarray(cache["rows"])[i]): np.asarray(cache["global"][i], dtype=np.float32)
                for i, (gid, _row) in enumerate(entries)
                if gid in set(r["graph_id"] for r in kept if r["kind"] == "O")
            }
            arrs = []
            for start in range(0, len(entries), args.batch_size):
                idx = list(range(start, min(start + args.batch_size, len(entries))))
                out, _local, _batch = predict_batch(
                    init_model, cache, idx, device=args.device, o_global=o_global, scopes=("G_FIXED",)
                )
                arrs.append(out["G_FIXED"])
            fixed = np.concatenate(arrs) if arrs else np.zeros(0)
            atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
            bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
            meta = {str(r["graph_id"]): r for r in kept}
            for i, (gid, _row) in enumerate(entries):
                r = meta[gid]
                rows_out.append(
                    {
                        "state": "init",
                        "split": split,
                        "graph_id": gid,
                        "row_id": int(r["row_id"]),
                        "kind": r["kind"],
                        "families": r["families"],
                        "remove": r["remove"],
                        "add": r["add"],
                        "scope": "G_FIXED",
                        "pred_standardised": float(fixed[i]),
                        "pred_raw": float(tgt["mean"] + tgt["scale"] * float(fixed[i])),
                        "n_atoms": int(atom_counts[i]),
                        "n_bonds": int(bond_counts[i]),
                        "n_components": int(np.asarray(cache["n_components"])[i]),
                    }
                )
            # NON-CONSTANT check on the actual O views (a degenerate init would
            # make the whole parameter-dependence test uninformative)
            o_idx = [i for i, (gid, _r) in enumerate(entries) if meta[gid]["kind"] == "O"]
            o_vals = fixed[o_idx]
            checks["views"].setdefault(split, {})["init_O_pred_std"] = float(np.std(o_vals))
            checks["views"][split]["init_O_pred_constant"] = bool(float(np.std(o_vals)) < 1e-9)
            runtime["timings_seconds"][f"scopes_init_{split}"] = float(time.perf_counter() - t0)
            C.log(f"scopes_init {split}: {len(entries)} graphs, O std={float(np.std(o_vals)):.4f}")
        write_csv(
            analysis_dir / "per_graph_predictions_init.csv",
            rows_out,
            ["state", "split", "graph_id", "row_id", "kind", "families", "remove", "add", "scope",
             "pred_standardised", "pred_raw", "n_atoms", "n_bonds", "n_components"],
        )
        init_after = C.DM.state_fingerprint(init_model.state_dict())
        checks["init_state_identity"] = {
            "hash_before": init_before,
            "hash_after": init_after,
            "expected": C.INIT_STATE_HASH,
            "unchanged": bool(init_before == init_after == C.INIT_STATE_HASH),
            "label": init_info.get("matches_recorded"),
        }
        checks["pass"] = bool(checks["pass"] and checks["init_state_identity"]["unchanged"])

    # ---- internal eight-cell decomposition (TRAINED only) --------------
    if "internal_trained" in tasks:
        internal_rows = read_internal_graph_ids(analysis_dir)
        rows_out = []
        for split in splits:
            t0 = time.perf_counter()
            entries, kept = build_split_cache(analysis_dir, graph_rows, split, raw_rows, spec)
            cache = F.build_cache(entries, spec, split=f"multi_motif_views_{split}")
            idx_by = {gid: i for i, (gid, _row) in enumerate(entries)}
            o_by_row = {
                int(r["row_id"]): idx_by[str(r["graph_id"])] for r in kept if str(r["kind"]) == "O"
            }
            kind_by_gid = {str(r["graph_id"]): str(r["kind"]) for r in kept}
            targets = [g for g in internal_rows.get(split, []) if g in idx_by]
            targets.sort(key=lambda g: (int(str(g).split(":")[0]), g))
            by_mol: dict[int, list[str]] = {}
            for g in targets:
                by_mol.setdefault(int(str(g).split(":")[0]), []).append(g)
            stats = {"n_views": 0, "max_abs_s_empty_vs_O": 0.0, "max_abs_s_full_vs_gfixed": 0.0}
            for rid in sorted(by_mol):
                batch_ids = by_mol[rid]
                idx = [idx_by[g] for g in batch_ids]
                o_i = o_by_row[rid]
                _out_v, x_v, batch_v = predict_batch(
                    model, cache, idx, device=args.device, o_global=None, scopes=(), return_local=True
                )
                _out_o, x_o, _batch_o = predict_batch(
                    model, cache, [o_i], device=args.device, o_global=None, scopes=("FULL",), return_local=True
                )
                x_o = x_o.repeat(len(batch_ids), 1)
                if x_v.shape != x_o.shape:
                    raise RuntimeError(
                        f"internal: O and view local inputs have different atom rows ({tuple(x_o.shape)} vs {tuple(x_v.shape)})"
                    )
                p0 = float(_out_o["FULL"][0])
                fixed_out, _l, _b = predict_batch(
                    model,
                    cache,
                    idx,
                    device=args.device,
                    o_global={rid: np.asarray(cache["global"][o_i], dtype=np.float32)},
                    scopes=("G_FIXED",),
                )
                p_fixed = fixed_out["G_FIXED"]
                x_o = x_o.to(args.device)
                x_v = x_v.to(args.device)
                # global42 is frozen at the molecule's own O value for every cell
                g_tile = torch.from_numpy(
                    np.tile(np.asarray(cache["global"][o_i], dtype=np.float32), (len(batch_ids), 1))
                ).to(args.device)
                batch_internal = dict(batch_v)
                batch_internal["global"] = g_tile
                for subset in SUBSETS:
                    mixed = mixed_local_input(x_o, x_v, group_map["groups"], subset)
                    override.value = mixed
                    with torch.inference_mode():
                        y = model(batch_internal)
                    override.value = None
                    arr = y.detach().float().cpu().numpy().reshape(-1)
                    for j, g in enumerate(batch_ids):
                        rows_out.append(
                            {
                                "state": "trained",
                                "split": split,
                                "graph_id": g,
                                "row_id": rid,
                                "kind": kind_by_gid.get(g, ""),
                                "subset": "".join(str(x) for x in subset) or "empty",
                                "pred_standardised": float(arr[j]),
                                "pred_raw": float(tgt["mean"] + tgt["scale"] * float(arr[j])),
                                "pred_O_standardised": float(p0),
                                "pred_gfixed_standardised": float(p_fixed[j]),
                            }
                        )
                    if subset == ():
                        stats["max_abs_s_empty_vs_O"] = max(
                            stats["max_abs_s_empty_vs_O"], float(np.abs(arr - float(p0)).max()) if arr.size else 0.0
                        )
                    if subset == (1, 2, 3):
                        stats["max_abs_s_full_vs_gfixed"] = max(
                            stats["max_abs_s_full_vs_gfixed"], float(np.abs(arr - p_fixed).max()) if arr.size else 0.0
                        )
                stats["n_views"] += len(batch_ids)
            blk = dict(stats)
            blk["tolerance"] = C.MOBIUS_TOL
            blk["pass"] = bool(
                stats["max_abs_s_empty_vs_O"] <= C.MOBIUS_TOL
                and stats["max_abs_s_full_vs_gfixed"] <= C.MOBIUS_TOL
            )
            checks["internal"][split] = blk
            checks["pass"] = bool(checks["pass"] and blk["pass"])
            runtime["timings_seconds"][f"internal_{split}"] = float(time.perf_counter() - t0)
            C.log(
                f"internal {split}: {stats['n_views']} views, "
                f"max|S=empty - O|={stats['max_abs_s_empty_vs_O']:.2e}, "
                f"max|S=123 - G_FIXED|={stats['max_abs_s_full_vs_gfixed']:.2e}"
            )
        write_csv(
            analysis_dir / "per_graph_internal.csv",
            rows_out,
            ["state", "split", "graph_id", "row_id", "kind", "subset", "pred_standardised", "pred_raw",
             "pred_O_standardised", "pred_gfixed_standardised"],
        )

    # ---- state identity -------------------------------------------------
    if need_trained:
        state_after = C.DM.state_fingerprint(model.state_dict())
        norm_after = C.DD.PREP_DATA.norm_fingerprint(C.frozen_norm(ck))
        if override is not None:
            override.close()
        checks["state_identity"] = {
            "state_hash_before": state_before,
            "state_hash_after": state_after,
            "expected_state_hash": C.STATE_HASH,
            "norm_fingerprint_before": norm_before,
            "norm_fingerprint_after": norm_after,
            "unchanged": bool(state_before == state_after == C.STATE_HASH and norm_before == norm_after),
            "model_training_steps": 0,
            "optimizer_created": False,
            "backward_calls": 0,
            "local_input_override_calls": int(override.calls) if override is not None else 0,
        }
        checks["pass"] = bool(checks["pass"] and checks["state_identity"]["unchanged"])

    total = float(time.perf_counter() - t_start)
    runtime.update(
        {
            "device": str(args.device),
            "environment": C.env_info(),
            "git_commit": C.git_commit(),
            "git_dirty": C.git_dirty(),
            "wall_seconds": total,
            "gpu_hours": float(total / 3600.0) if str(args.device).startswith("cuda") else 0.0,
        }
    )
    C.write_json(analysis_dir / f"runtime_{'-'.join(tasks)}.json", runtime)
    checks["runtime"] = runtime
    C.write_json(analysis_dir / f"infer_checks_{'-'.join(tasks)}.json", checks)
    C.log(f"infer done in {total:.1f}s; pass={checks['pass']}")
    return 0 if checks["pass"] else 2


def read_internal_graph_ids(analysis_dir: Path) -> dict[str, list[str]]:
    """Graphs that take part in the internal-path experiment (main pairs + C pool)."""
    pairs = read_csv(analysis_dir / "pair_catalog.csv")
    cal = read_csv(analysis_dir / "caliper_catalog.csv")
    out: dict[str, set[str]] = {"valid": set(), "test": set()}
    for r in pairs:
        if int(float(r["pop_main"] or 0)) != 1:
            continue
        out[r["split"]].add(str(r["t_graph_id"]))
        out[r["split"]].add(str(r["k_graph_id"]))
    for r in cal:
        out[r["split"]].add(str(r["c_graph_id"]))
    return {k: sorted(v) for k, v in out.items()}


def build_split_cache(
    analysis_dir: Path,
    graph_rows: Sequence[Mapping[str, str]],
    split: str,
    raw_rows: Sequence[Mapping[str, Any]],
    spec,
) -> tuple[list[tuple[str, dict[str, Any]]], list[Mapping[str, str]]]:
    kept = [r for r in graph_rows if str(r["split"]) == split]
    kept.sort(key=lambda r: (int(r["row_id"]), 0 if str(r["kind"]) == "O" else 1, str(r["kind"]), str(r["graph_id"])))
    entries: list[tuple[str, dict[str, Any]]] = []
    for r in kept:
        rid = int(r["row_id"])
        base = F.row_from_artifact({**dict(raw_rows[rid]), "mol_index": rid})
        base["n_components"] = G.summarise_raw(raw_rows[rid], rid).n_components
        if str(r["kind"]) == "O":
            entries.append((str(r["graph_id"]), base))
        else:
            cut, _info = R.edit_row(base, _parse_bonds(str(r["remove"])), _parse_bonds(str(r["add"])))
            entries.append((str(r["graph_id"]), cut))
    return entries, kept


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
    return p


if __name__ == "__main__":
    raise SystemExit(main())
