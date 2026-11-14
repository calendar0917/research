"""Structural groups (H1/H2/H3) and frozen-D2 source interventions.

Groups are defined *before* any prediction is read (see ``PROTOCOL.md``):

``H1``  a root atom whose radius-2 Morgan environment is absent from the whole
        official Train split (Train support 0)
``H2``  a root atom whose radius-2 environment is supported by >= 5 Train
        molecules while none of its radius-3 environments is supported at all
``H3``  atoms matched by the single SMARTS pattern ``[NX3]-[CX3](=[OX1])``

The environment support is a hashed-environment proxy (RDKit Morgan count
fingerprint, ``includeRedundantEnvironments=True``), not an exact isomorphism
statement.  A root with no valid radius-2 environment is reported as
``MISSING`` and never silently treated as unseen.

Group selection is Validation-only, on arm ``B``'s best-Val point, and freezes
before Test is read.  The interventions are internal edits of the prediction
function of a frozen molecule (layer-2 message source), never a retraining gain.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lp_common as C
import lp_data as D
import lp_analysis as A
import lp_model as M

import lcrc_data as PREP_DATA

GROUP_NAMES = ("H1", "H2", "H3")
GROUP_FLAGS = {"H1": "h1", "H2": "h2", "H3": "h3"}


# ---------------------------------------------------------------------------
# group tables
# ---------------------------------------------------------------------------


def membership_arrays(
    entries: Sequence[Mapping[str, Any]], split: str
) -> dict[str, Any]:
    rows = np.asarray([int(e["row"]) for e in entries], dtype=np.int64)
    heavy = np.asarray([int(e.get("n_heavy", 0)) for e in entries], dtype=np.int64)
    out = {"rows": rows, "heavy": heavy}
    for name in GROUP_NAMES:
        flag = GROUP_FLAGS[name]
        out[name] = np.asarray([bool(e.get(flag, False)) if "status" not in e else False for e in entries], dtype=bool)
        out[f"{name}_roots"] = {
            int(e["row"]): [int(a) for a in e.get(f"{flag}_roots", [])] for e in entries if "status" not in e
        }
    out["missing"] = {
        int(e["row"]): [int(a) for a in e.get("missing_r2_roots", [])] for e in entries if "status" not in e
    }
    out["n_missing_roots"] = int(sum(len(v) for v in out["missing"].values()))
    out["n_parse_failed"] = int(sum(1 for e in entries if e.get("status") == "PARSE_FAILED"))
    return out


def group_profile(
    errors: np.ndarray,
    membership: np.ndarray,
    heavy: np.ndarray,
    edges: Mapping[str, Any],
    train_marginal: np.ndarray,
) -> dict[str, Any]:
    profile = A.size_standardised_gap(errors, membership, heavy, edges, train_marginal)
    profile["n_group"] = int(membership.sum())
    profile["mae_group"] = float(errors[membership].mean()) if membership.any() else None
    profile["mae_other"] = float(errors[~membership].mean()) if (~membership).any() else None
    return profile


def group_bootstrap(
    errors: Mapping[str, np.ndarray],
    membership: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
    comparisons: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    """Scaffold-cluster bootstrap of the group MAE gap and of paired group deltas."""
    n_clusters = int(clusters.max()) + 1
    weights = rng.exponential(scale=1.0, size=(replicates, n_clusters))
    pos = membership
    neg = ~membership
    out: dict[str, Any] = {"n_group": int(pos.sum()), "n_other": int(neg.sum()), "models": {}}
    if pos.sum() == 0 or neg.sum() == 0:
        return {**out, "estimable": False}
    for arm, err in errors.items():
        gap = np.empty(replicates)
        for i in range(replicates):
            w = weights[i][clusters]
            wp, wn = w[pos], w[neg]
            if wp.sum() <= 0 or wn.sum() <= 0:
                gap[i] = np.nan
                continue
            gap[i] = float(np.sum(wp * err[pos]) / wp.sum() - np.sum(wn * err[neg]) / wn.sum())
        out["models"][arm] = {
            "gap_point": float(err[pos].mean() - err[neg].mean()),
            "gap_lo": float(np.nanpercentile(gap, 2.5)),
            "gap_hi": float(np.nanpercentile(gap, 97.5)),
        }
    out["paired_group_deltas"] = {}
    for name, (control, candidate) in comparisons.items():
        if control not in errors or candidate not in errors:
            continue
        ec, ed = errors[control], errors[candidate]
        delta = np.empty(replicates)
        for i in range(replicates):
            w = weights[i][clusters]
            wp, wn = w[pos], w[neg]
            if wp.sum() <= 0 or wn.sum() <= 0:
                delta[i] = np.nan
                continue
            before = float(np.sum(wp * ec[pos]) / wp.sum() - np.sum(wn * ec[neg]) / wn.sum())
            after = float(np.sum(wp * ed[pos]) / wp.sum() - np.sum(wn * ed[neg]) / wn.sum())
            delta[i] = before - after
        out["paired_group_deltas"][name] = {
            "point": float((ec[pos].mean() - ec[neg].mean()) - (ed[pos].mean() - ed[neg].mean())),
            "lo": float(np.nanpercentile(delta, 2.5)),
            "hi": float(np.nanpercentile(delta, 97.5)),
            "interpretation": "positive = candidate reduces the group gap more than the control",
        }
    out["estimable"] = True
    return out


def run_groups() -> dict[str, Any]:
    rows = A.S.load_split()["rows"]
    smiles = A.read_mol_csv_from_manifest()
    membership = A.group_membership(smiles, rows)
    arrays = {split: membership_arrays(membership["splits"][split], split) for split in C.SPLITS}

    train_heavy = arrays["train"]["heavy"]
    edges = A.size_bins(train_heavy)
    train_bins = A.assign_bins(train_heavy, edges)
    train_marginal = np.bincount(train_bins, minlength=3).astype(np.float64)
    train_marginal /= train_marginal.sum()

    arms = [arm for arm in A.MAIN_ARMS if (A.arm_dir(arm) / "metrics.json").is_file()]
    errors: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        for split in ("valid", "test"):
            pred = A.load_pred(arm, split, "best")
            if not np.array_equal(pred["rows"], arrays[split]["rows"]):
                raise ValueError(f"{arm}/{split}: prediction rows do not match the split row order")
            errors.setdefault(split, {})[arm] = np.abs(np.asarray(pred["raw"]) - np.asarray(pred["y_true"]))

    rng = np.random.default_rng(int(C.BOOTSTRAP_SEED) + 1)
    clusters = {
        split: A.scaffold_clusters([smiles[int(r)] for r in arrays[split]["rows"]])
        for split in ("valid", "test")
    }

    payload: dict[str, Any] = {
        "definitions": {
            "H1": "any valid radius-2 root with Train support 0",
            "H2": f"radius-2 support >= {C.GROUP_SUPPORT_MIN} and no valid radius-3 Train support",
            "H3": "SMARTS [NX3]-[CX3](=[OX1]) atom union",
            "support": "Morgan count fingerprint, radius 3, includeRedundantEnvironments; "
            "Train document frequency with leave-one-molecule-out",
            "morgan_radius_fine": list(C.MORGAN_RADIUS_FINE),
        },
        "support_summary": membership["support"],
        "size_bins": edges,
        "train_bin_marginal": train_marginal.tolist(),
        "n": {
            split: {
                "molecules": int(len(arrays[split]["rows"])),
                "missing_roots": arrays[split]["n_missing_roots"],
                "parse_failed": arrays[split]["n_parse_failed"],
                **{f"{name}_molecules": int(arrays[split][name].sum()) for name in GROUP_NAMES},
            }
            for split in C.SPLITS
        },
        "profiles": {},
        "bootstrap": {},
    }

    for name in GROUP_NAMES:
        payload["profiles"][name] = {}
        for split in ("valid", "test"):
            member = arrays[split][name]
            payload["profiles"][name][split] = {
                arm: group_profile(errors[split][arm], member, arrays[split]["heavy"], edges, train_marginal)
                for arm in arms
            }
        payload["bootstrap"][name] = {
            split: group_bootstrap(
                errors[split],
                arrays[split][name],
                clusters[split],
                replicates=int(C.BOOTSTRAP_REPLICATES),
                rng=np.random.default_rng(
                    int(C.BOOTSTRAP_SEED) + 100 * (GROUP_NAMES.index(name) + 1) + (0 if split == "valid" else 7)
                ),
                comparisons=A.CONTROLS,
            )
            for split in ("valid", "test")
        }

    # ---- Validation-only selection on arm B, then Test -------------------
    selection: dict[str, Any] = {"arm": "B", "split": "valid", "group": None, "candidates": {}}
    for name in GROUP_NAMES:
        profile = payload["profiles"][name]["valid"].get("B")
        if profile is None:
            selection["candidates"][name] = {"estimable": False, "reason": "arm B missing"}
            continue
        selection["candidates"][name] = {
            "estimable": bool(profile.get("estimable")),
            "gap": profile.get("gap"),
            "n_group": profile.get("n_group"),
            "n_other": profile.get("n_other"),
        }
    valid_candidates = [
        (name, payload["profiles"][name]["valid"]["B"])
        for name in GROUP_NAMES
        if payload["profiles"][name]["valid"].get("B", {}).get("estimable")
    ]
    eligible = [(name, profile) for name, profile in valid_candidates if float(profile["gap"]) >= C.GROUP_GAP_THRESHOLD]
    if not eligible:
        selection["label"] = "NO_CLEAR_FAILURE_GROUP"
        selection["test"] = {}
    else:
        best_name, best_profile = max(eligible, key=lambda item: (float(item[1]["gap"]), -GROUP_NAMES.index(item[0])))
        selection["group"] = best_name
        selection["selected_valid_gap"] = float(best_profile["gap"])
        selection["label"] = "FAILURE_GROUP_FROZEN"
        selection["test"] = payload["profiles"][best_name]["test"]
    payload["selection"] = selection

    C.GROUPS.mkdir(parents=True, exist_ok=True)
    C.write_json(C.GROUPS / "groups.json", payload)
    C.write_json(C.GROUPS / "membership_summary.json", {
        "n": payload["n"],
        "size_bins": edges,
        "support_summary": membership["support"],
    })
    return payload


# ---------------------------------------------------------------------------
# frozen-D2 source interventions
# ---------------------------------------------------------------------------


def _load_frozen(view: D.View, tag: str):
    path = A.arm_dir("D2") / f"checkpoint_{tag}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = M.build_arm(view.spec, view.train_norm, "D2")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _predict_variant(
    model,
    cache,
    device,
    *,
    norm: Mapping[str, Any],
    source_override=None,
    alt_state=None,
    alt_map=None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Whole-split prediction with an optional layer-2 source switch."""
    n = int(cache["n_molecules"])
    preds: list[np.ndarray] = []
    contribs: list[np.ndarray] = []
    rhos: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, n, C.EVAL_BATCH_SIZE):
                chunk = list(range(start, min(start + C.EVAL_BATCH_SIZE, n)))
                batch = D.to_device(PREP_DATA.collate(cache, chunk), device)
                mask = None
                if alt_map is not None:
                    mask = torch.zeros(int(batch["atom_feat"].shape[0]), dtype=torch.bool, device=device)
                    offsets = batch["atom_off"].tolist()
                    for i, local_row in enumerate(chunk):
                        for atom in alt_map.get(int(local_row), ()):
                            mask[offsets[i] + int(atom)] = True
                pred, aux = model(
                    batch,
                    source_override=source_override,
                    layer2_alt_state=alt_state,
                    layer2_alt_mask=mask,
                    return_aux=True,
                )
                preds.append(pred.detach().float().cpu().numpy().reshape(-1))
                contribs.append(aux["contrib"].detach().float().cpu().numpy().reshape(-1))
                rhos.append(aux["rho"].detach().float().cpu().numpy().reshape(-1))
    finally:
        model.train(was_training)
    # raw logD space: the head predicts the standardised target
    scale = float(norm["scale"])
    mean = float(norm["mean"])
    return (
        np.concatenate(preds).astype(np.float64) * scale + mean,
        {
            "contrib": np.concatenate(contribs).astype(np.float64) * scale,
            "rho": np.concatenate(rhos).astype(np.float64) * scale,
            "mean": mean,
        },
    )


def _degrees(view: D.View, split: str) -> dict[int, list[int]]:
    """Per-molecule atom degrees (physical bonds) in cache-local row order."""
    cache = view.caches[split]
    ptr = np.asarray(cache["bond_ptr"])
    bond_index = np.asarray(cache["bond_index"])
    out: dict[int, list[int]] = {}
    n = int(cache["n_molecules"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    for row in range(n):
        n_atoms = int(atom_ptr[row + 1] - atom_ptr[row])
        deg = [0] * n_atoms
        for u, v in bond_index[ptr[row] : ptr[row + 1]]:
            deg[int(u)] += 1
            deg[int(v)] += 1
        out[row] = deg
    return out


def _hash_pick(candidates: Sequence[int], k: int, salt: str) -> list[int]:
    """Fixed hash order selection (never looks at predictions or labels)."""
    def key(atom: int) -> str:
        return hashlib.sha256(f"{salt}|{atom}".encode("utf-8")).hexdigest()

    return sorted(candidates, key=key)[:k]


def matched_degree_masks(
    roots: Mapping[int, Sequence[int]],
    degrees: Mapping[int, Sequence[int]],
    salt: str,
) -> dict[str, Any]:
    """Within each molecule: equal-count degree matching of roots vs their complement."""
    target: dict[int, list[int]] = {}
    control: dict[int, list[int]] = {}
    per_bucket: dict[str, Any] = {}
    for row, root_atoms in roots.items():
        deg = degrees[row]
        root_set = set(int(a) for a in root_atoms)
        complement = [a for a in range(len(deg)) if a not in root_set]
        for bucket in (0, 1, 2, 3):
            want = lambda a, b=bucket: (deg[a] == b) if b < 3 else (deg[a] >= 3)  # noqa: E731
            left = sorted(a for a in root_set if want(a))
            right = sorted(a for a in complement if want(a))
            k = min(len(left), len(right))
            if k == 0:
                continue
            picked_t = _hash_pick(left, k, f"{salt}|t|{row}|{bucket}")
            picked_c = _hash_pick(right, k, f"{salt}|c|{row}|{bucket}")
            target.setdefault(row, []).extend(picked_t)
            control.setdefault(row, []).extend(picked_c)
            entry = per_bucket.setdefault(str(bucket), {"n_target": 0, "n_control": 0})
            entry["n_target"] += len(picked_t)
            entry["n_control"] += len(picked_c)
    return {"target": target, "control": control, "per_bucket": per_bucket}


def run_interventions(selection: Mapping[str, Any]) -> dict[str, Any]:
    view = D.build_view()
    device = torch.device("cpu")
    out: dict[str, Any] = {"checkpoints": {}, "group": selection.get("group")}

    group_name = selection.get("group")
    roots_by_split: dict[str, dict[int, list[int]]] = {}
    if group_name:
        rows = A.S.load_split()["rows"]
        smiles = A.read_mol_csv_from_manifest()
        membership = A.group_membership(smiles, rows)
        for split in C.SPLITS:
            arrays = membership_arrays(membership["splits"][split], split)
            roots_by_split[split] = dict(arrays[f"{group_name}_roots"])

    for tag in ("best", "last"):
        if not (A.arm_dir("D2") / f"checkpoint_{tag}.pt").is_file():
            continue
        model, payload = _load_frozen(view, tag)
        entry: dict[str, Any] = {
            "checkpoint": tag,
            "epoch": payload.get("epoch"),
            "state_hash": payload.get("state_hash"),
            "splits": {},
        }
        for split in ("valid", "test"):
            cache = view.caches[split]
            norm = view.target_norm
            normal, aux_n = _predict_variant(model, cache, device, norm=norm)
            reuse_all, _aux_r = _predict_variant(model, cache, device, norm=norm, source_override="reuse")
            fixed_all, _ = _predict_variant(model, cache, device, norm=norm, source_override="fixed")
            y = view.y[split]
            split_entry: dict[str, Any] = {
                "normal_rmse": A.rmse(normal, y),
                "normal_mae": A.mae(normal, y),
                "reuse_all_rmse": A.rmse(reuse_all, y),
                "reuse_all_mae": A.mae(reuse_all, y),
                "reuse_all_delta_pred_max": float(np.abs(reuse_all - normal).max()),
                "reuse_all_delta_pred_mean": float(np.abs(reuse_all - normal).mean()),
                "fixed_all_rmse": A.rmse(fixed_all, y),
                "fixed_all_mae": A.mae(fixed_all, y),
                "fixed_all_delta_pred_max": float(np.abs(fixed_all - normal).max()),
                "fixed_all_delta_pred_mean": float(np.abs(fixed_all - normal).mean()),
            }
            if group_name and roots_by_split.get(split):
                degrees = _degrees(view, split)
                # cache-local row index of every global row in this split
                global_rows = [int(r) for r in np.asarray(view.rows[split]).tolist()]
                local_of_global = {g: i for i, g in enumerate(global_rows)}
                roots_local = {
                    local_of_global[g]: atoms for g, atoms in roots_by_split[split].items() if g in local_of_global
                }
                matched = matched_degree_masks(roots_local, degrees, salt=f"{group_name}|{split}|{tag}")
                if not matched["target"]:
                    split_entry["match"] = {"label": "NO_MATCHABLE_CONTROL"}
                else:
                    target_pred, aux_t = _predict_variant(
                        model, cache, device, norm=norm, alt_state="reuse", alt_map=matched["target"]
                    )
                    control_pred, aux_c = _predict_variant(
                        model, cache, device, norm=norm, alt_state="reuse", alt_map=matched["control"]
                    )
                    identity_t = float(
                        np.abs((target_pred - normal) - ((aux_t["contrib"] - aux_n["contrib"]) + (aux_t["rho"] - aux_n["rho"]))).max()
                    )
                    identity_c = float(
                        np.abs((control_pred - normal) - ((aux_c["contrib"] - aux_n["contrib"]) + (aux_c["rho"] - aux_n["rho"]))).max()
                    )
                    split_entry["match"] = {
                        "label": "MATCHED",
                        "per_bucket": matched["per_bucket"],
                        "n_target_roots": int(sum(len(v) for v in matched["target"].values())),
                        "n_control_roots": int(sum(len(v) for v in matched["control"].values())),
                    }
                    split_entry["reuse_target"] = {
                        "rmse": A.rmse(target_pred, y),
                        "mae": A.mae(target_pred, y),
                        "delta_pred_max": float(np.abs(target_pred - normal).max()),
                        "delta_pred_mean": float(np.abs(target_pred - normal).mean()),
                        "delta_hidden_proxy": float(np.abs(aux_t["contrib"] - aux_n["contrib"]).max()),
                        "identity_max_abs_residual": identity_t,
                    }
                    split_entry["reuse_control"] = {
                        "rmse": A.rmse(control_pred, y),
                        "mae": A.mae(control_pred, y),
                        "delta_pred_max": float(np.abs(control_pred - normal).max()),
                        "delta_pred_mean": float(np.abs(control_pred - normal).mean()),
                        "delta_hidden_proxy": float(np.abs(aux_c["contrib"] - aux_n["contrib"]).max()),
                        "identity_max_abs_residual": identity_c,
                    }
                    split_entry["reuse_target_minus_control"] = {
                        "delta_pred_max_difference": float(
                            np.abs(target_pred - normal).max() - np.abs(control_pred - normal).max()
                        ),
                        "rmse_change_target": A.rmse(target_pred, y) - A.rmse(normal, y),
                        "rmse_change_control": A.rmse(control_pred, y) - A.rmse(normal, y),
                    }
            entry["splits"][split] = split_entry
        out["checkpoints"][tag] = entry

    out["notes"] = {
        "REUSE_ALL": "layer-2 source replaced by h0 for every directed edge (R1 semantics)",
        "FIXED_ALL": "layer-2 source replaced by the fixed extended source c_j (R2 semantics)",
        "REUSE_TARGET": "layer-2 source replaced by h0 only for edges entering the matched group roots",
        "REUSE_CONTROL": "the same replacement for degree-matched control roots in the same molecules",
        "identity": "Delta pred = Delta contrib + Delta rho is checked numerically",
        "scope": "internal intervention on a frozen prediction function; not a retraining gain",
    }
    C.INTERVENTIONS.mkdir(parents=True, exist_ok=True)
    C.write_json(C.INTERVENTIONS / "interventions.json", out)
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
