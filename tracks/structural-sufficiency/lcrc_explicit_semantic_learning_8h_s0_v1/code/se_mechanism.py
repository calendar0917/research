"""Mechanism analysis: support-conditioned slices, cross-code gradient coupling

kernel, and the real molecular-edit response.  CPU only, no training.

* **Support slices** use the *real* model codes with Train molecule document
  frequency (leave-one-out inside Train; N25/N50 use the corresponding Fit
  support).  Thresholds are frozen from Train-LOO terciles.
* **Gradient coupling kernel** takes only the ``path_emb`` bank, up to 64 codes
  (top-32 / bottom-32 Train df, ties broken by code), a fixed Rademacher
  direction (seed 20261115) and computes the cosine kernel
  ``K_ab = J_a . J_b / (|J_a||J_b|)`` of the per-code scalar w.r.t. the actual
  generator parameters.  Grouped by composition overlap (0/1/2 shared
  components of mid/b1/b2).
* **Edit response** re-maps the round-1 ``pairs.csv`` to the current official
  rows via canonical SMILES, recomputes components, and compares
  ``Delta pred`` with ``Delta y`` for LS / FS / FP / B at matched splits.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import se_common as C
import se_tables as T
import se_data as SED

import lp_common as LPC
import lp_data as LPD
import lcrc_data as PREP_DATA


# ---------------------------------------------------------------------------
# support slices
# ---------------------------------------------------------------------------


def _train_loo_support(train_cache) -> dict[str, Any]:
    """Train-LOO molecule document frequency for path/joint codes."""
    path_ptr = np.asarray(train_cache["path_ptr"])
    n = int(train_cache["n_molecules"])
    path_cat = np.asarray(train_cache["path_cat"])
    end63 = np.asarray(train_cache["path_end63"])
    df_path = defaultdict(int)
    df_joint = defaultdict(int)
    per_mol_path = []
    per_mol_joint = []
    for m in range(n):
        pcs = [int(x) for x in path_cat[path_ptr[m]:path_ptr[m + 1]]]
        es = [int(x) for x in end63[path_ptr[m]:path_ptr[m + 1]]]
        sp = set(pcs)
        sj = set(zip(pcs, es))
        per_mol_path.append(sp)
        per_mol_joint.append(sj)
        for p in sp:
            df_path[p] += 1
        for j in sj:
            df_joint[j] += 1
    return {"df_path": dict(df_path), "df_joint": dict(df_joint),
            "per_mol_path": per_mol_path, "per_mol_joint": per_mol_joint, "n": n}


def support_slices(run_name: str, *, budget: str = "N100") -> dict[str, Any]:
    """Per-molecule joint-path low-support fraction and size-standardised errors."""
    view = LPD.build_view()
    train = view.caches["train"]
    sup = _train_loo_support(train)
    # frozen terciles of the per-molecule joint low-support fraction on Train
    r_train = np.asarray([_low_support_frac_for(sup["per_mol_joint"][i], sup["df_joint"]) for i in range(sup["n"])])
    q1, q2 = np.quantile(r_train, [1 / 3, 2 / 3])
    out: dict[str, Any] = {"terciles": [float(q1), float(q2)], "budget": budget}
    for split in ("valid", "test"):
        cache = view.caches[split]
        path_ptr = np.asarray(cache["path_ptr"])
        path_cat = np.asarray(cache["path_cat"])
        end63 = np.asarray(cache["path_end63"])
        nm = int(cache["n_molecules"])
        r = np.empty(nm)
        for m in range(nm):
            pcs = [int(x) for x in path_cat[path_ptr[m]:path_ptr[m + 1]]]
            es = [int(x) for x in end63[path_ptr[m]:path_ptr[m + 1]]]
            r[m] = _low_support_frac_for(set(zip(pcs, es)), sup["df_joint"])
        bins = np.where(r <= q1, 0, np.where(r <= q2, 1, 2))
        entry = {"n": nm, "bin_counts": [int((bins == b).sum()) for b in (0, 1, 2)]}
        # size-standardised MAE by bin if the run exists
        if (C.ANALYSIS / "results" / run_name / "pred_valid_best.npz").is_file():
            d = np.load(C.ANALYSIS / "results" / run_name / f"pred_{split}_best.npz", allow_pickle=False)
            err = np.abs(np.asarray(d["raw"], float) - np.asarray(d["y_true"], float))
            entry["mae_by_bin"] = [float(err[bins == b].mean()) for b in (0, 1, 2)]
        out[split] = entry
    return out


def _low_support_frac_for(joint_set, df_joint, thresh: int = 5) -> float:
    if not joint_set:
        return 0.0
    low = sum(1 for j in joint_set if df_joint.get(j, 0) <= thresh)
    return low / len(joint_set)


# ---------------------------------------------------------------------------
# cross-code gradient coupling kernel
# ---------------------------------------------------------------------------


def gradient_kernel(family: str = "FS", *, max_codes: int = 64, seed: int = 20261115,
                    device: str = "cpu") -> dict[str, Any]:
    """Cosine kernel of the per-code path_emb scalar gradient w.r.t. generator theta."""
    view = LPD.build_view()
    train = view.caches["train"]
    sup = _train_loo_support(train)
    df = sup["df_path"]
    codes = sorted(df.keys())
    codes_by_df = sorted(codes, key=lambda c: (df[c], c))
    low = codes_by_df[: max_codes // 2]
    high = codes_by_df[-max_codes // 2:]
    chosen = sorted(set(low) | set(high))
    if len(chosen) > max_codes:
        chosen = chosen[:max_codes]

    perms = T.build_permutations()
    model = M_build(view, family, perms, device)
    gen = model.generator
    codes_t = torch.arange(C.N_PATH_CODES, device=device)
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 2, size=C.RANK).astype(np.float64) * 2 - 1
    q_t = torch.as_tensor(q / np.sqrt(C.RANK), dtype=torch.float32, device=device)

    params = [p for p in gen.parameters() if p.requires_grad]
    # for FP the composite-code bijection is applied to the code BEFORE decode,
    # so the gradient for original code c is taken at perm[c]
    path_perm = np.asarray(perms[T.perm_for_bank("path_emb")], dtype=np.int64) \
        if family == "FP" else None
    jac = {}
    for c in chosen:
        gen.zero_grad(set_to_none=True)
        table = gen.bank_raw("path_emb", codes_t)
        idx = int(path_perm[c]) if path_perm is not None else c
        s = (table[idx] * q_t).sum()
        s.backward(retain_graph=False)
        jac[c] = np.concatenate([
            (p.grad.detach().cpu().numpy().reshape(-1) if p.grad is not None else np.zeros(p.numel(), dtype=np.float32))
            for p in params
        ])
    norms = {c: float(np.linalg.norm(jac[c])) for c in chosen}
    # composition overlap grouping
    def comps(c):
        mid, b1, b2 = c // 25, (c % 25) // 5, c % 5
        return (mid, b1, b2)

    groups = {0: [], 1: [], 2: []}
    diag = {}
    for i, a in enumerate(chosen):
        diag[a] = float(np.dot(jac[a], jac[a]) / (norms[a] * norms[a] + 1e-12))
        for b in chosen[i + 1:]:
            k = float(np.dot(jac[a], jac[b]) / (norms[a] * norms[b] + 1e-12))
            ov = sum(1 for x, y in zip(comps(a), comps(b)) if x == y)
            groups[ov].append(k)
    return {
        "family": family, "n_codes": len(chosen),
        "df_of_chosen": {int(c): int(df.get(c, 0)) for c in chosen},
        "kernel_by_overlap": {str(k): {"n": len(v), "mean": float(np.mean(v)) if v else None,
                                       "median": float(np.median(v)) if v else None,
                                       "values": [round(x, 6) for x in v]}
                              for k, v in groups.items()},
        "diag_mean": float(np.mean(list(diag.values()))),
        "note": "positive K = coupled gradient directions; mechanism check, not chemical causality",
    }


def M_build(view, family, perms, device):
    import se_model as M

    m = M.build_semantic_model(view.spec, view.train_norm, family, permutations=perms)
    return m.to(device)


# ---------------------------------------------------------------------------
# real molecular edit response
# ---------------------------------------------------------------------------


def _canonical(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def edit_response(runs: Mapping[str, str]) -> dict[str, Any]:
    """Map round-1 pairs to current official rows; compare Delta pred vs Delta y."""
    src = C.EDIT / "analysis"
    pairs_path = src / "pairs.csv"
    if not pairs_path.is_file():
        return {"available": False, "reason": "pairs.csv missing"}
    import csv

    with open(pairs_path, newline="", encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))
    raw = LPD.build_view()
    from lp_source import read_mol_csv

    mol = read_mol_csv(C.HOME / "data/download/extracted/lipophilicity")
    smi2row = {}
    for i, s in enumerate(mol["smiles"]):
        smi2row.setdefault(_canonical(s), []).append(i)

    mapped = []
    for p in pairs:
        a = smi2row.get(_canonical(p.get("smiles_a", "")), [])
        b = smi2row.get(_canonical(p.get("smiles_b", "")), [])
        if len(a) == 1 and len(b) == 1:
            mapped.append({"row_a": a[0], "row_b": b[0]})
    # unique unordered pairs
    seen = set()
    uniq = []
    for m in mapped:
        key = tuple(sorted((m["row_a"], m["row_b"])))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    out: dict[str, Any] = {"n_pairs_raw": len(pairs), "n_mapped_unique": len(uniq)}
    y = raw.y
    split_of = {}
    for sp in LPC.SPLITS:
        for r in raw.rows[sp].tolist():
            split_of[int(r)] = sp
    # raw labels per global row id
    label_all = np.full(LPC.N_MOLECULES, np.nan)
    for sp in LPC.SPLITS:
        for r, v in zip(raw.rows[sp].tolist(), raw.y[sp].tolist()):
            label_all[int(r)] = float(v)
    deltas = []
    for m in uniq:
        ra, rb = int(m["row_a"]), int(m["row_b"])
        if not np.isfinite(label_all[ra]) or not np.isfinite(label_all[rb]):
            continue
        deltas.append({"ra": ra, "rb": rb, "dy": float(label_all[rb] - label_all[ra]),
                       "split_a": split_of.get(ra), "split_b": split_of.get(rb)})
    out["pairs"] = deltas
    # per-model edit MAE where both rows are available (any split), reported per split combo
    models = {}
    for name, run in runs.items():
        preds = {}
        for sp in LPC.SPLITS:
            p = C.ANALYSIS / "results" / run / f"pred_{sp}_best.npz"
            if p.is_file():
                d = np.load(p, allow_pickle=False)
                for r, v in zip(np.asarray(d["rows"]).tolist(), np.asarray(d["raw"]).tolist()):
                    preds[int(r)] = float(v)
        if preds:
            models[name] = preds
    res = {}
    for name, preds in models.items():
        cells: dict[str, dict[str, Any]] = defaultdict(lambda: {"errs": [], "agree": 0, "n": 0})
        for d in deltas:
            ra, rb = d["ra"], d["rb"]
            if ra not in preds or rb not in preds:
                continue
            dp = preds[rb] - preds[ra]
            combo = f"{d['split_a']}-{d['split_b']}"
            cell = cells[combo]
            cell["errs"].append(abs(dp - d["dy"]))
            if abs(d["dy"]) >= 0.1:
                cell["n"] += 1
                cell["agree"] += int(np.sign(dp) == np.sign(d["dy"]))
        res[name] = {
            combo: {"n_matched": len(c["errs"]),
                    "edit_mae": float(np.mean(c["errs"])) if c["errs"] else None,
                    "sign_n": c["n"], "sign_agreement": (c["agree"] / c["n"]) if c["n"] else None,
                    "LOW_SUPPORT": bool(len(c["errs"]) < 20)}
            for combo, c in cells.items()
        }
        all_errs = [e for c in cells.values() for e in c["errs"]]
        all_n = sum(c["n"] for c in cells.values())
        all_agree = sum(c["agree"] for c in cells.values())
        res[name]["ALL"] = {"n_matched": len(all_errs),
                            "edit_mae": float(np.mean(all_errs)) if all_errs else None,
                            "sign_n": all_n, "sign_agreement": (all_agree / all_n) if all_n else None}
    out["models"] = res
    out["note"] = ("components are recomputed on the mapped unordered pair graph; each cell <20 "
                   "components is flagged LOW_SUPPORT and only described")
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
