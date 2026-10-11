"""CSCL-v3d frozen official validation/test evaluation (zero training).

Protocol: ``configs/cscl_v3d_frozen_official_eval.yaml`` (frozen at commit
55e088b, before any official valid/test feature build or scoring).

Stages
------
freeze    : artifact hashes + strict checkpoint loads + fit-only stat cross-check
            + label-free structure-overlap audit (WL hash + exact isomorphism)
replay    : engineering gates on TRAIN data only (feature-entry replay, frozen
            F/S/O replay vs committed exports/MAEs)
extract   : official valid/test read-only inference inputs + frozen F/O/S
            predictions + conservation/additivity/gauge gates (y never read;
            ``ys=zeros`` is passed to the extractor)
score     : ONE formal scoring per split (reads official y once, after the
            predictions are hashed to disk) + paired bootstrap + judgment band
analyze   : attribution integrity + subgroups + frozen moment-usage sensitivity

Zero training: every stage runs eval() models under torch.no_grad(); no
optimizer exists in this module.  Checkpoint hashes are verified before and
after every extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_features as cf  # noqa: E402
import run_cscl_v3 as v3  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402
import strong_backbone_bucket_features as bf  # noqa: E402
import contribution_gauge as cg  # noqa: E402
import run_cscl_v3d as v3d  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
from run_cscl_v3d import SBucketContribHead, OBucketHead, collate_bucket  # noqa: E402

from torch_geometric.datasets import ZINC  # noqa: E402
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_compact_v4_training_sufficiency as ztraining,
    zinc_e2e_dictenv_p1 as p1run,
    zinc_static_dictionary_pair as sdp,
)
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr  # noqa: E402

PROTOCOL_ID = "cscl-v3d-frozen-official-eval"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "cscl_v3d_frozen_official_eval.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d_official_eval"
V3D_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d"

ENCODED_VALID = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_valid.pt"
ENCODED_TRAIN = rpv2.ENCODED_TRAIN

ATOL_SUM = 1e-5          # S additivity / raw-canonical tolerance (= cg.GAUGE_ATOL)
S_DEV_ATOL = 1e-4        # dev replay MAE tolerance
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
EQUIV_REGION = (-0.01, 0.01)
TOL_BAND = 0.02

RECORDED_DEV = {"F": 0.11641, "O_bucket": 0.11610, "S_bucket": 0.12694}

# frozen engineering tolerances (protocol yaml `engineering_gates`)
F_PRED_ATOL = 5.0e-5     # F replay vs stored GPU export (CPU float32 rounding)
S_PRED_ATOL = 1.0e-5     # S replay vs stored canonical dev predictions
MAE_ATOL = 1.0e-4        # S/O dev replay MAE vs recorded analysis numbers
FEATURE_REPLAY_ATOL = 1.0e-5  # CPU replay vs GPU-built cache (float32 fields)
GATE_THRESHOLDS = {
    "F_pred_atol": F_PRED_ATOL, "S_pred_atol": S_PRED_ATOL,
    "S_dev_mae_atol": MAE_ATOL, "O_dev_mae_atol": MAE_ATOL,
    "feature_replay_atol": FEATURE_REPLAY_ATOL, "additivity_atol": ATOL_SUM,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)


def _ckpt_sha() -> dict[str, str]:
    return {
        "F_soup": _sha(sba.SOUP_STATE),
        "O_bucket": _sha(V3D_DIR / "O_bucket_s0_best_state.pt"),
        "S_bucket": _sha(V3D_DIR / "S_bucket_s0_best_state.pt"),
        "gauge": _sha(V3D_DIR / "gauge_offsets_s0.pt"),
    }


def load_frozen_models(log=log):
    """Frozen F soup + O_bucket + gauged S_bucket (all eval, params frozen)."""
    model_f, prep, blob10 = sba.load_frozen_backbone(log=log)

    ns = v3d.prepare_all(log=log)

    o_head = OBucketHead(ns.vocab.n_total)
    o_state = torch.load(V3D_DIR / "O_bucket_s0_best_state.pt", map_location="cpu", weights_only=True)
    o_head.load_state_dict(o_state, strict=True)
    o_head.eval()
    for parameter in o_head.parameters():
        parameter.requires_grad_(False)

    s_head = SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    s_state = torch.load(V3D_DIR / "S_bucket_s0_best_state.pt", map_location="cpu", weights_only=True)
    s_head.load_state_dict(s_state, strict=True)
    s_head.eval()
    for parameter in s_head.parameters():
        parameter.requires_grad_(False)

    gauge_blob = torch.load(V3D_DIR / "gauge_offsets_s0.pt", map_location="cpu", weights_only=False)
    offsets = np.asarray(gauge_blob["offsets"], dtype=np.float64)
    assert offsets.shape == (ns.vocab.n_total,), "gauge offsets vocabulary mismatch"
    gauged = cg.GaugedHead(s_head, offsets)
    gauged.eval()
    for parameter in gauged.parameters():
        parameter.requires_grad_(False)
    log("[models] F soup + O_bucket + gauged S_bucket loaded (all frozen, eval mode)")
    return {
        "F": model_f, "O": o_head, "S_raw": s_head, "S": gauged,
        "prep": prep, "blob10": blob10, "ns": ns,
        "gauge_meta": gauge_blob["meta"],
    }


# ---------------------------------------------------------------------------
# stage: freeze
# ---------------------------------------------------------------------------

_OVERLAP_TRAIN_VALIDATE_N = 2000
_EXACT_CAP = 400


def _mol_graph(x, edge_index, edge_attr):
    import networkx as nx

    G = nx.Graph()
    for i, v in enumerate(np.asarray(x).reshape(-1).tolist()):
        G.add_node(i, a=int(v))
    src = np.asarray(edge_index)[0].tolist()
    dst = np.asarray(edge_index)[1].tolist()
    bs = np.asarray(edge_attr).reshape(-1).tolist()
    seen: dict[tuple[int, int], int] = {}
    symmetric = True
    for u, v, b in zip(src, dst, bs):
        key = (min(u, v), max(u, v))
        if key in seen:
            if seen[key] != int(b):
                symmetric = False
            continue
        seen[key] = int(b)
        G.add_edge(key[0], key[1], b=int(b))
    return G, symmetric


def _wl_hash(G) -> str:
    import networkx as nx

    return nx.weisfeiler_lehman_graph_hash(G, node_attr="a", edge_attr="b")


def _overlap_audit(dataset_train, dataset_val, dataset_test, log=log):
    """Label-free duplicate/overlap audit in model-input space (WL hash +
    exact label-graph isomorphism; PyG ZINC carries no smiles and RDKit is not
    installed).  Validated against the committed RDKit canonical-SMILES table
    on the first ``_OVERLAP_TRAIN_VALIDATE_N`` official-train molecules."""
    import networkx as nx

    def graphs(ds, n):
        out_g, sym = [], True
        for i in range(n):
            g, s = _mol_graph(ds[i].x.numpy(), ds[i].edge_index.numpy(), ds[i].edge_attr.numpy())
            out_g.append(g)
            sym = sym and s
        assert sym, "asymmetric edge_attr encountered (bond types differ per direction)"
        return out_g

    started = time.perf_counter()
    tr = graphs(dataset_train, len(dataset_train))
    va = graphs(dataset_val, len(dataset_val))
    te = graphs(dataset_test, len(dataset_test))
    hashes = {name: [_wl_hash(g) for g in gs] for name, gs in (("train", tr), ("val", va), ("test", te))}

    def exact(a, b):
        return nx.is_isomorphic(a, b, node_match=lambda x, y: x["a"] == y["a"], edge_match=lambda x, y: x["b"] == y["b"])

    def groups_within(gs, hs):
        by_hash: dict[str, list] = {}
        for i, h in enumerate(hs):
            by_hash.setdefault(h, []).append(i)
        dup_groups = []
        for idx in by_hash.values():
            if len(idx) < 2:
                continue
            confirmed = []
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    if exact(gs[idx[a]], gs[idx[b]]):
                        confirmed.append((idx[a], idx[b]))
            if confirmed:
                members = sorted({i for pair in confirmed for i in pair})
                dup_groups.append({"members": members[:64], "n": len(members), "pairs_confirmed": len(confirmed)})
        return {
            "n_molecules": len(gs), "n_wl_hash_groups": len(by_hash),
            "n_isomorphic_dup_molecules": sum(g["n"] for g in dup_groups),
            "groups": dup_groups,
        }

    def cross(gs_a, hs_a, gs_b, hs_b):
        by_hash_b: dict[str, list] = {}
        for j, h in enumerate(hs_b):
            by_hash_b.setdefault(h, []).append(j)
        cand = 0
        confirmed = []
        for i, h in enumerate(hs_a):
            for j in by_hash_b.get(h, ()):
                cand += 1
                if len(confirmed) < _EXACT_CAP and exact(gs_a[i], gs_b[j]):
                    confirmed.append([i, j])
        return {
            "n_wl_candidates": cand,
            "n_isomorphic_pairs_confirmed": len(confirmed),
            "note": ("exact-confirmation capped at %d pairs" % _EXACT_CAP) if cand > _EXACT_CAP else "all candidates exactly confirmed",
            "pairs_first": confirmed[:64],
        }

    # --- validity check vs the committed RDKit canonical-SMILES table ---
    table = np.load(REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz", allow_pickle=True)
    smiles = [str(s) for s in table["smiles"]]
    n_val = _OVERLAP_TRAIN_VALIDATE_N
    smi_groups: dict[str, list] = {}
    for i, s in enumerate(smiles[:n_val]):
        smi_groups.setdefault(s, []).append(i)
    hash_groups: dict[str, list] = {}
    for i, h in enumerate(hashes["train"][:n_val]):
        hash_groups.setdefault(h, []).append(i)
    purity = int(sum(1 for idx in smi_groups.values() if len({hashes["train"][i] for i in idx}) == 1))
    dup_pairs_checked = 0
    dup_smiles_disagree = 0
    for idx in hash_groups.values():
        if len(idx) < 2:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if exact(tr[idx[a]], tr[idx[b]]):
                    dup_pairs_checked += 1
                    dup_smiles_disagree += int(smiles[idx[a]] != smiles[idx[b]])
    validity = {
        "n_validated_molecules": n_val,
        "n_smiles_groups": len(smi_groups),
        "n_smiles_groups_single_wl_hash": purity,
        "smiles_group_wl_purity": purity / len(smi_groups),
        "n_isomorphic_dup_pairs_checked": dup_pairs_checked,
        "dup_pairs_with_differing_rdkit_smiles": dup_smiles_disagree,
        "note": "identical smiles must always co-hash (isomorphic graphs hash identically); a differing SMILES inside a confirmed dup pair indicates stereochemistry absent from the input graph (disclosed, not a method failure)",
        "ok": bool(purity == len(smi_groups)),
    }

    out = {
        "method": "weisfeiler-lehman graph hash (atom categories + bond types) candidates + exact label-graph isomorphism (networkx vf2); label-free; model-input-space identity",
        "disclosure": "isomorphism = the model's view of identical input (x/edge_index/edge_attr is all the model sees); not a claim of full chemical identity (stereochemistry/protonation not in the input)",
        "train": groups_within(tr, hashes["train"]),
        "valid": groups_within(va, hashes["val"]),
        "test": groups_within(te, hashes["test"]),
        "valid_vs_test": cross(va, hashes["val"], te, hashes["test"]),
        "valid_vs_train": cross(va, hashes["val"], tr, hashes["train"]),
        "test_vs_train": cross(te, hashes["test"], tr, hashes["train"]),
        "validity_vs_rdkit_table": validity,
        "seconds": time.perf_counter() - started,
    }
    log(f"[overlap] valid dups={out['valid']['n_isomorphic_dup_molecules']} test dups={out['test']['n_isomorphic_dup_molecules']} "
        f"valid~test={out['valid_vs_test']['n_isomorphic_pairs_confirmed']} valid~train={out['valid_vs_train']['n_isomorphic_pairs_confirmed']} "
        f"test~train={out['test_vs_train']['n_isomorphic_pairs_confirmed']} validity_ok={validity['ok']} "
        f"dup_smiles_disagree={dup_smiles_disagree} ({out['seconds']:.0f}s)")
    if not validity["ok"]:
        raise RuntimeError("overlap-audit validity check against the committed RDKit table failed")
    return out


def stage_freeze(log=log):
    started = time.perf_counter()
    models = load_frozen_models(log=log)
    ns = models["ns"]

    artifacts = {
        "F_soup": sba.SOUP_STATE, "O_bucket": V3D_DIR / "O_bucket_s0_best_state.pt",
        "S_bucket": V3D_DIR / "S_bucket_s0_best_state.pt", "gauge": V3D_DIR / "gauge_offsets_s0.pt",
        "prep_fit7200": sba.PREP_NPZ, "legacy_prep_10k": sba.LEGACY_PREP,
        "encoded_train": ENCODED_TRAIN, "encoded_valid": ENCODED_VALID,
        "env_train": REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/cache/env_train.pt",
        "env_valid": REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/cache/env_valid.pt",
        "anchor_stats": REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/anchor_stats.json",
        "test_records": REPO_ROOT / "tracks/ksvd/results/compact_v4_training_sufficiency/cache/v4_records_test.pkl.gz",
        "v3d_bucket_feature_cache": REPO_ROOT / "data/cache/cscl_v3d_bucket_features.pt",
        "smiles_table": REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz",
        "zinc_train_processed": REPO_ROOT / "data/ZINC/subset/processed/train.pt",
        "zinc_valid_processed": REPO_ROOT / "data/ZINC/subset/processed/val.pt",
        "zinc_test_processed": REPO_ROOT / "data/ZINC/subset/processed/test.pt",
        "code_eval": Path(__file__).resolve(),
        "code_backbone_bucket": _CODE_DIR / "strong_backbone_bucket_features.py",
        "code_backbone_attr": _CODE_DIR / "strong_backbone_attribution.py",
        "code_run_v3d": _CODE_DIR / "run_cscl_v3d.py",
        "code_gauge": _CODE_DIR / "contribution_gauge.py",
        "protocol": CONFIG_PATH,
    }
    manifest = {k: {"path": str(p.relative_to(REPO_ROOT)), "sha256": _sha(p)} for k, p in artifacts.items()}
    manifest["git_commit"] = rpv2.git_commit()
    manifest["protocol_id"] = PROTOCOL_ID
    manifest["zero_training"] = {"optimizer_steps": 0, "trainable_parameter_updates": 0, "seed1_runs": 0}
    manifest["models"] = {
        "F": {"n_params": int(sum(p.numel() for p in models["F"].parameters())), "frozen": True},
        "O_bucket": {"n_params": int(sum(p.numel() for p in models["O"].parameters())), "frozen": True},
        "S_bucket": {"n_params": int(sum(p.numel() for p in models["S_raw"].parameters())), "frozen": True},
    }

    # fit-only statistics cross-check vs the frozen gauge metadata
    # (hashes computed exactly like contribution_gauge.compute_fit_offsets)
    meta = models["gauge_meta"]
    vocab = ns.vocab
    fit_pos = v3._index_by_row(ns.features, ns.idx["fit_inner"])
    fit_batch = v3.collate(ns.features, fit_pos, vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    unit_type = fit_batch.unit_type.cpu().numpy().astype(np.int64)
    counts = np.bincount(unit_type, minlength=vocab.n_total).astype(np.int64)
    fit_unit_type_hash = hashlib.sha256(np.ascontiguousarray(unit_type).tobytes()).hexdigest()[:16]
    fit_rows_hash = hashlib.sha256(np.ascontiguousarray(np.asarray(sorted(int(r) for r in ns.idx["fit_inner"]), np.int64)).tobytes()).hexdigest()[:16]
    counts_hash = hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()[:16]
    stat_check = {
        "fit_unit_type_hash_recomputed": fit_unit_type_hash, "fit_unit_type_hash_frozen": meta["fit_unit_type_hash"],
        "fit_rows_hash_recomputed": fit_rows_hash, "fit_rows_hash_frozen": meta["fit_rows_hash"],
        "counts_hash_recomputed": counts_hash, "counts_hash_frozen": meta["counts_hash"],
        "n_fit_units": int(counts.sum()), "n_types_total": vocab.n_total, "n_known": vocab.n_known,
        "ok": bool(fit_unit_type_hash == meta["fit_unit_type_hash"] and fit_rows_hash == meta["fit_rows_hash"] and counts_hash == meta["counts_hash"]),
    }
    manifest["fit_only_statistics_cross_check"] = stat_check
    log(f"[freeze] fit-only stat cross-check ok={stat_check['ok']} (n_types={vocab.n_total}, known={vocab.n_known}, fit units={counts.sum()})")
    if not stat_check["ok"]:
        raise RuntimeError("fit-only statistics do not reproduce the frozen gauge metadata; refusing to evaluate")

    dataset_train = cf.load_official_train()
    dataset_val = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
    dataset_test = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="test")
    assert len(dataset_train) == 10000 and len(dataset_val) == 1000 and len(dataset_test) == 1000
    overlap = _overlap_audit(dataset_train, dataset_val, dataset_test, log=log)

    data_prov = {
        "official_dataset": {"source": "PyG ZINC subset=True", "root": str(cf.ZINC_ROOT),
                             "sizes": {"train": len(dataset_train), "val": len(dataset_val), "test": len(dataset_test)},
                             "processed_sha256": {k: manifest[f"zinc_{k}_processed"]["sha256"] for k in ("train", "valid", "test")}},
        "encoded_valid_stats_fitted_on": "official train (zinc_patch_path_pooling._phase_data fit_records)",
        "test_encoded_via": "zinc_compact_v4_training_sufficiency.build_encoded(train_records, test_records, base_config) + p1._attach_test_env (all standardizers fit on train records)",
        "overlap_audit": overlap,
    }
    _write_json(OUT / "freeze_manifest.json", manifest)
    _write_json(OUT / "data_provenance.json", data_prov)
    log(f"[freeze] done in {time.perf_counter() - started:.0f}s -> freeze_manifest.json / data_provenance.json")
    return manifest


# ---------------------------------------------------------------------------
# stage: replay (engineering gates on TRAIN data only, no official labels)
# ---------------------------------------------------------------------------

_FEATURE_COMPARISON_SKIP = {"y"}   # audit-only label field (0.0 by construction here)


def _build_mols_raw(dataset, rows: Sequence[int], smiles: Sequence[str] | None = None) -> list:
    """Label-free RINGCHAIN summaries straight from the official PyG graphs."""
    return [
        cf.molecule_units(
            int(i),
            dataset[i].x.numpy(),
            dataset[i].edge_index.numpy(),
            dataset[i].edge_attr.numpy(),
            str(smiles[i]) if smiles is not None else "",
        )
        for i in rows
    ]


def _features_equal(a, b) -> tuple[bool, float, str]:
    """Structural fields must match exactly; float fields compared by caller."""
    import dataclasses

    worst, worst_field = 0.0, ""
    exact = True
    for f in dataclasses.fields(bf.MolBucketFeatures):
        name = f.name
        if name in _FEATURE_COMPARISON_SKIP:
            continue
        va, vb = getattr(a, name), getattr(b, name)
        if isinstance(va, np.ndarray):
            if va.shape != vb.shape or not np.array_equal(va, vb):
                exact = False
                d = float(np.abs(va.astype(np.float64) - vb.astype(np.float64)).max()) if va.shape == vb.shape else np.inf
                if d > worst:
                    worst, worst_field = d, name
        elif isinstance(va, list):
            if va != vb:
                return False, np.inf, name
        elif va != vb:
            return False, np.inf, name
    return exact, worst, worst_field


def stage_replay(log=log) -> dict:
    """Gate 1 (feature entry) + Gate 2 (frozen F/S/O on train) -- train only."""
    started = time.perf_counter()
    models = load_frozen_models(log=log)
    ns = models["ns"]

    # ---- gate 2a: frozen F soup replay on monitor + dev (stored exports) ----
    train = rpv2.load_train_rows(log=log)
    rpv2.apply_prep_fit7200(train, models["prep"], models["blob10"])
    f_replay: dict[str, Any] = {}
    for name, path in (("dev", REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_dev_preds.npz"),
                       ("monitor", REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_monitor_preds.npz")):
        ref = np.load(path)
        rows = [train[int(i)] for i in ref["row"]]
        pred = rpv2.predict_raw(models["F"], rows, torch.device("cpu"))
        max_d = float(np.abs(pred - ref["pred"]).max())
        f_replay[name] = {"n": int(len(rows)), "max_abs_pred_diff": max_d, "ok": bool(max_d <= F_PRED_ATOL)}
        log(f"[replay:F:{name}] max|dpred|={max_d:.3e} (atol {F_PRED_ATOL:.0e})")

    # ---- gate 1: NEW inference entry on the first 64 train rows vs the cache ----
    dataset_train = cf.load_official_train()
    smiles = cf.load_canonical_smiles()
    subset = list(range(64))
    mols = _build_mols_raw(dataset_train, subset, smiles)
    batch = p1.env_collate([train[i] for i in subset])
    batch.y = torch.zeros(len(subset))
    feats_new, aud_new = bf.extract_batch_bucket(models["F"], batch, mols, [0.0] * len(subset), "cpu")
    blob = bf.load_bucket_cache(log=log)
    assert blob is not None
    feats_cache = blob["features"]
    assert [m.row for m in feats_cache[:64]] == subset
    bit_exact = True
    worst, worst_field = 0.0, ""
    for k in range(64):
        eq, w, wf = _features_equal(feats_new[k], feats_cache[k])
        bit_exact &= eq
        if w > worst:
            worst, worst_field = w, wf
    feature_replay = {
        "n": 64,
        "bit_exact_vs_cache": bool(bit_exact),
        "max_abs_float_diff": worst,
        "worst_field": worst_field,
        "atol": FEATURE_REPLAY_ATOL,
        "cache_build_device_note": "v3d cache was built on the res GPU1 (commit b516088); CPU replay differs only by float32 device rounding",
        "conservation_audit": aud_new,
        "ok": bool(bit_exact or worst <= FEATURE_REPLAY_ATOL),
    }
    log(f"[replay:features] bit_exact={bit_exact} max|d|={worst:.3e} ({worst_field}) atol={FEATURE_REPLAY_ATOL:.0e}")

    # ---- gate 2b: S/O replay on dev 2000 from the verified cache ----
    dev_ref = np.load(V3D_DIR / "S_bucket_s0_dev_preds.npz")
    lookup = {m.row: i for i, m in enumerate(ns.features)}
    picks = [lookup[int(r)] for r in dev_ref["row"]]
    batch = collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    with torch.no_grad():
        ps = models["S"](batch)["pred"].cpu().numpy().astype(np.float64)
        ps_raw = models["S_raw"](batch)["pred"].cpu().numpy().astype(np.float64)
        po = models["O"](batch)["pred"].cpu().numpy().astype(np.float64)
    s_max = float(np.abs(ps - dev_ref["pred"]).max())
    s_mae = float(np.abs(ps - dev_ref["y"]).mean())
    o_mae = float(np.abs(po - dev_ref["y"]).mean())
    gauge_dev = float(np.abs(ps - ps_raw).max())
    so_replay = {
        "n": int(len(picks)),
        "S_max_abs_pred_diff_vs_stored": s_max,
        "S_dev_mae": s_mae, "O_dev_mae": o_mae,
        "S_raw_vs_canonical_max_abs_pred_diff": gauge_dev,
        "recorded": RECORDED_DEV,
        "ok": bool(s_max <= S_PRED_ATOL and abs(s_mae - RECORDED_DEV["S_bucket"]) <= MAE_ATOL
                   and abs(o_mae - RECORDED_DEV["O_bucket"]) <= MAE_ATOL and gauge_dev <= ATOL_SUM),
    }
    log(f"[replay:S/O:dev] S max|dpred|={s_max:.3e} S MAE={s_mae:.5f} O MAE={o_mae:.5f} gauge|d|={gauge_dev:.2e}")

    audit = {
        "protocol_id": PROTOCOL_ID, "thresholds": GATE_THRESHOLDS,
        "f_replay": f_replay, "feature_entry_replay": feature_replay, "so_dev_replay": so_replay,
        "all_ok": bool(all(f_replay[n]["ok"] for n in f_replay) and feature_replay["ok"] and so_replay["ok"]),
        "seconds": time.perf_counter() - started,
    }
    log(f"[replay] all_ok={audit['all_ok']} ({audit['seconds']:.0f}s) -> input_replay_audit.json")
    if not audit["all_ok"]:
        raise RuntimeError("training-data replay gate failed; refusing official extraction")
    _write_json(OUT / "input_replay_audit.json", audit)
    return audit


# ---------------------------------------------------------------------------
# stage: extract (official valid/test inference inputs + frozen predictions)
# ---------------------------------------------------------------------------

def _load_official_rows(split: str, models, log=log):
    """Encoded + env-attached + fit7200-prepped rows; order == official split.

    valid : committed encoded_valid cache (stats fit on official train) + env_valid
    test  : historical test-unlock pipeline (all standardizers/vocab fit on
            official train records) + _attach_test_env (train anchor stats)
    Both then go through the ONE frozen preprocessing: legacy inverse -> fit7200
    transform (protocol sect. 4.3).  y is never read here.
    """
    if split == "valid":
        _train, rows, _audit = sdp.load_encoded(valid_subset=None)
        p1run.attach_env(rows, "valid")
        ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
        source = "committed encoded_valid.pt + env_valid.pt + apply_prep_fit7200"
    elif split == "test":
        train_records, _valid_records = ztraining.load_train_valid_records()
        test_records = ztraining.extract_test_records()
        _train_enc, rows, _audit = ztraining.build_encoded(train_records, test_records, ztraining.base_config())
        del train_records
        raw_test = list(zlr._load_zinc(cf.ZINC_ROOT, "test"))
        p1run._attach_test_env(rows, raw_test)
        ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="test")
        source = "v4_records_test.pkl.gz + build_encoded(fit=train records) + _attach_test_env + apply_prep_fit7200"
    else:
        raise ValueError(split)
    assert len(rows) == len(ds) == 1000, (split, len(rows), len(ds))
    rpv2.apply_prep_fit7200(rows, models["prep"], models["blob10"])
    log(f"[extract:{split}] rows built via {source}")

    # label-free row-alignment gates (structure only)
    n_atom_mismatch = n_atom_type_mismatch = 0
    for i in range(1000):
        if int(rows[i].dict_phi.shape[0]) != int(ds[i].x.shape[0]):
            n_atom_mismatch += 1
        elif not torch.equal(rows[i].dict_atom.long().reshape(-1), ds[i].x.long().reshape(-1)):
            n_atom_type_mismatch += 1
    log(f"[extract:{split}] alignment: atom_count_mismatch={n_atom_mismatch} atom_type_mismatch={n_atom_type_mismatch}")
    if n_atom_mismatch or n_atom_type_mismatch:
        raise RuntimeError(f"{split}: encoded rows do not align with the official PyG split (order or content)")
    return rows, ds, {"source": source, "atom_count_mismatches": n_atom_mismatch,
                      "atom_type_mismatches": n_atom_type_mismatch}


def _predict_split(split: str, device: str, log=log, limit: int | None = None) -> dict:
    """Frozen F/O/S predictions + conservation/additivity gates for one split.

    ``limit`` (engineering smoke only) restricts to the first N molecules and
    routes every output to ``*_smoke`` filenames; it can never overwrite or be
    mistaken for the formal artifacts, and it never scores.
    """
    started = time.perf_counter()
    n_mol = int(limit) if limit else 1000
    smoke = bool(limit)
    suffix = "_smoke" if smoke else ""
    models = load_frozen_models(log=log)
    ns = models["ns"]
    sha_before = _ckpt_sha()
    rows, ds, alignment = _load_official_rows(split, models, log=log)
    if smoke:
        rows, ds = rows[:n_mol], ds[:n_mol]

    # ---- F (Full-Y soup) ----
    f_pred = rpv2.predict_raw(models["F"], rows, torch.device(device))
    log(f"[extract:{split}{suffix}] F predictions done (n={len(f_pred)}, finite={bool(np.isfinite(f_pred).all())})")

    # ---- bucket features (y never read: ys = zeros) ----
    mols_raw = _build_mols_raw(ds, range(n_mol))
    features: list = []
    audit_totals: dict[str, float] = {}
    for s in range(0, n_mol, bf.EXTRACT_BATCH):
        chunk = list(range(s, min(s + bf.EXTRACT_BATCH, n_mol)))
        batch = p1.env_collate([rows[i] for i in chunk])
        batch.y = torch.zeros(len(chunk))
        ys_passed = [0.0] * len(chunk)   # y isolation: extraction never sees labels
        mols, aud = bf.extract_batch_bucket(models["F"], batch, [mols_raw[i] for i in chunk], ys_passed, device)
        assert all(m.y == 0.0 for m in mols), "y isolation violated in extraction"
        features.extend(mols)
        for k, v in aud.items():
            audit_totals[k] = max(audit_totals.get(k, 0.0), v)
    log(f"[extract:{split}] bucket features {len(features)} molecules; audit={bf.json_dumps_compact(audit_totals)}")
    failed = {k: v for k, v in audit_totals.items() if (
        v > bf.POOL_RTOL if k.endswith("_recon") or k.endswith("_containment") else v > bf.POOL_ATOL)}
    if failed:
        raise RuntimeError(f"{split}: conservation thresholds exceeded: {failed}")

    # ---- batch-composition invariance (first 16 molecules, two groupings) ----
    subset = list(range(min(16, n_mol)))
    b_all = p1.env_collate([rows[i] for i in subset]); b_all.y = torch.zeros(len(subset))
    f_all, _ = bf.extract_batch_bucket(models["F"], b_all, [mols_raw[i] for i in subset], [0.0] * len(subset), device)
    worst_bc = 0.0
    mismatch_fields: list[str] = []
    for g in range(4):
        sub = subset[g * 4:(g + 1) * 4]
        if not sub:
            continue
        b_part = p1.env_collate([rows[i] for i in sub]); b_part.y = torch.zeros(len(sub))
        f_part, _ = bf.extract_batch_bucket(models["F"], b_part, [mols_raw[i] for i in sub], [0.0] * len(sub), device)
        for j, m in enumerate(f_part):
            eq, w, wf = _features_equal(m, f_all[g * 4 + j])
            if not eq and not np.isfinite(w):
                mismatch_fields.append(f"g{g}j{j}:{wf}")   # structural mismatch
                worst_bc = np.inf
            else:
                worst_bc = max(worst_bc, w)                # float32 batch-shape drift
    batch_inv = {"n": len(subset), "max_abs_float_diff": float(worst_bc),
                 "structural_mismatch_fields": mismatch_fields[:16],
                 "ok": bool(np.isfinite(worst_bc) and worst_bc <= FEATURE_REPLAY_ATOL)}
    log(f"[extract:{split}{suffix}] batch-composition invariance max|d|={worst_bc:.3e} ok={batch_inv['ok']} {mismatch_fields[:4]}")

    # ---- S/O predictions + gauge invariance on the official split ----
    picks = list(range(n_mol))
    batch = collate_bucket(features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(torch.device(device))
    with torch.no_grad():
        out_s = models["S"](batch)
        out_s_raw = models["S_raw"](batch)
        out_o = models["O"](batch)
    s_pred = out_s["pred"].cpu().numpy().astype(np.float64)
    o_pred = out_o["pred"].cpu().numpy().astype(np.float64)
    gauge_check = {
        "max_abs_pred_diff_raw_vs_canonical": float(np.abs(out_s["pred"] - out_s_raw["pred"]).cpu().numpy().max()),
        "max_abs_gamma_diff": float(np.abs(out_s["gamma"] - out_s_raw["gamma"]).cpu().numpy().max()) if out_s["gamma"].numel() else 0.0,
        "max_abs_g_ctx_diff": float(np.abs(out_s["g_ctx"] - out_s_raw["g_ctx"]).cpu().numpy().max()),
        "max_abs_g_topo_diff": float(np.abs(out_s["g_topo"] - out_s_raw["g_topo"]).cpu().numpy().max()),
        "max_abs_bias_diff": float(np.abs(out_s["bias"] - out_s_raw["bias"]).cpu().numpy().max()),
        "max_abs_alpha_delta_identity_diff": float(
            np.abs(((out_s["alpha"] + out_s["delta"]) - (out_s_raw["alpha"] + out_s_raw["delta"])).cpu().numpy()).max()),
        "atol": ATOL_SUM,
    }
    gauge_check["ok"] = bool(
        gauge_check["max_abs_pred_diff_raw_vs_canonical"] <= ATOL_SUM
        and gauge_check["max_abs_alpha_delta_identity_diff"] <= ATOL_SUM
        and gauge_check["max_abs_gamma_diff"] == 0.0 and gauge_check["max_abs_g_ctx_diff"] == 0.0
        and gauge_check["max_abs_g_topo_diff"] == 0.0 and gauge_check["max_abs_bias_diff"] == 0.0)
    log(f"[extract:{split}] gauge invariance ok={gauge_check['ok']} ({gauge_check['max_abs_pred_diff_raw_vs_canonical']:.2e})")

    # ---- UNK routing stats (label-free) ----
    tids = batch.unit_type.cpu().numpy()
    n_known = ns.vocab.n_known
    unk_units = int((tids >= n_known).sum())
    global_unk_units = int((tids == ns.vocab.n_total - 1).sum())
    unk_stats = {
        "n_units": int(tids.size), "n_known_types": int(n_known), "n_vocab": int(ns.vocab.n_total),
        "unk_bucket_units": unk_units, "global_unk_units": global_unk_units,
        "unk_units_frac": unk_units / max(tids.size, 1),
    }
    umol = batch.unit_mol.cpu().numpy()
    mol_has_unk = np.zeros(batch.n_mols, dtype=bool)
    mol_has_unk[umol[tids >= n_known]] = True
    unk_stats["molecules_with_unk"] = int(mol_has_unk.sum())
    log(f"[extract:{split}{suffix}] UNK: {unk_stats['unk_bucket_units']}/{unk_stats['n_units']} units "
        f"(GLOBAL_UNK {unk_stats['global_unk_units']}); molecules with UNK = {unk_stats['molecules_with_unk']}")

    # ---- S contribution export + independent additivity reconstruction ----
    contrib_path = OUT / f"{split}{suffix}_S_bucket_contribs.npz"
    export_stats = _export_contributions(models, ns, features, picks, torch.device(device), contrib_path, split)
    recon = _verify_contrib_file(contrib_path, ATOL_SUM)
    log(f"[extract:{split}{suffix}] additivity from saved file: canonical={recon['additivity_max_abs_residual_canonical']:.2e} "
        f"raw={recon['additivity_max_abs_residual_raw']:.2e} ok={recon['ok']}")

    sha_after = _ckpt_sha()
    determinism = {"before": sha_before, "after": sha_after, "ok": bool(sha_before == sha_after)}
    log(f"[extract:{split}] checkpoint hashes unchanged: {determinism['ok']}")

    # ---- save prediction npz (per model + combined) ----
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    features_path = OUT / f"{split}{suffix}_inference_features.pt"
    torch.save({
        "feature_version": bf.FEATURE_VERSION, "protocol_id": PROTOCOL_ID, "split": split,
        "smoke": smoke, "n_molecules": n_mol,
        "code_fp_bucket_module": bf.code_fingerprint(),
        "frozen_checkpoint_sha16": {k: v[:16] for k, v in sha_before.items()},
        "data_fingerprint": {f"zinc_{split}_processed_sha256": _sha(REPO_ROOT / f"data/ZINC/subset/processed/{'val' if split == 'valid' else 'test'}.pt")},
        "audit": audit_totals, "alignment": alignment,
        "features": features,
    }, features_path)
    features_sha = _sha(features_path)
    meta_common = {
        "protocol_id": PROTOCOL_ID, "split": split, "smoke": smoke, "n_molecules": n_mol,
        "rows": np.arange(n_mol, dtype=np.int64),
        "checkpoint_sha256": sha_before, "input_features_sha256": features_sha,
        "feature_version": bf.FEATURE_VERSION, "generated_at": now,
        "data_fingerprint": _sha(REPO_ROOT / f"data/ZINC/subset/processed/{'val' if split == 'valid' else 'test'}.pt"),
        "y_in_file": False,
    }
    pred_files: dict[str, str] = {}
    for model_name, pred in (("F", f_pred), ("O_bucket", o_pred), ("S_bucket", s_pred)):
        path = OUT / f"{split}{suffix}_{model_name}_preds.npz"
        np.savez_compressed(path, model=model_name, pred=pred.astype(np.float64), **meta_common)
        pred_files[model_name] = path.name
    np.savez_compressed(
        OUT / f"{split}{suffix}_predictions.npz",
        row=meta_common["rows"], F_pred=f_pred, O_bucket_pred=o_pred, S_bucket_pred=s_pred,
        meta=json.dumps({k: v for k, v in meta_common.items() if k != "rows"}, default=str),
    )

    result = {
        "split": split, "alignment": alignment,
        "conservation_audit": audit_totals,
        "batch_composition_invariance": batch_inv,
        "gauge_invariance": gauge_check,
        "unk_stats": unk_stats,
        "contribution_export": {**export_stats, **recon, "file": contrib_path.name},
        "determinism": determinism,
        "prediction_files": pred_files,
        "features_file": {"name": features_path.name, "sha256": features_sha},
        "finiteness": {"F": bool(np.isfinite(f_pred).all()), "O": bool(np.isfinite(o_pred).all()), "S": bool(np.isfinite(s_pred).all())},
        "pred_lengths_ok": bool(len(f_pred) == len(o_pred) == len(s_pred) == n_mol),
        "smoke": smoke, "n_molecules": n_mol,
        "seconds": time.perf_counter() - started,
    }
    result["all_gates_ok"] = bool(
        result["gauge_invariance"]["ok"] and result["batch_composition_invariance"]["ok"]
        and result["contribution_export"]["ok"] and result["determinism"]["ok"]
        and all(result["finiteness"].values()) and result["pred_lengths_ok"]
        and not failed
    )
    _write_json(OUT / f"{split}{suffix}_extract_audit.json", result)
    log(f"[extract:{split}{suffix}] all_gates_ok={result['all_gates_ok']} ({result['seconds']:.0f}s)")
    if not result["all_gates_ok"]:
        raise RuntimeError(f"{split}: extraction engineering gates failed")
    return result


def _export_contributions(models, ns, features, picks: Sequence[int], device, path: Path, split: str) -> dict:
    """Strict-additivity export (canonical + raw) with unit atom membership.

    Mirrors ``run_cscl_v3d.export_gauged_contributions`` but consumes a caller
    supplied feature list (official-split inference features here).
    """
    gauged = models["S"]
    batch = collate_bucket(features, list(picks), ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    out = gauged(batch)
    pred = out["pred"].cpu().numpy().astype(np.float64)
    alpha = out["alpha"].cpu().numpy().astype(np.float64)
    delta = out["delta"].cpu().numpy().astype(np.float64)
    alpha_raw = out["alpha_raw"].cpu().numpy().astype(np.float64)
    delta_raw = out["delta_raw"].cpu().numpy().astype(np.float64)
    gamma = out["gamma"].cpu().numpy().astype(np.float64)
    g_ctx = out["g_ctx"].cpu().numpy().astype(np.float64)
    g_topo = out["g_topo"].cpu().numpy().astype(np.float64)
    bias = float(out["bias"].item())
    off = gauged.offset.detach().cpu().numpy().astype(np.float64)
    unit_type = batch.unit_type.cpu().numpy()
    unit_mol = batch.unit_mol.cpu().numpy()
    usum = np.zeros(batch.n_mols)
    np.add.at(usum, unit_mol, alpha + delta)
    usum_raw = np.zeros(batch.n_mols)
    np.add.at(usum_raw, unit_mol, alpha_raw + delta_raw)
    rsum = np.zeros(batch.n_mols)
    if len(gamma):
        np.add.at(rsum, batch.rel_mol.cpu().numpy(), gamma)
    recon = bias + usum + rsum + g_ctx + g_topo
    recon_raw = bias + usum_raw + rsum + g_ctx + g_topo
    atom_flat: list[int] = []
    atom_ptr = [0]
    unit_rows: list[int] = []
    for mi, fi in enumerate(picks):
        m = features[fi]
        for atoms in m.unit_atoms:
            atom_flat.extend(atoms)
            atom_ptr.append(len(atom_flat))
        unit_rows.extend([mi] * m.n_units)
    assert np.array_equal(np.asarray(unit_rows), unit_mol)
    np.savez_compressed(
        path,
        arm="S_bucket", protocol=PROTOCOL_ID, split=split,
        bias=bias,
        rows=np.asarray([features[i].row for i in picks], dtype=np.int64),
        pred=pred,
        unit_mol=unit_mol,
        unit_type=unit_type,
        unit_offset=off[unit_type],
        alpha=alpha, delta=delta,               # canonical
        alpha_raw=alpha_raw, delta_raw=delta_raw,
        unit_atom_flat=np.asarray(atom_flat, dtype=np.int64),
        unit_atom_ptr=np.asarray(atom_ptr, dtype=np.int64),
        rel_mol=batch.rel_mol.cpu().numpy(),
        gamma=gamma,
        g_ctx=g_ctx, g_topo=g_topo,
        g_ctx_per_feature=out["g_ctx_per_feature"].cpu().numpy().astype(np.float64),
        g_topo_per_feature=out["g_topo_per_feature"].cpu().numpy().astype(np.float64),
    )
    return {
        "additivity_max_abs_residual_canonical": float(np.abs(recon - pred).max()),
        "additivity_max_abs_residual_raw": float(np.abs(recon_raw - pred).max()),
        "global_bank_sum_residual": float(np.abs(
            out["g_ctx_per_feature"].cpu().numpy().astype(np.float64).sum(1)
            + out["g_topo_per_feature"].cpu().numpy().astype(np.float64).sum(1)
            - (g_ctx + g_topo)
        ).max()),
        "n_units": int(len(unit_type)), "n_relations": int(len(gamma)), "bias": bias,
    }


def _verify_contrib_file(path: Path, atol: float) -> dict:
    """INDEPENDENT additivity/gauge reconstruction from the SAVED npz only."""
    z = np.load(path, allow_pickle=False)
    pred = z["pred"].astype(np.float64)
    bias = float(z["bias"])
    usum = np.zeros(pred.shape[0])
    np.add.at(usum, z["unit_mol"], z["alpha"] + z["delta"])
    rsum = np.zeros(pred.shape[0])
    if z["gamma"].size:
        np.add.at(rsum, z["rel_mol"], z["gamma"])
    recon = bias + usum + rsum + z["g_ctx"] + z["g_topo"]
    res_can = float(np.abs(recon - pred).max())
    usum_raw = np.zeros(pred.shape[0])
    np.add.at(usum_raw, z["unit_mol"], z["alpha_raw"] + z["delta_raw"])
    res_raw = float(np.abs(bias + usum_raw + rsum + z["g_ctx"] + z["g_topo"] - pred).max())
    # gauge identities from the stored arrays
    off = z["unit_offset"]
    d_alpha = float(np.abs((z["alpha"] - z["alpha_raw"]) - off).max())
    d_delta = float(np.abs((z["delta_raw"] - z["delta"]) - off).max())
    # per-molecule shift cancellation: canonical and raw unit sums agree
    d_shift = float(np.abs(usum - usum_raw).max())
    # atom membership covers every atom of every molecule exactly once
    atom_ok = bool(np.all(np.diff(z["unit_atom_ptr"]) > 0))
    return {
        "additivity_max_abs_residual_canonical": res_can,
        "additivity_max_abs_residual_raw": res_raw,
        "alpha_offset_identity_max_abs": d_alpha,
        "delta_offset_identity_max_abs": d_delta,
        "unit_sum_shift_cancellation_max_abs": d_shift,
        "atom_membership_strictly_increasing": atom_ok,
        "ok": bool(res_can <= atol and res_raw <= atol and d_alpha <= atol
                   and d_delta <= atol and d_shift <= atol and atom_ok),
    }


# ---------------------------------------------------------------------------
# stage: score (ONE formal scoring per split; reads official y exactly once)
# ---------------------------------------------------------------------------

def _load_y(split: str) -> tuple[np.ndarray, object]:
    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val" if split == "valid" else "test")
    y = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))], dtype=np.float64)
    return y, ds


def _bootstrap_ci(d: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n_mol = d.size
    idx = rng.integers(0, n_mol, size=(n, n_mol))
    means = d[idx].mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "resamples": n, "seed": seed,
        "statistic": "signed per-molecule paired diff d_i=|e_A_i|-|e_B_i|; percentile bootstrap of the mean",
    }


def _score_split(split: str, reason: str | None = None, log=log) -> dict:
    """Read official y once for this split; metrics + paired bootstrap + band."""
    started = time.perf_counter()
    y, ds = _load_y(split)
    combined = np.load(OUT / f"{split}_predictions.npz", allow_pickle=False)
    meta = json.loads(str(combined["meta"]))
    preds = {name: combined[f"{name}_pred"].astype(np.float64) for name in ("F", "O_bucket", "S_bucket")}
    for name, p in preds.items():
        assert p.shape == y.shape and np.isfinite(p).all()
    assert np.array_equal(combined["row"], np.arange(1000)), "prediction row order broken"

    mae = {name: float(np.abs(p - y).mean()) for name, p in preds.items()}
    abserr = {name: np.abs(p - y) for name, p in preds.items()}
    mae_ci = {}
    for name, e in abserr.items():
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        idx = rng.integers(0, y.size, size=(BOOTSTRAP_N, y.size))
        means = e[idx].mean(axis=1)
        mae_ci[name] = {"mae": mae[name], "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                        "resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED}

    pairs = {}
    for a, b in (("S_bucket", "O_bucket"), ("S_bucket", "F"), ("O_bucket", "F")):
        d = abserr[a] - abserr[b]
        entry = _bootstrap_ci(d)
        entry["frac_a_better"] = float((abserr[a] < abserr[b]).mean())
        entry["n_molecules_a_strictly_better"] = int((abserr[a] < abserr[b]).sum())
        entry["n_molecules_b_strictly_better"] = int((abserr[b] < abserr[a]).sum())
        entry["n_molecules_tied"] = int((abserr[a] == abserr[b]).sum())
        pairs[f"{a}_minus_{b}"] = entry

    deltas_from_dev = {name: mae[name] - RECORDED_DEV[name] for name in mae}
    s_vs_o_abs = mae["S_bucket"] - mae["O_bucket"]
    s_vs_o_rel = s_vs_o_abs / mae["O_bucket"]

    metrics = {
        "protocol_id": PROTOCOL_ID, "split": split,
        "prediction_meta": meta,
        "n_molecules": int(y.size),
        "mae": mae,
        "mae_bootstrap_ci": mae_ci,
        "paired_differences": pairs,
        "deltas_from_internal_dev": deltas_from_dev,
        "S_minus_O_abs_increment": s_vs_o_abs,
        "S_minus_O_relative_increment": s_vs_o_rel,
        "scoring_read_y_from": f"PyG ZINC subset processed {'val' if split == 'valid' else 'test'}.pt (single read)",
        "scoring_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric_program_sha256": _sha(Path(__file__).resolve()),
    }
    _write_json(OUT / f"{split}_metrics.json", metrics)
    log(f"[score:{split}] MAE F={mae['F']:.5f} O={mae['O_bucket']:.5f} S={mae['S_bucket']:.5f}")
    log(f"[score:{split}] paired: S-O {pairs['S_bucket_minus_O_bucket']['mean_diff']:+.5f} "
        f"CI {pairs['S_bucket_minus_O_bucket']['ci95']}; S-F {pairs['S_bucket_minus_F']['mean_diff']:+.5f} "
        f"CI {pairs['S_bucket_minus_F']['ci95']}; O-F {pairs['O_bucket_minus_F']['mean_diff']:+.5f} "
        f"CI {pairs['O_bucket_minus_F']['ci95']}")

    # ---- one-formal-scoring access record ----
    rec_path = OUT / "test_access_record.json"
    record: dict[str, Any] = {}
    if rec_path.exists():
        record = json.loads(rec_path.read_text())
    splits_rec = record.setdefault("splits", {})
    key = "validation" if split == "valid" else "test"
    entry = {
        "first_formal_scoring_time": metrics["scoring_time"],
        "reason": reason,
        "prediction_files": {m: f"{split}_{m}_preds.npz" for m in preds},
        "prediction_combined_sha256": _sha(OUT / f"{split}_predictions.npz"),
        "official_split_source_sha256": _sha(REPO_ROOT / f"data/ZINC/subset/processed/{'val' if split == 'valid' else 'test'}.pt"),
        "metric_program_sha256": metrics["metric_program_sha256"],
        "protocol_frozen_commit": "55e088b",
        "user_authorised": True,
        "project_wide_pristine": False,
        "historical_contact_disclosure": (
            "the KSVD track historically read the official test split "
            "(tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_p1.py::unlock_test_stage, "
            "'project_wide_pristine': false); CSCL-v3d training/analysis never loaded "
            "official valid/test (train json discipline fields false; v2 driver never touches them)"
        ),
        "result_status": "frozen-version terminal result; not reusable to claim independence for any future configuration",
    }
    if key in splits_rec:
        prior = splits_rec[key]
        entry["re_run_of_first_scoring_at"] = prior["first_formal_scoring_time"]
        entry["re_run_reason"] = reason
        log(f"[score:{split}] WARNING: a formal scoring already exists for {key}; recorded as re-run (reason={reason})")
    splits_rec[key] = entry
    record["protocol_id"] = PROTOCOL_ID
    record.setdefault("model_ckpt_sha256", meta["checkpoint_sha256"])
    _write_json(rec_path, record)
    log(f"[score:{split}] done ({time.perf_counter() - started:.0f}s); access record updated")
    return metrics


# ---------------------------------------------------------------------------
# stage: analyze (attribution integrity + subgroups + frozen moment diagnostics)
# ---------------------------------------------------------------------------

def _group_mae(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    e = np.abs(pred[mask] - y[mask])
    return {"n": n, "mae": float(e.mean())}


def _paired_subgroup(pred_a, pred_b, y, mask) -> dict:
    if mask.sum() == 0:
        return {"n": 0}
    d = np.abs(pred_a[mask] - y[mask]) - np.abs(pred_b[mask] - y[mask])
    return {"n": int(mask.sum()), "mean_diff": float(d.mean()),
            "ci95": _bootstrap_ci(d, n=min(BOOTSTRAP_N, 2000))["ci95"]}


def _perturbed_features(features, kind: str, bucket: int | None = None):
    """Deep-copied features with a frozen label-free perturbation applied."""
    import copy

    out = []
    for m in features:
        m2 = copy.copy(m)
        if kind == "zero_unit_moments":
            m2.unit_bucket_sum_pv = np.zeros_like(m.unit_bucket_sum_pv)
            m2.unit_bucket_sum_pv2 = np.zeros_like(m.unit_bucket_sum_pv2)
        elif kind == "zero_unit_all":
            m2.unit_bucket_counts = np.zeros_like(m.unit_bucket_counts)
            m2.unit_bucket_sum_pv = np.zeros_like(m.unit_bucket_sum_pv)
            m2.unit_bucket_sum_pv2 = np.zeros_like(m.unit_bucket_sum_pv2)
        elif kind == "zero_rel_moments":
            m2.rel_bucket_sum_pv = np.zeros_like(m.rel_bucket_sum_pv)
            m2.rel_bucket_sum_pv2 = np.zeros_like(m.rel_bucket_sum_pv2)
        elif kind == "zero_rel_all":
            m2.rel_bucket_counts = np.zeros_like(m.rel_bucket_counts)
            m2.rel_bucket_sum_pv = np.zeros_like(m.rel_bucket_sum_pv)
            m2.rel_bucket_sum_pv2 = np.zeros_like(m.rel_bucket_sum_pv2)
        elif kind == "zero_unit_bucket_moments" and bucket is not None:
            m2.unit_bucket_sum_pv = m.unit_bucket_sum_pv.copy(); m2.unit_bucket_sum_pv[:, bucket, :] = 0.0
            m2.unit_bucket_sum_pv2 = m.unit_bucket_sum_pv2.copy(); m2.unit_bucket_sum_pv2[:, bucket, :] = 0.0
        elif kind == "zero_rel_bucket_moments" and bucket is not None:
            m2.rel_bucket_sum_pv = m.rel_bucket_sum_pv.copy(); m2.rel_bucket_sum_pv[:, bucket, :] = 0.0
            m2.rel_bucket_sum_pv2 = m.rel_bucket_sum_pv2.copy(); m2.rel_bucket_sum_pv2[:, bucket, :] = 0.0
        else:
            raise ValueError(kind)
        out.append(m2)
    return out


def _pred_on_features(models, ns, features, device):
    batch = collate_bucket(features, list(range(len(features))), ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    with torch.no_grad():
        return (models["S"](batch)["pred"].cpu().numpy().astype(np.float64),
                models["O"](batch)["pred"].cpu().numpy().astype(np.float64))


def _moment_sensitivity(models, ns, features_fit, features_valid, device, log=log) -> dict:
    """Frozen label-free sensitivity of the NEW per-bucket moment columns."""
    out: dict[str, Any] = {"note": ("zeroing = remove information; zero_moments keeps counts (synthetic task B ground "
                                    "truth was count-modulated), zero_all removes both; no training, no test labels")}
    for name, feats in (("fit_inner_1000", features_fit), ("valid_1000", features_valid)):
        p_s, p_o = _pred_on_features(models, ns, feats, device)
        res: dict[str, Any] = {"n": len(feats), "S": {}, "O": {}}

        def summ(base, pert):
            d = np.abs(pert - base)
            return {"mean_abs_dpred": float(d.mean()), "p90_abs_dpred": float(np.percentile(d, 90)),
                    "max_abs_dpred": float(d.max()), "frac_abs_dpred_gt_1e-4": float((d > 1e-4).mean())}

        for kind in ("zero_unit_moments", "zero_unit_all", "zero_rel_moments", "zero_rel_all"):
            ps2, po2 = _pred_on_features(models, ns, _perturbed_features(feats, kind), device)
            res["S"][kind] = summ(p_s, ps2)
            res["O"][kind] = summ(p_o, po2)
        for b in range(bf.N_BUCKETS):
            ps2, _ = _pred_on_features(models, ns, _perturbed_features(feats, "zero_rel_bucket_moments", b), device)
            res["S"][f"zero_rel_bucket_moments_b{b}"] = summ(p_s, ps2)
            ps3, _ = _pred_on_features(models, ns, _perturbed_features(feats, "zero_unit_bucket_moments", b), device)
            res["S"][f"zero_unit_bucket_moments_b{b}"] = summ(p_s, ps3)
        out[name] = res
        log(f"[sensitivity:{name}] S zero_unit_moments mean|dpred|={res['S']['zero_unit_moments']['mean_abs_dpred']:.4f} "
            f"zero_rel_moments {res['S']['zero_rel_moments']['mean_abs_dpred']:.4f}; "
            f"O zero_unit_moments {res['O']['zero_unit_moments']['mean_abs_dpred']:.4f}")
    return out


def stage_analyze(device: str = "cpu", log=log) -> dict:
    """Attribution integrity + subgroups + moment sensitivity (labels only via metrics files)."""
    started = time.perf_counter()
    models = load_frozen_models(log=log)
    ns = models["ns"]

    integrity: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "atol": ATOL_SUM}
    fit_pos = v3._index_by_row(ns.features, ns.idx["fit_inner"])
    fit_batch = v3.collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    fit_counts = np.bincount(fit_batch.unit_type.cpu().numpy().astype(np.int64), minlength=ns.vocab.n_total)

    subgroup: dict[str, Any] = {"protocol_id": PROTOCOL_ID}
    for split in ("valid", "test"):
        contrib = np.load(OUT / f"{split}_S_bucket_contribs.npz", allow_pickle=False)
        ver = _verify_contrib_file(OUT / f"{split}_S_bucket_contribs.npz", ATOL_SUM)
        ut = contrib["unit_type"].astype(np.int64)
        umol = contrib["unit_mol"].astype(np.int64)
        n_mols = contrib["pred"].shape[0]
        # alpha is a shared per-type scalar: spread within type must be ~0 (float32)
        alpha_spread = 0.0
        for t in np.unique(ut):
            m = ut == t
            if m.sum() > 1:
                alpha_spread = max(alpha_spread, float(np.ptp(contrib["alpha"][m])))
        per_type_fit_counts = fit_counts[ut]
        rsum = np.bincount(contrib["rel_mol"].astype(np.int64), weights=contrib["gamma"], minlength=n_mols) \
            if contrib["gamma"].size else np.zeros(n_mols)
        integrity[split] = {
            **ver,
            "alpha_within_type_spread_max": alpha_spread,
            "alpha_is_shared_type_scalar": bool(alpha_spread <= ATOL_SUM),
            "units_total": int(ut.size), "relations_total": int(contrib["gamma"].size),
            "units_of_fit_known_types": int((ut < ns.vocab.n_known).sum()),
            "units_of_unk_buckets": int((ut >= ns.vocab.n_known).sum()),
            "units_global_unk": int((ut == ns.vocab.n_total - 1).sum()),
            "fit_types_present": int(np.unique(ut[ut < ns.vocab.n_known]).size),
            "gamma_sum_per_molecule_mean": float(rsum.mean()),
            "gamma_sum_per_molecule_std": float(rsum.std()),
            "gamma_sum_per_molecule_abs_mean": float(np.abs(rsum).mean()),
            "per_unit_fit_instance_count_at_its_type": {
                "min": int(per_type_fit_counts.min()), "median": float(np.median(per_type_fit_counts))},
        }
        log(f"[integrity:{split}] additivity_ok={ver['ok']} alpha_scalar_ok={integrity[split]['alpha_is_shared_type_scalar']} "
            f"UNK units={integrity[split]['units_of_unk_buckets']}")

        # bonded (bucket-0) vs non-bonded relations need features; reload them
        feats = torch.load(OUT / f"{split}_inference_features.pt", map_location="cpu", weights_only=False)["features"]
        n_bonded = sum(int((m.rel_bucket_counts[:, 0] > 0).sum()) for m in feats)
        integrity[split]["relations_with_direct_bond"] = int(n_bonded)

        # ---- subgroups (labels read here for descriptive slices, post-scoring) ----
        y, _ds = _load_y(split)
        preds = np.load(OUT / f"{split}_predictions.npz", allow_pickle=False)
        pf, po, ps = (preds["F_pred"].astype(np.float64), preds["O_bucket_pred"].astype(np.float64),
                      preds["S_bucket_pred"].astype(np.float64))
        mol_has_unk = np.zeros(n_mols, dtype=bool)
        mol_has_unk[umol[ut >= ns.vocab.n_known]] = True
        n_unk = np.bincount(umol[ut >= ns.vocab.n_known], minlength=n_mols) if (ut >= ns.vocab.n_known).any() else np.zeros(n_mols, dtype=int)
        n_atoms = np.asarray([m.n_atoms for m in feats])
        n_ring_units = np.asarray([sum(1 for k in m.unit_kinds if int(k) == cf.KIND_RING) for m in feats])
        # molecule-level cyclomatic number: edges (intra + inter unit bonds) - atoms + 1
        edges = np.asarray([float(m.unit_desc[:, cf.ATOM_CATEGORIES:cf.ATOM_CATEGORIES + 4].sum() + m.rel_bond_hist.sum()) for m in feats])
        cyclo = edges - n_atoms + 1.0
        bucket0_share = np.asarray([float(m.unit_bucket_counts[:, 0].sum() + m.rel_bucket_counts[:, 0].sum()) / max(m.n_pairs, 1) for m in feats])
        sg: dict[str, Any] = {}
        for name, mask in (
            ("all", np.ones(n_mols, dtype=bool)),
            ("has_UNK", mol_has_unk), ("no_UNK", ~mol_has_unk),
            ("atoms_le_10", n_atoms <= 10), ("atoms_11_20", (n_atoms > 10) & (n_atoms <= 20)),
            ("atoms_gt_20", n_atoms > 20),
            ("ring_units_0", n_ring_units == 0), ("ring_units_1", n_ring_units == 1), ("ring_units_ge_2", n_ring_units >= 2),
            ("cyclomatic_le_3", cyclo <= 3), ("cyclomatic_ge_4", cyclo >= 4),
            ("bucket0_share_le_0.2", bucket0_share <= 0.2), ("bucket0_share_gt_0.2", bucket0_share > 0.2),
            ("y_le_minus3", y <= -3.0), ("y_gt_minus3", y > -3.0),
        ):
            sg[name] = {
                "n": int(mask.sum()),
                "mae": {m: _group_mae(p, y, mask) for m, p in (("F", pf), ("O", po), ("S", ps))},
                "S_minus_O": _paired_subgroup(ps, po, y, mask),
                "S_minus_F": _paired_subgroup(ps, pf, y, mask),
                "O_minus_F": _paired_subgroup(po, pf, y, mask),
            }
        subgroup[split] = {
            "factors": "label-free structural groups + descriptive y<=-3 slice (post-scoring, pre-declared)",
            "subgroups": sg,
            "unk_units_per_molecule_mean": float(n_unk.mean()),
        }

    # ---- frozen moment-usage sensitivity (train-domain + valid extrapolation) ----
    feats_valid = torch.load(OUT / "valid_inference_features.pt", map_location="cpu", weights_only=False)["features"]
    fit_sample = [ns.features[i] for i in fit_pos[:1000]]
    sensitivity = _moment_sensitivity(models, ns, fit_sample, feats_valid, device, log=log)

    result = {"integrity": integrity, "subgroup_analysis": subgroup, "moment_sensitivity": sensitivity,
              "seconds": time.perf_counter() - started}
    _write_json(OUT / "attribution_integrity.json", integrity)
    _write_json(OUT / "subgroup_analysis.json", subgroup)
    _write_json(OUT / "moment_sensitivity.json", sensitivity)
    log(f"[analyze] done ({result['seconds']:.0f}s)")
    return result


# ---------------------------------------------------------------------------
# final judgment + report helpers
# ---------------------------------------------------------------------------

def stage_judgment(log=log) -> dict:
    """Pre-frozen exploratory bands over the two scored splits (no adaptation)."""
    vm = json.loads((OUT / "valid_metrics.json").read_text())
    tm = json.loads((OUT / "test_metrics.json").read_text())
    so_t = tm["paired_differences"]["S_bucket_minus_O_bucket"]
    sf_t = tm["paired_differences"]["S_bucket_minus_F"]
    so_v = vm["paired_differences"]["S_bucket_minus_O_bucket"]
    sf_v = vm["paired_differences"]["S_bucket_minus_F"]

    direction_consistent = bool(
        np.sign(so_t["mean_diff"]) == np.sign(so_v["mean_diff"])
        and np.sign(sf_t["mean_diff"]) == np.sign(sf_v["mean_diff"])
        and abs(so_t["mean_diff"] - so_v["mean_diff"]) <= TOL_BAND
        and abs(sf_t["mean_diff"] - sf_v["mean_diff"]) <= TOL_BAND)
    strong = bool(so_t["ci95"][1] <= TOL_BAND and sf_t["ci95"][1] <= TOL_BAND and direction_consistent)
    limited = bool(not strong and so_t["mean_diff"] <= TOL_BAND and sf_t["mean_diff"] <= TOL_BAND)
    band = "strong_support" if strong else ("limited_support" if limited else "no_support")

    of_t = tm["paired_differences"]["O_bucket_minus_F"]
    o_f_equivalent = bool(EQUIV_REGION[0] <= of_t["ci95"][0] and of_t["ci95"][1] <= EQUIV_REGION[1])

    result = {
        "protocol_id": PROTOCOL_ID,
        "band": band,
        "band_definition": {
            "strong_support": f"test S-O and S-F paired CI upper <= {TOL_BAND}; validation direction consistent",
            "limited_support": f"S-O and S-F point estimates <= {TOL_BAND} but a CI upper bound exceeds it",
            "no_support": "a test gap clearly exceeds the tolerance band or integrity checks failed",
            "direction_consistent": "same sign of S-O and S-F point estimates on valid and test AND |point estimate difference across splits| <= 0.02 (pre-scoring operationalization of 'no serious contradiction')",
        },
        "direction_consistent": direction_consistent,
        "o_vs_f_equivalence_region": EQUIV_REGION,
        "o_vs_f_ci_within_region": o_f_equivalent,
        "o_vs_f_wording": ("statistically equivalent within [-0.01,+0.01]" if o_f_equivalent
                           else "equivalence NOT established; only the observed MAE difference and its uncertainty are reported"),
    }
    _write_json(OUT / "judgment.json", result)
    paired = {
        "protocol_id": PROTOCOL_ID,
        "note": "signed per-molecule paired differences d_i = |e_A_i| - |e_B_i|; percentile bootstrap (5000, seed 0); splits never mixed",
        "valid": vm["paired_differences"], "test": tm["paired_differences"],
        "mae": {"valid": vm.get("mae", {}), "test": tm.get("mae", {})},
        "deltas_from_internal_dev": {"valid": vm.get("deltas_from_internal_dev", {}), "test": tm.get("deltas_from_internal_dev", {})},
        "band": band, "o_vs_f": {k: result[k] for k in ("o_vs_f_equivalence_region", "o_vs_f_ci_within_region", "o_vs_f_wording")},
    }
    _write_json(OUT / "paired_comparisons.json", paired)
    log(f"[judgment] band={band} direction_consistent={direction_consistent} O-F equivalence={o_f_equivalent}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "replay", "extract", "score", "analyze", "judgment", "all"])
    parser.add_argument("--split", choices=["valid", "test"], help="for extract/score")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reason", default=None, help="scoring re-run reason (recorded, never silent)")
    parser.add_argument("--limit", type=int, default=None,
                        help="EXTRACT SMOKE ONLY: first N molecules into *_smoke files; never scores")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze":
        stage_freeze()
    elif args.stage == "replay":
        stage_replay()
    elif args.stage == "extract":
        assert args.split, "--split required"
        _predict_split(args.split, args.device, limit=args.limit)
    elif args.stage == "score":
        assert args.split, "--split required"
        _score_split(args.split, reason=args.reason)
    elif args.stage == "analyze":
        stage_analyze(args.device)
    elif args.stage == "judgment":
        stage_judgment()
    elif args.stage == "all":
        stage_freeze()
        stage_replay()
        _predict_split("valid", args.device)
        _predict_split("test", args.device)
        _score_split("valid", reason=args.reason)
        _score_split("test", reason=args.reason)
        stage_analyze(args.device)
        stage_judgment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
