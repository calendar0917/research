"""Driver for protocol ``cscl-bjoc-source-ownership-v1``.

  Phase 0 (CPU, label-free, frozen R3): ``phase0`` -> 0A/0B/0C + verdict
  Phase 1 (CPU gates):                  ``cpu-gates``
  freeze:                               ``freeze`` / ``verify-inputs``
  Phase 2 (GPU, cap 1.5 h):             ``arm`` -> ``predict`` -> ``score-monitor``
  mechanism audit (CPU):                ``mechanism``
  secondary descriptive valid compare:  ``valid-secondary``
  final:                                ``finalize``

Discipline: official test is never referenced; the official valid set is an
exposed development set used descriptively only; Monitor labels are read once
after all seed-0 predictions are frozen; the Locked split is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bjoc_source_attribution_feasibility_v1 as bsaf  # noqa: E402
import bjoc_source_ownership_v1 as own  # noqa: E402
import bjoc_structure_explanation_audit_v1 as v1  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = own.PROTOCOL_ID
TRACK = REPO_ROOT / "tracks/structure-contribution"
OUT = TRACK / "results" / "bjoc_source_ownership_v1"
CONFIG_PATH = TRACK / "configs" / "bjoc_source_ownership_v1.yaml"

DATA = TRACK / "results" / "bjoc_task_adaptive_ssl_v1"       # split D, objects, SSL, P3
F10K = TRACK / "results" / "bjoc_full10k_pretraining_transfer_v1"
REC = TRACK / "results" / "bjoc_recipe_tuning_v1"
BSAF = TRACK / "results" / "bjoc_source_attribution_feasibility_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"

SEEDS = (0, 1)
ARMS = ("N", "O", "S")
SPLIT_TAG = "D"
PHASE0_SEEDS = (0, 1)
PHASE0_MATCH_CONTROLS = 8
PHASE0_ALPHA = 1.0
PHASE0_RANDOM_DRAWS = 16
PHASE0_PERMUTATION_DRAWS = 1000
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
VALID_REFERENCE_R3 = {
    "seed0": 0.079101,
    "seed1": 0.080586,
    "source": "results/bjoc_recipe_tuning_v1/REPORT.md (B2 confirmation table)",
    "note": "frozen recipe-tuning R3, replayed through the same official-valid pipeline; replay guard tolerance 1e-5",
}

SPLIT_FINGERPRINTS = {
    "fit_sha256_rows": "9468656affcc2b7c69409f03698b1b6287b3a42dc6e164efd19dfb0190254525",
    "monitor_sha256_rows": "d196bc75ce898c7cfd61efa133372f25803d5bc1198210f60b6e0f981a400393",
    "locked_sha256_rows": "f3525ae882114350a57f7b0ddb6889656f40c05cebda21b2611a2c99cdfa22d5",
}
SSL_RECORDS = {0: "pretrain_D_s0.json", 1: "pretrain_D_s1.json"}


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(v1._j(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return v1.file_sha256(path)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return DATA / "split_D.json"


def objects_path(zone: str) -> Path:
    return DATA / f"objects_{zone}_D.pt"


def ssl_path(seed: int) -> Path:
    return DATA / f"pretrain_D_s{seed}.pt"


def ssl_record_path(seed: int) -> Path:
    return DATA / SSL_RECORDS[seed]


def p3_soup_path(seed: int) -> Path:
    return DATA / f"soup_P3_D_s{seed}.pt"


def p3_pred_path(seed: int) -> Path:
    return DATA / "predictions" / f"monitor_P3_D_s{seed}.npz"


def monitor_y_path() -> Path:
    return DATA / "monitor_y_D.npy"


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_D_s{seed}.pt"


def train_json_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_D_s{seed}.json"


def monitor_pred_path(arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"monitor_{arm}_D_s{seed}.npz"


def valid_pred_path(arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"valid_{arm}_D_s{seed}.npz"


def valid_pred_path_r3(seed: int) -> Path:
    return OUT / "predictions" / f"valid_R3_s{seed}.npz"


def valid_pred_path_p3(seed: int) -> Path:
    return OUT / "predictions" / f"valid_P3D_s{seed}.npz"


PRED_MANIFEST = OUT / "prediction_manifest.json"
MONITOR_LOCK = OUT / "monitor_read.lock"
MONITOR_Y_COPY = OUT / "monitor_y_read_once.npy"
VALID_LOCK = OUT / "valid_read.lock"
VALID_Y_COPY = OUT / "valid_y_read_once.npy"
FREEZE = OUT / "protocol_freeze.json"


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    return device


# ---------------------------------------------------------------------------
# input audit / freeze
# ---------------------------------------------------------------------------


def code_paths() -> list[Path]:
    return [
        TRACK / "code" / "bjoc_source_ownership_v1.py",
        TRACK / "code" / "run_bjoc_source_ownership_v1.py",
        TRACK / "tests" / "test_bjoc_source_ownership_v1.py",
        CONFIG_PATH,
    ]


def code_hashes() -> dict[str, str]:
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in code_paths()}


def audit_inputs() -> dict[str, Any]:
    split = read_json(split_path())
    observed = {k: split.get(k) for k in SPLIT_FINGERPRINTS}
    inputs = {
        "split_D": {
            "path": str(split_path().relative_to(REPO_ROOT)),
            "sha256": file_sha256(split_path()),
            "n_fit": len(split["fit_rows"]),
            "n_monitor": len(split["monitor_rows"]),
            "n_locked": len(split["locked_rows"]),
            "fingerprints_match": observed == SPLIT_FINGERPRINTS,
            "observed": observed,
        },
        "objects": {},
        "ssl": {},
        "phase0": {},
    }
    for zone in ("fit", "monitor", "locked"):
        p = objects_path(zone)
        inputs["objects"][zone] = {"path": str(p.relative_to(REPO_ROOT)), "exists": p.exists(), "sha256": file_sha256(p) if p.exists() else None}
    for seed in SEEDS:
        p = ssl_path(seed)
        rec = read_json(ssl_record_path(seed))
        inputs["ssl"][str(seed)] = {
            "path": str(p.relative_to(REPO_ROOT)),
            "exists": p.exists(),
            "sha256": file_sha256(p) if p.exists() else None,
            "recorded_encoder_final_sha256": rec.get("encoder_final_sha256"),
            "recorded_updates": rec.get("updates"),
            "recorded_fit_rows_only": rec.get("fit_rows_only"),
        }
        p3 = p3_soup_path(seed)
        inputs["ssl"][f"p3_soup_{seed}"] = {"path": str(p3.relative_to(REPO_ROOT)), "exists": p3.exists(), "sha256": file_sha256(p3) if p3.exists() else None}
        inputs["ssl"][f"p3_pred_{seed}"] = {"path": str(p3_pred_path(seed).relative_to(REPO_ROOT)), "exists": p3_pred_path(seed).exists()}
    for tag, p in (
        ("r3_soup0", REC / "soup_R3_seed0.pt"),
        ("r3_soup1", REC / "soup_R3_seed1.pt"),
        ("objects_train10k", F10K / "objects_train10k.pt"),
        ("prep_train10k", F10K / "prep_train10k.json"),
        ("objects_valid", F10K / "objects_valid.pt"),
        ("sample_manifest", BSAF / "sample_manifest.json"),
        ("encoded_train", ENCODED_TRAIN),
    ):
        inputs["phase0"][tag] = {"path": str(p.relative_to(REPO_ROOT)), "exists": p.exists(), "sha256": file_sha256(p) if p.exists() else None}
    ok = (
        inputs["split_D"]["fingerprints_match"]
        and all(v["exists"] for v in inputs["objects"].values())
        and all(v.get("exists") for k, v in inputs["ssl"].items() if k in {"0", "1"})
        and all(v["exists"] for v in inputs["phase0"].values())
    )
    inputs["all_present"] = ok
    return inputs


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    payload = audit_inputs()
    write_json(OUT / "input_audit.json", payload)
    if not payload["all_present"]:
        log("[verify-inputs] MISSING INPUTS")
        return 1
    log("[verify-inputs] all inputs present")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    audit = audit_inputs()
    if not audit["all_present"]:
        raise SystemExit("cannot freeze: missing inputs")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_sha256": code_hashes(),
        "input_audit_sha256": file_sha256(OUT / "input_audit.json"),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "monitor_labels_read": MONITOR_LOCK.exists(),
        "locked_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(FREEZE, payload)
    log("[freeze] protocol_freeze.json written")
    return 0


def verify_freeze() -> dict[str, Any]:
    if not FREEZE.exists():
        raise SystemExit("protocol not frozen: run freeze first")
    frozen = read_json(FREEZE)
    for path, sha in frozen["code_sha256"].items():
        if file_sha256(REPO_ROOT / path) != sha:
            raise SystemExit(f"frozen code changed: {path} (amend via a recorded protocol amendment)")
    if file_sha256(CONFIG_PATH) != frozen["config_sha256"]:
        raise SystemExit("frozen config changed")
    return frozen


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def _reference_w_map(model: src.BJOC, objs: Sequence[sro.MolObjects], rows: Sequence[int]) -> dict[int, torch.Tensor]:
    batch = v1.make_batch(objs, rows)
    feats = v1.run_features(model, batch)
    out: dict[int, torch.Tensor] = {}
    for g, row in enumerate(rows):
        w = feats["w"][batch["junc_batch"] == g]
        if w.shape[0] >= 1:
            out[int(row)] = w
    return out


def _subset_response(model: src.BJOC, pack: Mapping[str, Any], w_ref: torch.Tensor, subset: Sequence[int], alpha: float = PHASE0_ALPHA) -> np.ndarray:
    w = pack["w"]
    with torch.no_grad():
        pred_nat = float(bsaf.predict_null(model, pack).item())
        out = np.zeros(len(subset), dtype=np.float64)
        for t, j in enumerate(subset):
            j = int(j)
            w2 = w.clone()
            w2[j] = w[j] + float(alpha) * (w_ref[j % w_ref.shape[0]] - w[j])
            out[t] = float(bsaf.predict_null(model, pack, w=w2).item()) - pred_nat
    return out


def _top_k(score: torch.Tensor | np.ndarray, k: int) -> list[int]:
    arr = np.asarray(score).reshape(-1)
    return [int(i) for i in np.argsort(-arr, kind="stable")[:k]]


def _sign_test_one_sided(deltas: np.ndarray) -> dict[str, float]:
    d = np.asarray(deltas, dtype=np.float64)
    d = d[~np.isnan(d)]
    n = int(d.size)
    k = int((d > 0).sum())
    p = float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0**n)) if n else float("nan")
    return {"n_positive": k, "n": n, "p_one_sided": p}


def _bootstrap(deltas: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=np.float64)
    m = arr.mean()
    draws = rng.choice(arr, size=(n, arr.size), replace=True).mean(1)
    return {
        "mean": float(m),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


def phase0() -> dict[str, Any]:
    verify_freeze()
    cfg = load_config()
    objs = v1.load_objects()
    sample = read_json(BSAF / "sample_manifest.json")
    diag_rows = [int(r) for r in sample["diagnostic_rows"]]
    ref_rows = [int(r) for r in sample["reference_rows"]]
    norm = v1.load_norm()
    log(f"[phase0] corpus objects {len(objs)}; diagnostic {len(diag_rows)}; reference {len(ref_rows)}")
    counts = own.signature_corpus_counts(objs)
    corpus_total = int(sum(counts.values()))
    log(f"[phase0] signature classes {len(counts)} corpus junctions {corpus_total}")

    fact1 = {"per_seed": {}}
    b_records: list[dict[str, Any]] = []
    c_records: list[dict[str, Any]] = []
    for seed in PHASE0_SEEDS:
        model = v1.load_model(seed, norm)
        ref_w_map = _reference_w_map(model, objs, ref_rows)
        ref_valid = [(int(r), sample["reference_stats"][str(int(r))], ref_w_map[int(r)]) for r in ref_rows if int(r) in ref_w_map]
        occ_err, ig_occ_sp = [], []
        joint_fact = []
        for row in diag_rows:
            stats = sample["diagnostic_stats"][str(int(row))]
            ref_row, _, w_ref = min(
                ref_valid,
                key=lambda t: (abs(t[1]["n_atoms"] - stats["n_atoms"]) + abs(t[1]["n_bonds"] - stats["n_bonds"]) + abs(t[1]["n_junctions"] - stats["n_junctions"]), t[0]),
            )
            batch = v1.make_batch(objs, [int(row)])
            feats = v1.run_features(model, batch)
            pack = bsaf.pack_from_features(model, batch, feats, 0)
            n_j = int(pack["n_junctions"])
            if n_j < 1:
                continue
            w = pack["w"]
            mu = w.mean(0)
            ig = bsaf.ig_scores(model, pack, mu)
            occ = bsaf.occlusion_scores(model, pack, mu)
            occ_err.append(float((ig["ig"] + occ).abs().max() / max(float(occ.abs().mean()), 1e-9)))
            ig_occ_sp.append(float(np.corrcoef(np.argsort(np.argsort(ig["ig"].numpy())), np.argsort(np.argsort(occ.numpy())))[0, 1]))
            # Fact 2: joint reset -> first moment unchanged, second moment = mean^2
            with torch.no_grad():
                reset = mu.unsqueeze(0).repeat(n_j, 1)
                mu_reset = reset.mean(0)
                nu_reset = (reset * reset).mean(0)
                joint_delta = float(bsaf.predict_from_mu_nu(model, pack, mu, mu * mu).item()) - float(bsaf.predict_null(model, pack).item())
            joint_fact.append(
                {
                    "row": int(row),
                    "mu_reset_max_abs": float((mu_reset - mu).abs().max()),
                    "nu_reset_max_abs": float((nu_reset - mu * mu).abs().max()),
                    "joint_reset_delta": joint_delta,
                    "signed_ig_sum": float(ig["ig"].sum()),
                }
            )
            dev = own.dev_ranking(w).numpy()
            rare = own.rarity_ranking(objs[int(row)], counts, corpus_total)
            ig_abs = ig["ig"].abs().numpy()
            k = min(int(cfg["phase0"]["top_k"]), n_j)
            scores = {"IG": ig_abs, "dev": dev, "rare": rare}
            sets = {name: _top_k(arr, k) for name, arr in scores.items()}
            resp = {name: _subset_response(model, pack, w_ref, idxs) for name, idxs in sets.items()}
            complement = [i for i in range(n_j) if i not in set(sets["IG"])]
            rng = np.random.default_rng(70_000_000 + seed * 100_000 + int(row))
            rand_resp = []
            for _ in range(PHASE0_RANDOM_DRAWS):
                if complement:
                    draw = rng.choice(np.asarray(complement), size=min(k, len(complement)), replace=False)
                    rand_resp.extend(_subset_response(model, pack, w_ref, draw).tolist())
            # matched controls for IG top junctions
            sig_by_j = {j: own.junction_signature(objs[int(row)], j) for j in range(n_j)}
            deg_by_j = {j: own.center_degree(objs[int(row)], j) for j in range(n_j)}
            gaps, cover_hits, cover_total = [], 0, 0
            dev_exact_hits = 0
            dev_exact_total = 0
            ig_top_resp = resp["IG"]
            for pos, j in enumerate(sets["IG"]):
                cover_total += 1
                ctrl = own.matched_controls(
                    objs[int(row)],
                    j,
                    dev,
                    sig_by_j[j],
                    deg_by_j[j],
                    sig_by_j[j][3],
                    set(sets["IG"]),
                    PHASE0_MATCH_CONTROLS,
                    rng,
                )
                if not ctrl:
                    continue
                cover_hits += 1
                dev_exact_total += 1
                dev_exact_hits += sum(1 for c in ctrl if int(np.floor(dev[c] * 10.0)) == int(np.floor(dev[j] * 10.0)))
                ctrl_resp = _subset_response(model, pack, w_ref, ctrl)
                gaps.append(float(ig_top_resp[pos] - ctrl_resp.mean()))
            b_records.append(
                {
                    "seed": int(seed),
                    "row": int(row),
                    "reference_row": int(ref_row),
                    "n_junctions": n_j,
                    "top4_response": {name: float(np.abs(v).mean()) for name, v in resp.items()},
                    "random_response": float(np.abs(np.asarray(rand_resp, dtype=np.float64)).mean()) if rand_resp else None,
                    "matched_control_gap": float(np.mean(gaps)) if gaps else None,
                    "matched_control_covered": bool(cover_hits and cover_hits == cover_total),
                    "matched_cover_hits": cover_hits,
                    "matched_cover_total": cover_total,
                    "matched_dev_exact_hits": dev_exact_hits,
                    "matched_dev_exact_total": dev_exact_total,
                    "top4_sets": {name: idxs for name, idxs in sets.items()},
                    "mean_dev": float(dev.mean()),
                    "top4_mean_dev": {name: float(dev[idxs].mean()) for name, idxs in sets.items()},
                    "top4_mean_center_degree": {name: float(np.mean([deg_by_j[j] for j in idxs])) for name, idxs in sets.items()},
                    "top4_ring_fraction": {name: float(np.mean([sig_by_j[j][3] for j in idxs])) for name, idxs in sets.items()},
                }
            )
            for j in range(n_j):
                c_records.append(
                    {
                        "seed": int(seed),
                        "row": int(row),
                        "j": int(j),
                        "signature": list(sig_by_j[j]),
                        "degree": int(deg_by_j[j]),
                        "in_ring": int(sig_by_j[j][3]),
                        "ig": float(ig["ig"][j]),
                        "ig_centered": float(ig["ig"][j] - ig["ig"].mean()),
                        "ig_abs": float(ig_abs[j]),
                        "dev": float(dev[j]),
                        "rare": float(rare[j]),
                    }
                )
            if (len(b_records) % 32) == 0:
                log(f"[phase0] seed {seed}: {len(b_records)} molecules done")
        fact1["per_seed"][str(seed)] = {
            "ig_vs_occlusion_max_relative_error_mean": float(np.mean(occ_err)),
            "ig_vs_occ_rank_corr_mean": float(np.mean(ig_occ_sp)),
            "joint_reset_mu_max_abs": float(np.max([f["mu_reset_max_abs"] for f in joint_fact])),
            "joint_reset_nu_max_abs": float(np.max([f["nu_reset_max_abs"] for f in joint_fact])),
            "joint_reset_delta_mean": float(np.mean([f["joint_reset_delta"] for f in joint_fact])),
            "signed_ig_sum_mean": float(np.mean([f["signed_ig_sum"] for f in joint_fact])),
        }
    # ---- 0B aggregates ----
    per_mol: dict[tuple[int, int], dict[str, Any]] = {}
    for r in b_records:
        per_mol[(r["seed"], r["row"])] = r
    keys = sorted(per_mol)
    b_agg: dict[str, Any] = {"n_molecules": len(keys), "contrasts": {}}
    for name in ("dev", "rare"):
        d_ig = np.asarray([per_mol[k]["top4_response"]["IG"] - per_mol[k]["top4_response"][name] for k in keys])
        b_agg["contrasts"][f"IG_minus_{name}"] = {**_bootstrap(d_ig), **bsaf.sign_test(d_ig), **_sign_test_one_sided(d_ig)}
        b_agg["contrasts"][f"{name}_mean_response"] = float(np.mean([per_mol[k]["top4_response"][name] for k in keys]))
    d_ig_rand = np.asarray([per_mol[k]["top4_response"]["IG"] - per_mol[k]["random_response"] for k in keys if per_mol[k]["random_response"] is not None])
    b_agg["contrasts"]["IG_minus_random"] = {**_bootstrap(d_ig_rand), **bsaf.sign_test(d_ig_rand), **_sign_test_one_sided(d_ig_rand)}
    b_agg["contrasts"]["random_mean_response"] = float(np.mean([per_mol[k]["random_response"] for k in keys if per_mol[k]["random_response"] is not None]))
    b_agg["contrasts"]["IG_mean_response"] = float(np.mean([per_mol[k]["top4_response"]["IG"] for k in keys]))
    gaps = np.asarray([per_mol[k]["matched_control_gap"] for k in keys if per_mol[k]["matched_control_gap"] is not None])
    cover = float(np.mean([r["matched_cover_hits"] / max(r["matched_cover_total"], 1) for r in b_records]))
    dev_exact = float(sum(r["matched_dev_exact_hits"] for r in b_records) / max(sum(r["matched_dev_exact_total"] for r in b_records), 1))
    b_agg["matched_control"] = {
        **_bootstrap(gaps),
        **bsaf.sign_test(gaps),
        **_sign_test_one_sided(gaps),
        "coverage": cover,
        "coverage_definition": "fraction of IG top-4 junctions with >=1 same-molecule control matched on (signature, center degree, ring), dev-decile preferred",
        "n_matched_molecules": int(gaps.size),
        "n_matched_junctions": int(sum(r["matched_cover_hits"] for r in b_records)),
        "unit": "one gap per molecule (mean over its matched top-4 junctions); bootstrap/sign test over molecules",
        "dev_decile_exact_fraction": dev_exact,
    }
    # structural composition comparison
    comp: dict[str, Any] = {}
    for name in ("IG", "dev", "rare"):
        comp[name] = {
            "mean_top4_dev": float(np.mean([per_mol[k]["top4_mean_dev"][name] for k in keys])),
            "mean_top4_center_degree": float(np.mean([per_mol[k]["top4_mean_center_degree"][name] for k in keys])),
            "mean_top4_ring_fraction": float(np.mean([per_mol[k]["top4_ring_fraction"][name] for k in keys])),
        }
    b_agg["top4_composition"] = comp
    # overlap between IG top4 and baselines
    jac = []
    for k in keys:
        a = set(per_mol[k]["top4_sets"]["IG"])
        for name in ("dev", "rare"):
            bset = set(per_mol[k]["top4_sets"][name])
            jac.append(len(a & bset) / max(len(a | bset), 1))
    b_agg["mean_top4_jaccard"] = float(np.mean(jac))

    g_dev = b_agg["contrasts"]["IG_minus_dev"]
    g_rare = b_agg["contrasts"]["IG_minus_rare"]
    g_match = b_agg["matched_control"]
    thresholds = cfg["phase0"]["thresholds"]
    p0_beyond_dev = bool(g_dev["mean"] > 0 and g_dev["ci95_lower"] > 0 and g_dev["p_one_sided"] < thresholds["beyond"]["one_sided_sign_p_lt"])
    p0_beyond_rare = bool(g_rare["mean"] > 0 and g_rare["ci95_lower"] > 0 and g_rare["p_one_sided"] < thresholds["beyond"]["one_sided_sign_p_lt"])
    # AMENDMENT_01: coverage is a reported diagnostic, not a gate; the paired CI
    # already prices in the number of matched junctions.  See AMENDMENT_01.md.
    p0_matched = bool(g_match["mean"] > 0 and g_match["ci95_lower"] > 0)

    # ---- 0C signature statistics ----
    c_by_class: dict[tuple, list[dict[str, Any]]] = {}
    for r in c_records:
        c_by_class.setdefault(tuple(r["signature"]), []).append(r)
    classes = []
    for sig, rows in sorted(c_by_class.items(), key=lambda kv: -len(kv[1])):
        vals_abs = np.asarray([r["ig_abs"] for r in rows])
        vals_c = np.asarray([r["ig_centered"] for r in rows])
        classes.append(
            {
                "signature": list(sig),
                "corpus_count": int(counts.get(sig, 0)),
                "sample_count": len(rows),
                "corpus_freq": float(counts.get(sig, 0) / max(corpus_total, 1)),
                "mean_ig_abs": float(vals_abs.mean()),
                "mean_ig_centered": float(vals_c.mean()),
                "sign_agreement": float(abs(np.mean(np.sign(vals_c)))) if len(rows) else None,
                "note": "insufficient_frequency" if counts.get(sig, 0) < 30 else "ok",
            }
        )
    # top-decile enrichment with within-molecule permutation null
    top_flag = np.zeros(len(c_records), dtype=bool)
    by_mol: dict[tuple[int, int], list[int]] = {}
    for idx, r in enumerate(c_records):
        by_mol.setdefault((r["seed"], r["row"]), []).append(idx)
    for idxs in by_mol.values():
        vals = np.asarray([c_records[i]["ig_abs"] for i in idxs])
        k = max(1, int(round(0.1 * len(idxs))))
        top = np.argsort(-vals, kind="stable")[:k]
        for t in top:
            top_flag[idxs[int(t)]] = True
    frequent = [s for s in classes if s["corpus_count"] >= 100]
    sig_to_id = {tuple(s["signature"]): i for i, s in enumerate(classes)}
    rec_sig_ids = np.asarray([sig_to_id[tuple(r["signature"])] for r in c_records], dtype=np.int64)
    class_freq = np.asarray([s["corpus_freq"] for s in classes], dtype=np.float64)
    class_sample = np.asarray([s["sample_count"] for s in classes], dtype=np.int64)
    frequent_mask = np.asarray([s["corpus_count"] >= 100 for s in classes], dtype=bool)
    n_classes = len(classes)

    def enrichment_stat(flags: np.ndarray) -> float:
        n_top = int(flags.sum())
        if n_top == 0:
            return 0.0
        obs = np.bincount(rec_sig_ids[flags], minlength=n_classes).astype(np.float64)
        exp = n_top * class_freq
        mask = frequent_mask & (exp >= 3.0)
        if not mask.any():
            return 0.0
        return float(np.max(obs[mask] / np.maximum(exp[mask], 1e-9)))

    def sign_stat(values: np.ndarray) -> float:
        s = np.sign(values - np.median(values))
        num = np.bincount(rec_sig_ids, weights=s, minlength=n_classes)
        cnt = np.bincount(rec_sig_ids, minlength=n_classes)
        mask = class_sample >= 30
        if not mask.any():
            return 0.0
        agree = np.abs(num[mask] / np.maximum(cnt[mask], 1))
        return float(np.sum(cnt[mask] * agree) / np.sum(cnt[mask]))

    rng = np.random.default_rng(20261015)
    centered = np.asarray([r["ig_centered"] for r in c_records])
    ig_abs_all = np.asarray([r["ig_abs"] for r in c_records], dtype=np.float64)
    group_idx = [np.asarray(v, dtype=np.int64) for v in by_mol.values()]
    group_k = [max(1, int(round(0.1 * len(idx)))) for idx in group_idx]
    obs_enrich = enrichment_stat(top_flag)
    obs_sign = sign_stat(centered)
    null_enrich, null_sign = [], []
    for _ in range(PHASE0_PERMUTATION_DRAWS):
        perm_flags = top_flag.copy()
        perm_vals = centered.copy()
        for idxs, k in zip(group_idx, group_k):
            perm = rng.permutation(len(idxs))
            new_top = np.argsort(-ig_abs_all[idxs][perm], kind="stable")[:k]
            flags = np.zeros(len(idxs), dtype=bool)
            flags[new_top] = True
            perm_flags[idxs] = flags
            perm_vals[idxs] = centered[idxs][perm]
        null_enrich.append(enrichment_stat(perm_flags))
        null_sign.append(sign_stat(perm_vals))
    p_enrich = (1 + int(np.sum(np.asarray(null_enrich) >= obs_enrich))) / (1 + PHASE0_PERMUTATION_DRAWS)
    p_sign = (1 + int(np.sum(np.asarray(null_sign) >= obs_sign))) / (1 + PHASE0_PERMUTATION_DRAWS)
    c_agg = {
        "n_classes": len(classes),
        "n_classes_frequent": len(frequent),
        "n_classes_insufficient": int(sum(1 for s in classes if s["note"] == "insufficient_frequency")),
        "top_decile_count": int(top_flag.sum()),
        "enrichment_stat": float(obs_enrich),
        "enrichment_p": float(p_enrich),
        "enrichment_null_p95": float(np.quantile(null_enrich, 0.95)),
        "sign_stat": float(obs_sign),
        "sign_p": float(p_sign),
        "sign_null_p95": float(np.quantile(null_sign, 0.95)),
        "classes": classes,
    }
    p0c_enrichment = bool(p_enrich < thresholds["enrichment"]["p_lt"])
    p0c_sign = bool(p_sign < thresholds["sign_consistency"]["p_lt"])

    thresholds = cfg["phase0"]["thresholds"]
    if p0_beyond_dev and p0_beyond_rare and p0_matched and p0c_enrichment and p0c_sign:
        verdict = "SOURCE_SIGNAL_BEYOND_SIMPLE_BASELINES"
    elif (not p0_matched) and (
        b_agg["contrasts"]["dev_mean_response"] >= b_agg["contrasts"]["IG_mean_response"]
        or b_agg["contrasts"]["rare_mean_response"] >= b_agg["contrasts"]["IG_mean_response"]
    ):
        verdict = "SOURCE_SIGNAL_EXPLAINABLE_BY_SIMPLE_BASELINES"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "protocol": PROTOCOL_ID,
        "phase0_a": {**fact1, "fact1_statement": "IG_j is the exact single-junction baseline replacement difference", "fact2_statement": "joint reset keeps the first moment and collapses the second moment to mu_G^2"},
        "phase0_b": b_agg,
        "phase0_c": c_agg,
        "gates": {
            "P0_BEYOND_DEV": p0_beyond_dev,
            "P0_BEYOND_RARE": p0_beyond_rare,
            "P0_MATCHED": p0_matched,
            "P0C_ENRICHMENT": p0c_enrichment,
            "P0C_SIGN": p0c_sign,
        },
        "verdict": verdict,
        "records_b": b_records,
        "records_c": c_records,
        "labels_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def cmd_phase0(args: argparse.Namespace) -> int:
    payload = phase0()
    write_json(OUT / "phase0_results.json", payload)
    write_json(OUT / "phase0_decision.json", {k: payload[k] for k in ("protocol", "phase0_a", "phase0_b", "phase0_c", "gates", "verdict")})
    log(f"[phase0] verdict {payload['verdict']} gates {payload['gates']}")
    return 0


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _real_fit_objects(n: int = 128) -> list[sro.MolObjects]:
    objs = torch.load(objects_path("fit"), map_location="cpu", weights_only=False)
    return list(objs[:n])


def _real_fit_keys(n: int = 128) -> np.ndarray:
    return own.molecule_keys(_real_fit_objects(n))


def _permute_batch_bonds(batch: Mapping[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    out = own.clone_batch(batch)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    for key in ("bond_feat", "bond_batch", "bond_index"):
        out[key] = batch[key][perm]
    out["junc_bond"] = inv[batch["junc_bond"]]
    return out


def _permute_batch_junctions(batch: Mapping[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    out = own.clone_batch(batch)
    for key in ("junc_feat", "junc_batch", "junc_index", "junc_bond"):
        out[key] = batch[key][perm]
    return out


def _blockwise_perm(num: int, groups: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    perm = np.arange(num)
    for g in torch.unique(groups).tolist():
        idx = torch.nonzero(groups == g).reshape(-1).numpy()
        perm[idx] = idx[rng.permutation(len(idx))]
    return torch.as_tensor(perm, dtype=torch.long)


def cpu_gates(device: torch.device) -> dict[str, Any]:
    verify_freeze()
    norm = read_json(DATA / "prep_fit_D.json")
    objs = _real_fit_objects(128)
    keys = own.molecule_keys(objs)
    batch = own.batch_objects_with_keys(objs, keys)
    results: dict[str, Any] = {}
    models: dict[str, tuple[src.BJOC, own.BJOCSourceOwnership, dict[str, Any]]] = {}
    for mode in own.MODES:
        models[mode] = own.build_ownership_from_init(0, norm, device, mode, None)
    hashes = {mode: own.model_state_hash(model) for mode, (_, model, _) in models.items()}
    params = {mode: int(sum(p.numel() for p in model.parameters())) for mode, (_, model, _) in models.items()}
    results["arm_parity"] = {
        "init_hash_equal": len(set(hashes.values())) == 1,
        "parameter_counts": params,
        "expected": own.OWNERSHIP_EXPECTED_PARAMETERS,
        "pass": len(set(hashes.values())) == 1 and set(params.values()) == {own.OWNERSHIP_EXPECTED_PARAMETERS},
    }
    with torch.no_grad():
        p_base = models["N"][0](batch)
        diffs = {mode: float((model(batch) - p_base).abs().max()) for mode, (_, model, _) in models.items()}
    results["init_parity_vs_p3"] = {"max_abs_diff": diffs, "pass": diffs["N"] < 1e-6}
    rng = np.random.default_rng(0)
    bond_perm = _blockwise_perm(batch["bond_feat"].shape[0], batch["bond_batch"], rng)
    junc_perm = _blockwise_perm(batch["junc_feat"].shape[0], batch["junc_batch"], rng)
    perm_report = {}
    for mode in own.MODES:
        model = models[mode][1]
        with torch.no_grad():
            p0 = model(batch)
            pb = model(_permute_batch_bonds(batch, bond_perm))
            pj = model(_permute_batch_junctions(batch, junc_perm))
        perm_report[mode] = {"bond_order": float((p0 - pb).abs().max()), "junction_order": float((p0 - pj).abs().max())}
    results["permutation_invariance"] = {"max_abs_diff": perm_report, "pass": all(v < 1e-5 for d in perm_report.values() for v in d.values())}
    # ownership isolation (grouping level, real molecule)
    obj = objs[0]
    single = own.batch_objects_with_keys([obj], own.molecule_keys([obj]))
    if single["junc_feat"].shape[0] >= 1 and single["bond_feat"].shape[0] >= 2:
        edited = own.clone_batch(single)
        j0 = 0
        old = single["junc_bond"][j0].tolist()
        candidates = [b for b in range(single["bond_feat"].shape[0]) if b not in old]
        new = torch.as_tensor(candidates[:2] if len(candidates) >= 2 else [candidates[0], candidates[0]], dtype=torch.long)
        edited["junc_bond"][j0] = new
        touched = set(old) | set(new.tolist())
        with torch.no_grad():
            c0 = models["O"][1].forward_features(single)["c"]
            c1 = models["O"][1].forward_features(edited)["c"]
        isolated = all(torch.equal(c0[e], c1[e]) for e in range(c0.shape[0]) if e not in touched)
        changed = any(not torch.equal(c0[e], c1[e]) for e in touched)
        results["ownership_isolation"] = {"isolated": bool(isolated), "incident_changed": bool(changed), "pass": bool(isolated)}
    else:
        results["ownership_isolation"] = {"pass": True, "note": "molecule too small; covered by unit tests"}
    # global block separation
    with torch.no_grad():
        f0 = models["O"][1].forward_features(batch)
        f1 = models["O"][1].forward_features(batch, p_override=torch.randn_like(f0["p_global"]))
    glob = {
        "c_unchanged": bool(torch.equal(f0["c"], f1["c"])),
        "reader_local_unchanged": bool(((f0["reader_in"] - f1["reader_in"]).abs()[:, :321] == 0).all()),
        "global_block_changed": bool(((f0["reader_in"] - f1["reader_in"]).abs()[:, 321:] > 0).any()),
    }
    glob["pass"] = glob["c_unchanged"] and glob["reader_local_unchanged"] and glob["global_block_changed"]
    results["global_pathway_separation"] = glob
    # N/O delegation to frozen groupings
    with torch.no_grad():
        a = models["N"][1].encode_atoms(batch)
        b = models["N"][1].encode_bonds(a, batch)
        w = models["N"][1].encode_junctions(a, b, batch)
        Wn, Vn, nn = models["N"][1].group_to_bonds_ownership(w, batch, own.MODE_N)
        Wr, Vr, nr = src.BJOC.group_to_bonds(models["N"][1], w, batch, src.NULL)
        Wo, Vo, no = models["O"][1].group_to_bonds_ownership(w, batch, own.MODE_O)
        Wro, Vro, nro = src.BJOC.group_to_bonds(models["O"][1], w, batch, src.ORGANIZED)
    results["grouping_delegation"] = {
        "N_matches_null": bool(torch.equal(Wn, Wr) and torch.equal(Vn, Vr) and torch.equal(nn, nr)),
        "O_matches_organized": bool(torch.equal(Wo, Wro) and torch.equal(Vo, Vro) and torch.equal(no, nro)),
    }
    results["grouping_delegation"]["pass"] = results["grouping_delegation"]["N_matches_null"] and results["grouping_delegation"]["O_matches_organized"]
    # sham invariants + key collision audit
    with torch.no_grad():
        Ws, Vs, ns = models["S"][1].group_to_bonds_ownership(w, batch, own.MODE_S)
        jkey = models["S"][1]._junction_key(batch["junc_feat"], w)
        bkey = models["S"][1]._bond_key(
            batch["bond_feat"],
            batch["atom_feat"][batch["bond_index"][:, 0]],
            batch["atom_feat"][batch["bond_index"][:, 1]],
            ns,
        )
    desc_j = torch.cat([w, batch["junc_feat"]], dim=1)
    desc_b = models["S"][1].bond_descriptor(batch, ns)
    jcol = bcol = 0
    for g in torch.unique(batch["junc_batch"]).tolist():
        jm = batch["junc_batch"] == g
        kk = jkey[jm]
        order = torch.argsort(kk)
        for u, v in zip(order[:-1].tolist(), order[1:].tolist()):
            if torch.equal(kk[u], kk[v]) and not torch.equal(desc_j[jm][u], desc_j[jm][v]):
                jcol += 1
    for u in range(bkey.numel()):
        for v in range(u + 1, bkey.numel()):
            if torch.equal(bkey[u], bkey[v]) and not torch.equal(desc_b[u], desc_b[v]):
                bcol += 1
    sham = {
        "counts_preserved": bool(torch.equal(ns, no)),
        "total_slots_is_2nj": bool(float(ns.sum()) == 2.0 * float(batch["junc_feat"].shape[0])),
        "junction_key_collisions": int(jcol),
        "bond_key_collisions": int(bcol),
        "format_matches_O": bool(Ws.shape == Wo.shape and Vs.shape == Vo.shape),
    }
    sham["pass"] = sham["counts_preserved"] and sham["total_slots_is_2nj"] and jcol == 0 and bcol == 0 and sham["format_matches_O"]
    results["sham_invariants"] = sham
    # forbidden modules
    hits = {mode: own.forbidden_module_scan(model) for mode, (_, model, _) in models.items()}
    results["forbidden_modules"] = {"hits": hits, "pass": all(len(h) == 0 for h in hits.values())}
    results["decision"] = "CPU_GATES_PASS" if all(bool(results[k].get("pass")) for k in results if isinstance(results[k], dict) and "pass" in results[k]) else "CPU_GATES_FAIL"
    return results


def cmd_cpu_gates(args: argparse.Namespace) -> int:
    device = device_from_arg(args.device)
    payload = cpu_gates(device)
    write_json(OUT / "cpu_gates.json", payload)
    log(f"[cpu-gates] {payload['decision']}")
    return 0 if payload["decision"] == "CPU_GATES_PASS" else 1


# ---------------------------------------------------------------------------
# P3 replay
# ---------------------------------------------------------------------------


def cmd_p3_replay(args: argparse.Namespace) -> int:
    verify_freeze()
    device = device_from_arg(args.device)
    norm = read_json(DATA / "prep_fit_D.json")
    monitor = torch.load(objects_path("monitor"), map_location="cpu", weights_only=False)
    keys = own.molecule_keys(monitor)
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "per_seed": {}}
    for seed in SEEDS:
        model = bp.build_bjoc_null(seed, norm, device)
        model.load_state_dict(torch.load(p3_soup_path(seed), map_location=device, weights_only=False))
        pred = bp.predict_model(model, monitor, device)
        frozen = np.load(p3_pred_path(seed))["pred"]
        diff = float(np.abs(pred - frozen).max())
        payload["per_seed"][str(seed)] = {"max_abs_diff": diff, "replay_pass": diff <= 1e-5}
        log(f"[p3-replay] seed {seed} max|diff| {diff:.3e}")
    payload["decision"] = "P3_REPLAY_PASS" if all(v["replay_pass"] for v in payload["per_seed"].values()) else "P3_REPLAY_FAIL"
    write_json(OUT / "p3_replay.json", payload)
    return 0 if payload["decision"] == "P3_REPLAY_PASS" else 1


# ---------------------------------------------------------------------------
# supervised arms + prediction
# ---------------------------------------------------------------------------


def _load_split() -> dict[str, Any]:
    return read_json(split_path())


def cmd_arm(args: argparse.Namespace) -> int:
    verify_freeze()
    arm = str(args.arm).upper()
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    seed = int(args.seed)
    device = device_from_arg(args.device)
    split = _load_split()
    norm = read_json(DATA / "prep_fit_D.json")
    fit_objs = torch.load(objects_path("fit"), map_location="cpu", weights_only=False)
    if [int(o.mol_index) for o in fit_objs] != [int(i) for i in split["fit_rows"]]:
        raise SystemExit("fit object rows do not match the frozen split")
    rec = read_json(ssl_record_path(seed))
    ssl_state = torch.load(ssl_path(seed), map_location="cpu", weights_only=False)
    y_fit = bp.load_y_rows(ENCODED_TRAIN, split["fit_rows"])
    smoke = int(args.smoke_epochs) if args.smoke_epochs else None
    soup = soup_path(arm, seed)
    tj = train_json_path(arm, seed)
    if smoke:
        soup = OUT / "smoke" / f"soup_{arm}_D_s{seed}_smoke.pt"
        tj = OUT / "smoke" / f"train_{arm}_D_s{seed}_smoke.json"
    if soup.exists() or tj.exists():
        raise SystemExit(f"{arm} seed {seed} already exists; retraining is not allowed")
    payload = own.train_supervised_ownership(
        arm=arm,
        seed=seed,
        recipe_key="R3",
        train_objs=fit_objs,
        rows=split["fit_rows"],
        y_fit=y_fit,
        norm=norm,
        device=device,
        ssl_state=ssl_state,
        smoke_epochs=smoke,
    )
    if payload["encoder_used_sha256"] != rec["encoder_final_sha256"]:
        raise SystemExit("encoder used does not match the recorded SSL state")
    soup.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}, soup)
    payload.pop("soup_state", None)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = file_sha256(soup)
    payload["split_tag"] = SPLIT_TAG
    payload["ssl_record_sha256"] = file_sha256(ssl_record_path(seed))
    write_json(tj, payload)
    log(f"[{arm} s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s")
    return 0


def _load_arm_model(arm: str, seed: int, device: torch.device) -> own.BJOCSourceOwnership:
    norm = read_json(DATA / "prep_fit_D.json")
    _, model, _ = own.build_ownership_from_init(seed, norm, device, own.ARM_MODES[arm], None)
    model.load_state_dict(torch.load(soup_path(arm, seed), map_location=device, weights_only=False))
    return model


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze()
    device = device_from_arg(args.device)
    arms = [str(a).upper() for a in (args.arms or list(ARMS))]
    seeds = [int(s) for s in (args.seeds or [0])]
    zones: dict[str, Any] = {}
    monitor = torch.load(objects_path("monitor"), map_location="cpu", weights_only=False)
    zones["monitor"] = (monitor, own.molecule_keys(monitor))
    manifest = read_json(PRED_MANIFEST) if PRED_MANIFEST.exists() else {
        "protocol": PROTOCOL_ID,
        "entries": {},
        "monitor_labels_read": MONITOR_LOCK.exists(),
        "locked_labels_read": False,
        "official_test_loaded": False,
    }
    if args.valid:
        valid = torch.load(F10K / "objects_valid.pt", map_location="cpu", weights_only=False)
        zones["valid"] = (valid, own.molecule_keys(valid))
    for arm in arms:
        for seed in seeds:
            model = _load_arm_model(arm, seed, device)
            for zone, (objs, keys) in zones.items():
                pred = own.predict_model_ownership(model, objs, keys, device)
                if not np.isfinite(pred).all():
                    raise SystemExit(f"non-finite prediction {arm} {seed} {zone}")
                out = (monitor_pred_path if zone == "monitor" else valid_pred_path)(arm, seed)
                out.parent.mkdir(parents=True, exist_ok=True)
                key = f"{zone}:{arm}:{seed}"
                np.savez_compressed(out, row=np.arange(len(objs), dtype=np.int64), pred=pred.astype(np.float32))
                manifest["entries"][key] = {
                    "path": str(out.relative_to(REPO_ROOT)),
                    "sha256": file_sha256(out),
                    "soup_sha256": file_sha256(soup_path(arm, seed)),
                    "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                log(f"[predict] {zone} {arm} s{seed} -> {out.name}")
    write_json(PRED_MANIFEST, manifest)
    return 0


def _frozen_r3_valid_predictions(seed: int, device: torch.device) -> np.ndarray:
    norm = v1.load_norm()
    model = v1.load_model(seed, norm, "cpu")
    objs = torch.load(F10K / "objects_valid.pt", map_location="cpu", weights_only=False)
    return bp.predict_model(model, objs, "cpu")


def cmd_valid_reference(args: argparse.Namespace) -> int:
    verify_freeze()
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "per_seed": {}}
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    objs = torch.load(F10K / "objects_valid.pt", map_location="cpu", weights_only=False)
    keys = own.molecule_keys(objs)
    norm = read_json(DATA / "prep_fit_D.json")
    for seed in SEEDS:
        pred_r3 = _frozen_r3_valid_predictions(seed, torch.device("cpu"))
        np.savez_compressed(valid_pred_path_r3(seed), row=np.arange(len(objs), dtype=np.int64), pred=pred_r3.astype(np.float32))
        model = bp.build_bjoc_null(seed, norm, "cpu")
        model.load_state_dict(torch.load(p3_soup_path(seed), map_location="cpu", weights_only=False))
        pred_p3 = bp.predict_model(model, objs, "cpu")
        np.savez_compressed(valid_pred_path_p3(seed), row=np.arange(len(objs), dtype=np.int64), pred=pred_p3.astype(np.float32))
        payload["per_seed"][str(seed)] = {
            "r3_path": str(valid_pred_path_r3(seed).relative_to(REPO_ROOT)),
            "r3_sha256": file_sha256(valid_pred_path_r3(seed)),
            "p3_path": str(valid_pred_path_p3(seed).relative_to(REPO_ROOT)),
            "p3_sha256": file_sha256(valid_pred_path_p3(seed)),
        }
        log(f"[valid-reference] seed {seed} written")
    guard = {"reference": VALID_REFERENCE_R3, "tolerance": 1e-5, "pass": True}
    y_valid = np.load(F10K / "valid_y_read_once.npy")
    for seed in SEEDS:
        mae = float(np.abs(_load_pred(valid_pred_path_r3(seed)) - y_valid).mean())
        ref = float(VALID_REFERENCE_R3[f"seed{seed}"])
        ok = abs(mae - ref) <= 1e-5
        guard[f"seed{seed}_mae"] = mae
        guard[f"seed{seed}_reference"] = ref
        guard[f"seed{seed}_pass"] = ok
        guard["pass"] = guard["pass"] and ok
    payload["replay_guard"] = guard
    write_json(OUT / "valid_reference.json", payload)
    log(f"[valid-reference] replay guard {guard['pass']}")
    return 0 if guard["pass"] else 1


# ---------------------------------------------------------------------------
# Monitor scoring
# ---------------------------------------------------------------------------


def _ensure_monitor_y() -> np.ndarray:
    if not MONITOR_LOCK.exists():
        y = np.load(monitor_y_path())
        if y.shape[0] != len(_load_split()["monitor_rows"]):
            raise SystemExit("monitor label vector length mismatch")
        np.save(MONITOR_Y_COPY, y)
        MONITOR_LOCK.write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL_ID,
                    "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": str(monitor_y_path().relative_to(REPO_ROOT)),
                    "sha256": file_sha256(MONITOR_Y_COPY),
                    "rounds": ["seed0"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"[score-monitor] label read #1 y_sha256 {file_sha256(MONITOR_Y_COPY)}")
    else:
        lock = read_json(MONITOR_LOCK)
        if file_sha256(MONITOR_Y_COPY) != lock["sha256"]:
            raise SystemExit("cached monitor label file changed")
        log("[score-monitor] cached labels reused (no new label read)")
    return np.load(MONITOR_Y_COPY)


def _load_pred(path: Path) -> np.ndarray:
    return np.load(path)["pred"].astype(np.float64)


def _mae(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(pred - y)


def _contrast(name: str, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    err_a, err_b = _mae(a, y), _mae(b, y)
    d = err_a - err_b
    out = {"name": name, "mae_a": float(err_a.mean()), "mae_b": float(err_b.mean())}
    out.update(_bootstrap(d))
    out.update(bsaf.sign_test(d))
    out["frac_better"] = float(np.mean(d < 0))
    ordinary = y > TAIL_THRESHOLD
    tail = ~ordinary
    worst = np.argsort(-np.maximum(err_a, err_b), kind="stable")[:TOP_K_REMOVED]
    top_removed = np.zeros(len(y), dtype=bool)
    top_removed[worst] = True
    out["ordinary_delta"] = float(d[ordinary].mean())
    out["tail_delta"] = float(d[tail].mean()) if tail.any() else None
    out["top10_removed_delta"] = float(d[~top_removed].mean())
    out["rmse_a"] = float(np.sqrt((err_a**2).mean()))
    out["rmse_b"] = float(np.sqrt((err_b**2).mean()))
    out.update(_sign_test_one_sided(d))
    return out


def cmd_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze()
    round_name = str(args.round)
    seeds = [0] if round_name == "seed0" else [0, 1]
    y = _ensure_monitor_y()
    if round_name == "seed1" and MONITOR_LOCK.exists():
        lock = read_json(MONITOR_LOCK)
        rounds = list(lock.get("rounds", []))
        if "seed1" not in rounds:
            rounds.append("seed1")
            lock["rounds"] = rounds
            MONITOR_LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    preds: dict[str, dict[int, np.ndarray]] = {}
    for arm in ARMS:
        preds[arm] = {}
        for seed in seeds:
            p = monitor_pred_path(arm, seed)
            if not p.exists():
                raise SystemExit(f"missing frozen prediction {p}")
            preds[arm][seed] = _load_pred(p)
    p3 = {seed: _load_pred(p3_pred_path(seed)) for seed in SEEDS}
    metrics: dict[str, Any] = {"protocol": PROTOCOL_ID, "round": round_name, "n_monitor": int(len(y)), "arms": {}}
    for arm in ARMS:
        entry: dict[str, Any] = {"per_seed": {}}
        for seed in seeds:
            entry["per_seed"][str(seed)] = {"mae": float(_mae(preds[arm][seed], y).mean())}
        if len(seeds) == 2:
            mean_pred = np.mean([preds[arm][s] for s in seeds], axis=0)
            entry["seed_mean_mae"] = float(_mae(mean_pred, y).mean())
        metrics["arms"][arm] = entry
    offsets = {}
    for seed in seeds:
        contrasts = {
            "O_minus_N": (preds["O"][seed], preds["N"][seed]),
            "O_minus_S": (preds["O"][seed], preds["S"][seed]),
            "N_minus_S": (preds["N"][seed], preds["S"][seed]),
            "N_minus_P3": (preds["N"][seed], p3[seed]),
            "O_minus_P3": (preds["O"][seed], p3[seed]),
            "S_minus_P3": (preds["S"][seed], p3[seed]),
        }
        offsets[str(seed)] = {name: _contrast(name, a, b, y) for name, (a, b) in contrasts.items()}
    metrics["contrasts"] = offsets
    if len(seeds) == 2:
        seed_mean = {}
        for name, key_a, key_b in (
            ("O_minus_N", "O", "N"),
            ("O_minus_S", "O", "S"),
            ("N_minus_S", "N", "S"),
        ):
            a = np.mean([preds[key_a][s] for s in seeds], axis=0)
            b = np.mean([preds[key_b][s] for s in seeds], axis=0)
            seed_mean[name] = _contrast(name, a, b, y)
        metrics["seed_mean_contrasts"] = seed_mean
    split = _load_split()
    y_source = str(monitor_y_path().relative_to(REPO_ROOT))
    metrics["labels"] = {
        "read_once": True,
        "source": y_source,
        "sha256": file_sha256(MONITOR_Y_COPY),
        "historical_exposure": "split-D Monitor read by prior rounds; exploratory screen, not a blind test",
        "locked_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    # frozen gates
    c0 = offsets["0"]
    gates = {
        "ownership_contrast": bool(c0["O_minus_N"]["mean"] <= -0.002 and c0["O_minus_N"]["ci95_upper"] < 0),
        "sham_separation": bool(c0["O_minus_S"]["mean"] <= -0.002 and c0["O_minus_S"]["ci95_upper"] < 0),
        "not_tail_or_single_driven": bool(c0["O_minus_N"]["ordinary_delta"] < 0 and c0["O_minus_N"]["top10_removed_delta"] < 0),
        "parity_vs_p3": bool(c0["O_minus_P3"]["mae_a"] - c0["O_minus_P3"]["mae_b"] <= 0.004),
    }
    metrics["gates_seed0"] = gates
    if round_name == "seed0":
        metrics["decision"] = "SEED0_GATES_PASS" if all(gates.values()) else "SEED0_GATES_FAIL"
    else:
        sm = metrics.get("seed_mean_contrasts", {})
        gates1 = {
            "seed0_gate_record": all(gates.values()),
            "seed_means_negative": bool(
                sm.get("O_minus_N", {}).get("mean", 1.0) <= -0.002
                and sm.get("O_minus_N", {}).get("ci95_upper", 1.0) < 0
                and sm.get("O_minus_S", {}).get("mean", 1.0) < 0
            ),
            "same_sign_both_seeds": bool(
                offsets["0"]["O_minus_N"]["mean"] < 0
                and offsets["1"]["O_minus_N"]["mean"] < 0
                and offsets["0"]["O_minus_S"]["mean"] < 0
                and offsets["1"]["O_minus_S"]["mean"] < 0
            ),
        }
        metrics["gates_seed1"] = gates1
        metrics["decision"] = "SEED1_CONFIRMED" if all(gates1.values()) else "SEED1_NOT_CONFIRMED"
    write_json(OUT / f"monitor_metrics_{round_name}.json", metrics)
    log(f"[score-monitor {round_name}] decision {metrics['decision']}")
    return 0


# ---------------------------------------------------------------------------
# mechanism audit
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_mechanism(args: argparse.Namespace) -> int:
    verify_freeze()
    device = device_from_arg(args.device)
    seeds = [int(s) for s in (args.seeds or [0])]
    monitor = torch.load(objects_path("monitor"), map_location="cpu", weights_only=False)
    n = int(args.molecules)
    objs = list(monitor[:n])
    keys = own.molecule_keys(objs)
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "n_molecules": n, "per_arm_seed": {}}
    attributions: dict[tuple[str, int], np.ndarray] = {}
    for arm in ARMS:
        for seed in seeds:
            model = _load_arm_model(arm, seed, device)
            model.eval()
            faith, local_delta, global_delta, c_norm = [], [], [], []
            bond_grads = []
            for obj in objs:
                batch = own.batch_objects_with_keys([obj], own.molecule_keys([obj]))
                with torch.no_grad():
                    f0 = model.forward_features(batch)
                    p0 = model(batch)
                    p_local = model(batch, zero_local=True)
                    rnd = torch.randn_like(f0["p_global"])
                    p_global = model(batch, p_override=rnd)
                c = f0["c"]
                attr = c.norm(dim=1).numpy()
                deltas = np.zeros(c.shape[0])
                with torch.no_grad():
                    for e in range(c.shape[0]):
                        deltas[e] = float(model(batch, zero_bonds=torch.as_tensor([e])).item() - p0.item())
                faith.append(_spearman(attr, np.abs(deltas)))
                local_delta.append(float(abs(p_local.item() - p0.item())))
                global_delta.append(float(abs(p_global.item() - p0.item())))
                c_norm.append(attr)
                attributions[(arm, seed)] = np.concatenate([attributions.get((arm, seed), np.zeros(0)), attr])
            faith = np.asarray(faith)
            payload["per_arm_seed"][f"{arm}:{seed}"] = {
                "local_pathway_delta_mean_abs": float(np.mean(local_delta)),
                "global_block_delta_mean_abs": float(np.mean(global_delta)),
                "faithfulness_spearman_mean": float(np.nanmean(faith)),
                "faithfulness_spearman_median": float(np.nanmedian(faith)),
                "faithfulness_above_0.5_fraction": float(np.mean(np.asarray(faith) >= 0.5)),
                "mean_bond_attribution_norm": float(np.mean([a.mean() for a in c_norm])),
            }
    if len(seeds) == 2:
        for arm in ARMS:
            if (arm, 0) in attributions and (arm, 1) in attributions:
                payload.setdefault("cross_seed_attribution_spearman", {})[arm] = _spearman(attributions[(arm, 0)], attributions[(arm, 1)])
    metrics = {
        "local_pathway_deletion_faithfulness": float(np.nanmean([payload["per_arm_seed"][f"O:{s}"]["faithfulness_spearman_mean"] for s in seeds])),
        "min_local_pathway_delta": float(min(payload["per_arm_seed"][f"{a}:{s}"]["local_pathway_delta_mean_abs"] for a in ("O",) for s in seeds)),
    }
    payload["gates"] = {
        "faithfulness": bool(metrics["local_pathway_deletion_faithfulness"] >= 0.5),
        "local_pathway_used": bool(metrics["min_local_pathway_delta"] > 1e-6),
    }
    payload["metrics"] = metrics
    write_json(OUT / f"mechanism_s{'_'.join(str(s) for s in seeds)}.json", payload)
    log(f"[mechanism] gates {payload['gates']}")
    return 0


# ---------------------------------------------------------------------------
# secondary valid comparison (exposed development set)
# ---------------------------------------------------------------------------


def cmd_valid_secondary(args: argparse.Namespace) -> int:
    verify_freeze()
    if not (F10K / "valid_y_read_once.npy").exists():
        raise SystemExit("official-valid label cache missing")
    y = np.load(F10K / "valid_y_read_once.npy")
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "n_valid": int(len(y)), "arms": {}, "note": "official valid is an exposed development set; descriptive only"}
    sources: dict[str, dict[int, np.ndarray]] = {"R3": {}, "P3D": {}}
    for seed in SEEDS:
        sources["R3"][seed] = _load_pred(valid_pred_path_r3(seed))
        sources["P3D"][seed] = _load_pred(valid_pred_path_p3(seed))
    for arm in ARMS:
        for seed in SEEDS:
            sources[arm] = sources.get(arm, {})
            sources[arm][seed] = _load_pred(valid_pred_path(arm, seed))
    for name, per_seed in sources.items():
        if not per_seed:
            continue
        maes = {str(s): float(_mae(p, y).mean()) for s, p in per_seed.items()}
        entry = {"per_seed_mae": maes}
        if len(per_seed) == 2:
            mean_pred = np.mean(list(per_seed.values()), axis=0)
            entry["seed_mean_mae"] = float(_mae(mean_pred, y).mean())
        payload["arms"][name] = entry
    if "O" in sources and "R3" in sources:
        d = _mae(sources["O"][0], y) - _mae(sources["R3"][0], y)
        payload["O_minus_R3_seed0"] = {**_bootstrap(d), **bsaf.sign_test(d)}
    payload["reference_frozen_r3_official_valid"] = VALID_REFERENCE_R3
    guard = read_json(OUT / "valid_reference.json").get("replay_guard") if (OUT / "valid_reference.json").exists() else None
    if guard is not None and not guard.get("pass"):
        raise SystemExit("R3 valid replay guard failed; re-run valid-reference before reading this comparison")
    payload["replay_guard_pass"] = bool(guard and guard.get("pass"))
    write_json(OUT / "valid_secondary.json", payload)
    log("[valid-secondary] written")
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> int:
    verify_freeze()
    phase0 = read_json(OUT / "phase0_decision.json") if (OUT / "phase0_decision.json").exists() else {}
    cpu = read_json(OUT / "cpu_gates.json") if (OUT / "cpu_gates.json").exists() else {}
    monitor_files = sorted(OUT.glob("monitor_metrics_*.json"))
    monitor = [read_json(p) for p in monitor_files]
    mechanism_files = sorted(OUT.glob("mechanism_s*.json"))
    mechanism = [read_json(p) for p in mechanism_files]
    decision = {
        "protocol": PROTOCOL_ID,
        "phase0_verdict": phase0.get("verdict"),
        "cpu_gates": cpu.get("decision"),
        "monitor_rounds": [{"round": m.get("round"), "decision": m.get("decision"), "gates": m.get("gates_seed0")} for m in monitor],
        "mechanism_gates": [m.get("gates") for m in mechanism],
        "locked_read": False,
        "official_valid_loaded_for_secondary_only": (OUT / "valid_secondary.json").exists(),
        "official_test_loaded": False,
    }
    write_json(OUT / "final_decision.json", decision)
    log(f"[finalize] {json.dumps(decision, default=str)[:400]}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_bjoc_source_ownership_v1")
    p.add_argument("--device", default="auto")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-inputs")
    sub.add_parser("freeze")
    sub.add_parser("phase0")
    sub.add_parser("cpu-gates")
    sub.add_parser("p3-replay")
    a = sub.add_parser("arm")
    a.add_argument("--arm", required=True, choices=list(ARMS))
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--smoke-epochs", type=int, default=None)
    pr = sub.add_parser("predict")
    pr.add_argument("--arms", nargs="*", default=None)
    pr.add_argument("--seeds", nargs="*", type=int, default=None)
    pr.add_argument("--valid", action="store_true")
    sub.add_parser("valid-reference")
    s = sub.add_parser("score-monitor")
    s.add_argument("--round", required=True, choices=["seed0", "seed1"])
    m = sub.add_parser("mechanism")
    m.add_argument("--seeds", nargs="*", type=int, default=None)
    m.add_argument("--molecules", type=int, default=128)
    sub.add_parser("valid-secondary")
    sub.add_parser("finalize")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "verify-inputs": cmd_verify_inputs,
        "freeze": cmd_freeze,
        "phase0": cmd_phase0,
        "cpu-gates": cmd_cpu_gates,
        "p3-replay": cmd_p3_replay,
        "arm": cmd_arm,
        "predict": cmd_predict,
        "valid-reference": cmd_valid_reference,
        "score-monitor": cmd_score_monitor,
        "mechanism": cmd_mechanism,
        "valid-secondary": cmd_valid_secondary,
        "finalize": cmd_finalize,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
