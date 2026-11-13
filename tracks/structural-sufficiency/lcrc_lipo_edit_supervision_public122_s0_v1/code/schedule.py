"""Frozen auxiliary-pair and batch-order schedule.

One independent NumPy ``Generator(20261113)`` draws every real auxiliary batch
(32 pairs, component-uniform then pair-uniform within the component), every
token shuffle, and every re-draw.  The ordinary molecule batch order uses the
frozen R3 shuffle generator (seed 917000).

The N arm is a **token permutation** of the same real batch: the 64 endpoint
occurrences are shuffled and paired adjacently.  A shuffle is rejected if any
resulting edge is the same row, the same complete canonical molecule, or any
P_train real edit edge.  Rejection only changes pairing, never labels: the N
label difference is recomputed as z_v - z_u.

Nothing here reads a target value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import data as D
import pairs as P

SCHEDULE_NPZ = C.CACHE / "schedule.npz"


def _pair_edges(pairs: Sequence[Mapping[str, Any]]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(p["row_a"]), int(p["row_b"])))) for p in pairs}


def _batch_perms(n_rows: int, epochs: int, seed: int) -> np.ndarray:
    gen = torch.Generator().manual_seed(int(seed))
    out = np.empty((int(epochs), int(n_rows)), dtype=np.int64)
    for epoch in range(int(epochs)):
        out[epoch] = torch.randperm(int(n_rows), generator=gen).numpy().astype(np.int64)
    return out


def _bad_edge(u: int, v: int, canonical: Mapping[int, str], forbidden: set[tuple[int, int]]) -> str | None:
    if int(u) == int(v):
        return "same_row"
    if str(canonical[int(u)]) == str(canonical[int(v)]):
        return "same_canonical_molecule"
    if tuple(sorted((int(u), int(v)))) in forbidden:
        return "p_train_real_edit_edge"
    return None


def build_schedule(
    pairs: Sequence[Mapping[str, Any]] | None = None,
    *,
    seed: int = C.SCHEDULE_SEED,
    write: bool = True,
) -> dict[str, Any]:
    pairs = P.load_pairs() if pairs is None else list(pairs)
    p_train = sorted(P.p_train(pairs), key=lambda p: str(p["pair_id"]))
    if not p_train:
        raise RuntimeError("CONTROL_CONSTRUCTION_FAILED: P_train is empty")

    # component -> member pair indices (in deterministic pair_id order)
    by_comp: dict[int, list[int]] = {}
    for idx, p in enumerate(p_train):
        by_comp.setdefault(int(p["train_component"]), []).append(idx)
    comp_ids = sorted(by_comp)
    comp_members = [by_comp[c] for c in comp_ids]

    src = C.source_with_labels("none")
    canonical = P.canonical_smiles_map(src["smiles"])
    forbidden = _pair_edges(p_train)

    rng = np.random.default_rng(int(seed))
    n_steps = int(C.RECIPE["total_optimizer_steps"])
    q = int(C.AUX_PAIRS)
    real_idx = np.empty((n_steps, q), dtype=np.int64)
    random_u = np.empty((n_steps, q), dtype=np.int64)
    random_v = np.empty((n_steps, q), dtype=np.int64)
    shuffle_attempts = np.zeros(n_steps, dtype=np.int64)
    redraws = np.zeros(n_steps, dtype=np.int64)
    rejected_shuffles = 0
    rejected_edges = 0

    def draw_batch() -> np.ndarray:
        out = np.empty(q, dtype=np.int64)
        for i in range(q):
            c = int(rng.integers(0, len(comp_members)))
            members = comp_members[c]
            j = int(rng.integers(0, len(members)))
            out[i] = int(members[j])
        return out

    for step in range(n_steps):
        for redraw in range(65):
            batch = draw_batch()
            tokens = np.empty(2 * q, dtype=np.int64)
            for i, pi in enumerate(batch):
                p = p_train[int(pi)]
                tokens[2 * i] = int(p["row_a"])
                tokens[2 * i + 1] = int(p["row_b"])
            chosen = None
            attempts = 0
            for attempt in range(256):
                attempts += 1
                shuffled = rng.permutation(tokens)
                cand = shuffled.reshape(q, 2)
                bad = False
                for u, v in cand:
                    reason = _bad_edge(int(u), int(v), canonical, forbidden)
                    if reason is not None:
                        bad = True
                        rejected_edges += 1
                        break
                if not bad:
                    chosen = cand
                    break
                rejected_shuffles += 1
            if chosen is None:
                redraws[step] += 1
                continue
            real_idx[step] = batch
            random_u[step] = chosen[:, 0]
            random_v[step] = chosen[:, 1]
            shuffle_attempts[step] = attempts
            break
        else:
            raise RuntimeError(
                f"CONTROL_CONSTRUCTION_FAILED: step {step + 1} needed more than 64 real-batch redraws"
            )

    perms = _batch_perms(int(C.RECIPE["global_batch_size"]), int(C.RECIPE["epochs"]), C.SHUFFLE_SEED)

    # per-step endpoint multiset equality (the final token multiset is a permutation)
    multiset_ok = True
    for step in range(n_steps):
        real_tokens = []
        for pi in real_idx[step]:
            p = p_train[int(pi)]
            real_tokens.extend((int(p["row_a"]), int(p["row_b"])))
        random_tokens = []
        random_tokens.extend(int(x) for x in random_u[step])
        random_tokens.extend(int(x) for x in random_v[step])
        if sorted(real_tokens) != sorted(random_tokens):
            multiset_ok = False
            break

    forbidden_final = 0
    for step in range(n_steps):
        for u, v in zip(random_u[step], random_v[step]):
            if _bad_edge(int(u), int(v), canonical, forbidden) is not None:
                forbidden_final += 1

    redraw_steps = np.flatnonzero(redraws > 0)
    manifest = {
        "schema": "lcrc-lipo-edit-supervision-schedule-manifest-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "label_blind": True,
        "seeds": {
            "train": C.TRAIN_SEED,
            "schedule_numpy": int(seed),
            "shuffle_generator": C.SHUFFLE_SEED,
            "bootstrap": C.BOOTSTRAP_SEED,
        },
        "q_pairs_per_step": q,
        "n_steps": n_steps,
        "epochs": int(C.RECIPE["epochs"]),
        "steps_per_epoch": int(C.RECIPE["steps_per_epoch"]),
        "p_train": {
            "n_pairs": len(p_train),
            "n_components": len(comp_members),
            "pair_ids_sha256": C.sha256_text(C.stable_json([str(p["pair_id"]) for p in p_train])),
        },
        "component_sampling": "uniform component with replacement, then uniform pair within component",
        "real_pair_indices": {
            "shape": list(real_idx.shape),
            "dtype": "int64",
            "sha256": C.array_sha256(real_idx),
        },
        "random_pair_rows": {
            "u_shape": list(random_u.shape),
            "v_shape": list(random_v.shape),
            "u_sha256": C.array_sha256(random_u),
            "v_sha256": C.array_sha256(random_v),
        },
        "batch_order": {
            "shape": list(perms.shape),
            "dtype": "int64",
            "sha256": C.array_sha256(perms),
            "epoch1_first10": [int(x) for x in perms[0, :10]],
        },
        "n_steps_redrawn": int(len(redraw_steps)),
        "redraw_fraction": float(len(redraw_steps) / n_steps),
        "redraw_steps_head": [int(x) + 1 for x in redraw_steps[:50]],
        "n_redraws_total": int(redraws.sum()),
        "max_redraws_per_step": int(redraws.max()) if n_steps else 0,
        "n_rejected_shuffles": int(rejected_shuffles),
        "n_rejected_edges": int(rejected_edges),
        "max_shuffle_attempts_used": int(shuffle_attempts.max()) if n_steps else 0,
        "mean_shuffle_attempts": float(shuffle_attempts.mean()) if n_steps else 0.0,
        "endpoint_multiset_equal_all_steps": bool(multiset_ok),
        "random_pairs_with_forbidden_edge": int(forbidden_final),
        "checks": {
            "same_row": True,
            "same_canonical_molecule": True,
            "p_train_real_edit_edge": True,
            "max_shuffle_attempts": 256,
            "max_real_batch_redraws": 64,
        },
    }
    if not multiset_ok:
        raise RuntimeError("CONTROL_CONSTRUCTION_FAILED: N/R endpoint multisets differ")
    if forbidden_final:
        raise RuntimeError("CONTROL_CONSTRUCTION_FAILED: final N pair contains a forbidden edge")

    if write:
        C.CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            SCHEDULE_NPZ,
            real_pair_indices=real_idx,
            random_u=random_u,
            random_v=random_v,
            batch_perms=perms,
            shuffle_attempts=shuffle_attempts,
            redraws=redraws,
        )
        C.write_json(C.ANALYSIS / "schedule_manifest.json", manifest)
        C.log(
            f"[schedule] {n_steps} steps, {len(redraw_steps)} redrawn "
            f"({len(redraw_steps) / n_steps:.4%}), max attempts {manifest['max_shuffle_attempts_used']}"
        )
    return {"arrays": {
        "real_pair_indices": real_idx,
        "random_u": random_u,
        "random_v": random_v,
        "batch_perms": perms,
        "shuffle_attempts": shuffle_attempts,
        "redraws": redraws,
    }, "manifest": manifest, "p_train": p_train, "canonical": canonical}


def load_schedule(path: Path | None = None) -> dict[str, Any]:
    path = SCHEDULE_NPZ if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run code/prepare.py --cache-only first")
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    manifest = C.read_json(C.ANALYSIS / "schedule_manifest.json")
    checks = {
        "real_pair_indices": C.array_sha256(arrays["real_pair_indices"]),
        "random_u": C.array_sha256(arrays["random_u"]),
        "random_v": C.array_sha256(arrays["random_v"]),
        "batch_perms": C.array_sha256(arrays["batch_perms"]),
    }
    expected = {
        "real_pair_indices": manifest["real_pair_indices"]["sha256"],
        "random_u": manifest["random_pair_rows"]["u_sha256"],
        "random_v": manifest["random_pair_rows"]["v_sha256"],
        "batch_perms": manifest["batch_order"]["sha256"],
    }
    for key, got in checks.items():
        if got != expected[key]:
            raise ValueError(f"{path}: {key} sha256 {got} != frozen {expected[key]}")
    return {"arrays": arrays, "manifest": manifest, "checks": checks}


def verify_against(manifest: Mapping[str, Any], p_train: Sequence[Mapping[str, Any]], canonical) -> dict[str, Any]:
    """Re-check an existing schedule without rebuilding it (remote prepare)."""
    loaded = load_schedule()
    arrays = loaded["arrays"]
    m = loaded["manifest"]
    p_train = sorted(p_train, key=lambda p: str(p["pair_id"]))
    forbidden = _pair_edges(p_train)
    same_rows = 0
    same_canon = 0
    real_edges = 0
    multiset_ok = True
    for step in range(int(m["n_steps"])):
        for u, v in zip(arrays["random_u"][step], arrays["random_v"][step]):
            reason = _bad_edge(int(u), int(v), canonical, forbidden)
            if reason == "same_row":
                same_rows += 1
            elif reason == "same_canonical_molecule":
                same_canon += 1
            elif reason == "p_train_real_edit_edge":
                real_edges += 1
        real_tokens = []
        for pi in arrays["real_pair_indices"][step]:
            p = p_train[int(pi)]
            real_tokens.extend((int(p["row_a"]), int(p["row_b"])))
        rand_tokens = [int(x) for x in arrays["random_u"][step]] + [int(x) for x in arrays["random_v"][step]]
        if sorted(real_tokens) != sorted(rand_tokens):
            multiset_ok = False
    out = {
        "schedule_npz_sha256": C.sha256_file(SCHEDULE_NPZ),
        "real_pair_indices_sha256": loaded["checks"]["real_pair_indices"],
        "random_pair_rows_u_sha256": loaded["checks"]["random_u"],
        "random_pair_rows_v_sha256": loaded["checks"]["random_v"],
        "batch_order_sha256": loaded["checks"]["batch_perms"],
        "n_steps_redrawn": int(m["n_steps_redrawn"]),
        "endpoint_multiset_equal_all_steps": bool(multiset_ok),
        "random_pair_same_row_count": int(same_rows),
        "random_pair_same_canonical_count": int(same_canon),
        "random_pair_p_train_edge_count": int(real_edges),
        "pass": bool(multiset_ok and same_rows == 0 and same_canon == 0 and real_edges == 0),
    }
    return out


def ensure_schedule(
    p_train: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> bool:
    """Rebuild the schedule only if the ignored cache is absent.

    The committed manifest is the authority.  Rebuilding uses the committed
    P_train pairs and the frozen seeds, then checks the four array hashes before
    writing the cache.  Returns True iff a rebuild happened.
    """
    if SCHEDULE_NPZ.exists():
        return False
    built = build_schedule(p_train, write=False)
    got = {
        "real_pair_indices": built["manifest"]["real_pair_indices"]["sha256"],
        "random_u": built["manifest"]["random_pair_rows"]["u_sha256"],
        "random_v": built["manifest"]["random_pair_rows"]["v_sha256"],
        "batch_perms": built["manifest"]["batch_order"]["sha256"],
        "pair_ids": built["manifest"]["p_train"]["pair_ids_sha256"],
    }
    expected = {
        "real_pair_indices": manifest["real_pair_indices"]["sha256"],
        "random_u": manifest["random_pair_rows"]["u_sha256"],
        "random_v": manifest["random_pair_rows"]["v_sha256"],
        "batch_perms": manifest["batch_order"]["sha256"],
        "pair_ids": manifest["p_train"]["pair_ids_sha256"],
    }
    bad = {k: (got[k], expected[k]) for k in expected if got[k] != expected[k]}
    if bad:
        raise ValueError(f"rebuilt schedule differs from committed manifest: {bad}")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCHEDULE_NPZ, **built["arrays"])
    C.log("[schedule] rebuilt ignored cache from committed pairs; all four array hashes match")
    return True


__all__ = [
    "SCHEDULE_NPZ",
    "build_schedule",
    "ensure_schedule",
    "load_schedule",
    "verify_against",
]
