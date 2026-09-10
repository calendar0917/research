"""Compositional Patch Sharing Oracle (ZINC-12k, frozen compact-v4-hinge).

Stage mandate (see ``notes/compositional_patch_sharing_oracle.md``): this is a
**representation oracle / intervention** on the *frozen* compact-v4-hinge
validation-selected checkpoints (seeds 0-3).  It does **not** train any model,
does **not** change the loss/optimizer/graph head/topology channel, does
**not** re-design the patch representation and does **not** implement a
dictionary / KSVD / frequency gate.  The only operation is an auditable
replacement/blend of the **radius-2 exact patch embedding** (the output of
``model.typed_embedding``) at inference time, for rare/OOV patches only.

Hypothesis under test
---------------------
Exact patch identity is valuable, but rare/OOV exact tokens get very little
supervision.  *If* a rare/OOV exact patch embedding can borrow information from
structurally similar frequent patches, does frozen inference improve?

Intervention variants (pre-registered, do not widen):
  * NN1            : e_shared = e_exact(nearest frequent donor)
  * KNN8           : softmax(-d/T)-weighted mean of 8 nearest donors (T=1)
  * Blend50        : rare-seen 0.5*exact + 0.5*shared; OOV 1.0*shared
  * FrequentMean   : e_shared = mean of all frequent donor embeddings (control)
  * RandomDonor    : matched-k = mean of 8 *random* frequent donors (control)

Leakage / freeze rules (hard):
  * train defines vocabulary, frequency, structural descriptors, donor bank and
    embedding bank; validation is used only for frozen inference + diagnostics;
  * official **test is never loaded** (``TARGET_SPLIT`` is validation only);
  * donors are selected from graph/patch structure only -- never from targets,
    residuals or predictions;
  * frequency for a validation token is its official-train count only;
  * the four frozen selection checkpoints are reused, never retrained.

Stages (idempotent; ``--force`` to rebuild)::

    bank       train-only patch bank, coverage, donor bank, hierarchical search
    intervene  frozen 4-seed inference for baseline + 5 variants (bit-exact guard)
    metrics    overall + molecule-subgroup + disagreement + similarity analysis
    cases      improved/deteriorated molecule inspection
    figures    figures 1-6
    decision   Q1-Q15 + decision record
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit import (
    V4_RUN_IDS,
    _build_v4_model,
    _extract_v4_records,
    load_run_result,
    run_path,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ORACLE_ROOT = REPO_ROOT / "tracks/ksvd/results/compositional_patch_sharing_oracle"
FIG_DIR = ORACLE_ROOT / "figures"

SEEDS = (0, 1, 2, 3)

# Pre-registered thresholds (do not tune on validation results).
FREQUENT_DONOR_THRESHOLD = 20
DESCRIPTIVE_THRESHOLDS = (10, 20, 50)
RARE_TARGET_MAX = 5
KNN_K = 8
SOFTMAX_T = 1.0
BLEND_ALPHA = 0.5

# 146D shell descriptor layout (see ``_shell_descriptor``):
#   [atom_shell (3*28) | bond_shell (6*4) | root_atom (28) | incident_bonds (4) | scalars (6)]
ROOT_ATOM_OFFSET = 3 * zpp.ATOM_CATEGORIES + 6 * zpp.BOND_CATEGORIES  # 108
ROOT_ATOM_END = ROOT_ATOM_OFFSET + zpp.ATOM_CATEGORIES  # 136


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_marker(stage: str) -> Path:
    return ORACLE_ROOT / f"stage_{stage}.json"


def _mark_done(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["stage"] = stage
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_stage_marker(stage), result)
    return result


def _load_grams(path: Path) -> list[Any]:
    with gzip.open(path, "rb") as handle:
        return list(pickle.load(handle))


# --------------------------------------------------------------------------
# stage: bank
# --------------------------------------------------------------------------


@dataclass
class DonorBank:
    token_ids: np.ndarray  # [D] vocabulary token id (>=1)
    freqs: np.ndarray  # [D] official-train occurrence count
    desc_std: np.ndarray  # [D, 146] train-standardized descriptor prototype
    root_atom: np.ndarray  # [D] argmax(root_atom one-hot)
    parent_hex: list[str]  # [D] radius-1 parent certificate hex


def _patch_root_atom(descriptor: np.ndarray) -> int:
    block = descriptor[ROOT_ATOM_OFFSET:ROOT_ATOM_END]
    return int(np.argmax(block))


def _build_bank() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(4)
    train_records, valid_records = _extract_v4_records()

    typed_vocabulary = zpp._fit_vocabulary(
        train_records,
        "typed_certificate",
        maximum=8192,
        minimum_frequency=1,
    )
    parent_vocabulary = zpp._fit_vocabulary(
        train_records,
        "parent_certificate",
        maximum=2048,
        minimum_frequency=1,
    )

    counts: Counter[bytes] = Counter()
    descriptor_acc: dict[bytes, list[np.ndarray]] = defaultdict(list)
    parent_acc: dict[bytes, Counter[bytes]] = defaultdict(Counter)
    for record in train_records:
        for patch in record.patches:
            counts[patch.typed_certificate] += 1
            descriptor_acc[patch.typed_certificate].append(patch.shell_descriptor)
            parent_acc[patch.typed_certificate][patch.parent_certificate] += 1

    # --- audit: is the frozen typed certificate injective on rooted patches? ---
    # ``_typed_certificate`` stores ``pynauty.certificate`` of the *colored*
    # incidence graph.  Empirically the certificate is NOT a complete invariant
    # of the colored graph (pynauty's own ``isomorphic`` also compares color-cell
    # sizes), so one token can group patches with different root atoms and
    # different 146D descriptors.  We quantify it here rather than assuming
    # injectivity.
    token_root_set: dict[bytes, set[int]] = {}
    token_parent_set: dict[bytes, set[bytes]] = {}
    token_descriptor_spread = 0.0
    tokens_with_multiple_roots = 0
    tokens_with_multiple_parents = 0
    for cert, descriptors in descriptor_acc.items():
        arr = np.stack(descriptors, axis=0)
        roots = {_patch_root_atom(descriptor) for descriptor in descriptors}
        token_root_set[cert] = roots
        if len(roots) > 1:
            tokens_with_multiple_roots += 1
        if len(parent_acc[cert]) > 1:
            tokens_with_multiple_parents += 1
        token_parent_set[cert] = set(parent_acc[cert])
        if arr.shape[0] > 1:
            token_descriptor_spread = max(
                token_descriptor_spread, float(np.abs(arr - arr.mean(0)).max())
            )

    certificate_audit = {
        "unique_train_tokens": int(len(descriptor_acc)),
        "tokens_with_multiple_root_atoms": int(tokens_with_multiple_roots),
        "tokens_with_multiple_parent_certs": int(tokens_with_multiple_parents),
        "max_descriptor_spread_within_token": float(token_descriptor_spread),
        "finding": (
            "the frozen typed certificate (_typed_certificate = pynauty.certificate "
            "of the colored incidence graph) is NOT injective on rooted typed "
            "patches: one token can group patches with different root atoms and "
            "different 146D descriptors. pynauty's own isomorphic() also compares "
            "color-cell sizes, confirming the certificate alone is not a complete "
            "colored-graph invariant."
        ),
        "provenance": "audited 2026-09-10 in this stage; representation left frozen",
    }

    id_to_cert = {index: cert for cert, index in typed_vocabulary.items()}
    cert_to_id = dict(typed_vocabulary)

    # train-only standardization of the 146D shell descriptor (same fit the
    # frozen model uses for its patch encoder input).
    patch_standardizer = zpp.Standardizer.fit(zpp._patch_matrix(train_records))

    token_freq = np.zeros(len(typed_vocabulary) + 1, dtype=np.int64)
    token_root = np.zeros(len(typed_vocabulary) + 1, dtype=np.int64)
    token_desc = np.zeros(
        (len(typed_vocabulary) + 1, zpp.SHELL_WIDTH), dtype=np.float32
    )
    token_desc_std = np.zeros(
        (len(typed_vocabulary) + 1, zpp.SHELL_WIDTH), dtype=np.float32
    )
    token_parent_hex: dict[int, str] = {}
    for token_id, cert in id_to_cert.items():
        token_freq[token_id] = counts[cert]
        # robust prototype: mean over occurrences (tokens are not injective,
        # so a single "first" descriptor would be arbitrary).
        stack = np.stack(descriptor_acc[cert], axis=0)
        raw = stack.mean(axis=0)
        token_desc[token_id] = patch_standardizer.transform(raw[None, :])[0]
        if stack.shape[0] > 1:
            token_desc_std[token_id] = stack.std(axis=0)
        # modal root atom (argmax of the mean one-hot block == most common)
        token_root[token_id] = _patch_root_atom(raw)
        # modal (most frequent) radius-1 parent certificate
        token_parent_hex[token_id] = parent_acc[cert].most_common(1)[0][0].hex()
    # OOV row (id 0): no structure / zero frequency; never a donor.
    token_desc[0] = 0.0
    token_desc_std[0] = 0.0
    token_parent_hex[0] = ""

    donor_mask = token_freq >= FREQUENT_DONOR_THRESHOLD
    donor_ids = np.flatnonzero(donor_mask)
    donor_bank = DonorBank(
        token_ids=donor_ids.astype(np.int64),
        freqs=token_freq[donor_ids].astype(np.int64),
        desc_std=token_desc[donor_ids].astype(np.float32),
        root_atom=token_root[donor_ids].astype(np.int64),
        parent_hex=[token_parent_hex[int(i)] for i in donor_ids],
    )

    # group donors by (root_atom, parent) and by root_atom for the hierarchy.
    level1: dict[tuple[int, str], list[int]] = defaultdict(list)
    level2: dict[int, list[int]] = defaultdict(list)
    for local, (root, parent) in enumerate(zip(donor_bank.root_atom, donor_bank.parent_hex)):
        level1[(int(root), str(parent))].append(local)
        level2[int(root)].append(local)

    # ---- validation occurrence sweep -------------------------------------
    # A validation patch is an intervention target iff its *official-train*
    # count is <= 5 (rare-seen 1..5, OOV 0).  OOV occurrences all collapse to
    # token id 0, so donor matching is done per *occurrence* (or per token for
    # rare-seen, where the descriptor is token-unique -- verified above).
    rng_seed_base = 20260910
    target_rows: list[dict[str, Any]] = []
    per_molecule_target_counts = []
    per_molecule_rare_ratio = []
    for mol_index, record in enumerate(valid_records):
        n_patches = len(record.patches)
        n_target = 0
        n_rare_le5 = 0
        for patch_index, patch in enumerate(record.patches):
            cert = patch.typed_certificate
            token_id = int(cert_to_id.get(cert, 0))
            freq = int(token_freq[token_id]) if token_id > 0 else 0
            if freq <= RARE_TARGET_MAX:
                n_rare_le5 += 1
            if freq > RARE_TARGET_MAX:
                continue
            n_target += 1
            desc = patch_standardizer.transform(patch.shell_descriptor[None, :])[0]
            root = _patch_root_atom(patch.shell_descriptor)
            parent = patch.parent_certificate.hex()
            candidates = level1.get((root, parent))
            level = 1
            if not candidates:
                candidates = level2.get(root)
                level = 2
            if not candidates:
                candidates = list(range(len(donor_bank.token_ids)))
                level = 3
            cand = np.asarray(candidates, dtype=np.int64)
            dist = np.linalg.norm(
                donor_bank.desc_std[cand] - desc[None, :], axis=1
            )
            order = np.argsort(dist, kind="stable")
            n_avail = int(order.size)
            k = min(KNN_K, n_avail)
            knn_local = cand[order[:k]]
            knn_dist_local = dist[order[:k]]
            weights = np.exp(-(knn_dist_local - knn_dist_local.min()) / SOFTMAX_T)
            weights = weights / weights.sum()
            rng = np.random.default_rng(
                (rng_seed_base + token_id * 1000003 + mol_index * 1009 + patch_index)
                % (2**32)
            )
            rand_local = rng.choice(cand, size=k, replace=False)
            target_rows.append(
                {
                    "mol_index": mol_index,
                    "patch_index": patch_index,
                    "molecule_id": f"valid:{mol_index}",
                    "patch_token": token_id,
                    "train_freq": freq,
                    "group": ("oov" if freq == 0 else "rare_seen"),
                    "root_atom": int(root),
                    "donor_level": int(level),
                    "n_level_donors": int(n_avail),
                    "n_knn": int(k),
                    "nearest_local": int(knn_local[0]),
                    "nearest_token": int(donor_bank.token_ids[knn_local[0]]),
                    "nearest_distance": float(knn_dist_local[0]),
                    "knn_local": knn_local.tolist(),
                    "knn_weights": weights.tolist(),
                    "knn_dist": knn_dist_local.tolist(),
                    "rand_local": rand_local.tolist(),
                }
            )
        per_molecule_target_counts.append(n_target)
        per_molecule_rare_ratio.append(
            float(n_rare_le5) / max(float(n_patches), 1.0)
        )

    # pad per-target donor arrays to a fixed width for a compact npz.
    max_k = max((len(row["knn_local"]) for row in target_rows), default=1)
    n_targets = len(target_rows)
    knn_idx = np.full((n_targets, max_k), -1, dtype=np.int64)
    knn_w = np.zeros((n_targets, max_k), dtype=np.float64)
    knn_dist = np.full((n_targets, max_k), np.nan, dtype=np.float64)
    rand_idx = np.full((n_targets, max_k), -1, dtype=np.int64)
    for i, row in enumerate(target_rows):
        k = len(row["knn_local"])
        knn_idx[i, :k] = row["knn_local"]
        knn_w[i, :k] = row["knn_weights"]
        knn_dist[i, :k] = row["knn_dist"]
        rand_idx[i, :k] = row["rand_local"]
    nearest_local = np.asarray([row["nearest_local"] for row in target_rows], dtype=np.int64)
    np.savez_compressed(
        ORACLE_ROOT / "bank.npz",
        donor_token_ids=donor_bank.token_ids,
        donor_freqs=donor_bank.freqs,
        donor_desc_std=donor_bank.desc_std,
        donor_root_atom=donor_bank.root_atom,
        all_token_desc_mean=token_desc,
        all_token_desc_std=token_desc_std,
        target_mol_index=np.asarray([r["mol_index"] for r in target_rows], dtype=np.int64),
        target_patch_index=np.asarray(
            [r["patch_index"] for r in target_rows], dtype=np.int64
        ),
        target_token=np.asarray([r["patch_token"] for r in target_rows], dtype=np.int64),
        target_freq=np.asarray([r["train_freq"] for r in target_rows], dtype=np.int64),
        target_group=np.asarray(
            [1 if r["group"] == "oov" else 0 for r in target_rows], dtype=np.int64
        ),
        target_level=np.asarray([r["donor_level"] for r in target_rows], dtype=np.int64),
        target_n_level_donors=np.asarray(
            [r["n_level_donors"] for r in target_rows], dtype=np.int64
        ),
        target_nearest_local=nearest_local,
        target_nearest_distance=np.asarray(
            [r["nearest_distance"] for r in target_rows], dtype=np.float64
        ),
        target_knn_idx=knn_idx,
        target_knn_w=knn_w,
        target_knn_dist=knn_dist,
        target_rand_idx=rand_idx,
        per_molecule_target_count=np.asarray(per_molecule_target_counts, dtype=np.int64),
        per_molecule_rare_ratio=np.asarray(per_molecule_rare_ratio, dtype=np.float64),
    )

    freq_values = np.asarray(list(counts.values()), dtype=np.int64)
    coverage = {
        "train_total_unique_typed_tokens": int(len(counts)),
        "train_total_patch_occurrences": int(sum(counts.values())),
        "frequent_ge20_tokens": int((freq_values >= 20).sum()),
        "frequent_ge10_tokens": int((freq_values >= 10).sum()),
        "frequent_ge50_tokens": int((freq_values >= 50).sum()),
        "rare_le5_tokens": int((freq_values <= 5).sum()),
        "medium_6_19_tokens": int(((freq_values >= 6) & (freq_values <= 19)).sum()),
        "oov_validation_tokens": int(
            sum(1 for r in valid_records for p in r.patches if p.typed_certificate not in typed_vocabulary)
        ),
        "donor_bank_size": int(len(donor_bank.token_ids)),
        "valid_total_patch_occurrences": int(
            sum(len(r.patches) for r in valid_records)
        ),
        "valid_target_occurrences": int(n_targets),
        "valid_rare_seen_occurrences": int(
            sum(r["group"] == "rare_seen" for r in target_rows)
        ),
        "valid_oov_occurrences": int(sum(r["group"] == "oov" for r in target_rows)),
        "valid_molecules_any_affected": int(
            sum(c > 0 for c in per_molecule_target_counts)
        ),
    }

    # per-molecule affected flags (computed explicitly)
    rare_seen_affected = []
    oov_affected = []
    target_by_mol: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        target_by_mol[row["mol_index"]].append(row)
    for mol_index in range(len(valid_records)):
        rows = target_by_mol.get(mol_index, [])
        rare_seen_affected.append(any(r["group"] == "rare_seen" for r in rows))
        oov_affected.append(any(r["group"] == "oov" for r in rows))
    coverage["valid_molecules_rare_seen_affected"] = int(sum(rare_seen_affected))
    coverage["valid_molecules_oov_affected"] = int(sum(oov_affected))

    level_counts = Counter(r["donor_level"] for r in target_rows)
    level_counts_by_group = {
        g: dict(Counter(r["donor_level"] for r in target_rows if r["group"] == g))
        for g in ("rare_seen", "oov")
    }

    _write_json(
        ORACLE_ROOT / "coverage.json",
        {
            **coverage,
            "donor_constraint_level_counts": {str(k): int(v) for k, v in level_counts.items()},
            "donor_constraint_level_counts_by_group": {
                g: {str(k): int(v) for k, v in d.items()}
                for g, d in level_counts_by_group.items()
            },
            "thresholds": {
                "frequent_donor": FREQUENT_DONOR_THRESHOLD,
                "descriptive": list(DESCRIPTIVE_THRESHOLDS),
                "rare_target_max": RARE_TARGET_MAX,
                "knn_k": KNN_K,
                "softmax_T": SOFTMAX_T,
                "blend_alpha": BLEND_ALPHA,
            },
            "certificate_audit": certificate_audit,
            "vocabulary_size_with_oov": int(len(typed_vocabulary) + 1),
            "parent_vocabulary_size_with_oov": int(len(parent_vocabulary) + 1),
            "ready": True,
        },
    )

    return _mark_done(
        "bank",
        {
            "coverage": coverage,
            "donor_constraint_level_counts": {str(k): int(v) for k, v in level_counts.items()},
            "donor_constraint_level_counts_by_group": {
                g: {str(k): int(v) for k, v in d.items()}
                for g, d in level_counts_by_group.items()
            },
            "certificate_audit": certificate_audit,
            "n_targets": int(n_targets),
            "seconds": float(time.perf_counter() - started),
        },
    )


# --------------------------------------------------------------------------
# stage: intervene
# --------------------------------------------------------------------------


class _EmbeddingIntervention:
    """Forward hook that replaces rows of ``model.typed_embedding`` output.

    Only rows flagged ``active`` are replaced; every other row is returned
    bit-identically (``torch.where`` selects the original values).  When
    ``override`` is None the hook returns ``None`` and the module output is
    untouched, so a baseline run is bit-exact.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.override: torch.Tensor | None = None
        self.active: torch.Tensor | None = None
        self._handle = model.typed_embedding.register_forward_hook(self._hook)

    def _hook(
        self, module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> torch.Tensor | None:
        if self.override is None or self.active is None:
            return None
        return torch.where(self.active.unsqueeze(-1), self.override, output)

    def remove(self) -> None:
        self._handle.remove()


def _frozen_forward(
    model: torch.nn.Module,
    graphs: Sequence[Any],
    per_molecule_override: list[torch.Tensor | None] | None = None,
    per_molecule_active: list[torch.Tensor | None] | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    loader = DataLoader(list(graphs), batch_size=int(batch_size), shuffle=False)
    intervention = _EmbeddingIntervention(model)
    predictions: list[np.ndarray] = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            n = int(batch.num_graphs)
            if per_molecule_override is not None:
                rows_override = per_molecule_override[cursor : cursor + n]
                rows_active = per_molecule_active[cursor : cursor + n]
                override = torch.cat(rows_override, dim=0)
                active = torch.cat(rows_active, dim=0)
                if bool(active.any()):
                    intervention.override = override
                    intervention.active = active
                else:
                    intervention.override = None
                    intervention.active = None
            predictions.append(model(batch).cpu().numpy())
            cursor += n
    intervention.remove()
    prediction = np.concatenate(predictions).astype(np.float64)
    return zpp._quantile_point_prediction_numpy(prediction)


def _donor_embeddings(model: torch.nn.Module, donor_token_ids: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return model.typed_embedding(torch.from_numpy(donor_token_ids).long()).clone()


def _build_molecule_overrides(
    variant: str,
    n_molecules: int,
    per_molecule_patches: Sequence[int],
    target_mol: np.ndarray,
    target_patch: np.ndarray,
    target_token: np.ndarray,
    target_group: np.ndarray,
    knn_idx: np.ndarray,
    knn_w: np.ndarray,
    rand_idx: np.ndarray,
    donor_emb: torch.Tensor,
    typed_embedding: torch.nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    width = int(donor_emb.shape[1])
    overrides: list[torch.Tensor | None] = [None] * n_molecules
    actives: list[torch.Tensor | None] = [None] * n_molecules
    # bucket targets by molecule
    per_mol: dict[int, list[int]] = defaultdict(list)
    for i in range(target_mol.shape[0]):
        per_mol[int(target_mol[i])].append(i)

    donor_mean = donor_emb.mean(dim=0)
    for mol_index in range(n_molecules):
        n_patches = int(per_molecule_patches[mol_index])
        override = torch.zeros((n_patches, width), dtype=donor_emb.dtype)
        active = torch.zeros(n_patches, dtype=torch.bool)
        for i in per_mol.get(mol_index, []):
            p = int(target_patch[i])
            token = int(target_token[i])
            with torch.no_grad():
                e_exact = typed_embedding(torch.tensor([token], dtype=torch.long))[0]
            if variant == "baseline":
                active[p] = False
                continue
            if variant == "NN1":
                shared = donor_emb[int(knn_idx[i, 0])]
                value = shared
            elif variant == "KNN8":
                idx = torch.from_numpy(knn_idx[i][knn_idx[i] >= 0]).long()
                w = torch.from_numpy(knn_w[i][: idx.shape[0]]).to(donor_emb.dtype)
                value = (donor_emb[idx] * w[:, None]).sum(dim=0)
            elif variant == "Blend50":
                idx = torch.from_numpy(knn_idx[i][knn_idx[i] >= 0]).long()
                w = torch.from_numpy(knn_w[i][: idx.shape[0]]).to(donor_emb.dtype)
                shared = (donor_emb[idx] * w[:, None]).sum(dim=0)
                alpha = 0.0 if int(target_group[i]) == 1 else BLEND_ALPHA
                value = alpha * e_exact + (1.0 - alpha) * shared
            elif variant == "FrequentMean":
                value = donor_mean
            elif variant == "RandomDonor":
                idx = torch.from_numpy(rand_idx[i][rand_idx[i] >= 0]).long()
                value = donor_emb[idx].mean(dim=0)
            else:
                raise ValueError(f"unknown variant {variant!r}")
            override[p] = value.to(override.dtype)
            active[p] = True
        overrides[mol_index] = override
        actives[mol_index] = active
    return overrides, actives  # type: ignore[return-value]


def _intervene() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(4)
    bank = np.load(ORACLE_ROOT / "bank.npz")
    donor_token_ids = bank["donor_token_ids"]
    target_mol = bank["target_mol_index"]
    target_patch = bank["target_patch_index"]
    target_token = bank["target_token"]
    target_group = bank["target_group"]
    knn_idx = bank["target_knn_idx"]
    knn_w = bank["target_knn_w"]
    rand_idx = bank["target_rand_idx"]

    train_records, valid_records = _extract_v4_records()
    per_molecule_patches = [len(record.patches) for record in valid_records]
    n_molecules = len(valid_records)

    variants = ["baseline", "NN1", "KNN8", "Blend50", "FrequentMean", "RandomDonor"]
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        run_id = V4_RUN_IDS[seed]
        run_dir = run_path(run_id)
        config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
        result = load_run_result(run_id)
        run_preds = np.asarray(result["evaluation"]["valid"]["predictions"], dtype=np.float64)
        _fit, eval_data, audit = zpp._phase_data(train_records, valid_records, config=config)
        model = _build_v4_model(config, audit)
        state = torch.load(
            run_dir / "artifacts" / "legacy_full_result_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        n_params = int(sum(p.numel() for p in model.parameters()))
        if n_params != 99613:
            raise AssertionError(f"seed {seed}: params {n_params} != 99613")

        donor_emb = _donor_embeddings(model, donor_token_ids)
        if not bool(torch.isfinite(donor_emb).all()):
            raise AssertionError(f"seed {seed}: non-finite donor embeddings")

        seed_records: dict[str, Any] = {}
        baseline_preds = _frozen_forward(model, eval_data)
        max_diff = float(np.abs(baseline_preds - run_preds).max())
        if max_diff > 1e-9:
            raise AssertionError(
                f"seed {seed}: baseline forward deviates from recorded run "
                f"(max diff {max_diff}) -- fix inference path before any oracle"
            )
        seed_records["baseline"] = baseline_preds
        seed_records["baseline_max_abs_diff_vs_run"] = max_diff

        for variant in variants:
            if variant == "baseline":
                continue
            overrides, actives = _build_molecule_overrides(
                variant,
                n_molecules,
                per_molecule_patches,
                target_mol,
                target_patch,
                target_token,
                target_group,
                knn_idx,
                knn_w,
                rand_idx,
                donor_emb,
                model.typed_embedding,
            )
            if any(not bool(torch.isfinite(o).all()) for o in overrides):
                raise AssertionError(f"seed {seed} variant {variant}: non-finite override")
            preds = _frozen_forward(model, eval_data, overrides, actives)
            seed_records[variant] = preds

        # unaffected bit-identity guard (variant KNN8 as representative)
        affected_any = np.zeros(n_molecules, dtype=bool)
        for mol in target_mol:
            affected_any[int(mol)] = True
        knn8_preds = seed_records["KNN8"]
        if affected_any.any():
            unaffected_mask = ~affected_any
            diff = np.abs(knn8_preds[unaffected_mask] - baseline_preds[unaffected_mask])
            max_unaffected = float(diff.max())
        else:
            max_unaffected = 0.0
        seed_records["knn8_unaffected_max_abs_diff"] = max_unaffected

        per_seed[str(seed)] = {
            "run_id": run_id,
            "params": n_params,
            "baseline_max_abs_diff_vs_run": max_diff,
            "knn8_unaffected_max_abs_diff": max_unaffected,
            "records": seed_records,
        }
        print(
            f"[intervene] seed {seed} done (baseline diff {max_diff}, "
            f"unaffected diff {max_unaffected})",
            flush=True,
        )

    arrays = {}
    for seed in SEEDS:
        for variant in variants:
            arrays[f"pred_seed{seed}_{variant}"] = per_seed[str(seed)]["records"][variant]
    np.savez_compressed(ORACLE_ROOT / "predictions.npz", **arrays)
    _write_json(
        ORACLE_ROOT / "intervene_meta.json",
        {
            str(seed): {
                "run_id": per_seed[str(seed)]["run_id"],
                "params": per_seed[str(seed)]["params"],
                "baseline_max_abs_diff_vs_run": per_seed[str(seed)][
                    "baseline_max_abs_diff_vs_run"
                ],
                "knn8_unaffected_max_abs_diff": per_seed[str(seed)][
                    "knn8_unaffected_max_abs_diff"
                ],
            }
            for seed in SEEDS
        },
    )
    return _mark_done(
        "intervene",
        {
            "seeds": [int(s) for s in SEEDS],
            "variants": variants,
            "baseline_bit_exact_all_seeds": all(
                per_seed[str(s)]["baseline_max_abs_diff_vs_run"] == 0.0 for s in SEEDS
            ),
            "knn8_unaffected_bit_exact_all_seeds": all(
                per_seed[str(s)]["knn8_unaffected_max_abs_diff"] == 0.0 for s in SEEDS
            ),
            "seconds": float(time.perf_counter() - started),
        },
    )


# --------------------------------------------------------------------------
# stage: metrics
# --------------------------------------------------------------------------


def _load_targets() -> pd.DataFrame:
    bank = np.load(ORACLE_ROOT / "bank.npz")
    frame = pd.DataFrame(
        {
            "mol_index": bank["target_mol_index"],
            "patch_index": bank["target_patch_index"],
            "token": bank["target_token"],
            "train_freq": bank["target_freq"],
            "group": np.where(bank["target_group"] == 1, "oov", "rare_seen"),
            "level": bank["target_level"],
            "n_level_donors": bank["target_n_level_donors"],
            "nearest_local": bank["target_nearest_local"],
            "nearest_distance": bank["target_nearest_distance"],
        }
    )
    return frame


def _molecule_groups(targets: pd.DataFrame, n_molecules: int, rare_ratio: np.ndarray) -> pd.DataFrame:
    rare_seen = np.zeros(n_molecules, dtype=bool)
    oov = np.zeros(n_molecules, dtype=bool)
    for row in targets.itertuples(index=False):
        if row.group == "oov":
            oov[int(row.mol_index)] = True
        else:
            rare_seen[int(row.mol_index)] = True
    any_affected = rare_seen | oov
    quintile = np.full(n_molecules, -1, dtype=int)
    if any_affected.any():
        valid_vals = rare_ratio[any_affected]
        edges = np.quantile(valid_vals, [0.2, 0.4, 0.6, 0.8])
        quintile[any_affected] = np.searchsorted(edges, valid_vals, side="right")
    return pd.DataFrame(
        {
            "mol_index": np.arange(n_molecules),
            "rare_seen_affected": rare_seen,
            "oov_affected": oov,
            "any_affected": any_affected,
            "unaffected": ~any_affected,
            "rare_ratio": rare_ratio,
            "highest_rarity_quintile": quintile == 4,
        }
    )


def _metrics() -> dict[str, Any]:
    started = time.perf_counter()
    predictions = np.load(ORACLE_ROOT / "predictions.npz")
    bank = np.load(ORACLE_ROOT / "bank.npz")
    train_records, valid_records = _extract_v4_records()
    y = np.asarray([record.y for record in valid_records], dtype=np.float64)
    n_molecules = len(y)
    targets = _load_targets()
    rare_ratio = bank["per_molecule_rare_ratio"]
    groups = _molecule_groups(targets, n_molecules, rare_ratio)

    variants = ["baseline", "NN1", "KNN8", "Blend50", "FrequentMean", "RandomDonor"]

    # per-seed per-variant MAE
    per_seed_mae: dict[str, dict[str, float]] = {v: {} for v in variants}
    per_seed_abs: dict[str, dict[str, np.ndarray]] = {}
    preds_by_variant: dict[str, list[np.ndarray]] = {v: [] for v in variants}
    for v in variants:
        for seed in SEEDS:
            pred = predictions[f"pred_seed{seed}_{v}"]
            abs_err = np.abs(y - pred)
            per_seed_mae[v][str(seed)] = float(abs_err.mean())
            per_seed_abs.setdefault(v, {})[str(seed)] = abs_err
            preds_by_variant[v].append(pred)

    overall_delta = {}
    for v in variants:
        if v == "baseline":
            continue
        deltas = [
            per_seed_mae["baseline"][str(s)] - per_seed_mae[v][str(s)] for s in SEEDS
        ]
        overall_delta[v] = {
            "per_seed": {str(s): float(d) for s, d in zip(SEEDS, deltas)},
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas, ddof=1)),
            "n_positive": int(sum(d > 0 for d in deltas)),
            "seeds_same_direction": bool(
                all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
            ),
        }

    # subgroup tables (molecule-level mean abs error; mean over seeds)
    subgroup_masks = {
        "unaffected": groups["unaffected"].to_numpy(),
        "rare_seen_affected": groups["rare_seen_affected"].to_numpy(),
        "oov_affected": groups["oov_affected"].to_numpy(),
        "highest_rarity_quintile": groups["highest_rarity_quintile"].to_numpy(),
    }
    subgroup_table: dict[str, dict[str, Any]] = {}
    for name, mask in subgroup_masks.items():
        n = int(mask.sum())
        table: dict[str, Any] = {"n": n}
        for v in variants:
            mae = float(np.mean([per_seed_abs[v][str(s)][mask].mean() for s in SEEDS]))
            table[v] = mae
        for v in variants:
            if v == "baseline":
                continue
            deltas = [
                float(per_seed_abs["baseline"][str(s)][mask].mean())
                - float(per_seed_abs[v][str(s)][mask].mean())
                for s in SEEDS
            ]
            table[f"delta_{v}"] = float(np.mean(deltas))
            table[f"delta_{v}_per_seed"] = [float(d) for d in deltas]
            table[f"delta_{v}_n_positive"] = int(sum(d > 0 for d in deltas))
            table[f"delta_{v}_std"] = float(np.std(deltas, ddof=1))
        subgroup_table[name] = table

    # cross-seed disagreement (std across seeds) before vs after
    disagreement: dict[str, Any] = {}
    for name, mask in subgroup_masks.items():
        row: dict[str, Any] = {"n": int(mask.sum())}
        for v in variants:
            stack = np.stack(preds_by_variant[v], axis=0)  # [4, n_molecules]
            std = stack.std(axis=0, ddof=0)
            row[f"disagreement_{v}"] = float(std[mask].mean()) if mask.any() else float("nan")
        for v in variants:
            if v == "baseline":
                continue
            row[f"delta_{v}"] = (
                row[f"disagreement_{v}"] - row["disagreement_baseline"]
            )
        disagreement[name] = row

    # similarity-quality bins: molecule-level mean nearest donor distance
    mol_nearest = np.full(n_molecules, np.nan, dtype=np.float64)
    for mol, sub in targets.groupby("mol_index"):
        mol_nearest[int(mol)] = float(sub["nearest_distance"].mean())
    any_affected = groups["any_affected"].to_numpy()
    affected_idx = np.flatnonzero(any_affected)
    dist_edges = np.quantile(mol_nearest[affected_idx], [0.2, 0.4, 0.6, 0.8])
    dist_bin = np.full(n_molecules, -1, dtype=int)
    dist_bin[affected_idx] = np.searchsorted(
        dist_edges, mol_nearest[affected_idx], side="right"
    )
    similarity_bins = []
    for b in range(5):
        mask = dist_bin == b
        n_patches = int((targets["mol_index"].map(lambda m: dist_bin[int(m)] == b)).sum())
        entry = {
            "bin": b,
            "n_molecules": int(mask.sum()),
            "n_patches": n_patches,
            "mean_nearest_distance": float(mol_nearest[mask].mean()) if mask.any() else float("nan"),
        }
        for v in ("NN1", "KNN8", "Blend50"):
            deltas = [
                float(per_seed_abs["baseline"][str(s)][mask].mean())
                - float(per_seed_abs[v][str(s)][mask].mean())
                for s in SEEDS
            ] if mask.any() else [float("nan")]
            entry[f"delta_{v}"] = float(np.nanmean(deltas))
        similarity_bins.append(entry)

    # patch-level representation movement and nearest distance (per seed, NN1/KNN8)
    representation_movement = {}
    bank_targets = np.load(ORACLE_ROOT / "bank.npz")
    for v in ("NN1", "KNN8"):
        representation_movement[v] = {}
        for seed in SEEDS:
            run_dir = run_path(V4_RUN_IDS[seed])
            config = yaml.safe_load(
                (run_dir / "config.resolved.yaml").read_text(encoding="utf-8")
            )
            _fit, _eval, audit = zpp._phase_data(train_records, valid_records, config=config)
            # construct donor embeddings via the frozen checkpoint
            model = _build_v4_model(config, audit)
            model.load_state_dict(
                torch.load(
                    run_dir / "artifacts" / "legacy_full_result_selection_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            donor_emb = _donor_embeddings(model, bank_targets["donor_token_ids"])
            moves = []
            for i in range(bank_targets["target_token"].shape[0]):
                token = int(bank_targets["target_token"][i])
                with torch.no_grad():
                    e_exact = model.typed_embedding(torch.tensor([token])).clone()
                if v == "NN1":
                    shared = donor_emb[int(bank_targets["target_knn_idx"][i, 0])]
                else:
                    idx = bank_targets["target_knn_idx"][i]
                    idx = idx[idx >= 0]
                    w = torch.from_numpy(bank_targets["target_knn_w"][i][: idx.shape[0]]).float()
                    shared = (donor_emb[idx] * w[:, None]).sum(0)
                moves.append(float(torch.linalg.norm(e_exact[0] - shared)))
            representation_movement[v][str(seed)] = moves

    np.savez_compressed(
        ORACLE_ROOT / "movement.npz",
        **{
            f"{v}_seed{s}": np.asarray(representation_movement[v][str(s)])
            for v in ("NN1", "KNN8")
            for s in SEEDS
        },
    )

    result = {
        "overall_mae": per_seed_mae,
        "overall_delta": overall_delta,
        "subgroup_table": subgroup_table,
        "disagreement": disagreement,
        "similarity_bins": similarity_bins,
        "movement_summary": {
            v: {
                "mean": float(np.mean(representation_movement[v][str(s)]))
                for s in SEEDS
            }
            for v in ("NN1", "KNN8")
        },
    }
    _write_json(ORACLE_ROOT / "metrics.json", result)
    groups.to_csv(ORACLE_ROOT / "molecule_groups.csv", index=False)
    targets.to_csv(ORACLE_ROOT / "target_patches.csv", index=False)
    return _mark_done("metrics", {**result, "seconds": float(time.perf_counter() - started)})


# --------------------------------------------------------------------------
# stage: cases
# --------------------------------------------------------------------------


def _cases() -> dict[str, Any]:
    started = time.perf_counter()
    predictions = np.load(ORACLE_ROOT / "predictions.npz")
    train_records, valid_records = _extract_v4_records()
    y = np.asarray([record.y for record in valid_records], dtype=np.float64)
    targets = _load_targets()

    # molecule-level error change (baseline - KNN8), averaged over seeds
    base_err = np.mean(
        [np.abs(y - predictions[f"pred_seed{s}_baseline"]) for s in SEEDS], axis=0
    )
    knn_err = np.mean(
        [np.abs(y - predictions[f"pred_seed{s}_KNN8"]) for s in SEEDS], axis=0
    )
    delta = base_err - knn_err  # positive = improved
    affected = np.zeros(len(y), dtype=bool)
    affected[targets["mol_index"].to_numpy()] = True
    idx = np.flatnonzero(affected)
    order = idx[np.argsort(-delta[idx])]
    improved = order[:5]
    worsened = order[-5:]

    rows = []
    for label, indices in (("improved", improved), ("worsened", worsened)):
        for mol_index in indices:
            sub = targets[targets["mol_index"] == mol_index]
            patch_info = []
            for row in sub.itertuples(index=False):
                patch_info.append(
                    {
                        "patch_index": int(row.patch_index),
                        "group": row.group,
                        "train_freq": int(row.train_freq),
                        "donor_level": int(row.level),
                        "nearest_distance": float(row.nearest_distance),
                    }
                )
            rows.append(
                {
                    "case": label,
                    "molecule_id": f"valid:{int(mol_index)}",
                    "mol_index": int(mol_index),
                    "target": float(y[mol_index]),
                    "baseline_prediction_mean": float(
                        np.mean([predictions[f"pred_seed{s}_baseline"][mol_index] for s in SEEDS])
                    ),
                    "knn8_prediction_mean": float(
                        np.mean([predictions[f"pred_seed{s}_KNN8"][mol_index] for s in SEEDS])
                    ),
                    "baseline_error": float(base_err[mol_index]),
                    "knn8_error": float(knn_err[mol_index]),
                    "delta_error": float(delta[mol_index]),
                    "n_target_patches": int(sub.shape[0]),
                    "patch_info": patch_info,
                }
            )
    _write_json(ORACLE_ROOT / "cases.json", {"cases": rows})
    return _mark_done(
        "cases", {"n_improved_shown": 5, "n_worsened_shown": 5, "seconds": float(time.perf_counter() - started)}
    )


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------


def _figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    predictions = np.load(ORACLE_ROOT / "predictions.npz")
    train_records, valid_records = _extract_v4_records()
    y = np.asarray([record.y for record in valid_records], dtype=np.float64)
    targets = _load_targets()
    metrics = _read_json(ORACLE_ROOT / "metrics.json")

    # Figure 1: train patch frequency distribution (log-log)
    counts = Counter(
        patch.typed_certificate for record in train_records for patch in record.patches
    )
    freqs = np.asarray(sorted(counts.values(), reverse=True))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.arange(1, freqs.size + 1), freqs, lw=1)
    ax.axhline(FREQUENT_DONOR_THRESHOLD, color="C3", ls="--", lw=1, label="donor >= 20")
    ax.axhline(RARE_TARGET_MAX + 0.001, color="C2", ls=":", lw=1, label="rare <= 5")
    ax.set_xlabel("token rank (train)")
    ax.set_ylabel("train frequency")
    ax.set_title("Fig 1  radius-2 exact patch train-frequency distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_patch_frequency.png", dpi=150)
    plt.close(fig)

    # Figure 2: nearest donor distance distribution rare-seen vs OOV
    fig, ax = plt.subplots(figsize=(6, 4))
    for group, color in (("rare_seen", "C0"), ("oov", "C1")):
        vals = targets.loc[targets["group"] == group, "nearest_distance"].to_numpy()
        if vals.size:
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=f"{group} (n={vals.size})")
    ax.set_xlabel("standardized Euclidean distance to nearest frequent donor")
    ax.set_ylabel("patch occurrences")
    ax.set_title("Fig 2  nearest donor distance: rare-seen vs OOV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_donor_distance.png", dpi=150)
    plt.close(fig)

    # Figure 3: rare/OOV subgroup baseline vs KNN8 MAE
    subgroup = metrics["subgroup_table"]
    names = ["rare_seen_affected", "oov_affected", "highest_rarity_quintile"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(names))
    base_vals = [subgroup[n]["baseline"] for n in names]
    knn_vals = [subgroup[n]["KNN8"] for n in names]
    ax.bar(xs - 0.2, base_vals, width=0.4, label="baseline", color="C0")
    ax.bar(xs + 0.2, knn_vals, width=0.4, label="KNN8", color="C1")
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylabel("validation MAE")
    ax.set_title("Fig 3  rare/OOV subgroup MAE: baseline vs KNN8 (4-seed mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_subgroup_mae.png", dpi=150)
    plt.close(fig)

    # Figure 4: donor distance vs per-molecule error change
    mol_nearest = np.full(len(y), np.nan)
    for mol, sub in targets.groupby("mol_index"):
        mol_nearest[int(mol)] = float(sub["nearest_distance"].mean())
    base_err = np.mean([np.abs(y - predictions[f"pred_seed{s}_baseline"]) for s in SEEDS], axis=0)
    knn_err = np.mean([np.abs(y - predictions[f"pred_seed{s}_KNN8"]) for s in SEEDS], axis=0)
    delta = base_err - knn_err
    mask = ~np.isnan(mol_nearest)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mol_nearest[mask], delta[mask], s=8, alpha=0.4)
    bins = np.quantile(mol_nearest[mask], np.linspace(0, 1, 9))
    mids, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = mask & (mol_nearest >= lo) & (mol_nearest <= hi)
        if sel.any():
            mids.append(float(mol_nearest[sel].mean()))
            means.append(float(delta[sel].mean()))
    ax.plot(mids, means, "C3-o", lw=2, label="binned mean")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("mean nearest donor distance (molecule)")
    ax.set_ylabel("error change (baseline - KNN8); >0 = improved")
    ax.set_title("Fig 4  donor structural distance vs error change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_distance_vs_delta.png", dpi=150)
    plt.close(fig)

    # Figure 5: baseline vs sharing seed disagreement
    disagreement = metrics["disagreement"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(names))
    base_d = [disagreement[n]["disagreement_baseline"] for n in names]
    knn_d = [disagreement[n]["disagreement_KNN8"] for n in names]
    ax.bar(xs - 0.2, base_d, width=0.4, label="baseline", color="C0")
    ax.bar(xs + 0.2, knn_d, width=0.4, label="KNN8", color="C1")
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylabel("cross-seed prediction std")
    ax.set_title("Fig 5  cross-seed disagreement: baseline vs KNN8")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_disagreement.png", dpi=150)
    plt.close(fig)

    # Figure 6: KNN8 vs frequent-mean vs random-donor overall delta
    variants = ["NN1", "KNN8", "Blend50", "FrequentMean", "RandomDonor"]
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [metrics["overall_delta"][v]["mean"] for v in variants]
    stds = [metrics["overall_delta"][v]["std"] for v in variants]
    ax.bar(np.arange(len(variants)), means, yerr=stds, capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("overall ΔMAE (baseline - variant), 4-seed mean±std")
    ax.set_title("Fig 6  structural KNN vs control interventions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_variant_comparison.png", dpi=150)
    plt.close(fig)

    return _mark_done("figures", {"seconds": float(time.perf_counter() - started)})


# --------------------------------------------------------------------------
# stage: decision
# --------------------------------------------------------------------------


def _decision() -> dict[str, Any]:
    metrics = _read_json(ORACLE_ROOT / "metrics.json")
    coverage = _read_json(ORACLE_ROOT / "coverage.json")
    intervene_meta = _read_json(ORACLE_ROOT / "intervene_meta.json")
    subgroup = metrics["subgroup_table"]
    overall = metrics["overall_delta"]

    structural = ["NN1", "KNN8", "Blend50"]
    best_structural = max(structural, key=lambda v: overall[v]["mean"])
    best = overall[best_structural]
    rare_delta = subgroup["rare_seen_affected"][f"delta_{best_structural}"]
    oov_delta = subgroup["oov_affected"][f"delta_{best_structural}"]
    high_delta = subgroup["highest_rarity_quintile"][f"delta_{best_structural}"]
    targeted_gain = max(rare_delta, oov_delta, high_delta)

    knn = overall["KNN8"]["mean"]
    mean_control = overall["FrequentMean"]["mean"]
    random_control = overall["RandomDonor"]["mean"]
    knn_vs_mean = knn - mean_control
    knn_vs_random = knn - random_control
    knn_better_than_controls = knn > mean_control and knn > random_control

    n_pos = best["n_positive"]
    unaffected_max = max(
        float(intervene_meta[str(s)]["knn8_unaffected_max_abs_diff"]) for s in SEEDS
    )
    unaffected_ok = unaffected_max == 0.0

    if (
        best["mean"] >= 0.005
        and n_pos >= 3
        and targeted_gain >= 0.005
        and knn_better_than_controls
        and unaffected_ok
    ):
        decision = "STRONG GO - compositional patch representation"
    elif best["mean"] >= 0.003 and n_pos >= 3 and targeted_gain > 0:
        decision = "GO - compositional patch representation"
    elif (
        best["mean"] < 0.003
        and targeted_gain >= 0.02
        and n_pos >= 3
        and unaffected_ok
    ):
        decision = "TARGETED GO - compositional patch representation"
    elif (
        knn < 0.001
        and abs(rare_delta) < 0.005
        and abs(oov_delta) < 0.005
    ) or not knn_better_than_controls:
        decision = "NO-GO - compositional patch sharing"
    else:
        decision = "INCONCLUSIVE - weak/noisy signal"

    # interpretation taxonomy
    if knn_better_than_controls and best["mean"] > 0:
        interpretation = "A. structurally informed sharing"
    elif abs(knn - mean_control) < abs(knn_vs_random) and mean_control > 0:
        interpretation = "B. generic embedding shrinkage (KNN ~ frequent mean)"
    elif max(rare_delta, 0.0) > 0 and oov_delta > rare_delta + 0.005:
        interpretation = "C. OOV fallback only"
    elif rare_delta > oov_delta + 0.005 and rare_delta > 0:
        interpretation = "D. rare exact embedding instability"
    else:
        interpretation = "E. no useful sharing signal"

    qa = {
        "Q1_exact_embedding_path": (
            "radius-2 typed certificate -> _fit_vocabulary id (id 0 = OOV; ids "
            "1..V) -> model.typed_embedding = _HybridEmbedding(full table for "
            "id < 768, low-rank factorized table rank 4->16 for id >= 768); "
            "OOV id 0 is a learned full-table row.  The output e_patch is "
            "concatenated with the 146D standardized shell descriptor, the "
            "radius-1 parent embedding and the global patch context before "
            "the patch_encoder."
        ),
        "Q2_train_rare_le5_tokens": coverage["rare_le5_tokens"],
        "Q3_valid_molecules_affected": coverage["valid_molecules_any_affected"],
        "Q4_NN1_better_than_baseline": overall["NN1"]["mean"],
        "Q5_KNN8_better_than_NN1": overall["KNN8"]["mean"] - overall["NN1"]["mean"],
        "Q6_Blend50_better_than_replacement": overall["Blend50"]["mean"]
        - max(overall["NN1"]["mean"], overall["KNN8"]["mean"]),
        "Q7_rare_seen_subgroup_delta": rare_delta,
        "Q8_oov_subgroup_delta": oov_delta,
        "Q9_KNN8_vs_frequent_mean": knn_vs_mean,
        "Q10_KNN8_vs_random_donor": knn_vs_random,
        "Q11_closer_donor_higher_gain": metrics["similarity_bins"],
        "Q12_sharing_reduces_disagreement": {
            n: metrics["disagreement"][n]["delta_KNN8"] for n in metrics["disagreement"]
        },
        "Q13_unaffected_bit_identical": unaffected_ok,
        "Q14_interpretation": interpretation,
        "Q15_next_stage": {
            "A. structurally informed sharing": "design Shared Structural Base + Exact Identity Residual",
            "B. generic embedding shrinkage (KNN ~ frequent mean)": "shared prior / frequency-aware embedding shrinkage",
            "C. OOV fallback only": "OOV-specific structural encoder; keep seen exact embeddings",
            "D. rare exact embedding instability": "shared base + exact residual with frequency gate (later)",
            "E. no useful sharing signal": "close the compositional sharing / KSVD / subword / NN-embedding route",
        }.get(interpretation, "re-register a new stage"),
    }

    record = {
        "protocol_id": "luyin16-zinc-compositional-patch-sharing-oracle",
        "stage": "oracle-intervention",
        "frozen_model": {"runs": {str(s): V4_RUN_IDS[s] for s in SEEDS}, "params": 99613},
        "thresholds": coverage["thresholds"],
        "best_structural_variant": best_structural,
        "decision": decision,
        "interpretation": interpretation,
        "overall_delta_mean": best["mean"],
        "overall_delta_per_seed": best["per_seed"],
        "targeted_gain": targeted_gain,
        "knn_vs_frequent_mean": knn_vs_mean,
        "knn_vs_random_donor": knn_vs_random,
        "unaffected_bit_identical": unaffected_ok,
        "qa": qa,
        "coverage": coverage,
    }
    _write_json(ORACLE_ROOT / "decision_record.json", record)
    return _mark_done("decision", record)


STAGES = {
    "bank": _build_bank,
    "intervene": _intervene,
    "metrics": _metrics,
    "cases": _cases,
    "figures": _figures,
    "decision": _decision,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", default=list(STAGES), choices=list(STAGES) + [[]])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    ORACLE_ROOT.mkdir(parents=True, exist_ok=True)
    stages = args.stages or list(STAGES)
    for stage in stages:
        if not args.force and _stage_marker(stage).exists():
            print(f"[{stage}] already done (use --force to rerun)")
            continue
        print(f"[{stage}] running...", flush=True)
        STAGES[stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
