"""Fixed, label-free Train reference retrieval (Tanimoto on the 2048 Morgan bits).

Rules frozen by the protocol (round ``rd-pilot-small-v1``):

* similarity uses only the 2048 Morgan bits of the *reused* ECFP cache;
  descriptors and labels never enter it;
* every reference must come from Train and the query's own global row_id is
  excluded, together with every Train row whose stripped SMILES is *exactly*
  the query's stripped SMILES (this is exact-string deduplication, not an
  isomorphism claim);
* the remaining references are sorted by similarity descending and Train
  global row_id ascending, top 64 kept.

No label is read here, so changing any label cannot change a reference,
a K_eff or a weight.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import rd_common as C
import rd_data as D


def tanimoto_matrix(query_bits: np.ndarray, ref_bits: np.ndarray) -> np.ndarray:
    """Exact Tanimoto for 0/1 float32 rows.

    ``intersection`` is a float32 dot product, which is exact for 0/1 vectors
    of length 2048 (all partial sums are integers ``<= 2048 < 2**24``), so no
    uint8/int8 overflow can happen.  ``union == 0`` (both all-zero) maps to 0.
    """
    q = np.ascontiguousarray(query_bits, dtype=np.float32)
    r = np.ascontiguousarray(ref_bits, dtype=np.float32)
    inter = q @ r.T
    qs = q.sum(axis=1, dtype=np.float32)
    rs = r.sum(axis=1, dtype=np.float32)
    union = qs[:, None] + rs[None, :] - inter
    out = np.zeros(inter.shape, dtype=np.float32)
    positive = union > 0
    np.divide(inter, union, out=out, where=positive)
    return out


def _train_smiles_index(smiles: list[str]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, text in enumerate(smiles):
        index.setdefault(str(text).strip(), []).append(position)
    return index


def reference_table(
    task: D.TaskData,
    *,
    max_references: int = C.MAX_REFERENCES,
    block_size: int = 512,
) -> dict[str, Any]:
    """Build the frozen top-64 reference table for every split."""
    train_bits = task.bits["train"]
    train_row_id = task.row_id["train"].astype(np.int64)
    smiles_index = _train_smiles_index(task.smiles["train"])
    n_train = int(train_bits.shape[0])

    tables: dict[str, dict[str, np.ndarray]] = {}
    for split in C.SPLITS:
        qbits = task.bits[split]
        nq = int(qbits.shape[0])
        ref_local = np.full((nq, max_references), -1, dtype=np.int32)
        ref_row_id = np.full((nq, max_references), -1, dtype=np.int64)
        sim = np.zeros((nq, max_references), dtype=np.float32)
        eligible_count = np.zeros(nq, dtype=np.int32)
        same_smiles_excluded = np.zeros(nq, dtype=np.int32)
        for start in range(0, nq, block_size):
            stop = min(start + block_size, nq)
            block = tanimoto_matrix(qbits[start:stop], train_bits)
            for offset in range(stop - start):
                query = start + offset
                exclude = np.zeros(n_train, dtype=bool)
                if split == "train":
                    exclude[query] = True
                same = smiles_index.get(task.smiles[split][query], [])
                same_others = [position for position in same if not (split == "train" and position == query)]
                if same_others:
                    exclude[np.asarray(same_others, dtype=np.int64)] = True
                same_smiles_excluded[query] = len(same_others)
                valid = np.nonzero(~exclude)[0]
                eligible_count[query] = int(valid.size)
                if valid.size == 0:
                    continue
                scores = block[offset, valid]
                order = np.lexsort((train_row_id[valid], -scores))
                keep = valid[order[:max_references]]
                keep_scores = scores[order[:max_references]]
                count = int(keep.size)
                ref_local[query, :count] = keep.astype(np.int32)
                ref_row_id[query, :count] = train_row_id[keep]
                sim[query, :count] = keep_scores.astype(np.float32)
        order_check = np.all(ref_local[:, :1] >= 0) if nq else True
        tables[split] = {
            "ref_local": ref_local,
            "ref_row_id": ref_row_id,
            "sim": sim,
            "eligible_count": eligible_count,
            "same_smiles_excluded": same_smiles_excluded,
            "max_sim": np.where(eligible_count > 0, sim[:, 0], 0.0).astype(np.float32),
            "n_kept": np.minimum(eligible_count, max_references).astype(np.int32),
            "query_row_id": task.row_id[split].astype(np.int64),
            "all_have_a_reference": bool(order_check),
        }
    return {
        "dataset": task.dataset,
        "max_references": int(max_references),
        "similarity": "tanimoto on 2048 morgan radius-2 bits (label free)",
        "exclusion": "self global row_id + identical stripped SMILES in Train",
        "sort": "similarity descending, Train global row_id ascending",
        "splits": tables,
    }


def reference_hash(table: dict[str, Any]) -> str:
    payload = {
        split: {
            "ref_row_id": table["splits"][split]["ref_row_id"].tolist(),
            "sim": np.round(table["splits"][split]["sim"], 7).tolist(),
            "eligible_count": table["splits"][split]["eligible_count"].tolist(),
            "same_smiles_excluded": table["splits"][split]["same_smiles_excluded"].tolist(),
        }
        for split in C.SPLITS
    }
    return C.sha256_json(payload)


def k_effective(table_split: dict[str, np.ndarray], k: int) -> np.ndarray:
    return np.minimum(table_split["eligible_count"], k).astype(np.int32)


def config_weights(
    table_split: dict[str, np.ndarray],
    k: int,
    mode: str,
    *,
    tau: float = C.SOFTMAX_TAU,
) -> np.ndarray:
    """Weight matrix ``(n_query, max_references)``; every row sums to 1.

    Uniform: ``1 / K_eff`` over the first ``K_eff`` retained references.
    Softmax: ``exp((s - max s) / tau)`` normalised over the *same* ``K_eff``
    selected references, so the row sum is exactly 1 by construction.
    """
    if mode not in C.WEIGHT_MODES:
        raise KeyError(f"unknown weight mode {mode!r}")
    sim = np.asarray(table_split["sim"], dtype=np.float64)
    if k > sim.shape[1]:
        raise ValueError(f"K={k} exceeds the frozen table width {sim.shape[1]}")
    eligible = np.asarray(table_split["eligible_count"], dtype=np.int64)
    weights = np.zeros_like(sim, dtype=np.float64)
    for query in range(sim.shape[0]):
        k_eff = int(min(k, eligible[query]))
        if k_eff <= 0:
            continue
        scores = sim[query, :k_eff]
        if mode == "uniform":
            weights[query, :k_eff] = 1.0 / float(k_eff)
        else:
            shifted = scores - float(scores.max())
            values = np.exp(shifted / float(tau))
            weights[query, :k_eff] = values / values.sum()
    row_sums = weights.sum(axis=1)
    usable = row_sums > 0
    # float64 normalisation: the row sums are 1 up to double-precision rounding
    # (exactly 1 for the frozen K in {4, 16, 64} with uniform weights).
    if not np.allclose(row_sums[usable], 1.0, atol=0.0, rtol=1e-12):
        raise AssertionError("reference weights do not sum to 1")
    return weights


def config_summary(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in C.SPLITS:
        entry = table["splits"][split]
        for config in C.CONFIGS:
            k_eff = k_effective(entry, int(config["K"]))
            rows.append(
                {
                    "dataset": table["dataset"],
                    "split": split,
                    "config": config["name"],
                    "K": int(config["K"]),
                    "weights": config["weights"],
                    "n_queries": int(k_eff.size),
                    "k_eff_min": int(k_eff.min()) if k_eff.size else 0,
                    "k_eff_mean": float(k_eff.mean()) if k_eff.size else 0.0,
                    "k_eff_max": int(k_eff.max()) if k_eff.size else 0,
                    "eligible_min": int(entry["eligible_count"].min()) if k_eff.size else 0,
                    "same_smiles_excluded_total": int(entry["same_smiles_excluded"].sum()),
                    "queries_with_zero_reference": int((entry["eligible_count"] == 0).sum()),
                }
            )
    return rows


def config_weights_by_name(table: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    entry = table["splits"][split]
    return {
        config["name"]: config_weights(entry, int(config["K"]), str(config["weights"]))
        for config in C.CONFIGS
    }


def save_reference_table(table: dict[str, Any], path) -> dict[str, Any]:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"dataset": np.asarray(table["dataset"]), "max_references": np.asarray(table["max_references"])}
    for split in C.SPLITS:
        for key, value in table["splits"][split].items():
            if isinstance(value, np.ndarray):
                payload[f"{split}__{key}"] = value
    np.savez_compressed(target, **payload)
    return {"path": str(target), "sha256": C.sha256_file(target), "bytes": int(target.stat().st_size)}


def load_reference_table(path, dataset: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as handle:
        keys = list(handle.files)
        arrays = {key: handle[key] for key in keys}
    splits: dict[str, dict[str, Any]] = {}
    for split in C.SPLITS:
        entry: dict[str, Any] = {}
        for key in keys:
            prefix = f"{split}__"
            if key.startswith(prefix):
                entry[key[len(prefix):]] = arrays[key]
        splits[split] = entry
    return {
        "dataset": dataset,
        "max_references": int(arrays["max_references"]),
        "similarity": "tanimoto on 2048 morgan radius-2 bits (label free)",
        "exclusion": "self global row_id + identical stripped SMILES in Train",
        "sort": "similarity descending, Train global row_id ascending",
        "splits": splits,
    }


def iter_reference_rows(table_split: dict[str, np.ndarray]) -> Iterable[tuple[int, int, int, float]]:
    """Yield ``(query_local, ref_local, ref_row_id, similarity)`` for retained rows."""
    ref_local = table_split["ref_local"]
    ref_row_id = table_split["ref_row_id"]
    sim = table_split["sim"]
    n_kept = table_split["n_kept"]
    for query in range(ref_local.shape[0]):
        for slot in range(int(n_kept[query])):
            yield query, int(ref_local[query, slot]), int(ref_row_id[query, slot]), float(sim[query, slot])
