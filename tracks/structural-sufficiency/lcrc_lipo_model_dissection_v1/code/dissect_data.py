"""Structure caches, train-only normalisation and prediction helpers.

The structure pipeline is the frozen ``lcrc_cross_task_preparation_v1`` one:

    frozen raw attributes -> MolecularGraph -> build_molecule -> build_cache

with the frozen ``lipo_v1`` FeatureSpec.  Only the cache filenames are new
(one structure cache per public repeat and split).  Every cache is label-free:
targets live in a separate array that is loaded only for the split named by the
caller.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import dissect_common as C  # noqa: I001 (adds frozen prep/pilot code dirs to sys.path)
import dissect_source as S

import mb_raw_artifact as RAW
import mb_rdkit as R

import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec

CACHE_VERSION = "lcrc-lipo-dissection-v1"


# ---------------------------------------------------------------------------
# feature spec
# ---------------------------------------------------------------------------


def frozen_spec() -> FeatureSpec:
    """Load the frozen ``lipo_v1`` spec and prove the code reproduces its hash."""
    import mb_data as D  # frozen loader (hashes the committed JSON)

    spec = D.frozen_spec()
    if spec.hash() != C.FEATURE_SPEC_HASH:
        raise ValueError(f"lipo_v1 spec hash {spec.hash()} != frozen {C.FEATURE_SPEC_HASH}")
    if (spec.atom_feat_dim, spec.bond_feat_dim, spec.n_scalars, spec.global_dim) != (
        C.ATOM_FEAT_DIM,
        C.BOND_FEAT_DIM,
        C.ATOM_SCALAR_DIM,
        C.GLOBAL_DIM,
    ):
        raise ValueError("frozen lipo_v1 input widths changed")
    if (spec.n_nbr_codes, spec.n_path_codes, spec.n_bond_types) != (
        C.N_NBR_CODES,
        C.N_PATH_CODES,
        C.N_BOND_TYPES,
    ):
        raise ValueError("frozen lipo_v1 vocabulary changed")
    return spec


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------


def cache_path(repeat: int, split: str) -> Path:
    return C.CACHE / f"Lipophilicity_r{int(repeat)}_{split}.pt"


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_split_cache(
    repeat: int,
    split: str,
    rows: Sequence[int],
    smiles: Sequence[str],
    spec: FeatureSpec,
    *,
    save: bool = True,
) -> dict[str, Any]:
    if split not in ("train", "valid", "test"):
        raise ValueError("split must be train/valid/test")
    rows = [int(r) for r in rows]
    payload = RAW.load(verify=True)
    all_rows = payload["rows"]["Lipophilicity"]
    raw_rows = []
    for row in rows:
        item = all_rows[row]
        if int(item["mol_index"]) != row:
            raise ValueError(f"raw artifact row {row} has mol_index {item['mol_index']}")
        if str(item["smiles"]) != str(smiles[row]):
            raise ValueError(f"repeat {repeat}/{split}: artifact SMILES for row {row} differs from the CSV")
        raw_rows.append(item)
    graphs = [R.raw_row_to_graph(raw, int(raw["mol_index"]), spec) for raw in raw_rows]
    cache = PREP_DATA.build_cache(graphs, spec, split=f"Lipophilicity_r{int(repeat)}_{split}")
    cache["rows"] = np.asarray(rows, dtype=np.int64)
    cache["dataset"] = "Lipophilicity"
    cache["split_name"] = split
    cache["repeat"] = int(repeat)
    cache["cache_version"] = CACHE_VERSION
    cache["source_commit"] = C.CHEMBENCH_COMMIT
    cache["meta"]["labels_present"] = False
    cache["meta"]["row_ids"] = "global 0-based ChemBench CSV row index"
    PREP_DATA.validate_cache(cache, spec)
    if cache["molecule_ids"] != [str(r) for r in rows]:
        raise ValueError(f"repeat {repeat}/{split}: molecule_ids do not match source row ids")
    cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
    if save:
        _atomic_save(cache, cache_path(repeat, split))
    return cache


def load_or_build_split_cache(
    repeat: int,
    split: str,
    rows: Sequence[int],
    smiles: Sequence[str],
    spec: FeatureSpec,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    path = cache_path(repeat, split)
    if path.exists() and not rebuild:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{path}: unexpected cache version {cache.get('cache_version')!r}")
        if int(cache.get("repeat", -1)) != int(repeat):
            raise ValueError(f"{path}: cached repeat differs")
        if list(np.asarray(cache["rows"]).tolist()) != [int(r) for r in rows]:
            raise ValueError(f"{path}: cached row ids differ from the published split")
        if cache["meta"].get("feature_spec_hash") != spec.hash():
            raise ValueError(f"{path}: cached feature spec hash differs")
        if cache["meta"].get("labels_present"):
            raise ValueError(f"{path}: structure cache must be label-free")
        return cache
    return build_split_cache(repeat, split, rows, smiles, spec)


# ---------------------------------------------------------------------------
# normalisation (Train structures only)
# ---------------------------------------------------------------------------


def target_normaliser(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    scale = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate train target distribution")
    return {
        "mean": mean,
        "scale": scale,
        "n": int(y.size),
        "source": "train_split_only",
        "valid_or_test_used": False,
        "formula": "z = (y - mean) / scale; y_hat_raw = mean + scale * z_hat",
    }


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


class View:
    def __init__(
        self,
        *,
        repeat: int,
        spec: FeatureSpec,
        train_cache: Mapping[str, Any],
        valid_cache: Mapping[str, Any],
        y_train: np.ndarray,
        y_valid: np.ndarray,
        train_norm: Mapping[str, Any],
        target_norm: Mapping[str, Any],
        rows: Mapping[str, Sequence[int]],
        source_meta: Mapping[str, Any],
    ) -> None:
        self.repeat = int(repeat)
        self.spec = spec
        self.train_cache = train_cache
        self.valid_cache = valid_cache
        self.y_train = np.asarray(y_train, dtype=np.float64)
        self.y_valid = np.asarray(y_valid, dtype=np.float64)
        self.train_norm = train_norm
        self.target_norm = target_norm
        self.rows = {k: list(v) for k, v in rows.items()}
        self.source_meta = dict(source_meta)

    def audit(self) -> dict[str, Any]:
        return {
            "repeat": self.repeat,
            "n_train": int(self.train_cache["n_molecules"]),
            "n_valid": int(self.valid_cache["n_molecules"]),
            "train_cache_fingerprint": self.train_cache["meta"]["content_fingerprint"],
            "valid_cache_fingerprint": self.valid_cache["meta"]["content_fingerprint"],
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(self.train_norm),
            "target_norm": dict(self.target_norm),
            "rows": {
                "train": {"n": len(self.rows["train"]), "sha256": C.row_sha256(self.rows["train"])},
                "valid": {"n": len(self.rows["valid"]), "sha256": C.row_sha256(self.rows["valid"])},
                "test": {"n": len(self.rows["test"]), "sha256": C.row_sha256(self.rows["test"])},
            },
            "feature_spec_hash": self.spec.hash(),
            "labels_in_structure_cache": False,
            "test_targets_loaded": False,
            "test_structure_cache_built": False,
            "source": self.source_meta,
        }


def build_view(repeat: int = C.MAIN_REPEAT, *, rebuild: bool = False) -> View:
    """Train+Valid view.  The Test target column is never decoded here."""
    spec = frozen_spec()
    src = S.load_lipop(repeat, labels="train_valid")
    rows = src["rows"]
    train_cache = load_or_build_split_cache(
        repeat, "train", rows["train"], src["smiles"], spec, rebuild=rebuild
    )
    valid_cache = load_or_build_split_cache(
        repeat, "valid", rows["valid"], src["smiles"], spec, rebuild=rebuild
    )
    # The frozen ``collate`` adds offsets from bond_ptr / atom_ptr, so the train
    # cache must not be paired with a different valid cache.  Fingerprints are
    # recorded per cache and re-checked at load time.
    if list(np.asarray(train_cache["rows"]).tolist()) != rows["train"]:
        raise ValueError("train cache rows != published split")
    if list(np.asarray(valid_cache["rows"]).tolist()) != rows["valid"]:
        raise ValueError("valid cache rows != published split")
    train_norm = PREP_DATA.fit_norm(train_cache, spec, source=f"Lipophilicity_r{repeat}_train_structures_only")
    if train_norm.get("valid_or_test_used") is not False:
        raise ValueError("train normalisation must not use Valid/Test")
    return View(
        repeat=int(repeat),
        spec=spec,
        train_cache=train_cache,
        valid_cache=valid_cache,
        y_train=src["y"]["train"],
        y_valid=src["y"]["valid"],
        train_norm=train_norm,
        target_norm=target_normaliser(src["y"]["train"]),
        rows=rows,
        source_meta={
            "source_commit": src["source_commit"],
            "csv_sha256": src["csv_sha256"],
            "split_file": src["split_file"],
            "split_file_sha256": src["split_file_sha256"],
            "split_rows_sha256": src["split_rows_sha256"],
            "split_sizes": src["sizes"],
            "label_column": src["label_column"],
            "units": src["units"],
        },
    )


def build_test_cache(repeat: int, *, rebuild: bool = False) -> dict[str, Any]:
    """Label-free Test structure cache (no target byte decoded)."""
    spec = frozen_spec()
    src = S.load_lipop(repeat, labels="none")
    rows = src["rows"]["test"]
    return load_or_build_split_cache(repeat, "test", rows, src["smiles"], spec, rebuild=rebuild)


def load_test(repeat: int, *, rebuild: bool = False) -> tuple[dict[str, Any], np.ndarray, list[str], list[int]]:
    """Test structure cache + targets + SMILES + row ids.  Frozen scoring only."""
    spec = frozen_spec()
    cache = build_test_cache(repeat, rebuild=rebuild)
    src = S.load_lipop(repeat, labels="all")
    rows = src["rows"]["test"]
    if [int(x) for x in np.asarray(cache["rows"]).tolist()] != [int(x) for x in rows]:
        raise ValueError("test cache row ids differ from the frozen published split")
    smiles = [src["smiles"][r] for r in rows]
    return cache, np.asarray(src["y"]["test"], dtype=np.float64), smiles, rows


# ---------------------------------------------------------------------------
# batching / prediction
# ---------------------------------------------------------------------------


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_scores(model, cache: Mapping[str, Any], device, *, batch_size: int = 256) -> np.ndarray:
    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = to_device(PREP_DATA.collate(cache, rows), device)
            pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def predict_batch(model, cache: Mapping[str, Any], rows: Sequence[int], device) -> np.ndarray:
    batch = to_device(PREP_DATA.collate(cache, list(rows)), device)
    with torch.no_grad():
        return model(batch).detach().float().cpu().numpy().reshape(-1)


__all__ = [
    "CACHE_VERSION",
    "frozen_spec",
    "cache_path",
    "build_split_cache",
    "load_or_build_split_cache",
    "target_normaliser",
    "View",
    "build_view",
    "build_test_cache",
    "load_test",
    "to_device",
    "predict_scores",
    "predict_batch",
]
