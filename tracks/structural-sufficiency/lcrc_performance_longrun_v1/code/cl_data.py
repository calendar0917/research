"""Data views for the performance long-run.

The frozen graph path is reused byte-for-byte.  Two kinds of view are exposed:

``base``
    exactly the Sprint-02 ``lipo_v1`` view (train + valid), with the frozen
    Train-only standardiser.  Used by BASE / RHOSHAPE / MULTICONTRIB8 so their
    input tensors are identical to the historical baseline.

``chem8``
    the same caches with the eight committed CHEMLOCAL8 channels appended to
    ``atom_feat``.  The base columns are untouched; the standardiser is
    re-fitted (and required to be identical, because scalars/globals did not
    change).

Test caches follow the same rule.  Test labels are read only by the training
driver after the best-Validation checkpoint has been frozen.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import cl_common as C  # noqa: I001

import mb2_data as D
import mb2_metrics as MET
import lcrc_data as PREP_DATA
import ta_data as TD

import cl_chem8 as CH


def _source_meta(view) -> dict[str, Any]:
    if hasattr(view, "source_meta"):
        return dict(view.source_meta)
    if hasattr(view, "split_meta"):
        return dict(view.split_meta)
    return {}


def base_view(dataset: str, split: str = "published"):
    """Load the frozen train+valid view for one split (base ``lipo_v1``)."""
    if split == "published":
        return D.build_view(dataset)
    if split.startswith("lipo-repeat:"):
        if dataset != "Lipophilicity":
            raise ValueError("lipo-repeat splits are Lipophilicity only")
        suffix = int(split.split(":", 1)[1])
        if suffix not in (123, 124, 122):
            raise ValueError("lipo repeat suffix must be 122/123/124")
        frozen = TD.lipo_repeat_split(suffix)
        rows = frozen["rows"]
        tv = TD.build_view_rows(
            dataset,
            {"train": rows["train"], "valid": rows["valid"]},
            tag=f"lipo_r{suffix}",
            split_meta={
                "kind": "published_repeat",
                "suffix": suffix,
                "file": frozen["file"],
                "sha256": frozen["sha256"],
                "rows_sha256": frozen["rows_sha256"],
            },
        )
        return D.DatasetView(
            dataset=dataset,
            spec=tv.spec,
            train_cache=tv.train_cache,
            valid_cache=tv.valid_cache,
            targets=tv.targets,
            train_norm=tv.train_norm,
            target_norm=tv.target_norm,
            rows={"train": tv.rows["train"], "valid": tv.rows["valid"]},
            label_columns=tv.label_columns,
            units=tv.units,
            source_meta={**dict(tv.split_meta), "cache_meta": tv.cache_meta},
        )
    raise ValueError(f"unknown split {split!r}")


def extend_view(view, dataset: str):
    """Return a chem8 view over the exact same rows and caches."""
    spec = CH.chem8_spec(view.spec)
    train_cache = CH.extend_cache(view.train_cache, spec, dataset=dataset)
    valid_cache = CH.extend_cache(view.valid_cache, spec, dataset=dataset)
    norm = PREP_DATA.fit_norm(train_cache, spec, source=str(view.train_norm.get("source")))
    base_fp = PREP_DATA.norm_fingerprint(view.train_norm)
    ext_fp = PREP_DATA.norm_fingerprint(norm)
    if base_fp != ext_fp:
        raise ValueError(
            f"{dataset}: chem8 changed the Train-only standardiser fingerprint "
            f"({ext_fp} != {base_fp}); the extension must not touch scalars/globals"
        )
    src = _source_meta(view)
    src = {
        **src,
        "chem8": {
            "channel_names": list(CH.CHANNEL_NAMES),
            "artifact_manifest": C.track_rel(C.CHEM8_MANIFEST),
            "base_spec_hash": view.spec.hash(),
            "extended_spec_hash": spec.hash(),
            "train_cache_base_fingerprint": view.train_cache["meta"].get("content_fingerprint"),
            "train_cache_extended_fingerprint": train_cache["meta"].get("content_fingerprint"),
            "valid_cache_base_fingerprint": view.valid_cache["meta"].get("content_fingerprint"),
            "valid_cache_extended_fingerprint": valid_cache["meta"].get("content_fingerprint"),
            "train_norm_fingerprint_unchanged": True,
        },
    }
    return D.DatasetView(
        dataset=dataset,
        spec=spec,
        train_cache=train_cache,
        valid_cache=valid_cache,
        targets=view.targets,
        train_norm=norm,
        target_norm=view.target_norm,
        rows=dict(view.rows),
        label_columns=view.label_columns,
        units=view.units,
        source_meta=src,
    )


def view_for_arm(dataset: str, split: str, arm: str):
    """Base view for readout-only arms; chem8 view for local-chemistry arms.

    Returns ``(view, base_spec, uses_chem8)`` so the model builder can copy the
    BASE tensors from the *un-extended* atom width.
    """
    base = base_view(dataset, split)
    if arm in ("chem8", "combo_chem_contrib", "combo_chem_rho"):
        return extend_view(base, dataset), base.spec, True
    if arm in ("base", "rhoshape", "multicontrib", "combo_rho_contrib"):
        return base, base.spec, False
    raise ValueError(f"unknown arm {arm!r}")


def test_view(dataset: str, split: str = "published"):
    """Label-free test cache plus its labels, for one explicit split.

    Returns ``(cache, y_test, rows, base_view)``.  The caller decides when this
    is called; the training driver calls it only after the best-Validation
    checkpoint has been captured.
    """
    if split == "published":
        base = D.build_view(dataset)
        cache = D.build_test_structure_cache(dataset)
        cache, y, rows = D.load_test_cache(dataset, cache=cache)
        return cache, np.asarray(y, dtype=np.float64), [int(r) for r in rows], base
    if split.startswith("lipo-repeat:"):
        suffix = int(split.split(":", 1)[1])
        frozen = TD.lipo_repeat_split(suffix)
        base = base_view(dataset, split)
        cache, y, rows = TD.build_test_view(
            dataset,
            frozen["rows"]["test"],
            tag=f"lipo_r{suffix}",
            split_meta={"kind": "published_repeat", "suffix": suffix},
        )
        if [int(r) for r in rows] != [int(r) for r in frozen["rows"]["test"]]:
            raise ValueError("test rows differ from the frozen published repeat")
        return cache, np.asarray(y, dtype=np.float64), [int(r) for r in rows], base
    raise ValueError(f"unknown split {split!r}")


def extend_test_cache(cache: Mapping[str, Any], dataset: str, spec):
    return CH.extend_cache(cache, spec, dataset=dataset)


def evaluate_predictions(
    task: str,
    raw: np.ndarray,
    y: np.ndarray,
    target_norm: Mapping[str, Any] | None,
    task_names: Sequence[str],
) -> dict[str, Any]:
    return MET.task_metrics(task, raw, y, target_norm, list(task_names))


def load_checkpoint_payload(path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


__all__ = [
    "base_view",
    "extend_view",
    "view_for_arm",
    "test_view",
    "extend_test_cache",
    "evaluate_predictions",
    "load_checkpoint_payload",
]
