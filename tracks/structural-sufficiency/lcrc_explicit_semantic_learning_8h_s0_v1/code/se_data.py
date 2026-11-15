"""Data views for the round: Lipo (official scaffold) and official ESOL.

Reuses the frozen ``lcrc_lipo_context_small_s0_v1`` structure caches and
``lipo_v1`` feature spec read-only.  Adds:

* nested label-budget subsets of the official Train rows, frozen by
  ``sha256('LCRC-semantic-budget-v1|' + global_row_id)``;
* a Fit-only target normaliser for budget runs (structure normalisation keeps
  its frozen all-official-Train form and is disclosed);
* an official ``ogbg-molesol`` view with the same label-free schema/codes and a
  Train-fitted target normaliser.

Nothing here selects on Valid/Test.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import se_common as C
import lp_common as LPC
import lp_data as LPD
import lp_source as LPS

OGB_MOLESOL = "ogbg-molesol"
MOLESOL_DIRNAME = "ogbg_molesol"
MOLESOL_SPLIT = {"train": 902, "valid": 113, "test": 113}
NESTED_BUDGET_PREFIX = "LCRC-semantic-budget-v1|"


# ---------------------------------------------------------------------------
# lipo
# ---------------------------------------------------------------------------


def nested_order(train_rows: Sequence[int]) -> list[int]:
    """Frozen nested subset order: sha256('LCRC-semantic-budget-v1|'+row_id)."""
    rows = [int(r) for r in train_rows]

    def key(r: int) -> str:
        return hashlib.sha256(f"{NESTED_BUDGET_PREFIX}{r}".encode("utf-8")).hexdigest()

    return sorted(rows, key=key)


def nested_fit_rows(train_rows: Sequence[int], fraction: float) -> list[int]:
    order = nested_order(train_rows)
    n = int(round(len(order) * fraction))
    if fraction >= 0.999:
        return order
    return order[:n]


class BudgetView:
    """Official split + a Fit subset of the official Train rows.

    ``fit_rows`` are official global row ids (members of the Train split).
    Caches are shared with the full view (they only depend on official split
    row ids).  ``y_fit`` are the Fit labels; the target normaliser uses them.
    """

    def __init__(self, base: LPD.View, fit_rows: Sequence[int], budget: str):
        self.base = base
        self.budget = budget
        self.fit_rows = [int(r) for r in fit_rows]
        self.rows = {k: np.asarray(base.rows[k], dtype=np.int64) for k in LPC.SPLITS}
        self.caches = base.caches
        self.spec = base.spec
        self.train_norm = base.train_norm
        # row id -> position in official train split
        self._train_pos = {int(r): i for i, r in enumerate(self.rows["train"].tolist())}
        self.fit_positions = np.asarray([self._train_pos[r] for r in self.fit_rows], dtype=np.int64)
        y = base.y["train"][self.fit_positions]
        mean = float(y.mean())
        scale = float(y.std())
        self.target_norm = {
            "mean": mean,
            "scale": scale,
            "source": f"Fit subset ({len(self.fit_rows)} official Train rows)",
            "valid_or_test_used": False,
            "n": int(y.size),
            "structure_normalisation": "frozen all-official-Train (label-free); disclosed",
        }
        self.y = {k: np.asarray(base.y[k], dtype=np.float64) for k in LPC.SPLITS}

    # convenience ------------------------------------------------------
    def y_scaled(self, split: str) -> torch.Tensor:
        y = self.y[split]
        return torch.as_tensor((y - self.target_norm["mean"]) / self.target_norm["scale"], dtype=torch.float32)

    def y_fit_scaled(self) -> torch.Tensor:
        y = self.y["train"][self.fit_positions]
        return torch.as_tensor((y - self.target_norm["mean"]) / self.target_norm["scale"], dtype=torch.float32)

    def to_raw(self, standardised: np.ndarray) -> np.ndarray:
        return np.asarray(standardised, dtype=np.float64) * float(self.target_norm["scale"]) + float(
            self.target_norm["mean"]
        )

    def audit(self) -> dict[str, Any]:
        out = self.base.audit()
        out["budget"] = {
            "name": self.budget,
            "n_fit": len(self.fit_rows),
            "fit_rows_sha256": LPC.row_sha256(self.fit_rows),
            "target_norm": dict(self.target_norm),
        }
        return out


def lipo_budget_view(budget: str, *, base: LPD.View | None = None) -> BudgetView:
    base = base or LPD.build_view()
    train_rows = base.rows["train"].tolist()
    if budget == "N100":
        fit = train_rows
    elif budget == "N50":
        fit = nested_fit_rows(train_rows, 0.5)
    elif budget == "N25":
        fit = nested_fit_rows(train_rows, 0.25)
    else:
        raise ValueError(budget)
    return BudgetView(base, fit, budget)


# ---------------------------------------------------------------------------
# official ESOL
# ---------------------------------------------------------------------------


def _esol_download_root() -> Any:
    return C.REPO / "data" / "ogb" / MOLESOL_DIRNAME


#: committed label-free ESOL artifacts (built once by ``se_esol_prep.py``)
ESOL_RAW = C.DATA / "esol_raw_v1.json.gz"
ESOL_LABELS = C.DATA / "esol_labels_v1.npz"
ESOL_SPLIT = C.DATA / "esol_split_v1.json"


def build_esol_view(*, rebuild: bool = False) -> LPD.View:
    """Official ogbg-molesol with the same lipo_v1 schema/codes.

    Structures come from the committed label-free artifact (built once locally
    with RDKit by ``se_esol_prep.py``); the GPU hosts have no RDKit.  The *only*
    thing re-fitted for this task is the Train target normaliser.
    """
    import gzip

    if not ESOL_RAW.is_file():
        raise FileNotFoundError(f"missing {C.track_rel(ESOL_RAW)}; run code/se_esol_prep.py locally first")
    with gzip.open(ESOL_RAW, "rt", encoding="utf-8") as fh:
        payload = json.loads(fh.read())
    if payload.get("label_free") is not True:
        raise PermissionError("ESOL raw artifact is not declared label-free")
    raw_rows = payload["rows"]
    split_payload = C.read_json(ESOL_SPLIT)
    labels = np.load(ESOL_LABELS, allow_pickle=False)
    y_all = np.asarray(labels["y"], dtype=np.float64)
    rows_by_split = {k: [int(x) for x in split_payload["rows"][k]] for k in LPC.SPLITS}
    sizes = {k: len(v) for k, v in rows_by_split.items()}
    if sizes != MOLESOL_SPLIT:
        raise ValueError(f"molesol split sizes {sizes} != official {MOLESOL_SPLIT}")

    import sys

    for p in (str(C.PREP_CODE),):
        if p not in sys.path:
            sys.path.insert(0, p)
    import adapters as PREP_ADAPTERS
    import lcrc_common as PREP_COMMON
    import lcrc_data as PREP_DATA

    spec = LPD.frozen_spec()
    graphs_all = {}
    for row in sorted({r for v in rows_by_split.values() for r in v}):
        graphs_all[row] = PREP_ADAPTERS.lipo_row_to_graph(raw_rows[row], int(row), spec)

    caches: dict[str, Any] = {}
    rows_out: dict[str, np.ndarray] = {}
    for split in LPC.SPLITS:
        rws = rows_by_split[split]
        graphs = [graphs_all[r] for r in rws]
        cache = PREP_DATA.build_cache(graphs, spec, split=f"{OGB_MOLESOL}_{split}")
        cache["rows"] = np.asarray(rws, dtype=np.int64)
        cache["dataset"] = OGB_MOLESOL
        cache["split_name"] = split
        cache["cache_version"] = "lcrc-esol-view-v1"
        cache["meta"]["labels_present"] = False
        cache["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(cache)
        PREP_DATA.validate_cache(cache, spec)
        caches[split] = cache
        rows_out[split] = np.asarray(rws, dtype=np.int64)

    train_norm = PREP_DATA.fit_norm(caches["train"], spec, source=f"{OGB_MOLESOL}_official_train_structures_only")
    y = {k: y_all[rows_out[k]] for k in LPC.SPLITS}
    target_norm = LPD.target_normaliser(y["train"])
    return LPD.View(spec=spec, caches=caches, train_norm=train_norm, target_norm=target_norm, rows=rows_out, y=y)


def _esol_split(root) -> dict[str, list[int]]:
    import gzip

    out: dict[str, list[int]] = {}
    for split in LPC.SPLITS:
        path = root / "split" / "scaffold" / f"{split}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            out[split] = [int(float(line.strip())) for line in fh if line.strip()]
    sizes = {k: len(v) for k, v in out.items()}
    if sizes != MOLESOL_SPLIT:
        raise ValueError(f"molesol split sizes {sizes} != official {MOLESOL_SPLIT}")
    return out


# ---------------------------------------------------------------------------
# unseen/support statistics (real model codes, Train molecule df)
# ---------------------------------------------------------------------------


def code_support(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Per-molecule document frequency of path/nbr/end/joint codes in one cache."""
    from collections import Counter

    path_ptr = np.asarray(cache["path_ptr"])
    n = int(cache["n_molecules"])
    path_cat = np.asarray(cache["path_cat"])
    end63 = np.asarray(cache["path_end63"])
    nbr_code = np.asarray(cache["nbr_code"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_df: Counter = Counter()
    joint_df: Counter = Counter()
    nbr_df: Counter = Counter()
    end_df: Counter = Counter()
    for m in range(n):
        pcs = set(int(x) for x in path_cat[path_ptr[m]:path_ptr[m + 1]])
        es = set(int(x) for x in end63[path_ptr[m]:path_ptr[m + 1]])
        path_df.update(pcs)
        end_df.update(es)
        joint_df.update({(int(p), int(e)) for p, e in zip(path_cat[path_ptr[m]:path_ptr[m + 1]], end63[path_ptr[m]:path_ptr[m + 1]])})
        nbr_df.update(set(int(x) for x in nbr_code[nbr_ptr[m]:nbr_ptr[m + 1]]))
    return {"path_df": path_df, "nbr_df": nbr_df, "end_df": end_df, "joint_df": joint_df}


__all__ = [name for name in dir() if not name.startswith("_")]
