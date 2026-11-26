"""The fragment-conditional response models (RESP family), reused unchanged.

Two operators, one parameter layout (198,262 parameters each), one shared
zero-``w`` initial state::

    PRODUCT     T = sum_j w_j q_j k_j / sqrt(16),        q = tanh(U z_S), k = tanh(V z_C)
    FUSION_MLP  T = sum_j w_j tanh((U z_S + V z_C)_j / sqrt(2)) / sqrt(16)
    s(C,S)      = b(C) + a(S) + T,     F(G) = mean over the frozen <=4 selected views
                                       (original fallback head when no legal view exists)

``PRODUCT`` is the historical ``RESP`` arm and ``FUSION_MLP`` the historical
``JOINT`` arm of ``response_dictionary_lipo_followup_s0_v1``; both classes are
imported, not re-implemented.  Only the row subset is new: this round trains on
an arbitrary set of official row ids, with a target scaler fitted on that set.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sr_common as C

for _p in (str(C.RD_SMALL_CODE), str(C.RD_FOLLOW_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rd_common as RC  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402
import fu_model as FM  # noqa: E402

MODE = {"PRODUCT": "RESP", "FUSION_MLP": "JOINT"}
#: Frozen architecture of ``response_dictionary_small_s0_v1`` (never searched).
ARCH = {"hidden": 96, "latent": 16, "n_layers": 3, "dropout": 0.10}
EXPECTED_PARAMETERS = 198262
EXPECTED_INIT_HASH = "ccc8171d4cdc47daadd226c382947a0e3e62f6c9266dcad6f952ba98a989072c"


def load_fragment_cache(dataset: str) -> dict[str, Any]:
    expected = (C.EXPECTED_SHA["lipo_frag_cache"] if dataset == "ogbg-mollipo"
                else C.EXPECTED_SHA["esol_frag_cache"])
    got = C.sha256_file(C.DATASET_FRAG[dataset])
    if got != expected:
        raise RuntimeError(f"{dataset} fragment cache sha256 {got} != {expected}")
    path = C.DATASET_FRAG[dataset]
    with np.load(path, allow_pickle=True) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["_sha256"] = got
    payload["_path"] = str(path.relative_to(C.REPO_ROOT))
    return payload


@dataclass
class Subset:
    """One molecule subset expressed in official row ids."""

    name: str
    row_id: np.ndarray
    label_raw: np.ndarray
    label_std: np.ndarray
    split: RT.Split

    def local(self, rows: Sequence[int]) -> np.ndarray:
        """Official row ids -> positions inside this subset (must be a subset)."""
        rows = np.asarray(rows, dtype=np.int64)
        pos = np.searchsorted(self.row_id, rows)
        if not np.array_equal(self.row_id[pos], rows):
            raise RuntimeError("requested rows are not part of this RESP subset")
        return pos


def subset_split(payload: Mapping[str, Any], split_name: str, rows: Sequence[int],
                 scaler: Mapping[str, float]) -> Subset:
    rows = np.asarray(sorted(int(r) for r in rows), dtype=np.int64)
    all_rows = np.asarray(payload[f"{split_name}_mol_row_id"], dtype=np.int64)
    pos = np.searchsorted(all_rows, rows)
    if not np.array_equal(all_rows[pos], rows):
        raise RuntimeError(f"rows not all present in the {split_name} fragment split")
    mol_ptr_all = np.asarray(payload[f"{split_name}_mol_view_ptr"], dtype=np.int64)
    view_s = np.asarray(payload[f"{split_name}_view_s"], dtype=np.int64)
    view_c = np.asarray(payload[f"{split_name}_view_c"], dtype=np.int64)
    fallback = np.asarray(payload[f"{split_name}_mol_fallback"], dtype=np.int64)
    labels = np.asarray(payload[f"{split_name}_mol_label_raw"], dtype=np.float64)[pos]

    mol_ptr = np.zeros(len(pos) + 1, dtype=np.int64)
    s_out, c_out = [], []
    fb_out = np.full(len(pos), -1, dtype=np.int64)
    for j, p in enumerate(pos.tolist()):
        a, b = int(mol_ptr_all[p]), int(mol_ptr_all[p + 1])
        mol_ptr[j + 1] = mol_ptr[j] + (b - a)
        if a == b:
            fb_out[j] = int(fallback[p])
        else:
            s_out.append(view_s[a:b])
            c_out.append(view_c[a:b])
    view_s_out = np.concatenate(s_out) if s_out else np.zeros(0, dtype=np.int64)
    view_c_out = np.concatenate(c_out) if c_out else np.zeros(0, dtype=np.int64)
    split = RT.Split(
        row_id=rows,
        label_raw=labels,
        label_std=(labels - float(scaler["mean"])) / float(scaler["std"]),
        mol_ptr=mol_ptr,
        view_s=view_s_out,
        view_c=view_c_out,
        fallback=fb_out,
    )
    return Subset(split_name, rows, labels, split.label_std, split)


def canonical_init(dims: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The committed seed-0 initial state (hash-pinned by the parent round)."""
    path = C.RD_SMALL_DIR / "data" / "canonical_init_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    got = RM.state_hash(state)
    if got != EXPECTED_INIT_HASH:
        raise RuntimeError(f"fragment-model init hash {got} != {EXPECTED_INIT_HASH}")
    return state


class RespBundle:
    """Everything a RESP arm needs, pinned to one arbitrary molecule subset."""

    def __init__(self, dataset: str, fit_rows: Sequence[int], scaler: Mapping[str, float],
                 split_name: str = "train") -> None:
        self.dataset = dataset
        self.payload = load_fragment_cache(dataset)
        self.dims = RM.ogb_category_dims()
        self.frag_data = RM.fragment_data_list(self.payload)
        self.scaler = {"mean": float(scaler["mean"]), "std": float(scaler["std"])}
        self.split_name = split_name
        self.fit = subset_split(self.payload, split_name, fit_rows, scaler)

    def subset(self, rows: Sequence[int], name: str) -> Subset:
        return subset_split(self.payload, self.split_name, rows, self.scaler)

    def subset_of(self, split_name: str, rows: Sequence[int], name: str) -> Subset:
        return subset_split(self.payload, split_name, rows, self.scaler)


def build_model(dims, init_state, operator: str,
                config: Mapping[str, Any] | None = None) -> torch.nn.Module:
    mode = MODE[operator]
    model_config = FM.model_config(dict(ARCH), mode)
    model = FM.build_model(dims, model_config, init_state)
    n = RM.n_parameters(model)
    if n != EXPECTED_PARAMETERS:
        raise RuntimeError(f"{operator} has {n} parameters, expected {EXPECTED_PARAMETERS}")
    return model


# ---------------------------------------------------------------------------
# differentiable prediction on an arbitrary molecule subset
# ---------------------------------------------------------------------------


def predict_tensor(model, bundle: RespBundle, subset: Subset,
                   mol_indices: np.ndarray, device: Any) -> torch.Tensor:
    """Differentiable ``F(G)`` in standardised units for the given molecules."""
    unique_ids: list[int] = []
    unique_pos: dict[int, int] = {}
    view_local_s: list[int] = []
    view_local_c: list[int] = []
    view_mol_local: list[int] = []
    fb_local = np.full(len(mol_indices), -1, dtype=np.int64)
    split = subset.split
    for j, mol in enumerate(np.asarray(mol_indices, dtype=np.int64).tolist()):
        a, b = int(split.mol_ptr[mol]), int(split.mol_ptr[mol + 1])
        if a == b:
            frag = int(split.fallback[mol])
            if frag < 0:
                raise ValueError(f"molecule {mol} has neither views nor a fallback fragment")
            if frag not in unique_pos:
                unique_pos[frag] = len(unique_ids)
                unique_ids.append(frag)
            fb_local[j] = unique_pos[frag]
            continue
        for v in range(a, b):
            for sid in (int(split.view_s[v]), int(split.view_c[v])):
                if sid not in unique_pos:
                    unique_pos[sid] = len(unique_ids)
                    unique_ids.append(sid)
            view_local_s.append(unique_pos[int(split.view_s[v])])
            view_local_c.append(unique_pos[int(split.view_c[v])])
            view_mol_local.append(j)

    batch = RT.fragment_batch(bundle.frag_data, unique_ids).to(device)
    z = model.encoder(batch)
    out = torch.zeros(len(mol_indices), device=device, dtype=z.dtype)
    counts = np.zeros(len(mol_indices), dtype=np.float64)
    if view_local_s:
        z_s = z[torch.as_tensor(view_local_s, device=device)]
        z_c = z[torch.as_tensor(view_local_c, device=device)]
        scores = model.view_scores(z_s, z_c)
        view_mol = torch.as_tensor(view_mol_local, device=device, dtype=torch.long)
        sums = torch.zeros(len(mol_indices), device=device, dtype=scores.dtype).index_add(0, view_mol, scores)
        cnt = torch.zeros(len(mol_indices), device=device, dtype=scores.dtype).index_add(
            0, view_mol, torch.ones_like(scores))
        counts = cnt.detach().cpu().numpy()
        out = out + sums / cnt.clamp(min=1.0)
    fb_mask = fb_local >= 0
    if bool(fb_mask.any()):
        idx = torch.as_tensor(fb_local[fb_mask], device=device, dtype=torch.long)
        rows = torch.as_tensor(np.nonzero(fb_mask)[0], device=device, dtype=torch.long)
        out = out.index_copy(0, rows, model.fallback_scores(z[idx]))
    return out


@torch.no_grad()
def predict_std(model, bundle: RespBundle, subset: Subset, rows: np.ndarray,
                device: Any, batch_size: int = 256) -> np.ndarray:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros(rows.size, dtype=np.float64)
    start = 0
    while start < rows.size:
        idx = np.arange(start, min(rows.size, start + int(batch_size)))
        out[idx] = predict_tensor(model, bundle, subset, idx, device).double().cpu().numpy()
        start += idx.size
    return out


def base_loss(model, bundle: RespBundle, subset: Subset, rows: np.ndarray,
              device: Any) -> torch.Tensor:
    """The frozen per-molecule view-mean MSE on standardised labels."""
    return RT.molecule_loss(model, _DatasetView(bundle), subset.split,
                            np.asarray(rows, dtype=np.int64), device)


class _DatasetView:
    """Minimal stand-in exposing the attributes ``rd_train.molecule_loss`` uses."""

    def __init__(self, bundle: RespBundle) -> None:
        self.frag_data = bundle.frag_data
        self.dims = bundle.dims
        self.scaler = bundle.scaler
