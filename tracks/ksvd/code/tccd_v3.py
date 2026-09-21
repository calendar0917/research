#!/usr/bin/env python
"""TCCD-v3 strong-local bridge library.

This module keeps the TCCD-v2 prototype/composition machinery fixed while
replacing only the weak 714-D local representation with a frozen B-full
SharedStructuralPatchEncoder output plus a matched 16 -> 64 linear adapter.
Official ZINC test is never loaded.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v1 as V1
from tracks.ksvd.code import tccd_v2 as V2
from tracks.ksvd.code.run_tccd_v0 import load_or_build_records, records_cache_path
from tracks.ksvd.experiments.luyin16 import zinc_shared_structural_patch_encoder as SSPE

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_VERSION = "tccd_v3"
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/tccd_v3"
CACHE_DIR = RESULTS_DIR / "cache"

STRONG_LOCAL_WIDTH = 16
D_LOCAL = 64
K_PROTO = 64
N_REL = V2.N_REL
TEMP_MIN = V2.TEMP_MIN
TEMP_SPAN = V2.TEMP_SPAN
TEMP_INIT = V2.TEMP_INIT
EPS = V2.EPS
PROTO_INIT_SEED = V2.PROTO_INIT_SEED

TCCD_V2_PROTO_BEST = 0.287337
TCCD_V2_PROTO_SOUP = 0.261988
TCCD_V2_SOUP_REFERENCE = 0.261988
CANONICAL_GPU1_BASELINE = 0.119818
STRONG_REFERENCE_SEED0_SOUP = 0.11981802638241788
STRONG_REFERENCE_2SEED_SOUP = 0.11897220489243046
STRONG_SOURCE_COMMIT = "0aa71c81e845fd334c8ecd2bf7904c149277c289"
STRONG_CHECKPOINT = SSPE.RESULTS_DIR / "soup_states/sspe_seed0_top5_soup.pt"
STRONG_GRAPH_CACHE = SSPE.CACHE_DIR
STRONG_CACHE_SCHEMA = SSPE.CACHE_SCHEMA_VERSION

LOCAL_PASS_TOL = 1.0e-6
PERM_PASS_TOL = 1.0e-5
BATCH_PASS_TOL = 1.0e-5
DETERMINISM_PASS_TOL = 0.0

L1_MAX = 0.20
L1_DELTA = 0.07
L2_MAX = 0.24
L2_DELTA = 0.04
PROTO_PRESERVE = 0.015
PROTO_HARMFUL = 0.03
COMP_MIN = 0.01


@dataclass(frozen=True)
class StructExample:
    """One graph's concatenated rooted-patch structural tensors."""

    graph_id: int
    atom: np.ndarray
    root: np.ndarray
    dist: np.ndarray
    patch: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    bond: np.ndarray
    n_patches: int
    n_nodes: int
    n_edges: int
    exterior_signature: str = ""


class StructBatch:
    """Minimal tensor container consumed by SharedStructuralPatchEncoder."""

    def __init__(
        self,
        *,
        struct_atom: torch.Tensor,
        struct_root: torch.Tensor,
        struct_dist: torch.Tensor,
        struct_patch: torch.Tensor,
        struct_src: torch.Tensor,
        struct_dst: torch.Tensor,
        struct_bond: torch.Tensor,
        patch_counts: list[int],
        graph_ids: list[int],
    ) -> None:
        self.struct_atom = struct_atom
        self.struct_root = struct_root
        self.struct_dist = struct_dist
        self.struct_patch = struct_patch
        self.struct_src = struct_src
        self.struct_dst = struct_dst
        self.struct_bond = struct_bond
        self.patch_counts = list(patch_counts)
        self.graph_ids = list(graph_ids)

    def to(self, device: str | torch.device) -> "StructBatch":
        device = torch.device(device)
        return StructBatch(
            struct_atom=self.struct_atom.to(device),
            struct_root=self.struct_root.to(device),
            struct_dist=self.struct_dist.to(device),
            struct_patch=self.struct_patch.to(device),
            struct_src=self.struct_src.to(device),
            struct_dst=self.struct_dst.to(device),
            struct_bond=self.struct_bond.to(device),
            patch_counts=self.patch_counts,
            graph_ids=self.graph_ids,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def initial_temperature_logit() -> float:
    q = (TEMP_INIT - TEMP_MIN) / TEMP_SPAN
    return float(math.log(q / (1.0 - q)))


def _graphs_for_split(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_bundle, valid_bundle, meta = SSPE.extract_records()
    if split == "train":
        return list(train_bundle["graphs"]), dict(meta)
    if split == "valid":
        return list(valid_bundle["graphs"]), dict(meta)
    raise ValueError("official test is forbidden")


def struct_example_from_plain(graph: Mapping[str, Any] | Any, graph_id: int, *, exterior_signature: str = "") -> StructExample:
    def field(name: str) -> Any:
        if isinstance(graph, Mapping):
            return graph[name]
        return getattr(graph, name)

    atom = np.asarray(field("atom"), dtype=np.int64)
    root = np.asarray(field("root"), dtype=np.int64)
    dist = np.asarray(field("dist"), dtype=np.int64)
    patch = np.asarray(field("patch"), dtype=np.int64)
    bond = np.asarray(field("bond"), dtype=np.int64)
    edge_patch = np.asarray(field("edge_patch"), dtype=np.int64)
    src_local = np.asarray(field("src"), dtype=np.int64)
    dst_local = np.asarray(field("dst"), dtype=np.int64)
    n_patches = int(field("n_patches"))
    counts = np.bincount(patch, minlength=n_patches).astype(np.int64)
    node_start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    src = src_local + node_start[edge_patch]
    dst = dst_local + node_start[edge_patch]
    return StructExample(
        graph_id=int(graph_id),
        atom=atom,
        root=root,
        dist=dist,
        patch=patch,
        src=src,
        dst=dst,
        bond=bond,
        n_patches=n_patches,
        n_nodes=int(field("n_nodes")),
        n_edges=int(field("n_edges")),
        exterior_signature=str(exterior_signature),
    )


def collate_struct(examples: Sequence[StructExample]) -> StructBatch:
    node_offset = 0
    patch_offset = 0
    atoms: list[torch.Tensor] = []
    roots: list[torch.Tensor] = []
    dists: list[torch.Tensor] = []
    patches: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    dsts: list[torch.Tensor] = []
    bonds: list[torch.Tensor] = []
    patch_counts: list[int] = []
    graph_ids: list[int] = []
    for example in examples:
        atoms.append(torch.as_tensor(example.atom, dtype=torch.long))
        roots.append(torch.as_tensor(example.root, dtype=torch.long))
        dists.append(torch.as_tensor(example.dist, dtype=torch.long))
        patches.append(torch.as_tensor(example.patch + patch_offset, dtype=torch.long))
        srcs.append(torch.as_tensor(example.src + node_offset, dtype=torch.long))
        dsts.append(torch.as_tensor(example.dst + node_offset, dtype=torch.long))
        bonds.append(torch.as_tensor(example.bond, dtype=torch.long))
        patch_counts.append(int(example.n_patches))
        graph_ids.append(int(example.graph_id))
        node_offset += int(example.n_nodes)
        patch_offset += int(example.n_patches)
    empty = torch.zeros(0, dtype=torch.long)
    return StructBatch(
        struct_atom=torch.cat(atoms) if atoms else empty,
        struct_root=torch.cat(roots) if roots else empty,
        struct_dist=torch.cat(dists) if dists else empty,
        struct_patch=torch.cat(patches) if patches else empty,
        struct_src=torch.cat(srcs) if srcs else empty,
        struct_dst=torch.cat(dsts) if dsts else empty,
        struct_bond=torch.cat(bonds) if bonds else empty,
        patch_counts=patch_counts,
        graph_ids=graph_ids,
    )


def strong_encoder(device: str | torch.device = "cpu") -> tuple[nn.Module, nn.Module]:
    if not STRONG_CHECKPOINT.exists():
        raise FileNotFoundError(f"frozen strong checkpoint missing: {STRONG_CHECKPOINT}")
    model = SSPE.build_candidate(0)
    state = torch.load(STRONG_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = model.structural_encoder
    if encoder is None:
        raise RuntimeError("B-full model has no structural encoder")
    encoder.eval()
    return model, encoder


def _embedding_cache_path(split: str) -> Path:
    return CACHE_DIR / f"strong_local_{split}.pkl.gz"


def _embedding_metadata(split: str, graphs_meta: Mapping[str, Any], n_graphs: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema": "tccd_v3_strong_local_embeddings_v1",
        "split": split,
        "n_graphs": int(n_graphs),
        "strong_checkpoint": str(STRONG_CHECKPOINT.relative_to(REPO_ROOT)),
        "strong_checkpoint_sha256": _sha256_file(STRONG_CHECKPOINT),
        "strong_source_commit": STRONG_SOURCE_COMMIT,
        "strong_encoder_class": "SharedStructuralPatchEncoder",
        "strong_tensor": "structural_encoder.forward -> e_struct",
        "strong_width": STRONG_LOCAL_WIDTH,
        "radius": 2,
        "graph_cache_schema": STRONG_CACHE_SCHEMA,
        "graph_cache_metadata": graphs_meta,
        "official_test_loaded": False,
        "relation_metadata_reference": f"tccd_v3/{split}/graph_index_and_centre_id",
    }


def extract_embeddings(split: str, device: str, *, batch_size: int = 128) -> tuple[list[np.ndarray], dict[str, Any]]:
    graphs, graph_meta = _graphs_for_split(split)
    _full_model, encoder = strong_encoder(device)
    examples = [struct_example_from_plain(graph, i) for i, graph in enumerate(graphs)]
    loader = torch.utils.data.DataLoader(
        examples,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_struct,
    )
    embeddings: list[np.ndarray | None] = [None] * len(examples)
    with torch.no_grad():
        for batch in loader:
            out = encoder(batch.to(device)).detach().cpu().numpy().astype(np.float32)
            offset = 0
            for graph_id, count in zip(batch.graph_ids, batch.patch_counts):
                embeddings[int(graph_id)] = out[offset : offset + int(count)].copy()
                offset += int(count)
    if any(item is None for item in embeddings):
        raise RuntimeError(f"missing embedding rows for split={split}")
    final = [item for item in embeddings if item is not None]
    if any(item.shape[1] != STRONG_LOCAL_WIDTH for item in final):
        raise RuntimeError("strong-local embedding width mismatch")
    metadata = _embedding_metadata(split, graph_meta, len(final))
    metadata["n_patches"] = int(sum(item.shape[0] for item in final))
    metadata["graph_ids"] = list(range(len(final)))
    metadata["centre_ids"] = [list(range(int(item.shape[0]))) for item in final]
    return final, metadata


def load_or_build_embeddings(split: str, device: str, *, batch_size: int = 128, force: bool = False) -> tuple[list[np.ndarray], dict[str, Any]]:
    path = _embedding_cache_path(split)
    if path.exists() and not force:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        meta = dict(payload["metadata"])
        if (
            meta.get("cache_schema") == "tccd_v3_strong_local_embeddings_v1"
            and meta.get("strong_checkpoint_sha256") == _sha256_file(STRONG_CHECKPOINT)
            and meta.get("n_graphs") == len(payload["embeddings"])
        ):
            return [np.asarray(x, dtype=np.float32) for x in payload["embeddings"]], meta
    embeddings, metadata = extract_embeddings(split, device, batch_size=batch_size)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump({"metadata": metadata, "embeddings": embeddings}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(CACHE_DIR / f"strong_local_{split}_metadata.json", metadata)
    return embeddings, metadata


def _official_records(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records, train_meta = V2.load_train_records()
    train_mols, _ = T.load_mols(data_root, "train")
    valid_mols, valid_y = T.load_mols(data_root, "valid")
    layout = V2.frozen_layout()
    atom_index, bond_index = T.category_catalog(train_mols)
    if len(atom_index) != layout.n_atom or len(bond_index) != layout.n_bond:
        raise RuntimeError("frozen TCCD layout category mismatch")
    valid_records, valid_cached = load_or_build_records(
        "valid",
        valid_mols,
        layout,
        atom_index,
        bond_index,
        valid_y,
        len(valid_mols),
    )
    meta = {
        "train_records_source": str(V1.V0_TRAIN_CACHE.relative_to(REPO_ROOT)),
        "train_meta": train_meta,
        "valid_records_cached": bool(valid_cached),
        "valid_records_source": str(
            records_cache_path("valid", layout, len(valid_records)).relative_to(REPO_ROOT)
        ),
        "n_train": len(train_records),
        "n_valid": len(valid_records),
        "official_test_loaded": False,
    }
    return list(train_records), list(valid_records), meta


def build_bridge_records(data_root: Path, device: str, *, force_embeddings: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records, valid_records, protocol_meta = _official_records(data_root)
    train_e, train_e_meta = load_or_build_embeddings("train", device, force=force_embeddings)
    valid_e, valid_e_meta = load_or_build_embeddings("valid", device, force=force_embeddings)
    if len(train_records) != len(train_e) or len(valid_records) != len(valid_e):
        raise RuntimeError("strong embedding / TCCD record graph count mismatch")
    out_train: list[dict[str, Any]] = []
    out_valid: list[dict[str, Any]] = []
    for records, embeddings, split, target in (
        (train_records, train_e, "train", out_train),
        (valid_records, valid_e, "valid", out_valid),
    ):
        for graph_id, (record, embedding) in enumerate(zip(records, embeddings)):
            if int(record["n"]) != int(embedding.shape[0]):
                raise RuntimeError(f"{split} graph {graph_id}: centre count mismatch")
            copied = dict(record)
            copied["X"] = np.asarray(embedding, dtype=np.float32)
            copied["graph_index"] = int(graph_id)
            copied["split"] = split
            copied["centre_ids"] = list(range(int(record["n"])))
            target.append(copied)
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": protocol_meta,
        "train_embedding_cache": train_e_meta,
        "valid_embedding_cache": valid_e_meta,
        "input_width": STRONG_LOCAL_WIDTH,
        "adapter_width": D_LOCAL,
        "official_test_loaded": False,
    }
    return out_train, out_valid, meta


def _compose_latent_padded(Z_flat: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    B, N, _ = batch["X_pad"].shape
    Z = Z_flat.reshape(B, N, D_LOCAL)
    valid = batch["valid"]
    Z = Z * valid.unsqueeze(-1).to(Z.dtype)
    M = torch.einsum("bnk,brnm,bml->brkl", Z, batch["R_pad"], Z)
    pair = M[:, :, batch["iu0"], batch["iu1"]].reshape(B, -1)
    return torch.cat([Z.sum(dim=1), pair], dim=1)


class DenseBridgeModel(nn.Module):
    def __init__(self, input_width: int = STRONG_LOCAL_WIDTH, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(int(seed) + 17001)
        self.adapter = nn.Linear(int(input_width), D_LOCAL)
        self.head = nn.Linear(T.h_dim(K_PROTO, N_REL), 1)
        self.uses_latent_bypass = False
        self.graph_feature_kind = "dense_latent_composition_only"
        self.input_width = int(input_width)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.adapter(X)

    def graph_features(self, Z: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return _compose_latent_padded(Z, batch)

    def forward_padded(self, batch: Mapping[str, torch.Tensor], **_: Any):
        X = batch["X_pad"].reshape(-1, self.input_width)
        Z = self.encode(X)
        h = self.graph_features(Z, batch)
        pred = self.head(h).reshape(-1)
        return pred, Z, h

    def renormalize_(self) -> None:
        return None


class PrototypeBridgeModel(nn.Module):
    def __init__(self, input_width: int = STRONG_LOCAL_WIDTH, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(int(seed) + 17001)
        self.adapter = nn.Linear(int(input_width), D_LOCAL)
        P = V2.normalized_gaussian(D_LOCAL, K_PROTO, PROTO_INIT_SEED)
        self.P = nn.Parameter(torch.as_tensor(P.copy()))
        self.temp_logit = nn.Parameter(torch.tensor(initial_temperature_logit(), dtype=torch.float32))
        self.head = nn.Linear(T.h_dim(K_PROTO, N_REL), 1)
        self.uses_latent_bypass = False
        self.graph_feature_kind = "assignment_composition_only"
        self.input_width = int(input_width)
        self.K = K_PROTO
        self.d = D_LOCAL
        self.proto_seed = int(PROTO_INIT_SEED)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.adapter(X)

    def temperature(self) -> torch.Tensor:
        return TEMP_MIN + TEMP_SPAN * torch.sigmoid(self.temp_logit)

    def assign(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.encode(X)
        Zbar = torch.nn.functional.normalize(Z, dim=-1, eps=EPS)
        Pbar = torch.nn.functional.normalize(self.P, dim=0, eps=EPS)
        logits = (Zbar @ Pbar) / self.temperature()
        return torch.softmax(logits, dim=-1)

    def graph_features(
        self,
        C: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        shuffle: bool = False,
        indices: Sequence[int] | None = None,
        records: Sequence[Mapping[str, Any]] | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        C3 = C.reshape(batch["X_pad"].shape[0], batch["X_pad"].shape[1], K_PROTO)
        if shuffle:
            if indices is None or records is None:
                raise ValueError("shuffle requires indices and records")
            C3 = V2.shuffle_assignments(C3, records, indices, seed, C.device)
        return V2.compose_padded(C3, batch["R_pad"], batch["valid"], batch["iu0"], batch["iu1"])

    def forward_padded(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        shuffle: bool = False,
        indices: Sequence[int] | None = None,
        records: Sequence[Mapping[str, Any]] | None = None,
        seed: int = 0,
    ):
        X = batch["X_pad"].reshape(-1, self.input_width)
        C = self.assign(X)
        h = self.graph_features(C, batch, shuffle=shuffle, indices=indices, records=records, seed=seed)
        pred = self.head(h).reshape(-1)
        return pred, C, h

    def renormalize_(self) -> None:
        with torch.no_grad():
            self.P.copy_(self.P / self.P.norm(dim=0, keepdim=True).clamp_min(1e-6))


def make_batch(records: Sequence[Mapping[str, Any]], indices: Sequence[int], device: str):
    return V2.make_batch(records, indices, device)


def evaluate_mae(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: str,
    *,
    batch: int = 64,
    shuffle: bool = False,
    seed: int = 0,
) -> float:
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_batch(records, chunk, device)
            if isinstance(model, PrototypeBridgeModel):
                pred, _, _ = model.forward_padded(b, shuffle=shuffle, indices=chunk, records=records, seed=seed)
            else:
                pred, _, _ = model.forward_padded(b)
            errors.append((pred - b["y"]).abs().cpu().numpy())
    return float(np.concatenate(errors).mean())


def _initial_regularization(
    model: PrototypeBridgeModel,
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    device: str,
    *,
    batch: int,
) -> dict[str, Any]:
    cal_idx = list(train_indices[: min(int(batch), len(train_indices))])
    b = make_batch(records, cal_idx, device)
    with torch.no_grad():
        pred, C_flat, _ = model.forward_padded(b)
        valid = b["valid"].reshape(-1)
        task = float((pred - b["y"]).abs().mean())
        local = float(V2.local_entropy(C_flat, valid))
        balance = float(V2.balance_kl(C_flat, valid))
    lam_local = 0.05 * task / max(local, EPS)
    lam_balance = 0.05 * task / max(balance, V2.BALANCE_FLOOR)
    return {
        "calibration_graphs": len(cal_idx),
        "initial_task": task,
        "initial_local_entropy": local,
        "initial_balance_kl": balance,
        "lambda_local": float(lam_local),
        "lambda_balance": float(lam_balance),
        "initial_local_contribution": float(lam_local * local),
        "initial_balance_contribution": float(lam_balance * balance),
        "balance_floor": V2.BALANCE_FLOOR,
    }


def train_bridge(
    model: nn.Module,
    records_train: Sequence[Mapping[str, Any]],
    records_valid: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    valid_indices: Sequence[int],
    device: str,
    *,
    seed: int = 0,
    max_epochs: int = T.MAX_EPOCHS,
    patience: int = T.PATIENCE,
    batch: int = T.BATCH,
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    log=print,
) -> V2.PrototypeTrainResult:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    model.to(device)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    is_proto = isinstance(model, PrototypeBridgeModel)
    reg = _initial_regularization(model, records_train, train_indices, device, batch=batch) if is_proto else {}
    lam_local = float(reg.get("lambda_local", 0.0))
    lam_balance = float(reg.get("lambda_balance", 0.0))
    best = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    top: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.time()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(train_indices))
        ep_loss = ep_task = ep_local = ep_balance = 0.0
        ep_n = 0
        for start in range(0, len(order), int(batch)):
            sel = [train_indices[i] for i in order[start : start + int(batch)]]
            b = make_batch(records_train, sel, device)
            opt.zero_grad(set_to_none=True)
            if is_proto:
                pred, C_flat, _ = model.forward_padded(b)
                valid = b["valid"].reshape(-1)
                task = (pred - b["y"]).abs().mean()
                local = V2.local_entropy(C_flat, valid)
                balance = V2.balance_kl(C_flat, valid)
                loss = task + lam_local * local + lam_balance * balance
                ep_local += float(local.detach()) * len(sel)
                ep_balance += float(balance.detach()) * len(sel)
            else:
                pred, _, _ = model.forward_padded(b)
                task = (pred - b["y"]).abs().mean()
                loss = task
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            opt.step()
            model.renormalize_()
            ep_loss += float(loss.detach()) * len(sel)
            ep_task += float(task.detach()) * len(sel)
            ep_n += len(sel)
        valid_mae = evaluate_mae(model, records_valid, valid_indices, device, batch=64)
        row = {
            "epoch": int(epoch),
            "train_loss": ep_loss / max(ep_n, 1),
            "train_task": ep_task / max(ep_n, 1),
            "train_local_entropy": ep_local / max(ep_n, 1),
            "train_balance_kl": ep_balance / max(ep_n, 1),
            "valid": valid_mae,
            "temperature": float(model.temperature().detach()) if is_proto else None,
        }
        history.append(row)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top.append((float(valid_mae), int(epoch), state))
        top.sort(key=lambda item: (item[0], item[1]))
        top = top[: T.TOP_K_SOUP]
        if valid_mae < best - 1e-9:
            best = float(valid_mae)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(state)
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == int(max_epochs):
            tau = f" tau={row['temperature']:.5f}" if row["temperature"] is not None else ""
            log(f"[{model.__class__.__name__}] epoch={epoch:03d} train={row['train_task']:.6f} valid={valid_mae:.6f} best={best:.6f}@{best_epoch}{tau}")
        if stale >= int(patience):
            break
    soup_state = None
    soup_valid = None
    members: list[int] = []
    if top:
        members = [epoch for _, epoch, _ in top]
        keys = top[0][2].keys()
        soup_state = {key: sum(state[key].float() for _, _, state in top) / len(top) for key in keys}
        backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(soup_state)
        soup_valid = evaluate_mae(model, records_valid, valid_indices, device, batch=64)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return V2.PrototypeTrainResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_valid is None else float(soup_valid),
        soup_members=members,
        train_history=history,
        state_best=best_state,
        state_soup=soup_state,
        regularization=reg,
        wall_s=time.time() - started,
    )


def build_model(arm: str, seed: int = 0) -> nn.Module:
    if arm == "dense":
        return DenseBridgeModel(seed=seed)
    if arm == "prototype":
        return PrototypeBridgeModel(seed=seed)
    raise ValueError(arm)


def collect_assignments(
    model: PrototypeBridgeModel,
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: str,
    *,
    batch: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    chunks: list[np.ndarray] = []
    graph_ids: list[int] = []
    centre_ids: list[int] = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch)):
            chunk = list(indices[start : start + int(batch)])
            b = make_batch(records, chunk, device)
            _, C_flat, _ = model.forward_padded(b)
            valid = b["valid"].reshape(-1)
            chunks.append(C_flat[valid].cpu().numpy())
            for gi in chunk:
                n = int(records[gi]["n"])
                graph_ids.extend([int(gi)] * n)
                centre_ids.extend(list(range(n)))
    return (
        np.concatenate(chunks, axis=0),
        np.asarray(graph_ids, dtype=np.int64),
        np.asarray(centre_ids, dtype=np.int64),
        np.arange(len(graph_ids), dtype=np.int64),
    )


def vocabulary_diagnostics(
    model: PrototypeBridgeModel,
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: str,
) -> dict[str, Any]:
    C, graph_ids, centre_ids, flat_ids = collect_assignments(model, records, indices, device)
    usage = V2.usage_metrics(C, graph_ids)
    coherence = V2.semantic_coherence(C, records, list(indices))
    order = np.argsort(C.mean(axis=0))[::-1]
    top_real: dict[str, Any] = {}
    for k in range(K_PROTO):
        take = min(50, C.shape[0])
        idx = np.argsort(C[:, k])[::-1][:take]
        rows = []
        for j in idx:
            gi = int(graph_ids[j])
            ci = int(centre_ids[j])
            key = records[gi]["keys"][ci]
            rows.append({
                "flat_patch_index": int(j),
                "graph_id": gi,
                "centre_id": ci,
                "assignment": float(C[j, k]),
                "key_hex": bytes(key).hex(),
            })
        top_real[str(k)] = rows
    return {
        "usage": usage,
        "learned_temperature": float(model.temperature().detach()),
        "semantic_coherence_summary": coherence.get("_summary", {}),
        "semantic_coherence": coherence,
        "top_real_patches_per_prototype": top_real,
        "top8_prototypes_by_usage": order[:8].astype(int).tolist(),
        "n_patches": int(C.shape[0]),
    }


def classify_decision(strong_proto: float) -> str:
    delta_local = float(TCCD_V2_PROTO_BEST - strong_proto)
    if strong_proto <= L1_MAX and delta_local >= L1_DELTA:
        return "L1"
    if strong_proto <= L2_MAX and delta_local >= L2_DELTA:
        return "L2"
    return "L3"


def interpretation_proto_gap(gap: float) -> str:
    if gap <= 0.0:
        return "prototype has non-positive gap and positive inductive-bias evidence"
    if gap <= PROTO_PRESERVE:
        return "prototype vocabulary essentially preserves strong-local predictive ability"
    if gap > PROTO_HARMFUL:
        return "prototype vocabulary is harmful under strong local representation"
    return "prototype gap is intermediate / unresolved under the registered thresholds"


def interpretation_composition(gain: float) -> str:
    if gain > COMP_MIN:
        return "assignment-sensitive composition remains materially useful"
    return "stronger local embedding may already encode much of the prior composition signal"


def _locality_graph_pair() -> tuple[StructExample, StructExample]:
    """Build two actual graphs with the same radius-2 rooted patch.

    The root patch is the path 0-1-2.  Both graphs add node 3 outside the
    radius-2 patch; only the exterior node/edge attributes differ.
    """
    from torch_geometric.data import Data

    def make_data(exterior_atom: int, exterior_bond: int) -> Data:
        edges = [(0, 1), (1, 2), (2, 3)]
        directed = edges + [(b, a) for a, b in edges]
        return Data(
            edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
            x=torch.tensor([0, 1, 2, exterior_atom], dtype=torch.long),
            edge_attr=torch.tensor(
                [0, 1, exterior_bond, 0, 1, exterior_bond], dtype=torch.long
            ),
            num_nodes=4,
            y=torch.tensor([0.0]),
        )

    graphs_a = SSPE._patch_graphs_from_dataset([make_data(3, 2)])
    graphs_b = SSPE._patch_graphs_from_dataset([make_data(7, 3)])

    def root_patch(graph: Any, label: str) -> StructExample:
        mask_nodes = graph.patch == 0
        mask_edges = graph.edge_patch == 0
        nodes = np.flatnonzero(mask_nodes)
        remap = {int(old): i for i, old in enumerate(nodes.tolist())}
        src = np.asarray([remap[int(x)] for x in graph.src[mask_edges]], dtype=np.int64)
        dst = np.asarray([remap[int(x)] for x in graph.dst[mask_edges]], dtype=np.int64)
        return StructExample(
            0,
            graph.atom[mask_nodes],
            graph.root[mask_nodes],
            graph.dist[mask_nodes],
            np.zeros(int(nodes.size), dtype=np.int64),
            src,
            dst,
            graph.bond[mask_edges],
            1,
            int(nodes.size),
            int(mask_edges.sum()),
            label,
        )

    return root_patch(graphs_a[0], "A"), root_patch(graphs_b[0], "B")


def gate0_checks() -> dict[str, Any]:
    """Gate 0 correctness checks; no official test loader is touched."""
    torch.manual_seed(20260921)
    atom = np.asarray([0, 1, 2], dtype=np.int64)
    root = np.asarray([1, 0, 0], dtype=np.int64)
    dist = np.asarray([0, 1, 2], dtype=np.int64)
    patch = np.asarray([0, 0, 0], dtype=np.int64)
    src = np.asarray([0, 1, 1, 2], dtype=np.int64)
    dst = np.asarray([1, 0, 2, 1], dtype=np.int64)
    bond = np.asarray([0, 0, 1, 1], dtype=np.int64)
    base = StructExample(0, atom, root, dist, patch, src, dst, bond, 1, 3, 4, "outside-A")
    ext = StructExample(0, atom.copy(), root.copy(), dist.copy(), patch.copy(), src.copy(), dst.copy(), bond.copy(), 1, 3, 4, "outside-B")
    model, encoder = strong_encoder("cpu") if STRONG_CHECKPOINT.exists() else (None, None)
    checks: dict[str, Any] = {
        "official_test_loaded": False,
        "checkpoint_present": bool(STRONG_CHECKPOINT.exists()),
    }
    if encoder is None:
        checks.update({"pure_local": False, "permutation_invariant": False, "batch_invariant": False, "deterministic": False})
        return checks
    with torch.no_grad():
        e0 = encoder(collate_struct([base]))
        e1 = encoder(collate_struct([ext]))
        eb = encoder(collate_struct([base, ext]))[:1]
        reload_model, reload_encoder = strong_encoder("cpu")
        er = reload_encoder(collate_struct([base]))
    checks["synthetic_tensor_exterior_delta"] = float((e0 - e1).abs().max())
    checks["synthetic_tensor_exterior_invariant"] = checks["synthetic_tensor_exterior_delta"] <= LOCAL_PASS_TOL

    graph_a, graph_b = _locality_graph_pair()
    with torch.no_grad():
        eg_a = encoder(collate_struct([graph_a]))
        eg_b = encoder(collate_struct([graph_b]))
    checks["graph_exterior_max_delta"] = float((eg_a - eg_b).abs().max())
    checks["pure_local_max_delta"] = checks["graph_exterior_max_delta"]
    checks["pure_local"] = checks["pure_local_max_delta"] <= LOCAL_PASS_TOL
    checks["batch_max_delta"] = float((e0 - eb).abs().max())
    checks["batch_invariant"] = checks["batch_max_delta"] <= BATCH_PASS_TOL
    checks["determinism_max_delta"] = float((e0 - er).abs().max())
    checks["deterministic"] = checks["determinism_max_delta"] <= DETERMINISM_PASS_TOL
    # Relabel node ids and edge endpoints consistently.
    perm = np.asarray([0, 2, 1], dtype=np.int64)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.size)
    relabeled = StructExample(
        0,
        atom[perm],
        root[perm],
        dist[perm],
        patch[perm],
        inverse[src],
        inverse[dst],
        bond.copy(),
        1,
        3,
        4,
        "outside-A",
    )
    with torch.no_grad():
        ep = encoder(collate_struct([relabeled]))
    checks["permutation_max_delta"] = float((e0 - ep).abs().max())
    checks["permutation_invariant"] = checks["permutation_max_delta"] <= PERM_PASS_TOL
    checks["frozen_encoder"] = all(not parameter.requires_grad for parameter in model.parameters())
    checks["encoder_width"] = int(e0.shape[1]) == STRONG_LOCAL_WIDTH
    checks["all_pass"] = bool(all(value for key, value in checks.items() if isinstance(value, bool) and key != "official_test_loaded"))
    return checks
