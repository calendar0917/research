"""Build bootstrap-aligned consensus KSVD node tokens for MolHIV.

All dictionaries and alignment statistics are fitted exclusively on official
train patches.  Official-test graphs are never vectorized or sparse-coded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .build_molhiv_node_tokens import _reservoir_train_patches
from .data_molhiv import load_molhiv
from .ksvd import _omp, ksvd
from .molhiv_node_tokens import centered_ego_vector, graph_node_offsets


def _fit_consensus_members(
    Y: np.ndarray,
    n_members: int,
    bootstrap_fraction: float,
    n_atoms: int,
    sparsity: int,
    n_iter: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Fit one full-pool anchor and train-only subsampled KSVD members."""
    if n_members < 2:
        raise ValueError("consensus requires at least two members")
    if not 0.25 <= bootstrap_fraction <= 1.0:
        raise ValueError("bootstrap_fraction must be in [0.25, 1.0]")
    rng = np.random.default_rng(seed + 314159)
    dictionaries: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for member in range(n_members):
        if member == 0:
            indices = np.arange(Y.shape[1], dtype=np.int64)
            sampling = "full_anchor_pool"
        else:
            n_take = max(n_atoms, int(round(bootstrap_fraction * Y.shape[1])))
            indices = np.sort(rng.choice(Y.shape[1], size=n_take, replace=False))
            sampling = "without_replacement_subsample"
        D, _, info = ksvd(
            Y[:, indices],
            n_atoms=n_atoms,
            T=sparsity,
            T_min=1,
            n_iter=n_iter,
            seed=seed + 1009 * member,
        )
        dictionaries.append(D)
        infos.append({
            "member": member,
            "sampling": sampling,
            "n_source_patches": int(indices.size),
            "source_index_sha256": __import__("hashlib").sha256(
                np.ascontiguousarray(indices).view(np.uint8)
            ).hexdigest(),
            **info,
        })
    return dictionaries, infos


def _align_to_anchor(
    dictionaries: list[np.ndarray],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Hungarian-match atoms by absolute cosine and correct their signs."""
    anchor = dictionaries[0]
    aligned = [anchor.copy()]
    stats: list[dict[str, Any]] = [{
        "member": 0,
        "mean_abs_cosine_to_anchor": 1.0,
        "minimum_abs_cosine_to_anchor": 1.0,
        "maximum_abs_cosine_to_anchor": 1.0,
        "permutation": list(range(anchor.shape[1])),
        "signs": [1.0] * anchor.shape[1],
    }]
    for member, D in enumerate(dictionaries[1:], 1):
        similarity = anchor.T @ D
        rows, cols = linear_sum_assignment(-np.abs(similarity))
        if not np.array_equal(rows, np.arange(anchor.shape[1])):
            raise RuntimeError("unexpected incomplete anchor assignment")
        signs = np.sign(similarity[rows, cols])
        signs[signs == 0] = 1.0
        aligned_D = D[:, cols] * signs[None, :]
        matched = np.abs(similarity[rows, cols])
        aligned.append(aligned_D)
        stats.append({
            "member": member,
            "mean_abs_cosine_to_anchor": float(matched.mean()),
            "minimum_abs_cosine_to_anchor": float(matched.min()),
            "maximum_abs_cosine_to_anchor": float(matched.max()),
            "median_abs_cosine_to_anchor": float(np.median(matched)),
            "permutation": cols.astype(int).tolist(),
            "signs": signs.astype(float).tolist(),
        })
    return aligned, stats


def _mean_consensus_dictionary(aligned: list[np.ndarray]) -> np.ndarray:
    D = np.mean(np.stack(aligned, axis=0), axis=0)
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    if np.any(norms < 1e-12):
        raise RuntimeError("degenerate mean consensus atom")
    return D / norms


def _topk(code: np.ndarray, k: int) -> np.ndarray:
    if k >= code.size:
        return code
    keep = np.argpartition(np.abs(code), -k)[-k:]
    out = np.zeros_like(code)
    out[keep] = code[keep]
    return out


def _consensus_code(
    aligned: list[np.ndarray], y: np.ndarray, sparsity: int,
) -> tuple[np.ndarray, float]:
    codes = np.stack([_omp(D, y, sparsity) for D in aligned], axis=0)
    code = _topk(codes.mean(axis=0), sparsity)
    supports = np.abs(codes) > 1e-10
    # Fraction of selected coordinates shared by at least two members.
    retained = np.abs(code) > 1e-10
    agreement = float(np.mean(supports[:, retained].sum(axis=0) >= 2)) if retained.any() else 0.0
    return code, agreement


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000, help="0 means full dataset")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--n-atoms", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=3)
    ap.add_argument("--max-train-patches", type=int, default=6000)
    ap.add_argument("--consensus-members", type=int, default=3)
    ap.add_argument("--bootstrap-fraction", type=float, default=0.8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features were not loaded")
    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)

    Ytr, patch_stats = _reservoir_train_patches(
        graphs, tr, bundle.node_feats, bundle.edge_feats,
        radius=args.radius, max_nodes=args.max_nodes,
        capacity=args.max_train_patches, seed=args.dict_seed,
        max_patches_per_graph=0,
    )
    print(f"train patch pool {Ytr.shape}; raw={patch_stats['n_train_node_patches_raw']}", flush=True)

    members, member_info = _fit_consensus_members(
        Ytr, n_members=args.consensus_members,
        bootstrap_fraction=args.bootstrap_fraction,
        n_atoms=args.n_atoms, sparsity=args.sparsity,
        n_iter=args.ksvd_iter, seed=args.dict_seed,
    )
    aligned, alignment_info = _align_to_anchor(members)
    D_consensus = _mean_consensus_dictionary(aligned)
    print(
        "aligned dictionary mean cosines="
        + str([round(x["mean_abs_cosine_to_anchor"], 4) for x in alignment_info]),
        flush=True,
    )

    offsets = graph_node_offsets(graphs)
    total_nodes = int(offsets[-1])
    tokens_atoms = np.zeros((total_nodes, args.n_atoms), dtype=np.float32)
    tokens_codes = np.zeros((total_nodes, args.n_atoms), dtype=np.float32)
    encode_indices = np.concatenate([tr, va])
    encoded_nodes = 0
    agreement_sum = 0.0
    atoms_recon_sum = 0.0
    codes_anchor_recon_sum = 0.0
    for count, raw_i in enumerate(encode_indices, 1):
        i = int(raw_i)
        g = graphs[i]
        for raw_u in g.nodes:
            u = int(raw_u)
            y = np.asarray(centered_ego_vector(
                g, u, bundle.node_feats[i], bundle.edge_feats[i],
                radius=args.radius, max_nodes=args.max_nodes,
            ), dtype=np.float64)
            y /= max(float(np.linalg.norm(y)), 1e-12)
            row = int(offsets[i] + u)
            atom_code = _omp(D_consensus, y, args.sparsity)
            code_code, agreement = _consensus_code(aligned, y, args.sparsity)
            tokens_atoms[row] = atom_code.astype(np.float32)
            tokens_codes[row] = code_code.astype(np.float32)
            agreement_sum += agreement
            atoms_recon_sum += float(np.dot(y - D_consensus @ atom_code, y - D_consensus @ atom_code))
            codes_anchor_recon_sum += float(np.dot(y - aligned[0] @ code_code, y - aligned[0] @ code_code))
            encoded_nodes += 1
        if count % 500 == 0 or count == len(encode_indices):
            print(f"encoded graphs {count}/{len(encode_indices)}; nodes={encoded_nodes}", flush=True)

    for i in te:
        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
        if np.any(tokens_atoms[lo:hi] != 0) or np.any(tokens_codes[lo:hi] != 0):
            raise AssertionError("consensus cache encoded official-test nodes")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "offsets": offsets,
        "original_indices": np.asarray(bundle.meta["original_indices"], dtype=np.int64),
        "train_indices": tr,
        "valid_indices": va,
        "test_indices": te,
        "dictionary_ksvd_consensus_atoms": D_consensus.astype(np.float32),
        "dictionary_ksvd_consensus_codes": D_consensus.astype(np.float32),
        "tokens_ksvd_consensus_atoms": tokens_atoms,
        "tokens_ksvd_consensus_codes": tokens_codes,
    }
    for member, D in enumerate(aligned):
        archive[f"dictionary_ksvd_consensus_aligned_member{member}"] = D.astype(np.float32)
    np.savez_compressed(output, **archive)

    meta = {
        "protocol_id": "molhiv-bootstrap-aligned-consensus-ksvd-node-tokens-v1",
        "test_policy": "official test node patches were not vectorized or sparse-coded",
        "config": vars(args),
        "data_meta": bundle.meta,
        "patch_stats": patch_stats,
        "families": ["ksvd_consensus_atoms", "ksvd_consensus_codes"],
        "member_info": member_info,
        "alignment_info": alignment_info,
        "consensus_info": {
            "atom_consensus": "normalized mean of sign-corrected Hungarian-aligned atoms",
            "code_consensus": "mean of aligned member OMP codes followed by absolute top-T pruning",
            "mean_retained_support_majority_agreement": agreement_sum / max(encoded_nodes, 1),
            "encoded_atom_consensus_mean_squared_reconstruction_error": atoms_recon_sum / max(encoded_nodes, 1),
            "encoded_code_consensus_anchor_mean_squared_reconstruction_error": codes_anchor_recon_sum / max(encoded_nodes, 1),
        },
        "total_nodes": total_nodes,
        "encoded_train_valid_nodes": encoded_nodes,
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
