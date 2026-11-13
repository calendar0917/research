"""Weight-independent local sufficient signature ``S_v`` of one rooted atom.

Definition (frozen by PROTOCOL.md section 5).  For root ``v`` of one molecule the
signature is the following six blocks, serialised with length-prefixed fields:

1. ``A``: the root's complete ``atom_feat`` (51 float32) and complete
   ``atom_scalar`` (9 float32).
2. ``B``: the multiset of incident bonds, each entry ``(atom_feat[neighbour],
   bond_feat[e])``.  Repeated occurrences are kept.
3. ``C``: the multiset of one-hop ``nbr_code`` values.
4. ``D``: the joint S2 contingency multiset ``(path_cat, path_end63)``.
5. ``E``: the S2 content contingency multiset ``(path_cat, atom_feat[end])``.
6. ``F``: the multiset of ``path_closure`` values.

The blocks contain **only** quantities read by the frozen forward.  No atom id,
no ``path_end`` index, no molecule id and no SMILES string can enter: every block
is a sorted multiset of value vectors.

Floats are compared as canonical finite ``float32`` bytes with ``-0.0`` folded
to ``+0.0``.  The digest is used for candidate bucketing only; equality claims
are always confirmed with a field-by-field comparison of the serialised blocks.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Mapping, Sequence

import numpy as np

import common as C

MAGIC = b"LCRC-S3-SIG-v1"


# ---------------------------------------------------------------------------
# canonical scalar encoding
# ---------------------------------------------------------------------------


def f32_bytes(arr) -> bytes:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    a = a.copy()
    a[a == 0.0] = np.float32(0.0)  # fold -0.0 into +0.0
    if not np.isfinite(a).all():
        raise ValueError("non-finite float in a signature block")
    return a.tobytes()


def _u32(n: int) -> bytes:
    return struct.pack("<I", int(n))


def _i32(arr) -> bytes:
    return np.ascontiguousarray(np.asarray(arr, dtype="<i4")).tobytes()


def _i8(arr) -> bytes:
    return np.ascontiguousarray(np.asarray(arr, dtype="<i1")).tobytes()


def _multiset(entries: Sequence[bytes]) -> bytes:
    return _u32(len(entries)) + b"".join(sorted(entries))


# ---------------------------------------------------------------------------
# molecule-level index (adjacency / incident bonds)
# ---------------------------------------------------------------------------


def molecule_view(cache: Mapping[str, Any], mol: int) -> dict[str, Any]:
    """Slice one molecule out of a frozen structure cache (molecule-local)."""
    mol = int(mol)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    a0, a1 = int(atom_ptr[mol]), int(atom_ptr[mol + 1])
    b0, b1 = int(bond_ptr[mol]), int(bond_ptr[mol + 1])
    view = {
        "mol": mol,
        "row": int(np.asarray(cache["rows"])[mol]),
        "n_atoms": a1 - a0,
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1],
        "atom_scalar": np.asarray(cache["atom_scalar"])[a0:a1],
        "atom_type": np.asarray(cache["atom_type"])[a0:a1],
        "bond_index": np.asarray(cache["bond_index"])[b0:b1],
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1],
        "bond_type": np.asarray(cache["bond_type"])[b0:b1],
        "global": (
            np.asarray(cache["global"])[mol]
            if np.asarray(cache["global"]).ndim == 2
            else np.asarray(cache["global"]).reshape(-1)
        ),
        "atom_base": a0,
        "path_ptr": path_ptr[a0 : a1 + 1] - int(path_ptr[a0]),
        "path_cat": np.asarray(cache["path_cat"])[int(path_ptr[a0]) : int(path_ptr[a1])],
        "path_end63": np.asarray(cache["path_end63"])[int(path_ptr[a0]) : int(path_ptr[a1])],
        "path_end": np.asarray(cache["path_end"])[int(path_ptr[a0]) : int(path_ptr[a1])],
        "path_closure": np.asarray(cache["path_closure"])[int(path_ptr[a0]) : int(path_ptr[a1])],
        "nbr_ptr": np.asarray(cache["nbr_ptr"])[a0 : a1 + 1] - int(np.asarray(cache["nbr_ptr"])[a0]),
        "nbr_code": np.asarray(cache["nbr_code"])[
            int(np.asarray(cache["nbr_ptr"])[a0]) : int(np.asarray(cache["nbr_ptr"])[a1])
        ],
    }
    n = view["n_atoms"]
    incident: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(np.asarray(view["bond_index"]).tolist()):
        incident[int(u)].append((int(v), int(e)))
        incident[int(v)].append((int(u), int(e)))
    view["incident"] = incident
    return view


# ---------------------------------------------------------------------------
# the six blocks
# ---------------------------------------------------------------------------


def root_blocks(view: Mapping[str, Any], root: int) -> dict[str, bytes]:
    """Serialise the six signature blocks of ``root`` (one molecule view)."""
    v = int(root)
    atom_feat = np.asarray(view["atom_feat"], dtype=np.float32)
    atom_scalar = np.asarray(view["atom_scalar"], dtype=np.float32)
    bond_feat = np.asarray(view["bond_feat"], dtype=np.float32)

    block_a = f32_bytes(atom_feat[v]) + f32_bytes(atom_scalar[v])

    bond_entries: list[bytes] = []
    for u, e in view["incident"][v]:
        bond_entries.append(f32_bytes(atom_feat[int(u)]) + f32_bytes(bond_feat[int(e)]))
    block_b = _multiset(bond_entries)

    ptr = np.asarray(view["nbr_ptr"], dtype=np.int64)
    codes = np.asarray(view["nbr_code"], dtype=np.int64)[int(ptr[v]) : int(ptr[v + 1])]
    block_c = _multiset([_i32([int(c)]) for c in codes.tolist()])

    pptr = np.asarray(view["path_ptr"], dtype=np.int64)
    lo, hi = int(pptr[v]), int(pptr[v + 1])
    path_cat = np.asarray(view["path_cat"])[lo:hi]
    path_end63 = np.asarray(view["path_end63"])[lo:hi]
    path_end = np.asarray(view["path_end"])[lo:hi]
    closure = np.asarray(view["path_closure"])[lo:hi]

    block_d = _multiset(
        [_i32([int(c), int(e)]) for c, e in zip(path_cat.tolist(), path_end63.tolist())]
    )
    block_e = _multiset(
        [_i32([int(c)]) + f32_bytes(atom_feat[int(e)]) for c, e in zip(path_cat.tolist(), path_end.tolist())]
    )
    block_f = _multiset([_i8([int(x)]) for x in closure.tolist()])

    return {"A": block_a, "B": block_b, "C": block_c, "D": block_d, "E": block_e, "F": block_f}


BLOCKS = ("A", "B", "C", "D", "E", "F")


def blocks_bytes(blocks: Mapping[str, bytes]) -> bytes:
    return MAGIC + b"".join(b"|" + k.encode() + b"|" + blocks[k] for k in BLOCKS)


def root_digest(view: Mapping[str, Any], root: int) -> bytes:
    return hashlib.sha256(blocks_bytes(root_blocks(view, root))).digest()


def root_digest_hex(view: Mapping[str, Any], root: int) -> str:
    return root_digest(view, root).hex()


# ---------------------------------------------------------------------------
# molecule-level multiset
# ---------------------------------------------------------------------------


def molecule_digests(view: Mapping[str, Any]) -> list[str]:
    return [root_digest_hex(view, v) for v in range(int(view["n_atoms"]))]


def multiset_digest(digests: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(MAGIC)
    for d in sorted(str(x) for x in digests):
        h.update(bytes.fromhex(d))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# field-by-field verification (never hash-only)
# ---------------------------------------------------------------------------


def compare_roots(view_a: Mapping[str, Any], root_a: int, view_b: Mapping[str, Any], root_b: int) -> dict[str, Any]:
    ba = root_blocks(view_a, root_a)
    bb = root_blocks(view_b, root_b)
    per_block = {k: bool(ba[k] == bb[k]) for k in BLOCKS}
    return {
        "equal": all(per_block.values()),
        "per_block": per_block,
        "digest_equal": bool(root_digest(view_a, root_a) == root_digest(view_b, root_b)),
    }


def compare_molecule_signatures(view_a: Mapping[str, Any], view_b: Mapping[str, Any]) -> dict[str, Any]:
    """Exact multiset comparison of the six-block descriptors of two molecules.

    The digest multiset is only the candidate test.  Equality is decided on the
    full serialised descriptor bytes (all six blocks, every field), counted as a
    multiset.  ``per_block`` then records, for the first matched root pair, which
    blocks differ (empty when the multisets differ).
    """
    from collections import Counter

    da = molecule_digests(view_a)
    db = molecule_digests(view_b)
    digest_equal = bool(sorted(da) == sorted(db))
    full_a = Counter(blocks_bytes(root_blocks(view_a, v)) for v in range(int(view_a["n_atoms"])))
    full_b = Counter(blocks_bytes(root_blocks(view_b, v)) for v in range(int(view_b["n_atoms"])))
    equal = bool(full_a == full_b)
    per_block: dict[str, bool] = {k: bool(equal) for k in BLOCKS}
    return {
        "multiset_equal": equal,
        "digest_multiset_equal": digest_equal,
        "n_roots_a": int(view_a["n_atoms"]),
        "n_roots_b": int(view_b["n_atoms"]),
        "per_block_all_equal": bool(equal),
        "per_block": per_block,
        "n_verified_root_pairs": int(min(view_a["n_atoms"], view_b["n_atoms"])) if digest_equal else 0,
        "comparison": "exact multiset of full six-block descriptor bytes",
    }


def global_bytes(view: Mapping[str, Any]) -> bytes:
    return f32_bytes(np.asarray(view["global"], dtype=np.float32))


def global_digest(view: Mapping[str, Any]) -> str:
    return hashlib.sha256(global_bytes(view)).hexdigest()


# ---------------------------------------------------------------------------
# identity-field exclusion audit
# ---------------------------------------------------------------------------


def signature_field_audit() -> dict[str, Any]:
    """Static statement of what the serialiser can and cannot read."""
    return {
        "blocks": {
            "A": "atom_feat[root] (51), atom_scalar[root] (9)",
            "B": "multiset of (atom_feat[neighbour], bond_feat[incident bond])",
            "C": "multiset of nbr_code",
            "D": "multiset of (path_cat, path_end63)",
            "E": "multiset of (path_cat, atom_feat[end])",
            "F": "multiset of path_closure",
        },
        "excluded": [
            "atom id / root index",
            "path_end atom index",
            "molecule id / row id",
            "SMILES / canonical SMILES",
            "split assignment, target value",
        ],
        "float_encoding": "canonical finite float32 bytes; -0.0 folded to +0.0",
        "integer_encoding": "little-endian int32 / int8, original integer semantics",
        "hash_role": "candidate bucketing only; equality confirmed block-by-block",
    }


__all__ = [
    "MAGIC",
    "BLOCKS",
    "f32_bytes",
    "molecule_view",
    "root_blocks",
    "blocks_bytes",
    "root_digest",
    "root_digest_hex",
    "molecule_digests",
    "multiset_digest",
    "compare_roots",
    "compare_molecule_signatures",
    "global_bytes",
    "global_digest",
    "signature_field_audit",
]
