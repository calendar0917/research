"""D3 control: a fixed-salt random connected partition with the same size cap.

The partition is drawn per molecule by growing connected pieces from random seed
atoms, capped at ``random_partition.max_atoms``.  The template table is interned
from Train pieces only; Valid reuses it, so unseen Valid layouts get new ids just
like the frozen pipeline.  Labels are never read.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import wp_common as C
import wp_data as D

try:  # the inherited round's word machinery
    import ws_words as W
except ModuleNotFoundError:  # pragma: no cover - path is set by wp_common
    import sys

    sys.path.insert(0, str(C.OLD_ROUND / "code"))
    import ws_words as W

WORDS_ATOM_COLS = W.WORDS_ATOM_COLS


def subset_payload(raw: Mapping[str, Any], graphs: Sequence[int]) -> dict[str, np.ndarray]:
    """Raw-style payload restricted to ``graphs`` (graph-local indices preserved)."""
    node_ptr = np.asarray(raw["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    bond_u = np.asarray(raw["bond_u"], dtype=np.int64)
    bond_v = np.asarray(raw["bond_v"], dtype=np.int64)
    atom_parts, bu_parts, bv_parts, bf_parts = [], [], [], []
    node_out, bond_out = [0], [0]
    n_atoms = n_bonds = 0
    for g in graphs:
        a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        atom_parts.append(atom_feat[a0:a1])
        bu_parts.append(bond_u[b0:b1])  # cache endpoints are graph-local
        bv_parts.append(bond_v[b0:b1])
        bf_parts.append(bond_feat[b0:b1])
        n_atoms += a1 - a0
        n_bonds += b1 - b0
        node_out.append(n_atoms)
        bond_out.append(n_bonds)
    return {
        "node_ptr": np.asarray(node_out, dtype=np.int64),
        "bond_ptr": np.asarray(bond_out, dtype=np.int64),
        "atom_feat": np.concatenate(atom_parts, axis=0),
        "bond_u": np.concatenate(bu_parts),
        "bond_v": np.concatenate(bv_parts),
        "bond_feat": np.concatenate(bf_parts, axis=0),
    }


def random_states(molecules: Sequence[Any], graphs: Iterable[int], table: Any,
                  rng: np.random.Generator, max_atoms: int) -> dict[int, Any]:
    states = {}
    for g in graphs:
        mol = molecules[g]
        n = int(mol.n_atoms)
        taken = np.zeros(n, dtype=bool)
        pieces: list[list[int]] = []
        for start in rng.permutation(n).tolist():
            if taken[start]:
                continue
            piece = [int(start)]
            taken[start] = True
            frontier = [o for o, _ in mol.adj[start] if not taken[o]]
            while len(piece) < max_atoms and frontier:
                pick = int(rng.integers(len(frontier)))
                atom = int(frontier.pop(pick))
                if taken[atom]:
                    continue
                piece.append(atom)
                taken[atom] = True
                frontier.extend(o for o, _ in mol.adj[atom] if not taken[o])
            pieces.append(sorted(piece))
        state = W.PartState()
        for piece in pieces:
            pos = {atom: index for index, atom in enumerate(piece)}
            labels = np.stack([mol.atom_feat[atom][WORDS_ATOM_COLS] for atom in piece])
            bu, bv, bt = [], [], []
            for atom in piece:
                for other, bk in mol.adj[atom]:
                    if other in pos and pos[atom] < pos[other]:
                        bu.append(pos[atom])
                        bv.append(pos[other])
                        bt.append(int(mol.bond_feat[bk, 0]))
            tid = table.intern(labels, np.asarray(bu), np.asarray(bv), np.asarray(bt))
            state.pieces.append(piece)
            state.tmpl.append(tid)
        states[g] = state
    return states


def rebuild_splits(data: Mapping[str, Any], splits: Mapping[str, Any], cfg: Mapping[str, Any]):
    """Return (splits, template dict) for the random connected partition."""
    import ws_model as OM

    words, raw = data["words"], data["raw"]
    dims = C.official_dims()
    n_bond_codes = int(dims["bond"][0])
    salt = int(cfg["random_partition"]["salt"])
    max_atoms = int(cfg["random_partition"]["max_atoms"])
    table = W.TemplateTable()
    out: dict[str, Any] = {}
    for split in ("train", "valid"):
        graphs = [int(i) for i in raw[f"{split}_idx"]]
        molecules = W.load_molecules(subset_payload(raw, graphs))
        rng = np.random.default_rng(salt + (0 if split == "train" else 1))
        states = random_states(molecules, range(len(molecules)), table, rng, max_atoms)
        payload = W.build_split(molecules, range(len(molecules)), states, table)
        base = {k: np.asarray(words[f"base/{split}/{k}"])
                for k in ("atom_feat", "bond_u", "bond_v", "bond_feat",
                          "graph_n_atoms", "graph_n_bonds")}
        arrays = {f"word/{split}/{k}": v for k, v in payload.items()}
        arrays.update({f"base/{split}/{k}": v for k, v in base.items()})
        arrays[f"row_ids/{split}"] = np.asarray(words[f"row_ids/{split}"], dtype=np.int64)
        arrays[f"labels/{split}"] = np.asarray(words[f"labels/{split}"], dtype=np.float64)
        out[split] = arrays
    template = OM.template_arrays(table, dims, n_bond_codes)
    dist = D.template_dist_table({f"tmpl/{k}": v for k, v in template.items()})
    template["dist"] = dist
    template["n_templates"] = np.asarray([[int(len(table))]])
    for split, payload in out.items():
        out[split] = D.split_from_arrays(payload, dist, split) if isinstance(payload, dict) else payload
    return out, template


def summary(splits: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"partition": "random_connected", "splits": {}}
    for split, data in splits.items():
        arr = data.arrays
        sizes = np.diff(arr["site_ptr"])
        inc = np.diff(arr["inc_ptr"])
        report["splits"][split] = {
            "n_graphs": int(len(data)),
            "n_words": int(arr["word_tmpl"].shape[0]),
            "word_size_histogram": {str(int(k)): int(v) for k, v in
                                    zip(*np.unique(sizes, return_counts=True))},
            "port_count_histogram": {str(int(k)): int(v) for k, v in
                                     zip(*np.unique(inc, return_counts=True))},
            "boundary_incidence_ratio": float(len(inc) / max(1, arr["word_tmpl"].shape[0])),
            "words_per_molecule": float(arr["word_tmpl"].shape[0] / max(1, len(data))),
        }
    report["n_templates"] = int(np.asarray(template["n_templates"]).reshape(-1)[0])
    return report
