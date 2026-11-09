"""P2 — Train-only radius-2 WL support of local structural environments.

Definition (frozen in PROTOCOL.md section 6):

* atom initial colour: atomic number, formal charge, aromaticity, total H count,
  total degree, in-ring flag;
* bond label: bond type, aromaticity, conjugation, in-ring flag;
* two refinement rounds: ``sha256(canonical_json([self_colour, sorted
  [(bond_label, neighbour_colour), ...]]))`` -> ``WL2_ID`` per atom;
* ``train_molecule_frequency[WL2_ID]`` counts **distinct Train molecules** that
  contain the colour (a repeated colour inside one molecule counts once);
* per Valid/Test molecule: ``fraction_unseen`` (freq == 0), ``fraction_rare``
  (0 < freq <= 4), ``median_train_support``, ``n_heavy_atoms``.

Only the official mapping ``smiles`` column and the official split row ids are
read.  No activity label is touched here.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger

import lc_common as C

RDLogger.DisableLog("rdApp.*")

SUPPORT_PATH = C.LOCAL / "wl2_train_support.npz"
SUMMARY_PATH = C.ANALYSIS / "train_wl2_frequency_summary.json"
VALID_CSV = C.ANALYSIS / "structure_support_valid.csv"
TEST_CSV = C.ANALYSIS / "structure_support_test.csv"
STRUCT_SUMMARY = C.ANALYSIS / "structure_support_summary.json"

RDKIT_VERSION = __import__("rdkit").__version__


def _sha(text: str) -> bytes:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).digest()


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def atom_signature(atom: Chem.Atom) -> str:
    return _canonical(
        [
            int(atom.GetAtomicNum()),
            int(atom.GetFormalCharge()),
            int(bool(atom.GetIsAromatic())),
            int(atom.GetTotalNumHs()),
            int(atom.GetTotalDegree()),
            int(bool(atom.IsInRing())),
        ]
    )


def bond_signature(bond: Chem.Bond) -> str:
    try:
        conj = bool(bond.GetIsConjugated())
    except Exception:
        conj = False
    return _canonical(
        [
            str(bond.GetBondType()),
            int(bool(bond.GetIsAromatic())),
            int(conj),
            int(bool(bond.IsInRing())),
        ]
    )


def parse_smiles(smiles: str) -> tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, "sanitized"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES even without sanitisation")
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    return mol, "unsanitized_fallback"


def wl2_root_colours(mol: Chem.Mol, *, rounds: int = 2) -> list[bytes]:
    """One stable WL2 colour (sha256 digest) per atom, rooted at radius ``rounds``."""
    atoms = list(mol.GetAtoms())
    colours = [_sha(atom_signature(a)) for a in atoms]
    adj: list[list[tuple[str, int]]] = [[] for _ in atoms]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lab = bond_signature(bond)
        adj[i].append((lab, j))
        adj[j].append((lab, i))
    for _ in range(rounds):
        new: list[bytes] = []
        for i in range(len(atoms)):
            neigh = sorted([adj[i][k][0], colours[adj[i][k][1]].hex()] for k in range(len(adj[i])))
            new.append(_sha(_canonical([colours[i].hex(), neigh])))
        colours = new
    return colours


def read_smiles() -> list[str]:
    import pandas as pd

    frame = pd.read_csv(C.OGB_MAPPING, usecols=["smiles"])
    return [str(x) for x in frame["smiles"].tolist()]


def read_split_rows(split: str) -> np.ndarray:
    import pandas as pd

    arr = pd.read_csv(C.SPLIT_FILES[split], header=None).to_numpy().reshape(-1)
    return arr.astype(np.int64)


def build_train_support(smiles: list[str], train_rows: np.ndarray) -> tuple[Counter, dict[str, Any]]:
    support: Counter = Counter()
    parse_modes: Counter = Counter()
    n_atoms_total = 0
    n_zero_atom = 0
    t0 = time.perf_counter()
    for idx, row in enumerate(train_rows):
        smi = smiles[int(row)]
        try:
            mol, mode = parse_smiles(smi)
        except Exception:
            parse_modes["failed"] += 1
            continue
        parse_modes[mode] += 1
        colours = wl2_root_colours(mol)
        n_atoms_total += len(colours)
        if not colours:
            n_zero_atom += 1
        for colour in set(colours):  # distinct-within-molecule only
            support[colour] += 1
        if (idx + 1) % 50_000 == 0:
            print(f"  [wl2-train] {idx + 1}/{train_rows.size} molecules, {len(support)} distinct WL2", flush=True)
    meta = {
        "n_train_molecules": int(train_rows.size),
        "n_train_atoms": int(n_atoms_total),
        "n_zero_atom_train": int(n_zero_atom),
        "parse_modes": dict(parse_modes),
        "n_distinct_wl2": int(len(support)),
        "distinct_with_support_1": int(sum(1 for v in support.values() if v == 1)),
        "distinct_with_support_le_4": int(sum(1 for v in support.values() if v <= C.RARE_SUPPORT_MAX)),
        "max_support": int(max(support.values())) if support else 0,
        "build_seconds": float(time.perf_counter() - t0),
        "definition": {
            "rooted": True,
            "rounds": 2,
            "atom_features": ["atomic_number", "formal_charge", "aromatic", "total_h", "total_degree", "in_ring"],
            "bond_features": ["bond_type", "aromatic", "conjugated", "in_ring"],
            "hash": "sha256 over canonical JSON of [self_colour_hex, sorted [(bond_label, neighbour_colour_hex)]]",
            "molecule_frequency": "number of distinct Train molecules containing the WL2 colour (repeat inside one molecule counts once)",
            "labels_read": False,
        },
    }
    return support, meta


def support_features(
    smiles: list[str], rows: np.ndarray, support: Counter, *, split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    parse_modes: Counter = Counter()
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        smi = smiles[int(row)]
        rec: dict[str, Any] = {
            "split": split,
            "split_index": int(i),
            "ogb_row": int(row),
            "n_heavy_atoms": None,
            "n_atoms": None,
            "n_distinct_wl2": None,
            "fraction_unseen": None,
            "fraction_rare": None,
            "median_train_support": None,
            "min_train_support": None,
            "max_train_support": None,
            "parse_mode": None,
            "novelty_group": "UNRESOLVED",
        }
        try:
            mol, mode = parse_smiles(smi)
        except Exception:
            parse_modes["failed"] += 1
            rec["parse_mode"] = "failed"
            out.append(rec)
            continue
        parse_modes[mode] += 1
        colours = wl2_root_colours(mol)
        n_atoms = len(colours)
        n_heavy = int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1))
        rec["parse_mode"] = mode
        rec["n_atoms"] = n_atoms
        rec["n_heavy_atoms"] = n_heavy
        if n_atoms == 0:
            out.append(rec)
            continue
        supports = np.fromiter((support.get(c, 0) for c in colours), dtype=np.int64, count=n_atoms)
        frac_unseen = float((supports == 0).mean())
        rec.update(
            {
                "n_distinct_wl2": int(len(set(colours))),
                "fraction_unseen": frac_unseen,
                "fraction_rare": float((supports <= C.RARE_SUPPORT_MAX).mean()),
                "median_train_support": float(np.median(supports)),
                "min_train_support": int(supports.min()),
                "max_train_support": int(supports.max()),
                "novelty_group": C.novelty_group(frac_unseen),
            }
        )
        out.append(rec)
        if (i + 1) % 10_000 == 0:
            print(f"  [wl2-{split}] {i + 1}/{rows.size}", flush=True)
    meta = {
        "n_rows": int(rows.size),
        "parse_modes": dict(parse_modes),
        "seconds": float(time.perf_counter() - t0),
        "labels_read": False,
    }
    return out, meta


def save_support(support: Counter) -> str:
    C.LOCAL.mkdir(parents=True, exist_ok=True)
    keys = np.array([k for k in support.keys()], dtype="S32")
    values = np.fromiter((support[k] for k in support.keys()), dtype=np.int64, count=len(support))
    tmp = SUPPORT_PATH.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, keys=keys, values=values)
    Path(tmp).replace(SUPPORT_PATH)
    return C.sha256_file(SUPPORT_PATH)


def load_support() -> Counter:
    with np.load(SUPPORT_PATH, allow_pickle=False) as z:
        keys = z["keys"]
        values = z["values"]
    return Counter({bytes(k): int(v) for k, v in zip(keys.tolist(), values.tolist())})


def _distribution(rows: Sequence[dict[str, Any]], key: str, groups: Iterable[str]) -> dict[str, Any]:
    out = {}
    for g in groups:
        vals = [r[key] for r in rows if r["novelty_group"] == g and r.get(key) is not None]
        out[g] = C.quantiles(vals) if key != "fraction_rare" else C.quantiles(vals)
    return out


def _group_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(r["novelty_group"] for r in rows)
    return {g: int(counts.get(g, 0)) for g in C.NOVELTY_GROUPS}


def run(*, limit_train: int | None = None) -> dict[str, Any]:
    smiles = read_smiles()
    splits = {name: read_split_rows(name) for name in ("train", "valid", "test")}
    if limit_train is not None:
        splits["train"] = splits["train"][: int(limit_train)]
        print(f"[wl2] DEBUG limit_train={limit_train}")

    support, train_meta = build_train_support(smiles, splits["train"])
    support_sha = save_support(support)

    features: dict[str, list[dict[str, Any]]] = {}
    feature_meta: dict[str, Any] = {}
    for split in ("valid", "test"):
        rows, meta = support_features(smiles, splits[split], support, split=split)
        features[split] = rows
        feature_meta[split] = meta
        C.save_csv(
            VALID_CSV if split == "valid" else TEST_CSV,
            rows,
            columns=[
                "split",
                "split_index",
                "ogb_row",
                "parse_mode",
                "n_heavy_atoms",
                "n_atoms",
                "n_distinct_wl2",
                "fraction_unseen",
                "fraction_rare",
                "median_train_support",
                "min_train_support",
                "max_train_support",
                "novelty_group",
            ],
        )

    # Group summaries (molecule level; labels are not used here).
    groups_meta = {}
    for split in ("valid", "test"):
        rows = features[split]
        groups_meta[split] = {
            "counts": _group_counts(rows),
            "n_heavy_atoms": {
                g: C.quantiles([r["n_heavy_atoms"] for r in rows if r["novelty_group"] == g])
                for g in C.NOVELTY_GROUPS
            },
            "fraction_rare": {
                g: C.quantiles([r["fraction_rare"] for r in rows if r["novelty_group"] == g])
                for g in C.NOVELTY_GROUPS
            },
            "median_train_support": {
                g: C.quantiles([r["median_train_support"] for r in rows if r["novelty_group"] == g])
                for g in C.NOVELTY_GROUPS
            },
            "fraction_unseen_histogram": _hist(
                [r["fraction_unseen"] for r in rows if r.get("fraction_unseen") is not None],
                edges=(0.0, 1e-12, 0.05, 0.10, 0.25, 0.50, 1.0),
            ),
        }

    summary = {
        "schema": "lcrc-pcba-bottleneck-diagnosis-wl2-support-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "rdkit": RDKIT_VERSION,
        "sources": {
            "mapping": C.file_record(C.OGB_MAPPING, "official MolPCBA mapping; smiles column only"),
            "train_split": C.file_record(C.SPLIT_FILES["train"], "official scaffold Train rows"),
            "valid_split": C.file_record(C.SPLIT_FILES["valid"], "official scaffold Valid rows"),
            "test_split": C.file_record(C.SPLIT_FILES["test"], "official scaffold Test rows"),
        },
        "train_support": {**train_meta, "support_npz": C.rel(SUPPORT_PATH), "support_npz_sha256": support_sha},
        "features": feature_meta,
        "groups": groups_meta,
        "thresholds": {
            "SEEN": "fraction_unseen == 0",
            "PARTIAL_NOVEL": f"0 < fraction_unseen <= {C.NOVELTY_PARTIAL_MAX}",
            "HIGH_NOVEL": f"fraction_unseen > {C.NOVELTY_PARTIAL_MAX}",
            "rare_max_train_molecule_frequency": C.RARE_SUPPORT_MAX,
        },
        "labels_read": False,
        "scope_note": "WL2 is a structural-support proxy, not an isomorphism certificate and not the network's true input.",
    }
    C.write_json(SUMMARY_PATH, summary)
    C.write_json(
        STRUCT_SUMMARY,
        {
            "schema": "lcrc-pcba-bottleneck-diagnosis-structure-support-v1",
            "round_id": C.ROUND_ID,
            "result_label": C.RESULT_LABEL,
            "created_utc": C.utcnow(),
            "per_split": groups_meta,
            "train_support_meta": {k: v for k, v in train_meta.items() if k != "definition"},
            "files": {
                "valid": C.rel(VALID_CSV),
                "test": C.rel(TEST_CSV),
                "frequency_summary": C.rel(SUMMARY_PATH),
                "support_npz": C.rel(SUPPORT_PATH),
            },
            "labels_read": False,
        },
    )
    return summary


def _hist(values: Sequence[float], edges: Sequence[float]) -> dict[str, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    labels = [f"[{edges[i]}, {edges[i + 1]})" if i + 1 < len(edges) - 1 else f"[{edges[i]}, {edges[i + 1]}]" for i in range(len(edges) - 1)]
    out = {}
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        if i + 1 == len(edges) - 1:
            mask = (arr >= lo) & (arr <= hi)
        else:
            mask = (arr >= lo) & (arr < hi)
        out[lab] = int(mask.sum())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-train", type=int, default=None, help="debug only; breaks the frozen definition")
    args = parser.parse_args(argv)
    s = run(limit_train=args.limit_train)
    print(
        f"[wl2] train molecules {s['train_support']['n_train_molecules']} "
        f"atoms {s['train_support']['n_train_atoms']} distinct {s['train_support']['n_distinct_wl2']} "
        f"({s['train_support']['build_seconds']:.1f}s)"
    )
    for split in ("valid", "test"):
        print(f"[wl2] {split} groups {s['groups'][split]['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
