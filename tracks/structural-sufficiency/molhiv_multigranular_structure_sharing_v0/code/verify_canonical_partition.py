"""Close the canonical-key question: is ``canonical_key`` the same *partition* as
the naive "minimum over all node permutations" reference?

The two keys are different canonical forms -- ``canonical_key`` only permutes
inside 1-WL cells, so its tuple value can legitimately be larger than the
unrestricted minimum.  Comparing tuples therefore says nothing about type
identity.  The property that *does* matter is that both induce the same
equivalence classes:

    canonical_key(a) == canonical_key(b)  <=>  brute_force_key(a) == brute_force_key(b)

which is what this script tests, for both granularities, on a fixed sample of
molecules.  Written to ``analysis/canonical_partition_check.json``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

import msds_common as C
import msds_data as D
import msds_enum as E

N_MOLECULES = 160
MAX_INSTANCES_PER_MOLECULE = 140


def main() -> None:
    t0 = time.perf_counter()
    molecules = D.load_molecules()
    order = np.argsort(molecules.n_atoms, kind="stable")
    eligible = [int(g) for g in order if int(molecules.n_atoms[g]) >= 6]
    step = max(1, len(eligible) // N_MOLECULES)
    sample = sorted(eligible[::step][:N_MOLECULES])
    rng = np.random.default_rng(C.SEED)

    report: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "phase": "canonical-key partition check (Phase A supplementary verification)",
        "sampled_molecules": len(sample),
        "sample_rule": (f"every {step}-th molecule of the size-sorted eligible list (n_atoms>=6); at most "
                        f"{MAX_INSTANCES_PER_MOLECULE} instances per molecule drawn uniformly at random (seed {C.SEED})"),
        "n_atoms_sampled": {
            "min": int(min(molecules.n_atoms[g] for g in sample)),
            "median": float(np.median([int(molecules.n_atoms[g]) for g in sample])),
            "max": int(max(molecules.n_atoms[g] for g in sample)),
        },
        "instances_per_molecule_cap": MAX_INSTANCES_PER_MOLECULE,
        "granularities": {},
    }
    for gran, atom_cols, bond_cols in (
        ("COARSE", C.COARSE_ATOM_COLUMNS, C.COARSE_BOND_COLUMNS),
        ("FULL", tuple(range(9)), (0, 1, 2)),
    ):
        canon_of: list[tuple] = []
        brute_of: list[tuple] = []
        raw_equal = 0
        for gid in sample:
            n = int(molecules.n_atoms[gid])
            bu, bv, bf = molecules.bonds(gid)
            atom = molecules.atoms(gid)
            adj: dict[int, dict[int, tuple]] = defaultdict(dict)
            for a, b, f in zip(bu.tolist(), bv.tolist(), bf.tolist()):
                adj[a][b] = tuple(int(x) for x in f)
                adj[b][a] = tuple(int(x) for x in f)
            instances = list(E.instances_for_molecule(n, bu, bv))
            if len(instances) > MAX_INSTANCES_PER_MOLECULE:
                pick = np.sort(rng.choice(len(instances), size=MAX_INSTANCES_PER_MOLECULE, replace=False))
                instances = [instances[int(i)] for i in pick]
            for fam_idx, nodes in instances:
                bond_attr = {(a, b): adj[a][b] for a in nodes for b in nodes if b > a and b in adj[a]}
                ck = E.canonical_key(nodes, atom, bond_attr, atom_cols, bond_cols)
                bk = E.brute_force_key(nodes, atom, bond_attr, atom_cols, bond_cols)
                canon_of.append(ck)
                brute_of.append(bk)
                raw_equal += int(ck == bk)
        # partition comparison
        c2b: dict[tuple, set] = defaultdict(set)
        b2c: dict[tuple, set] = defaultdict(set)
        for ck, bk in zip(canon_of, brute_of):
            c2b[ck].add(bk)
            b2c[bk].add(ck)
        max_class_inflation = max(len(v) for v in b2c.values())
        merged = sum(1 for v in c2b.values() if len(v) > 1)
        report["granularities"][gran] = {
            "n_instances": len(canon_of),
            "n_canonical_types": len(c2b),
            "n_brute_force_types": len(b2c),
            "same_number_of_types": len(c2b) == len(b2c),
            "partitions_identical": merged == 0 and max_class_inflation == 1,
            "canonical_types_merging_distinct_brute_types": merged,
            "max_brute_types_inside_one_canonical_type": max_class_inflation,
            "max_canonical_types_inside_one_brute_type": max(len(v) for v in c2b.values()),
            "raw_tuple_equality_rate": raw_equal / max(len(canon_of), 1),
            "note": "raw_tuple_equality is expected to be < 1: different canonical forms, same partition",
        }
        print(f"[{gran}] {report['granularities'][gran]}", flush=True)
    report["seconds"] = time.perf_counter() - t0
    report["git"] = C.git_state()
    report["environment"] = C.environment()
    C.write_json(C.ANALYSIS / "canonical_partition_check.json", report)
    ok = all(v["partitions_identical"] for v in report["granularities"].values())
    print(f"[check] partitions identical: {ok} in {report['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
