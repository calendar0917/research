"""LEGACY cscl-v0 unit signature — reference only, do not use in pipelines.

This module preserves the exact cscl-v0 signature implementation (commit
6549c04 era) so that the before/after representation audit
(``run_cscl_v1.py audit``) can quantify what the fix changed.  It is kept in a
separate file, clearly marked legacy, so the production pipeline can never
import it by accident.

Known defects (reproduced with minimal counterexamples in
``notes/correctness_information_audit_v1.md`` and pinned by regression tests):

* colours are dense-remapped by first-appearance order over the local atom
  ordering, and the signature keeps only the multiset of dense colour ids →
  * relabel variance: the same labelled graph can produce different
    signatures (e.g. an N–C–N chain);
  * chemistry erasure: units differing only in atom types (C–O vs C–N vs
    C–F) or bond types (C–C vs C=C vs C#C) collide.

The v0 pipeline also fed *doubled* (both-direction) PyG edges into
``intra_bonds``; :func:`v0_unit_signature` accepts exactly what the v0 code
passed, so the audit reproduces v0 behaviour faithfully.
"""

from __future__ import annotations

import hashlib

WL_ROUNDS = 4


def v0_unit_signature(kind: int, atoms: list[int], atom_types: list[int], intra_bonds: list[tuple[int, int, int]]) -> str:
    """Bit-exact reproduction of cscl-v0 ``unit_signature`` (legacy, defective)."""
    local = {a: i for i, a in enumerate(sorted(atoms))}
    n = len(atoms)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, bt in intra_bonds:
        iu, iv = local[u], local[v]
        adj[iu].append((iv, int(bt)))
        adj[iv].append((iu, int(bt)))
    colors = [int(atom_types[a]) for a in atoms]
    for _ in range(WL_ROUNDS):
        new_colors: list[str] = []
        for i in range(n):
            neigh = sorted(f"{bt}:{colors[j]}" for j, bt in adj[i])
            new_colors.append(f"{colors[i]}|{'-'.join(neigh)}")
        remap: dict[str, int] = {}
        for s in new_colors:
            if s not in remap:
                remap[s] = len(remap)
        colors = [remap[s] for s in new_colors]
    final = ",".join(sorted(str(c) for c in colors))
    digest = hashlib.sha256(final.encode()).hexdigest()[:16]
    return f"{kind}|{n}|{digest}"
