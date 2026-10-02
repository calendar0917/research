"""Label-free bounded raw features for the train-only object diagnostic.

The ring features are composition, within-ring co-occurrence and cyclic
bond-pair separation, not complete-ring token IDs or label-derived features.
"""
from __future__ import annotations

import hashlib
import numpy as np
from typed_cycle_reference import chordless_cycles

HASH_WIDTH = 128
BOND_PAIRS = [(a,b) for a in range(4) for b in range(a,4)]
TYPED_RING_WIDTH = 8*64 + 8*5*10 + HASH_WIDTH  # 1040


def _bucket(key):
    payload=",".join(map(str,key)).encode("ascii")
    digest=hashlib.blake2b(payload,digest_size=8,person=b"ringprobe-v1").digest()
    return int.from_bytes(digest,"little") % HASH_WIDTH


def edge_type_counts(graph):
    x=np.zeros(HASH_WIDTH)
    for u,v,b in graph.edges:
        a,c=sorted((graph.atoms[u],graph.atoms[v]))
        x[_bucket((0,a,b,c))]+=1
    return x


def ring_features(graph):
    untyped=np.zeros(8)
    typed=np.zeros(TYPED_RING_WIDTH)
    bond_map=graph.bonds()
    for cycle in chordless_cycles(graph):
        ell=len(cycle)
        size=ell-3
        untyped[size]+=1
        atoms=np.bincount([graph.atoms[v] for v in cycle],minlength=28)
        bonds=[bond_map[tuple(sorted((cycle[t],cycle[(t+1)%ell])))] for t in range(ell)]
        bond_counts=np.bincount(bonds,minlength=4)
        counts=np.concatenate([atoms,bond_counts])
        typed[64*size:64*(size+1)]+=np.concatenate([counts,counts*counts])
        for i in range(ell):
            for j in range(i+1,ell):
                gap=min(j-i,ell-j+i)
                pair=tuple(sorted((bonds[i],bonds[j])))
                typed[512+size*50+(gap-1)*10+BOND_PAIRS.index(pair)]+=1
        for i,b in enumerate(bonds):
            a,c=sorted((graph.atoms[cycle[i]],graph.atoms[cycle[(i+1)%ell]]))
            typed[912+_bucket((ell,a,b,c))]+=1
    return untyped,typed
