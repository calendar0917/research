"""Controlled sanity check for relation-regularized dictionary learning.

Patch content contains a strong nuisance factor and a weak relation-relevant
factor.  Relations are determined only by the weak factor.  The check asks
whether relation regularization can preserve that weak factor better than a
matched ordinary dictionary without reading held-out relations at encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ksvd import ksvd
from .relational_ksvd import relational_ksvd, sparse_codes


def make_basis(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Make the shared content coordinates for train and held-out graphs.

    A held-out split must contain new noisy patch instances in the *same*
    feature space.  Re-sampling this basis per split would instead test
    arbitrary rotations of the input coordinates, and makes any relation
    representation needlessly impossible to transfer.
    """
    rng = np.random.default_rng(seed)
    strong = rng.normal(size=(2, dim))
    strong /= np.linalg.norm(strong, axis=1, keepdims=True)
    weak = rng.normal(size=dim)
    weak -= strong.T @ (strong @ weak)
    weak /= np.linalg.norm(weak)
    return strong, weak


def make_split(
    n_graphs: int,
    strong: np.ndarray,
    weak: np.ndarray,
    weak_signal: float,
    noise: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    dim = strong.shape[1]
    graphs = []
    for _ in range(n_graphs):
        # Every graph has identical latent composition: two copies of each
        # (strong group, weak bit) combination.
        types = np.array([(major, bit) for major in range(2) for bit in range(2) for _ in range(2)])
        rng.shuffle(types)
        Y = np.stack(
            [
                strong[major] + (1.0 if bit else -1.0) * weak_signal * weak
                + noise * rng.normal(size=dim)
                for major, bit in types
            ],
            axis=1,
        )
        Y /= np.maximum(np.linalg.norm(Y, axis=0, keepdims=True), 1e-12)
        relation = np.zeros((8, 8, 2), dtype=np.float64)
        for i in range(8):
            for j in range(i + 1, 8):
                relation[i, j, 0] = relation[j, i, 0] = types[i, 1] == types[j, 1]
                relation[i, j, 1] = relation[j, i, 1] = np.all(types[i] == types[j])
        graphs.append((Y, relation))
    return graphs


def flatten(graphs, shuffle: bool, seed: int):
    rng = np.random.default_rng(seed); ys=[]; pairs=[]; labels=[]; offset=0
    for Y,A in graphs:
        p=Y.shape[1]; perm=rng.permutation(p) if shuffle else np.arange(p); A=A[np.ix_(perm,perm)]
        ys.append(Y)
        for i in range(p):
            for j in range(i+1,p): pairs.append((offset+i,offset+j)); labels.append(A[i,j])
        offset+=p
    return np.concatenate(ys,1),np.asarray(pairs),np.asarray(labels)


def pair_data(graphs,D,sparsity):
    Xs=[]; ys=[]
    for Y,A in graphs:
        Z=sparse_codes(D,Y,sparsity).T
        for i in range(len(Z)):
            for j in range(i+1,len(Z)):
                Xs.append(np.r_[np.abs(Z[i]-Z[j]),Z[i]*Z[j]]); ys.append(A[i,j])
    return np.asarray(Xs),np.asarray(ys)


def raw_pair_data(graphs):
    """Upper reference: relation decoding before any dictionary compression."""
    Xs=[]; ys=[]
    for Y,A in graphs:
        Z=Y.T
        for i in range(len(Z)):
            for j in range(i+1,len(Z)):
                Xs.append(np.r_[np.abs(Z[i]-Z[j]),Z[i]*Z[j]]); ys.append(A[i,j])
    return np.asarray(Xs),np.asarray(ys)


def probe(train,valid):
    Xtr,ytr=train; Xva,yva=valid; vals=[]
    for r in range(ytr.shape[1]):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=0))
        m.fit(Xtr,ytr[:,r]); vals.append(float(roc_auc_score(yva[:,r],m.predict_proba(Xva)[:,1])))
    return vals,float(np.mean(vals))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weak-signal',type=float,default=0.15); ap.add_argument('--noise',type=float,default=0.35); ap.add_argument('--relation-weight',type=float,default=2.0); ap.add_argument('--output',required=True); args=ap.parse_args()
    strong, weak = make_basis(24, 31415)
    tr=make_split(160,strong,weak,args.weak_signal,args.noise,0); va=make_split(80,strong,weak,args.weak_signal,args.noise,1); Y,p,L=flatten(tr,False,0); _,_,S=flatten(tr,True,9)
    ordinary,_,_=ksvd(Y,n_atoms=8,T=2,T_min=1,n_iter=4,seed=0)
    relational,ri=relational_ksvd(Y,p,L,n_atoms=8,sparsity=2,relation_weight=args.relation_weight,outer_iter=4,code_steps=30,seed=0)
    shuffled,si=relational_ksvd(Y,p,S,n_atoms=8,sparsity=2,relation_weight=args.relation_weight,outer_iter=4,code_steps=30,seed=0)
    out={'protocol_id':'relational-ksvd-controlled-sanity-v1','config':vars(args),'methods':{},'fit':{'relational':ri,'shuffled':si}}
    out['methods']['raw_content']={'relation_auc':probe(raw_pair_data(tr),raw_pair_data(va))}
    for name,D in [('ordinary',ordinary),('relational',relational),('shuffled',shuffled)]: out['methods'][name]={'relation_auc':probe(pair_data(tr,D,2),pair_data(va,D,2))}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)); print(json.dumps(out['methods'],indent=2))
if __name__=='__main__': main()
