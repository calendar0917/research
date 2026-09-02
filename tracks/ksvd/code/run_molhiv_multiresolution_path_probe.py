"""Multi-resolution typed-path vocabularies on exact official-train folds.

The previous full-feature path model transferred better than rooted motifs but
still made each word overly specific. This probe compares:
  * full: all 9 atom and all 3 bond categorical features, paths 1..4;
  * core: atom number/charge/hybridisation/aromatic/ring and bond type/conjugation,
    paths 1..4 or 1..5;
  * lean: atom number/aromatic/ring and bond type/conjugation, paths 1..5;
  * full+core: a multi-resolution union.

No official valid/test graph is accessed. A shuffled-node core vocabulary is a
negative structural control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path: sys.path.insert(0, str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes, transformed
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths, path_signature, sparse_from_counters

PROFILES = {
    "full": ((0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 2)),
    "core": ((0, 3, 6, 7, 8), (0, 2)),
    "lean": ((0, 7, 8), (0, 2)),
}


def sha(x: np.ndarray) -> str: return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def fit_eval(xf: sparse.csr_matrix, yf: np.ndarray, xh: sparse.csr_matrix, yh: np.ndarray, c: float) -> tuple[dict[str, Any], np.ndarray]:
    m = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0); m.fit(xf, yf)
    pf=m.predict_proba(xf)[:,1]; ph=m.predict_proba(xh)[:,1]
    return {"trainable_parameters":int(xf.shape[1]+1),"fit_auc":float(roc_auc_score(yf,pf)),"heldout_auc":float(roc_auc_score(yh,ph)),"coefficient_l2":float(np.linalg.norm(m.coef_)),"n_iter":np.asarray(m.n_iter_,dtype=int).tolist()},ph


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a=np.asarray([r["heldout_auc"] for r in rows]); f=np.asarray([r["fit_auc"] for r in rows]); p=np.asarray([r["trainable_parameters"] for r in rows])
    return {"fold_auc":a.tolist(),"mean_auc":float(a.mean()),"sample_std_auc":float(a.std(ddof=1)),"min_fold_auc":float(a.min()),"mean_fit_auc":float(f.mean()),"mean_parameters":float(p.mean())}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--fold-cache",default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz"); ap.add_argument("--min-df",type=int,default=5); ap.add_argument("--output",default="tracks/ksvd/results/molhiv/multiresolution_path_fulltrain_probe_20260729.json"); args=ap.parse_args(); t0=time.time(); repo=Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only(); from ogb.graphproppred import GraphPropPredDataset
    dataset=GraphPropPredDataset(name="ogbg-molhiv",root=str(repo/"data"/"ogb")); split=dataset.get_idx_split(); train_graph_indices=np.asarray(split["train"],dtype=np.int64); forbidden=np.concatenate([np.asarray(split["valid"],dtype=np.int64),np.asarray(split["test"],dtype=np.int64)]); y=np.asarray(dataset.labels).reshape(-1).astype(np.int64)[train_graph_indices]
    pos=np.full(len(dataset),-1,dtype=np.int64); pos[train_graph_indices]=np.arange(len(train_graph_indices))
    with np.load(args.fold_cache,allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"],dtype=np.int64),train_graph_indices): raise ValueError("train mismatch")
        outer=[]
        for fold in range(3):
            fg=np.asarray(z[f"fold_{fold}_train_indices"],dtype=np.int64); hg=np.asarray(z[f"fold_{fold}_valid_indices"],dtype=np.int64); outer.append((pos[fg],pos[hg],fg,hg))
    if np.any(pos[forbidden]>=0): raise AssertionError("forbidden mapping")

    vocabs={name:{} for name in PROFILES}; lengths={name:[] for name in PROFILES}; rows={name:[] for name in PROFILES}
    shuffled_vocab:dict[tuple[int,bytes],int]={}; shuffled_lengths:list[int]=[]; shuffled_rows:list[Counter[int]]=[]
    for row,gi in enumerate(train_graph_indices.tolist()):
        g,_=dataset[int(gi)]; nf=np.asarray(g["node_feat"],dtype=np.int64); ei=np.asarray(g["edge_index"],dtype=np.int64); ef=np.asarray(g["edge_feat"],dtype=np.int64); paths,edge_pos=enumerate_simple_paths(ei,len(nf),5)
        rng=np.random.default_rng(np.uint64(20260729)^np.uint64((int(gi)+1)*0x9E3779B1)); perm=rng.permutation(len(nf))
        for name,(atom_idx,bond_idx) in PROFILES.items():
            atom_labels=[digest(b"A"+feature_bytes(nf[u,list(atom_idx)])) for u in range(len(nf))]
            edge_labels={key:digest(b"B"+feature_bytes(ef[j,list(bond_idx)])) for key,j in edge_pos.items()}
            counts:Counter[int]=Counter(); limit=4 if name=="full" else 5
            for length in range(1,limit+1):
                for nodes in paths[length]:
                    key=(length,path_signature(nodes,atom_labels,edge_labels)); col=vocabs[name].get(key)
                    if col is None: col=len(vocabs[name]); vocabs[name][key]=col; lengths[name].append(length)
                    counts[col]+=1
            rows[name].append(counts)
            if name=="core":
                shuffled_atom=[atom_labels[int(perm[u])] for u in range(len(nf))]; scounts:Counter[int]=Counter()
                for length in range(1,6):
                    for nodes in paths[length]:
                        key=(length,path_signature(nodes,shuffled_atom,edge_labels)); col=shuffled_vocab.get(key)
                        if col is None: col=len(shuffled_vocab); shuffled_vocab[key]=col; shuffled_lengths.append(length)
                        scounts[col]+=1
                shuffled_rows.append(scounts)
        if (row+1)%5000==0: print(f"extracted {row+1}/{len(train_graph_indices)} "+" ".join(f"{k}={len(v)}" for k,v in vocabs.items()),flush=True)
    mats={name:sparse_from_counters(rows[name],len(vocabs[name])) for name in PROFILES}; smat=sparse_from_counters(shuffled_rows,len(shuffled_vocab)); length_arr={name:np.asarray(lengths[name],dtype=np.int8) for name in PROFILES}; slength=np.asarray(shuffled_lengths,dtype=np.int8)
    out:dict[str,Any]={"protocol_id":"molhiv-multiresolution-typed-paths-v1","date":"2026-07-29","scope":"exact official-train only; three scaffold folds","official_valid_evaluations":0,"official_test_evaluations":0,"isolation":"dataset graph items accessed only for official train","config":vars(args),"profiles":{name:{"atom_feature_indices":list(a),"bond_feature_indices":list(b),"raw_vocab":int(mats[name].shape[1])} for name,(a,b) in PROFILES.items()},"raw_shuffled_core_vocab":int(smat.shape[1]),"folds":[],"aggregate":{}}
    all_results:dict[str,list[dict[str,Any]]]={}
    specs=[("full1_4",[("full",4)]),("core1_4",[("core",4)]),("core1_5",[("core",5)]),("lean1_5",[("lean",5)]),("full1_4_plus_core1_5",[("full",4),("core",5)])]
    for fold,(fit,held,fg,hg) in enumerate(outer):
        print(f"outer fold {fold}: fit={len(fit)} held={len(held)}",flush=True); fold_out={"fold":fold,"results":{},"feature_counts":{}}
        blocks:dict[tuple[str,int],tuple[sparse.csr_matrix,sparse.csr_matrix,int]]={}
        for name,maxlen in {(n,l) for _,parts in specs for n,l in parts}:
            df=np.asarray((mats[name][fit]>0).sum(axis=0)).ravel(); cols=np.flatnonzero((df>=args.min_df)&(length_arr[name]<=maxlen)); blocks[(name,maxlen)]=(transformed(mats[name][fit][:,cols],"binary"),transformed(mats[name][held][:,cols],"binary"),int(len(cols)))
        for spec,parts in specs:
            bf=[blocks[p][0] for p in parts]; bh=[blocks[p][1] for p in parts]; xf=bf[0] if len(bf)==1 else sparse.hstack(bf,format="csr"); xh=bh[0] if len(bh)==1 else sparse.hstack(bh,format="csr"); fold_out["feature_counts"][spec]={f"{n}_max{l}":blocks[(n,l)][2] for n,l in parts}
            for c in (0.01,0.03):
                key=f"{spec}_binary_c{c:g}"; rr,ph=fit_eval(xf,y[fit],xh,y[held],c); rr.update({"C":c,"spec":spec,"heldout_probabilities":ph.tolist()}); fold_out["results"][key]=rr; all_results.setdefault(key,[]).append(rr)
        sdf=np.asarray((smat[fit]>0).sum(axis=0)).ravel(); scols=np.flatnonzero((sdf>=args.min_df)&(slength<=5)); key="control_shuffled_core1_5_binary_c0.03"; rr,ph=fit_eval(transformed(smat[fit][:,scols],"binary"),y[fit],transformed(smat[held][:,scols],"binary"),y[held],0.03); rr.update({"C":0.03,"n_features":int(len(scols)),"heldout_probabilities":ph.tolist()}); fold_out["results"][key]=rr; all_results.setdefault(key,[]).append(rr); out["folds"].append(fold_out)
    out["aggregate"]={k:summarize(v) for k,v in all_results.items() if len(v)==3}; out["ranking"]=sorted(({"candidate":k,**v} for k,v in out["aggregate"].items()),key=lambda r:(r["mean_auc"],r["min_fold_auc"]),reverse=True); out["elapsed_sec"]=time.time()-t0; Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps({"output":args.output,"profiles":out["profiles"],"top20":out["ranking"][:20],"elapsed_sec":out["elapsed_sec"]},indent=2))

if __name__=="__main__": main()
