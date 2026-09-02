"""Scaffold-jackknife coefficient stabilization for the exact path model.

The representation and final capacity remain identical to the exact full-path
linear model.  Multiple leave-scaffold-part-out logistic fits are used only to
estimate a more stable *single* coefficient vector.  Coefficients are averaged
back into one predictor; no multi-model inference or GNN is used.

All feature filtering and all submodel fits occur inside each outer official-
train scaffold fold.  Official valid/test graphs are never accessed.
"""
from __future__ import annotations

import argparse
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
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths, path_signature, sparse_from_counters


def binary(x: sparse.csr_matrix) -> sparse.csr_matrix:
    z=x.astype(np.float32,copy=True); z.data.fill(1.0); return z


def fit_lr(x: sparse.csr_matrix, y: np.ndarray, c: float=0.03) -> LogisticRegression:
    m=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=0)
    m.fit(x,y); return m


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y,np.asarray(score).ravel()))


def summary(rows: list[dict[str,Any]]) -> dict[str,Any]:
    a=np.asarray([r['heldout_auc'] for r in rows],dtype=np.float64)
    f=np.asarray([r['fit_auc'] for r in rows],dtype=np.float64)
    return {'fold_auc':a.tolist(),'mean_auc':float(a.mean()),'sample_std_auc':float(a.std(ddof=1)),'min_fold_auc':float(a.min()),'mean_fit_auc':float(f.mean()),'mean_parameters':float(np.mean([r['parameters'] for r in rows]))}


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold-cache',default='tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz')
    ap.add_argument('--min-df',type=int,default=5)
    ap.add_argument('--output',default='tracks/ksvd/results/molhiv/path_scaffold_jackknife_fulltrain_probe_20260729.json')
    args=ap.parse_args();t0=time.time();repo=Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only();from ogb.graphproppred import GraphPropPredDataset
    ds=GraphPropPredDataset(name='ogbg-molhiv',root=str(repo/'data'/'ogb'));sp=ds.get_idx_split();train=np.asarray(sp['train'],dtype=np.int64)
    forbidden=np.concatenate([np.asarray(sp['valid'],dtype=np.int64),np.asarray(sp['test'],dtype=np.int64)])
    y=np.asarray(ds.labels).reshape(-1).astype(np.int64)[train];pos=np.full(len(ds),-1,dtype=np.int64);pos[train]=np.arange(len(train))
    with np.load(args.fold_cache,allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z['official_train_indices'],dtype=np.int64),train):raise ValueError('train mismatch')
        groups=np.asarray(z['train_scaffold_groups']).astype(str);outer=[]
        for fold in range(3):
            outer.append((pos[np.asarray(z[f'fold_{fold}_train_indices'],dtype=np.int64)],pos[np.asarray(z[f'fold_{fold}_valid_indices'],dtype=np.int64)]))
    if np.any(pos[forbidden]>=0):raise AssertionError('forbidden contamination')

    vocab={};rows=[]
    for row,gi in enumerate(train.tolist()):
        g,_=ds[int(gi)];nf=np.asarray(g['node_feat'],dtype=np.int64);ei=np.asarray(g['edge_index'],dtype=np.int64);ef=np.asarray(g['edge_feat'],dtype=np.int64)
        paths,ep=enumerate_simple_paths(ei,len(nf),4);al=[digest(b'A'+feature_bytes(nf[u])) for u in range(len(nf))];el={k:digest(b'B'+feature_bytes(ef[j])) for k,j in ep.items()};cnt=Counter()
        for length in range(1,5):
            for nodes in paths[length]:
                key=(length,path_signature(nodes,al,el));col=vocab.get(key)
                if col is None:col=len(vocab);vocab[key]=col
                cnt[col]+=1
        rows.append(cnt)
        if (row+1)%5000==0:print(f'extracted {row+1}/{len(train)} vocab={len(vocab)}',flush=True)
    x=sparse_from_counters(rows,len(vocab));del rows,vocab

    out={'protocol_id':'molhiv-exact-path-scaffold-jackknife-coefficient-stability-v1','date':'2026-07-29','scope':'official-train only; three outer scaffold folds','official_valid_evaluations':0,'official_test_evaluations':0,'isolation':'dataset graph items accessed only for official train','representation':'exact full OGB-feature paths length1-4, binary','final_predictor':'one averaged coefficient vector, not multi-model inference','config':vars(args),'raw_vocab':int(x.shape[1]),'folds':[],'aggregate':{}}
    allr={}
    for fold,(fit,held) in enumerate(outer):
        df=np.asarray((x[fit]>0).sum(axis=0)).ravel();cols=np.flatnonzero(df>=args.min_df);xf=binary(x[fit][:,cols]);xh=binary(x[held][:,cols]);yf=y[fit];yh=y[held]
        base=fit_lr(xf,yf);wb=base.coef_.ravel().astype(np.float64);bb=float(base.intercept_[0]);base_fit=np.asarray(xf@wb).ravel()+bb;base_held=np.asarray(xh@wb).ravel()+bb
        fold_out={'fold':fold,'n_fit':int(len(fit)),'n_held':int(len(held)),'n_features':int(len(cols)),'results':{},'jackknife':{}}
        def rec(name,w,b,meta=None):
            sf=np.asarray(xf@w).ravel()+b;sh=np.asarray(xh@w).ravel()+b;r={'fit_auc':auc(yf,sf),'heldout_auc':auc(yh,sh),'parameters':int(len(w)+1),'coefficient_l2':float(np.linalg.norm(w))}
            if meta:r.update(meta)
            fold_out['results'][name]=r;allr.setdefault(name,[]).append(r);print(f'  {name}: {r["heldout_auc"]:.6f}',flush=True)
        print(f'outer fold {fold}: fit={len(fit)} held={len(held)} features={len(cols)}',flush=True);rec('baseline_fullfit_c0.03',wb,bb)
        for n_splits in (5,10):
            splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=20260729+100*fold+n_splits)
            ws=[];bs=[];submeta=[]
            for j,(keep,drop) in enumerate(splitter.split(np.zeros(len(fit)),yf,groups[fit])):
                m=fit_lr(xf[keep],yf[keep]);ws.append(m.coef_.ravel().astype(np.float64));bs.append(float(m.intercept_[0]));submeta.append({'submodel':j,'n_fit':int(len(keep)),'n_drop':int(len(drop)),'n_fit_positive':int(yf[keep].sum()),'n_drop_positive':int(yf[drop].sum())})
            W=np.stack(ws);wm=W.mean(axis=0);bm=float(np.mean(bs));sd=W.std(axis=0,ddof=1)
            name=f'jackknife{n_splits}_coefficient_mean';rec(name,wm,bm,{'n_subfits':n_splits,'coefficient_mean_std':float(sd.mean()),'coefficient_median_std':float(np.median(sd))})
            stability=np.abs(wm)/(np.abs(wm)+sd+1e-8);wshr=wm*stability
            rec(f'jackknife{n_splits}_stability_shrunk',wshr,bm,{'n_subfits':n_splits,'mean_stability':float(stability.mean())})
            rec(f'fullfit_half_jackknife{n_splits}_half',0.5*wb+0.5*wm,0.5*bb+0.5*bm,{'n_subfits':n_splits})
            fold_out['jackknife'][str(n_splits)]={'subfits':submeta,'mean_pairwise_coefficient_std':float(sd.mean()),'mean_stability':float(stability.mean())}
        out['folds'].append(fold_out)
    out['aggregate']={k:summary(v) for k,v in allr.items()};out['ranking']=sorted(({'candidate':k,**v} for k,v in out['aggregate'].items()),key=lambda r:(r['mean_auc'],r['min_fold_auc']),reverse=True);out['elapsed_sec']=time.time()-t0
    Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps({'output':args.output,'ranking':out['ranking'],'elapsed_sec':out['elapsed_sec']},indent=2),flush=True)
if __name__=='__main__':main()
