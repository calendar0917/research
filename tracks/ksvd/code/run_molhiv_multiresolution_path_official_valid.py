"""Two predeclared multi-resolution path candidates on official validation.

Advancement gate (declared before this run): compared with the already frozen
full-path baseline valid AUC 0.7935375024495394, require either +0.003 absolute
or valid AUC >= 0.80 before any further official-test evaluation.
Official test graph items are never accessed.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
_TRACK=Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path: sys.path.insert(0,str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes, transformed
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths, path_signature, sparse_from_counters
PROFILES={"full":((0,1,2,3,4,5,6,7,8),(0,1,2),4),"core":((0,3,6,7,8),(0,2),5)}
def sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def encode(dataset:Any,indices:np.ndarray,vocabs:dict[str,dict],add:bool):
 rows={n:[] for n in PROFILES}
 for row,gi in enumerate(indices.tolist()):
  g,_=dataset[int(gi)]; nf=np.asarray(g['node_feat'],dtype=np.int64); ei=np.asarray(g['edge_index'],dtype=np.int64); ef=np.asarray(g['edge_feat'],dtype=np.int64); paths,edge_pos=enumerate_simple_paths(ei,len(nf),5)
  for name,(ai,bi,limit) in PROFILES.items():
   al=[digest(b'A'+feature_bytes(nf[u,list(ai)])) for u in range(len(nf))]; el={k:digest(b'B'+feature_bytes(ef[j,list(bi)])) for k,j in edge_pos.items()}; counts=Counter()
   for length in range(1,limit+1):
    for nodes in paths[length]:
     key=(length,path_signature(nodes,al,el)); col=vocabs[name].get(key)
     if col is None and add: col=len(vocabs[name]); vocabs[name][key]=col
     if col is not None: counts[col]+=1
   rows[name].append(counts)
  if (row+1)%5000==0: print(f'encoded {row+1}/{len(indices)} add={add} '+ ' '.join(f'{n}={len(v)}' for n,v in vocabs.items()),flush=True)
 return rows
def sparse_hashes(x): return {'shape':list(x.shape),'nnz':int(x.nnz),'data_sha256':sha(x.data),'indices_sha256':sha(x.indices),'indptr_sha256':sha(x.indptr)}
def fit(name,xf,yf,xv,yv,vi,c,meta):
 m=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=0); m.fit(xf,yf); pf=m.predict_proba(xf)[:,1]; pv=m.predict_proba(xv)[:,1]
 return {'candidate':name,**meta,'C':c,'trainable_parameters':int(xf.shape[1]+1),'fit_auc':float(roc_auc_score(yf,pf)),'official_valid_auc':float(roc_auc_score(yv,pv)),'coefficient_sha256':sha(np.asarray(m.coef_,dtype=np.float64)),'intercept_sha256':sha(np.asarray(m.intercept_,dtype=np.float64)),'n_iter':np.asarray(m.n_iter_,dtype=int).tolist(),'train_feature_hashes':sparse_hashes(xf),'official_valid_indices':vi.astype(int).tolist(),'official_valid_y':yv.astype(int).tolist(),'official_valid_probabilities':pv.tolist(),'official_valid_probability_sha256':sha(pv)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--min-df',type=int,default=5); ap.add_argument('--baseline-valid',type=float,default=0.7935375024495394); ap.add_argument('--required-gain',type=float,default=0.003); ap.add_argument('--output',default='tracks/ksvd/results/molhiv/multiresolution_path_official_valid_20260729.json'); args=ap.parse_args(); t0=time.time(); repo=Path(__file__).resolve().parents[3]; _patch_torch_load_weights_only(); from ogb.graphproppred import GraphPropPredDataset
 ds=GraphPropPredDataset(name='ogbg-molhiv',root=str(repo/'data'/'ogb')); sp=ds.get_idx_split(); tr=np.asarray(sp['train'],dtype=np.int64); va=np.asarray(sp['valid'],dtype=np.int64); te=np.asarray(sp['test'],dtype=np.int64); labels=np.asarray(ds.labels).reshape(-1).astype(np.int64); yt=labels[tr]; yv=labels[va]
 vocabs={n:{} for n in PROFILES}; rt=encode(ds,tr,vocabs,True); rv=encode(ds,va,vocabs,False)
 if set(te.tolist())&(set(tr.tolist())|set(va.tolist())): raise AssertionError('split overlap')
 mats={}; counts={}
 for name in PROFILES:
  at=sparse_from_counters(rt[name],len(vocabs[name])); av=sparse_from_counters(rv[name],len(vocabs[name])); df=np.asarray((at>0).sum(axis=0)).ravel(); cols=np.flatnonzero(df>=args.min_df); mats[name]=(transformed(at[:,cols],'binary'),transformed(av[:,cols],'binary')); counts[name]=int(len(cols))
 candidates=[('full1_4_plus_core1_5_binary_c0.01',sparse.hstack([mats['full'][0],mats['core'][0]],format='csr'),sparse.hstack([mats['full'][1],mats['core'][1]],format='csr'),0.01,{'representation':'full1_4_plus_core1_5','n_full_features':counts['full'],'n_core_features':counts['core']}),('core1_5_binary_c0.03',mats['core'][0],mats['core'][1],0.03,{'representation':'core1_5','n_full_features':0,'n_core_features':counts['core']})]
 results=[]
 for name,xf,xv,c,meta in candidates:
  r=fit(name,xf,yt,xv,yv,va,c,meta); results.append(r); print(f"{name}: valid={r['official_valid_auc']:.6f} fit={r['fit_auc']:.6f} params={r['trainable_parameters']}",flush=True)
 best=sorted(results,key=lambda r:(-r['official_valid_auc'],r['trainable_parameters']))[0]; threshold=min(0.8,args.baseline_valid+args.required_gain); advance=bool(best['official_valid_auc']>=0.8 or best['official_valid_auc']>=args.baseline_valid+args.required_gain)
 out={'protocol_id':'molhiv-multiresolution-path-official-valid-v1','date':'2026-07-29','status':'official_valid_gate_complete','official_valid_evaluations':len(results),'official_test_evaluations':0,'test_isolation':'dataset graph items accessed only for official train and valid','candidate_policy':'two candidates frozen from official-train three-fold experiment','advancement_gate':{'baseline_valid_auc':args.baseline_valid,'required_gain':args.required_gain,'alternative_absolute_auc':0.8,'effective_minimum':threshold},'advance_to_test':advance,'n_train':int(len(tr)),'n_valid':int(len(va)),'eligible_features':counts,'results':results,'best_candidate':best,'elapsed_sec':time.time()-t0}; Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps({'output':args.output,'best':{'candidate':best['candidate'],'valid_auc':best['official_valid_auc'],'parameters':best['trainable_parameters']},'advance_to_test':advance,'elapsed_sec':out['elapsed_sec']},indent=2))
if __name__=='__main__': main()
