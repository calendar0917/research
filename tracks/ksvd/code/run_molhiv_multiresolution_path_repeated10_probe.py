"""Repeated 90/10 scaffold evaluation for full/core path representations.

The earlier three scaffold folds leave out 30--40% of official-train, unlike
the official split where the model is fitted on roughly 80% of all molecules.
This probe uses ten deterministic 10-fold scaffold partitions and takes one
held fold from each seed.  It compares only four predeclared candidates and
never accesses official valid/test graphs.
"""
from __future__ import annotations
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
_TRACK=Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:sys.path.insert(0,str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_stable_exact_motif_probe import digest,feature_bytes
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths,path_signature,sparse_from_counters
FULL_A=tuple(range(9));FULL_B=tuple(range(3));CORE_A=(0,3,6,7,8);CORE_B=(0,2)
def binary(x):
 z=x.astype(np.float32,copy=True);z.data.fill(1.0);return z
def fitauc(xf,yf,xh,yh,c):
 m=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=0);m.fit(xf,yf);return float(roc_auc_score(yh,m.predict_proba(xh)[:,1])),float(roc_auc_score(yf,m.predict_proba(xf)[:,1]))
def summ(rows):
 a=np.asarray([r['heldout_auc'] for r in rows]);f=np.asarray([r['fit_auc'] for r in rows]);return {'auc':a.tolist(),'mean_auc':float(a.mean()),'sample_std_auc':float(a.std(ddof=1)),'min_auc':float(a.min()),'max_auc':float(a.max()),'median_auc':float(np.median(a)),'mean_fit_auc':float(f.mean()),'mean_parameters':float(np.mean([r['parameters'] for r in rows]))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold-cache',default='tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz');ap.add_argument('--min-df',type=int,default=5);ap.add_argument('--repeats',type=int,default=10);ap.add_argument('--output',default='tracks/ksvd/results/molhiv/multiresolution_path_repeated10_fulltrain_probe_20260729.json');args=ap.parse_args();t0=time.time();repo=Path(__file__).resolve().parents[3]
 _patch_torch_load_weights_only();from ogb.graphproppred import GraphPropPredDataset
 ds=GraphPropPredDataset(name='ogbg-molhiv',root=str(repo/'data'/'ogb'));sp=ds.get_idx_split();train=np.asarray(sp['train'],dtype=np.int64);forbidden=np.concatenate([np.asarray(sp['valid'],dtype=np.int64),np.asarray(sp['test'],dtype=np.int64)]);y=np.asarray(ds.labels).reshape(-1).astype(np.int64)[train];pos=np.full(len(ds),-1,dtype=np.int64);pos[train]=np.arange(len(train))
 with np.load(args.fold_cache,allow_pickle=False) as z:
  if not np.array_equal(np.asarray(z['official_train_indices'],dtype=np.int64),train):raise ValueError('train mismatch')
  groups=np.asarray(z['train_scaffold_groups']).astype(str)
 if np.any(pos[forbidden]>=0):raise AssertionError('forbidden')
 voc={'full':{},'core':{}};rows={'full':[],'core':[]}
 for ri,gi in enumerate(train.tolist()):
  g,_=ds[int(gi)];nf=np.asarray(g['node_feat'],dtype=np.int64);ei=np.asarray(g['edge_index'],dtype=np.int64);ef=np.asarray(g['edge_feat'],dtype=np.int64);paths,ep=enumerate_simple_paths(ei,len(nf),5);cnt={'full':Counter(),'core':Counter()}
  for name,aa,bb,maxl in [('full',FULL_A,FULL_B,4),('core',CORE_A,CORE_B,5)]:
   al=[digest(b'A'+feature_bytes(nf[u,list(aa)])) for u in range(len(nf))];el={k:digest(b'B'+feature_bytes(ef[j,list(bb)])) for k,j in ep.items()}
   for length in range(1,maxl+1):
    for nodes in paths[length]:
     key=(length,path_signature(nodes,al,el));col=voc[name].get(key)
     if col is None:col=len(voc[name]);voc[name][key]=col
     cnt[name][col]+=1
  for name in rows:rows[name].append(cnt[name])
  if (ri+1)%5000==0:print(f'extracted {ri+1}/{len(train)} full={len(voc["full"])} core={len(voc["core"])}',flush=True)
 mats={n:sparse_from_counters(rows[n],len(voc[n])) for n in rows};del rows,voc
 specs={'full_c0.01':('full',0.01),'full_c0.03':('full',0.03),'core_c0.03':('core',0.03),'full_plus_core_c0.01':('concat',0.01)};allr={k:[] for k in specs};splits=[]
 for repeat in range(args.repeats):
  seed=20260729+repeat;sg=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=seed);fit,held=next(sg.split(np.zeros(len(train)),y,groups));blocks={};counts={}
  for n,m in mats.items():
   df=np.asarray((m[fit]>0).sum(axis=0)).ravel();cols=np.flatnonzero(df>=args.min_df);blocks[n]=(binary(m[fit][:,cols]),binary(m[held][:,cols]));counts[n]=int(len(cols))
  splits.append({'repeat':repeat,'seed':seed,'n_fit':int(len(fit)),'n_held':int(len(held)),'fit_positive':int(y[fit].sum()),'held_positive':int(y[held].sum()),'n_fit_scaffolds':int(len(set(groups[fit].tolist()))),'n_held_scaffolds':int(len(set(groups[held].tolist()))),'features':counts})
  print(f'repeat {repeat}: fit={len(fit)} held={len(held)} pos={y[held].sum()}',flush=True)
  for key,(kind,c) in specs.items():
   if kind=='concat':xf=sparse.hstack([blocks['full'][0],blocks['core'][0]],format='csr');xh=sparse.hstack([blocks['full'][1],blocks['core'][1]],format='csr')
   else:xf,xh=blocks[kind]
   ah,af=fitauc(xf,y[fit],xh,y[held],c);r={'repeat':repeat,'heldout_auc':ah,'fit_auc':af,'parameters':int(xf.shape[1]+1)};allr[key].append(r);print(f'  {key}: {ah:.6f}',flush=True)
 out={'protocol_id':'molhiv-multiresolution-path-repeated-90-10-scaffold-v1','date':'2026-07-29','scope':'official-train only; ten repeated 10-fold scaffold partitions','official_valid_evaluations':0,'official_test_evaluations':0,'isolation':'dataset graph items accessed only for official train','config':vars(args),'splits':splits,'results':allr,'aggregate':{k:summ(v) for k,v in allr.items()}}
 out['ranking']=sorted(({'candidate':k,**v} for k,v in out['aggregate'].items()),key=lambda r:(r['mean_auc'],r['min_auc']),reverse=True);out['elapsed_sec']=time.time()-t0;Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps({'output':args.output,'ranking':out['ranking'],'elapsed_sec':out['elapsed_sec']},indent=2),flush=True)
if __name__=='__main__':main()
