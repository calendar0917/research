"""Frozen retrospective official-test evaluation for core typed paths."""
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
from code.run_molhiv_stable_exact_motif_probe import digest,feature_bytes,transformed
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths,path_signature,sparse_from_counters
def sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def sparse_hashes(x): return {'shape':list(x.shape),'nnz':int(x.nnz),'data_sha256':sha(x.data),'indices_sha256':sha(x.indices),'indptr_sha256':sha(x.indptr)}
def encode(ds:Any,indices:np.ndarray,vocab:dict,add:bool,ai:list[int],bi:list[int],maxlen:int):
 rows=[]
 for row,gi in enumerate(indices.tolist()):
  g,_=ds[int(gi)]; nf=np.asarray(g['node_feat'],dtype=np.int64); ei=np.asarray(g['edge_index'],dtype=np.int64); ef=np.asarray(g['edge_feat'],dtype=np.int64); paths,ep=enumerate_simple_paths(ei,len(nf),maxlen); al=[digest(b'A'+feature_bytes(nf[u,ai])) for u in range(len(nf))]; el={k:digest(b'B'+feature_bytes(ef[j,bi])) for k,j in ep.items()}; counts=Counter()
  for length in range(1,maxlen+1):
   for nodes in paths[length]:
    key=(length,path_signature(nodes,al,el)); col=vocab.get(key)
    if col is None and add: col=len(vocab); vocab[key]=col
    if col is not None: counts[col]+=1
  rows.append(counts)
  if (row+1)%5000==0: print(f'encoded {row+1}/{len(indices)} add={add} vocab={len(vocab)}',flush=True)
 return rows
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--freeze',default='tracks/ksvd/results/molhiv/core_path_official_test_freeze_v1.json'); ap.add_argument('--output',default='tracks/ksvd/results/molhiv/core_path_official_test_20260729.json'); args=ap.parse_args(); t0=time.time(); fp=Path(args.freeze); actual=hashlib.sha256(fp.read_bytes()).hexdigest(); expected=Path(args.freeze+'.sha256').read_text().split()[0]
 if actual!=expected: raise ValueError('freeze hash mismatch')
 fr=json.loads(fp.read_text()); cfg=fr['config']; repo=Path(__file__).resolve().parents[3]; _patch_torch_load_weights_only(); from ogb.graphproppred import GraphPropPredDataset
 ds=GraphPropPredDataset(name='ogbg-molhiv',root=str(repo/'data'/'ogb')); sp=ds.get_idx_split(); tr=np.asarray(sp['train'],dtype=np.int64); va=np.asarray(sp['valid'],dtype=np.int64); te=np.asarray(sp['test'],dtype=np.int64); labels=np.asarray(ds.labels).reshape(-1).astype(np.int64); yt=labels[tr]; ye=labels[te]
 if set(va.tolist())&(set(tr.tolist())|set(te.tolist())): raise AssertionError('split overlap')
 vocab={}; rt=encode(ds,tr,vocab,True,cfg['atom_feature_indices'],cfg['bond_feature_indices'],cfg['max_path_length']); re=encode(ds,te,vocab,False,cfg['atom_feature_indices'],cfg['bond_feature_indices'],cfg['max_path_length']); at=sparse_from_counters(rt,len(vocab)); ae=sparse_from_counters(re,len(vocab)); df=np.asarray((at>0).sum(axis=0)).ravel(); cols=np.flatnonzero(df>=cfg['min_df']); xt=transformed(at[:,cols],cfg['mode']); xe=transformed(ae[:,cols],cfg['mode']); hashes=sparse_hashes(xt)
 if hashes!=fr['expected_train_feature_hashes']: raise ValueError(f'train features mismatch: {hashes} != {fr["expected_train_feature_hashes"]}')
 m=LogisticRegression(C=cfg['C'],class_weight=cfg['class_weight'],solver=cfg['solver'],max_iter=cfg['max_iter'],random_state=cfg['random_state']); m.fit(xt,yt); ch=sha(np.asarray(m.coef_,dtype=np.float64)); ih=sha(np.asarray(m.intercept_,dtype=np.float64)); ni=np.asarray(m.n_iter_,dtype=int).tolist()
 if ch!=fr['expected_coefficient_sha256'] or ih!=fr['expected_intercept_sha256'] or ni!=fr['expected_n_iter']: raise ValueError('classifier reproduction mismatch')
 pt=m.predict_proba(xt)[:,1]; pe=m.predict_proba(xe)[:,1]; out={'protocol_id':'molhiv-core-path-official-test-v1','date':'2026-07-29','status':'retrospective_controlled_frozen_evaluation_complete','freeze_manifest':args.freeze,'freeze_manifest_sha256':actual,'selected_candidate':fr['selected_candidate'],'fit_split':'exact official train','evaluation_split':'exact official test','official_valid_evaluations_in_this_runner':0,'official_test_evaluations_in_this_runner':1,'historical_test_disclosure':fr['disclosure'],'valid_isolation':'official valid graph items were not accessed','n_train':int(len(tr)),'n_test':int(len(te)),'n_train_positive':int(yt.sum()),'n_test_positive':int(ye.sum()),'n_path_features':int(len(cols)),'trainable_parameters':int(xt.shape[1]+1),'fit_auc':float(roc_auc_score(yt,pt)),'official_test_auc':float(roc_auc_score(ye,pe)),'test_indices':te.astype(int).tolist(),'test_y':ye.astype(int).tolist(),'test_probabilities':pe.tolist(),'test_probability_sha256':sha(pe),'train_feature_hashes':hashes,'test_feature_hashes':sparse_hashes(xe),'coefficient_sha256':ch,'intercept_sha256':ih,'n_iter':ni,'elapsed_sec':time.time()-t0}; Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps({'output':args.output,'freeze_sha256':actual,'official_valid_auc_frozen_selection':fr['selected_candidate']['official_valid_auc'],'official_test_auc':out['official_test_auc'],'parameters':out['trainable_parameters'],'elapsed_sec':out['elapsed_sec']},indent=2))
if __name__=='__main__': main()
