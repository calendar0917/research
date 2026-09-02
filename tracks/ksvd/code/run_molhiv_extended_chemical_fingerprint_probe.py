"""Extended chemical-structure fingerprints on official-train scaffold folds.

This runner is train-only development: official valid/test molecules are not
parsed or fingerprinted.  It tests whether path, atom-pair, torsion, and count
fingerprints provide a more scaffold-stable signal than Morgan bit vectors.
"""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_TRACK=Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path: sys.path.insert(0,str(_TRACK))
from code.data_molhiv import load_molhiv

def sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def rank01(x): return rankdata(np.asarray(x),method='average')/len(x)
def summary(v):
 a=np.asarray(v,float); return {'fold_auc':a.tolist(),'mean_auc':float(a.mean()),'sample_std_auc':float(a.std(ddof=1)),'min_fold_auc':float(a.min())}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold-cache',default='tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz'); ap.add_argument('--bits',type=int,default=2048); ap.add_argument('--output',default='tracks/ksvd/results/molhiv/extended_chemical_fingerprint_fulltrain_probe_20260729.json'); args=ap.parse_args(); t0=time.time()
 repo=Path(__file__).resolve().parents[3]; bundle=load_molhiv(repo/'data'/'ogb',max_graphs=None,seed=0,with_features=False)
 y=np.asarray(bundle.y,dtype=np.int64); orig=np.asarray(bundle.meta['original_indices'],dtype=np.int64); train=np.asarray(bundle.split['train'],dtype=np.int64); forbidden=np.concatenate([bundle.split['valid'],bundle.split['test']]).astype(np.int64)
 with np.load(args.fold_cache,allow_pickle=False) as z:
  if not np.array_equal(orig,np.asarray(z['original_indices'],dtype=np.int64)): raise ValueError('fold cache mismatch')
  folds=[(np.asarray(z[f'fold_{f}_train_indices'],dtype=np.int64),np.asarray(z[f'fold_{f}_valid_indices'],dtype=np.int64)) for f in range(3)]
 if any(set(a.tolist())|set(b.tolist())!=set(train.tolist()) for a,b in folds): raise ValueError('folds do not partition official train')
 from rdkit import Chem,RDLogger
 from rdkit.Chem import rdFingerprintGenerator,MACCSkeys
 RDLogger.DisableLog('rdApp.*'); mapping=pd.read_csv(repo/'data'/'ogb'/'ogbg_molhiv'/'mapping'/'mol.csv.gz'); smiles=mapping.iloc[orig]['smiles'].astype(str).tolist(); mols=[None]*len(smiles); fallback=[]
 for i in train.tolist():
  m=Chem.MolFromSmiles(smiles[int(i)])
  if m is None:
   m=Chem.MolFromSmiles(smiles[int(i)],sanitize=False)
   if m is not None: m.UpdatePropertyCache(strict=False); Chem.GetSymmSSSR(m); fallback.append(int(i))
  if m is None: raise RuntimeError(f'parse failed {i}')
  mols[int(i)]=m
 if any(mols[int(i)] is not None for i in forbidden.tolist()): raise AssertionError('forbidden molecule parsed')
 generators={
  'morgan_bit_r1':('bit',rdFingerprintGenerator.GetMorganGenerator(radius=1,fpSize=args.bits,includeChirality=True)),
  'morgan_bit_r2':('bit',rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=args.bits,includeChirality=True)),
  'morgan_bit_r3':('bit',rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=args.bits,includeChirality=True)),
  'morgan_count_r2':('count',rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=args.bits,includeChirality=True)),
  'morgan_count_r3':('count',rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=args.bits,includeChirality=True)),
  'rdkit_path_bit':('bit',rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=args.bits,minPath=1,maxPath=7,useHs=True,branchedPaths=True)),
  'atom_pair_bit':('bit',rdFingerprintGenerator.GetAtomPairGenerator(fpSize=args.bits,includeChirality=True,minDistance=1,maxDistance=30)),
  'topological_torsion_bit':('bit',rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=args.bits,includeChirality=True,torsionAtomCount=4)),
 }
 out={'protocol_id':'molhiv-extended-chemical-fingerprint-officialtrain-scaffold-v1','date':'2026-07-29','scope':'exact official train internal scaffold folds only','official_valid_evaluations':0,'official_test_evaluations':0,'config':vars(args),'n_train':int(len(train)),'n_train_positive':int(y[train].sum()),'fallback_unsanitized_local_indices':fallback,'variants':{},'aggregate':{},'ensembles':{}}
 c_grid=(0.001,0.003,0.01)
 # Store compact held-out predictions in memory for later fixed rank ensembles.
 predictions={}
 for name,(kind,generator) in generators.items():
  print('fingerprint',name,flush=True); x=np.zeros((len(smiles),args.bits),dtype=np.uint16 if kind=='count' else np.uint8)
  for i in train.tolist():
   arr=generator.GetCountFingerprintAsNumPy(mols[int(i)]) if kind=='count' else generator.GetFingerprintAsNumPy(mols[int(i)])
   if kind=='count': arr=np.minimum(arr,65535).astype(np.uint16)
   x[int(i)]=arr
  if np.any(x[forbidden]): raise AssertionError('forbidden rows nonzero')
  xf=np.log1p(x).astype(np.float32) if kind=='count' else x.astype(np.float32)
  variant={'kind':kind,'dimension':args.bits,'official_train_feature_sha256':sha(x[train]),'transformation':'log1p' if kind=='count' else 'binary','candidates':{}}
  for c in c_grid:
   rows=[]
   for fold,(fit,held) in enumerate(folds):
    model=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=0); model.fit(xf[fit],y[fit]); pfit=model.predict_proba(xf[fit])[:,1]; p=model.predict_proba(xf[held])[:,1]
    rows.append({'fold':fold,'fit_auc':float(roc_auc_score(y[fit],pfit)),'heldout_auc':float(roc_auc_score(y[held],p)),'heldout_indices':held.tolist(),'heldout_y':y[held].astype(int).tolist(),'heldout_probabilities':p.tolist(),'heldout_probability_sha256':sha(p),'coefficient_sha256':sha(np.asarray(model.coef_,dtype=np.float64))})
   key=f'c{c:g}'; variant['candidates'][key]={'folds':rows,**summary([r['heldout_auc'] for r in rows])}
  best=max(variant['candidates'],key=lambda k:(variant['candidates'][k]['mean_auc'],-abs(float(k[1:])-0.003)))
  variant['selected_by_mean_internal_auc']=best; out['variants'][name]=variant; out['aggregate'][name]={'selected_c':float(best[1:]),**{k:v for k,v in variant['candidates'][best].items() if k!='folds'}}
  predictions[name]=[np.asarray(r['heldout_probabilities']) for r in variant['candidates'][best]['folds']]
  print(name,best,out['aggregate'][name]['fold_auc'],out['aggregate'][name]['mean_auc'],flush=True)
 # MACCS separately (167 dimensions).
 name='maccs_bit'; x=np.zeros((len(smiles),167),dtype=np.uint8)
 for i in train.tolist():
  bv=MACCSkeys.GenMACCSKeys(mols[int(i)]); arr=np.zeros(167,dtype=np.int8); from rdkit import DataStructs; DataStructs.ConvertToNumpyArray(bv,arr); x[int(i)]=arr
 xf=x.astype(np.float32); variant={'kind':'bit','dimension':167,'official_train_feature_sha256':sha(x[train]),'transformation':'binary','candidates':{}}
 for c in c_grid:
  rows=[]
  for fold,(fit,held) in enumerate(folds):
   model=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=0); model.fit(xf[fit],y[fit]); pfit=model.predict_proba(xf[fit])[:,1]; p=model.predict_proba(xf[held])[:,1]
   rows.append({'fold':fold,'fit_auc':float(roc_auc_score(y[fit],pfit)),'heldout_auc':float(roc_auc_score(y[held],p)),'heldout_indices':held.tolist(),'heldout_y':y[held].astype(int).tolist(),'heldout_probabilities':p.tolist(),'heldout_probability_sha256':sha(p),'coefficient_sha256':sha(np.asarray(model.coef_,dtype=np.float64))})
  key=f'c{c:g}'; variant['candidates'][key]={'folds':rows,**summary([r['heldout_auc'] for r in rows])}
 best=max(variant['candidates'],key=lambda k:variant['candidates'][k]['mean_auc']); variant['selected_by_mean_internal_auc']=best; out['variants'][name]=variant; out['aggregate'][name]={'selected_c':float(best[1:]),**{k:v for k,v in variant['candidates'][best].items() if k!='folds'}}; predictions[name]=[np.asarray(r['heldout_probabilities']) for r in variant['candidates'][best]['folds']]
 # Fixed equal-rank ensembles: all pairs, plus leave-one-family broad ensembles. No fitted per-fold weights.
 names=list(predictions); pair=[]
 for i,a in enumerate(names):
  for b in names[i+1:]:
   vals=[]
   for f in range(3): vals.append(float(roc_auc_score(np.asarray(out['variants'][a]['candidates'][out['variants'][a]['selected_by_mean_internal_auc']]['folds'][f]['heldout_y']),0.5*rank01(predictions[a][f])+0.5*rank01(predictions[b][f]))))
   pair.append({'members':[a,b],**summary(vals)})
 out['ensembles']['equal_rank_pairs_top30']=sorted(pair,key=lambda r:r['mean_auc'],reverse=True)[:30]
 families={
  'morgan_bits':['morgan_bit_r1','morgan_bit_r2','morgan_bit_r3'],
  'distance_structure':['atom_pair_bit','topological_torsion_bit','rdkit_path_bit'],
  'broad_all':['morgan_bit_r1','morgan_bit_r2','morgan_bit_r3','morgan_count_r2','morgan_count_r3','rdkit_path_bit','atom_pair_bit','topological_torsion_bit','maccs_bit'],
  'compact_diverse':['morgan_bit_r2','rdkit_path_bit','atom_pair_bit','topological_torsion_bit','maccs_bit'],
 }
 for key,members in families.items():
  vals=[]
  for f in range(3):
   yy=np.asarray(out['variants'][members[0]]['candidates'][out['variants'][members[0]]['selected_by_mean_internal_auc']]['folds'][f]['heldout_y']); score=np.mean([rank01(predictions[m][f]) for m in members],axis=0); vals.append(float(roc_auc_score(yy,score)))
  out['ensembles'][key]={'members':members,**summary(vals)}
 out['ranking']=sorted(({'variant':k,**v} for k,v in out['aggregate'].items()),key=lambda r:r['mean_auc'],reverse=True); out['elapsed_sec']=time.time()-t0; Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps({'ranking':out['ranking'],'ensembles':{k:v for k,v in out['ensembles'].items() if k!='equal_rank_pairs_top30'},'top_pairs':out['ensembles']['equal_rank_pairs_top30'][:10],'output':args.output,'elapsed_sec':out['elapsed_sec']},indent=2))
if __name__=='__main__': main()
