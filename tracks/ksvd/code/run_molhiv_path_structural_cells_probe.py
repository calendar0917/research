"""Independent typed-path model augmented with branch and cycle cells.

Full-feature simple paths (length 1..4) are the base representation. Two small,
coarser structural vocabularies add information paths cannot express directly:
  * star cells: a center atom plus the unordered set of all incident typed
    neighbour bonds/atoms, explicitly representing branching;
  * cycle cells: canonical typed sequences for a deterministic cycle basis,
    explicitly representing ring closure.
Core atom/bond profiles are used for transferability. Official valid/test graph
items are never accessed. A within-molecule atom-assignment shuffle is the
negative structural control.
"""
from __future__ import annotations
import argparse, hashlib, json, struct, sys, time
from collections import Counter
from pathlib import Path
from typing import Any
import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
_TRACK=Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path: sys.path.insert(0,str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_stable_exact_motif_probe import digest,feature_bytes,transformed
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths,path_signature,sparse_from_counters
FULL_A=(0,1,2,3,4,5,6,7,8); FULL_B=(0,1,2); CORE_A=(0,3,6,7,8); CORE_B=(0,2)
def sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def star_signature(u:int,nbrs:list[set[int]],atom:list[bytes],edges:dict[tuple[int,int],bytes])->bytes:
 msg=[]
 for v in nbrs[u]: msg.append(edges[(u,v) if u<v else (v,u)]+atom[v])
 msg.sort(); return digest(b'S'+atom[u]+struct.pack('<H',len(msg))+b''.join(msg))
def cycle_signature(cycle:list[int],atom:list[bytes],edges:dict[tuple[int,int],bytes])->bytes:
 n=len(cycle); variants=[]
 for direction in (1,-1):
  seq=cycle if direction==1 else list(reversed(cycle))
  for start in range(n):
   rot=seq[start:]+seq[:start]; p=bytearray(b'C'+bytes([n]))
   for i,u in enumerate(rot):
    v=rot[(i+1)%n]; p+=atom[u]; p+=edges[(u,v) if u<v else (v,u)]
   variants.append(bytes(p))
 return digest(min(variants))
def topology(edge_index:np.ndarray,n:int):
 nbrs=[set() for _ in range(n)]; G=nx.Graph(); G.add_nodes_from(range(n)); first={}
 for j in range(edge_index.shape[1]):
  u,v=int(edge_index[0,j]),int(edge_index[1,j])
  if u==v: continue
  key=(u,v) if u<v else (v,u)
  if key not in first: first[key]=j; G.add_edge(*key)
  nbrs[u].add(v);nbrs[v].add(u)
 cycles=nx.cycle_basis(G,root=min(G.nodes) if n else None)
 return nbrs,first,cycles
def fit_eval(xf,yf,xh,yh,c):
 m=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=0);m.fit(xf,yf);pf=m.predict_proba(xf)[:,1];ph=m.predict_proba(xh)[:,1]
 return {'trainable_parameters':int(xf.shape[1]+1),'fit_auc':float(roc_auc_score(yf,pf)),'heldout_auc':float(roc_auc_score(yh,ph)),'coefficient_l2':float(np.linalg.norm(m.coef_)),'n_iter':np.asarray(m.n_iter_,dtype=int).tolist()},ph
def summarize(rs):
 a=np.asarray([r['heldout_auc'] for r in rs]);f=np.asarray([r['fit_auc'] for r in rs]);p=np.asarray([r['trainable_parameters'] for r in rs]);return {'fold_auc':a.tolist(),'mean_auc':float(a.mean()),'sample_std_auc':float(a.std(ddof=1)),'min_fold_auc':float(a.min()),'mean_fit_auc':float(f.mean()),'mean_parameters':float(p.mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold-cache',default='tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz');ap.add_argument('--min-df',type=int,default=5);ap.add_argument('--output',default='tracks/ksvd/results/molhiv/path_structural_cells_fulltrain_probe_20260729.json');args=ap.parse_args();t0=time.time();repo=Path(__file__).resolve().parents[3];_patch_torch_load_weights_only();from ogb.graphproppred import GraphPropPredDataset
 ds=GraphPropPredDataset(name='ogbg-molhiv',root=str(repo/'data'/'ogb'));sp=ds.get_idx_split();train=np.asarray(sp['train'],dtype=np.int64);forbidden=np.concatenate([np.asarray(sp['valid'],dtype=np.int64),np.asarray(sp['test'],dtype=np.int64)]);y=np.asarray(ds.labels).reshape(-1).astype(np.int64)[train];pos=np.full(len(ds),-1,dtype=np.int64);pos[train]=np.arange(len(train))
 with np.load(args.fold_cache,allow_pickle=False) as z:
  if not np.array_equal(np.asarray(z['official_train_indices'],dtype=np.int64),train):raise ValueError('train mismatch')
  outer=[]
  for fold in range(3):
   fg=np.asarray(z[f'fold_{fold}_train_indices'],dtype=np.int64);hg=np.asarray(z[f'fold_{fold}_valid_indices'],dtype=np.int64);outer.append((pos[fg],pos[hg]))
 if np.any(pos[forbidden]>=0):raise AssertionError('forbidden')
 voc={n:{} for n in ['path','star','cycle','shuffle_star','shuffle_cycle']};rows={n:[] for n in voc};cycle_counts=[]
 for row,gi in enumerate(train.tolist()):
  g,_=ds[int(gi)];nf=np.asarray(g['node_feat'],dtype=np.int64);ei=np.asarray(g['edge_index'],dtype=np.int64);ef=np.asarray(g['edge_feat'],dtype=np.int64);paths,ep=enumerate_simple_paths(ei,len(nf),4);nbrs,first,cycles=topology(ei,len(nf));cycle_counts.append(len(cycles))
  fal=[digest(b'A'+feature_bytes(nf[u,list(FULL_A)])) for u in range(len(nf))];fel={k:digest(b'B'+feature_bytes(ef[j,list(FULL_B)])) for k,j in ep.items()};cal=[digest(b'A'+feature_bytes(nf[u,list(CORE_A)])) for u in range(len(nf))];cel={k:digest(b'B'+feature_bytes(ef[j,list(CORE_B)])) for k,j in first.items()};rng=np.random.default_rng(np.uint64(20260729)^np.uint64((int(gi)+1)*0x9E3779B1));perm=rng.permutation(len(nf));sal=[cal[int(perm[u])] for u in range(len(nf))]
  counts={n:Counter() for n in voc}
  for length in range(1,5):
   for nodes in paths[length]:
    sig=path_signature(nodes,fal,fel);key=(length,sig);col=voc['path'].get(key)
    if col is None:col=len(voc['path']);voc['path'][key]=col
    counts['path'][col]+=1
  for u in range(len(nf)):
   for name,al in [('star',cal),('shuffle_star',sal)]:
    sig=star_signature(u,nbrs,al,cel);col=voc[name].get(sig)
    if col is None:col=len(voc[name]);voc[name][sig]=col
    counts[name][col]+=1
  for cyc in cycles:
   for name,al in [('cycle',cal),('shuffle_cycle',sal)]:
    sig=cycle_signature(cyc,al,cel);key=(len(cyc),sig);col=voc[name].get(key)
    if col is None:col=len(voc[name]);voc[name][key]=col
    counts[name][col]+=1
  for n in voc:rows[n].append(counts[n])
  if (row+1)%5000==0:print(f"extracted {row+1}/{len(train)} "+' '.join(f'{n}={len(v)}' for n,v in voc.items()),flush=True)
 mats={n:sparse_from_counters(rows[n],len(voc[n])) for n in voc};out={'protocol_id':'molhiv-full-path-core-structural-cells-v1','date':'2026-07-29','scope':'exact official-train only; three scaffold folds','official_valid_evaluations':0,'official_test_evaluations':0,'isolation':'dataset graph items accessed only for official train','config':vars(args),'raw_vocab':{n:int(m.shape[1]) for n,m in mats.items()},'mean_cycle_basis_size':float(np.mean(cycle_counts)),'folds':[],'aggregate':{}};allr={}
 for fold,(fit,held) in enumerate(outer):
  print(f'outer fold {fold}: fit={len(fit)} held={len(held)}',flush=True);blocks={};counts={}
  for n,m in mats.items():
   df=np.asarray((m[fit]>0).sum(axis=0)).ravel();cols=np.flatnonzero(df>=args.min_df);blocks[n]=(transformed(m[fit][:,cols],'binary'),transformed(m[held][:,cols],'binary'));counts[n]=int(len(cols))
  specs={'path':['path'],'path_plus_star':['path','star'],'path_plus_cycle':['path','cycle'],'path_plus_star_cycle':['path','star','cycle'],'control_path_plus_shuffled_cells':['path','shuffle_star','shuffle_cycle'],'star_cycle_only':['star','cycle']};fo={'fold':fold,'feature_counts':counts,'results':{}}
  for spec,names in specs.items():
   xf=blocks[names[0]][0] if len(names)==1 else sparse.hstack([blocks[n][0] for n in names],format='csr');xh=blocks[names[0]][1] if len(names)==1 else sparse.hstack([blocks[n][1] for n in names],format='csr')
   cs=(0.01,0.03) if spec in ['path','path_plus_star','path_plus_cycle','path_plus_star_cycle'] else (0.03,)
   for c in cs:
    key=f'{spec}_binary_c{c:g}';rr,ph=fit_eval(xf,y[fit],xh,y[held],c);rr.update({'C':c,'spec':spec,'heldout_probabilities':ph.tolist()});fo['results'][key]=rr;allr.setdefault(key,[]).append(rr)
  out['folds'].append(fo)
 out['aggregate']={k:summarize(v) for k,v in allr.items() if len(v)==3};out['ranking']=sorted(({'candidate':k,**v} for k,v in out['aggregate'].items()),key=lambda r:(r['mean_auc'],r['min_fold_auc']),reverse=True);out['elapsed_sec']=time.time()-t0;Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps({'output':args.output,'raw_vocab':out['raw_vocab'],'top20':out['ranking'][:20],'elapsed_sec':out['elapsed_sec']},indent=2))
if __name__=='__main__':main()
