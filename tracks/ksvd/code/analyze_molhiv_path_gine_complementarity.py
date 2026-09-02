"""Retrospective diagnostic of independent path models vs frozen GINE."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
R=Path('tracks/ksvd/results/molhiv')
def sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def rank01(x): return rankdata(np.asarray(x),method='average')/len(x)
def load_gine():
 vs=[];ts=[];yv=yt=None
 for seed in range(5):
  d=json.loads((R/f'frozen_test_gine_h70_seed{seed}.json').read_text()); v=np.asarray(d['official_valid_predictions'],float); t=np.asarray(d['official_test_predictions'],float); a=np.asarray(d['official_valid_y'],int); b=np.asarray(d['official_test_y'],int)
  if yv is None: yv,yt=a,b
  else: assert np.array_equal(yv,a) and np.array_equal(yt,b)
  vs.append(v);ts.append(t)
 return yv,yt,np.mean(vs,axis=0),np.mean(ts,axis=0)
def load_path(valid_file,test_file):
 v=json.loads((R/valid_file).read_text()); t=json.loads((R/test_file).read_text()); s=v.get('selected_candidate',v.get('best_candidate')); pv=np.asarray(s['official_valid_probabilities'],float); yv=np.asarray(s['official_valid_y'],int); pt=np.asarray(t['test_probabilities'],float); yt=np.asarray(t['test_y'],int); return yv,yt,pv,pt

yv,yt,gv,gt=load_gine(); y1,t1,fv,ft=load_path('exact_path_official_valid_20260729.json','exact_path_official_test_20260729.json'); y2,t2,cv,ct=load_path('multiresolution_path_official_valid_20260729.json','core_path_official_test_20260729.json'); assert np.array_equal(yv,y1) and np.array_equal(yv,y2) and np.array_equal(yt,t1) and np.array_equal(yt,t2)
models={'gine_h70_5seed':(gv,gt),'full_path':(fv,ft),'core_path':(cv,ct)}; out={'protocol_id':'molhiv-independent-path-gine-retrospective-complementarity-v1','date':'2026-07-29','status':'retrospective_diagnostic_test_already_exposed','models':{},'pair_blends':{},'three_way':[]}
for n,(v,t) in models.items(): out['models'][n]={'valid_auc':float(roc_auc_score(yv,v)),'test_auc':float(roc_auc_score(yt,t)),'valid_test_gap':float(roc_auc_score(yv,v)-roc_auc_score(yt,t))}
for pn in [('gine_h70_5seed','full_path'),('gine_h70_5seed','core_path'),('full_path','core_path')]:
 a,b=pn; rows=[]
 for mode in ['probability','rank']:
  av,at=models[a]; bv,bt=models[b]
  if mode=='rank': av,at,bv,bt=rank01(av),rank01(at),rank01(bv),rank01(bt)
  for w in np.linspace(0,1,11):
   rows.append({'mode':mode,'weight_first':float(w),'valid_auc':float(roc_auc_score(yv,w*av+(1-w)*bv)),'test_auc':float(roc_auc_score(yt,w*at+(1-w)*bt))})
 out['pair_blends'][f'{a}__{b}']={'valid_selected':sorted(rows,key=lambda r:(-r['valid_auc'],r['mode'],r['weight_first']))[0],'test_oracle':sorted(rows,key=lambda r:-r['test_auc'])[0],'fixed_equal':[r for r in rows if r['weight_first']==0.5]}
for wg in np.linspace(0,1,11):
 for wf in np.linspace(0,1-wg,int(round((1-wg)*10))+1):
  wc=1-wg-wf
  for mode in ['probability','rank']:
   vals=[]
   for v,t in models.values(): vals.append((rank01(v),rank01(t)) if mode=='rank' else (v,t))
   pv=wg*vals[0][0]+wf*vals[1][0]+wc*vals[2][0]; pt=wg*vals[0][1]+wf*vals[1][1]+wc*vals[2][1]
   out['three_way'].append({'mode':mode,'weights':{'gine':float(wg),'full_path':float(wf),'core_path':float(wc)},'valid_auc':float(roc_auc_score(yv,pv)),'test_auc':float(roc_auc_score(yt,pt))})
out['three_way_valid_selected']=sorted(out['three_way'],key=lambda r:(-r['valid_auc'],r['mode']))[0];out['three_way_test_oracle']=sorted(out['three_way'],key=lambda r:-r['test_auc'])[0]
(R/'path_gine_complementarity_20260729.json').write_text(json.dumps(out,indent=2));print(json.dumps({'models':out['models'],'pairs':{k:v['valid_selected'] for k,v in out['pair_blends'].items()},'three_way_valid_selected':out['three_way_valid_selected'],'three_way_test_oracle':out['three_way_test_oracle']},indent=2))
