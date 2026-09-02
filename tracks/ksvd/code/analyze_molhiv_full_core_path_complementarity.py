"""Retrospective diagnostic for the frozen full+core path model and GINE."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
R=Path('tracks/ksvd/results/molhiv')
def rank01(x):return rankdata(np.asarray(x),method='average')/len(x)
vs=[];ts=[];yv=yt=None
for seed in range(5):
 d=json.loads((R/f'frozen_test_gine_h70_seed{seed}.json').read_text());v=np.asarray(d['official_valid_predictions'],float);t=np.asarray(d['official_test_predictions'],float);a=np.asarray(d['official_valid_y'],int);b=np.asarray(d['official_test_y'],int)
 if yv is None:yv,yt=a,b
 else:assert np.array_equal(yv,a) and np.array_equal(yt,b)
 vs.append(v);ts.append(t)
gv,gt=np.mean(vs,axis=0),np.mean(ts,axis=0)
vd=json.loads((R/'multiresolution_path_official_valid_20260729.json').read_text());vr=next(x for x in vd['results'] if x['candidate']=='full1_4_plus_core1_5_binary_c0.01');pv=np.asarray(vr['official_valid_probabilities'],float);yv2=np.asarray(vr['official_valid_y'],int)
td=json.loads((R/'full_core_path_official_test_20260729.json').read_text());pt=np.asarray(td['test_probabilities'],float);yt2=np.asarray(td['test_y'],int);assert np.array_equal(yv,yv2) and np.array_equal(yt,yt2)
rows=[]
for mode in ('probability','rank'):
 av,at,bv,bt=gv,gt,pv,pt
 if mode=='rank':av,at,bv,bt=rank01(av),rank01(at),rank01(bv),rank01(bt)
 for w in np.linspace(0,1,11):rows.append({'mode':mode,'gine_weight':float(w),'path_weight':float(1-w),'valid_auc':float(roc_auc_score(yv,w*av+(1-w)*bv)),'test_auc':float(roc_auc_score(yt,w*at+(1-w)*bt))})
out={'protocol_id':'molhiv-full-core-path-gine-retrospective-complementarity-v1','date':'2026-07-29','status':'retrospective_diagnostic_test_already_exposed','models':{'gine_h70_5seed':{'valid_auc':float(roc_auc_score(yv,gv)),'test_auc':float(roc_auc_score(yt,gt))},'full_core_path':{'valid_auc':float(roc_auc_score(yv,pv)),'test_auc':float(roc_auc_score(yt,pt))}},'grid':rows,'valid_selected':max(rows,key=lambda r:(r['valid_auc'],r['mode'])),'test_oracle_diagnostic_only':max(rows,key=lambda r:r['test_auc'])}
(R/'full_core_path_gine_complementarity_20260729.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['models','valid_selected','test_oracle_diagnostic_only']},indent=2))
