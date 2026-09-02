"""Retrospective official split complementarity after frozen Morgan test run.

Test labels are already exposed in this repository.  Test-side oracle rows are
strictly diagnostic and must never be presented as model-selection results.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path: sys.path.insert(0, str(_CODE))
from analyze_molhiv_official_model_complementarity import _family, _old_extractor, _result_extractor


def rank01(x): return rankdata(np.asarray(x), method='average') / len(x)
def auc(y,x): return float(roc_auc_score(y,x))
def choose(rows):
    return max(rows,key=lambda r:(r['valid_auc'], -(abs(r['morgan_weight']-0.2))))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-dir',default='tracks/ksvd/results/molhiv'); ap.add_argument('--output',default='tracks/ksvd/results/molhiv/morgan_official_complementarity_20260729.json'); a=ap.parse_args()
    root=Path(a.results_dir)
    families={}
    specs=[
      ('gine_h70_5seed',sorted(glob.glob(str(root/'frozen_test_gine_h70_seed*.json'))),_old_extractor),
      ('ksvd_node_token_5seed',sorted(glob.glob(str(root/'frozen_test_ksvd_seed*.json'))),_old_extractor),
      ('stable_realprototype_3bank',sorted(glob.glob(str(root/'stable_realproto_full_official_bank*_seed0.json'))),_result_extractor('stable_random_node_mil')),
    ]
    for name,paths,ext in specs: families[name]=_family(name,paths,ext)
    vd=json.loads((root/'chemical_fingerprint_official_valid_20260729.json').read_text())['candidates']['morgan_r2_c0.01']
    td=json.loads((root/'chemical_fingerprint_official_test_20260729.json').read_text())
    families['morgan_r2_c0.01']={
      'official_valid':{'labels':np.asarray(vd['official_valid_y']), 'probabilities':np.asarray(vd['official_valid_probabilities']), 'auc':vd['official_valid_auc']},
      'official_test':{'labels':np.asarray(td['test_y']), 'probabilities':np.asarray(td['test_probabilities']), 'auc':td['official_test_auc']},
    }
    ref=families['morgan_r2_c0.01']
    for name,f in families.items():
      for split in ('official_valid','official_test'):
       if not np.array_equal(ref[split]['labels'],f[split]['labels']): raise ValueError((name,split,'label mismatch'))
    yv=ref['official_valid']['labels']; yt=ref['official_test']['labels']; grid=np.linspace(0,1,11)
    out={'protocol_id':'molhiv-morgan-saved-official-complementarity-retrospective-v1','date':'2026-07-29','status':'retrospective diagnostic; test is exposed and not used for claims','families':{},'pairwise_with_morgan':{},'three_family_simplex':{}}
    for n,f in families.items(): out['families'][n]={'official_valid_auc':auc(yv,f['official_valid']['probabilities']),'official_test_auc':auc(yt,f['official_test']['probabilities'])}
    m='morgan_r2_c0.01'
    for other in [x for x in families if x!=m]:
      row={'correlation':{},'probability_grid':{},'rank_grid':{}}
      for split in ('official_valid','official_test'):
       x=families[m][split]['probabilities']; z=families[other][split]['probabilities']
       row['correlation'][split]={'pearson':float(np.corrcoef(x,z)[0,1]),'spearman':float(spearmanr(x,z).statistic)}
      for mode in ('probability','rank'):
       candidates=[]
       for w in grid:
        mv=families[m]['official_valid']['probabilities']; ov=families[other]['official_valid']['probabilities']; mt=families[m]['official_test']['probabilities']; ot=families[other]['official_test']['probabilities']
        if mode=='rank': mv,ov,mt,ot=map(rank01,(mv,ov,mt,ot))
        candidates.append({'morgan_weight':float(w),'valid_auc':auc(yv,w*mv+(1-w)*ov),'test_auc':auc(yt,w*mt+(1-w)*ot)})
       row[f'{mode}_grid']={'valid_selected':choose(candidates),'test_oracle_diagnostic_only':max(candidates,key=lambda r:r['test_auc']),'all':candidates}
      out['pairwise_with_morgan'][other]=row
    # Three-way GINE / stable real-prototype / Morgan, coarse and fixed grid.
    names=['gine_h70_5seed','stable_realprototype_3bank','morgan_r2_c0.01']
    for mode in ('probability','rank'):
      rows=[]
      va=[families[n]['official_valid']['probabilities'] for n in names]; ta=[families[n]['official_test']['probabilities'] for n in names]
      if mode=='rank': va=list(map(rank01,va)); ta=list(map(rank01,ta))
      for i in range(11):
       for j in range(11-i):
        k=10-i-j; w=np.asarray([i,j,k])/10
        rows.append({'weights':w.tolist(),'valid_auc':auc(yv,sum(q*x for q,x in zip(w,va))),'test_auc':auc(yt,sum(q*x for q,x in zip(w,ta)))})
      selected=max(rows,key=lambda r:r['valid_auc']); oracle=max(rows,key=lambda r:r['test_auc'])
      out['three_family_simplex'][mode]={'families':names,'valid_selected':selected,'test_oracle_diagnostic_only':oracle}
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'families':out['families'],'pairs':{k:{m:v[m+'_grid']['valid_selected'] for m in ('probability','rank')} for k,v in out['pairwise_with_morgan'].items()},'three_family':out['three_family_simplex'],'output':a.output},indent=2))
if __name__=='__main__': main()
