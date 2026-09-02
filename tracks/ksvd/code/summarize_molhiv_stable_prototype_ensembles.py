"""Summarize official-train scaffold development prototype ensembles."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    return np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))


def load(path: Path):
    d=json.loads(path.read_text())
    h=d['results']['fixed_random']['heldout']
    return d,np.asarray(h['graph_indices'],dtype=np.int64),np.asarray(h['labels'],dtype=np.float64),np.asarray(h['scores'],dtype=np.float64)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result-dir',default='results/molhiv')
    ap.add_argument('--output',default='results/molhiv/stable_prototype_ensemble_scaffold_summary.json')
    args=ap.parse_args()
    root=Path(args.result_dir)
    configs=[(20260728,0),(20260728,1),(20260728,2),(20260729,0),(20260730,0)]
    names={
      'model_seed_ensemble':[(20260728,0),(20260728,1),(20260728,2)],
      'prototype_bank_ensemble':[(20260728,0),(20260729,0),(20260730,0)],
      'five_model_combined':configs,
    }
    folds=[]
    for fold in range(3):
      loaded={}
      ref_i=ref_y=None
      for proto,model in configs:
        p=root/f'stable_fixedrandom_bank{proto}_fold{fold}_modelseed{model}.json'
        d,i,y,s=load(p)
        if ref_i is None: ref_i,ref_y=i,y
        if not np.array_equal(i,ref_i) or not np.array_equal(y,ref_y):
          raise ValueError(f'prediction alignment mismatch: {p}')
        loaded[(proto,model)]={'auc':float(roc_auc_score(y,s)),'scores':s,'prob':sigmoid(s),'path':str(p)}
      ensembles={}
      for name,members in names.items():
        prob=np.mean([loaded[x]['prob'] for x in members],axis=0)
        ensembles[name]={
          'auc':float(roc_auc_score(ref_y,prob)),
          'members':[{'prototype_seed':p,'model_seed':m} for p,m in members],
        }
      folds.append({
        'fold':fold,
        'n_graphs':int(len(ref_y)),
        'n_positive':int(ref_y.sum()),
        'individual_auc':{f'bank{p}_seed{m}':loaded[(p,m)]['auc'] for p,m in configs},
        'ensembles':ensembles,
      })
    summary={
      'protocol_id':'molhiv-label-free-stable-real-prototype-ensemble-scaffold-v1',
      'date':'2026-07-28',
      'data':'8k stratified subset; official-train-only outer scaffold folds; no official valid/test evaluation',
      'ensemble_rule':'arithmetic mean of sigmoid probabilities',
      'folds':folds,
      'aggregate':{},
    }
    for name in names:
      vals=np.array([x['ensembles'][name]['auc'] for x in folds])
      summary['aggregate'][name]={'fold_auc':vals.tolist(),'mean_auc':float(vals.mean()),'std_auc':float(vals.std())}
    for p,m in configs:
      key=f'bank{p}_seed{m}'
      vals=np.array([x['individual_auc'][key] for x in folds])
      summary['aggregate'][key]={'fold_auc':vals.tolist(),'mean_auc':float(vals.mean()),'std_auc':float(vals.std())}
    baseline=np.array(summary['aggregate']['bank20260728_seed0']['fold_auc'])
    for name in names:
      vals=np.array(summary['aggregate'][name]['fold_auc'])
      summary['aggregate'][name]['delta_vs_fixed_bank_seed0']=float((vals-baseline).mean())
      summary['aggregate'][name]['fold_wins_vs_fixed_bank_seed0']=int(np.sum(vals>baseline))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['aggregate'],indent=2))

if __name__=='__main__': main()
