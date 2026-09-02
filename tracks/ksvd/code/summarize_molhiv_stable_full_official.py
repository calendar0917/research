"""Summarize frozen full-data real-prototype and matched GINE ensembles."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def member_stats(values,ensemble_auc):
    values=np.asarray(values,dtype=np.float64)
    return {'mean':float(values.mean()),'sample_std':float(values.std(ddof=1)),'range':float(values.max()-values.min()),'ensemble_minus_member_mean':float(ensemble_auc-values.mean())}

def aligned(rows,key):
    idx=np.asarray(rows[0][key]['indices'],dtype=np.int64)
    y=np.asarray(rows[0][key]['y'],dtype=np.int64)
    probs=[]
    for row in rows:
        cur=row[key]
        if not np.array_equal(np.asarray(cur['indices'],dtype=np.int64),idx): raise ValueError(f'{key} index mismatch')
        if not np.array_equal(np.asarray(cur['y'],dtype=np.int64),y): raise ValueError(f'{key} label mismatch')
        probs.append(np.asarray(cur['probabilities'],dtype=np.float64))
    mean=np.mean(probs,axis=0)
    return {'auc':float(roc_auc_score(y,mean)),'n_graphs':int(len(y)),'n_positive':int(y.sum()),'indices':idx.tolist(),'y':y.tolist(),'probabilities':mean.tolist()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result-dir',default='results/molhiv');ap.add_argument('--include-gine',action='store_true');ap.add_argument('--output',default='results/molhiv/stable_realproto_full_official_ensemble_summary.json');args=ap.parse_args();root=Path(args.result_dir)
    seeds=[20260728,20260729,20260730];docs=[];rows=[]
    for s in seeds:
        p=root/f'stable_realproto_full_official_bank{s}_seed0.json';d=json.loads(p.read_text());docs.append(d);rows.append(d['results']['stable_random_node_mil'])
    out={'protocol_id':'molhiv-stable-real-prototype-full-official-ensemble-v1','date':'2026-07-28','policy':{'members':'prototype seeds 20260728/20260729/20260730; model seed 0','ensemble':'arithmetic mean of probabilities','full_dataset_graphs':41127,'development':'selected only on 8k official-train internal scaffold folds','official_test_status':'controlled frozen evaluation, not untouched because repository earlier inspected official test'},'prototype_members':[]}
    for s,d,r in zip(seeds,docs,rows): out['prototype_members'].append({'prototype_seed':s,'path':d['output'],'valid_auc':r['official_valid']['auc'],'test_auc':r['official_test']['auc'],'prototype_sha256':r['prototype_sha256']})
    out['prototype_ensemble']={'official_valid':aligned(rows,'official_valid'),'official_test':aligned(rows,'official_test')}
    if args.include_gine:
        grows=[];gm=[]
        for s in (0,1,2):
            p=root/f'fixed_epoch30_gine_full_official_seed{s}.json';d=json.loads(p.read_text());grows.append(d);gm.append({'seed':s,'path':str(p),'valid_auc':d['official_valid']['auc'],'test_auc':d['official_test']['auc']})
        out['gine_members']=gm;out['gine_ensemble']={'official_valid':aligned(grows,'official_valid'),'official_test':aligned(grows,'official_test')}
        out['comparison']={'prototype_minus_gine_valid_auc':out['prototype_ensemble']['official_valid']['auc']-out['gine_ensemble']['official_valid']['auc'],'prototype_minus_gine_test_auc':out['prototype_ensemble']['official_test']['auc']-out['gine_ensemble']['official_test']['auc']}
        out['member_statistics']={}
        for family,members in (('prototype',out['prototype_members']),('gine',out['gine_members'])):
            out['member_statistics'][family]={}
            for short,split in (('valid','official_valid'),('test','official_test')):
                values=[m[f'{short}_auc'] for m in members]
                out['member_statistics'][family][split]=member_stats(values,out[f'{family}_ensemble'][split]['auc'])
        expected={'official_valid':(4113,81),'official_test':(4113,130)}
        split_audit={}
        for split,(expected_n,expected_pos) in expected.items():
            pmetrics=out['prototype_ensemble'][split];gmetrics=out['gine_ensemble'][split]
            if pmetrics['indices'] != gmetrics['indices']: raise ValueError(f'{split} cross-family index mismatch')
            if pmetrics['y'] != gmetrics['y']: raise ValueError(f'{split} cross-family label mismatch')
            if pmetrics['n_graphs'] != expected_n or pmetrics['n_positive'] != expected_pos: raise ValueError(f'{split} count mismatch')
            split_audit[split]={'n_graphs':expected_n,'n_positive':expected_pos,'all_six_members_indices_and_labels_aligned':True}
        out['audits']={'official_train_graphs':32901,'official_train_positive':1232,'split_alignment':split_audit,'sklearn_auc_recomputed_from_saved_probabilities':True}
    path=Path(args.output);path.write_text(json.dumps(out,indent=2))
    concise={'prototype_members':out['prototype_members'],'prototype_ensemble':{split:{k:v for k,v in metrics.items() if k in ('auc','n_graphs','n_positive')} for split,metrics in out['prototype_ensemble'].items()}}
    if args.include_gine:
        concise['gine_members']=out['gine_members']
        concise['gine_ensemble']={split:{k:v for k,v in metrics.items() if k in ('auc','n_graphs','n_positive')} for split,metrics in out['gine_ensemble'].items()}
        concise['comparison']=out['comparison']
        concise['member_statistics']=out['member_statistics']
        concise['audits']=out['audits']
    print(json.dumps(concise,indent=2))
if __name__=='__main__':main()
