"""Summarize the predeclared 3-seed official terminal evaluation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def stats(x):
    a=np.asarray(x,dtype=float)
    return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),'minimum':float(a.min()),'maximum':float(a.max())}
def ensemble(rows,key):
    ys=[np.asarray(r[key]['y'],dtype=int) for r in rows]; idx=[np.asarray(r[key]['indices'],dtype=int) for r in rows]; ps=[np.asarray(r[key]['probabilities'],dtype=float) for r in rows]
    if not all(np.array_equal(ys[0],x) for x in ys[1:]) or not all(np.array_equal(idx[0],x) for x in idx[1:]): raise ValueError(f'{key} alignment mismatch')
    p=np.mean(np.stack(ps),axis=0)
    return {'auc':float(roc_auc_score(ys[0],p)),'aggregation':'arithmetic mean of seed 0/1/2 probabilities','n_models':len(rows)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-dir',default='results/molhiv'); ap.add_argument('--output',default='results/molhiv/rawpatch_pca64_official_terminal_3seed_summary.json'); args=ap.parse_args(); root=Path(args.results_dir)
    mil=[json.loads((root/f'rawpatch_pca64_official_terminal_seed{s}.json').read_text()) for s in range(3)]
    gine=[json.loads((root/f'fixed_epoch30_gine_official_terminal_seed{s}.json').read_text()) for s in range(3)]
    families={'random_node_mil':[r['results']['random_node_mil'] for r in mil],'ksvd_node_mil':[r['results']['ksvd_node_mil'] for r in mil],'fixed_epoch30_gine':gine}
    out={'protocol_id':'molhiv-gnnfree-pca64-official-terminal-3seed-summary-v1','date':'2026-07-28','scope':'8,000-graph stratified development subset using remapped OGB official train/valid/test membership; not the full 41,127-graph benchmark','test_status':'terminal frozen evaluation; no post-test tuning or rerun','families':{}}
    for name,rows in families.items():
        out['families'][name]={}
        for split in ('official_valid','official_test'):
            out['families'][name][split]=stats([r[split]['auc'] for r in rows]); out['families'][name][split+'_probability_ensemble']=ensemble(rows,split)
    for split in ('official_valid','official_test'):
        out.setdefault('paired_seed_deltas',{})[f'random_minus_gine_{split}']=(np.asarray(out['families']['random_node_mil'][split]['values'])-np.asarray(out['families']['fixed_epoch30_gine'][split]['values'])).tolist()
        out['paired_seed_deltas'][f'ksvd_minus_gine_{split}']=(np.asarray(out['families']['ksvd_node_mil'][split]['values'])-np.asarray(out['families']['fixed_epoch30_gine'][split]['values'])).tolist()
        out['paired_seed_deltas'][f'ksvd_minus_random_{split}']=(np.asarray(out['families']['ksvd_node_mil'][split]['values'])-np.asarray(out['families']['random_node_mil'][split]['values'])).tolist()
    out['ensemble_deltas']={}
    for split in ('official_valid','official_test'):
        suf=split+'_probability_ensemble'; f=out['families']; out['ensemble_deltas'][f'random_minus_gine_{split}']=f['random_node_mil'][suf]['auc']-f['fixed_epoch30_gine'][suf]['auc']; out['ensemble_deltas'][f'ksvd_minus_gine_{split}']=f['ksvd_node_mil'][suf]['auc']-f['fixed_epoch30_gine'][suf]['auc']; out['ensemble_deltas'][f'ksvd_minus_random_{split}']=f['ksvd_node_mil'][suf]['auc']-f['random_node_mil'][suf]['auc']
    Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
