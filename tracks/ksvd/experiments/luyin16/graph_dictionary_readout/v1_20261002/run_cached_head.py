"""Two-stage cached-feature fit/evaluate CLI. No validation opened during fit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .prototype_dictionary import (PrototypeDictionary, BlockScaler, fit_mae,
                                      folded_predict, FULL_BLOCK_WIDTHS,
                                      N_PROTOTYPES, LAMBDA_GRID, SEED)
except ImportError:
    from prototype_dictionary import (PrototypeDictionary, BlockScaler, fit_mae,
                                      folded_predict, FULL_BLOCK_WIDTHS,
                                      N_PROTOTYPES, LAMBDA_GRID, SEED)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(path, split, size):
    with np.load(path, allow_pickle=False) as z:
        if str(z['split'].item()) != split or bool(z['official_test_loaded'].item()):
            raise RuntimeError('wrong split or official test exposure')
        if not bool(z['frozen_backbone'].item()):
            raise RuntimeError('feature cache must be from a frozen backbone')
        r,y,baseline = (z[key].astype(np.float64) for key in ['R','y','p_base'])
        if r.shape!=(size,814) or y.shape!=(size,) or baseline.shape!=(size,):
            raise RuntimeError('Full reader cache has incorrect geometry')
        if not all(np.isfinite(a).all() for a in (r,y,baseline)):
            raise RuntimeError('nonfinite feature cache')
        groups = z['group_ids'].copy() if 'group_ids' in z.files else np.arange(size)
        ids = z['ids'].copy()
        if groups.shape!=(size,) or ids.shape!=(size,):
            raise RuntimeError('unaligned group IDs or molecule IDs')
        sha = str(z['checkpoint_sha'].item())
        if len(sha)!=64:
            raise RuntimeError('missing full checkpoint hash')
    return r,y,baseline,groups,ids,sha


def save_model(path, model, coef, median, checkpoint_sha, selected_lambda):
    np.savez(path,mean=model.scaler.mean,scale=model.scaler.scale,
             keep=model.scaler.keep,weights=model.scaler.weights,
             block_widths=np.asarray(model.scaler.block_widths),
             centers=model.centers,inverse_root=model.inverse_root,
             bandwidth_squared=np.asarray(model.bandwidth_squared),
             selected_rows=model.selected_rows,spectrum=model.spectrum,
             coef=coef,target_median=np.asarray(median),
             checkpoint_sha=np.asarray(checkpoint_sha),
             selected_lambda=np.asarray(selected_lambda))


def load_model(path):
    with np.load(path,allow_pickle=False) as z:
        scaler=BlockScaler(z['mean'],z['scale'],z['keep'],z['weights'],tuple(z['block_widths']))
        model=PrototypeDictionary(scaler,z['centers'],z['inverse_root'],
                                  float(z['bandwidth_squared']),z['selected_rows'],z['spectrum'])
        return model,z['coef'],float(z['target_median']),str(z['checkpoint_sha'])


def fit(args):
    r,y,_,groups,ids,checkpoint_sha=load_features(args.train,'train',10000)
    folder=args.out
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.npz').exists():
        raise RuntimeError('refusing to overwrite a fitted model')
    rng=np.random.default_rng(SEED)
    unique=np.unique(groups)
    rng.shuffle(unique)
    dev_groups=unique[::10]
    dev=np.flatnonzero(np.isin(groups,dev_groups))
    head_fit=np.flatnonzero(~np.isin(groups,dev_groups))
    dictionary=PrototypeDictionary.fit(r[head_fit])
    a,dev_a=dictionary.design(r[head_fit]),dictionary.design(r[dev])
    median=float(np.median(y[head_fit]))
    candidates=[]
    for lam in LAMBDA_GRID:
        coef,report=fit_mae(a,y[head_fit]-median,lam)
        if report['status']!='CONVERGED':
            (folder/'INCOMPLETE_SOLVER.json').write_text(json.dumps(report,indent=2)+'\n')
            raise RuntimeError('uncertified solver; stop without calling the family a negative')
        report['head_dev_MAE']=float(np.mean(abs(median+dev_a@coef-y[dev])))
        candidates.append(report)
    selected=min(candidates,key=lambda row:(row['head_dev_MAE'], -row['lambda_value']))['lambda_value']
    # Refit selected head once on all 10k; official valid remains unopened.
    dictionary=PrototypeDictionary.fit(r)
    median=float(np.median(y))
    coef,report=fit_mae(dictionary.design(r),y-median,selected)
    if report['status']!='CONVERGED':
        (folder/'INCOMPLETE_SOLVER.json').write_text(json.dumps(report,indent=2)+'\n')
        raise RuntimeError('final convex fit did not meet its precision gate')
    model_path=folder/'model.npz'
    save_model(model_path,dictionary,coef,median,checkpoint_sha,selected)
    result=dict(scope='head-fit on pretrained frozen features; NOT end-to-end OOF',
                official_test_loaded=False,train_cache_sha=digest(args.train),
                checkpoint_sha=checkpoint_sha,model_sha=digest(model_path),
                seed=SEED,n_prototypes=N_PROTOTYPES,
                feature_width=814,n_head_fit=len(head_fit),n_head_dev=len(dev),
                head_dev_note='Backbone has seen these train labels; this dev metric only selects head regularization, not a generalization gate',
                candidates=candidates,selected_lambda=selected,final_solver=report,
                train_MAE=float(np.mean(abs(folded_predict(dictionary,r,coef,median)-y))),
                valid_opened_during_fit=False)
    (folder/'FIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def evaluate(args):
    fit_record=json.loads((args.model.parent/'FIT.json').read_text())
    if fit_record['model_sha']!=digest(args.model):
        raise RuntimeError('head changed after fit was locked')
    model,coef,median,checkpoint_sha=load_model(args.model)
    r,y,baseline,groups,ids,valid_sha=load_features(args.valid,'valid',1000)
    if valid_sha!=checkpoint_sha:
        raise RuntimeError('train and valid use different backbones')
    pred=folded_predict(model,r,coef,median)
    base_error,new_error=abs(baseline-y),abs(pred-y)
    delta=base_error-new_error
    # Fixed ID-based diagnostic bins, not selected by labels or errors.
    order=np.argsort(ids.astype(str),kind='stable')
    bin_gains=[float(delta[order[b::5]].mean()) for b in range(5)]
    base_mae,new_mae=float(base_error.mean()),float(new_error.mean())
    gain=base_mae-new_mae
    passes=gain>=.006 and new_mae<=.113 and sum(x>0 for x in bin_gains)>=4
    result=dict(scope='single reused-official-valid screen; exploratory, not confirmatory',
                official_test_loaded=False,valid_cache_sha=digest(args.valid),
                model_sha=digest(args.model),checkpoint_sha=checkpoint_sha,
                original_MAE=base_mae,graph_dictionary_MAE=new_mae,gain=gain,
                five_ID_bin_gains=bin_gains,better_fraction=float(np.mean(delta>0)),
                gain_median=float(np.median(delta)),
                verdict='GRAPH_DICTIONARY_READOUT_PROMISING' if passes else 'GRAPH_DICTIONARY_READOUT_STOP',
                limitation='One backbone seed; historical validation reuse; negative result only excludes this fixed dictionary/head family')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez(args.out.with_suffix('.predictions.npz'),ids=ids,y=y,p_base=baseline,p_graph_dictionary=pred)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('fit'); p.add_argument('--train',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--model',type=Path,required=True);p.add_argument('--valid',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    fit(args) if args.stage=='fit' else evaluate(args)


if __name__=='__main__':main()
