"""Train-only nested ridge diagnostic for a NEW static typed-cycle object.

Input features must be label-free raw graph functions, never a neural feature
trained on all 10k labels. No official valid/test is accepted by the CLI.
This diagnostic does not validate the final shared neural dictionary model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

RIDGE_GRID = (1e-3, 1e-2, 1e-1)  # mean-squared loss + lambda*||w||^2
SPLIT_SEED = 20261002


def feature_transform_fit(x):
    # Fixed signed compression, then fit-only standardization.
    z = np.arcsinh(x)
    mean, std = z.mean(0), z.std(0)
    keep = (std > 1e-8) & (np.count_nonzero(x, axis=0) >= 5)
    return dict(mean=mean[keep], scale=std[keep], keep=keep)


def feature_transform(x, fit):
    return (np.arcsinh(x[:, fit["keep"]])-fit["mean"])/fit["scale"]


def ridge_path(train_x, train_y, eval_x, lambdas=RIDGE_GRID):
    fit = feature_transform_fit(train_x)
    x, e = feature_transform(train_x,fit), feature_transform(eval_x,fit)
    y_mean = float(train_y.mean())
    # Fit loss is averaged per molecule; regularization is scale-consistent.
    gram = x.T @ x / len(x)
    rhs = x.T @ (train_y-y_mean) / len(x)
    outputs = {}
    for lam in lambdas:
        matrix = gram.copy()
        matrix.flat[::len(matrix)+1] += lam
        w = cho_solve(cho_factor(matrix,lower=True,check_finite=False),rhs,check_finite=False)
        outputs[lam] = e @ w + y_mean
    return outputs, int(x.shape[1])


def outer_folds(group_ids):
    rng = np.random.default_rng(SPLIT_SEED)
    groups = np.unique(group_ids)
    rng.shuffle(groups)
    for fold in range(3):
        outer_groups = groups[fold::3]
        remainder = groups[np.isin(groups,outer_groups,invert=True)]
        # One fixed inner dev split used only to choose each arm's ridge lambda.
        dev_groups = remainder[::5]
        fit_groups = remainder[np.isin(remainder,dev_groups,invert=True)]
        yield tuple(np.flatnonzero(np.isin(group_ids,g)) for g in [fit_groups,dev_groups,outer_groups])


def run_probe(x_base, x_typed, y, group_ids, *, scope="synthetic"):
    if len(y) != len(x_base) or len(y) != len(x_typed) or len(y) != len(group_ids):
        raise ValueError("unaligned molecule rows")
    if not all(np.isfinite(x).all() for x in [x_base,x_typed,y]):
        raise ValueError("non-finite feature or label")
    features = {"BASE_WITH_UNTYPED_CYCLES":x_base,
                "BASE_PLUS_TYPED_CYCLES":np.concatenate([x_base,x_typed],axis=1)}
    all_predictions = {name:np.empty(len(y)) for name in features}
    rows = []
    for fold,(fit,dev,heldout) in enumerate(outer_folds(group_ids)):
        row={"fold":fold,"n_fit":len(fit),"n_dev":len(dev),"n_outer":len(heldout),"arms":{}}
        outer_train=np.concatenate([fit,dev])
        for name,x in features.items():
            path,_=ridge_path(x[fit],y[fit],x[dev])
            selected=min(RIDGE_GRID,key=lambda lam:float(np.mean(abs(path[lam]-y[dev]))))
            predictions,n_features=ridge_path(x[outer_train],y[outer_train],x[heldout],(selected,))
            pred=predictions[selected]
            all_predictions[name][heldout]=pred
            row["arms"][name]=dict(lambda_value=selected,kept_features=n_features,
                                    MAE=float(np.mean(abs(pred-y[heldout]))))
        row["gain"]=row["arms"]["BASE_WITH_UNTYPED_CYCLES"]["MAE"]-row["arms"]["BASE_PLUS_TYPED_CYCLES"]["MAE"]
        rows.append(row)
    mae={name:float(np.mean(abs(pred-y))) for name,pred in all_predictions.items()}
    gain=mae["BASE_WITH_UNTYPED_CYCLES"]-mae["BASE_PLUS_TYPED_CYCLES"]
    positive_folds=sum(row["gain"]>0 for row in rows)
    gate=bool(gain>=.003 and positive_folds>=2)
    return dict(scope=scope,fit_universe="official train only" if scope=="official_train" else "synthetic",
                official_test_loaded=False,n_graphs=len(y),folds=rows,MAE=mae,
                gain=gain,positive_folds=positive_folds,
                gate="BUY_ONE_TYPED_CYCLE_SCREEN" if gate else "DO_NOT_BUY_CURRENT_TYPED_CYCLE_SCREEN",
                limitation="Raw-feature diagnostic; neither Small/Full residual attribution nor neural-model performance evidence")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("features",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.features,allow_pickle=False) as z:
        if str(z["split"].item()) != "train" or bool(z["official_test_loaded"].item()):
            raise RuntimeError("probe accepts only official train, test blocked")
        if len(z["y"]) != 10000 or not bool(z["label_free_features"].item()):
            raise RuntimeError("requires all 10k train and verified label-free raw features")
        report=run_probe(z["X_base"].astype(float),z["X_typed"].astype(float),z["y"].astype(float),
                         z["group_ids"],scope="official_train")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ["MAE","gain","positive_folds","gate"]},indent=2))


if __name__=="__main__":
    main()
