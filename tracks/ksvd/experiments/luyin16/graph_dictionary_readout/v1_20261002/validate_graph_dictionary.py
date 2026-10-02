"""Numerical/functional checks only; synthetic data, not ZINC accuracy."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import LinearConstraint, minimize
try:
    from .prototype_dictionary import (PrototypeDictionary, fit_mae, folded_predict,
                                      FULL_BLOCK_WIDTHS, squared_distance)
except ImportError:
    from prototype_dictionary import (PrototypeDictionary, fit_mae, folded_predict,
                                      FULL_BLOCK_WIDTHS, squared_distance)


def main():
    rng = np.random.default_rng(18)
    report = {'scope':'synthetic reference validation; no ZINC or PyTorch execution',
              'official_test_loaded':False}
    # Independent analytic scalar optimum: |c-1|+c^2 has optimum 0.5.
    scalar, cert = fit_mae(np.ones((9,1)), np.ones(9), 2., gap_tolerance=1e-9)
    assert abs(scalar[0]-.5) < 5e-5 and cert['status']=='CONVERGED'
    report['analytic_MAE_optimum'] = float(scalar[0])

    # Independent constrained optimizer with explicit absolute-error epigraph.
    a = np.column_stack([np.ones(30), rng.normal(size=(30,3))])
    y = a@np.array([.2,-.1,.3,.15])+rng.normal(scale=.04,size=30)
    lam = .03
    coef, cert = fit_mae(a,y,lam,gap_tolerance=1e-8)
    n,k = a.shape
    constraint = LinearConstraint(np.vstack([np.column_stack([a,-np.eye(n)]),
                                            np.column_stack([-a,-np.eye(n)])]),
                                  -np.inf,np.concatenate([y,-y]))
    start = np.concatenate([np.zeros(k),np.abs(y)+.1])
    def objective(z): return np.mean(z[k:])+.5*lam*(z[:k]@z[:k])
    def gradient(z): return np.concatenate([lam*z[:k],np.full(n,1/n)])
    independent = minimize(objective,start,jac=gradient,constraints=constraint,
                           method='SLSQP',options={'ftol':1e-11,'maxiter':500})
    difference = abs(cert['certificate']['primal']-independent.fun)
    assert independent.success and difference < 1e-7
    report['independent_epigraph_objective_delta'] = float(difference)
    report['MAE_primal_dual_gap'] = cert['certificate']['gap']

    # Functional benchmark: a smooth interaction; no target-defined dictionary.
    xy = rng.uniform(-2,2,size=(1600,2))
    raw = np.column_stack([xy,np.zeros((len(xy),2))])
    clean = .2*np.sin(1.5*xy[:,0])*np.cos(xy[:,1])
    target = clean + rng.normal(scale=.01,size=len(xy))
    train, heldout = raw[:1200], raw[1200:]
    d = PrototypeDictionary.fit(train,(2,2),n_prototypes=128)
    median = float(np.median(target[:1200]))
    beta, fit = fit_mae(d.design(train),target[:1200]-median,1e-5,
                       gap_tolerance=1e-6,time_budget_s=30)
    assert fit['status']=='CONVERGED'
    p = folded_predict(d,heldout,beta,median)
    direct = median+d.design(heldout)@beta
    fold_error = float(np.max(abs(p-direct)))
    assert fold_error < 1e-9
    p_reordered = folded_predict(d,heldout[::-1],beta,median)[::-1]
    assert np.max(abs(p-p_reordered)) < 1e-10
    pieces = np.concatenate([folded_predict(d,part,beta,median)
                             for part in np.array_split(heldout,7)])
    assert np.max(abs(p-pieces)) < 1e-10
    linear_beta, _ = fit_mae(np.column_stack([np.ones(len(train)),train[:,:2]]),
                            target[:1200]-median,1e-5)
    linear = median+np.column_stack([np.ones(len(heldout)),heldout[:,:2]])@linear_beta
    mae, linear_mae = float(np.mean(abs(p-target[1200:]))),float(np.mean(abs(linear-target[1200:])))
    assert mae < .04 and mae < linear_mae
    report['synthetic_interaction'] = {'direct_dictionary_MAE':mae,
                                      'linear_MAE':linear_mae,
                                      'solver':fit}
    strong_mae=float(np.mean(abs(clean[1200:]-target[1200:])))
    report['weak_probe_false_promotion_witness'] = {
        'weak_reference_MAE':linear_mae,
        'strong_reference_MAE':strong_mae,
        'candidate_MAE':mae,
        'gain_against_weak':linear_mae-mae,
        'gain_against_strong':strong_mae-mae,
        'interpretation':'Positive gain against a weak reference does not imply a gain against an already-strong predictor',
    }
    assert linear_mae-mae > .003 and strong_mae-mae < .003
    report['folded_inference_max_delta'] = fold_error
    report['inference_batch_and_order_checks'] = 'passed'

    # Nyström implementation vs independent pseudoinverse kernel identity.
    centers = d.centers
    w = np.exp(-squared_distance(centers,centers)/(2*d.bandwidth_squared))
    c = d.kernel(heldout[:20])
    lhs = d.codes(heldout[:20])@d.codes(heldout[:20]).T
    rhs = c@np.linalg.pinv(w,rcond=1e-8)@c.T
    nystrom_error = float(np.max(abs(lhs-rhs)))
    assert nystrom_error < 1e-6
    report['independent_Nystrom_identity_max_delta'] = nystrom_error
    duplicate_raw = np.tile(train[:16],(10,1))
    duplicates = PrototypeDictionary.fit(duplicate_raw,(2,2),n_prototypes=64)
    assert np.isfinite(duplicates.codes(heldout)).all()
    report['duplicate_prototype_pseudoinverse'] = 'passed'

    # Actual intended array dimensions, not a ZINC timing or performance result.
    start = time.perf_counter()
    n_scale = 10000
    latent = rng.normal(size=(n_scale,12))
    raw_scale = latent@rng.normal(scale=.1,size=(12,sum(FULL_BLOCK_WIDTHS)))
    raw_scale[:,288] = 0  # C6-like constant slot
    scale_dictionary = PrototypeDictionary.fit(raw_scale)
    design = scale_dictionary.design(raw_scale)
    generated_y = .1*latent[:,0]+.08*np.sin(latent[:,1])+rng.normal(scale=.02,size=n_scale)
    _, scale_fit = fit_mae(design,generated_y,1e-4,time_budget_s=60)
    report['shape_timing'] = {'raw_shape':list(raw_scale.shape),
                              'dictionary_shape':list(scale_dictionary.centers.shape),
                              'code_shape':list(design.shape),
                              'seconds':time.perf_counter()-start,
                              'solver':scale_fit,
                              'scope':'synthetic correlated arrays, not real ZINC throughput'}
    assert scale_fit['status']=='CONVERGED'
    report['all_gates_passed'] = True
    Path(__file__).with_name('graph_dictionary_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
