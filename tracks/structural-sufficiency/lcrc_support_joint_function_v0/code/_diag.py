import sys, traceback
sys.path.insert(0,'tracks/structural-sufficiency/lcrc_support_joint_function_v0/code')
import torch
import sj_common as C, sj_support as S, sj_model as M, sj_train as T
import sj_gates as G
dev = torch.device('cuda:0')
for task in ('hiv','lipo'):
    view, tr, va, te, spec = S.task_caches(task)
    sup = S.build_task_support(task)
    models = {a: M.make_arm(task, view, sup, a, 0, dev)[0] for a in C.ARMS}
    print(task, 'devices', {a: str(next(models[a].parameters()).device) for a in C.ARMS})
    cache = G.subset_cache(tr, list(range(64)), spec)
    batch = T.to_device(C.PREP_DATA.collate(cache, list(range(64))), dev)
    print(task, 'batch dev', str(batch['nbr_code'].device), str(batch['path_cat'].device))
    for a in C.ARMS:
        try:
            with torch.no_grad():
                out = models[a](batch, return_aux=True)
            print(task, a, 'forward ok', [float(x.detach().abs().max()) for x in (out[0],)])
        except Exception as exc:
            print(task, a, 'FAILED', type(exc).__name__, str(exc)[:200])
            traceback.print_exc(limit=3)
