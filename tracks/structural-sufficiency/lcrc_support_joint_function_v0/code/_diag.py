import sys
sys.path.insert(0,'tracks/structural-sufficiency/lcrc_support_joint_function_v0/code')
import torch, numpy as np
import sj_common as C, sj_support as S, sj_model as M, sj_train as T
import sj_gates as G
dev = torch.device('cuda:0')
view, tr, va, te, spec = S.task_caches('hiv')
sup = S.build_task_support('hiv')
m, info = M.make_arm('hiv', view, sup, 'A', 0, dev)
print('model device', next(m.parameters()).device)
cache = G.subset_cache(tr, list(range(4)), spec)
raw = C.PREP_DATA.collate(cache, list(range(4)))
print('raw types', {k: (type(v).__name__, getattr(v,'device',None)) for k,v in raw.items()})
b = T.to_device(raw, dev)
print('after to_device', {k: getattr(v,'device',None) for k,v in b.items()})
print('nbr_emb device', m.local.nbr_emb.weight.device)
