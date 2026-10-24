import sys, numpy as np, torch, json
sys.path.insert(0,'.')
import esoc_data as D, esoc_models as M, esoc_train as T
dev='cuda:0'
from pathlib import Path
ANALYSIS = Path(__file__).resolve().parent.parent / 'analysis'
scales=T.read_json(ANALYSIS/'scales.json')
tr=D.SplitTensors(D.load_cache('train'), dev)
order=T.batch_order(0,1,tr.n,32)
batches=[order[i*32:(i+1)*32] for i in range(10)]
sched=T.schedule_fn(1000,5,2e-3,2e-5)

def run(mode, label):
    m=M.build_model(0,'real',scales).to(dev)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout): mod.p=0.0
    m.drop_path=0.0
    net = torch.compile(m, mode=mode, dynamic=False) if mode else m
    base = getattr(net,'_orig_mod',net)
    opt=T.make_optimizer(base,2e-3,1e-5)
    losses=[]; p0=None
    for i,rows in enumerate(batches):
        b=tr.batch(rows)
        for g in opt.param_groups: g['lr']=sched(i)
        p=net(b); loss=(p-b['y']).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(),5.0); opt.step()
        losses.append(float(loss.detach()))
        if i==0: p0=[v.detach().clone() for v in base.parameters()]
    print(label, [round(v,5) for v in losses])
    return losses, [v.detach().clone() for v in base.parameters()], p0

l_e, par_e, p0_e = run(None,'eager   ')
l_c, par_c, p0_c = run('reduce-overhead','cg      ')
l_d, par_d, p0_d = run('default','default ')
print('first-loss diff eager vs cg  ', abs(l_e[0]-l_c[0]))
print('first-loss diff eager vs def ', abs(l_e[0]-l_d[0]))
print('after-step1 param diff e/c   ', max(float((a-b).abs().max()) for a,b in zip(p0_e,p0_c)))
print('after-step1 param diff e/d   ', max(float((a-b).abs().max()) for a,b in zip(p0_e,p0_d)))
print('final param diff e/c         ', max(float((a-b).abs().max()) for a,b in zip(par_e,par_c)))
print('final param diff e/d         ', max(float((a-b).abs().max()) for a,b in zip(par_e,par_d)))
