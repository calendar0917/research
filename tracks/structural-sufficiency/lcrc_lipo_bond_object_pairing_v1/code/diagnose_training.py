import sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0,'code')
import common as C, data as D, fit_labels, model as M
torch.set_num_threads(4)
spec=C.load_frozen_spec(); norm=C.load_frozen_norm(); caches=D.load_caches(spec)
fit=fit_labels.load_fit_labels(arm_guard=True); y_fit=fit['y']; tn=D.load_target_norm(verify=False)
fit_rows=[int(r) for r in caches['fit']['rows']]; n=C.SPLIT_SIZES['fit']; tp=D.fit_training_positions(n); cache=caches['fit']['cache']
for arm in ['A','R','N']:
    if arm=='A':
        torch.manual_seed(0); model=C.prep_model.LCRC(spec,norm,C.prep_model.DEFAULT_CONFIG)
    else:
        model,_=M.build_model(arm,spec,norm,0,'cpu')
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=C.LR,weight_decay=C.WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=C.TOTAL_UPDATES,eta_min=C.LR_MIN)
    gen=torch.Generator().manual_seed(0+C.TRAIN_SHUFFLE_OFFSET)
    print('ARM',arm)
    for ep in range(1,5):
        sq=0.; ab=0.; seen=0; dx=0.
        for rows in D.shuffle_batches(n,C.BATCH_SIZE,gen):
            plan=D.plan_fit_batch(rows,train_positions=tp,cache=cache,fit_global_rows=fit_rows)
            info=D.assemble_fit_batch(cache,y_fit,plan)
            batch=D.to_device(info['batch'],'cpu')
            y=torch.as_tensor((info['y']-tn['mean'])/tn['scale'],dtype=torch.float32)
            if arm=='A':
                pred=model(batch)
            else:
                pred,aux=model(batch,return_aux=True); dx=max(dx,float(aux['delta_bond'].abs().max()) if aux['delta_bond'].numel() else 0.)
            loss=F.mse_loss(pred,y); opt.zero_grad(set_to_none=True); loss.backward()
            gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),C.GRAD_CLIP)); opt.step(); sched.step()
            e=(pred-y).detach(); sq+=float((e*e).sum()); ab+=float(e.abs().sum()); seen+=e.numel()
        print(ep,'mse',sq/seen,'mae',ab/seen,'delta_max',dx)
