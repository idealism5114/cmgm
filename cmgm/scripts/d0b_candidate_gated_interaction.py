"""Temporal base + gated ST correction. Default synthetic preflight; --run fits once."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from cmgm.models.gated_interaction_residual import VARIANT
from cmgm.models.candidate_moe_fusion import VARIANT as ORIGINAL
from cmgm.scripts import d0b_candidate_moe_bottleneck16 as shared
from cmgm.scripts.d0b_gated_interaction_audit import make_model, initialization, sanity, counts
from cmgm.scripts.baseline_protocol import prediction_loss
from cmgm.scripts.d0b_candidate_moe_diagnostics import routing, distribution
from cmgm.scripts.formal_v2_protocol import atomic_json, sha, metrics
from cmgm.training.train import train

ROOT=Path(__file__).resolve().parents[2]
OUT_ROOT=ROOT/'experiments/d0b_candidate_gated_interaction_residual'
REFERENCE=ROOT/'experiments/d0b_candidate_2expert_moe/20260921_151711/results.json'
REFERENCE_CP=ROOT/'checkpoints/switching_latent_balanced_candidate_2expert_moe_best.pt'
PROTOCOL={k:v for k,v in shared.PROTOCOL.items() if k not in ('variant','expert_hidden_dim','future_controls','future_control_policy')}
PROTOCOL.update(variant=VARIANT, temporal_expert_hidden_dim=64,
                interaction='Wo(Dropout(ReLU(Us(s))*ReLU(Ut(t)), .1)); three bias-free Linear64,64',
                candidates='e_T=e_base=t+F_T(t); e_ST=same e_base+delta_ST',
                fusion='raw candidate mixture; equivalent to e_base+pi_ST*delta_ST',
                interaction_initialization='standard Linear; output NOT zero-initialized',
                description='Temporal base plus gated spatial-temporal interaction correction',
                hypothesis='Joint candidate-relation and interaction-path experiment; no single-subchange attribution')
ZERO_NOTE='ZeroCorrection 是冻结 checkpoint 的推理干预，不能替代重新训练后的消融，也不能证明 backbone 机制贡献恢复。'


def source_record():
    result=shared.source_record()
    for p in (Path(__file__),ROOT/'cmgm/scripts/d0b_gated_interaction_audit.py',ROOT/'cmgm/scripts/d0b_candidate_moe_bottleneck16.py'):
        result['hashes'][str(p.relative_to(ROOT))]=sha(p)
    return result


def reference_audit(data_audit, seed, data, report=REFERENCE, checkpoint=REFERENCE_CP):
    """Reuse only audited original Candidate TRAIN/VAL metrics. No TEST access/eval."""
    result=dict(status='PENDING',report=str(report),checkpoint=str(checkpoint))
    try:
        from cmgm.scripts.d0b_candidate_2expert_moe import PROTOCOL as expected
        r=json.loads(Path(report).read_text())
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        md=cp['metadata'];h=cp['history']
        if r['config']!=expected or md['config']!=expected or expected['seed']!=seed:
            raise ValueError('Reference protocol/seed mismatch')
        if md['variant']!=ORIGINAL or md['seed']!=seed or r['config']['variant']!=ORIGINAL:
            raise ValueError('Not the original Candidate64 variant')
        if Path(r['checkpoint']).resolve()!=Path(checkpoint).resolve() or sha(checkpoint)!=r['checkpoint_sha256']:
            raise ValueError('Reference checkpoint identity mismatch')
        if r['status']!='COMPLETE' or not r['trained'] or not r['sanity']['PASS'] or not r['best_sanity']['PASS']:
            raise ValueError('Reference incomplete/invalid')
        if md['source_hashes']!=r['source_hashes']:
            raise ValueError('Reference source provenance mismatch')
        if md['data']!=r['data'] or any(h[k]!=r['history'][k] for k in h):
            raise ValueError('Reference metadata/history mismatch')
        if len(h['val_loss'])<200 and len(h['val_loss'])-cp['best_epoch']<10:
            raise ValueError('Reference training not complete')
        if cp['best_val_loss']!=min(h['val_loss']) or h['val_loss'][cp['best_epoch']-1]!=cp['best_val_loss']:
            raise ValueError('Reference selection mismatch')
        for epoch,beta in enumerate(h['switch_beta'],1):
            if abs(beta-.0005*min(1.,max(0.,(epoch-1)/19)))>1e-15:
                raise ValueError('Reference native KL schedule mismatch')
        for s in ('train','val'):
            if r['data']['split_fingerprint'][s]!=data_audit['split_fingerprint'][s]:
                raise ValueError(s+' data fingerprint mismatch')
        canonical=lambda a:[(v['commodity'],v.get('node_index',v.get('full_node')),v.get('target_index',v.get('target_output'))) for v in a['mapping']]
        if canonical(r['data'])!=canonical(data_audit):
            raise ValueError('Commodity order mismatch')
        if r['source_hashes']['cmgm/training/metric_standard.py']!=sha(ROOT/'cmgm/training/metric_standard.py'):
            raise ValueError('Reference evaluator definition drift')
        original=make_model(data,seed,ORIGINAL)
        original.load_state_dict(cp['model_state_dict'],strict=True)
        result.update(status='PASS',checkpoint_sha256=sha(checkpoint),report_sha256=sha(report),
                      metrics={s:r['evaluation']['metrics'][s] for s in ('train','val')},
                      best_epoch=cp['best_epoch'],selection_val_huber=cp['best_val_loss'],parameters=sum(p.numel() for p in original.parameters()),
                      evaluation_policy='Existing audited TRAIN/VAL results only; no reference retraining or TEST metrics')
    except (OSError,KeyError,ValueError,RuntimeError) as exc:
        result['reason']=f'{type(exc).__name__}: {exc}'
    return result


def correction_statistics(base,delta,effective,fused):
    base,delta,effective,fused=[v.double().cpu() for v in (base,delta,effective,fused)]
    nb,nd,ne,nh=[v.norm(dim=-1) for v in (base,delta,effective,fused)]
    valid=(nb>0)&(nd>0)
    cosine=(base[valid]*delta[valid]).sum(-1)/(nb[valid]*nd[valid])
    return dict(norms={k:distribution(v.numpy()) for k,v in zip(('e_base','delta_ST','effective_correction','h'),(nb,nd,ne,nh))},
                effective_to_base_ratio=distribution((ne/(nb+1e-8)).numpy()),ratio_epsilon=1e-8,
                cosine=dict(valid_count=int(valid.sum()),undefined_zero_norm_count=int((~valid).sum()),
                            distribution=distribution(cosine.numpy()) if valid.any() else None,
                            policy='Zero-norm pairs excluded and counted; no invented cosine values'))


@torch.no_grad()
def evaluate_frozen(model,data,seed,device,out):
    """Persist formal predictions, then use the SAME eval-forward base for intervention."""
    model.eval();saved={};representations={};result={}
    for split,loader in shared.data_loaders(data,seed,full=True).items():
        pred_rows=[];target_rows=[];pi_rows=[];base=[];delta=[];effective=[];fused=[];loss=[]
        for x,y in loader:
            p=model(x.to(device));c=model.candidate_moe_fusion.last
            if p.shape!=y.shape or p.shape[1:]!=(4,24) or not torch.isfinite(p).all():
                raise ValueError('Invalid frozen prediction')
            pred_rows.append(p.cpu().numpy());target_rows.append(y.numpy());pi_rows.append(c['pi'].cpu().numpy())
            loss.append(float(prediction_loss(p,y.to(device))))
            for destination,key in ((base,'e_base'),(delta,'delta_ST'),(effective,'effective_correction'),(fused,'h_moe')):
                destination.append(c[key].cpu())
        p,y,pi=np.concatenate(pred_rows),np.concatenate(target_rows),np.concatenate(pi_rows)
        b,d,e,h=[torch.cat(rows) for rows in (base,delta,effective,fused)]
        if not all(torch.isfinite(v).all() for v in (b,d,e,h)):
            raise ValueError('Nonfinite frozen representations')
        result[split]=dict(metrics=metrics(p,y),prediction_only_huber_batch_mean=float(np.mean(loss)),
                           routing=routing(pi),correction=correction_statistics(b,d,e,h))
        saved.update({split+'_prediction':p,split+'_target':y,split+'_pi':pi})
        representations[split]=b
    np.savez_compressed(out/'formal_train_val_predictions.npz',**saved)
    # No second backbone/base evaluation; shared head in eval mode, same cached base.
    intervention={};zero_arrays={}
    for split,base in representations.items():
        p=torch.cat([model.head(v.to(device)).reshape(-1,4,24).cpu() for v in base.split(64)]).numpy()
        y=saved[split+'_target'];m=metrics(p,y)
        intervention[split]=dict(metrics=m,zero_minus_formal_MAE={h:m[h]['MAE']-result[split]['metrics'][h]['MAE'] for h in m})
        zero_arrays[split+'_prediction']=p
    np.savez_compressed(out/'zero_correction_predictions.npz',**zero_arrays)
    return dict(formal=result,zero_correction=dict(note=ZERO_NOTE,same_eval_forward_base=True,shared_head=True,results=intervention))


def report(r,out):
    lines=['# 时序基准＋门控时空交互修正','',f"状态：{r['status']}",'',
           '这是候选关系与交互路径的整体结构实验，不预设预测改善、动态专家选择成功或 backbone 消融贡献恢复。',
           '`e_base=t+F_T(t)`；`delta=Wo(Dropout(ReLU(Us(s))*ReLU(Ut(t))))`；',
           '`e_T=e_base`，`e_ST=e_base+delta`；`h=pi_T*e_T+pi_ST*e_ST=e_base+pi_ST*delta`。',
           '三个交互 Linear 均无 bias、使用默认初始化；零输入性质只针对投影后的 s/t。',
           '原 backbone、TemporalResidualExpert、router、shared head 与训练协议保持不变。',
           '训练：四周期等权 Huber(.02)+native Switch KL；选择/早停/调度：prediction-only VAL Huber batch mean。',
           f"Candidate64 参照：{r.get('reference',{}).get('status','PENDING')}。",f"{r.get('reference',{}).get('reason','')}",
           f"Checkpoint：{r.get('checkpoint','未训练')}；best epoch：{r.get('best_epoch','N/A')}；训练秒数：{r.get('training_seconds','N/A')}。",'',
           '| Split | Horizon | MAE | MSE | RMSE | Hit% |','|---|---:|---:|---:|---:|---:|']
    ev=r.get('evaluation',{})
    for s,row in ev.get('formal',{}).items():
        for h,m in row['metrics'].items():
            lines.append(f"| {s} | {h}d | {m['MAE']:.10g} | {m['MSE']:.10g} | {m['RMSE']:.10g} | {100*m['Hit']:.6f} |")
        lines += ['',f"{s} prediction-only Huber（batch mean）：{row['prediction_only_huber_batch_mean']:.12g}",'']
    if ev and r['reference']['status']=='PASS':
        ref=r['reference'];lines+=['','Candidate64 对照（变化%=100×(新−原)/原；MAE 负值更好）：','',
            '| Split | Horizon | Candidate64 MAE | 新 MAE | 变化% |','|---|---:|---:|---:|---:|']
        for s in ('train','val'):
            for h in ('1','5','10','20'):
                old=ref['metrics'][s][h]['MAE'];new=ev['formal'][s]['metrics'][h]['MAE']
                lines.append(f'| {s} | {h}d | {old:.10g} | {new:.10g} | {100*(new-old)/old:+.5f} |')
        lines+=['',f"正式 VAL 选择损失：原 {ref['selection_val_huber']:.12g}；新 {r['best_val_loss']:.12g}。"]
    if ev:
        lines+=['','ZeroCorrection（只读）：','',ZERO_NOTE,'',
                '| Split | Horizon | 正式 MAE | ZeroCorrection MAE | 干预−正式 |','|---|---:|---:|---:|---:|']
        for s,row in ev['zero_correction']['results'].items():
            for h,m in row['metrics'].items():
                lines.append(f"| {s} | {h}d | {ev['formal'][s]['metrics'][h]['MAE']:.10g} | {m['MAE']:.10g} | {row['zero_minus_formal_MAE'][h]:+.10g} |")
    lines += ['','完整 routing/修正范数分布、比值及零范数余弦处理见 diagnostics.json。',
              '范围：TRAIN/VAL only；不构建或评价 TEST，不运行任何消融或搜索。',
              '限制：单 seed；尚未训练时没有预测结论；两个结构变化作为整体研究，不能单独归因。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'results.json',r)


def preflight(seed,out):
    shared.check_config()
    data=dict(n_nodes=284,market_indices=dict(stock=(0,248),bond=(248,260),commodity=(260,284)))
    model,init=initialization(data,seed)
    g=torch.Generator().manual_seed(seed+1)
    checks=sanity(model,torch.randn(2,20,284,21,generator=g),torch.randn(2,4,24,generator=g)*.02)
    out.mkdir(parents=True,exist_ok=False)
    r=dict(status='SYNTHETIC PREFLIGHT PASS — NO TRAINING',config={**PROTOCOL,'seed':seed},
           initialization=init,sanity=checks,source=source_record(),reference=dict(status='PENDING',reason='Synthetic preflight does not read real data/reference artifacts'))
    for key in ('config','initialization','sanity','source'):
        atomic_json(out/(key+'_audit.json' if key in ('initialization','sanity') else key+'.json'),r[key])
    report(r,out)
    print(json.dumps(dict(output=str(out),status=r['status'],parameters=init['new'],delta=init['delta'],shared_max_abs_diff=init['shared_max_abs_diff']),indent=2))


def finish(model,data,r,out,device):
    before=sha(r['checkpoint'])
    if before!=r['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    r['evaluation']=evaluate_frozen(model,data,r['config']['seed'],device,out)
    if sha(r['checkpoint'])!=before:
        raise ValueError('Checkpoint changed during read-only evaluation')
    actual=r['evaluation']['formal']['val']['prediction_only_huber_batch_mean']
    if abs(actual-r['best_val_loss'])>1e-8:
        raise ValueError('Restored best checkpoint VAL objective differs from selection history; STOP')
    r['restored_val_huber_error']=actual-r['best_val_loss']
    r['status']='COMPLETE — TRAIN/VAL ONLY'
    atomic_json(out/'train_val_metrics.json',{s:row['metrics'] for s,row in r['evaluation']['formal'].items()})
    atomic_json(out/'diagnostics.json',{s:{k:row[k] for k in ('routing','correction')} for s,row in r['evaluation']['formal'].items()})
    atomic_json(out/'zero_correction.json',r['evaluation']['zero_correction'])
    report(r,out)


def execute(args):
    shared.check_config();device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    data,audit=shared.train_val_data(args.data_audit)
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory; do not overwrite any experiment')
    out.mkdir(parents=True)
    model,init=initialization(data,args.seed);model.to(device)
    x,y=next(iter(shared.data_loaders(data,args.seed,full=True)['val']))
    checks=sanity(model,x[:2].to(device),y[:2].to(device))
    ref=reference_audit(audit,args.seed,data,args.reference_results,args.reference_checkpoint)
    r=dict(status='DATA PREFLIGHT PASS',config={**PROTOCOL,'seed':args.seed},data=audit,
           source=source_record(),initialization=init,sanity=checks,reference=ref)
    for filename,value in [('config',r['config']),('data_audit',audit),('source_hashes',r['source']),
                           ('initialization_audit',init),('structural_sanity',checks),('reference_provenance',ref)]:
        atomic_json(out/(filename+'.json'),value)
    report(r,out)
    if not args.run:
        print(f'DATA PREFLIGHT PASS, no training: {out}');return
    OUT_ROOT.mkdir(parents=True,exist_ok=True)
    receipt=OUT_ROOT/f'seed{args.seed}_formal_run.json'
    with (OUT_ROOT/f'seed{args.seed}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists():raise RuntimeError(f'Formal run already reserved: {receipt}; review, never auto retry')
        cp=ROOT/'checkpoints/candidate_gated_interaction_residual'/f'seed{args.seed}'/out.name/(VARIANT+'_best.pt')
        cp.parent.mkdir(parents=True,exist_ok=True)
        if cp.exists():raise FileExistsError(cp)
        atomic_json(receipt,dict(status='STARTED',output=str(out),checkpoint=str(cp)))
        r.update(status='TRAINING',checkpoint=str(cp));report(r,out)
        del model
        model=make_model(data,args.seed).to(device)  # Reset RNG after audits, not from fitted weights.
        loaders=shared.data_loaders(data,args.seed)
        def callback(history):
            atomic_json(out/'training_history.json',history)
            atomic_json(out/'routing_history.json',history['candidate_routing_history'])
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        try:
            history=train(model,loaders['train'],loaders['val'],torch.empty((2,0),dtype=torch.long,device=device),
                          torch.empty(0,device=device),device,num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,
                          checkpoint_path=str(cp),checkpoint_metadata=dict(variant=VARIANT,seed=args.seed,
                          config=r['config'],data=audit,source=r['source']),epoch_history_callback=callback)
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.perf_counter()-start
            saved=torch.load(cp,map_location=device,weights_only=False)
            model.load_state_dict(saved['model_state_dict'],strict=True)
            model.switching_latent_transformer.set_epoch(saved['best_epoch'])
            saved.update(training_complete=True,training_seconds=seconds,history=history)
            temp=cp.with_suffix('.tmp');torch.save(saved,temp);temp.replace(cp)
            r.update(status='TRAINING COMPLETE; CHECKPOINT FROZEN',training_complete=True,training_seconds=seconds,
                     checkpoint_sha256=sha(cp),best_epoch=saved['best_epoch'],best_val_loss=saved['best_val_loss'])
            callback(history);report(r,out)
            atomic_json(out/'best_checkpoint_metadata.json',{k:r[k] for k in ('checkpoint','checkpoint_sha256','best_epoch','best_val_loss','training_seconds','training_complete')})
            atomic_json(receipt,dict(status='TRAINING COMPLETE',output=str(out),checkpoint=str(cp),sha256=sha(cp)))
            finish(model,data,r,out,device)
        except BaseException as exc:
            r.update(status='EVALUATION INTERRUPTED' if r.get('training_complete') else 'INTERRUPTED — REVIEW REQUIRED',error=f'{type(exc).__name__}: {exc}')
            report(r,out);raise


def evaluate_completed(args):
    out=args.evaluate_completed.resolve();r=json.loads((out/'results.json').read_text())
    if r['config']!={**PROTOCOL,'seed':r['config']['seed']} or not r.get('training_complete'):
        raise ValueError('Expected a completed run of this protocol')
    if r['checkpoint_sha256']!=sha(r['checkpoint']) or r['source']['hashes']!=source_record()['hashes']:
        raise ValueError('Checkpoint/source drift')
    data,audit=shared.train_val_data(args.data_audit)
    if audit!=r['data']:raise ValueError('Data drift')
    device=torch.device(args.device);model=make_model(data,r['config']['seed']).to(device)
    cp=torch.load(r['checkpoint'],map_location=device,weights_only=False)
    if cp['metadata']['config']!=r['config'] or not cp.get('training_complete'):
        raise ValueError('Checkpoint metadata mismatch')
    model.load_state_dict(cp['model_state_dict'],strict=True)
    model.switching_latent_transformer.set_epoch(cp['best_epoch'])
    finish(model,data,r,out,device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--run',action='store_true')
    group.add_argument('--data-preflight',action='store_true')
    group.add_argument('--evaluate-completed',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path)
    p.add_argument('--data-audit',type=Path,default=shared.DEFAULT_AUDIT)
    p.add_argument('--reference-results',type=Path,default=REFERENCE)
    p.add_argument('--reference-checkpoint',type=Path,default=REFERENCE_CP)
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.output is None:
        mode='run' if args.run else 'data_preflight' if args.data_preflight else 'synthetic_preflight'
        args.output=OUT_ROOT/f'{mode}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.evaluate_completed:evaluate_completed(args)
    elif args.run or args.data_preflight:execute(args)
    else:preflight(args.seed,args.output)


if __name__=='__main__':main()
