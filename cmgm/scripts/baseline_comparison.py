"""Deep V3: eight fixed baselines plus read-only Candidate MoE. Default is preflight, --run fits once."""
import argparse
from datetime import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from cmgm import config
from cmgm.models.comparison_baselines import ORDER,TRAINABLE,MODEL_CONFIGS,INPUT_VIEWS,make_model
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.candidate_moe_fusion import VARIANT as BASE
from cmgm.scripts.baseline_protocol import seed_all,loaders,parameter_counts,sanity,evaluate,train_one,data_audit
from cmgm.scripts.d0b_5d_error_regime_diagnostic import ROOT,checkpoint_payload,sha256
from cmgm.scripts.formal_v2_protocol import atomic_json
from cmgm.scripts.baseline_runtime import RUNTIME_PROTOCOL,environment,measure_inference

OURS='D0B-Candidate-Aware 2-Expert MoE'
KEYS=dict(RNN='rnn',GRU='gru',LSTM='lstm',VanillaTransformer='transformer',GraphWaveNet='graph_wavenet',MTGNN='mtgnn',MSGNet='msgnet',CrossGNN='crossgnn')
# Compatibility only for historical formal_v2 script imports. NOT the V3 reference.
REFERENCE=dict(MAE=.0219947838,MSE=.000912875418,RMSE=.0302138283,Hit=.468401487)
PROTOCOL=dict(suite='baseline_comparison_deep_v3',models=list(ORDER),ours_variant=BASE,
    seed=42,seq_len=20,batch_size=64,epochs=200,patience=10,lr=1e-4,weight_decay=1e-5,
    loss='sum four Huber(delta=.02), no baseline auxiliary loss',optimizer='Adam, one parameter group',
    scheduler='ReduceLROnPlateau(mode=min,factor=.5,patience=5)',selection='mean of batch-mean multi-horizon VAL Huber; VAL5 logging only',
    train_shuffle=False,train_drop_last=True,full_evaluation_drop_last=False,adapter_std='population, correction=0',
    flatten_order='token-major, then feature-major: channel=21*token+feature',
    scalar_mts_adapter='each model owns one shared trainable Linear(21,1)',forecast_slots=[1,5,10,20],
    determinism='Python/NumPy/PyTorch/CUDA/loader seed42, cudnn deterministic=True, benchmark=False',
    ours='frozen existing checkpoint; never fit',error_policy='INVALID and STOP; no automatic retry/fallback')


def flush(r,out):
    from cmgm.scripts.baseline_report import write_report
    write_report(r,out)
    atomic_json(out/'results.json',r)
    artifacts=dict(sanity_checks=dict(data=r.get('data_audit'),models=r['sanity']),source_hashes=r.get('source_hashes',{}),
        data_audit=r.get('data_audit',{}),model_configs=MODEL_CONFIGS,protocol=PROTOCOL,
        baseline_provenance=r.get('baseline_provenance',{}),ours_reference_verification=r.get('reference_check',{}),
        runtime_protocol=RUNTIME_PROTOCOL,runtime_measurements={n:v['runtime'] for n,v in r.get('models',{}).items() if 'runtime' in v})
    for key,value in artifacts.items():atomic_json(out/(key+'.json'),value)


def sources():
    names=['cmgm/config.py','cmgm/models/comparison_baselines.py','cmgm/scripts/baseline_protocol.py',
        'cmgm/scripts/baseline_comparison.py','cmgm/scripts/baseline_report.py','cmgm/scripts/baseline_runtime.py','cmgm/training/metric_standard.py',
        'cmgm/data/data_loader.py','cmgm/data/feature_builder.py','cmgm/models/hetero_mixhop_model.py',
        'cmgm/models/candidate_moe_fusion.py','cmgm/models/switching_latent_transformer.py',
        'cmgm/graph/adaptive_graph.py','cmgm/models/model.py','cmgm/scripts/main_ablation.py']
    names+=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'cmgm/models/baselines').iterdir() if p.suffix in ('.py','.json','.md'))
    return {n:sha256(ROOT/n) for n in names}


def completed_training(path,name,audit,source_hashes,expected_hash):
    """Only recover completed V3 training whose recorded identity/hash exactly match."""
    if path is None or not path.exists():return None
    if expected_hash is None or sha256(path)!=expected_hash:raise ValueError('Missing/mismatched checkpoint hash; STOP review')
    cp=checkpoint_payload(path)
    if not cp.get('training_complete'):return None
    md=cp['metadata']
    if (md.get('model')!=name or md.get('protocol')!=PROTOCOL or md.get('configuration')!=MODEL_CONFIGS[name]
        or md.get('data_fingerprint')!=audit['split_fingerprint'] or md.get('source_hashes')!=source_hashes):
        raise ValueError('Completed V3 training identity/source differs; STOP')
    return cp


def audit_ours(path,report_path,audit):
    old=json.loads(report_path.read_text());cp=checkpoint_payload(path);md=cp.get('metadata',{})
    candidate_data={**audit,'mapping':[dict(commodity=v['commodity'],node_index=v['full_node'],target_index=v['target_output'],output_index=v['target_output']) for v in audit['mapping']]}
    checks=dict(complete=old.get('status')=='COMPLETE' and old.get('trained') is True,
        variant=old.get('config',{}).get('variant')==BASE and md.get('variant')==BASE,
        checkpoint_hash=sha256(path)==old.get('checkpoint_sha256'),
        data=old.get('data')==candidate_data and md.get('data')==candidate_data,
        checkpoint_config=md.get('config')==old.get('config'),source_metadata=md.get('source_hashes')==old.get('source_hashes'),
        initial_sanity=old.get('sanity',{}).get('PASS') is True,best_sanity=old.get('best_sanity',{}).get('PASS') is True,
        best_epoch=cp.get('best_epoch')==old.get('best_checkpoint_metadata',{}).get('best_epoch'))
    if not all(checks.values()):raise ValueError('OURS provenance mismatch: '+str(checks))
    provenance=dict(checkpoint=str(path.resolve()),checkpoint_sha256=sha256(path),report=str(report_path.resolve()),
        report_sha256=sha256(report_path),checks=checks,metadata=md,best_epoch=cp['best_epoch'])
    return cp,old,provenance


def run_suite(args,device,data):
    if (config.SEQ_LEN,config.FEATURE_DIM,tuple(config.MULTI_HORIZONS),config.HUBER_DELTA,config.LEARNING_RATE,config.WEIGHT_DECAY)!=(20,21,(1,5,10,20),.02,1e-4,1e-5):raise ValueError('Frozen formal protocol differs')
    if config.TARGET_TYPE!='return' or config.LOSS_TYPE!='huber':raise ValueError('Expected original return/Huber pipeline')
    run=bool(getattr(args,'run',False))
    if run and args.sanity_only:raise ValueError('--run and --sanity-only are mutually exclusive')
    if run and device.type!='cuda':raise ValueError('Formal training requires CUDA; no CPU fallback')
    outroot=Path(args.output_dir);cpdir=Path(args.checkpoint_dir);ours_path=Path(args.ours_checkpoint);ours_report=Path(args.ours_results)
    audit=data_audit(data);hashes=sources();before=sha256(ours_path)
    if not audit['PASS']:raise ValueError('INVALID data/commodity order')
    cp,old,provenance=audit_ours(ours_path,ours_report,audit)
    if args.resume:
        options=sorted(outroot.glob('*/results.json'))
        if args.resume=='latest' and not options:raise ValueError('No prepared V3 suite')
        out=options[-1].parent if args.resume=='latest' else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=PROTOCOL or r['source_hashes']!=hashes or r['data_audit']!=audit or r['model_configs']!=MODEL_CONFIGS:
            raise ValueError('Resume protocol/data/config/source mismatch; no automatic reuse')
        if r['ours_provenance']!=provenance or r['checkpoint_dir']!=str(cpdir.resolve()):raise ValueError('Resume artifact destination/reference mismatch')
        if any(v in ('RUNNING','INVALID') for v in r['model_status'].values()):raise ValueError('Interrupted/invalid V3 run requires review, never an automatic restart')
    else:
        if any(outroot.glob('*/results.json')):raise ValueError('Existing V3 suite: use --resume latest')
        if any(cpdir.glob('*.pt')):raise ValueError('Existing V3 checkpoints without explicit matching resume; STOP')
        out=outroot/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        r=dict(status='PREFLIGHT',protocol=PROTOCOL,models={},sanity={},model_status={},data_audit=audit,source_hashes=hashes,
            model_configs=MODEL_CONFIGS,ours_provenance=provenance,checkpoint_dir=str(cpdir.resolve()),training_executed=False,
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            baseline_provenance=json.loads((ROOT/'cmgm/models/baselines/provenance.json').read_text()))
    full=loaders(data,full=True);cs,ce=data['market_indices']['commodity'];names=data['feature_names'][cs:ce]
    fixed=next(iter(full['val']))[0][:4].to(device)
    r['fixed_sanity_input_shape']=list(fixed.shape);r['last_launch_device']=str(device);flush(r,out)
    current=OURS
    try:
        if OURS not in r['models']:
            mi=data['market_indices'];m=HeteroMixHopCMGM(data['n_nodes'],24,n_stock=mi['stock'][1]-mi['stock'][0],n_bond=mi['bond'][1]-mi['bond'][0],variant=BASE).to(device)
            counts=parameter_counts(m);m.load_state_dict(cp['model_state_dict'],strict=True);m.eval();m.requires_grad_(False)
            reference=evaluate(m,full,device,names);expected=old['evaluation']['metrics']
            errors={s:{h:{k:abs(v-expected[s][h][k]) for k,v in values.items()} for h,values in hm.items()} for s,hm in reference['metrics'].items()}
            ok=all(np.isclose(v,expected[s][h][k],rtol=1e-5,atol=1e-9) for s,hm in reference['metrics'].items() for h,values in hm.items() for k,v in values.items())
            r['reference_check']=dict(PASS=bool(ok),actual=reference['metrics'],expected=expected,absolute_errors=errors,tolerance=dict(rtol=1e-5,atol=1e-9))
            if not ok:raise ValueError('STOP: OURS reference mismatch')
            r['models'][OURS]=dict(**reference,parameters=counts,training=dict(best_epoch=cp['best_epoch'],train_seconds=None,seconds_per_epoch_mean=None),
                input_view='native full-node input (B,20,N,21); frozen audited Candidate MoE',frozen_in_suite=True)
            r['model_status'][OURS]='FROZEN_REFERENCE';del m;gc.collect();flush(r,out)
        del cp,old
        # Audit every architecture before any optimizer is created; CPU PREPARED can resume on CUDA.
        for name in ORDER:
            current=name
            if r['model_status'].get(name) in ('COMPLETE','TRAINED'):continue
            if r['sanity'].get(name,{}).get('device')==str(device) and r['sanity'][name].get('PASS'):continue
            seed_all();m=make_model(name,data['market_indices']).to(device);check=sanity(m,fixed)
            r['sanity'][name]=dict(initial=check,PASS=check['PASS'],device=str(device))
            r['models'][name]=dict(parameters=parameter_counts(m),input_view=INPUT_VIEWS[name],configuration=MODEL_CONFIGS[name])
            if not check['PASS']:raise AssertionError(name+' initial sanity failed')
            r['model_status'][name]='PREPARED';del m;gc.collect();flush(r,out)
        if not run:
            r['status']='COMPLETE' if all(r['model_status'].get(n)=='COMPLETE' for n in ORDER) else 'PREPARED'
            flush(r,out);print('PREPARED: no optimizer created; eight fixed baseline fits require --run. Output:',out,flush=True)
            return r
        # Timings from different hosts/devices/settings must never be silently mixed.
        for entry in r['models'].values():
            if 'runtime' in entry and (entry['runtime']['environment']!=environment(device) or entry['runtime']['protocol']!=RUNTIME_PROTOCOL):
                raise ValueError('Runtime hardware/protocol mismatch; STOP review before comparing timings')
        current=OURS
        if 'runtime' not in r['models'][OURS]:
            mi=data['market_indices']
            m=HeteroMixHopCMGM(data['n_nodes'],24,n_stock=mi['stock'][1]-mi['stock'][0],n_bond=mi['bond'][1]-mi['bond'][0],variant=BASE).to(device)
            try:
                m.load_state_dict(checkpoint_payload(ours_path)['model_state_dict'],strict=True)
                m.requires_grad_(False)
                r['models'][OURS]['runtime']=measure_inference(m,full['test'],device)
                flush(r,out)
            finally:
                del m;gc.collect();torch.cuda.empty_cache()
        for name in ORDER:
            current=name;entry=r['models'][name];path=cpdir/f'{KEYS[name]}_best.pt'
            recovered=completed_training(path,name,audit,hashes,entry.get('checkpoint_sha256')) if path.exists() else None
            if r['model_status'].get(name)=='COMPLETE':
                if recovered is None:raise ValueError('Completed checkpoint missing/invalid')
                if 'runtime' not in entry:
                    m=make_model(name,data['market_indices']).to(device)
                    try:
                        m.load_state_dict(recovered['model_state_dict'],strict=True)
                        entry['runtime']=measure_inference(m,full['test'],device);flush(r,out)
                    finally:
                        del m;gc.collect();torch.cuda.empty_cache()
                print(name,'COMPLETE: reused, no training/evaluation',flush=True);continue
            seed_all();m=make_model(name,data['market_indices']).to(device)
            try:
                if recovered is not None:
                    m.load_state_dict(recovered['model_state_dict'],strict=True);entry['training']=recovered['training_summary']
                    atomic_json(out/f'history_{KEYS[name]}.json',recovered['history'])
                else:
                    if path.exists():raise ValueError('Unverified checkpoint; STOP')
                    cpdir.mkdir(parents=True,exist_ok=True);r['model_status'][name]='RUNNING';r['status']='TRAINING '+name;r['training_executed']=True;flush(r,out)
                    l=loaders(data)
                    metadata=dict(model=name,configuration=MODEL_CONFIGS[name],protocol=PROTOCOL,source_hashes=hashes,
                                  data_fingerprint=audit['split_fingerprint'],git_sha=r['git_sha'])
                    summary,history=train_one(m,l['train'],l['val'],device,path,metadata,
                        on_epoch=lambda history,key=KEYS[name]:atomic_json(out/f'history_{key}.json',history))
                    entry['training']=summary;entry['checkpoint_sha256']=sha256(path)
                    r['model_status'][name]='TRAINED';flush(r,out)
                best=sanity(m,fixed);r['sanity'][name].update(best=best,PASS=best['PASS'])
                if not best['PASS']:raise AssertionError('Restored best sanity failed')
                # Formal TEST for each new baseline occurs only here, once after selection.
                evaluation=evaluate(m,full,device,names);entry.update(evaluation)
                r['model_status'][name]='COMPLETE';flush(r,out)
                entry['runtime']=measure_inference(m,full['test'],device);flush(r,out)
            finally:
                del m;gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
        r['status']='COMPLETE';flush(r,out)
        print('All eight fixed runs and frozen Ours comparison complete. STOP.',out,flush=True)
        return r
    except BaseException as exc:
        r['model_status'][current]='INVALID';r['status']='INVALID: STOP review required'
        r.setdefault('errors',[]).append(dict(model=current,type=type(exc).__name__,message=str(exc),device=str(device)))
        flush(r,out);raise
    finally:
        if sha256(ours_path)!=before:raise AssertionError('OURS checkpoint modified')
        r['ours_checkpoint_unchanged']=True;flush(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from cmgm.scripts.main_ablation import build_data
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/baseline_comparison_deep_v3')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints/baselines_deep_v3')
    p.add_argument('--ours-checkpoint',type=Path,default=ROOT/'checkpoints'/f'{BASE}_best.pt')
    p.add_argument('--ours-results',type=Path,default=ROOT/'experiments/d0b_candidate_2expert_moe/20260921_151711/results.json')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--sanity-only',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--no-cuda',action='store_true');p.add_argument('--resume',help='Prepared V3 directory or latest; completed models never repeat')
    p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    if not args.run:args.sanity_only=True
    if args.run and (args.no_cuda or not torch.cuda.is_available()):raise ValueError('Formal training requires CUDA')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'.active.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seed_all();data=build_data(args);run_suite(args,device,data)


if __name__=='__main__':main()
