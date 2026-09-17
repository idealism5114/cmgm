"""Faithful TCN repair; preserve eight formal rows; one corrected seed42 fit only."""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import torch
from cmgm.models.formal_baselines_v2 import make_neural
from cmgm.scripts.baseline_protocol import seed_all,loaders
from cmgm.scripts.formal_v2_audit import ROOT,sanity
from cmgm.scripts.formal_v2_protocol import sha,atomic_json,input_audit
from cmgm.scripts.formal_baseline_single_run import save,process_model,release

TCN_ROOT=ROOT/'third_party/baselines/tcn'


def verify_provenance():
    p=json.loads((TCN_ROOT/'provenance.json').read_text())
    for file,digest in p['files'].items():
        if sha(TCN_ROOT/file)!=digest:raise ValueError('TCN source provenance mismatch: '+file)
    return p


def source_hashes():
    paths=['cmgm/models/formal_baselines_v2.py','cmgm/scripts/formal_tcn_repair.py','cmgm/scripts/baseline_protocol.py',
           'cmgm/scripts/formal_baseline_single_run.py','cmgm/scripts/formal_v2_protocol.py','third_party/baselines/tcn/tcn.py']
    return {p:sha(ROOT/p) for p in paths}


def prepare(args,data):
    p=verify_provenance();audit=input_audit(data);hashes=source_hashes()
    source=sorted(args.source_dir.glob('*/results.json'))[-1] if args.source=='latest' else Path(args.source)
    old=json.loads(source.read_text())
    if old['status']!='COMPLETE' or len(old['models'])!=9:raise ValueError('Expected complete fixed-config source suite')
    if old['input_audit']!=audit:raise ValueError('Original data/split/input equality mismatch')
    if old['current_source_hashes']['cmgm/scripts/baseline_protocol.py']!=hashes['cmgm/scripts/baseline_protocol.py']:raise ValueError('Original training protocol changed')
    if args.resume:
        paths=sorted(args.output_dir.glob('*/results.json'));out=paths[-1].parent if args.resume=='latest' and paths else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['repair']['source_manifest_sha256']!=sha(source) or r['repair']['implementation_hashes']!=hashes or r['input_audit']!=audit:raise ValueError('Repair source/configuration changed')
    else:
        if any(args.output_dir.glob('*/results.json')):raise ValueError('One repair only; resume latest, do not create another fit')
        out=args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
        r=copy.deepcopy(old)
        invalid=r['models'].pop('TCN');invalid.update(status='INVALID_FOR_FINAL_BASELINE_TABLE',reason='architecture fidelity issue')
        r['invalidated_TCN']=invalid;r['jobs']={};r['sanity'].pop('TCN',None)
        r['status']='TCN REPAIR PREPARED: corrected single fit pending'
        r['checkpoint_dir']=str(args.checkpoint_dir/out.name)
        r['repair']=dict(name='FORMAL TCN BASELINE REPAIR + SINGLE RERUN',source_manifest=str(source),source_manifest_sha256=sha(source),implementation_hashes=hashes,
            reason='architecture fidelity issue',permitted_fit='TCN only, seed42 once; no performance-based retry',other_eight='unchanged results and checkpoints; no fitting or reevaluation')
        r['provenance']['TCN_corrected']=p;r['provenance']['classes']['TCN']='locuslab/TCN TemporalConvNet; WeightNorm+Chomp; direct N*21 input'
        r['current_source_hashes']=hashes
        r['computation']=dict(new_fits_required=1,grid_runs=0,repeated_seeds=0,other_model_training=0)
        for n,row in r['plan'].items():
            row.update(action='Retrain' if n=='TCN' else 'Reuse',reason='Architecture fidelity repair only' if n=='TCN' else 'Original formal row preserved exactly; no reevaluation')
        r['plan']['TCN'].update(sanity_PASS=False,fixed_config_matches=False)
        atomic_json(source.parent/'TCN_INVALIDATION.json',dict(status='INVALID_FOR_FINAL_BASELINE_TABLE',reason='architecture fidelity issue',old_checkpoint=invalid['path'],old_sha256=invalid['sha256'],replacement_report=str(out/'REPORT.md')))
        legacy=source.parent/'REPORT.md';text=legacy.read_text()
        if not text.startswith('> TCN INVALID_FOR_FINAL_BASELINE_TABLE'):
            legacy.write_text('> TCN INVALID_FOR_FINAL_BASELINE_TABLE: architecture fidelity issue. This table is a historical snapshot; the other eight results remain valid. Corrected TCN amendment: '+str(out/'REPORT.md')+'\n\n'+text)
    for n,e in r['models'].items():
        if n!='TCN' and e!=old['models'][n]:raise AssertionError('Non-TCN result changed')
        if sha(e['path'])!=e['sha256']:raise AssertionError('Completed checkpoint changed: '+n)
    if sha(r['invalidated_TCN']['path'])!=r['invalidated_TCN']['sha256']:raise AssertionError('Old TCN checkpoint changed')
    atomic_json(out/'protocol.json',r['protocol']);atomic_json(out/'fixed_configurations.json',r['fixed_configurations'])
    atomic_json(out/'baseline_provenance.json',r['provenance']);atomic_json(out/'input_equality_audit.json',audit)
    (out/'ADAPTATION_NOTES.md').write_text((TCN_ROOT/'ADAPTATION_NOTES.md').read_text())
    save(r,out);return r,out


def preflight(r,out,data,device):
    seed_all(42);m=make_neural('TCN',data,device).eval();x=next(iter(loaders(data,full=True)['val']))[0][:2].to(device)
    norms=[]
    for i,b in enumerate(m.tcn.network):
        assert b.conv1.in_channels==(data['n_nodes']*21 if i==0 else 128)
        assert b.conv1.dilation==(2**i,) and b.conv2.dilation==(2**i,)
        assert b.chomp1.chomp_size==2*2**i and b.chomp2.chomp_size==2*2**i
        assert (b.downsample is not None)==(i==0)
        for name,c in [('conv1',b.conv1),('conv2',b.conv2),*([('downsample',b.downsample)] if b.downsample is not None else [])]:
            w=c.weight.detach().clone();sd=float(w.std());norms.append(dict(block=i,layer=name,mean=float(w.mean()),std=sd))
            assert .009<sd<.011
            if name!='downsample':assert hasattr(c,'parametrizations')
    before={n:p.clone() for n,p in m.named_parameters()}
    check=sanity('TCN',m,x)
    assert all(torch.equal(before[n],p) for n,p in m.named_parameters())
    assert check['PASS'],check
    # Explicitly verify effective weights survive parametrization recomputation.
    for i,b in enumerate(m.tcn.network):
        for name in ('conv1','conv2'):
            assert float(getattr(b,name).weight.std())==next(row['std'] for row in norms if row['block']==i and row['layer']==name)
    r['sanity']['TCN']=dict(initial=check,initial_weight_statistics=norms,params=sum(p.numel() for p in m.parameters()),PASS=True)
    r['plan']['TCN']['sanity_PASS']=True
    del m;release();save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--cpu-check',action='store_true');p.add_argument('--resume')
    p.add_argument('--source',default='latest');p.add_argument('--source-dir',type=Path,default=ROOT/'experiments/formal_baseline_benchmark_v2_single_run')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/formal_tcn_repair');p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints/formal_tcn_repair')
    p.set_defaults(batch_size=64,seq_len=20,retry_invalid=False);args=p.parse_args()
    if args.cpu_check and args.run:raise ValueError('Formal rerun must use GPU')
    if not args.cpu_check and not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no CPU fit fallback')
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    with (args.checkpoint_dir/'.active.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from cmgm.scripts.main_ablation import build_data
        seed_all(42);data=build_data(args);device=torch.device('cpu' if args.cpu_check else 'cuda');r,out=prepare(args,data)
        old=json.loads(Path(r['repair']['source_manifest']).read_text())
        try:
            if 'TCN' not in r['models']:
                preflight(r,out,data,device)
                if args.run:process_model('TCN',r,out,data,device,args,allow_fit=True)
            r['status']='COMPLETE: TCN corrected once; other eight rows unchanged' if 'TCN' in r['models'] else 'PREPARED: one corrected TCN GPU fit pending'
            save(r,out)
        except BaseException as e:
            r['status']='STOP: '+type(e).__name__;r.setdefault('errors',[]).append(str(e));save(r,out);raise
        finally:
            for n,v in old['models'].items():
                if sha(v['path'])!=v['sha256']:raise AssertionError('Original checkpoint changed: '+n)
                if n!='TCN' and r['models'][n]!=v:raise AssertionError('Non-TCN row changed')
        print('TCN repair report:',out,'STOP',flush=True)


if __name__=='__main__':main()
