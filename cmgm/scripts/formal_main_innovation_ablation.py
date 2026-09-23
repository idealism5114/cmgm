"""Revised main ablations: audit/re-evaluate by default; --run fits only missing controls."""
import argparse
import hashlib
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from cmgm import config
from cmgm.models.formal_d0b_main_ablation import NAMES,KEY,REUSE,NEW,DEFINITIONS,FULL,CANDIDATE,GLOBAL,GLOBAL_FULL,GLOBAL_NAMES,GLOBAL_DEFINITIONS,TS_VARIANT,TS_FULL,TS_NAMES,TS_DEFINITIONS,MainInnovationAblation
from cmgm.scripts.formal_main_ablation_audit import init_audit,sanity
from cmgm.scripts.baseline_protocol import seed_all,loaders,data_audit
from cmgm.scripts.formal_v2_protocol import atomic_json,sha,metrics
from cmgm.scripts.formal_ablation_study import PROTOCOL as OLD_PROTOCOL,free
from cmgm.training.train import train_epoch,validate_epoch
from cmgm.training.metric_standard import population_metrics
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL={**OLD_PROTOCOL,'name':'D0B Candidate-Aware 2-Expert MoE FORMAL MAIN-INNOVATION ABLATION',
          'configurations':list(NAMES),'formal_reference':CANDIDATE,
          'reuse_policy':'Only audited Candidate Full / exact global mixture; old Adaptive-Gate internal ablations forbidden',
          'selection':'prediction-only four-horizon VAL Huber batch mean',
          'KL':'Native beta_max=.0005 warmup20; uniform p/prior mathematically zero; all other variants retain native KL',
          'edge_propagation_control':'standard single-hop GCN x2; same learned directed A; D^-1/2(A+I)D^-1/2; no MixHop',
          'moe_auxiliary_loss':None,'routing_warmup':None,'near_tie_relative_percent':.1}


GLOBAL_PROTOCOL={**{k:v for k,v in PROTOCOL.items() if k!='edge_propagation_control'},
    'name':'Global Dual-Expert Temporal Routing Redundancy — Post-hoc Mechanism Study',
    'mode':'global-temporal','configurations':list(GLOBAL_NAMES),'formal_reference':GLOBAL,
    'reuse_policy':'Audited native Global Full only; three new Global temporal interventions; never retrain Full',
    'interpretation':'Post-hoc mechanism verification; three single seed42 runs; no causal proof or statistical significance'}


TS_PROTOCOL={**GLOBAL_PROTOCOL,
    'name':'T+S Global Dual-Expert — Post-hoc Architecture-Mechanism Study',
    'mode':'ts-global-expert','configurations':list(TS_NAMES),'formal_reference':TS_VARIANT,
    'reuse_policy':'Five new seed42 configurations, completed exact TS runs resume only; historical architectures are read-only references',
    'edge_propagation_control':'ordinary MixHopPropagation(64,64,K=2,beta=.05) x2; same learned adjacency',
    'interpretation':'Post-hoc architecture-mechanism study; one Full and four targeted controls; no tuning or causal proof'}


def suite_mode(r):
    ref=r.get('protocol',{}).get('formal_reference')
    return 'ts-global-expert' if ref==TS_VARIANT else 'global-temporal' if ref==GLOBAL else 'candidate'


def suite_names(r):
    return {'candidate':NAMES,'global-temporal':GLOBAL_NAMES,'ts-global-expert':TS_NAMES}[suite_mode(r)]


def suite_definitions(r):
    return {'candidate':DEFINITIONS,'global-temporal':GLOBAL_DEFINITIONS,'ts-global-expert':TS_DEFINITIONS}[suite_mode(r)]


def training_order(r):
    if suite_mode(r)=='ts-global-expert':return (TS_FULL,*TS_NAMES[:-1])
    if suite_mode(r)=='global-temporal':return GLOBAL_NAMES
    return (*NEW,*(n for n in NAMES if n not in NEW))


def reference_paths(args):
    if getattr(args,'mode','candidate') in ('global-temporal','ts-global-expert'):
        return ((GLOBAL_FULL,args.global_results,args.global_checkpoint),)
    return ((FULL,args.full_results,args.full_checkpoint),('w/o Candidate-Aware Routing',args.global_results,args.global_checkpoint))


def reference_rows(r):
    return r['references'] if suite_mode(r)=='ts-global-expert' else r['reuse']


def save(r,out):
    from cmgm.scripts.formal_main_ablation_report import report
    atomic_json(out/'results.json',r);atomic_json(out/'partial_results.json',r);report(r,out)


def audit_data(data):
    if (config.SEQ_LEN,config.FEATURE_DIM,config.MULTI_HORIZONS)!=(20,21,[1,5,10,20]):raise ValueError('Frozen data config changed')
    audit=data_audit(data)
    audit['mapping']=[dict(commodity=v['commodity'],node_index=v['full_node'],target_index=v['target_output'],output_index=v['target_output']) for v in audit['mapping']]
    if not audit['PASS']:raise ValueError('Data/order audit failed')
    return json.loads(json.dumps(audit))


def audit_reuse(name,source,path,audit):
    """Only the two completed native Candidate-family references are eligible."""
    row=dict(source_experiment=str(source.parent),source_report=str(source),path=str(path),
        old_checkpoint_found=path.exists(),definition_exact_match=False,reused=False)
    if not source.exists() or not path.exists():
        row['reason']='No completed native reference artifact';return row
    old=json.loads(source.read_text());cp=torch.load(path,map_location='cpu',weights_only=False);md=cp.get('metadata',{})
    # Imports are local: the historical runners import this module's audit_data.
    if name==FULL:
        from cmgm.scripts.d0b_candidate_2expert_moe import PROTOCOL as expected
    else:
        from cmgm.scripts.d0b_candidate_2expert_global_mixture import PROTOCOL as expected
    history=cp.get('history',{});vals=history.get('val_loss',[]);best_epoch=cp.get('best_epoch',0)
    completed=(bool(vals) and 0<best_epoch<=len(vals) and
        (len(vals)>=200 or len(vals)-best_epoch>=10) and
        np.isclose(cp.get('best_val_loss',float('inf')),min(vals),rtol=0,atol=1e-12))
    checks=dict(variant=md.get('variant')==REUSE[name],seed42=md.get('seed')==42,
        complete=old.get('status')=='COMPLETE' and old.get('trained') is True and completed and cp.get('training_complete',True) is True,
        protocol=md.get('config')==expected==old.get('config'),data=md.get('data')==audit==old.get('data'),
        source_metadata=md.get('source_hashes')==old.get('source_hashes'),
        history=all(v==old.get('history',{}).get(k) for k,v in history.items()) and
            set(old.get('history',{}))-set(history)<={'train_time'},
        best_epoch=best_epoch==old.get('best_checkpoint_metadata',{}).get('best_epoch'),
        initialization=old.get('initialization',{}).get('PASS') is True,
        sanity=old.get('sanity',{}).get('PASS') is True and old.get('best_sanity',{}).get('PASS') is True)
    if name!=FULL:checks['expert_initialization']=old.get('expert_initialization',{}).get('PASS') is True
    if sha(path)!=old.get('checkpoint_sha256'):raise ValueError('Reference checkpoint hash mismatch; STOP rather than retrain')
    # Record historical source drift explicitly. Current strict-load/native equivalence,
    # shared initialization and fresh predictions must pass before accepting reuse.
    drift=[s for s,h in old.get('source_hashes',{}).items() if not (ROOT/s).exists() or sha(ROOT/s)!=h]
    core=['cmgm/models/candidate_moe_fusion.py','cmgm/models/switching_latent_transformer.py']
    if name!=FULL:core.append('cmgm/models/global_mixture_fusion.py')
    # Adding TS classes is explicitly permitted, but the entire pre-existing
    # Global module prefix must be byte-for-byte the audited historical file.
    global_file='cmgm/models/global_mixture_fusion.py'
    prefix=(ROOT/global_file).read_bytes().split(b'\n\nTS_VARIANT =',1)[0]
    append_only=(name!=FULL and global_file in drift and
                 hashlib.sha256(prefix).hexdigest()==old.get('source_hashes',{}).get(global_file))
    checks['core_implementation_unchanged']=all(s in old.get('source_hashes',{}) and
        (s not in drift or (s==global_file and append_only)) for s in core)
    row['global_source_append_only_audit']=dict(needed=global_file in drift and name!=FULL,
        original_prefix_sha256=hashlib.sha256(prefix).hexdigest(),PASS=append_only or global_file not in drift)
    row.update(checks=checks,reused=all(checks.values()),definition_exact_match=checks['variant'] and checks['protocol'],
        sha256=sha(path),source_report_sha256=sha(source),best_epoch=best_epoch,
        historical_source_hashes=old.get('source_hashes',{}),source_drift=drift,
        expected_metrics=old.get('evaluation',{}).get('metrics',{}),
        training_complete=checks['complete'],completion_evidence='Legacy checkpoint has no completion flag: audited COMPLETE/trained report, identical checkpoint history, valid best epoch and early-stop/epoch-budget termination; report-only train_time is added after checkpoint serialization',
        report_only_history_keys=sorted(set(old.get('history',{}))-set(history)),
        runtime=dict(best_epoch=best_epoch,train_seconds=old.get('history',{}).get('train_time'),seconds_per_epoch=None))
    row['reason']='Eligible pending strict load and fresh reproduction; no retraining' if row['reused'] else 'REUSE REJECTED: '+','.join(k for k,v in checks.items() if not v)
    return row


def reuse_unchanged_for_gcn(r,source):
    """Explicit protocol revision: preserve ten completed rows, replace only Edge control."""
    import copy
    source=Path(source)
    if source.is_dir():source=source/'results.json'
    old=json.loads(source.read_text())
    previous_protocol={k:v for k,v in PROTOCOL.items() if k!='edge_propagation_control'}
    if old.get('status')!='COMPLETE' or old.get('errors') or old.get('protocol')!=previous_protocol or old.get('data')!=r['data']:
        raise ValueError('GCN revision requires the completed matching Candidate-MoE suite')
    allowed={'cmgm/models/formal_d0b_main_ablation.py','cmgm/scripts/formal_main_innovation_ablation.py',
        'cmgm/scripts/formal_main_ablation_audit.py','cmgm/scripts/formal_main_ablation_report.py'}
    changed={s for s,h in r['source_hashes'].items() if old.get('source_hashes',{}).get(s)!=h}
    if set(old.get('source_hashes',{}))!=set(r['source_hashes']) or not changed<=allowed:
        raise ValueError('Unrelated source changed; no automatic migration')
    if set(old.get('results',{}))!=set(NAMES):raise ValueError('Incomplete prior table')
    excluded='w/o EdgeAttnMixHop'
    for n in NAMES:
        entry=old['results'][n];path=Path(entry['path'])
        if sha(path)!=entry['sha256'] or not entry.get('sanity',{}).get('PASS'):
            raise ValueError('Prior checkpoint hash/sanity failed: '+n)
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if n not in REUSE:
            expected=dict(name=n,definition=DEFINITIONS[n],seed=42,protocol=previous_protocol,data=r['data'],source_hashes=old['source_hashes'])
            if n==excluded:
                expected['definition']='Same learned A into two ordinary MixHopPropagation(64,64,K=2,beta=.05); Candidate MoE retained'
            if cp.get('metadata')!=expected or cp.get('training_complete') is not True:
                raise ValueError('Prior training definition/protocol mismatch: '+n)
        else:
            row=r['reuse'][n]
            if not row['reused'] or row['sha256']!=entry['sha256']:
                raise ValueError('Native reference differs: '+n)
        if n==excluded:continue
        # Exact state compatibility is checked by the caller using the actual dataset.
        r['results'][n]=copy.deepcopy(entry)
        r['results'][n].update(reused=True,retained_unchanged_from=str(source.parent))
        if n not in REUSE:
            r['reuse'][n]=dict(old_checkpoint_found=True,definition_exact_match=True,reused=True,reuse_status='PASS',
                reason='Same intervention/architecture/data/protocol; only the separate Edge control changed',
                source_experiment=str(source.parent),source_report=str(source),source_report_sha256=sha(source),
                path=str(path),sha256=entry['sha256'],best_epoch=cp['best_epoch'])
    r['revision']=dict(reason='User changed EdgeAttnMixHop comparator from ordinary MixHop to standard GCN',
        previous_results=str(source),previous_results_sha256=sha(source),changed_sources=sorted(changed),
        superseded_edge_result=old['results'][excluded],retained_variants=[n for n in NAMES if n!=excluded],
        pending_variant=excluded,old_artifacts_unchanged=True)



def audit_cross_architecture(args,audit,mechanisms=GLOBAL_NAMES[:-1]):
    """Read historical evidence, never select/retrain historical configurations."""
    result={}
    for label,path,full in (('Old Adaptive Fusion',args.old_gate_results,'Full D0B'),
                            ('Dynamic Candidate MoE',args.dynamic_results,FULL)):
        path=Path(path);old=json.loads(path.read_text())
        if old.get('status')!='COMPLETE' or old.get('errors') or old.get('data')!=audit:
            raise ValueError('Historical completion/data audit failed: '+str(path))
        for key in ('seed','seq_len','features','horizons','optimizer','lr','weight_decay',
                    'batch_size','epochs','patience','scheduler','prediction_loss','train_shuffle','train_drop_last'):
            if old['protocol'].get(key)!=GLOBAL_PROTOCOL.get(key):raise ValueError('Historical protocol mismatch: '+key)
        rows={};provenance={}
        for n in (*mechanisms,full):
            e=old['results'][n];checkpoint=Path(e['path'])
            if sha(checkpoint)!=e['sha256'] or not e.get('sanity',{}).get('PASS'):
                raise ValueError('Historical checkpoint/sanity mismatch: '+n)
            cp=torch.load(checkpoint,map_location='cpu',weights_only=False);md=cp.get('metadata',{})
            record=old.get('reuse',{}).get(n,{})
            src_path=path;src=old;old_name=n
            if e.get('reused'):
                src_path=Path(record.get('source_report') or str(Path(e['source_experiment'])/'results.json'))
                src=json.loads(src_path.read_text());old_name=record.get('old_name',n)
                if not record.get('reused') or not record.get('definition_exact_match'):
                    raise ValueError('Historical reuse was not audited: '+n)
            native_candidate=n==FULL
            if native_candidate:
                reference=audit_reuse(FULL,src_path,checkpoint,audit)
                if not reference['reused'] or not record.get('fresh_reproduction',{}).get('PASS') or not record.get('strict_load'):
                    raise ValueError('Historical Candidate Full is not verified')
                source_metric=src['evaluation']['metrics']['test']['5']
            else:
                if (cp.get('training_complete') is not True or md.get('seed')!=42 or md.get('data')!=audit
                    or md.get('name')!=old_name or md.get('protocol')!=src.get('protocol')
                    or md.get('source_hashes')!=src.get('source_hashes') or src.get('status')!='COMPLETE'):
                    raise ValueError('Historical metadata/completion mismatch: '+n)
                source_entry=src['results'][old_name]
                if source_entry['sha256']!=e['sha256'] or not source_entry['sanity']['PASS']:
                    raise ValueError('Historical source result mismatch: '+n)
                expected_definition=('Same learned A into two ordinary MixHopPropagation(64,64,K=2,beta=.05); Candidate MoE retained' if n=='w/o EdgeAttnMixHop' else DEFINITIONS.get(n))
                if label=='Dynamic Candidate MoE' and md.get('definition')!=expected_definition:
                    raise ValueError('Historical Candidate intervention differs: '+n)
                source_metric=source_entry['metrics']['test']['5']
            if n=='w/o EdgeAttnMixHop' and not md.get('definition','').startswith('Same learned A into two ordinary MixHopPropagation(64,64,K=2,beta=.05)'):
                raise ValueError('Historical Edge comparator is not ordinary MixHop')
            state=cp['model_state_dict']
            is_candidate=any(k.startswith('candidate_moe_fusion.') for k in state)
            if is_candidate!=(label=='Dynamic Candidate MoE') or any(k.startswith('global_mixture_fusion.') for k in state):
                raise ValueError('Historical fusion architecture mismatch: '+n)
            metric=e['metrics']['test']['5']
            if not all(np.isclose(metric[k],source_metric[k],rtol=1e-5,atol=1e-9) for k in ('MAE','MSE','RMSE','Hit')):
                raise ValueError('Historical source metrics differ: '+n)
            rows[n]=metric
            provenance[n]=dict(checkpoint=str(checkpoint),sha256=e['sha256'],source_report=str(src_path),
                source_report_sha256=sha(src_path),best_epoch=cp['best_epoch'],PASS=True)
        base=rows[full]['MAE']
        result[label]=dict(source_report=str(path),source_report_sha256=sha(path),full_name=full,
            metrics=rows,relative_deltas={n:100*(rows[n]['MAE']-base)/base for n in mechanisms},
            provenance=provenance,PASS=True,scope='Historical audited results; explanatory only; no retraining or model selection')
    return result



def audit_ts_history(args,audit):
    """Exact old MixHop comparator; never substitute the later GCN experiment."""
    result=audit_cross_architecture(args,audit,mechanisms=TS_NAMES[:-1])
    source=Path(args.tst_global_results);old=json.loads(source.read_text())
    if old.get('protocol')!=GLOBAL_PROTOCOL or old.get('data')!=audit or old.get('status')!='COMPLETE' or old.get('errors'):
        raise ValueError('T+ST Global temporal study is not a completed exact-protocol reference')
    rows={};provenance={}
    for n in GLOBAL_NAMES:
        e=old['results'][n];path=Path(e['path'])
        if sha(path)!=e['sha256'] or not e.get('sanity',{}).get('PASS'):
            raise ValueError('T+ST Global historical checkpoint/sanity failed: '+n)
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if n==GLOBAL_FULL:
            record=old['reuse'][n]
            native=audit_reuse(n,Path(record['source_report']),path,audit)
            if not native['reused'] or not record.get('strict_load') or not record.get('fresh_reproduction',{}).get('PASS'):
                raise ValueError('T+ST Global Full reference not audited')
        else:
            expected=dict(name=n,definition=GLOBAL_DEFINITIONS[n],seed=42,protocol=GLOBAL_PROTOCOL,
                data=audit,source_hashes=old['source_hashes'])
            if cp.get('metadata')!=expected or cp.get('training_complete') is not True:
                raise ValueError('T+ST Global ablation definition/metadata mismatch: '+n)
        state=cp['model_state_dict']
        if not any(k.startswith('global_mixture_fusion.interaction_expert.') for k in state) or any('candidate_moe_fusion' in k or 'spatial_expert' in k for k in state):
            raise ValueError('Historical expert architecture mismatch')
        rows[n]=e['metrics']['test']['5']
        provenance[n]=dict(checkpoint=str(path),sha256=e['sha256'],best_epoch=cp['best_epoch'],
            definition=GLOBAL_DEFINITIONS[n],PASS=True)
    base=rows[GLOBAL_FULL]['MAE']
    result['T+ST Global Mixture']=dict(source_report=str(source),source_report_sha256=sha(source),full_name=GLOBAL_FULL,
        metrics=rows,relative_deltas={n:100*(rows[n]['MAE']-base)/base for n in GLOBAL_NAMES[:-1]},
        provenance=provenance,PASS=True,edge_control='N/A; no completed ordinary-MixHop control in this formal study')
    return result


def setup(args,data):
    if (config.TARGET_TYPE,config.LOSS_TYPE,config.HUBER_DELTA,config.LEARNING_RATE,config.WEIGHT_DECAY)!=('return','huber',.02,1e-4,1e-5):
        raise ValueError('Frozen loss/optimizer configuration changed')
    if args.resume and getattr(args,'revise_edge_gcn_from',None):raise ValueError('Choose resume OR explicit GCN revision')
    mode=getattr(args,'mode','candidate');protocol={'candidate':PROTOCOL,'global-temporal':GLOBAL_PROTOCOL,'ts-global-expert':TS_PROTOCOL}[mode]
    names=suite_names({'protocol':protocol})
    if mode!='candidate' and getattr(args,'revise_edge_gcn_from',None):raise ValueError('GCN revision does not belong to the targeted global experiment')
    audit=audit_data(data)
    sources=['cmgm/models/formal_d0b_main_ablation.py','cmgm/scripts/formal_main_innovation_ablation.py',
        'cmgm/scripts/formal_main_ablation_audit.py','cmgm/scripts/formal_main_ablation_report.py',
        'cmgm/models/candidate_moe_fusion.py','cmgm/models/global_mixture_fusion.py','cmgm/models/hetero_mixhop_model.py',
        'cmgm/models/switching_latent_transformer.py','cmgm/models/model.py','cmgm/graph/adaptive_graph.py',
        'cmgm/training/train.py','cmgm/training/metric_standard.py','cmgm/scripts/baseline_protocol.py',
        'cmgm/scripts/main_ablation.py','cmgm/data/data_loader.py','cmgm/data/feature_builder.py','cmgm/config.py']
    hashes={s:sha(ROOT/s) for s in sources}
    if args.resume:
        options=[]
        for path in sorted(args.output_dir.glob('*/results.json')):
            if json.loads(path.read_text()).get('protocol')==protocol:options.append(path)
        if args.resume=='latest' and not options:raise ValueError('No prepared experiment for the selected mode; a different fusion architecture is ineligible')
        out=options[-1].parent if args.resume=='latest' else Path(args.resume)
        r=json.loads((out/'results.json').read_text())
        if r['protocol']!=protocol or r['data']!=audit or r['source_hashes']!=hashes:raise ValueError('Resume data/protocol/implementation mismatch; STOP review')
        if r.get('errors'):raise ValueError('Invalid/interrupted experiment requires review, no silent rerun')
        if r['checkpoint_dir']!=str(args.checkpoint_dir/out.name):raise ValueError('Checkpoint destination changed')
        for name,source,path in reference_paths(args):
            if Path(reference_rows(r)[name]['path']).resolve()!=path.resolve() or Path(reference_rows(r)[name]['source_report']).resolve()!=source.resolve():
                raise ValueError('Resume reference paths changed')
    else:
        for path in args.output_dir.glob('*/results.json'):
            if json.loads(path.read_text()).get('protocol')==protocol:raise ValueError('Use --resume latest; duplicate suite for selected mode forbidden')
        out=args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
        reuse={n:audit_reuse(n,source,path,audit) for n,source,path in reference_paths(args)}
        full=GLOBAL_FULL if mode!='candidate' else FULL
        if not reuse[full]['reused']:raise ValueError('Formal Full reference audit failed: '+str(reuse[full]))
        refs=reuse if mode=='ts-global-expert' else None
        if refs is not None:reuse={}
        for n in names:
            if n in reuse:continue
            reuse[n]=dict(old_checkpoint_found=False,definition_exact_match=False,reused=False,
            reason='Must estimate under this exact Full architecture; other fusion architectures are ineligible')
        r=dict(protocol=protocol,data=audit,source_hashes=hashes,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            reuse=reuse,jobs={},results={},initialization={},sanity={},status='PREPARING',checkpoint_dir=str(args.checkpoint_dir/out.name))
        if refs is not None:r.update(references=refs,reference_results={})
        if getattr(args,'revise_edge_gcn_from',None):
            reuse_unchanged_for_gcn(r,args.revise_edge_gcn_from)
            for name,entry in r['results'].items():
                seed_all(42);model=MainInnovationAblation(name,data)
                model.load_state_dict(torch.load(entry['path'],map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
                del model
    if mode in ('global-temporal','ts-global-expert'):
        historical=audit_ts_history(args,audit) if mode=='ts-global-expert' else audit_cross_architecture(args,audit)
        if r.get('cross_architecture') not in (None,historical):raise ValueError('Historical explanatory artifacts changed')
        r['cross_architecture']=historical
    save(r,out);return r,out


@torch.no_grad()
def evaluate(model,data,device):
    result=dict(metrics={},per_commodity=[],mechanism={})
    name=model.main_name
    for split,loader in loaders(data,full=True).items():
        preds=[];targets=[];probs=[];longs=[];micros=[];disagreements=[];routing=[];expert_norms=[];model.eval()
        for batch in loader:
            pred=model(batch[0].to(device));y=batch[1]
            if pred.shape!=y.shape or not torch.isfinite(pred).all():raise ValueError('Prediction shape/finite failure')
            preds.append(pred.cpu().numpy());targets.append(y.numpy());b=model.switching_latent_transformer
            if split=='test' and hasattr(model,'global_mixture_fusion'):
                d=model.global_mixture_fusion.last
                expert_norms.append(torch.stack([d[k].norm(dim=-1) for k in ('e_T','e_S' if model.ablation_mode=='ts-global-expert' else 'e_ST')],-1).cpu())
            if hasattr(model,'candidate_moe_fusion'):routing.append(model.candidate_moe_fusion.last['pi'].cpu())
            if split=='test' and (model.ablation_mode in ('global-temporal','ts-global-expert') or name in (FULL,'w/o Adaptive Regime Routing')):
                probs.append(b.last_regime_probabilities.cpu())
            if split=='test' and (model.ablation_mode in ('global-temporal','ts-global-expert') or name in (FULL,'w/o Regime-Specific Transitions')):
                c=b.last_latent_candidates
                disagreements.append(torch.stack([(c[:,:,i]-c[:,:,j]).abs().mean(-1) for i,j in ((0,1),(0,2),(1,2))],-1).cpu())
            if model.ablation_mode in ('global-temporal','ts-global-expert') or name in (FULL,'w/o Balanced Readout'):
                longs.append(b.last_h_long.norm(dim=-1).cpu());micros.append(b.last_h_micro.norm(dim=-1).cpu())
        p,y=np.concatenate(preds),np.concatenate(targets);result['metrics'][split]=metrics(p,y)
        if split=='test':
            idx=config.MULTI_HORIZONS.index(5);cs,ce=data['market_indices']['commodity']
            result['per_commodity']=[dict(commodity=str(data['feature_names'][cs+i]),**population_metrics(p[:,idx,i],y[:,idx,i])) for i in range(ce-cs)]
        if expert_norms:
            norms=torch.cat(expert_norms).double().mean(0)
            result['mechanism']['test_expert_norms']=(dict(temporal_expert_mean_L2=float(norms[0]),spatial_expert_mean_L2=float(norms[1]),spatial_temporal_norm_ratio=float(norms[1]/(norms[0]+1e-8))) if model.ablation_mode=='ts-global-expert' else dict(temporal_mean_L2=float(norms[0]),interaction_mean_L2=float(norms[1])))
        if routing:
            pi=torch.cat(routing).double()
            result['mechanism'].setdefault('router',{})[split]=dict(mean_pi=pi.mean(0).tolist(),std_pi=pi.std(0,unbiased=False).tolist(),
                P10=torch.quantile(pi,.1,dim=0).tolist(),P50=torch.quantile(pi,.5,dim=0).tolist(),P90=torch.quantile(pi,.9,dim=0).tolist(),
                entropy=float(-(pi*(pi+1e-8).log()).sum(-1).mean()))
        if hasattr(model,'global_mixture_fusion'):
            result['mechanism']['global_mixture']=dict(model.global_mixture_fusion.weight_diagnostics(),scope='one global pair across every sample; not sample-dependent')
        if hasattr(model,'simple_fusion'):
            result['mechanism']['simple_fusion']=dict(parameters=sum(p.numel() for p in model.simple_fusion.parameters()),
                experts_absent=not hasattr(model,'candidate_moe_fusion'),router_absent=not hasattr(model,'candidate_moe_fusion'),input='[s||t]',shared_head='unchanged')
        if probs:
            prob=torch.cat(probs).double()
            result['mechanism']['regime']=dict(mean_p=prob.mean((0,1)).tolist(),entropy=float(-(prob*torch.log(prob+1e-8)).sum(-1).mean()),
                hard_occupancy=torch.bincount(prob.argmax(-1).flatten(),minlength=3).double().div(prob.shape[0]*prob.shape[1]).tolist(),
                occupancy_note='This is deterministic tie breaking under uniform probabilities, not learned collapse.' if name=='w/o Adaptive Regime Routing' else 'Descriptive learned regime occupancy.')
        if disagreements:result['mechanism']['candidate_pairwise_mean_absolute_disagreement']=float(torch.cat(disagreements).double().mean())
        if longs:
            h=float(torch.cat(longs).double().mean());z=float(torch.cat(micros).double().mean())
            result['mechanism'][split]=dict(long_norm=h,micro_norm=z,micro_long_ratio=z/(h+1e-8))
    return result


def preflight(r,out,data,device):
    x,y=next(iter(loaders(data,full=True)['train']))[:2];x=x[:2].to(device);y=y[:2].to(device)
    for n in suite_names(r):
        if r['sanity'].get(n,{}).get('PASS') and r['sanity'][n].get('device')==str(device):continue
        model,initial=init_audit(n,data,device,x,mode=suite_mode(r));r['initialization'][n]=initial
        if not initial['PASS']:save(r,out);raise AssertionError(n+' shared initialization failed')
        check=sanity(model,x,y);check.update(device=str(device),split='TRAIN');r['sanity'][n]=check
        print('[Revised sanity]',n,'params',initial['total_instantiated'],'shared diff',initial['shared_parameter_initial_max_abs_diff'],'PASS',check['PASS'],flush=True)
        del model;free();save(r,out)
        if not check['PASS']:raise AssertionError(n+' sanity failed')


def evaluate_reused(r,out,data,device):
    ts=suite_mode(r)=='ts-global-expert'
    destination=r['reference_results'] if ts else r['results']
    for n in ((GLOBAL_FULL,) if suite_mode(r) in ('global-temporal','ts-global-expert') else (FULL,'w/o Candidate-Aware Routing')):
        row=reference_rows(r)[n]
        if not row['reused']:continue
        if sha(row['path'])!=row['sha256'] or sha(row['source_report'])!=row['source_report_sha256']:
            raise ValueError('Read-only reference artifact changed')
        if n in destination:continue
        cp=torch.load(row['path'],map_location='cpu',weights_only=False)
        seed_all(42);m=MainInnovationAblation(n,data,mode='global-temporal' if ts else suite_mode(r)).to(device);m.load_state_dict(cp['model_state_dict'],strict=True)
        x,y=next(iter(loaders(data,full=True)['train']))[:2]
        check=sanity(m,x[:2].to(device),y[:2].to(device))
        if not check['PASS']:raise AssertionError(n+' reused best sanity failed')
        result=evaluate(m,data,device);expected=row['expected_metrics']
        errors={s:{h:{k:abs(v-expected[s][h][k]) for k,v in values.items()} for h,values in hm.items()} for s,hm in result['metrics'].items()}
        ok=all(np.isclose(v,expected[s][h][k],rtol=1e-5,atol=1e-9) for s,hm in result['metrics'].items() for h,values in hm.items() for k,v in values.items())
        row['fresh_reproduction']=dict(PASS=bool(ok),absolute_errors=errors,rtol=1e-5,atol=1e-9)
        if not ok:save(r,out);raise ValueError(n+' fresh evaluation does not reproduce audited artifact; STOP')
        row['strict_load']=True;row['reuse_status']='PASS'
        row['sha256_after_fresh_evaluation']=sha(row['path'])
        if row['sha256_after_fresh_evaluation']!=row['sha256']:raise ValueError('Read-only Full checkpoint changed')
        destination[n]=dict(result,path=row['path'],sha256=row['sha256'],reused=True,source_experiment=row['source_experiment'],runtime=row['runtime'],sanity=check)
        print('[Fresh reference]',n,'TEST5 MAE',result['metrics']['test']['5']['MAE'],flush=True)
        del m,cp;free();save(r,out)


def fit(n,m,data,device,path,metadata,out):
    optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    criterion=torch.nn.HuberLoss(delta=.02);sources=loaders(data);history=[];best=float('inf');stale=0;start=time.perf_counter()
    edge_index=torch.empty((2,0),dtype=torch.long,device=device);edge_weight=torch.empty(0,device=device)
    for epoch in range(1,201):
        tic=time.perf_counter();beta=m.switching_latent_transformer.set_epoch(epoch)
        train=train_epoch(m,sources['train'],edge_index,edge_weight,optimizer,criterion,device)
        val=validate_epoch(m,sources['val'],edge_index,edge_weight,criterion,device)
        if not np.isfinite(train) or not np.isfinite(val):raise FloatingPointError('Invalid nonfinite training run')
        from cmgm.scripts.baseline_protocol import validation
        _,secondary=validation(m,sources['val'],device)
        scheduler.step(val)
        row=dict(epoch=epoch,train_total_loss=train,val_objective=val,val5_MAE=secondary['MAE'],val5_MSE=secondary['MSE'],LR=optimizer.param_groups[0]['lr'],switch_beta=0 if m.disable_switch_kl else beta,seconds=time.perf_counter()-tic)
        history.append(row)
        if val<best:
            best=val;stale=0
            cp=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},best_epoch=epoch,best_val_objective=val,metadata=metadata,training_complete=False)
            temp=Path(str(path)+'.tmp');torch.save(cp,temp);os.replace(temp,path)
        else:stale+=1
        atomic_json(out/f'history_{KEY[n]}.json',history)
        print('[Formal ablation]',n,'epoch',epoch,'train',train,'val',val,'secondary VAL5',secondary['MAE'],flush=True)
        if stale>=10:break
    cp=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(cp['model_state_dict']);m.switching_latent_transformer.set_epoch(cp['best_epoch'])
    cp.update(training_complete=True,history=history,runtime=dict(best_epoch=cp['best_epoch'],train_seconds=time.perf_counter()-start,seconds_per_epoch=float(np.mean([h['seconds'] for h in history])),val5_at_best=history[cp['best_epoch']-1]['val5_MAE']))
    temp=Path(str(path)+'.tmp');torch.save(cp,temp);os.replace(temp,path)
    return cp


def run_one(n,r,out,data,device,args):
    if n in r['results']:
        if sha(r['results'][n]['path'])!=r['results'][n]['sha256']:raise ValueError('Completed artifact changed')
        return
    if n in (FULL,GLOBAL_FULL):raise AssertionError('Formal Full must never train')
    if n not in suite_names(r):raise ValueError('Variant outside selected experiment')
    if r['reuse'][n]['reused']:raise AssertionError('Eligible checkpoint must be reevaluated, never retrained')
    seed_all(42);m=MainInnovationAblation(n,data,mode=suite_mode(r)).to(device)
    path=(ROOT/'checkpoints'/f'{TS_VARIANT}_best.pt') if n==TS_FULL else Path(r['checkpoint_dir'])/(KEY[n]+'_seed42.pt')
    path.parent.mkdir(parents=True,exist_ok=True)
    md=dict(name=n,definition=suite_definitions(r)[n],seed=42,protocol=r['protocol'],data=r['data'],source_hashes=r['source_hashes'])
    cp=None
    if path.exists():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if cp['metadata']!=md:raise ValueError('Checkpoint implementation/data/protocol mismatch')
        if not cp['training_complete']:raise ValueError('Interrupted run; STOP for manual implementation/resource review. No automatic retraining.')
        m.load_state_dict(cp['model_state_dict'],strict=True)
    if cp is None:
        if n in r['jobs']:raise ValueError('Previously started run cannot be silently restarted')
        r['jobs'][n]=dict(status='RUNNING',path=str(path));r['status']='TRAINING '+n;save(r,out)
        cp=fit(n,m,data,device,path,md,out)
    r['jobs'][n]=dict(status='FITTED',path=str(path))
    if suite_mode(r)=='ts-global-expert':atomic_json(out/f'training_history_{KEY[n]}.json',cp['history'])
    save(r,out)
    x,y=next(iter(loaders(data,full=True)['val']))[:2]
    check=sanity(m,x[:2].to(device),y[:2].to(device))
    if not check['PASS']:
        r['sanity'][n]['best']=check;save(r,out);raise AssertionError(n+' best-checkpoint sanity failed')
    result=evaluate(m,data,device)
    r['results'][n]=dict(result,path=str(path),sha256=sha(path),reused=False,runtime=cp['runtime'],sanity=check)
    if suite_mode(r)=='ts-global-expert':
        r['results'][n]['best_checkpoint_metadata']={k:cp[k] for k in ('best_epoch','best_val_objective','training_complete','metadata')}
    r['jobs'][n]['status']='DONE';print('[Completed]',n,flush=True)
    del m,cp;free();save(r,out)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--resume');p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--mode',choices=('candidate','global-temporal','ts-global-expert'),default='candidate')
    p.add_argument('--old-gate-results',type=Path,default=ROOT/'experiments/formal_main_innovation_ablation/20260916_154242/results.json')
    p.add_argument('--tst-global-results',type=Path,default=ROOT/'experiments/formal_main_innovation_ablation/global_temporal/20260923_155317/results.json')
    p.add_argument('--dynamic-results',type=Path,default=ROOT/'experiments/formal_main_innovation_ablation/20260922_151741/results.json')
    p.add_argument('--revise-edge-gcn-from',type=Path,help='Completed previous Candidate suite; keep ten rows and prepare only the revised GCN control')
    p.add_argument('--full-results',type=Path,default=ROOT/'experiments/d0b_candidate_2expert_moe/20260921_151711/results.json')
    p.add_argument('--full-checkpoint',type=Path,default=ROOT/'checkpoints'/f'{CANDIDATE}_best.pt')
    p.add_argument('--global-results',type=Path,default=ROOT/'experiments/d0b_candidate_2expert_global_mixture/20260921_214032/results.json')
    p.add_argument('--global-checkpoint',type=Path,default=ROOT/'checkpoints'/f'{GLOBAL}_best.pt')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--checkpoint-dir',type=Path,default=None)
    p.set_defaults(batch_size=64,seq_len=20);args=p.parse_args()
    args.output_dir=args.output_dir or ROOT/('experiments/d0b_ts_global_2expert_mechanism' if args.mode=='ts-global-expert' else 'experiments/formal_main_innovation_ablation')
    args.checkpoint_dir=args.checkpoint_dir or ROOT/('checkpoints/formal_main_innovation_ablation/ts_global' if args.mode=='ts-global-expert' else 'checkpoints/formal_main_innovation_ablation')
    if args.mode=='global-temporal':
        args.output_dir=args.output_dir/'global_temporal'
        args.checkpoint_dir=args.checkpoint_dir/'global_temporal'
    if args.cpu_check and args.run:raise ValueError('CPU formal training prohibited')
    if not args.cpu_check and not torch.cuda.is_available():raise ValueError('GPU unavailable; no CPU training fallback')
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    # Share the old ablation lock too, so these two D0B-family suites cannot overlap.
    old_lock=ROOT/'checkpoints/formal_ablation/.active.lock';old_lock.parent.mkdir(parents=True,exist_ok=True)
    with old_lock.open('a+') as old, (args.checkpoint_dir/'.active.lock').open('a+') as lock:
        for handle in (old,lock):
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('A D0B formal-ablation process is active; do not run concurrently')
        from cmgm.scripts.main_ablation import build_data
        seed_all(42);data=build_data(args);device=torch.device('cpu' if args.cpu_check else 'cuda');r,out=setup(args,data)
        try:
            print('REUSE:',[n for n in suite_names(r) if r['reuse'][n]['reused']],flush=True)
            print('NEED NEW TRAINING:',[n for n in suite_names(r) if not r['reuse'][n]['reused'] and n not in r['results']],flush=True)
            if suite_mode(r)=='ts-global-expert':print('READ-ONLY T+ST REFERENCE:',list(r['references']),flush=True)
            evaluate_reused(r,out,data,device);preflight(r,out,data,device)
            if args.run:
                for n in training_order(r):run_one(n,r,out,data,device,args)
            pending=[n for n in suite_names(r) if n not in r['results']]
            r['status']='COMPLETE' if not pending else 'PREPARED: '+str(len(pending))+' unique runs pending'
            save(r,out)
        except BaseException as exc:
            r['status']='STOP '+type(exc).__name__;r.setdefault('errors',[]).append(str(exc));save(r,out);raise
        finally:
            for row in reference_rows(r).values():
                if row['reused'] and (sha(row['path'])!=row['sha256'] or sha(row['source_report'])!=row['source_report_sha256']):raise AssertionError('Read-only source artifact changed')
        print('Revised report:',out,'STOP',flush=True)


if __name__=='__main__':main()
