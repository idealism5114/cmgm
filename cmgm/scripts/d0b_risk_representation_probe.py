"""One frozen D0B extraction followed by the prescribed TRAIN-only linear probes."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess

import numpy as np
import torch

from cmgm.scripts.d0b_5d_error_regime_diagnostic import (
    ROOT,VARIANTS,prepare_data,validate_causal_features,discover_checkpoint,
    load_model,collect,sha256,state_checksum,
)
from cmgm.scripts.d0b_tail_predictability_amplitude import add_commodity_signals
from cmgm.scripts.d0b_volatility_conditional_analysis import baseline_check
from cmgm.scripts.d0b_5d_error_regime_analysis import save_results
from cmgm.config import MULTI_HORIZONS

REPRESENTATIONS=('h_temporal','h_spatial','h_fused','h_long','h_micro','h_comm')


class RepresentationCollector:
    """Hooks return None and only detach/copy actual native-path tensors."""
    def __init__(self,model,commodity_nodes):
        self.model=model
        self.nodes=np.asarray(commodity_nodes,dtype=int)
        self.arrays={key:[] for key in REPRESENTATIONS}
        self.handles=[]

    def _save(self,key,value):
        assert not torch.is_grad_enabled(),'Extraction must run under no_grad'
        assert value.shape[-1]==64
        self.arrays[key].append(value.detach().cpu().numpy().copy())

    def __enter__(self):
        def spatial_input(module,args):
            nodes=args[0]
            assert nodes.ndim==3,'Expected true (B,N,64) nodes, not batch-averaged nodes'
            assert nodes.shape[1]==self.model.num_nodes
            self._save('h_comm',nodes[:,self.nodes,:])
        def spatial_output(module,args,output):
            assert output.ndim==2
            self._save('h_spatial',output)
        def fused_input(module,args):
            assert args[0].ndim==2
            self._save('h_fused',args[0])
        def temporal_output(module,args,output):
            assert output.ndim==2
            self._save('h_temporal',output)
            self._save('h_long',module.last_h_long)
            self._save('h_micro',module.last_h_micro)
        self.handles=[self.model.type_pool.register_forward_pre_hook(spatial_input),
            self.model.type_pool.register_forward_hook(spatial_output),
            self.model.head.register_forward_pre_hook(fused_input),
            self.model.switching_latent_transformer.register_forward_hook(temporal_output)]
        return self

    def __exit__(self,*args):
        for handle in self.handles:
            handle.remove()

    def concatenated(self):
        counts={key:len(value) for key,value in self.arrays.items()}
        assert len(set(counts.values()))==1,f'Unequal native hook call counts: {counts}'
        result={key:np.concatenate(value,axis=0) for key,value in self.arrays.items()}
        assert len({value.shape[0] for value in result.values()})==1
        return result


@torch.no_grad()
def audit_cached_readout(model,representations,frame,device):
    """Replay only existing fusion/head on exports, not another input forward."""
    h,hs,ht=(torch.from_numpy(representations[k]).to(device)
             for k in ('h_fused','h_spatial','h_temporal'))
    gate=torch.sigmoid(model.gate_fc(torch.cat([hs,ht],dim=-1)))
    expected=gate*model.lstm_proj(ht)+(1-gate)*model.gcn_proj(hs)
    pred=model.head(h).view(-1,len(MULTI_HORIZONS),model.n_commodities)[:,MULTI_HORIZONS.index(5)]
    result=dict(cached_fused_vs_native_fusion_max_diff=float((h-expected).abs().max()),
        cached_head_replay_vs_observations_max_diff=float(np.max(np.abs(pred.cpu().numpy().reshape(-1)-frame.prediction.to_numpy()))),
        full_D0B_forward_repeated=False)
    assert max(v for k,v in result.items() if k.endswith('diff'))<1e-7
    return result


def extract(args):
    path=discover_checkpoint(VARIANTS['D0B'],args.checkpoint_dir,args.checkpoint)
    if path is None:
        raise FileNotFoundError('Official D0B checkpoint missing')
    torch.manual_seed(42)
    device=torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    datasets,prices,mi,fingerprint=prepare_data(args.prepared_data)
    sanity=validate_causal_features(datasets,prices,mi)
    model,metadata=load_model('D0B',path,mi,device)
    before={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
    cs,ce=mi['commodity']
    assert model.n_stock+model.n_bond==cs
    assert model.n_commodities==ce-cs
    commodity_map=[dict(commodity_index=i,node_index=node,name=str(prices.columns[node])) for i,node in enumerate(range(cs,ce))]
    for ds in datasets.values():
        assert ds.commodity_start==cs and ds.commodity_end==ce
        assert ds.market_indices==mi
    with RepresentationCollector(model,range(cs,ce)) as collector:
        frame=collect(model,'D0B',datasets,prices,mi,device)
    representations=collector.concatenated()
    frame=add_commodity_signals(frame,datasets,prices,mi)
    alignment_audit=audit_cached_readout(model,representations,frame,device)
    origins=frame[['split','sample_index','forecast_origin']].drop_duplicates().reset_index(drop=True)
    assert len(origins)==len(representations['h_fused'])
    # Native collect verifies each date and target against raw prices. Here verify
    # the expansion identity, commodity names and local node index before probes.
    keys=frame[['split','sample_index','forecast_origin','commodity_index','commodity']]
    for split,ds in datasets.items():
        sub=keys[keys.split==split]
        assert np.array_equal(sub.sample_index,np.repeat(np.arange(len(ds)),ce-cs))
        assert np.array_equal(sub.commodity_index,np.tile(np.arange(ce-cs),len(ds)))
        assert np.array_equal(sub.commodity,np.tile([v['name'] for v in commodity_map],len(ds)))
    after=state_checksum(model)
    max_diff=max(float((t.detach().cpu()-before[name]).abs().max()) for name,t in model.state_dict().items())
    assert max_diff==0 and after==metadata['state_sha256_before']
    assert sha256(path)==metadata['file_sha256_before']
    metadata.update(model_parameter_max_diff=max_diff,state_sha256_after=after,file_sha256_after=sha256(path))
    output=args.output_dir or ROOT/'experiments/d0b_risk_representation_probe'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'observation_records.csv',index=False)
    origins.to_csv(output/'representation_origins.csv',index=False)
    np.savez_compressed(output/'frozen_representations.npz',**representations)
    check=baseline_check(frame)
    meta=dict(checkpoint=metadata,native_reference_check=check,source_fingerprint=fingerprint,
        causal_feature_sanity=sanity,commodity_map=commodity_map,primary_index=MULTI_HORIZONS.index(5),
        diagnostic_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        representation_shapes={k:list(v.shape) for k,v in representations.items()},
        split_origins={k:len(v) for k,v in datasets.items()},commodity_count=ce-cs,
        extraction='One pass per original TRAIN/VAL/TEST observation; read-only native forward hooks, no batch averaging',
        representation_alignment_audit=alignment_audit,
        extraction_locations=dict(h_spatial='type_pool output',h_comm='type_pool input, after native GNN + gcn_norm, verified commodity indices',
            h_fused='actual input of existing prediction head',h_temporal='native switching branch output',
            h_long='balanced long readout',h_micro='balanced micro readout'))
    save_results(meta,output/'extraction_metadata.json')
    if not check['PASS']:
        (output/'REPORT.md').write_text('STOP: native D0B reference mismatch. No probes fitted.\n'+str(check),encoding='utf-8')
        raise RuntimeError('Native reference mismatch; no probes fitted')
    return frame,representations,meta,output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,default=ROOT/'checkpoints')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--prepared-data',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-cuda',action='store_true')
    args=parser.parse_args()
    frame,reps,metadata,output=extract(args)
    from cmgm.scripts.d0b_risk_probe_analysis import analyze,write_report
    from cmgm.scripts.d0b_risk_probe_report import attach_reviewed_conclusion
    r=analyze(frame,reps,output)
    r.update(metadata)
    r['implementation_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in (
        Path(__file__),ROOT/'cmgm/scripts/d0b_risk_probe_analysis.py',ROOT/'cmgm/scripts/d0b_risk_probe_report.py',
        ROOT/'cmgm/scripts/d0b_5d_error_regime_diagnostic.py',ROOT/'cmgm/scripts/d0b_5d_error_regime_analysis.py',
        ROOT/'cmgm/scripts/d0b_tail_predictability_amplitude.py',ROOT/'cmgm/scripts/d0b_tail_amplitude_analysis.py')}
    save_results(r,output/'results.json')
    attach_reviewed_conclusion(r)
    save_results(r,output/'results.json');write_report(r,output)
    print(f"[D0B risk probe] Report: {output/'REPORT.md'}",flush=True)


if __name__=='__main__':
    main()
