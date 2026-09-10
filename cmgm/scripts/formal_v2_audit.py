"""Read-only provenance, information and causal/batch checks. No fitting."""
import copy
import importlib.metadata
import json
from pathlib import Path
import torch
from torch import nn
from cmgm.scripts.formal_v2_protocol import sha

ROOT=Path(__file__).resolve().parents[2]


def provenance():
    root=ROOT/'third_party/baselines'
    official=json.loads((root/'baseline_provenance.json').read_text())
    for model,p in official.items():
        for name,record in p['files'].items():
            if sha(root/model/name)!=record['local_sha256']:
                raise ValueError(f'Official code provenance mismatch: {model}/{name}; STOP')
    versions={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','scikit-learn','xgboost','joblib')}
    return dict(official=official,versions=versions,adaptations='third_party/baselines/ADAPTATION_NOTES.md',
        classes=dict(Ridge='sklearn.linear_model.Ridge',RF='sklearn.ensemble.RandomForestRegressor',
            XGBoost='sklearn.multioutput.MultiOutputRegressor(xgboost.XGBRegressor)',LSTM='torch.nn.LSTM',
            TCN='cmgm.models.formal_baselines_v2.TCN: causal residual dilated TCN',Transformer='torch.nn.TransformerEncoder, activation=relu'),PASS=True)


def same_rng(fn,x):
    devices=[x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        return fn()


@torch.no_grad()
def convolution_audit(m,x):
    """Check each official temporal kernel's right-edge alignment in isolation."""
    records=[]
    for name,conv in m.core.named_modules():
        if not isinstance(conv,nn.Conv2d) or conv.kernel_size[-1]<=1:continue
        value=torch.randn(2,conv.in_channels,3,48,device=x.device,dtype=x.dtype)
        future=value.clone();future[:,:,:,32:]+=11
        a,b=conv(value),conv(future)
        valid=32-(conv.kernel_size[-1]-1)*conv.dilation[-1]
        if valid<=0:raise ValueError('No causal prefix in convolution audit')
        error=float((a[...,:valid]-b[...,:valid]).abs().max())
        records.append(dict(layer=name,kernel=list(conv.kernel_size),dilation=list(conv.dilation),prefix_max_diff=error))
    return dict(kernels=records,PASS=bool(records) and all(r['prefix_max_diff']==0 for r in records),
        scope='Official temporal convolution kernels are causal relative to right edges. MTGNN full-window LayerNorm and skip paths do not claim streaming prefix invariance; all forecast inputs end at the origin.')


@torch.no_grad()
def _sanity(name,m,x):
    m.eval();p=same_rng(lambda:m(x),x);perm=torch.arange(len(x)-1,-1,-1,device=x.device)
    batch=float((same_rng(lambda:m(x[perm]),x)-p[perm]).abs().max())
    single=float((same_rng(lambda:m(x[:1]),x)-p[:1]).abs().max())
    shape=list(p.shape);finite=bool(torch.isfinite(p).all());causal={}
    if hasattr(m,'temporal_states'):
        future=x.clone();future[:,10:]=-3*future[:,10:]+5
        a=same_rng(lambda:m.temporal_states(x),x);b=same_rng(lambda:m.temporal_states(future),x)
        error=float((a[:,:10]-b[:,:10]).abs().max());causal=dict(prefix_cutoff=10,prefix_max_diff=error,PASS=error<1e-6)
    elif name=='D0B':
        from cmgm.scripts.d0b_hybrid_graph_prior import full_sanity
        original=same_rng(lambda:full_sanity(m,x),x)
        causal=dict(original_D0B_sanity=original,PASS=original['PASS'])
    else:causal=same_rng(lambda:convolution_audit(m,x),x)
    r=dict(Finite=finite,OutputShape=shape,BatchPerm=batch,SingleSample=single,Causal=causal,
        FullInformation=True,CommodityOrdering='dataset mapping, horizon-major/commodity-minor; graph wrapper index_select',
        comparison_randomness='Common RNG realization; official MTGNN stochastic eval graph preserved',output_scale=float(p.abs().max()))
    if name in ('Graph WaveNet','MTGNN'):r['full_graph_nodes']=x.shape[2]
    r['PASS']=finite and shape==[len(x),4,24] and batch<1e-6 and single<1e-6 and causal['PASS']
    return r


def sanity(name,m,x):
    mode=m.training
    try:
        r=same_rng(lambda:_sanity(name,m,x),x)
        if not r['PASS'] and r['Finite'] and r['Causal']['PASS']:
            audit=same_rng(lambda:_sanity(name,copy.deepcopy(m).double(),x.double()),x)
            bound=32*torch.finfo(x.dtype).eps*max(1.,r['output_scale'])
            r['float64_audit']=audit;r['fp32_bound']=bound
            r['PASS']=audit['PASS'] and max(r['BatchPerm'],r['SingleSample'])<=bound
        return r
    finally:m.train(mode)
