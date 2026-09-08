"""D0B-HuberHorizonScaleDiagnostic: checkpoint-only residual and autograd probes.

No training, interventions, new variants, optimizer, or parameter updates.
Mandatory experiment metrics from this experiment onward: MAE / MSE / RMSE / Hit%.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_previous_state_diagnostics import _load_checkpoint
from cmgm.scripts.d0e_diagnostics import gradient_groups, flattened_gradients
from cmgm.training.metric_standard import population_metrics, STANDARD
from cmgm.training.train import _prediction_loss

ROOT = Path(__file__).resolve().parents[2]
VARIANT = 'switching_latent_balanced_readout'
DELTA = .02
EPS = 1e-12
THRESHOLDS = (.005, .010, .015, .020, .025, .030, .040, .050)
BINS = (0, .005, .010, .015, .020, .030, .040, .050, np.inf)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {'mean': float(x.mean()), 'std': float(x.std()),
            **{f'P{p}': float(np.percentile(x, p)) for p in (25, 50, 75, 90, 95, 99)},
            'min': float(x.min()), 'max': float(x.max())}


def robust_scale(x):
    x = np.asarray(x, dtype=np.float64)
    mad = float(np.median(np.abs(x - np.median(x))))
    return {'std': float(x.std()), 'MAD': mad, 'robust_std': 1.4826 * mad}


def residual_statistics(prediction, target, delta=DELTA):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    m = population_metrics(p, y)
    e = p - y
    a = np.abs(e)
    g = np.where(a <= delta, e, delta * np.sign(e))
    ga = np.abs(g)
    huber = np.where(a <= delta, .5 * e ** 2, delta * (a - .5 * delta))
    q, c = float(np.mean(a <= delta)), float(np.mean(a > delta))
    boundary = float(np.mean(a == delta))
    saturation = float(np.mean(ga == delta))
    # Equality belongs to Q but its derivative is already exactly delta.
    assert abs(q + c - 1) < 1e-14
    assert abs(saturation - c - boundary) < 1e-14
    centered = e - e.mean()
    es = float(e.std())
    return {'metrics': m, 'count': int(e.size), 'abs_residual': summary(a),
            'signed_residual': {'mean': float(e.mean()), 'std': es,
                                'skewness': float(np.mean(centered ** 3) / es ** 3) if es else None,
                                'positive_fraction': float(np.mean(e > 0)), 'negative_fraction': float(np.mean(e < 0))},
            'target': {'mean': float(y.mean()), 'std': float(y.std()), 'abs': summary(np.abs(y))},
            'residual_to_target': {'median_abs': float(np.median(a) / (np.median(np.abs(y)) + EPS)),
                                   'std': float(es / (y.std() + EPS))},
            'quadratic_fraction': q, 'linear_fraction': c, 'boundary_fraction': boundary,
            'coverage': {str(t): float(np.mean(a <= t)) for t in THRESHOLDS},
            'huber_loss': float(huber.mean()),
            'huber_to_mse_compression': float(huber.mean() / (m['MSE'] + EPS)),
            'gradient': {'mean_abs': float(ga.mean()), 'median_abs': float(np.median(ga)),
                         'P90_abs': float(np.percentile(ga, 90)), 'P95_abs': float(np.percentile(ga, 95)),
                         'rms': float(np.sqrt(np.mean(g ** 2))), 'saturated_fraction': saturation,
                         'clip_ratio': float(ga.mean() / (a.mean() + EPS))},
            'sanity': {'rmse_mse_error': abs(m['RMSE'] ** 2 - m['MSE']),
                       'region_sum_error': abs(q + c - 1),
                       'saturation_error_corrected_for_boundary': abs(saturation - c - boundary)},
            'histogram': {'bin_edges': [None if np.isinf(v) else v for v in BINS],
                          'last_edge': '+infinity', 'convention': '[left,right), last bin closed',
                          'counts': np.histogram(a, BINS)[0].tolist()}}


def network_gradients(model, batch, device, horizon_loss=None):
    assert not model.training
    b = model.switching_latent_transformer
    groups = gradient_groups(model)
    groups.update({'Market Encoder': list(b.market_encoder.parameters()),
                   'LongMemory Transformer': list(b.long_memory.layers.parameters()),
                   'Base RPE': [b.long_memory.base_rpe], 'state readout': list(b.state_readout.parameters()),
                   'spatial branch': [p for name in ('type_proj', 'temporal_score', 'graph_learner',
                                                     'attn_mixhop1', 'attn_mixhop2', 'gcn_norm', 'type_pool')
                                      for p in getattr(model, name).parameters()],
                   'fusion/gate': [p for name in ('gate_fc', 'lstm_proj', 'gcn_proj')
                                   for p in getattr(model, name).parameters()],
                   'prediction head': list(model.head.parameters())})
    names = {id(p): name for name, p in model.named_parameters()}
    assert set(names) == {id(p) for params in groups.values() for p in params}
    x, y = (v.to(device) for v in batch[:2])
    vectors, losses = {}, {}
    with torch.enable_grad():
        for h in config.MULTI_HORIZONS:
            i = config.MULTI_HORIZONS.index(h)
            prediction = model(x)
            loss = (torch.nn.functional.huber_loss(prediction[:, i], y[:, i], delta=DELTA)
                    if horizon_loss is None else horizon_loss(prediction, y, h))
            vectors[str(h)] = flattened_gradients(loss, groups)
            losses[str(h)] = loss.item()
    norms = {h: {k: v.norm().item() for k, v in values.items()} for h, values in vectors.items()}
    ratios = {h: {k: n / (norms['5'][k] + EPS) for k, n in values.items()} for h, values in norms.items()}
    cosines = {k: {str(h): float(torch.nn.functional.cosine_similarity(vectors['5'][k], vectors[str(h)][k], dim=0, eps=EPS))
                   for h in (1, 10, 20)} for k in groups}
    return {'norms': norms, 'ratios_to_5d': ratios, 'cosines_vs_5d': cosines, 'losses': losses,
            'parameter_names': {k: [names[id(p)] for p in params] for k, params in groups.items()},
            'method': ('eval; raw mean single-horizon Huber; autograd.grad only; no KL' if horizon_loss is None
                       else 'eval; supplied single-horizon objective; autograd.grad only; no KL'),
            'note': 'Groups overlap (combined/readout groups); do not sum module norms. Base RPE excluded from Transformer layer group.'}


def distribution_variation(values):
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()), 'std': float(values.std()), 'min': float(values.min()),
            'max': float(values.max()), 'CV': float(values.std() / (values.mean() + EPS))}


def split_statistics(p, y, commodity_names, model):
    rows = {str(h): residual_statistics(p[:, config.MULTI_HORIZONS.index(h)], y[:, config.MULTI_HORIZONS.index(h)])
            for h in config.MULTI_HORIZONS}
    huber_sum = sum(v['huber_loss'] for v in rows.values())
    mse_sum = sum(v['metrics']['MSE'] for v in rows.values())
    with torch.no_grad():
        criterion = torch.nn.HuberLoss(delta=DELTA)
        helper_loss = _prediction_loss(model, torch.from_numpy(p.astype(np.float64)),
                                       torch.from_numpy(y.astype(np.float64)), criterion).item()
    assert abs(helper_loss - huber_sum) < 1e-12
    for v in rows.values():
        v.update(huber_share=v['huber_loss'] / huber_sum, mse_share=v['metrics']['MSE'] / mse_sum)
    i = config.MULTI_HORIZONS.index(5)
    commodity = []
    for j, name in enumerate(commodity_names):
        r = residual_statistics(p[:, i, j], y[:, i, j])
        commodity.append({'index': j, 'name': str(name), **r['metrics'], 'residual_std': r['signed_residual']['std'],
                          'median_abs_residual': r['abs_residual']['P50'], 'linear_fraction': r['linear_fraction'],
                          'huber_loss': r['huber_loss']})
    sorted_c = sorted(commodity, key=lambda v: v['MSE'])
    legacy = {}
    for h in config.MULTI_HORIZONS:
        idx = config.MULTI_HORIZONS.index(h)
        e = p[:, idx].astype(np.float64) - y[:, idx].astype(np.float64)
        valid = np.abs(y[:, idx]) > 1e-8
        legacy[str(h)] = {'RMSE_mean_asset_legacy': float(np.sqrt(np.mean(e ** 2, axis=0)).mean()),
                          'Hit_masked_legacy': float(np.mean(np.sign(p[:, idx][valid]) == np.sign(y[:, idx][valid]))) if valid.any() else None}
    return {'samples': len(p), 'horizons': rows, 'sum_huber': huber_sum,
            'training_helper_loss': helper_loss, 'training_helper_error': abs(helper_loss - huber_sum),
            'commodity_5d': commodity, 'commodity_heterogeneity': {k: distribution_variation([v[k] for v in commodity])
                                                                 for k in ('MAE', 'MSE', 'linear_fraction')},
            'horizon_MSE_variation': distribution_variation([v['metrics']['MSE'] for v in rows.values()]),
            'highest_MSE_commodities': sorted_c[-5:][::-1], 'lowest_MSE_commodities': sorted_c[:5],
            'legacy_metrics_for_historical_comparison_only': legacy}


def markdown_table(headers, rows):
    def fmt(v):
        if isinstance(v, float): return f'{v:.9g}'
        return str(v).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in rows]) + '\n\n'


def write_report(r, path):
    text = ['# D0B-HuberHorizonScaleDiagnostic\n\n']
    def section(title): text.append('## ' + title + '\n\n')
    def paragraph(t): text.append(t + '\n\n')
    def table(headers, rows): text.append(markdown_table(headers, rows))
    def allrows(fn):
        return [[s, h, *fn(v)] for s, sp in r['splits'].items() for h, v in sp['horizons'].items()]
    section('1. Checkpoint / data sanity')
    paragraph('NO TRAINING / NO MODEL MODIFICATION / NO PARAMETER UPDATE / NO NEW MODEL VARIANT.')
    table(['Field', 'Value'], [[k, json.dumps(v, ensure_ascii=False)] for k, v in r['provenance'].items()])
    table(['Integrity', 'Value'], list(r['integrity'].items()))
    paragraph('Checkpoint 缺失的 seed / Git SHA 不作推断；运行 seed / Git SHA 单独记录。TRAIN 包含尾 batch。重叠窗口是描述性总体，不能当独立重复实验。')
    section('2. Mandatory metrics: MAE/MSE/RMSE/Hit')
    paragraph('Mandatory experiment metrics from this experiment onward: **MAE / MSE / RMSE / Hit%**. ' + STANDARD)
    table(['Split', 'Horizon', 'MAE', 'MSE', 'RMSE', 'Hit%'], allrows(lambda v: [v['metrics'][k] for k in ('MAE','MSE','RMSE')] + [100*v['metrics']['Hit']]))
    paragraph('旧报告 RMSE 是商品各自 RMSE 的均值，旧 Hit 排除 |target|≤1e-8；当前 pooled RMSE 与不掩码 Hit 按本轮要求计算。数值口径变化不代表预测变化。旧口径核对：')
    table(['Split','Horizon','Legacy RMSE','Legacy Hit%'], [[s,h,v['RMSE_mean_asset_legacy'],100*v['Hit_masked_legacy'] if v['Hit_masked_legacy'] is not None else None] for s,sp in r['splits'].items() for h,v in sp['legacy_metrics_for_historical_comparison_only'].items()])
    section('3. Residual scale by horizon')
    table(['Split','Horizon','mean |e|','std |e|','P25','median/P50','P75','P90','P95','P99','max'], allrows(lambda v: [v['abs_residual'][k] for k in ('mean','std','P25','P50','P75','P90','P95','P99','max')]))
    table(['Split','Horizon','signed mean','signed std','skewness','positive fraction','negative fraction'], allrows(lambda v: list(v['signed_residual'].values())))
    table(['Split','Horizon','target mean','target std','median |y|','mean |y|','P75 |y|','P90 |y|','P95 |y|','R median','R std'], allrows(lambda v: [v['target']['mean'],v['target']['std']] + [v['target']['abs'][k] for k in ('P50','mean','P75','P90','P95')] + list(v['residual_to_target'].values())))
    table(['Horizon','TRAIN scale source','std','MAD','1.4826 MAD','delta/std','delta/robust std'], [[h,k,v['std'],v['MAD'],v['robust_std'],DELTA/(v['std']+EPS),DELTA/(v['robust_std']+EPS)] for h,ss in r['train_scales'].items() for k,v in ss.items()])
    paragraph('以下标准化仅使用 TRAIN scale；VAL/TEST 未用于估计 scale。没有生成新 delta 配置。')
    table(['Split','Horizon','TRAIN scale denominator','normalized median |e|','P90','P95'], [[s,h,k,*v.values()] for s,sp in r['splits'].items() for h,row in sp['horizons'].items() for k,v in row['normalized_residual'].items()])
    section('4. Huber quadratic-vs-linear region')
    table(['Split','Horizon','quadratic %','linear/clipped %'], allrows(lambda v: [100*v['quadratic_fraction'],100*v['linear_fraction']]))
    paragraph('CDF thresholds 是描述性覆盖率横坐标，不是 candidate delta。')
    table(['Split','Horizon',*[str(v) for v in THRESHOLDS]], allrows(lambda v: list(v['coverage'].values())))
    section('5. Huber loss contribution')
    paragraph('**equal horizon coefficients ≠ equal horizon loss contribution ≠ equal optimization influence**。当前 objective 保持 Σ L_h；无 KL 混入此处。完整总体 mean 与 training helper 在同一完整张量上核对；不同于旧 epoch 日志对 batch means 的等权平均。')
    table(['Split','Horizon','Huber loss','Huber share','MSE','MSE share','MAE','RMSE','Q','C'], allrows(lambda v: [v['huber_loss'],v['huber_share'],v['metrics']['MSE'],v['mse_share'],v['metrics']['MAE'],v['metrics']['RMSE'],v['quadratic_fraction'],v['linear_fraction']]))
    table(['Split','sum Huber','training helper','abs difference'], [[s,sp['sum_huber'],sp['training_helper_loss'],sp['training_helper_error']] for s,sp in r['splits'].items()])
    section('6. MSE contribution')
    paragraph('MSE share 只是对照，不是训练 objective。Huber 小误差项为 0.5 e²，因此 compression 的绝对数值不是 loss weight。')
    table(['Split','Horizon','MSE share','Huber share','Huber/MSE compression'], allrows(lambda v: [v['mse_share'],v['huber_share'],v['huber_to_mse_compression']]))
    section('7. Huber gradient magnitude')
    paragraph('解析 g 是未 reduction 的单个 residual 导数；mean loss 对 prediction 的导数另除以样本×商品数。ClipRatio 的分母 |e| 对应 0.5 e² 的导数幅值，普通 MSE 导数为 2e。精确 |e|=delta 属于 quadratic 区，但 |g| 也等于 delta；saturation = linear + boundary mass。')
    table(['Split','Horizon','mean |g|','median |g|','P90 |g|','P95 |g|','RMS g','saturated fraction','ClipRatio','boundary mass'], allrows(lambda v: list(v['gradient'].values())+[v['boundary_fraction']]))
    section('8. Network gradient magnitude')
    g = r['network_gradients']
    paragraph(g['method'] + '. ' + g['note'] + ' 固定 TEST batch；范数依赖 Jacobian 和模块参数规模，不能仅由 residual g 推断。')
    table(['Module','1d norm','5d norm','10d norm','20d norm','1/5','10/5','20/5','cos5,1','cos5,10','cos5,20'], [[k,*[g['norms'][str(h)][k] for h in config.MULTI_HORIZONS],*[g['ratios_to_5d'][str(h)][k] for h in (1,10,20)],*g['cosines_vs_5d'][k].values()] for k in g['norms']['5']])
    section('9. Commodity heterogeneity')
    for s,sp in r['splits'].items():
        paragraph(s+' 5d：')
        table(['Commodity','MAE','MSE','RMSE','Hit%','residual std','median |e|','clipped fraction','Huber'], [[v['name'],v['MAE'],v['MSE'],v['RMSE'],100*v['Hit'],v['residual_std'],v['median_abs_residual'],v['linear_fraction'],v['huber_loss']] for v in sp['commodity_5d']])
        table(['Quantity','mean','std','min','max','CV'], [[k,*v.values()] for k,v in {**sp['commodity_heterogeneity'],'across-horizon pooled MSE':sp['horizon_MSE_variation']}.items()])
        table(['Rank group','Commodity','MSE'], [[label,v['name'],v['MSE']] for label,key in [('highest','highest_MSE_commodities'),('lowest','lowest_MSE_commodities')] for v in sp[key]])
    section('10. Interpretation')
    for t in r.get('interpretation', ['Measured results are complete; interpretation pending review.']): paragraph(t)
    section('11. Decision')
    paragraph(r.get('decision', 'Pending measured-statistic review; no subsequent experiment authorized.'))
    paragraph('STOP. 未修改 loss / delta / switch KL / model / optimizer，未生成候选训练配置。')
    path.write_text(''.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/switching_latent_balanced_readout_best.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/d0b_huber_horizon_scale'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--device', default='cpu', choices=('cpu','cuda'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    assert config.MULTI_HORIZONS == [1,5,10,20] and config.SEQ_LEN == 20
    assert config.HUBER_DELTA == DELTA and config.LOSS_TYPE == 'huber'
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    start = time.time()
    digest = file_hash(args.checkpoint)
    from cmgm.scripts.main_ablation import build_data
    data = build_data(SimpleNamespace(batch_size=64, seq_len=20, seed=42))
    loaders = {s.upper(): DataLoader(loader.dataset, batch_size=64, shuffle=False, drop_last=False)
               for s,loader in data['loaders'].items()}
    assert {s:len(v.dataset) for s,v in loaders.items()} == {'TRAIN':1396,'VAL':268,'TEST':269}
    assert data['n_nodes'] == 284 and data['n_commodities'] == 24 and config.FEATURE_DIM == 21
    market = data['market_indices']
    model = HeteroMixHopCMGM(284,24,n_stock=market['stock'][1],n_bond=market['bond'][1]-market['bond'][0],
                             feat_dim=21,variant=VARIANT).to(device).eval()
    payload = _load_checkpoint(model,args.checkpoint,device)
    metadata = payload.get('metadata') or {}
    assert metadata.get('variant', VARIANT) == VARIANT
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    branch = model.switching_latent_transformer
    assert branch.balanced_readout and not branch.regime_filter.learnable_sticky_alpha
    assert branch.regime_filter.sticky_alpha == .5
    fixed = next(iter(loaders['TEST']))
    names = data['feature_names'][market['commodity'][0]:market['commodity'][1]]
    report = {'provenance': {'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest,
              'loaded_variant':VARIANT,'checkpoint_metadata':metadata,'best_epoch':payload.get('best_epoch'),
              'checkpoint_seed':metadata.get('seed',payload.get('seed','unknown')),
              'checkpoint_git_sha':metadata.get('git_sha',metadata.get('git SHA',payload.get('git_sha','unknown'))),
              'runtime_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'runtime_seed':42,'parameter_count':sum(p.numel() for p in model.parameters()),
              'horizons':config.MULTI_HORIZONS,'delta':DELTA,'fixed_TEST_batch_shape':list(fixed[0].shape),
              'fixed_TEST_target_shape':list(fixed[1].shape),'torch':torch.__version__,'device':str(device),
              'threads':args.threads,'metric_standard':STANDARD,
              'source_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__),ROOT/'cmgm/config.py',ROOT/'cmgm/training/train.py',ROOT/'cmgm/training/metric_standard.py',*sorted((ROOT/'cmgm/models').glob('*.py'))]}},
              'splits':{},'train_scales':{}}
    for s,loader in loaders.items():
        ps,ys = [],[]
        with torch.no_grad():
            for x,y in loader:
                assert list(x.shape[1:]) == [20,284,21]
                p = model(x.to(device)).cpu()
                assert p.shape == y.shape and list(p.shape[1:]) == [4,24]
                ps.append(p.numpy());ys.append(y.numpy())
        p,y = np.concatenate(ps),np.concatenate(ys)
        # Full arrays preserve original float32; statistics promote before subtraction.
        np.savez_compressed(args.output/f'{s}_residuals.npz',prediction=p,target=y,
                            residual=p.astype(np.float64)-y.astype(np.float64),horizons=config.MULTI_HORIZONS)
        report['splits'][s] = split_statistics(p,y,names,model)
        for h in config.MULTI_HORIZONS:
            i = config.MULTI_HORIZONS.index(h)
            e = p[:,i].astype(np.float64)-y[:,i].astype(np.float64)
            if s == 'TRAIN': report['train_scales'][str(h)] = {'target':robust_scale(y[:,i]),'residual':robust_scale(e)}
            norms = {}
            for source,scales in report['train_scales'][str(h)].items():
                for kind in ('std','robust_std'):
                    norms[f'{source}_{kind}'] = {f'P{q}':float(np.percentile(np.abs(e)/(scales[kind]+EPS),q)) for q in (50,90,95)}
            report['splits'][s]['horizons'][str(h)]['normalized_residual'] = norms
        print(f'[D0B Huber {s}] n={len(p)} '+json.dumps({h:v['metrics'] for h,v in report['splits'][s]['horizons'].items()}),flush=True)
    report['network_gradients'] = network_gradients(model,fixed,device)
    differences = {k:float((v.detach().cpu()-initial[k]).abs().max()) for k,v in model.state_dict().items()}
    report['integrity'] = {'max_state_dict_diff':max(differences.values()),'checkpoint_unchanged':file_hash(args.checkpoint)==digest,
                           'all_parameter_grads_none':all(p.grad is None for p in model.parameters()),
                           'all_modules_eval':all(not m.training for m in model.modules()),
                           'max_RMSE_squared_minus_MSE':max(v['sanity']['rmse_mse_error'] for sp in report['splits'].values() for v in sp['horizons'].values()),
                           'elapsed_seconds':time.time()-start}
    assert report['integrity']['max_state_dict_diff'] == 0
    assert all(report['integrity'][k] for k in ('checkpoint_unchanged','all_parameter_grads_none','all_modules_eval'))
    (args.output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    commodity_rows = [{'split':s,**v} for s,sp in report['splits'].items() for v in sp['commodity_5d']]
    with (args.output/'commodity_5d_error.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(commodity_rows[0]));writer.writeheader();writer.writerows(commodity_rows)
    residual_rows = [{'split':s,'horizon':h,**v['metrics'],**v['abs_residual'],'Q':v['quadratic_fraction'],'C':v['linear_fraction'],
                      'Huber':v['huber_loss'],'Huber_share':v['huber_share'],'MSE_share':v['mse_share']} for s,sp in report['splits'].items() for h,v in sp['horizons'].items()]
    with (args.output/'residual_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(residual_rows[0]));writer.writeheader();writer.writerows(residual_rows)
    write_report(report,args.output/'REPORT.md')
    print('[D0B Huber integrity] '+json.dumps(report['integrity']),flush=True)
    print('[D0B Huber report] '+str(args.output/'REPORT.md'),flush=True)


if __name__ == '__main__':
    main()
