"""TRAIN-only Ridge selection, followed by frozen B TRAIN/VAL evaluation.

No optimizer, backward, TEST loader, or D0B training exists in this entry.
Default is synthetic preflight; --run explicitly enables the small probe fits.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from cmgm.scripts.d0b_short_long_expert_diagnostics import (
    ROOT, EXPERIMENT_ROOT, SOURCE_FILES, HORIZONS, GROUPS, GROUP_NAMES,
    TIE_ATOL, TIE_RTOL, METRIC_ATOL, REPRODUCTION_ATOL, REPRODUCTION_RTOL,
    capture_normal, predict_frozen_inputs, verify_group_injection,
    expert_losses, best_candidates, choose_predictions, metrics, state_digest,
    discover_index, load_run, load_validated_data, compare_saved,
    model_kwargs, sha256, write_json, write_csv, route_statistics, check_protocol,
)

ALPHAS = (.001, .1, 1., 10., 100.)
STD_EPS = 1e-12
SCORE_TIE_ATOL = 1e-12


def advantages(losses):
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 3 or losses.shape[1:] != (2, 3):
        raise ValueError('Expected (origins,2,3) expert losses')
    return losses[..., 1:] - losses[..., :1]


def hard_choices(predicted):
    """Lowest candidate index within an absolute 1e-12 score tie."""
    predicted = np.asarray(predicted, dtype=np.float64)
    scores = np.concatenate((np.zeros((*predicted.shape[:-1], 1)), predicted), axis=-1)
    return (scores <= scores.min(-1, keepdims=True) + SCORE_TIE_ATOL).argmax(-1)


def choice_weights(choices):
    return np.eye(3, dtype=np.float64)[np.asarray(choices, dtype=int)]


def soft_weights(predicted, tau):
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.ndim != 3 or predicted.shape[1:] != (2, 2):
        raise ValueError('Expected (origins,2,2) predicted advantages')
    scores = np.concatenate((np.zeros((len(predicted), 2, 1)), predicted), axis=-1)
    result = np.full_like(scores, 1 / 3)
    for g in range(2):
        if tau[g] > STD_EPS:
            logits = -scores[:, g] / tau[g]
            logits -= logits.max(-1, keepdims=True)
            exp = np.exp(logits)
            result[:, g] = exp / exp.sum(-1, keepdims=True)
    return result


def temporal_folds(origin_rows, dates, horizon):
    """Purge labels ending at/after each validation prediction origin row.

    These are rows in the already audited trading calendar, not natural days.
    Dataset label row is origin_row+horizon. Equality is also purged.
    """
    rows = np.asarray(origin_rows, dtype=int)
    dates = np.asarray(dates, dtype='datetime64[ns]')
    n = len(rows)
    if (n < 30 or len(dates) != n or not np.all(np.diff(rows) == 1)
            or not np.all(np.diff(dates) > np.timedelta64(0, 'ns'))):
        raise ValueError('Need chronological, contiguous audited trading-row origins')
    boundaries = (int(n * .50), int(n * .66), int(n * .83), n)
    result = []
    for number, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        available = np.arange(start)
        retained = available[rows[available] + horizon < rows[start]]
        purged = available[rows[available] + horizon >= rows[start]]
        valid = np.arange(start, stop)
        if not len(retained) or not len(valid):
            raise ValueError('Empty temporal fold after label-overlap purge')
        end_row = int(rows[retained[-1]] + horizon)
        end_index = int(end_row - rows[0])
        result.append(dict(fold=number, max_horizon=horizon,
            train_indices=retained.tolist(), purged_indices=purged.tolist(),
            validation_indices=valid.tolist(), train_count=len(retained),
            validation_count=len(valid), validation_first_origin_row=int(rows[start]),
            validation_first_origin_date=str(dates[start]),
            validation_last_origin_date=str(dates[stop - 1]),
            last_retained_train_label_end_row=end_row,
            last_retained_train_label_end_date=str(dates[end_index]),
            purge_rule='origin_row + group_max_horizon < first_validation_origin_row'))
    return result


@dataclass
class LinearProbe:
    alpha: float
    mask: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    fitted_origins: int

    @classmethod
    def fit(cls, x, y, alpha):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid probe TRAIN inputs or two-output targets')
        xs, ys = StandardScaler().fit(x), StandardScaler().fit(y)
        mask = np.sqrt(xs.var_) > STD_EPS
        xz, yz = xs.transform(x)[:, mask], ys.transform(y)
        if mask.any():
            estimator = Ridge(alpha=alpha, fit_intercept=True, solver='svd').fit(xz, yz)
            coef, intercept = estimator.coef_, estimator.intercept_
        else:
            coef, intercept = np.zeros((2, 0)), yz.mean(0)
        return cls(alpha, mask, xs.mean_, xs.scale_, ys.mean_, ys.scale_, coef, intercept, len(x))

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mask):
            raise ValueError('Probe input dimension mismatch')
        standardized = ((x - self.x_mean) / self.x_scale)[:, self.mask]
        return (standardized @ self.coef.T + self.intercept) * self.y_scale + self.y_mean

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(float(z['alpha']), z['mask'], z['x_mean'], z['x_scale'],
                       z['y_mean'], z['y_scale'], z['coef'], z['intercept'], int(z['fitted_origins']))


def fit_train_probes(x, d, rows, dates):
    """No VAL argument: fit transforms/CV/full refits on TRAIN alone."""
    probes, records = [], {}
    for g, group in enumerate(GROUP_NAMES):
        folds = temporal_folds(rows, dates, max(HORIZONS[i] for i in GROUPS[g]))
        candidates = []
        for alpha in ALPHAS:
            total, count, details = 0., 0, []
            for fold in folds:
                train, valid = np.asarray(fold['train_indices']), np.asarray(fold['validation_indices'])
                probe = LinearProbe.fit(x[train], d[train, g], alpha)
                residual = probe.predict(x[valid]) - d[valid, g]
                mse = float(np.mean(residual ** 2))
                total += mse * len(valid); count += len(valid)
                details.append(dict(fold=fold['fold'], original_scale_MSE=mse,
                    validation_origins=len(valid), scaler_fit_origins=probe.fitted_origins,
                    x_mean=probe.x_mean.tolist(), x_scale=probe.x_scale.tolist(),
                    retained_dimensions=np.flatnonzero(probe.mask).tolist(),
                    y_mean=probe.y_mean.tolist(), y_scale=probe.y_scale.tolist()))
            candidates.append(dict(alpha=alpha, origin_weighted_original_scale_MSE=total / count, folds=details))
        selected = min(candidates, key=lambda r: r['origin_weighted_original_scale_MSE'])['alpha']
        probes.append(LinearProbe.fit(x, d[:, g], selected))
        records[group] = dict(selected_alpha=selected, candidates=candidates, folds=folds)
    return probes, records


@torch.inference_mode()
def capture_router_input(model, loader, device):
    """Hook the actual checkpoint's MLP inputs, after its learned LayerNorms."""
    f = model.short_long_dynamic_moe
    if not f.residual or not f.dynamic:
        raise ValueError('This probe requires frozen B-dynamic')
    short, long = [], []
    def collect(destination):
        def hook(module, args):
            destination.append(args[0].cpu().numpy().copy())
        return hook
    hs = f.router.short.register_forward_pre_hook(collect(short))
    hl = f.router.long.register_forward_pre_hook(collect(long))
    try:
        score, arrays, batches = capture_normal(model, loader, device)
    finally:
        hs.remove(); hl.remove()
    x, xl = np.concatenate(short), np.concatenate(long)
    if x.shape != (len(arrays['target']), 128) or not np.array_equal(x, xl) or not np.isfinite(x).all():
        raise ValueError('Actual short/long router inputs do not align')
    arrays['router_input'] = x
    return score, arrays, batches


def prediction_fit_rows(split, d, predicted, constant):
    rows = []
    for g, group in enumerate(GROUP_NAMES):
        for k, candidate in enumerate(('state', 'interaction')):
            actual = d[:, g, k]
            base_error = constant[:, g, k] - actual
            baseline_mse = float(np.mean(base_error ** 2))
            for name, p in (('constant_TRAIN_mean', constant), ('ridge', predicted)):
                e = p[:, g, k] - actual
                mse, var = float(np.mean(e ** 2)), float(np.var(actual))
                nonzero = np.abs(actual) > SCORE_TIE_ATOL
                true_better = actual < -SCORE_TIE_ATOL
                guessed = p[:, g, k] < -SCORE_TIE_ATOL
                rows.append(dict(split=split, group=group, candidate=candidate, predictor=name,
                    MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt(mse)), MSE=mse,
                    R2=1 - mse / var if np.std(actual) > STD_EPS else None,
                    target_std=float(np.std(actual)),
                    delta_MSE_vs_constant=mse - baseline_mse,
                    relative_MSE_improvement_percent=100 * (baseline_mse - mse) / baseline_mse if baseline_mse > STD_EPS ** 2 else None,
                    sign_accuracy_nonzero=float((true_better[nonzero] == guessed[nonzero]).mean()) if nonzero.any() else None,
                    sign_nonzero_origins=int(nonzero.sum()),
                    true_better_fraction=float(true_better.mean()),
                    predicted_better_fraction=float(guessed.mean()),
                    recall_better=float(guessed[true_better].mean()) if true_better.any() else None,
                    recall_not_better=float((~guessed[~true_better]).mean()) if (~true_better).any() else None))
    return rows


def routing_rows(split, name, weights, losses):
    weights = np.asarray(weights, dtype=np.float64)
    n = len(losses)
    if weights.shape == (2, 3):
        weights = np.broadcast_to(weights, (n, 2, 3))
    if weights.shape != (n, 2, 3):
        raise ValueError('Misaligned strategy routes')
    summary = route_statistics(weights)
    oracle, tied, _ = best_candidates(losses)
    hard = weights.argmax(-1)
    rows = []
    for g, group in enumerate(GROUP_NAMES):
        non_tie = tied[:, g].sum(-1) == 1
        selected = losses[np.arange(n), g, hard[:, g]]
        regret = selected - losses[:, g].min(-1)
        for k, candidate in enumerate(('shared', 'state', 'interaction')):
            rows.append(dict(split=split, strategy=name, group=group, candidate=candidate,
                mean=summary['mean'][g][k], std=summary['std'][g][k],
                P10=summary['quantiles']['0.1'][g][k], P50=summary['quantiles']['0.5'][g][k],
                P90=summary['quantiles']['0.9'][g][k],
                mean_sample_entropy_nats=summary['mean_sample_entropy_nats'][g],
                entropy_of_mean_weights_nats=summary['entropy_of_mean_weights_nats'][g],
                argmax_selection_fraction=float((hard[:, g] == k).mean()),
                oracle_agreement_non_ties=float((hard[non_tie, g] == oracle[non_tie, g]).mean()) if non_tie.any() else None,
                non_tie_origins=int(non_tie.sum()), mean_hard_choice_regret=float(regret.mean()),
                semantics='Agreement/regret describe argmax hard choice, not soft mixture loss'))
    return rows


def discover_diagnostics(explicit):
    paths = [explicit] if explicit else sorted(EXPERIMENT_ROOT.glob('expert_diagnostics_*/diagnostic_file_index.json'))
    candidates = []
    for path in paths:
        path = path / 'diagnostic_file_index.json' if path.is_dir() else path
        if not path.exists():
            continue
        indexed = json.loads(path.read_text())
        if not {'audit.json', 'provenance.json', 'b-dynamic_train_predictions.npz',
                'b-dynamic_val_predictions.npz', 'validated_train_val_cache.npz'} <= indexed.keys():
            continue
        audit = json.loads(Path(indexed['audit.json']).read_text())
        if audit['status'] == 'COMPLETE' and audit.get('original_artifacts_unchanged') and audit.get('model_parameters_unchanged'):
            candidates.append((path, indexed, audit))
    if len(candidates) != 1:
        raise ValueError(f'Need one completed audited diagnostic index; found {len(candidates)}; pass --diagnostics')
    return candidates[0]


def build_report(summary, fit_rows, routing):
    val, reference = summary['scores']['val'], summary['references']
    primary = val['ridge_hard_MAIN']['horizons']['5']['MAE']
    b = val['original_B_dynamic']['horizons']['5']['MAE']
    d0b, three = (reference[a]['val']['horizons']['5']['MAE'] for a in ('d0b', 'three-global'))
    rrows = [r for r in fit_rows if r['split'] == 'val' and r['predictor'] == 'ridge']
    better = [r['group'] + '/' + r['candidate'] for r in rrows if r['delta_MSE_vs_constant'] < 0]
    no_predictive_gain = len(better) == 0
    lines = ['# 冻结 B 专家的相对优势预测探针（TRAIN/VAL）', '',
        ('结论：本次线性探针没有提取到优于常量预测器的可泛化优势信号；现有证据不足以支持扩大 router 或替换基线。不能由此断言所有非线性 router 都无效。' if no_predictive_gain else
         '结论：部分优势目标显示线性可预测信号；是否可利用仍需结合预先指定硬选择的实际 MAE 以及软路由的均值/打乱干预判断，不能只凭预测损失差或赢家识别率判断。'), '',
        f"现有输入的可利用优势信号：线性探针在 VAL 上有 {len(better)}/4 个损失差目标的 MSE 低于 TRAIN 均值常量预测器（{', '.join(better) or '无'}）。这只评价本次线性提取能力，不能排除其他非线性信号。",
        f'预先指定主策略 Ridge 硬选择的 VAL MAE@5d={primary:.9f}，相对原 B Δ={primary-b:+.9f}；相对 D0B Δ={primary-d0b:+.9f}；相对三专家全局静态 Δ={primary-three:+.9f}。',
        f"主策略{'降低' if primary < b else '没有降低'}原 B 的 5d MAE，{'超过' if primary < d0b else '没有超过'}D0B；不根据 VAL 选择另一个辅助策略取代主策略。", '',
        f'相对原 B 的 5d MAE 改善率={(b-primary)/b*100:+.4f}%；相对 D0B={(d0b-primary)/d0b*100:+.4f}%；相对三专家全局静态={(three-primary)/three*100:+.4f}%。正值表示误差降低。',
        '主策略各周期相对原 B 的 MAE 变化：' + '，'.join(f"{h}d {val['ridge_hard_MAIN']['horizons'][str(h)]['MAE']-val['original_B_dynamic']['horizons'][str(h)]['MAE']:+.9f}" for h in HORIZONS) + '。', '',
        '## 预先固定的拟合规则', '',
        f"实际 router 输入为 checkpoint 中 LN_s/LN_t 后送入两组 MLP 的同一个 128 维向量。short/long 的 alpha 分别为 {summary['selected_alpha']}，候选固定为 {list(ALPHAS)}，只按 TRAIN 时间折原尺度 MSE 选择。",
        f"TRAIN 损失差 RMS 温度 short/long={summary['tau']}，不是通过 VAL 调参。TRAIN 最佳固定专家={summary['train_fixed_choice']}（0共享/1状态/2交互）。",
        '输入与目标 StandardScaler、近零维度处理只拟合每折保留的 TRAIN；最终重新拟合完整 TRAIN 并保存参数后才评价 VAL。交易行标签结束位置按 short 5、long 20 清除边界重叠；不使用自然日减法。', '',
        '## VAL 策略比较', '',
        '| 策略 | short Huber | long Huber | MAE@1d | MAE@5d | MAE@10d | MAE@20d | RMSE@5d | Hit@5d % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, score in {**val, **{a: reference[a]['val'] for a in reference}}.items():
        h = score['horizons']
        lines.append(f"| {name} | {score['group_huber'][0]:.9g} | {score['group_huber'][1]:.9g} | "
            + ' | '.join(f"{h[str(i)]['MAE']:.9f}" for i in HORIZONS)
            + f" | {h['5']['RMSE']:.9f} | {100*h['5']['Hit']:.4f} |")
    lines += ['', '## 优势预测能力（VAL，原损失差尺度）', '',
        '| 组/目标 | Ridge RMSE | Constant RMSE | Ridge R² | MSE相对改善 % | 符号识别率 % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rrows:
        c = next(c for c in fit_rows if c['split']=='val' and c['predictor']=='constant_TRAIN_mean'
                 and c['group']==r['group'] and c['candidate']==r['candidate'])
        fmt = lambda v: '不可定义' if v is None else f'{v:.5g}'
        lines.append(f"| {r['group']}/{r['candidate']} | {r['RMSE']:.9g} | {c['RMSE']:.9g} | {fmt(r['R2'])} | "
            f"{fmt(r['relative_MSE_improvement_percent'])} | {fmt(100*r['sign_accuracy_nonzero'] if r['sign_accuracy_nonzero'] is not None else None)} |")
    soft = val['ridge_soft_AUX']['horizons']['5']['MAE']
    fixed = val['ridge_soft_TRAIN_mean']['horizons']['5']['MAE']
    shuffled = val['ridge_soft_paired_shuffle']['horizons']['5']['MAE']
    lines += ['', '## 样本条件化是否转化为收益', '',
        f'辅助 Ridge 软路由 MAE@5d={soft:.9f}；固定其 TRAIN 平均权重={fixed:.9f}，Δ(固定−动态)={fixed-soft:+.9f}；整对 short/long 权重打乱={shuffled:.9f}，Δ(打乱−动态)={shuffled-soft:+.9f}。',
        '这些值必须结合主策略与损失差预测解读：只有固定平均权重改善，或打乱不损害结果，均不支持逐样本路由有效。辅助软路由不是新的主模型。', '',
        '## 硬选择与理想选择的一致性（VAL）', '',
        '| 策略 | 组 | 非并列一致率 % | 硬选择平均 Huber regret |', '|---|---|---:|---:|']
    for r in routing:
        if r['split']=='val' and r['candidate']=='shared' and r['strategy'] in ('original_B_dynamic','constant_advantage_hard','ridge_hard_MAIN','ridge_soft_AUX'):
            agreement = r['oracle_agreement_non_ties']
            lines.append(f"| {r['strategy']} | {r['group']} | {100*agreement:.4f} | {r['mean_hard_choice_regret']:.9g} |" if agreement is not None else
                         f"| {r['strategy']} | {r['group']} | 不可定义 | {r['mean_hard_choice_regret']:.9g} |")
    lines += ['', '## 输入变化与数据审计', '',
        f"TRAIN/VAL origins={summary['origin_counts']}；两组 router 实际输入完全相同。TRAIN/VAL 近零方差维度数={summary['near_zero_dimensions']}；逐维标准差见 router_input_statistics.json。",
        'audit.json 记录 checkpoint、所有原始结果、模型 state_dict 的前后哈希；全部冻结，无 backward、optimizer 或 TEST 读取。正常预测先复现上一轮产物，再拟合探针。', '',
        '## 证据边界', '',
        '- 冻结 B 原先见过完整 TRAIN；内部时间验证只选择探针 alpha，不是整个预测系统的无泄漏交叉验证。VAL 曾用于方案探索，本次仍是探索性评价。',
        '- 理想单专家选择使用相应区间标签，不能部署，不是所有软融合方法的理论上界；按组 Huber 选择不保证 5d MAE 最低。',
        '- B 的共享候选是共同训练后的 z_base，不等于独立 D0B。软路由经过真实表示融合和原非线性 head，不平均最终预测。',
        '- 组内 Huber 是样本等权的两个周期×商品均值，区别于历史选模的四周期求和与 batch-mean；收益率小数尺度，Hit JSON/CSV 为比例，展示为百分数。',
        '- 损失差/硬选择并列用固定 1e-12 绝对容差；真实专家损失的最优并列还用 1e-9 相对容差。R² 可为负，近零方差时 null。',
        '- 不作统计显著性或经济因果声明，不扩大 router、不训练原模型、不访问 TEST。', '',
        '完整 TRAIN/VAL 各周期、各策略指标见 strategy_comparison.csv；probe_fit_metrics.csv 保存两种目标的预测统计；routing_diagnostics.csv 保存路由分布、硬一致率和 regret。per_origin_predictions.npz 可复算全部统计；probe_short.npz/probe_long.npz 保存 scaler、掩码和 Ridge 参数。']
    return '\n'.join(lines) + '\n'


def run(args):
    started = time.perf_counter()
    check_protocol()
    diag_path, di, previous_audit = discover_diagnostics(args.diagnostics)
    index_path, index = discover_index(args.file_index or Path(previous_audit['file_index']))
    out = args.output or EXPERIMENT_ROOT / f'advantage_probe_seed42_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    audit = dict(status='RUNNING', test_evaluated=False, original_model_training=False, probe_fitting=True)
    write_json(out / 'audit.json', audit)
    print(f'PROBE_DIRECTORY {out.resolve()}', flush=True)
    protected = {Path(p): sha256(Path(p)) for p in di.values()}
    protected.update({Path(index_path): sha256(index_path), Path(diag_path): sha256(diag_path)})
    for item in index.values():
        for key in ('result', 'config', 'training_history', 'checkpoint', 'TRAIN_predictions', 'VAL_predictions'):
            protected[Path(item[key])] = sha256(Path(item[key]))
    try:
        torch.set_num_threads(args.threads)
        model, cfg, result, model_provenance = load_run('b-dynamic', index['b-dynamic'], args.device)
        if result['best_epoch'] != 81 or model_provenance['checkpoint_sha256'] != previous_audit['models']['b-dynamic']['checkpoint_sha256']:
            raise ValueError('Not the audited B seed42 epoch81 checkpoint')
        env = cfg['environment']
        torch.backends.cudnn.deterministic = env['cudnn_deterministic']
        torch.backends.cudnn.benchmark = env['cudnn_benchmark']
        reference = Path(cfg['reference_config'])
        if sha256(reference) != cfg['reference_config_sha256']:
            raise ValueError('Original reference config hash mismatch')
        data, fingerprint = load_validated_data(Path(di['validated_train_val_cache.npz']), cfg['data_audit'], reference)
        if json.loads(json.dumps(model_kwargs(data))) != cfg['model_kwargs']:
            raise ValueError('Actual dataset/model config differs')
        match = re.fullmatch(r'sum_h Huber\(delta=([0-9.]+)\)', cfg['protocol']['prediction_loss'])
        if match is None:
            raise ValueError('Original Huber delta absent')
        delta = float(match.group(1))
        config = dict(variant=cfg['variant'], seed=42, best_epoch=result['best_epoch'],
            probe='two independent two-output Ridge regressors', alphas=list(ALPHAS),
            alpha_selection='TRAIN purged expanding folds, origin-weighted original-scale mean MSE',
            fold_boundaries=[.50,.66,.83,1.], zero_variance_std_epsilon=STD_EPS,
            scaler='StandardScaler for inputs and targets; fold TRAIN only; final full TRAIN',
            ridge_solver='svd', temperature='full TRAIN two-target pooled RMS; no tuning',
            main_strategy='ridge_hard_MAIN', auxiliary_strategy='ridge_soft_AUX',
            permutation_seed=args.permutation_seed, feature='actual router u=[checkpoint LN_s(s),checkpoint LN_t(t)]',
            horizons=list(HORIZONS), groups=[[1,5],[10,20]], huber_delta=delta,
            hard_tie_atol=SCORE_TIE_ATOL, expert_loss_tie_atol=TIE_ATOL, expert_loss_tie_rtol=TIE_RTOL,
            target_scale='decimal cumulative return', original_training_protocol=cfg['protocol'],
            test_evaluated=False, original_model_training=False)
        write_json(out / 'probe_config.json', config)
        before = state_digest(model)
        collected, batches, features, candidates, losses = {}, {}, {}, {}, {}
        input_stats = {}
        for split in ('train', 'val'):
            score, arrays, frozen = capture_router_input(model, data['loaders'][split], args.device)
            arrays.update(origin_dates=data['origin_dates'][split], commodity_order=np.asarray(fingerprint['commodity_order'], dtype=str))
            with np.load(di[f'b-dynamic_{split}_predictions.npz'], allow_pickle=False) as z:
                saved = {k: z[k] for k in ('prediction','target','pi','origin_dates','commodity_order','expert_predictions','expert_losses')}
            baseline = metrics(saved['prediction'], saved['target'], delta)
            difference = compare_saved(arrays, Path(di[f'b-dynamic_{split}_predictions.npz']), score, baseline['horizons'])
            losses[split] = expert_losses(saved['expert_predictions'], arrays['target'], delta)
            if not np.array_equal(losses[split], saved['expert_losses']):
                raise ValueError('Saved expert losses do not reproduce in float64')
            audit.setdefault('normal_reproduction', {})[split] = dict(PASS=True, max_abs_prediction_difference=difference)
            audit.setdefault('one_hot_injection', {})[split] = verify_group_injection(model, data['loaders'][split], frozen, arrays, args.device)
            # Reuse saved candidates only after full frozen fusion/head reproduction.
            candidate_differences = []
            for k in range(3):
                current = predict_frozen_inputs(model, frozen, np.eye(3)[k][None].repeat(2, 0), args.device)
                np.testing.assert_allclose(current, saved['expert_predictions'][:,k], atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL)
                candidate_differences.append(float(np.abs(current.astype(np.float64)-saved['expert_predictions'][:,k]).max()))
                candidate_score = metrics(current, arrays['target'], delta)
                saved_score = metrics(saved['expert_predictions'][:,k], arrays['target'], delta)
                for h in candidate_score['horizons']:
                    for key, value in candidate_score['horizons'][h].items():
                        old = saved_score['horizons'][h][key]
                        if (value != old if key == 'Hit' else abs(value - old) > METRIC_ATOL):
                            raise ValueError('Reused candidate population metrics do not reproduce')
            audit.setdefault('candidate_reproduction', {})[split] = dict(PASS=True, max_abs_differences=candidate_differences)
            collected[split], batches[split] = arrays, frozen
            features[split], candidates[split] = arrays['router_input'].astype(np.float64), saved['expert_predictions']
            std = features[split].std(0)
            input_stats[split] = dict(per_dimension_std=std.tolist(), near_zero_count=int((std <= STD_EPS).sum()),
                                      near_zero_dimensions=np.flatnonzero(std <= STD_EPS).tolist())
            print(f'NORMAL_REPRODUCED {split} {len(arrays["target"])} origins; input {features[split].shape}', flush=True)

        # Only TRAIN targets enter CV, fit, constant predictions, tau or fixed choices.
        d = {s: advantages(losses[s]) for s in ('train','val')}
        ds = data['loaders']['train'].dataset
        if tuple(ds.horizons) != HORIZONS or len(ds.prices) != len(d['train']) + ds.seq_len + max(HORIZONS) - 1:
            raise ValueError('Trading-row target semantics changed')
        rows = np.arange(len(d['train'])) + ds.seq_len - 1
        with threadpool_limits(limits=args.threads):
            probes, cv = fit_train_probes(features['train'], d['train'], rows, data['origin_dates']['train'])
        tau = np.sqrt(np.mean(d['train'] ** 2, axis=(0,2)))
        constant_mean = d['train'].mean(0)
        train_choice = best_candidates(losses['train'].mean(0))[0]
        write_json(out / 'temporal_folds.json', cv)
        write_json(out / 'router_input_statistics.json', input_stats)
        for g, group in enumerate(GROUP_NAMES):
            path = out / f'probe_{group}.npz'
            probes[g].save(path)
            for split in ('train','val'):
                np.testing.assert_array_equal(probes[g].predict(features[split]), LinearProbe.load(path).predict(features[split]))
        write_json(out / 'TRAIN_LOCKED.json', dict(selected_alpha=[p.alpha for p in probes],
            tau=tau.tolist(), tau_degenerate=(tau <= STD_EPS).tolist(), constant_advantage=constant_mean.tolist(),
            train_best_fixed_choice=train_choice.tolist(), train_origin_count=len(d['train']),
            fit_source='TRAIN only', probe_hashes={g:sha256(out/f'probe_{g}.npz') for g in GROUP_NAMES}))
        print(f'TRAIN_FIT_LOCKED alpha={[p.alpha for p in probes]} tau={tau.tolist()}', flush=True)
        predicted = {s: np.stack([p.predict(features[s]) for p in probes], axis=1) for s in ('train','val')}
        soft = {s: soft_weights(predicted[s], tau) for s in ('train','val')}
        probe_mean = soft['train'].mean(0)
        original_mean = collected['train']['pi'].mean(0, dtype=np.float64)
        constant = {s:np.broadcast_to(constant_mean, d[s].shape) for s in ('train','val')}
        permutation = {s:np.random.default_rng(args.permutation_seed).permutation(len(d[s])) for s in ('train','val')}
        scores, reference_scores, fit_rows, strategy_rows, routing, packed = {}, {}, [], [], [], {}
        for split in ('train','val'):
            arrays, cp = collected[split], candidates[split]
            n, y = len(cp), arrays['target']
            oracle = best_candidates(losses[split])[0]
            route_map = dict(original_B_dynamic=arrays['pi'], original_B_TRAIN_mean=original_mean,
                TRAIN_best_fixed=choice_weights(train_choice), constant_advantage_hard=choice_weights(hard_choices(constant[split])),
                ridge_hard_MAIN=choice_weights(hard_choices(predicted[split])), ridge_soft_AUX=soft[split],
                ridge_soft_TRAIN_mean=probe_mean, oracle_EX_POST=choice_weights(oracle),
                ridge_soft_paired_shuffle=soft[split][permutation[split]],
                constant_advantage_soft_AUX=soft_weights(constant[split], tau))
            strategy_preds = dict(original_B_dynamic=arrays['prediction'],
                TRAIN_best_fixed=choose_predictions(cp, train_choice),
                constant_advantage_hard=choose_predictions(cp, hard_choices(constant[split])),
                ridge_hard_MAIN=choose_predictions(cp, hard_choices(predicted[split])),
                oracle_EX_POST=choose_predictions(cp, oracle))
            for name in route_map:
                if name not in strategy_preds:
                    strategy_preds[name] = predict_frozen_inputs(model, batches[split], route_map[name], args.device)
            # Entire hard selection must equal actual one-hot representation/head inference.
            hard_pred = predict_frozen_inputs(model, batches[split], route_map['ridge_hard_MAIN'], args.device)
            np.testing.assert_allclose(hard_pred, strategy_preds['ridge_hard_MAIN'], atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL)
            scores[split] = {name:metrics(p,y,delta) for name,p in strategy_preds.items()}
            fit_rows.extend(prediction_fit_rows(split,d[split],predicted[split],constant[split]))
            for name, weights in route_map.items():
                routing.extend(routing_rows(split,name,weights,losses[split]))
            packed.update({f'{split}_{k}':v for k,v in dict(router_input=features[split], target=y,
                origin_dates=arrays['origin_dates'], commodity_order=arrays['commodity_order'],
                expert_predictions=cp, expert_losses=losses[split], true_advantage=d[split],
                ridge_advantage=predicted[split], constant_advantage=constant[split],
                original_pi=arrays['pi'], ridge_pi=soft[split], ridge_hard_choice=hard_choices(predicted[split]),
                oracle_choice=oracle, route_permutation=permutation[split],
                oracle_tied_candidates=best_candidates(losses[split])[1]).items()})
            packed.update({f'{split}_prediction_{name}':p for name,p in strategy_preds.items()})
            print(f'FROZEN_STRATEGIES {split} complete', flush=True)

        # Original references: strict checkpoint/metadata audit; reuse only matching NPZ metrics.
        for alias in ('d0b','three-global'):
            ref_model, rcfg, rr, rp = load_run(alias,index[alias],'cpu')
            del ref_model
            if rcfg['data_audit'] != fingerprint or rcfg['protocol'] != cfg['protocol']:
                raise ValueError('Reference data/protocol conflict')
            audit.setdefault('references',{})[alias] = rp
            reference_scores[alias] = {}
            for split in ('train','val'):
                with np.load(index[alias][split.upper()+'_predictions'],allow_pickle=False) as z:
                    if (not np.array_equal(z['target'],collected[split]['target'])
                            or not np.array_equal(z['origin_dates'],collected[split]['origin_dates'])
                            or not np.array_equal(z['commodity_order'],collected[split]['commodity_order'])):
                        raise ValueError('Reference targets/dates/order conflict')
                    ref_score = metrics(z['prediction'],z['target'],delta)
                if ref_score['horizons'] != rr['metrics'][split]:
                    raise ValueError('Reference saved population metrics not reproduced')
                reference_scores[alias][split] = ref_score
        for split in ('train','val'):
            comparisons = {**scores[split], **{a:rs[split] for a,rs in reference_scores.items()}}
            for name, score in comparisons.items():
                for g, ids in enumerate(GROUPS):
                    for i in ids:
                        h = str(HORIZONS[i]); m = score['horizons'][h]
                        strategy_rows.append(dict(split=split,strategy=name,group=GROUP_NAMES[g],horizon=HORIZONS[i],
                            group_Huber_sample_mean=score['group_huber'][g],**m,Hit_percent=100*m['Hit'],
                            delta_MAE_vs_original_B=m['MAE']-scores[split]['original_B_dynamic']['horizons'][h]['MAE'],
                            delta_MAE_vs_D0B=m['MAE']-reference_scores['d0b'][split]['horizons'][h]['MAE'],
                            delta_MAE_vs_three_global=m['MAE']-reference_scores['three-global'][split]['horizons'][h]['MAE'],
                            label_conditioned_selection=name=='oracle_EX_POST',main_strategy=name=='ridge_hard_MAIN'))
        after = state_digest(model)
        if before != after or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen model state changed')
        if any(sha256(p) != digest for p,digest in protected.items()):
            raise ValueError('Original artifact/checkpoint changed')
        summary = dict(scores=scores,references=reference_scores,selected_alpha=[p.alpha for p in probes],
            tau=tau.tolist(),tau_degenerate=(tau <= STD_EPS).tolist(),train_fixed_choice=train_choice.tolist(),
            original_TRAIN_mean_pi=original_mean.tolist(),ridge_soft_TRAIN_mean_pi=probe_mean.tolist(),
            origin_counts={s:len(features[s]) for s in features},
            near_zero_dimensions={s:input_stats[s]['near_zero_count'] for s in input_stats})
        for filename, table in (('probe_fit_metrics.csv',fit_rows),('strategy_comparison.csv',strategy_rows),('routing_diagnostics.csv',routing)):
            write_csv(out/filename,table)
        np.savez_compressed(out/'per_origin_predictions.npz',**packed)
        write_json(out/'probe_results.json',summary)
        (out/'probe_summary.md').write_text(build_report(summary,fit_rows,routing),encoding='utf-8')
        provenance = dict(checkpoint=model_provenance, previous_diagnostic_index=str(diag_path),
            previous_diagnostic_provenance=json.loads(Path(di['provenance.json']).read_text()),
            original_file_index=str(index_path), original_file_index_sha256=sha256(index_path),
            data_audit=fingerprint, calendar_audit=data['calendar_audit'],
            source_sha256={p:sha256(ROOT/p) for p in (*SOURCE_FILES,
                'cmgm/scripts/d0b_short_long_expert_diagnostics.py','cmgm/scripts/d0b_frozen_b_advantage_probe.py')},
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                sklearn=sklearn.__version__,device=args.device,threads=args.threads,
                cuda=torch.version.cuda,autocast=False,cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
            protected_artifact_sha256={str(p):v for p,v in protected.items()},
            target_scale='decimal cumulative return',hit_storage='fraction',
            reproduction_atol=REPRODUCTION_ATOL,reproduction_rtol=REPRODUCTION_RTOL,metric_atol=METRIC_ATOL,
            invocation=sys.argv,test_evaluated=False,original_model_training=False,
            probe_fit_only=True,elapsed_seconds=time.perf_counter()-started)
        write_json(out/'provenance.json',provenance)
        audit.update(status='COMPLETE',model_state_before=before,model_state_after=after,
            model_parameters_unchanged=True,original_artifacts_unchanged=True,checkpoint_sha256_before=model_provenance['checkpoint_sha256'],
            checkpoint_sha256_after=sha256(Path(index['b-dynamic']['checkpoint'])),data_audit=fingerprint,
            origins=summary['origin_counts'],features_shape={s:list(features[s].shape) for s in features},
            alpha_selected_on_TRAIN_only=True,probe_restore_exact=True,
            elapsed_seconds=time.perf_counter()-started)
        write_json(out/'audit.json',audit)
        write_json(out/'file_index.json',{p.name:dict(path=str(p.resolve()),sha256=sha256(p)) for p in out.iterdir() if p.is_file()})
        print('COMPLETE '+str(out/'probe_summary.md'),flush=True)
    except Exception as e:
        audit.update(status='FAILED_STOP',error=f'{type(e).__name__}: {e}',elapsed_seconds=time.perf_counter()-started)
        write_json(out/'audit.json',audit)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Fit only the linear probes; frozen TRAIN/VAL inference, never TEST')
    parser.add_argument('--diagnostics',type=Path,help='Completed diagnostic directory or diagnostic_file_index.json')
    parser.add_argument('--file-index',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--permutation-seed',type=int,default=42)
    args = parser.parse_args()
    if args.run:
        run(args)
    else:
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_frozen_b_advantage_probe.py'],cwd=ROOT,check=True)
        print('SYNTHETIC_PREFLIGHT PASS; probe fitting and real data NOT_RUN')


if __name__ == '__main__':
    main()
