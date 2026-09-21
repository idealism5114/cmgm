"""Fixed comparison and descriptive MoE evidence; never model-selection logic."""
import json
from cmgm.scripts.formal_main_ablation_report import table,csv_file
from cmgm.scripts.formal_v2_protocol import atomic_json


def report(r,out):
    evaluation=r.get('evaluation',{});stats=evaluation.get('diagnostics',{})
    allmetrics=evaluation.get('metrics',{});test=allmetrics.get('test',{}).get('5')
    rows=[]
    for n,c in r['controls'].items():
        m=c['metrics']['test']['5'];rows.append(dict(Variant=n,MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],Hit_percent=100*m['Hit'],Source='existing formal control'))
    rows.append(dict(Variant='3-Expert Dense MoE Fusion',MAE=test['MAE'] if test else None,MSE=test['MSE'] if test else None,
        RMSE=test['RMSE'] if test else None,Hit_percent=100*test['Hit'] if test else None,Source='new seed42 single run' if test else 'PENDING'))
    csv_file(out/'fusion_comparison.csv',rows)
    files=dict(config=r['config'],source_hashes=r['source_hashes'],initialization_audit=r['initialization'],
        structural_sanity=dict(initial=r['sanity'],best=r.get('best_sanity'),data=r['data']),
        best_checkpoint_metadata=r.get('best_checkpoint_metadata',{}),test_metrics=allmetrics.get('test',{}),
        moe_diagnostics=stats,reference_provenance=r['controls'])
    if r.get('history'):
        files['training_history']=r['history'];files['routing_history']=r['history']['moe_routing_history']
    for name,value in files.items():atomic_json(out/(name+'.json'),value)
    for name in ('training_history','routing_history'):
        if not (out/(name+'.json')).exists():atomic_json(out/(name+'.json'),[])
    answers=[]
    names=['Adaptive Gated Fusion / Full D0B','Fixed Equal Fusion','TemporalOnly']
    for n in names:
        if not test:answers.append(f'MoE vs {n}: PENDING.')
        else:
            reference=r['controls'][n]['metrics']['test']['5']['MAE'];delta=test['MAE']-reference
            answers.append(f'MoE vs {n}: '+('lower' if delta<0 else 'higher' if delta>0 else 'equal')+f' TEST5 MAE; signed delta={delta:.10g}, relative={100*delta/reference:.6g}%.')
    mechanism=stats.get('test')
    if mechanism:
        p=mechanism['mean_pi'];labels=r['config']['expert_order'];largest=labels[max(range(3),key=lambda k:p[k])]
        entropy=mechanism['routing_entropy'];hard=mechanism['hard_occupancy'];d=mechanism['disagreement']
        answers += [f'Largest mean learned routing weight: {largest}. Hard occupancy T/S/ST={hard}.',
            f'Routing concentration: mean pi={p}; entropy={entropy:.8g} versus uniform log(3)={mechanism["uniform_entropy"]:.8g}; effective expert count={mechanism["effective_expert_count"]:.8g}. No collapse cutoff was preregistered: these measurements describe concentration without treating hard occupancy alone as collapse.',
            f'Expert representation disagreement: {d}. '+('All measured representations coincide.' if max(d.values())==0 else 'Measured expert outputs differ; this alone does not establish distinct predictive functions.'),
            f'Interaction routing: mean={p[2]:.8g}, median={mechanism["pi_quantiles"]["0.5"][2]:.8g}, P90={mechanism["pi_quantiles"]["0.9"][2]:.8g}, max={mechanism["pi_max"][2]:.8g}. Descriptive evidence, not a causal contribution estimate.',
            f'Spatial routing: mean={p[1]:.8g}, P90={mechanism["pi_quantiles"]["0.9"][1]:.8g}, max={mechanism["pi_max"][1]:.8g}; largest-weight fraction={hard[1]:.8g}.',
            f'TEST mean learned pi_T/pi_S/pi_ST={p}; effective pi={mechanism["mean_effective_pi"]}.',
            f'TEST learned routing entropy={entropy:.10g}.',
            f'The routing profile favors {largest} on average. Routing weights and expert disagreement are descriptive; this single comparison cannot causally attribute any improvement to temporal, spatial or interaction specialization.',
            'Accept this fixed-design result. No second seed, rerun for poor performance, expert-count/router/lambda/dropout search or other model follows.']
    else:
        answers += ['PENDING: trained routing profile.','PENDING: routing concentration/collapse evidence.',
                    'PENDING: expert representation disagreement.','PENDING: Interaction routing weight.',
                    'PENDING: Spatial usage across samples.','PENDING: TEST mean pi_T/pi_S/pi_ST.',
                    'PENDING: TEST routing entropy.','PENDING: descriptive source of any improvement.',
                    'If MoE fails, retain the result and STOP; no automatic tuning or retry.']
    text='# D0B-MoEFusion\n\nStatus: '+r['status']+'\n\n'
    text+='## 1. Experiment Objective\n\nDoes expert-specialized MoE fusion outperform the original adaptive gated fusion? One fixed design, seed42, one formal run.\n\n'
    text+='## 2. Fixed D0B Components\n\nSpatial TempWeighted/AdaptiveGraph/EdgeAttnMixHop, temporal encoder/causal Transformer/Base RPE/Markov/G1–G3/Z/Balanced Readout, W_s/W_t and the shared prediction head retain their native definitions. Structure is frozen, not parameter training: all active parameters train end-to-end. Fusion experts do not share weights with latent regime generators. Full D0B keeps its original gate.\n\n'
    text+='## 3. MoE Architecture\n\n'
    text+=r'''\[
s=W_sh_s,\quad t=W_th_t,\quad e_T=E_T(t),\quad e_S=E_S(s),\quad e_{ST}=E_{ST}([s\Vert t]).
\]
\[
\pi=\operatorname{Softmax}(R([h_s\Vert h_t])),\quad
\pi^{eff}=(1-\gamma)U_3+\gamma\pi,\quad \gamma=\min(1,e/10).
\]
\[
h_{MoE}=\pi_T^{eff}e_T+\pi_S^{eff}e_S+\pi_{ST}^{eff}e_{ST},\quad
\hat Y=f_{head}(h_{MoE}).
\]
\[
L=\sum_h\operatorname{Huber}_{.02}(\hat Y_h,Y_h)+L_{switch}+10^{-4}L_{balance},\quad
L_{balance}=\sum_k\bar\pi_k\log((\bar\pi_k+10^{-8})/(1/3)).
\]

T/S experts: residual Linear(64,64) → ReLU → Dropout(.1) → Linear(64,64).
ST: Linear(128,64) → ReLU → Dropout(.1) → Linear(64,64), no residual.
Router: Linear(128,32) → ReLU → Linear(32,3), softmax; all experts execute.
One original shared Linear64/ReLU/Dropout(.3)/Linear96 head follows the fused representation.

'''
    a=r['initialization'];text+='## 4. Initialization Audit\n\n'+table([{k:a.get(k) for k in ('baseline_parameters','moe_parameters','delta_parameters','shared_max_abs_diff','mismatch_count','PASS')}])+'\nShared native modules are constructed first; only the new variant removes gate_fc. New fusion parameters use ordinary PyTorch Linear initialization. Initial prediction equality is not required.\n\n'
    text+='## 5. Structural Sanity\n\n'+table([dict(Stage='initial',PASS=r['sanity'].get('PASS'),Device=r['sanity'].get('device')),dict(Stage='best',PASS=r.get('best_sanity',{}).get('PASS'),Device=r['sanity'].get('device'))])+'\nShape/formula, dense autograd connectivity, raw-vs-projected input wiring, normalization, warm-up, balance, batch/single-sample and temporal prefix causality are recorded in structural_sanity.json. Learned collapse after training is reported, not used to invalidate an otherwise correctly wired model. Commodity mapping and data fingerprints match the formal controls.\n\n'
    text+='## 6. Training Protocol\n\n```json\n'+json.dumps(r['config'],indent=2)+'\n```\n\nCheckpoint/early-stop/scheduler use prediction-only four-horizon VAL Huber; neither balance nor Switch KL enters selection. Learned pi controls balance, not warm-up pi_eff. The saved best epoch restores its exact gamma for VAL/TEST, including early best epochs. No gradient clipping or separate optimizer groups.\n\n'
    text+='## 7. TEST Multi-Horizon Metrics\n\n'+table([dict(Horizon=h,MAE=m['MAE'],MSE=m['MSE'],RMSE=m['RMSE'],Hit_percent=100*m['Hit']) for h,m in allmetrics.get('test',{}).items()])+'\nPooled origins × 24 commodities; direct MSE, RMSE=sqrt(MSE), unmasked sign Hit including zero targets.\n\n'
    text+='## 8. Fusion Comparison\n\n'+table(rows)+'\nControl values are read from audited existing formal artifacts, not hard-coded or retrained. Exact report/checkpoint paths, hashes and original training metadata are recorded in reference_provenance.json.\n\n'
    text+='## 9. Routing Statistics\n\n'+table([dict(Split=s,MeanPi=d['mean_pi'],MeanEffectivePi=d['mean_effective_pi'],Entropy=d['routing_entropy'],HardOccupancy=d['hard_occupancy'],Gamma=d['gamma_warmup']) for s,d in stats.items()])+'\nTRAIN epoch diagnostics cover the original drop-last training batches; final TRAIN diagnostics cover all origins. Mean routing/entropy pool observations; balance_loss is the mean batch loss used by the objective.\n\n'
    text+='## 10. Expert Specialization Diagnostics\n\n'+(json.dumps({k:mechanism[k] for k in ('expert_mean_L2','h_moe_mean_L2','disagreement','pi_quantiles')},indent=2) if mechanism else 'PENDING.')+'\n\n'
    text+='## 11. Interpretation\n\n'
    if test:
        full=r['controls'][names[0]]['metrics']['test']['5']['MAE'];temporal=r['controls']['TemporalOnly']['metrics']['test']['5']['MAE']
        if test['MAE']<full:
            text+='Under the fixed single-run protocol, the MoE fusion yields lower MAE than the original adaptive gated fusion.\n\n'
            if test['MAE']>=temporal:text+='Although the MoE improves branch fusion relative to the original gated fusion, the experiment does not establish a positive net contribution of the spatial branch over TemporalOnly.\n\n'
        else:text+='MoE does not improve TEST5 MAE over the original adaptive gated fusion under this fixed protocol. Accept the result; do not tune or rerun.\n\n'
        if mechanism['mean_pi'][0]==max(mechanism['mean_pi']):text+='The temporal specialist has the largest mean weight, consistent with the previous branch-level evidence favoring TemporalOnly. The actual probability and entropy values determine how concentrated this preference is.\n\n'
    else:text+='PENDING: no formal MoE training has been executed. No forecasting conclusion is available.\n\n'
    text+='## 12. Required Answers\n\n'+''.join(f'{i}. {a}\n\n' for i,a in enumerate(answers,1))
    text+='## 13. Limitations\n\nSingle run, no significance or multi-seed robustness claim. MoE has additional parameters and the prescribed balance regularizer/warm-up; this experiment does not isolate expert specialization from capacity/regularization effects. Router weights are not causal attributions, and different representations alone do not prove useful specialization. Existing controls are audited frozen results.\n\nSTOP\n'
    (out/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
