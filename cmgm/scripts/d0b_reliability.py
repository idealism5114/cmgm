"""Frozen D0B interpretability and error-risk study.

Default is synthetic preflight. Real-data stages require --run. No optimizer
is constructed for D0B. The separate risk stage fits only a logistic monitor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import (align_markets, load_bond_prices,
    load_commodity_prices, load_stock_prices)
from cmgm.interpretability.d0b_reliability import (
    D0BTrace, HORIZONS, LAYER_NAMES, block_mismatch, entropy,
    fixed_gate_shapley, market_descriptors, matched_random_sources,
    normalize_masked_attention, paired_moving_block_bootstrap,
    paired_risk_block_bootstrap,
    regime_summary, risk_evaluation, source_intervention,
    target_gradient_scores, train_only_state_alignment,
    weighted_regime_profile)
from cmgm.interpretability.d0b_graph import graph_study
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.scripts.d0b_node_history_readout import (_data_fingerprint,
    _model_kwargs, BASE_VARIANT)
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT=ROOT/"checkpoints/d0b_node_history_readout/switching_latent_balanced_readout_seed42_20261006_231320_782593/best.pt"
EXPECTED_SHA="6af1ca4264fb4f62ff0dabe05ffd3aa8d43c24c3b3b28464be936cc669065021"
REFERENCE_RESULT=ROOT/"experiments/d0b_node_history_readout/switching_latent_balanced_readout_seed42_20261006_231320_782593/result.json"
SOURCES=("cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/switching_latent_transformer.py",
    "cmgm/models/switching_transformer.py",
    "cmgm/graph/adaptive_graph.py",
    "cmgm/models/model.py",
    "cmgm/interpretability/d0b_reliability.py",
    "cmgm/interpretability/d0b_graph.py",
    "cmgm/scripts/d0b_reliability.py",
    "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/d0b_node_history_readout.py",
    "cmgm/config.py",
    "cmgm/data/data_loader.py","cmgm/training/metric_standard.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    no_overwrite(path)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def no_overwrite(path):
    if Path(path).exists(): raise FileExistsError(f"Refusing to overwrite {path}")


def require_completed_extract(out, splits, device):
    """Fail before costly data loading when a downstream stage lacks its input."""
    for split in splits:
        npz=out/f"extract_{split}.npz"
        report=out/f"extract_{split}.json"
        if not npz.is_file() or not report.is_file():
            raise RuntimeError(
                f"Missing completed {split.upper()} extraction in {out}. "
                f"Run: ../.venv/bin/python -m cmgm.scripts.d0b_reliability "
                f"--stage extract --split {split} --run --device {device} "
                f"--output-dir {out}")
        if json.loads(report.read_text()).get("status")!="COMPLETE":
            raise RuntimeError(f"{report} is not marked COMPLETE; inspect extraction")


def require_current_source_audit(out):
    """Do not mix post-fix results with artifacts audited under earlier code."""
    path=out/"audit.json"
    if path.exists():
        recorded=json.loads(path.read_text()).get("source_hashes")
        current={s:digest(ROOT/s) for s in SOURCES}
        if recorded!=current:
            raise RuntimeError(
                f"{path} was created under different source code. "
                "Preserve it and choose a NEW --output-dir for this run.")


def synthetic_preflight():
    """No real data or checkpoint. Model prediction, attribution, graph and risk logic."""
    torch.manual_seed(42)
    m=HeteroMixHopCMGM(num_nodes=30,n_stock=4,n_bond=2,n_commodities=24,
                      commodity_indices=(6,30),variant=BASE_VARIANT).eval()
    m.requires_grad_(False)
    x=torch.randn(2,20,30,21)
    with torch.no_grad():
        reference=m(x)
        with D0BTrace(m) as cap: t=cap.forward(x)
        assert torch.allclose(t["prediction"],reference,rtol=0,atol=0)
        assert t["posterior"].shape==(2,20,3)
        assert t["applied_route"].shape==(2,20,3)
        assert t["gate"].shape==(2,64)
        assert len(t["source_summary"])==4
        assert all(z.shape==(2,m.attn_mixhop1.n_heads,30)
                   for z in t["source_summary"])
        sh=fixed_gate_shapley(m.head,t["a"],t["b"])
        assert sh["completeness_error"].max()<1e-6
        assert sh["prediction"].shape==(2,96)
        p=t["posterior"].clone()
        with D0BTrace(m) as cap:
            forced=cap.forward(x,forced_route=torch.tensor([1.,0.,0.]))
        assert torch.equal(forced["posterior"],p)
        assert torch.all(forced["applied_route"][:,:]==torch.tensor([1.,0.,0.]))
        assert (forced["prediction"]-reference).abs().max()>0
        with D0BTrace(m) as cap:
            h_removed=cap.forward(x,zero_component="H")["prediction"]
        assert h_removed.shape==reference.shape
        a=torch.ones(1,4,4,2)/4
        masked=normalize_masked_attention(a,torch.ones(4,dtype=torch.bool))
        assert torch.allclose(masked.sum(2),torch.ones(1,4,2))
        assert torch.allclose(torch.diagonal(masked,dim1=1,dim2=2),
                              torch.ones(1,2,4))
        assert torch.isfinite(masked).all()
        with source_intervention(m,torch.tensor([True]*6+[False]*24)):
            intervention=m(x)
        assert intervention.shape==reference.shape and torch.isfinite(intervention).all()
    # Frozen parameter tensors, but live alpha through a grad-enabled input.
    xg=x[:1].clone().requires_grad_()
    with D0BTrace(m,live_attention=True) as cap:
        g=cap.forward(xg)
        score=target_gradient_scores(g["prediction"],cap.attentions,1,0)
        assert score.shape==(1,30) and torch.isfinite(score).all()
    assert all(v.grad is None for v in m.parameters())
    p=np.array([[.9,.05,.05],[.1,.8,.1],[.1,.1,.8]]*20)
    assert len(regime_summary(p)["soft_occupancy"])==3
    z=np.arange(80*24).reshape(80,24)
    ci=paired_moving_block_bootstrap(z,block=40,draws=5)
    assert len(ci["low"])==24
    for row in z:
        assert len(row)==24
    cal=dict(error5=np.r_[np.zeros(20),np.ones(20)],B=np.random.default_rng(1).normal(size=(40,2)))
    ev=dict(error5=np.r_[np.zeros(20),np.ones(20)],B=np.random.default_rng(2).normal(size=(40,2)))
    for key in ("R","G","A"):
        cal[key]=np.random.default_rng(len(key)+3).normal(size=(40,2))
        ev[key]=np.random.default_rng(len(key)+4).normal(size=(40,2))
    assert len(risk_evaluation(cal,ev))==8
    return dict(PASS=True,prediction_capture_exact=True,shapley_complete=True,
        forced_route_reaches_Z=True,posterior_unchanged_by_forcing=True,
        graph_source_axis=2,all_mask_self_loop_finite=True,
        graph_gradient_live=True,frozen_parameter_gradients_none=True,
        grouped_bootstrap=True,risk_combinations=8,real_data="NOT_RUN")


def load_context(checkpoint, device, seed=42, reference_result=REFERENCE_RESULT):
    checkpoint=Path(checkpoint).resolve()
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    cp_hash=digest(checkpoint)
    if checkpoint==DEFAULT_CHECKPOINT.resolve() and cp_hash!=EXPECTED_SHA:
        raise RuntimeError("Canonical seed42 checkpoint SHA256 mismatch")
    reference_result=Path(reference_result)
    if not reference_result.exists():raise FileNotFoundError("Original result metadata absent")
    prior=json.loads(reference_result.read_text())
    if prior["model"]["variant"]!=BASE_VARIANT or prior["model"]["seed"]!=seed:
        raise RuntimeError("Original reference identity mismatch")
    if checkpoint==DEFAULT_CHECKPOINT.resolve():
        recorded=prior.get("checkpoint_sha256")
        if recorded and recorded!=cp_hash:raise RuntimeError("Original checkpoint metadata hash mismatch")
    data=build_data(SimpleNamespace(batch_size=64,seq_len=20,seed=seed))
    fp=_data_fingerprint(data,include_test=False)
    if fp!=prior["model"]["data_fingerprint"]:
        raise RuntimeError("Data/commodity order differs from original seed42 run")
    kwargs=_model_kwargs(data)
    model=HeteroMixHopCMGM(**kwargs,variant=BASE_VARIANT)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(payload["model_state_dict"],strict=True)
    if payload.get("best_epoch")!=prior["best_epoch"]:
        raise RuntimeError("Best epoch differs from audited seed42 result")
    model.requires_grad_(False);model.to(device).eval()
    raw,mi=align_markets(load_stock_prices(config.STOCK_FILE),
        load_bond_prices(config.BOND_FILE),load_commodity_prices(config.COMMODITY_FILE))
    if list(raw.columns)!=data["feature_names"]:raise RuntimeError("Raw asset order differs")
    if len(raw)!=sum(len(v) for v in (
        data["loaders"]["train"].dataset.raw_prices,
        data["loaders"]["val"].dataset.raw_prices,
        data["loaders"]["test"].dataset.raw_prices)):
        raise RuntimeError("Calendar mismatch")
    split_calendar={}
    for split_name in ("train","val","test"):
        origin,label20=origin_dates(raw,split_name,data["loaders"][split_name].dataset)
        split_calendar[split_name]=dict(origins=len(origin),
            first_origin=str(origin[0]),last_origin=str(origin[-1]),
            first_20d_label_date=str(label20[0]),last_20d_label_date=str(label20[-1]))
    audit=dict(status="AUDITED",seed=seed,reference_result=str(reference_result),
        checkpoint=str(checkpoint),checkpoint_sha256=cp_hash,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        worktree_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        source_hashes={s:digest(ROOT/s) for s in SOURCES},
        train_val_data_fingerprint=fp,
        dates=dict(start=str(raw.index[0].date()),end=str(raw.index[-1].date()),
                   split_calendar=split_calendar),
        asset_order=[str(s) for s in raw.columns],
        target_unit="decimal return, clipped to [-1,1] by existing data pipeline",
        target_definition="for each h, raw_price[t+h]/max(abs(raw_price[t]),1e-8)-1, clipped to [-1,1]",
        horizons=list(HORIZONS),metric="pooled origins × commodities; RMSE=sqrt(population MSE); unmasked sign Hit",
        checkpoint_selection="prediction-only four-horizon VAL Huber, original batch-mean aggregation",
        historical_val=prior["metrics"]["val"],bootstrap="NOT_RUN",
        intervention_reference=dict(train_mean_gate="TRAIN origins",
            train_mean_route="TRAIN origins and historical timesteps",
            route_mismatch_seed=42,graph_random_seed=42,
            graph_target_horizon=5),
        test_evaluated_in_this_study=False)
    return model,data,raw,mi,audit


def origin_dates(raw, split, dataset):
    return reconstructed_origin_dates(raw,split,len(dataset))


def reconstructed_origin_dates(raw, split, origin_count):
    """Rebuild dates from the audited chronological calendar, without NPZ pickle."""
    n=len(raw);a=int(n*config.TRAIN_RATIO);b=a+int(n*config.VAL_RATIO)
    start={"train":0,"val":a,"test":b}[split]
    end={"train":a,"val":b,"test":n}[split]
    if origin_count!=end-start-20-max(HORIZONS)+1:
        raise RuntimeError("Split origin count mismatch")
    rows=start+np.arange(origin_count)+19
    return (np.asarray(raw.index[rows].strftime("%Y-%m-%d"),dtype="U10"),
            np.asarray(raw.index[rows+20].strftime("%Y-%m-%d"),dtype="U10"))


def write_safe_extract_archive(source, destination, dates, label20):
    """Copy numeric results exactly; rebuild date arrays without unpickling them."""
    source,destination=Path(source),Path(destination)
    no_overwrite(destination)
    with np.load(source,allow_pickle=False) as old:
        if not {"date_last","target_end20"}.issubset(old.files):
            raise RuntimeError(f"Missing expected date fields in {source}")
        arrays={k:old[k] for k in old.files if k not in ("date_last","target_end20")}
    count=len(arrays["prediction"])
    dates=np.asarray(dates,dtype="U10")
    label20=np.asarray(label20,dtype="U10")
    if dates.shape!=(count,) or label20.shape!=(count,):
        raise RuntimeError("Reconstructed dates do not match extracted origins")
    arrays.update(date_last=dates,target_end20=label20)
    temporary=destination.with_name(destination.name+".tmp")
    no_overwrite(temporary)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle,**arrays)
    with np.load(temporary,allow_pickle=False) as rebuilt:
        if set(rebuilt.files)!=set(arrays) or any(
                not np.array_equal(rebuilt[k],value) for k,value in arrays.items()):
            raise RuntimeError("Repacked extraction differs from source arrays")
    temporary.replace(destination)
    return dict(source_archive=str(source),source_sha256=digest(source),
                safe_archive=str(destination),safe_sha256=digest(destination),
                origins=count,changed_fields=["date_last","target_end20"],
                other_arrays_exact=True)


def migrate_date_archives(source_dir, destination_dir):
    """Create a separate, audited continuation from completed TRAIN/VAL files."""
    source_dir=Path(source_dir).resolve()
    destination_dir=Path(destination_dir).resolve()
    if source_dir==destination_dir or (destination_dir.exists() and any(destination_dir.iterdir())):
        raise FileExistsError("Choose an empty NEW --output-dir; original files stay untouched")
    original_audit=source_dir/"audit.json"
    if not original_audit.is_file():raise FileNotFoundError(original_audit)
    audit=json.loads(original_audit.read_text())
    if audit.get("status")!="AUDITED" or digest(audit["checkpoint"])!=audit["checkpoint_sha256"]:
        raise RuntimeError("Source audit or frozen checkpoint identity is invalid")
    prior=json.loads(Path(audit["reference_result"]).read_text())
    if (prior["model"]["variant"]!=BASE_VARIANT or
            audit["train_val_data_fingerprint"]!=prior["model"]["data_fingerprint"]):
        raise RuntimeError("Source experiment and formal D0B reference disagree")
    raw,mi=align_markets(load_stock_prices(config.STOCK_FILE),
        load_bond_prices(config.BOND_FILE),load_commodity_prices(config.COMMODITY_FILE))
    if list(raw.columns)!=audit["asset_order"]:
        raise RuntimeError("Calendar asset order differs from extraction audit")
    prepared={}
    for split in ("train","val"):
        report_path=source_dir/f"extract_{split}.json"
        archive_path=source_dir/f"extract_{split}.npz"
        if not report_path.is_file() or not archive_path.is_file():
            raise FileNotFoundError(f"Completed {split} extraction missing")
        report=json.loads(report_path.read_text())
        if report.get("status")!="COMPLETE" or report.get("split")!=split:
            raise RuntimeError(f"{split} extraction not COMPLETE")
        dates,label20=reconstructed_origin_dates(raw,split,report["n_origins"])
        calendar=audit["dates"]["split_calendar"][split]
        if (dates[0]!=calendar["first_origin"] or dates[-1]!=calendar["last_origin"] or
                label20[0]!=calendar["first_20d_label_date"] or
                label20[-1]!=calendar["last_20d_label_date"]):
            raise RuntimeError(f"{split} reconstructed calendar differs from audit")
        with np.load(archive_path,allow_pickle=False) as old:
            if not np.array_equal(old["commodity_order"],
                    np.asarray(audit["train_val_data_fingerprint"]["commodity_order"])):
                raise RuntimeError(f"{split} commodity order differs from audit")
            if old["prediction"].shape!=(report["n_origins"],4,24):
                raise RuntimeError(f"{split} prediction shape differs from report")
            for i,h in enumerate(HORIZONS):
                metrics=population_metrics(old["prediction"][:,i],old["target"][:,i])
                if any(abs(metrics[k]-report["metrics"][str(h)][k])>1e-12 for k in metrics):
                    raise RuntimeError(f"{split} metrics differ from report")
        prepared[split]=(report,archive_path,dates,label20)
    destination_dir.mkdir(parents=True,exist_ok=True)
    migrations={}
    for split,(report,archive_path,dates,label20) in prepared.items():
        destination_archive=destination_dir/f"extract_{split}.npz"
        migrations[split]=write_safe_extract_archive(
            archive_path,destination_archive,dates,label20)
        copied=dict(report,path=str(destination_archive),
            date_repair="Calendar strings reconstructed from audited raw chronology; numeric arrays unchanged",
            source_report=str(source_dir/f"extract_{split}.json"),
            source_report_sha256=digest(source_dir/f"extract_{split}.json"))
        atomic_json(destination_dir/f"extract_{split}.json",copied)
    revised=dict(audit,source_hashes={s:digest(ROOT/s) for s in SOURCES},
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        worktree_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        migration=dict(source_dir=str(source_dir),source_audit_sha256=digest(original_audit),
            source_hashes=audit["source_hashes"],archive_repair=migrations,
            model_inference_repeated=False,test_evaluated=False))
    atomic_json(destination_dir/"audit.json",revised)
    return dict(status="COMPLETE",output_dir=str(destination_dir),
                archive_repair=migrations,model_inference_repeated=False)


def _np(x):
    return x.detach().cpu().numpy()


def extract(model,data,raw,mi,split,out,device):
    ds=data["loaders"][split].dataset
    dates,label20=origin_dates(raw,split,ds)
    loader=DataLoader(ds,batch_size=32,shuffle=False,drop_last=False)
    arrays={k:[] for k in ("prediction","target","posterior","applied_route",
        "prior","gate","s","t","a","b","phi_t","phi_s","phi_baseline",
        "temporal_abs_share","cancellation","source_attention",
        "source_attention_by_layer_hop_head","B","R","G")}
    previous_terminal_p=None
    for start,batch in enumerate(loader):
        lo=start*32
        x,y=batch[0].to(device),batch[1]
        if start==0:
            with torch.inference_mode():reference_prediction=model(x)
        with torch.inference_mode(),D0BTrace(model) as cap:
            tr=cap.forward(x)
            if start==0:
                torch.testing.assert_close(tr["prediction"],reference_prediction,
                                           rtol=0,atol=0)
            sh=fixed_gate_shapley(model.head,tr["a"],tr["b"])
            torch.testing.assert_close(tr["prediction"].reshape(len(x),-1),
                                       sh["prediction"],rtol=1e-5,atol=1e-7)
            if sh["completeness_error"].max()>1e-6:raise RuntimeError("Shapley incomplete")
            source_detail=torch.stack(tr["source_summary"],dim=1).reshape(
                len(x),2,2,model.attn_mixhop1.n_heads,model.num_nodes)
            source=source_detail.mean(dim=(1,2,3))
            values=dict(prediction=tr["prediction"],target=y,
                posterior=tr["posterior"],applied_route=tr["applied_route"],
                prior=tr["prior"],gate=tr["gate"],s=tr["s"],t=tr["t"],
                a=tr["a"],b=tr["b"],phi_t=sh["phi_temporal"].reshape_as(y),
                phi_s=sh["phi_spatial"].reshape_as(y),
                phi_baseline=sh["baseline"].reshape_as(y),
                temporal_abs_share=sh["temporal_abs_share"].reshape_as(y),
                cancellation=sh["cancellation"].reshape_as(y),
                source_attention=source,
                source_attention_by_layer_hop_head=source_detail)
            for key,val in values.items():arrays[key].append(_np(val))
            for j in range(len(x)):
                i=lo+j
                origin=i+19
                desc=market_descriptors(ds.raw_prices[i:i+20],
                    [mi[k] for k in ("stock","bond","commodity")])
                bvec=[desc["vol_commodity_20"],
                      float(tr["prediction"][j,1].abs().mean())]
                p=_np(tr["posterior"][j,-1])
                p_prev=p if previous_terminal_p is None else previous_terminal_p
                rvec=[float(entropy(p)),float(np.abs(p-p_prev).sum())]
                previous_terminal_p=p.copy()
                gvec=[float(sh["temporal_abs_share"][j].mean()),
                      float(sh["cancellation"][j].mean())]
                arrays["B"].append(bvec);arrays["R"].append(rvec);arrays["G"].append(gvec)
        if start==0 or (start+1)%10==0 or lo+len(x)==len(ds):
            print(f"[extract {split}] {lo+len(x)}/{len(ds)} origins",flush=True)
    packed={k:np.concatenate(v) if k not in ("B","R","G") else np.asarray(v,dtype=np.float32)
            for k,v in arrays.items()}
    packed.update(date_last=dates,target_end20=label20,
                  commodity_order=np.asarray(data["feature_names"][mi["commodity"][0]:mi["commodity"][1]]),
                  horizons=np.asarray(HORIZONS))
    path=out/f"extract_{split}.npz";no_overwrite(path)
    np.savez_compressed(path,**packed)
    mae={str(h):population_metrics(packed["prediction"][:,i],packed["target"][:,i])
         for i,h in enumerate(HORIZONS)}
    report=dict(status="COMPLETE",split=split,n_origins=len(ds),metrics=mae,
        posterior_terminal_summary=regime_summary(packed["posterior"][:,-1]),
        capture_prediction_equal_original=True,full_attention_saved=False,
        path=str(path),test_evaluated=split=="test")
    atomic_json(out/f"extract_{split}.json",report)
    return report


def state_analysis(out,raw,mi,split):
    train=np.load(out/"extract_train.npz",allow_pickle=False)
    other=np.load(out/f"extract_{split}.npz",allow_pickle=False)
    def descriptions(z):
        loc=raw.index.get_indexer(z["date_last"])
        if (loc<19).any():raise RuntimeError("Invalid descriptor date")
        return np.array([list(market_descriptors(raw.values[i-19:i+1],
                   [mi[k] for k in ("stock","bond","commodity")]).values()) for i in loc])
    dt,do=descriptions(train),descriptions(other)
    pt,po=train["posterior"][:,-1],other["posterior"][:,-1]
    actual=weighted_regime_profile(pt,dt)
    shifted=weighted_regime_profile(block_mismatch(pt),dt)
    report=dict(status="COMPLETE",train=regime_summary(pt),
        evaluation=regime_summary(po),train_weighted_profile=actual.tolist(),
        evaluation_profile_fixed_states=weighted_regime_profile(po,do).tolist(),
        train_block_mismatch_profile=shifted.tolist(),
        descriptor_names=list(market_descriptors(raw.values[0:20],
              [mi[k] for k in ("stock","bond","commodity")])) ,
        state_labels=["State 1","State 2","State 3"],
        causal_claim=False,split=split)
    atomic_json(out/f"state_{split}.json",report);return report


def route_interventions(model,data,split,out,device):
    train=np.load(out/"extract_train.npz",allow_pickle=False)
    ref=np.load(out/f"extract_{split}.npz",allow_pickle=False)
    mean=train["posterior"].mean(axis=(0,1)).astype(np.float32)
    ds=data["loaders"][split].dataset
    rng=np.random.default_rng(42)
    order=np.arange(len(ds));blocks=[order[i:i+40] for i in range(0,len(ds),40)]
    shifted=np.concatenate(blocks[1:]+blocks[:1])
    # Each complete posterior window is moved as a unit; only Z mixture uses it.
    options={"TRAIN_mean_Z_route":np.broadcast_to(mean,(len(ds),20,3)),
             "block_shifted_Z_route":ref["posterior"][shifted]}
    base=ref["prediction"];target=ref["target"]
    results={}
    for name,forced in options.items():
        preds=[];gates=[];applied=[];post=[]
        loader=DataLoader(ds,batch_size=32,shuffle=False,drop_last=False)
        for n,batch in enumerate(loader):
            x=batch[0].to(device);f=torch.as_tensor(forced[n*32:n*32+len(x)],device=device)
            with torch.inference_mode(),D0BTrace(model) as cap:
                tr=cap.forward(x,forced_route=f)
            for key,value,dest in (("prediction",tr["prediction"],preds),
                                   ("gate",tr["gate"],gates),
                                   ("applied_route",tr["applied_route"],applied),
                                   ("posterior",tr["posterior"],post)):
                dest.append(_np(value))
        pred=np.concatenate(preds);gate=np.concatenate(gates)
        vals={str(h):dict(delta_MAE=population_metrics(pred[:,i],target[:,i])["MAE"]-
                      population_metrics(base[:,i],target[:,i])["MAE"],
                      mean_abs_prediction_change=float(np.abs(pred[:,i]-base[:,i]).mean()))
              for i,h in enumerate(HORIZONS)}
        results[name]=dict(metrics=vals,mean_abs_gate_change=float(np.abs(gate-ref["gate"]).mean()),
            mean_original_posterior=np.concatenate(post).mean(axis=(0,1)).tolist(),
            mean_applied_route=np.concatenate(applied).mean(axis=(0,1)).tolist())
    atomic_json(out/f"route_{split}.json",dict(status="COMPLETE",
        semantics="forced_probabilities replaces mixture weights entering the full Z recursion; original Markov filter posterior continues unchanged",
        train_reference_mean=mean.tolist(),interventions=results))


def branch_interventions(model,data,split,out,device):
    train=np.load(out/"extract_train.npz",allow_pickle=False)
    ref=np.load(out/f"extract_{split}.npz",allow_pickle=False)
    gmean=torch.as_tensor(train["gate"].mean(0),device=device)
    a0=torch.as_tensor(train["a"].mean(0),device=device)[None]
    b0=torch.as_tensor(train["b"].mean(0),device=device)[None]
    ds=data["loaders"][split].dataset;loader=DataLoader(ds,batch_size=32,shuffle=False)
    pred={k:[] for k in ("TRAIN_mean_gate","without_weighted_T",
                         "without_weighted_S","without_H_readout",
                         "without_Z_readout")}
    sensitivity=[];mean_phi_t=[];mean_phi_s=[];mean_baseline=[]
    for batch in loader:
        x=batch[0].to(device)
        with torch.inference_mode(),D0BTrace(model) as cap:tr=cap.forward(x)
        sh=fixed_gate_shapley(model.head,tr["a"],tr["b"],
                              a_reference=a0,b_reference=b0)
        sensitivity.append(float(sh["completeness_error"].max()))
        mean_phi_t.append(_np(sh["phi_temporal"].reshape(len(x),4,-1)))
        mean_phi_s.append(_np(sh["phi_spatial"].reshape(len(x),4,-1)))
        mean_baseline.append(_np(sh["baseline"].reshape(len(x),4,-1)))
        s,t=tr["s"],tr["t"]
        variants=dict(TRAIN_mean_gate=gmean*t+(1-gmean)*s,
                      without_weighted_T=tr["b"],
                      without_weighted_S=tr["a"])
        with torch.inference_mode():
            for name,fused in variants.items():
                pred[name].append(_np(model.head(fused).view(len(x),4,-1)))
            for name,part in (("without_H_readout","H"),("without_Z_readout","Z")):
                pred[name].append(_np(model._switching_latent_transformer_forward(
                    x,zero_readout_component=part)))
    baseline=ref["prediction"];target=ref["target"]
    results={}
    for name,pieces in pred.items():
        value=np.concatenate(pieces)
        results[name]={str(h):dict(delta_MAE=population_metrics(value[:,i],target[:,i])["MAE"]-
             population_metrics(baseline[:,i],target[:,i])["MAE"],
             mean_abs_prediction_change=float(np.abs(value[:,i]-baseline[:,i]).mean()))
             for i,h in enumerate(HORIZONS)}
    mean_path=out/f"branch_TRAIN_mean_reference_{split}.npz"
    no_overwrite(mean_path)
    np.savez_compressed(mean_path,phi_temporal=np.concatenate(mean_phi_t),
        phi_spatial=np.concatenate(mean_phi_s),baseline=np.concatenate(mean_baseline),
        a_reference=_np(a0),b_reference=_np(b0))
    posterior=ref["posterior"][:,-1].astype(np.float64)
    by_state={}
    for k in range(3):
        weights=posterior[:,k]/posterior[:,k].sum()
        by_state[f"State {k+1}"]=dict(
            mean_signed_temporal_5d=float(np.sum(weights[:,None]*ref["phi_t"][:,1])/ref["phi_t"].shape[2]),
            mean_signed_spatial_5d=float(np.sum(weights[:,None]*ref["phi_s"][:,1])/ref["phi_s"].shape[2]),
            mean_abs_temporal_5d=float(np.sum(weights[:,None]*np.abs(ref["phi_t"][:,1]))/ref["phi_t"].shape[2]),
            mean_abs_spatial_5d=float(np.sum(weights[:,None]*np.abs(ref["phi_s"][:,1]))/ref["phi_s"].shape[2]))
    atomic_json(out/f"branch_{split}.json",dict(status="COMPLETE",
        TRAIN_mean_gate=train["gate"].mean(0).tolist(),
        TRAIN_mean_reference_completeness_max=max(sensitivity),
        TRAIN_mean_reference_artifact=str(mean_path),
        posterior_weighted_contributions_by_state=by_state,
        fixed_gate_local_decomposition="extract split NPZ contains phi_t, phi_s, baseline, share, cancellation",
        interventions_recompute_gate_for_HZ_only=True,results=results))


def run_risk(out, calibration="val", evaluation="test", bootstrap=False):
    audit_path=out/"audit.json"
    if not audit_path.exists():raise RuntimeError("Audited frozen-checkpoint manifest missing")
    audit=json.loads(audit_path.read_text())
    if audit.get("status")!="AUDITED" or digest(audit["checkpoint"])!=audit["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint identity changed")
    for split in (calibration,evaluation):
        result_path=out/f"extract_{split}.json"
        graph_path=out/f"graph_{split}_5d.json"
        if not result_path.exists() or not graph_path.exists():
            raise RuntimeError(f"Missing completed extraction/graph provenance for {split}")
        if json.loads(result_path.read_text()).get("status")!="COMPLETE" or \
           json.loads(graph_path.read_text()).get("status")!="COMPLETE":
            raise RuntimeError(f"Incomplete extraction/graph provenance for {split}")
    c=np.load(out/f"extract_{calibration}.npz",allow_pickle=False)
    e=np.load(out/f"extract_{evaluation}.npz",allow_pickle=False)
    if c["target_end20"].max()>=e["date_last"].min():
        raise RuntimeError("Calibration 20-day labels overlap evaluation origins")
    def pack(z,split):
        graph_path=out/f"graph_features_{split}.npz"
        if not graph_path.exists():
            raise RuntimeError(f"A features absent: run graph stage on {split} first")
        with np.load(graph_path,allow_pickle=False) as graph:
            if not np.array_equal(graph["index"],np.arange(len(z["prediction"]))):
                raise RuntimeError("Graph features do not cover every origin in chronological order")
            A=graph["A"].copy()
        return dict(error5=np.abs(z["prediction"][:,1]-z["target"][:,1]).mean(1),
                    **{k:z[k] for k in ("B","R","G")},A=A)
    result=risk_evaluation(pack(c,calibration),pack(e,evaluation))
    report=dict(status="COMPLETE",calibration=calibration,evaluation=evaluation,
        threshold_source="calibration 80th percentile only",model="fixed C=1 logistic",
        exploratory=bool(evaluation in ("val","test")),
        coverage_note="Selective MAE is conditional on accepted dates, not full-model MAE",
        results=result,bootstrap="NOT_RUN")
    if bootstrap:
        # All monitors use identical sampled dates; no independent commodity draw.
        report["bootstrap"]=paired_risk_block_bootstrap(
            np.abs(e["prediction"][:,1]-e["target"][:,1]).mean(1),
            {k:v["risk_score"] for k,v in result.items()},block=40,draws=2000)
    atomic_json(out/f"risk_{calibration}_to_{evaluation}.json",report)
    return report


def repeatability(out, other_dirs, raw, mi):
    """Align seed labels on TRAIN profiles; evaluate alignment on VAL only."""
    from scipy.spatial.distance import jensenshannon
    base_train=np.load(out/"extract_train.npz",allow_pickle=False)
    base_val=np.load(out/"extract_val.npz",allow_pickle=False)
    loc=raw.index.get_indexer(base_train["date_last"])
    if (loc<19).any():raise RuntimeError("TRAIN dates absent from raw calendar")
    d=np.array([list(market_descriptors(raw.values[i-19:i+1],
                  [mi[k] for k in ("stock","bond","commodity")]).values()) for i in loc])
    results={}
    reference_audit=json.loads((out/"audit.json").read_text())
    for other_dir in other_dirs:
        other_dir=Path(other_dir)
        alt_audit=json.loads((other_dir/"audit.json").read_text())
        if alt_audit.get("seed") not in (43,44) or alt_audit.get("status")!="AUDITED":
            raise RuntimeError("Other state trajectory is not an audited seed43/44 D0B run")
        if alt_audit["train_val_data_fingerprint"]!=reference_audit["train_val_data_fingerprint"]:
            raise RuntimeError("Cross-seed data fingerprints differ")
        if digest(alt_audit["checkpoint"])!=alt_audit["checkpoint_sha256"]:
            raise RuntimeError("Other checkpoint hash changed")
        ot=np.load(other_dir/"extract_train.npz",allow_pickle=False)
        ov=np.load(other_dir/"extract_val.npz",allow_pickle=False)
        if not np.array_equal(base_train["date_last"],ot["date_last"]) or not np.array_equal(
                base_val["date_last"],ov["date_last"]):
            raise RuntimeError("Cross-seed origin dates differ")
        aligned=train_only_state_alignment(base_train["posterior"][:,-1],
                           ot["posterior"][:,-1],d)
        p=base_val["posterior"][:,-1]
        q=ov["posterior"][:,-1][:,aligned["permutation"]]
        aligned["VAL_mean_JSD"]=float(np.square(jensenshannon(p,q,axis=1)).mean())
        aligned["VAL_hard_agreement"]=float((p.argmax(1)==q.argmax(1)).mean())
        results[str(other_dir)]=aligned
    report=dict(status="COMPLETE",reference=str(out),comparisons=results,
                test_used_for_alignment=False)
    atomic_json(out/"state_repeatability.json",report)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("preflight","migrate-dates","extract","state",
                    "route","branch","graph","risk","repeatability"),default="preflight")
    ap.add_argument("--run",action="store_true")
    ap.add_argument("--split",choices=("train","val","test"),default="val")
    ap.add_argument("--checkpoint",type=Path,default=DEFAULT_CHECKPOINT)
    ap.add_argument("--reference-result",type=Path,default=REFERENCE_RESULT)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir",type=Path,default=ROOT/"experiments/d0b_reliability/frozen_seed42")
    ap.add_argument("--source-dir",type=Path,help="Completed old extraction for --stage migrate-dates")
    ap.add_argument("--calibration",choices=("train","val"),default="val")
    ap.add_argument("--evaluation",choices=("val","test"),default="test")
    ap.add_argument("--bootstrap",action="store_true")
    ap.add_argument("--case-index",type=int,action="append")
    ap.add_argument("--horizon",type=int,choices=HORIZONS,default=5)
    ap.add_argument("--other-dir",type=Path,action="append",default=[])
    args=ap.parse_args()
    if args.stage=="preflight":
        print(json.dumps(synthetic_preflight(),ensure_ascii=False,indent=2));return
    if not args.run:raise SystemExit("Real-data stages require explicit --run")
    if args.stage=="migrate-dates":
        if args.source_dir is None:raise ValueError("--source-dir is required for migrate-dates")
        print(json.dumps(migrate_date_archives(args.source_dir,args.output_dir),
                         ensure_ascii=False)[:3000]);return
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    require_current_source_audit(out)
    if args.stage=="risk":
        print(json.dumps(run_risk(out,args.calibration,args.evaluation,args.bootstrap),
                         ensure_ascii=False)[:3000]);return
    if args.stage in ("state","route","branch"):
        require_completed_extract(out,("train",args.split),args.device)
    elif args.stage=="graph":
        require_completed_extract(out,(args.split,),args.device)
    elif args.stage=="repeatability":
        require_completed_extract(out,("train","val"),args.device)
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    model,data,raw,mi,audit=load_context(args.checkpoint,device,args.seed,args.reference_result)
    if not (out/"audit.json").exists():atomic_json(out/"audit.json",audit)
    elif json.loads((out/"audit.json").read_text())["checkpoint_sha256"]!=audit["checkpoint_sha256"]:
        raise RuntimeError("Output directory belongs to a different checkpoint")
    if args.stage=="extract":result=extract(model,data,raw,mi,args.split,out,device)
    elif args.stage=="state":result=state_analysis(out,raw,mi,args.split)
    elif args.stage=="route":result=route_interventions(model,data,args.split,out,device)
    elif args.stage=="branch":result=branch_interventions(model,data,args.split,out,device)
    elif args.stage=="graph":
        result=graph_study(model,data,raw,mi,args.split,out,device,
                           args.case_index,args.horizon)
    elif args.stage=="repeatability":
        if not args.other_dir:raise ValueError("Supply --other-dir for seed43/44 artifact roots")
        result=repeatability(out,args.other_dir,raw,mi)
    else:raise AssertionError(args.stage)
    print(json.dumps(result,ensure_ascii=False,default=str)[:3000])


if __name__=="__main__":main()
