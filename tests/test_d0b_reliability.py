"""Small, synthetic-only audits for frozen D0B interpretation."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cmgm.interpretability.d0b_reliability import (
    D0BTrace, fixed_gate_shapley, normalize_masked_attention,
    paired_moving_block_bootstrap, risk_evaluation, source_intervention,
    train_only_state_alignment)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.data.data_loader import MarketSequenceDataset
from cmgm.scripts.d0b_reliability import (
    synthetic_preflight,extract,require_completed_extract,
    require_current_source_audit,write_safe_extract_archive)


def test_end_to_end_synthetic_preflight():
    assert synthetic_preflight()["PASS"]


def test_fixed_gate_train_mean_reference_formula():
    torch.manual_seed(7)
    head=torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.Tanh(),
                             torch.nn.Linear(6,3))
    a,b=torch.randn(5,4),torch.randn(5,4)
    a0,b0=torch.randn(1,4),torch.randn(1,4)
    result=fixed_gate_shapley(head,a,b,a_reference=a0,b_reference=b0)
    expected=.5*((head(a+b0)-head(a0+b0))+(head(a+b)-head(a0+b)))
    torch.testing.assert_close(result["phi_temporal"],expected)
    torch.testing.assert_close(result["phi_temporal"]+result["phi_spatial"],
                               head(a+b)-head(a0+b0),atol=1e-6,rtol=1e-6)


def test_receiver_source_direction_and_all_mask_boundary():
    a=torch.zeros(1,4,4,1)
    a[0,0,2,0]=1.0  # receiver 0 reads source 2
    a[0,1,2,0]=1.0
    a[0,2,2,0]=1.0
    a[0,3,2,0]=1.0
    assert a.mean((1,3)).argmax().item()==2
    removed=normalize_masked_attention(a,torch.tensor([0,0,1,0],dtype=torch.bool))
    assert torch.isfinite(removed).all()
    torch.testing.assert_close(removed.sum(2),torch.ones(1,4,1))
    all_removed=normalize_masked_attention(torch.ones_like(a)/4,
                     torch.ones(4,dtype=torch.bool))
    torch.testing.assert_close(all_removed.sum(2),torch.ones(1,4,1))
    torch.testing.assert_close(all_removed[0,:,:,0],torch.eye(4))
    batched=torch.cat((a,torch.ones_like(a)/4),dim=0)
    per_sample=normalize_masked_attention(batched,torch.tensor(
        [[0,0,1,0],[1,1,1,1]],dtype=torch.bool))
    torch.testing.assert_close(per_sample.sum(2),torch.ones(2,4,1))
    torch.testing.assert_close(per_sample[1,:,:,0],torch.eye(4))


def test_graph_intervention_changes_prediction_and_restores_path():
    torch.manual_seed(42)
    model=HeteroMixHopCMGM(num_nodes=30,n_stock=4,n_bond=2,
          n_commodities=24,commodity_indices=(6,30),
          variant="switching_latent_balanced_readout").eval()
    model.requires_grad_(False)
    x=torch.randn(2,20,30,21)
    with torch.no_grad():
        original=model(x)
        with source_intervention(model,torch.tensor([True]*6+[False]*24)):
            changed=model(x)
        restored=model(x)
    assert (changed-original).abs().max()>0
    torch.testing.assert_close(restored,original,rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_train_only_alignment_and_date_group_bootstrap():
    p=np.tile(np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]),(20,1))
    d=np.tile(np.eye(3),(20,1))
    q=p[:,[2,0,1]]
    aligned=train_only_state_alignment(p,q,d)
    assert aligned["permutation"]==[1,2,0]
    x=np.repeat(np.arange(80)[:,None],24,axis=1)
    x[:,1:]+=1000
    ci=paired_moving_block_bootstrap(x,block=40,draws=50,seed=4)
    assert abs((ci["low"][1]-ci["low"][0])-1000)<1e-9
    assert abs((ci["high"][1]-ci["high"][0])-1000)<1e-9


def test_risk_threshold_comes_from_calibration_only():
    rng=np.random.default_rng(8)
    cal=dict(error5=np.r_[np.zeros(20),np.ones(20)])
    ev=dict(error5=np.r_[np.zeros(20),np.ones(20)])
    for name in ("B","R","G","A"):
        cal[name]=rng.normal(size=(40,2))
        ev[name]=rng.normal(size=(40,2))
    first=risk_evaluation(cal,ev)["B"]
    ev["error5"]=ev["error5"]*100
    second=risk_evaluation(cal,ev)["B"]
    assert first["threshold"]==second["threshold"]
    assert second["MAE_at_80pct_coverage"]==100*first["MAE_at_80pct_coverage"]


def test_synthetic_split_extraction_artifact(tmp_path):
    torch.manual_seed(1)
    rng=np.random.default_rng(1)
    raw=100*np.exp(rng.normal(0,.01,size=(600,30)).cumsum(0))
    feature=rng.normal(size=(600,30,21)).astype(np.float32)
    names=[f"asset_{i}" for i in range(30)]
    frame=pd.DataFrame(raw,index=pd.date_range("2025-01-01",periods=600),columns=names)
    mi={"stock":(0,4),"bond":(4,6),"commodity":(6,30)}
    datasets={}
    for name,(a,b) in {"train":(0,420),"val":(420,510),"test":(510,600)}.items():
        datasets[name]=MarketSequenceDataset(raw[a:b],mi,20,
            feature_matrix=feature[a:b],raw_prices=raw[a:b],target_type="return",
            horizons=[1,5,10,20])
    data={"feature_names":names,"loaders":{k:DataLoader(v,batch_size=32)
                                           for k,v in datasets.items()}}
    model=HeteroMixHopCMGM(num_nodes=30,n_stock=4,n_bond=2,n_commodities=24,
        commodity_indices=(6,30),variant="switching_latent_balanced_readout").eval()
    model.requires_grad_(False)
    report=extract(model,data,frame,mi,"val",tmp_path,torch.device("cpu"))
    assert report["n_origins"]==51
    with np.load(tmp_path/"extract_val.npz",allow_pickle=False) as z:
        assert z["prediction"].shape==(51,4,24)
        assert z["posterior"].shape==(51,20,3)
        assert z["source_attention"].shape==(51,30)
        assert z["source_attention_by_layer_hop_head"].shape==(
            51,2,2,model.attn_mixhop1.n_heads,30)
        assert z["B"].shape==(51,2) and z["G"].shape==(51,2)
        assert z["date_last"].dtype.kind=="U"
        assert z["target_end20"].dtype.kind=="U"
        np.testing.assert_allclose(z["R"][32,1],np.abs(
            z["posterior"][32,-1]-z["posterior"][31,-1]).sum(),atol=1e-6)
        assert np.isfinite(z["prediction"]).all()


def test_date_archive_repair_keeps_numeric_arrays_exact_without_pickle(tmp_path):
    source=tmp_path/"old.npz"
    safe=tmp_path/"safe.npz"
    original=np.arange(2*4*24,dtype=np.float32).reshape(2,4,24)
    np.savez_compressed(source,prediction=original,
        date_last=np.array(["2025-01-01","2025-01-02"],dtype=object),
        target_end20=np.array(["2025-01-29","2025-01-30"],dtype=object))
    summary=write_safe_extract_archive(source,safe,
        ["2025-01-01","2025-01-02"],["2025-01-29","2025-01-30"])
    assert summary["other_arrays_exact"]
    with np.load(safe,allow_pickle=False) as z:
        np.testing.assert_array_equal(z["prediction"],original)
        assert z["date_last"].dtype.kind=="U"
        assert z["target_end20"].dtype.kind=="U"


def test_missing_extraction_reports_exact_prerequisite(tmp_path):
    try:
        require_completed_extract(tmp_path,("train","val"),"cuda:0")
    except RuntimeError as exc:
        assert "--stage extract --split train --run --device cuda:0" in str(exc)
    else:
        raise AssertionError("Missing TRAIN extraction must stop downstream stages")


def test_old_source_audit_requires_new_output_directory(tmp_path):
    (tmp_path/"audit.json").write_text('{"source_hashes": {}}')
    try:
        require_current_source_audit(tmp_path)
    except RuntimeError as exc:
        assert "NEW --output-dir" in str(exc)
    else:
        raise AssertionError("Source mismatch must not mix experiment artifacts")
