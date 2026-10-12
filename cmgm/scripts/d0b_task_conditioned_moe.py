"""Matched D0B / three-expert routing study. Default: synthetic preflight only.

Only explicit --run fits one seed/variant. This entry never evaluates TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.task_conditioned_moe import (
    ALL_VARIANTS, BASE_VARIANT, EXPERT_NAMES, MOE_VARIANTS,
    SHORT_LONG_VARIANT,
)
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / "d0b_task_conditioned_moe"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "d0b_task_conditioned_moe"
SOURCE_FILES = (
    "cmgm/config.py", "cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/task_conditioned_moe.py", "cmgm/models/horizon_tensor_fusion.py",
    "cmgm/models/switching_latent_transformer.py", "cmgm/training/train.py",
    "cmgm/training/metric_standard.py", "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/d0b_task_conditioned_moe.py",
)
PROTOCOL = {
    "seq_len": 20, "feature_dim": 21, "target_type": "return",
    "horizons": [1, 5, 10, 20], "batch_size": 64,
    "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-5,
    "max_epochs": 200, "early_stopping_patience": 10,
    "scheduler": "ReduceLROnPlateau(min,factor=0.5,patience=5)",
    "prediction_loss": "sum_h Huber(delta=0.02)",
    "native_switch_kl": "beta_max=5e-4,warmup=20 epochs",
    "checkpoint_selection": "prediction-only four-horizon VAL Huber, batch-mean",
    "train_shuffle": False, "train_drop_last": True,
    "full_eval_shuffle": False, "full_eval_drop_last": False,
    "moe_warmup": False, "moe_auxiliary_loss": False,
}
VAL_GATE = {
    "primary": "VAL MAE@5d (decimal return)",
    "strictly_lower_than": [BASE_VARIANT, "switching_latent_balanced_task_moe_global"],
    "secondary_reference": BASE_VARIANT,
    "simultaneous_10d_20d_degradation_threshold_relative": 0.02,
    "pass_condition": "5d strictly lower than both references AND NOT (10d and 20d each more than 2% higher than D0B)",
    "test_policy": "No TEST evaluation in this entry; any later TEST is exploratory because the project has viewed TEST previously.",
}


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_kwargs(data):
    c0, c1 = data["market_indices"]["commodity"]
    s0, s1 = data["market_indices"]["stock"]
    b0, b1 = data["market_indices"]["bond"]
    return dict(num_nodes=int(data["n_nodes"]), n_commodities=int(data["n_commodities"]),
                n_stock=int(s1-s0), n_bond=int(b1-b0), commodity_indices=(int(c0),int(c1)))


def data_audit(data):
    names = [str(n) for n in data["feature_names"]]
    c0, c1 = data["market_indices"]["commodity"]
    splits = {}
    for split in ("train", "val"):
        dataset = data["loaders"][split].dataset
        digest = hashlib.sha256()
        for field in ("feature_matrix", "prices", "raw_prices"):
            value = getattr(dataset, field, None)
            if value is not None:
                array = np.ascontiguousarray(value)
                digest.update(field.encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
        splits[split] = dict(origins=len(dataset), sha256=digest.hexdigest(),
                             seq_len=int(dataset.seq_len), horizons=list(dataset.horizons),
                             target_type=dataset.target_type)
    return {
        "splits": splits, "node_order": names,
        "commodity_order": names[c0:c1], "commodity_indices": [int(c0),int(c1)],
        "market_indices": {k:list(map(int,v)) for k,v in data["market_indices"].items()},
        "input_shape": ["B",20,int(data["n_nodes"]),21],
        "output_shape": ["B",4,int(data["n_commodities"])],
        "test_metrics_accessed": False,
    }


def initialization_audit(kwargs, variant, seed):
    # Isolate audit constructors from the RNG stream used by the actual fit.
    cpu, py, np_state = torch.random.get_rng_state(), random.getstate(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_seed(seed)
        reference = HeteroMixHopCMGM(variant=BASE_VARIANT, **kwargs)
        set_seed(seed)
        candidate = HeteroMixHopCMGM(variant=variant, **kwargs)
        before, after = reference.state_dict(), candidate.state_dict()
        shared = [key for key in before if not key.startswith("gate_fc.")]
        differences = {key:(None if key not in after else float((before[key]-after[key]).abs().max()))
                       for key in shared if key not in after or not torch.equal(before[key],after[key])}
        return {"shared_tensor_count":len(shared),
                "shared_parameter_count":sum(before[key].numel() for key in shared),
                "mismatch_count":len(differences), "different":differences,
                "shared_max_abs_diff":max((value for value in differences.values() if value is not None),default=0.0),
                "PASS":not differences}
    finally:
        torch.random.set_rng_state(cpu)
        random.setstate(py)
        np.random.set_state(np_state)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def full_loader(loader):
    return DataLoader(loader.dataset, batch_size=config.BATCH_SIZE,
                      shuffle=False, drop_last=False, num_workers=0)


def route_summary(routes):
    if routes is None:
        return None
    route = np.concatenate(routes,axis=0).astype(np.float64)
    if route.ndim != 3 or route.shape[1:] != (4,3):
        raise ValueError("Unexpected routing shape")
    eps = 1e-12
    short, long = route[:,:2].mean((0,1)),route[:,2:].mean((0,1))
    middle = .5*(short+long)
    kl = lambda p,q: float(np.sum(p*np.log((p+eps)/(q+eps))))
    return {
        "expert_order": list(EXPERT_NAMES), "horizon_order": list(config.MULTI_HORIZONS),
        "mean_by_horizon": route.mean(0).tolist(),
        "mean_short": short.tolist(), "mean_long":long.tolist(),
        "entropy_by_horizon":(-np.sum(route*np.log(route+eps),axis=-1).mean(0)).tolist(),
        "short_long_js_nats":.5*(kl(short,middle)+kl(long,middle)),
        "sample_std_by_horizon":route.std(axis=0).tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets, routes = [],[],[]
    for batch in full_loader(loader):
        x,y = batch[:2]
        prediction = model(x.to(device))
        if tuple(prediction.shape) != tuple(y.shape) or not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite or misaligned prediction")
        predictions.append(prediction.cpu().numpy())
        targets.append(y.numpy())
        if model.variant in MOE_VARIANTS:
            routes.append(model.task_moe_fusion.last_routing.cpu().numpy())
    p,y = np.concatenate(predictions),np.concatenate(targets)
    metrics = {str(h):population_metrics(p[:,i],y[:,i])
               for i,h in enumerate(config.MULTI_HORIZONS)}
    return metrics, route_summary(routes if routes else None)


def synthetic_preflight():
    kwargs = dict(num_nodes=30,n_commodities=24,n_stock=4,n_bond=2,
                  commodity_indices=(6,30))
    x = torch.randn(2,20,30,21)
    outputs = {}
    for variant in ALL_VARIANTS:
        torch.manual_seed(42)
        model = HeteroMixHopCMGM(variant=variant,**kwargs).eval()
        with torch.inference_mode():
            output = model(x)
        assert tuple(output.shape)==(2,4,24) and torch.isfinite(output).all()
        outputs[variant]=output
        assert initialization_audit(kwargs,variant,42)["PASS"]
    print("Synthetic preflight PASS: all five variants, shapes, finite values, shared initialization.")
    print("No dataset loaded, no formal training, no TEST evaluation.")


def check_protocol():
    actual=(config.SEQ_LEN,config.FEATURE_DIM,tuple(config.MULTI_HORIZONS),
            config.TARGET_TYPE,config.LOSS_TYPE,config.HUBER_DELTA,
            config.BATCH_SIZE,config.LEARNING_RATE,config.WEIGHT_DECAY,
            config.NUM_EPOCHS,config.PATIENCE)
    expected=(20,21,(1,5,10,20),"return","huber",.02,64,1e-4,1e-5,200,10)
    if actual != expected:
        raise ValueError(f"D0B protocol changed: {actual} != {expected}")


def compare_results(paths, output):
    """Locked seed-42 VAL decision only; never reads TEST or selects an epoch."""
    if len(paths) != len(ALL_VARIANTS):
        raise ValueError("Provide exactly five seed-42 result.json paths")
    by_variant = {}
    configs = {}
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        config_record = json.loads((path.parent/"config.json").read_text(encoding="utf-8"))
        variant = item["variant"]
        if variant not in ALL_VARIANTS or variant in by_variant or item["seed"] != 42:
            raise ValueError("Invalid or duplicate variant / seed in comparison")
        if item["status"] != "COMPLETE" or item["test_evaluated"]:
            raise ValueError("Expected complete TRAIN/VAL-only result")
        if sha256(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {variant}")
        by_variant[variant],configs[variant]=item,config_record
    if set(by_variant) != set(ALL_VARIANTS):
        raise ValueError(f"Missing variants: {set(ALL_VARIANTS)-set(by_variant)}")
    control = configs[BASE_VARIANT]
    for variant,record in configs.items():
        for key in ("protocol","data_audit","git_commit","source_sha256"):
            if record[key] != control[key]:
                raise ValueError(f"Incomparable {key}: {variant}")
    mae = lambda v,h:float(by_variant[v]["metrics"]["val"][str(h)]["MAE"])
    candidate = SHORT_LONG_VARIANT
    primary = mae(candidate,5) < mae(BASE_VARIANT,5) and mae(candidate,5) < mae("switching_latent_balanced_task_moe_global",5)
    degradation = {str(h):(mae(candidate,h)-mae(BASE_VARIANT,h))/mae(BASE_VARIANT,h)
                   for h in (10,20)}
    both_bad = all(value > VAL_GATE["simultaneous_10d_20d_degradation_threshold_relative"]
                   for value in degradation.values())
    decision = {
        "status":"COMPLETE", "seed":42, "test_evaluated":False,
        "val_gate":VAL_GATE, "pass":bool(primary and not both_bad),
        "primary_5d_pass":bool(primary), "relative_degradation_vs_d0b":degradation,
        "both_long_horizons_over_threshold":bool(both_bad),
        "rows":[{
            "variant":variant,
            "val_mae":{str(h):mae(variant,h) for h in (1,5,10,20)},
            "val_prediction_huber_batch_mean":by_variant[variant]["best_val_prediction_huber_batch_mean"],
            "best_epoch":by_variant[variant]["best_epoch"],
            "parameters_total":by_variant[variant]["parameters_total"],
            "result":str(next(path for path in paths if json.loads(path.read_text())["variant"]==variant)),
        } for variant in ALL_VARIANTS],
        "interpretation":"Seed-42 VAL screening only; no TEST or statistical claim",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    write_json(output,decision)
    print(f"VAL gate pass={decision['pass']}; comparison={output}")


def run(args):
    check_protocol()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    # Existing factory constructs chronological splits; only TRAIN/VAL loaders
    # are iterated or fingerprinted. TEST labels and results are never evaluated.
    data=build_data(SimpleNamespace(batch_size=64,seq_len=20,seed=args.seed))
    audit=data_audit(data)
    kwargs=model_kwargs(data)
    init=initialization_audit(kwargs,args.variant,args.seed)
    if not init["PASS"]:
        raise RuntimeError(f"Shared initialization mismatch: {init}")
    set_seed(args.seed)
    model=HeteroMixHopCMGM(variant=args.variant,**kwargs).to(args.device)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}"
    run_name=f"{args.variant}_seed{args.seed}_{stamp}"
    output=EXPERIMENT_ROOT/run_name
    checkpoint=CHECKPOINT_ROOT/run_name/"best.pt"
    output.mkdir(parents=True,exist_ok=False)
    checkpoint.parent.mkdir(parents=True,exist_ok=False)
    metadata={
        "variant":args.variant,"seed":args.seed,"model_kwargs":kwargs,
        "protocol":PROTOCOL,"val_gate":VAL_GATE,"data_audit":audit,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        "source_sha256":{name:sha256(ROOT/name) for name in SOURCE_FILES},
        "parameters_total":sum(p.numel() for p in model.parameters()),
        "parameters_trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "experts_parameters":sum(p.numel() for p in model.task_moe_fusion.temporal_expert.parameters())+
            sum(p.numel() for p in model.task_moe_fusion.spatial_expert.parameters())+
            sum(p.numel() for p in model.task_moe_fusion.interaction_expert.parameters())
            if args.variant in MOE_VARIANTS else 0,
        "initialization_audit":init,
        "target_scale":"decimal return", "hit_storage":"fraction",
        "metric_population":"all origins x all 24 commodities; no zero mask",
        "test_evaluated":False,
    }
    write_json(output/"config.json",metadata)
    write_json(output/"data_audit.json",audit)
    write_json(output/"initialization_audit.json",init)
    started=time.perf_counter()
    history=train(model,data["loaders"]["train"],data["loaders"]["val"],
                  torch.empty(2,0,dtype=torch.long),torch.empty(0),torch.device(args.device),
                  num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,
                  checkpoint_path=str(checkpoint),checkpoint_metadata=metadata)
    seconds=time.perf_counter()-started
    payload=torch.load(checkpoint,map_location=args.device,weights_only=False)
    if payload.get("metadata",{}).get("variant") != args.variant or payload["best_epoch"] != history["best_epoch"]:
        raise RuntimeError("Checkpoint identity/best-epoch mismatch")
    model.load_state_dict(payload["model_state_dict"],strict=True)
    metrics,routing={},{}
    for split in ("train","val"):
        metrics[split],routing[split]=evaluate(model,data["loaders"][split],torch.device(args.device))
    interventions={}
    if args.variant == SHORT_LONG_VARIANT:
        fusion=model.task_moe_fusion
        for name in ("swap_short_long",*EXPERT_NAMES):
            with fusion.intervention(name):
                values,_=evaluate(model,data["loaders"]["val"],torch.device(args.device))
            interventions[name]=values
    result={
        "status":"COMPLETE","variant":args.variant,"seed":args.seed,
        "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256(checkpoint),
        "best_epoch":int(payload["best_epoch"]),
        "best_val_prediction_huber_batch_mean":float(payload["best_val_loss"]),
        "training_seconds":seconds,"metrics":metrics,"routing":routing,
        "val_frozen_interventions":interventions,"test_evaluated":False,
        "parameters_total":metadata["parameters_total"],
        "parameters_trainable":metadata["parameters_trainable"],
    }
    write_json(output/"training_history.json",history)
    write_json(output/"metrics.json",metrics)
    write_json(output/"routing_diagnostics.json",routing)
    write_json(output/"val_frozen_interventions.json",interventions)
    write_json(output/"result.json",result)
    print(f"COMPLETE {args.variant} seed={args.seed}; VAL MAE@5d={metrics['val']['5']['MAE']:.9g}")
    print(f"Result: {output}\nCheckpoint: {checkpoint}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="explicitly fit one TRAIN/VAL run")
    parser.add_argument("--variant",choices=ALL_VARIANTS,default=SHORT_LONG_VARIANT)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--compare-results",nargs=5,type=Path,metavar="RESULT_JSON",
                        help="five frozen seed-42 result.json files; VAL-only decision")
    parser.add_argument("--comparison-output",type=Path,
                        default=EXPERIMENT_ROOT/"seed42_val_comparison.json")
    args=parser.parse_args()
    if args.compare_results:
        if args.run:
            parser.error("--compare-results cannot be combined with --run")
        compare_results(args.compare_results,args.comparison_output)
    elif args.run:
        run(args)
    else:
        synthetic_preflight()


if __name__=="__main__":
    main()
