"""Six matched D0B/four-task MoE controls; default is synthetic preflight.

Only --run performs one TRAIN/VAL fit. This entry has no TEST evaluation path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.four_task_moe import (
    EXPERT_NAMES as FOUR_EXPERT_NAMES, FILM_VARIANT,
    FOUR_TASK_VARIANTS, GLOBAL_VARIANT as FOUR_GLOBAL,
    PER_HORIZON_VARIANT as FOUR_PER_HORIZON,
)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.task_conditioned_moe import (
    BASE_VARIANT, GLOBAL_VARIANT as THREE_GLOBAL,
    PER_HORIZON_VARIANT as THREE_PER_HORIZON,
    EXPERT_NAMES as THREE_EXPERT_NAMES,
)
from cmgm.scripts.d0b_task_conditioned_moe import (
    PROTOCOL, check_protocol, data_audit, full_loader,
    initialization_audit, model_kwargs, sha256, write_json,
)
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT=ROOT/"experiments"/"d0b_four_task_moe"
CHECKPOINT_ROOT=ROOT/"checkpoints"/"d0b_four_task_moe"
VARIANTS=(BASE_VARIANT,THREE_GLOBAL,THREE_PER_HORIZON,
          FOUR_GLOBAL,FOUR_PER_HORIZON,FILM_VARIANT)
VAL_GATE={
    "primary_metric":"VAL MAE@5d (decimal return)",
    "candidate":FOUR_PER_HORIZON,
    "strictly_lower_than":[BASE_VARIANT,FOUR_GLOBAL],
    "pass_condition":"candidate VAL MAE@5d strictly lower than both references",
    "secondary_horizons":"report 1d,10d,20d; no additional screening threshold",
    "test_policy":"No TEST in this entry; any later test comparison is exploratory due to previous project TEST exposure",
}
SOURCE_FILES=(
    "cmgm/config.py", "cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/switching_latent_transformer.py",
    "cmgm/models/task_conditioned_moe.py", "cmgm/models/four_task_moe.py",
    "cmgm/models/horizon_tensor_fusion.py", "cmgm/training/train.py",
    "cmgm/training/metric_standard.py", "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/d0b_task_conditioned_moe.py",
    "cmgm/scripts/d0b_four_task_moe.py",
)


def parameters(model):
    count=lambda module:sum(p.numel() for p in module.parameters())
    result={"total":count(model),
            "trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if model.variant in FOUR_TASK_VARIANTS:
        module=model.four_task_moe_fusion
        result["experts"]={
            "recent_change":count(module.recent_expert),
            "long_memory":count(module.long_expert),
            "spatial_structure":count(module.spatial_expert),
            "low_rank_interaction":count(module.interaction_expert),
        }
        result["router"] = module.logits.numel()
        result["film"] = (module.film_scale_delta.numel()+module.film_shift.numel()
                          if model.variant==FILM_VARIANT else 0)
    elif model.variant in (THREE_GLOBAL,THREE_PER_HORIZON):
        module=model.task_moe_fusion
        result["experts"]={name:count(getattr(module,attribute)) for name,attribute in (
            ("temporal","temporal_expert"),("spatial","spatial_expert"),
            ("interaction","interaction_expert"))}
        result["router"]=module.logits.numel()
        result["film"]=0
    return result


def route_matrix(model):
    if model.variant in FOUR_TASK_VARIANTS:
        return (model.four_task_moe_fusion.routing_weights().detach().cpu().numpy(),
                FOUR_EXPERT_NAMES)
    if model.variant in (THREE_GLOBAL,THREE_PER_HORIZON):
        # Static matched controls: do not infer routing from any sample.
        return (torch.softmax(model.task_moe_fusion.logits,dim=-1)
                .expand(4,-1).detach().cpu().numpy(),THREE_EXPERT_NAMES)
    return None,None


def route_statistics(model):
    weights,names=route_matrix(model)
    if weights is None:
        return None
    eps=1e-12
    if weights.shape!=(4,len(names)) or not np.isfinite(weights).all():
        raise ValueError("Invalid horizon routing matrix")
    js={}
    for i,j in combinations(range(4),2):
        p,q=weights[i].astype(np.float64),weights[j].astype(np.float64)
        mean=.5*(p+q)
        js[f"{config.MULTI_HORIZONS[i]}d_vs_{config.MULTI_HORIZONS[j]}d"]=float(
            .5*np.sum(p*np.log((p+eps)/(mean+eps)))+
            .5*np.sum(q*np.log((q+eps)/(mean+eps))))
    return {
        "expert_order":list(names),"horizon_order":list(config.MULTI_HORIZONS),
        "weights_by_horizon":weights.tolist(),
        "entropy_nats_by_horizon":(-np.sum(weights*np.log(weights+eps),axis=-1)).tolist(),
        "pairwise_js_nats":js,"sample_dependent":False,
    }


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval()
    predictions,targets=[],[]
    for batch in full_loader(loader):
        x,y=batch[:2]
        prediction=model(x.to(device))
        if prediction.shape!=y.shape or not torch.isfinite(prediction).all():
            raise ValueError("Invalid prediction shape or nonfinite values")
        predictions.append(prediction.cpu().numpy())
        targets.append(y.numpy())
    p,y=np.concatenate(predictions),np.concatenate(targets)
    return {str(h):population_metrics(p[:,i],y[:,i])
            for i,h in enumerate(config.MULTI_HORIZONS)}


def synthetic_preflight():
    kwargs=dict(num_nodes=30,n_commodities=24,n_stock=4,n_bond=2,commodity_indices=(6,30))
    x=torch.randn(2,20,30,21)
    for variant in VARIANTS:
        set_seed(42)
        model=HeteroMixHopCMGM(variant=variant,**kwargs).eval()
        with torch.inference_mode():
            prediction=model(x)
        if prediction.shape!=(2,4,24) or not torch.isfinite(prediction).all():
            raise AssertionError(f"Synthetic forward failed: {variant}")
        if not initialization_audit(kwargs,variant,42)["PASS"]:
            raise AssertionError(f"Shared D0B initialization differs: {variant}")
    print("Four-task synthetic preflight PASS: six variants, full outputs, shared D0B initialization.")
    print("No market dataset or TEST result accessed; no formal training.")


def matched_initialization(kwargs,seed):
    # The four-expert global/per-task/FiLM variants must share every expert
    # tensor at construction. This audit is independent of the fitted model.
    states={}
    for variant in FOUR_TASK_VARIANTS:
        set_seed(seed)
        states[variant]={name:value.clone() for name,value in
            HeteroMixHopCMGM(variant=variant,**kwargs).four_task_moe_fusion.state_dict().items()
            if name.startswith(("recent_expert.","long_expert.",
                                "spatial_expert.","interaction_expert."))}
    reference=states[FOUR_GLOBAL]
    mismatches={variant:[key for key,value in reference.items()
                         if key not in state or not torch.equal(value,state[key])]
                for variant,state in states.items()}
    return {"expert_tensor_count":len(reference),"mismatches":mismatches,
            "PASS":all(not names for names in mismatches.values())}


def run(args):
    check_protocol()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    set_seed(args.seed)
    # Reuse the frozen D0B split/feature/target factory. Only TRAIN/VAL
    # datasets are iterated; no TEST metric or prediction is computed.
    data=build_data(SimpleNamespace(batch_size=64,seq_len=20,seed=args.seed))
    kwargs=model_kwargs(data)
    fingerprint=data_audit(data)
    shared=initialization_audit(kwargs,args.variant,args.seed)
    matched=matched_initialization(kwargs,args.seed)
    if not shared["PASS"] or not matched["PASS"]:
        raise RuntimeError(f"Initialization audit failed: shared={shared}; matched={matched}")
    set_seed(args.seed)
    model=HeteroMixHopCMGM(variant=args.variant,**kwargs).to(args.device)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}"
    name=f"{args.variant}_seed{args.seed}_{stamp}"
    output=EXPERIMENT_ROOT/name
    checkpoint=CHECKPOINT_ROOT/name/"best.pt"
    output.mkdir(parents=True,exist_ok=False)
    checkpoint.parent.mkdir(parents=True,exist_ok=False)
    metadata={
        "variant":args.variant,"seed":args.seed,"model_kwargs":kwargs,
        "protocol":PROTOCOL,"val_gate":VAL_GATE,"data_audit":fingerprint,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        "source_sha256":{name:sha256(ROOT/name) for name in SOURCE_FILES},
        "parameters":parameters(model),"shared_initialization_audit":shared,
        "four_expert_initialization_audit":matched,
        "target_scale":"decimal return", "hit_storage":"fraction",
        "metric_population":"all origins x all commodities; zero targets included",
        "test_evaluated":False,
    }
    for filename,content in (("config.json",metadata),("data_audit.json",fingerprint),
                             ("initialization_audit.json",{"shared":shared,"matched":matched})):
        write_json(output/filename,content)
    started=time.perf_counter()
    history=train(model,data["loaders"]["train"],data["loaders"]["val"],
                  torch.empty(2,0,dtype=torch.long),torch.empty(0),torch.device(args.device),
                  num_epochs=200,lr=1e-4,weight_decay=1e-5,patience=10,
                  checkpoint_path=str(checkpoint),checkpoint_metadata=metadata)
    train_seconds=time.perf_counter()-started
    payload=torch.load(checkpoint,map_location=args.device,weights_only=False)
    if (payload.get("metadata",{}).get("variant")!=args.variant
            or payload.get("best_epoch")!=history["best_epoch"]):
        raise RuntimeError("Saved checkpoint variant/best-epoch mismatch")
    model.load_state_dict(payload["model_state_dict"],strict=True)
    model.eval()
    device=torch.device(args.device)
    metrics={split:evaluate(model,data["loaders"][split],device) for split in ("train","val")}
    routing=route_statistics(model)
    swaps={}
    if args.variant in (THREE_PER_HORIZON,FOUR_PER_HORIZON,FILM_VARIANT):
        fusion=(model.four_task_moe_fusion if args.variant in FOUR_TASK_VARIANTS
                else model.task_moe_fusion)
        for i,j in combinations(range(4),2):
            with fusion.swap_tasks(i,j):
                alternative=evaluate(model,data["loaders"]["val"],device)
            swaps[f"{config.MULTI_HORIZONS[i]}d_vs_{config.MULTI_HORIZONS[j]}d"]={
                "metrics":alternative,
                "delta_mae_vs_original":{str(h):alternative[str(h)]["MAE"]-metrics["val"][str(h)]["MAE"]
                                         for h in config.MULTI_HORIZONS},
            }
    result={
        "status":"COMPLETE","variant":args.variant,"seed":args.seed,
        "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256(checkpoint),
        "best_epoch":int(payload["best_epoch"]),
        "best_val_prediction_huber_batch_mean":float(payload["best_val_loss"]),
        "train_seconds":train_seconds,"parameters":metadata["parameters"],
        "metrics":metrics,"routing":routing,"val_frozen_route_swaps":swaps,
        "test_evaluated":False,
    }
    for filename,content in (("training_history.json",history),("metrics.json",metrics),
                             ("routing_diagnostics.json",routing),
                             ("val_route_swaps.json",swaps),("result.json",result)):
        write_json(output/filename,content)
    print(f"COMPLETE {args.variant} seed={args.seed}; VAL MAE@5d={metrics['val']['5']['MAE']:.9g}")
    print(f"Result: {output}\nCheckpoint: {checkpoint}")


def compare_results(paths,output):
    if len(paths)!=len(VARIANTS):
        raise ValueError("Provide exactly six seed-42 result.json files")
    results,configs={},{}
    for path in paths:
        item=json.loads(path.read_text(encoding="utf-8"))
        variant=item["variant"]
        if (variant not in VARIANTS or variant in results or item["seed"]!=42
                or item["status"]!="COMPLETE" or item["test_evaluated"]):
            raise ValueError("Invalid or duplicate seed-42 TRAIN/VAL result")
        if sha256(Path(item["checkpoint"]))!=item["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {variant}")
        results[variant]=item
        configs[variant]=json.loads((path.parent/"config.json").read_text(encoding="utf-8"))
    if set(results)!=set(VARIANTS):
        raise ValueError(f"Missing variants: {set(VARIANTS)-set(results)}")
    reference=configs[BASE_VARIANT]
    for variant,record in configs.items():
        for field in ("protocol","data_audit","git_commit","source_sha256","seed"):
            if record[field]!=reference[field]:
                raise ValueError(f"Incomparable {field}: {variant}")
    mae=lambda variant,h:float(results[variant]["metrics"]["val"][str(h)]["MAE"])
    passed=mae(FOUR_PER_HORIZON,5)<mae(BASE_VARIANT,5) and mae(FOUR_PER_HORIZON,5)<mae(FOUR_GLOBAL,5)
    summary={"status":"COMPLETE","seed":42,"val_gate":VAL_GATE,"pass":bool(passed),
             "test_evaluated":False,"rows":[{
                 "variant":variant,"val_mae":{str(h):mae(variant,h) for h in config.MULTI_HORIZONS},
                 "best_val_prediction_huber_batch_mean":results[variant]["best_val_prediction_huber_batch_mean"],
                 "best_epoch":results[variant]["best_epoch"],
                 "parameters":results[variant]["parameters"],
                 "train_seconds":results[variant]["train_seconds"],
             } for variant in VARIANTS]}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    write_json(output,summary)
    print(f"VAL gate pass={passed}; comparison={output}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="explicitly fit one TRAIN/VAL model")
    parser.add_argument("--variant",choices=VARIANTS,default=FOUR_PER_HORIZON)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--compare-results",nargs=6,type=Path,metavar="RESULT_JSON")
    parser.add_argument("--comparison-output",type=Path,
                        default=EXPERIMENT_ROOT/"seed42_val_comparison.json")
    args=parser.parse_args()
    if args.compare_results:
        if args.run:
            parser.error("Comparison cannot be combined with --run")
        compare_results(args.compare_results,args.comparison_output)
    elif args.run:
        run(args)
    else:
        synthetic_preflight()


if __name__=="__main__":
    main()
