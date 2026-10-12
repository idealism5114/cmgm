"""Matched short/long task-mask MoE study. Default: synthetic preflight only.

Explicit --run fits one TRAIN/VAL configuration. There is no TEST path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.short_long_mask_moe import (
    EXPERT_NAMES, FIXED_VARIANT, GLOBAL_VARIANT, LEARNED_VARIANT,
    MASK_MOE_VARIANTS,
)
from cmgm.models.task_conditioned_moe import BASE_VARIANT, GLOBAL_VARIANT as THREE_GLOBAL
from cmgm.scripts.d0b_task_conditioned_moe import (
    PROTOCOL, check_protocol, data_audit, full_loader, initialization_audit,
    model_kwargs, sha256, write_json,
)
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / "d0b_short_long_mask_moe"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "d0b_short_long_mask_moe"
VARIANTS = (BASE_VARIANT, THREE_GLOBAL, *MASK_MOE_VARIANTS)
SOURCE_FILES = (
    "cmgm/config.py", "cmgm/models/model.py", "cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/switching_latent_transformer.py",
    "cmgm/models/task_conditioned_moe.py", "cmgm/models/short_long_mask_moe.py",
    "cmgm/models/horizon_tensor_fusion.py", "cmgm/training/train.py",
    "cmgm/training/metric_standard.py", "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/d0b_task_conditioned_moe.py",
    "cmgm/scripts/d0b_short_long_mask_moe.py",
)
VAL_RULE = {
    "primary": "VAL MAE@5d, decimal return",
    "checkpoint_selection": PROTOCOL["checkpoint_selection"],
    "comparisons": [BASE_VARIANT, THREE_GLOBAL, GLOBAL_VARIANT,
                    FIXED_VARIANT, LEARNED_VARIANT],
    "interpretation": (
        "If learned mask beats D0B but not five-expert global, attribute the "
        "gain to expert design, not task routing. If it does not beat the "
        "matched three-expert global control, report that as well."
    ),
    "test_policy": "No TEST access; later TEST after frozen VAL choice is exploratory.",
}


def parameters(model):
    count = lambda module: sum(p.numel() for p in module.parameters())
    report = {"total": count(model),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if model.variant in MASK_MOE_VARIANTS:
        fusion = model.short_long_mask_moe
        report["experts"] = {name: count(getattr(fusion, name + "_expert"))
                             for name in EXPERT_NAMES}
        report["fusion_norm"] = count(fusion.fusion_norm)
        report["routing_logits"] = (4 if model.variant != FIXED_VARIANT else 0)
    return report


def matched_initialization(kwargs, seed):
    states = {}
    for variant in MASK_MOE_VARIANTS:
        set_seed(seed)
        module = HeteroMixHopCMGM(variant=variant, **kwargs).short_long_mask_moe
        states[variant] = {name: value.detach().clone() for name, value in module.state_dict().items()
                           if name.startswith((*[part + "_expert." for part in EXPERT_NAMES],
                                               "fusion_norm."))}
    reference = states[GLOBAL_VARIANT]
    mismatches = {variant: [name for name, value in reference.items()
                            if name not in state or not torch.equal(value, state[name])]
                  for variant, state in states.items()}
    return {"shared_expert_tensor_count": len(reference), "mismatches": mismatches,
            "PASS": all(not names for names in mismatches.values())}


def routing(model):
    if model.variant not in MASK_MOE_VARIANTS:
        return None
    weights = model.short_long_mask_moe.routing_weights()
    return {"sample_dependent": False, "expert_order": list(EXPERT_NAMES),
            "access": {"short": ["shared", "short_memory", "short_shock"],
                       "long": ["shared", "long_memory", "long_structure"]},
            "weights": {group: value.detach().cpu().tolist() for group, value in weights.items()},
            "entropy_nats": {group: float(-(value * value.clamp_min(1e-12).log()).sum())
                             for group, value in weights.items()}}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    norm_sums = {name: 0.0 for name in EXPERT_NAMES}
    count = 0
    for batch in full_loader(loader):
        x, y = batch[:2]
        prediction = model(x.to(device))
        if prediction.shape != y.shape or not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite or misaligned TRAIN/VAL prediction")
        predictions.append(prediction.cpu().numpy())
        targets.append(y.numpy())
        if model.variant in MASK_MOE_VARIANTS:
            experts = model.short_long_mask_moe.last_experts
            for name, value in experts.items():
                norm_sums[name] += float(value.norm(dim=-1).sum())
            count += len(x)
    p, y = np.concatenate(predictions), np.concatenate(targets)
    metrics = {str(h): population_metrics(p[:, i], y[:, i])
               for i, h in enumerate(config.MULTI_HORIZONS)}
    norms = {name: total / count for name, total in norm_sums.items()} if count else None
    return metrics, norms


def preflight():
    kwargs = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2,
                  commodity_indices=(6, 30))
    x = torch.randn(2, 20, 30, 21)
    for variant in VARIANTS:
        set_seed(42)
        model = HeteroMixHopCMGM(variant=variant, **kwargs).eval()
        with torch.inference_mode():
            prediction = model(x)
        if prediction.shape != (2, 4, 24) or not torch.isfinite(prediction).all():
            raise AssertionError(f"Invalid synthetic prediction: {variant}")
        if not initialization_audit(kwargs, variant, 42)["PASS"]:
            raise AssertionError(f"Shared D0B initialization changed: {variant}")
    if not matched_initialization(kwargs, 42)["PASS"]:
        raise AssertionError("Five-expert initialization differs across routes")
    print("Short/long task-mask synthetic preflight PASS: five variants; no training or TEST.")


def run(args):
    check_protocol()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    data = build_data(SimpleNamespace(batch_size=64, seq_len=20, seed=args.seed))
    kwargs = model_kwargs(data)
    fingerprint = data_audit(data)
    shared = initialization_audit(kwargs, args.variant, args.seed)
    matched = matched_initialization(kwargs, args.seed)
    if not shared["PASS"] or not matched["PASS"]:
        raise RuntimeError(f"Initialization audit failed: {shared}; {matched}")
    set_seed(args.seed)
    model = HeteroMixHopCMGM(variant=args.variant, **kwargs).to(args.device)
    name = f"{args.variant}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    output = EXPERIMENT_ROOT / name
    checkpoint = CHECKPOINT_ROOT / name / "best.pt"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    metadata = {
        "variant": args.variant, "seed": args.seed, "model_kwargs": kwargs,
        "protocol": PROTOCOL, "val_rule": VAL_RULE, "data_audit": fingerprint,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "parameters": parameters(model), "shared_initialization_audit": shared,
        "five_expert_initialization_audit": matched,
        "target_scale": "decimal return", "hit_storage": "fraction",
        "test_evaluated": False,
    }
    for filename, value in (("config.json", metadata), ("data_audit.json", fingerprint),
                            ("initialization_audit.json", {"shared": shared, "matched": matched})):
        write_json(output / filename, value)
    started = time.perf_counter()
    history = train(model, data["loaders"]["train"], data["loaders"]["val"],
                    torch.empty(2, 0, dtype=torch.long), torch.empty(0),
                    torch.device(args.device), num_epochs=200, lr=1e-4,
                    weight_decay=1e-5, patience=10, checkpoint_path=str(checkpoint),
                    checkpoint_metadata=metadata)
    train_seconds = time.perf_counter() - started
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    if (payload.get("metadata", {}).get("variant") != args.variant
            or payload.get("best_epoch") != history["best_epoch"]):
        raise RuntimeError("Checkpoint variant or best epoch mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    device = torch.device(args.device)
    metrics, norms = {}, {}
    for split in ("train", "val"):
        metrics[split], norms[split] = evaluate(model, data["loaders"][split], device)
    route = routing(model)
    diagnostics = {}
    if args.variant in (FIXED_VARIANT, LEARNED_VARIANT):
        fusion = model.short_long_mask_moe
        original = metrics["val"]
        for name in ("swap_short_long", "remove_short_memory", "remove_short_shock",
                     "remove_long_memory", "remove_long_structure"):
            with fusion.intervention(name):
                alternative, _ = evaluate(model, data["loaders"]["val"], device)
            diagnostics[name] = {
                "metrics": alternative,
                "delta_mae": {str(h): alternative[str(h)]["MAE"] - original[str(h)]["MAE"]
                              for h in config.MULTI_HORIZONS},
            }
    result = {
        "status": "COMPLETE", "variant": args.variant, "seed": args.seed,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_prediction_huber_batch_mean": float(payload["best_val_loss"]),
        "train_seconds": train_seconds, "parameters": metadata["parameters"],
        "metrics": metrics, "routing": route, "expert_norms": norms,
        "val_frozen_interventions": diagnostics, "test_evaluated": False,
    }
    for filename, value in (("training_history.json", history), ("metrics.json", metrics),
                            ("routing_diagnostics.json", route), ("expert_norms.json", norms),
                            ("val_frozen_interventions.json", diagnostics), ("result.json", result)):
        write_json(output / filename, value)
    print(f"COMPLETE {args.variant} seed={args.seed}; VAL MAE@5d={metrics['val']['5']['MAE']:.9g}")
    print(f"Result: {output}\nCheckpoint: {checkpoint}")


def compare(paths, output):
    if len(paths) != len(VARIANTS):
        raise ValueError("Provide exactly five result.json paths")
    results, configs = {}, {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        variant = result["variant"]
        if (variant not in VARIANTS or variant in results or result["seed"] != 42
                or result["status"] != "COMPLETE" or result["test_evaluated"]
                or sha256(Path(result["checkpoint"])) != result["checkpoint_sha256"]):
            raise ValueError(f"Invalid or incomparable result: {path}")
        results[variant] = result
        configs[variant] = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    if set(results) != set(VARIANTS):
        raise ValueError(f"Missing variants: {set(VARIANTS)-set(results)}")
    reference = configs[BASE_VARIANT]
    for variant, record in configs.items():
        for key in ("protocol", "data_audit", "git_commit", "source_sha256", "seed"):
            if record[key] != reference[key]:
                raise ValueError(f"Incomparable {key}: {variant}")
    mae = lambda variant, h: float(results[variant]["metrics"]["val"][str(h)]["MAE"])
    main = mae(LEARNED_VARIANT, 5)
    summary = {
        "status": "COMPLETE", "seed": 42, "test_evaluated": False,
        "val_rule": VAL_RULE,
        "learned_beats_d0b": main < mae(BASE_VARIANT, 5),
        "learned_beats_five_expert_global": main < mae(GLOBAL_VARIANT, 5),
        "learned_beats_three_expert_global": main < mae(THREE_GLOBAL, 5),
        "rows": [{"variant": variant,
                  "val_mae": {str(h): mae(variant, h) for h in config.MULTI_HORIZONS},
                  "val_huber": results[variant]["best_val_prediction_huber_batch_mean"],
                  "best_epoch": results[variant]["best_epoch"],
                  "parameters": results[variant]["parameters"],
                  "train_seconds": results[variant]["train_seconds"]}
                 for variant in VARIANTS],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    write_json(output, summary)
    print(f"VAL comparison: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly train one TRAIN/VAL model")
    parser.add_argument("--variant", choices=VARIANTS, default=LEARNED_VARIANT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare-results", nargs=5, type=Path, metavar="RESULT_JSON")
    parser.add_argument("--comparison-output", type=Path,
                        default=EXPERIMENT_ROOT / "seed42_val_comparison.json")
    args = parser.parse_args()
    if args.compare_results:
        if args.run:
            parser.error("Comparison cannot be combined with training")
        compare(args.compare_results, args.comparison_output)
    elif args.run:
        run(args)
    else:
        preflight()


if __name__ == "__main__":
    main()
