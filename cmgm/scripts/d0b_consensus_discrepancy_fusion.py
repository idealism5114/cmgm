"""Compare D0B gated, mean, and consensus-discrepancy fusion.

Default execution is synthetic preflight only. A formal fit requires --run.
TEST evaluation is separate and requires --evaluate-test after that fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.consensus_discrepancy_fusion import (
    ConsensusDiscrepancyFusion, MEAN_VARIANT, RESIDUAL_VARIANT, mean_fusion,
)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / "d0b_consensus_discrepancy_fusion"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "d0b_consensus_discrepancy_fusion"
BASE_VARIANT = "switching_latent_balanced_readout"
VARIANTS = (BASE_VARIANT, MEAN_VARIANT, RESIDUAL_VARIANT)
DISPLAY = {
    BASE_VARIANT: "D0B-BalancedLatentReadout (adaptive gate)",
    MEAN_VARIANT: "D0B-MeanFusion",
    RESIDUAL_VARIANT: "D0B-ConsensusDiscrepancyFusion",
}
PROTOCOL = {
    "seq_len": 20, "feature_dim": 21, "horizons": [1, 5, 10, 20],
    "target_type": "return", "split": "existing chronological TRAIN/VAL/TEST from main_ablation.build_data",
    "batch_size": 64, "optimizer": "Adam",
    "lr": 1e-4, "weight_decay": 1e-5, "max_epochs": 200, "patience": 10,
    "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
    "prediction_loss": "sum of four horizon Huber losses, delta=0.02",
    "switch_kl": "native D0B KL, beta_max=5e-4, warmup=20 epochs",
    "checkpoint_selection": "existing batch-mean four-horizon validation prediction Huber",
    "train_shuffle": False, "train_drop_last": True,
    "full_evaluation_drop_last": False,
}
SOURCE_FILES = (
    "cmgm/config.py", "cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/consensus_discrepancy_fusion.py",
    "cmgm/models/switching_latent_transformer.py", "cmgm/training/train.py",
    "cmgm/training/metric_standard.py", "cmgm/data/data_loader.py",
    "cmgm/data/feature_builder.py", "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/d0b_consensus_discrepancy_fusion.py",
)


def _dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _source_hashes() -> dict:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def _shared_init_audit(reference: nn.Module, candidate: nn.Module) -> dict:
    ref, cur = dict(reference.named_parameters()), dict(candidate.named_parameters())
    excluded = {"gate_fc.weight", "gate_fc.bias"}
    if candidate.variant == RESIDUAL_VARIANT:
        excluded.update(name for name in cur if name.startswith("consensus_discrepancy_fusion."))
    compared = [name for name in ref if name not in excluded]
    missing = [name for name in compared if name not in cur]
    differences = {name: float((ref[name] - cur[name]).abs().max())
                   for name in compared if name in cur and not torch.equal(ref[name], cur[name])}
    audit = {
        "shared_parameter_count": sum(ref[name].numel() for name in compared if name in cur),
        "compared_tensors": len(compared), "missing_parameter_names": missing,
        "mismatched_parameter_max_abs_diff": differences,
        "shared_max_abs_diff": max(differences.values(), default=0.0),
        "mismatch_count": len(missing) + len(differences),
        "PASS": not missing and not differences,
    }
    return audit


def _model_kwargs(n_nodes: int, n_commodities: int, market_indices: dict) -> dict:
    return dict(
        num_nodes=n_nodes, n_commodities=n_commodities,
        n_stock=market_indices["stock"][1] - market_indices["stock"][0],
        n_bond=market_indices["bond"][1] - market_indices["bond"][0],
    )


def _synthetic_preflight() -> None:
    torch.manual_seed(421)
    module = ConsensusDiscrepancyFusion().eval()
    s, t = torch.randn(3, 64), torch.randn(3, 64)
    fused, parts = module(s, t, return_components=True)
    c, d = 0.5 * (s + t), s - t
    context = module.A_c(c) + module.b
    residual = torch.tanh(context + module.A_d(d)) - torch.tanh(context)
    expected = c + module.B(residual)
    torch.testing.assert_close(fused, expected)
    assert fused.shape == (3, 64) and parts["d"].shape == (3, 64)
    assert sum(p.numel() for p in module.parameters()) == 3_088
    with torch.no_grad():
        module.B.weight.normal_()
    equal = torch.randn(3, 64)
    equal_fused, equal_parts = module(equal, equal, return_components=True)
    torch.testing.assert_close(equal_parts["delta"], torch.zeros_like(equal))
    torch.testing.assert_close(equal_fused, equal)
    torch.testing.assert_close(mean_fusion(s, t), 0.5 * (s + t))

    kwargs = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2)
    torch.manual_seed(19)
    baseline = HeteroMixHopCMGM(**kwargs, variant=BASE_VARIANT)
    torch.manual_seed(19)
    mean = HeteroMixHopCMGM(**kwargs, variant=MEAN_VARIANT)
    torch.manual_seed(19)
    full = HeteroMixHopCMGM(**kwargs, variant=RESIDUAL_VARIANT)
    assert _shared_init_audit(baseline, mean)["PASS"]
    assert _shared_init_audit(baseline, full)["PASS"]
    assert "gate_fc" in baseline._modules and "gate_fc" not in mean._modules
    assert "gate_fc" not in full._modules
    assert sum(p.numel() for p in full.consensus_discrepancy_fusion.parameters()) == 3_088
    x = torch.randn(2, config.SEQ_LEN, 30, config.FEATURE_DIM)
    baseline.eval(); mean.eval(); full.eval()
    with torch.no_grad():
        assert baseline(x).shape == mean(x).shape == full(x).shape == (2, 4, 24)
        torch.testing.assert_close(mean(x), full(x), rtol=0, atol=0)
        torch.testing.assert_close(full(x.flip(0)).flip(0), full(x))
        branch = full.switching_latent_transformer
        assert branch.set_epoch(1) >= 0 and torch.isfinite(branch.switch_loss())
    print("Synthetic preflight PASS: formula, zero-discrepancy, parameter count, shared init, shapes, native switching KL.")
    print("No dataset was loaded; no optimizer/training/test evaluation was run.")


def _fingerprint(data: dict) -> dict:
    market_indices = data["market_indices"]
    names = data["feature_names"]
    cs, ce = market_indices["commodity"]
    if ce - cs != data["n_commodities"]:
        raise ValueError("Commodity range and dataset target dimension disagree")
    splits = {}
    for split in ("train", "val", "test"):
        ds = data["loaders"][split].dataset
        digest = hashlib.sha256()
        for field in ("feature_matrix", "prices", "raw_prices"):
            value = getattr(ds, field, None)
            if value is not None:
                arr = np.ascontiguousarray(value)
                digest.update(field.encode())
                digest.update(arr.tobytes())
        splits[split] = {
            "sha256": digest.hexdigest(), "origins": len(ds),
            "seq_len": int(ds.seq_len), "horizons": list(ds.horizons),
            "target_type": ds.target_type,
        }
    return {
        "splits": splits,
        "commodity_order": [str(names[i]) for i in range(cs, ce)],
        "commodity_mapping": [
            {"commodity": str(names[cs + i]), "node_index": cs + i,
             "target_index": i, "output_index": i} for i in range(ce - cs)
        ],
        "node_count": int(data["n_nodes"]),
        "feature_dim": config.FEATURE_DIM,
        "input_feature_names": ["price", "ret_1d", "ret_5d", "ret_10d", "ret_20d",
            "vol_5d", "vol_10d", "vol_20d", "zscore_5d", "zscore_10d", "zscore_20d",
            "price_ma5", "price_ma10", "price_ma20", "price_ma60", "rsi_14",
            "bb_position", "skewness_20d", "kurtosis_20d", "roc_10d", "percentile_20d"],
    }


def _full_loader(loader):
    return DataLoader(loader.dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      drop_last=False, num_workers=0)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        pred = model(x)
        if pred.shape != y.shape or not torch.isfinite(pred).all():
            raise ValueError(f"Invalid predictions: pred={tuple(pred.shape)}, target={tuple(y.shape)}")
        predictions.append(pred.cpu().numpy())
        targets.append(y.numpy())
    p, y = np.concatenate(predictions), np.concatenate(targets)
    return {str(h): population_metrics(p[:, i], y[:, i])
            for i, h in enumerate(config.MULTI_HORIZONS)}


@torch.no_grad()
def _fusion_diagnostics(model, loader, device):
    rows = {key: [] for key in ("consensus", "discrepancy", "delta", "ratio")}
    model.eval()
    for batch in loader:
        x = batch[0].to(device)
        hs = model._temp_weighted_spatial(x)
        ht = model.switching_latent_transformer(x)
        s, t = model.gcn_proj(hs), model.lstm_proj(ht)
        c = .5 * (s + t)
        d = s - t
        if model.variant == RESIDUAL_VARIANT:
            _, parts = model.consensus_discrepancy_fusion(s, t, return_components=True)
            delta = parts["delta"]
        else:
            delta = torch.zeros_like(c)
        cn, dn, qn = c.norm(dim=-1), d.norm(dim=-1), delta.norm(dim=-1)
        rows["consensus"].append(cn.cpu())
        rows["discrepancy"].append(dn.cpu())
        rows["delta"].append(qn.cpu())
        rows["ratio"].append((qn / (cn + 1e-8)).cpu())
    cat = {k: torch.cat(v) for k, v in rows.items()}
    return {
        "consensus_l2_mean_per_sample": float(cat["consensus"].mean()),
        "discrepancy_l2_mean_per_sample": float(cat["discrepancy"].mean()),
        "delta_l2_mean_per_sample": float(cat["delta"].mean()),
        "delta_to_consensus_ratio_mean_per_sample": float(cat["ratio"].mean()),
        "delta_to_consensus_ratio_median_per_sample": float(cat["ratio"].median()),
        "delta_to_consensus_ratio_max_per_sample": float(cat["ratio"].max()),
        "sample_count": int(cat["ratio"].numel()),
        "ratio_reduction": "per-sample L2 ratio, summarized over split samples",
    }


def run_experiment(args) -> None:
    if args.variant not in VARIANTS:
        raise ValueError(f"Unsupported variant {args.variant}")
    if (config.SEQ_LEN, config.FEATURE_DIM, tuple(config.MULTI_HORIZONS),
        config.TARGET_TYPE, config.LOSS_TYPE, config.HUBER_DELTA,
        config.BATCH_SIZE, config.LEARNING_RATE, config.WEIGHT_DECAY,
        config.NUM_EPOCHS, config.PATIENCE) != (
        20, 21, (1, 5, 10, 20), "return", "huber", .02, 64, 1e-4, 1e-5, 200, 10):
        raise ValueError("Project config differs from the frozen D0B comparison protocol")
    if args.evaluate_test and not args.run:
        raise ValueError("TEST evaluation is available only as an explicit post-fit --run option")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    device = torch.device(args.device)
    set_seed(args.seed)
    from cmgm.scripts.main_ablation import build_data
    data = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                      seq_len=config.SEQ_LEN, seed=args.seed))
    data_audit = _fingerprint(data)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    run_name = f"{args.variant}_seed{args.seed}_{stamp}"
    output = EXPERIMENT_ROOT / run_name
    checkpoint = CHECKPOINT_ROOT / run_name / "best.pt"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint.parent.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    kwargs = _model_kwargs(data["n_nodes"], data["n_commodities"], data["market_indices"])
    reference = HeteroMixHopCMGM(**kwargs, variant=BASE_VARIANT)
    set_seed(args.seed)
    model = HeteroMixHopCMGM(**kwargs, variant=args.variant)
    initialization = _shared_init_audit(reference, model)
    del reference
    if not initialization["PASS"]:
        raise RuntimeError("Shared initialization audit failed; stopping before training")
    model = model.to(device)

    source_hashes = _source_hashes()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = {
        "variant": args.variant, "display_name": DISPLAY[args.variant], "seed": args.seed,
        "protocol": {**PROTOCOL, "seed": args.seed, "device": str(device)},
        "git_sha": git_sha, "source_hashes": source_hashes,
        "data_fingerprint": data_audit, "initialization_audit": initialization,
        "metric_definition": "pooled sample x commodity; Hit stored as fraction [0,1]",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "fusion_parameter_count": (sum(p.numel() for p in model.consensus_discrepancy_fusion.parameters())
                                   if args.variant == RESIDUAL_VARIANT else 0),
    }
    _dump(output / "config.json", metadata)
    _dump(output / "data_audit.json", data_audit)
    _dump(output / "initialization_audit.json", initialization)
    start = time.perf_counter()
    history = train(
        model, data["loaders"]["train"], data["loaders"]["val"],
        torch.empty(2, 0, dtype=torch.long), torch.empty(0), device,
        num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY, patience=config.PATIENCE,
        checkpoint_path=str(checkpoint), checkpoint_metadata=metadata,
    )
    seconds = time.perf_counter() - start
    splits = ("train", "val", "test") if args.evaluate_test else ("train", "val")
    metrics = {split: _evaluate(model, _full_loader(data["loaders"][split]), device)
               for split in splits}
    diagnostics = {split: _fusion_diagnostics(model, _full_loader(data["loaders"][split]), device)
                   for split in ("train", "val")}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (int(payload["best_epoch"]) != int(history["best_epoch"])
            or not np.isclose(float(payload["best_val_loss"]), min(history["val_loss"]))):
        raise RuntimeError("Checkpoint selection metadata disagrees with training history")
    payload.update(training_complete=True, training_seconds=seconds,
                   train_val_test_metrics=metrics)
    torch.save(payload, checkpoint)
    _dump(output / "training_history.json", history)
    _dump(output / "metrics.json", metrics)
    _dump(output / "fusion_diagnostics.json", diagnostics)
    _dump(output / "result.json", {
        "status": "COMPLETE", "model": metadata,
        "checkpoint": str(checkpoint.resolve()),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_prediction_loss": float(payload["best_val_loss"]),
        "training_seconds": seconds, "metrics": metrics,
        "fusion_diagnostics": diagnostics,
        "test_evaluated": bool(args.evaluate_test),
    })
    print(f"COMPLETE: {args.variant}; best epoch={payload['best_epoch']}; results={output}")
    print(f"Checkpoint: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--run", action="store_true", help="run one fixed-protocol fit")
    choice.add_argument("--preflight", action="store_true", help="synthetic-only checks (default)")
    parser.add_argument("--variant", choices=VARIANTS, default=RESIDUAL_VARIANT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="evaluate TEST once after the frozen best checkpoint is restored")
    args = parser.parse_args()
    if args.evaluate_test and not args.run:
        parser.error("--evaluate-test requires --run; TEST is evaluated only after a frozen best checkpoint")
    if args.run:
        run_experiment(args)
    else:
        _synthetic_preflight()


if __name__ == "__main__":
    main()
