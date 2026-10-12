"""Controlled D0B global-concat and commodity-preserving history readout study.

Default invocation is synthetic preflight only. Pass --run to fit one selected
variant. TEST metrics are computed only when --evaluate-test is also supplied.
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
from torch import nn
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.node_history_readout import (
    ALL_VARIANTS, CONCAT_VARIANT, NODE_ATTENTION_VARIANT,
    NODE_MEAN_VARIANT, attention_diagnostics,
)
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / "d0b_node_history_readout"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "d0b_node_history_readout"
BASE_VARIANT = "switching_latent_balanced_readout"
MEAN_FUSION_VARIANT = "switching_latent_balanced_mean_fusion"
VARIANTS = (BASE_VARIANT, MEAN_FUSION_VARIANT, *ALL_VARIANTS)
DISPLAY = {
    BASE_VARIANT: "D0B-BalancedLatentReadout",
    MEAN_FUSION_VARIANT: "D0B-MeanFusion",
    CONCAT_VARIANT: "D0B-GlobalConcatFusion",
    NODE_MEAN_VARIANT: "D0B-NodeHistoryMean",
    NODE_ATTENTION_VARIANT: "D0B-CommodityConditionedHistoryAttention",
}
PROTOCOL = {
    "seq_len": 20, "feature_dim": 21, "horizons": [1, 5, 10, 20],
    "target_type": "return", "optimizer": "Adam", "lr": 1e-4,
    "weight_decay": 1e-5, "batch_size": 64, "max_epochs": 200,
    "early_stopping_patience": 10,
    "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
    "loss": "sum of four Huber horizon losses, delta=.02",
    "checkpoint_selection": "existing prediction-only four-horizon VAL Huber; existing batch-mean aggregation",
    "native_switch_kl": "enabled; beta_max=5e-4, warmup=20 epochs",
    "train_shuffle": False, "train_drop_last": True,
    "full_split_evaluation_drop_last": False,
}
SOURCE_FILES = (
    "cmgm/config.py", "cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/switching_latent_transformer.py",
    "cmgm/models/node_history_readout.py", "cmgm/training/train.py",
    "cmgm/training/metric_standard.py", "cmgm/data/data_loader.py",
    "cmgm/data/feature_builder.py", "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/d0b_node_history_readout.py",
)
INPUT_FEATURE_NAMES = (
    "price", "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_10d", "vol_20d", "zscore_5d", "zscore_10d", "zscore_20d",
    "price_ma5", "price_ma10", "price_ma20", "price_ma60", "rsi_14",
    "bb_position", "skewness_20d", "kurtosis_20d", "roc_10d", "percentile_20d",
)


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n", encoding="utf-8")


def _hashes() -> dict:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def _model_kwargs(data: dict) -> dict:
    start, end = data["market_indices"]["commodity"]
    return {
        "num_nodes": data["n_nodes"],
        "n_commodities": data["n_commodities"],
        "n_stock": data["market_indices"]["stock"][1] - data["market_indices"]["stock"][0],
        "n_bond": data["market_indices"]["bond"][1] - data["market_indices"]["bond"][0],
        "commodity_indices": (int(start), int(end)),
    }


def _make_model(variant: str, kwargs: dict) -> HeteroMixHopCMGM:
    return HeteroMixHopCMGM(**kwargs, variant=variant)


def _init_audit(reference: nn.Module, candidate: nn.Module) -> dict:
    ref, cur = dict(reference.named_parameters()), dict(candidate.named_parameters())
    excluded = {"gate_fc.weight", "gate_fc.bias"}
    excluded.update(name for name in cur if name.startswith(("global_concat_fusion.", "node_history_readout.")))
    if candidate.variant == NODE_ATTENTION_VARIANT:
        excluded = {"gate_fc.weight", "gate_fc.bias"}
        excluded.update({"node_history_readout.W_q.weight", "node_history_readout.W_k.weight"})
    compared = [name for name in ref if name not in excluded]
    missing = [name for name in compared if name not in cur]
    mismatches = {
        name: float((ref[name] - cur[name]).abs().max())
        for name in compared if name in cur and not torch.equal(ref[name], cur[name])
    }
    return {
        "compared_parameter_count": sum(ref[name].numel() for name in compared if name in cur),
        "compared_tensor_count": len(compared), "missing": missing,
        "mismatched_max_abs_diff": mismatches,
        "shared_max_abs_diff": max(mismatches.values(), default=0.0),
        "mismatch_count": len(missing) + len(mismatches),
        "PASS": not missing and not mismatches,
    }


def _audit_seed(variant: str, kwargs: dict, seed: int) -> dict:
    # Audit construction is isolated from the subsequent training RNG stream.
    cpu_state = torch.random.get_rng_state()
    py_state, np_state = random.getstate(), np.random.get_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_seed(seed)
        ref = _make_model(BASE_VARIANT, kwargs)
        set_seed(seed)
        candidate = _make_model(variant, kwargs)
        audit = _init_audit(ref, candidate)
        if variant in (NODE_MEAN_VARIANT, NODE_ATTENTION_VARIANT):
            audit["new_module_parameter_count"] = sum(
                p.numel() for n, p in candidate.named_parameters()
                if n.startswith("node_history_readout.")
            )
        return audit
    finally:
        torch.random.set_rng_state(cpu_state)
        random.setstate(py_state)
        np.random.set_state(np_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _data_fingerprint(data: dict, include_test: bool = False) -> dict:
    names = data["feature_names"]
    c0, c1 = data["market_indices"]["commodity"]
    splits = {}
    for split in (("train", "val", "test") if include_test else ("train", "val")):
        dataset = data["loaders"][split].dataset
        digest = hashlib.sha256()
        for field in ("feature_matrix", "prices", "raw_prices"):
            value = getattr(dataset, field, None)
            if value is not None:
                digest.update(field.encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        splits[split] = {
            "sha256": digest.hexdigest(), "origins": len(dataset),
            "seq_len": int(dataset.seq_len), "horizons": list(dataset.horizons),
            "target_type": dataset.target_type,
        }
    return {
        "splits": splits, "test_fingerprint_included": bool(include_test),
        "node_count": int(data["n_nodes"]), "feature_dim": int(config.FEATURE_DIM),
        "commodity_order": [str(names[i]) for i in range(c0, c1)],
        "commodity_mapping": [
            {"commodity": str(names[c0 + i]), "node_index": c0 + i,
             "target_index": i, "output_index": i}
            for i in range(c1 - c0)
        ],
        "market_indices": {k: list(map(int, v)) for k, v in data["market_indices"].items()},
        "input_feature_names": list(INPUT_FEATURE_NAMES),
    }


def _loader(loader):
    return DataLoader(loader.dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      drop_last=False, num_workers=0)


@torch.no_grad()
def _evaluate_split(model, loader, device, *, force_uniform_attention=False):
    model.eval()
    predictions, targets = [], []
    aux_rows = []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        if model.variant == NODE_ATTENTION_VARIANT:
            pred, aux = model.node_history_forward(
                x, force_uniform_attention=force_uniform_attention
            )
            if not force_uniform_attention:
                aux_rows.append({k: v.detach().cpu() for k, v in aux.items() if torch.is_tensor(v)})
        else:
            pred = model(x)
        if pred.shape != y.shape or not torch.isfinite(pred).all():
            raise ValueError(f"Invalid prediction shape/values {tuple(pred.shape)} vs {tuple(y.shape)}")
        predictions.append(pred.cpu().numpy())
        targets.append(y.numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    metrics = {str(h): population_metrics(prediction[:, i], target[:, i])
               for i, h in enumerate(config.MULTI_HORIZONS)}
    if model.variant != NODE_ATTENTION_VARIANT or force_uniform_attention:
        return metrics, None, prediction
    tensors = {k: torch.cat([row[k] for row in aux_rows], dim=0) for k in aux_rows[0]}
    diag = attention_diagnostics(tensors["attention"])
    diag.update({
        "s_nodes_l2_mean": float(tensors["s_nodes"].norm(dim=-1).mean()),
        "u_nodes_l2_mean": float(tensors["u_nodes"].norm(dim=-1).mean()),
        "s_global_l2_mean": float(tensors["s_global"].norm(dim=-1).mean()),
        "t_global_l2_mean": float(tensors["t_global"].norm(dim=-1).mean()),
    })
    return metrics, diag, prediction


def _synthetic_preflight() -> None:
    kwargs = dict(num_nodes=30, n_commodities=24, n_stock=4, n_bond=2,
                  commodity_indices=(6, 30))
    variants = VARIANTS
    models = {}
    for variant in variants:
        torch.manual_seed(17)
        model = _make_model(variant, kwargs).eval()
        models[variant] = model
    x = torch.randn(2, config.SEQ_LEN, 30, config.FEATURE_DIM)
    outputs = {}
    for variant, model in models.items():
        with torch.no_grad():
            outputs[variant] = model(x)
        assert outputs[variant].shape == (2, 4, 24)
        assert torch.isfinite(outputs[variant]).all()

    for variant in (CONCAT_VARIANT, NODE_MEAN_VARIANT, NODE_ATTENTION_VARIANT):
        audit = _audit_seed(variant, kwargs, 17)
        assert audit["PASS"], (variant, audit)
    mean, attention = models[NODE_MEAN_VARIANT], models[NODE_ATTENTION_VARIANT]
    for name in ("W_v", "W_o", "fusion_projection"):
        left = getattr(mean.node_history_readout, name)
        right = getattr(attention.node_history_readout, name)
        assert torch.equal(left.weight, right.weight)
        if left.bias is not None:
            assert torch.equal(left.bias, right.bias)
    assert sum(p.numel() for p in attention.node_history_readout.parameters()) - sum(
        p.numel() for p in mean.node_history_readout.parameters()
    ) == 6144
    with torch.no_grad():
        pred_uniform, aux = attention.node_history_forward(x, force_uniform_attention=True)
        pred_mean = mean(x)
        torch.testing.assert_close(pred_uniform, pred_mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(aux["attention"].sum(-1), torch.ones(2, 24), atol=1e-6, rtol=0)
        torch.testing.assert_close(attention(x.flip(0)).flip(0), attention(x), atol=2e-5, rtol=2e-5)
        alone = attention(x[:1])
        torch.testing.assert_close(alone, attention(x)[:1], atol=2e-5, rtol=2e-5)

    # Output-row sanity: repeated global representation exactly recovers the
    # corresponding rows of the original head for each commodity and horizon.
    base = models[BASE_VARIANT]
    shared = torch.randn(2, 64)
    repeated = shared[:, None, :].expand(-1, 24, -1)
    from cmgm.models.node_history_readout import apply_original_head_per_node
    with torch.no_grad():
        by_node = apply_original_head_per_node(base.head, repeated, 4, 24)
        direct = base.head(shared).view(2, 4, 24)
        torch.testing.assert_close(by_node, direct, rtol=0, atol=1e-6)

    real_shape_kwargs = dict(num_nodes=284, n_commodities=24, n_stock=248, n_bond=12,
                             commodity_indices=(260, 284))
    parameter_counts = {}
    for variant in VARIANTS:
        model = _make_model(variant, real_shape_kwargs)
        parameter_counts[variant] = sum(p.numel() for p in model.parameters())

    # The ordinary baseline still strictly round-trips its own checkpoint keys.
    base_roundtrip = HeteroMixHopCMGM(**kwargs, variant=BASE_VARIANT)
    base_roundtrip.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(base_roundtrip.eval()(x), outputs[BASE_VARIANT])
    print("Node-history synthetic preflight PASS: five variants, shapes, finite values, shared init, output-row order, uniform attention equivalence, batch/single consistency, strict baseline state round-trip.")
    print("Actual N=284 parameter counts: " + json.dumps(parameter_counts, sort_keys=True))
    print("No dataset loaded; no optimizer step, formal fit, or TEST evaluation.")


def run_experiment(args) -> None:
    if args.variant not in VARIANTS:
        raise ValueError(f"Unknown variant {args.variant!r}")
    if args.evaluate_test and not args.run:
        raise ValueError("TEST evaluation requires explicit --run --evaluate-test")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    if (config.SEQ_LEN, config.FEATURE_DIM, tuple(config.MULTI_HORIZONS),
        config.TARGET_TYPE, config.LOSS_TYPE, config.HUBER_DELTA,
        config.BATCH_SIZE, config.LEARNING_RATE, config.WEIGHT_DECAY,
        config.NUM_EPOCHS, config.PATIENCE) != (
        20, 21, (1, 5, 10, 20), "return", "huber", .02, 64, 1e-4, 1e-5, 200, 10):
        raise ValueError("The configured D0B training protocol differs from this frozen experiment")

    data = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                      seq_len=config.SEQ_LEN, seed=args.seed))
    data_fp = _data_fingerprint(data, include_test=args.evaluate_test)
    kwargs = _model_kwargs(data)
    initialization_audit = _audit_seed(args.variant, kwargs, args.seed)
    if not initialization_audit["PASS"]:
        raise RuntimeError(f"Shared initialization audit failed: {initialization_audit}")

    # Restore the exact seeded constructor RNG stream after read-only audit.
    set_seed(args.seed)
    model = _make_model(args.variant, kwargs)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    run_name = f"{args.variant}_seed{args.seed}_{stamp}"
    output = EXPERIMENT_ROOT / run_name
    checkpoint = CHECKPOINT_ROOT / run_name / "best.pt"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = model.to(device)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = {
        "variant": args.variant, "display_name": DISPLAY[args.variant], "seed": args.seed,
        "protocol": {**PROTOCOL, "seed": args.seed, "device": str(device)},
        "git_sha": git_sha, "source_hashes": _hashes(),
        "data_fingerprint": data_fp,
        "initialization_audit": initialization_audit,
        "commodity_order": data_fp["commodity_order"],
        "commodity_mapping": data_fp["commodity_mapping"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "metric_definition": "pooled origins x commodities; Hit stored as fraction",
        "prediction_head_row_order": "linear output row h*Nc+c reshaped as (H,Nc), validated against repeated-global direct head",
    }
    _json(output / "config.json", metadata)
    _json(output / "data_audit.json", data_fp)
    _json(output / "initialization_audit.json", initialization_audit)

    start_time = time.perf_counter()
    history = train(
        model, data["loaders"]["train"], data["loaders"]["val"],
        torch.empty(2, 0, dtype=torch.long), torch.empty(0), device,
        num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY, patience=config.PATIENCE,
        checkpoint_path=str(checkpoint), checkpoint_metadata=metadata,
    )
    training_seconds = time.perf_counter() - start_time
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    split_names = ["train", "val"] + (["test"] if args.evaluate_test else [])
    metrics, diagnostics = {}, {}
    predictions = {}
    for split in split_names:
        metrics[split], diagnostics[split], predictions[split] = _evaluate_split(
            model, _loader(data["loaders"][split]), device
        )
    if args.variant == NODE_ATTENTION_VARIANT:
        # Frozen-model intervention only: force uniform time weights after the
        # checkpoint is selected. It is not the independently trained mean model.
        val_loader = _loader(data["loaders"]["val"])
        uniform_metrics, _, uniform_predictions = _evaluate_split(
            model, val_loader, device, force_uniform_attention=True
        )
        diagnostics["val"]["force_uniform_attention_intervention"] = {
            "metrics": uniform_metrics,
            "prediction_mae_abs_difference_by_horizon": {
                str(h): float(np.abs(predictions["val"][:, i] - uniform_predictions[:, i]).mean())
                for i, h in enumerate(config.MULTI_HORIZONS)
            },
            "interpretation": "frozen trained attention checkpoint with attention replaced by uniform weights; not an independently trained uniform-history model",
        }

    result = {
        "status": "COMPLETE", "model": metadata,
        "checkpoint": str(checkpoint.resolve()),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_prediction_loss": float(payload["best_val_loss"]),
        "training_seconds": training_seconds,
        "metrics": metrics, "readout_diagnostics": diagnostics,
        "test_evaluated": bool(args.evaluate_test),
    }
    _json(output / "training_history.json", history)
    _json(output / "metrics.json", metrics)
    _json(output / "readout_diagnostics.json", diagnostics)
    _json(output / "result.json", result)
    print(f"COMPLETE: {args.variant}; best epoch={payload['best_epoch']}; results={output}")
    print(f"Checkpoint: {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="fit one selected variant")
    parser.add_argument("--variant", choices=VARIANTS, default=BASE_VARIANT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="explicitly evaluate TEST after restoring the best VAL checkpoint")
    args = parser.parse_args()
    if not args.run:
        _synthetic_preflight()
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
