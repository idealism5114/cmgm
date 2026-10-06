"""Preflight by default; run the fixed D0B-HorizonTensorFusion experiment with --run."""
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
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.horizon_tensor_fusion import DISPLAY_NAME, VARIANT
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import train

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments" / "d0b_horizon_tensor_fusion"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "d0b_horizon_tensor_fusion"
FROZEN_PROTOCOL = dict(seed_default=42, seq_len=20, feature_dim=21,
    horizons=[1, 5, 10, 20], batch_size=64, optimizer="Adam",
    learning_rate=1e-4, weight_decay=1e-5, max_epochs=200, patience=10,
    scheduler="ReduceLROnPlateau(mode=min,factor=.5,patience=5)",
    prediction_loss="sum of four horizon Huber losses, delta=.02",
    switch_kl="native D0B KL, beta_max=5e-4, warmup=20 epochs",
    selection="batch-mean four-horizon VAL prediction Huber; VAL5 is diagnostic only",
    train_shuffle=False, train_drop_last=True, full_metrics_drop_last=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_hashes() -> dict:
    files = ["cmgm/config.py", "cmgm/models/hetero_mixhop_model.py",
             "cmgm/models/horizon_tensor_fusion.py",
             "cmgm/models/switching_latent_transformer.py",
             "cmgm/training/train.py", "cmgm/training/metric_standard.py",
             "cmgm/data/data_loader.py", "cmgm/data/feature_builder.py",
             "cmgm/scripts/main_ablation.py"]
    return {name: _sha256_bytes((ROOT / name).read_bytes()) for name in files}


def _json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _synthetic_model(variant: str, seed: int):
    torch.manual_seed(seed)
    return HeteroMixHopCMGM(30, 24, n_stock=4, n_bond=2, variant=variant)


def _shared_initialization_audit(reference, model) -> dict:
    ref = dict(reference.named_parameters())
    new = dict(model.named_parameters())
    shared_names = sorted(name for name in ref if not name.startswith("gate_fc."))
    missing = sorted(name for name in shared_names if name not in new)
    changed = [name for name in shared_names if name in new and not torch.equal(ref[name], new[name])]
    max_diff = max((float((ref[name] - new[name]).abs().max())
                    for name in shared_names if name in new), default=0.0)
    return dict(shared_parameter_count=sum(ref[name].numel() for name in shared_names if name in new),
        shared_parameter_tensor_count=len(shared_names), shared_max_abs_diff=max_diff,
        mismatch_count=len(changed) + len(missing), mismatch_parameters=changed,
        missing_shared_parameters=missing,
        removed_parameters=sorted(name for name in ref if name not in new),
        added_parameters=sorted(name for name in new if name not in ref),
        PASS=not changed and not missing and max_diff == 0.0)


def synthetic_preflight() -> dict:
    """Small no-data/no-training smoke check; detailed checks live in pytest."""
    original = _synthetic_model("switching_latent_balanced_readout", 42)
    tensor = _synthetic_model(VARIANT, 42)
    old, new = dict(original.named_parameters()), dict(tensor.named_parameters())
    audit = _shared_initialization_audit(original, tensor)
    assert audit["PASS"]
    assert "gate_fc" not in tensor._modules
    assert tensor.horizon_tensor_fusion is not None
    assert sum(p.numel() for p in tensor.horizon_tensor_fusion.parameters()) == 21_832
    assert sum(p.numel() for p in new.values()) - sum(p.numel() for p in old.values()) == 13_576
    x = torch.randn(2, config.SEQ_LEN, 30, config.FEATURE_DIM)
    tensor.eval()
    with torch.no_grad():
        y = tensor(x)
        yp = tensor(x.flip(0)).flip(0)
    torch.testing.assert_close(y, yp)
    assert y.shape == (2, 4, 24) and torch.isfinite(y).all()
    result = dict(variant=VARIANT, display_name=DISPLAY_NAME, output_shape=list(y.shape),
                  fusion_params=21_832, gate_params_removed=8_256,
                  net_parameter_increase=13_576,
                  initialization_audit=audit,
                  batch_permutation_max_abs_diff=float((y - yp).abs().max()))
    del original, tensor
    torch.manual_seed(42)
    full_base = HeteroMixHopCMGM(284, 24, n_stock=248, n_bond=12,
                                 variant="switching_latent_balanced_readout")
    torch.manual_seed(42)
    full_tensor = HeteroMixHopCMGM(284, 24, n_stock=248, n_bond=12, variant=VARIANT)
    result["N284_parameter_counts"] = {
        "readout": sum(p.numel() for p in full_base.parameters()),
        "horizon_tensor": sum(p.numel() for p in full_tensor.parameters()),
    }
    assert result["N284_parameter_counts"] == {"readout": 520_549, "horizon_tensor": 534_125}
    del full_base, full_tensor
    print(json.dumps(result, indent=2))
    return result


def _data_fingerprint(data) -> dict:
    audit = {"mapping": [], "splits": {}}
    cs, ce = data["market_indices"]["commodity"]
    names = data["feature_names"]
    if ce - cs != 24:
        raise ValueError(f"Expected 24 commodities, got {ce-cs}")
    for i in range(24):
        audit["mapping"].append(dict(commodity=str(names[cs + i]), node_index=cs + i,
                                     target_index=i, output_index=i))
    # Deliberately fingerprints and reports only TRAIN/VAL; TEST is never evaluated.
    for split in ("train", "val"):
        ds = data["loaders"][split].dataset
        digest = hashlib.sha256()
        for name in ("feature_matrix", "prices", "raw_prices"):
            value = getattr(ds, name, None)
            if value is not None:
                arr = np.ascontiguousarray(value)
                digest.update(name.encode())
                digest.update(arr.tobytes())
        audit["splits"][split] = dict(sha256=digest.hexdigest(), origins=len(ds),
            seq_len=ds.seq_len, horizons=list(ds.horizons), target_type=ds.target_type,
            timeline=len(ds.raw_prices) if ds.raw_prices is not None else None)
    audit["node_names"] = [str(x) for x in names]
    audit["input_feature_names"] = ["price", "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "vol_5d", "vol_10d", "vol_20d", "zscore_5d", "zscore_10d", "zscore_20d",
        "price_ma5", "price_ma10", "price_ma20", "price_ma60", "rsi_14",
        "bb_position", "skewness_20d", "kurtosis_20d", "roc_10d", "percentile_20d"]
    if len(audit["input_feature_names"]) != config.FEATURE_DIM:
        raise AssertionError("Recorded feature list does not match FEATURE_DIM")
    return audit


def _full_loader(loader):
    return DataLoader(loader.dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      drop_last=False, num_workers=0)


@torch.no_grad()
def _evaluate_split(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        p = model(x)
        if p.shape != y.shape or p.shape[1:] != (4, 24):
            raise ValueError(f"Output/target mismatch: {tuple(p.shape)} vs {tuple(y.shape)}")
        if not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite prediction")
        predictions.append(p.cpu().numpy())
        targets.append(y.numpy())
    p, y = np.concatenate(predictions), np.concatenate(targets)
    return {str(h): population_metrics(p[:, i], y[:, i])
            for i, h in enumerate(config.MULTI_HORIZONS)}


@torch.no_grad()
def _fusion_diagnostics(model, loader, device):
    model.eval()
    module = model.horizon_tensor_fusion
    inter, linear = [], []
    intervention_metrics = {"interaction_disabled": [], "embedding_zero": []}
    intervention_targets = []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        hs = model._temp_weighted_spatial(x)
        ht = model.switching_latent_transformer(x)
        s, t = model.gcn_proj(hs), model.lstm_proj(ht)
        _, c = module(s, t, return_components=True)
        inter.append(c["interaction"].norm(dim=-1).cpu().numpy())
        linear.append(c["linear"].norm(dim=-1).cpu().numpy())
        for key, kwargs in (("interaction_disabled", {"disable_interaction": True}),
                             ("embedding_zero", {"zero_embedding": True})):
            fused = module(s, t, **kwargs)
            intervention_metrics[key].append(model._predict_from_horizon_states(fused).cpu().numpy())
        intervention_targets.append(y.numpy())
    inter = np.concatenate(inter, axis=0)
    linear = np.concatenate(linear, axis=0)
    targets = np.concatenate(intervention_targets, axis=0)
    frozen = {}
    for name, chunks in intervention_metrics.items():
        pred = np.concatenate(chunks, axis=0)
        frozen[name] = {str(h): population_metrics(pred[:, i], targets[:, i])
                        for i, h in enumerate(config.MULTI_HORIZONS)}
    horizons = list(config.MULTI_HORIZONS)
    k = module.effective_kernels().cpu().numpy()
    return dict(interaction_norm_by_horizon=inter.mean(axis=0).tolist(),
        linear_norm_by_horizon=linear.mean(axis=0).tolist(),
        interaction_to_linear_norm_ratio=(inter.mean(axis=0) / np.maximum(linear.mean(axis=0), 1e-12)).tolist(),
        horizon_embedding=module.E.detach().cpu().tolist(),
        horizon_embedding_norm=float(module.E.detach().norm()),
        kernel_component_norms={name: float(getattr(module, name).detach().norm())
                                for name in ("G0", "G1", "G2")},
        effective_kernel_pairwise_frobenius={
            f"{i}_{j}": float(np.linalg.norm(k[i] - k[j]))
            for i in range(len(horizons)) for j in range(i + 1, len(horizons))},
        frozen_interventions=frozen,
        interpretation="Frozen interventions describe this fitted model's reliance only; they are not retrained structural ablations.")


def run_experiment(args):
    if (config.SEQ_LEN, config.FEATURE_DIM, tuple(config.MULTI_HORIZONS), config.TARGET_TYPE,
        config.LOSS_TYPE, config.HUBER_DELTA, config.BATCH_SIZE, config.LEARNING_RATE,
        config.WEIGHT_DECAY, config.NUM_EPOCHS, config.PATIENCE) != (
        20, 21, (1, 5, 10, 20), "return", "huber", .02, 64, 1e-4, 1e-5, 200, 10):
        raise ValueError("Repository config differs from the frozen horizon tensor protocol")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    device = torch.device(args.device)
    set_seed(args.seed)
    from cmgm.scripts.main_ablation import build_data
    data = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE, seq_len=config.SEQ_LEN,
                                      seed=args.seed))
    audit = _data_fingerprint(data)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    out = EXPERIMENT_ROOT / f"run_seed{args.seed}_{stamp}"
    checkpoint = CHECKPOINT_ROOT / f"run_seed{args.seed}_{stamp}" / "best.pt"
    out.mkdir(parents=True, exist_ok=False)
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    source_hashes = _source_hashes()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    mi = data["market_indices"]
    set_seed(args.seed)
    reference = HeteroMixHopCMGM(data["n_nodes"], data["n_commodities"],
        n_stock=mi["stock"][1] - mi["stock"][0], n_bond=mi["bond"][1] - mi["bond"][0],
        variant="switching_latent_balanced_readout")
    set_seed(args.seed)
    model = HeteroMixHopCMGM(data["n_nodes"], data["n_commodities"],
        n_stock=mi["stock"][1] - mi["stock"][0], n_bond=mi["bond"][1] - mi["bond"][0],
        variant=VARIANT)
    init_audit = _shared_initialization_audit(reference, model)
    del reference
    if not init_audit["PASS"]:
        raise ValueError("Shared D0B initialization differs; stop before training")
    _json_dump(out / "initialization_audit.json", init_audit)
    model = model.to(device)
    counts = {"total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "fusion": sum(p.numel() for p in model.horizon_tensor_fusion.parameters()),
              "gate_removed_reference_count": 8_256}
    if "gate_fc" in model._modules:
        raise AssertionError("New fusion must not retain gate_fc")
    protocol = {**FROZEN_PROTOCOL, "seed": args.seed, "device": str(device)}
    metadata = dict(variant=VARIANT, display_name=DISPLAY_NAME, protocol=protocol,
        source_hashes=source_hashes, git_sha=git_sha, data_fingerprint=audit,
        parameter_counts=counts, initialization_audit=init_audit,
        horizons=list(config.MULTI_HORIZONS))
    _json_dump(out / "config.json", metadata)
    start = time.perf_counter()
    history = train(model, data["loaders"]["train"], data["loaders"]["val"],
        torch.empty(2, 0, dtype=torch.long), torch.empty(0), device,
        num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY, patience=config.PATIENCE,
        checkpoint_path=str(checkpoint), checkpoint_metadata=metadata)
    elapsed = time.perf_counter() - start
    # Full-population diagnostics are TRAIN/VAL only. No TEST loader is iterated.
    metrics = {split: _evaluate_split(model, _full_loader(data["loaders"][split]), device)
               for split in ("train", "val")}
    diagnostics = {split: _fusion_diagnostics(model, _full_loader(data["loaders"][split]), device)
                   for split in ("train", "val")}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training_complete"] = True
    payload["training_seconds"] = elapsed
    payload["train_val_metrics"] = metrics
    torch.save(payload, checkpoint)
    best_epoch = int(payload["best_epoch"])
    best_val_prediction_loss = float(payload["best_val_loss"])
    history_best_epoch = int(history["best_epoch"])
    history_best_loss = min(float(v) for v in history["val_loss"])
    if best_epoch != history_best_epoch or not np.isclose(
            best_val_prediction_loss, history_best_loss, rtol=1e-7, atol=1e-12):
        raise RuntimeError("Saved checkpoint selection metadata disagrees with training history")
    _json_dump(out / "training_history.json", history)
    _json_dump(out / "train_val_metrics.json", metrics)
    _json_dump(out / "fusion_diagnostics.json", diagnostics)
    _json_dump(out / "result.json", dict(status="COMPLETE", model=metadata,
        checkpoint=str(checkpoint.resolve()), best_epoch=best_epoch,
        best_val_prediction_loss=best_val_prediction_loss,
        training_seconds=elapsed, train_val_metrics=metrics,
        fusion_diagnostics=diagnostics, test_evaluated=False))
    print(f"COMPLETE: TRAIN/VAL only. Results: {out}")
    print(f"Checkpoint: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="run one fixed seed training and TRAIN/VAL evaluation")
    mode.add_argument("--preflight", action="store_true", help="synthetic-only checks (default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.run:
        run_experiment(args)
    else:
        synthetic_preflight()


if __name__ == "__main__":
    main()
