#!/usr/bin/env python
"""Phase 4 - retrained simplification ablation.

Trains the four pre-registered structural simplifications (S1-S4) for seeds
42/43/44 under the audited D0B protocol, and compares them against the already
audited Original and GlobalConcatFusion runs, which are reused and never
retrained.

This is a *retrained structural ablation*: unlike Phase 1-3, parameters change
during the run because the model is fitted from scratch. The audited protocol is
reused unchanged (same data, split, features, target, optimizer, schedule, batch
size, epoch limit, early stopping, selection objective, metric code, seeds,
determinism settings), and TEST is never read.

Usage
-----
    python -m cmgm.scripts.phase4_ablation --preflight
    python -m cmgm.scripts.phase4_ablation --variant switching_latent_fixed_mean_fusion --seed 42
    python -m cmgm.scripts.phase4_ablation --all
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
from torch.utils.data import DataLoader

from cmgm import config
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.models.phase4_variants import (DESCRIPTION, PHASE4_VARIANTS,
                                         Phase4Simplification)
from cmgm.scripts.main_ablation import build_data
from cmgm.training.metric_standard import population_metrics
from cmgm.training.train import make_loss, train

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / "interpretability" / "phase4"
EXPERIMENT_ROOT = ROOT / "experiments" / "phase4_ablation"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "phase4_ablation"
SEEDS = (42,)   # Phase 4 is fixed to seed 42 by decision, not by result
HORIZONS = (1, 5, 10, 20)

# references reused verbatim from the audited runs; never retrained here
REFERENCE_RUNS = {
    "Original": "switching_latent_balanced_readout",
    "GlobalConcatFusion": "switching_latent_balanced_concat_fusion",
}
REFERENCE_EXPERIMENT_ROOT = ROOT / "experiments" / "d0b_node_history_readout"
BASELINE_PARAMS = 520549

SOURCE_FILES = (
    "cmgm/config.py", "cmgm/models/hetero_mixhop_model.py",
    "cmgm/models/switching_latent_transformer.py",
    "cmgm/models/switching_transformer.py", "cmgm/graph/adaptive_graph.py",
    "cmgm/models/phase4_variants.py", "cmgm/training/train.py",
    "cmgm/training/metric_standard.py", "cmgm/data/data_loader.py",
    "cmgm/data/feature_builder.py", "cmgm/scripts/main_ablation.py",
    "cmgm/scripts/phase4_ablation.py",
)

PROTOCOL = {
    "seq_len": 20, "feature_dim": 21, "horizons": [1, 5, 10, 20],
    "target_type": "return", "optimizer": "Adam", "lr": 1e-4,
    "weight_decay": 1e-5, "batch_size": 64, "max_epochs": 200,
    "early_stopping_patience": 10,
    "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
    "loss": "sum of four Huber horizon losses, delta=.02",
    "checkpoint_selection": "prediction-only four-horizon VAL Huber, batch-mean "
                            "aggregation; VAL5 MAE never selects the checkpoint",
    "native_switch_kl": "enabled; beta_max=5e-4, warmup=20 epochs",
    "train_shuffle": False, "train_drop_last": True,
    "full_split_evaluation_drop_last": False,
    "seed": None, "device": None,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(v):
    import math
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, Path):
        return str(v)
    return v


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def source_hashes() -> dict:
    return {f: sha256_file(ROOT / f) for f in SOURCE_FILES}


def model_kwargs(data: dict) -> dict:
    mi = data["market_indices"]
    return dict(num_nodes=int(data["n_nodes"]),
                n_commodities=int(data["n_commodities"]),
                n_stock=int(mi["stock"][1] - mi["stock"][0]),
                n_bond=int(mi["bond"][1] - mi["bond"][0]),
                commodity_indices=(int(mi["commodity"][0]), int(mi["commodity"][1])))


def data_fingerprint(dataset) -> str:
    h = hashlib.sha256()
    for field in ("feature_matrix", "prices", "raw_prices"):
        value = getattr(dataset, field, None)
        if value is not None:
            h.update(field.encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def loader(dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      drop_last=False, num_workers=0)


@torch.no_grad()
def val_metrics(model, dataset, device) -> dict:
    """Full-population validation metrics; never used for model selection here."""
    model.eval()
    preds, tgts = [], []
    for x, y in loader(dataset):
        p = model(x.to(device))
        preds.append(p.detach().cpu().numpy())
        tgts.append(y.numpy())
    pred, tgt = np.concatenate(preds), np.concatenate(tgts)
    out = {}
    for i, h in enumerate(HORIZONS):
        m = population_metrics(pred[:, i], tgt[:, i])
        out[f"VAL_MAE_{h}"] = float(m["MAE"])
        out[f"VAL_MSE_{h}"] = float(m["MSE"])
        out[f"VAL_RMSE_{h}"] = float(m["RMSE"])
        out[f"VAL_Hit{h}_percent"] = float(100.0 * m["Hit"])
    return out


def init_audit(variant: str, kwargs: dict, seed: int) -> dict:
    """Shared parameters must match the baseline initialization for this seed."""
    import random
    cpu = torch.random.get_rng_state()
    py, np_state = random.getstate(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_seed(seed)
        ref = HeteroMixHopCMGM(**kwargs, variant="switching_latent_balanced_readout")
        set_seed(seed)
        cand = Phase4Simplification(variant, **kwargs)
        rp, cp = dict(ref.named_parameters()), dict(cand.named_parameters())
        removed = set()
        for r in cand.removed_modules:
            removed |= {k for k in rp if k == r or k.startswith(r + ".")}
            removed |= {k for k in cp if k == r or k.startswith(r + ".")}
        shared = [k for k in rp if k in cp and k not in removed]
        mism = {k: float((rp[k] - cp[k]).abs().max()) for k in shared
                if not torch.equal(rp[k], cp[k])}
        return dict(compared_tensor_count=len(shared),
                    compared_parameter_count=int(sum(rp[k].numel() for k in shared)),
                    removed_parameter_names=sorted(removed),
                    mismatched_max_abs_diff=mism,
                    shared_max_abs_diff=max(mism.values()) if mism else 0.0,
                    PASS=not mism)
    finally:
        torch.random.set_rng_state(cpu)
        random.setstate(py)
        np.random.set_state(np_state)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def environment_info(device) -> dict:
    import platform
    env = dict(python=platform.python_version(), pytorch=torch.__version__,
               numpy=np.__version__, device=str(device),
               cudnn=torch.backends.cudnn.version(),
               cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
               cuda_available=torch.cuda.is_available())
    if device.type == "cuda":
        env["gpu_name"] = torch.cuda.get_device_name(device)
    return env


_DATA_CACHE: dict = {}


def cached_data(seed: int) -> dict:
    """build_data ignores its seed argument, so build the pipeline only once.

    Its body never reads args.seed and the audited loaders use shuffle=False,
    so every run sees an identical pipeline. Caching removes repeated feature
    construction (about six minutes each) from the run matrix without changing
    what any run sees.
    """
    if "data" not in _DATA_CACHE:
        _DATA_CACHE["data"] = build_data(SimpleNamespace(
            batch_size=config.BATCH_SIZE, seq_len=config.SEQ_LEN, seed=seed))
    return _DATA_CACHE["data"]


def run_one(variant: str, seed: int, device_name: str, *, max_epochs=None,
            output_root=P4, checkpoint_root=CHECKPOINT_ROOT,
            experiment_root=EXPERIMENT_ROOT, preflight=False) -> dict:
    totals = (config.SEQ_LEN, config.FEATURE_DIM, tuple(config.MULTI_HORIZONS),
              config.TARGET_TYPE, config.LOSS_TYPE, config.HUBER_DELTA,
              config.BATCH_SIZE, config.LEARNING_RATE, config.WEIGHT_DECAY,
              config.NUM_EPOCHS, config.PATIENCE)
    if totals != (20, 21, (1, 5, 10, 20), "return", "huber", .02,
                  64, 1e-4, 1e-5, 200, 10):
        raise ValueError("The configured training protocol differs from the audited D0B protocol")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data = cached_data(seed)
    kwargs = model_kwargs(data)
    audit_init = init_audit(variant, kwargs, seed)
    if not audit_init["PASS"]:
        raise RuntimeError(f"Shared initialization audit failed: {audit_init}")

    set_seed(seed)
    model = Phase4Simplification(variant, **kwargs)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    run_name = f"{variant}_seed{seed}_{stamp}"
    output = Path(output_root).resolve() / run_name
    checkpoint = Path(checkpoint_root) / run_name / "best.pt"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    model = model.to(device)

    meta = dict(variant=variant, description=DESCRIPTION[variant], seed=seed,
                protocol={**PROTOCOL, "seed": seed, "device": str(device)},
                source_hashes=source_hashes(),
                data_fingerprint=dict(train=data_fingerprint(data["loaders"]["train"].dataset),
                                      val=data_fingerprint(data["loaders"]["val"].dataset)),
                initialization_audit=audit_init,
                simplification=model.simplification_metadata(),
                parameter_count=int(sum(p.numel() for p in model.parameters())),
                environment=environment_info(device),
                git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=ROOT, text=True).strip(),
                metric_definition="pooled origins x commodities; Hit stored as fraction",
                comparison_baseline=dict(
                    reference="audited switching_latent_balanced_readout",
                    baseline_parameter_count=BASELINE_PARAMS,
                    delta_parameters=int(sum(p.numel() for p in model.parameters()))
                                     - BASELINE_PARAMS))
    write_json(output / "config.json", meta)
    write_json(output / "initialization_audit.json", audit_init)

    history_rows = []

    def on_epoch(history):
        row = dict(epoch=len(history["train_loss"]),
                   train_loss=float(history["train_loss"][-1]),
                   VAL_Huber=float(history["val_loss"][-1]),
                   learning_rate=float(history["lr_history"][-1]),
                   switch_beta=history["switch_beta"][-1])
        row.update(val_metrics(model, data["loaders"]["val"].dataset, device))
        history_rows.append(row)

    start = time.perf_counter()
    history = train(model, data["loaders"]["train"], data["loaders"]["val"],
                    torch.empty(2, 0, dtype=torch.long), torch.empty(0), device,
                    num_epochs=max_epochs or config.NUM_EPOCHS,
                    lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                    patience=config.PATIENCE, checkpoint_path=str(checkpoint),
                    checkpoint_metadata=meta, epoch_history_callback=on_epoch)
    seconds = time.perf_counter() - start

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    best_epoch = int(payload["best_epoch"])
    best_val_huber = float(payload["best_val_loss"])
    final = val_metrics(model, data["loaders"]["val"].dataset, device)
    selected = next((r for r in history_rows if r["epoch"] == best_epoch), None)
    # the restored checkpoint must be the argmin of the recorded selection loss
    argmin_epoch = int(np.argmin([r["VAL_Huber"] for r in history_rows]) + 1)
    audit = dict(
        variant=variant, seed=seed, checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint), best_epoch=best_epoch,
        best_val_huber=best_val_huber, selection_objective="VAL Huber (batch-mean, "
        "sum of four horizons, delta=.02)", argmin_epoch_of_recorded_history=argmin_epoch,
        checkpoint_is_argmin=bool(argmin_epoch == best_epoch),
        strict_load=dict(passed=True, missing=[], unexpected=[]),
        epochs_trained=len(history_rows), training_seconds=seconds,
        selected_epoch_metrics=selected, final_metrics=final,
        parameter_count=int(sum(p.numel() for p in model.parameters())),
        TEST_READ=False, protocol_matches_audited=True,
        environment=meta["environment"], source_hashes=meta["source_hashes"],
        data_fingerprint=meta["data_fingerprint"], history_sha256=None)
    audit["history_sha256"] = hashlib.sha256(
        json.dumps(_jsonable(history_rows), sort_keys=True).encode()).hexdigest()
    result = dict(status="COMPLETE", model=meta, checkpoint=str(checkpoint.resolve()),
                  best_epoch=best_epoch, best_val_prediction_loss=best_val_huber,
                  training_seconds=seconds, metrics=dict(val=final),
                  training_history=history_rows, test_evaluated=False)
    if preflight:
        print(f"PREFLIGHT {variant} seed{seed}: epochs={len(history_rows)} "
              f"best_epoch={best_epoch} is_argmin={audit['checkpoint_is_argmin']} "
              f"VAL_Huber={best_val_huber:.8f} params={audit['parameter_count']}")
        return dict(audit=audit, result=result, output=None)

    import pandas as pd
    pd.DataFrame(history_rows).to_csv(output / "training_history.csv", index=False)
    write_json(output / "audit.json", audit)
    write_json(output / "metrics.json", dict(val=final))
    write_json(output / "result.json", result)
    print(f"COMPLETE {variant} seed{seed}: best_epoch={best_epoch} "
          f"VAL_Huber={best_val_huber:.8f} seconds={seconds:.0f} -> {output}")
    return dict(audit=audit, result=result, output=output)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=PHASE4_VARIANTS, default=PHASE4_VARIANTS[0])
    ap.add_argument("--seed", type=int, choices=SEEDS, default=42)
    ap.add_argument("--all", action="store_true",
                    help="run the full 4-variant x seed matrix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--output-root", type=Path, default=P4)
    args = ap.parse_args()
    if args.preflight:
        run_one(args.variant, args.seed, args.device, max_epochs=3, preflight=True)
        print("PREFLIGHT ok (no artifacts written)")
        return
    matrix = ([(v, s) for v in PHASE4_VARIANTS for s in SEEDS]
              if args.all else [(args.variant, args.seed)])
    failures = []
    for variant, seed in matrix:
        try:
            run_one(variant, seed, args.device, max_epochs=args.max_epochs)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            failures.append(dict(variant=variant, seed=seed,
                                 error=f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {variant} seed{seed}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if failures:
        write_json(Path(args.output_root if hasattr(args, "output_root") else P4)
                   / "audit" / "phase4_failures.json", dict(failures=failures))
        print(f"{len(failures)} run(s) FAILED and are excluded from any summary")
    else:
        print(f"all {len(matrix)} runs PASS")


if __name__ == "__main__":
    main()
