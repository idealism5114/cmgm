"""Checkpoint-only 5d error decomposition. No fitting, backward or model changes.

Run from the repository root with ``python -m cmgm.scripts.d0b_5d_error_regime_diagnostic``.
The comparator is required by default; --allow-missing-comparator produces an
explicitly INCOMPLETE D0B-only report, never a fabricated two-model conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cmgm.config import (BOND_FILE, COMMODITY_FILE, STOCK_FILE, MULTI_HORIZONS,
                         FEATURE_DIM, SEQ_LEN, BATCH_SIZE, TARGET_TYPE)
from cmgm.data.data_loader import (MarketSequenceDataset, align_markets,
    load_stock_prices, load_bond_prices, load_commodity_prices)
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"D0B": "switching_latent_balanced_readout",
            "TempWeighted": "temporal_weighted_graph"}
KEYS = ["split", "sample_index", "forecast_origin", "commodity_index", "commodity"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_checksum(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_fingerprint():
    paths = [STOCK_FILE, BOND_FILE, COMMODITY_FILE] + [ROOT / p for p in (
        "cmgm/config.py", "cmgm/data/data_loader.py", "cmgm/data/feature_builder.py",
        "cmgm/scripts/main_ablation.py")]
    return {str(p.resolve()): sha256(p) for p in paths}


def checkpoint_payload(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def discover_checkpoint(variant, directory, explicit=None):
    """Use registry metadata first, then actual filenames and embedded metadata.

    Ambiguity is an error. Never guess among multiple matching weights.
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    directory = Path(directory).resolve()
    registry = directory / "registry.json"
    if registry.is_file():
        obj = json.loads(registry.read_text())
        record = obj.get(variant, obj.get("checkpoints", {}).get(variant))
        if record is not None:
            raw = record if isinstance(record, str) else record.get("path", record.get("checkpoint_path"))
            if raw is None:
                raise ValueError(f"Registry entry has no path: {variant}")
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = registry.parent / p
            if not p.is_file():
                raise FileNotFoundError(f"Registry checkpoint missing: {p}")
            return p.resolve()
    matches = []
    for p in sorted(directory.rglob("*")):
        if p.suffix not in (".pt", ".pth", ".ckpt"):
            continue
        if p.stem == variant + "_best":
            matches.append(p)
            continue
        payload = checkpoint_payload(p)
        metadata = payload.get("metadata", payload.get("checkpoint_metadata", {}))
        recorded = metadata.get("variant", payload.get("variant"))
        if recorded == variant:
            matches.append(p)
    if len(matches) > 1:
        raise ValueError(f"Multiple checkpoints for {variant}; specify the official best: {matches}")
    return matches[0] if matches else None


def prepare_data(cache=None):
    """Optional numeric-only cache is accepted only with matching source/data hashes."""
    fingerprint = source_fingerprint()
    if cache and Path(cache).is_file():
        with np.load(cache, allow_pickle=False) as a:
            meta = json.loads(str(a["metadata"]))
            if meta["fingerprint"] != fingerprint or meta["seq_len"] != SEQ_LEN or meta["batch_size"] != BATCH_SIZE:
                raise ValueError("Prepared data fingerprint/config mismatch; rebuild with the official pipeline")
            mi = {k: tuple(v) for k, v in meta["market_indices"].items()}
            datasets = {s: MarketSequenceDataset(a["prices_" + s], mi, SEQ_LEN,
                feature_matrix=a["features_" + s], raw_prices=a["raw_" + s],
                target_type=TARGET_TYPE, horizons=MULTI_HORIZONS) for s in ("train", "val", "test")}
            names = meta["names"]
    else:
        from cmgm.scripts.main_ablation import build_data
        data = build_data(SimpleNamespace(seed=42, batch_size=BATCH_SIZE, seq_len=SEQ_LEN))
        datasets = {s: loader.dataset for s, loader in data["loaders"].items()}
        mi, names = data["market_indices"], data["feature_names"]
        if cache:
            arrays = {}
            for s, ds in datasets.items():
                arrays.update({"features_" + s: ds.feature_matrix, "prices_" + s: ds.prices,
                               "raw_" + s: ds.raw_prices})
            meta = dict(fingerprint=fingerprint, market_indices=mi, names=names,
                        seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
            arrays["metadata"] = np.array(json.dumps(meta))
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **arrays)
    if source_fingerprint() != fingerprint:
        raise RuntimeError("Source or data changed during preparation")
    prices, real_mi = align_markets(load_stock_prices(STOCK_FILE), load_bond_prices(BOND_FILE),
                                   load_commodity_prices(COMMODITY_FILE))
    assert mi == real_mi and list(prices.columns) == names
    raw = np.concatenate([datasets[s].raw_prices for s in ("train", "val", "test")])
    np.testing.assert_array_equal(raw, prices.values)
    return datasets, prices, mi, fingerprint


def causal_timeline(prices, mi):
    """Diagnostic columns only; never passed to model or fitted on future data.

    Volatility reproduces existing feature_builder vol_5d/vol_20d in raw units
    (same pct_change, rolling ddof=1, min_periods and clip), avoiding averaging
    asset-specific z-scores as though they were physical return volatility.
    """
    prices = pd.DataFrame(prices)
    ret = prices.pct_change().fillna(0.0)
    past5 = prices.pct_change(5).clip(-5, 5).fillna(0.0)
    cs, ce = mi["commodity"]
    context = {}
    for w in (5, 20):
        context[f"vol{w}"] = ret.rolling(w, min_periods=1).std().fillna(0).clip(0, 5).iloc[:, cs:ce].mean(axis=1).to_numpy()
    for market, (a, b) in mi.items():
        context[f"dispersion_{market}"] = ret.iloc[:, a:b].std(axis=1, ddof=0).rolling(5, min_periods=1).mean().to_numpy()
        context[f"past5_{market}"] = past5.iloc[:, a:b].mean(axis=1).to_numpy()
    return context, past5.iloc[:, cs:ce].to_numpy()


def validate_causal_features(datasets, prices, mi):
    """Verify diagnostic volatility is the raw-unit version of actual X features."""
    from cmgm.config import FEAT_ZSCORE_EPS
    raw = prices.values
    daily = pd.DataFrame(raw).pct_change().fillna(0.0)
    train_end = len(datasets["train"].raw_prices)
    xfeatures = np.concatenate([datasets[s].feature_matrix for s in ("train", "val", "test")])
    # feature_builder's documented order; assertions detect a changed pipeline.
    differences = {}
    for window, feature_index in ((5, 5), (20, 7)):
        vol = daily.rolling(window, min_periods=1).std().fillna(0).clip(0, 5).to_numpy(dtype=np.float32)
        mean = vol[:train_end].mean(axis=0, keepdims=True)
        std = np.maximum(vol[:train_end].std(axis=0, keepdims=True), FEAT_ZSCORE_EPS)
        expected = (vol-mean)/std
        difference = float(np.max(np.abs(expected-xfeatures[:, :, feature_index])))
        # float32 reductions over strided feature arrays can round differently.
        np.testing.assert_allclose(expected, xfeatures[:, :, feature_index], rtol=1e-5, atol=1e-5)
        differences[f"vol{window}_standardized_X_max_diff"] = difference
    cut = len(raw)//2
    original, past = causal_timeline(raw, mi)
    changed = raw.copy()
    changed[cut:] *= 1.75
    altered, altered_past = causal_timeline(changed, mi)
    for key in original:
        np.testing.assert_array_equal(original[key][:cut], altered[key][:cut])
    np.testing.assert_array_equal(past[:cut], altered_past[:cut])
    return {**differences, "diagnostic_timeline_future_perturbation_prefix_max_diff": 0.0,
            "future_perturbation_cut": str(prices.index[cut]),
            "scope": "Diagnostic past-return/volatility/dispersion columns; original model forward unchanged"}


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index][:2]
        return index, x, y


def latent_context(branch):
    p = branch.last_regime_probabilities.detach().cpu().double().numpy()
    prior = branch.last_regime_priors.detach().cpu().double().numpy()
    z = branch.last_latent_states.detach().cpu().double().numpy()
    move = np.abs(np.diff(p, axis=1)).sum(-1)
    dz = np.linalg.norm(np.diff(z, axis=1), axis=-1)
    last, pr = p[:, -1], prior[:, -1]
    sort = np.sort(last, axis=-1)
    result = {f"p{k}": last[:, k] for k in range(last.shape[1])}
    result.update(state=last.argmax(-1), confidence=last.max(-1),
        entropy=-(last * np.log(np.maximum(last, 1e-12))).sum(-1),
        margin=sort[:, -1] - sort[:, -2], movement=move[:, -1],
        transition_score=move[:, -5:].mean(-1), transition_max=move[:, -5:].max(-1),
        kl_surprise=(last * np.log(np.maximum(last, 1e-12) / np.maximum(pr, 1e-12))).sum(-1),
        l1_surprise=np.abs(last - pr).sum(-1), delta_z=dz[:, -1], mean_delta_z5=dz[:, -5:].mean(-1))
    return result


def load_model(label, path, mi, device):
    payload = checkpoint_payload(path)
    metadata = payload.get("metadata", payload.get("checkpoint_metadata", {}))
    recorded = metadata.get("variant", payload.get("variant"))
    if recorded is not None and recorded != VARIANTS[label]:
        raise ValueError(f"Checkpoint variant mismatch: {recorded} != {VARIANTS[label]}")
    model = HeteroMixHopCMGM(mi["commodity"][1], mi["commodity"][1] - mi["commodity"][0],
        n_stock=mi["stock"][1], n_bond=mi["bond"][1]-mi["bond"][0],
        feat_dim=FEATURE_DIM, variant=VARIANTS[label])
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    result = {"variant": VARIANTS[label], "checkpoint_path": str(path.resolve()),
        "best_epoch": payload.get("best_epoch", metadata.get("best_epoch")),
        "seed": metadata.get("seed", payload.get("seed")),
        "checkpoint_git_sha": metadata.get("git_sha", payload.get("git_sha")),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "file_sha256_before": sha256(path), "state_sha256_before": state_checksum(model),
        "variant_metadata_available": recorded is not None,
        "missing_metadata_policy": "null means unavailable; do not infer training seed/SHA"}
    return model, result


def aligned_merge(left, right):
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise AssertionError("Duplicate observation identities")
    left = left.assign(_diagnostic_order=np.arange(len(left)))
    out = left.merge(right, on=KEYS, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_check"))
    assert (out["_merge"] == "both").all(), "Forecast origin/commodity mismatch"
    np.testing.assert_array_equal(out["target"], out["target_check"])
    return out.sort_values("_diagnostic_order").drop(columns=["target_check", "_merge", "_diagnostic_order"]).reset_index(drop=True)


@torch.no_grad()
def collect(model, label, datasets, prices, mi, device):
    idx_5 = MULTI_HORIZONS.index(5)
    cs, ce = mi["commodity"]
    names = np.asarray(prices.columns[cs:ce], dtype=str)
    timeline, past5 = causal_timeline(prices.values, mi)
    records, offset = [], 0
    for split in ("train", "val", "test"):
        ds = datasets[split]
        assert ds.horizons == MULTI_HORIZONS and ds.target_type == "return"
        loader = DataLoader(IndexedDataset(ds), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        for indices, x, y in loader:
            sample_ids = indices.numpy()
            origins = offset + sample_ids + ds.seq_len - 1
            dates = prices.index[origins].strftime("%Y-%m-%d").to_numpy()
            target = y[:, idx_5].numpy()
            expected = (prices.values[origins + 5, cs:ce] / np.maximum(np.abs(prices.values[origins, cs:ce]), 1e-8) - 1).clip(-1, 1).astype(np.float32)
            np.testing.assert_array_equal(target, expected)
            assert np.array_equal(ds.raw_prices[sample_ids + ds.seq_len - 1, cs:ce], prices.values[origins, cs:ce])
            prediction = model(x.to(device)).cpu().numpy()
            assert prediction.shape == y.shape
            prediction = prediction[:, idx_5]
            row = {"split": split, "sample_index": np.repeat(sample_ids, ce-cs),
                "forecast_origin": np.repeat(dates, ce-cs),
                "commodity_index": np.tile(np.arange(ce-cs), len(sample_ids)),
                "commodity": np.tile(names, len(sample_ids)), "target": target.reshape(-1),
                label: prediction.reshape(-1)}
            if label == "D0B":
                row["past5"] = past5[origins].reshape(-1)
                for key, value in timeline.items():
                    row[key] = np.repeat(value[origins], ce-cs)
                for key, value in latent_context(model.switching_latent_transformer).items():
                    row[key] = np.repeat(value, ce-cs)
            records.append(pd.DataFrame(row))
        print(f"[D0B error decomposition] {label} {split.upper()}: {len(ds)} origins × {ce-cs} commodities", flush=True)
        offset += len(ds.raw_prices)
    result = pd.concat(records, ignore_index=True)
    assert not result.duplicated(KEYS).any()
    assert np.isfinite(result.select_dtypes(include=np.number)).all().all()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--d0b-checkpoint", type=Path)
    parser.add_argument("--tempweighted-checkpoint", type=Path)
    parser.add_argument("--allow-missing-comparator", action="store_true",
                        help="Explicitly incomplete D0B-only report; no A–G conclusion")
    parser.add_argument("--prepared-data", type=Path, help="Numeric NPZ cache with verified source/data fingerprints")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    paths = {label: discover_checkpoint(variant, args.checkpoint_dir,
             args.d0b_checkpoint if label == "D0B" else args.tempweighted_checkpoint)
             for label, variant in VARIANTS.items()}
    if paths["D0B"] is None:
        parser.error("Official D0B best checkpoint is missing")
    if paths["TempWeighted"] is None and not args.allow_missing_comparator:
        parser.error("Official temporal_weighted_graph checkpoint not found. Supply --tempweighted-checkpoint. No training fallback.")
    torch.manual_seed(42)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    datasets, prices, mi, fingerprint = prepare_data(args.prepared_data)
    feature_sanity = validate_causal_features(datasets, prices, mi)
    metadata, frames = {}, {}
    for label, path in paths.items():
        if path is None:
            continue
        model, info = load_model(label, path, mi, device)
        frames[label] = collect(model, label, datasets, prices, mi, device)
        info.update(file_sha256_after=sha256(path), state_sha256_after=state_checksum(model))
        assert info["file_sha256_before"] == info["file_sha256_after"]
        assert info["state_sha256_before"] == info["state_sha256_after"]
        info["checkpoint_and_state_unchanged"] = True
        metadata[label] = info
        del model
    observations = frames["D0B"]
    if "TempWeighted" in frames:
        observations = aligned_merge(observations, frames["TempWeighted"])
    output = args.output_dir or ROOT / "experiments/d0b_5d_error_regime" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    from cmgm.scripts.d0b_5d_error_regime_analysis import analyze, write_report, save_results
    result = analyze(observations, output)
    result.update(checkpoints=metadata, data_source_fingerprint=fingerprint,
        diagnostic_git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        implementation_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "cmgm/scripts/d0b_5d_error_regime_analysis.py")},
        device=str(device), causal_feature_sanity=feature_sanity,
        primary_horizon=5, primary_index=MULTI_HORIZONS.index(5),
        fixed_batch_shape=list(next(iter(DataLoader(datasets["test"], batch_size=BATCH_SIZE)))[0].shape),
        alignment={"joined_on": KEYS, "target_exact_equal": True,
            "dates_and_targets_reconstructed_from_raw_prices": True,
            "two_model_identity_assertions": "PASS" if "TempWeighted" in frames else "NOT AVAILABLE: comparator missing",
            "full_population_including_last_train_batch": True},
        status="COMPLETE" if "TempWeighted" in frames else "INCOMPLETE: official TempWeighted checkpoint missing")
    save_results(result, output / "results.json")
    write_report(result, output)
    print(f"[D0B error decomposition] {result['status']}; report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
