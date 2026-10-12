#!/usr/bin/env python
"""Why does the recomputed data fingerprint differ from the recorded one?

The audited runs recorded a SHA256 over the concatenated bytes of
``feature_matrix``, ``prices`` and ``raw_prices`` for the train and val
splits.  This extraction host recomputes a different value even though the
input CSVs, the pipeline source and the split sizes are unchanged, and even
though the frozen model's own VAL selection loss reproduces exactly.

This script rebuilds the data once and re-hashes it under plausible
byte-level representations (signed-zero normalisation, float64 widening,
field orderings, field subsets) to test whether the recorded digest is
recoverable.  It never modifies data or model code.

    python -m cmgm.scripts.phase1_data_fingerprint_diagnostic
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cmgm import config
from cmgm.scripts.main_ablation import build_data

ROOT = Path(__file__).resolve().parents[2]
RECORDED = {}
for seed, name in ((42, "switching_latent_balanced_readout_seed42_20261006_231320_782593"),):
    result = json.loads((ROOT / "experiments" / "d0b_node_history_readout" / name
                         / "result.json").read_text())
    RECORDED = {k: v["sha256"] for k, v in
                result["model"]["data_fingerprint"]["splits"].items()}

FIELDS = ("feature_matrix", "prices", "raw_prices")


def transforms():
    yield "plain", lambda x: x
    yield "negzero_normalised", lambda x: np.where(
        x == 0, np.float32(0.0), x).astype(x.dtype, copy=False)
    yield "float64", lambda x: x.astype(np.float64)
    yield "float32", lambda x: x.astype(np.float32, copy=False)


def digest(dataset, fields, tf):
    """tf may be a single callable or a {field: callable} mapping."""
    h = hashlib.sha256()
    for f in fields:
        v = getattr(dataset, f, None)
        if v is None:
            return None
        fn = tf[f] if isinstance(tf, dict) else tf
        h.update(f.encode())
        h.update(np.ascontiguousarray(fn(v)).tobytes())
    return h.hexdigest()


def main() -> None:
    data = build_data(SimpleNamespace(batch_size=config.BATCH_SIZE,
                                      seq_len=config.SEQ_LEN, seed=42))
    for split in ("train", "val"):
        ds = data["loaders"][split].dataset
        recorded = RECORDED[split]
        print(f"=== {split}: recorded {recorded}")
        for f in FIELDS:
            v = getattr(ds, f)
            negz = int(np.count_nonzero((v == 0) & np.signbit(v)))
            print(f"    {f}: shape {v.shape} dtype {v.dtype} "
                  f"C_CONTIGUOUS={v.flags['C_CONTIGUOUS']} negative_zeros={negz}")
        matches = []
        # (a) uniform transform, any field order, any non-empty subset
        for size in (3, 2, 1):
            for fields in itertools.permutations(FIELDS, size):
                for name, tf in transforms():
                    if digest(ds, fields, tf) == recorded:
                        matches.append((fields, name))
        # (b) a different transform per field, full field set, any order
        named = dict(transforms())
        for combo in itertools.product(named, repeat=len(FIELDS)):
            tfd = {f: named[c] for f, c in zip(FIELDS, combo)}
            for fields in itertools.permutations(FIELDS):
                if digest(ds, fields, tfd) == recorded:
                    matches.append((fields, tfd))
        print(f"    combined-digest reproductions: "
              f"{[str(m) for m in matches] if matches else 'NONE'}")
        # per-field digests, so a future reader can compare field by field
        for f in FIELDS:
            v = getattr(ds, f)
            print(f"    per-field sha256 {f}: "
                  f"{hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}")
    print()
    print("Interpretation: a match on the combined digest identifies the exact "
          "byte-level representation the audited run hashed. No match means the "
          "difference is environmental and the recorded per-field arrays were "
          "never stored, so it cannot be resolved without them.")


if __name__ == "__main__":
    main()
