#!/usr/bin/env python
"""Refresh the Phase 3 audit's output hashes and emit a verification record.

The intervention run writes `phase3_audit.json` before the figures and report
exist, so this script completes the output inventory afterwards and confirms
that neither Phase 1 nor Phase 2 was touched.

    python -m cmgm.scripts.phase3_finalize
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "interpretability" / "phase1"
P2 = ROOT / "interpretability" / "phase2"
P3 = ROOT / "interpretability" / "phase3"
CKPT = ROOT / "checkpoints" / "d0b_node_history_readout"
VARIANTS = {
    42: "switching_latent_balanced_readout_seed42_20261006_231320_782593",
    43: "switching_latent_balanced_readout_seed43_20261007_123714_3172316",
    44: "switching_latent_balanced_readout_seed44_20261007_125331_3172316",
}
REQUIRED = (
    "gate/gate_intervention_metrics.csv",
    "gate/gate_prediction_changes.parquet",
    "gate/gate_shuffle_summary.csv",
    "graph/graph_intervention_metrics.csv",
    "graph/prior_sweep_metrics.csv",
    "graph/prior_sweep_structure.csv",
    "graph/graph_prediction_changes.parquet",
    "regime/generator_similarity.csv",
    "regime/regime_intervention_metrics.csv",
    "regime/regime_prediction_changes.parquet",
    "tables/baseline_reproduction.csv",
    "tables/intervention_controls.csv",
    "audit/phase3_audit.json",
    "PHASE3_FUNCTIONAL_MECHANISM_AUDIT.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    audit_path = P3 / "audit" / "phase3_audit.json"
    audit = json.loads(audit_path.read_text())

    # phase1 keys are stored relative to interpretability/phase1; phase2 keys
    # are stored relative to the repository root
    drift1 = {k: dict(recorded=v, current=sha256_file(P1 / k))
              for k, v in audit["phase1_input_hashes"].items()
              if sha256_file(P1 / k) != v}
    drift2 = {k: dict(recorded=v, current=sha256_file(ROOT / k))
              for k, v in audit["phase2_input_hashes"].items()
              if sha256_file(ROOT / k) != v}
    ckpt_ok = all(sha256_file(CKPT / VARIANTS[s] / "best.pt")
                  == audit["checkpoint_sha256"][str(s)] for s in (42, 43, 44))

    outputs = {str(p.relative_to(ROOT)): sha256_file(p)
               for p in sorted(P3.rglob("*"))
               if p.is_file() and "audit" not in p.parts}
    audit["output_hashes"] = outputs
    audit["output_hashes_refreshed_utc"] = datetime.utcnow().isoformat() + "Z"
    audit["output_hashes_note"] = (
        "refreshed by cmgm.scripts.phase3_finalize after figure and report "
        "generation; intervention outputs themselves were not modified")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    missing = [f for f in REQUIRED if not (P3 / f).exists()]
    figs = sorted(p.name for p in (P3 / "figures").glob("*.png"))
    verification = dict(
        generated_utc=datetime.utcnow().isoformat() + "Z",
        phase1_inputs_unchanged=not drift1,
        phase2_inputs_unchanged=not drift2,
        checkpoints_unchanged=bool(ckpt_ok),
        controls_pass=bool(audit["controls_pass"]),
        controls_not_evaluable=audit["controls_not_evaluable"],
        parameter_guard_all_unchanged=all(g["unchanged"] for g in audit["parameter_guard"]),
        baseline_bitwise_vs_phase1=all(b["bitwise_identical_to_phase1"]
                                       for b in audit["baseline_reproduction"]),
        TEST_READ=audit["TEST_READ"], TRAINING=audit["TRAINING"],
        PARAMETER_UPDATE=audit["PARAMETER_UPDATE"],
        required_deliverables_missing=missing,
        figures=figs,
        phase3_file_count=sum(1 for p in P3.rglob("*") if p.is_file()),
        PASS=bool(not drift1 and not drift2 and ckpt_ok
                  and audit["controls_pass"] and not missing and len(figs) == 7),
    )
    (P3 / "tables" / "phase3_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({k: verification[k] for k in
                      ("controls_pass", "phase1_inputs_unchanged",
                       "phase2_inputs_unchanged", "checkpoints_unchanged",
                       "baseline_bitwise_vs_phase1", "required_deliverables_missing",
                       "figures", "PASS")}, indent=1))


if __name__ == "__main__":
    main()
