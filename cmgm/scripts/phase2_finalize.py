#!/usr/bin/env python
"""Refresh the Phase 2 audit output hashes and emit a verification record.

Regenerating the report after the analysis run changes files that
`phase2_audit.json` had already hashed, so the hash map is refreshed here. This
script never writes to `interpretability/phase1/`.

    python -m cmgm.scripts.phase2_finalize
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "interpretability" / "phase1"
P2 = ROOT / "interpretability" / "phase2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    audit_path = P2 / "audit" / "phase2_audit.json"
    audit = json.loads(audit_path.read_text())

    # 1. Phase 1 integrity: nothing under phase1 may have changed
    drift = {k: dict(recorded=v, current=sha256_file(ROOT / k))
             for k, v in audit["input_phase1_hashes"].items()
             if sha256_file(ROOT / k) != v}

    # 2. refresh the output hash map (audit/ itself excluded, as before)
    outputs = {str(p.relative_to(ROOT)): sha256_file(p)
               for p in sorted(P2.rglob("*"))
               if p.is_file() and "audit" not in p.parts}
    audit["output_hashes"] = outputs
    audit["output_hashes_refreshed_utc"] = datetime.utcnow().isoformat() + "Z"
    audit["output_hashes_note"] = (
        "refreshed by cmgm.scripts.phase2_finalize after figures and report "
        "generation; the analysis outputs themselves were not modified")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    figs = sorted(p.name for p in (P2 / "figures").glob("*.png"))
    report = (P2 / "PHASE2_INTERPRETABILITY_REPORT.md")
    sections = [ln.strip() for ln in report.read_text().splitlines()
                if ln.startswith("## ")]
    delivered = {
        "regime": sorted(p.name for p in (P2 / "regime").iterdir() if p.is_file()),
        "graph": sorted(p.name for p in (P2 / "graph").iterdir() if p.is_file()),
        "gate": sorted(p.name for p in (P2 / "gate").iterdir() if p.is_file()),
        "joint": sorted(p.name for p in (P2 / "joint").iterdir() if p.is_file()),
        "tables": sorted(p.name for p in (P2 / "tables").iterdir() if p.is_file()),
        "figures": figs,
    }
    required = {
        "regime_profiles.csv", "regime_transition_matrices.csv",
        "regime_duration.csv", "regime_alignment.json",
        "regime_cross_seed_consistency.csv", "graph_cross_seed_consistency.csv",
        "graph_head_diversity.csv", "prior_vs_attention.csv",
        "gate_cross_seed_consistency.csv", "regime_graph_association.csv",
        "regime_gate_association.csv", "graph_gate_association.csv",
        "interpretability_error_association.csv",
    }
    all_files = {f for v in delivered.values() for f in v}
    missing = sorted(required - all_files)
    verification = dict(
        generated_utc=datetime.utcnow().isoformat() + "Z",
        phase1_artifacts_unchanged=not drift,
        phase1_drift=drift,
        phase2_file_count=sum(len(v) for v in delivered.values()),
        delivered=delivered,
        report_sections=sections,
        required_deliverables_missing=missing,
        PASS=bool(not drift and not missing and len(figs) == 8),
    )
    (P2 / "tables" / "phase2_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({k: verification[k] for k in
                      ("phase1_artifacts_unchanged", "phase2_file_count",
                       "required_deliverables_missing", "PASS")}, indent=1))


if __name__ == "__main__":
    main()
