#!/usr/bin/env python
"""Phase 5 report and audit."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "interpretability" / "phase5"
FD = P5 / "frozen_diagnostics"
FC = P5 / "fusion_controls"
SEEDS = (42, 43, 44)
REPRO_FLOOR_PCT = 0.061        # measured in Phase 4 for same-seed retraining


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tbl(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    dim = pd.read_csv(FD / "effective_dimension.csv")
    dyn = pd.read_csv(FD / "stage_dynamics.csv")
    geo = pd.read_csv(FD / "geometry_preservation.csv")
    knn = pd.read_csv(FD / "knn_retention.csv")
    cka = pd.read_csv(FD / "representation_cka.csv")
    pro = pd.read_csv(FD / "linear_probe_metrics.csv")
    ctl = pd.read_csv(FC / "fusion_control_metrics.csv")
    rep = pd.read_csv(FD / "fusion_control_representation_metrics.csv")
    base_h = float(ctl[ctl.model == "Original"].VAL_Huber.iloc[0])

    def dm(stage, col="normalized"):
        return dim[dim.stage == stage][col].mean()

    def dy(stage):
        return dyn[dyn.stage == stage]["mean"].mean()

    def ge(pair):
        return geo[geo.pair == pair].spearman.mean()

    def kn(pair, k=10):
        s = knn[(knn.pair == pair) & (knn.k == k)]
        return s.retention.mean() if len(s) else float("nan")

    def ck(pair):
        s = cka[cka.pair == pair]
        return s.cka.mean() if len(s) else float("nan")

    prim = pro[pro.alpha == 1.0].groupby("representation").probe_Huber.mean()

    L = []
    L.append("# Phase 5 - Information Bottleneck and Aggregation Diagnostics")
    L.append("")
    L.append(f"Generated {datetime.utcnow().isoformat()}Z.")
    L.append("")
    L.append("## 1. Scope")
    L.append("")
    L.append("- **Diagnostic phase, not an architecture search.** Phase 5A is "
             "read-only over the frozen checkpoints and Phase 1 artifacts; Phase 5B "
             "trains three pre-registered fusion controls, nothing else.")
    L.append("- 5A uses seeds 42/43/44 for cross-seed descriptive consistency. "
             "5B is **seed 42 only, n = 1**, so no robustness claim is made there.")
    L.append(f"- TEST_READ = false. TRAINING_PHASE5A = false. "
             f"TRAINING_PHASE5B = true (three controls only).")
    L.append("- Vocabulary: variance and effective dimensionality are **not** "
             "information. Every statement below concerns representation "
             "compression, variation attenuation, geometry distortion or linear "
             "predictive accessibility.")
    L.append("")
    L.append("### 1.1 A premise correction, stated up front")
    L.append("")
    L.append("The Phase 5 brief states that GlobalConcatFusion preserves 128 "
             "dimensions into the head while the Original compresses to 64. **That "
             "is not what the frozen code does.** GlobalConcatFusion computes "
             "`fused = F.gelu(global_concat_fusion(cat([s, t])))` with "
             "`global_concat_fusion = nn.Linear(128, 64)`, so it outputs **64** "
             "dimensions into the **same 64-input head**, at an identical 520549 "
             "parameters. Phase 5B's representation extraction confirms this "
             "directly: GCF's post-fusion width is 64.")
    L.append("")
    L.append("Consequently the dimensional-bottleneck hypothesis (128 vs 64) is "
             "answered before any control is run: GCF wins *while being 64 wide*. "
             "The brief's control C1 is also not an independent test of the width "
             "hypothesis, because C1 is GCF minus the GELU. Both facts are carried "
             "into the interpretation below rather than silently corrected.")
    L.append("")

    L.append("## 2. Motivation")
    L.append("")
    L.append("Phase 3 showed that on the frozen checkpoint the learned gate, the "
             "adaptive graph prior and the Markov router are close to functionally "
             "inert, while both branches individually matter. Phase 4 showed that "
             "removing each of those mechanisms individually costs only 0.10-0.21 "
             "percent (1-3x the 0.061 percent reproducibility floor), that removing "
             "all three costs 2.27 percent, and that GlobalConcatFusion beats the "
             "Original on seed 42 by 1.39 percent and on all three audited seeds. "
             "Phase 5 asks *where* representation structure is lost, and whether "
             "GCF's advantage comes from width or from the fusion operator.")
    L.append("")

    L.append("## 3. Spatial aggregation")
    L.append("")
    rows = [
        ("pre-GNN nodes (flattened)", f"{dm('S_node_prgnn'):.4f}", f"{dy('S_node_prgnn'):.4f}"),
        ("pre-pool nodes (flattened)", f"{dm('S_node_prepool'):.4f}", f"{dy('S_node_prepool'):.5f}"),
        ("stock mean pool", f"{dm('S_pool_stock'):.4f}", f"{dy('S_pool_stock'):.5f}"),
        ("spatial readout h_s", f"{dm('S_hs'):.4f}", f"{dy('S_hs'):.5f}"),
        ("spatial projected s", f"{dm('S_s'):.4f}", f"{dy('S_s'):.5f}"),
    ]
    L.append(tbl(rows, ["Stage", "normalised effective dim (PR/D)", "mean step / sqrt(d)"]))
    L.append("")
    L.append(f"Pooling flattens the node tensor from 284x64 to 64. Sample geometry "
             f"survives it well: the Spearman correlation of pairwise sample "
             f"distances between the pre-pooling nodes and h_s is "
             f"{ge('S_node_prepool->S_hs'):.3f}, and k=10 neighbourhood retention is "
             f"{kn('S_node_prepool->S_hs'):.3f}. Linear CKA between the two is "
             f"{ck('S_node_prepool|S_hs'):.3f}.")
    L.append("")
    L.append(f"**Compression is real but geometry-preserving.** Normalised effective "
             f"dimension is low everywhere (0.00-0.02), because sample variation "
             f"already lives in very few directions before pooling; the flattened "
             f"pre-pool tensor has a participation ratio of only "
             f"{dm('S_node_prepool', 'participation_ratio'):.2f} of 18176 "
             f"dimensions. Step dynamics fall from {dy('S_node_prepool'):.5f} at the "
             f"pre-pool nodes to {dy('S_hs'):.5f} at h_s. The `/sqrt(d)` normaliser "
             f"is dimension-dependent, so magnitudes are not comparable across "
             f"stages of different width.")
    L.append("")

    L.append("## 4. Temporal aggregation")
    L.append("")
    rows = [
        ("H_T", f"{dm('T_H'):.4f}", f"{dy('T_H'):.5f}"),
        ("Z_T", f"{dm('T_Z'):.4f}", f"{dy('T_Z'):.5f}"),
        ("[H_T;Z_T] balanced concat", f"{dm('T_cat'):.4f}", f"{dy('T_cat'):.5f}"),
        ("h_t", f"{dm('T_ht'):.4f}", f"{dy('T_ht'):.5f}"),
        ("t", f"{dm('T_t'):.4f}", f"{dy('T_t'):.5f}"),
    ]
    L.append(tbl(rows, ["Stage", "normalised effective dim (PR/D)", "mean step / sqrt(d)"]))
    L.append("")
    L.append(f"The balanced readout does not collapse the temporal representation: "
             f"normalised effective dimension *rises* from "
             f"{dm('T_cat'):.4f} at [H_T;Z_T] to {dm('T_ht'):.4f} at h_t, geometry "
             f"survives with Spearman {ge('T_cat->T_ht'):.3f} and k=10 retention "
             f"{kn('T_cat->T_ht'):.3f}, and CKA is {ck('T_cat|T_ht'):.3f}. The two "
             f"readout halves are themselves similar rather than independent: "
             f"CKA(u_H, u_Z) = {ck('T_uH|T_uZ'):.3f}.")
    L.append("")

    L.append("## 5. Fusion compression")
    L.append("")
    L.append(f"`[s;t]` (128) to `f` (64): normalised effective dimension moves from "
             f"{dm('F1_s_t'):.4f} to {dm('F2_fused'):.4f}, sample geometry is "
             f"essentially untouched (Spearman {ge('F1_s_t->F2_fused'):.4f}, k=10 "
             f"retention {kn('F1_s_t->F2_fused'):.3f}), and CKA is "
             f"{ck('F1_s_t|F2_fused'):.3f}.")
    L.append("")
    L.append("**The fusion compression is the least destructive operation in the "
             "pipeline, not the most.** It is the stage with the highest geometry "
             "retention and the highest CKA. The two branches are individually "
             "quite different (CKA(h_s, h_t) = "
             f"{ck('S_hs|T_ht'):.3f}, CKA(s, t) = {ck('S_s|T_t'):.3f}), so the "
             "fusion is combining dissimilar content, but it does so without "
             "distorting sample geometry.")
    L.append("")

    L.append("## 6. Predictive accessibility")
    L.append("")
    L.append("A ridge regression (alpha = 1.0, pre-registered) is fitted on **TRAIN** "
             "representations and evaluated on **VAL**; it is a diagnostic probe, "
             "not the main model. Re-extracted VAL representations were verified "
             "**bitwise identical** to the Phase 1 artifacts for all three seeds.")
    L.append("")
    order = ["T_ht", "T_Z", "T_cat", "T_H", "S_node_prepool_stock_pca64",
             "S_node_prepool_commodity_pca64", "F2_fused", "F1_s_t", "F0_hs_ht", "S_hs"]
    rows = [(r, f"{prim[r]:.6f}") for r in order if r in prim.index]
    L.append(tbl(rows, ["Representation", "ridge probe Huber on VAL (alpha=1.0)"]))
    L.append("")
    L.append(f"The trained full model reaches {base_h:.5f} on the same split. The "
             f"temporal representations are the linearly accessible ones: h_t "
             f"({prim['T_ht']:.5f}) and Z_T ({prim['T_Z']:.5f}) approach the full "
             f"model, while the spatial readout h_s is an order of magnitude worse "
             f"({prim['S_hs']:.5f}) and drags down every representation containing "
             f"it ([h_s;h_t] {prim['F0_hs_ht']:.5f}, [s;t] "
             f"{prim['F1_s_t']:.5f}, fused {prim['F2_fused']:.5f}).")
    L.append("")
    L.append("**Probe soundness control.** `gcn_proj` is a linear map, so a ridge "
             "probe on `s = gcn_proj(h_s)` can be no more informative than one on "
             "`h_s`; if the two disagreed, the probe would be mis-specified. They "
             "agree closely at every regularization strength (seed 42: 0.0458 for "
             "`h_s` versus 0.0438 for `s` at alpha = 1.0; 0.0067 versus 0.0066 at "
             "alpha = 100), which confirms the probe is not the source of the low "
             "accessibility of the spatial readout.")
    L.append("")
    L.append("**Alpha sensitivity (all reported, none selected on VAL).** Probe "
             "Huber is strongly regularization-dependent, so the pre-registered "
             "alpha = 1.0 figures above substantially understate what is "
             "recoverable. Seed 42, `h_s`: 0.0580 / 0.0576 / 0.0547 / 0.0458 / "
             "0.0243 / 0.0067 for alpha = 0.001 / 0.01 / 0.1 / 1 / 10 / 100. The "
             "ranking across representations is stable under this sweep; the "
             "absolute level is not.")
    L.append("")
    L.append("A caveat that must accompany this table: a linear probe measures only "
             "*linear* accessibility, and the model's head is a two-layer MLP. A "
             "representation can be poorly linearly decodable and still be useful "
             "nonlinearly. The comparison across representations of the same probe "
             "class is the meaningful one.")
    L.append("")

    L.append("## 7. Minimal fusion controls (seed 42, n = 1)")
    L.append("")
    rows = []
    for _, r in ctl.iterrows():
        rows.append((r.model, r.fusion_width, r.has_dynamic_gate, r.additive_mixing,
                     f"{r.VAL_Huber:.8f}", f"{r.improvement_pct:+.2f}%",
                     int(r.parameter_count)))
    L.append(tbl(rows, ["Model", "fusion width", "dynamic gate", "additive mixing",
                        "VAL Huber", "vs Original", "params"]))
    L.append("")
    L.append(f"Read against the {REPRO_FLOOR_PCT}% same-seed reproducibility floor "
             f"measured in Phase 4:")
    L.append("")
    for txt in (
        f"GlobalConcatFusion is best (+1.39%, ~23x floor).",
        f"C2 GatedConcat128 is **indistinguishable from Original** "
        f"({ctl[ctl.model=='GatedConcat128'].improvement_pct.iloc[0]:+.3f}%, far "
        f"inside the floor): preserving the gate at 128 dimensions buys nothing.",
        f"C3 MeanConcat128 is *worse* than Original "
        f"({ctl[ctl.model=='MeanConcat128'].improvement_pct.iloc[0]:+.2f}%) and "
        f"worse than the 64-dimensional mean sum S1: concatenating without the gate "
        f"hurts.",
        f"C1 ConcatLinear64 is the worst model tested "
        f"({ctl[ctl.model=='ConcatLinear64'].improvement_pct.iloc[0]:+.2f}%): a "
        f"learned 128-to-64 linear mixing with no nonlinearity is worse than simply "
        f"summing the branches.",
        f"S1 mean sum is {ctl[ctl.model=='S1-FixedMeanFusion'].improvement_pct.iloc[0]:+.2f}%, "
        f"about 2x the floor.",
    ):
        L.append(f"- {txt}")
    L.append("")
    L.append("C1 is GCF minus its GELU: with no activation between the concat "
             "projection and the head's first layer, C1's fusion collapses into a "
             "single linear map. The contrast between C1 (-1.51%) and GCF (+1.39%) "
             "is therefore the effect of adding one nonlinearity to the fusion, and "
             "it is the largest single contrast in Phase 5B.")
    L.append("")
    L.append("Representation diagnostics per control (pre-fusion geometry to "
             "post-fusion geometry, Spearman):")
    L.append("")
    gm_ = json.loads((FC / "representation_meta.json").read_text())
    rows = []
    for m, v in gm_.items():
        pr = rep[(rep["model"] == m) & (rep["stage"] == "pre_fusion")]["participation_ratio"]
        rows.append((m, v["pre_width"], v["post_width"],
                     f"{float(pr.iloc[0]):.4f}" if len(pr) else "-"))
    L.append(tbl(rows, ["Model", "pre width", "post width", "pre participation ratio"]))
    L.append("")
    gm = json.loads((FC / "representation_meta.json").read_text())
    rows = [(m, f"{v['geometry_spearman']:.4f}") for m, v in gm.items()]
    L.append(tbl(rows, ["Model", "geometry Spearman, pre to post fusion"]))
    L.append("")
    L.append("**Geometry preservation does not track performance.** C3 preserves "
             "sample geometry perfectly (1.000, it is a pure rescaling) and loses "
             "0.79 percent, while GCF distorts it most (0.952) and wins 1.39 "
             "percent. Compression that perturbs geometry is not the failure mode.")
    L.append("")

    L.append("## 8. Dimensional versus operator bottleneck")
    L.append("")
    L.append("**Dimensional bottleneck: not supported.** GCF is itself 64 wide and "
             "matches the Original's parameter count, and C2, which does carry 128 "
             "dimensions into the head, is statistically indistinguishable from the "
             "Original. Width is not the lever.")
    L.append("")
    L.append("**Operator bottleneck: partially supported, with a specific "
             "mechanism.** The Original's fusion is channel-wise: output channel j "
             "is a convex combination of t_j and s_j only, so it can never mix "
             "channel j of one branch with channel k of the other. GCF's "
             "`Linear(128->64)` mixes everything. But mixing alone is not enough: "
             "C1 mixes channels linearly and performs worst of all. Only GCF mixes "
             "channels **and** applies a nonlinearity, and only GCF wins. The "
             "evidence therefore points to a *nonlinear cross-channel* fusion, not "
             "to width and not to additive mixing as such.")
    L.append("")

    L.append("## 9. Pooling hypothesis")
    L.append("")
    L.append(f"**Spatial pooling bottleneck: partially supported, and not via "
             f"geometry.** Pooling preserves sample geometry (Spearman "
             f"{ge('S_node_prepool->S_hs'):.3f}, k=10 retention "
             f"{kn('S_node_prepool->S_hs'):.3f}, CKA "
             f"{ck('S_node_prepool|S_hs'):.3f}) and preserves normalised effective "
             f"dimension, yet h_s is the least linearly accessible representation "
             f"measured ({prim['S_hs']:.5f} versus {prim['T_ht']:.5f} for h_t). So "
             f"what pooling destroys is not geometry or dimensionality but simple "
             f"linear decodability of the target direction. Because the head is "
             f"nonlinear, that need not make h_s useless, and Phase 3's spatial-only "
             f"condition reaching about 0.0019 shows the spatial branch is not "
             f"discardable.")
    L.append("")

    L.append("## 10. Limitations")
    L.append("")
    for t in ("Phase 5B is n = 1 (seed 42). Individual control differences are "
              "reported against the 0.061% reproducibility floor, not as "
              "significance, and none of them establishes robustness.",
              "Linear probes measure only linear accessibility; they are not "
              "information-theoretic and cannot see nonlinearly encoded content.",
              "Variance and effective dimensionality are not information; they "
              "describe dispersion, not content.",
              "All representation diagnostics are observational. No causal claim "
              "about the model or any market follows from them.",
              "TRAIN and VAL only. TEST was never read.",
              "The frozen model source files were modified by another party during "
              "this session (a new `lag_factor_fusion` variant and optional "
              "`return_memory`/`return_live_probabilities` arguments). The changes "
              "only add variants and default-off options, and the Phase 5A VAL "
              "re-extraction remained bitwise identical to Phase 1, so no Phase 5 "
              "number is affected; the source-hash record is nonetheless not "
              "comparable to the Phase 1 record."):
        L.append(f"- {t}")
    L.append("")
    L.append("## 11. Answers to the five headline questions")
    L.append("")
    L.append(f"**A. Spatial pooling** (284x64 to h_s^64): loses almost no geometry "
             f"(Spearman {ge('S_node_prepool->S_hs'):.3f}) and no normalised "
             f"effective dimension, but attenuates step dynamics from "
             f"{dy('S_node_prepool'):.5f} to {dy('S_hs'):.5f} and produces the least "
             f"linearly accessible representation in the pipeline.")
    L.append("")
    L.append(f"**B. Temporal compression** ([H_T;Z_T] to h_t): mild. Effective "
             f"dimension rises, geometry is retained (Spearman "
             f"{ge('T_cat->T_ht'):.3f}), and h_t is the most linearly accessible "
             f"representation measured.")
    L.append("")
    L.append(f"**C. Final fusion** ([s;t]_128 to f_64): **not** the worst bottleneck. "
             f"It has the highest geometry retention ({ge('F1_s_t->F2_fused'):.4f}) "
             f"and the highest CKA ({ck('F1_s_t|F2_fused'):.3f}) of any stage.")
    L.append("")
    L.append("**D. Controls**: see the table in section 7. GCF +1.39%, Original 0, "
             "C2 -0.01%, S1 -0.12%, C3 -0.79%, C1 -1.51%.")
    L.append("")
    L.append("**E. Final diagnosis**: option 4 is best supported (the fusion "
             "operator, specifically the absence of a nonlinear cross-channel "
             "mixing), with option 1 partially supported (spatial pooling degrades "
             "linear accessibility even though it preserves geometry). Option 3 "
             "(64-dimensional width) is contradicted, since the best model is "
             "itself 64 wide. Option 2 is not supported.")
    L.append("")
    L.append("![Spatial pipeline](figures/fig01_spatial_pipeline.png)")
    L.append("")
    L.append("![Temporal pipeline](figures/fig02_temporal_pipeline.png)")
    L.append("")
    L.append("![Geometry retention](figures/fig03_geometry_retention.png)")
    L.append("")
    L.append("![Probe accessibility](figures/fig04_probe_accessibility.png)")
    L.append("")
    L.append("![Fusion controls](figures/fig05_fusion_controls.png)")
    L.append("")
    L.append("![Width and operator map](figures/fig06_width_operator_map.png)")
    L.append("")
    (P5 / "PHASE5_INFORMATION_BOTTLENECK_DIAGNOSTICS.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")

    # ---- audit
    p1 = json.loads((ROOT / "interpretability" / "phase1" / "seed42" /
                     "audit.json").read_text())
    phase1_now = {f: sha(ROOT / f) for f in
                  ("cmgm/models/switching_latent_transformer.py",
                   "cmgm/models/hetero_mixhop_model.py")}
    a5a = json.loads((P5 / "audit" / "phase5a_audit.json").read_text())
    new_runs = []
    for d in sorted((ROOT / "experiments" / "phase5b_controls").glob("*_seed*")):
        a = json.loads((d / "audit.json").read_text())
        ck = Path(a["checkpoint"])
        new_runs.append(dict(variant=a["variant"], seed=a["seed"],
                             run_dir=str(d.relative_to(ROOT)),
                             checkpoint=str(ck.relative_to(ROOT)),
                             checkpoint_sha256=a["checkpoint_sha256"],
                             best_epoch=a["best_epoch"],
                             checkpoint_is_argmin=a["checkpoint_is_argmin"],
                             best_val_huber=a["best_val_huber"],
                             parameter_count=a["parameter_count"],
                             history_sha256=a["history_sha256"]))
    audit = dict(
        phase=5, generated_utc=datetime.utcnow().isoformat() + "Z",
        seeds_5a=list(SEEDS), seed_5b=42,
        TEST_READ=False, TRAINING_PHASE5A=False, TRAINING_PHASE5B=True,
        phase5a=dict(protocol="read-only over frozen checkpoints and Phase 1 artifacts",
                     probe_model="ridge regression, closed form",
                     probe_alpha_primary=1.0, probe_alpha_sensitivity=[0.1, 1.0, 10.0],
                     probe_fit="TRAIN representations; evaluated on VAL",
                     pca_dims=[64, 128], pca_fit="TRAIN only",
                     val_extraction_bitwise_identical_to_phase1=
                         a5a.get("val_extraction_verified"),
                     val_extraction_checks=a5a.get("val_extraction_bitwise_checks"),
                     phase1_unchanged=a5a.get("phase1_unchanged")),
        phase5b=dict(preregistered_variants=[r["variant"] for r in new_runs],
                     run_count=len(new_runs), single_seed=True,
                     new_runs=new_runs,
                     reused_references=["Original seed42", "S1-FixedMeanFusion "
                                        "(Phase 4 seed42)", "GlobalConcatFusion seed42"]),
        premise_correction="GlobalConcatFusion outputs 64 dimensions through "
                           "Linear(128,64)+GELU and the shared 64-input head; it does "
                           "not preserve 128 dimensions",
        reproducibility_floor_pct=REPRO_FLOOR_PCT,
        source_drift=dict(
            note="frozen model sources were modified by another party mid-session",
            phase1_recorded_switching_latent_transformer=
                p1["recorded_source_hashes"]["cmgm/models/switching_latent_transformer.py"],
            current_switching_latent_transformer=phase1_now["cmgm/models/switching_latent_transformer.py"],
            phase1_recorded_hetero_mixhop=
                p1["recorded_source_hashes"]["cmgm/models/hetero_mixhop_model.py"],
            current_hetero_mixhop=phase1_now["cmgm/models/hetero_mixhop_model.py"],
            impact="changes add new variants and default-off return flags only; "
                   "Phase 5A VAL re-extraction stayed bitwise identical to Phase 1"),
        output_hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(P5.rglob("*"))
                       if p.is_file() and "audit" not in p.parts},
        figures=sorted(p.name for p in (P5 / "figures").glob("*.png")),
    )
    audit["PASS"] = bool(
        len(new_runs) == 3
        and all(r["checkpoint_is_argmin"] for r in new_runs)
        and a5a.get("val_extraction_verified")
        and audit["phase5a"]["phase1_unchanged"]
        and len(audit["figures"]) == 6)
    (P5 / "audit").mkdir(parents=True, exist_ok=True)
    (P5 / "audit" / "phase5_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report + audit written; PASS =", audit["PASS"])


if __name__ == "__main__":
    main()
