"""
Strict nested-model verification:
  comm_output_residual's base path must EXACTLY reproduce the original
  edge_attn model given the SAME checkpoint / weights / batch.

Pipeline:
  1. Train edge_attn briefly and save its checkpoint
  2. Build comm_output_residual, load the SAME checkpoint (strict=False —
     only the residual_head/residual_alpha are missing, they do not
     participate in the base path)
  3. Same batch, eval() + no_grad():
       original edge_attn  → pred_original
       comm_output_residual → base_pred  (alpha-independent)
  4. Report max_abs_diff / mean_abs_diff / MAE_original / MAE_base
     PASS threshold: max_abs_diff < 1e-6

If FAIL, the script dumps per-module intermediate tensors for both
models so the divergence point can be located.

Run:  python scripts/verify_nested.py [--epochs 2]
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from cmgm.config import FEATURE_DIM, TARGET_TYPE
from cmgm.data.data_loader import set_seed
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.training.train import train
from cmgm.scripts.main_ablation import build_data

CKPT = 'experiments/regime_analysis/edge_attn_chk.pt'


def collect_intermediates(model, x, variant):
    """Manual forward that records every stage (for divergence location)."""
    B, T, N, _ = x.shape
    n_stock, n_bond = model.n_stock, model.n_bond
    fut = n_stock + n_bond
    out = {}

    A = model.graph_learner() if model.use_learn_graph else model.static_A
    x_gcn = x.mean(dim=0).permute(1, 0, 2)
    x_proj = model.type_proj(x_gcn, n_stock, n_bond)
    x_proj = x_proj.mean(dim=1)
    h1 = torch.relu(model.attn_mixhop1(x_proj, A))
    h2 = model.attn_mixhop2(h1, A)
    h = model.gcn_norm(h2)
    out['h'] = h
    h_global = model.type_pool(h)
    gcn_out = h_global.unsqueeze(0).expand(B, -1)
    out['gcn_out'] = gcn_out

    x_seq = x.reshape(B, T, -1)
    lstm_out, (h_n, _) = model.temporal(x_seq)
    lstm_out = h_n[-1]
    out['lstm_out'] = lstm_out

    combined = torch.cat([gcn_out, lstm_out], dim=-1)
    gate = torch.sigmoid(model.gate_fc(combined))
    fused = gate * model.lstm_proj(lstm_out) + (1 - gate) * model.gcn_proj(gcn_out)
    out['fused'] = fused

    base_pred = model.head(fused)
    if model.n_horizons > 1:
        base_pred = base_pred.view(B, model.n_horizons, model.n_commodities)
    else:
        base_pred = base_pred.view(B, model.n_commodities)
    out['base_pred'] = base_pred

    # residual part (comm_output_residual only)
    if variant == 'comm_output_residual':
        h_comm = h[fut:].unsqueeze(0).expand(B, -1, -1)
        residual = model.residual_head(h_comm)
        if model.n_horizons > 1:
            residual = residual.permute(0, 2, 1)
        else:
            residual = residual.squeeze(-1)
        out['residual'] = residual
        out['final'] = base_pred + model.residual_alpha * residual
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  verify epochs: {args.epochs}")

    # ── 1. Data (same pipeline as main_ablation) ──
    dargs = argparse.Namespace(batch_size=64, seq_len=20)
    data = build_data(dargs, FEATURE_DIM)
    loaders = data['loaders']
    n_stock = data['market_indices']['stock'][1] - data['market_indices']['stock'][0]
    n_bond = data['market_indices']['bond'][1] - data['market_indices']['bond'][0]

    # ── 2. Train edge_attn briefly and save checkpoint ──
    set_seed(args.seed)
    model_orig = HeteroMixHopCMGM(data['n_nodes'], data['n_commodities'],
                                  n_stock=n_stock, n_bond=n_bond,
                                  variant='edge_attn').to(device)
    dummy = torch.empty(2, 0, dtype=torch.long), torch.zeros(0)
    train(model_orig, loaders['train'], loaders['val'],
          dummy[0], dummy[1], device, num_epochs=args.epochs, patience=100)
    torch.save(model_orig.state_dict(), CKPT)
    print(f"[Saved] edge_attn checkpoint → {CKPT}")

    # ── 3. Build comm_output_residual, load SAME checkpoint ──
    model_res = HeteroMixHopCMGM(data['n_nodes'], data['n_commodities'],
                                 n_stock=n_stock, n_bond=n_bond,
                                 variant='comm_output_residual').to(device)
    missing, unexpected = model_res.load_state_dict(
        torch.load(CKPT, map_location=device), strict=False)
    print(f"[Load] missing={sorted(missing)}  unexpected={sorted(unexpected)}")
    assert all('residual' in k for k in missing), \
        f"unexpected missing keys beyond residual modules: {missing}"

    # ── 4. Same batch, eval + no_grad comparison ──
    model_orig.eval()
    model_res.eval()
    max_diff, mean_diff, max_rel = 0.0, 0.0, 0.0
    maes_o, maes_b = [], []
    all_orig, all_base = [], []
    n_batches = 0
    with torch.no_grad():
        for batch in loaders['test']:
            xb, yb = batch[0].to(device), batch[1]
            p_orig = model_orig(xb)
            p_res = model_res(xb)
            p_base = model_res.last_base_pred
            assert p_orig.shape == p_base.shape, (p_orig.shape, p_base.shape)

            diff = (p_orig - p_base).abs()
            max_diff = max(max_diff, diff.max().item())
            mean_diff += diff.mean().item()
            denom = p_orig.abs().clamp(min=1e-8)
            max_rel = max(max_rel, (diff / denom).max().item())

            y_np = yb.numpy()
            p_orig_np = p_orig.cpu().numpy()
            p_base_np = p_base.cpu().numpy()
            if y_np.ndim == 3:  # extract primary horizon from both
                from cmgm.config import MULTI_HORIZONS, TARGET_HORIZON
                h_idx = MULTI_HORIZONS.index(TARGET_HORIZON)
                y_np = y_np[:, h_idx, :]
                p_orig_np = p_orig_np[:, h_idx, :]
                p_base_np = p_base_np[:, h_idx, :]
            maes_o.append(np.mean(np.abs(y_np - p_orig_np)))
            maes_b.append(np.mean(np.abs(y_np - p_base_np)))
            all_orig.append(p_orig_np)
            all_base.append(p_base_np)
            n_batches += 1

    mean_diff /= n_batches
    mae_orig = float(np.mean(maes_o))
    mae_base = float(np.mean(maes_b))
    ao = np.concatenate(all_orig)
    ab = np.concatenate(all_base)

    print("\n" + "=" * 60)
    print("NESTED-MODEL VERIFICATION (same checkpoint)")
    print("=" * 60)
    print(f"  max_abs_diff   = {max_diff:.3e}")
    print(f"  mean_abs_diff  = {mean_diff:.3e}")
    print(f"  max_rel_diff   = {max_rel:.3e}")
    print(f"  MAE_original   = {mae_orig:.6f}")
    print(f"  MAE_base       = {mae_base:.6f}")
    print(f"  pred stats (original): mean={ao.mean():.6f} std={ao.std():.6f} "
          f"min={ao.min():.6f} max={ao.max():.6f}")
    print(f"  pred stats (base)    : mean={ab.mean():.6f} std={ab.std():.6f} "
          f"min={ab.min():.6f} max={ab.max():.6f}")
    ok = max_diff < 1e-6
    print(f"  RESULT         = {'PASS ✅' if ok else 'FAIL ❌'}")
    print("=" * 60)

    # ── 5. If FAIL: locate divergence per module ──
    if not ok:
        print("\n[Divergence location — per-module comparison]")
        with torch.no_grad():
            xb = next(iter(loaders['test']))[0][:8].to(device)
            o = collect_intermediates(model_orig, xb, 'edge_attn')
            r = collect_intermediates(model_res, xb, 'comm_output_residual')
            for key in o:
                if key in r:
                    d = (o[key] - r[key]).abs().max().item()
                    flag = "OK " if d < 1e-6 else "DIFF"
                    print(f"  {key:<14s} {flag}  max_abs_diff={d:.3e}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
