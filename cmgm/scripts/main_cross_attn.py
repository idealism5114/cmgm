#!/usr/bin/env python3
"""
CMGM_CrossAttn — Cross-Attention Fusion between GCN and LSTM.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m cmgm.main_cross_attn
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmgm.config import NUM_EPOCHS, BATCH_SIZE, SEQ_LEN, RANDOM_SEED, PATIENCE
from cmgm.data.data_loader import set_seed, create_data_loaders
from cmgm.graph.graph_builder import build_graph
from cmgm.models.cross_attn_model import CMGM_CrossAttn
from cmgm.training.train import train
from cmgm.training.evaluate import evaluate, evaluate_per_commodity
from cmgm.experiment_logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='volatility_adjusted')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"[Config] Device: {device} | Method: {args.method}")

    # ── Data ──
    data = create_data_loaders(batch_size=args.batch_size, seq_len=args.seq_len)
    graph = build_graph(data['train_returns'], data['market_indices'], method=args.method)
    ei, ew = graph['edge_index'], graph['edge_weight']

    # ── Model ──
    model = CMGM_CrossAttn(data['n_nodes'], data['n_commodities'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CMGM_CrossAttn | Params: {total_params:,}")
    print(f"Fusion: Cross-Attention(LSTM→GCN_steps) → [attended || LSTM]")

    # ── Train ──
    history = train(model, data['train_loader'], data['val_loader'],
                     ei, ew, device, num_epochs=args.epochs, patience=args.patience)

    # ── Evaluate ──
    eval_results = evaluate(model, data['test_loader'], ei, ew, data['scaler'],
                             data['market_indices'], device, compute_ci=True,
                             model_name='CMGM_CrossAttn')

    if not device.type.startswith('cpu'):
        pass  # per-commodity skipped for brevity

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("CMGM_CrossAttn Complete")
    print(f"{'=' * 60}")
    print(f"Test MAE (norm): {eval_results['metrics_norm']['MAE']:.6f}")
    print(f"Test MSE (norm): {eval_results['metrics_norm']['MSE']:.6f}")
    print(f"Best val loss:   {min(history['val_loss']):.6f}")
    print(f"{'=' * 60}")

    logger = ExperimentLogger()
    results = [('CMGM_CrossAttn', history.get('train_time', 0),
                 eval_results['metrics_norm'], eval_results['metrics_orig'])]
    config = {
        'method': args.method, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'seed': args.seed, 'tag': args.tag,
        'version': 'cross_attn',
    }
    logger.log_run(config, results)

    return eval_results


if __name__ == '__main__':
    main()
