"""Command-line entry point for the retained CMGM baselines."""

import argparse

from cmgm.config import BATCH_SIZE, CORRELATION_METHOD, NUM_EPOCHS, PATIENCE, RANDOM_SEED, SEQ_LEN
from cmgm.scripts.main_hetero_mixhop import run_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines and HeteroMixHop")
    parser.add_argument("--method", default=CORRELATION_METHOD)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--tag", default="")
    # Keep the historical CLI accepted. The legacy runner set --all=True by
    # default, so these flags never narrowed execution; the unified runner
    # intentionally preserves that effective behavior.
    parser.add_argument("--lr", action="store_true")
    parser.add_argument("--svr", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--bilstm", action="store_true")
    parser.add_argument("--gcn", action="store_true")
    parser.add_argument("--gcn-gat", action="store_true")
    parser.add_argument("--cmgm", action="store_true")
    parser.add_argument("--all", action="store_true", default=True)
    return parser.parse_args()


def main():
    return run_comparison(parse_args())


if __name__ == "__main__":
    main()
