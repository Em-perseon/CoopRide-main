"""
Sample DiDi order matrix to create a comparable-demand dataset.

Usage:
  python sample_didi_orders.py --input ../data/DiDi_day1_grid121.pkl --output ../data/DiDi_day1_grid121_sample35.pkl --rate 0.35 --seed 42
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def sample_orders(input_path: Path, output_path: Path, rate: float, seed: int) -> None:
    if rate <= 0 or rate > 1:
        raise ValueError("rate must be in (0, 1].")

    with input_path.open("rb") as f:
        data = pickle.load(f)

    order = data["order"].astype(np.int64)
    rng = np.random.default_rng(seed)
    sampled = rng.binomial(order, rate).astype(np.int32)

    total_before = int(order.sum())
    total_after = int(sampled.sum())

    # Preserve required keys and add shape for load_envs_custom.
    out = {
        "duration": data["duration"],
        "price": data["price"],
        "neighbor": data["neighbor"],
        "order": sampled,
        "shape": (11, 11),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[sample_didi_orders] rate={rate}, seed={seed}")
    print(f"[sample_didi_orders] total_orders: {total_before} -> {total_after} ({total_after / max(1, total_before):.3f})")
    print(f"[sample_didi_orders] output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input DiDi pkl file")
    parser.add_argument("--output", type=str, required=True, help="Output sampled pkl file")
    parser.add_argument("--rate", type=float, default=0.35, help="Sampling rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sample_orders(Path(args.input), Path(args.output), args.rate, args.seed)


if __name__ == "__main__":
    main()

