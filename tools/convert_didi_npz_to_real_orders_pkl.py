"""
Convert LLM-A-HDRL DiDi NPZ to CoopRide real_orders PKL.

real_orders format: [origin_grid, dest_grid, start_time, duration_slots, price]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def convert(npz_path: Path, base_pkl_path: Path, output_pkl_path: Path, sample_rate: float, seed: int) -> None:
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("sample_rate must be in (0, 1].")

    data = np.load(npz_path, allow_pickle=True)
    slot_offsets = data["slot_offsets"]
    origin = data["origin_grid"]
    dest = data["dest_grid"]
    trip_time = data["trip_time"] if "trip_time" in data.files else None
    fare_amount = data["fare_amount"] if "fare_amount" in data.files else None
    slot_minutes = int(data["slot_minutes"]) if "slot_minutes" in data.files else 10

    with base_pkl_path.open("rb") as f:
        base = pickle.load(f)
    neighbor = base["neighbor"]
    price_param = base["price"]
    duration_param = base["duration"]

    grid_rows = int(data["grid_rows"]) if "grid_rows" in data.files else 11
    grid_cols = int(data["grid_cols"]) if "grid_cols" in data.files else 11
    grid_num = grid_rows * grid_cols

    n_slots = len(slot_offsets) - 1
    order_param = np.zeros((grid_num, grid_num, n_slots), dtype=np.int32)
    real_orders = []

    rng = np.random.default_rng(seed)
    slot_seconds = slot_minutes * 60

    total_before = 0
    total_after = 0

    for slot in range(n_slots):
        start = int(slot_offsets[slot])
        end = int(slot_offsets[slot + 1])
        if end <= start:
            continue
        indices = np.arange(start, end)
        total_before += len(indices)

        if sample_rate < 1.0:
            sample_size = max(1, int(len(indices) * sample_rate))
            indices = rng.choice(indices, size=sample_size, replace=False)
        total_after += len(indices)

        for idx in indices:
            o = int(origin[idx])
            d = int(dest[idx])
            price = float(fare_amount[idx]) if fare_amount is not None else 0.0
            if trip_time is not None:
                duration_slots = max(1, int(np.ceil(float(trip_time[idx]) / slot_seconds)))
            else:
                duration_slots = 1
            real_orders.append([o, d, int(slot), int(duration_slots), price])
            order_param[o, d, slot] += 1

    output = {
        "duration": duration_param,
        "price": price_param,
        "neighbor": neighbor,
        "order": order_param,
        "shape": (grid_rows, grid_cols),
        "real_orders": np.array(real_orders, dtype=np.float64),
        "metadata": {
            "source_npz": str(npz_path),
            "sample_rate": sample_rate,
            "seed": seed,
            "slot_minutes": slot_minutes,
            "total_orders_before": int(total_before),
            "total_orders_after": int(total_after),
        },
    }

    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl_path.open("wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[convert_didi_npz_to_real_orders] total_orders: {total_before} -> {total_after}")
    print(f"[convert_didi_npz_to_real_orders] output: {output_pkl_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="LLM-A-HDRL DiDi NPZ path")
    parser.add_argument("--base-pkl", required=True, help="Base DiDi pkl (for neighbor/price/duration)")
    parser.add_argument("--output", required=True, help="Output PKL with real_orders")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sample rate for orders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    convert(Path(args.npz), Path(args.base_pkl), Path(args.output), args.sample_rate, args.seed)


if __name__ == "__main__":
    main()

