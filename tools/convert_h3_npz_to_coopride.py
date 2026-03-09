import argparse
import os
import pickle
from datetime import datetime

import numpy as np

try:
    import h3
except Exception as exc:
    raise SystemExit("h3 is required to build neighbor distances.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-path",
        type=str,
        default="../LLM-A-HDRL/data/nyc_tlc_2015_01_h3.npz",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="../CoopRide-main/data/NYC2015Jan_h3_289.pkl",
    )
    parser.add_argument("--start-date", type=str, default="2015-01-22")
    parser.add_argument("--end-date", type=str, default="2015-01-24")
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--target-max-orders", type=int, default=100)
    return parser.parse_args()


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _choose_shape(n_cells: int) -> tuple[int, int]:
    side = int(np.ceil(np.sqrt(n_cells)))
    rows = side
    cols = int(np.ceil(n_cells / rows))
    return rows, cols


def build_neighbor(res8_h3: list[str], max_distance: int) -> np.ndarray:
    n_cells = len(res8_h3)
    neighbor = np.zeros((n_cells, n_cells), dtype=np.int32)
    for i, src in enumerate(res8_h3):
        neighbor[i, i] = 0
        for j, dst in enumerate(res8_h3):
            if i == j:
                continue
            dist = h3.grid_distance(src, dst)
            if dist is None:
                neighbor[i, j] = max_distance
            else:
                neighbor[i, j] = int(min(dist, max_distance))
    return neighbor


def main() -> None:
    args = parse_args()
    npz_path = os.path.abspath(args.npz_path)
    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as data:
        origin_res8 = data["origin_res8"]
        dest_res8 = data["dest_res8"]
        fare_amount = data["fare_amount"]
        slot_offsets = data["slot_offsets"]
        slot_minutes = int(data["slot_minutes"])
        start_ts = int(data["start_ts"])
        res8_h3 = [str(x) for x in data["res8_h3"].tolist()]
        # 获取行程时间（如果有）
        trip_time = data["trip_time"] if "trip_time" in data else None

    n_cells = len(res8_h3)
    slots_per_day = int(1440 // slot_minutes)
    total_slots = len(slot_offsets) - 1

    start_date = _parse_date(args.start_date).date()
    end_date = _parse_date(args.end_date).date()
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    order_param_sum = np.zeros((n_cells, n_cells, slots_per_day), dtype=np.int64)
    day_count = 0
    
    # 收集真实订单数据 (用于 generate_order=0 模式)
    real_orders_list = []

    start_day_idx = int((start_date - datetime.utcfromtimestamp(start_ts).date()).days)
    end_day_idx = int((end_date - datetime.utcfromtimestamp(start_ts).date()).days)

    # 预先构建邻居距离矩阵（用于计算 duration）
    neighbor_temp = build_neighbor(res8_h3, args.max_distance).astype(np.int32)
    
    for day_idx in range(start_day_idx, end_day_idx + 1):
        day_slot_start = day_idx * slots_per_day
        day_slot_end = day_slot_start + slots_per_day
        if day_slot_start < 0 or day_slot_end > total_slots:
            continue
        for slot_in_day in range(slots_per_day):
            slot_idx = day_slot_start + slot_in_day
            start_idx = int(slot_offsets[slot_idx])
            end_idx = int(slot_offsets[slot_idx + 1])
            if end_idx <= start_idx:
                continue
            origin = origin_res8[start_idx:end_idx].astype(np.int64, copy=False)
            dest = dest_res8[start_idx:end_idx].astype(np.int64, copy=False)
            flat = origin * n_cells + dest
            counts = np.bincount(flat, minlength=n_cells * n_cells)
            order_param_sum[:, :, slot_in_day] += counts.reshape(n_cells, n_cells)
            
            # 收集真实订单: [origin_grid, dest_grid, start_time, duration, price]
            slot_fare = fare_amount[start_idx:end_idx]
            for i in range(len(origin)):
                origin_grid = int(origin[i])
                dest_grid = int(dest[i])
                price = float(slot_fare[i])
                
                # 计算 duration（基于网格距离，单位为 10 分钟步长）
                if trip_time is not None:
                    duration = max(1, int(trip_time[start_idx + i] // 600))  # 转换为 10 分钟单位
                else:
                    # 使用邻居距离作为 duration
                    duration = max(1, int(neighbor_temp[origin_grid, dest_grid]))
                
                # start_time 是一天中的 slot 索引 (0-143)
                real_orders_list.append([origin_grid, dest_grid, slot_in_day, duration, price])
        day_count += 1

    if day_count == 0:
        raise RuntimeError("No slots found in the requested date range.")

    order_param = np.rint(order_param_sum / day_count).astype(np.int32)
    target_max = int(args.target_max_orders)
    if target_max > 0:
        order_num = order_param.sum(axis=1)
        current_max = int(order_num.max()) if order_num.size > 0 else 0
        if current_max > target_max:
            scale = target_max / float(current_max)
            order_param = np.floor(order_param.astype(np.float64) * scale).astype(np.int32)

    neighbor = build_neighbor(res8_h3, args.max_distance).astype(np.int32)

    max_dist = args.max_distance
    price_param = np.zeros((max_dist + 2, 2), dtype=np.float64)
    global_mean = float(np.mean(fare_amount))
    global_std = float(np.std(fare_amount)) if float(np.std(fare_amount)) > 0 else 1.0

    # Compute distance bins for all trips in range
    fare_list = []
    dist_list = []
    for day_idx in range(start_day_idx, end_day_idx + 1):
        day_slot_start = day_idx * slots_per_day
        day_slot_end = day_slot_start + slots_per_day
        if day_slot_start < 0 or day_slot_end > total_slots:
            continue
        start_idx = int(slot_offsets[day_slot_start])
        end_idx = int(slot_offsets[day_slot_end])
        if end_idx <= start_idx:
            continue
        origin = origin_res8[start_idx:end_idx].astype(np.int64, copy=False)
        dest = dest_res8[start_idx:end_idx].astype(np.int64, copy=False)
        dist = neighbor[origin, dest]
        dist = np.minimum(dist, max_dist)
        dist_list.append(dist)
        fare_list.append(fare_amount[start_idx:end_idx])

    if fare_list:
        dist_all = np.concatenate(dist_list)
        fare_all = np.concatenate(fare_list)
        for d in range(max_dist + 1):
            mask = dist_all == d
            if np.any(mask):
                mean = float(np.mean(fare_all[mask]))
                std = float(np.std(fare_all[mask]))
                price_param[d, 0] = mean
                price_param[d, 1] = std if std > 0 else global_std
            else:
                price_param[d, 0] = global_mean
                price_param[d, 1] = global_std
    else:
        price_param[:, 0] = global_mean
        price_param[:, 1] = global_std

    shape = _choose_shape(n_cells)
    
    # 转换真实订单为 numpy 数组
    real_orders = np.array(real_orders_list, dtype=np.float64)
    
    data_param = {
        "neighbor": neighbor,
        "price": price_param,
        "order": order_param,
        "shape": shape,
        "real_orders": real_orders,  # 新增：真实订单数据 [origin, dest, time, duration, price]
    }

    with open(out_path, "wb") as f:
        pickle.dump(data_param, f, protocol=4)

    print(f"Saved: {out_path}")
    print(f"Date range: {args.start_date} to {args.end_date} (days={day_count})")
    print(f"Grid cells: {n_cells}, shape: {shape}, slots/day: {slots_per_day}")
    print(f"Real orders: {len(real_orders)}")
    if len(real_orders) > 0:
        prices = real_orders[:, 4]
        print(f"  Price mean: {np.mean(prices):.2f}, std: {np.std(prices):.2f}")


if __name__ == "__main__":
    main()
