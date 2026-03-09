"""
将 NPZ 数据转换为 CoopRide 的 real_orders 格式
real_orders 格式: [[origin_grid, dest_grid, start_time, duration, price], ...]
"""
import numpy as np
import pickle
import sys
import os
sys.path.append('../')

def convert_npz_to_real_orders(npz_path, output_pkl_path, start_slot=3600, end_slot=3744):
    """
    从 NPZ 数据生成包含真实价格的 real_orders 列表
    
    Args:
        npz_path: NPZ 数据文件路径
        output_pkl_path: 输出的 PKL 文件路径（包含 real_orders）
        start_slot: 起始 slot（Jan 26 = 3600）
        end_slot: 结束 slot（Jan 26 = 3744）
    """
    print(f"Loading NPZ data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # 获取 slot_offsets
    slot_offsets = data['slot_offsets']
    origin_res8 = data['origin_res8']
    dest_res8 = data['dest_res8']
    fare_amount = data['fare_amount']
    trip_time = data.get('trip_time', None)
    
    # 需要网格映射（从 res8 H3 索引到 CoopRide 网格 ID）
    # 这里假设使用已有的 PKL 文件中的网格映射
    existing_pkl_path = "../data/NYC2015Jan26_h3_289_aligned.pkl"
    if os.path.exists(existing_pkl_path):
        with open(existing_pkl_path, 'rb') as f:
            existing_data = pickle.load(f)
        
        # 获取网格映射（从 res8 索引到网格 ID）
        # 需要从 order_param 推断映射关系
        order_param = existing_data['order']
        grid_num = order_param.shape[0]
        
        # 构建 res8 到网格 ID 的映射
        # 这里需要知道 res8_h3 的索引如何映射到网格
        res8_h3 = data.get('res8_h3', None)
        if res8_h3 is None:
            print("Warning: res8_h3 not found, using direct mapping")
            res8_to_grid = {}  # 需要从其他地方获取
        else:
            # 简化：假设 res8 索引直接对应网格 ID（需要根据实际情况调整）
            unique_res8 = np.unique(res8_h3)
            res8_to_grid = {res8: idx for idx, res8 in enumerate(unique_res8)}
    else:
        print(f"Warning: {existing_pkl_path} not found, cannot get grid mapping")
        return
    
    # 收集 Jan 26 的订单
    real_orders = []
    
    for slot_idx in range(start_slot, end_slot):
        if slot_idx >= len(slot_offsets) - 1:
            break
        
        start_idx = int(slot_offsets[slot_idx])
        end_idx = int(slot_offsets[slot_idx + 1])
        
        if end_idx <= start_idx:
            continue
        
        # 获取该 slot 的订单
        slot_origin = origin_res8[start_idx:end_idx]
        slot_dest = dest_res8[start_idx:end_idx]
        slot_fare = fare_amount[start_idx:end_idx]
        slot_time = trip_time[start_idx:end_idx] if trip_time is not None else None
        
        # 转换为 real_orders 格式
        for i in range(len(slot_origin)):
            origin_res8_idx = slot_origin[i]
            dest_res8_idx = slot_dest[i]
            price = float(slot_fare[i])
            
            # 映射到网格 ID
            origin_grid = res8_to_grid.get(origin_res8_idx, -1)
            dest_grid = res8_to_grid.get(dest_res8_idx, -1)
            
            if origin_grid < 0 or dest_grid < 0:
                continue  # 跳过无法映射的订单
            
            # 计算 duration（以 10 分钟为单位）
            if slot_time is not None:
                duration = max(1, int(slot_time[i] // 600))  # 转换为 10-min 单位
            else:
                duration = 1  # 默认值
            
            # start_time 是 slot 在一天中的索引 (0-143)
            start_time = slot_idx % 144
            
            # real_orders 格式: [origin_grid, dest_grid, start_time, duration, price]
            real_orders.append([int(origin_grid), int(dest_grid), int(start_time), int(duration), float(price)])
    
    print(f"Generated {len(real_orders)} real orders with actual prices")
    print(f"Price statistics:")
    prices = [order[4] for order in real_orders]
    print(f"  Mean: {np.mean(prices):.2f}")
    print(f"  Median: {np.median(prices):.2f}")
    print(f"  Std: {np.std(prices):.2f}")
    
    # 保存到 PKL 文件
    output_data = {
        'real_orders': np.array(real_orders),
        'metadata': {
            'source': npz_path,
            'start_slot': start_slot,
            'end_slot': end_slot,
            'total_orders': len(real_orders),
            'price_mean': float(np.mean(prices)),
            'price_std': float(np.std(prices))
        }
    }
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Saved to: {output_pkl_path}")
    return real_orders


if __name__ == '__main__':
    npz_path = "../../LLM-A-HDRL/data/nyc_tlc_2015_01_h3.npz"
    output_pkl_path = "../data/NYC2015Jan26_real_orders.pkl"
    
    convert_npz_to_real_orders(npz_path, output_pkl_path, start_slot=3600, end_slot=3744)

