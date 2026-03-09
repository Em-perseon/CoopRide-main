"""
Test script: Verify CoopRide correctly uses real order prices
"""
import sys
import os
sys.path.append('../')
import numpy as np
import pickle

# Test 1: Check if data file contains real_orders
print("=" * 60)
print("Test 1: Check data file")
print("=" * 60)

data_path = "../data/NYC2015Jan26_h3_289_real_orders.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f"数据文件: {data_path}")
print(f"包含的字段: {list(data.keys())}")

if 'real_orders' in data:
    real_orders = data['real_orders']
    print(f"\n[OK] real orders count: {len(real_orders)}")
    prices = real_orders[:, 4]
    print(f"  price mean: {np.mean(prices):.2f}")
    print(f"  price std: {np.std(prices):.2f}")
    print(f"  price range: [{np.min(prices):.2f}, {np.max(prices):.2f}]")
else:
    print("[FAIL] No real_orders in data file!")

# Test 2: Load env and check day_orders
print("\n" + "=" * 60)
print("Test 2: Load env (with real orders)")
print("=" * 60)

from load_data import load_envs_custom

env, M, N, _, grid_num = load_envs_custom(data_path, driver_num=2000, use_real_orders=True)

print(f"Grid num: {grid_num}")
print(f"env.real_orders length: {len(env.real_orders)}")
print(f"env.day_orders length: {len(env.day_orders)}")

if len(env.day_orders) > 0:
    total_day_orders = sum(len(slot) for slot in env.day_orders)
    print(f"day_orders total orders: {total_day_orders}")
    
    # Check prices
    all_prices = []
    for slot_orders in env.day_orders:
        for order in slot_orders:
            # order format: [start_node, end_node, start_time, duration, price]
            all_prices.append(order[4])
    
    if all_prices:
        print(f"\n[OK] day_orders price stats:")
        print(f"  mean: {np.mean(all_prices):.2f}")
        print(f"  std: {np.std(all_prices):.2f}")
        print(f"  range: [{np.min(all_prices):.2f}, {np.max(all_prices):.2f}]")
else:
    print("[FAIL] day_orders is empty!")

# Test 3: Compare with sampled prices
print("\n" + "=" * 60)
print("Test 3: Compare sampled prices")
print("=" * 60)

env_sampled, _, _, _, _ = load_envs_custom(data_path, driver_num=2000, use_real_orders=False)

# Sampled prices come from price_param
price_param = data['price']
print(f"price_param shape: {price_param.shape}")
print("Sampled price stats (by distance):")
for d in range(min(5, len(price_param))):
    print(f"  distance {d}: mean={price_param[d, 0]:.2f}, std={price_param[d, 1]:.2f}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
print("\nIf price mean ~= 9.59, real order mode works correctly.")
print("If price mean ~= 4.37, still using sampled prices.")

