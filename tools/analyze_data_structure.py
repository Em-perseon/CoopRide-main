"""
数据结构分析脚本
用于分析 CoopRide 项目中的 PKL 数据文件结构
"""

import os
import pickle
import numpy as np
from collections import Counter


def print_section(title):
    """打印分隔线"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def analyze_pkl_file(file_path, data_name):
    """
    分析单个 PKL 文件的结构
    
    Args:
        file_path: PKL 文件路径
        data_name: 数据集名称
    """
    print_section(f"分析 {data_name} 数据集: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    try:
        with open(file_path, 'rb') as handle:
            data_param = pickle.load(handle)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    print(f"\n✅ 成功加载 {data_name} 数据")
    
    # 分析顶层键
    if isinstance(data_param, dict):
        print(f"\n📋 顶层键 ({len(data_param)} 个):")
        for key in data_param.keys():
            value = data_param[key]
            print(f"  - '{key}': {type(value).__name__}")
            
            # 如果是数组，显示形状
            if isinstance(value, np.ndarray):
                print(f"    shape: {value.shape}, dtype: {value.dtype}")
                print(f"    min: {value.min()}, max: {value.max()}, mean: {value.mean():.2f}")
            # 如果是列表，显示长度
            elif isinstance(value, list):
                print(f"    length: {len(value)}")
                if len(value) > 0:
                    print(f"    first element type: {type(value[0])}")
                    if isinstance(value[0], (list, tuple)):
                        print(f"    first element length: {len(value[0])}")
        print()
        
        # 详细分析每个键
        for key, value in data_param.items():
            analyze_data_key(key, value, indent=2)
    else:
        print(f"\n⚠️  数据不是字典类型: {type(data_param)}")
        print(f"内容: {data_param}")


def analyze_data_key(key, value, indent=0):
    """
    分析单个数据键的详细内容
    
    Args:
        key: 键名
        value: 键值
        indent: 缩进层级
    """
    prefix = "  " * indent
    
    # 分析 order_param
    if key == 'order':
        print(f"\n{prefix}📦 订单分布矩阵 (order_param):")
        print(f"{prefix}  - shape: {value.shape} -> (出发地, 目的地, 时间段)")
        print(f"{prefix}  - dtype: {value.dtype}")
        print(f"{prefix}  - 总订单数: {int(value.sum())}")
        print(f"{prefix}  - 非零订单数: {np.count_nonzero(value)}")
        print(f"{prefix}  - 订单范围: {value.min()} ~ {value.max()}")
        
        # 按时间维度统计
        time_orders = value.sum(axis=(0, 1))
        print(f"{prefix}  - 各时间段订单统计 (144个10分钟时间片):")
        print(f"{prefix}    min: {time_orders.min()}, max: {time_orders.max()}, mean: {time_orders.mean():.2f}")
        
        # 按出发地统计
        origin_orders = value.sum(axis=(1, 2))
        print(f"{prefix}  - 各出发地订单统计 ({value.shape[0]}个网格):")
        print(f"{prefix}    min: {origin_orders.min()}, max: {origin_orders.max()}, mean: {origin_orders.mean():.2f}")
        
        # 高订单网格
        top_origins = np.argsort(origin_orders)[-5:][::-1]
        print(f"{prefix}  - 订单最多的5个出发地: {top_origins}, 订单数: {origin_orders[top_origins]}")
    
    # 分析 price_param
    elif key == 'price':
        print(f"\n{prefix}💰 价格参数矩阵 (price_param):")
        print(f"{prefix}  - shape: {value.shape} -> (网格, 邻居级别)")
        print(f"{prefix}  - dtype: {value.dtype}")
        print(f"{prefix}  - 价格范围: {value.min()} ~ {value.max()}")
        print(f"{prefix}  - 价格均值: {value.mean():.2f}")
        
        # 按邻居级别统计价格
        for i in range(min(7, value.shape[1])):
            prices = value[:, i]
            print(f"{prefix}  - 邻居级别 {i}: min={prices.min()}, max={prices.max()}, mean={prices.mean():.2f}")
    
    # 分析 neighbor
    elif key == 'neighbor':
        print(f"\n{prefix}🗺️  邻居距离矩阵 (neighbor):")
        print(f"{prefix}  - shape: {value.shape} -> (网格, 网格)")
        print(f"{prefix}  - dtype: {value.dtype}")
        print(f"{prefix}  - 距离范围: {value.min()} ~ {value.max()}")
        
        # 统计不同距离的频率
        unique, counts = np.unique(value, return_counts=True)
        print(f"{prefix}  - 距离分布:")
        for dist, count in sorted(zip(unique, counts))[:15]:  # 只显示前15个
            if dist >= 100:
                print(f"{prefix}    距离 {dist}: {count} 次 (不可达)")
            else:
                print(f"{prefix}    距离 {dist}: {count} 次")
        
        # 最大可通行距离
        valid_distances = value[value < 100]
        if len(valid_distances) > 0:
            l_max = int(valid_distances.max())
            print(f"{prefix}  - 最大可通行距离 (l_max): {l_max}")
    
    # 分析 duration
    elif key == 'duration':
        print(f"\n{prefix}⏱️  持续时间参数 (duration):")
        print(f"{prefix}  - shape: {value.shape}")
        print(f"{prefix}  - dtype: {value.dtype}")
        print(f"{prefix}  - 持续时间范围: {value.min()} ~ {value.max()}")
        print(f"{prefix}  - 持续时间均值: {value.mean():.2f}")
    
    # 分析 shape
    elif key == 'shape':
        print(f"\n{prefix}📐 网格形状 (shape):")
        print(f"{prefix}  - 值: {value}")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            print(f"{prefix}  - 网格尺寸: {value[0]} 行 × {value[1]} 列 = {value[0] * value[1]} 个网格")
    
    # 分析 real_orders
    elif key == 'real_orders':
        print(f"\n{prefix}📝 真实订单数据 (real_orders):")
        if isinstance(value, np.ndarray):
            print(f"{prefix}  - shape: {value.shape}")
            print(f"{prefix}  - dtype: {value.dtype}")
            print(f"{prefix}  - 订单总数: {len(value)}")
            
            if len(value) > 0:
                print(f"{prefix}  - 订单示例 (前3条):")
                for i, order in enumerate(value[:3]):
                    print(f"{prefix}    订单 {i+1}: {order}")
                    if isinstance(order, (list, tuple)) and len(order) >= 5:
                        print(f"{prefix}      格式: (出发地, 目的地, 开始时间, 结束时间, 价格)")
                        print(f"{prefix}      出发地={order[0]}, 目的地={order[1]}, 时间={order[2]}~{order[3]}, 价格={order[4]:.2f}")
                
                # 统计价格分布
                if len(value) > 0:
                    prices = [o[4] for o in value if isinstance(o, (list, tuple)) and len(o) >= 5]
                    if prices:
                        print(f"{prefix}  - 价格统计: min={min(prices):.2f}, max={max(prices):.2f}, mean={np.mean(prices):.2f}")
        elif isinstance(value, list):
            print(f"{prefix}  - 订单总数: {len(value)}")
            if len(value) > 0:
                print(f"{prefix}  - 第一条订单: {value[0]}")
                if isinstance(value[0], (list, tuple)):
                    print(f"{prefix}  - 订单字段数: {len(value[0])}")
        else:
            print(f"{prefix}  - 类型: {type(value)}")


def analyze_all_data_files():
    """分析所有可用的数据文件"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # 定义要分析的数据文件列表
    data_files = [
        ("NYU_grid143.pkl", "NYU 143网格"),
        ("DiDi_day1_grid121.pkl", "DiDi 121网格"),
        ("NYC2015Jan_h3_289.pkl", "NYC 289网格"),
        ("NYC2015Jan26_h3_289_real_orders.pkl", "NYC 真实订单"),
        ("DiDi_day1_grid121_b2_real_orders.pkl", "DiDi 真实订单 b2"),
    ]
    
    # 添加子目录中的文件
    sub_dirs = ["DiDi", "NYU"]
    for subdir in sub_dirs:
        dir_path = os.path.join(data_dir, subdir)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith('.pkl'):
                    full_path = os.path.join(dir_path, file)
                    if full_path not in [os.path.join(data_dir, f[0]) for f in data_files]:
                        data_files.append((os.path.join(subdir, file), f"{subdir} - {file}"))
    
    # 分析每个文件
    for file_name, data_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        analyze_pkl_file(file_path, data_name)
        print("\n")


def summarize_data_structure():
    """总结数据结构"""
    print_section("数据结构总结")
    print("""
    CoopRide PKL 数据文件结构:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ 顶层键 (字典类型):                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │  'order'      : 订单分布矩阵                                       │
    │                - shape: (出发地, 目的地, 时间段)                   │
    │                - dtype: int32/float32                            │
    │                - 值: 表示从网格i到网格j在时间段t的订单数量           │
    │                                                                  │
    │  'price'      : 价格参数矩阵                                       │
    │                - shape: (网格, 邻居级别)                          │
    │                - dtype: float64                                 │
    │                - 值: 表示从网格到k级邻居网格的价格                  │
    │                - 第0级: 同网格，第1-6级: 不同距离的邻居            │
    │                                                                  │
    │  'neighbor'   : 邻居距离矩阵                                       │
    │                - shape: (网格, 网格)                              │
    │                - dtype: int32/float64                            │
    │                - 值: 0=同一网格, 1-6=可通行距离, >=100=不可达     │
    │                                                                  │
    │  'shape'      : 网格形状                                           │
    │                - type: tuple/list                                │
    │                - 值: (M, N) 表示M行N列的网格                       │
    │                                                                  │
    │  'duration'   : 持续时间参数 (部分数据集)                           │
    │                - shape: (网格, 网格) 或其他                         │
    │                - 值: 表示订单的持续时间                             │
    │                                                                  │
    │  'real_orders': 真实订单数据 (部分数据集)                           │
    │                - type: list 或 array                              │
    │                - 每个订单: (出发地, 目的地, 开始时间, 结束时间, 价格) │
    └─────────────────────────────────────────────────────────────────┘
    
    时间段说明:
    - 通常为 144 个时间段
    - 每个时间段代表 10 分钟
    - 总共覆盖 24 小时 (144 × 10分钟 = 1440分钟 = 24小时)
    
    网格编号:
    - 网格ID范围: 0 到 (M×N-1)
    - 例如 143 网格: ID 0-142
    - 例如 289 网格: ID 0-288
    
    邻居级别说明:
    - 0级: 同一网格
    - 1-6级: 不同距离的邻居网格
    - >=100: 不可达/太远
    """)


if __name__ == '__main__':
    print("="*80)
    print("  CoopRide 数据结构分析工具")
    print("="*80)
    
    # 总结数据结构
    summarize_data_structure()
    
    # 分析所有数据文件
    print("\n正在分析数据文件...")
    analyze_all_data_files()
    
    print("\n" + "="*80)
    print("  分析完成!")
    print("="*80)