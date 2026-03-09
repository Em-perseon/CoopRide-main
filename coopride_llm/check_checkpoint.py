"""
工具脚本：查看MAPPO模型checkpoint的详细结构和维度信息
使用方法: python tools/check_checkpoint.py <checkpoint_path>
"""

import torch
import sys
from collections import OrderedDict

def print_tensor_info(name, tensor):
    """打印张量的详细信息"""
    print(f"  ├─ {name}")
    print(f"  │  ├─ Shape: {tensor.shape}")
    print(f"  │  ├─ Dtype: {tensor.dtype}")
    print(f"  │  └─ Device: {tensor.device}")

def analyze_state_dict(state_dict, prefix=""):
    """分析state_dict的结构"""
    print(f"\n{prefix}包含 {len(state_dict)} 个参数:")
    total_params = 0
    
    for key, value in state_dict.items():
        param_count = value.numel()
        total_params += param_count
        print_tensor_info(key, value)
    
    print(f"\n{prefix}总参数量: {total_params:,} ({total_params / 1024 / 1024:.2f}M)")
    return total_params

def analyze_checkpoint(checkpoint_path):
    """分析checkpoint文件"""
    print(f"="*80)
    print(f"分析checkpoint: {checkpoint_path}")
    print(f"="*80)
    
    # 加载checkpoint
    try:
        state = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 打印顶层键
    print(f"\n【顶层结构】")
    print(f"Checkpoint包含的键:")
    for key in state.keys():
        value = state[key]
        if isinstance(value, OrderedDict):
            print(f"  ├─ '{key}': OrderedDict (state_dict)")
        elif isinstance(value, int):
            print(f"  ├─ '{key}': int (value={value})")
        elif isinstance(value, dict):
            print(f"  ├─ '{key}': dict (优化器状态)")
        else:
            print(f"  ├─ '{key}': {type(value).__name__}")
    
    # 如果有step信息
    if 'step' in state:
        print(f"\n【训练信息】")
        print(f"训练步数: {state['step']}")
    
    # 分析Actor网络
    if 'actor net' in state:
        print(f"\n{'='*80}")
        print(f"【Actor网络】")
        print(f"="*80)
        actor_params = analyze_state_dict(state['actor net'], "Actor")
    
    # 分析Critic网络
    if 'critic net' in state:
        print(f"\n{'='*80}")
        print(f"【Critic网络】")
        print(f"="*80)
        critic_params = analyze_state_dict(state['critic net'], "Critic")
    
    # 分析优化器状态
    if 'actor optimizer' in state:
        print(f"\n{'='*80}")
        print(f"【Actor优化器】")
        print(f"="*80)
        opt_state = state['actor optimizer']
        print(f"优化器状态键: {list(opt_state.keys())}")
        if 'state' in opt_state:
            print(f"参数状态数量: {len(opt_state['state'])}")
        if 'param_groups' in opt_state:
            print(f"参数组数量: {len(opt_state['param_groups'])}")
            if len(opt_state['param_groups']) > 0:
                pg = opt_state['param_groups'][0]
                print(f"  lr: {pg.get('lr', 'N/A')}")
                print(f"  betas: {pg.get('betas', 'N/A')}")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"【总结】")
    print(f"="*80)
    if 'actor net' in state and 'critic net' in state:
        total = actor_params + critic_params
        print(f"Actor参数量: {actor_params:,} ({actor_params/total*100:.1f}%)")
        print(f"Critic参数量: {critic_params:,} ({critic_params/total*100:.1f}%)")
        print(f"总参数量: {total:,} ({total / 1024 / 1024:.2f}M)")

def main():
    # if len(sys.argv) < 2:
        # print("使用方法: python tools/check_checkpoint.py <checkpoint_path>")
        # print("\n示例:")
        # print("  python tools/check_checkpoint.py logs/synthetic/grid143/xxx/param.pkl")
        # sys.exit(1)
    test_base_dir = '../logs/synthetic/grid143/EnvStat326_OD143_FMRLmerge_Batch1000_Gamma0.97_Lambda0.95_Iter1_Ir0.001_Step144_Ent0.005_Minibatch5_Parallel5mix_MDP0_StateEmb2_Meta0global_DGCNAC_relufeaNor1_20260103_02-57'
    
    # checkpoint_path = sys.argv[1]
    analyze_checkpoint(test_base_dir + '/Best.pkl')

if __name__ == "__main__":
    main()




# =================================== interface =================================== #
def change_weight(checkpoint):
    target_key = 'order_layer.order_layer2.weight'

    # ==========================================
    # 第一步：将权重导出为二维 List
    # ==========================================

    # 获取 Tensor 对象
    tensor_data = checkpoint[target_key]
    
    # 转换为 List
    # .detach(): 从计算图中分离，防止梯度报错
    # .cpu(): 确保数据在 CPU 上（如果原本在 GPU，直接转 numpy/list 会报错）
    # .tolist(): 转换为 Python 原生 List
    weight_as_list = tensor_data.detach().cpu().tolist()

    print(f"导出成功！数据类型: {type(weight_as_list)}")
    print(f"列表维度: {len(weight_as_list)} 行, {len(weight_as_list[0])} 列")
    
    # ==========================================
    # 第二步：修改 List (此处作为演示，将所有值 = 1)
    # ==========================================

    # 对二维 list 进行简单的遍历修改
    for i in range(len(weight_as_list)):
        for j in range(len(weight_as_list[0])):
            weight_as_list[i][j] = 1.0  # 你的自定义修改逻辑放这里


    # ==========================================
    # 第三步：将修改后的 List 导回 Checkpoint
    # ==========================================

    # 1. 转回 Tensor
    # 注意：必须指定 dtype=torch.float32，因为 Python list 默认小数是 float64 (double)，
    # 而 PyTorch 模型通常使用 float32。如果不指定，模型加载时会报错。
    new_tensor = torch.tensor(weight_as_list, dtype=torch.float32)

    # 2. 形状检查 (非常重要！防止手动修改时改坏了维度)
    original_shape = checkpoint[target_key].shape
    if new_tensor.shape != original_shape:
        raise ValueError(f"维度不匹配！原维度: {original_shape}, 新维度: {new_tensor.shape}")

    # 3. 覆盖原 Checkpoint 中的值
    checkpoint[target_key] = new_tensor

    print("\n导入成功！Checkpoint 已更新。")
    print(f"新 Tensor 属性: {checkpoint[target_key].dtype}")
    return checkpoint