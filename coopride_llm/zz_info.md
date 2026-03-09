# run_CoTa.py 代码结构分析

## 文件概述
`run_CoTa.py` 是 CoopRide 项目的核心训练脚本，使用 MAPPO (Multi-Agent PPO) 算法进行协同出租车调度系统的训练和测试。

## 目录结构
```
run/
└── run_CoTa.py  # 主训练脚本
```

---

## 代码结构

### 1. 导入部分
```python
- 环境设置 (MKL_NUM_THREADS, CUDA_VISIBLE_DEVICES)
- 标准库 (os, argparse, sys, time, pickle)
- 自定义模块:
  - simulator.envs: 环境模拟器
  - tools.create_envs: 环境创建工具
  - tools.load_data: 数据加载工具
  - algo.MAPPO: MAPPO 算法实现
  - tools.logfile: 日志记录工具
- PyTorch 相关 (torch, tensorboard)
```

### 2. 核心函数

#### 2.1 `get_parameter()` - 参数配置函数
**功能**: 定义和管理所有训练/测试参数

**参数分类**:
- **测试参数**: 
  - `test_dir`: 测试日志目录
  - `model_dir`: 模型权重路径
  - `test`: 是否为测试模式
  - `TEST_ITER`, `TEST_SEED`: 测试迭代次数和随机种子

- **基础训练参数**:
  - `MAX_ITER`: 最大迭代次数 (6000)
  - `resume_iter`: 恢复训练的起始迭代
  - `device`: 计算设备 (gpu/cpu)

- **环境参数**:
  - `dispatch_interval`: 决策间隔 (分钟)
  - `TIME_LEN`: 一天总决策次数 (1440/间隔)
  - `grid_num`: 网格数量 (143/121/100)
  - `driver_num`: 司机数量 (对应不同网格数量)
  - `env_seed`: 环境随机种子
  - `dynamic_env`: 是否使用动态环境

- **RL 算法参数**:
  - `batch_size`: 批次大小
  - `actor_lr`, `critic_lr`, `meta_lr`: 学习率
  - `train_actor_iters`, `train_critic_iters`, `train_phi_iters`: 训练迭代次数
  - `gamma`: 折扣因子 (0.97)
  - `lam`: GAE lambda (0.95)
  - `max_grad_norm`: 梯度裁剪 (10)
  - `clip_ratio`: PPO 裁剪比率 (0.2)
  - `ent_factor`: 熵系数 (0.005)
  - `steps_per_epoch`: 每个 epoch 的步数 (144)
  - `minibatch_num`: 小批次数量 (5)
  - `parallel_episode`: 并行回合数 (5)
  - `parallel_way`: 并行方式 ('mix')

- **网络架构参数**:
  - `use_orthogonal`: 是否使用正交初始化
  - `use_value_clip`: 是否使用值裁剪
  - `use_valuenorm`: 是否使用值归一化
  - `use_huberloss`: 是否使用 Huber 损失
  - `use_lr_anneal`: 是否使用学习率退火
  - `use_GAEreturn`: 是否使用 GAE
  - `use_rnn`: 是否使用 RNN (GRU)
  - `use_GAT`, `use_GCN`, `use_DGCN`: 图神经网络选择
  - `adj_rank`: 邻接阶数
  - `activate_fun`: 激活函数 ('relu')

- **状态和特征参数**:
  - `feature_normal`: 特征归一化方式 (1/2/3)
  - `use_state_diff`: 是否使用状态差值
  - `use_order_time`: 是否使用订单时间
  - `new_order_entropy`: 是否使用新订单熵
  - `order_grid`: 是否使用订单网格
  - `use_mdp`: 是否使用 MDP (0/1/2)
  - `state_emb_choose`: 状态嵌入选择 (2)

- **合作机制参数 (Meta-learning)**:
  - `meta_choose`: meta 学习策略选择 (0-7)
    - 0: 无 meta
    - 1,2,3: 圆环加权 (不同归一化)
    - 4,5: 圆饼加权 (0~K阶/1~K阶)
    - 6,7: 圆环+圆饼结合
  - `meta_scope`: meta 作用范围 (4)
  - `team_rank`: 团队排名 (0)
  - `global_share`: 是否全局共享

- **奖励和匹配参数**:
  - `FM_mode`: 匹配模式 ('local'/'RLmerge'/'RLsplit')
  - `reward_scale`: 奖励缩放 (5)
  - `ORR_reward`: 是否使用 ORR 奖励
  - `ORR_reward_effi`: ORR 奖励效率 (1)
  - `only_ORR`: 是否仅使用 ORR

- **日志控制**:
  - `log_feature`: 是否记录特征
  - `log_distribution`: 是否记录分布
  - `log_phi`: 是否记录 phi
  - `log_name`: 日志名称 (自动生成)

**日志名称生成逻辑**:
根据参数组合自动生成唯一的日志名称，格式如:
```
EnvStat{seed}_OD{grid}_FM{mode}_Batch{size}_Gamma{gamma}_...
```

#### 2.2 `train()` - 训练函数
**功能**: 执行完整的训练循环

**参数**:
- `env`: 环境
- `agent`: MAPPO 智能体
- `writer`: TensorBoard writer
- `args`: 配置参数
- `device`: 计算设备

**训练流程**:
```
1. 初始化最佳性能指标 (best_gmv, best_orr)
2. 外层循环: iteration from resume_iter to MAX_ITER
   ├── 设置随机种子
   ├── 环境重置 (env.reset)
   ├── 初始化状态和隐藏状态
   ├── 时间步循环: T from 0 to TIME_LEN-1
   │   ├── Agent 采样动作 (agent.action)
   │   ├── 执行环境步 (env.step)
   │   ├── 计算指标 (KL, entropy, GMV, ORR)
   │   ├── 计算奖励 (可能包含 ORR_reward)
   │   ├── 存储转换 (agent.buffer.push)
   │   ├── 计算 bootstrap value (若 epoch 结束)
   │   ├── 更新状态
   │   └── T += 1
   ├── 更新智能体 (agent.update)
   ├── 保存日志和特征
   ├── 保存最佳模型
   └── 记录 TensorBoard 指标
```

**关键特性**:
- 支持并行训练队列 (`parallel_queue`)
- 支持 ORR 奖励机制
- 支持 GAE 计算
- 支持多种网络架构 (DGCN, GAT, GCN, RNN)
- 自动保存最佳模型

#### 2.3 `test()` - 测试函数
**功能**: 在测试模式下评估模型性能

**参数**: 与 `train()` 相同

**测试流程**:
```
1. 设置固定随机种子 (TEST_SEED)
2. 循环: iteration from TEST_SEED to TEST_SEED + TEST_ITER
   ├── 环境重置
   ├── 初始化状态和隐藏状态
   ├── 时间步循环
   │   ├── Agent 采样动作 (无训练)
   │   ├── 执行环境步
   │   ├── 计算指标
   │   ├── 更新状态
   │   └── T += 1
   ├── 保存分布日志
   └── 输出性能指标
```

**与训练的区别**:
- 不更新模型参数
- 不存储转换到 buffer
- 不进行 bootstrap value 计算
- 使用固定随机种子保证可重复性

### 3. 主程序入口 (`if __name__ == "__main__"`)

**执行流程**:
```
1. 获取参数 (get_parameter)
2. 设置计算设备 (GPU/CPU)
3. 创建环境 (根据 grid_num 选择):
   ├── 100: create_OD()
   ├── 36: create_OD_36()
   ├── 121: load_envs_DiDi121()
   └── 143: load_envs_NYU143()
4. 设置车队帮助模式 (env.fleet_help)
5. 创建 TensorBoard writer
6. 初始化 MAPPO Agent
7. 移动模型到指定设备
8. 根据 args.test 选择:
   ├── True: 加载模型 → 运行 test()
   └── False: 初始化日志 → 运行 train()
```

---

## 数据流图

```
[环境数据] → [数据加载] → [环境创建]
                          ↓
                      [环境对象]
                          ↓
              +---------------------------+
              |      MAPPO Agent          |
              +---------------------------+
              |  - Actor Network          |
              |  - Critic Network         |
              |  - Meta Network           |
              |  - Buffer                 |
              |  - Value Normalizer       |
              +---------------------------+
                          ↓
              [训练/测试循环]
                          ↓
        [动作采样] → [环境交互] → [奖励计算] → [参数更新]
                          ↓
              [日志记录] → [模型保存] → [TensorBoard]
```

---

## 关键特性总结

### 1. 多智能体强化学习
- 使用 MAPPO 算法进行多智能体协作
- 支持集中式训练、分布式执行 (CTDE)

### 2. 图神经网络
- 支持 DGCN (Deep Graph Convolutional Network)
- 支持邻域聚合和信息传播

### 3. 元学习机制
- 7 种不同的 meta 策略
- 支持圆环加权、圆饼加权、混合策略

### 4. 灵活的奖励设计
- 基础 GMV 奖励
- ORR (Order Response Rate) 奖励
- 支持奖励缩放

### 5. 多种数据集支持
- NYU143 (纽约大学数据)
- DiDi121 (滴滴数据)
- Synthetic (合成数据)

### 6. 完善的日志系统
- TensorBoard 实时监控
- 特征记录
- 分布记录
- 模型自动保存

---

## 使用示例

### 训练
```bash
python run/run_CoTa.py
```

### 测试
```python
# 在 get_parameter() 中设置:
args.test = True
args.model_dir = 'path/to/model.pkl'
```

---

## 依赖关系

```
run_CoTa.py
├── simulator/envs.py          # 环境定义
├── simulator/envs_real.py     # 真实环境
├── tools/create_envs.py       # 环境创建
├── tools/load_data.py         # 数据加载
├── algo/MAPPO.py              # MAPPO 算法
├── tools/logfile.py           # 日志工具
└── data/*.pkl                 # 数据文件
```

---

## 性能指标

训练和测试过程中记录的关键指标:
- **GMV**: 总订单价值
- **ORR**: 订单响应率
- **KL**: KL 散度 (订单分布 vs. 司机分布)
- **Entropy**: 供给/需求熵
- **Loss**: Actor/Critic/Meta 损失
- **Value**: Value function 预测值

---

## 配置建议

### 快速实验
- `MAX_ITER = 100`
- `parallel_episode = 1`
- `grid_num = 100`

### 正式训练
- `MAX_ITER = 6000`
- `parallel_episode = 5`
- `grid_num = 143`

### 测试
- `TEST_ITER = 10`
- `TEST_SEED = 固定值`
- `test = True`

---

## 注意事项

1. **CUDA 设置**: 文件硬编码了 `CUDA_VISIBLE_DEVICES='0'`，需要根据实际 GPU 索引调整
2. **路径依赖**: 使用相对路径 `../`，需要在正确的目录下运行
3. **日志目录**: 会自动创建，避免重复运行导致的覆盖问题
4. **内存管理**: `memory_size` 根据并行回合数和步数计算
5. **数据集**: 不同 `grid_num` 需要对应的数据集存在

---

## 扩展性

该脚本设计良好，易于扩展:
1. 添加新的环境: 在主程序入口添加新的环境创建逻辑
2. 修改奖励函数: 在 `train()` 和 `test()` 中修改奖励计算部分
3. 添加新的网络架构: 修改 Agent 定义和参数设置
4. 自定义日志: 扩展 `logfile` 工具类

---

**文档生成时间**: 2026-01-28  
**Python 版本**: 3.x  
**PyTorch 版本**: 1.x