# MAPPO算法代码框架总结

## 目录

- [1. 整体架构](#1-整体架构)
  - [1.1 文件结构](#11-文件结构)
  - [1.2 算法特点](#12-算法特点)
  - [1.3 网络层级关系](#13-网络层级关系)
  - [1.4 网络调用关系](#14-网络调用关系)
  - [1.5 子网络调用链](#15-子网络调用链)
  - [1.6 网络输入输出对照表](#16-网络输入输出对照表)

- [2. 核心类结构](#2-核心类结构)
  - [2.1 PPO类](#21-ppo类)
  - [2.2 Actor类](#22-actor类)
  - [2.3 Critic类](#23-critic类)
  - [2.4 Replay_buffer类](#24-replay_buffer类)

- [3. 数据流向](#3-数据流向)
  - [3.1 推理阶段数据流](#31-推理阶段数据流)
  - [3.2 训练阶段数据流](#32-训练阶段数据流)

- [4. 网络结构](#4-网络结构)
  - [4.1 Actor网络](#41-actor网络)
  - [4.2 Critic网络](#42-critic网络)
  - [4.3 嵌入层模块](#43-嵌入层模块)

- [5. 训练流程](#5-训练流程)
  - [5.1 训练循环](#51-训练循环)
  - [5.2 Actor更新](#52-actor更新)
  - [5.3 Critic更新](#53-critic更新)
  - [5.4 Meta更新](#54-meta更新)

- [6. 文件输入输出](#6-文件输入输出)
  - [6.1 文件输入](#61-文件输入)
  - [6.2 文件输出](#62-文件输出)
  - [6.3 输入输出映射表](#63-输入输出映射表)

- [7. 关键技术点](#7-关键技术点)
  - [7.1 GAE](#71-gae)
  - [7.2 PPO裁剪](#72-ppo裁剪)
  - [7.3 图神经网络](#73-图神经网络)
  - [7.4 元学习](#74-元学习)
  - [7.5 辅助任务](#75-辅助任务)
  - [7.6 价值标准化](#76-价值标准化)

- [8. 神经网络详细信息](#8-神经网络详细信息)
  - [8.1 网络总览](#81-网络总览)
  - [8.2 Actor网络](#82-actor网络)
  - [8.3 Critic网络](#83-critic网络)
  - [8.4 DeepMdpAgent](#84-deepmdpagent)
  - [8.5 权重管理](#85-权重管理)
  - [8.6 优化器配置](#86-优化器配置)
  - [8.7 张量维度总结](#87-张量维度总结)
  - [8.8 参数量估计](#88-参数量估计)

- [9. 使用示例](#9-使用示例)
  - [9.1 初始化](#91-初始化)
  - [9.2 训练](#92-训练)
  - [9.3 推理](#93-推理)
  - [9.4 保存加载](#94-保存加载)

- [10. 总结](#10-总结)

---

## 1. 整体架构

MAPPO (Multi-Agent PPO) 是用于协同网约车调度系统的多智能体强化学习算法。

### 1.1 文件结构

```
algo/MAPPO.py
├── 辅助函数 (lines 13-73)
├── 嵌入层模块
│   ├── order_embedding (订单嵌入)
│   ├── order_embedding2 (订单嵌入v2)
│   └── state_embedding (状态嵌入)
├── 表示层模块
│   ├── state_representation1 (状态表征v1)
│   └── state_representation2 (状态表征v2)
├── 网络层
│   ├── RNNLayer (循环神经网络)
│   ├── GATLayer (图注意力网络)
│   ├── DGCNLayer (扩散图卷积GRU)
│   ├── GCNLayer (图卷积网络)
│   └── NeighborLayer (邻居聚合层)
├── 策略网络
│   └── Actor (订单匹配策略)
├── 价值网络
│   ├── Critic0 (简化版)
│   └── Critic (完整版)
├── 辅助智能体
│   ├── MdpAgent (MDP智能体)
│   └── DeepMdpAgent (深度MDP)
├── 主算法类
│   └── PPO (算法协调器)
└── 经验回放
    └── Replay_buffer (数据缓冲)
```

### 1.2 算法特点

- **多智能体协同**: 多个网格区域的智能体协同决策
- **集中式训练，分布式执行**: 训练时集中式，执行时分布式
- **图神经网络**: 支持GAT/GCN/DGCN捕获空间关系
- **元学习**: 支持Phi函数实现多层级价值估计
- **辅助任务**: 支持状态预测等辅助损失

### 1.3 网络层级关系

```
PPO类 (协调器)
    │
    ├── Actor (策略网络)
    │   ├── state_layer → state_representation1
    │   │   ├── state_embedding
    │   │   │   ├── time_embedding
    │   │   │   ├── grid_embedding
    │   │   │   └── contin_embedding
    │   │   ├── [可选] NeighborLayer
    │   │   ├── [可选] RNNLayer
    │   │   ├── [可选] GATLayer
    │   │   ├── [可选] GCNLayer
    │   │   └── [可选] DGCNLayer
    │   │
    │   └── order_layer → order_embedding/order_embedding2
    │       ├── grid_embedding (origin)
    │       ├── grid_embedding (dest)
    │       └── contin_embedding
    │
    ├── Critic (价值网络)
    │   ├── state_layer → state_representation1 (同Actor)
    │   │   └── [相同的子网络结构]
    │   │
    │   ├── local_value_layer (线性层)
    │   ├── [可选] Phi_layer (元学习权重)
    │   ├── [可选] global_fc_layer (全局特征)
    │   └── [可选] global_value_layer (全局价值)
    │
    ├── [可选] DeepMdpAgent
    │   ├── time_embedding
    │   ├── grid_embedding
    │   ├── state_layer1/2
    │   └── value_layer
    │
    └── Replay_buffer (数据存储)
```

```
【顶层结构】

Checkpoint包含的键:

  ├─ 'step': int (value=0)

  ├─ 'actor net': OrderedDict (state_dict)

  ├─ 'critic net': OrderedDict (state_dict)

  ├─ 'actor optimizer': dict (优化器状态)

  ├─ 'critic optimizer': dict (优化器状态)



【训练信息】

训练步数: 0



================================================================================

【Actor网络】

================================================================================



Actor包含 21 个参数:

  ├─ order_layer.grid_embedding.weight

  │  ├─ Shape: torch.Size([143, 128])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ order_layer.contin_embedding.weight

  │  ├─ Shape: torch.Size([128, 6])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ order_layer.contin_embedding.bias

  │  ├─ Shape: torch.Size([128])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ order_layer.order_layer2.weight

  │  ├─ Shape: torch.Size([128, 384])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ order_layer.order_layer2.bias

  │  ├─ Shape: torch.Size([128])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ order_layer.order_layer3.weight

  │  ├─ Shape: torch.Size([128, 128])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ order_layer.order_layer3.bias

  │  ├─ Shape: torch.Size([128])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.grid_embedding.weight

  │  ├─ Shape: torch.Size([143, 64])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.time_embedding.weight

  │  ├─ Shape: torch.Size([144, 64])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.contin_embedding.weight

  │  ├─ Shape: torch.Size([64, 22])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.contin_embedding.bias

  │  ├─ Shape: torch.Size([64])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.gnn_layer.GCN1.gconv.weight

  │  ├─ Shape: torch.Size([64, 192])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.gnn_layer.GCN1.gconv.bias

  │  ├─ Shape: torch.Size([64])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.gnn_layer.GCN2.gconv.weight

  │  ├─ Shape: torch.Size([64, 192])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.gnn_layer.GCN2.gconv.bias

  │  ├─ Shape: torch.Size([64])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.gnn_layer.fc.weight

  │  ├─ Shape: torch.Size([64, 64])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.gnn_layer.fc.bias

  │  ├─ Shape: torch.Size([64])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.state_fc1.weight

  │  ├─ Shape: torch.Size([128, 192])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.state_fc1.bias

  │  ├─ Shape: torch.Size([128])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.state_fc2.weight

  │  ├─ Shape: torch.Size([128, 128])

  │  ├─ Dtype: torch.float32

  │  └─ Device: cpu

  ├─ state_layer.state_fc2.bias

  │  ├─ Shape: torch.Size([128])

  │  ├─ Dtype: torch.float32
```

### 1.4 网络调用关系

| 阶段 | 入口方法 | 调用网络 | 调用子网络 | 输出 |
|------|----------|----------|------------|------|
| 推理 | PPO.action() | Actor.forward() | state_layer<br>order_layer | probs<br>hidden_state |
| | | Critic.forward() | state_layer<br>local_value_layer | value_local<br>hidden_state |
| | | Critic.get_global_value() | global_fc_layer<br>global_value_layer | value_global<br>hidden_state |
| 训练 | PPO.update() | compute_loss_actor() | Actor.multi_mask_forward() | loss_actor |
| | | compute_loss_critic() | Critic.get_local_value() | loss_critic |
| | | | Critic.get_global_value() | loss_critic_global |
| | | compute_loss_phi() | Critic.get_phi() | loss_phi |

### 1.5 子网络调用链

#### Actor.forward() 调用链

```
Actor.forward(state, order, mask)
    ↓
state_layer.forward(state)
    ↓
state_representation1.forward()
    ↓
├─→ state_embedding.forward()
│    ├─→ time_embedding() [N, E]
│    ├─→ grid_embedding() [N, E]
│    └─→ contin_embedding() [N, E]
├─→ [可选] NeighborLayer.forward() [N, E]
├─→ [可选] RNNLayer.forward() [N, E]
├─→ [可选] GATLayer.forward() [N, E]
├─→ [可选] GCNLayer.forward() [N, E]
└─→ [可选] DGCNLayer.forward() [N, E]
    ↓
state_emb [N, E]
```

```
order_layer.forward(order)
    ↓
order_embedding.forward()
    ↓
├─→ grid_embedding(origin) [B, M, E]
├─→ grid_embedding(dest) [B, M, E]
└─→ contin_embedding(contin) [B, M, E]
    ↓
concat → FC1 → FC2 → order_emb [B, M, E]
```

```
最终计算:
MatMul(state_emb, order_emb^T) → compatibility [N, M]
    ↓
Softmax → probs [N, M]
```

#### Critic.forward() 调用链

```
Critic.forward(state)
    ↓
state_layer.forward(state) [同Actor]
    ↓
state_emb [N, E]
    ↓
local_value_layer(state_emb) → local_value [N, K+1]
```

### 1.6 网络输入输出对照表

| 调用者 | 被调用者 | 输入维度 | 输出维度 | 用途 |
|--------|----------|----------|----------|------|
| PPO.action | Actor.forward | state[N,S]<br>order[N,M,O]<br>mask[N,M] | probs[N,M]<br>hidden_state | 生成订单匹配概率 |
| PPO.action | Critic.forward | state[N,S] | value_local[N,K+1]<br>hidden_state | 估计局部价值 |
| PPO.action | Critic.get_global_value | state[N,S] | value_global[1,1]<br>hidden_state | 估计全局价值 |
| Actor.forward | state_layer | state[N,S] | state_emb[N,E] | 状态编码 |
| Actor.forward | order_layer | order[N,M,O] | order_emb[N,M,E] | 订单编码 |
| PPO.compute_loss_actor | Actor.multi_mask_forward | state[B,S]<br>order[B,M,O]<br>mask[B,M,M] | probs[B,N,M,M] | 计算新策略概率 |
| PPO.compute_loss_critic | Critic.get_local_value | state[B,S] | new_value[B,K+1] | 计算新价值 |
| PPO.compute_loss_critic | Critic.get_global_value | state[B,S] | new_value[1,1] | 计算新全局价值 |
| PPO.compute_loss_phi | Critic.get_phi | state[B,S] | phi[B,K+1] | 生成元学习权重 |
| PPO.process_order | DeepMdpAgent.get_value | mdp_state[T,2] | mdp_value[T,1] | 添加MDP价值特征 |

**图例说明**:
- N: agent_num (智能体数量)
- M: max_order_num (最大订单数)
- B: batch_size (批量大小)
- S: state_dim (状态维度)
- O: order_dim (订单维度)
- E: embedding_dim (嵌入维度)
- K: meta_scope (元学习范围)
- T: time_steps (时间步数)

---

## 2. 核心类结构

### 2.1 PPO类

**代码位置**: ~line 890-1350

**功能**: MAPPO算法的主协调器类

**核心属性**:
```python
# 环境参数
self.agent_num = args.grid_num          # 智能体数量(网格数)
self.driver_num = args.driver_num      # 司机数量
self.max_order_num = 100                # 最大订单数

# 超参数
self.gamma = args.gamma                 # 折扣因子
self.lam = args.lam                     # GAE参数
self.clip_ratio = args.clip_ratio       # PPO裁剪参数
self.ent_factor = args.ent_factor       # 熵系数

# 网络组件
self.actor = Actor(...)                 # 策略网络
self.critic = Critic(...)               # 价值网络
self.buffer = Replay_buffer(...)       # 经验回放缓冲区
```

**主要方法**:
- `action()`: 生成动作(订单匹配)
- `update()`: 更新网络参数
- `compute_loss_actor()`: 计算策略损失
- `compute_loss_critic()`: 计算价值损失
- `compute_loss_phi()`: 计算元学习损失

### 2.2 Actor类

**代码位置**: ~line 430-530

**功能**: 订单匹配策略网络

**核心结构**:
```python
class Actor(nn.Module):
    def __init__(self, ...):
        # 状态嵌入层
        self.state_layer = state_representation1(...)
        
        # 订单嵌入层
        self.order_layer = order_embedding(...)
    
    def forward(self, state, order, mask):
        # 1. 嵌入状态和订单
        state_emb = self.state_layer(state)  # [N, E]
        order_emb = self.order_layer(order)  # [N, M, E]
        
        # 2. 计算兼容性分数
        compatibility = torch.matmul(
            state_emb[:, None, :], 
            order_emb.transpose(-2, -1)
        )  # [N, M]
        
        # 3. 生成动作概率分布
        probs = F.softmax(compatibility, dim=-1)  # [N, M]
        return probs, hidden_state
```

### 2.3 Critic类

**代码位置**: ~line 593-735

**功能**: 价值估计网络，支持局部和全局价值

**核心结构**:
```python
class Critic(nn.Module):
    def __init__(self, ..., meta_scope=3, meta_choose=0, ...):
        # 状态嵌入层
        self.state_layer = state_representation1(...)
        
        # 局部价值层
        self.local_value_layer = nn.Linear(embedding_dim, meta_scope+1)
        
        # 如果启用元学习
        if meta_choose > 0:
            self.Phi_layer = nn.Linear(embedding_dim, meta_scope+1)
            self.global_fc_layer = nn.Linear(global_emb_dim, embedding_dim)
            self.global_value_layer = nn.Linear(embedding_dim, 1)
    
    def forward(self, state, adj, hidden_state):
        state_emb = self.state_layer(state)
        local_value = self.local_value_layer(state_emb)
        return local_value, hidden_state
    
    def get_global_value(self, state, adj, hidden_state):
        global_value = self.global_value_layer(...)
        return global_value, hidden_state
    
    def get_phi(self, state):
        phi = self.Phi_layer(state_emb)
        return phi
```

### 2.4 Replay_buffer类

**代码位置**: ~line 873-1350

**功能**: 存储训练数据的缓冲区

**核心结构**:
```python
class Replay_buffer:
    def __init__(self, capacity, state_dim, order_dim, ...):
        # 存储池
        self.state_pool = torch.zeros(...)
        self.order_pool = torch.zeros(...)
        self.action_pool = torch.zeros(...)
        self.reward_pool = torch.zeros(...)
        self.value_local_pool = torch.zeros(...)
        self.value_global_pool = torch.zeros(...)
        self.oldp_pool = torch.zeros(...)
    
    def push(self, ...):
        # 存储单步数据
    
    def finish_path_local(self, last_local_val):
        # 计算局部优势和回报 (使用GAE)
    
    def finish_path_global(self, last_global_val):
        # 计算全局优势和回报
    
    def get(self, device='cpu'):
        # 获取所有数据用于训练
```

---

## 3. 数据流向

### 3.1 推理阶段数据流

```
环境数据
  ├─ states [N, S]
  ├─ orders [N, M, O]
  └─ masks [N, M]
    ↓
PPO.action()
    ├─→ Actor.forward()
    │     ├─→ state_layer → state_emb [N, E]
    │     ├─→ order_layer → order_emb [N, M, E]
    │     ├─→ MatMul → compatibility [N, M]
    │     └─→ Softmax → probs [N, M]
    │
    └─→ Critic.forward()
          ├─→ state_layer → state_emb [N, E]
          └─→ local_value_layer → value_local [N, K+1]
    ↓
返回: probs, value_local, value_global
```

### 3.2 训练阶段数据流

```
数据收集阶段
  ├─ buffer.push(state, order, action, reward, value_local, value_global, oldp)
  └─ 重复多步
    ↓
优势计算
  ├─ finish_path_local(last_local_val)
  │   └─ 使用GAE计算局部优势
  └─ finish_path_global(last_global_val)
      └─ 使用GAE计算全局优势
    ↓
网络更新 (update())
  ├─ buffer.get() → 获取训练数据
  │
  ├─ Actor更新
  │   ├─ compute_loss_actor()
  │   │   ├─→ Actor.multi_mask_forward() → newp
  │   │   ├─→ ratio = newp/oldp
  │   │   ├─→ PPO裁剪损失
  │   │   └─→ loss_actor
  │   └─ actor_optimizer.step()
  │
  ├─ Critic更新
  │   ├─ compute_loss_critic(get_local_value)
  │   │   ├─→ new_value_local
  │   │   └─→ loss_critic_local
  │   ├─ compute_loss_critic(get_global_value)
  │   │   ├─→ new_value_global
  │   │   └─→ loss_critic_global
  │   └─ critic_optimizer.step()
  │
  └─ Meta更新 (可选)
      ├─ compute_loss_phi()
      │   ├─→ Critic.get_phi() → phi
      │   ├─→ adv_coop = adv_local * phi
      │   └─→ loss_phi
      └─ meta_optimizer.step()
```

---

## 4. 网络结构

### 4.1 Actor网络

| 层级 | 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 1 | state_layer | state[N,S] | state_emb[N,E] | 状态编码 |
| 2 | order_layer | order[N,M,O] | order_emb[N,M,E] | 订单编码 |
| 3 | MatMul | state_emb[N,E]<br>order_emb[N,M,E] | compatibility[N,M] | 兼容性计算 |
| 4 | Softmax+Mask | compatibility[N,M] | probs[N,M] | 概率分布 |

### 4.2 Critic网络

| 层级 | 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 1 | state_layer | state[N,S] | state_emb[N,E] | 状态编码 |
| 2 | local_value_layer | state_emb[N,E] | local_value[N,K+1] | 局部价值 |
| 3 | Phi_layer (可选) | state_emb[N,E] | phi[N,K+1] | 元学习权重 |
| 4 | global_fc_layer (可选) | global_emb | global_emb_f | 全局特征 |
| 5 | global_value_layer (可选) | global_emb_f | global_value[1,1] | 全局价值 |

### 4.3 嵌入层模块

#### state_embedding

```
输入: state[N,S]
  ├─ time_embedding(time) → [N, E]
  ├─ grid_embedding(grid_id) → [N, E]
  └─ contin_embedding(contin) → [N, E]
输出: concat → [N, 3E]
```

#### order_embedding

```
输入: order[N,M,O]
  ├─ grid_embedding(origin_grid) → [N, M, E]
  ├─ grid_embedding(dest_grid) → [N, M, E]
  └─ contin_embedding(contin) → [N, M, E]
输出: concat → FC1 → FC2 → [N, M, E]
```

---

## 5. 训练流程

### 5.1 训练循环

```python
# 数据收集阶段
for step in range(episode_length):
    states, orders, masks = env.get_data()
    processed_state = process_state(states, t)
    processed_order, mask = process_order(orders, t)
    
    actions, value_local, value_global, oldp, ... = ppo.action(...)
    rewards = env.step(actions)
    
    ppo.buffer.push(processed_state, processed_order, actions, 
                    rewards, value_local, value_global, oldp, ...)

# 优势计算
ppo.buffer.finish_path_local(last_local_val)
ppo.buffer.finish_path_global(last_global_val)

# 网络更新
ppo.update(device, writer)
```

### 5.2 Actor更新

```python
# 1. 计算新策略概率
probs = self.actor.multi_mask_forward(state, order, mask_order)
newp = torch.gather(probs, -1, action)[..., -1]

# 2. 计算概率比
ratio = newp / oldp

# 3. PPO裁剪损失
clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantage
loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()

# 4. 熵正则化
ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
loss_pi -= ent_factor * ent

# 5. 反向传播
self.actor_optimizer.zero_grad()
loss_pi.backward()
nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
self.actor_optimizer.step()
```

### 5.3 Critic更新

```python
# 局部价值更新
new_value = self.critic.get_local_value(state)
error = ret - new_value
loss = mse_loss(error).mean()

self.critic_optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
self.critic_optimizer.step()

# 全局价值更新 (如果启用)
if meta_choose > 0:
    new_value_global = self.critic.get_global_value(state)
    loss_global = mse_loss(ret_global - new_value_global).mean()
    
    self.critic_global_optimizer.zero_grad()
    loss_global.backward()
    self.critic_global_optimizer.step()
```

### 5.4 Meta更新

```python
# 1. 计算新策略梯度
loss_term1 = -(ratio_new * adv_global).mean()
grad_term1 = torch.autograd.grad(loss_term1, self.actor_new.parameters())

# 2. 计算旧策略梯度
loss_term2 = ratio_old.mean()
grad_term2 = torch.autograd.grad(loss_term2, self.actor_old.parameters())

# 3. Meta梯度
grad_total = sum((g1 * g2).sum() for g1, g2 in zip(grad_term1, grad_term2))

# 4. 计算协作优势
phi = self.critic.get_phi(state_critic)
adv_coop = (adv_local * phi * neighbor_num).sum(-1)
adv_coop = adv_coop / (phi * neighbor_num).sum(-1)

# 5. Meta损失
loss_phi = grad_total.detach() * adv_coop.mean()

# 6. 反向传播
self.meta_optimizer.zero_grad()
loss_phi.backward()
self.meta_optimizer.step()
```

---

## 6. 文件输入输出

### 6.1 文件输入

#### 6.1.1 数据集文件

**数据集文件格式**：`.pkl`（Pickle格式）

| 文件名 | 路径 | 包含字段 | 说明 | 使用模块 |
|--------|------|----------|------|----------|
| NYU_grid143.pkl | data/ | neighbor, price, order, shape | 纽约大学143网格数据集 | load_envs_NYU143() |
| DiDi_day1_grid121.pkl | data/ | neighbor, price, order, shape | 滴滴121网格数据集 | load_envs_DiDi121() |
| NYC2015Jan26_h3_289_real_orders.pkl | data/ | neighbor, price, order, real_orders, shape | 纽约289网格真实订单数据集 | load_envs_custom() |
| 自定义数据集 | data/ | neighbor, price, order, real_orders, shape | 用户自定义数据 | load_envs_custom() |

**数据集文件详细说明**：

```python
# 数据集文件结构 (以NYU_grid143.pkl为例)
{
    'neighbor': np.ndarray,    # shape: (143, 143)
        # 含义：网格之间的距离层级矩阵
        # 值范围：0-6 (0=同一网格，1-6=不同距离层级，100+=不可达)
        # 用途：用于构建邻接矩阵，支持GAT/GCN/DGCN图神经网络
        # 读取位置：load_envs_NYU143() -> env.neighbor_dis
    
    'price': np.ndarray,      # shape: (10, 2)
        # 含义：不同距离层级的订单价格倍率
        # 格式：[distance_level, price_multiplier]
        # 用途：根据订单起点终点的距离层级计算订单价格
        # 读取位置：load_envs_NYU143() -> env.price_param
    
    'order': np.ndarray,      # shape: (143, 143, 144)
        # 含义：三维订单需求矩阵 (起点网格, 终点网格, 时间步)
        # 维度说明：
        #   - 第一维(143)：起点网格ID
        #   - 第二维(143)：终点网格ID
        #   - 第三维(144)：时间步(1440分钟/10分钟间隔=144步)
        # 值：该时间步从起点到终点的订单数量
        # 用途：环境在每个时间步生成订单的主要依据
        # 读取位置：load_envs_NYU143() -> env.order_param
        # 数据流向：env.order_param -> env.reset() -> order_states -> PPO.process_order()
    
    'shape': tuple,           # e.g., (13, 11)
        # 含义：网格地图的行列数
        # 用途：将一维网格ID映射到二维坐标
        # 读取位置：load_envs_NYU143() -> M, N
}

# 真实订单数据集额外字段
{
    'real_orders': list,      # shape: [N]
        # 含义：真实订单列表
        # 格式：每条订单为 [origin_grid, dest_grid, pickup_time, duration, price, service_type]
        #   - origin_grid: 起点网格ID (int)
        #   - dest_grid: 终点网格ID (int)
        #   - pickup_time: 接单时间 (int)
        #   - duration: 订单持续时间/车程 (int)
        #   - price: 订单真实价格 (float)
        #   - service_type: 服务类型 (int, 可选)
        # 用途：提供真实的订单价格和时空分布，替代采样价格
        # 采样率：可通过 real_order_sample_rate 参数控制 (默认0.1=10%采样)
        # 读取位置：load_envs_custom() -> env.order_real -> env.utility_bootstrap_oneday_order()
        # 数据流向：env.day_orders -> env.get_orders_by_id() -> action_ids -> env.step()
}
```

**数据加载流程**：

```
run_MAPPO.py
    ↓
根据 args.grid_num 选择加载函数
    ├─ grid_num=100: create_OD()
    ├─ grid_num=36: create_OD_36()
    ├─ grid_num=121: load_envs_DiDi121()
    ├─ grid_num=143: load_envs_NYU143()
    └─ 自定义路径: load_envs_custom()
        ↓
加载 .pkl 文件
    ↓
创建 CityReal 环境对象
    ↓
env.reset(mode='PPO2', generate_order=0/1)
    ├─ generate_order=0: 使用真实订单 (real_orders)
    └─ generate_order=1: 使用采样订单 (order_param)
        ↓
返回环境数据
    ├─ states_node: 各网格状态
    ├─ order_states: 各网格订单列表
    └─ order_idx: 订单索引
```

#### 6.1.2 特征归一化文件

**文件路径**：`save/feature_{grid_num}.pkl`

| 文件名 | 用途 | 使用模块 |
|--------|------|----------|
| feature_143.pkl | 143网格的特征最大值范围 | PPO.load_feature_scope() |

**文件结构**：

```python
# feature_143.pkl 结构
{
    'state': np.ndarray,   # shape: (feature_dim,)
        # 含义：状态特征各维度的绝对值最大值
        # 用途：对状态特征进行归一化 (state /= scope)
        # 读取位置：PPO.__init__() -> self.feature_scope
        # 数据流向：PPO.process_state() -> s[:,1:] /= state_scope
    
    'order': np.ndarray,   # shape: (feature_dim,)
        # 含义：订单特征各维度的绝对值最大值
        # 用途：对订单特征进行归一化 (order /= scope)
        # 读取位置：PPO.__init__() -> self.feature_scope
        # 数据流向：PPO.process_order() -> order[:,:,contin_dim:] /= order_scope
}
```

**归一化方式**（feature_normal=3）：

```python
# PPO.process_state()
state[:,1:] /= state_scope[None,1:]  # 状态特征归一化

# PPO.process_order()
order[:,:,contin_dim:] /= order_scope[None,None,contin_dim:]  # 订单特征归一化
```

#### 6.1.3 预训练模型文件

**文件路径**：`{save_dir}/{save_name}.pkl`

| 文件名 | 用途 | 使用模块 |
|--------|------|----------|
| Best.pkl | 最佳性能模型 | PPO.load_param() |
| param.pkl | 定期保存的检查点 | PPO.load_param() |
| iter_{N}.pkl | 定期保存的迭代检查点 | PPO.load_param() |

**文件结构**：

```python
# 模型文件结构
{
    'step': int,                           # 训练步数
    'actor net': state_dict,               # Actor网络权重
        # 包含：state_layer, order_layer 等所有参数
        # 读取位置：PPO.load_param() -> self.actor.load_state_dict()
        # 用途：订单匹配策略推理
    
    'critic net': state_dict,              # Critic网络权重
        # 包含：state_layer, local_value_layer, Phi_layer, global_fc_layer, global_value_layer
        # 读取位置：PPO.load_param() -> self.critic.load_state_dict()
        # 用途：价值估计和元学习权重计算
    
    'actor optimizer': state_dict,         # Actor优化器状态
        # 读取位置：PPO.load_param() -> self.actor_optimizer.load_state_dict()
        # 用途：恢复训练时的优化器状态
    
    'critic optimizer': state_dict         # Critic优化器状态
        # 读取位置：PPO.load_param() -> self.critic_optimizer.load_state_dict()
        # 用途：恢复训练时的优化器状态
}
```

#### 6.1.4 命令行参数

| 参数名称 | 类型 | 默认值 | 说明 | 使用模块 |
|---------|------|--------|------|----------|
| grid_num | int | 143 | 网格数量 | 所有模块 |
| driver_num | int | 2000 | 司机数量 | 环境初始化 |
| batch_size | int | 1000 | 批量大小 | PPO.update() |
| actor_lr | float | 0.001 | Actor学习率 | 优化器配置 |
| critic_lr | float | 0.001 | Critic学习率 | 优化器配置 |
| gamma | float | 0.97 | 折扣因子 | GAE计算 |
| lam | float | 0.95 | GAE参数 | GAE计算 |
| clip_ratio | float | 0.2 | PPO裁剪参数 | PPO裁剪损失 |
| ent_factor | float | 0.005 | 熵系数 | 策略损失 |
| MAX_ITER | int | 6000 | 最大训练迭代数 | 训练循环 |
| steps_per_epoch | int | 144 | 每epoch步数 | 训练循环 |
| use_mdp | int | 0 | MDP模式(0/1/2) | MDP模块 |
| feature_normal | int | 3 | 特征归一化方式 | 特征处理 |
| use_real_orders | bool | False | 是否使用真实订单 | 环境初始化 |
| real_order_sample_rate | float | 0.1 | 真实订单采样率 | 环境初始化 |
| custom_data_path | str | None | 自定义数据路径 | 环境初始化 |
| test | bool | False | 是否测试模式 | run_MAPPO.py |
| model_dir | str | - | 模型加载路径 | 测试模式 |
| resume_iter | int | 0 | 恢复训练的迭代数 | 训练模式 |

**环境变量控制**：

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| COOPRIDE_MAX_ITER | 最大训练迭代数 | `export COOPRIDE_MAX_ITER=10000` |
| COOPRIDE_SAVE_INTERVAL | 模型保存间隔 | `export COOPRIDE_SAVE_INTERVAL=20` |
| COOPRIDE_TEST | 是否测试模式 | `export COOPRIDE_TEST=1` |
| COOPRIDE_MODEL_PATH | 模型路径 | `export COOPRIDE_MODEL_PATH=/path/to/model.pkl` |
| COOPRIDE_GRID_NUM | 网格数量 | `export COOPRIDE_GRID_NUM=289` |
| COOPRIDE_DRIVER_NUM | 司机数量 | `export COOPRIDE_DRIVER_NUM=3000` |
| COOPRIDE_DATA_PATH | 自定义数据路径 | `export COOPRIDE_DATA_PATH=/path/to/data.pkl` |
| COOPRIDE_USE_REAL_ORDERS | 使用真实订单 | `export COOPRIDE_USE_REAL_ORDERS=1` |
| COOPRIDE_REAL_ORDER_SAMPLE_RATE | 真实订单采样率 | `export COOPRIDE_REAL_ORDER_SAMPLE_RATE=0.2` |
| COOPRIDE_RESUME_PATH | 恢复训练路径 | `export COOPRIDE_RESUME_PATH=/path/to/checkpoint.pkl` |
| COOPRIDE_RESUME_ITER | 恢复训练迭代数 | `export COOPRIDE_RESUME_ITER=500` |
| COOPRIDE_DISABLE_NORM | 禁用归一化 | `export COOPRIDE_DISABLE_NORM=1` |

### 6.2 文件输出

#### 6.2.1 模型权重文件

**文件路径**：`{log_dir}/{save_name}.pkl`

| 文件名 | 生成时机 | 包含内容 | 用途 |
|--------|----------|----------|------|
| Best.pkl | GMV创新高时 | 完整模型+优化器 | 最佳性能模型 |
| param.pkl | 每轮训练结束 | 完整模型+优化器 | 定期备份 |
| iter_{N}.pkl | 按间隔保存 | 完整模型+优化器 | 检查点恢复 |

**生成位置**：`run_MAPPO.py`

```python
# 训练时保存
if np.sum(gmv) > best_gmv:
    agent.save_param(args.log_dir, 'Best')

if (iteration + 1) % args.model_save_interval == 0:
    agent.save_param(args.log_dir, f'iter_{iteration + 1}')

# 每轮结束保存
agent.save_param(args.log_dir, 'param')
```

**保存内容**：

```python
state = {
    'step': self.step,
    'actor net': self.actor.state_dict(),
    'critic net': self.critic.state_dict(),
    'actor optimizer': self.actor_optimizer.state_dict(),
    'critic optimizer': self.critic_optimizer.state_dict()
}
```

#### 6.2.2 MDP模型文件

**文件路径**：`{log_dir}/MDP.pkl`

| 模式 | 生成位置 | 包含内容 | 用途 |
|------|----------|----------|------|
| 表格MDP (use_mdp=1) | MdpAgent.save_param() | value_state, n_state | 订单价值估计 |
| 深度MDP (use_mdp=2) | DeepMdpAgent.save_param() | 网络权重+优化器 | 深度价值估计 |

**表格MDP文件结构**（use_mdp=1）：

```python
# MdpAgent保存的文件
{
    'value': np.ndarray,    # shape: (time_len+1, node_num)
        # 含义：每个时间步每个网格的状态价值
        # 用途：通过查表快速估计订单的预期价值
        
    'num': np.ndarray      # shape: (time_len+1, node_num)
        # 含义：每个时间步每个网格的访问次数
        # 用途：用于增量更新价值估计
}
```

**深度MDP文件结构**（use_mdp=2）：

```python
# DeepMdpAgent保存的文件
{
    'net': state_dict,     # 深度神经网络权重
        # 包含：grid_embedding, time_embedding, state_layer1, state_layer2, value_layer
        # 用途：通过神经网络估计状态价值
    
    'optimizer': state_dict # 优化器状态
        # 用途：恢复训练时的优化器状态
}
```

**MDP模型作用详解**：

MDP（Markov Decision Process）模型用于估计接单的长期价值，为订单匹配提供额外的价值特征。

**核心功能**：

1. **订单价值估计**：
   - 计算接单的即时价值（订单价格）
   - 计算接单后的预期未来价值（目标网格的价值状态）
   - 计算接单的机会成本（当前网格的价值状态）

2. **价值计算公式**：
   ```python
   # 表格MDP
   value = price + γ^duration × value[time+duration][dest] - value[time][origin]
   # 其中：
   # - price: 订单价格
   # - γ: 折扣因子
   # - duration: 订单持续时间
   # - value[t][g]: 时间t网格g的状态价值
   ```

3. **数据流向**：
   ```
   订单状态 (origin, dest, price, duration)
       ↓
   DeepMdpAgent.get_value(mdp_state)
       ├─ 计算当前状态价值: value_cur = MDP[t][origin]
       ├─ 计算目标状态价值: value_next = MDP[t+duration][dest]
       └─ 计算订单优势: adv = price + γ^duration × value_next - value_cur
       ↓
   将adv作为额外特征添加到订单
       ↓
   订单状态维度增加: [origin, dest, price, duration, type, entropy, adv]
       ↓
   进入Actor网络进行订单匹配决策
   ```

4. **训练方式**：
   - **表格MDP**：增量更新，类似Q-learning
   - **深度MDP**：监督学习，使用MSE损失训练神经网络
   - **目标标签**：实际接单后获得的回报（GMV）

5. **优势**：
   - 提供长期视角的订单价值评估
   - 平衡短期收益和长期收益
   - 帮助智能体选择更有潜力的订单
   - 特别适用于低频订单区域的决策

#### 6.2.3 TensorBoard日志文件

**文件路径**：`{log_dir}/events.out.tfevents.*`

| 日志类型 | 记录内容 | TensorBoard标签 |
|---------|----------|----------------|
| 训练指标 | ORR, GMV | train ORR, train GMV |
| 损失函数 | actor_loss, critic_loss | train actor loss, train critic local loss, train critic global loss |
| 优化指标 | 熵, KL散度 | train entropy, train kl |
| 优势统计 | 优势均值/标准差 | train adv mean, train adv std |
| 返回值 | 局部/全局返回 | train return local, train return global |
| MDP损失 | MDP价值损失 | train mdp value |
| Phi值 | 元学习权重 | 记录在logs/phi/目录 |

**生成位置**：`run_MAPPO.py`

```python
# TensorBoard记录
writer.add_scalar('train ORR', order_response_rates[-1], iteration)
writer.add_scalar('train GMV', np.sum(gmv), iteration)
writer.add_scalar('train actor loss', loss_actor, self.step)
writer.add_scalar('train critic local loss', loss_critic_local, self.step)
writer.add_scalar('train critic global loss', loss_critic_global, self.step)
writer.add_scalar('train phi loss', loss_phi, self.step)
writer.add_scalar('train entropy', entropy, self.step)
writer.add_scalar('train kl', kl, self.step)
```

**查看方式**：

```bash
# 启动TensorBoard
tensorboard --logdir=logs/synthetic/grid143/{exp_name}

# 浏览器访问
http://localhost:6006
```

#### 6.2.4 设置文件

**文件路径**：`{log_dir}/setting.txt`

| 内容 | 说明 |
|------|------|
| 完整参数列表 | 记录训练时所有args参数 |
| 参数值 | 每个参数的具体数值 |

**生成位置**：`run_MAPPO.py`

```python
# 保存训练设置
with open(log_dir + '/setting.txt', 'w') as f:
    for key, value in args_dict.items():
        f.writelines(key + ' : ' + str(value) + '\n')
```

**用途**：
- 训练结果的可复现性
- 实验对比和调试
- 记录超参数配置

#### 6.2.5 特征日志文件

**文件路径**：`{log_dir}/feature/`

| 文件名 | 内容 | 生成位置 |
|--------|------|----------|
| feature.txt | 状态和订单特征的统计信息 | logs.save_feature() |

**用途**：监控特征分布，辅助调试

#### 6.2.6 分布日志文件

**文件路径**：`{log_dir}/distribution/{iteration}.csv`

| 文件名 | 内容 | 生成位置 |
|--------|------|----------|
| {iteration}.csv | 每时间步的奖励、司机数、订单数分布 | logs.save_log_distribution() |

**数据格式**：

```csv
time_step,grid_id,reward,driver_num,order_num
0,0,10.5,5,3
0,1,8.2,4,2
...
```

#### 6.2.7 Phi日志文件

**文件路径**：`{log_dir}/phi/`

| 文件名 | 内容 | 生成位置 |
|--------|------|----------|
| phi_{step}.csv | 元学习权重Phi的分布 | logs.save_log_phi() |
| phi_{iteration}.npy | Phi函数完整数据 | logs.save_full_phi() |

**数据格式**（phi_{step}.csv）：

```csv
phi_0,phi_1,phi_2,phi_3,phi_4
0.1,0.2,0.3,0.25,0.15
...
```

### 6.3 输入输出映射表

#### 6.3.1 数据流向总图

```
【输入层】
├─ 数据集文件 (.pkl)
│   ├─ neighbor → env.neighbor_dis → adj → GNN模块
│   ├─ price → env.price_param → 订单价格计算
│   ├─ order → env.order_param → 订单生成
│   └─ real_orders → env.order_real → 真实订单
│
├─ 特征文件 (feature_143.pkl)
│   ├─ state_scope → PPO.process_state() → 状态归一化
│   └─ order_scope → PPO.process_order() → 订单归一化
│
├─ 预训练模型 (.pkl)
│   ├─ actor net → self.actor → 策略推理
│   ├─ critic net → self.critic → 价值估计
│   └─ optimizer states → 训练恢复
│
└─ 命令行参数/环境变量
    ├─ 网络结构参数 → Actor/Critic初始化
    ├─ 训练超参数 → 优化器/损失函数
    └─ 模式控制 → 训练/测试/MDP等

【处理层】
├─ CityReal环境
│   ├─ reset() → states, orders, order_idx
│   ├─ get_orders_by_id() → 执行的订单列表
│   └─ step() → next_states, rewards
│
├─ PPO类
│   ├─ process_state() → 归一化状态张量
│   ├─ process_order() → 归一化订单张量
│   └─ action() → action, values, logp
│
├─ Actor网络
│   ├─ state_layer → state_embedding
│   ├─ order_layer → order_embedding
│   └─ forward() → probs
│
└─ Critic网络
    ├─ get_local_value() → local_value
    ├─ get_global_value() → global_value
    └─ get_phi() → meta_weights

【输出层】
├─ 模型权重
│   ├─ Best.pkl → 最佳性能检查点
│   ├─ param.pkl → 定期备份
│   └─ MDP.pkl → MDP模型
│
├─ 训练日志
│   ├─ events.out.tfevents.* → TensorBoard日志
│   ├─ setting.txt → 超参数记录
│   ├─ feature/ → 特征统计
│   ├─ distribution/ → 分布日志
│   └─ phi/ → 元学习权重
│
└─ 评估指标
    ├─ train ORR → 训练订单响应率
    ├─ train GMV → 训练总交易额
    ├─ Best_ORR → 最佳ORR
    └─ Best_GMV → 最佳GMV
```

#### 6.3.2 模块间数据传递表

| 源模块 | 目标模块 | 数据 | 传递方式 | 说明 |
|--------|----------|------|----------|------|
| env | PPO.process_state() | states_node | 返回值 | 原始网格状态 |
| env | PPO.process_order() | order_states | 返回值 | 原始订单列表 |
| PPO.process_state() | Actor.forward() | state张量 | 参数 | 归一化后的状态 |
| PPO.process_order() | Actor.forward() | order张量 | 参数 | 归一化后的订单 |
| Actor.forward() | env.get_orders_by_id() | action_ids | 返回值 | 选中的订单ID |
| env.step() | PPO.buffer.push() | rewards | 参数 | 环境奖励 |
| PPO.buffer | PPO.update() | batch数据 | 返回值 | 训练批次 |
| PPO.update() | writer.add_scalar() | 损失/指标 | 参数 | TensorBoard记录 |
| PPO.save_param() | 磁盘 | 模型权重 | 文件写入 | 保存检查点 |
| MDP | PPO.process_order() | 订单优势 | 特征 | MDP价值特征 |
| PPO | MDP.push() | 订单状态 | 参数 | MDP训练数据 |
| PPO | MDP.update() | - | 调用 | MDP更新 |

#### 6.3.3 测试数据输入路径与格式

**测试模式启动**：

```bash
# 通过环境变量启用测试模式
export COOPRIDE_TEST=1
export COOPRIDE_MODEL_PATH=/path/to/Best.pkl
python run/run_MAPPO.py
```

**测试数据输入**：

| 输入项 | 路径/格式 | 说明 | 代码位置 |
|--------|-----------|------|----------|
| 预训练模型 | args.model_dir | Best.pkl 或 iter_{N}.pkl | run_MAPPO.py line 47 |
| 数据集 | data/*.pkl | 同训练时的数据集 | run_MAPPO.py line 350-368 |
| 测试种子 | args.TEST_SEED | 默认1314520 | run_MAPPO.py line 51 |
| 测试迭代数 | args.TEST_ITER | 默认1 | run_MAPPO.py line 50 |

**测试函数**（test()）：

```python
# run_MAPPO.py line 220-311

def test(env, agent, writer=None, args=None, device='cpu'):
    # 1. 设置随机种子
    np.random.seed(args.TEST_SEED)
    
    # 2. 加载预训练模型
    agent.load_param(args.model_dir)
    
    # 3. 测试循环
    for iteration in range(args.TEST_ITER):
        # 重置环境
        states_node, _, order_states, order_idx, ... = env.reset(
            mode='PPO2', 
            generate_order=0  # 使用真实订单
        )
        
        # 4. 推理循环
        for T in range(args.TIME_LEN):
            # 生成动作（不采样，贪婪策略）
            action, ..., action_ids, ... = agent.action(
                state, order, ...,
                device=device,
                sample=False,  # 贪婪策略
                random_action=False
            )
            
            # 5. 执行动作
            orders = env.get_orders_by_id(action_ids)
            next_states_node, ... = env.step(orders, generate_order=0, mode='PPO2')
            
            # 6. 记录指标
            gmv.append(env.gmv)
            order_response_rates.append(env.order_response_rate)
    
    # 7. 输出测试结果
    print(f'Test GMV: {np.sum(gmv)}, Test ORR: {order_response_rates[-1]}')
```

**测试输出**：

| 输出项 | 格式 | 说明 |
|--------|------|------|
| GMV | float | 测试总交易额 |
| ORR | float | 测试订单响应率 |
| distribution/ | CSV | 每时间步的分布数据 |
| 时间消耗 | float | 测试用时 |

**测试命令示例**：

```bash
# 1. 使用默认测试配置
export COOPRIDE_TEST=1
python run/run_MAPPO.py

# 2. 指定模型路径
export COOPRIDE_TEST=1
export COOPRIDE_MODEL_PATH=logs/synthetic/grid143/xxx/Best.pkl
python run/run_MAPPO.py

# 3. 指定数据集
export COOPRIDE_TEST=1
export COOPRIDE_DATA_PATH=data/NYC2015Jan26_h3_289_real_orders.pkl
export COOPRIDE_MODEL_PATH=logs/xxx/Best.pkl
python run/run_MAPPO.py

# 4. 使用真实订单测试
export COOPRIDE_TEST=1
export COOPRIDE_USE_REAL_ORDERS=1
export COOPRIDE_REAL_ORDER_SAMPLE_RATE=1.0
python run/run_MAPPO.py
```

---

## 7. 关键技术点

### 7.1 GAE (Generalized Advantage Estimation)

**功能**: 计算优势函数，减少方差

```python
deltas = reward + gamma * value_next - value
advantage = torch.zeros_like(deltas)
advantage[:, -1] = deltas[:, -1]

for i in range(deltas.shape[1]-2, -1, -1):
    advantage[:, i] = deltas[:, i] + advantage[:, i+1] * (gamma * lam)
```

**优点**:
- 减少方差
- 自适应平衡偏差和方差

### 7.2 PPO裁剪

**功能**: 限制策略更新幅度

```python
ratio = new_prob / old_prob
clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantage
loss = -torch.min(ratio * advantage, clip_adv).mean()
```

**作用**:
- 防止策略更新过大
- 稳定训练过程

### 7.3 图神经网络

| 类型 | 用途 | 特点 |
|------|------|------|
| GAT | 图注意力 | 多头注意力机制 |
| GCN | 图卷积 | 邻居聚合 |
| DGCN | 扩散图卷积 | 捕获时空依赖 |

### 7.4 元学习

**功能**: 动态加权局部价值

```python
phi = self.critic.get_phi(state_critic)
adv_coop = (adv_local * phi * neighbor_num).sum(-1)
adv_coop = adv_coop / (phi * neighbor_num).sum(-1)
```

**作用**:
- 自适应平衡局部和全局价值
- 支持多层级决策

### 7.5 辅助任务

**功能**: 状态预测辅助损失

```python
next_state_pred = net.auxiliary_emb(state)
next_state_label = net.get_state_emb(next_state).detach()
auxi_loss = mse_loss(next_state_pred - next_state_label).mean()
```

**作用**:
- 提升状态表征质量
- 加速训练收敛

### 7.6 价值标准化

**功能**: PopArt风格归一化

```python
value_normalizer.update(ret)
ret = value_normalizer.normalize(ret)
```

**作用**:
- 稳定价值估计
- 提升训练稳定性

---

## 8. 神经网络详细信息

### 8.1 网络总览

| 网络名称 | 代码位置 | 功能 | 训练/推理 | 参数量 |
|---------|---------|------|----------|--------|
| Actor | ~line 430-530 | 订单匹配策略 | 训练+推理 | ~700K |
| Critic | ~line 593-735 | 价值估计 | 训练+推理 | ~1M |
| DeepMdpAgent | ~line 770-870 | MDP价值估计 | 训练(可选) | ~92K |

### 8.2 Actor网络

**基本信息**:
- 代码位置: ~line 430-530
- 权重加载: ~line 1100-1120
- 权重保存: ~line 920
- 权重读入: ~line 945

**输入输出**:
- 输入: state[N,S], order[N,M,O], mask[N,M]
- 输出: probs[N,M], hidden_state

**依赖子网络**:
- state_representation1: 状态表征
- order_embedding: 订单表征

### 8.3 Critic网络

**基本信息**:
- 代码位置: ~line 593-735
- 权重加载: ~line 1120-1140
- 权重保存: ~line 920
- 权重读入: ~line 945

**输入输出**:
- forward(): 输入state[N,S], 输出value_local[N,K+1]
- get_global_value(): 输入state[N,S], 输出value_global[1,1]
- get_phi(): 输入state[N,S], 输出phi[N,K+1]

**依赖子网络**:
- state_representation1: 状态表征
- local_value_layer: 局部价值
- global_fc_layer: 全局特征(可选)
- global_value_layer: 全局价值(可选)
- Phi_layer: 元学习权重(可选)

### 8.4 DeepMdpAgent

**基本信息**:
- 代码位置: ~line 770-870
- 权重加载: ~line 1060-1065
- 权重保存: ~line 850

**输入输出**:
- get_value(): 输入state[T,D,2], 输出value[T,D,1]
- update(): 输入state, targetV, 输出loss

**功能**:
- MDP价值估计
- 为订单添加MDP价值特征

### 8.5 权重管理

**保存格式**:
```python
state = {
    'step': self.step,
    'actor net': self.actor.state_dict(),
    'critic net': self.critic.state_dict(),
    'actor optimizer': self.actor_optimizer.state_dict(),
    'critic optimizer': self.critic_optimizer.state_dict()
}
torch.save(state, save_dir + '/' + save_name + '.pkl')
```

**加载格式**:
```python
state = torch.load(load_dir)
self.actor.load_state_dict(state['actor net'])
self.critic.load_state_dict(state['critic net'])
```

### 8.6 优化器配置

| 网络 | 优化器 | 学习率 | 更新方法 |
|------|--------|--------|----------|
| Actor | Adam | actor_lr(0.001) | actor_optimizer.step() |
| Critic(局部) | Adam | critic_lr(0.001) | critic_optimizer.step() |
| Critic(全局) | Adam | critic_lr(0.001) | critic_global_optimizer.step() |
| Meta(Phi) | Adam | meta_lr(0.001) | meta_optimizer.step() |
| DeepMdpAgent | Adam | 0.001 | optimizer.step() |

### 8.7 张量维度总结

| 网络 | 输入维度 | 输出维度 | 说明 |
|------|---------|---------|------|
| Actor | state[N,S]<br>order[N,M,O] | probs[N,M] | N=agent_num, M=max_order_num |
| Critic | state[N,S] | local_value[N,K+1] | K=meta_scope |
| DeepMdpAgent | state[T,D,2] | value[T,D,1] | T=time, D=driver_num |

### 8.8 参数量估计

**假设配置**:
- grid_num = 143
- embedding_dim = 128
- meta_scope = 3
- max_order_num = 100

| 网络 | 各层参数 | 总计 |
|------|----------|------|
| Actor | state_rep1(~500K)<br>+order_emb(~200K) | ~700K |
| Critic | state_rep1(~500K)<br>+local(~0.5K)<br>+global(~500K)<br>+phi(~0.5K) | ~1M |
| DeepMdpAgent | embeddings(~27K)<br>+fc(~65K) | ~92K |
| 总计 | | ~1.8M |

---

## 9. 使用示例

### 9.1 初始化

```python
from algo.MAPPO import PPO

ppo = PPO(env, args, device='cuda')
```

### 9.2 训练

```python
for episode in range(num_episodes):
    for step in range(episode_length):
        state, order, mask = env.step()
        action, value_local, value_global, oldp, ... = ppo.action(...)
        reward = env.execute(action)
        ppo.buffer.push(state, order, action, reward, ...)
    
    ppo.buffer.finish_path_local(...)
    ppo.buffer.finish_path_global(...)
    ppo.update(device, writer)
```

### 9.3 推理

```python
action, value_local, value_global, ... = ppo.action(
    state, order, mask, device='cuda', 
    sample=False  # 贪婪策略
)
```

### 9.4 保存加载

```python
# 保存
ppo.save_param(save_dir, 'model')

# 加载
ppo.load_param(load_dir)
```

---

## 10. 总结

### 核心特点

1. **多智能体协同**: 支持多个网格区域的智能体协同决策
2. **图神经网络**: 利用空间邻接关系增强表征学习
3. **PPO算法**: 稳定的策略优化算法
4. **元学习**: 支持多层级价值估计
5. **辅助任务**: 提升训练效果和稳定性

### 适用场景

- 网约车动态定价与订单分配
- 城市交通资源调度
- 大规模多智能体决策系统

### 扩展性

- 支持多种图神经网络 (GAT/GCN/DGCN)
- 支持RNN处理序列数据
- 支持多种辅助任务
- 可配置的元学习机制

---

**文档生成时间**: 2026-01-30
**代码版本**: CoopRide-main