import os
import numpy as np
import pickle 
import sys 
sys.path.append('../')
from simulator.envs import *
sys.path.append('../coopride_llm')
from log import Logger

def load_envs_DiDi121(driver_num=2000):
    # 场景控制开关定义
    enable_tidal = False          # 控制潮汐通勤场景
    enable_burst = False          # 控制局部突发需求场景
    enable_long_tail = False      # 控制长尾跨区需求场景
    enable_supply_decay = False   # 控制全局运力衰减场景
    Logger.info("=" * 50)  # 分隔线
    Logger.info(f"enable_tidal: {enable_tidal}, enable_burst: {enable_burst}, enable_long_tail: {enable_long_tail}")  # 分隔线
    Logger.info("=" * 50)  # 分隔线

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    primary_path = os.path.join(data_dir, "DiDi", "DiDi_day1_grid121.pkl")
    fallback_path = os.path.join(data_dir, "DiDi_day1_grid121.pkl")
    data_path = primary_path if os.path.exists(primary_path) else fallback_path
    with open(data_path, 'rb') as handle:
        data_param = pickle.load(handle) 
    dura_param = data_param['duration'] # 每
    price_param = data_param['price']   # 每级邻居的价格    第0级表示自己网格 0~l_max
    neighbor = data_param['neighbor']    # neighbor>=100 表示不可达的订单
    order_param = data_param['order']   # shape=(11,11,144)  表示 (出发地，目的地，出发时间)
    l_max = 8   # 最大通勤跨邻居数
    M,N = 11,11
    price_param[:,1]/=2     # 减小订单的方差
    np.random.seed(0)
    # 减小order param 数量
    commute1 = order_param.astype(np.float32)
    commute1[(neighbor==0)] = 0
    index = commute1>=3
    commute1[index] = (commute1[index]-3)*0.2+3
    index = commute1>=2
    commute1 = np.round(commute1/2+0.1)
    commute1[neighbor==100]=0
    random_delete = np.random.randint(0,3,(commute1.shape))
    random_delete[commute1.sum(-1).sum(-1)<3000] = 0
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 添加与邻居相关的随机数据
    random_grid_num = np.random.randint(1,6,M*N)
    random_prob = np.zeros((M*N,M*N))
    
    # ---------------------------------------------------------
    # 场景三：长尾跨区需求场景控制逻辑 (部分1/2)
    # 物理意义：模拟城郊长途出行需求激增对运力周转率的拉伸效应
    # ---------------------------------------------------------
    if enable_long_tail:
        # 将概率权重向高层级邻接节点偏移
        prob_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.2, 0.14]
    else:
        prob_list = [0.05, 0.2, 0.4, 0.15, 0.1, 0.05, 0.05]
    
    for k in range(7):
        index= neighbor==k
        random_prob+= prob_list[k]/np.sum(index,axis=-1,keepdims=True)*index
    random_add = np.zeros(commute1.shape)
    for i in range(M*N):
        sample = np.random.choice(M*N,size = (random_grid_num[i],commute1.shape[-1]),replace=True, p=random_prob[i])    
        for t in range(commute1.shape[-1]):
            random_add[i,sample[:,t],t] = 1
    commute1+=random_add
    # 删除数量多的
    random_delete = np.random.randint(0,2,(commute1.shape))
    #random_delete[random_delete<=1] = 0
    index = commute1.sum(1)<=25
    random_delete = random_delete.swapaxes(1,2)
    random_delete[index] = 0
    random_delete = random_delete.swapaxes(1,2)
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    # 全部删除一点点
    random_delete = np.random.randint(0,5,(commute1.shape))
    
    # ---------------------------------------------------------
    # 场景三：长尾跨区需求场景控制逻辑 (部分2/2)
    # ---------------------------------------------------------
    if enable_long_tail:
        # 对跨度大于等于4个网格的订单实施保护机制
        random_delete[neighbor >= 4] = 0
    
    random_delete[random_delete<=3] = 0
    random_delete[random_delete>3] = 1
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    
    # ---------------------------------------------------------
    # 场景一：潮汐交通控制逻辑
    # 物理映射：早晚高峰期间居住区与商业区之间的定向通勤洪峰
    # ---------------------------------------------------------
    if enable_tidal:
        total_grids = M * N
        commercial_threshold = total_grids // 3
        
        # 构建一维网格索引数组
        commercial_grids = np.arange(commercial_threshold)
        residential_grids = np.arange(commercial_threshold, total_grids)
        
        # 封装时间切片对象以提升代码复用性与可读性
        morning_peak_slice = slice(42, 55)
        evening_peak_slice = slice(102, 115)
        
        # 早高峰：居住区向商业区的单向流量放大
        # 利用 reshape(-1, 1) 将源节点数组转为列向量，触发 NumPy 广播机制完成子矩阵的批量标量乘法
        commute1[residential_grids.reshape(-1, 1), commercial_grids, morning_peak_slice] *= 30.0
            
        # 晚高峰：商业区向居住区的单向流量放大
        commute1[commercial_grids.reshape(-1, 1), residential_grids, evening_peak_slice] *= 30.0

    # ---------------------------------------------------------
    # 场景二：局部突发需求控制逻辑
    # 物理映射：大型集会散场引发的瞬时高并发跨区出行需求
    # ---------------------------------------------------------
    if enable_burst:
        event_grid_1, event_grid_2 = 45, 88
        burst_slice = slice(50, 65)
        
        # 提取合法的目的地掩码：距离在 1 到 l_max 之间（排除自身和不可达网格）
        valid_dest_1 = (neighbor[event_grid_1] > 0) & (neighbor[event_grid_1] <= l_max)
        valid_dest_2 = (neighbor[event_grid_2] > 0) & (neighbor[event_grid_2] <= l_max)
        
        # 仅对合法范围内的目的地注入瞬时脉冲流量
        commute1[event_grid_1, valid_dest_1, burst_slice] += 100
        commute1[event_grid_2, valid_dest_2, burst_slice] += 100
    
    order_param = commute1.astype(np.int32)
    # 初始化司机数量
    driver_param=np.zeros(M*N,dtype=np.int32)+1
    order_num = np.sum(np.sum(order_param, axis=1), axis=1)
    driver_param[order_num>=100]= driver_param[order_num>=100]+3
    driver_param[order_num>=400]= driver_param[order_num>=400]+3
    driver_param[order_num>=800]= driver_param[order_num>=800]+3
    driver_param = driver_param*driver_num/np.sum(driver_param)
    driver_param = driver_param.astype(np.int32)
    random_add = np.random.choice(M*N, driver_num-np.sum(driver_param), replace = True)
    for dri in random_add:
        driver_param[dri] += 1
    # 统计数量特别多的网格id
    large_grid_dist={i:0 for i in range(121)}
    a= np.sum(order_param, axis=1)
    for i in range(144):
        b = np.where(a[:,i]>=100)[0]
        for n in b:
            large_grid_dist[n]+=1
    large_grid=[]
    for k,v in large_grid_dist.items():
        if v>0:
            large_grid.append(k)
    print('订单数量: {} , 司机数量: {}'.format( np.sum(order_num), np.sum(driver_param)))
    Logger.info(f"订单数量: {np.sum(order_num)}, 司机数量: {np.sum(driver_param)}")

    # 处理为envs的参数
    mapped_matrix_int = np.arange(M*N)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(M,N))
    order_num = np.sum(order_param, axis=1)
    order_num_dict = []
    for t in range(144):
        order_num_dict.append( {i:[order_num[i,t]] for i in range(M*N)} )
    idle_driver_location_mat = np.zeros((144, M*N))
    for t in range(144):
        idle_driver_location_mat[t] = driver_param
    order_time = [0.2, 0.2, 0.15,       # 没用
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]
    n_side = 6
    order_real = []
    onoff_driver_location_mat=[]
    env = CityReal(mapped_matrix_int, order_num_dict, [], idle_driver_location_mat,
                   order_time, price_param, l_max, M, N, n_side, 144, 1, np.array(order_real),
                   np.array(onoff_driver_location_mat), neighbor_dis = neighbor , order_param=order_param ,fleet_help=False)
    return env, M, N, None, M*N



def load_envs_NYU143(driver_num=2000):
    """
    加载 NYU 143 网格数据集并创建仿真环境
    
    数据处理流程说明：
    1. 数据加载：从 pickle 文件加载包含价格、邻居关系、订单分布等信息的参数
    2. 订单处理：对原始订单数据进行多步调整，包括数量压缩、随机增删、邻居相关扰动等
    3. 司机初始化：根据订单量分布初始化司机位置，并随机调整至指定总数
    4. 环境构建：将处理后的数据封装为 CityReal 仿真环境对象
    
    Args:
        driver_num: 初始司机数量，默认为 2000
        
    Returns:
        env: CityReal 仿真环境对象
        M: 网格行数
        N: 网格列数
        None: 保留字段（未使用）
        M*N: 网格总数
    """
    # 场景控制开关定义
    enable_tidal = False          # 控制潮汐通勤场景
    enable_burst = False          # 控制局部突发需求场景
    enable_long_tail = False      # 控制长尾跨区需求场景
    enable_supply_decay = False   # 控制全局运力衰减场景
    Logger.info("=" * 50)  # 分隔线
    Logger.info(f"enable_tidal: {enable_tidal}, enable_burst: {enable_burst}, enable_long_tail: {enable_long_tail}")  # 分隔线
    Logger.info("=" * 50)  # 分隔线

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    primary_path = os.path.join(data_dir, "NYU", "NYU_grid143.pkl")
    fallback_path = os.path.join(data_dir, "NYU_grid143.pkl")
    data_path = primary_path if os.path.exists(primary_path) else fallback_path
    with open(data_path, 'rb') as handle:
        data_param = pickle.load(handle) 
    # 从数据文件中提取关键参数
    price_param = data_param['price']       # 每级邻居的价格，第0级表示自己网格，范围 0~l_max
    neighbor = data_param['neighbor']        # 邻居距离矩阵，>=100 表示不可达的订单
    order_param = data_param['order']       # 订单分布矩阵，shape=(出发地, 目的地, 出发时间)
    M,N = data_param['shape']                # 网格尺寸：M行N列
    l_max = 6                                # 最大通勤跨邻居级数
    
    # ========== 步骤1：减小订单参数数量 ==========
    np.random.seed(0)                         # 固定随机种子保证可复现
    commute1 = order_param.astype(np.float32) # 转换为浮点类型便于处理
    
    # 将不可达网格的订单清零（neighbor==0 表示同一网格内的订单不处理）
    commute1[(neighbor==0)] = 0
    
    # 对大订单数量进行压缩（大于等于3的订单）
    index = commute1>=3
    commute1[index] = (commute1[index]-3)*0.2+3  # 使用0.2系数压缩，保持基数为3
    
    # 对中等订单数量进行压缩（大于等于2的订单）
    index = commute1>=2
    commute1 = np.round(commute1/2+0.1)         # 除以2后四舍五入
    
    # 将完全不可达的订单清零（neighbor>=100）
    commute1[neighbor==100]=0
    # 随机删除部分订单（0-2个），但保留订单总量较少的区域（<3000）
    random_delete = np.random.randint(0,3,(commute1.shape))
    random_delete[commute1.sum(-1).sum(-1)<3000] = 0  # 订单总量少的区域不删除
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0  # 防止订单数为负
    
    # 针对网格68进行额外的随机删除（0-1个）
    random_delete = np.random.randint(0,2,(commute1.shape[1],commute1.shape[2]))
    commute1[68]-=random_delete
    commute1[commute1<0] = 0  # 防止订单数为负
    # ========== 步骤2：添加与邻居相关的随机订单数据 ==========
    
    # 为每个网格随机生成1-5个额外订单
    random_grid_num = np.random.randint(1,6,M*N)
    
    # 构建基于邻居距离的概率分布矩阵
    random_prob = np.zeros((M*N,M*N))
    
    # ---------------------------------------------------------
    # 场景三：长尾跨区需求场景控制逻辑 (部分1/2)
    # 物理意义：模拟城郊长途出行需求激增对运力周转率的拉伸效应
    # ---------------------------------------------------------
    if enable_long_tail:
        # 将概率权重向高层级邻接节点偏移
        prob_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.2, 0.14]
    else:
        prob_list = [0.05, 0.2, 0.4, 0.15, 0.1, 0.05, 0.05]
    
    for k in range(7):
        index= neighbor==k
        # 将权重归一化后分配给对应邻居级别的网格
        random_prob+= prob_list[k]/np.sum(index,axis=-1,keepdims=True)*index
    
    # 根据概率分布生成随机订单
    random_add = np.zeros(commute1.shape)
    for i in range(M*N):
        # 为网格i在每个时间段随机选择目标网格
        sample = np.random.choice(M*N,size = (random_grid_num[i],commute1.shape[-1]),replace=True, p=random_prob[i])    
        for t in range(commute1.shape[-1]):
            random_add[i,sample[:,t],t] = 1  # 添加1个订单
    commute1+=random_add
    # ========== 步骤3：删除订单数量较多的区域 ==========
    
    random_delete = np.random.randint(0,2,(commute1.shape))
    # 找出总订单量<=25的网格（订单较少），不对这些网格进行随机删除
    index = commute1.sum(1)<=25
    random_delete = random_delete.swapaxes(1,2)   # 交换轴以应用索引
    random_delete[index] = 0                      # 订单少的网格不删除
    random_delete = random_delete.swapaxes(1,2)  # 恢复原始轴顺序
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0
    
    # ========== 步骤4：全局少量删除订单（约20%概率）==========
    random_delete = np.random.randint(0,5,(commute1.shape))
    # ---------------------------------------------------------
    # 场景三：长尾跨区需求场景控制逻辑 (部分2/2)
    # ---------------------------------------------------------
    if enable_long_tail:
        # 对跨度大于等于4个网格的订单实施保护机制
        random_delete[neighbor >= 4] = 0
    random_delete[random_delete<=3] = 0   # 0-3不删除
    random_delete[random_delete>3] = 1    # 4才删除，即约20%删除率
    commute1 = commute1-random_delete
    commute1[commute1<0] = 0

    # ---------------------------------------------------------
    # 场景一：潮汐交通控制逻辑
    # 物理映射：早晚高峰期间居住区与商业区之间的定向通勤洪峰
    # ---------------------------------------------------------
    if enable_tidal:
        total_grids = M * N
        commercial_threshold = total_grids // 3
        
        # 构建一维网格索引数组
        commercial_grids = np.arange(commercial_threshold)
        residential_grids = np.arange(commercial_threshold, total_grids)
        
        # 封装时间切片对象以提升代码复用性与可读性
        morning_peak_slice = slice(42, 55)
        evening_peak_slice = slice(102, 115)
        
        # 早高峰：居住区向商业区的单向流量放大
        # 利用 reshape(-1, 1) 将源节点数组转为列向量，触发 NumPy 广播机制完成子矩阵的批量标量乘法
        commute1[residential_grids.reshape(-1, 1), commercial_grids, morning_peak_slice] *= 30.0
            
        # 晚高峰：商业区向居住区的单向流量放大
        commute1[commercial_grids.reshape(-1, 1), residential_grids, evening_peak_slice] *= 30.0

    # ---------------------------------------------------------
    # 场景二：局部突发需求控制逻辑
    # 物理映射：大型集会散场引发的瞬时高并发跨区出行需求
    # ---------------------------------------------------------
    if enable_burst:
        event_grid_1, event_grid_2 = 45, 88
        burst_slice = slice(50, 65)
        
        # 提取合法的目的地掩码：距离在 1 到 l_max 之间（排除自身和不可达网格）
        valid_dest_1 = (neighbor[event_grid_1] > 0) & (neighbor[event_grid_1] <= l_max)
        valid_dest_2 = (neighbor[event_grid_2] > 0) & (neighbor[event_grid_2] <= l_max)
        
        # 仅对合法范围内的目的地注入瞬时脉冲流量
        commute1[event_grid_1, valid_dest_1, burst_slice] += 100
        commute1[event_grid_2, valid_dest_2, burst_slice] += 100

    # 转换回整数类型
    order_param = commute1.astype(np.int32)
    # ========== 步骤5：初始化司机数量 ==========
    
    # 初始均匀分配：每个网格1个司机
    driver_param=np.ones(M*N,dtype=np.int32)
    
    # 按比例缩放至目标司机总数
    driver_param = driver_param*driver_num/np.sum(driver_param)
    driver_param = driver_param.astype(np.int32)
    
    # 处理四舍五入后的余数，随机分配到某些网格
    random_add = np.random.choice(M*N, driver_num-np.sum(driver_param), replace = True)
    for dri in random_add:
        driver_param[dri] += 1
    # ========== 步骤6：统计高订单网格 ==========
    
    # 找出在不同时间段订单数>=100的网格
    large_grid_dist={i:0 for i in range(121)}
    a= np.sum(order_param, axis=1)  # 按目的地聚合
    for i in range(144):
        b = np.where(a[:,i]>=100)[0]  # 查找订单>=100的出发地
        for n in b:
            large_grid_dist[n]+=1
    large_grid=[]
    for k,v in large_grid_dist.items():
        if v>0:
            large_grid.append(k)
    order_num = order_param.sum()
    print('订单数量: {} , 司机数量: {}'.format( np.sum(order_num), np.sum(driver_param)))
    Logger.info(f"订单数量: {np.sum(order_num)}, 司机数量: {np.sum(driver_param)}")

    # ========== 步骤7：转换为环境所需参数格式 ==========
    
    # 创建网格ID映射矩阵
    mapped_matrix_int = np.arange(M*N)
    mapped_matrix_int=np.reshape(mapped_matrix_int,(M,N))
    
    # 计算每个网格在每个时间段的订单总数
    order_num = np.sum(order_param, axis=1)
    
    # 转换为字典列表格式：[{网格ID: [订单数]}, ...]，每个时间片一个字典
    order_num_dict = []
    for t in range(144):
        order_num_dict.append( {i:[order_num[i,t]] for i in range(M*N)} )
    # 构建空闲司机位置矩阵：(时间段, 网格)
    # 假设每个时间段开始时，司机分布相同
    idle_driver_location_mat = np.zeros((144, M*N))
    for t in range(144):
        idle_driver_location_mat[t] = driver_param
    
    # 订单时长概率分布（9类），当前未使用
    order_time = [0.2, 0.2, 0.15,       # 没用
                  0.15, 0.1, 0.1,
                  0.05, 0.04, 0.01]
    n_side = 6
    
    # 真实订单和上下线司机（当前未使用）
    order_real = []
    onoff_driver_location_mat=[]
    # 创建 CityReal 仿真环境对象
    env = CityReal(
        mapped_matrix_int,           # 网格ID映射
        order_num_dict,              # 订单数量字典
        [],                          # 真实订单（未使用）
        idle_driver_location_mat,    # 空闲司机位置矩阵
        order_time,                  # 订单时长分布
        price_param,                 # 价格参数
        l_max,                       # 最大通勤距离
        M, N,                        # 网格尺寸
        n_side,                      # 邻居边数
        144,                         # 时间步数
        1,                           # 订单概率
        np.array(order_real),        # 真实订单数组（空）
        np.array(onoff_driver_location_mat),  # 上下线司机（空）
        neighbor_dis=neighbor,       # 邻居距离矩阵
        order_param=order_param,     # 订单参数矩阵
        fleet_help=False             # 不使用车队帮助
    )
    
    return env, M, N, None, M*N


def load_envs_custom(data_path, driver_num=2000, use_real_orders=False, real_order_sample_rate=0.1):
    """
    加载自定义数据集
    
    Args:
        data_path: PKL 数据文件路径
        driver_num: 司机数量
        use_real_orders: 是否使用真实订单数据（包含真实价格）
        real_order_sample_rate: 真实订单采样率 (0.0-1.0)，默认 0.1 (10%)
    """
    with open(data_path, 'rb') as handle:
        data_param = pickle.load(handle)
    neighbor = data_param['neighbor']
    price_param = data_param['price']
    order_param = data_param['order']
    shape = data_param.get('shape', None)
    
    # 加载真实订单数据（如果有且启用）
    real_orders_data = data_param.get('real_orders', None)
    if use_real_orders and real_orders_data is not None and len(real_orders_data) > 0:
        order_real_full = real_orders_data.tolist() if hasattr(real_orders_data, 'tolist') else list(real_orders_data)
        
        # 对真实订单进行采样以提高训练速度
        if real_order_sample_rate < 1.0:
            np.random.seed(42)  # 固定种子保证可复现
            sample_size = max(1, int(len(order_real_full) * real_order_sample_rate))
            indices = np.random.choice(len(order_real_full), sample_size, replace=False)
            order_real = [order_real_full[i] for i in indices]
            print(f"[load_envs_custom] 真实订单采样: {len(order_real_full)} -> {len(order_real)} ({real_order_sample_rate*100:.0f}%)")
        else:
            order_real = order_real_full
            print(f"[load_envs_custom] 使用全部真实订单: {len(order_real)} 条")
        
        if len(order_real) > 0:
            prices = [o[4] for o in order_real]
            print(f"  真实价格统计: mean={np.mean(prices):.2f}, std={np.std(prices):.2f}")
    else:
        order_real = []
        if use_real_orders:
            print(f"[load_envs_custom] 警告: 数据文件中没有 real_orders，将使用采样价格")

    grid_num = int(order_param.shape[0])
    if shape is None:
        M, N = grid_num, 1
    else:
        M, N = shape
        if M * N != grid_num:
            M, N = grid_num, 1

    l_max = int(np.max(neighbor[neighbor < 100])) if np.any(neighbor < 100) else 1
    l_max = max(1, min(l_max, 8))

    order_param = order_param.astype(np.int32)
    total_orders = int(order_param.sum())
    print(f"[load_envs_custom] order_param total_orders: {total_orders}")

    order_num = np.sum(order_param, axis=1)
    order_num_dict = []
    for t in range(order_param.shape[2]):
        order_num_dict.append({i: [order_num[i, t]] for i in range(grid_num)})

    origin_volume = order_param.sum(axis=(1, 2)).astype(np.float64)
    if origin_volume.sum() > 0:
        driver_param = origin_volume / origin_volume.sum() * driver_num
    else:
        driver_param = np.ones(grid_num, dtype=np.float64) * driver_num / grid_num
    driver_param = np.floor(driver_param).astype(np.int32)
    remainder = driver_num - int(driver_param.sum())
    if remainder > 0:
        add_idx = np.random.choice(grid_num, remainder, replace=True)
        for idx in add_idx:
            driver_param[idx] += 1

    idle_driver_location_mat = np.zeros((order_param.shape[2], grid_num))
    for t in range(order_param.shape[2]):
        idle_driver_location_mat[t] = driver_param

    order_time = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.04, 0.01]
    n_side = 6
    onoff_driver_location_mat = []
    mapped_matrix_int = np.arange(M * N).reshape(M, N)
    
    # 设置采样概率（如果使用真实订单，概率为 1 表示使用全部订单）
    probability = 1.0 if use_real_orders and len(order_real) > 0 else 1.0 / 28
    prob_env = os.environ.get("COOPRIDE_ORDER_PROB")
    if prob_env:
        try:
            probability = float(prob_env)
            print(f"[load_envs_custom] override probability={probability}")
        except ValueError:
            print(f"[load_envs_custom] invalid COOPRIDE_ORDER_PROB={prob_env}, keep {probability}")
    
    env = CityReal(
        mapped_matrix_int,
        order_num_dict,
        [],
        idle_driver_location_mat,
        order_time,
        price_param,
        l_max,
        M,
        N,
        n_side,
        order_param.shape[2],
        probability,
        np.array(order_real),
        np.array(onoff_driver_location_mat),
        neighbor_dis=neighbor,
        order_param=order_param,
        fleet_help=False,
    )
    
    # 如果使用真实订单，预先生成 day_orders
    if use_real_orders and len(order_real) > 0:
        env.utility_bootstrap_oneday_order()
        print(f"[load_envs_custom] 已预生成 day_orders")
    
    return env, M, N, None, grid_num



if __name__ == '__main__':     

    load_envs_DiDi121(2000)
    #load_envs_NYU143(2000)
