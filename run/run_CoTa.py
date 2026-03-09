import os
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
from copyreg import pickle
import os.path as osp
import sys
sys.path.append('../')
from simulator.envs import *
from tools.create_envs import *
from tools.load_data import *
from algo.MAPPO import *
from tools import logfile
import torch
import pickle
import time
import setproctitle
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../coopride_llm')
from log import Logger

os.environ["CUDA_VISIBLE_DEVICES"]='0'
setproctitle.setproctitle("didi@wjw")


def count_action_orders_and_drivers(orders, env):
    """
    统计 action 中各类订单的数量（各个节点之和）和空闲司机数量
    
    Args:
        orders: 环境返回的订单列表，每个元素是一个订单列表（对应一个节点）
        env: 环境对象，包含所有节点信息
    
    Returns:
        order_stats: 订单统计字典
            - 'real_actions': 真实订单数量 (service_type=-1)
            - 'fake_actions': 虚假订单数量 (service_type=0)
            - 'fleet_actions': 车队订单数量 (service_type>0)
            - 'total_actions': 总订单数量
        total_idle_drivers: 所有节点的空闲司机总数
    """
    # 初始化订单统计
    action_stats = {
        'real_actions': 0,    # 真实订单
        'fake_actions': 0,    # 虚假订单
        'fleet_actions': 0,    # 车队订单
        'total_actions': 0     # 总订单数
    }
    
    # 统计 action 中各类订单的数量
    for node_orders in orders:
        for order in node_orders:
            service_type = order.get_service_type()
            if service_type == -1:
                # 真实订单
                action_stats['real_actions'] += 1
            elif service_type == 0:
                # 虚假订单
                action_stats['fake_actions'] += 1
            elif service_type > 0:
                # 车队订单
                action_stats['fleet_actions'] += 1
            action_stats['total_actions'] += 1
    
    # 统计所有节点的空闲司机数量
    total_idle_drivers = sum(node.idle_driver_num for node in env.nodes if node is not None)
    
    return action_stats, total_idle_drivers


def get_parameter():
    """
    参数配置函数：定义和管理所有训练/测试的参数
    
    Returns:
        args: 包含所有配置参数的对象
    
    功能说明：
        1. 使用 argparse 解析命令行参数
        2. 设置训练和测试的各种超参数
        3. 配置环境、算法、网络架构等参数
        4. 自动生成日志目录和日志名称
        5. 保存配置到 setting.txt 文件
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    args= parser.parse_args()
    # ========== ！！！！！ProPS参数！！！！！ ==========

    # 匹配模式：
    # 'local': 本地匹配
    # 'RLmerge': RL 合并匹配（车队可以调度）
    # 'RLsplit': RL 分离匹配
    args.FM_mode = ['local','RLmerge','RLsplit' ][0]

    # ========== 基础训练参数 ==========
    # 最大迭代次数
    args.MAX_ITER=6000

    # ========== 测试相关参数 ==========
    # 测试基准目录路径（公共前缀）
    # test_base_dir = '../logs/synthetic/grid143/EnvStat326_OD143_FMRLmerge_Batch1000_Gamma0.97_Lambda0.95_Iter1_Ir0.001_Step144_Ent0.005_Minibatch5_Parallel5mix_MDP0_StateEmb2_Meta0global_DGCNAC_relufeaNor1_20260103_02-57'
    # test_base_dir = '../coopride_llm/ckpts/test143'
    test_base_dir = '../coopride_llm/ckpts/test121'
    
    # 测试日志目录（仅用于查看）
    args.test_dir = test_base_dir
    
    # 模型权重文件路径（用于加载测试模型）
    args.model_dir = test_base_dir + '/Best.pkl'
    
    # 是否为测试模式（True=测试，False=训练）
    # args.test = False
    args.test = True
    
    # 测试迭代次数
    args.TEST_ITER=50
    
    # 测试随机种子（保证测试可重复）
    args.TEST_SEED = 1314520

    # 恢复训练的起始迭代（用于断点续训）
    args.resume_iter=0
    
    # 计算设备：'gpu' 或 'cpu'
    args.device='gpu'
    # 是否考虑相邻网格接单
    args.neighbor_dispatch=False
    
    # 是否考虑车辆的随机下线
    args.onoff_driver=False
    
    # 日志名称（注释掉的是历史日志名称）
    #args.log_name='M2_a0.01_reward2_t2_gamma0_value_noprice_noentropy'
    #args.log_name='advnormal_gradMean_iter10_lr3e-4_step144_clipno_batchall3_parallel1_minibatch1'
    args.log_name='debug'

    # ========== 环境相关参数 ==========
    # 决策间隔（分钟）：每隔多少分钟做一次调度决策
    args.dispatch_interval= 10
    
    # 车辆速度（km/min），等于决策间隔
    args.speed=args.dispatch_interval
    
    # 等待时间（分钟），等于决策间隔
    args.wait_time=args.dispatch_interval
    
    # 一天的总决策次数（1440分钟 / 决策间隔）
    args.TIME_LEN = int(1440//args.dispatch_interval)
    
    # 网格数量（决定了使用哪个数据集）
    # 可选值：36, 100, 121, 143
    args.grid_num= 121
    # 司机数量字典：不同网格规模对应不同的司机数量
    driver_dict = {
        143:2000,  # NYU143 数据集
        121:1500,  # DiDi121 数据集
        100:1000   # 合成数据
    }
    args.driver_num=driver_dict[args.grid_num]
    
    # 城市开始时间（0 表示从一天开始）
    args.city_time_start=0
    
    # 是否使用动态环境（False=静态环境，True=动态环境）
    args.dynamic_env = False
    
    # 环境随机种子字典：不同网格规模使用不同的固定种子
    seed_dict = {
        143:326,
        121:6,
        100:16
    }
    args.env_seed = seed_dict[args.grid_num]

    # ========== RL 算法参数 ==========
    # 批次大小
    args.batch_size=int(1000)
    
    # Actor 网络学习率
    args.actor_lr=1e-3
    
    # Critic 网络学习率
    args.critic_lr=1e-3
    
    # Meta 网络学习率
    args.meta_lr = 1e-3
    
    # Actor 网络每次更新训练迭代次数
    args.train_actor_iters=1
    
    # Critic 网络每次更新训练迭代次数
    args.train_critic_iters=1
    
    # Meta 网络每次更新训练迭代次数
    args.train_phi_iters=1
    
    # 确保批次大小为整数
    args.batch_size= int(args.batch_size)
    
    # 折扣因子（gamma）：对未来奖励的折扣程度
    args.gamma=0.97
    
    # GAE lambda：控制优势估计的偏差-方差权衡
    args.lam=0.95
    
    # 梯度裁剪的最大范数（防止梯度爆炸）
    args.max_grad_norm = 10
    
    # PPO 裁剪比率（控制策略更新幅度）
    args.clip_ratio=0.2
    
    # 熵系数（鼓励探索）
    args.ent_factor=0.005
    
    # 是否对优势进行归一化
    args.adv_normal=True
    
    # 是否使用 PPO 裁剪
    args.clip=True
    
    # 每个 epoch 的步数（一天的时间步数）
    args.steps_per_epoch=144
    
    # 梯度聚合方式：'mean' 或 'sum'
    args.grad_multi='mean'
    
    # 小批次数量（每次更新时将数据分成多少个小批次）
    #args.minibatch_num= int(round(args.steps_per_epoch*args.grid_num/args.batch_size))
    args.minibatch_num=5
    
    # 并行回合数（并行训练的回合数）
    args.parallel_episode=5
    
    # 并行方式：'mix'（混合）或 'mean'（平均）
    args.parallel_way='mix'
    
    # 是否使用并行队列（True=先进先出，False=按顺序）
    args.parallel_queue=True
    
    # 是否使用返回值缩放
    args.return_scale=False
    # ========== 网络架构参数 ==========
    # 是否使用正交初始化（提高训练稳定性）
    args.use_orthogonal=True
    
    # 是否使用值裁剪（防止值函数过大）
    args.use_value_clip=True
    
    # 是否使用值归一化（RunningNorm）
    args.use_valuenorm=True
    
    # 是否使用 Huber 损失（对异常值不敏感）
    args.use_huberloss=False
    
    # 是否使用学习率退火
    args.use_lr_anneal=False
    
    # 是否使用 GAE（广义优势估计）
    args.use_GAEreturn=True
    
    # 是否使用 RNN（GRU）
    args.use_rnn=False
    
    # 是否使用 GAT（图注意力网络）
    args.use_GAT=False
    
    # 是否使用 GCN（图卷积网络）
    args.use_GCN=False
    
    # 是否使用 DGCN（深度图卷积网络）
    args.use_DGCN = True
    
    # 是否使用 Dropout
    args.use_dropout=False
    
    # 是否使用全局嵌入
    args.global_emb = False
    
    # ========== 辅助损失参数 ==========
    # 是否使用辅助损失
    args.use_auxi = False
    
    # 辅助损失类型：'huber', 'mse', 'cos'
    args.auxi_loss = ['huber','mse','cos'] [1]
    
    # 辅助损失权重
    args.auxi_effi=0.01
    
    # 虚假订单辅助损失
    args.use_fake_auxi=0
    
    # 正则化类型：'None', 'L1', 'L2', 'L1state', 'L2state'
    args.use_regularize = ['None','L1','L2', 'L1state', 'L2state'] [0]
    
    # 正则化系数
    args.regularize_alpha = 1e-1

    # 激活函数：'relu', 'tanh', 'sigmoid' 等
    args.activate_fun='relu'
    
    # ========== 邻域和中心化参数 ==========
    # 是否使用邻居状态信息
    args.use_neighbor_state = False
    
    # 邻接阶数（使用多少阶邻居）
    args.adj_rank = 3
    
    # 特征合并方式：'cat'（拼接）或 'res'（残差）
    args.merge_method = 'cat'
    
    # Actor 是否中心化（集中式训练）
    args.actor_centralize = True
    
    # Critic 是否中心化（集中式训练）
    args.critic_centralize = True

    # ========== 奖励和匹配参数 ==========
    # 奖励缩放因子
    args.reward_scale=5
    
    # 经验回放大小
    args.memory_size = int(args.TIME_LEN*args.parallel_episode)
    
    # 匹配模式：
    # 'local': 本地匹配
    # 'RLmerge': RL 合并匹配（车队可以调度）
    # 'RLsplit': RL 分离匹配
    # args.FM_mode = ['local','RLmerge','RLsplit' ][0]
    
    # 是否移除虚假订单
    args.remove_fake_order=False
    
    # 是否使用 ORR（订单响应率）奖励
    args.ORR_reward=False
    
    # ORR 奖励效率系数
    args.ORR_reward_effi=1
    
    # 是否只使用 ORR 奖励
    args.only_ORR=False

    # ========== 特征相关参数 ==========
    # 特征归一化方式：
    # 1: 归一化到 [0,1]
    # 2: 标准化
    # 3: 加载历史归一化参数
    args.feature_normal = 3
    
    # 是否使用状态差值作为状态补充
    args.use_state_diff = False
    
    # 是否使用订单时间特征
    args.use_order_time = False
    
    # 是否使用新订单熵
    args.new_order_entropy=True
    
    # 是否使用订单网格
    args.order_grid=True
    
    # 是否使用 MDP（马尔可夫决策过程）：
    # 0: 不使用
    # 1: 表格 MDP
    # 2: Deep MDP
    args.use_mdp = 0
    
    # 是否更新值函数
    args.update_value=True
    
    # 在特征中去除的项：['fea', 'time', 'id']
    args.rm_state = []
    
    # 在网络中去除的状态：AC0123
    args.state_remove = ''

    # ========== 状态表征参数 ==========
    # 状态嵌入选择：
    # 0: 无嵌入
    # 1: 简单嵌入
    # 2: 复杂嵌入
    args.state_emb_choose = 2

    # ========== 日志控制参数 ==========
    # 是否记录特征
    args.log_feature = True
    
    # 是否记录分布
    args.log_distribution = False
    
    # 是否记录 phi（meta 参数）
    args.log_phi = False

    # ========== 合作机制参数（Meta-learning） ==========
    '''
    meta_choose 策略说明：
        0: 无 meta（此时看 team_rank 和 global_share）
        1, 2, 3: 圆环加权（不同归一化方法）
        4, 5: 圆饼加权
            - 4: 0~K 阶（包含当前节点）
            - 5: 1~K 阶（不包含当前节点）
        6, 7: 圆环和圆饼结合（低阶圆饼，高阶圆环）
            - 6: 前 1 阶圆饼，其余圆环
            - 7: 前 2 阶圆饼，其余圆环
    '''
    # Meta 策略选择
    args.meta_choose =0
    
    # Meta 作用范围（最多考虑多少阶邻居）
    args.meta_scope = 4
    
    # 团队排名（用于计算 ORR 奖励时考虑的邻域范围）
    args.team_rank=0
    
    # 是否全局共享（True=所有智能体共享，False=不共享）
    args.global_share=False

    # ========== 日志名称生成 ==========
    # 构建日志名称字典（用于生成唯一的日志名称）
    log_name_dict={
        'OD': args.grid_num,              # 网格数量
        'FM': args.FM_mode,               # 匹配模式
        'Batch': args.batch_size,         # 批次大小
        'Gamma':args.gamma,                # 折扣因子
        'Lambda':args.lam,                # GAE lambda
        'Iter': args.train_actor_iters,   # 训练迭代次数
        'Ir': args.actor_lr,              # Actor 学习率
        'Step': args.steps_per_epoch,     # 每个 epoch 的步数
        'Ent': args.ent_factor,           # 熵系数
        'Minibatch': args.minibatch_num,   # 小批次数量
        'Parallel': str(args.parallel_episode)+args.parallel_way,  # 并行设置
        'MDP': str(args.use_mdp),         # MDP 设置
        'StateEmb': str(args.state_emb_choose),  # 状态嵌入
    }
    
    # 开始构建日志名称
    args.log_name=''
    
    # 根据环境类型添加前缀
    if args.dynamic_env:
        args.log_name+= 'EnvDyna_'      # 动态环境
    else:
        args.log_name+= 'EnvStat{}_'.format(args.env_seed)  # 静态环境（带种子）
    
    # 添加所有基础参数
    for k,v in log_name_dict.items():
        args.log_name+= k+str(v)+'_'
    
    # 添加 Meta 参数
    args.log_name+='Meta'+str(args.meta_choose)
    if args.meta_choose==0:
        # 无 meta 的情况
        if args.global_share:
            args.log_name+='global'
        else:
            args.log_name+=str(args.team_rank)
    else:
        # 有 meta 的情况
        args.log_name+=str(args.meta_scope)
        args.log_name+= '_LR'+str(args.meta_lr)
    # 添加可选参数到日志名称
    #args.log_name+='seed0'
    #args.log_name+='_car50'
    
    # 是否移除网格
    if args.order_grid==False:
        args.log_name+='_RmGrid'
    
    # 是否只使用 ORR
    if args.only_ORR:
        args.log_name+='_onlyORR'
    
    # 是否更新值函数
    if args.update_value:
        pass  # 已默认包含，注释掉
        #args.log_name+='_UpVal'
    
    # 是否移除状态
    if args.state_remove != '':
        args.log_name += '_StateRm'+args.state_remove
    
    # 是否使用新订单熵
    #args.log_name+='_KLNEW'
    if args.new_order_entropy:
        pass  # 已默认包含，注释掉
        #args.log_name+='_NewEntropy'
    
    # 是否使用状态差值
    if args.use_state_diff :
        args.log_name+='_StateDiff'
    
    # 网络架构相关
    if args.use_orthogonal==True:
        pass  # 已默认包含，注释掉
        #args.log_name+='_OrthoInit'
    if args.use_value_clip:
        pass  # 已默认包含，注释掉
        #args.log_name+='_ValueClip'
    if args.use_valuenorm:
        pass  # 已默认包含，注释掉
        #args.log_name+='_ValueNorm'
    if args.use_huberloss:
        pass  # 已默认包含，注释掉
        #args.log_name+='_Huberloss'
    if args.use_lr_anneal:
        args.log_name+='_LRAnneal'
    if args.use_GAEreturn:
        pass  # 已默认包含，注释掉
        #args.log_name+='_GAEreturn'
    
    # RNN 和图神经网络
    if args.use_rnn:
        args.log_name+='_GRU2'
    if args.use_GAT:
        args.log_name+='_GATnew'
    if args.use_GCN:
        args.log_name+='_GCN'+str(args.adj_rank)
    if args.use_DGCN:
        args.log_name+='_DGCN'
        if args.actor_centralize:
            args.log_name+='A'  # Actor 中心化
        if args.critic_centralize:
            args.log_name+='C'  # Critic 中心化
    
    # 邻居状态
    if args.use_neighbor_state:
        args.log_name+='_Statenew'+str(args.adj_rank)
    
    # 正则化和辅助损失
    if args.use_regularize != 'None':
        args.log_name+='_'+args.use_regularize+str(args.regularize_alpha)
    if args.use_auxi:
        args.log_name+= 'Auxi'+args.auxi_loss+str(args.auxi_effi)
    if args.use_fake_auxi >0:
        args.log_name+= 'FakeNewAuxi'+str(args.use_fake_auxi)
    
    # 激活函数和特征归一化
    args.log_name+= '_'+args.activate_fun
    args.log_name+= 'feaNor'+str(args.feature_normal)
    
    # 移除的特征
    if len(args.rm_state)>0:
        args.log_name+='_RM'
        for s in args.rm_state:
            args.log_name+=s
        

    # ========== 日志目录创建 ==========
    # 添加合并方法（已注释）
    #args.log_name+= '_'+args.merge_method
    #args.log_name+='_GAE'

    # 历史日志名称（已注释）
    #args.log_name='advnormal_gradMean_iter10_lr3e-4_step144_clipno_batchall3_parallel1_minibatch1'
    #args.log_name='debug'

    # 获取当前时间
    current_time = time.strftime("%Y%m%d_%H-%M")
    
    # 构建日志目录路径
    #log_dir = '../logs/' + "{}".format(current_time)  # 使用时间戳
    log_dir = '../logs/' + 'synthetic/'+'grid'+str(args.grid_num)+'/'+args.log_name  # 使用参数名称
    
    # 如果目录已存在，添加时间戳后缀
    if os.path.exists(log_dir):
        log_dir+='_'+current_time
    
    # 保存日志目录路径
    args.log_dir=log_dir
    
    # 创建日志目录（包括所有父目录）
    mkdir_p(log_dir)
    print ("log dir is {}".format(log_dir))
    
    # ========== 保存配置参数 ==========
    args.writer_logs=False
    if args.writer_logs:
        # 将所有参数保存到 setting.txt 文件
        args_dict=args.__dict__
        with open(log_dir+'/setting.txt','w') as f:
            for key, value in args_dict.items():
                f.writelines(key+' : '+str(value)+'\n')

    return args


def train(env, agent , writer=None,args=None,device='cpu'):

    best_gmv=0
    best_orr=0
    if args.return_scale:
        record_return=test(env,agent, test_iter=1,args=args,device=device)/20
        record_return[record_return==0]=1
    for iteration in np.arange(args.resume_iter,args.MAX_ITER):
        t_begin=time.time()
        print('\n---- ROUND: #{} ----'.format(iteration))
        RANDOM_SEED = iteration*1000
        if args.dynamic_env:
            env.reset_randomseed(RANDOM_SEED)
        else:
            env.reset_randomseed(args.env_seed*1000)

        gmv = []
        fake_orr = []
        fleet_orr = []
        kl = []
        entropy = []
        order_response_rates = []
        T = 0

        states_node, _, order_states, order_idx, order_feature, global_order_states = env.reset(mode='PPO2')
        state=agent.process_state(states_node,T)  # state dim= (grid_num, 119)
        state_rnn_actor = torch.zeros((1,agent.agent_num,agent.hidden_dim),dtype=torch.float)
        state_rnn_critic = torch.zeros((1,agent.agent_num,agent.hidden_dim),dtype=torch.float)
        order,mask_order=agent.process_order(order_states,T)
        order=agent.remove_order_grid(order)
        mask_order= agent.mask_fake(order, mask_order)


        for T in np.arange(args.TIME_LEN):
            assert len(order_idx)==args.grid_num ,'dim error'
            assert len(order_states)==args.grid_num ,'dim error'
            for i in range(len(order_idx)):
                assert len(order_idx[i])==len(order_states[i]), 'dim error'

            #t0=time.time()
            action, local_value, global_value ,logp, mask_agent, mask_order_multi, mask_action ,  mask_entropy ,next_state_rnn_actor, next_state_rnn_critic ,action_ids, selected_ids = agent.action(state,order,state_rnn_actor, state_rnn_critic,mask_order,order_idx ,device,sample=True,random_action=False, FM_mode = args.FM_mode)
            
            if args.use_mdp>0 and args.update_value:
                agent.MDP.push(order_states,selected_ids)
                #MDP.update_value(order_states,selected_ids,env)

            #t1=time.time()
            orders = env.get_orders_by_id(action_ids)

            next_states_node,  next_order_states, next_order_idx, next_order_feature= env.step(orders, generate_order=1, mode='PPO2')

            #t2=time.time()

            # distribution should gotten after step
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            entropy.append(entr_value)
            kl.append(kl_value)
            gmv.append(env.gmv)
            fake_orr.append(env.fake_response_rate)
            fleet_orr.append(env.fleet_response_rate)
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)

            # store transition
            if T==args.TIME_LEN-1:
                done=True
            else:
                done=False

            reward = torch.Tensor([ node.gmv for node in env.nodes] ) 
            if args.log_distribution:
                driver_num= torch.Tensor([node.idle_driver_num for node in env.nodes])
                order_num= torch.Tensor([node.real_order_num for node in env.nodes])
                agent.logs.push_log_distribution(T,reward, driver_num,order_num)

            if args.ORR_reward==True:
                ORR_reward=torch.zeros_like(reward)
                driver_num= torch.Tensor([node.idle_driver_num for node in env.nodes])+1e-5
                order_num= torch.Tensor([node.real_order_num for node in env.nodes])+1e-5
                driver_order=torch.stack([driver_num,order_num],dim=1)
                ORR_entropy= torch.min(driver_order,dim=1)[0]/torch.max(driver_order,dim=1)[0]
                '''
                ORR_entropy= ORR_entropy*torch.log(ORR_entropy)

                global_entropy= torch.min(torch.sum(driver_order,dim=0))/torch.max(torch.sum(driver_order,dim=0))
                global_entropy = global_entropy*torch.log(global_entropy)
                ORR_entropy= torch.abs(ORR_entropy-global_entropy)
                order_num/=torch.sum(order_num)
                driver_num/=torch.sum(driver_num)
                ORR_KL = torch.sum(order_num * torch.log(order_num / driver_num))
                '''
                for i in range(args.grid_num):
                    num=1
                    ORR_reward[i]=ORR_entropy[i]
                    for rank in range(args.team_rank):
                        neighb= env.nodes[i].layers_neighbors_id[rank]
                        num+=len(neighb)
                        ORR_reward[i]+= torch.sum(ORR_entropy[neighb])
                    ORR_reward[i]/=num
                #ORR_reward= -ORR_reward*10-ORR_KL+2.5
                reward+= ORR_reward*args.ORR_reward_effi
                if args.only_ORR:
                    reward= ORR_reward*args.ORR_reward_effi

            #print(0)
            if args.return_scale:
                reward/=record_return
            else:
                reward/=args.reward_scale

            next_state=agent.process_state(next_states_node,T+1)  # state dim= (grid_num, 119)
            next_order,next_order_mask=agent.process_order(next_order_states, T+1)
            next_order=agent.remove_order_grid(next_order)
            next_order_mask= agent.mask_fake(next_order, next_order_mask)

            agent.buffer.push(state, next_state ,order, action, reward[:,None], local_value, global_value ,logp , mask_order_multi, mask_action, mask_agent, mask_entropy ,state_rnn_actor.squeeze(0), state_rnn_critic.squeeze(0))

            epoch_ended = (T%args.steps_per_epoch)== (args.steps_per_epoch-1)
            done = T==args.TIME_LEN-1
            if done or epoch_ended:
                if done:
                    next_local_value = torch.zeros((agent.agent_num,agent.meta_scope+1))
                    next_global_value = torch.zeros((1,1))
                elif epoch_ended:
                    with torch.no_grad():
                        next_local_value,_ = agent.critic(next_state.to(device), agent.adj , next_state_rnn_critic.to(device))
                        next_global_value, _ = agent.critic.get_global_value(next_state.to(device), agent.adj , next_state_rnn_critic.to(device))
                    next_local_value = next_local_value.detach().cpu()
                    next_global_value = next_global_value.detach().cpu()
                agent.buffer.finish_path_local(next_local_value)
                if args.meta_choose >0:
                    agent.buffer.path_start_idx = agent.buffer.record_start_idx
                    agent.buffer.finish_path_global(next_global_value)
                #agent.update(device,writer)

            #t3=time.time()
            #print(t1-t0,t2-t0,t3-t0)

            states_node=next_states_node
            order_idx=next_order_idx
            order_states=next_order_states
            order_feature=next_order_feature
            state=next_state
            order=next_order
            mask_order=next_order_mask
            state_rnn_actor = next_state_rnn_actor
            state_rnn_critic = next_state_rnn_critic
            T += 1


        if args.parallel_queue==False:
            if (iteration+1)%args.parallel_episode==0:
                agent.update(device,writer)
        else:
            if (iteration+1)>=args.parallel_episode:
                agent.update(device,writer)
                #agent.buffer

        if args.log_feature:
            agent.logs.save_feature()
        if args.log_distribution and iteration%50==0:
            agent.logs.save_log_distribution(name=iteration,dir='distribution')
        if args.log_phi and iteration%50==0:
            agent.logs.save_full_phi(name=iteration,dir='phi')

        if args.use_mdp==2:
            writer.add_scalar('train mdp value',agent.MDP.update(device),iteration) 

        t_end=time.time()

        if np.sum(gmv)> best_gmv:
            best_gmv=np.sum(gmv)
            best_orr=order_response_rates[-1]
            agent.save_param(args.log_dir,'Best')
        print('>>> Time: [{0:<.4f}] Mean_ORR: [{1:<.4f}] GMV: [{2:<.4f}] Best_ORR: [{3:<.4f}] Best_GMV: [{4:<.4f}]'.format(
            t_end-t_begin,order_response_rates[-1], np.sum(gmv),best_orr,best_gmv ))
        agent.save_param(args.log_dir,'param')
        if args.use_mdp>0:
            agent.MDP.save_param(args.log_dir)
        writer.add_scalar('train ORR',order_response_rates[-1],iteration)
        writer.add_scalar('train GMV',np.sum(gmv),iteration)
        #writer.add_scalar('train KL',np.mean(kl),iteration)
        #writer.add_scalar('train Suply/demand',np.mean(entropy),iteration)


def log_test_info(args):
    """
    打印测试执行的关键信息
    
    Args:
        args: 配置参数对象
    
    功能说明：
        调用 log 模块记录测试的关键配置信息，包括执行模式、网格参数、
        迭代次数、是否使用 meta、是否使用 MDP 等
    """
    Logger.info("=" * 60)  # 分隔线
    Logger.info("TEST EXECUTION INFO")  # 测试信息标题
    Logger.info("=" * 60)  # 分隔线
    
    Logger.info("-" * 50)  # 分隔线
    Logger.info(f"Test Iterations: {args.TEST_ITER}")  # 测试迭代次数，决定运行多少个测试回合
    Logger.info(f"FM Mode: {args.FM_mode}")  # 是否开启重定位
    Logger.info("-" * 50)  # 分隔线
    
    # 执行模式
    mode = "TEST" if args.test else "TRAIN"
    Logger.info(f"Execution Mode: {mode}")  # 当前执行模式：TEST为测试模式，TRAIN为训练模式
    
    # 网格参数
    Logger.info(f"Grid Number: {args.grid_num}")  # 网格数量，决定了使用哪个数据集，可选值：36, 100, 121, 143
    Logger.info(f"Driver Number: {args.driver_num}")  # 司机数量，不同网格规模对应不同司机数（143网格对应2000，121对应1500，100对应1000）
    # Logger.info(f"Grid Layout: {args.M} x {args.N}")  # 网格布局的行列尺寸
    Logger.info(f"Environment Type: {'Dynamic' if args.dynamic_env else 'Static'}")  # 环境类型，Static为静态环境（固定场景），Dynamic为动态环境（场景变化）
    if not args.dynamic_env:
        Logger.info(f"Environment Seed: {args.env_seed}")  # 静态环境的随机种子，用于保证场景可重现（不同网格使用不同固定种子）
    
    # 测试相关参数
    Logger.info(f"Test Seed: {args.TEST_SEED}")  # 测试随机种子，用于保证测试结果可重复
    Logger.info(f"Model Path: {args.model_dir}")  # 预训练模型权重文件的路径
    
    # 时间相关参数
    Logger.info(f"Dispatch Interval: {args.dispatch_interval} min")  # 调度决策间隔时间（分钟），每隔多少分钟做一次车辆调度决策
    Logger.info(f"Time Steps per Day: {args.TIME_LEN}")  # 一天的总时间步数（1440分钟除以决策间隔）
    
    # Meta 参数（合作机制参数）
    has_meta = args.meta_choose > 0
    Logger.info(f"Meta Learning: {'Enabled' if has_meta else 'Disabled'}")  # 元学习是否启用，用于实现多智能体合作机制
    if has_meta:
        Logger.info(f"Meta Strategy: {args.meta_choose}")  # 元学习策略类型，0=无meta，1-3=圆环加权，4-5=圆饼加权，6-7=混合策略
        Logger.info(f"Meta Scope: {args.meta_scope}")  # 元学习作用范围，最多考虑多少阶邻居进行合作
    else:
        Logger.info(f"Team Rank: {args.team_rank}")  # 团队排名，用于计算ORR奖励时考虑的邻域范围（当不使用meta时生效）
        Logger.info(f"Global Share: {args.global_share}")  # 是否全局共享参数（当不使用meta时生效）
    
    # MDP 参数（马尔可夫决策过程参数）
    if args.use_mdp == 0:
        Logger.info(f"MDP: Disabled")  # 不使用MDP辅助值函数估计
    elif args.use_mdp == 1:
        Logger.info(f"MDP: Tabular MDP")  # 使用表格型MDP，基于Q表进行值函数估计
    elif args.use_mdp == 2:
        Logger.info(f"MDP: Deep MDP")  # 使用深度MDP，基于神经网络进行值函数估计
    
    # 网络架构参数
    Logger.info(f"State Embedding: {args.state_emb_choose}")  # 状态嵌入类型，0=无嵌入，1=简单嵌入，2=复杂嵌入
    Logger.info(f"Use RNN: {args.use_rnn}")  # 是否使用循环神经网络（GRU）处理序列状态
    Logger.info(f"Use GAT: {args.use_GAT}")  # 是否使用图注意力网络捕获节点间关系
    Logger.info(f"Use GCN: {args.use_GCN}")  # 是否使用图卷积网络进行特征提取
    Logger.info(f"Use DGCN: {args.use_DGCN}")  # 是否使用深度图卷积网络进行多层特征提取
    Logger.info(f"Actor Centralized: {args.actor_centralize}")  # Actor网络是否中心化，集中式训练时使用全局信息
    Logger.info(f"Critic Centralized: {args.critic_centralize}")  # Critic网络是否中心化，集中式训练时使用全局信息
    
    # 匹配模式参数
    Logger.info(f"Fleet Matching Mode: {args.FM_mode}")  # 车队匹配模式，local=本地匹配，RLmerge=RL合并匹配（可车队调度），RLsplit=RL分离匹配
    
    # 计算设备
    Logger.info(f"Device: {args.device.upper()}")  # 计算设备，GPU使用CUDA加速，CPU使用普通计算
    
    # 日志信息
    Logger.info(f"Log Directory: {args.log_dir}")  # 日志文件保存目录
    
    Logger.info("=" * 60)  # 分隔线
    Logger.info("TEST STARTED")  # 测试开始提示
    Logger.info("=" * 60)  # 分隔线
    
    # ========== 关键参数详细说明 ==========
    """
    args.team_rank: 团队排名，用于计算ORR奖励时考虑的邻域范围（当不使用meta时生效）
        生效位置: 在train()和test()函数的ORR_reward计算中，通过env.nodes[i].layers_neighbors_id[rank]获取邻居节点
        不同值影响: 
            - rank=0: 只考虑自身节点的ORR熵，无合作
            - rank=1: 考虑自身和1阶邻居的ORR熵加权平均
            - rank>1: 考虑自身和1~rank阶所有邻居的ORR熵加权平均
        值越大，合作范围越广，能够利用更大区域的信息，但计算开销增加
    
    args.global_share: 是否全局共享参数（当不使用meta时生效）
        生效位置: 在Critic网络初始化和更新时，影响value函数参数是否所有智能体共享
        不同值影响:
            - True: 所有网格共享同一套value网络参数
              优点: 参数量少，泛化能力强，训练稳定
              缺点: 可能无法捕获不同网格的差异化特征
            - False: 每个网格有独立的value参数
              优点: 拟合能力强，可以学习每个网格的独特模式
              缺点: 参数量大，易过拟合，训练不稳定
    
    args.FM_mode: 车队匹配模式，决定车辆如何接受跨网格订单
        生效位置: 在agent.action()函数的action_ids生成和env.step()的订单分配中
        可选值及影响:
            - 'local': 本地匹配
              作用: 每个网格的车辆只能接受本网格的订单
              特点: 无跨网格调度，实现简单，但无法利用邻近网格的车辆/订单资源
            - 'RLmerge': RL合并匹配（车队可以调度）
              作用: 车队可以跨网格调度车辆来服务订单
              特点: 实现了跨网格合作，能动态平衡供需，提高整体ORR，但决策空间更大
            - 'RLsplit': RL分离匹配
              作用: 订单和车辆的匹配在不同阶段分离处理
              特点: 提供另一种合作机制，可能在某些场景下表现更好
    """

def update_ckpt(agent):
    
    # 保存当前模型参数
    pkl_name = 'Best_' + time.strftime("%Y%m%d_%H-%M-%S") + '.pkl'
    agent.save_param(args.test_dir, pkl_name)
    #覆盖旧的Best.pkl, 便于循环训练
    agent.save_param(args.test_dir, 'Best.pkl')


def test(env, agent , writer=None,args=None,device='cpu'):
    """
    测试函数：在测试模式下评估模型性能
    
    Args:
        env: 环境对象
        agent: MAPPO 智能体
        writer: TensorBoard writer (可选)
        args: 配置参数
        device: 计算设备 ('cpu' 或 'cuda')
    
    Returns:
        无直接返回值，但会打印测试结果并保存日志
    
    功能说明：
        1. 使用固定随机种子保证测试可重复性
        2. 不进行模型参数更新
        3. 评估模型在多个回合中的表现
        4. 记录关键指标（GMV, ORR, KL, Entropy等）
    """
    # 设置固定随机种子，确保测试结果可重复
    np.random.seed(args.TEST_SEED)

    # 初始化最佳性能指标
    best_gmv=0
    best_orr=0

    # 初始化所有迭代的 ORR 和 GMV 记录数组
    all_orr_records = []         # 存储每次迭代的 ORR
    all_gmv_records = []         # 存储每次迭代的 GMV

    # 外层循环：测试迭代次数
    for iteration in np.arange(args.TEST_SEED, args.TEST_SEED+args.TEST_ITER):
        # 记录测试开始时间
        t_begin=time.time()
        Logger.info("=" * 50)  # 分隔线，标记新的测试回合开始
        Logger.info(f"TEST ROUND #{iteration}")  # 当前测试回合编号
        Logger.info("=" * 50)  # 分隔线
        print('\n---- ROUND: #{} ----'.format(iteration))

        # 计算随机种子（每次迭代使用不同种子）
        RANDOM_SEED = iteration*1000

        # 根据环境类型重置随机种子
        if args.dynamic_env:
            env.reset_randomseed(RANDOM_SEED)  # 动态环境使用计算出的随机种子
        else:
            env.reset_randomseed(args.env_seed*1000)  # 静态环境使用固定环境种子

        # 初始化指标列表
        gmv = []                    # 总订单价值
        fake_orr = []               # 虚假订单响应率
        fleet_orr = []              # 车队订单响应率
        kl = []                     # KL散度
        entropy = []                # 供给/需求熵
        order_response_rates = []   # 订单响应率
        T = 0                       # 当前时间步

        # 环境重置，获取初始状态
        states_node, _, order_states, order_idx, order_feature, global_order_states = env.reset(mode='PPO2')

        # 处理初始状态，state维度为 (grid_num, 119)
        state=agent.process_state(states_node,T)

        # 初始化 RNN 隐藏状态（用于 Actor）
        state_rnn_actor = torch.zeros((1,agent.agent_num,agent.hidden_dim),dtype=torch.float)

        # 初始化 RNN 隐藏状态（用于 Critic）
        state_rnn_critic = torch.zeros((1,agent.agent_num,agent.hidden_dim),dtype=torch.float)

        # 处理订单状态
        order,mask_order=agent.process_order(order_states,T)

        # 移除订单网格信息
        order=agent.remove_order_grid(order)

        # 屏蔽虚假订单
        mask_order= agent.mask_fake(order, mask_order)

        # 内层循环：一天的时间步
        for T in np.arange(args.TIME_LEN):
            # 断言检查维度一致性
            assert len(order_idx)==args.grid_num ,'dim error'
            assert len(order_states)==args.grid_num ,'dim error'
            for i in range(len(order_idx)):
                assert len(order_idx[i])==len(order_states[i]), 'dim error'

            # t0=time.time()
            # Agent 采样动作
            # 参数说明：
            #   - sample=True: 使用采样而非确定性动作
            #   - random_action=False: 不使用随机动作
            #   - FM_mode: 匹配模式（local/RLmerge/RLsplit）
            (
                action,
                local_value,
                global_value,
                logp,
                mask_agent,
                mask_order_multi,
                mask_action,
                mask_entropy,
                next_state_rnn_actor,
                next_state_rnn_critic,
                action_ids,
                selected_ids,
            ) = agent.action(
                state,
                order,
                state_rnn_actor,
                state_rnn_critic,
                mask_order,
                order_idx,
                device,
                sample=True,
                random_action=False,
                FM_mode=args.FM_mode,
            )

            # t1=time.time()
            # 根据动作 ID 获取订单
            orders = env.get_orders_by_id(action_ids)

            # 统计 action 中各类订单(不代表最终执行)的数量和空闲司机数量
            action_stats, total_idle_drivers = count_action_orders_and_drivers(orders, env)

            # 执行环境步，生成订单并更新环境状态
            next_states_node,  next_order_states, next_order_idx, next_order_feature= env.step(orders, generate_order=1, mode='PPO2')

            # t2=time.time()

            # 记录当前时间步的关键信息
            Logger.info_daily(f"TimeStep {T:03d}----------------")  # 当前时间步编号（000-143），格式化为3位数

            order_stats = {
                'real_orders': int(sum(orders.real_order_num for orders in env.nodes)),      # 真实订单
                'fake_orders': int(sum(orders.fake_order_num for orders in env.nodes)),      # 虚假订单
                'fleet_orders': int(sum(orders.fleet_order_num for orders in env.nodes)),    # 车队订单
                'total_orders': int(sum(len(orders.orders) for orders in env.nodes))         # 总订单数
            }            

            Logger.info_daily(f"  Total Idle Drivers: {total_idle_drivers}")  # 总空闲司机数，表示当前未接单的司机数量
            Logger.info_daily(f"  Order  Stats: {order_stats}")  # action订单统计信息，包含各类订单的数量
            Logger.info_daily(f"  Action Stats: {action_stats}")  # action订单统计信息，包含各类订单的数量

            Logger.info_daily(f"  Current GMV: {env.gmv:.2f}")  # 当前时间步的总GMV（美元），衡量系统当前时刻创造的经济价值
            Logger.info_daily(f"  Current ORR: {env.order_response_rate:.4f}")  # 当前时间步的订单响应率，衡量当前时刻订单满足比例
            # Logger.info_daily(f"  Fake ORR: {env.fake_response_rate:.4f}")  # 当前时间步的虚假订单响应率，衡量系统对虚拟订单的响应能力
            # Logger.info_daily(f"  Fleet ORR: {env.fleet_response_rate:.4f}")  # 当前时间步的车队订单响应率，衡量车队整体服务能力

            # 记录指标
            gmv.append(env.gmv)                          # 记录总订单价值，所有网格当前时间步的GMV之和
            fake_orr.append(env.fake_response_rate)      # 记录虚假订单响应率，系统对虚拟订单的响应比例
            fleet_orr.append(env.fleet_response_rate)    # 记录车队订单响应率，车队整体服务订单的比例
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)  # 记录订单响应率，已服务订单数除以总订单数

            # 判断回合是否结束
            if T==args.TIME_LEN-1:
                done=True
            else:
                done=False

            # 计算基础奖励（基于 GMV）
            reward = torch.Tensor([ node.gmv for node in env.nodes] ) 

            # 获取空闲司机数和真实订单数
            driver_num= torch.Tensor([node.idle_driver_num for node in env.nodes])
            order_num= torch.Tensor([node.real_order_num for node in env.nodes])

            # 记录分布日志
            agent.logs.push_log_distribution(T,reward, driver_num,order_num)

            # 如果启用 ORR（订单响应率）奖励
            if args.ORR_reward==True:
                ORR_reward=torch.zeros_like(reward)
                driver_num= torch.Tensor([node.idle_driver_num for node in env.nodes])+1e-5
                order_num= torch.Tensor([node.real_order_num for node in env.nodes])+1e-5
                driver_order=torch.stack([driver_num,order_num],dim=1)

                # 计算 ORR 熵：衡量供需平衡程度
                # 使用 min(driver, order) / max(driver, order) 作为熵指标
                ORR_entropy= torch.min(driver_order,dim=1)[0]/torch.max(driver_order,dim=1)[0]

                '''
                ORR_entropy= ORR_entropy*torch.log(ORR_entropy)

                global_entropy= torch.min(torch.sum(driver_order,dim=0))/torch.max(torch.sum(driver_order,dim=0))
                global_entropy = global_entropy*torch.log(global_entropy)
                ORR_entropy= torch.abs(ORR_entropy-global_entropy)
                order_num/=torch.sum(order_num)
                driver_num/=torch.sum(driver_num)
                ORR_KL = torch.sum(order_num * torch.log(order_num / driver_num))
                '''

                # 考虑邻域的 ORR（如果设置了 team_rank）
                for i in range(args.grid_num):
                    num=1
                    ORR_reward[i]=ORR_entropy[i]
                    for rank in range(args.team_rank):
                        neighb= env.nodes[i].layers_neighbors_id[rank]
                        num+=len(neighb)
                        ORR_reward[i]+= torch.sum(ORR_entropy[neighb])
                    ORR_reward[i]/=num
                # ORR_reward= -ORR_reward*10-ORR_KL+2.5

                # 将 ORR 奖励加到基础奖励上
                reward+= ORR_reward*args.ORR_reward_effi

                # 如果只使用 ORR 奖励
                if args.only_ORR:
                    reward= ORR_reward*args.ORR_reward_effi

            # print(0)
            # 奖励归一化/缩放
            if args.return_scale:
                reward/=record_return
            else:
                reward/=args.reward_scale

            # 处理下一状态
            next_state=agent.process_state(next_states_node,T+1)  # state dim= (grid_num, 119)

            # 处理下一订单状态
            next_order,next_order_mask=agent.process_order(next_order_states, T+1)
            next_order=agent.remove_order_grid(next_order)
            next_order_mask= agent.mask_fake(next_order, next_order_mask)

            '''
            # 注释掉的代码：测试模式不存储转换到 buffer
            agent.buffer.push(state, next_state ,order, action, reward[:,None], local_value, global_value ,logp , mask_order_multi, mask_action, mask_agent, mask_entropy ,state_rnn_actor.squeeze(0), state_rnn_critic.squeeze(0))

            epoch_ended = (T%args.steps_per_epoch)== (args.steps_per_epoch-1)
            done = T==args.TIME_LEN-1
            if done or epoch_ended:
                if done:
                    next_local_value = torch.zeros((agent.agent_num,agent.meta_scope+1))
                    next_global_value = torch.zeros((1,1))
                elif epoch_ended:
                    with torch.no_grad():
                        next_local_value,_ = agent.critic(next_state.to(device), agent.adj , next_state_rnn_critic.to(device))
                        next_global_value, _ = agent.critic.get_global_value(next_state.to(device), agent.adj , next_state_rnn_critic.to(device))
                    next_local_value = next_local_value.detach().cpu()
                    next_global_value = next_global_value.detach().cpu()
                agent.buffer.finish_path_local(next_local_value)
                if args.meta_choose >0:
                    agent.buffer.path_start_idx = agent.buffer.record_start_idx
                    agent.buffer.finish_path_global(next_global_value)
                #agent.update(device,writer)
            '''

            # t3=time.time()
            # print(t1-t0,t2-t0,t3-t0)

            # 更新状态变量为下一状态
            states_node=next_states_node
            order_idx=next_order_idx
            order_states=next_order_states
            order_feature=next_order_feature
            state=next_state
            order=next_order
            mask_order=next_order_mask
            state_rnn_actor = next_state_rnn_actor
            state_rnn_critic = next_state_rnn_critic
            T += 1

        '''
        # 注释掉的代码：测试模式不进行模型更新
        if args.log_feature:
            agent.logs.save_feature()

        if args.parallel_queue==False:
            if (iteration+1)%args.parallel_episode==0:
                agent.update(device,writer)
        else:
            if (iteration+1)>=args.parallel_episode:
                agent.update(device,writer)
                #agent.buffer

        if args.use_mdp==2:
            writer.add_scalar('train mdp value',agent.MDP.update(device),iteration) 
        '''

        # 记录测试结束时间
        t_end=time.time()

        # 更新最佳性能指标
        if np.sum(gmv)> best_gmv:
            best_gmv=np.sum(gmv)
            best_orr=order_response_rates[-1]
        
        # 记录当前迭代的 ORR 和 GMV
        all_orr_records.append(order_response_rates[-1])
        all_gmv_records.append(np.sum(gmv))

        # 打印测试结果
        print('>>> Time: [{0:<.4f}] Mean_ORR: [{1:<.4f}] GMV: [{2:<.4f}] Best_ORR: [{3:<.4f}] Best_GMV: [{4:<.4f}]'.format(
            t_end-t_begin,order_response_rates[-1], np.sum(gmv),best_orr,best_gmv ))

        # 使用 Logger.info 记录当前 iter 的统计信息
        Logger.info(f"Round #{iteration} Summary")  # 当前回合总结
        Logger.info(f"  Execution Time: {t_end-t_begin:.4f}s")  # 本回合执行时间（秒），从回合开始到结束的总耗时
        Logger.info(f"  Current ORR: {order_response_rates[-1]:.4f}")  # 当前回合的订单响应率=已服务订单数/总订单数，衡量系统满足订单需求的能力，值越接近1越好
        Logger.info(f"  Current GMV: {np.sum(gmv):.4f}")  # 当前回合的总订单价值=所有成功服务的订单价值之和，衡量系统创造的经济价值，值越大收益越高
        Logger.info(f"  Best ORR: {best_orr:.4f}")  # 历史最佳订单响应率，所有测试回合中达到的最高订单响应率
        Logger.info(f"  Best GMV: {best_gmv:.4f}")  # 历史最佳总订单价值，所有测试回合中达到的最高订单价值
        # Logger.info(f"  Mean Fake ORR: {np.mean(fake_orr):.4f}")  # 平均虚假订单响应率=系统对虚拟订单的响应比例（虚拟订单是系统生成的测试用订单），用于验证系统的鲁棒性和在压力下的表现
        # Logger.info(f"  Mean Fleet ORR: {np.mean(fleet_orr):.4f}")  # 平均车队订单响应率=车队整体完成订单的比例，衡量车队整体服务能力和调度效率
        Logger.info("=" * 50)  # 分隔线，标记回合结束

        # agent.save_param(args.log_dir,'param')

        # 保存分布日志
        agent.logs.save_log_distribution('distribution')

        Logger.info("=" * 50)  # 分隔线，标记测试回合结束
        Logger.info(f"TEST ROUND #{iteration} END")  # 当前测试回合编号
        Logger.info("=" * 50)  # 分隔线
    Logger.info("=" * 50)  # 分隔线
    Logger.info(f"ALL TESTS COMPLETED")  # 所有测试回合完成提示
    Logger.info("=" * 50)  # 分隔线
    
    # 打印所有迭代的 ORR 和 GMV 记录
    Logger.info(f"ALL TESTS RECORDS (Total {len(all_orr_records)} iterations)")
    Logger.info("=" * 50)
    for i, (orr, gmv) in enumerate(zip(all_orr_records, all_gmv_records)):
        Logger.info(f"Iteration #{i+args.TEST_SEED}: ORR = {orr:.4f}, GMV = {gmv:.2f}")
    Logger.info("=" * 50)


if __name__ == "__main__":
    """
    主函数入口：程序的执行起点
    
    功能流程：
        1. 获取配置参数
        2. 设置计算设备（GPU/CPU）
        3. 创建环境（根据 grid_num 选择不同的数据集）
        4. 初始化 TensorBoard writer
        5. 初始化 MAPPO Agent
        6. 根据 test 参数选择训练或测试模式
    """
    # ========== 步骤 1: 获取配置参数 ==========
    args=get_parameter()

    # ========== 步骤 2: 设置计算设备 ==========
    if args.device=='gpu':
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    # ========== 步骤 3: 创建环境 ==========
    '''
    注释掉的代码：真实环境创建方式
    dataset=kdd18(args)
    dataset.build_dataset(args)
    env=CityReal(dataset=dataset,args=args)
    '''
    
    # 根据 grid_num 选择不同的环境创建函数
    if args.grid_num==100:
        # 创建 100 网格的合成环境
        env, args.M, args.N, _, args.grid_num=create_OD()
    elif args.grid_num==36:
        # 创建 36 网格的合成环境
        env, args.M, args.N, _, args.grid_num=create_OD_36()
    elif args.grid_num == 121:
        # 加载 DiDi 121 网格的真实环境
        env, M, N, _, args.grid_num=load_envs_DiDi121(driver_num=args.driver_num)
    elif args.grid_num == 143:
        # 加载 NYU 143 网格的真实环境
        env, M, N, _, args.grid_num=load_envs_NYU143(driver_num=args.driver_num)
    
    # 设置车队帮助模式（如果匹配模式不是 local，则启用车队调度）
    env.fleet_help= args.FM_mode != 'local'

    # ========== 步骤 4: 初始化 TensorBoard writer ==========
    if args.writer_logs:
        writer=SummaryWriter(args.log_dir)
    else:
        writer=None

    # ========== 步骤 5: 初始化 MAPPO Agent ==========
    agent=PPO(env,args,device)

    # 注释掉的代码：MDP 相关
    #MDP=MdpAgent(args.TIME_LEN,args.grid_num,args.gamma)
    #if args.order_value:
        #MDP.load_param('../logs/synthetic/MDP/OD+localFM/MDPsave.pkl')
                        #logs/synthetic/MDP/OD+randomFM/MDP.pkl
    #agent.MDP=MDP
    #agent=None
    
    # 将模型移动到指定设备
    agent.move_device(device)
    
    # ========== 步骤 6: 选择训练或测试模式 ==========
    if args.test:
        # 测试模式
        logs = logfile.logs(args.log_dir, args)
        agent.logs = logs
        # 打印测试配置信息
        log_test_info(args)
        # 加载预训练模型
        agent.load_param(args.model_dir)
        # 运行测试
        test(env,agent,writer=writer, args=args,device=device)
    else:
        # 训练模式
        logs = logfile.logs(args.log_dir, args)
        agent.logs = logs
        # 运行训练
        train(env,agent,writer=writer,args=args,device=device)

    # 注释掉的代码：设置 agent 的步数（用于恢复训练）
    #agent.step=args.resume_iter
