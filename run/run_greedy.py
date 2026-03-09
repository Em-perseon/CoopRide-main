"""
Edited by Jerry Jin: run MDP
编辑者: Jerry Jin: 运行MDP算法
"""

#import tensorflow as tf  # 注释掉的TensorFlow导入
import argparse  # 命令行参数解析模块
import os.path as osp  # 路径操作模块
import sys  # 系统相关模块
sys.path.append('../')  # 将父目录添加到Python路径中
from simulator.envs import *  # 导入模拟器环境相关模块
from tools.load_data import *  # 导入数据加载工具
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录器
import torch  # PyTorch深度学习框架
import numpy as np  # NumPy数值计算库
#from algo.base import SummaryObj  # 注释掉的摘要对象导入

from tools.create_envs import *  # 导入环境创建工具

# 设置基础目录路径
base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
log_dir = osp.join(base_dir, 'log')  # 日志目录
data_dir = osp.join(base_dir, 'data')  # 数据目录


def running_example(args, training_round=1400, fleet_help=False):
    """
    运行贪婪算法的示例函数
    
    参数:
        args: 命令行参数对象
        training_round: 训练轮数，默认为1400
        fleet_help: 是否启用车队管理，默认为False
    """

    # 设置日志目录路径（使用绝对路径避免 Windows 兼容性问题）
    log_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'logs', 'synthetic', 'greedy', 'debug')
    mkdir_p(log_dir)  # 创建日志目录（如果不存在）
    writer=SummaryWriter(log_dir)  # 初始化TensorBoard写入器

    # 设置随机种子以确保实验可复现
    random.seed(0)
    np.random.seed(0)

    # 根据网格数量创建不同规模的环境
    if args.grid_num == 100:
        # 创建100网格的OD(起点-终点)环境
        env, M, N, central_node_ids, _ = create_OD(fleet_help)
    elif args.grid_num == 121:
        # 加载滴滴121网格的真实数据环境
        env, args.M, args.N, _, args.grid_num=load_envs_DiDi121(driver_num=args.driver_num)
    elif args.grid_num == 143:
        # 加载NYU(纽约大学)143网格的真实数据环境
        env, args.M, args.N, _, args.grid_num=load_envs_NYU143( driver_num=args.driver_num)

    # 初始化模型（已注释，原为TensorFlow相关配置）
    #config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

    # 初始化摘要对象（已注释）
    #summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    #summary.register(['KL', 'Entropy', 'Fleet-ORR', 'Fake-ORR', 'ORR', 'GMV'])
    
    # 重置环境的随机种子
    env.reset_randomseed(0)

    # 开始训练迭代循环
    for iteration in range(training_round):
        print('\n---- ROUND: #{} ----'.format(iteration))

        # 初始化各项指标的列表
        order_response_rates = []  # 订单响应率列表
        T = 0  # 时间步计数器
        max_iter = 144  # 最大迭代次数（144个时间步，对应一天的时间片）

        # 重置环境，获取初始状态
        states_node, states, order_list, order_idx, order_feature, global_order_states = env.reset()

        # 初始化各轮次的指标存储列表
        gmv = []  # 总交易额
        fake_orr = []  # 模拟响应率
        fleet_orr = []  # 车队响应率
        kl = []  # KL散度（用于衡量订单分布和司机分布的差异）
        entropy = []  # 熵值（用于衡量分布的随机性）

        # grid_num=100  # 网格数量
        grid_num = args.grid_num  # 网格数量

        # 进入每个时间步的主循环
        while T < max_iter:
            '''
            全局订单状态格式: [order_id, begin id, end id, price, duration]
            订单ID对格式: [(节点id, 订单id), (), ...] * 有效节点数
            订单列表格式: [订单类 * K] * 有效节点数
            '''
            
            # 贪婪策略：为每个节点选择价格最高的订单
            action_ids=[]  # 存储所有节点的动作ID列表
            for node_id in range(grid_num):
                # 将该节点的订单特征转换为PyTorch张量
                feature=torch.Tensor(order_list[node_id])
                # 按价格（第2列）降序排序
                sort_logit,rank = torch.sort(feature[:,2],descending=True)
                # 获取排序后的订单ID列表
                action_ids.append([order_idx[node_id][id] for id in rank])

            # 根据动作ID获取对应的订单
            orders = env.get_orders_by_id(action_ids)

            # 执行一步环境交互
            # mode='SAC' 返回: (next_states_node, order_states, order_idx, order_feature)
            next_states_node, order_list, order_idx, order_feature = env.step(orders, generate_order=1, mode='SAC')
            
            # 获取订单分布（在step之后调用）
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            
            # 分离订单分布和司机分布
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            
            # 计算KL散度：衡量订单分布与司机分布的差异
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            
            # 记录各项指标
            kl.append(kl_value)  # KL散度
            entropy.append(entr_value)  # 熵值
            gmv.append(env.gmv)  # 总交易额
            fake_orr.append(env.fake_response_rate)  # 模拟响应率
            fleet_orr.append(env.fleet_response_rate)  # 车队响应率
            
            # 记录有效的订单响应率（大于等于0）
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)

            # 每50个时间步打印一次状态信息
            if T % 50 == 0:
                print(
                    'City_time: [{0:<5d}], Order_response_rate: [{1:<.4f}], KL: [{2:<.4f}], Entropy: [{3:<.4f}], Fake_orr: [{4:<.4f}], Fleet_arr: [{5:<.4f}], Idle_drivers: [{6}], Ori_order_num: [{7}], Fleet_drivers: [{8}]'.format(
                        env.city_time - 1, env.order_response_rate, kl_value, entr_value, env.fake_response_rate,
                        env.fleet_response_rate, env.ori_idle, env.ori_order_num, env.ori_fleet
                    ))

            T += 1  # 时间步递增
        
        # 打印本轮训练的汇总指标
        print('>>> Mean_ORR: [{0:<.6f}] GMV: [{1}] Mean_KL: [{2}] Mean_Entropy: [{3}]'.format(
            np.mean(order_response_rates), np.sum(gmv), np.mean(kl), np.mean(entropy)))
        
        # 将指标记录到TensorBoard
        writer.add_scalar('train ORR',np.mean(order_response_rates),iteration)
        writer.add_scalar('train GMV',np.sum(gmv),iteration)
        writer.add_scalar('train KL',np.mean(kl),iteration)
        writer.add_scalar('train Entropy',np.mean(entropy),iteration)

        '''
        注释掉的摘要写入代码
        summary.write({
            'KL': np.mean(kl),
            'Entropy': np.mean(entropy),
            'Fake-ORR': np.mean(fake_orr),
            'Fleet-ORR': np.mean(fleet_orr),
            'ORR': np.mean(order_response_rates),
            'GMV': np.sum(gmv)
        }, iteration)
        '''
        # model.train()  # 注释掉的模型训练代码


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加算法类型参数（可选: MDP）
    parser.add_argument('-a', '--algo', type=str, default='MDP', help='Algorithm Type, choices: MDP')
    # 添加训练轮数参数
    parser.add_argument('-t', '--train_round', type=int, help='Training round limit', default=1400)
    # 添加车队管理触发参数
    parser.add_argument('-f', '--fleet_help', type=bool, help='Trigger for fleet management', default=False)
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置司机数量和网格数量
    args.driver_num = 2000
    args.grid_num = 143
    args.train_round = 3
    
    # 运行贪婪算法示例
    running_example(args, args.train_round)


"""
================================================================================
代码整体流程逻辑说明
================================================================================

本脚本实现了一个基于贪婪策略的网约车调度算法，用于模拟和优化司机与订单的匹配过程。

整体流程可分为以下几个阶段：

1. 初始化阶段：
   - 导入必要的库（NumPy、PyTorch、TensorBoard等）和自定义模块
   - 设置基础路径（日志目录、数据目录）
   - 创建日志目录并初始化TensorBoard写入器
   - 设置随机种子以确保实验可复现性

2. 环境创建阶段：
   - 根据命令行参数中的grid_num创建不同规模的环境：
     * grid_num=100: 创建合成的100网格OD（起点-终点）环境
     * grid_num=121: 加载滴滴121网格的真实数据环境
     * grid_num=143: 加载NYU143网格的真实数据环境
   - 重置环境的随机种子

3. 主训练循环（外层循环）：
   - 进行training_round轮训练（默认1400轮）
   - 每轮训练开始时重置环境，获取初始状态
   - 初始化本轮的指标存储列表（GMV、ORR、KL散度、熵等）

4. 时间步循环（内层循环，144个时间步对应一天）：
   贪婪策略执行步骤：
   a) 为每个节点遍历所有订单
   b) 按订单价格降序排序
   c) 为每个节点选择价格最高的订单作为动作
   d) 通过env.step()执行动作，更新环境状态
   e) 获取订单分布和司机分布，计算KL散度（衡量供需匹配程度）
   f) 记录各项性能指标（ORR、GMV、KL、Entropy等）
   g) 每50个时间步打印当前状态

5. 结果记录阶段：
   - 每轮训练结束后计算平均指标
   - 使用TensorBoard记录训练过程中的各项指标趋势
   - 打印本轮训练的汇总统计信息

6. 命令行入口：
   - 使用argparse解析命令行参数
   - 设置默认参数（司机数2000，网格121）
   - 调用running_example函数启动训练

核心思想：
该贪婪算法在每个时间步为每个节点简单选择价格最高的订单，不考虑长期收益。
这种策略作为基线方法，用于与更复杂的方法（如MAPPO、MDP等）进行对比。
通过记录KL散度可以评估订单分布与司机分布的匹配程度，较小的KL散度表示供需匹配更好。

================================================================================
"""