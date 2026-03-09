"""
Edited by Jerry Jin: run MDP
"""

#import tensorflow as tf
import argparse
import os.path as osp
import sys
sys.path.append('../')
from simulator.envs import *
from torch.utils.tensorboard import SummaryWriter
from algo.non_nueral.mdp import MdpAgent
from simulator.objects import Order
#from algo.base import SummaryObj
from tools.load_data import *
from tools.create_envs import *
from tools import logfile
import torch

base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, 'data')


def running_example(algo, training_round=1400, fleet_help=False):

    random.seed(0)
    np.random.seed(0)
    fleet_help=True
    random_fleet=True
    dynamic_env = False
    env_seed = 326
    scale = 1

    log_dir = '../logs/' + 'synthetic/'  +'grid'+str(args.grid_num)+'/MDP/'+'OD+randomFM'
    if dynamic_env:
        log_dir+='_EnvDync'
    else:
        log_dir+='_EnvStat'+str(env_seed)
    log_dir+='_Scale'+str(scale)
    current_time = time.strftime("%Y%m%d_%H-%M")
    if os.path.exists(log_dir):
        log_dir+='_'+current_time
    print('logdir : ',log_dir)
    mkdir_p(log_dir)
    writer=SummaryWriter(log_dir)
    logs = logs = logfile.logs(log_dir, args)

    if args.grid_num == 100:
        env, M, N, central_node_ids, _ = create_OD(fleet_help,scale=scale)
    elif args.grid_num == 121:
        env, M, N, _, args.grid_num=load_envs_DiDi121(driver_num=args.driver_num)
    elif args.grid_num == 143:
        env, M, N, _, args.grid_num=load_envs_NYU143(driver_num=args.driver_num)
    env.fleet_help = fleet_help
    # initialize the model
    #config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

    #summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    #summary.register(['KL', 'Entropy', 'Fleet-ORR', 'Fake-ORR', 'ORR', 'GMV'])

    if algo == 'MDP':
        model = MdpAgent(144, M*N)
    else:
        raise Exception('Unaccepted algo type: {}'.format(algo))
    
    #model.load_param('../logs/synthetic/MDP/OD+localFM/MDP.pkl')

    env.reset_randomseed(0)

    for iteration in range(training_round):
        print('\n---- ROUND: #{} ----'.format(iteration))
        if dynamic_env:
            env.reset_randomseed(iteration*1000)
        else:
            env.reset_randomseed(env_seed*1000 )
        order_response_rates = []
        T = 0
        max_iter = 144

        states_node, states, order_list, order_idx, _, global_order_states = env.reset()

        gmv = []
        fake_orr = []
        fleet_orr = []
        kl = []
        entropy = []

        while T < max_iter:
            '''
            global_order_states= [order_id, begin id, end id, price, duration]
            order_id_pairs = [ (node id , order id),(),() ]* valid nodes 
            orders = [ order class * K]*valid nodes
            '''
            order_id_pairs = model.act(env.city_time, global_order_states)

            driver_num=[env.nodes[i].idle_driver_num for i in range(env.n_nodes)]
            order_id_pairs = [order_id_pairs[i][:driver_num[i]] for i in range(env.n_nodes)]

            orders = env.get_orders_by_id(order_id_pairs)
            
            if random_fleet and fleet_help:
                for id in range(env.n_nodes):
                    for i,o in enumerate(orders[id]):
                        if o.get_service_type()>=0:
                            end_pos=np.random.choice(env.nodes[id].layers_neighbors_id[0],1 ).item()
                            o=Order(env.nodes[id], env.nodes[end_pos],o.get_begin_time(), 1, o.get_price(), 0, service_type=1)
                            orders[id][i]=o


            next_global_order_states, node_states = env.step(orders, generate_order=1, mode='MDP')
            # distribution should gotten after step
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            kl.append(kl_value)
            entropy.append(entr_value)
            gmv.append(env.gmv)
            fake_orr.append(env.fake_response_rate)
            fleet_orr.append(env.fleet_response_rate)
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)

            reward = torch.Tensor([ node.gmv for node in env.nodes] ) 
            driver_num= torch.Tensor([node.idle_driver_num for node in env.nodes])
            order_num= torch.Tensor([node.real_order_num for node in env.nodes])
            logs.push_log_distribution(T,reward, driver_num,order_num)
            
            '''
            if T % 50 == 0:
                print(
                    'City_time: [{0:<5d}], Order_response_rate: [{1:<.4f}], KL: [{2:<.4f}], Entropy: [{3:<.4f}], Fake_orr: [{4:<.4f}], Fleet_arr: [{5:<.4f}], Idle_drivers: [{6}], Ori_order_num: [{7}], Fleet_drivers: [{8}]'.format(
                        env.city_time - 1, env.order_response_rate, kl_value, entr_value, env.fake_response_rate,
                        env.fleet_response_rate, env.ori_idle, env.ori_order_num, env.ori_fleet
                    ))
            '''
            global_order_states = next_global_order_states

            T += 1
        logs.save_log_distribution('distribution')
        print('>>> Mean_ORR: [{0:<.6f}] GMV: [{1}] Mean_KL: [{2}] Mean_Entropy: [{3}]'.format(
            order_response_rates[-1], np.sum(gmv), np.mean(kl), np.mean(entropy)))
        writer.add_scalar('train ORR',order_response_rates[-1],iteration)
        writer.add_scalar('train GMV',np.sum(gmv),iteration)
        writer.add_scalar('train KL',np.mean(kl),iteration)
        writer.add_scalar('train Entropy',np.mean(entropy),iteration)
        if iteration%10==0:
            model.save_MDP(log_dir)
        '''
        summary.write({
            'KL': np.mean(kl),
            'Entropy': np.mean(entropy),
            'Fake-ORR': np.mean(fake_orr),
            'Fleet-ORR': np.mean(fleet_orr),
            'ORR': np.mean(order_response_rates),
            'GMV': np.sum(gmv)
        }, iteration)
        '''
        # model.train()
        model.store_transitions()
        model.train()


def test_mdp(args):

    random.seed(1314520)
    np.random.seed(1314520)
    torch.manual_seed(1314520)
    Test_iter = 5
    fleet_help=True
    random_fleet=True
    dynamic_env = False
    env_seed = 326
    scale = 1
    model_dir = '../save/grid143/OD+randomFM_EnvStat326_Scale1/MDP.pkl'
    print(model_dir)

    '''
    log_dir = '../logs/' + 'synthetic/'  +'grid'+str(args.grid_num)+'/MDP/'+'OD+randomFM'
    if dynamic_env:
        log_dir+='_EnvDync'
    else:
        log_dir+='_EnvStat'+str(env_seed)
    log_dir+='_Scale'+str(scale)
    current_time = time.strftime("%Y%m%d_%H-%M")
    if os.path.exists(log_dir):
        log_dir+='_'+current_time
    print('logdir : ',log_dir)
    mkdir_p(log_dir)
    writer=SummaryWriter(log_dir)
    '''
    #logs = logs = logfile.logs(log_dir, args)

    if args.grid_num == 100:
        env, M, N, central_node_ids, _ = create_OD(fleet_help,scale=scale)
    elif args.grid_num == 121:
        env, M, N, _, args.grid_num=load_envs_DiDi121(driver_num=args.driver_num)
    elif args.grid_num == 143:
        env, M, N, _, args.grid_num=load_envs_NYU143(driver_num=args.driver_num)
    env.fleet_help = fleet_help

    model = MdpAgent(144, M*N)
    model.load_param(model_dir)
    
    env.reset_randomseed(0)
    record_orr = []
    record_gmv = []
    for iteration in range(Test_iter):
        print('\n---- ROUND: #{} ----'.format(iteration))
        if dynamic_env:
            env.reset_randomseed(iteration*1000)
        else:
            env.reset_randomseed(env_seed*1000 )
        order_response_rates = []
        T = 0
        max_iter = 144

        states_node, states, order_list, order_idx, _, global_order_states = env.reset()

        gmv = []
        fake_orr = []
        fleet_orr = []
        kl = []
        entropy = []

        while T < max_iter:
            '''
            global_order_states= [order_id, begin id, end id, price, duration]
            order_id_pairs = [ (node id , order id),(),() ]* valid nodes 
            orders = [ order class * K]*valid nodes
            '''
            order_id_pairs = model.act(env.city_time, global_order_states)

            driver_num=[env.nodes[i].idle_driver_num for i in range(env.n_nodes)]
            order_id_pairs = [order_id_pairs[i][:driver_num[i]] for i in range(env.n_nodes)]

            orders = env.get_orders_by_id(order_id_pairs)
            
            if random_fleet and fleet_help:
                for id in range(env.n_nodes):
                    for i,o in enumerate(orders[id]):
                        if o.get_service_type()>=0:
                            #end_pos=np.random.choice(env.nodes[id].layers_neighbors_id[0],1 ).item()
                            nei = env.nodes[id].layers_neighbors_id[0]
                            end_pos = nei[torch.randperm(len(nei))[0].item()]
                            o=Order(env.nodes[id], env.nodes[end_pos],o.get_begin_time(), 1, o.get_price(), 0, service_type=1)
                            orders[id][i]=o

            next_global_order_states, node_states = env.step(orders, generate_order=1, mode='MDP')
            # distribution should gotten after step
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            kl.append(kl_value)
            entropy.append(entr_value)
            gmv.append(env.gmv)
            fake_orr.append(env.fake_response_rate)
            fleet_orr.append(env.fleet_response_rate)
            order_response_rates.append(env.order_response_rate)

            reward = torch.Tensor([ node.gmv for node in env.nodes] ) 
            driver_num= torch.Tensor([node.idle_driver_num for node in env.nodes])
            order_num= torch.Tensor([node.real_order_num for node in env.nodes])
            #logs.push_log_distribution(T,reward, driver_num,order_num)
            
            global_order_states = next_global_order_states
            T += 1
        #logs.save_log_distribution('distribution')
        record_gmv.append(np.sum(gmv))
        record_orr.append(order_response_rates[-1])
        print('>>> Mean_ORR: [{0:<.6f}] GMV: [{1}] Mean_KL: [{2}] Mean_Entropy: [{3}]'.format(
            order_response_rates[-1], np.sum(gmv), np.mean(kl), np.mean(entropy)))
        '''
        writer.add_scalar('train ORR',order_response_rates[-1],iteration)
        writer.add_scalar('train GMV',np.sum(gmv),iteration)
        writer.add_scalar('train KL',np.mean(kl),iteration)
        writer.add_scalar('train Entropy',np.mean(entropy),iteration)
        '''
    record_gmv = np.array(record_gmv)
    record_orr = np.array(record_orr)
    print('GMV mean : {} , GMV std : {} '.format(record_gmv.mean(),record_gmv.std()))
    print('ORR mean : {} , ORR std : {} '.format(record_orr.mean(),record_orr.std()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='MDP', help='Algorithm Type, choices: MDP')
    parser.add_argument('-t', '--train_round', type=int, help='Training round limit', default=1400)
    parser.add_argument('-f', '--fleet_help', type=bool, help='Trigger for fleet management', default=False)
    args = parser.parse_args()
    args.driver_num = 2000
    args.grid_num = 143
    args.TIME_LEN = 144
    args.parallel_episode = 0
    args.meta_scope = 0
    args.mode = 'test'
    if args.mode == 'train':
        running_example(args.algo, args.train_round)
    elif args.mode=='test':
        test_mdp(args)
