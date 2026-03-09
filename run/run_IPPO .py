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

os.environ["CUDA_VISIBLE_DEVICES"]='0'
setproctitle.setproctitle("didi@wjw")


def get_parameter():
    parser = argparse.ArgumentParser()
    args= parser.parse_args()

    # parameter
    args.MAX_ITER=6000

    # test 相关
    args.test_dir = '../logs/synthetic/grid143/EnvStat326_OD143_FMRLmerge_Batch2000_Gamma0.97_Lambda0.95_Iter1_Ir0.001_Step144_Ent0.005_Minibatch5_Parallel5mix_MDP0_StateEmb2_Meta04_diffu1_DGCNC_relufeaNor3'
    args.model_dir = '../logs/synthetic/grid143/EnvStat326_OD143_FMRLmerge_Batch2000_Gamma0.97_Lambda0.95_Iter1_Ir0.001_Step144_Ent0.005_Minibatch5_Parallel5mix_MDP0_StateEmb2_Meta04_diffu1_DGCNC_relufeaNor3/Best.pkl'
    args.test = False
    args.TEST_ITER=1
    args.TEST_SEED = 1314520

    args.resume_iter=0
    args.device='gpu'
    args.neighbor_dispatch=False        # 是否考虑相邻网格接单
    args.onoff_driver=False             # 不考虑车辆的随机下线
    #args.log_name='M2_a0.01_reward2_t2_gamma0_value_noprice_noentropy'
    #args.log_name='advnormal_gradMean_iter10_lr3e-4_step144_clipno_batchall3_parallel1_minibatch1'
    args.log_name='debug'

    # 环境相关
    args.dispatch_interval= 10   # 决策间隔/min
    args.speed=args.dispatch_interval
    args.wait_time=args.dispatch_interval
    args.TIME_LEN = int(1440//args.dispatch_interval)   # 一天的总决策次数
    args.grid_num= 143         # 网格数量，决定了数据集
    driver_dict = {
        143:2000,
        121:1500,
        100:1000
    }
    args.driver_num=driver_dict[args.grid_num]        # 改变初始化司机数量
    args.city_time_start=0      # 一天的开始时间，时间的总长度是TIME_LEN
    args.dynamic_env = False
    seed_dict = {
        143:326,
        121:6,
        100:16
    }
    args.env_seed = seed_dict[args.grid_num]

    args.batch_size=int(1000)
    args.actor_lr=1e-3
    args.critic_lr=1e-3
    args.meta_lr = 1e-3
    args.train_actor_iters=1
    args.train_critic_iters=1
    args.train_phi_iters=1
    args.batch_size= int(args.batch_size)
    args.gamma=0.97
    args.lam=0.95
    args.max_grad_norm = 10
    args.clip_ratio=0.2
    args.ent_factor=0.005
    args.adv_normal=True
    args.clip=True
    args.steps_per_epoch=144
    args.grad_multi='mean'   # sum or mean
    #args.minibatch_num= int(round(args.steps_per_epoch*args.grid_num/args.batch_size))
    args.minibatch_num=5
    args.parallel_episode=5
    args.parallel_way='mix'    # mix, mean
    args.parallel_queue=True
    args.return_scale=False
    args.use_orthogonal=True
    args.use_value_clip=True
    args.use_valuenorm=True
    args.use_huberloss=False
    args.use_lr_anneal=False
    args.use_GAEreturn=True
    args.use_rnn=False
    args.use_GAT=False
    args.use_GCN=False
    args.use_DGCN = True
    args.use_dropout=False
    args.global_emb = False
    args.use_auxi = False
    args.auxi_loss = ['huber','mse','cos'] [1]
    args.auxi_effi=0.01
    args.use_fake_auxi=0
    args.use_regularize = ['None','L1','L2', 'L1state', 'L2state'] [0]
    args.regularize_alpha = 1e-1

    args.activate_fun='relu'
    
    args.use_neighbor_state = False     # 表示使用固定的多少阶邻居的信息作为状态
    args.adj_rank = 3
    args.merge_method = 'cat' # ['cat','res']
    args.actor_centralize = True
    args.critic_centralize = True

    args.reward_scale=5
    args.memory_size = int(args.TIME_LEN*args.parallel_episode)
    args.FM_mode = ['local','RLmerge','RLsplit' ][1]
    args.remove_fake_order=False
    args.ORR_reward=False
    args.ORR_reward_effi=1
    args.only_ORR=False

    # 特征相关
    args.feature_normal = 3 # 1和2是不同的归一化，3是加载历史的
    args.use_state_diff = False  # 将前后两状态的差值作为状态补充
    args.use_order_time = False
    args.new_order_entropy=True
    args.order_grid=True
    args.use_mdp = 0        # 0表示无，1表示表格，2表示deep
    args.update_value=True
    args.rm_state = []        # 在特征里去除 ['fea','time','id']
    args.state_remove = ''    # 在网络里去除 AC0123

    # 状态表征
    args.state_emb_choose = 2

    # 控制Logs
    args.log_feature = True
    args.log_distribution = False
    args.log_phi = False

    # 控制合作
    '''
    meta_choose : 0
        无meta 此时看team rank 和 global share
    meta_choose : 1,2,3
        都是圆环加权,1,2,3是不同的归一化方法
    meta_choose : 4, 5
        圆饼加权
        4 是 0~K阶
        5 是 1~K阶
    meta_choose : 6,7
        圆环和圆饼结合, 低阶圆饼, 高阶圆环
        6 是 前1阶圆饼
        7 是 前2阶圆饼
    '''
    # meta choose 0:无meta 此时看team rank 和 global share
    args.meta_choose =0
    args.meta_scope = 4
    args.team_rank=0
    args.global_share=False

    log_name_dict={
        'OD': args.grid_num,
        'FM': args.FM_mode,
        'Batch': args.batch_size,
        #'Advnorm': '' if args.adv_normal else 'NO',
        #'Grad': args.grad_multi,
        'Gamma':args.gamma,
        'Lambda':args.lam,
        'Iter': args.train_actor_iters,
        'Ir': args.actor_lr,
        'Step': args.steps_per_epoch,
        #'Clipnew': args.clip_ratio if args.clip else 'NO',
        'Ent': args.ent_factor,
        'Minibatch': args.minibatch_num,
        'Parallel': str(args.parallel_episode)+args.parallel_way,
        #'Rscale':args.reward_scale,
        'MDP': str(args.use_mdp),
        #'queue': '' if args.parallel_queue else 'NO',
        #'TeamR': 'share' if args.full_share else args.team_reward_factor,
        #'TeamRank': 'global' if args.global_share else args.team_rank,
        #'ORR': args.ORR_reward_effi if args.ORR_reward else 'NO' ,
        #'Actor': 'Cen' if args.actor_centralize else 'Decen',
        #'Critic': 'Cen' if args.critic_centralize else 'Decen',
        #'Auxi': args.auxi_loss+str(args.auxi_effi) if args.use_auxi else 'No',
        #'FakeNewAuxi': args.use_fake_auxi
        'StateEmb': str(args.state_emb_choose),
        #'Meta': str(args.meta_choose)+'+'
    }
    args.log_name=''
    if args.dynamic_env:
        args.log_name+= 'EnvDyna_'
    else:
        args.log_name+= 'EnvStat{}_'.format(args.env_seed)
    for k,v in log_name_dict.items():
        args.log_name+= k+str(v)+'_'
    args.log_name+='Meta'+str(args.meta_choose)
    if args.meta_choose==0:
        if args.global_share:
            args.log_name+='global'
        else:
            args.log_name+=str(args.team_rank)
    else:
        args.log_name+=str(args.meta_scope)
        args.log_name+= '_LR'+str(args.meta_lr)
    #args.log_name+='seed0'
    #args.log_name+='_car50'
    if args.order_grid==False:
        args.log_name+='_RmGrid'
    if args.only_ORR:
        args.log_name+='_onlyORR'
    if args.update_value:
        pass
        #args.log_name+='_UpVal'
    if args.state_remove != '':
        args.log_name += '_StateRm'+args.state_remove
    #args.log_name+='_KLNEW'
    if args.new_order_entropy:
        #args.log_name+='_NewEntropy'
        pass
    if args.use_state_diff :
        args.log_name+='_StateDiff'
    if args.use_orthogonal==True:
        pass
        #args.log_name+='_OrthoInit'
    if args.use_value_clip:
        pass
        #args.log_name+='_ValueClip'
    if args.use_valuenorm:
        pass
        #args.log_name+='_ValueNorm'
    if args.use_huberloss:
        pass
        #args.log_name+='_Huberloss'
    if args.use_lr_anneal:
        args.log_name+='_LRAnneal'
    if args.use_GAEreturn:
        pass
        #args.log_name+='_GAEreturn'
    if args.use_rnn:
        args.log_name+='_GRU2'
    if args.use_GAT:
        args.log_name+='_GATnew'
    if args.use_GCN:
        args.log_name+='_GCN'+str(args.adj_rank)
    if args.use_DGCN:
        args.log_name+='_DGCN'
        if args.actor_centralize:
            args.log_name+='A'
        if args.critic_centralize:
            args.log_name+='C'
    if args.use_neighbor_state:
        args.log_name+='_Statenew'+str(args.adj_rank)
    if args.use_regularize != 'None':
        args.log_name+='_'+args.use_regularize+str(args.regularize_alpha)
    if args.use_auxi:
        args.log_name+= 'Auxi'+args.auxi_loss+str(args.auxi_effi)
    if args.use_fake_auxi >0:
        args.log_name+= 'FakeNewAuxi'+str(args.use_fake_auxi)
    args.log_name+= '_'+args.activate_fun
    args.log_name+= 'feaNor'+str(args.feature_normal)
    if len(args.rm_state)>0:
        args.log_name+='_RM'
        for s in args.rm_state:
            args.log_name+=s
        

    #args.log_name+= '_'+args.merge_method
    #args.log_name+='_GAE'

    #args.log_name='advnormal_gradMean_iter10_lr3e-4_step144_clipno_batchall3_parallel1_minibatch1'
    #args.log_name='debug'

    current_time = time.strftime("%Y%m%d_%H-%M")
    #log_dir = '../logs/' + "{}".format(current_time)
    log_dir = '../logs/' + 'synthetic/'+'grid'+str(args.grid_num)+'/'+args.log_name
    if os.path.exists(log_dir):
        log_dir+='_'+current_time
    args.log_dir=log_dir
    mkdir_p(log_dir)
    print ("log dir is {}".format(log_dir))
    
    args.writer_logs=True
    if args.writer_logs:
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


def test(env, agent , writer=None,args=None,device='cpu'):
    np.random.seed(args.TEST_SEED)
    best_gmv=0
    best_orr=0
    for iteration in np.arange(args.TEST_SEED, args.TEST_SEED+args.TEST_ITER):
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

            '''
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

        '''
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

        t_end=time.time()

        if np.sum(gmv)> best_gmv:
            best_gmv=np.sum(gmv)
            best_orr=order_response_rates[-1]
        print('>>> Time: [{0:<.4f}] Mean_ORR: [{1:<.4f}] GMV: [{2:<.4f}] Best_ORR: [{3:<.4f}] Best_GMV: [{4:<.4f}]'.format(
            t_end-t_begin,order_response_rates[-1], np.sum(gmv),best_orr,best_gmv ))
        #agent.save_param(args.log_dir,'param')
        agent.logs.save_log_distribution('distribution')
        #writer.add_scalar('train ORR',order_response_rates[-1],iteration)
        #writer.add_scalar('train GMV',np.sum(gmv),iteration)
        #writer.add_scalar('train KL',np.mean(kl),iteration)
        #writer.add_scalar('train Suply/demand',np.mean(entropy),iteration)


if __name__ == "__main__":
    args=get_parameter()

    if args.device=='gpu':
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    '''
    dataset=kdd18(args)
    dataset.build_dataset(args)
    env=CityReal(dataset=dataset,args=args)
    '''
    if args.grid_num==100:
        env, args.M, args.N, _, args.grid_num=create_OD()
    elif args.grid_num==36:
        env, args.M, args.N, _, args.grid_num=create_OD_36()
    elif args.grid_num == 121:
        env, M, N, _, args.grid_num=load_envs_DiDi121(driver_num=args.driver_num)
    elif args.grid_num == 143:
        env, M, N, _, args.grid_num=load_envs_NYU143(driver_num=args.driver_num)
    
    env.fleet_help= args.FM_mode != 'local'

    if args.writer_logs:
        writer=SummaryWriter(args.log_dir)
    else:
        writer=None

    agent=PPO(env,args,device)

    #MDP=MdpAgent(args.TIME_LEN,args.grid_num,args.gamma)
    #if args.order_value:
        #MDP.load_param('../logs/synthetic/MDP/OD+localFM/MDPsave.pkl')
                        #logs/synthetic/MDP/OD+randomFM/MDP.pkl
    #agent.MDP=MDP
    #agent=None
    agent.move_device(device)
    
    if args.test:
        logs = logfile.logs(args.log_dir, args)
        agent.logs = logs
        agent.load_param(args.model_dir)
        test(env,agent,writer=writer, args=args,device=device)
    else:
        logs = logfile.logs(args.log_dir, args)
        agent.logs = logs
        train(env,agent,writer=writer,args=args,device=device)

    #agent.step=args.resume_iter