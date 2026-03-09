import torch
import numpy as np
import pickle
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        pass

class logs:
    def __init__(self,logdir, args):
        self.logdir = logdir
        self.args = args
        self.prepare_log_phi()
        self.prepare_log_distribution()
        self.prepare_full_phi()
        self.record_state = None
        self.record_order = None

    def prepare_log_phi(self):
        with open(self.logdir+'/log_phi.txt','w') as f:
            f.write('record phi'+'\n')

    def save_log_phi(self, iter, phi):
        with open(self.logdir+'/log_phi.txt','a') as f:
            f.write('iter'+str(iter)+':')
            for k in phi:
                f.write(str(format(k.item(),'.6f'))+',')
            f.write('\n')

    def prepare_log_phi(self):
        with open(self.logdir+'/log_phi.txt','w') as f:
            f.write('record phi'+'\n')

    def save_log_phi(self, iter, phi):
        with open(self.logdir+'/log_phi.txt','a') as f:
            f.write('iter'+str(iter)+':')
            for k in phi:
                f.write(str(format(k.item(),'.6f'))+',')
            f.write('\n')

    def prepare_log_distribution(self):
        self.distribution = {
            'reward':np.zeros((self.args.TIME_LEN, self.args.grid_num)),
            'driver':np.zeros((self.args.TIME_LEN, self.args.grid_num)),
            'order':np.zeros((self.args.TIME_LEN, self.args.grid_num))
            }

    def push_log_distribution(self,t,r,d,o):
        self.distribution['reward'][t]=r
        self.distribution['driver'][t]=d
        self.distribution['order'][t]=o

    def save_log_distribution(self,name,dir=None):
        if dir==None:
            dir = self.logdir+'/{}.pkl'.format(name)
        else:
            mkdir_p(self.logdir+'/{}'.format(dir))
            dir = self.logdir+'/{}/{}.pkl'.format(dir,name)
        with open(dir,'wb') as f:
            pickle.dump(self.distribution,f)

    def prepare_full_phi(self):
        self.phi = np.zeros((self.args.TIME_LEN*self.args.parallel_episode, self.args.grid_num, self.args.meta_scope+1))

    def push_full_phi(self,phi):
        self.phi = phi

    def save_full_phi(self,name,dir=None):
        if dir==None:
            dir = self.logdir+'/{}.pkl'.format(name)
        else:
            mkdir_p(self.logdir+'/{}'.format(dir))
            dir = self.logdir+'/{}/{}.pkl'.format(dir,name)
        with open(dir,'wb') as f:
            pickle.dump(self.phi,f)

    def record_state_minmax(self, state):
        if type(state) == np.ndarray:
            state = torch.Tensor(state)
        state_max = torch.max(state,dim=0)[0]
        state_min = torch.min(state,dim=0)[0]
        if self.record_state is None:
            self.record_state = torch.stack([state_min,state_max], dim=0)
        else:
            for i in range(len(state_max)):
                if state_min[i]< self.record_state[0,i]:
                    self.record_state[0,i] = state_min[i]
                if state_max[i]> self.record_state[1,i]:
                    self.record_state[1,i] = state_max[i]

    def record_order_minmax(self, order):
        order_max = torch.max(order,dim=0)[0]
        order_min = torch.min(order,dim=0)[0]
        if self.record_order is None:
            self.record_order = torch.stack([order_min,order_max], dim=0)
        else:
            for i in range(len(order_max)):
                if order_min[i]< self.record_order[0,i]:
                    self.record_order[0,i] = order_min[i]
                if order_max[i]> self.record_order[1,i]:
                    self.record_order[1,i] = order_max[i]

    def save_feature(self):
        feature = {
            'state':self.record_state,
            'order':self.record_order
        }
        with open(self.logdir+'/feature_{}.pkl'.format(self.args.grid_num), 'wb') as f:
            pickle.dump(feature,f)


if __name__ == "__main__":
    with open('../logs/synthetic/grid143/debug_20230125_11-02/phi/0.pkl', 'rb') as f:
        feature = pickle.load(f)
    state = feature['state']
    order = feature['order']
    print(0)