from operator import index
import random, os
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Normal,MultivariateNormal,Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import distributions as pyd, float32
from algo.utils.valuenorm import ValueNorm
from algo.utils.layers import GraphAttentionLayer, GraphConvolutionLayer , DCGRUCell
# from algo.utils.GNN import GraphConvolution
from copy import deepcopy
from collections import namedtuple
import math
import scipy.signal

import sys

def layer_init(layer, std=np.sqrt(2), bias_const=0.0, init=True):
    if init==False:
        return layer
    torch.nn.init.orthogonal_(layer.weight, std)
    if 'bias' in dir(layer):
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def normalize(input,flag):
    if flag==False:
        return input
    else:
        mean=torch.mean(input).detach()
        #std=torch.sqrt(torch.mean((input-mean)**2)).detach()
        std = input.std().detach()
        if std==0:
            std=1
        return (input-mean)/std

def huber_loss(e, d):
    a = (torch.abs(e) <= d).float()
    b = (torch.abs(e) > d).float()
    return a*e**2/2 + b*d*(torch.abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def cos_loss(pred, label):
    return torch.sum(pred*label,dim=-1)/torch.sqrt(torch.sum(pred**2, dim=-1)*torch.sum(label**2, dim=-1))

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr-(0.5e-4)) * (epoch / float(total_num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class order_embedding(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim, contin_dim, activate_fun=F.relu , gain=1 ,init=True):
        super(order_embedding,self).__init__()
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.contin_dim = int(contin_dim)
        self.activate = activate_fun

        self.grid_embedding= layer_init(nn.Embedding(grid_dim,embedding_dim),std=gain, bias_const=0, init=init)
        self.contin_embedding= layer_init(nn.Linear(self.contin_dim,embedding_dim),std=gain, bias_const=0, init=init)
        self.order_layer2= layer_init( nn.Linear(3*embedding_dim,1*embedding_dim), std=gain, bias_const=0, init=init )
        self.order_layer3= layer_init( nn.Linear(1*embedding_dim,1*embedding_dim), std=1, bias_const=0, init=init )

    def forward(self,order):
        '''
        grid= order[:,:,:2].long()
        contin=order[:,:,2:].float()
        '''
        grid= order[...,:2].long()
        contin=order[...,2:].float()
        grid_emb= self.activate(self.grid_embedding(grid))
        contin_emb = self.activate(self.contin_embedding(contin))
        #order_emb=torch.cat([grid_emb[:,:,0,:],grid_emb[:,:,1,:],contin_emb],dim=-1)
        order_emb=torch.cat([grid_emb[...,0,:],grid_emb[...,1,:],contin_emb],dim=-1)
        order_emb= self.activate(self.order_layer2(order_emb))
        order_emb = self.order_layer3(order_emb)
        return order_emb

class order_embedding2(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim, contin_dim, activate_fun=F.relu , gain=1 ,init=True):
        super(order_embedding2,self).__init__()
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.contin_dim = int(contin_dim)
        self.activate = activate_fun

        self.grid_embedding= layer_init(nn.Embedding(grid_dim,embedding_dim),std=gain, bias_const=0, init=init)
        self.time_embedding= layer_init(nn.Embedding(time_dim,embedding_dim),std=gain, bias_const=0, init=init)
        self.contin_embedding= layer_init(nn.Linear(self.contin_dim,embedding_dim),std=gain, bias_const=0, init=init)
        self.order_layer2= layer_init( nn.Linear(4*embedding_dim,1*embedding_dim), std=gain, bias_const=0, init=init )
        self.order_layer3= layer_init( nn.Linear(1*embedding_dim,1*embedding_dim), std=1, bias_const=0, init=init )

    def forward(self,order):
        '''
        grid= order[:,:,:2].long()
        contin=order[:,:,2:].float()
        '''
        grid= order[...,1:3].long()
        time = order[...,0].long()
        contin=order[...,3:].float()
        grid_emb= self.activate(self.grid_embedding(grid))
        time_emb = self.activate(self.time_embedding(time))
        contin_emb = self.activate(self.contin_embedding(contin))
        #order_emb=torch.cat([grid_emb[:,:,0,:],grid_emb[:,:,1,:],contin_emb],dim=-1)
        order_emb=torch.cat([time_emb ,grid_emb[...,0,:],grid_emb[...,1,:],contin_emb],dim=-1)
        order_emb= self.activate(self.order_layer2(order_emb))
        order_emb = self.order_layer3(order_emb)
        return order_emb

class state_embedding(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim ,contin_dim, activate_fun=F.relu , gain=1 ,init=True, llm_dim=0, llm_proj_dim=64):
        super(state_embedding,self).__init__()
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.contin_dim = int(contin_dim)
        self.activate = activate_fun
        self.llm_dim = int(llm_dim) if llm_dim else 0
        self.llm_proj_dim = int(llm_proj_dim) if self.llm_dim else 0
        self.base_contin_dim = self.contin_dim - self.llm_dim if self.llm_dim else self.contin_dim
        if self.base_contin_dim < 0:
            self.base_contin_dim = 0

        self.grid_embedding = layer_init( nn.Embedding(grid_dim,embedding_dim), std=gain, bias_const=0, init=init)
        self.time_embedding = layer_init( nn.Embedding(time_dim,embedding_dim), std=gain, bias_const=0, init=init )
        if self.llm_dim > 0:
            self.llm_proj = layer_init(nn.Linear(self.llm_dim, self.llm_proj_dim), std=gain, bias_const=0, init=init)
            self.contin_embedding = layer_init(
                nn.Linear(self.base_contin_dim + self.llm_proj_dim, embedding_dim), std=gain, bias_const=0, init=init
            )
        else:
            self.contin_embedding = layer_init( nn.Linear(self.contin_dim,embedding_dim), std=gain, bias_const=0, init=init)
        self.state_layer2 = layer_init( nn.Linear(3*embedding_dim,1*embedding_dim), std=gain, bias_const=0, init=init)

    def forward(self,state):
        '''
        time=state[:,0].long()
        grid= state[:,1].long()
        contin=state[:,2:].float()
        '''
        time=state[...,0].long()
        grid= state[...,1].long()
        contin=state[...,2:].float()
        time_emb= self.activate(self.time_embedding(time))
        grid_emb= self.activate(self.grid_embedding(grid))
        if self.llm_dim > 0:
            base = contin[..., :self.base_contin_dim]
            llm = contin[..., self.base_contin_dim:self.base_contin_dim + self.llm_dim]
            llm_feat = self.activate(self.llm_proj(llm * 10.0))
            if self.base_contin_dim > 0:
                contin_in = torch.cat([base, llm_feat], dim=-1)
            else:
                contin_in = llm_feat
        else:
            contin_in = contin
        contin_emb = self.activate(self.contin_embedding(contin_in))
        state_emb=torch.cat([time_emb,grid_emb,contin_emb],dim=-1)
        state_emb= self.activate(self.state_layer2(state_emb))
        return state_emb


class RNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num, init=True):
        super(RNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim= output_dim
        self.layer_num=layer_num
        self.init=init

        self.rnn = nn.GRU(input_dim, output_dim, num_layers=layer_num)
        #self.norm = nn.LayerNorm(outputs_dim)
        if self.init:
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    def forward(self, input,hidden):
        '''
        input: (seq length, batch size, feature dim)
        hidden:(layer num,  batch size, hidden dim)
        output:(seq length, batch size, output dim)
        '''
        output,hidden= self.rnn(input.unsqueeze(0),hidden)
        return output.squeeze(0), hidden

class GATLayer(nn.Module):
    def __init__(self, nfeat, nhid, output_dim, dropout, alpha, nheads, use_dropout=False, activate_fun=F.relu , gain=1 ,init=True):
        """Dense version of GAT."""
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.use_dropout= use_dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, use_dropout=use_dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.activate = activate_fun
        self.out_att = GraphAttentionLayer(nhid * nheads, output_dim, dropout=dropout, alpha=alpha, concat=False, use_dropout=use_dropout)
        self.fc = layer_init( nn.Linear(output_dim,output_dim), std=gain, bias_const=0, init=init)

    def forward(self, x, adj):
        if self.use_dropout:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        if self.use_dropout:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x,adj))
        return self.activate( self.fc(x) )
        #x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)


class DGCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim ,output_dim,num_nodes, adj, activate_fun=F.relu , gain=1 ,init=True):
        super(DGCNLayer, self).__init__()
        self.input_dim = input_dim # 1433
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.GCN1 = DCGRUCell(self.input_dim, self.hidden_dim , adj, max_diffusion_step=1, num_nodes=num_nodes) #torch.tanh
        self.GCN2 = DCGRUCell(self.hidden_dim, self.output_dim , adj, max_diffusion_step=1, num_nodes=num_nodes)
        self.activate = activate_fun
        self.fc = layer_init( nn.Linear(output_dim,output_dim), std=gain, bias_const=0, init=init)

    def forward(self, x):
        x = self.GCN1(x)
        x = self.GCN2(x)
        x =  self.fc(x) 
        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim ,output_dim, activate_fun=F.relu , gain=1 ,init=True):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim # 1433
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.GCN1 = GraphConvolutionLayer(self.input_dim, self.hidden_dim ,activation=F.relu) #torch.tanh
        self.GCN2 = GraphConvolutionLayer(self.hidden_dim, output_dim, activation=F.relu)
        self.activate = activate_fun
        self.fc = layer_init( nn.Linear(output_dim,output_dim), std=gain, bias_const=0, init=init)

    def forward(self, x, support):
        x = self.GCN1(x, support)
        x = self.GCN2(x, support)
        x = self.activate( self.fc(x) )
        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss


class NeighborLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activate_fun=F.relu , gain=1 , init=True):
        super(NeighborLayer, self).__init__()  
        self.input_dim = input_dim
        self.output_dim= output_dim
        self.init=init    
        self.activate = activate_fun
        self.fc = layer_init( nn.Linear(input_dim,output_dim), std=gain, bias_const=0, init=init)

    def forward(self, x, adj):
        output = torch.matmul(adj,x)/torch.sum(adj,dim=-1,keepdim=True)
        output = self.activate( self.fc(output) )
        return output


class state_representation1(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim , state_contin_dim ,  init=True, use_rnn=False, use_GAT=False, use_GCN=False , use_DGCN=False, adj=None , use_neighbor_state=False , merge_method = 'cat' , use_auxi = False , use_dropout=False, activate_fun='relu', global_emb = False, llm_dim=0, llm_proj_dim=64):
        super(state_representation1,self).__init__()
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.use_rnn=use_rnn
        self.use_GAT=use_GAT
        self.use_GCN=use_GCN
        self.use_DGCN=use_DGCN
        self.adj = adj
        self.use_neighbor_state = use_neighbor_state
        self.use_dropout=use_dropout
        self.merge_method = merge_method
        self.use_auxi = use_auxi
        self.global_emb = global_emb
        if activate_fun =='tanh':
            self.activate = torch.tanh
        elif activate_fun=='relu':
            self.activate = F.relu
        gain = nn.init.calculate_gain(activate_fun)
        self.state_layer = state_embedding(grid_dim,time_dim,embedding_dim, state_contin_dim, self.activate, gain ,init, llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)
        self.key_embedding_dim=embedding_dim
        if self.use_neighbor_state:
            self.neighbor_layer = NeighborLayer(embedding_dim, embedding_dim,  self.activate, gain ,init)
            self.key_embedding_dim=embedding_dim*2
        if self.use_rnn:
            self.rnn_layer=RNNLayer(embedding_dim, embedding_dim,1)
            self.key_embedding_dim=embedding_dim*2
        if self.use_GAT:
            self.gnn_layer= GATLayer(nfeat=embedding_dim, nhid= int(embedding_dim/8), output_dim=embedding_dim, dropout=0, alpha=0.2, nheads=8, use_dropout=use_dropout, activate_fun=self.activate, gain=gain ,init=init)
            self.key_embedding_dim=embedding_dim*2
        if self.use_GCN:
            self.gnn_layer = GCNLayer( input_dim=embedding_dim, hidden_dim=embedding_dim, output_dim=embedding_dim, activate_fun=self.activate, gain=gain ,init=init )
            self.key_embedding_dim=embedding_dim*2
        if self.use_DGCN:
            self.gnn_layer = DGCNLayer(input_dim=embedding_dim, hidden_dim=embedding_dim, output_dim=embedding_dim, num_nodes=grid_dim, adj=adj)
        if self.merge_method == 'res':
            self.key_embedding_dim=embedding_dim
        self.key_layer = layer_init( nn.Linear(self.key_embedding_dim,embedding_dim), std=1, bias_const=0, init=init)
        if self.use_auxi:
            self.auxiliary_layer = layer_init( nn.Linear(self.embedding_dim,embedding_dim), std=gain, bias_const=0, init=init)

    def forward(self, state):
        state_emb = self.state_layer(state)
        if self.use_neighbor_state:
            neighbor_emb = self.neighbor_layer(state_emb, adj)
            if self.merge_method == 'cat':
                state_emb = torch.cat([state_emb, neighbor_emb], dim=-1)
            elif self.merge_method == 'res':
                state_emb = state_emb+neighbor_emb
        if self.use_rnn:
            rnn_emb, hidden_state = self.rnn_layer(state_emb,hidden_state)
            if self.merge_method == 'cat':
                state_emb = torch.cat([state_emb,rnn_emb],dim=-1)     #shape=(agent num, 2*embedding dim)
            elif self.merge_method == 'res':
                state_emb = state_emb+rnn_emb
        if self.use_GAT or self.use_GCN:
            # adj= (N,N)
            gnn_embedding = self.gnn_layer(state_emb, adj)  #shape=(agent num, embedding dim)
            if self.merge_method == 'cat':
                state_emb = torch.cat([state_emb,gnn_embedding],dim=-1)     #shape=(agent num, 2*embedding dim)
            elif self.merge_method == 'res':
                state_emb = state_emb+gnn_embedding
        if self.use_DGCN:
            state_emb = self.gnn_layer(state_emb).squeeze(0)
        state_emb = self.key_layer(state_emb)
        return state_emb


class state_representation2(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim ,contin_dim, use_DGCN=False, adj=None, llm_dim=0, llm_proj_dim=64):
        super(state_representation2,self).__init__()
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.contin_dim = int(contin_dim)
        self.llm_dim = int(llm_dim) if llm_dim else 0
        self.llm_proj_dim = int(llm_proj_dim) if self.llm_dim else 0
        self.base_contin_dim = self.contin_dim - self.llm_dim if self.llm_dim else self.contin_dim
        if self.base_contin_dim < 0:
            self.base_contin_dim = 0
        self.use_DGCN = use_DGCN
        self.activate = F.relu
        gain = nn.init.calculate_gain('relu')
        self.grid_embedding = layer_init( nn.Embedding(grid_dim,embedding_dim//2), std=gain, bias_const=0, init=True)
        self.time_embedding = layer_init( nn.Embedding(time_dim,embedding_dim//2), std=gain, bias_const=0, init=True )
        if self.llm_dim > 0:
            self.llm_proj = layer_init(nn.Linear(self.llm_dim, self.llm_proj_dim), std=gain, bias_const=0, init=True)
            contin_in_dim = self.base_contin_dim + self.llm_proj_dim
        else:
            contin_in_dim = self.contin_dim
        self.contin_embedding = layer_init( nn.Linear(contin_in_dim,embedding_dim//2), std=gain, bias_const=0, init=True)
        if self.use_DGCN:
            self.gnn_layer = DGCNLayer(input_dim=embedding_dim//2, hidden_dim=embedding_dim//2, output_dim=embedding_dim//2, num_nodes=grid_dim, adj=adj)
        self.state_fc1 = layer_init( nn.Linear(embedding_dim//2*3,embedding_dim), std=gain, bias_const=0, init=True)
        self.state_fc2 = layer_init( nn.Linear(embedding_dim,embedding_dim), std=1, bias_const=0, init=True)

    def forward(self, state):
        time=state[...,0].long()
        grid= state[...,1].long()
        contin=state[...,2:].float()

        time_emb= self.time_embedding(time)
        grid_emb= self.grid_embedding(grid)
        if self.llm_dim > 0:
            base = contin[..., :self.base_contin_dim]
            llm = contin[..., self.base_contin_dim:self.base_contin_dim + self.llm_dim]
            llm_feat = self.activate(self.llm_proj(llm * 10.0))
            if self.base_contin_dim > 0:
                contin_in = torch.cat([base, llm_feat], dim=-1)
            else:
                contin_in = llm_feat
        else:
            contin_in = contin
        contin_emb = self.contin_embedding(contin_in)
        if self.use_DGCN:
            contin_emb = self.gnn_layer(contin_emb)
            if len(state.shape)==2:
                contin_emb = contin_emb.squeeze(0)
        state_emb = torch.cat([time_emb, grid_emb, contin_emb],dim=-1)
        state_emb = self.activate( self.state_fc1(state_emb) )
        state_emb = self.state_fc2(state_emb)
        return state_emb

    def global_emb(self,state):
        time=state[...,0].long()
        contin=state[...,2:].float()
        time_emb= self.time_embedding(time)
        if self.llm_dim > 0:
            base = contin[..., :self.base_contin_dim]
            llm = contin[..., self.base_contin_dim:self.base_contin_dim + self.llm_dim]
            llm_feat = self.activate(self.llm_proj(llm * 10.0))
            if self.base_contin_dim > 0:
                contin_in = torch.cat([base, llm_feat], dim=-1)
            else:
                contin_in = llm_feat
        else:
            contin_in = contin
        contin_emb = self.contin_embedding(contin_in)
        if self.use_DGCN:
            contin_emb = self.gnn_layer(contin_emb)
        return time_emb, contin_emb


class Actor(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim , state_contin_dim , order_contin_dim, init=True, use_rnn=False, use_GAT=False, use_GCN=False , use_DGCN=False, adj=None , use_neighbor_state=False , merge_method = 'cat' , use_auxi = False , use_dropout=False, activate_fun='relu', global_emb = False, order_emb_choose=1 , state_emb_choose=1, state_remove ='', llm_dim=0, llm_proj_dim=64):
        super(Actor, self).__init__()
        # order : [origin grid id, des grid id, pickup time ,duration, price ]
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.use_rnn=use_rnn
        self.use_GAT=use_GAT
        self.use_GCN=use_GCN
        self.use_DGCN=use_DGCN
        self.adj = adj
        self.use_neighbor_state = use_neighbor_state
        self.use_dropout=use_dropout
        self.merge_method = merge_method
        self.use_auxi = use_auxi
        self.global_emb = global_emb
        self.state_emb_choose = state_emb_choose
        self.state_remove = state_remove
        if activate_fun =='tanh':
            self.activate = torch.tanh
        elif activate_fun=='relu':
            self.activate = F.relu
        gain = nn.init.calculate_gain(activate_fun)

        if order_emb_choose==1:
            self.order_layer = order_embedding(grid_dim,time_dim,embedding_dim, order_contin_dim, self.activate, gain ,init)
        elif order_emb_choose==2:
            self.order_layer = order_embedding2(grid_dim,time_dim,embedding_dim, order_contin_dim, self.activate, gain ,init)
        if state_emb_choose==1:
            self.state_layer = state_representation1(grid_dim,time_dim,embedding_dim , state_contin_dim ,  init, use_rnn, use_GAT, use_GCN, use_DGCN, adj , use_neighbor_state , merge_method  , use_auxi  , use_dropout, activate_fun, global_emb, llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)
        elif state_emb_choose==2:
            self.state_layer = state_representation2(grid_dim,time_dim,embedding_dim ,state_contin_dim, use_DGCN, adj, llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)

    def forward(self, state, order, mask , adj=None ,hidden_state=None ,scale=True, return_logit=False):
        # key embedding
        if mask.dtype is not torch.bool:
            mask=mask.bool()
        state = self.state_wrapper(state)
        state_emb = self.state_layer(state)
        order_emb = self.order_layer(order) # [batch size, order length, emb size]
        compatibility = torch.squeeze(torch.matmul(state_emb[:,None,:], order_emb.transpose(-2, -1)),dim=1)
        if scale:
            compatibility/=math.sqrt(state_emb.size(-1))
        compatibility -= compatibility.max(-1)[0][:,None]
        compatibility[~mask]=-math.inf
        if self.global_emb:
            probs=F.softmax(compatibility, dim=-1)  # [batch size, order length]
            global_emb = torch.sum(order_emb*(probs[:,:,None]), dim=1)  # [batch size, emb size]
            compatibility = torch.squeeze(torch.matmul(global_emb[:,None,:], order_emb.transpose(-2, -1)),dim=1) # [batch size, order length]
            if scale:
                compatibility/=math.sqrt(state_emb.size(-1))
            compatibility[~mask]=-math.inf
        if return_logit:
            return compatibility, hidden_state
        else:
            probs=F.softmax(compatibility, dim=-1)  # [batch size, order length]
            return probs, hidden_state

    def get_state_emb(self, state):
        state_emb = self.state_layer(state)
        return state_emb

    def auxiliary_emb(self, state):
        state_emb = self.state_layer(state)
        auxi_emb = self.activate( self.auxiliary_layer(state_emb) )
        return auxi_emb

    def multi_mask_forward(self, state, order, mask , adj=None,hidden_state=None, scale=True):
        # key embedding
        if mask.dtype is not torch.bool:
            mask=mask.bool()
        state = self.state_wrapper(state)
        state_emb = self.state_layer(state)   # [batch/grid num, grid_num, emb size]
        order_emb = self.order_layer(order) # [batch/grid num, grid_num, order ength , emb size]
        #compatibility = torch.matmul(state_emb[:,None,:], order_emb.transpose(-2, -1))
        order_emb = order_emb.transpose(-2, -1) # [batch/grid num, grid_num,  emb size, order ength ]
        compatibility = torch.matmul(state_emb[...,None,:], order_emb)  # [batch/grid num, grid_num, 1, order length]
        if scale:
            compatibility/=math.sqrt(state_emb.size(-1))
        #compatibility= compatibility.repeat(1,mask.shape[1],1)
        compatibility -= torch.max(compatibility, dim=-1, keepdim=True)[0]
        if self.global_emb:
            compatibility[~mask[...,0:1,:]]=-math.inf
            probs=F.softmax(compatibility, dim=-1)  # [batch size, 1, order length]
            alpha = probs+0.00
            alpha[~mask[...,0:1,:]] = 0
            global_emb = order_emb*alpha
            global_emb = torch.sum(global_emb,dim=-1)
            # = torch.sum(order_emb*probs, dim=-1)  # [batch size, emb size]
            compatibility = torch.matmul(global_emb[...,None,:], order_emb) # [batch size, grid num , 1, order length]
            if scale:
                compatibility/=math.sqrt(state_emb.size(-1))
        repeat_shape = [1 for _ in compatibility.shape]
        repeat_shape[-2]= mask.shape[-2]
        compatibility= compatibility.repeat(tuple(repeat_shape))    # [batch , grid num , orderr length, order length]
        compatibility[~mask]=-math.inf
        probs=F.softmax(compatibility, dim=-1)
        return probs, hidden_state

    def _distribution(self, state, order, mask):
        probs = self.forward(state, order, mask)
        return Categorical(probs=probs)

    def state_wrapper(self,state):
        if 'A' in self.state_remove:
            if '0' in self.state_remove:
                state[...,0] = 0    # delete time
            if '1' in self.state_remove:
                state[...,1] = 0    # delete grid id
            if '2' in self.state_remove:
                state[...,2:] = 0
            if '3' in self.state_remove:
                state[...,2:4] = 0
        return state


class Critic0(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim , state_contin_dim , order_contin_dim, init=True, use_rnn=False, use_GAT=False,  use_GCN=False ,  use_DGCN=False, adj=None ,use_neighbor_state=False , merge_method = 'cat' , use_auxi = False, activate_fun='relu', state_emb_choose=1,  state_remove ='', llm_dim=0, llm_proj_dim=64):
        super(Critic0, self).__init__()
        # order : [origin grid id, des grid id, pickup time ,duration, price ]
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.use_rnn = use_rnn
        self.use_GAT = use_GAT
        self.use_GCN = use_GCN
        self.use_DGCN= use_DGCN
        self.adj = adj
        self.use_neighbor_state = use_neighbor_state
        self.merge_method = merge_method
        self.use_auxi = use_auxi
        self.state_emb_choose = state_emb_choose
        self.state_remove = state_remove
        if activate_fun =='tanh':
            self.activate = torch.tanh
        elif activate_fun=='relu':
            self.activate = F.relu
        gain = nn.init.calculate_gain(activate_fun)

        if state_emb_choose==1:
            self.state_layer = state_representation1(grid_dim,time_dim,embedding_dim , state_contin_dim ,  init, use_rnn, use_GAT, use_GCN, use_DGCN, adj , use_neighbor_state , merge_method  , use_auxi  , use_dropout, activate_fun, global_emb, llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)
        elif state_emb_choose==2:
            self.state_layer = state_representation2(grid_dim,time_dim,embedding_dim ,state_contin_dim, use_DGCN, adj, llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)

        self.value_layer = layer_init( nn.Linear(self.embedding_dim,1), std=1, bias_const=0, init=init)
        if self.use_auxi:
            self.auxiliary_layer = layer_init( nn.Linear(self.embedding_dim,embedding_dim), std=gain, bias_const=0, init=init)

    def forward(self, state, adj=None , hidden_state=None):
        state_emb = self.state_layer(state)
        value = self.value_layer(self.activate(state_emb))
        return value, 0,hidden_state

    def get_state_emb(self, state):
        state_emb = self.state_layer(state)
        return state_emb

    def auxiliary_emb(self, state):
        state_emb = self.state_layer(state)
        auxi_emb = self.activate( self.auxiliary_layer(state_emb) )
        return auxi_emb


class Critic(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim , state_contin_dim , order_contin_dim, init=True, use_rnn=False, use_GAT=False,  use_GCN=False ,  use_DGCN=False, adj=None ,use_neighbor_state=False , merge_method = 'cat' , use_auxi = False, activate_fun='relu', state_emb_choose=1, meta_scope = 3 , meta_choose = 0, state_remove='', llm_dim=0, llm_proj_dim=64):
        super(Critic, self).__init__()
        # order : [origin grid id, des grid id, pickup time ,duration, price ]
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        self.use_rnn = use_rnn
        self.use_GAT = use_GAT
        self.use_GCN = use_GCN
        self.use_DGCN= use_DGCN
        self.adj = adj
        self.use_neighbor_state = use_neighbor_state
        self.merge_method = merge_method
        self.use_auxi = use_auxi
        self.state_emb_choose = state_emb_choose
        self.state_remove = state_remove
        if activate_fun =='tanh':
            self.activate = torch.tanh
        elif activate_fun=='relu':
            self.activate = F.relu
        gain = nn.init.calculate_gain(activate_fun)
        self.meta_scope = meta_scope
        self.meta_choose = meta_choose

        if state_emb_choose==1:
            self.state_layer = state_representation1(grid_dim,time_dim,embedding_dim , state_contin_dim ,  init, use_rnn, use_GAT, use_GCN, use_DGCN, adj , use_neighbor_state , merge_method  , use_auxi  , use_dropout, activate_fun, global_emb, llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)
        elif state_emb_choose==2:
            self.state_layer = state_representation2(grid_dim,time_dim,embedding_dim ,state_contin_dim, use_DGCN, adj, llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)

        self.local_value_layer = layer_init( nn.Linear(self.embedding_dim,self.meta_scope+1), std=1, bias_const=0, init=init)
        if self.meta_choose>0:
            self.Phi_layer = layer_init( nn.Linear(self.embedding_dim,self.meta_scope+1), std=1, bias_const=0, init=init)
            global_emb_dim = embedding_dim if self.use_DGCN else embedding_dim//2*(self.grid_dim+1)
            self.global_fc_layer = layer_init( nn.Linear(global_emb_dim,self.embedding_dim), std=gain, bias_const=0, init=init)
            self.global_value_layer = layer_init( nn.Linear(self.embedding_dim,1), std=1, bias_const=0, init=init)

    def forward(self, state, adj=None , hidden_state=None):
        state = self.state_wrapper(state)
        state_emb = self.state_layer(state)
        local_value = self.local_value_layer(self.activate(state_emb))
        return local_value, hidden_state

    def get_local_value(self, state, adj=None , hidden_state=None):
        state = self.state_wrapper(state)
        state_emb = self.state_layer(state)
        local_value = self.local_value_layer(self.activate(state_emb))
        return local_value, hidden_state

    def get_global_value(self,state,adj=None , hidden_state=None):
        if self.meta_choose==0:
            return torch.zeros((1,1)).float(), hidden_state
        if len(state.shape)==2:
            state = state[None,:,:]
        state = self.state_wrapper(state)
        time_emb, contin_emb = self.state_layer.global_emb(state)
        if self.use_DGCN:
            state_emb = torch.cat([time_emb[:,0],contin_emb.mean(1)],dim=-1)
        else:
            state_emb = torch.cat([time_emb[:,0],contin_emb.reshape(contin_emb.shape[0],-1)],dim=-1)
        state_emb = self.activate( self.global_fc_layer(state_emb) )
        global_value = self.global_value_layer( state_emb )
        return global_value, hidden_state

    def get_state_emb(self, state):
        state_emb = self.state_layer(state)
        return state_emb

    def get_phi(self, state):
        state = self.state_wrapper(state)
        state_emb = self.activate( self.state_layer(state) )
        if self.meta_choose<4:  # 圆环
            Phi = torch.sigmoid( self.Phi_layer( state_emb ) )
        elif self.meta_choose==4:
            Phi = F.softmax( self.Phi_layer( state_emb ),dim=-1 )
        elif self.meta_choose==5:
            Phi_emb = self.Phi_layer( state_emb )
            Phi_emb[...,0] = -math.inf
            Phi = F.softmax( Phi_emb ,dim=-1 )
        elif self.meta_choose==6:
            Phi_emb = self.Phi_layer( state_emb )
            Phi_emb[...,0] = -math.inf
            Phi = F.softmax( Phi_emb ,dim=-1 )
        elif self.meta_choose==7:
            Phi_emb = self.Phi_layer( state_emb )
            Phi_emb[...,:2] = -math.inf
            Phi = F.softmax( Phi_emb ,dim=-1 )
        return Phi

    def state_wrapper(self,state):
        if 'C' in self.state_remove:
            if '0' in self.state_remove:
                state[...,0] = 0    # delete time
            if '1' in self.state_remove:
                state[...,1] = 0    # delete grid id
            if '2' in self.state_remove:
                state[...,2:] = 0
            if '3' in self.state_remove:
                state[...,2:4] = 0
        return state

class MdpAgent(object):
    def __init__(self, time_len, node_num, gamma=0.99):
        self.gamma = gamma  # discount for future value
        self.time_len=time_len
        self.node_num=node_num
        self.value_state = np.zeros([time_len + 1, node_num ])
        self.n_state = np.zeros([time_len + 1, node_num ])
        self.cur_time=0
        self.value_iter=[]

    def get_value(self,order):
        # [begin node, end node, price, duration ,service type]
        value= order[2] + pow(self.gamma,order[3])*self.value_state[min(self.cur_time+order[3],self.time_len), order[1]]  -self.value_state[self.cur_time, order[0]]
        #value= pow(self.gamma,order[3])*self.value_state[min(self.cur_time+order[3],self.time_len), order[1]]  -self.value_state[self.cur_time, order[0]]
        return value

    def update_value(self,order,selected_ids,env):
        value_record=[]
        for _node_id in env.get_node_ids():
            num= min(env.nodes[_node_id].idle_driver_num, len(selected_ids[_node_id]))
            for k in range(num):
                id= selected_ids[_node_id][k]
                o=order[_node_id][id]
                #self.n_state[self.cur_time, o[0]] += 1
                value=self.get_value(o)
                td= value
                #self.value_state[self.cur_time,o[0]]+= 1/self.n_state[self.cur_time, o[0]]*td
                self.value_state[self.cur_time,o[0]] = 199/200* self.value_state[self.cur_time, o[0]]+ 1/200*td
                value_record.append(value)
        self.value_iter.append(np.mean(value_record))

    def save_param(self,dir):
        save_dict={
            'value':self.value_state,
            'num':self.n_state
        }
        with open(dir+'/'+'MDP.pkl','wb') as f:
            pickle.dump(save_dict,f)

    def load_param(self,dir):
        with open(dir,'rb') as f:
            MDP_param=pickle.load(f)
        self.value_state= MDP_param['value']
        self.n_state = MDP_param['num']


class DeepMdpAgent(nn.Module):
    def __init__(self,grid_dim,time_dim,embedding_dim , driver_num,init=True,  activate_fun='relu'):
        super(DeepMdpAgent, self).__init__()
        self.grid_dim=grid_dim
        self.time_dim=time_dim
        self.embedding_dim=embedding_dim
        if activate_fun =='tanh':
            self.activate = torch.tanh
        elif activate_fun=='relu':
            self.activate = F.relu
        gain = nn.init.calculate_gain(activate_fun)
        self.grid_embedding = layer_init( nn.Embedding(grid_dim,embedding_dim//2), std=1, bias_const=0, init=init)
        self.time_embedding = layer_init( nn.Embedding(time_dim,embedding_dim//2), std=1, bias_const=0, init=init )
        self.state_layer1 = layer_init( nn.Linear(embedding_dim,embedding_dim), std=gain, bias_const=0, init=init)
        self.state_layer2 = layer_init( nn.Linear(embedding_dim,embedding_dim), std=gain, bias_const=0, init=init)
        self.value_layer = layer_init( nn.Linear(embedding_dim,1), std=1, bias_const=0, init=init)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, eps=1e-5)
        self.state_pool = torch.zeros((time_dim,driver_num,2)).long()
        self.targetV_pool = torch.zeros((time_dim,driver_num,1)).float()
        self.mask_pool = torch.zeros((time_dim,driver_num)).bool()
        self.ptr=0
        self.step=0
        
    def get_value(self, state):
        state = state.long()    # [time, grid]
        time_emb= self.time_embedding(state[...,0])
        grid_emb= self.grid_embedding(state[...,1])
        state_emb=torch.cat([time_emb,grid_emb],dim=-1)
        state_emb = self.activate(self.state_layer1(state_emb))
        state_emb = self.activate(self.state_layer2(state_emb))
        value = self.value_layer(state_emb)
        return value

    def memory_info(self, state, target):
        self.state_info = state.clone()
        self.target_info = target.clone()

    def push(self, orders, select):
        mask = torch.zeros(self.target_info.shape[0]).bool()
        order_num=0
        for i,o in enumerate(orders):
            for s in select[i]:
                mask[s+order_num]=1
            order_num+=len(o)
        select_num = torch.sum(mask)
        self.state_pool[self.ptr,:select_num] = self.state_info[mask]
        self.targetV_pool[self.ptr,:select_num] = self.target_info[mask][:,None]
        self.mask_pool[self.ptr,:select_num] = 1
        self.ptr+=1
        self.ptr = self.ptr%self.time_dim

    def update(self, device, batch_size=5000, iters=5):
        state = self.state_pool[self.mask_pool].to(device)
        targetV = self.targetV_pool[self.mask_pool].to(device)
        record_loss=[]
        for iter in range(iters):
            for index in BatchSampler(SubsetRandomSampler(range(state.shape[0])), batch_size, True):
                batch_s = state[index]
                batch_tarV = targetV[index]
                batch_curV = self.get_value(batch_s)
                loss = mse_loss(batch_curV-batch_tarV).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 10)
                self.optimizer.step()
                record_loss.append(loss.item())
        return np.mean(record_loss)

    def save_param(self,dir):
        save_dict={
            'net':self.state_dict(),
            'optimizer':self.optimizer.state_dict()
        }
        torch.save(save_dict,dir+'/'+'MDP.pkl')

    def load_param(self,dir):
        with open(dir,'rb') as f:
            MDP_param=pickle.load(f)
        self.value_state= MDP_param['value']
        self.n_state = MDP_param['num']


class PPO:
    """ build value network
    """
    def __init__(self,
                 env,
                 args,
                 device):
        self.set_seed(0)
        self.args = args
        # param for env
        self.agent_num=args.grid_num
        self.TIME_LEN=args.TIME_LEN
        self.use_mdp=args.use_mdp
        self.use_order_time = args.use_order_time
        self.order_grid=args.order_grid
        self.new_order_entropy=args.new_order_entropy
        self.remove_fake_order=args.remove_fake_order
        self.use_state_diff = args.use_state_diff

        self.state_dim=env.get_state_space_node()+1
        self.order_dim=6

        if self.use_state_diff:
            self.state_diff_dim = self.state_dim-2  # 去除时间、id
            self.state_dim += self.state_diff_dim
        if self.use_mdp>0:
            self.order_dim+=1
        if self.new_order_entropy:
            self.order_dim+=2
        if self.use_order_time:
            self.order_dim+=1

        if args.grid_num==100:
            self.max_order_num=60
        elif args.grid_num==121:
            self.max_order_num=100
        elif args.grid_num == 143:
            self.max_order_num = 100
        else:
            self.max_order_num = 100
        max_order_env = os.environ.get("COOPRIDE_MAX_ORDER_NUM")
        if max_order_env:
            try:
                self.max_order_num = int(max_order_env)
            except ValueError:
                pass
        self.driver_num = args.driver_num
        self.action_dim=self.max_order_num

        self.hidden_dim=128
        ratio_env = os.environ.get("COOPRIDE_HIGH_VALUE_RATIO")
        self.high_value_ratio = 0.0
        if ratio_env:
            try:
                self.high_value_ratio = float(ratio_env)
            except ValueError:
                self.high_value_ratio = 0.0
        self.high_value_ratio = max(0.0, min(1.0, self.high_value_ratio))
        min_orr_env = os.environ.get("COOPRIDE_MIN_ORR")
        self.min_orr_threshold = None
        if min_orr_env:
            try:
                self.min_orr_threshold = float(min_orr_env)
            except ValueError:
                self.min_orr_threshold = None

        # param for hyperparameter
        self.total_steps=args.MAX_ITER
        self.memory_size=args.memory_size
        self.batch_size=args.batch_size
        self.actor_lr=args.actor_lr
        self.critic_lr=args.critic_lr
        self.meta_lr = args.meta_lr
        self.train_actor_iters = args.train_actor_iters
        self.train_critic_iters = args.train_critic_iters
        self.train_phi_iters = args.train_phi_iters
        self.batch_size= int(args.batch_size)
        self.gamma= args.gamma
        self.lam= args.lam
        self.max_grad_norm = args.max_grad_norm
        self.clip_ratio = args.clip_ratio
        self.ent_factor=args.ent_factor
        self.adv_normal = args.adv_normal
        self.clip = args.clip
        self.grad_multi=args.grad_multi   # sum or mean
        self.minibatch_num=args.minibatch_num
        self.parallel_episode= args.parallel_episode
        self.parallel_way = args.parallel_way
        self.parallel_queue=args.parallel_queue

        self.use_orthogonal=args.use_orthogonal
        self.use_value_clip=args.use_value_clip
        self.use_valuenorm=args.use_valuenorm
        self.use_huberloss=args.use_huberloss
        self.huber_delta=10
        self.use_lr_anneal = args.use_lr_anneal
        self.use_GAEreturn = args.use_GAEreturn
        self.use_rnn = args.use_rnn
        self.use_GAT = args.use_GAT
        self.use_GCN = args.use_GCN
        self.use_DGCN = args.use_DGCN
        self.use_dropout=args.use_dropout
        self.use_neighbor_state = args.use_neighbor_state
        self.adj_rank = args.adj_rank
        self.merge_method = args.merge_method
        self.use_auxi = args.use_auxi
        self.auxi_loss = args.auxi_loss
        self.auxi_effi = args.auxi_effi
        self.use_regularize = args.use_regularize
        self.regularize_alpha = args.regularize_alpha
        self.use_fake_auxi = args.use_fake_auxi
        self.activate_fun = args.activate_fun
        self.feature_normal = args.feature_normal
        self.rm_state = args.rm_state
        self.global_emb = args.global_emb
        self.llm_dim = int(getattr(env, "global_instruction_dim", 0) or 0)
        try:
            self.llm_proj_dim = int(os.environ.get("COOPRIDE_LLM_PROJ_DIM", "64"))
        except ValueError:
            self.llm_proj_dim = 64
        if self.use_order_time:
            order_emb_choose=2
            order_contin_dim = self.order_dim-3
        else:
            order_emb_choose=1
            order_contin_dim = self.order_dim-2
        self.state_emb_choose = args.state_emb_choose
        self.state_remove = args.state_remove

        self.actor_decen = not args.actor_centralize
        self.critic_decen = not args.critic_centralize

        if self.use_GAT:
            self.adj_rank=1
        if self.use_GCN:
            self.adj_rank=args.adj_rank

        self.device=device
        self.env = env
        self.adj=self.compute_neighbor_tensor()
        if self.use_GCN:
            self.adj = self.normalize_adj(self.compute_edge_tensor())
        if self.use_DGCN:
            self.adj = self.compute_edge_tensor()

        self.meta_scope = args.meta_scope
        self.meta_choose = args.meta_choose
        if self.meta_choose==0:
            self.meta_scope=0

        # optimizers
        if self.use_mdp==0:
            self.MDP = None
        elif self.use_mdp==1:
            self.MDP = MdpAgent()
        elif self.use_mdp==2:
            self.MDP = DeepMdpAgent(self.agent_num, self.TIME_LEN, embedding_dim=128, driver_num=self.driver_num ,init=self.use_orthogonal, activate_fun=self.activate_fun)

        self.actor = Actor(self.agent_num, self.TIME_LEN, 128, self.state_dim-2, 
                            order_contin_dim ,
                            self.use_orthogonal, 
                            self.use_rnn, 
                            self.use_GAT and (not self.actor_decen), 
                            self.use_GCN and (not self.actor_decen), 
                            self.use_DGCN and (not self.actor_decen), 
                            self.adj,
                            self.use_neighbor_state and (not self.actor_decen), 
                            self.merge_method , 
                            self.use_auxi or self.use_fake_auxi>0 ,
                            activate_fun = self.activate_fun,
                            global_emb= self.global_emb,
                            order_emb_choose=order_emb_choose,
                            state_emb_choose=self.state_emb_choose,
                            state_remove = self.state_remove,
                            llm_dim=self.llm_dim,
                            llm_proj_dim=self.llm_proj_dim) 
        if self.meta_choose >0:
            self.actor_old = Actor(self.agent_num, self.TIME_LEN, 128, self.state_dim-2, 
                            order_contin_dim ,
                            self.use_orthogonal, 
                            self.use_rnn, 
                            self.use_GAT and (not self.actor_decen), 
                            self.use_GCN and (not self.actor_decen), 
                            self.use_DGCN and (not self.actor_decen), 
                            self.adj,
                            self.use_neighbor_state and (not self.actor_decen), 
                            self.merge_method , 
                            self.use_auxi or self.use_fake_auxi>0 ,
                            activate_fun = self.activate_fun,
                            global_emb= self.global_emb,
                            order_emb_choose=order_emb_choose,
                            state_emb_choose=self.state_emb_choose,
                            state_remove = self.state_remove,
                            llm_dim=self.llm_dim,
                            llm_proj_dim=self.llm_proj_dim) 
            self.actor_new = Actor(self.agent_num, self.TIME_LEN, 128, self.state_dim-2, 
                            order_contin_dim ,
                            self.use_orthogonal, 
                            self.use_rnn, 
                            self.use_GAT and (not self.actor_decen), 
                            self.use_GCN and (not self.actor_decen), 
                            self.use_DGCN and (not self.actor_decen), 
                            self.adj,
                            self.use_neighbor_state and (not self.actor_decen), 
                            self.merge_method , 
                            self.use_auxi or self.use_fake_auxi>0 ,
                            activate_fun = self.activate_fun,
                            global_emb= self.global_emb,
                            order_emb_choose=order_emb_choose,
                            state_emb_choose=self.state_emb_choose,
                            state_remove = self.state_remove,
                            llm_dim=self.llm_dim,
                            llm_proj_dim=self.llm_proj_dim) 
        self.critic = Critic(self.agent_num, self.TIME_LEN, 128, self.state_dim-2, order_contin_dim,self.use_orthogonal, 
                            self.use_rnn, 
                            self.use_GAT and (not self.critic_decen), 
                            self.use_GCN and (not self.critic_decen), 
                            self.use_DGCN and (not self.critic_decen), 
                            self.adj,
                            self.use_neighbor_state and (not self.critic_decen), 
                            self.merge_method , 
                            self.use_auxi or self.use_fake_auxi>0,
                            activate_fun = self.activate_fun,
                            state_emb_choose=self.state_emb_choose,
                            meta_scope=self.meta_scope,
                            meta_choose=self.meta_choose,
                            state_remove = self.state_remove,
                            llm_dim=self.llm_dim,
                            llm_proj_dim=self.llm_proj_dim)

        # Set up optimizers for policy and value function
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)

        self.critic_local_param = list(self.critic.state_layer.parameters())+ list(self.critic.local_value_layer.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic_local_param, lr=self.critic_lr, eps=1e-5)

        if self.meta_choose>0:
            self.meta_param = list(self.critic.Phi_layer.parameters())
            self.critic_global_param = list(self.critic.global_value_layer.parameters())+ list(self.critic.global_fc_layer.parameters())
            self.meta_optimizer = torch.optim.Adam(self.meta_param, lr=self.meta_lr)
            self.critic_global_optimizer = torch.optim.Adam(self.critic_global_param, lr=self.critic_lr, eps=1e-5)
            self.update_policy(self.actor_old,self.actor)
            self.update_policy(self.actor_new,self.actor)
        else:
            self.meta_optimizer = None
            self.critic_global_optimizer = None

        if self.use_valuenorm:
            self.value_local_normalizer = ValueNorm(self.meta_scope+1).to(self.device)
            self.value_global_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_local_normalizer = None
            self.value_global_normalizer = None

        self.buffer= Replay_buffer(args.memory_size, self.state_dim, self.order_dim ,self.action_dim, self.hidden_dim,
                            self.max_order_num, self.agent_num, self.gamma, self.lam,
                            adv_normal = self.adv_normal, 
                            parallel_queue=self.parallel_queue, 
                            value_local_normalizer=self.value_local_normalizer, 
                            value_global_normalizer=self.value_global_normalizer, 
                            use_GAEreturn=self.use_GAEreturn ,
                            actor_decen=self.actor_decen, 
                            critic_decen=self.critic_decen, 
                            local_value_dim=self.meta_scope+1,
                            adj = self.env.neighbor_dis,
                            args = args)

        self.neighbor_num=None

        if self.feature_normal==3:
            self.feature_scope = self.load_feature_scope()

        self.step = 0

        print('PPO init')

    def set_seed(self,seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def move_device(self,device):
        self.actor=self.actor.to(device)
        self.critic=self.critic.to(device)
        if self.use_mdp==2:
            self.MDP = self.MDP.to(device)
        if self.meta_choose>0:
            self.actor_old = self.actor_old.to(device)
            self.actor_new = self.actor_new.to(device)

    def save_param(self, save_dir,save_name='param'):
        state = {
            'step': self.step,
            'actor net':self.actor.state_dict(),
            'critic net':self.critic.state_dict() ,
            'actor optimizer':self.actor_optimizer.state_dict(),
            'critic optimizer': self.critic_optimizer.state_dict()
            }
        torch.save(state,save_dir+'/'+save_name+'.pkl')

    def load_param(self,load_dir,resume=False):
        state=torch.load(load_dir)
        # try:
        #     state['actor net'] = change_weight(state['actor net'])
        # except Exception as e:
        #     print(f"Error in changing weight: {e}")

        self.actor.load_state_dict(state['actor net'])
        self.critic.load_state_dict(state['critic net'])

    def update_policy(self,net_old,net_new):
        net_old.load_state_dict(net_new.state_dict())

    def load_feature_scope(self,dir='../save'):
        dir = dir+'/feature_{}.pkl'.format(self.agent_num)
        with open(dir, 'rb') as f:
            feature = pickle.load(f)
        state = torch.max(torch.abs(feature['state']),dim=0)[0]
        state[state==0]=5
        order = torch.max(torch.abs(feature['order']),dim=0)[0]
        order[order==0]=5
        feature_scope = {
            'state':state,
            'order':order
        }
        return feature_scope

    def check_grad(self,net):
        for name, parms in net.named_parameters(): 
            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                ' -->grad_value:',parms.grad)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = adj.sum(dim=-1, keepdim=True) # D
        d_inv_sqrt = torch.pow(rowsum, -0.5) # D^-0.5
        d_inv_sqrt[rowsum==0] = 0.
        normalize_adj = d_inv_sqrt*adj*(d_inv_sqrt.T)
        return normalize_adj

    def compute_edge_tensor(self):
        if self.env.order_param is not None:
            adj = torch.from_numpy( self.env.order_param.sum(-1) ).float()
            adj[adj<30]=0
        else:
            neighbors = [node.layers_neighbors_id for node in self.env.nodes]
            adj= torch.zeros((self.agent_num,self.agent_num),dtype=torch.float32)
            for i in range(self.agent_num):
                # adj[i,i]=1
                for k in range(self.env.l_max):
                    index= adj[i]>0
                    adj[i][index] = adj[i][index]+self.env.order_time_dist[k]
                    adj[i,neighbors[i][k]]=self.env.order_time_dist[k]
            adj = adj/torch.sum(adj,dim=-1,keepdim=True)
            order_num = torch.zeros(self.agent_num, dtype=torch.float32)
            for order_dist in self.env.order_num_dist:
                for i in range(self.agent_num):
                    order_num[i]+=order_dist[i][0]
            order_num/=len(self.env.order_num_dist)
            adj = adj*order_num[:,None]
            adj[adj<0.05]=0
        return adj.to(self.device)

    def compute_neighbor_tensor(self):
        neighbors = [node.layers_neighbors_id for node in self.env.nodes]
        adj= torch.zeros((self.agent_num,self.agent_num),dtype=torch.float32)
        for i in range(self.agent_num):
            adj[i,i]=1
            for k in range(self.adj_rank):
                adj[i,neighbors[i][k]]=1
        '''
        if self.phi_method==0:
            neighbor_tensor=torch.zeros((self.coop_scope+1,self.agent_num,self.agent_num),dtype=torch.float)
            for i in range(self.agent_num):
                neighbor_tensor[0,i,i]=1
                for rank in range(self.coop_scope):
                    neighbor_tensor[rank+1,i,neighbors[i][rank]]=1
            neighbor_tensor=neighbor_tensor.to(self.device)
            neighbor_tensor/= torch.sum(neighbor_tensor,dim=2,keepdim=True)
            neighbor_tensor=neighbor_tensor[:,:,:,None].transpose(0,3).squeeze(0)[:,:,None,:]
        elif self.phi_method==1:
            neighbor_tensor=torch.zeros((self.agent_num,self.agent_num),dtype=torch.float)
            for i in range(self.agent_num):
                neighbor_tensor[i,i]=1
                for rank in range(self.coop_scope):
                    neighbor_tensor[i,neighbors[i][rank]]=1
            neighbor_tensor=neighbor_tensor.to(self.device)
            neighbor_tensor/=torch.sum(neighbor_tensor,dim=1,keepdim=True)
        '''
        return adj.to(self.device)

    def normalize_feature(self,feature):
        feature_min= torch.min(feature,dim=0, keepdims=True)[0]
        feature-=feature_min
        feature_max = torch.max(feature,dim=0, keepdims=True)[0]
        feature_max[feature_max==0]=1
        feature/= feature_max
        feature = (feature-0.5)*2
        return feature

    def process_state(self,s, t):
        s=np.stack(s,axis=0)
        if self.args.log_feature:
            self.logs.record_state_minmax(s)
        if self.feature_normal==1:
            feature_max= np.max(s[:,1:],axis=0, keepdims=True)
            feature_max[feature_max==0]=1
            s[:,1:]/=feature_max
        elif self.feature_normal==2:
            norm_s = torch.from_numpy(s[:,1:])
            norm_s = torch.clamp(norm_s,-20,20)
            norm_s = self.normalize_feature(norm_s)
            s[:,1:] = norm_s.numpy()
        elif self.feature_normal==3:
            state_scope = self.feature_scope['state'].numpy()
            feature_dim = s.shape[1] - 1
            scope_dim = state_scope.shape[0] - 1
            if feature_dim > scope_dim:
                pad = np.ones((feature_dim - scope_dim,), dtype=state_scope.dtype)
                state_scope = np.concatenate([state_scope, pad], axis=0)
            s[:,1:] /= state_scope[None,1:]
        if self.use_state_diff:
            if t==0:
                self.record_last_state=np.zeros((s.shape[0], self.state_diff_dim))
            else:
                self.record_last_state = s[:,1:1+self.state_diff_dim]-self.record_last_state
            s = np.concatenate([s,self.record_last_state], axis=-1)
            self.record_last_state = s[:,1:1+self.state_diff_dim]

        if 'fea' in self.rm_state:
            s[:,1:] = 0
        if 'time' in self.rm_state:
            t=0
        if 'id' in self.rm_state:
            s[:,0] = 0
        '''
        onehot_grid_id = np.eye(self.agent_num)
        if self.state_time==0:
            state= np.concatenate([onehot_grid_id,s[:,1:]],axis=1)
        elif self.state_time==1:
            time = np.zeros((s.shape[0],1),dtype=float)
            time[:,0]= t/self.TIME_LEN
            state= np.concatenate([onehot_grid_id, time ,s[:,1:]],axis=1)
        elif self.state_time==2:
            time= np.zeros((s.shape[0],self.TIME_LEN))
            time[:, int(t)]=1
            state= np.concatenate([onehot_grid_id, time ,s[:,1:]],axis=1)
        '''
        time = np.zeros((s.shape[0],1),dtype=float)
        time[:,0]=t
        state=np.concatenate([ time ,s],axis=1)
        return torch.Tensor(state)

    def add_order_value(self,order_state):
        if self.use_mdp==0:
            return order_state
        for i in range(len(order_state)):
            for j,o in enumerate(order_state[i]):
                o+= [self.MDP.get_value(o)]
        return order_state

    def remove_order_grid(self,order):
        if self.order_grid:
            return order
        else:
            order[:,:,:2]=0
            return order

    def mask_fake(self,order,mask):
        if self.remove_fake_order==False:
            return mask
        else:
            return mask & (order[:,:,4]<0)

    def add_new_entropy(self,env,order):
        driver_num= torch.Tensor([node.idle_driver_num for node in env.nodes])+1e-5
        order_num= torch.Tensor([node.real_order_num for node in env.nodes])+1e-5
        driver_order=torch.stack([driver_num,order_num],dim=1)
        ORR_entropy= torch.min(driver_order,dim=1)[0]/torch.max(driver_order,dim=1)[0]
        node=order[:,:,:2].long()
        entropy_feature= ORR_entropy[node[:,:,1]]-ORR_entropy[node[:,:,0]]
        driver_num_feature = driver_num[node[:,:,1]]-driver_num[node[:,:,0]]
        order_num_feature = order_num[node[:,:,1]]-order_num[node[:,:,0]]
        order[:,:,5]=entropy_feature
        order= torch.cat([order,driver_num_feature[:,:,None],order_num_feature[:,:,None]],-1)
        return order

    def add_mdp(self, order_state,  t):
        t = t%self.TIME_LEN
        if self.use_mdp==0:
            return order_state
        # order_state.shape = (order num, 7)    7 is 特征维度 begin node, end node, price, duration ,service type, entropy]
        mdp_state = torch.zeros((order_state.shape[0],4), dtype=torch.long)  # [t0,s0,t1,s1]
        mdp_state[:,0] = t
        mdp_state[:,2] = t+order_state[:,3]
        mdp_state[:,1] = order_state[:,0]
        mdp_state[:,3] = order_state[:,1]
        mdp_state[:,2] = torch.clamp(mdp_state[:,2],0,self.TIME_LEN-1)
        if self.use_mdp==2:
            with torch.no_grad():
                cur_value = self.MDP.get_value(mdp_state[:,0:2].to(self.device)).cpu().squeeze(1)
                next_value = self.MDP.get_value(mdp_state[:,2:4].to(self.device)).cpu().squeeze(1)
            target_value = next_value*torch.pow(self.gamma,mdp_state[:,2]-mdp_state[:,0])
            target_value = target_value+order_state[:,2]
            adv = target_value-cur_value
            self.MDP.memory_info(mdp_state[:,:2], target_value)
            order_state[:,-1] = adv
            return order_state

    def process_order(self,order_state, t):
        # [begin node, end node, price, duration ,service type, entropy]
        order_num = []
        trimmed_state = []
        for i in range(len(order_state)):
            cur_state = order_state[i]
            if len(cur_state) > self.max_order_num:
                cur_state = cur_state[:self.max_order_num]
            trimmed_state.append(cur_state)
            order_num.append(len(cur_state))
        # order_dim_origin= self.order_dim-2 if self.new_order_entropy else self.order_dim
        if self.use_mdp:
            order_dim_origin = 7
        else:
            order_dim_origin = 6
        order=torch.zeros((self.agent_num,self.max_order_num,order_dim_origin),dtype=float32)
        mask=torch.zeros((self.agent_num,self.max_order_num),dtype=torch.bool)
        # if self.use_mdp:
        #    order_dim_state = order_dim_origin-1
        # else:
        #    order_dim_state = order_dim_origin
        for i in range(len(trimmed_state)):
            order[i,:order_num[i], :7]= torch.Tensor(trimmed_state[i])
            mask[i,:order_num[i]]=1
        if self.use_mdp>0:
            order[mask] = self.add_mdp(order[mask], t)
        if self.new_order_entropy:
            order=self.add_new_entropy(self.env,order)
        if self.use_order_time:
            time_fea = torch.ones((order.shape[0],order.shape[1],1))*t
            order = torch.cat([time_fea,order],dim=-1)
        contin_dim = 2 if not self.use_order_time else 3 
        if self.args.log_feature:
            self.logs.record_order_minmax(order[mask])
        if self.feature_normal==1:
            order[:,:,contin_dim:]=torch.clamp(order[:,:,contin_dim:],-10,10)
            feature_scale= torch.max(torch.abs(order[:,:,contin_dim:]))
            feature_scale[feature_scale==0]=1
            order[:,:,contin_dim:]/=feature_scale
        elif self.feature_normal==2:
            norm_order=torch.clamp(order[:,:,contin_dim:],-20,20)
            norm_order = self.normalize_feature(norm_order.view(-1,norm_order.shape[-1]))
            order[:,:,contin_dim:] = norm_order.view(order.shape[0],order.shape[1], norm_order.shape[-1])
        elif self.feature_normal==3:
            order_scope = self.feature_scope['order']
            order[:,:,contin_dim:]/= order_scope[None,None,contin_dim:]
        return order,mask

    def action_process(self,action):
        low = self.action_space_low
        high = self.action_space_high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action

    def action(
        self,
        state,
        order,
        state_rnn_actor,
        state_rnn_critic,
        mask,
        order_idx,
        device="cpu",
        random_action=False,
        sample=True,
        MDP=None,
        need_full_prob=False,
        FM_mode="local",
    ):
        """
        Compute current action for all grids give states
        
        ============ state 字段说明 ============
        state: 网格状态特征，shape = (grid_num, state_dim)
        state_dim = 5 + 3*l_max (+ llm_dim)
        
        字段索引说明:
        - state[..., 0]: 时间步 (time step)
        - state[..., 1]: 网格ID (grid id)
        - state[..., 2]: 空闲司机数量 (idle driver num)
        - state[..., 3]: 真实订单数量 (real order num)
        - state[..., 4]: 本地熵值 (local entropy = idle_driver_num / (real_order_num + idle_driver_num))
        - state[..., 5]: 相对全局熵的绝对差 (abs(local_entropy - global_entropy))
        - state[..., 6:6+l_max]: 价格分布 (price distribution, l_max个价格区间的概率分布)
        - state[..., 6+l_max:6+2*l_max]: 时间分布 (time distribution, l_max个时长的概率分布)
        - state[..., 6+2*l_max:6+3*l_max]: 目的地层级分布 (end layer distribution, l_max个目的地层级的概率分布)
        - state[..., 6+3*l_max:]: 全局指令向量 (global instruction vector, 可选, 维度由llm_dim决定)
        
        ============ order 字段说明 ============
        order: 订单特征，shape = (grid_num, max_order_num, order_dim)
        order_dim = 6 或 7 (取决于use_order_time和use_mdp参数)
        
        字段索引说明 (use_order_time=False时):
        - order[..., 0]: 起点网格ID (origin grid id)
        - order[..., 1]: 终点网格ID (destination grid id)
        - order[..., 2]: 订单价格 (order price)
        - order[..., 3]: 订单时长 (order duration, 单位: 时间步)
        - order[..., 4]: 服务类型 (service type):
            * -1: 本地真实订单
            * 0: 车队管理订单
            * 1: 相邻网格订单
            * <0: 假订单/留在本地选项
        - order[..., 5]: 熵值差异 (entropy difference = end_entropy - start_entropy, 如果new_order_entropy=True)
        - order[..., 6]: 司机数量差异 (driver num diff = end_driver_num - start_driver_num, 如果new_order_entropy=True)
        - order[..., 7]: 订单数量差异 (order num diff = end_order_num - start_order_num, 如果new_order_entropy=True)
        - order[..., -1]: MDP优势值 (MDP advantage value, 如果use_mdp>0)
        
        字段索引说明 (use_order_time=True时):
        - order[..., 0]: 时间步 (time step)
        - order[..., 1]: 起点网格ID (origin grid id)
        - order[..., 2]: 终点网格ID (destination grid id)
        - order[..., 3]: 订单价格 (order price)
        - order[..., 4]: 订单时长 (order duration)
        - order[..., 5]: 服务类型 (service type)
        - order[..., 6:]: 其他特征 (同上，索引相应后移)
        
        ============ 参数说明 ============
        :param state: grid_num x state_dim, 网格状态特征
        :param order: grid_num x max_order_num x order_dim, 订单特征
        :param mask: grid_num x max_order_num, 订单有效掩码 (True表示有效订单)
        :return:
        """

        # ========== 初始化阶段 ==========
        mask=mask.bool()  # 确保mask是bool类型

        # 随机动作模式（主要用于测试或探索）
        if random_action:
            # 生成随机动作，范围在[-1, 1]之间均匀分布
            action=torch.randn(state.shape[0],self.action_dim).uniform_(-1,1)
        else:
            # ========== 网络前向推理 ==========
            state=state.to(device)  # 将状态移到目标设备
            with torch.no_grad():  # 不计算梯度，节省内存
                    # Actor网络：计算策略logits和更新后的RNN隐藏状态
                    # return_logit=True: 返回未经过softmax的logits
                    logits, state_rnn_actor = self.actor(state, order.to(device), mask.to(device), self.adj ,state_rnn_actor.to(device), return_logit = True)
                # Critic网络：计算本地价值函数
                value_local, state_rnn_critic = self.critic(state, self.adj , state_rnn_critic.to(device))
                # Critic网络：计算全局价值函数
                value_global, state_rnn_critic = self.critic.get_global_value(state, self.adj , state_rnn_critic.to(device))

        # ========== 将结果移到CPU ==========
        logits=logits.cpu()  # 策略logits移到CPU
        # logits=torch.exp(logits)
        value_local=value_local.cpu()  # 本地价值移到CPU
        value_global = value_global.cpu()  # 全局价值移到CPU

        # ========== 初始化动作和相关变量 ==========
        action=torch.zeros((self.agent_num,self.max_order_num),dtype=torch.float32)  # 动作向量
        mask_order= torch.zeros((mask.shape[0],mask.shape[1],mask.shape[1]),dtype=torch.bool)  # 订单掩码（三维）
        mask_action = torch.zeros((self.agent_num,self.max_order_num),dtype=torch.bool)  # 动作掩码（哪些司机有有效动作）
        mask_entropy = torch.zeros((self.agent_num,self.max_order_num),dtype=torch.bool)  # 熵计算掩码
        driver_record=torch.zeros((self.agent_num,),dtype=torch.long)  # 记录每个网格实际分配的司机数量
        oldp = torch.zeros((self.agent_num,self.max_order_num),dtype=torch.float32)  # 旧概率（用于PPO的ratio计算）
        mask_agent=torch.ones((self.agent_num,),dtype=torch.bool)  # 智能体掩码（哪些智能体参与训练）
        action_ids=[]  # 实际选择的订单ID列表
        selected_idx=[]  # 选择索引列表（用于MDP学习）

        # ========== 订单响应率（ORR）自适应调整 ==========
        high_ratio = self.high_value_ratio  # 初始化高价值比例
        # 如果设置了最小ORR阈值，检查当前ORR是否低于阈值
        if self.min_orr_threshold is not None:
            current_orr = getattr(self.env, "order_response_rate", None)  # 获取当前订单响应率
            # 如果当前ORR低于阈值，禁用高价值优先策略，优先保证订单响应率
            if current_orr is not None and current_orr >= 0 and current_orr < self.min_orr_threshold:
                high_ratio = 0.0
        # sample orders
        # ========== FM_mode 说明 ==========
        # FM_mode 决定司机分配策略，主要有三种模式：
        # 1. 'local': 基础本地模式，司机从所有可用订单中采样（包括假订单）
        # 2. 'RLmerge': 高价值优先混合模式
        #    - 根据high_ratio比例，将部分司机分配给高价值真实订单
        #    - 剩余司机从所有订单池中选择（包括假订单和车队订单）
        #    - 允许灵活调度，但优先保证高价值订单
        # 3. 'RLsplit': 硬分段模式
        #    - 前real_num个司机只能从真实订单中选择
        #    - 剩余司机只能从车队订单和假订单中选择
        #    - 严格分离订单类型，无交叉
        # ====================================
        if FM_mode=='local':
            # ========== 本地模式：司机从所有可用订单中采样 ==========
            # 策略：每个司机独立地从订单池中选择一个订单（包括假订单）
            # 选择后订单从池中移除，保证不重复选择
            # ==============================================
            for i in range(state.shape[0]):
                max_driver_num= self.max_order_num  # 每个网格的最大司机数量限制
                driver_num = self.env.nodes[i].idle_driver_num  # 当前网格的空闲司机数量
                driver_num = min(driver_num,max_driver_num)  # 不超过最大限制
                driver_record[i]=driver_num  # 记录实际分配的司机数量
                # fake_num= self.env.nodes[i].fleet_order_num+1       # 假订单数量（车队订单+1个留本地选项）
                # real_num= self.env.nodes[i].real_order_num  # 真实订单数量

                if driver_num==0 or len(order_idx[i])==1:  # 没有司机或只有假订单
                    choose=[0]  # 选择留本地（索引0）
                    mask_agent[i]=0  # 该智能体不参与训练
                else:
                    choose=[]  # 存储选择的订单索引
                    logit=logits[i][mask[i]].clone()  # 获取有效订单的logit
                    prob = F.softmax(logit, dim=-1)  # 计算概率分布
                    mask_d= mask[i].clone()  # 动态掩码，用于标记已选择的订单
                    mask_entropy[i,0]=1  # 标记需要计算熵的订单（假订单）

                    # 为每个司机选择订单
                    for d in range(driver_num):
                        mask_order[i,d]=mask_d  # 记录当前司机可选订单的掩码

                        # 根据sample参数决定采样方式
                        if sample:
                            # 采样模式：根据概率分布随机采样
                            choose.append(torch.multinomial(prob,1 , replacement=True))
                        else:
                            # 贪婪模式：选择概率最大的订单
                            choose.append(torch.argmax(prob))

                        mask_action[i,d]=1  # 标记该司机有有效动作
                        oldp[i,d]=prob[choose[-1]]  # 记录旧概率，用于PPO更新

                        # 如果选择的不是假订单（索引>0），则从候选池中移除
                        if choose[-1]>0:
                            mask_d[choose[-1]]=0  # 将该订单设为不可选
                            logit[choose[-1]]= -math.inf  # 将logit设为负无穷
                            prob = F.softmax(logit, dim=-1)  # 重新计算概率分布

                        # 如果假订单概率为1，说明没有其他可选订单，提前结束
                        if prob[0]==1 :
                            break

                action[i,:len(choose)] = torch.Tensor(choose)  # 设置动作向量
                action_ids.append([ order_idx[i][idx]  for idx in choose])  # 记录实际选择的订单ID

                # 构建选择索引（用于MDP学习）
                select=[]   # 给MDP学的选择索引
                if driver_num==0:
                    select=[]
                elif len(order_idx[i])==1:
                    select=[0]*driver_num  # 所有司机都选择留本地
                else:
                    # 前len(choose)个司机选择choose中的订单，剩余司机选择留本地
                    select = choose+ ([0]*(driver_num-len(choose)))
                selected_idx.append(select)
        elif FM_mode=='RLmerge':
            # ========== RLmerge模式：高价值优先混合 ==========
            # 策略：
            # 1. 将司机分为两组：
            #    - high_quota个司机：强制从高价值真实订单池中选择
            #    - 剩余司机：从全部订单池选择（包括假订单）
            # 2. 高价值订单池：按价格排序后的前high_count个真实订单
            # 3. 优势：保证高价值订单优先被服务，同时保留灵活性
            # ==============================================

            for i in range(state.shape[0]):
                max_driver_num= self.max_order_num
                driver_num = self.env.nodes[i].idle_driver_num
                driver_num = min(driver_num,max_driver_num)
                driver_record[i]=driver_num
                fake_num= self.env.nodes[i].fleet_order_num+1       # the last  fake order is stay local
                real_num= self.env.nodes[i].real_order_num

                if driver_num==0 :
                    choose=[0]
                    mask_agent[i]=0
                else:
                    choose=[]
                    mask_d= mask[i].clone()
                    mask_entropy[i,0]=1
                    logit_full = logits[i].clone()
                    logit_full[~mask_d] = -math.inf
                    high_indices = None
                    high_quota = 0
                    if high_ratio > 0 and len(order_idx[i]) > fake_num:
                        real_indices = list(range(fake_num, len(order_idx[i])))
                        if real_indices:
                            high_count = max(1, int(math.ceil(len(real_indices) * high_ratio)))
                            high_indices = set(real_indices[:high_count])
                            high_quota = max(1, int(math.ceil(driver_num * high_ratio)))
                            high_quota = min(high_quota, len(real_indices), driver_num)
                    for d in range(driver_num):
                        mask_order[i,d]=mask_d
                        use_high = high_indices is not None and d < high_quota
                        candidate_logit = logit_full
                        if use_high:
                            candidate_logit = logit_full.clone()
                            high_mask = torch.zeros_like(mask_d)
                            high_mask[list(high_indices)] = True
                            candidate_logit[~high_mask] = -math.inf
                            if torch.isneginf(candidate_logit).all():
                                candidate_logit = logit_full
                        prob = F.softmax(candidate_logit, dim=-1)
                        if sample:
                            chosen_idx = int(torch.multinomial(prob,1 , replacement=True).item())
                        else:
                            chosen_idx = int(torch.argmax(prob).item())
                        choose.append(chosen_idx)
                        mask_action[i,d]=1
                        oldp[i,d]=prob[chosen_idx]
                        if chosen_idx>= fake_num:
                            mask_d[chosen_idx]=0
                            logit_full[chosen_idx]= -math.inf
                action[i,:len(choose)] = torch.Tensor(choose)
                action_ids.append([ order_idx[i][idx]  for idx in choose])
                select=[]   # 给MDP学
                if driver_num==0:
                    select=[]
                else:
                    select = choose
                selected_idx.append(select)
        elif FM_mode=='RLsplit':
            # ========== RLsplit模式：硬分段分配 ==========
            # 策略：
            # 1. 将司机严格分为两组，无交叉：
            #    - 第一组（前real_num个司机）：只能从真实订单选择，不能选假订单
            #    - 第二组（剩余司机）：只能从车队订单+假订单选择，不能选真实订单
            # 2. 每个司机选择后，对应订单从池中移除（mask_d设为0）
            # 3. 特点：严格分离，保证一定数量的司机只服务真实订单
            # ==============================================
            for i in range(state.shape[0]):
                max_driver_num=self.max_order_num
                driver_num = self.env.nodes[i].idle_driver_num
                driver_num = min(driver_num,max_driver_num)
                driver_record[i]=driver_num
                fake_num= self.env.nodes[i].fleet_order_num+1       # the last  fake order is stay local
                real_num= self.env.nodes[i].real_order_num
                choose=[]
                if driver_num==0:
                    choose=[0]
                    mask_agent[i]=0
                choose=[]
                for d in range(driver_num):
                    if d<real_num:
                        if d==0:
                            logit=logits[i][mask[i]].clone()
                            logit[:fake_num]=-math.inf  # for real_order
                            prob = F.softmax(logit, dim=-1)
                            mask_d= mask[i].clone()
                            mask_d[:fake_num]=0     # for real_order
                            mask_entropy[i,d]=1
                        mask_order[i,d]=mask_d
                        choose.append(torch.multinomial(prob,1 , replacement=True))
                        mask_action[i,d]=1
                        oldp[i,d]=prob[choose[-1]]
                        mask_d[choose[-1]]=0
                        logit[choose[-1]]=-math.inf
                        prob = F.softmax(logit, dim=-1)
                    else:
                        mask_action[i,:d]=0     # driver num > real num
                        if d==real_num:
                            logit=logits[i][mask[i]].clone()
                            logit[fake_num:]=-math.inf  # for real_order
                            mask_d=mask[i].clone()
                            mask_d[fake_num:]=0
                            prob = F.softmax(logit, dim=-1)
                            mask_entropy[i,:d]=1
                        mask_order[i,d]=mask_d
                        choose.append(torch.multinomial(prob,1 , replacement=True))
                        mask_action[i,d]=1
                        oldp[i,d]=prob[choose[-1]]
                action[i,:len(choose)] = torch.Tensor(choose)
                action_ids.append([ order_idx[i][idx]  for idx in choose])
                select=[]   # 给MDP学
                if driver_num==0:
                    select=[]
                else:
                    select = choose+ ([0]*(driver_num-len(choose)))
                selected_idx.append(select)
        if need_full_prob:
            return action, value, oldp, mask_agent, mask_order, mask_action ,action_ids, probs, driver_record
        return action, value_local, value_global ,oldp, mask_agent, mask_order, mask_action, mask_entropy , state_rnn_actor.cpu(), state_rnn_critic.cpu() ,action_ids,selected_idx

    def soft_update_params(self,net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

    def compute_regularize(self,model):
        loss = 0   
        for name, param in model.named_parameters():
            flag = False
            if 'weight' in name:
                if 'state' in self.use_regularize:
                    if 'state' in name:
                        flag = True
                else:
                    flag = True
            if flag:
                if 'L1' in self.use_regularize:
                    loss += torch.abs(param).sum()
                elif 'L2' in self.use_regularize:
                    loss += (param**2).sum()/2
        return loss

    def add_L1_loss(self, model, l1_alpha):
        l1_loss = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_loss.append(torch.abs(param).sum())
        return l1_alpha*sum(l1_loss)

    def add_L2_loss(self, model, l1_alpha):
        l1_loss = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_loss.append((param**2).sum()/2)
        return l1_alpha*sum(l1_loss)

    def split_batch(self,index,data,device='cpu'):
        batch={}
        for key,value in data.items():
            if len(value.shape)==0:
                batch[key] = value
            else:
                batch[key]=value[index]
        return batch

    def meta_prepare(self,data):
        phi = torch.zeros(data['value_local'].shape, device=data['value_local'].device, dtype=torch.float32)
        # phi shape=(batch num, agent num, meta scope+1)
        length = data['state_critic'].shape[0]
        batch_size = int(self.batch_size/self.agent_num)
        batch_num =  int(np.ceil(length/batch_size))
        with torch.no_grad():
            for i in range(batch_num):
                state = data['state_critic'][batch_size*i: min(batch_size*(i+1), length)]
                phi[batch_size*i: min(batch_size*(i+1), length)] = self.critic.get_phi(state)
        if self.args.log_phi:
            self.logs.push_full_phi(phi.cpu().numpy())
        if self.neighbor_num is None:       # shape=(agent num, meta_scope+1)
            self.neighbor_num = torch.zeros((self.agent_num,self.meta_scope+1)).float()
            for i in range(self.agent_num):
                for k in range(self.meta_scope+1):
                    index = self.env.neighbor_dis[i]==k
                    self.neighbor_num[i,k] = np.sum(index)
            self.neighbor_num = self.neighbor_num.to(data['adv_local'].device).unsqueeze(0) #  shape=(1,agent num, meta_scope+1)
        if self.meta_choose==1:
            adv_coop = (data['adv_local']*phi*self.neighbor_num).sum(-1)
            adv_coop = adv_coop/ (phi*self.neighbor_num).sum(-1) # 归一化
        elif self.meta_choose==2:
            adv_coop = (data['adv_local']*phi*self.neighbor_num).sum(-1)
            adv_coop = adv_coop/ (self.neighbor_num).sum(-1) # 归一化
        elif self.meta_choose==3:
            adv_coop = (data['adv_local']*phi*self.neighbor_num).sum(-1)
            adv_coop = adv_coop/ 20 # 归一化
        elif self.meta_choose==4:
            adv_coop = (data['adv_local']*phi).sum(-1)
        elif self.meta_choose==5:
            adv_coop = (data['adv_local']*phi)[...,1:].sum(-1)
        elif self.meta_choose==6:
            adv_coop = (data['adv_local']*phi)[...,1:].sum(-1)
        elif self.meta_choose==7:
            adv_coop = (data['adv_local']*phi)[...,2:].sum(-1)
        data['adv_coop_mean'] = adv_coop.mean().detach()
        data['adv_coop_std'] = adv_coop.std().detach()
        data['adv_coop'] = normalize(adv_coop.unsqueeze(2), self.adv_normal)
        data['adv_global'] = normalize(data['adv_global'], self.adv_normal)
        data['phi'] = phi
        return data

    def update(self, device='cpu',writer=None):
        if self.use_lr_anneal:
            update_linear_schedule(self.actor_optimizer, self.step, self.total_steps, self.actor_lr)
            update_linear_schedule(self.critic_optimizer, self.step, self.total_steps, self.critic_lr)

        data=self.buffer.get(device)
        record_entropy=[]
        record_return_local=[]
        record_return_global=[]
        record_actor_loss_origin=[]
        record_critic_local_loss_origin=[]
        record_critic_global_loss_origin=[]
        record_phi_loss_origin=[]
        record_actor_auxi_loss=[]
        record_critic_auxi_loss=[]

        if self.meta_choose>0:
            data = self.meta_prepare(data)
            # 保存policy，用于计算meta gradient梯度
            self.update_policy(self.actor_old,self.actor)
        else:
            data['adv_coop'] = data['adv_local']
            data['adv_coop_mean'] = data['adv_coop'].mean().detach()
            data['adv_coop_std'] = data['adv_coop'].std().detach()
            data['adv_coop'] = normalize(data['adv_coop'], self.adv_normal)
            # if self.actor_decen:
            #    data['adv_coop'] = data['adv_coop'].reshape((-1,1))

        # Train policy with multiple steps of gradient descent
        data_actor={
            'state':data['state_actor'],
            'next_state':data['next_state_actor'],
            'order':data['order'],
            'action':data['action'],
            'advantage':data['adv_coop'],
            'oldp':data['oldp'],
            'mask_order':data['mask_order'],
            'mask_action':data['mask_action'],
            'mask_agent':data['mask_agent'],
            'mask_entropy':data['mask_entropy'],
            'state_rnn':data['state_rnn_actor']
        }
        data_size = data_actor['state'].shape[0]
        if self.parallel_way=='mean':
            data_size= int(data_size/self.parallel_episode)
        if not self.actor_decen:
            batch_size = int(self.batch_size/self.agent_num)
        else:
            batch_size = self.batch_size
        for iter in range(self.train_actor_iters):
            record_actor_loss=[]
            record_ratio_max=[]
            record_ratio_mean=[]
            record_KL=[]
            self.actor_optimizer.zero_grad()
            batch_num= int(np.round(data_size/batch_size/self.minibatch_num))
            cnt=0
            thread=1 if self.parallel_way=='mix' else self.parallel_episode
            for index in BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, True):
                record_KL=[]
                for _ in range(thread):
                    # self.actor_optimizer.zero_grad()
                    loss_actor, actor_info = self.compute_loss_actor(self.split_batch(index,data_actor))
                    kl = actor_info['kl']
                    loss_actor/=(batch_num*thread)
                    loss_actor.backward()
                    if iter==0:
                        record_entropy.append(actor_info['entropy'])
                        record_actor_loss_origin.append(loss_actor.item())
                    record_KL.append(actor_info['kl'])
                    record_actor_loss.append(loss_actor.item())
                    record_ratio_max.append(actor_info['ratio_max'])
                    record_ratio_mean.append(actor_info['ratio_mean'])
                    record_actor_auxi_loss.append( actor_info['auxi_loss'] )
                    # loss_iter+=loss_actor
                if (cnt+1)%batch_num==0:
                    if np.mean(record_KL)<-0.01:
                        self.actor_optimizer.zero_grad()   
                        # continue
                    else:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()
                        self.actor_optimizer.zero_grad()              
                cnt+=1
            # if np.mean(record_KL)<-0.01:
            # break

        # Value function learning
        data_critic_local={
            'state':data['state_critic'],
            'next_state':data['next_state_critic'],
            'ret':data['ret_local'],
            'value':data['value_local'],
            'state_rnn':data['state_rnn_critic']
        }
        data_size = data_critic_local['state'].shape[0]
        if self.parallel_way=='mean':
            data_size= int(data_size/self.parallel_episode)
        if not self.critic_decen:
            batch_size = int(self.batch_size/self.agent_num)
        else:
            batch_size = self.batch_size
        for iter in range(self.train_critic_iters):
            record_critic_local_loss=[]
            self.critic_optimizer.zero_grad()
            batch_num= int(np.round(data_size/batch_size/self.minibatch_num))
            cnt=0
            thread=1 if self.parallel_way=='mix' else self.parallel_episode
            for index in BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, True):
                for _ in range(thread):
                    loss_critic, critic_info = self.compute_loss_critic(self.split_batch(index,data_critic_local), self.critic.get_local_value , self.value_local_normalizer)
                    loss_critic/=(batch_num*thread)
                    loss_critic.backward()
                    if iter==0:
                        record_critic_local_loss_origin.append(loss_critic.item())
                        record_return_local.append(critic_info['ret'])
                    record_critic_local_loss.append(loss_critic.item())
                    record_critic_auxi_loss.append( critic_info['auxi_loss'] )
                # mpi_avg_grads(ac.v)    # average grads across MPI processes
                if (cnt+1)%batch_num==0:
                    # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, norm_type=2)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()
                cnt+=1

        if self.meta_choose>0:
            record_critic_global_loss_origin, record_return_global = self.update_global_critic(data)
            record_phi_loss_origin, record_phi_loss = self.update_phi(data)
            self.logs.save_log_phi(self.step, data['phi'].reshape(-1,self.meta_scope+1).mean(0))
        else:
            record_critic_global_loss_origin, record_return_global = 0,0
            record_phi_loss_origin, record_phi_loss = 0,0

        writer.add_scalar('train actor loss', np.mean(record_actor_loss_origin), global_step=self.step)
        writer.add_scalar('train critic local loss', np.mean(record_critic_local_loss_origin), global_step=self.step)
        writer.add_scalar('train critic global loss', np.mean(record_critic_global_loss_origin), global_step=self.step)
        writer.add_scalar('train phi loss', np.mean(record_phi_loss_origin), global_step=self.step)
        writer.add_scalar('train entropy', np.mean(record_entropy), global_step=self.step)
        writer.add_scalar('train kl', np.mean(record_KL), global_step=self.step)
        # writer.add_scalar('train delta actor loss', np.mean(record_actor_loss)-np.mean(record_actor_loss_origin), global_step=self.step)
        # writer.add_scalar('train delta critic loss', np.mean(record_critic_loss)-np.mean(record_critic_loss_origin), global_step=self.step)
        writer.add_scalar('train ratio max', np.mean(record_ratio_max) , global_step=self.step)
        writer.add_scalar('train ratio mean', np.mean(record_ratio_mean) , global_step=self.step)
        writer.add_scalar('train adv mean', data['adv_coop_mean'].mean() , global_step=self.step)
        writer.add_scalar('train adv std', data['adv_coop_std'].std() , global_step=self.step)
        writer.add_scalar('train return local', np.mean(record_return_local) , global_step=self.step)
        writer.add_scalar('train return global', np.mean(record_return_global) , global_step=self.step)
        writer.add_scalar('train actor auxi loss', np.mean(record_actor_auxi_loss) , global_step=self.step)
        writer.add_scalar('train critic auxi loss', np.mean(record_critic_auxi_loss) , global_step=self.step)
        self.step += 1

    def update_global_critic(self,data):
        # Global Value function learning
        record_critic_global_loss_origin=[]
        record_return_global = []
        data_critic_global={
            'state':data['state_critic'],       # [batch num, agent num, state dim]
            'next_state':data['next_state_critic'],
            'ret':data['ret_global'],           # [batch num, 1]
            'value':data['value_global'],       # [batch num, 1]
            'state_rnn':data['state_rnn_critic']    # [batch num, agent num, hidden dim]
        }
        data_size = data_critic_global['state'].shape[0]
        batch_size = int(self.batch_size/self.agent_num)
        for iter in range(self.train_critic_iters):
            record_critic_global_loss=[]
            self.critic_global_optimizer.zero_grad()
            batch_num= int(np.round(data_size/batch_size/self.minibatch_num))
            cnt=0
            thread=1 if self.parallel_way=='mix' else self.parallel_episode
            for index in BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, True):
                for _ in range(thread):
                    loss_critic, critic_info = self.compute_loss_critic(self.split_batch(index,data_critic_global),self.critic.get_global_value , self.value_global_normalizer)
                    loss_critic/=(batch_num*thread)
                    loss_critic.backward()
                    if iter==0:
                        record_critic_global_loss_origin.append(loss_critic.item())
                        record_return_global.append(critic_info['ret'])
                    record_critic_global_loss.append(loss_critic.item())
                    # record_critic_auxi_loss.append( critic_info['auxi_loss'] )
                # mpi_avg_grads(ac.v)    # average grads across MPI processes
                if (cnt+1)%batch_num==0:
                    # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, norm_type=2)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_global_optimizer.step()
                    self.critic_global_optimizer.zero_grad()
                cnt+=1
        return record_critic_global_loss_origin, record_return_global

    def update_phi(self,data):
        record_phi_loss_origin=[]
        # phi function learning
        data_phi={
            'state_actor':data['state_actor'],
            'state_critic':data['state_critic'],
            'order':data['order'],
            'action':data['action'],
            'adv_global':data['adv_global'],
            'adv_local':data['adv_local'],
            'adv_coop_mean':data['adv_coop_mean'],
            'adv_coop_std':data['adv_coop_std'],
            'oldp':data['oldp'],
            'mask_order':data['mask_order'],
            'mask_action':data['mask_action'],
            'mask_agent':data['mask_agent']
        }   
        self.update_policy(self.actor_new,self.actor)
        data_size = data_phi['state_critic'].shape[0]
        batch_size = int(self.batch_size/self.agent_num)
        for iter in range(self.train_phi_iters):
            record_phi_loss=[]
            self.meta_optimizer.zero_grad()
            batch_num= int(np.round(data_size/batch_size))
            cnt=0
            thread=1 if self.parallel_way=='mix' else self.parallel_episode
            for index in BatchSampler(SubsetRandomSampler(range(data_size)), batch_size, True):
                for _ in range(thread):
                    loss_phi = self.compute_loss_phi(self.split_batch(index,data_phi))
                    loss_phi/=(batch_num*thread)
                    # loss_phi.backward(retain_graph=True)
                    loss_phi.backward()
                    if iter==0:
                        record_phi_loss_origin.append(loss_phi.item())
                    record_phi_loss.append(loss_phi.item())
                # mpi_avg_grads(ac.v)    # average grads across MPI processes
                if (cnt+1)%batch_num==0:
                    # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, norm_type=2)
                    self.meta_optimizer.step()
                    self.meta_optimizer.zero_grad()
                cnt+=1
        return record_phi_loss_origin, record_phi_loss

    def compute_loss_phi(self,data):
        state_actor, state_critic, order, action, adv_global, adv_local, adv_coop_mean, adv_coop_std,  oldp, mask_order, mask_action, mask_agent = data['state_actor'],data['state_critic'], data['order'], data['action'], data['adv_global'], data['adv_local'], data['adv_coop_mean'], data['adv_coop_std'] , data['oldp'],data['mask_order'], data['mask_action'] ,data['mask_agent']
        # term1
        probs,_ = self.actor_new.multi_mask_forward(state_actor, order, mask_order) 
        # probs shape(batch, agent num, order num, order num)
        # action shape(batch, agent num, order num)
        # newp shape(batch, agent num, order num)
        newp = torch.gather(probs, -1, action[...,None]).squeeze(-1)
        ratio = newp/oldp
        ratio[~mask_action]=0
        if self.clip:
            # adv_global [batch,1] -> [batch,1,1]
            # ratio [batch, agent num, order num]
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_global[:,:,None]
            # clip_adv=ratio*advantage
            loss_term1= -(torch.min(ratio * adv_global[:,:,None], clip_adv))
            loss_term1[~mask_action]=0
            loss_term1 = (torch.sum(loss_term1,dim=-1)/torch.sum(mask_action,dim=-1))[mask_agent].mean()
        else:
            ratio=torch.sum(ratio,dim=1,keepdim=True)/torch.sum(mask_action,dim=1,keepdim=True)
            loss_term1 = -(ratio * adv_global)[mask_agent].mean()
        # term2
        probs, _ = self.actor_old.multi_mask_forward(state_actor, order, mask_order)
        # probs [batch, agent num, order num, order num]
        # action [batch, agent num, order num]
        # newp [batch, agent num, order num]
        newp = torch.gather(probs, -1, action[...,None]).squeeze(-1)
        ratio = newp/oldp
        ratio[~mask_action]=0
        loss_term2 = (torch.sum(ratio,dim=-1)/torch.sum(mask_action,dim=-1))[mask_agent].mean()
        # use train rule
        self.actor_new.zero_grad()
        grad_term1 = torch.autograd.grad(loss_term1, self.actor_new.parameters())
        self.actor_old.zero_grad()
        grad_term2 = torch.autograd.grad(loss_term2, self.actor_old.parameters())
        grad_total = 0
        for grad1, grad2 in zip(grad_term1, grad_term2):
            grad_total += (grad1 * grad2).sum()
        # compute phi and adv_coop
        phi = self.critic.get_phi(state_critic)     # phi [batch, agent num, value dim]
        # adv_local [batch, agent num, value dim]
        # neighbor_num [1,agent num, value dim]
        if self.meta_choose==1:
            adv_coop = (data['adv_local']*phi*self.neighbor_num).sum(-1)
            adv_coop = adv_coop/ (phi*self.neighbor_num).sum(-1) # 归一化
        elif self.meta_choose==2:
            adv_coop = (data['adv_local']*phi*self.neighbor_num).sum(-1)
            adv_coop = adv_coop/ (self.neighbor_num).sum(-1) # 归一化
        elif self.meta_choose==3:
            adv_coop = (data['adv_local']*phi*self.neighbor_num).sum(-1)
            adv_coop = adv_coop/ 20 # 归一化
        elif self.meta_choose==4:
            adv_coop = (data['adv_local']*phi).sum(-1)
        elif self.meta_choose==5:
            adv_coop = (data['adv_local']*phi)[...,1:].sum(-1)
        elif self.meta_choose==6:
            adv_coop = (data['adv_local']*phi)[...,1:].sum(-1)
        elif self.meta_choose==7:
            adv_coop = (data['adv_local']*phi)[...,2:].sum(-1)
        if self.adv_normal:
            adv_coop = (adv_coop-adv_coop_mean)/adv_coop_std
        # loss of meta gradient
        loss_phi = grad_total.detach() * adv_coop[mask_agent].mean()
        return loss_phi

    def compute_loss_actor(self,data):
        state, next_state, order, action, advantage, oldp, mask_order, mask_action, mask_agent, mask_entropy , hidden_rnn_actor = data['state'], data['next_state'] ,data['order'], data['action'], data['advantage'], data['oldp'],data['mask_order'], data['mask_action'] ,data['mask_agent'], data['mask_entropy'] ,data['state_rnn']
        # Policy loss
        probs,_ = self.actor.multi_mask_forward(state, order, mask_order, self.adj,hidden_rnn_actor.unsqueeze(0))
        # probs shape(batch, agent num, order num, order num)
        # action shape(batch, agent num, order num)
        newp = torch.gather(probs, -1, action[...,None]).squeeze(-1)  # newp shape(batch, agent num, order num)
        ratio = newp/oldp
        ratio[~mask_action]=0   # mask_action shape(batch, agent num, order num)
        ratio_diff = torch.abs(ratio - 1)
        if mask_action.any():
            ratio_max = torch.max(ratio_diff[mask_action]).item()
            ratio_mean = torch.mean(ratio_diff[mask_action]).item()
        else:
            ratio_max = 0.0
            ratio_mean = 0.0
        if self.grad_multi=='sum':
            ratio=torch.sum(ratio,dim=1,keepdim=True)
            if self.clip:
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantage
                clip_adv=ratio*advantage
                loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()
            else:
                loss_pi = -(ratio * advantage)[mask_agent].mean()
        elif self.grad_multi=='mean':
            # ratio=torch.sum(ratio,dim=1,keepdim=True)/torch.sum(mask_action,dim=1,keepdim=True)
            if self.clip:
                # advantage shape[batch, agent num, 1]
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantage
                # clip_adv=ratio*advantage
                loss_pi= -(torch.min(ratio * advantage, clip_adv))
                loss_pi[~mask_action]=0
                denom = torch.sum(mask_action,dim=-1,keepdim=True).clamp(min=1)
                loss_pi = (torch.sum(loss_pi,dim=-1)/denom.squeeze(-1))[mask_agent].mean()
                ratio=torch.sum(ratio,dim=-1,keepdim=True)/denom
            else:
                denom = torch.sum(mask_action,dim=1,keepdim=True).clamp(min=1)
                ratio=torch.sum(ratio,dim=1,keepdim=True)/denom
                loss_pi = -(ratio * advantage)[mask_agent].mean()
        # ent= -torch.sum((probs[:,0]+1e-12)*torch.log(probs[:,0]+1e-12),dim=1)
        ent= -torch.sum((probs[mask_entropy]+1e-12)*torch.log(probs[mask_entropy]+1e-12),dim=-1)
        # ent[~mask_action[:,0]]=0
        ent= ent.mean()
        # ent=  -torch.sum((probs[:,:,0]+1e-12)*torch.log(probs[:,:,0]+1e-12),dim=1)[mask_agent].mean()
        loss_pi-= self.ent_factor*ent
        if self.use_auxi:
            auxi_loss = self.compute_loss_auxi(self.actor, state, next_state)
            loss_pi += self.auxi_effi*auxi_loss
        else:
            auxi_loss = np.array([0])
        if self.use_regularize != 'None':
            loss_pi+=  self.regularize_alpha * self.compute_regularize(self.actor)
        if self.use_fake_auxi>0:
            loss_pi += self.auxi_effi*self.compute_fake_loss(self.actor, state, next_state)

        # Useful extra info
        approx_kl = torch.log(ratio[mask_agent]).mean().item()
        # approx_kl=0
        entropy=  ent.item()
        # ratio_max= torch.max(torch.abs(ratio.detach())).item()
        # entropy = pi.entropy().mean().item()
        # clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        # clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, auxi_loss=auxi_loss.item() ,entropy=entropy, ratio_max=ratio_max, ratio_mean=ratio_mean)
        return loss_pi, pi_info

    def compute_loss_auxi(self, net ,state, next_state):
        with torch.no_grad():
            next_state_label = net.get_state_emb(next_state).detach()
        next_state_pred = net.auxiliary_emb(state)
        if self.auxi_loss =='huber':
            error_pred = next_state_pred - next_state_label
            auxi_loss = torch.sum( huber_loss(error_pred,self.huber_delta), dim=-1 ).mean()
        elif self.auxi_loss =='mse':
            error_pred = next_state_pred - next_state_label
            auxi_loss = torch.sum(mse_loss(error_pred),dim=-1).mean()
        elif self.auxi_loss =='cos':
            auxi_loss = cos_loss(next_state_pred, next_state_label).mean()
        return auxi_loss

    def compute_fake_loss(self, net, state, next_state):
        next_state_label = net.get_state_emb(next_state)
        next_state_pred = net.auxiliary_emb(state)
        e=next_state_pred 
        d=next_state_label
        if self.use_fake_auxi == 1:
            fake_loss = e**2/2 + d*(torch.abs(e)-d/2)
        elif self.use_fake_auxi == 2:
            e=e.detach()
            fake_loss = e**2/2 + d*(torch.abs(e)-d/2)
        elif self.use_fake_auxi == 3:
            d=d.detach()
            fake_loss = e**2/2 + d*(torch.abs(e)-d/2)
        elif self.use_fake_auxi == 4:
            a = (torch.abs(e) <= d).float()
            b = (torch.abs(e) > d).float()
            fake_loss = a*e**2/2 + b*d*(torch.abs(e)-d/2)
        elif self.use_fake_auxi == 5:
            e=e.detach()
            a = (torch.abs(e) <= d).float()
            b = (torch.abs(e) > d).float()
            fake_loss = a*e**2/2 + b*d*(torch.abs(e)-d/2)
        elif self.use_fake_auxi == 6:
            d=d.detach()
            a = (torch.abs(e) <= d).float()
            b = (torch.abs(e) > d).float()
            fake_loss = a*e**2/2 + b*d*(torch.abs(e)-d/2)
        elif self.use_fake_auxi == 7:
            fake_loss = -d**2/2
        elif self.use_fake_auxi == 8:
            fake_loss = -torch.abs(d)
        return fake_loss.mean()

    def compute_loss_critic(self,data, critic_fun,value_normalizer):
        state, next_state, ret, old_value, hidden_rnn_critic = data['state'], data['next_state'] ,data['ret'], data['value'], data['state_rnn']
        critic_info = dict(ret=ret.mean().item())
        new_value,_ =critic_fun(state, self.adj ,hidden_rnn_critic.unsqueeze(0))
        if self.use_valuenorm:
            value_normalizer.update(ret)
            ret = value_normalizer.normalize(ret)
        if self.use_value_clip:
            value_pred_clipped = old_value + (new_value - old_value).clamp(-self.clip_ratio, self.clip_ratio)
            error_clipped = ret - value_pred_clipped
            error_original = ret - new_value
            if self.use_huberloss:
                value_loss_clipped = huber_loss(error_clipped,self.huber_delta)
                value_loss_original = huber_loss(error_original,self.huber_delta)
            else:
                value_loss_clipped = mse_loss(error_clipped)
                value_loss_original = mse_loss(error_original)
            value_loss = torch.max(value_loss_original, value_loss_clipped).mean()
        else:
            error_original = ret - new_value
            if self.use_huberloss:
                value_loss = huber_loss(error_original,self.huber_delta).mean()
            else:
                value_loss = mse_loss(error_original).mean()
        if self.use_auxi:
            auxi_loss = self.compute_loss_auxi(self.critic, state, next_state)
            value_loss += self.auxi_effi*auxi_loss
        else:
            auxi_loss = np.array([0])
        if self.use_fake_auxi>0:
            value_loss += self.auxi_effi*self.compute_fake_loss(self.critic, state, next_state)
        if self.use_regularize != 'None':
            value_loss+=  self.regularize_alpha * self.compute_regularize(self.critic)
        critic_info['auxi_loss'] = auxi_loss.item()
        return value_loss,critic_info


class Replay_buffer():
    def __init__(self, capacity,state_dim, order_dim,action_dim, hidden_dim,max_order_num,agent_num , gamma=0.99, lam=0.95 , adv_normal=True, parallel_queue=False, value_local_normalizer=None, value_global_normalizer=None ,use_GAEreturn=False ,actor_decen=True, critic_decen=True, local_value_dim=1, adj=None, args=None ):
        self.capacity = capacity
        self.agent_num=agent_num
        self.order_dim=order_dim
        self.max_order_num=max_order_num
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.hidden_dim=hidden_dim
        self.local_value_dim = local_value_dim
        assert self.action_dim==self.max_order_num , 'action dim error'

        self.state_pool = torch.zeros((self.agent_num,self.capacity, state_dim)).float()
        self.next_state_pool = torch.zeros((self.agent_num,self.capacity, state_dim)).float()
        self.order_pool = torch.zeros((self.agent_num,self.capacity, max_order_num ,order_dim)).float()
        self.state_rnn_actor_pool = torch.zeros((self.agent_num,self.capacity, hidden_dim)).float()
        self.state_rnn_critic_pool = torch.zeros((self.agent_num,self.capacity, hidden_dim)).float()
        self.action_pool = torch.zeros((self.agent_num,self.capacity, max_order_num)).long()
        self.reward_pool = torch.zeros((self.agent_num,self.capacity, 1)).float()
        self.advantage_local_pool =  torch.zeros((self.agent_num,self.capacity, local_value_dim)).float()
        self.advantage_global_pool =  torch.zeros((self.capacity, 1)).float()
        self.return_local_pool = torch.zeros((self.agent_num,self.capacity, local_value_dim)).float()
        self.return_global_pool = torch.zeros((self.capacity, 1)).float()
        self.value_local_pool = torch.zeros((self.agent_num,self.capacity, local_value_dim)).float()
        self.value_global_pool = torch.zeros((self.capacity, 1)).float()
        self.oldp_pool =  torch.zeros((self.agent_num,self.capacity, max_order_num)).float()
        self.mask_order_pool =  torch.zeros((self.agent_num,self.capacity, max_order_num,max_order_num),dtype=torch.bool)
        self.mask_action_pool =  torch.zeros((self.agent_num,self.capacity, max_order_num),dtype=torch.bool)
        self.mask_agent_pool =  torch.zeros((self.agent_num,self.capacity),dtype=torch.bool)
        self.mask_entropy_pool = torch.zeros((self.agent_num,self.capacity, max_order_num),dtype=torch.bool)
        self.gamma=gamma
        self.lam=lam
        self.ptr=0
        self.path_start_idx=0
        self.record_start_idx = 0
        self.adv_normal = adv_normal
        self.parallel_queue=parallel_queue
        self.value_local_normalizer=value_local_normalizer
        self.value_global_normalizer=value_global_normalizer
        self.use_GAEreturn=use_GAEreturn
        self.actor_decen = actor_decen
        self.critic_decen = critic_decen
        if adj is not None:
            self.neighbor = torch.zeros((local_value_dim,agent_num,agent_num),  dtype=torch.float32)
            adj = torch.from_numpy(adj)     # (agent_num,agent_num)
            if args.meta_choose>0 and args.meta_choose <=3:
                for i in range(local_value_dim):
                    index =  adj==i
                    for j in range(agent_num):
                        self.neighbor[i,j][index[j]] = 1/torch.sum(index[j],dim=-1)
            elif args.meta_choose==0:
                assert local_value_dim==1
                for i in range(args.team_rank+1):
                    index = adj==i
                    for j in range(agent_num):
                        self.neighbor[0,j][index[j]] = 1
                self.neighbor/= torch.sum(self.neighbor, dim=-1, keepdim=True)
            elif args.meta_choose==4 or args.meta_choose==5:
                for i in range(local_value_dim):
                    index =  adj<=i
                    for j in range(agent_num):
                        self.neighbor[i,j][index[j]] = 1/torch.sum(index[j],dim=-1)
            elif args.meta_choose==6:
                for i in range(local_value_dim):
                    if i<=1:
                        index =  adj<=i
                    else:
                        index = adj==i
                    for j in range(agent_num):
                        self.neighbor[i,j][index[j]] = 1/torch.sum(index[j],dim=-1)
            elif args.meta_choose==7:
                for i in range(local_value_dim):
                    if i<=2:
                        index =  adj<=i
                    else:
                        index = adj==i
                    for j in range(agent_num):
                        self.neighbor[i,j][index[j]] = 1/torch.sum(index[j],dim=-1)

    def condition_reshape(self,tensor, shape, condition):
        if condition:
            return torch.reshape(tensor,shape)
        else:
            return tensor.transpose(0,1)

    def push(self, state, next_state ,order ,action, reward, value_local, value_global, p , mask_order , mask_action, mask_agent, mask_entropy , state_rnn_actor, state_rnn_critic):
        assert self.ptr < self.capacity
        self.state_pool[:,self.ptr]=state
        self.next_state_pool[:, self.ptr] = next_state
        self.order_pool[:,self.ptr]=order
        self.action_pool[:,self.ptr]=action
        self.reward_pool[:,self.ptr]=reward
        self.value_local_pool[:,self.ptr]=value_local   # (grid num, value dim)
        self.value_global_pool[self.ptr]=value_global # (1,1)
        self.oldp_pool[:,self.ptr]=p
        self.mask_order_pool[:,self.ptr]=mask_order
        self.mask_agent_pool[:,self.ptr]=mask_agent
        self.mask_action_pool[:,self.ptr]=mask_action
        self.mask_entropy_pool[:,self.ptr]=mask_entropy
        self.state_rnn_actor_pool[:, self.ptr]=state_rnn_actor
        self.state_rnn_critic_pool[:,self.ptr]=state_rnn_critic
        self.ptr+=1

    def finish_path_local(self, last_local_val=0):
        #path_slice = slice(self.path_start_idx, self.ptr)
        #reward_self = torch.cat([self.reward_pool[:,self.path_start_idx:self.ptr], last_local_val[:,None,:]],dim=1)
        value_local = torch.cat([self.value_local_pool[:,self.path_start_idx:self.ptr], last_local_val[:,None,:]],dim=1)
        if self.value_local_normalizer is not None:
            value_local= self.value_local_normalizer.denormalize(value_local)   # shape(grid num, 144+1, value dim)
        # the next two lines implement GAE-Lambda advantage calculation
        '''
        reward_self : (grid num, 144, 1)
        neighbor : (value_dim, grid num, grid num)
        '''
        reward_self = self.reward_pool[:,self.path_start_idx:self.ptr]  # shape(grid num, 144, 1)
        # (1,144,1,grid num) * (grid_num, 1, value_dim , grid num) -> (grid num, 144, value_dim)
        reward_local = (reward_self.permute(2,1,0)[...,None,:]*self.neighbor.permute(1,0,2)[:,None,...]).sum(-1)
        reward_local = torch.cat([reward_local,last_local_val[:,None,:]],dim=1)
        #reward_local = torch.matmul(self.neighbor, reward_self.permute(2,0,1))  # shape(value dim, grid num, 144)
        #reward_local = torch.cat([reward_local.permute(1,2,0),last_local_val[:,None,:]], dim=1)
        deltas_local = reward_local[:,:-1] + self.gamma * value_local[:,1:] - value_local[:,:-1]
        advantage_local= torch.zeros(deltas_local.shape,dtype=torch.float32)
        advantage_local[:, -1]=deltas_local[:,-1]
        ret_local = torch.zeros(deltas_local.shape,dtype=torch.float32)
        ret_local[:,-1] = reward_local[:,-2]
        for i in range(deltas_local.shape[1]-2, -1,-1):
            advantage_local[:,i] = deltas_local[:,i]+ advantage_local[:,i+1]*(self.gamma*self.lam)
            ret_local[:,i] = self.gamma*ret_local[:,i+1] + reward_local[:,i]
        #self.adv_buf[:,self.path_start_idx:self.prt] = scipy.signal.lfilter(deltas, self.gamma * self.lam)
        self.advantage_local_pool[:,self.path_start_idx:self.ptr] = advantage_local
        if self.use_GAEreturn:
            self.return_local_pool[:,self.path_start_idx:self.ptr] = advantage_local+value_local[:,:-1]
        else:
            self.return_local_pool[:,self.path_start_idx:self.ptr] = ret_local
        self.record_start_idx = self.path_start_idx
        self.path_start_idx=self.ptr

    def finish_path_global(self, last_global_val=0):
        #path_slice = slice(self.path_start_idx, self.ptr)
        reward_global = torch.cat([self.reward_pool[:,self.path_start_idx:self.ptr].mean(0), last_global_val],dim=0)    # shape(144+1,1)
        value_global = torch.cat([self.value_global_pool[self.path_start_idx:self.ptr], last_global_val],dim=0) # shape(144+1,1)
        if self.value_global_normalizer is not None:
            value_global= self.value_global_normalizer.denormalize(value_global)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas_global = reward_global[:-1] + self.gamma * value_global[1:] - value_global[:-1]  # shape(144,1)
        advantage_global= torch.zeros(deltas_global.shape,dtype=torch.float32)
        advantage_global[-1]=deltas_global[-1]          # [seq num, 1]
        ret_global = torch.zeros(deltas_global.shape,dtype=torch.float32)   # [seq num, 1]
        ret_global[-1] = reward_global[-2]
        for i in range(deltas_global.shape[0]-2, -1,-1):
            advantage_global[i] = deltas_global[i]+ advantage_global[i+1]*(self.gamma*self.lam)
            ret_global[i] = self.gamma*ret_global[i+1] + reward_global[i]
        #self.adv_buf[:,self.path_start_idx:self.prt] = scipy.signal.lfilter(deltas, self.gamma * self.lam)
        self.advantage_global_pool[self.path_start_idx:self.ptr] = advantage_global
        if self.use_GAEreturn:
            self.return_global_pool[self.path_start_idx:self.ptr] = advantage_global+value_global[:-1]
        else:
            self.return_global_pool[self.path_start_idx:self.ptr] = ret_global
        self.path_start_idx=self.ptr

    def sample(self, batch_size):
        index = np.random.choice(range(min(self.capacity,self.num_transition)), batch_size, replace=False)
        bn_s, bn_a, bn_r, bn_s_, bn_seq_ , bn_d , bn_d_seq= self.state_pool[index], self.action_pool[index], self.reward_pool[index],\
                                        self.next_state_pool[index], self.next_seq_pool[index]   ,self.done_pool[index] , self.done_seq_pool[index]

        return bn_s, bn_a, bn_r, bn_s_, bn_seq_ ,bn_d , bn_d_seq

    def normalize(self,input, flag):
        if flag==False:
            return input
        else:
            mean=torch.mean(input)
            std=torch.sqrt(torch.mean((input-mean)**2))
            if std==0:
                std=1
            return (input-mean)/std

    def get(self,device='cpu',writer=None):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr<= self.capacity

        record_ptr=self.ptr
        if self.parallel_queue:
            self.ptr=self.capacity

        '''
        if self.adv_normal:
            adv_mean=torch.mean(self.advantage_pool[:,:self.ptr])
            adv_std= torch.sqrt(torch.mean((self.advantage_pool[:,:self.ptr]-adv_mean)**2))
            if adv_std==0:
                adv_std=1
            self.advantage_pool = (self.advantage_pool-adv_mean)/adv_std
            #self.advantage_pool = self.advantage_pool/adv_std
        '''
        '''
        data = dict(
            state= torch.reshape(self.state_pool[:,:self.ptr],(-1,self.state_dim)).to(device), 
            order= torch.reshape(self.order_pool[:,:self.ptr],(-1,self.max_order_num,self.order_dim)).to(device),
            action= torch.reshape(self.action_pool[:,:self.ptr],(-1,self.max_order_num)).to(device),
            ret= torch.reshape(self.return_pool[:,:self.ptr],(-1,1)).to(device),
            value= torch.reshape(self.value_pool[:,:self.ptr],(-1,1)).to(device),
            #advantage= torch.reshape(self.advantage_pool[:,:self.ptr],(-1,1)).to(device),
            advantage= torch.reshape(self.normalize(self.advantage_pool[:,:self.ptr],self.adv_normal),(-1,1)).to(device),
            oldp= torch.reshape(self.oldp_pool[:,:self.ptr],(-1,self.max_order_num)).to(device),
            mask_order= torch.reshape(self.mask_order_pool[:,:self.ptr],(-1,self.max_order_num, self.max_order_num)).to(device),
            mask_agent= torch.reshape(self.mask_agent_pool[:,:self.ptr],(-1,)).to(device),
            mask_action= torch.reshape(self.mask_action_pool[:,:self.ptr],(-1,self.max_order_num)).to(device),
            state_rnn_actor = torch.reshape(self.state_rnn_actor_pool[:,:self.ptr],(-1,self.hidden_dim)).to(device),
            state_rnn_critic = torch.reshape(self.state_rnn_critic_pool[:,:self.ptr],(-1,self.hidden_dim)).to(device)
            )
        '''
        data = dict(
            state_actor= self.condition_reshape(self.state_pool[:,:self.ptr],(-1,self.state_dim), self.actor_decen).to(device), 
            next_state_actor= self.condition_reshape(self.next_state_pool[:,:self.ptr],(-1,self.state_dim), self.actor_decen).to(device), 
            state_critic = self.condition_reshape(self.state_pool[:,:self.ptr],(-1,self.state_dim), self.critic_decen).to(device), 
            next_state_critic = self.condition_reshape(self.next_state_pool[:,:self.ptr],(-1,self.state_dim), self.critic_decen).to(device), 
            order= self.condition_reshape(self.order_pool[:,:self.ptr],(-1,self.max_order_num,self.order_dim), self.actor_decen).to(device),
            action= self.condition_reshape(self.action_pool[:,:self.ptr],(-1,self.max_order_num), self.actor_decen).to(device),
            ret_local= self.condition_reshape(self.return_local_pool[:,:self.ptr],(-1,self.local_value_dim), self.critic_decen).to(device),
            value_local= self.condition_reshape(self.value_local_pool[:,:self.ptr],(-1,self.local_value_dim), self.critic_decen).to(device),
            adv_local = self.condition_reshape(self.advantage_local_pool[:,:self.ptr],(-1,self.local_value_dim), self.actor_decen).to(device),
            ret_global = self.return_global_pool[:self.ptr].to(device),
            value_global = self.value_global_pool[:self.ptr].to(device),
            adv_global = self.advantage_global_pool[:self.ptr].to(device),
            #advantage= self.condition_reshape(self.normalize(self.advantage_pool[:,:self.ptr],self.adv_normal),(-1,1), self.actor_decen).to(device),
            oldp= self.condition_reshape(self.oldp_pool[:,:self.ptr],(-1,self.max_order_num), self.actor_decen).to(device),
            mask_order= self.condition_reshape(self.mask_order_pool[:,:self.ptr],(-1,self.max_order_num, self.max_order_num), self.actor_decen).to(device),
            mask_agent= self.condition_reshape(self.mask_agent_pool[:,:self.ptr],(-1,), self.actor_decen).to(device),
            mask_action= self.condition_reshape(self.mask_action_pool[:,:self.ptr],(-1,self.max_order_num), self.actor_decen).to(device),
            mask_entropy= self.condition_reshape(self.mask_entropy_pool[:,:self.ptr],(-1,self.max_order_num), self.actor_decen).to(device),
            state_rnn_actor = self.condition_reshape(self.state_rnn_actor_pool[:,:self.ptr],(-1,self.hidden_dim), self.actor_decen).to(device),
            state_rnn_critic = self.condition_reshape(self.state_rnn_critic_pool[:,:self.ptr],(-1,self.hidden_dim), self.critic_decen).to(device)
            )
        #size=self.ptr*self.agent_num
        if self.parallel_queue:
            self.ptr=record_ptr
            if self.ptr==self.capacity:
                self.ptr=0
                self.path_start_idx=0
        else:
            self.ptr=0
            self.path_start_idx=0
        return data


if __name__ == "__main__":
    prob=torch.Tensor([1,2,3])/0
    prob[0]=1
    prob=torch.softmax(prob,0)
    a=torch.multinomial(prob,1 , replacement=True)
    print(a)
    '''

    原始数据是origin_dataset
    有效数据对应的idx存在数组 valid_idx

    np.random.shuffle : 将valid_idx 打乱
    打乱之后按照batch大小取就好了
    假设每次取出的是 index
    然后 valid_idx[index]就得到了 origin_dataset中要取的哪些行
    origin_dataset[valid_idx[index]] 就是只有有效数据的

    '''
