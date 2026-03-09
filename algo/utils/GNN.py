import  torch
from    torch import nn
from    torch.nn import functional as F
import math


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def lap(adj, time_dep):
    if(time_dep):
        for m in range(adj.shape[0]):
            for n in range(adj.shape[1]):
                for i in range(adj.shape[2]):
                    adj[m][n][i][i] = 0
                adj_sum_0 = adj[m][n].sum(0).pow(-0.5)
                adj_sum_1 = adj[m][n].sum(1).pow(-0.5)
                adj_sum_0[adj_sum_0 == float('inf')] = 0
                adj_sum_1[adj_sum_1 == float('inf')] = 0
                adj[m][n] = torch.einsum("ab,bc,cd->ad", (torch.diag(adj_sum_0), adj[m][n], torch.diag(adj_sum_1)))
    else:
        for i in range(adj.shape[0]):
            adj[i][i] = 0
        adj_sum_0 = adj.sum(0).pow(-0.5)
        adj_sum_1 = adj.sum(1).pow(-0.5)
        adj_sum_0[adj_sum_0 == float('inf')] = 0
        adj_sum_1[adj_sum_1 == float('inf')] = 0
        adj = torch.einsum("ab,bc,cd->ad", (torch.diag(adj_sum_0), adj, torch.diag(adj_sum_1)))
    return adj
    
class ChebConv2D(torch.nn.Module):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{\hat{X}}_k \cdot
        \mathbf{\Theta}_k

    where :math:`\mathbf{\hat{X}}_k` is computed recursively by

    .. math::
        \mathbf{\hat{X}}_0 &= \mathbf{X}

        \mathbf{\hat{X}}_1 &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{\hat{X}}_k &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{\hat{X}}_{k-1} - \mathbf{\hat{X}}_{k-2}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K1 (int): Left Chebyshev filter size, *i.e.* number of hops.
        K2 (int): Right Chebyshev filter size, *i.e.* number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K1, K2, device, bias=True):
        super(ChebConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.linear = nn.Linear(in_channels * K1 * K2, out_channels)
        self.K1 = K1
        self.K2 = K2

    def reset_parameters(self):

        size = self.in_channels * self.weight.size(0) * self.weight.size(1)
        
        uniform(size, self.weight)
        uniform(size, self.bias)
    r"""
    ChebConv x: [num_nodes, num_node_features]
    ChebConv2D x: [batch_size, num_nodes, num_nodes, num_node_features (in_channels)]
    """
    def forward(self, x, adj):#edge_index, edge_weight=None):
        time_dep = False
        if(adj.shape[0] == 2):
            time_dep = True
            
        # deal with graph
        num_nodes = adj.shape[-1]
#         adj = adj / 8
#         print('adj shape before lap:', adj.shape)
        adj = lap(adj.clone(), time_dep)
#         print('adj shape after lap:', adj.shape)
# #         print "num_nodes, num_edges, K, batch_size:", num_nodes, num_edges, K, batch_size
        
#         if edge_weight is None:
#             edge_weight = x.new_ones((num_edges, ))
#         edge_weight = edge_weight.view(-1)
#         assert edge_weight.size(0) == edge_index.size(1)

#         deg = degree(row, num_nodes, dtype=x.dtype)

#         # Compute normalized and rescaled Laplacian.
#         print "deg.shape:", deg.shape
#         deg = deg.pow(-0.5)
    
# #         print deg == float('inf')
#         deg[deg == float('inf')] = 0
#         lap = -deg[row] * edge_weight * deg[col]
#         print "lap:", lap
        # Construct Tx: K, K, batch_size, num_nodes, num_nodes, num_node_features (in_channels)
        # adj(time_dep): 2, batch_size, num_nodes, num_nodes
        Tx = torch.zeros(self.K1, self.K2, x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(self.device)
        # Perform filter operation recurrently. 
        ## GCN
        if(not time_dep):
            for k1 in range(self.K1):
                k2 = 0
                if k1 == 0:
                    Tx[k1][k2] = x.clone()
                elif k1 == 1:
                    Tx[k1][k2] = torch.einsum("abcd,be->aecd", (Tx[k1-1][k2].clone(), adj))
                elif k1 in range(2, self.K1):
                    Tx[k1][k2] = 2 * torch.einsum("abcd,be->aecd", (Tx[k1-1][k2].clone(), adj)) - Tx[k1-2][k2].clone()
                for k2 in range(1, self.K2):
                    if k2 == 1:
#                         print Tx[k1][k2-1].shape
#                         print adj.shape
                        Tx[k1][k2] = torch.einsum("abcd,ec->abed", (Tx[k1][k2-1].clone(), adj))
                    elif k2 in range(2, self.K2):
                        Tx[k1][k2] = 2 * torch.einsum("abcd,ec->abed", (Tx[k1][k2-1].clone(), adj)) - Tx[k1][k2-2].clone()
        else:
            for k1 in range(self.K1):
                k2 = 0
                if k1 == 0:
                    Tx[k1][k2] = x.clone()
                elif k1 == 1:
                    Tx[k1][k2] = torch.einsum("abcd,abe->aecd", (Tx[k1-1][k2].clone(), adj[0]))
                elif k1 in range(2, self.K1):
                    Tx[k1][k2] = 2 * torch.einsum("abcd,abe->aecd", (Tx[k1-1][k2].clone(), adj[0])) - Tx[k1-2][k2].clone()
                for k2 in range(1, self.K2):
                    if k2 == 1:
#                         print('Tx[k1][k2-1].shape:', Tx[k1][k2-1].shape)
#                         print('Tx[k1][k2-1]:', Tx[k1][k2-1])
#                         print('adj[1].shape:', adj[1].shape)
#                         print('adj[1]:', adj[1])
                        Tx[k1][k2] = torch.einsum("abcd,aec->abed", (Tx[k1][k2-1].clone(), adj[1]))
                    elif k2 in range(2, self.K2):
                        Tx[k1][k2] = 2 * torch.einsum("abcd,aec->abed", (Tx[k1][k2-1].clone(), adj[1])) - Tx[k1][k2-2].clone()

                    
# #         print "Tx.shape: ", Tx

        # manipulation on x
        Tx2 = Tx.permute([2, 3, 4, 0, 1, 5]).reshape(-1, num_nodes, num_nodes, x.shape[3]*self.K1*self.K2)
        out = self.linear(Tx2)#.reshape(-1, x.shape[3]*self.K1*self.K2))
#         print "out.shape:", out.shape

        r"""
        ============original codes for 1D GCN============
        Tx_0 = x
        out = torch.mm(Tx_0, self.weight[0])

        if K > 1:
            Tx_1 = spmm(edge_index, lap, num_nodes, x)
            out = out + torch.mm(Tx_1, self.weight[1])

        for k in range(2, K):
            Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
            out = out + torch.mm(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        """
        
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels)




class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 use_dropout=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.featureless = featureless
        self.use_dropout = use_dropout

        '''
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        '''
        self.weight = layer_init( nn.Linear(3*embedding_dim,1*embedding_dim), std=gain, bias_const=0, init=init)


    def forward(self, x, support):
        # print('inputs:', inputs)
        if self.training and self.use_dropout:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            xw = torch.matmul(x, self.weight)
        else:
            xw = self.weight

        out = torch.matmul(support, xw)

        '''
        if self.bias is not None:
            out += self.bias
        '''
            
        return self.activation(out)


class GCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim ,output_dim):
        super(GCNLayer, self).__init__()

        self.input_dim = input_dim # 1433
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(GraphConvolution(self.input_dim, self.hidden_dim ,activation=F.relu),
                                    GraphConvolution(self.hidden, output_dim, activation=F.relu)
                                    )

    def forward(self, x, support):
        x = self.layers(x, support)

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
