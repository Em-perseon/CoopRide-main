import  torch
from    torch import nn
from    torch.nn import functional as F

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
