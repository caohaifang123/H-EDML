"""Euclidean layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax
from torch_scatter import scatter_add
import geoopt.manifolds.poincare.math as pmath
from utils.train_utils import glorot, zeros, add_self_loops


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
            self.in_features, self.out_features
        )


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class Attention(Module):
    def __init__(self, input_channel, reduction=2):
        super(Attention, self).__init__()
        input_channel = input_channel * 2

        self.fc = nn.Sequential(
            nn.Linear(input_channel, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2, bias=False),
        )
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x_h, x_e):
        # x_h = pmath.logmap0(x_h, c=args.c)
        px = [x_h, x_e]
        px = torch.cat(px, dim=-1)
        attention = self.fc(px)
        attention = self.sm(attention)
        return attention


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1)
        return probs


class GATConv(MessagePassing):
    """The graph attentional operator from the "Graph Attention Networks"
    Implementation based on torch_geometric
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 act=None):
        super(GATConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if concat:
            self.out_channels = out_channels // heads
        else:
            self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, heads * self.out_channels, bias=bias)
        self.att = Parameter(torch.Tensor(1, 2 * heads * self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)
        self.linear.reset_parameters()

    def forward(self, input):
        """"""
        x, adj = input
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index = adj._indices()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x)
        out = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        out = self.act(out)
        return out, adj

    def message(self, edge_index_i, x_i, x_j, num_nodes):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1).reshape(-1, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.out_channels)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GCNConv(MessagePassing):
    """The graph convolutional operator from the "Semi-supervised Classification with Graph Convolutional Networks"
    Implementation based on torch_geometric
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 dropout=0,
                 bias=True,
                 act=None):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes)
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, input, edge_weight=None):
        """"""
        x, adj = input
        edge_index = adj._indices()
        x = torch.matmul(x, self.weight)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        out = self.propagate(edge_index, x=x, norm=norm)
        out = self.act(out)
        return out, adj

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SGConv(MessagePassing):
    """The simple graph convolutional operator from the "Simplifying Graph Convolutional Networks"
    Implementation based on torch_geometric
    """

    def __init__(self, in_channels, out_channels, K=1, cached=False, bias=True, dropout=0, act=None):
        super(SGConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        if bias:
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.dropout = dropout
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, input, edge_weight=None):
        """"""
        x, adj = input
        edge_index = adj._indices()

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached:
            x = self.lin(x)

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = GCNConv.norm(edge_index, x.size(0),
                                            edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        if self.cached:
            x = self.lin(self.cached_result)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(x)
        return x, adj

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class SAGEConv(MessagePassing):
    """The GraphSAGE operator from the "Inductive Representation Learning on Large Graphs"
    Implementation based on torch_geometric
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True, dropout=0, act=None):
        super(SAGEConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = torch.nn.Linear(in_channels, out_channels, bias=False)

        self.dropout = dropout
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, input, edge_weight=None):
        """"""
        x, adj = input
        edge_index = adj._indices()

        if torch.is_tensor(x):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_rel(out) + self.lin_root(x[1])

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.act(out)
        return out, adj

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
