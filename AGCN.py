# -*- coding: utf-8 -*-
"""
Created on 14/3/2023
@author: ZhizhuoYin
"""

import torch
import math
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros


class AGCN(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    
    def __init__(self, in_channels, out_channels, heads=4, dropout = 0, concat = True, beta = True, improved=False, cached=False,
                 bias=True, **kwargs):
        super(AGCN, self).__init__(aggr='add', **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.weight = Parameter(torch.Tensor(2, in_channels, out_channels))
        self.out_linear = torch.nn.Linear(2 * out_channels, out_channels)
        self.heads = heads
        print("Heads: " + str(heads))
        self.beta = beta
        self.dropout = dropout
        self.concat = False
        self.root_weight = False
        self.lin_key = torch.nn.Linear(out_channels, heads * out_channels)
        self.lin_query = torch.nn.Linear(out_channels, heads * out_channels)
        self.lin_value = torch.nn.Linear(out_channels, heads * out_channels)
        self.m0_alpha = Parameter(torch.Tensor(out_channels))
        self.m1_alpha = Parameter(torch.Tensor(out_channels))
        if self.concat:
            self.lin_skip = torch.nn.Linear(out_channels, heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = torch.nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = torch.nn.Linear(out_channels, out_channels, bias=bias)
            if self.beta:
                self.lin_beta = torch.nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.lin_l = torch.nn.Linear(out_channels, out_channels, bias=bias)
        self.lin_r = torch.nn.Linear(out_channels, out_channels, bias=False)
        self.rnn0 = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.rnn1 = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):

        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.rnn0.reset_parameters()
        self.rnn1.reset_parameters()
        glorot(self.weight)
        zeros(self.bias)
        #self.beta = torch.rand(1)
        self.cached_result = None
        self.cached_num_edges = None
    
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)
        
        fill_value = 1 if not improved else 2
        #print(edge_index, edge_weight)
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], edge_weight
    
    def forward(self, x, edge_index, edge_weight=None, u=None):
        """"""
        #print(x.size(),u.size())

        x = x.view(-1, self.out_channels)
        u = u.view(-1, self.out_channels)
        #x=x+u

        #item = item.view(-1, self.out_channels)
        x0 = x#torch.matmul(torch.cat((x, u), dim=-1), self.weight[0])
        x0 = (x0,x0)
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
        
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index1, norm, edge_weight1 = self.norm(edge_index, x.size(0), edge_weight[0],
                                          self.improved, x.dtype)
            self.cached_result = edge_index1, norm
        
        edge_index1, norm = self.cached_result

        #print(norm.size(),x.size(),u.size())
        self.flow = 'source_to_target'
        #x0_norm = F.normalize(x0[0],p=2,dim=1)
        #x0_cat = torch.cat((x0, u), dim=-1)
        m0 = self.propagate(edge_index1, x=x0[0], norm=norm, edge_weight=edge_weight1)
        m_u0 = F.normalize(u)#self.propagate(edge_index1, x=u, norm=norm, edge_weight=edge_weight1))
        #if self.concat:
        #    m0 = m0.view(-1, self.heads * self.out_channels)
        #else:
        #    m0 = m0.mean(dim=1)
        m0_alpha = F.softmax(F.leaky_relu(m0*m_u0).sum(dim=-1).view(1,-1),dim=1).view(-1,1)

        m0 = m0 + m0_alpha * m_u0
        #m0 = self.rnn0(m0, x0[1])
        if self.root_weight:
            x_r = self.lin_skip(x0[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([m0, x_r, m0 - x_r], dim=-1))
                beta = F.leaky_relu(beta)
                m0 = beta * x_r + (1 - beta) * m0
            else:
                m0 += x_r

        #torch.matmul(torch.cat((m0, m_u0), dim=-1), self.weight[0])
        #m0 = self.lin_l(m0)
        #rm0 = self.rnn(x0, m0)
        #m0 += self.lin_r(x0[1])
        #m0 = F.normalize(m0,p=2,dim=-1)
        #m0 = self.rnn(x0_norm, m0)
        
        x1 = x#torch.matmul(torch.cat((x, u), dim=-1), self.weight[1])
        x1 = (x1,x1)
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
        
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index2, norm, edge_weight2 = self.norm(edge_index, x.size(0), edge_weight[1],
                                          self.improved, x.dtype)
            self.cached_result = edge_index2, norm
        
        edge_index2, norm = self.cached_result
        
        self.flow = 'target_to_source'
        #x1_norm = F.normalize(x1[0],p=2,dim=1)
        #x1_cat = torch.cat((x1, u), dim=-1)

        m1 = self.propagate(edge_index = edge_index2, x=x1[0], norm=norm, edge_weight=edge_weight2)
        m_u1 = F.normalize(u)#self.propagate(edge_index=edge_index2, x=u, norm=norm, edge_weight=edge_weight2))
        m1_alpha = F.softmax(F.leaky_relu(m1 * m_u1).sum(dim=-1).view(1, -1), dim=1).view(-1, 1)
        m1 = m1 + m1_alpha*m_u1
        #m1 = self.rnn1(m1, x1[1])
        #if self.concat:
        #    m1 = m1.view(-1, self.heads * self.out_channels)
        #else:
        #    m1 = m1.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x1[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([m1, x_r, m1 - x_r], dim=-1))
                beta = F.leaky_relu(beta)
                m1 = beta * x_r + (1 - beta) * m1
            else:
                m1 += x_r

        #m1 = torch.matmul(torch.cat((m1, m_u1), dim=-1), self.weight[1])
        #m1 = self.lin_l(m1)
        #rm1 = self.rnn(m1,x1[])
        #m1 += self.lin_r(x1[1])
        #m1 = F.normalize(m1, p=2, dim=-1)
        #m1 = self.rnn(x1_norm, m1)
        #print(m0.size(), m1.size())

        return self.out_linear(torch.cat((m0,m1),dim=-1))
    
    def message(self, edge_index_i, x_i, x_j, norm, edge_weight):

        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        #if self.lin_edge is not None:
        #    edge_weight = self.lin_edge(edge_weight).view(-1, self.heads,
        #                                              self.out_channels)
        #key *= edge_weight.view(-1,1,1)
        alpha = F.relu((query * key).sum(dim=-1)) / math.sqrt(self.out_channels)
        alpha = F.softmax(alpha, dim=0)
        #alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = F.normalize(alpha)
	    #out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        #out += norm.view(-1, self.heads, 1)*self.lin_value(x_j).view(-1,self.heads,self.out_channels)
        #if edge_weight is not None:
        #    out *= edge_weight.view(-1,1,1)

        #out *= alpha.view(-1, self.heads, 1)
        #out = out.view(-1, self.out_channels)# + norm.view(-1, self.heads)*x_j
        gcnheads = torch.stack([norm.view(-1,1)*x_j for i in range(self.heads)],1)#self.lin_value(norm.view(-1,1)*x_j).view(-1, self.heads, self.out_channels)#
        multiheadout = gcnheads*alpha.view(-1, self.heads, 1)
        multiheadout = multiheadout.mean(dim=1)
        return multiheadout#out.view(-1, self.out_channels,1)#edge_weight.view(-1,1) * out.view(-1, self.out_channels)#(norm.view(-1, self.heads , 1) * out).view(-1, self.heads * self.out_channels)

    def message_and_aggregate(self, adj_t, x):
        return torch.matmul(adj_t, x, reduce=self.aggr)
    
    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
