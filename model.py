# -*- coding: utf-8 -*-
"""
Created on 14/3/2023
@author: ZhizhuoYin
"""

import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from AGCN import AGCN
from GEN import GENConv
from torch_geometric.nn import GatedGraphConv,SAGEConv,GATConv,GCNConv


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.user_linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, node_embedding, item_embedding_table, sections, num_count, user_embedding, max_item_id, u_n_repeat):
        v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(node_embedding)))    # |V|_i * 1
        s_g_whole = num_count.view(-1, 1) * alpha * node_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        return s_h


class Embedding2ScoreWithU(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2ScoreWithU, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, 1)
        self.W_2 = nn.Linear(3*self.hidden_size + self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_5 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.user_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.user_out = nn.Linear(2 * self.hidden_size, self.hidden_size)


    def forward(self, node_embedding, global_node_embedding, item_embedding_table, sections, num_count, user_embedding, max_item_id,
                u_n_repeat):
        if list(sections.size())[0] == 1:
            u_n_repeat = u_n_repeat.view(1, -1)
            #global_node_embedding = global_node_embedding.view(1,-1)
            node_embedding = node_embedding.view(-1, self.hidden_size)
            v_n_repeat = tuple(node_embedding[-1].repeat(sections[0], 1))
            alpha = self.W_1(
                torch.sigmoid(self.W_2(torch.cat((torch.cat(v_n_repeat).view(sections[0], -1), node_embedding, global_node_embedding, u_n_repeat), dim=-1))))
        else:
            v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))  # split whole x back into graphs G_i
            v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)  # repeat |V|_i times for the last node embedding
            alpha = self.W_1(
                torch.sigmoid(self.W_2(torch.cat((torch.cat(v_n_repeat, dim=0), node_embedding, global_node_embedding, u_n_repeat), dim=-1))))
            # vl: last node vector of each subgraph; vi: node vector of each node; u: user vector of each subgraph
            # alpha: weight of each node

        s_g_whole = num_count.view(-1, 1) * alpha * (node_embedding)  # |V|_i * hidden_size

        if list(sections.size())[0] == 1:
            s_g = tuple(torch.sum(s_g_whole.view(-1, self.hidden_size), dim=0).view(1, -1))
            u_n = tuple(torch.mean(u_n_repeat.view(-1, self.hidden_size), dim=0).view(1, -1))
            stack_u_n = torch.cat(u_n, dim=0)
            s_h = self.W_5(torch.cat((node_embedding[-1].view(1,-1), s_g[0].view(-1,self.hidden_size)), dim=-1)) #,u_n[0].view(-1, self.hidden_size)
        else:
            s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))  # split whole s_g into graphs G_i
            s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split) # sum node vectors in graph G_i as s_g

            u_n_split = torch.split(u_n_repeat,tuple(sections.cpu().numpy()))
            u_n = tuple(torch.mean(embeddings, dim=0).view(1, -1) for embeddings in u_n_split)
            stack_u_n = torch.cat(u_n,dim=0)

            v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i) # obtain last node tuple
            stack_v_n = torch.cat(v_n, dim=0)
            s_h = self.W_5(torch.cat((stack_v_n, torch.cat(s_g, dim=0)),dim=1)) # concatenate last node of graph and graph s_g
        
        #s_h += self.user_linear(user_embedding).tanh() # residual addition
        return s_h


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_item: the number of items in the whole item set for embedding layer.
        n_user: the number of users
    """
    def __init__(self, opt, hidden_size, n_item, is_user = False, is_item = True, basicmodel = "BA-GCN", n_user=None, heads=None, device=None, u=1):
        super(GNNModel, self).__init__()
        self.opt = opt
        self.globalItemEmbedding = Parameter(torch.rand([n_item,hidden_size]),requires_grad=True)
        self.is_user = is_user
        self.is_item = is_item
        self.basicmodel = basicmodel
        self.global_item_edge_list = 0
        self.global_user_edge_list = 0
        self.fixed_user = torch.zeros(200)
        self.alphaweight = Parameter(torch.Tensor(hidden_size,1))
        self.device = device
        self.hidden_size, self.n_item, self.n_user, self.heads, self.u = hidden_size, n_item, n_user, heads, u
        self.globalitemembedding = nn.Embedding(self.n_item, self.hidden_size)
        self.item_embedding = nn.Embedding(self.n_item, self.hidden_size)
        self.item_embeddingConv = GatedGraphConv(self.hidden_size, num_layers=1)
        self.user_embeddingConv = GATConv(self.hidden_size, self.hidden_size, heads=1, dropout=0.6)
        self.sagelayer = SAGEConv(self.hidden_size,self.hidden_size)
        self.sagelayerout = SAGEConv(self.hidden_size, self.hidden_size)
        self.dropoutlayer = nn.Dropout(p=0.7)
        self.Wa = nn.Embedding(self.hidden_size, self.hidden_size)
        self.Ua = nn.Embedding(self.hidden_size, self.hidden_size)

        self.va = nn.Embedding(self.hidden_size,1)
        self.vb = nn.Embedding(self.hidden_size, 1)
        self.beta = Parameter(torch.Tensor(n_item))


        if self.n_user:
            self.user_embedding = nn.Embedding(self.n_user, self.hidden_size)
        if self.u > 0:
            self.gnn = []
            for i in range(self.u):
                if basicmodel == "BA-GCN":
                    self.gnn.append(AGCN(2 * self.hidden_size, self.hidden_size, heads=1).to(device))
                elif basicmodel == "GCN":
                    self.gnn.append(GCNConv(self.hidden_size,self.hidden_size).to(device))
                elif basicmodel == "GAT":
                    self.gnn.append(GATConv(self.hidden_size,self.hidden_size).to(device))
        else:
            self.gnn = AGCN(self.hidden_size, self.hidden_size)
        self.e2s = Embedding2ScoreWithU(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data, train_flag = False, item=0, user_edge_list=None, is_user = False,is_item = True, item_edge_index=0, max_item_id=0):
        x, edge_index, batch, edge_count, in_degree_inv, out_degree_inv, sequence, num_count, userid = \
            data.x - 1, data.edge_index, data.batch, data.edge_count, data.in_degree_inv, data.out_degree_inv,\
            data.sequence, data.num_count, data.userid

        hidden = self.item_embedding(x).squeeze()
        if train_flag==True:
            if is_user:
                self.global_user_edge_list = user_edge_list
                globalusr = self.user_embeddingConv(self.user_embedding.weight,user_edge_list)
                globalusr = F.normalize(globalusr)
                globalusr = self.sagelayer(globalusr, user_edge_list)
                globalusr = self.sagelayerout(globalusr, user_edge_list)
            else:
                globalusr = self.user_embedding.weight
            if is_item:
                self.global_item_edge_list = item_edge_index
                globalie = self.item_embeddingConv(self.globalitemembedding.weight, self.global_item_edge_list)
            else:
                globalie = self.globalitemembedding.weight
        else:
            globalie = self.globalitemembedding.weight
            globalusr = self.user_embedding.weight

        if self.opt.isfixeduser == 1:
            globalusr = torch.zeros_like(globalusr)
        if self.opt.isfixeditem == 1:
            globalie = torch.zeros_like(globalie)
        itemfeat = globalie[x].squeeze()

        u = globalusr[userid].squeeze()
        sections = torch.bincount(batch) # count node number of each subgraph

        if self.u > 0:
            for layer in range(self.u):
                if list(sections.size())[0] == 1:
                    u_n_repeat = tuple(u.view(1, -1).repeat(sections[0], 1))
                else:
                    # constructing user vector for each node
                    u_n_repeat = tuple(u.view(1, -1).repeat(int(times), 1) for (u, times) in zip(u, sections))

                itemAttention = F.softmax(
                    F.leaky_relu(torch.matmul(torch.matmul(itemfeat,self.Wa.weight)+torch.matmul(torch.cat(u_n_repeat, dim=0),self.Ua.weight), self.va.weight)).view(1, -1),
                    dim=1).view(-1, 1)

                hidden = hidden + (itemAttention * itemfeat)
                if self.basicmodel == "BA-GCN":
                    hidden = self.gnn[layer](hidden, edge_index,
                                             [edge_count * in_degree_inv, edge_count * out_degree_inv],
                                             u=torch.cat(u_n_repeat, dim=0))
                elif self.basicmodel == "GCN":
                    edge_weight = edge_count * out_degree_inv
                    hidden = self.gnn[layer](hidden, edge_index, edge_weight)
                elif self.basicmodel == "GAT":
                    edge_weight = edge_count * out_degree_inv
                    hidden = self.gnn[layer](hidden, edge_index)
                if self.heads is not None:
                    hidden = torch.stack(hidden.chunk(self.heads, dim=-1), dim=1).mean(dim=1)

                u = self.e2s(hidden, itemAttention*itemfeat, self.item_embedding, sections, num_count, u, max_item_id, torch.cat(u_n_repeat, dim=0))
        else:
            hidden = self.gnn(hidden, edge_index, [edge_count * in_degree_inv, edge_count * out_degree_inv], u=None)
            if self.heads is not None:
                hidden = torch.stack(hidden.chunk(self.heads, dim=-1), dim=1).mean(dim=1)
            u = self.e2s(hidden, itemfeat, self.item_embedding, sections, num_count, u, max_item_id)
        
        z_i_hat = torch.mm(u, self.item_embedding.weight[:max_item_id].transpose(1, 0))

        return z_i_hat
