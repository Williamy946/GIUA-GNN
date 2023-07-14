# -*- coding: utf-8 -*-
"""
Created on 14/3/2023
@author: ZhizhuoYin
"""

import numpy as np
import torch
import pandas as pd
from torch.nn.functional import softmax
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from scipy.stats import wasserstein_distance


def forward(model, loader, device, writer, epoch, user_edge_dict = None,is_user = False, is_item=True, optimizer=None, train_flag=True, max_item_id=0, last_update=0):
    if train_flag:
        model.train()
    else:
        model.eval()
        hit20, mrr20, hit10, mrr10, hit5, mrr5, hit1, mrr1 = [], [], [], [], [], [], [], []

    mean_loss = 0.0
    itemlist = []
    edgerepeat = []
    item_edges = [[], []]
    edgelist = [[] for i in range(max_item_id+1)] # elements [receiver,times,index]
    globalItem2IndMapper = {}
    itemindex = 0

    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        x = [it[0] for it in batch.x.tolist()]
        edge_index = batch.edge_index.tolist()
        edge_count = batch.edge_count.tolist()
        itemlist += list(filter(lambda d: d not in globalItem2IndMapper,x))

        item = itemlist
        if is_item:
            for it in x:
                if it not in globalItem2IndMapper:
                    globalItem2IndMapper[it] = itemindex
                    itemindex += 1

            for k in range(len(edge_index[0])):
                isexist = 0
                for receiver in edgelist[x[edge_index[0][k]]]:
                    if receiver[0] == x[edge_index[1][k]]:
                        receiver[1] += 1
                        isexist = 1
                        break
                if not isexist:
                    item_edges[0] += [x[edge_index[0][k]]-1]
                    item_edges[1] += [x[edge_index[1][k]]-1]
                    edgelist[x[edge_index[0][k]]].append([x[edge_index[1][k]],1,len(edgerepeat)])
                    edgerepeat += [edge_count[k]]

        usredgelist = [[], []]
        userid = batch.userid.tolist()
        if is_user == True:
            for u in userid:
                for v in userid:
                    v = int(v)
                    u = int(u)
                    if (u in user_edge_dict) and (v in user_edge_dict):
                        if v in user_edge_dict[u]['in']:
                            usredgelist[0].append(v)
                            usredgelist[1].append(u)
                        if v in user_edge_dict[u]['out']:
                            usredgelist[0].append(u)
                            usredgelist[1].append(v)
        if is_item:
            usredgelist = torch.tensor(usredgelist, dtype=torch.long)
            item_edge_index = torch.tensor(item_edges,dtype=torch.long)
            item = torch.tensor(item,dtype=torch.long)
            scores = model(batch.to(device),train_flag=train_flag, is_user=is_user, is_item=is_item, user_edge_list=usredgelist.to(device) ,item=item.to(device),item_edge_index=item_edge_index.to(device), max_item_id=max_item_id)
        else:
            scores = model(batch.to(device), train_flag=train_flag, is_user=is_user, is_item=is_item, max_item_id=max_item_id)
        targets = batch.y - 1
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), last_update + i)
        else:
            sub_scores = scores.topk(20)[1]    # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]  # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit10.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(5)[1]  # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit5.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr5.append(0)
                else:
                    mrr5.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(1)[1]  # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit1.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr1.append(0)
                else:
                    mrr1.append(1 / (np.where(score == target)[0][0] + 1))

        mean_loss += loss / batch.num_graphs

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit20 = np.mean(hit20) * 100
        mrr20 = np.mean(mrr20) * 100
        print(str(hit20)+'\t'+str(mrr20))
        writer.add_scalar('index/hit20', hit20, epoch)
        writer.add_scalar('index/mrr20', mrr20, epoch)
        hit10 = np.mean(hit10) * 100
        mrr10 = np.mean(mrr10) * 100
        print(str(hit10)+'\t'+str(mrr10))
        writer.add_scalar('index/hit10', hit10, epoch)
        writer.add_scalar('index/mrr10', mrr10, epoch)
        hit5 = np.mean(hit5) * 100
        mrr5 = np.mean(mrr5) * 100
        print(str(hit5)+'\t'+str(mrr5))
        writer.add_scalar('index/hit5', hit5, epoch)
        writer.add_scalar('index/mrr5', mrr5, epoch)
        hit1 = np.mean(hit1) * 100
        mrr1 = np.mean(mrr1) * 100
        print(str(hit1)+'\t'+str(mrr1))
        writer.add_scalar('index/hit1', hit1, epoch)
        writer.add_scalar('index/mrr1', mrr1, epoch)
        return [[hit20,hit10,hit5,hit1],[mrr20,mrr10,mrr5,mrr1],epoch]
    return []

def forward_entropy(model, loader, device, max_item_id=0):
    for i, batch in enumerate(loader):
        scores = softmax(model(batch.to(device), train_flag=False, max_item_id=max_item_id), dim=1)
        dis_score = Categorical(scores)
        if i == 0:
            entropy = dis_score.entropy()
        else:
            entropy = torch.cat((entropy, dis_score.entropy()))
    
    pro = entropy.cpu().detach().numpy()
    weights = np.exp((pd.Series(pro).rank() / len(pro)).values)
    return weights / np.sum(weights)


def forward_cross_entropy(model, loader, device, max_item_id=0):
    for i, batch in enumerate(loader):
        scores = softmax(model(batch.to(device),train_flag=False, max_item_id= max_item_id), dim=1)
        targets = batch.y - 1
        if i == 0:
            cross_entropy = torch.nn.functional.cross_entropy(scores, targets, reduction='none')
        else:
            cross_entropy = torch.cat((cross_entropy, torch.nn.functional.cross_entropy(scores, targets, reduction='none')))

    pro = cross_entropy.cpu().detach().numpy()
    return pro / pro.sum()


def forward_wass(model, loader, device, max_item_id=0):
    distance = []
    for i, batch in enumerate(loader):

        scores = softmax(model(batch.to(device), train_flag=False, max_item_id = max_item_id), dim=1)
        targets = batch.y - 1

        targets_1hot = torch.zeros_like(scores).scatter_(1, targets.view(-1, 1), 1).cpu().numpy()
        distance += list(wasserstein_distance(score, target) for score, target in zip(scores.cpu().numpy(), targets_1hot))

    weights = np.exp((pd.Series(distance).rank() / len(distance)).values)
    return weights / np.sum(weights)
