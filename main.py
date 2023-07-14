# -*- coding: utf-8 -*-
"""
Created on 14/3/2023
@author: ZhizhuoYin
"""

import argparse
import logging
import time
import pickle
from tqdm import tqdm
from model import GNNModel
from train import forward
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from se_data_process import load_data_valid, load_testdata
from reservoir import Reservoir
from sampling import *

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='lastfm', help='dataset name: gowalla/lastfm')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=200, help='hidden state size')
parser.add_argument('--epoch', type=int, default=2, help='the number of training epochs. 2 for lastfm, tmall, 4 for gowalla')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--u', type=int, default=1, help='the number of layer with u')
parser.add_argument('--res_size', type=int, default=100, help='the denominator of the reservoir size')
parser.add_argument('--win_size', type=int, default=1, help='the denominator of the window size')
parser.add_argument('--isonline', type=int, default=1, help='1 for streaming model, 0 for offline')
parser.add_argument('--isfixeduser', type=int, default=0, help='1 for fixed user, 0 for none')
parser.add_argument('--isfixeditem', type=int, default=0, help='1 for fixed item, 0 for none')
parser.add_argument('--device', type=int, default=0, help='the number of device')
opt = parser.parse_args()
logging.warning(opt)


def main():
    assert opt.dataset in ['gowalla', 'lastfm', 'tmall']
    device = torch.device('cuda:'+str(opt.device) if torch.cuda.is_available() else 'cpu')
    if opt.dataset == 'gowalla':
        is_user = False #False for not processing user embedding, better performance
        is_item = True
    elif opt.dataset == 'lastfm':
        is_user = False 
        is_item = True
    else:
        is_user = False 
        is_item = True
    print(is_user,is_item)
    usernetdict = {}
    cur_dir = os.getcwd()
    print(opt)
    if is_user:
        user_edge_list = [[],[]]
        usernetfile = open(cur_dir + '/../datasets/' + opt.dataset + '/raw/usernet.txt', 'rb')
        usernet = pickle.load(usernetfile)
        usernetfile.close()
        for sender,receiver in list(zip(usernet[0],usernet[1])):
            user_edge_list[0].append(sender)
            user_edge_list[1].append(receiver)
        for index in range(len(user_edge_list[0])):
            if user_edge_list[0][index] not in usernetdict:
                usernetdict[user_edge_list[0][index]] = {'in':{},'out':{user_edge_list[1][index]:True}}
            else:
                usernetdict[user_edge_list[0][index]]['out'][user_edge_list[1][index]] = True

            if user_edge_list[1][index] not in usernetdict:
                usernetdict[user_edge_list[1][index]] = {'in': {user_edge_list[0][index]:True}, 'out': {}}
            else:
                usernetdict[user_edge_list[1][index]]['in'][user_edge_list[0][index]] = True



    train_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    train_for_res, _ = load_data_valid(
        os.path.expanduser(os.path.normpath(cur_dir + '/../datasets/' + opt.dataset + '/raw/train.txt')), 0)
    max_train_item = max(max(max(train_for_res[0])), max(train_for_res[1]))
    max_train_user = max(train_for_res[2])
    print(max_train_item,max_train_user)
    test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='test1')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    test_for_res = load_testdata(
        os.path.expanduser(os.path.normpath(cur_dir + '/../datasets/' + opt.dataset + '/raw/test1.txt')))
    max_item = max(max(max(test_for_res[0])), max(test_for_res[1]))
    max_user = max(test_for_res[2])
    pre_max_item = max_train_item
    pre_max_user = max_train_user


    log_dir = cur_dir + '/../log/' + str(opt.dataset) + '/paper200/' + '_fix_new_entropy(rank)_on_union+' + str(opt.u) + 'tanh*u_AGCN***GAG-win' + str(opt.win_size) \
              + '***concat3_linear_tanh_in_e2s_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)
    
    if opt.dataset == 'gowalla':
        n_item = 30160
        n_user = 57000
    elif opt.dataset == 'lastfm':
        n_item = 11000
        n_user = 1000
    else:
        n_item = 10000
        n_user = 200000
    model = GNNModel(opt=opt, hidden_size=opt.hidden_size, is_user=is_user, is_item=is_item, n_item=n_item, n_user=n_user, device=device ,u=opt.u).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.lr_dc_step], gamma=opt.lr_dc)
    
    logging.warning(model)

    # offline training on 'train' and test on 'test1'
    logging.warning('*********Begin offline training*********')
    updates_per_epoch = len(train_loader)
    updates_count = 0

    for train_epoch in tqdm(range(opt.epoch)):
        forward(model, train_loader, device, writer, train_epoch, optimizer=optimizer,
                user_edge_dict=usernetdict, is_user=is_user, is_item=is_item,
                train_flag=True, max_item_id=n_item, last_update=updates_count)
        scheduler.step()
        updates_count += updates_per_epoch
        with torch.no_grad():
            forward(model, test_loader, device, writer, train_epoch, user_edge_dict=usernetdict, is_user=is_user, is_item=is_item,
                    train_flag=False, max_item_id=max_item)

    test1_loader = test_loader
    # reservoir construction with 'train'
    logging.warning('*********Constructing the reservoir with offline training data*********')
    if opt.isonline == 1:
        res = Reservoir(train_for_res, opt.res_size, is_user, usernetdict)
        res.update(train_for_res)

    # test and online training on 'test2~5'
    logging.warning('*********Begin online training*********')
    now = time.asctime()

    for test_epoch in tqdm(range(1, 6)):
        if test_epoch != 1:
            test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='test' + str(test_epoch))
            test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
            
            test_for_res = load_testdata(
                os.path.expanduser(os.path.normpath(
                    cur_dir + '/../datasets/' + opt.dataset + '/raw/test' + str(test_epoch) + '.txt')))
            pre_max_item = max_item
            pre_max_user = max_user
            max_item = max(max(max(test_for_res[0])), max(test_for_res[1]))
            max_user = max(test_for_res[2])
            
            # test on the current test set
            # no need to test on test1 because it's done in the online training part
            # epoch + 10 is a number only for the visualization convenience
            t1 = time.time()
            with torch.no_grad():
                forward(model, test_loader, device, writer, test_epoch ,
                        train_flag=False, max_item_id=n_item)

            t2 = time.time()
            print(t2-t1)
        if opt.isonline == 1:
            # reservoir sampling
            sampled_data = fix_new_entropy_on_union(cur_dir, now, opt, model, device, res.data, test_for_res,
                                                    len(test_for_res[0]) // opt.win_size, pre_max_item, pre_max_user,
                                                    ent='wass')

            # cast the sampled set to dataset
            sampled_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset,
                                                 phrase='sampled' + now,
                                                 sampled_data=sampled_data)
            sampled_loader = DataLoader(sampled_dataset, batch_size=opt.batch_size, shuffle=True)

            for params in optimizer.param_groups:  # 遍历Optimizer中的每一组参数
                params['lr'] = opt.lr
            # update with the sampled set
            t1 = time.time()
            forward(model, sampled_loader, device, writer, test_epoch + opt.epoch, optimizer=optimizer,
                    user_edge_dict=usernetdict, is_user=is_user, is_item=is_item,
                    train_flag=True, max_item_id=n_item, last_update=updates_count)
            t2 = time.time()
            print(t2 - t1)
            updates_count += len(test_loader)

            scheduler.step()

            res.update(test_for_res)
            os.remove('../datasets/' + opt.dataset + '/processed/sampled' + now + '.pt')


if __name__ == '__main__':
    main()
