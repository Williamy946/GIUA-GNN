# -*- coding: utf-8 -*-
"""
Created on 14/3/2023
@author: ZhizhuoYin
"""

import numpy as np


class Reservoir(object):
    def __init__(self, train, size_denominator, is_user, usernetdict):
        super(Reservoir, self).__init__()
        self.is_user = False#is_user
        self.userdict = usernetdict
        #self.maxlen = max([len(usernetdict[user]['in']) for user in usernetdict])
        self.weightlist = []
        self.r_size = len(train[0]) / size_denominator
        self.t = 0
        self.data = ([], [], [])

    def add(self, x, y, u):  # one list represents one sample
        # global t
        random = np.random.rand()

        if self.is_user:
            if u not in self.userdict:
                self.userdict[u] = {'in': {-1: True}, 'out': {}}
            userweight = random * np.arctan(len(self.userdict[u]['in']))*2/np.pi
        else:

            userweight = random

        if self.t < self.r_size:
            self.data[0].append(x)
            self.data[1].append(y)
            self.data[2].append(u)
            self.weightlist.append(userweight)
        else:
            p = self.r_size / self.t
            s = False
            if self.is_user:
                if userweight > min(self.weightlist):
                    s = True
                    index = self.weightlist.index(min(self.weightlist))
                    self.weightlist[index] = userweight
            else:
                random = np.random.rand()
                if random <= p:
                    s = True
                    index = int(random * (len(self.data[0]) - 1))
            if s:
                self.data[0][index] = x
                self.data[1][index] = y
                self.data[2][index] = u
        self.t += 1

    def update(self, data):
        for index in range(len(data[0])):
            x = data[0][index]
            y = data[1][index]
            user = data[2][index]
            self.add(x, y, user)
        x = 1
