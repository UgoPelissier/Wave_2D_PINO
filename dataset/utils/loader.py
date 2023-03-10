# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:53:04 2023

@author: ugo.pelissier
"""

import torch
import numpy as np

def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt

class DataLoader2D(object):
    def __init__(self, data, nx=128, nt=100, sub=1, sub_t=1):
        self.sub = sub
        self.sub_t = sub_t            
        s = nx
        # if nx is odd
        if (s % 2) == 1:
            s = s - 1
        self.S = s // sub
        self.T = nt // sub_t
        self.T += 1
        data = data[:, 0:self.T:sub_t, 0:self.S:sub, 0:self.S:sub]
        self.data = data.permute(0, 2, 3, 1)
        
    def make_loader(self, n_sample, batch_size, start=0, train=True):
        a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
        u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, 1]),
                            gridy.repeat([n_sample, 1, 1, 1, 1]),
                            gridt.repeat([n_sample, 1, 1, 1, 1]),
                            a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader