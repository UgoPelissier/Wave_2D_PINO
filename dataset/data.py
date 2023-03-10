# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:32:01 2023

@author: ugo.pelissier
"""

import torch

from .utils.random_field import GRF_Mattern
from .utils.wave_2D import WaveEq2D
from functorch import vmap
from .utils.load import load_config
from .utils.loader import DataLoader2D

def data_generator(dim, N, L, Nu, l, sigma, device, Nsamples, Nx, Ny, dt, save_int):
    grf = GRF_Mattern(dim, N, length=L, nu=Nu, l=l, sigma=sigma, boundary="periodic", device=device)
    U0 = grf.sample(Nsamples)
    u0 = U0.cpu().float()
    u0 = u0.reshape([u0.shape[0],1,u0.shape[1],u0.shape[2]])
    invar = {'IC': u0}
    
    wave_eq = WaveEq2D(Nx=Nx, Ny=Ny, dt=dt, device=device)
    U = vmap(wave_eq.wave_driver, in_dims=(0, None))(U0, save_int)
    u = U.cpu().float()
    outvar = {'sol': u}
    
    return invar, outvar