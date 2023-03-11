# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:32:01 2023

@author: ugo.pelissier
"""

import torch

from utils.random_field import GRF_Mattern
from utils.wave_2D import WaveEq2D
from functorch import vmap
from utils.load import load_config
from utils.loader import DataLoader2D

import modulus
from modulus.hydra import ModulusConfig

dim = 2
N = 128
Nx = 128
Ny = 128
l = 0.1
L = 1.0
sigma = 1.0
Nu = None
Nsamples = 2
dt = 1.0e-4
save_int = int(1e-2/dt)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def data_generator(dim, N, L, Nu, l, sigma, device, Nsamples, Nx, Ny, dt, save_int, phase, i):
    grf = GRF_Mattern(dim, N, length=L, nu=Nu, l=l, sigma=sigma, boundary="periodic", device=device)
    U0 = grf.sample(Nsamples)
    u0 = U0.cpu().float()
    u0 = u0.reshape([u0.shape[0],1,u0.shape[1],u0.shape[2]])
    invar = {'IC': u0}
    
    wave_eq = WaveEq2D(Nx=Nx, Ny=Ny, dt=dt, device=device)
    U = vmap(wave_eq.wave_driver, in_dims=(0, None))(U0, save_int)
    u = U.cpu().float()
    outvar = {'sol': u}
    
    torch.save(invar, 'invar_'+phase+'_'+str(i)+'.pt')
    torch.save(outvar, 'outvar_'+phase+'_'+str(i)+'.pt')

@modulus.main(config_path="../conf", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    for i in range(10):
        data_generator(dim, N, L, Nu, l, sigma, device, cfg['custom']['ntrain'], Nx, Ny, dt, save_int, 'train', i)
        data_generator(dim, N, L, Nu, l, sigma, device, cfg['custom']['ntest'], Nx, Ny, dt, save_int, 'test', i)
        
    invar_train = torch.load("invar_train_0.pt")
    invar_train["IC"].shape
    for i in range(1,10):
        name = "invar_train_" + str(i) + ".pt"
        temp = torch.load(name)
        invar_train["IC"] = torch.cat((invar_train["IC"], temp["IC"]), 0)
        print(invar_train["IC"].shape)
    torch.save(invar_train, 'invar_train.pt')
    
    outvar_train = torch.load("outvar_train_0.pt")
    outvar_train["sol"].shape
    for i in range(1,10):
        name = "outvar_train_" + str(i) + ".pt"
        temp = torch.load(name)
        outvar_train["sol"] = torch.cat((outvar_train["sol"], temp["sol"]), 0)
        print(outvar_train["sol"].shape)
    torch.save(outvar_train, 'outvar_train.pt')
    
    invar_test = torch.load("invar_test_0.pt")
    invar_test["IC"].shape
    for i in range(1,10):
        name = "invar_test_" + str(i) + ".pt"
        temp = torch.load(name)
        invar_test["IC"] = torch.cat((invar_test["IC"], temp["IC"]), 0)
        print(invar_test["IC"].shape)
    torch.save(invar_test, 'invar_test.pt')
    
    outvar_test = torch.load("outvar_test_0.pt")
    outvar_test["sol"].shape
    for i in range(1,10):
        name = "outvar_test_" + str(i) + ".pt"
        temp = torch.load(name)
        outvar_test["sol"] = torch.cat((outvar_test["sol"], temp["sol"]), 0)
        print(outvar_test["sol"].shape)
    torch.save(outvar_test, 'outvar_test.pt')
    
if __name__ == "__main__":
    run()