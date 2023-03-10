# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:38:47 2023

@author: ugo.pelissier
"""

import torch
import matplotlib.pyplot as plt

class WaveEq2D():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                 ymin=0,
                 ymax=1,
                 Nx=100,
                 Ny=100,
                 c=1.0,
                 dt=1e-3,
                 tend=1.0,
                 device=None,
                 dtype=torch.float64,
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Nx = Nx
        self.Ny = Ny
        x = torch.linspace(xmin, xmax, Nx+1, device=device, dtype=dtype)
        y = torch.linspace(ymin, ymax, Ny+1, device=device, dtype=dtype)
        self.x = x
        self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.X, self.Y = torch.meshgrid(x,y,indexing='ij')
        self.c = c
        self.phi = torch.zeros_like(self.X[:Nx,:Ny], device=device)
        self.psi = torch.zeros_like(self.phi, device=device)
        self.phi0 = torch.zeros_like(self.phi, device=device)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.Phi = []
        self.T = []
        self.device = device
        
    

    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx

    def Dy(self, data):
        data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
        return data_dy


    def Dxy(self, data):
        data_dxy = self.CD_ij(data, axis_i=0, axis_j=1, dx=self.dx, dy=self.dy)
        return data_dxy
        

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    def Dyy(self, data):
        data_dyy = self.CD_ii(data,axis=1, dx=self.dy)
        return data_dyy

    
    def wave_calc_RHS(self, phi, psi):
        phi_xx = self.Dxx(phi)
        phi_yy = self.Dyy(phi)
        
        psi_RHS = self.c**2 * (phi_xx + phi_yy)
        phi_RHS = psi
        return phi_RHS, psi_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        


    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def wave_rk4(self, phi, psi, t=0):
        phi_RHS1, psi_RHS1 = self.wave_calc_RHS(phi, psi)
        t1 = t + 0.5*self.dt
        phi1 = self.update_field(phi, phi_RHS1, step_frac=0.5)
        psi1 = self.update_field(psi, psi_RHS1, step_frac=0.5)
        
        phi_RHS2, psi_RHS2 = self.wave_calc_RHS(phi1, psi1)
        t2 = t + 0.5*self.dt
        phi2 = self.update_field(phi, phi_RHS2, step_frac=0.5)
        psi2 = self.update_field(psi, psi_RHS2, step_frac=0.5)
        
        phi_RHS3, psi_RHS3 = self.wave_calc_RHS(phi2, psi2)
        t3 = t + self.dt
        phi3 = self.update_field(phi, phi_RHS3, step_frac=1.0)
        psi3 = self.update_field(psi, psi_RHS3, step_frac=1.0)
        
        phi_RHS4, psi_RHS4 = self.wave_calc_RHS(phi3, psi3)
        
        t_new = t + self.dt
        psi_new = self.rk4_merge_RHS(psi, psi_RHS1, psi_RHS2, psi_RHS3, psi_RHS4)
        phi_new = self.rk4_merge_RHS(phi, phi_RHS1, phi_RHS2, phi_RHS3, phi_RHS4)
        
        return phi_new, psi_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        
        c = plt.pcolormesh(self.X, self.Y, self.phi, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()

        
    def wave_driver(self, phi0, save_interval=10, plot_interval=0):
        self.phi0 = phi0[:self.Nx,:self.Ny]
        self.phi = self.phi0
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{phi}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.Phi.append(self.phi)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
            self.phi, self.psi, self.t = self.wave_rk4(self.phi, self.psi, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{phi}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.Phi.append(self.phi)
                self.T.append(self.t)

        return torch.stack(self.Phi)