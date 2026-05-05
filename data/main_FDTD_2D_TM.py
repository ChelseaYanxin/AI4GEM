# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:03:15 2019

@author: yangzf
"""

from data.constants_CPU import floattype, complextype
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0
from scipy.constants import speed_of_light as c0

import time

nmax=2000
dx = 1.0E-3
dy = 1.0E-3

dt = 0.99 / (c0*(1.0/dx**2+1.0/dy**2)**0.5)

nx=100
ny=100

# make the source point at the center
Is = round(nx/2-1)
Js = round(ny/2-1)

EM_2D_mode = "TM"

if EM_2D_mode == "TE":
    Ex = np.zeros((nx, ny + 1), dtype=floattype)
    Ey = np.zeros((nx + 1, ny), dtype=floattype)
    Hz = np.zeros((nx, ny), dtype=floattype)
     
if EM_2D_mode == "TM":
    Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)
    Hx = np.zeros((nx + 1, ny), dtype=floattype)
    Hy = np.zeros((nx, ny + 1), dtype=floattype)

################
## coefficients
################

CA_Ex = np.zeros((nx, ny + 1), dtype=floattype)
CB_Ex = np.zeros((nx, ny + 1), dtype=floattype)

CA_Ey = np.zeros((nx + 1, ny), dtype=floattype)
CB_Ey = np.zeros((nx + 1, ny), dtype=floattype)


sig_x = np.zeros((nx, ny + 1), dtype=floattype)
sig_y = np.zeros((nx + 1, ny), dtype=floattype)

CA_Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)
CB_Ez = np.zeros((nx + 1, ny + 1), dtype=floattype)


sig_z = np.zeros((nx + 1, ny + 1), dtype=floattype)

eps = np.zeros((nx + 1, ny + 1), dtype=floattype)

DA = 1
DB = (dt/(m0))

for i in range(0, nx):
    for j in range (0, ny):
#        sig_x[i,j] = 0 
#        sig_y[i,j] = 0 
        sig_z[i,j] = 0 
        eps[i,j] = 1*e0
#        CA_Ex[i,j] = (1-sig_x[i,j]*dt/(2*eps[i,j]))/(1+sig_x[i,j]*dt/(2*eps[i,j]))
#        CB_Ex[i,j] = (dt/eps[i,j])/(1+sig_x[i,j]*dt/(2*eps[i,j]))
#        CA_Ey[i,j] = (1-sig_x[i,j]*dt/(2*eps[i,j]))/(1+sig_y[i,j]*dt/(2*eps[i,j]))
#        CB_Ey[i,j] = (dt/eps[i,j])/(1+sig_x[i,j]*dt/(2*eps[i,j]))


        CA_Ez[i,j] = (1-sig_z[i,j]*dt/(2*eps[i,j]))/(1+sig_z[i,j]*dt/(2*eps[i,j]))
        CB_Ez[i,j] = (dt/eps[i,j])/(1+sig_z[i,j]*dt/(2*eps[i,j]))



# Source RHSden_ex = np.zeros((nx), dtype=floattype)
den_hx = np.zeros((nx), dtype=floattype)
den_hx[:] = 1/dy

den_hy = np.zeros((ny), dtype=floattype)
den_hy[:]  = 1/dy

den_ey = np.zeros((ny), dtype=floattype)
den_ey [:] = 1/dy
  
den_ex = np.zeros((nx), dtype=floattype)
den_ex[:]  = 1/dx

rtau=50.0e-12
tau=rtau/dt
ndelay=3*tau
srcconst=-dt*(3.0e11)
source = np.zeros((nmax+1), dtype=floattype)

for n in range(1, nmax+1):
    source[n]= (n-ndelay)*(np.exp(-((n-ndelay)**2/(tau**2))))

RecordEz = np.zeros((nmax+1), dtype=floattype)

n = 0
for n in range(1,nmax+1): 
    t = time.time()
    
    for i in range(0, nx):
        for j in range(0, ny):
#            Hz[i, j] = DA * Hz[i, j] - DB * (Ey[i + 1, j] - Ey[i, j]) * den_hx[i] + DB * (Ex[i, j + 1] - Ex[i, j]) * den_hy[j]
            Hx[i, j] = DA * Hx[i, j] - DB * (Ez[i, j + 1] - Ez[i, j]) * den_hy[j]
            Hy[i, j] = DA * Hy[i, j] + DB * (Ez[i + 1, j] - Ez[i, j]) * den_hx[i]
                
    for i in range(1, nx):
        for j in range(1, ny):
#            Ex[i, j] = CA_Ex[i,j] * Ex[i, j] + CB_Ex[i,j] * (Hz[i, j] - Hz[i, j - 1]) * den_ey[j]
            Ez[i, j] = CA_Ez[i,j] * Ez[i, j] + CB_Ez[i,j] * ( (Hy[i, j] - Hy[i - 1, j]) * den_ex[i] - (Hx[i, j] - Hx[i, j - 1]) * den_ey[j])


    elapsed = time.time() - t
    print(elapsed)
    
    #Source
    Ez[Is, Js] = Ez[Is, Js] + CB_Ez[i,j] * source[n]  
    
    RecordEz[n] = Ez[Is, Js]  
    print(n)   
    print(Ez[20, 20])  

    if np.remainder(n,1) == 0:
        plt.clf()
#        plt.subplot(221)
        plt.imshow(Hx[:, :])
        plt.title("FDTD 2D TE mode, Time Step: "+str(n))
        plt.colorbar()
#        plt.pause(0.05)
#        plt.subplot(222)
#        plt.imshow(Ex[:, :])
#        plt.colorbar()
#        plt.pause(0.05)
#        plt.subplot(223)
#        plt.imshow(Ey[:, :])
#        plt.colorbar()
        plt.draw()
        plt.pause(0.05)


