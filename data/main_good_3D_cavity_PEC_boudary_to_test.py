# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 00:11:41 2018

@author: yangzf
"""

from constants import floattype, complextype
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0


 
#update E H coefficients
c0 = (2.99792458e8)

nmax=500
dx=(2e-3)
dy=(2e-3)
dz=(2e-3)

nx=20
ny=20
nz=20

Is = 10-1
Js = 10-1
Ks = 10-1

Ex = np.zeros((nx, ny + 1, nz + 1), dtype=floattype)
Ey = np.zeros((nx + 1, ny, nz + 1), dtype=floattype)
Ez = np.zeros((nx + 1, ny + 1, nz), dtype=floattype)
Hx = np.zeros((nx + 1, ny, nz ), dtype=floattype)
Hy = np.zeros((nx, ny + 1, nz), dtype=floattype)
Hz = np.zeros((nx, ny, nz + 1), dtype=floattype)

updatecoeffsH = np.zeros((6, 6), dtype=floattype)
updatecoeffsE = np.zeros((6, 6), dtype=floattype)

ID =  np.zeros((6,nx + 1, ny + 1, nz + 1), dtype=np.int32)

dt = 0.99 / (c0*np.sqrt(1.0/(dx*dx)+1.0/(dy*dy)+1.0/(dz*dz)))
dt = dx/(2.0*c0)

#update coefficients

# Default material constitutive parameters (free_space)
er = 1.0
se = 0.0
mr = 1.0
sm = 0.0




def update_H_coeff(IDnum,dx,dy,dz,dt,sm):
    if IDnum == 1:
        mr = 1
        sm = 0
        HA = (m0 * mr / dt) + 0.5 * sm
        HB = (m0 * mr / dt) - 0.5 * sm
        DA = HB / HA
        DBx = (1 / dx) * 1 / HA
        DBy = (1 / dy) * 1 / HA
        DBz = (1 / dz) * 1 / HA
        srcm = 1 / HA
        updatecoeffsH[IDnum, 0] = DA
        updatecoeffsH[IDnum, 1] = DBx
        updatecoeffsH[IDnum, 2] = DBy
        updatecoeffsH[IDnum, 3] = DBz

def update_E_coeff(IDnum,dx,dy,dz,dt,se):
    if IDnum == 1:
        er = 1
        se = 0
        EA = (e0 * er / dt) + 0.5 * se
        EB = (e0 * er / dt) - 0.5 * se
        CA = EB / EA
        CBx = (1 / dx) * 1 / EA
        CBy = (1 / dy) * 1 / EA
        CBz = (1 / dz) * 1 / EA
        srce = 1 / EA
        updatecoeffsE[IDnum, 0] = CA
        updatecoeffsE[IDnum, 1] = CBx
        updatecoeffsE[IDnum, 2] = CBy
        updatecoeffsE[IDnum, 3] = CBz
        updatecoeffsE[IDnum, 4] = srce
    if IDnum == 2:
        CA = 0
        CBx = 0
        CBy = 0
        CBz = 0
        srce = 0
        updatecoeffsE[IDnum, 0] = CA
        updatecoeffsE[IDnum, 1] = CBx
        updatecoeffsE[IDnum, 2] = CBy
        updatecoeffsE[IDnum, 3] = CBz


#construct the ID E
for i in range(0, nx+1):
    for j in range(0, ny+1):
        for k in range(0, nz+1):
            ID[0, i, j, k] = 1
            ID[1, i, j, k] = 1
            ID[2, i, j, k] = 1
            update_E_coeff(ID[0, i, j, k],dx,dy,dz,dt,0)
            update_E_coeff(ID[1, i, j, k],dx,dy,dz,dt,0)
            update_E_coeff(ID[2, i, j, k],dx,dy,dz,dt,0)
#for j in range(1, ny):
#    for k in range(1, nz):
#        ID[0, 0, j, k] = 1
#
#for i in range(1, nx):
#    for k in range(1, nz):
#        ID[1, i, 0, k] = 1
#        
#for i in range(1, nx):
#    for j in range(1, ny):
#        ID[2, i, j, 0] = 1
        
#construct the ID H
for i in range(0, nx+1):
    for j in range(0, ny+1):
        for k in range(0, nz+1):
            ID[3, i, j, k] = 1
            ID[4, i, j, k] = 1
            ID[5, i, j, k] = 1
            update_H_coeff(ID[3, i , j, k],dx,dy,dz,dt,0)
            update_H_coeff(ID[4, i , j ,k],dx,dy,dz,dt,0)
            update_H_coeff(ID[5, i , j, k],dx,dy,dz,dt,0)
            
# 1 free space        # 2 PEC

#for i in range(0, nx+1):
#    for j in range(0, ny+1):
#        for k in range(0, nz+1):
#            if i==0:
#                ID[0, 0, j, k] = 2
#                update_E_coeff(ID[0, 0, j, k],dx,dy,dz,dt,0)
#                
#            
#for j in range(0, ny+1):
#    for k in range(0, nz+1):
#        ID[0, 0, j, k] = 2
#        update_E_coeff(ID[0, 0, j, k],dx,dy,dz,dt,0)
#        ID[0, nx, j, k] = 2
#        update_E_coeff(ID[0, nx, j, k],dx,dy,dz,dt,0)
#
#for i in range(0, nx+1):
#    for k in range(0, nz+1):
#        ID[1, i, 0, k] = 2
#        update_E_coeff(ID[1, i, 0, k],dx,dy,dz,dt,0)
#        ID[1, i, ny, k] = 2
#        update_E_coeff(ID[1, i, ny, k],dx,dy,dz,dt,0)
#        
#for i in range(0, nx+1):
#    for j in range(0, ny+1):
#        ID[2, i, j, 0] = 2         
#        update_E_coeff(ID[2, i, j, 0],dx,dy,dz,dt,0)
#        ID[2, i, j, nz] = 2  
#        update_E_coeff(ID[2, i, j, nz],dx,dy,dz,dt,0)
        
    
# Source RHS
rtau=(50.0e-12)
tau=rtau/dt
ndelay=3*tau
srcconst=-dt*(3.0e11)
source = np.zeros((nmax+1), dtype=floattype)
for n in range(1, nmax+1):
    source[n]= srcconst*(n-ndelay)*(np.exp(-((n-ndelay)**2/(tau**2))))
    
RecordEz = np.zeros((nmax+1), dtype=floattype)
n = 0
for n in range(1,nmax+1): 
       
    # update the electric fields
    for i in range(0, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                materialEx = ID[0, i, j, k]
                Ex[i, j, k] = updatecoeffsE[materialEx, 0] * Ex[i, j, k] + updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])

    # Ex components at i = 0
#    for j in range(0, ny):
#        for k in range(0, nz):
#            materialEx = ID[0, 0, j, k]
#            Ex[0, j, k] = updatecoeffsE[materialEx, 0] * Ex[0, j, k] + updatecoeffsE[materialEx, 2] * (Hz[0, j, k] - Hz[0, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[0, j, k] - Hy[0, j, k - 1])
#    

    for i in range(1, nx):
        for j in range(0, ny):
            for k in range(1, nz):
                materialEy = ID[1, i, j, k]
                Ey[i, j, k] = updatecoeffsE[materialEy, 0] * Ey[i, j, k] + updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])

    # Ey components at j = 0
#    for i in range(0, nx):
#        for k in range(0, nz):
#            materialEy = ID[1, i, 0, k]
#            Ey[i, 0, k] = updatecoeffsE[materialEy, 0] * Ey[i, 0, k] + updatecoeffsE[materialEy, 3] * (Hx[i, 0, k] - Hx[i, 0, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, 0, k] - Hz[i - 1, 0, k])
#    

    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(0, nz):
                materialEz = ID[2, i, j, k]
                Ez[i, j, k] = updatecoeffsE[materialEz, 0] * Ez[i, j, k] + updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])
    
    # Ez components at k = 0
#    for i in range(0, nx):
#        for j in range(0, ny):
#            materialEz = ID[2, i, j, 0]
#            Ez[i, j, 0] = updatecoeffsE[materialEz, 0] * Ez[i, j, 0] + updatecoeffsE[materialEz, 1] * (Hy[i, j, 0] - Hy[i - 1, j, 0]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, 0] - Hx[i, j - 1, 0])
  
    #Source
    Ez[Is, Js, Ks] = Ez[Is, Js, Ks] + source[n]
    
    # update the magnetic fields
    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                materialHx = ID[3, i ,j, k]
                materialHy = ID[4, i, j, k]
                materialHz = ID[5, i, j, k]
                Hx[i, j, k] = updatecoeffsH[materialHx, 0] * Hx[i, j, k] - updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) + updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k])
                Hy[i, j, k] = updatecoeffsH[materialHy, 0] * Hy[i, j, k] - updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) + updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k])
                Hz[i, j, k] = updatecoeffsH[materialHz, 0] * Hz[i, j, k] - updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) + updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k])
#    n = n + 1
    RecordEz[n] = Ez[11, 11, 9]
    print(n)   
    print(Ez[11, 11, 9])
    
    if n==1:
        fig = plt.figure(1)
        plt.imshow(Ez[:, :, 4])
        plt.colorbar()
        plt.show()
    if np.remainder(n,5) == 0:
        plt.imshow(Ez[:, :, 10])
        plt.pause(0.1)
        plt.show()



