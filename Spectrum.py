# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 19:10:06 2018

@author: ls
"""

import numpy as np
import matplotlib.pyplot as plt

def freq_cp(Mx,My,Mz,export_x,export_y,export_z,T,
            f_cutoff,X,x0,x1,cell):
    if export_z == 1:
        M0_matrix = Mz
    elif export_y == 1:
        M0_matrix = My
    else:
        M0_matrix = Mx
    Ms = 1e3
    M0_matrix = M0_matrix*Ms
    M, n = M0_matrix.shape
    N = int((x1-x0)/cell)
    M_matrix = np.zeros((M,N))
    start = 0
    end = N
    count = 0
    aveMatrix = np.zeros((M,N))
    while end <= n:
        M_matrix = M0_matrix[:,start:end]
        
        fs = 1 / T / (1e9)
        #n = np.linspace(0,M-1,M)
        fftMatrix = np.zeros((M,N))
        #aveMatrix = np.zeros((1,N))
        for i in range(0,N):
            fftMatrix[:,i] = np.abs(np.fft.fft(M_matrix[:,i]))
        aveMatrix = aveMatrix + fftMatrix
        start += N
        end += N
        count += 1
    aveMatrix = aveMatrix / count
    aveMatrix = 10 * np.log10(aveMatrix / np.max(aveMatrix))
    
    # elimate reflection
    M_mean = np.mean(aveMatrix)
    M_max = np.max(aveMatrix)
    aveMatrix = np.clip(aveMatrix,M_mean,M_max)
    
    X_neglim, X_poslim = x0/X*100, x1/X*100
    Y_neglim, Y_poslim = 0, fs
    extent = [X_neglim,X_poslim,Y_neglim,Y_poslim]
    plt.figure()
    plt.rcParams['figure.figsize'] = (9.0,8.0)
    plt.imshow(aveMatrix, cmap = plt.cm.jet, origin='upper',extent=extent)
    plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       ['0.1', '0.2', '0.3', '0.4','0.5', '0.6','0.7','0.8','0.9','1.0'], \
       fontsize = 19)
    plt.yticks(fontsize = 19)
    plt.ylim(0,f_cutoff)
    plt.xlim(X_neglim,X_poslim)
    plt.colorbar(shrink = 0.5)
    plt.xlabel(r'$\mathrm{Normalized\enspace distance\enspace (a.u.)}$',
                          fontsize = 19)
    plt.ylabel(r'$\mathrm{Frequency\enspace (GHz)}$',fontsize = 19)
    plt.savefig('Frequency spectrum.eps', dpi = 500)
    plt.show()
    return aveMatrix
    
    