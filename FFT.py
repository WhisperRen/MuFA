# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:01:13 2018

@author: zhiwei ren
"""

import numpy as np
import matplotlib.pyplot as plt

def fft3(Mx,My,Mz,export_x,export_y,export_z,T,f_min,f_max):
    if export_z == 1:
        M_matrix = Mz
    elif export_y == 1:
        M_matrix = My
    else:
        M_matrix = Mx
    M, N = M_matrix.shape
    fs = 1 / T
    n = np.linspace(0,M-51,M-50)
    wn = fs * n / M
    fftMatrix = np.zeros((N,M))
    aveMatrix = np.zeros((1,N))
    for i in range(0,N):
        fftMatrix[i,:] = np.abs(np.fft.fft(M_matrix[:,i]))
    aveMatrix = np.mean(fftMatrix[:,0:M-50], axis = 0,keepdims = True)
    aveMatrix = aveMatrix / np.max(aveMatrix)
    plt.figure()
    plt.rcParams['figure.figsize'] = (9.0,8.5)
    wn = wn/1e9
    plt.plot(wn,aveMatrix.T,linewidth = 2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'$\mathrm{Frequency\enspace (GHz)}$',fontsize=18)
    plt.ylabel(r'$\mathrm{Normalized\enspace FFT\enspace power\enspace (arb.units)}$',
               fontsize=18)
    plt.xlim((f_min,f_max))
    plt.savefig('FFT power.eps',dpi = 500)
    plt.show()
    return aveMatrix

if __name__ == '__main__':
    M = np.linspace(0, 2 * np.pi, 30).reshape((5,6))
    wave = np.cos(M)
    T = 1
    fft3(M,T)
    print('hello')