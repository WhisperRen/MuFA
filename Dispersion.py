# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:18:50 2018

@author: zhiwei ren
"""

import numpy as np
import matplotlib.pyplot as plt

def fftsc(Mx,My,Mz,export_x,export_y,export_z,T,X,
          T_resample,X_resample,win,Resample_Switch,
          f_cutoff,klim,x0,x1,cell):
    if export_z == 1:
        M0_matrix = Mz
    elif export_y == 1:
        M0_matrix = My
    else:
        M0_matrix = Mx
    M, n = M0_matrix.shape
    N = int((x1-x0)/cell)
    M_matrix = np.zeros((M,N))
    start = 0
    end = N
    count = 0
    while end <= n:
        M_matrix = M_matrix + M0_matrix[:,start:end]
        start += N
        end += N
        count += 1
    M_matrix = M_matrix / count
    
    Ms = 1e3
    M_matrix = M_matrix * Ms
    fs, ks = 1 / T, 1 / (2*X)
    M_fft, N_fft = M, N
    Matrix = np.zeros((M_fft,N_fft))
    
    # resampling
    if Resample_Switch == 1:
        M_fft, N_fft = round(M / T_resample), round(N / X_resample)
        j = 0
        for i in range(0,N-1):
            if np.mod(i,X_resample) == 0:
                Matrix[:,j] = M_matrix[:,i]
                j += 1
    else:
        Matrix = M_matrix
    
    # window function
    if win == 1:
        from scipy.signal import chebwin as chebwin
        attenuation = 50.0
        win1 = chebwin(M_fft,attenuation)
        win2 = chebwin(N_fft,attenuation)
        win1 = win1[np.newaxis,:]
        win2 = win2[np.newaxis,:]
        win1 = win1.T
        win = np.dot(win1, win2)
        Matrix = Matrix * win
    else:
        win = win
    
    # 2D FFT
    fftMatrix = np.zeros((M_fft,N_fft))
    fftMatrix = np.fft.fft2(Matrix)
    fftMatrix = np.abs(np.fft.fftshift(fftMatrix))
    fs = fs / (2e9)
    ks = -2*np.pi*ks/(1e9)
    fftMatrix = fftMatrix[0:round(M_fft/2), 0:N_fft-1]
    fftMatrix = 10 * np.log10(fftMatrix / np.max(fftMatrix))
    
    # elimate reflection
    M_mean = np.mean(fftMatrix)
    M_max = np.max(fftMatrix)
    fftMatrix = np.clip(fftMatrix,M_mean,M_max)
    
    # image show
    X_neglim, X_poslim = ks*2e2, -ks*2e2
    Y_neglim, Y_poslim = 0, fs
    extent = [X_neglim,X_poslim,Y_neglim,Y_poslim]
    plt.figure()
    plt.rcParams['figure.figsize'] = (9.0,8.5)
    plt.imshow(fftMatrix, cmap = plt.cm.jet, origin='upper',extent=extent)
    
    klim = klim*2e2
    if klim in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
        plt.xticks([20, 40, 60, 80, 100, 
                    120, 140, 160, 180, 200],
           ['0.1', '0.2', '0.3', '0.4','0.5', '0.6',
            '0.7','0.8','0.9','1.0'],fontsize = 18)
    elif klim in [2, 6, 10, 14, 18]:
        plt.xticks([2, 6, 10, 14, 18],
           ['0.01', '0.03', '0.05','0.07','0.09'],fontsize = 18)
    else:
        plt.xticks([4, 8, 12, 16],
           ['0.02', '0.04', '0.06','0.08'],fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlim(0,klim)
    plt.ylim(0,f_cutoff)
    plt.colorbar(shrink = 0.5)
    plt.xlabel(r"$\mathrm{Wave\enspace vector}\enspace k_x\enspace \mathrm{(nm^{-1})}$",
               fontsize = 17)
    plt.ylabel(r'$\mathrm{Frequency\enspace (GHz)}$',fontsize = 17)
    plt.savefig('Dispersion curve.eps', dpi=500)
    plt.show()
    return fftMatrix

if __name__ == '__main__':
    x = y = np.arange(-3.0,3.0,0.025)
    X,Y = np.meshgrid(x,y)
    Z1 = np.exp(-X**2-Y**2)
    Z2 = np.exp(-(X-1)**2 - (Y-1)**2)
    Z = (Z1 - Z2)*2
    fftsc(1,1,Z,0,0,1,0.025,0.025,1,1,0,1)