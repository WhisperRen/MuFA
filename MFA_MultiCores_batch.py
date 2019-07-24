# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:35:46 2018

@author: ls
"""
# =============================================================================
#                    Dir (using \\ or / instead of \)
# =============================================================================
Dir = 'E:\\MuFA\\MFA'
destination = 'E:\\MuFA\\MFA'

# =============================================================================
#                  General geometry parameters (in meter)
# =============================================================================
X = 1500e-9
Y = 30e-9
Z = 10e-9

x_cell = 1.5e-9
y_cell = 1.5e-9
z_cell = 10e-9

x_start, x_end = 500e-9,1400e-9
y_start, y_end = 13.5e-9,16.5e-9
z_start, z_end = 0,10e-9

# =============================================================================
#    General time parameters (integer, 'SP'=sampling period, in second) 
# =============================================================================
StartStage = 1
EndStage = 2500

time_SP = 2e-12

# =============================================================================
#                    FFT particular parameters (in GHz)
# =============================================================================
f_min = 0
f_max = 80

# =============================================================================
#                 Dispersion curve particular parameters
# =============================================================================
# space_SP (in meter)
space_SP = 1.5e-9
# f_cutoff (in GHz)
f_cutoff_dispersion = 80
#   klim   (in nm^-1)
klim = 0.4
# xxx_resample (integer)
time_resample = 1
space_resample = 1

# =============================================================================
#              Frequency spectrum particular parameters (in GHz)
# =============================================================================
f_cutoff_colormap = 80

# =============================================================================
#                   General control ('1'= on, '0'= off)
# =============================================================================
export_x = 0
export_y = 0
export_z = 1

FFT_Switch = 0
Dispersion_Switch = 0
Spectrum_Switch = 1

# sub-control for dispersion curve function
win = 0
resampling = 0

# =============================================================================
#      Number of CPU cores used (if equals 0, means using all cores)
# =============================================================================
CoreNum = 3



# =============================================================================
# =============================================================================
# =============================================================================
#  Below are functional codes. For normal execution, do not modify them.
# =============================================================================
# =============================================================================
# =============================================================================
from functools import partial
import numpy as np
import os
from multiprocessing import Pool

X, Y, Z = float(X),float(Y),float(Z)
A,B,C = float(x_cell),float(y_cell),float(z_cell)
x_start,x_end = float(x_start),float(x_end)
y_start,y_end = float(y_start),float(y_end)
z_start,z_end = float(z_start),float(z_end)
StartStage,EndStage = int(StartStage),int(EndStage)
T1,X_SP = float(time_SP),float(space_SP)
f_cutoff_dispersion, klim = float(f_cutoff_dispersion),float(klim)
f_cutoff_colormap = float(f_cutoff_colormap)
T_resample,X_resample = int(time_resample),int(space_resample)

def fileload(name,colum):
    colum = int(colum)
    file = np.loadtxt(name, dtype = float,
                      comments = '#', usecols = (colum,))
    return file

def multicore(dirc,startstage,endstage,
              exportX,exportY,exportZ,num):
    if exportZ == 1:
        column = '2'
    elif exportY == 1:
        column = '1'
    else:
        column = '0'

    files = []
    for i,j,k in os.walk(dirc):
        files.append(k)
    files = files[0]
    files.sort()
    files_sort = files[2:]
    files_new = []
    for i in range(startstage-1,endstage):
        files_new.append(dirc+ '/' + files_sort[i])
    if num == 0:
        pool = Pool()
        matrices = pool.map(partial(fileload,colum=column),files_new)
        pool.close()
        pool.join()
    else:
        pool = Pool(processes = num)
        matrices = pool.map(partial(fileload,colum=column),files_new)
        pool.close()
        pool.join()
    return matrices

if __name__ == '__main__':
    subDir = []
    for i, j, k in os.walk(Dir):
        subDir.append(j)
    subDir = subDir[0]
    for index in range(len(subDir)):
        subDir[index] = Dir + '/' + subDir[index]
    
    for address in enumerate(subDir):
        matrices = multicore(address[1],StartStage,EndStage,
                             export_x,export_y,export_z,CoreNum)
        
        from GetDoc2_0 import GetDocs
        Mx,My,Mz = GetDocs(matrices,StartStage,EndStage,X,Y,Z,x_start,
                           y_start,z_start,x_end,y_end,z_end,A,B,C,
                           export_x,export_y,export_z)
    
        if FFT_Switch == 1:
            from FFT import fft3
            fft3(Mx,My,Mz,export_x,export_y,export_z,T1,f_min,f_max,
                 destination,address[0])
        else:
            FFT_Switch = FFT_Switch
            
        if Dispersion_Switch == 1:
            from Dispersion import fftsc
            fftsc(Mx,My,Mz,export_x,export_y,export_z,T1,X_SP,
                  T_resample,X_resample,win,resampling,
                  f_cutoff_dispersion,klim,x_start,x_end,A,
                  destination,address[0])
        else:
            Dispersion_Switch = Dispersion_Switch
        if Spectrum_Switch == 1:
            from Spectrum import freq_cp
            freq_cp(Mx,My,Mz,export_x,export_y,export_z,T1,
                    f_cutoff_colormap,X,x_start,x_end,A,
                    destination,address[0])
        else:
            Spectrum_Switch = Spectrum_Switch
        
        from gc import collect
        del matrices,Mx,My,Mz
        collect()
