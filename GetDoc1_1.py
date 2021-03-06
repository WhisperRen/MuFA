# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:32:40 2018

@author: zhiwei ren
"""
import os
import numpy as np
import time

def display_time(func):
    def wrapper(*args):
        t1 = time.time()
        result = func(*args)
        t2 = time.time()
        print('Total time: {:.6} s'.format(t2 - t1))
        return result
    return wrapper

def getfiles(dirc0):
    files = []
    for i,j,k in os.walk(dirc0):
        files.append(k)
    return files

def fileload(dirc,files,i,colum):
    file = np.loadtxt(str(dirc + '\\' + files[0][i+1]), dtype = float,
                      comments = '#', usecols = (colum,))
    return file

def kernel(x_num,y_num,z_num,a,b,c,x0,y0,z0,M,N,
           M_matrix,file,i,startstage):
    for s in range(1,z_num+1):
        for t in range(1,y_num+1):
            for k in range(1,x_num+1):
                y1 = int((s-1) * x_num * y_num + (t-1) * x_num + k)
                y2 = int(x0/a + k + (y0/b + t - 1) * M + (z0/c + s - 1)*M*N)
                M_matrix[i - startstage, y1 - 1] = file[y2-1]
    return M_matrix

@display_time
def GetDocs(dirc,startstage,endstage,x,y,z,x0,y0,z0,x1,y1,z1,a,b,c,
            exportX,exportY,exportZ):
    # total cell number
    M, N = x / a, y / b
    # desired cell number
    x_num = int(x1/a - x0/a)
    y_num = int(y1/b - y0/b)
    z_num = int(z1/c - z0/c)
    # desired time duration and cell number
    mclum = endstage - startstage
    nclum = x_num * y_num * z_num
    
    if exportX == 1:
        Mx_matrix = np.zeros((mclum+1,nclum))
    else:
        Mx_matrix = 0
    
    if exportY == 1:
        My_matrix = np.zeros((mclum+1,nclum))
    else:
        My_matrix = 0
    
    if exportZ == 1:
        Mz_matrix = np.zeros((mclum+1,nclum))
    else:
        Mz_matrix = 0
    
    # get files name list
    files = getfiles(dirc)
    
    # main loop
    for i in range(startstage,endstage+1):
        
        if exportZ == 1:
            colum = 2
            file = fileload(dirc,files,i,colum)
            Mz_matrix = kernel(x_num,y_num,z_num,a,b,c,x0,y0,z0,M,N,
                               Mz_matrix,file,i,startstage)
        else:
            Mz_matrix = Mz_matrix
        
        if exportY == 1:
            colum = 1
            file = fileload(dirc,files,i,colum)
            My_matrix = kernel(x_num,y_num,z_num,a,b,c,x0,y0,z0,M,N,
                               My_matrix,file,i,startstage)
        else:
            My_matrix = My_matrix
        
        if exportX == 1:
            colum = 0
            file = fileload(dirc,files,i,colum)
            Mx_matrix = kernel(x_num,y_num,z_num,a,b,c,x0,y0,z0,M,N,
                               Mx_matrix,file,i,startstage)
        else:
            Mx_matrix = Mx_matrix
        
    return Mx_matrix, My_matrix, Mz_matrix

if __name__ == '__main__':
    
    GetDocs(dirc,startstage,endstage,x,y,z,x0,y0,z0,x1,y1,z1,a,b,c,
            exportX,exportY,exportZ)