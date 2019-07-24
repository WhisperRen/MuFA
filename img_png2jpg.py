# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:37:36 2019

@author: ls
"""

from PIL import Image
import os

Dir = 'E:\\Python36\\Object_detection_data\\magnonic_for_CV\\dispersion_D3_jet'
save_Dir = 'C:\\Users\\ls\\Desktop\\fig'

files = []
for i,j,k in os.walk(Dir):
    files.append(k)
files = files[0]

for i in files:
    im = Image.open(Dir+'\\'+i)
    im = im.convert('RGB')
    im.save(save_Dir + '\\' + i[:-3]+'jpg')