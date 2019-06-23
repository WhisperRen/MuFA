# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:15:41 2018

@author: ls
"""
import tkinter as tk
from GetDoc1_1 import GetDocs

# =============================================================================
#                          GUI part
# =============================================================================
# create the window
window = tk.Tk()
window.title('MuFA')
window.geometry('900x650')

# checkbutton function for analysis type
FFT_1D_Switch = 0
Dispersion_Switch = 0
Freq_colormap_Switch = 0
def check_analysis_type():
    global FFT_1D_Switch
    global Dispersion_Switch
    global Freq_colormap_Switch
    if var1.get() == 1:
        FFT_1D_Switch = 1
        e18['state'] = 'disabled'
        e20['state'] = 'normal'
        e21['state'] = 'normal'
        
    else:
        FFT_1D_Switch = 0
        e18['state'] = 'normal'
    if var1.get() == 2:
        Dispersion_Switch = 1
        c7['state'] = 'normal'
        c8['state'] = 'normal'
        e15['state'] = 'normal'
        e19['state'] = 'normal'
        e20['state'] = 'disabled'
        e21['state'] = 'disabled'
        
    else:
        Dispersion_Switch = 0
        c7['state'] = 'disabled'
        c8['state'] = 'disabled'
        e15['state'] = 'disabled'
        e19['state'] = 'disabled'
    if var1.get() == 3:
        Freq_colormap_Switch = 1
        e20['state'] = 'disabled'
        e21['state'] = 'disabled'
    else:
        Freq_colormap_Switch = 0

# if the dispersion curve with window function
win = 0
def if_win():
    global win
    if var7.get() == 1:
        win = 1
    else:
        win = 0

# create checkbutton for analysis type
var1 = tk.IntVar()
var7 = tk.IntVar()
c1 = tk.Radiobutton(window,text = 'FFT',
                    variable = var1,value = 1,
                    command = check_analysis_type,font=('Arial',14))
c2 = tk.Radiobutton(window,text = 'Dispersion Curve',
                    variable = var1,value = 2,
                    command = check_analysis_type,font=('Arial',14))
c3 = tk.Radiobutton(window,text = 'Spectrum',
                    variable = var1,value = 3,
                    command = check_analysis_type,font=('Arial',14))
c7 = tk.Checkbutton(window,text = 'Win',
                    variable = var7,onvalue = 1,offvalue = 0,
                    command = if_win,font=('Arial',14),state='disabled')
c1.place(x = 0,y = 20)
c2.place(x = 0,y = 50)
c3.place(x = 0,y = 80)
c7.place(x = 190,y = 50)

# Dir entry
e = tk.Entry(window,show = None,font=('Arial',12),width=200)
e.place(x = 290, y = 86)
l = tk.Label(window,text = 'Dir ( using \\\ or / instead of \\ )',
             font=('Arial',12),width=22,height=2).place(x = 290,y = 44)

# geometry entries 1
# colum 1
index1 = 0
createVar = locals()
for i in range(0,6):
    createVar['e'+str(i)] = tk.Entry(window,width=12,show = None,
              font=('Arial',12))
    createVar['e'+str(i)].place(x=100,y=150+index1)
    index1 += 30

labels_1 = {'l2':'X (m)','l3':'Y (m)','l4':'Z (m)',
            'l5':'cell_x (m)','l6':'cell_y (m)','l7':'cell_z (m)'}

index2 = 0
for key in labels_1:
    key = tk.Label(window,text=labels_1[key],font=('Arial',12),
                   width=10).place(x=0,y=148+index2)
    index2 += 30

# geometry entries 2
# colum 2
index3 = 0
for i in range(6,12):
    createVar['e'+str(i)] = tk.Entry(window,width=12,show = None,
              font=('Arial',12))
    createVar['e'+str(i)].place(x=420,y=150+index3)
    index3 += 30

labels_2 = {'l8':'x_start (m)','l9':'y_start (m)','l10':'z_start (m)' 
         ,'l11':'x_end (m)','l12':'y_end (m)','l13':'z_end (m)'}

index4 = 0
for key in labels_2:
    key = tk.Label(window,text=labels_2[key],font=('Arial',12),
                   width=10).place(x=320,y=148+index4)
    index4 += 30

# time and sampling period entries 
# colum 1
index5 = 0
for i in range(12,16):
    createVar['e'+str(i)] = tk.Entry(window,width=12,show = None,
              font=('Arial',12))
    createVar['e'+str(i)].place(x=101,y=400+index5)
    index5 += 30
default15 = tk.StringVar()
default15.set('0')
e15['state'] = 'disabled'
e15['textvariable'] = default15

labels_3 = {'l14':'startStage','l15':'endStage',
            'l16':'time SP (s)','l17':'space SP (m)'}

index6 = 0
for key in labels_3:
    key = tk.Label(window,text=labels_3[key],font=('Arial',12),
                   width=10).place(x=0,y=398+index6)
    index6 += 30

# checkbutton function for deciding export component
# colum 2 
export_x = 0
export_y = 0
export_z = 0
def export_component():
    global export_x
    global export_y
    global export_z
    if var4.get() == 1:
        export_x = 1
    else:
        export_x = 0
    if var4.get() == 2:
        export_y = 1
    else:
        export_y = 0
    if var4.get() == 3:
        export_z = 1
    else:
        export_z = 0

Resample_Switch = 0
def resample():
    global Resample_Switch
    if var8.get() == 1:
        Resample_Switch = 1
        e16['state'] = 'normal'
        e17['state'] = 'normal'
    else:
        Resample_Switch = 0
        e16['state'] = 'disabled'
        e17['state'] = 'disabled'
# create checkbutton for deciding export component and re-sampling
var4 = tk.IntVar()
var8 = tk.IntVar()
c4 = tk.Radiobutton(window,text = 'Export_x',
                    variable = var4,value = 1,
                    command = export_component,font=('Arial',14))
c5 = tk.Radiobutton(window,text = 'Export_y',
                    variable = var4,value = 2,
                    command = export_component,font=('Arial',14))
c6 = tk.Radiobutton(window,text = 'Export_z',
                    variable = var4,value = 3,
                    command = export_component,font=('Arial',14))

c8 = tk.Checkbutton(window,text = 'Re-sampling',
                    variable = var8,onvalue = 1,offvalue = 0,
                    command = resample,font=('Arial',14),state='disabled')
c4.place(x = 640,y = 395)
c5.place(x = 640,y = 425)
c6.place(x = 640,y = 455)
c8.place(x = 325,y = 390)

# create entry for resample
default16 = tk.StringVar()
default16.set('0')
default17 = tk.StringVar()
default17.set('0')
e16 = tk.Entry(window,width=12,textvariable=default16,
               show = None,font=('Arial',12),state='disabled')
e17 = tk.Entry(window,width=12,textvariable=default17,
               show = None,font=('Arial',12),state='disabled')
e16.place(x = 420,y = 430)
e17.place(x = 420,y = 460)
tk.Label(window,text='T-resample',font=('Arial',12),
         width=10).place(x=320,y=428)
tk.Label(window,text='S-resample',font=('Arial',12),
         width=10).place(x=320,y=458)

# create entry for cutoff freq and klim
tk.Label(window,text='f_min (GHz)',font=('Arial',12),
         width=10).place(x=640,y=208)
tk.Label(window,text='f_max (GHz)',font=('Arial',12),
         width=10).place(x=640,y=238)
tk.Label(window,text='f_cutoff (GHz)',font=('Arial',12),
         width=10).place(x=640,y=268)
tk.Label(window,text='k-lim (nm-1)',font=('Arial',12),
         width=10).place(x=640,y=298)

default18 = tk.StringVar()
default18.set('0')
default19 = tk.StringVar()
default19.set('0')
e18 = tk.Entry(window,width=12,textvariable=default18,
               show = None,font=('Arial',12),state='disabled')
e19 = tk.Entry(window,width=12,textvariable=default19,
               show = None,font=('Arial',12),state='disabled')
e18.place(x = 740, y = 270)
e19.place(x = 740, y = 300)

default20 = tk.StringVar()
default20.set('0')
default21 = tk.StringVar()
default21.set('0')
e20 = tk.Entry(window,width=12,textvariable=default20,
               show = None,font=('Arial',12),state='disabled')
e21 = tk.Entry(window,width=12,textvariable=default21,
               show = None,font=('Arial',12),state='disabled')
e20.place(x = 740, y = 210)
e21.place(x = 740, y = 240)
# =============================================================================
# =============================================================================
'''
unadded function
'''

# =============================================================================
# =============================================================================
Dir = ''
X,Y,Z = 0.0, 0.0, 0.0
x_start, y_start, z_start = 0.0, 0.0, 0.0
A, B, C = 0.0, 0.0, 0.0
x_end, y_end, z_end = 0.0, 0.0, 0.0
startStage,endStage = int(0),int(0)
T1 = 0.0
X_period = 0.0
T_resample, X_resample = 0, 0
f_cutoff, klim = 100, 0.6
# button function for assigning parameters from GUI to import functions
def assign(FFT_1D_Switch,Dispersion_Switch,Freq_colormap_Switch):
    global Dir, X,Y,Z, x_start, y_start, z_start, A, B, C
    global x_end, y_end, y_end, startStage,endStage, T1, X_period
    global Mx, My, Mz
    global T_resample, X_resample
    global f_cutoff, klim
    Dir = str(e.get())
    X, Y, Z = float(e0.get()), float(e1.get()), float(e2.get())
    A, B, C = float(e3.get()),float(e4.get()),float(e5.get())
    x_start, y_start, z_start = float(e6.get()), float(e7.get()), float(e8.get())
    x_end, y_end, z_end = float(e9.get()), float(e10.get()), float(e11.get())
    startStage,endStage = int(e12.get()), int(e13.get())
    T1, X_period = float(e14.get()), float(e15.get())
    T_resample, X_resample = int(e16.get()), int(e17.get())
    f_cutoff, klim = float(e18.get()), float(e19.get())
    f_min, f_max = float(e20.get()), float(e21.get())
    Mx,My,Mz = GetDocs(Dir,startStage,endStage,X,Y,Z,x_start,y_start,z_start,
                       x_end,y_end,z_end,A,B,C,export_x,export_y,export_z)
    if FFT_1D_Switch == 1:
        from FFT import fft3
        # the f_min,F_max has not been added to the GUI
        fft3(Mx,My,Mz,export_x,export_y,export_z,T1,f_min,f_max)
    else:
        FFT_1D_Switch = FFT_1D_Switch
    
    if Dispersion_Switch == 1:
        from Dispersion import fftsc
        fftsc(Mx,My,Mz,export_x,export_y,export_z,T1,X_period,
              T_resample,X_resample,win,Resample_Switch,f_cutoff,
              klim,x_start,x_end,A)
    else:
        Dispersion_Switch = Dispersion_Switch
    if Freq_colormap_Switch == 1:
        from Spectrum import freq_cp
        freq_cp(Mx,My,Mz,export_x,export_y,export_z,T1,
                f_cutoff,X,x_start,x_end,A)
    else:
        Freq_colormap_Switch = Freq_colormap_Switch

if __name__ == '__main__':
    # create button for 'run' function
    c = tk.Button(window,text='OK',width=8,height=2,font=('Arial',14),
                  command = lambda: assign(FFT_1D_Switch,Dispersion_Switch,
                                           Freq_colormap_Switch))
    c.place(x = 640, y = 530)

    window.mainloop()
