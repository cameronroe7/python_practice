# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:01:47 2021

@author: Cameron
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft


def rossler(t,x,epsilon):
    return [-(x[1]+x[2]), x[0] + .2 * x[1], .2 + (x[0] - epsilon)*x[2]]

x0 = np.array([2,2,0])
tspan = np.array([0,5000])

epsilons = np.array([2.25,3.1,4,4.8])
#epsilons = np.array([2])

for idx,epsilon in enumerate(epsilons):
    sol = solve_ivp(rossler,tspan,x0,args=(epsilon,),rtol=1e-6,atol=1e-9)
    t = sol.t
    x = sol.y
    N2 = 2 ** np.ceil(np.log2(abs(len(t))))
    T = t[-1,] - t[0,]
    dt2 = T/N2
    t2 = dt2 * np.arange(0,N2 - 1)
    interp_func = interp1d(t,x[0,:],kind='cubic')
    x2 = interp_func(t2)
    a = fft(x2)
    e = a * np.conj(a)/ (N2**2)
    maxpower = np.max(e)
    idx_max = np.argmax(e)
    omega= 2*np.pi/T * (np.arange(0,N2/2))
    T1 = 2*np.pi / omega[idx_max]
    idx_transient = np.argmin(abs(t - 15*T1))
    x_ss = x[:,idx_transient:-1]
    
    #create figure for run
    fig = plt.figure(figsize=(6,10),dpi=200)
    #fig.suptitle('$\epsilon = {:2g}$'.format(epsilon),fontsize=20)
    
    #plot state space
    ax1 = fig.add_subplot(3,1,1,projection='3d')
    ax1.plot(x_ss[0,:],x_ss[1,:],x_ss[2,:],linewidth=.4)
    ax1.grid(False)
    plt.setp(ax1,xlabel='$x_1$',ylabel='$x_2$',zlabel='$x_3$')
    ax1.set_title('$\epsilon = {:2g}$'.format(epsilon),fontsize=16)
    
    
    #plot frequency spectrum
    ax2 = fig.add_subplot(3,1,2)
    ax2.loglog(omega,e[np.arange(0,N2//2).astype('int')],linewidth=.4)
    ax2.set_xlabel('$\omega$')
    ax2.set_ylabel('Power')
    
    #display dominant frequency on figure
    ax3 = fig.add_subplot(3,1,3)
    ax3.text(.5,.5,'Dominant frequency $\omega_{{dominant}} = {0:.2f}$ Hz\n Dominant period $T_{{dominant}} = {1:.2f}$ s'.format(omega[idx_max],T1),horizontalalignment=
             'center',fontsize=16)
    ax3.axis('off')
    
    fig.savefig('eps = {:2g}.png'.format(epsilon) ,dpi=200,linewidth=0.5)
