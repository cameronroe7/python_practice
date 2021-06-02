# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:29:39 2020

@author: Cameron
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def odefun(t,x,f_0,omega):
    alpha = .1
    zeta = .1
    omega_0 = 1
    return [x[1], f_0 * np.cos(omega*t) - 2* zeta * omega_0 * x[1] - omega_0**2 * (x[0] + alpha*x[0]**3)]

x0 = [0,0]
tmax = 150
tspan = [0,tmax]
f_0 = .7
omegas = (.5,1.24,1.25,2)
index = 1
plt.figure(figsize=(9,6))

for omega in omegas:
    sol = solve_ivp(odefun,tspan,x0,rtol=1e-6,atol = 1e-9, args=(f_0,omega))
    plt.subplot(2,2,index)
    plt.subplots_adjust(bottom=-.15)
    plt.plot(sol.t,sol.y[0,:])
    plt.axis([0, tmax,-3,3 ])
    plt.xticks(np.linspace(0,tmax,5),)
    index = index + 1
