# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

R = .5
Cd = .8
g = 9.81
d = 2.5e-2
Ad = np.pi * (d ** 2) / 4
Qins = np.array([1,2,3]) * 1e-3
tspan = [0, 3600]
h0 = [0];

def fill_cyl(t,h):
    return 1 / (np.pi * (R ** 2)) * (Qin - Cd*Ad*np.sqrt(2*g*h))

plt.rcdefaults()

for i in range(len(Qins)):
    Qin = Qins[i]
    sol = solve_ivp(fill_cyl,tspan,h0, rtol = 1e-6, atol = 1e-9)
    plt.plot(sol.t,sol.y.reshape(sol.t.shape))
    h_inf = 1 / (2*g) * (Qin/(Cd*Ad)) ** 2
    err = (abs((sol.y[:,-1] - h_inf)/h_inf)).item()
    print('Qin value: {0:.1g} L/s Relative error: {1:.2g}'.format(Qin*1000,err))
plt.legend(['Q = .001 $m^{{3}}/s$','Q = .002 $m^{{3}}/s$','Q = .003 $m^{{3}}/s$'])
plt.xlabel('Time, [s]')
plt.ylabel('Tank fill height, [m]')

#task 2

def fill_sphere(t,h):
    if (h < 2*R):
        return Qin/(np.pi*h * (2*R - h))
    else:
        return 0
    
Qins = np.array([1,2,3]) * 1e-2
R = 2
h0 = [1e-6]    
fig,ax = plt.subplots()
for i in range(len(Qins)):
    Qin = Qins[i]
    sol = solve_ivp(fill_sphere,tspan,h0, method='RK45',rtol = 1e-6, atol = 1e-9)
    fillTime = 4*np.pi*(R**3)/(3*Qin)
    ax.plot(sol.t, sol.y.reshape(sol.t.shape), label=r'Qin = {:.1g}'.format(Qin))
    ax.plot(np.array([fillTime,fillTime]),np.array([0, 4]),'--k',label=r'$t_{fill,theoretical}$' + ' for Qin = {:.1g}'.format(Qin))
    ax.set(xlim=(0, 3600),ylim=(0,4))
    ax.set(xticks=(np.linspace(0,3600,10)))    
ax.legend()
ax.set_ylabel('Fill height, [m]')
ax.set_xlabel('Time, [s]')
