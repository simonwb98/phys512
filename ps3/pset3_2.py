import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from  math import log10, floor

t_i = 0
t_f = 10

tau = [4.468*10**12, 24.1*(1/365), 6.7*(1/24)*(1/365), 2.455*10**5,
       7.538*10**4, 1.6*10**3, 38235/365, 3.1/60*(1/24)*(1/365), 26.8/60*(1/24)*(1/365),
       19.9/60*(1/24)*(1/365), 164.3/1e-6*(1/60)*(1/24)*(1/365), 
       22.3, 5.015*10**3, 1.38376*10**5/365, 10**60] # list of all half lifes in yrs

tau_alt = [t/tau[0] for t in tau] # normalized half life units (wrt to U-238)
tau_alt2 = [t/tau[3] for t in tau] # normalized wrt to U-234

labels = ['U-238', 'Th-234', 'Pa-234', 'U-234', 
          'Th-230', 'Ra-226', 'Rn-222', 'Po-218', 
          'Pb-214', 'Bi-214', 'Po-214', 'Pb-210', 
          'Bi-210', 'Po-210', 'Pb-206']

products = np.zeros(len(tau))
products[0] = 1

init = products # initialize products


def decay(t, y, tau = tau_alt2):
    # given y (vector of all products amount) and tau as above, 
    # calculate dy/dt for all species
    # if species n decreases by an amount dy/dt, species n + 1
    # will increase by that amount
    dydt = np.zeros(len(tau))
    for i in range(len(tau)):
        if i == 0:
            dydt[i] = -y[i]/tau[i]
        elif i == len(tau):
            dydt[i] = y[i - 1]/tau[i - 1]
        else:
            dydt[i] = y[i - 1]/tau[i - 1] - y[i]/tau[i]
    return dydt
               


soln = integrate.solve_ivp(decay, (t_i, t_f), np.asarray(init), method='Radau', t_eval = np.linspace(t_i, t_f, 1001))
y = soln.y
t = soln.t

# Plotting all constituents of the decay chain

# for i, f in enumerate(y):
#     largest = float('{:.1e}'.format((max(f))))
#     base10 = log10(largest)
#     plt.plot(t, f, label='[' + labels[i] + ']') # + r'$\times 10^{' + str(abs(floor(base10))) + r'}$'
# plt.plot(t, np.exp(-t/tau_alt[0]), 'o', label='[U-238] expected decay')
# plt.grid()
# plt.xlabel(r'Time ($1/\tau$)')
# plt.ylabel('Fractional amount of element')
# plt.xscale('log')
# # plt.ylim(0.8, 1.05)
# plt.xlim(1e-2, t_f)
# plt.yscale('log')

# plt.legend(ncol=2, bbox_to_anchor = (1, 0.8))
# plt.savefig('decay.jpg', dpi=300, bbox_inches='tight')
# plt.clf()

# Plotting ratio of th-230 to u-234

plt.clf()
ratio = np.zeros(len(y[3]))
ratio[1::] = y[4][1::]/y[3][1::]
# print(ratio)
plt.plot(t, ratio, label='Ratio [Th-230]/[U-234]')
# plt.xscale('log')
# plt.xlim(0, 1e-4)
# plt.ylim(0, 0.3)

# time = np.linspace(0, t_f, 10001)
# factor = tau[0]/tau[3]
# print(factor)
# plt.plot(time, np.exp(factor*time) - 1)
plt.xlabel(r'Time ($1/\tau$)')
plt.ylabel('Population Ratio')
plt.legend()
plt.savefig('ratio_th_u.jpg', dpi=300)

# Plotting ratio of pb-206 to u-238

# plt.clf()
# plt.plot(t/tau[0], (y[-1]/y[0]), 'o', label='[Pb-206]/[U-238]')
# exp = np.linspace(0, 13.5, 10001)
# time = 10**exp/tau[0]
# plt.plot(time, np.exp(time) - 1, color='orange', label='Approximation')
# plt.xlabel(r'Time ($1/\tau$)')
# plt.ylabel('Population Ratio')
# plt.legend()
# plt.savefig('ratio.jpg', dpi=300)

# plt.xscale('log')

# plt.yscale('log')

