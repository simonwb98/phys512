import numpy as np
from matplotlib import pyplot as plt

N = int(1e7) # number of samples we will draw
M = 1
target = np.exp
custom_rand = []
x = np.random.uniform(low=0, high=np.pi/2, size=N)
trials = 0.80475*np.tan(np.pi*(x))

# trials = np.random.standard_cauchy(size=N)
trials = trials[trials >= 0]

u = np.random.rand(len(trials))

def lorentz(x):
    return 1/(1+(x/0.80475)**2)

target_dist = target(-trials)
scaled_lorentz = M*lorentz(trials)

s = target_dist/scaled_lorentz

keep = trials[u < s]
      
# plt.clf()
# custom_rand = custom_rand[np.array([custom_rand]) <= 7]
plt.hist(keep, density=True, bins = 300,)
x = np.linspace(0, max(keep), 1001)
plt.plot(x, target(-x), label=r"$e^{-x}$")
plt.plot(x, M*lorentz(x), label=r"Lorentzian")
plt.xlim(0, 5)
plt.legend()
plt.xlabel('x')
plt.ylabel('PDF(x)')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(4, 0.15, f"{len(keep)/len(trials)*100:.2f} %", fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig('decay_lorentz_best.png', dpi=300)

# do the same but for power law distribution

# x = np.random.uniform(low = 0, high = 1, size = N)
# power = np.e
# M = (power/np.e)**power + 0.000001
# exp = 1/(1 - power)

# trials = x**(exp)
# u = np.random.rand(N)

# def power_law(x, pwr):
#     return x**(-pwr)

# target_dist = target(-trials)
# scaled_pwr = M*power_law(trials, power)
# s = target_dist/scaled_pwr

# keep = trials[u < s] - 1


# plt.clf()
# plt.hist(keep, density=True, bins = 300,)
# x = np.linspace(0.001, max(keep), 1001)
# plt.plot(x, target(-x), label=r"$e^{-x}$")
# plt.plot(x, M*power_law(x, power), label=rf"Power law, $\alpha$ = {power:.2f}, C={M:.2f}")
# plt.xlim(0, 5)
# y_max = 1.5
# plt.ylim(0, y_max)
# plt.legend()
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(0.1, 0.95*y_max, f"{len(keep)/N*100:.2f} %", fontsize=14,
#         verticalalignment='top', bbox=props)
# plt.xlabel('x')
# plt.ylabel('PDF(x)')
# plt.savefig(f'power_law_{power}.png', dpi=300)

# alpha = np.linspace(1.1, 4, 21)
# compare = []
# x = np.random.uniform(low = 0, high = 1, size = N)
# u = np.random.rand(N)
# for power in alpha:
#     M = (power/np.e)**power + 1e-6
#     exp = 1/(1 - power)

#     trials = x**(exp)

#     target_dist = target(-trials)
#     scaled_pwr = M*power_law(trials, power)
#     s = target_dist/scaled_pwr

#     keep = trials[u < s] - 1
#     compare.append(len(keep)/N)
    
# plt.clf()
# plt.plot(alpha, compare)
# plt.xlabel(r'$\alpha$')
# plt.ylabel('Efficiency')
# plt.savefig('power_law_efficiency.jpg', dpi=300)