import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import expon

# ratio of uniforms method

N = int(1e7)

u = np.random.uniform(low=0, high=1, size = N)
v = np.random.uniform(low=-2/np.e, high = 2/np.e, size = N)

ratio = v/u
f = expon.pdf(ratio)
x = ratio[u < np.sqrt(f)]

# Plot Results
plt.plot()
# Cut far numbers to get a better plot.
plt.hist(x[np.abs(x)<10], bins=300, density=True)
xs = np.linspace(0, 5, 1000)
plt.plot(xs, np.exp(-xs), label=r'$e^{-x}$')
plt.xlim(min(xs), max(xs))
plt.legend()
plt.xlabel('x')
plt.ylabel('PDF(x)')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(4, 0.15, f"{len(x)/N*100:.2f} %", fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig('ratio_of_uniforms.jpg', dpi=300)

keep = u**2 < f
# Plot bounding box and uniform samples
plt.clf()
plt.scatter(u[keep],v[keep],alpha=0.03,s=0.4)
plt.vlines([0, 1],ymin = -2/np.e, ymax = 2/np.e)
plt.hlines([-2/np.e, 2/np.e],xmin = 0,xmax = 1)