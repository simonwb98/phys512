import numpy as np
from matplotlib import pyplot as plt

chi_sq = np.loadtxt('chi_square.txt')
freq = np.fft.rfftfreq(len(chi_sq))
pwr_sp = np.abs(np.fft.rfft(chi_sq))

plt.loglog(freq, pwr_sp)
plt.xlabel('Frequency')
plt.ylabel(r'Power spectrum of $\chi^2$')
plt.savefig('chi_squared.jpg', dpi=300)