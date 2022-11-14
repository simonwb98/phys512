import numpy as np
from matplotlib import pyplot as plt

N = 100
x = np.arange(0, N)
def sine(x, f0):
    return np.sin(2*np.pi*f0*x)
f0 = 0.30 # set main frequency in Hz
f1 = 0.306 # offset frequency

y1 = sine(x, f0)
y2 = sine(x, f1)
# plt.plot(x, y1)
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'$f(\tau)$')

dft_y1 = np.fft.fft(y1)
dft_y2 = np.fft.fft(y2)
f = np.arange(0, N)/N
# plt.plot(f, np.abs(dft_y1)/N, '.', label=f'$f = {f0}$ Hz')
plt.plot(f, np.abs(dft_y2)/N, '.', color = '#ff7f0e', label=f'$f = {f1}$ Hz')
plt.xlabel(r'$f = v/N$ [Hz]')
plt.ylabel(r'Power spectrum $S(v)$')
# plt.savefig('power_spectrum.jpg', dpi = 300)

# windowing
def hann_window(x, N):
    return 0.5*(1-np.cos(2*np.pi*x/N))

# multiply window function with signal in real space
window_sig = y2*hann_window(x, N)

dft_w = np.fft.fft(window_sig)
plt.plot(f, np.abs(dft_w)/N, '.', color = '#2ca02c', label='with Hann window')
# plt.savefig('pwr_window.jpg', dpi = 300)

# part (e)
dft_wind = np.fft.fft(hann_window(x, N))
# print(np.abs(dft_w))
plt.plot(f, np.abs(dft_wind)/N, '.', color = '#d62728', label='Hann window')
plt.legend()
plt.savefig('fft_window.jpg', dpi=300)
