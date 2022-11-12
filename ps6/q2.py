import numpy as np
from matplotlib import pyplot as plt

def fft_shift(arr, shift):
    # according to the convolution theorem, we can
    # write the convolution of two functions conv(f, g)
    # as the inverse fourier transrom of the product of 
    # their respective fourier transforms, like 
    # h(t) = conv(f, g)(t) = IFT(F*G)
    g = np.zeros(len(arr))
    g[shift] = 1
    arr_ft = np.fft.fft(arr)
    g_ft = np.fft.fft(g)
    return np.fft.ifft(arr_ft*g_ft)

def correlation(f, g):
    # the correlation f°g can be computed with the
    # fourier transforms of f and g
    dft_f = np.fft.fft(f)
    dft_g = np.fft.fft(g)
    return np.fft.ifft(dft_f*np.conj(dft_g))

x = np.arange(0, 100)
sigma = 5
x0 = 30
y = (1/sigma*np.sqrt(2*np.pi))**5*np.exp(-0.5*((x - x0)/sigma)**2)
shift = 30
y_shift = fft_shift(y, shift)


plt.plot(x, y, label='Gaussian')
plt.plot(x, y_shift, label='shifted Gaussian')

# plt.plot(x, correlation(y, y), label='autocorrelation')
# plt.plot(x, correlation(y, y_shift), label=f'cross-correlation - shift {shift}')
plt.legend()
plt.xlabel(r'$\tau$' + "\'")
plt.ylabel(r'$(f_1\star f_2)(\tau$' + "\')")
plt.savefig(f'corr_gauss_{shift}.jpg', dpi=300)