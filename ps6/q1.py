import numpy as np
from matplotlib import pyplot as plt

def fft_shift(arr, shift):
    # according to the convolution theorem, we can
    # write the convolution of two functions conv(f, g)
    # as the inverse fourier transrom of the product of 
    # their respective fourier transforms, like 
    # h(t) = conv(f, g)(t) = IFT(F*G)
    g = np.zeros(len(arr))
    g[len(arr)//2 + shift] = 1
    arr_ft = np.fft.fft(arr)
    g_ft = np.fft.fft(g)
    return np.fft.ifft(arr_ft*g_ft)


x = np.linspace(-10, 10, 1000)
y = np.exp(-0.5*x**2)
y_shift = fft_shift(y, -200)

plt.plot(x, y, label='Gaussian')
plt.plot(x, y_shift, label='shifted Gaussian')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('shift_gauss.jpg', dpi=300)