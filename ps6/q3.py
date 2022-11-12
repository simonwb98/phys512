import numpy as np

def no_wrapping(arr1, arr2):
    # assuming that len(arr1) = len(arr2) = N, 
    # one can append N zeros to both arrays to 
    # get rid of wrapping-around effects.
    N = len(arr1)
    zeros = np.zeros(N)
    a1 = np.append(arr1, zeros)
    a2 = np.append(arr2, zeros)
    dft_a1 = np.fft.fft(a1)
    dft_a2 = np.fft.fft(a2)
    
    # calculate convolution and ignore second 
    # half of values
    conv = 1/N*np.fft.ifft(dft_a1*dft_a2)[:N]
    return conv

