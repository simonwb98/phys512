import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1/2, 1, 1001)
u = 4*x - 3 # to rescale domain
y = np.log2(x)
deg = 50
coeffs = np.polynomial.chebyshev.chebfit(u, y, deg)
# retaining up to 9th order
chebs = np.empty([len(x),10])
chebs[:,0]=1
chebs[:,1]=u
for i in range(1,9):
    chebs[:,i+1]=2*u*chebs[:,i]-chebs[:,i-1]

model = coeffs[0:10]*chebs

y2 = []
for array in model:
    y2.append(sum(array))

def mylog2(x):
    (mantissa, exponent) = np.frexp(x)
    variable = 4*mantissa - 3
    cheb = np.empty(10)
    cheb[0]=1
    cheb[1]=variable
    for i in range(1, 9):
        cheb[i+1] = 2*variable*cheb[i] - cheb[i-1]
    model = coeffs[0:10]*cheb
    term1 = sum(model)
    return term1 + exponent
    
print(mylog2(4))
# plt.plot(x, y2, label='model')
# plt.plot(x, y, label='np.log2')
# plt.ylabel('y')
# plt.xlabel('x')
# plt.legend()
# plt.savefig('log2.jpg', dpi=300)


