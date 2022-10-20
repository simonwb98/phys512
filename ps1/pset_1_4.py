import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

def lorentzian(x):
    return 1/(1 + x**2)

### PARAMETER INITIALIZATION ###

n = 8# order numerator for rational function fit
m = 10 # order denominator for rational function fit
deg = 2 # polynomial degree
fun = np.cos

x = np.linspace(-np.pi/2, np.pi/2, n + m + 1) # define (coarsely spaced) x values
x_fine = np.linspace(-np.pi/2, np.pi/2, 1001) # define x_fine values for fit evaluation
y = fun(x) 
y_fine = fun(x_fine)
plt.plot(x_fine, y_fine, label='lorentzian')

### MODEL FUNCTIONS ###

# fitting polynomial
fit_1 = np.polyfit(x, y, deg, full=True) # generate fit params
y_1 = np.polyval(fit_1[0], x_fine) # with fit params, generate y_1 array
plt.plot(x_fine, y_1, label='polynomial')
error_1 = np.std(y_1 - y_fine) # get std

# fitting cubic spline
fit_2 = splrep(x, y) # generate spline 
y_2 = splev(x_fine, fit_2) # generate y_2 array
plt.plot(x_fine, y_2, label='cubic spline')
error_2 = np.std(y_2 - y_fine) # get std

# fitting rational like in-class example (ratfit_class.py)
pcols = [x**k for k in range(n + 1)] # generate x values that multiply with constants (p0, p1, etc.)
pmat = np.vstack(pcols) # stack vertically

qcols = [-x**k*y for k in range(1, m + 1)] # same here
qmat = np.vstack(qcols)
mat = np.hstack([pmat.T, qmat.T]) # create final matrix that multiplies the coeff to give y
coeff = np.linalg.inv(mat)@y # invert matrix product to solve for the coeffs

num = np.polyval(np.flipud(coeff[:n + 1]), x_fine) # evaluate p
denom = 1 + x_fine*np.polyval(np.flipud(coeff[n + 1:]), x_fine) # evaluate q
y_3 = num/denom  
plt.plot(x_fine, y_3, label='rational')
error_3 = np.std(y_3 - y_fine)

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'lower left')
plt.savefig('higher_order_lorentz_fit_models.jpg', dpi = 300)
print('\nThe errors (std) are \npolynomial model: ', error_1,
      '\nb-spline: ', error_2, '\nrational function: ', error_3)