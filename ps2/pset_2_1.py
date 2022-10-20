import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
e_f = 1e-15

class SpacingTooSmallError(Exception):
    '''Raised when the dx spacing between x values becomes too small'''
    pass


def integrate_adaptive(fun, a, b, tol, args=None, extra=None):
    # extra is a dict with keys being x and values being y
    # for first call of function, init the dict
    if extra == None:
        extra = {}
    # most function body works similar to how the integrator in class was written
    x = np.linspace(a, b, 5)
    dx = x[1] - x[0]
    if dx < e_f:
        raise SpacingTooSmallError('The spacing of x-values is smaller then machine precision. This can happen with singularities in the domain.')
    for x_value in x:
        if x_value not in extra.keys():
            try:
                # make sure an error is raised for the specific value
                with np.errstate(divide='raise'):
                    extra[x_value] = fun(x_value, args)
            except ZeroDivisionError or FloatingPointError:
                extra[x_value] = np.nan
    # 3 point integral
    i1 = (extra[x[0]] + 4*extra[x[2]] + extra[x[4]])/3*(2*dx)
    # 5 point integral 
    i2 = (extra[x[0]] + 4*extra[x[1]] + 2*extra[x[2]] + 4*extra[x[3]] + extra[x[4]])/3*dx
    err = abs(i1 - i2)
    if err < tol:
        return i2
    else:
        mid = (a + b)/2
        int1 = integrate_adaptive(fun, a, mid, tol/2, args=args, extra=extra)
        int2 = integrate_adaptive(fun, mid, b, tol/2, args=args, extra=extra)
        return int1 + int2

distances = np.linspace(0, 5, 1001)
tol = 1e-5

dE = lambda u, z : (z - u)/(1 + z**2 - 2*z*u)**(3/2)
E_field1 = []
E_field2 = []
# print(integrate_adaptive(dE, -1, 1, args = (0.5,), tol=tol))

for distance in distances:
    try:
        y1 = integrate_adaptive(dE, -1, 1, tol = tol, args=distance)
    except:
        y1 = np.nan
    y, err = quad(dE, -1, 1, args=(distance,))
    E_field1.append(y)
    E_field2.append(y1)

norm_E1 = [float(value)/max(E_field1) for value in E_field1]
norm_E2 = [float(value)/max(E_field2) + 0.2 for value in E_field2]

plt.plot(distances, norm_E1,  label='quad integrator')
plt.plot(distances, norm_E2,  label='numerical integrator (vertically shifted)')
plt.xlabel('Distance (R)')
plt.ylabel('E (normalized)')
plt.legend()
plt.savefig('E_field_vs_distance.jpg', dpi=300)