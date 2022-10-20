import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def integrate(fun,a,b,tol):
    # print('calling function from ',a,b)
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    #do the 3-point integral
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    if myerr<tol:
        return i2
    else:
        mid=(a+b)/2
        int1=integrate(fun,a,mid,tol/2)
        int2=integrate(fun,mid,b,tol/2)
        return int1+int2

distances = np.linspace(0, 5, 1001)
tol = 1e-5

dE = lambda u, z : (z - u)/(1 + z**2 - 2*z*u)**(3/2)
E_field = []

for distance in distances:
    # y1 = integrate(dE, -1, 1, tol = tol, args = (distance,))
    y, err = quad(dE, -1, 1, args=(distance,))
    E_field.append(y)

norm_E = [float(value)/max(E_field) for value in E_field]

plt.plot(distances, norm_E,  label='quad')
plt.xlabel('Distance (R)')
plt.ylabel('E (normalized)')
plt.legend()