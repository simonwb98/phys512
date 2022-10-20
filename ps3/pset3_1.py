import numpy as np
import matplotlib.pyplot as plt

deriv = lambda x, y: y/(1 + x**2)
steps1 = 201 # for rk4step
steps2 = 200//3 + 1 # for rk4stepd
start = -20
end = 20
x1 = np.linspace(start, end, steps1) # x ranges
x2 = np.linspace(start, end, steps2)
dx1 = x1[1] - x1[0] # dx spacings
dx2 = x2[1] - x2[0]
y1 = np.zeros(steps1) # initialize
y1[0] = 1
y2 = np.zeros(steps2)
y2[0] = 1
c0 = 1/(np.exp(np.arctan(-20)))
true1 = c0*np.exp(np.arctan(x1)) # calculate true values
true2 = c0*np.exp(np.arctan(x2))

def rk4_step(fun, x, y, h):
    # as done in lecture
    k1 = fun(x, y)*h
    k2 = h*fun(x + h/2, y + k1/2)
    k3 = h*fun(x + h/2, y + k2/2)
    k4 = h*fun(x + h, y + k3)
    dy = (k1 + 2*k2 + 2*k3 + k4)/6
    return y + dy

def rk4_stepd(fun, x, y, h):
    # calls rk4_step three times
    result1 = rk4_step(fun, x, y, h)
    temp = rk4_step(fun, x, y, h/2)
    result2 = rk4_step(fun, x + h/2, temp, h/2)
    # found values for the linear combination of the two 
    a = -1.0/14
    b = 15.0/14
    improved_result = a*result1 + b*result2
    return improved_result


# calling integrator
for index in range(len(y1) - 1):
    y1[index+1] = rk4_step(deriv, x1[index], y1[index], dx1)
for index in range(len(y2) - 1):
    y2[index+1] = rk4_stepd(deriv, x2[index], y2[index], dx2)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (5, 5), sharex=True, constrained_layout=True)
ax1.plot(x1, y1, label=r'rk4_step')
ax1.plot(x2, y2, label='rk4_stepd')
ax1.plot(x1, true1, label='analytical')
ax2.plot(x1, abs(y1-true1)*1e4, label='rk4_step')
ax2.plot(x2, abs(y2-true2)*1e4, label='rk4_stepd')
ax2.set_xlabel('x')
ax2.set_ylabel(r'residual ($\times 10^{-4}$)')

ax1.set_ylabel('y')
ax1.legend()
plt.savefig('model_comparison.jpg', dpi=300)


    
