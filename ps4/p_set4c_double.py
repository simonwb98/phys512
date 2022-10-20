import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import derivative

input_file = 'sidebands.npz'

stuff = np.load(input_file)
time = stuff['time']
d = stuff['signal']
iterator = 51 # number of fitting calls
model_params = np.zeros([iterator, 6]) # define model matrix

def single_lorentzian(t, m):
    return m[0]/(1 + ((t - m[1])**2)/(m[2]**2))

def grad_lorentz(t, m):
    # model params m
    # m[0] = a      m[3] = t_0
    # m[1] = b      m[4] = dt
    # m[2] = c      m[5] = w
    assert(len(m) == 6)
    fun = single_lorentzian
    f_single = lambda amp, center, width : fun(t, np.array([amp, center, width]))
    f = lambda m : (
        f_single(m[0], m[3], m[5])
        )
    pred = f(m)
    # make empty array to store gradient values
    grad = np.empty([t.size, 6]) 
    # numerical derivs
    step = 1e-10
    for i, v in enumerate(m):
        dm = np.zeros(len(m))
        dm[i] = step
        grad[:,i] = (f(m + dm) - f(m))/step
    return pred, grad
    

plt.plot(time, d, label='data')
# # initial guesses for the parameters can be read from the plot
t_0 = 1.8e-4
a = 1.5
b = 0.1
c = b
dt = 0.5e-4
w = 1e-4
# plt.plot(time, lorentzian(time, t_0, a, w)[0])

model_params[0] = (a, b, c, t_0, dt, w)
error = []
for i in range(iterator):
    pred, grad = grad_lorentz(time, model_params[i])
    r = d - pred
    err = sum(r**2)
    error.append(err)
    r = np.matrix(r).transpose()
    grad = np.matrix(grad)
    lhs = grad.transpose()*grad
    rhs = grad.transpose()*r
    
    dm = np.linalg.pinv(lhs)@rhs
    dm = dm.reshape(6,)
    if i == iterator - 1:
        break
    else:
        model_params[i + 1] = model_params[i] + dm

plt.plot(time, grad_lorentz(time, model_params[-1])[0], label='fit')
plt.plot(time, grad_lorentz(time, model_params[0])[0], label='guess')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (au)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('tripple_lorentzian_fit.jpg', dpi=300)

plt.clf()
plt.plot(error)
plt.yscale('log')
plt.ylabel('RSS (au)')
plt.xlabel('Fit Iterations')
plt.savefig('error_single_lorentz_numerical.jpg', dpi=300)