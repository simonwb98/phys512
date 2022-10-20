import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import derivative

input_file = 'sidebands.npz'

stuff = np.load(input_file)
time = stuff['time']
d = stuff['signal']
iterator = 51 # number of fitting calls
model_params = np.zeros([iterator, 3]) # define model matrix


def lorentzian(t, m):
    a, t_0, w = m
    d = a/(1 + ((t - t_0)**2)/(w**2))
    grad = np.zeros([t.size, 3]) # make empty array to store gradient values
    # numerical derivs
    # f = lambda a, t_0, w : a/(1 + ((t - t_0)**2)/(w**2))
    # make subforms that contain only one variable
    f_a = lambda a : a/(1 + ((t - t_0)**2)/(w**2))
    f_t = lambda t_0 : a/(1 + ((t - t_0)**2)/(w**2))
    f_w = lambda w : a/(1 + ((t - t_0)**2)/(w**2))
    
    grad[:,0] = derivative(f_a, a, dx=1e-6)
    grad[:,1] = derivative(f_t, t_0, dx=1e-6) # deriv t_0
    grad[:,2] = derivative(f_w, w, dx=1e-8) # deriv w
    return d, grad

plt.plot(time, d, label='data')
# # initial guesses for the parameters can be read from the plot
t_0 = 1.8e-4
a = 1.5
w = 1e-4
# plt.plot(time, lorentzian(time, t_0, a, w)[0])

model_params[0] = (a, t_0, w)
error = []
for i in range(iterator):
    pred, grad = lorentzian(time, model_params[i])
    r = d - pred
    err = sum(r**2)
    error.append(err)
    r = np.matrix(r).transpose()
    grad = np.matrix(grad)
    
    lhs = grad.transpose()*grad
    rhs = grad.transpose()*r
    
    dm = np.linalg.pinv(lhs)@rhs
    dm = dm.reshape(3,)
    if i == iterator - 1:
        break
    else:
        model_params[i + 1] = model_params[i] + dm


plt.plot(time, lorentzian(time, model_params[-1])[0], label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (au)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('numerical_lorentzian_fit.jpg', dpi=300)

plt.clf()
plt.plot(error)
plt.yscale('log')
plt.ylabel('RSS (au)')
plt.xlabel('Fit Iterations')
plt.savefig('error_numerical.jpg', dpi=300)