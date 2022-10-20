import numpy as np
from matplotlib import pyplot as plt

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
    grad[:,0] = (w**2)/((w**2) + (t - t_0)**2) # derivative w.r.t. a
    grad[:,1] = 2*a*w**2*(t - t_0)/(((w**2) + (t - t_0)**2)**2) # deriv t_0
    grad[:,2] = 2*a*w*(t - t_0)**2/(((w**2) + (t - t_0)**2)**2) # deriv w
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
    # calculate residual (serves as noise covariance matrix)
    r = d - pred
    err = np.mean(np.abs(r))
    error.append(err)
    # cast matrices in readily available forms
    r = np.matrix(r).transpose()
    grad = np.matrix(grad)
    # compute linear system components
    lhs = grad.transpose()*grad
    rhs = grad.transpose()*r
    # solve it for dm
    dm = np.linalg.inv(lhs)@rhs
    # output is matrix, reshape for next step
    dm = dm.reshape(3,)
    if i == iterator - 1:
        break
    else:
        model_params[i + 1] = model_params[i] + dm


par_errs=np.sqrt(np.diag(np.linalg.inv(lhs)*err**2))

plt.plot(time, lorentzian(time, model_params[-1])[0], label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (au)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig('single_lorentzian_fit.jpg', dpi=300)

# plt.clf()
# plt.plot(error)
# plt.yscale('log')
# plt.ylabel('RSS (au)')
# plt.xlabel('Fit Iterations')
# plt.savefig('error.jpg', dpi=300)