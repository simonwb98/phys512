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
        f_single(m[0], m[3], m[5]) +
        f_single(m[1], m[3] + m[4], m[5]) + 
        f_single(m[2], m[3] - m[4], m[5]))
    pred = f(m)
    # make empty array to store gradient values
    grad = np.empty([t.size, 6]) 
    # numerical derivs
    step = 1e-10
    for i, v in enumerate(m):
        dm = np.zeros(len(m))
        dm[i] = step
        grad[:,i] = (f(m + dm) - f(m - dm))/(2*step)
    return pred, grad
    

plt.plot(time, d, label='data')
# # initial guesses for the parameters can be read from the plot
t_0 = 1.92e-4
a = 1.42
b = 0.1
c = b
dt = 0.5e-4
w = 1.79e-5
# plt.plot(time, lorentzian(time, t_0, a, w)[0])

model_params[0] = (a, b, c, t_0, dt, w)
error = []
for i in range(iterator):
    pred, grad = grad_lorentz(time, model_params[i])
    r = d - pred
    std = np.mean(np.abs(r))
    # just for plotting purposes
    error.append(std**2)
    r = np.matrix(r).transpose()
    grad = np.matrix(grad)
    lhs = grad.transpose()*grad
    rhs = grad.transpose()*r
    
    dm = np.linalg.inv(lhs)@rhs
    dm = dm.reshape(6,)
    if i == iterator - 1:
        break
    else:
        model_params[i + 1] = model_params[i] + dm

plt.plot(time, grad_lorentz(time, model_params[-1])[0], label='best fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (au)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

# calculate covariance matrix
cov = np.linalg.inv(lhs)*error[-1]
sampler = 6
for j in range(sampler):
    alt_params = np.random.multivariate_normal(model_params[-1], cov)
    plt.plot(time, grad_lorentz(time, alt_params)[0], label=f"perturbation {j + 1}")

plt.xlim(1.8e-4, 2.0e-4)
plt.ylim(1, 1.5)
plt.legend(ncol=2)
plt.savefig('params_perturb_zoomed.jpg', dpi=300)

# calculate std for best fit model
chi_squared = 1/std*r.T@r
# generating chi-sq
sampler = 10001
chi_sq = []
diff_chi = []
for j in range(sampler):
    alt_params = np.random.multivariate_normal(model_params[-1], cov)
    alt_pred = grad_lorentz(time, alt_params)[0]
    res = d - alt_pred   
    alt_chi = 1/std*res.T@res
    chi_sq.append(alt_chi)
    diff_chi.append(alt_chi - float(chi_squared))
# plt.clf()
# plt.hist(chi_sq, bins=100)
# plt.xlabel(r'$\chi^2$')
# plt.ylabel('Frequency')
# plt.savefig('histogram-chi-squared.jpg', dpi=300)

plt.clf()
plt.hist(diff_chi, bins=100)
plt.xlabel(r'$\Delta\chi^2$')
plt.ylabel('Frequency')
plt.savefig('delta_chi-hist_step_-10.jpg', dpi=300)

# print(f"{model_params[-1] = }")
# print(f"{errs = }")

# plt.clf()
# plt.plot(error)
# plt.yscale('log')
# plt.ylabel('RSS (au)')
# plt.xlabel('Fit Iterations')
# plt.savefig('error_tripple_lorentz_numerical.jpg', dpi=300)

# plt.clf()
# plt.plot(time, r, label='Residual of Triple Lorentzian Model')
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# plt.xlabel('Time (s)')
# plt.ylabel('Residual (au)')
# plt.legend()
# plt.savefig('residual_triple_lorentz.jpg', dpi=300)
