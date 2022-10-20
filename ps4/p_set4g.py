import p_set4d_e as prev
import numpy as np
from matplotlib import pyplot as plt
import corner

input_file = 'sidebands.npz'

stuff = np.load(input_file)
time = stuff['time']
d = stuff['signal']

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

# from previous answer, generate step size array and initial values
par_best= prev.model_params[-1]
par_errs = np.sqrt(np.diag(np.linalg.inv(prev.lhs)))


def get_chi_sq(data, params):
    pred, grad = grad_lorentz(time, params)
    chisq = np.sum((data - pred)**2)
    return chisq

n_steps = 100000
steps_taken = 0
chi_sq = []
par = []
# initialize parameter list by using best fit +/- a little something
init_par = par_best + 3.0*np.random.randn(len(par_best))*par_errs
chi_sq.append(get_chi_sq(d, init_par))
par.append(init_par)


while steps_taken < n_steps:
    # generate new parameters
    par_new = par[-1] + 1.0*np.random.randn(len(par_best))*par_errs
    # calculate new chi squared
    new_chi = get_chi_sq(d, par_new)
    delta_chi = new_chi - chi_sq[-1]
    prob_step = np.exp(-0.5*delta_chi)
    # if prob_step is greater then one, chi will have increased
    # and we want to discard that step, otherwise accept with a certain
    # probability:
    if prob_step > np.random.rand(1):
        par.append(par_new)
        chi_sq.append(new_chi)
    else:
        # if step is not taken, copy last entries for params and chi_sq
        par.append(par[-1])
        chi_sq.append(chi_sq[-1])
    steps_taken += 1
    if steps_taken % 10 == 0:
        print(f"At {steps_taken/n_steps*100:.2f} percent")

plt.plot(chi_sq)
plt.xlabel('Iterations')
plt.ylabel(r'$\chi^2$')
plt.savefig('chi_squared.jpg', dpi=300)

plt.clf()
par = np.array(par)
corner.corner(par, labels=['a', 'b', 'c', '$t_0$', 'dt', 'w'], show_titles=True, title_fmt='.2E')
plt.savefig('corner_plots.jpg', dpi=300)

