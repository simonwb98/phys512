import numpy as np
from newton import curv, best, errs, spec, get_spectrum
import corner 
from matplotlib import pyplot as plt


def get_chi_sq(data, params):
    data = get_spectrum(params)[:len(spec)]
    chisq = np.sum((data - spec)**2/errs**2)
    return chisq

n_steps = 10000
steps_taken = 0
chi_sq = []
par = []
# initialize parameter list by using best fit +/- a little something
# par_errs = np.sqrt(np.diag(np.linalg.inv(curv)))
init_par = np.random.multivariate_normal(best, np.linalg.inv(curv)/2)
# init_par = best + 3.0*np.random.randn(len(best))*par_errs
chi_sq.append(get_chi_sq(spec, init_par))
par.append(init_par)


while steps_taken < n_steps:
    # generate new parameters
    step = np.random.multivariate_normal(np.zeros(best.size), np.linalg.inv(curv)/2)
    par_new = par[-1] + step
    # calculate new chi squared
    new_chi = get_chi_sq(spec, par_new)
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
    np.savetxt('mcmc_params.txt', par)
    np.savetxt('chi_square.txt', chi_sq)
    if steps_taken % 100 == 0:
        print(f"At {steps_taken/n_steps*100:.2f} percent")
        # corner.corner(par, labels=[r'$H_0$', r'$\Omega_bh^2$', r'$\Omega_ch^2$', r'$\tau$', r'$A_s$', r'$n_s$'], show_titles=True, title_fmt='.2E')
        


plt.plot(chi_sq)
plt.xlabel('Iterations')
plt.ylabel(r'$\chi^2$')
plt.savefig('chi_squared.jpg', dpi=300)

plt.clf()
par = np.array(par)
corner.corner(par, labels=[r'$H_0$', r'$\Omega_bh^2$', r'$\Omega_ch^2$', r'$\tau$', r'$A_s$', r'$n_s$'], show_titles=True, title_fmt='.2E')
plt.savefig('corner_plots.jpg', dpi=300)


