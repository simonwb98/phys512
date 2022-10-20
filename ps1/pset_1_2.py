import numpy as np



def ndiff(fun, x, full = False, args = None):
    e_f = 1e-15 # machine precission
    # initial guess for dx
    dx = 1e-5
    # define derivative
    def deriv(fun, x, dx, args):
        return (fun(x + dx, args) - fun(x - dx, args))/(2*dx)
    # using the centered third order derivative for error
    def third_deriv(fun, x, dx, args):
        return (fun(x + 2*dx, args) - 2*fun(x + dx, args) + 2*fun(x - dx, args) - fun(x - 2*dx, args))/(2*dx**3)
    # iterating once
    dx_opt = np.power(e_f*abs(fun(x, args)/third_deriv(fun, x, dx, args)), 1/3)
    if dx_opt == 0 or dx_opt == np.inf:
        dx_opt = 1e-5
    error = e_f*abs(fun(x, args)/dx_opt) + abs(third_deriv(fun, x, dx_opt, args))*dx_opt**2
    if full:
        return (deriv(fun, x, dx_opt, args), dx_opt, error)
    else:
        return deriv(fun, x, dx_opt, args)
    
