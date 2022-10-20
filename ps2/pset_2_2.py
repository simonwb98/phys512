import numpy as np

class SpacingTooSmallError(Exception):
    '''Raised when the dx spacing between x values becomes too small'''
    pass

def inverse_root(x, args):
    return 1/np.sqrt(abs(x))

def square(x, args=None):
    return x**2

fun = np.sin
a = 0
b = np.pi/2
tol = 1e-15
e_f = 1e-15

def integrate(fun,a,b,tol,first=True):
    # print('calling function from ',a,b)
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    #do the 3-point integral
    global fun_calls
    if first:
        fun_calls = 0
    fun_calls += 5
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    
    if myerr<tol:
        return i2
    else:
        mid=(a+b)/2
        int1=integrate(fun,a,mid,tol/2, False)
        int2=integrate(fun,mid,b,tol/2, False)
        return int1+int2

def integrate_adaptive(fun, a, b, tol, args=None, extra=None):
    # extra is a dict with keys being x and values being y
    global func_calls
    # for first call of function, init the dict
    if extra == None:
        func_calls = 0
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
            func_calls += 1
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
    
integrate(fun, a, b, tol)
print('\nIntegral of', fun.__name__, 'from ', a, 'to', str(round(b, 2)) + ': ',  integrate_adaptive(fun, a, b, tol))
print('number of fun calls (lazy): ', fun_calls)
print('number of fun calls (adaptive): ', func_calls)
