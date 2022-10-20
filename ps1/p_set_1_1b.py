import numpy as np
import matplotlib.pyplot as plt

# 64-bit, double precision machine --> fractional error, epsilon_m ~1e-16 (ask in tutorial).
# Assuming that epsilon_f = epsilon_m (as mentioned in Numerical Recipes, Section 5.7).

fun = np.exp
multiplier = 0.01
delta = 10**(np.linspace(-15, 0, 1001)) # make array to test best value
x = 1
temp = x + delta
delta = temp - x # for delta to be represented by an exact number in binary

numerical_derivative = 1/(12*delta)*(-fun(multiplier*(x + 2*delta)) + 8*fun(multiplier*(x + delta)) - 8*fun(multiplier*(x - delta)) + fun(multiplier*(x - 2*delta)))

plt.loglog(delta, abs(numerical_derivative - multiplier*fun(multiplier*x)), label=r'$f(x) = e^{0.01x}$')
plt.xlabel(r'$\delta$')
plt.ylabel('e')
plt.legend()

plt.savefig('error_2.jpg', dpi=300)