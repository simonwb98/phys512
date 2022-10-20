import numpy as np

positions = np.loadtxt('dish_zenith.txt')

# define x,y,z values
x = np.zeros(len(positions))
y = np.zeros(len(positions))
z = np.zeros(len(positions))
for i, l in enumerate(positions):
    x[i] = l[0]
    y[i] = l[1]
    z[i] = l[2]
    
# define model matrix A
A = np.empty([len(x), 4])
A[:,0] = x**0
A[:,1] = x
A[:,2] = y
A[:,3] = x**2 + y**2

lhs = A.T@A
rhs = A.T@z
m = np.linalg.inv(lhs)@rhs # [k_1, k_2, k_3, k_4]
pred = A@m


# in principle, the covariance matrix needs to be 
# computed from the outer product |n><n| where 
# n is the residual in each data point. A rough
# method we used in class should however also be
# sufficient:
N = np.mean((z-pred)**2) # get mean of difference
# calculate params errors using A and A.T
par_errs=np.sqrt(N*np.diag(np.linalg.inv(lhs)@rhs))
