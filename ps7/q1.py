import numpy as np
from matplotlib import pyplot as plt

rand_nums = np.loadtxt('rand_points.txt')

x = rand_nums[:,0]
y = rand_nums[:,1]
z = rand_nums[:,2]

# fig = plt.figure(figsize=(15, 15))
# ax = plt.axes(projection='3d')
# ax.scatter(x, y, z, marker='.')

# comparing with same size as in given data
size = len(x)
limit = int(1e8 + 1)

py_rand = np.random.randint(0, limit, size=(size, 3))

x = py_rand[:,0]
y = py_rand[:,1]
z = py_rand[:,2]

fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, marker='.')
