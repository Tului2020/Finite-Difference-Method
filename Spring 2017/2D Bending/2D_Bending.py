import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

w = 100
n = 50
L = 1.0
dx = L/(n-1)
n2 = n**2
EI = 1


x_axis = np.zeros(shape=(n, n))
y_axis = np.zeros(shape=(n, n))
base = np.arange(0, L+dx/10, dx)
for row in range(n):
    for col in range(n):
        x_axis[row][col] = base[col]
        y_axis[row][col] = base[-1 - row]


def bcs(x):
    return 0 * x


def bcn(x):
    return 0 * x


def bce(x):
    return 0 * x


def bcw(x):
    return 0 * x


def create_matrix_vector():
    retm = np.identity(n2)
    retv = np.multiply(np.ones(n2), -w*dx**2)
    change = [[0], [n-1]]
    while change[0][-1] < n2-n:
        change[0].append(change[0][-1] + n)
        change[1].append(change[1][-1] + n)
    for i in range(n):
        retv[change[0][i]] = bcs(i*dx)
        retv[change[1][i]] = bcn(i*dx)
        retv[change[0][0] + i] = bcw(i*dx)
        retv[change[0][-1] + i] = bcw(i * dx)
    for c1 in range(1, len(change[0])-1):
        for c2 in range(1, n-1):
            s = change[0][c1] + c2
            retm[s][s] = -4
            retm[s][s - 1] = 1
            retm[s][s + 1] = 1
            retm[s][s - n] = 1
            retm[s][s + n] = 1
    return retm, retv


vec = create_matrix_vector()
mat = vec[0]
vec = vec[1]

moment = np.multiply(np.linalg.solve(mat, vec), dx**2 / EI)
z = np.zeros(shape=(n, n))
for row in range(n):
    for col in range(n):
        i = n**2 - col - n * row - 1
        z[row][col] = moment[i]
fig = pl.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_axis, y_axis, z)
pl.title('2D Moment')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()



u = np.linalg.solve(mat, moment)
z = np.zeros(shape=(n, n))
for row in range(n):
    for col in range(n):
        i = n**2 - col - n * row - 1
        z[row][col] = u[i]


fig = pl.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_axis, y_axis, z)
pl.title('2D Bending')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()













