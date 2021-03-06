import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

z_0 = 1
h = 1
n = 40
L = 1.0
dx = L/(n-1)
base = np.multiply(np.arange(-n+1, n), dx)
n = len(base)
x_axis = np.zeros(shape=(n, n))
y_axis = np.zeros(shape=(n, n))
n2 = n**2


def analytical(y1):
    return z_0*(1 - (y1/h)**2)


for row in range(n):
    for col in range(n):
        x_axis[row][col] = base[col]
        y_axis[row][col] = base[-1-row]


def bcs(x):
    return 0 * x


def bcn(x):
    return 0 * x


def bce(x):
    return analytical(x)


def bcw(x):
    return analytical(x)


def create_matrix_vector():
    retm = np.identity(n2)
    retv = np.multiply(np.ones(n2), -2*z_0/(h*h)*dx**2)
    change = [[0], [n-1]]
    while change[0][-1] < n2-n:
        change[0].append(change[0][-1] + n)
        change[1].append(change[1][-1] + n)
    for i in range(n):
        retv[change[0][i]] = bcs(i*dx)
        retv[change[1][i]] = bcn(i*dx)
        retv[change[0][0] + i] = bcw((i-int(n/2))*dx)
        retv[change[0][-1] + i] = bce((i-int(n/2))*dx)
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
u = np.linalg.solve(mat, vec)
z = u.reshape(n, n)
a = np.zeros(shape=(n, n))
e = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(n):
        a[i][j] = analytical((j - int(n/2)) * dx)
        e[i][j] = abs(a[i][j] - z[i][j])


fig = pl.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_axis, y_axis, z)
#surf = ax.plot_surface(x_axis, y_axis, a)
#surf = ax.plot_surface(x_axis, y_axis, e)
pl.title('2D Torsion Numerical')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()

fig = pl.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x_axis, y_axis, z)
surf = ax.plot_surface(x_axis, y_axis, a)
#surf = ax.plot_surface(x_axis, y_axis, e)
pl.title('2D Torsion Analytical')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()


fig = pl.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x_axis, y_axis, z)
#surf = ax.plot_surface(x_axis, y_axis, a)
surf = ax.plot_surface(x_axis, y_axis, e)
pl.title('2D Torsion Error')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()












