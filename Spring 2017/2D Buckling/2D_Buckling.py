import numpy as np
import matplotlib.pyplot as pl
import math
from mpl_toolkits.mplot3d import Axes3D


n = 30
L = 1.0
dx = L/(n-1)
dx2 = dx**2
n2 = n**2
rep = 200
x_axis = np.zeros(shape=(n, n))
y_axis = np.zeros(shape=(n, n))
base = np.arange(0, L+dx/10, dx)
for row in range(n):
    for col in range(n):
        x_axis[row][col] = base[col]
        y_axis[row][col] = base[-1 - row]

def create_matrix():
    retm = np.identity(n2)
    change = [[0], [n-1], []]
    while change[0][-1] < n2-n:
        change[0].append(change[0][-1] + n)
        change[1].append(change[1][-1] + n)
    for i in range(n):
        change[2].append(change[0][i])
        change[2].append(change[1][i])
    for c1 in range(1, len(change[0])-1):
        for c2 in range(1, n-1):
            s = change[0][c1] + c2
            retm[s][s] = -4 / dx2
            retm[s][s - 1] = 1 / dx2
            retm[s][s + 1] = 1 / dx2
            retm[s][s - n] = 1 / dx2
            retm[s][s + n] = 1 / dx2
    for c in change[2]:
        retm = np.delete(retm, n2 - c - 1, axis=0)
        retm = np.delete(retm, n2 - c - 1, axis=1)

    for i in range(n - 2):
        retm = np.delete(retm, 0, axis=1)
        retm = np.delete(retm, 0, axis=0)
        retm = np.delete(retm, len(retm) - 1, axis=1)
        retm = np.delete(retm, len(retm) - 1, axis=0)
    return retm

matrix = create_matrix()
x = np.arange(float(len(matrix)))


def inverse_power_method(a1=matrix, x1=x, r=rep):
    at = np.linalg.inv(a1)
    for i in range(r):
        x11 = at.dot(x1)
        if i == rep-1:
            lam = x1.dot(x1)/x1.dot(x11)
        x1 = x11/(np.linalg.norm(x11))
    return lam, x1

eig = inverse_power_method()
print(eig[0])
u = eig[1]
u = np.divide(u, np.max(u))
z = np.zeros(shape=(n, n))
lu = int(math.sqrt(len(u)))
for row in range(lu):
    for col in range(lu):
        z[row + 1][col + 1] = u[row * lu + col]
z = np.rot90(z)
z = np.rot90(z)


def analytical(x1, y1):
    pi = math.pi
    ret = math.sin(pi*x1) * math.sin(pi*y1)
    return ret

anal = np.zeros(shape=(n, n))
e = np.zeros(shape=(n, n))
for row in range(n):
    for col in range(n):
        anal[row][col] = analytical(x_axis[row][col], y_axis[row][col])
        e[row][col] = abs(anal[row][col] - z[row][col])





fig = pl.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_axis, y_axis, z)
pl.title('Numerical 2D Buckling')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()

fig = pl.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_axis, y_axis, anal)
pl.title('Analytical 2D Buckling')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()

fig = pl.figure()
ax = fig.gca(projection='3d')
pl.title('Error')
surf = ax.plot_surface(x_axis, y_axis, e)
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()














