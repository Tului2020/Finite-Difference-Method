import numpy as np
import matplotlib.pyplot as pl
import math
from mpl_toolkits.mplot3d import Axes3D


rep = 200

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


def inverse_power_method(a1, x1, r=rep):
    at = np.linalg.inv(a1)
    for i in range(r):
        x11 = at.dot(x1)
        if i == rep-1:
            lam = x1.dot(x1)/x1.dot(x11)
        x1 = x11/(np.linalg.norm(x11))
    return lam, x1


itr = 50
s = np.zeros(itr)
a = np.multiply(np.ones(itr), 2*math.pi**2)
xs = np.arange(4, itr+4)

for n in range(4, itr+4):
    L = 1.0
    dx = L/(n-1)
    dx2 = dx**2
    n2 = n**2
    x_axis = np.zeros(shape=(n, n))
    y_axis = np.zeros(shape=(n, n))
    base = np.arange(0, L+dx/10, dx)
    for row in range(n):
        for col in range(n):
            x_axis[row][col] = base[col]
            y_axis[row][col] = base[-1 - row]
    matrix = create_matrix()
    x = np.arange(float(len(matrix)))
    s[n-4] = abs(inverse_power_method(matrix, x)[0])



fig = pl.figure()
pl.plot(xs, s, 'b')
pl.plot(xs, a, 'r')
ax = fig.add_subplot(111)
pl.title('2D Eigenvalue')
pl.ylabel('eigenvalue')
pl.xlabel('number of elements, n')
pl.grid(True)
pl.show()


















