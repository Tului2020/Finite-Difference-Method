import matplotlib.pyplot as pl
import numpy as np
import math

#inputs

w = 78970.5                             # weight/length (Newtons/meter) steel
L = 1                                   # meters
EI = 7897050                            # property
n = [100]                              # nxn matrix


def deflection(x1, w1=w, ei=EI):
    return -w1/ei*(math.pow(x1, 4) - 2 * math.pow(x1, 3) + x1) / 24


# Program
def matrix_a(n1, l1=L):
    dx1 = (l1/(n1-1.0))**2
    ret = np.identity(n1)
    for i in range(1, n1-1):
        ret[i][i] = -2 / dx1
        ret[i][i - 1] = ret[i][i + 1] = 1 / dx1
    return ret
# print(matrix(n[0]))


def vector_w(n1, w1=w):
    ret = np.multiply(np.ones(n1), -w1)
    ret[0] = ret[-1] = 0
    return ret
# print(vector_w(n[0]))


def axis(n1, l1=L):
    return np.multiply(np.arange(n1), l1/(n1 - 1.0))

A = [[0] for i in range(len(n))]
v = [[0] for i in range(len(n))]
m = [[0] for i in range(len(n))]
u = [[0] for i in range(len(n))]
x = [[0] for i in range(len(n))]

for i in range(len(n)):
    x[i] = axis(n[i])
    A[i] = matrix_a(n[i])
    v[i] = vector_w(n[i])
    m[i] = np.linalg.solve(A[i], v[i])
    A[i] = np.multiply(A[i], EI)
    u[i] = np.linalg.solve(A[i], m[i])

xa = axis(100)
a = np.zeros(100)
for i in range(len(xa)):
    a[i] = deflection(xa[i])


pl.plot(x[0], u[0], 'r', label=str(len(x[0])))
#pl.plot(x[1], u[1], 'g', label=str(len(x[1])))
#pl.plot(xa, a, 'r', label='analytical')
#pl.legend(bbox_to_anchor=(0.80, 1.15), loc=2, borderaxespad=0.)
pl.grid(True)
pl.xlabel('Length')
pl.ylabel('Deflection')
pl.title('Finite Difference Method')
pl.show()










