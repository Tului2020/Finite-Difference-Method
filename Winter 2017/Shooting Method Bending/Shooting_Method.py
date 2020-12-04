import matplotlib.pyplot as pl
import numpy as np
from random import random


n = 5000
l = 1.0
dx = l/(n-1)
w = 100
ei = 100000


def create_matrix(n1=n):
    ret = np.multiply(np.identity(n1), -2)
    for i in range(1, n1-1):
        ret[i][i + 1] = 1
        ret[i][i - 1] = 1
    ret[0][0] = ret[-1][-1] = 1
    return ret


def create_w(n1=n, w1=w, dx1=dx):
    ret = np.multiply(np.ones(n1), -w1 * dx1**2)

    return ret


def create_axis(n1=n, dx1=dx):
    ret = []
    for i in range(n1):
        ret.append(i * dx1)
    return ret


m = create_matrix()
w = create_w()
x = create_axis()
moment = np.linalg.solve(m, w)
def shoot(i, dis, dx1=dx, ei1=ei):
    dis.append(moment[i] * dx1**2 / ei1 + 2 * dis[-1] - dis[-2])

#for i in range(n):
#    print(m[i], w[i])


"""
pl.plot(x, np.divide(w, dx**2))
pl.title('Distributed Loading')
pl.grid(True)
pl.show()


pl.plot(x, moment)
pl.title('Bending Moment')
pl.grid(True)
pl.show()
"""

g0 = random()/10000
g1 = -random()/10000

y_0 = 0
discrete = [[y_0, dx*g0+y_0], [y_0, dx*g1+y_0], [y_0]]
for i in range(n-2):
    shoot(i + 2, discrete[0])
    shoot(i + 2, discrete[1])


s = (discrete[1][-1] - discrete[0][-1])/(g0 - g1)
g = discrete[1][-1]/s + g1

discrete[2].append(dx*g+y_0)
for i in range(n-2):
    shoot(i + 2, discrete[2])


pl.plot(x, discrete[0])
pl.plot(x, discrete[1])
pl.title('Deflection')
pl.grid(True)
pl.show()


pl.plot(x, discrete[2])
pl.title('Deflection, slope = ' + str(g))
pl.grid(True)
pl.show()






















