import matplotlib.pyplot as pl
from random import random
import numpy as np
import math


# This program uses the bisection method


n = 100
L = 1.0
dx = L/(n-1)
g = -1
tol = 0.00001
x = np.arange(0, 1+dx/2, dx)


def shoot(eig):
    ret = np.zeros(n)
    ret[1] = g
    for x1 in range(2, n):
        ret[x1] = ret[x1-1] * (2 - eig*dx**2) - ret[x1-2]
    return ret


def bm(l, u, om=random(), c=1, t=tol):
    m = (u + l) / 2
    fl = shoot(l)[-1]
    fm = shoot(m)[-1]
    fu = shoot(u)[-1]
    if c == 1:
        if fl * fm < 0:
            return bm(l, m, m, c+1)
        elif fu * fm < 0:
            return bm(m, u, m, c+1)
        else:
            return ' Choose different points '
    elif abs((m-om)/m)*100 < t:
        return m #, c
    elif fl * fm < 0:
        return bm(l, m, m, c+1)
    elif fu * fm < 0:
        return bm(m, u, m, c+1)
    else:
        return ' Choose different points '



e = bm(9, 10)
u = shoot(e)
er = (math.pi**2 - e ) * 100 / math.pi**2

pl.plot(x, u)
pl.title('Eigenvalue: ' + str(e) + '\n error: ' + str(er))
pl.grid(True)
pl.show()


































