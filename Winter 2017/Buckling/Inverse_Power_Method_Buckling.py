import numpy as np
import math
import matplotlib.pyplot as pl
from pprint import pprint

# get smallest eigenvalue
repet = 5
lambx = np.arange(1, repet+1)
lambval = np.zeros(repet)
realval = np.multiply(np.ones(repet), -math.pi**2)

for rep in range(1, repet+1):
    n = 1000
    L = 1.0
    dx = L/(n-1)
    dx2 = dx ** 2
    EI = 10000



    def create_a1(n1=n-2):
        ret = [[0]*n1 for i in range(n1)]
        for i in range(n1):
            try:
                ret[i][i - 1] = 1 / dx2
                ret[i][i - 0] = -2 / dx2
                ret[i][i + 1] = 1 / dx2
            except IndexError:
                None
        return np.array(ret)


    def create_a(n1=n):
        ret = [[0]*n1 for i in range(n1)]
        for i in range(n1):
            if i == 0 or i == n1-1:
                ret[i][i] = 1
            else:
                ret[i][i - 1] = 1 / dx2
                ret[i][i - 0] = -2 / dx2
                ret[i][i + 1] = 1 / dx2
        return np.array(ret)


    def create_x(n1=n, dx1=dx):
        ret = []
        for i in range(n1):
            ret.append(i * dx1)
        return ret


    def inverse_power_method(a1, x1, r=rep):
        at = np.linalg.inv(a1)
        for i in range(r):
            x11 = at.dot(x1)
            if i == rep-1:
                lam = x1.dot(x1)/x1.dot(x11)
            x1 = x11/(np.linalg.norm(x11))
        return x1, lam


    def power_method(a1, x1, r=rep):
        for i in range(r):
            s1 = a1.dot(x1)
            lam1 = np.linalg.norm(s1)
            x1 = np.divide(s1, lam1)
        lam1 = np.divide(lam1, np.amax(lam1))
        return x1, lam1


    a = create_a()
    x = np.arange(float(n-2))

    a1 = np.array([[0]*(n-2) for i in range(n-2)])
    for r in range(n-2):
        for c in range(n-2):
            a1[r][c] = a[r+1][c+1]
    #print(a1)
    xa = create_x()
    p = inverse_power_method(a1, x)
    lambval[rep-1] = p[1]

    y = np.array(p[0])
    x = np.zeros(n)
    for s in range(n-2):
        x[s+1] = y[s]

pl.plot(lambx, lambval)
pl.plot(lambx, realval)
pl.xlabel('Number of iteration')
pl.ylabel('Eigenvalue')
pl.title('Number of nodes: ' + str(n) + ', Iteration: ' + str(rep))
pl.grid(True)
pl.show()



