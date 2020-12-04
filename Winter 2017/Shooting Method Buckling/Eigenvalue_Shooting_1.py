import matplotlib.pyplot as pl
import numpy as np
import math

# This program starts at 9.89 and steps down until gets value


n = 100
L = 1.0
dx = L/(n-1)
eig = 9.89
g = 1
step = -0.000001

def shoot():
    ret = np.zeros(n)
    ret[1] = g
    for x1 in range(2, n):
        ret[x1] = ret[x1-1] * (2 - eig*dx**2) - ret[x1-2]
    return ret

x = np.arange(0, 1+dx/2, dx)
u = np.zeros(n)
p = abs(shoot()[-1])
eig += step
c = abs(shoot()[-1])


while p > c:
    u = np.zeros(n)
    p = abs(shoot()[-1])
    eig += step
    c = abs(shoot()[-1])

u = shoot()


pl.plot(x, u)
pl.title('Eigenvalue: ' + str(eig))
pl.grid(True)
pl.show()


































