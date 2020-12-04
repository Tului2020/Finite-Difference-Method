import matplotlib.pyplot as pl
import numpy as np
import math

#inputs
L = 1
n = 400
dy = L/(n-1.0)


def axis(n1=n, dy1=dy):
    return np.multiply(np.arange(n1), dy1)


def shoot(im1, im0, lam, dy1=dy):
    return (2 - lam * dy1**2) * im0 - im1

x = axis()
u = np.zeros(n)
u[1] = -0.008
u[-1] = 1
lamb = 9
u1 = np.zeros(n)
u1[1] = -0.008
lamb1 = 9.5
u2 = np.zeros(n)
u2[1] = -0.008
lamb2 = math.pi**2

for i in range(2, n):
    u[i] = shoot(u[i - 2], u[i - 1], lamb)
    u1[i] = shoot(u1[i - 2], u1[i - 1], lamb1)
    u2[i] = shoot(u2[i - 2], u2[i - 1], lamb2)


pl.plot(u, x, 'r', label=lamb)
pl.plot(u1, x, 'g', label=lamb1)
pl.plot(u2, x, 'k', label=lamb2)
pl.legend(bbox_to_anchor=(0.80, 1.15), loc=2, borderaxespad=0.)
pl.title('Shooting Method')
pl.xlabel('Deflection')
pl.ylabel('Length')
pl.grid(True)
pl.show()

