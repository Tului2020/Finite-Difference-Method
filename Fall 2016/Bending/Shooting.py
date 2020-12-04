import matplotlib.pyplot as pl
import numpy as np
import math
from random import random
#inputs

w = 78970.5                             # weight/length (Newtons/meter) steel
L = 1                                   # meters
EI = 7897050                            # property
n = 100                                 # nxn matrix
dx = L/(n - 1.0)
dx2 = dx**2


def shoot_moment(mm1, mm0, dx21=dx2, w1=w):
    return -w1 * dx21 + 2 * mm0 - mm1


def shoot_deflection(um1, um0, im0, dx21=dx2, ei=EI):
    return im0 * dx21 / ei + 2 * um0 - um1

x = np.multiply(np.arange(n), dx)
bm1 = np.zeros(n)
bm1[1] = 400*random()

bm2 = np.zeros(n)
bm2[1] = 400*random()

for i in range(2, n):
    bm1[i] = shoot_moment(bm1[i - 2], bm1[i - 1])
    bm2[i] = shoot_moment(bm2[i - 2], bm2[i - 1])

bm = np.zeros(n)
bm[1] = bm2[1] - bm2[-1]*(bm2[1] - bm1[1])/(bm2[-1] - bm1[-1])
for i in range(2, n):
    bm[i] = shoot_moment(bm[i - 2], bm[i - 1])

"""
pl.plot(x, bm1, 'k', label='Guess 1')
pl.plot(x, bm2, 'b', label='Guess 2')
pl.plot(x, bm,  'r', label='Interpolation')
pl.legend(bbox_to_anchor=(0.80, 1.15), loc=2, borderaxespad=0.)
pl.grid(True)
pl.xlabel('Length')
pl.ylabel('Bending Moment')
pl.title('Shooting Method for Bending Moment')
pl.show()
"""


u1 = np.zeros(n)
u1[1] = -random()/100000
u2 = np.zeros(n)
u2[1] = -random()/100000
u = np.zeros(n)

for i in range(2, n):
    u1[i] = shoot_deflection(u1[i - 2], u1[i - 1], bm[i])
    u2[i] = shoot_deflection(u2[i - 2], u2[i - 1], bm[i])

u[1] = u2[1] - u2[-1]*(u2[1] - u1[1])/(u2[-1] - u1[-1])
for i in range(2, n):
    u[i] = shoot_deflection(u[i - 2], u[i - 1], bm[i])


#pl.plot(x, u1, 'k', label='Guess 1')
#pl.plot(x, u2, 'b', label='Guess 2')
pl.plot(x, u,  'r', label='Interpolation')
#pl.legend(bbox_to_anchor=(0.80, 1.15), loc=2, borderaxespad=0.)
pl.grid(True)
pl.xlabel('Length')
pl.ylabel('Deflection')
pl.title('Shooting Method')
pl.show()


