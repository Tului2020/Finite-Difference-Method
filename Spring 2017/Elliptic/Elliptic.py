import numpy as np
import matplotlib.pyplot as pl
import math
from mpl_toolkits.mplot3d import Axes3D

c = 1
r = 1
t = 2 * math.pi
nt = 31    # looks nicer of it is an odd number
nr = 20
dt = t / (nt - 1)
dr = r / (nr - 1)
n = (nt-1) * nr

theta = np.arange(0, t, dt)
radius = np.arange(0, r, dr)



def elliptic(r1, t1, c1=c):
    return np.array([c1 * math.cosh(r1) * math.cos(t1), c1 * math.sinh(r1) * math.sin(t1)])



if not theta[-1] == 2 * math.pi:
    theta = np.append(theta, 2 * math.pi)

if not radius[-1] == r:
    radius = np.append(radius, r)

x, y = [], []
for ti in theta:
    x1, y1 = [], []
    for ri in radius:
        x1.append(elliptic(ri, ti)[0])
        y1.append(elliptic(ri, ti)[1])
    x.append(x1)
    y.append(y1)
    #pl.plot(x1, y1, color='g')
x = np.round(x, 4)
y = np.round(y, 4)
C = -2/(np.max(x)**2) - 2/(np.max(y)**2)
for j in range(len(x[0])):
    elx, ely = [], []
    for i in range(len(x)):
        elx.append(x[i][j])
        ely.append(y[i][j])
    #pl.plot(elx, ely)
#pl.show()


matrix = np.identity(n)
vector = np.zeros(n)
change = np.arange(0, n, nr)

for i in change:
    for j in range(nr-1):
        jloc = int((i + j)/nr)
        iloc = (i+j)%nr
        if iloc * dr > 0.7:
            hk = elliptic(iloc*dr, jloc*dt)
            h_a, k_a = elliptic((iloc+1)*dr, jloc*dt), elliptic(iloc*dr, (jloc+1)*dt)
            h_b, k_b = elliptic((iloc-1)*dr, jloc*dt), elliptic(iloc*dr, (jloc-1)*dt)
            h1, h2 = np.linalg.norm(hk - h_b), np.linalg.norm(hk - h_a)
            k1, k2 = np.linalg.norm(hk - k_b), np.linalg.norm(hk - k_a)
            h, k = (h1 + h2) / 2, (k1 + k2) / 2
            matrix[i + j][i + j] = -2 / (h1 * h2) - 2 / (k1 * k2)
            matrix[i + j][i + j + 1] = 1 / (h2 * h)
            matrix[i + j][i + j - 1] = 1 / (h1 * h)
            matrix[i + j][(i + j + nr) % n] = 1 / (k2 * k)
            matrix[i + j][(i + j - nr) % n] = 1 / (k1 * k)
            vector[i + j] = C


u = np.linalg.solve(matrix, vector).reshape(nt-1, nr)
u = np.insert(u, len(u)-1, u[0], axis=0)

fig = pl.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, u)
pl.title('Biharmonic Elliptic')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()







