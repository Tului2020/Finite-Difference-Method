import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import xlsxwriter

n = 100
L = 1.0
h = 2*L/(n-1)
n2 = n**2
C = 56
x_axis = np.zeros(shape=(n, n))
y_axis = np.zeros(shape=(n, n))
base = np.arange(-L, L, h)
if not base[-1] == L:
    base = np.append(base, L)
for row in range(n):
    for col in range(n):
        x_axis[row][col] = y_axis[-1-col][row] = base[col]

def analytical(x1, y1):
    return x1**4 + y1**4 + (x1**2)*(y1**2)


def excel(s, whatev='s'):
    book = xlsxwriter.Workbook('/Users/tului/Desktop/'+whatev+'.xlsx')
    ws = book.add_worksheet('Sheet1')
    try:
        for i in range(len(s)):
            for j in range(len(s[i])):
                ws.write_number(i+2, j+2, s[i][j])
    except TypeError:
        for i in range(len(s)):
            ws.write_number(i+2, 2, s[i])
    book.close()


def create_matrix_vector(n1=n):
    retm = np.identity(n2)
    retv = np.zeros(n2)
    # 0 = bcw, 1 = bcs, 2 = bcn, 3 = bce
    bc_list = [np.arange(1, n1 - 1),
              np.multiply(np.arange(n1), n1),
              np.multiply(np.arange(1, n1+1), n1) - np.ones(n1).astype(int),
              np.arange(n1**2 - n1 + 1, n1**2 - 1)]

    # 0 = bcw, 1 = bcs, 2 = bcn, 3 = bce
    inner = [np.delete(np.delete(np.add([bc_list[0]], n1), 0), -1),
             np.add(np.delete(np.delete(bc_list[1], 0), -1), 1),
             np.subtract(np.delete(np.delete(bc_list[2], 0), -1), 1),
             np.delete(np.delete(np.subtract([bc_list[3]], n1), 0), -1)]


    apply_eqn = []
    for i in range(n1-4):
        s = np.arange(n1-4) + np.add(np.multiply(np.ones(n1-4), 2*n1+2), n1*i)
        apply_eqn.append(s)

    # west
    for j in bc_list[0]:
        retv[j] = analytical(-L, -L + (j % n1) * h)

    # south
    for j in bc_list[1]:
        retv[j] = analytical(-L + int(j / n1) * h, -L)

    # north
    for j in bc_list[2]:
        retv[j] = analytical(-L + int(j / n1) * h, L)

    # east
    for j in bc_list[3]:
        retv[j] = analytical(L, -L + (j % n1) * h)

    # inner west
    for j in inner[0]:
        retv[j] = analytical(-L + h, -L + (j % n1) * h)

    # inner south
    for j in inner[1]:
        retv[j] = analytical(-L + int(j / n1) * h, - L + h)

    # inner north
    for j in inner[2]:
        retv[j] = analytical(-L + int(j / n1) * h, L - h)

    # inner east
    for j in inner[3]:
        retv[j] = analytical(L - h, -L + (j % n1) * h)


    for i in apply_eqn:
        for j in i:
            r = int(j)
            retv[r] = C * h **4
            retm[r][r] = 20
            retm[r][r - 2 * n1] = retm[r][r - 2] =retm[r][r + 2 * n1] = retm[r][r + 2] =1
            retm[r][r - n1 - 1] = retm[r][r - n1 + 1] = retm[r][r + n1 + 1] = retm[r][r + n1 - 1] = 2
            retm[r][r - n1] = retm[r][r - 1] = retm[r][r + n1] = retm[r][r + 1] = -8
    return retm, retv

matrix, vector = create_matrix_vector()
z = np.rot90(np.linalg.solve(matrix, vector).reshape(n, n))
#excel(vector, 's1')
#excel(matrix)

a = np.zeros(shape=(n, n))
e = np.zeros(shape=(n, n))

for i in range(n):
    for j in range(n):
        a[i][j] = analytical(x_axis[i][j], y_axis[i][j])
        e[i][j] = abs(a[i][j] - z[i][j])


#excel(a, 's2')
fig = pl.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x_axis, y_axis, z, color='r')
#surf = ax.plot_surface(x_axis, y_axis, a)
surf = ax.plot_surface(x_axis, y_axis, e)
pl.title('Biharmonic')
pl.xlabel('X-Axis')
pl.ylabel('Y-Axis')
pl.show()








