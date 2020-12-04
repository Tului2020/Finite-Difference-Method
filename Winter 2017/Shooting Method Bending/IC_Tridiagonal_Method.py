import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np

n = 100
L = 1.0
dx = L / (n - 1)
dx4 = dx * dx * dx * dx


def create_x(n1=n, dx1=dx):
    ret = []
    for i in range(n1):
        ret.append(i * dx1)
    return ret


def create_five(n1=n):
    ret = [[0]*n1 for i in range(n1)]
    for row in range(2, n1-2):
        for col in range(n1):
            if row == col:
                ret[row][col] = 6
            elif abs(row-col) == 1:
                ret[row][col] = -4
            elif abs(row - col) == 2:
                ret[row][col] = 1
    return ret


def young_modulus(x1):
    return 1000 + x1 * 0


def second_moment_area(x1):
    return 1000 + x1 * 0


def distributed_loading(x1):
    return -100000000 + x1 * 0


def matrix_solver(matrix1, vector1): #this will work with any typ of matrix.
    n1 = len(vector1)
    m1 = [[0]*n1 for i in range(n1)]
    v1 = []
    for row in range(n1):
        v1.append(vector1[row])
        for col in range(n1):
            m1[row][col] = matrix1[row][col]

    for i in range(n1):
        s1 = m1[i][i]
        for col in range(n1):
            m1[i][col] /= s1
        v1[i] /= s1

        for row in range(n1):
            if row != i:
                s1 = m1[row][i]
                for col in range(n1):
                    m1[row][col] -= s1 * m1[i][col]
                v1[row] -= s1 * v1[i]

    return v1


def create_w(n1=n, dx1=dx, dx41=dx4):
    ret = []
    for i in range(n1):
        ret.append(dx41 * distributed_loading(i * dx1) / (young_modulus(i * dx1) * second_moment_area(i * dx1)))
    return ret


"""def get_ay(f, dx1=dx, l1=L):
    iw = integral(f)[-1]
    temp = []
    for i1 in range(len(f)):
        temp.append(f[i1]*i1*dx1)
    iwx = integral(temp)[-1]/l1
    return iwx - iw"""


Ay = -distributed_loading(0) * L / 2
matrix = create_five(n)             # creates matrix
w = create_w()                      # create w
xa = create_x()

# Simply supported ----------------------------------- 1

title = "Simply Supported Beam"
# Deflection - Left Side
matrix[0][0] = 1
w[0] = 0

# Moment - Left Side
matrix[1][0] = 1
matrix[1][1] = -2
matrix[1][2] = 1
w[1] = 0

# Shear - Left Side
matrix[-2][0] = -1
matrix[-2][1] = 2
matrix[-2][3] = -2
matrix[-2][4] = 1
w[-2] = Ay * 2 * dx * dx * dx / (young_modulus(xa[-2]) * second_moment_area(xa[-2]))


# Slope - Left Side
matrix[-1][0] = -1
matrix[-1][1] = 1

for g in range(-4246000, -4240000):
    w[-1] = g * dx / (young_modulus(xa[-1]) * second_moment_area(xa[-1]))
    u = np.linalg.solve(matrix, w)
    if abs(u[-1]) < 0.0000001:
        guess = g
        print(g)
        break

# Simply supported ----------------------------------- 2
guess = -4245358
w[-1] = guess * dx / (young_modulus(xa[-1]) * second_moment_area(xa[-1]))
u = np.linalg.solve(matrix, w)


plt.plot(xa, u, 'r')
plt.title(title)
plt.grid(True)
plt.show()





