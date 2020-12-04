# Finds the Dominant Eigenvalue of matrixA
import matplotlib.pyplot as plt
import numpy as np
import math
from pprint import pprint


L = 1                                       # meters
initial = 1                                 # multiplier for initial u vector
rep = 1000
n = 100                                     # nxn matrix
dx = float(L) / (n - 1)
dx2 = dx * dx


def createMatrixA(n):
    global matrixA
    matrixA = [[0] * (n) for i in range(n)]
    last = len(matrixA[0]) - 1
    dx2 = math.pow(last, -2)
    for row in range(n):
        for col in range(n):
            if row - col == 1 or row - col == -1:
                matrixA[row][col] = -1 / dx2
            if col == row:
                matrixA[row][col] = 2 / dx2
    matrixA[0][0] = 1
    matrixA[last][last] = 1
    matrixA[0][1] = 0
    matrixA[last][last - 1] = 0


# initial u assumption != 0
u_0 = np.transpose(np.ones(n) * initial)
u = u_0
mat_lam = np.ones(rep)
x = np.ones(rep)

for j in range(rep):
    x[j] = j


createMatrixA(n)


for i in range(rep):
    u_1 = matrixA @ np.transpose(u_0)  # calculation of next u value
    u_sq = u_1 @ np.transpose(u_1)  # squared value of next u value
    lam = (u_0 @ u_1) / (u_0 @ u_0)
    # print(u_0 @ u_1)
    u_1 = u_1 / np.sqrt(u_sq)  # normalized version of next u value

    u_0 = u_1
    mat_lam[i] = lam

eigval, eigvec = np.linalg.eig(matrixA)
print(max(eigval))

plt.figure()
plt.plot(x, mat_lam, 'k')
plt.title("largest eigenvalue, repetitions " + str(rep))
plt.ylabel("eigenvalue")
plt.xlabel("repetitions")
plt.grid(True)
plt.show()


rep = 10


# initial u assumption != 0
u_0 = np.transpose(np.ones(n) * initial)
u = u_0
mat_lam = np.ones(rep)
x = np.ones(rep)
j = 0
for j in range(rep):
    x[j] = j


createMatrixA(n)

matrixA = np.linalg.inv(matrixA)

for i in range(rep):
    u_1 = matrixA @ np.transpose(u_0)  # calculation of next u value
    u_sq = u_1 @ np.transpose(u_1)  # squared value of next u value
    lam = (u_0 @ u_1) / (u_0 @ u_0)
    # print(u_0 @ u_1)
    u_1 = u_1 / np.sqrt(u_sq)  # normalized version of next u value

    u_0 = u_1
    mat_lam[i] = lam ** (-1)

eigval, eigvec = np.linalg.eig(matrixA)


plt.figure()
plt.plot(x, mat_lam, 'k')
plt.title("largest eigenvalue, repetitions " + str(rep))
plt.ylabel("eigenvalue")
plt.xlabel("repetitions")
plt.grid(True)
plt.show()





