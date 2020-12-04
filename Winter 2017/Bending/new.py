"""import matplotlib.pyplot as plt
import numpy as np
import math
import pprint

#inputs

L = 1                                   #meters
n = 20                                  #nxn matrix at least 3x3
dx = float(L)/(n-1)
dx2 = dx*dx

def trisolver(X, D):
    N = np.size(X, 0)
    Y = np.transpose(np.empty(N))
    A = np.diagonal(X, -1).copy()
    B = np.diagonal(X).copy()
    C = np.diagonal(X, 1).copy()
    F = [0]*n
    for i in range(N):
        F[i]=D[i]
    for i in range(0, N - 1):
        mult = A[i] / B[i]
        B[i + 1] = B[i + 1] - (C[i] * mult)
        F[i + 1] = F[i + 1] - (F[i] * mult)
    for j in range(0, N):
        I = N - 1 - j
        if I == (N - 1):
            Y[I] = F[I] / B[I]
        else:
            Y[I] = (F[I] - C[I] * Y[I + 1]) / B[I]
    return list(Y)
def create_a(n):
    m = [[0] * (n) for i in range(n)]
    last = len(m[0]) - 1
    m[0][0] = 1
    m[last][last] = 1
    m[1][0] = 1 / dx2
    m[last - 1][last] = 1 / dx2
    for row in range(1, last):
        for column in range(1, last):
            if column + 1 == row or column - 1 == row:
                m[row][column] = 1 / dx2
            if column == row:
                m[row][column] = -2 / dx2 - 1
    return m

def create_u(n,i=0):
    ret = [i]*n
    ret[0] = 1
    ret[-1] = math.cosh(1)
    return ret

def createXaxis(n,dx):
    matrixx=[dx]*(n)
    for c in range (n):
        matrixx[c]=c*dx
    return matrixx
def f(x):
    return math.cosh(x)
def createContinuous(n):
    continuous = [0] * n
    for x in range(0, n):
        rx = x / (float(n - 1))
        continuous[x] = f(rx)
    return continuous



def cubic_spline(n):
    m = [[0] * n for i in range(n)]
    last = len(m[0]) - 1
    m[0][0] = 1
    m[last][last] = 1
    m[1][0] = 1 / dx2
    m[last - 1][last] = 1 / dx2
    f = (1 / dx2) - (1 / 6)
    s = - (2 / dx2 + 4 / 6)
    for row in range(1, last):
        for column in range(1, last):
            if column + 1 == row or column - 1 == row:
                m[row][column] = f
            if column == row:
                m[row][column] = s
    m[1][0]=f
    m[-2][-1]=f

    return m


def max_error(a, b):
    me = 0
    if len(a) == len(b):
        for s in range(len(A)):
            if abs(a[s] - b[s]) > me:
                me = abs((a[s] - b[s])/a[s])
    return me


c = cubic_spline(n)
print(dx2)
for i in range(len(c[0])):
    print(c[i])


A = create_a(n)
print(dx2)
for i in range(len(c[0])):
    print(A[i])
u = create_u(n)
discrete = trisolver(A, u)
cubic = trisolver(c, u)
xa = createXaxis(n,dx)
continuous = createContinuous(n)


print(max_error(discrete, continuous), max_error(cubic, continuous))



plt.plot(xa, continuous,'b')
plt.plot(xa, discrete,'g')
#plt.plot(xa, cubic,'r')
plt.title("linear, size: " + str(n))
#plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()"""

# Does not have actual nonlinear solution, compares to linear solution

import matplotlib.pyplot as plt
import numpy as np
import math


w = -7500  # weight/length (Newtons/meter) steel
E = 20
I = 5
L = 1  # meters
n = 5  # nxn matrix
dx = float(L) / (n - 1)
dx2 = dx * dx
repetition = 0


# Program
def trisolver(X, D):
    N = np.size(X, 0)
    Y = np.transpose(np.empty(N))

    A = np.diagonal(X, -1).copy()
    B = np.diagonal(X).copy()
    C = np.diagonal(X, 1).copy()

    for i in range(0, N - 1):
        mult = A[i] / B[i]
        B[i + 1] = B[i + 1] - (C[i] * mult)
        D[i + 1] = D[i + 1] - (D[i] * mult)

    for j in range(0, N):
        I = N - 1 - j
        if I == (N - 1):
            Y[I] = D[I] / B[I]
        else:
            Y[I] = (D[I] - C[I] * Y[I + 1]) / B[I]

    return Y


def createMatrixA(n):
    global matrixA
    matrixA = [[0] * n for i in range(n)]
    matrixA[0][0] = 1
    matrixA[n - 1][n - 1] = 1
    matrixA[1][0] = 1 / (dx2)
    matrixA[n - 2][n - 1] = 1 / (dx2)

    for row in range(1, len(matrixA[0]) - 1):
        for column in range(1, len(matrixA[0]) - 1):
            if (column + 1 == row or column - 1 == row):
                matrixA[row][column] = 1 / (dx2)

            if (column == row):
                matrixA[row][column] = -2 / (dx2)


def createMatrixD():
    global matrixD
    global matrixA
    rows = len(matrixA[0])
    matrixD = [float(w)] * rows
    matrixD[0] = 0
    matrixD[len(matrixD) - 1] = 0


def createMatrixDp():
    global matrixD
    global U
    global M
    matrixDp = np.zeros(len(M))
    for i in range(1, len(M) - 1):
        input = M[i] * (1 + ((U[i + 1] - U[i - 1]) / (2 * dx)) ** 2) ** (3 / 2)
        matrixD[i] = input
    matrixD[0] = 0
    matrixD[len(matrixDp) - 1] = 0


def createXaxis(n, dx):
    global x
    x = [dx] * n
    for c in range(n):
        x[c] = c * dx


# M Implementation

createXaxis(n, dx)
createMatrixA(n)
createMatrixD()
M = trisolver(matrixA, matrixD)
# U Implementation

M = M / (E * I)
M1 = M.copy()
U = trisolver(matrixA, M1)
plt.figure()
plt.plot(x, U, 'k', label='Approx')
# print(U, 'Linear')

for k in range(repetition):
    createMatrixA(n)
    createMatrixDp()
    U = trisolver(matrixA, matrixD)
    plt.plot(x, U, 'k')
    plt.title('Nonlinear Bending')
    # print(U, 'Repetition')

u = np.zeros(n)
for i in range(0, n - 1):
    u[i] = (w * x[i] ** 4 / (24 * E * I)) - (w * x[i] ** 3 / (12 * E * I)) + x[i] * (
    (w / (12 * E * I)) - (w / (24 * E * I)))
plt.plot(x, u, 'r', label='Exact')
plt.legend()
plt.grid(True)
plt.show()



# def actualDeflection(x):
#    c=w*E*I*(4*math.pow(x,3)-6*math.pow(x,2)+1)/24
#    return -math.sqrt(1-c*c)+math.sqrt(1-math.pow(w*E*I/24,2))
# real = actualDeflection(x)
# def actualV(x):
#    c = float(-w*E*I * (4 * math.pow(x, 3) - 6 * math.pow(x, 2) + 1) / 24)
#    return c/math.sqrt(1-c*c)
# real = float( actualV(x))
# plt.plot(x,real)
# plt.plot(x, real)













