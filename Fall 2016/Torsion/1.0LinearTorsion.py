import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.linalg as la

#inputs
L=1
T=1
c1=1
c2=1
n=10
dx= float(L)/(n-1)
d=c2*math.pow(dx,-2)

#functions
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
def createMatrixA(n):
    global matrixA
    global last
    matrixA = [[0]*(n) for i in range(n)]
    last=len(matrixA[0])-1
    for row in range(n):
        for col in range(n):
            if (row-col==1 or row-col==-1):
                matrixA[row][col] = -d
            if (col==row):
                matrixA[row][col] = (c1+2*d)
    matrixA[0][0] = 1
    matrixA[last][last] = 1
    matrixA[0][1] = 0
    matrixA[last][last - 1] = -1
def createMatrixT():
    global matrixT
    matrixT=[T]*n
    matrixT[0]=0
    matrixT[last]=0
def actualTorsion(x):
    return 1-(math.pow(math.e,x)+math.pow(math.e,2-x))/(1+math.pow(math.e,2))
def createXaxis(n,dx):
    global matrixx
    matrixx=[dx]*(n)
    for c in range (n):
        matrixx[c]=c*dx
def createContinuous():
    global continuous
    continuous=[0]*n
    for i in range(len(continuous)):
        continuous[i]=actualTorsion(matrixx[i])



#implementation
createMatrixA(n)
createMatrixT()
discrete2=trisolver(matrixA,matrixT)
discrete=la.solve(matrixA,matrixT)
createXaxis(n,dx)
createContinuous()

#graph
plt.plot(matrixx, discrete,'r')
plt.plot(matrixx, continuous,'b')
plt.title("Linear Torsion, size " + str(n))
#plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()
