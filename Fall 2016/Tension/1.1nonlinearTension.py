import matplotlib.pyplot as plt
import numpy as np
import math

#inputs
w = 78970.5                             #weight/length (Newtons/meter) steel
L = 1                                   #meters
H = 18970                               #horizontal tension in Newtons
n = 100                                   #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
repetition = 10
wh=w/H


# Program
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
    matrixA[0][0]=1
    matrixA[last][last]=1
    matrixA[1][0] = 1 / (dx2)
    matrixA[last - 1][last] = 1 / (dx2)

    for row in range(1,last):
        for column in range(1,last):
            if (column +1 == row or column-1==row):
                matrixA[row][column] = 1/ (dx2)
            if (column == row):
                matrixA[row][column] = -2/ (dx2)
def createMatrixD():
    global matrixD
    matrixD = [wh]*n
    matrixD[0]=0
    matrixD[last]=0
def createXaxis(n,dx):
    global matrixx
    matrixx=[dx]*(n)
    for c in range (n):
        matrixx[c]=c*dx
def actualFunction(x):
    e = math.e
    c = math.log((1 - math.pow(e, -w / H)) / (math.pow(e, w / H) - 1), e) / 2
    u = H/w*(math.cosh(w*x/ H + c) - math.cosh(c))
    return u
def createMatrixDp():
    matrixD[0]=0
    matrixD[last]=0
    for i in range(1,last):
        CT=math.sqrt(1+math.pow((discrete[i+1]-discrete[i-1])/(2*dx),2))*wh
        matrixD[i]=CT
def createContinuous():
    global continuous
    continuous = [0] * (n)
    for x in range(0, n):
        rx = x / (float(n - 1))
        continuous[x] = actualFunction(rx)


#Implementation
createMatrixA(n)
createMatrixD()
createXaxis(n,dx)
createContinuous()
discrete = trisolver(matrixA, matrixD)

for i in range(repetition):
    createMatrixDp()
    if(i==0 or i==repetition-1):
        print matrixD
    discrete = trisolver(matrixA, matrixD)




























# printing and plotting








plt.plot(matrixx, discrete,'r')
plt.plot(matrixx, continuous,'b')
plt.title("nonlinear, size: " + str(n) + ', repetition:' + str(repetition))
#plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()


















