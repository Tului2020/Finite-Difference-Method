import matplotlib.pyplot as plt
import numpy as np
import math
from pprint import pprint

#inputs

w = 78970.5                             #weight/length (Newtons/meter) steel
L = 1                                   #meters
EI = 10000                              #horizontal tension in Newtons
n = 100                                 #nxn matrix
dx = float(L)/(n-1)
dx2 = dx * dx
wei = w / EI
repetition = 2




# Program
def trisolver(m2, v2):
    n = len(v2)
    m1 = [[0]*n for i in range(n)]
    v1 = [0]*n
    ret = []
    for row in range(len(m2)):
        v1[row] = v2[row]
        for col in range(len(m2[row])):
            m1[row][col] = m2[row][col]

    for i in range(n-1):
        s = float(m1[i+1][i]) / m1[i][i]
        for z in range(n):
            m1[i+1][z] -= s*m1[i][z]
        v1[i+1] -= s*v1[i]

    for i in range(1, n):
        s1 = float(m1[n-i-1][n-i]) / m1[n-i][n-i]
        m1[n - i - 1][n - i] -= s1 * m1[n-i][n-i]
        v1[n - i - 1] -= s1 * v1[n-i]
    for i in range(n):
        ret.append(v1[i]/m1[i][i])
    return ret

def trisolver2(X, D):
    N = np.size(X, 0)
    Y = np.transpose(np.empty(N))
    A = np.diagonal(X, -1).copy()
    B = np.diagonal(X).copy()
    C = np.diagonal(X, 1).copy()
    F = [0]*N
    for i in range(N-1):
        F[i]=B[i]
    for i in range(0, N - 1):
        mult = A[i] / B[i]
        B[i + 1] = B[i + 1] - (C[i] * mult)
        F[i + 1] = F[i + 1] - (F[i] * mult)
    for j in range(0, N):
        I = N - 1 - j
        if I == (N - 1):
            Y[I] = D[I] / B[I]
        else:
            Y[I] = (D[I] - C[I] * Y[I + 1]) / B[I]
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
def createMatrixW():
    global matrixW
    matrixW=[-w]*n
    matrixW[0]=0
    matrixW[last]=0
def createMatrixAA():
    for row in range(n):
        for col in range(n):
            matrixA[row][col]=matrixA[row][col]*EI
def createContinuous():
    global continuous
    continuous = [0] * (n)
    for x in range(0, n):
        rx = x / (float(n - 1))
        continuous[x] = actualV(rx)
def actualV(x):
    c = -wei * (4 * math.pow(x, 3) - 6 * math.pow(x, 2) + 1) / 24
    return c/math.sqrt(1-c*c)
def actualDeflection(x):
    c=-wei*(4*math.pow(x,3)-6*math.pow(x,2)+1)/24
    return -math.sqrt(1-c*c)+math.sqrt(1-math.pow(wei/24,2))
def actualBending(x):
    return -w*(math.pow(x,2)-x)/2
def createXaxis(n,dx):
    global matrixx
    matrixx=[dx]*(n)
    for c in range (n):
        matrixx[c]=c*dx
def createMatrixBp():
    B[0]=0.0
    B[last]=0.0
    for i in range(1,last):
        CT=math.pow(1+math.pow((discrete[i+1]-discrete[i-1])/(2*dx),2),3/2)
        #print CT
        B[i]=CT*B[i]
        #print B[i]
def createIntegral(vector):
    sv=[0]*n
    for i in range(1,n):
        rx = 1.0 / (n)
        sv[i]=sv[i-1]+rx*(vector[i]-vector[i-1])/2
    return sv
def createDerivative(vector):
    sv=[0]*n
    rx = 1.0 / (n)
    sv[0]=(vector[1]-vector[0])/(rx)
    for i in range(1,n):
        sv[i]=(vector[i]-vector[i-1])/rx
    return sv

createXaxis(n, dx)
createMatrixA(n)
createMatrixW()
pprint(matrixA)
print(matrixW)
B = trisolver(matrixA, matrixW)
createMatrixAA()
discrete = trisolver(matrixA, B)
createContinuous()





for i in range(repetition):
    createMatrixBp()
    discrete = trisolver(matrixA, B)
discrete = createDerivative(discrete)




plt.plot(matrixx, discrete,'r')
plt.plot(matrixx, continuous,'b')
plt.title("nonlinear, size: " + str(n)+ " repetition: "+ str(repetition))
#plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()

