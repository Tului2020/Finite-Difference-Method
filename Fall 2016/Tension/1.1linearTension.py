import matplotlib.pyplot as plt
import numpy as np

#inputs

w = 78970.5                             #weight/length (Newtons/meter) steel
L = 1                                   #meters
H = 78970.5                              #horizontal tension in Newtons
n = 100                                   #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx


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
    matrixA = [[0]*n for i in range(n)]
    matrixA[0][0]=1
    matrixA[n-1][n-1]=1
    matrixA[1][0]=1
    matrixA[n-2][n-1]=1

    for row in range(1,len(matrixA[0])-1):
        for column in range(1,len(matrixA[0])-1):
            if (column +1 == row or column-1==row):
                matrixA[row][column] = 1/(dx2)

            if (column == row):
                matrixA[row][column] = -2/(dx2)
def createMatrixD():
    global matrixD
    global matrixA
    rows = len(matrixA[0])
    matrixD = [float(w/H)]*rows
    matrixD[0]=0
    matrixD[len(matrixD)-1]=0
def createXaxis(n,dx):
    global x
    x=[dx]*n
    for c in range (n):
        x[c]=c*dx
def actualDef(x):
    return w*x*(x-1)/(2*H)

# Implementation
createMatrixA(n)
createMatrixD()

print matrixD

res=trisolver(matrixA,matrixD)
createXaxis(n,dx)
cont=[0]*n
for i in range(n):
    cont[i]=actualDef(i*dx)

# plotting


for i in range(len(matrixA[0])):
    print matrixA[i]
print ""
print matrixD
print ""
"""
print res
"""
print res
print cont

plt.plot(x, res,'r')
#plt.plot(x, cont,'b')
plt.title("linear, size " + str(n))
plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()
















