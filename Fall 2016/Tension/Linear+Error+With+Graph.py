
# coding: utf-8

# In[146]:

import matplotlib.pyplot as plt
import numpy as np
import math 
#get_ipython().magic('matplotlib notebook')
#inputs
def routine(num):    
    w = 78970.5                             #weight/length (Newtons/meter) steel
    L = 1                                   #meters
    H = 78970.5                             #horizontal tension in Newtons
    n = num                                 #nxn matrix
    dx= float(L)/(n-1)
    dx2=dx*dx

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
        matrixA = [[0]*n for i in range(n)]
        matrixA[0][0]=1
        matrixA[n-1][n-1]=1
        matrixA[1][0]=1/(dx2)
        matrixA[n-2][n-1]=1/(dx2)
    
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
    
    
    # Implementation
    createMatrixA(n)
    createMatrixD()
    res=trisolver(matrixA,matrixD)
    createXaxis(n,dx)

    u = np.zeros(n)
    ex = np.linspace(0 , 1, num=n)
    for i in range (0,n):
        e = math.e
        c = math.log((1 - math.pow(e, -w / H)) / (math.pow(e, w / H) - 1), e) / 2
        u[i] = H/w*(math.cosh(w*ex[i]/ H + c) - math.cosh(c))
    
    plt.plot(ex,u, 'r')
    plt.plot(ex,res, 'k')
    plt.title('Red is exact, Black is linear approx')
    return res, u , dx
n = 100
w = 78970.5                             #weight/length (Newtons/meter) steel
L = 1                                   #meters
H = 78970.5                            #horizontal tension in Newtons                                #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
res, u , dx = routine(n)    


#empty arrays for difference calculations
dif = np.zeros(n)
error = np.zeros(n)
c = np.arange(n-1, 1, -1)

DX = np.zeros(n-2)
DX[0] = dx
error = np.zeros(n-2)


#calculating difference/error
for k in range (0 , n-2):
    for m in range (0,c[k]):
        dif[m] = (res[m] - u[m])**2
    error[k] = 1/(c[k]) * sum(dif)
    #print(DX)
    res, u , DX[k] = routine(c[k])
    m = 0
plt.figure()
plt.plot(DX,error)
plt.ylabel('Mean Square Error')
plt.xlabel('dx step size')

#plt.plot(x,error)


# In[ ]:



