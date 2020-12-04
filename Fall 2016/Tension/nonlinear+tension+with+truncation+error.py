
# coding: utf-8

# In[12]:

import matplotlib.pyplot as plt
import numpy as np
import math
get_ipython().magic('matplotlib notebook')
#inputs
def routine(num,rep):
    w = 78970.5                          #weight/length (Newtons/meter) steel
    L = 1                                  #meters
    H = 10000                              #horizontal tension in Newtons
    n = num                           #nxn matrix
    dx= float(L)/(n-1)
    dx2=dx*dx
    repetition = rep
    
    
    
    
    # Program
    def trisolver(X, D):
        N = np.size(X, 0)
        Y = np.transpose(np.empty(N))
    
        A = np.diagonal(X, -1).copy()
        B = np.diagonal(X).copy()
        C = np.diagonal(X, 1).copy()
        #D = Dee.copy()
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
    def createMatrixDp():
        global matrixD
        global res
        for i in range(1,len(res)-1):
            input = float(w/H)*np.sqrt((1 + ((res[i+1] - res[i-1])/(2*dx))**2))
            matrixD[i]=input
        matrixD[0]=0
        matrixD[len(matrixD)-1]=0
    #exact solution calculation
    ex = np.linspace(0 , 1, num=n-1)
    u = np.zeros(n-1)

    for i in range (0,n-1):
        e = math.e
        c = math.log((1 - math.pow(e, -w / H)) / (math.pow(e, w / H) - 1), e) / 2
        u[i] = H/w*(math.cosh(w*ex[i]/ H + c) - math.cosh(c)) 


  

    
    # Implementation

    createMatrixA(n)
    createMatrixD()
    res = trisolver(matrixA, matrixD)
    return res, u, dx2
w = 78970.5                          #weight/length (Newtons/meter) steel
L = 1                                  #meters
H = 10000                              #horizontal tension in Newtons
n = 50                        #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
repetition=7
res, u , dx = routine(n,repetition)

def trisolver(X, D):
    N = np.size(X, 0)
    Y = np.transpose(np.empty(N))
    
    A = np.diagonal(X, -1).copy()
    B = np.diagonal(X).copy()
    C = np.diagonal(X, 1).copy()
        #D = Dee.copy()
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
        
def createMatrixDp():
    global matrixD
    global res
    for i in range(1,len(res)-1):
        input = float(w/H)*np.sqrt((1 + ((res[i+1] - res[i-1])/(2*dx))**2))
        matrixD[i]=input
    matrixD[0]=0
    matrixD[len(matrixD)-1]=0
    #exact solution calculation
ex = np.linspace(0 , 1, num=n-1)
u = np.zeros(n-1)
for i in range (0,n-1):
    e = math.e
    c = math.log((1 - math.pow(e, -w / H)) / (math.pow(e, w / H) - 1), e) / 2
    u[i] = H/w*(math.cosh(w*ex[i]/ H + c) - math.cosh(c)) 

trudif = np.zeros(n)
d = np.arange(n-1, 1, -1)

DX = np.zeros(n-2)
DX[0] = dx
truerror = np.zeros(n-2)


#calculating difference/error
for k in range (0 , n-2):
    for m in range (0,d[k]):
        trudif[m] = (res[m] - u[m])**2
    truerror[k] = (1/(d[k]) * sum(trudif))
    print(truerror)
    res, u , DX[k] = routine(d[k],repetition)
    m = 0
plt.figure()
plt.plot(DX,truerror)
plt.ylabel('Mean Square Error')
plt.xlabel('dx step size')






m = 0
n = 100
res, u , dx = routine(n,repetition)
#Preparing for convergence error
dif = np.zeros(n)
error = np.zeros(repetition+1)
for m in range (0,n-1):
    dif[m] = (res[m] - u[m])**2
error[0] = 1/(n) * sum(dif)
error[0] = np.log(error[0])

#Repetition and convergence error calculation
for i in range(repetition):
    createMatrixA(n)
    createMatrixDp()
    res=trisolver(matrixA,matrixD)
    for m in range (0,n-1):
        dif[m] = (res[m] - u[m])**2
    error[i+1] = 1/(m) * sum(dif)
createXaxis(n,dx)






# printing and plotting

"""
for i in range(len(matrixA[0])):
    print matrixA[i]
print ""
print matrixD
print ""
print res
"""

#plt.figure()
#plt.plot(x, res)
#plt.title("nonlinear, size " + str(n))
##plt.axis([0, 1, -1, 0])
#plt.grid(True)
#plt.show()
#
#
##empty arrays for difference calculations
#scale = np.arange(0, repetition+1,1)
#ex = np.linspace(0 , 1, num=99)
#
#print(np.size(ex), np.size(u))
#plt.plot(ex,u, 'r')
#plt.figure()
#plt.plot(scale, error)
#plt.title('Nonlinear convergence error')
#plt.ylabel('Error')
#plt.xlabel('Repetitions')
#


# In[ ]:



