
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import math
get_ipython().magic('matplotlib notebook')
#inputs

w = 78970.5                          #weight/length (Newtons/meter) steel
L = 1                                  #meters
H = 10000                              #horizontal tension in Newtons
n = 100                           #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
repetition = 4




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
ex = np.linspace(0 , 1, num=100)
u = np.zeros(100)



for i in range (0,100):
    e = math.e
    c = math.log((1 - math.pow(e, -w / H)) / (math.pow(e, w / H) - 1), e) / 2
    u[i] = H/w*(math.cosh(w*ex[i]/ H + c) - math.cosh(c))   

    
# Implementation

createMatrixA(n)
createMatrixD()
res = trisolver(matrixA, matrixD)

for i in range(repetition):   
    createMatrixA(n)
    createMatrixDp()
    res=trisolver(matrixA,matrixD)
    #print(matrixD)
createXaxis(n,dx)


dif = np.zeros(n)
error = np.zeros(repetition+1)
for m in range (0,n-1):
    dif[m] = (res[m] - u[m])**2
error[0] = 1/(n) * sum(dif)


#Repetition and convergence error calculation
for i in range(repetition):
    createMatrixA(n)
    createMatrixDp()
    res=trisolver(matrixA,matrixD)
    for m in range (0,n-1):
        dif[m] = (res[m] - u[m])**2
    error[i+1] = 1/(m) * sum(dif)
    print(error)
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

plt.figure()
plt.plot(x, res)
plt.title("nonlinear, size " + str(n))
#plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()


#empty arrays for difference calculations
scale = np.arange(0, repetition+1,1)


print(np.size(ex), np.size(u))
plt.plot(ex,u, 'r')
plt.figure()
plt.plot(scale, error)
plt.title('Nonlinear convergence error')
plt.ylabel('Error')
plt.xlabel('Repetitions')


# In[ ]:



