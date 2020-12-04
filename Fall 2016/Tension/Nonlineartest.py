
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
repetition = 5




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
        input = float(w/H)*math.sqrt((1+((res[i+1]-res[i-1])/(2*dx))**2))
        matrixD[i]=input
    matrixD[0]=0
    matrixD[len(matrixD)-1]=0







#exact solution calculation
ex = np.linspace(0 , 1, num=100)
u = np.zeros(100)

fac = (w/H)
C1 = fac*np.sqrt(2) - fac*math.asinh(1) - fac
C2 = fac

for i in range (0,100):
    u[i] = fac*ex[i]*math.asinh(ex[i]) - fac*np.sqrt(ex[i]**2+1) + C1*ex[i] + C2    

    
# Implementation

createMatrixA(n)
createMatrixD()
res = trisolver(matrixA, matrixD)
diff = np.zeros(n)
error = np.zeros(repetition)

for i in range(repetition):   
    createMatrixA(n)
    createMatrixDp()
    res=trisolver(matrixA,matrixD)
    #print(matrixD)
createXaxis(n,dx)


#reps = np.arange(0, repetition, 1)
#print( reps )
#plt.figure()
#plt.plot(reps, error)








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
#diff = np.zeros(n)
#error = np.zeros(n)
#c = np.zeros(n)
#
##calculating difference/error
#for j in range (0,n):    
#    diff[j] = abs(u[j] - res[j])
#    error[j] = np.log(diff[j])
#    c[j] = error[j]/(dx*n)    
    
plt.plot(ex,u, 'r')


# In[ ]:



