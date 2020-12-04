
# coding: utf-8

# In[45]:

import matplotlib.pyplot as plt
import numpy as np
#%matplotlib notebook
#inputs

T = 1                           #weight/length (Newtons/meter) steel
C1= .1
C2= .5
L = 1                                   #meters
n = 100                                 #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
lam1 = np.sqrt(C1/C2)
lam2 = -np.sqrt(C1/C2)

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

def createMatrixM(n):
    global matrixM
    matrixM = [[0]*n for i in range(n)]
    matrixM[0][0]= 1
    matrixM[n-1][n-1]= 1
    matrixM[1][0]=-C2/(dx2)
    matrixM[n-2][n-1]=-C2/(dx2)

    for row in range(1,len(matrixM[0])-1):
        for column in range(1,len(matrixM[0])-1):
            if (column +1 == row or column-1==row):
                matrixM[row][column] = -C2/(dx2)

            if (column == row):
                matrixM[row][column] = (2*C2/(dx2))+C1
def createMatrixD1():
    global matrixD1
    global matrixM
    rows = len(matrixM[0])
    matrixD1= np.zeros(n)
    matrixD1 = [float(T)]*rows
    matrixD1[0]=0
    #matrixD1[len(matrixD1)-1]=0
def createXaxis(n,dx):
    global x
    x=[dx]*n
    for c in range (n):
        x[c]=c*dx


# M Implementation
createXaxis(n,dx)
createMatrixM(n)
createMatrixD1()

res=trisolver(matrixM,matrixD1)



B= ((-T/C1)*(lam1*np.exp(lam1*L)))/(lam1*np.exp(lam1*L) - lam2*np.exp(lam2*L))
A = -B - T/C1
u = np.zeros(n)

for i in range(0, n):
    u[i] = A*np.exp(lam1*x[i]) + B*np.exp(lam2*x[i]) + T/C1
print(u)

plt.plot(x,res,'k', label=('Linear'))
plt.title("Linear Torsion, size " + str(n))
plt.plot(x,u,'b', label='Actual')
plt.legend()



# In[ ]:



