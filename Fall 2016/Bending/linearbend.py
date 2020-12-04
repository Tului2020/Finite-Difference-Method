
# coding: utf-8

# # Linear

# In[11]:

import matplotlib.pyplot as plt
import numpy as np
#get_ipython().magic('matplotlib notebook')
#inputs

w = .1                           #weight/length (Newtons/meter) steel
E = 200000000
I = 5000
L = 1                                   #meters
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
    matrixM[0][0]=1
    matrixM[n-1][n-1]=1
    matrixM[1][0]=1/(dx2)
    matrixM[n-2][n-1]=1/(dx2)

    for row in range(1,len(matrixM[0])-1):
        for column in range(1,len(matrixM[0])-1):
            if (column +1 == row or column-1==row):
                matrixM[row][column] = 1/(dx2)

            if (column == row):
                matrixM[row][column] = -2/(dx2)
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
def createMatrixD1():
    global matrixD1
    global matrixM
    rows = len(matrixM[0])
    matrixD1= np.zeros(n)
    matrixD1 = [float(w)]*rows
    #for i in range(1,n-1):
    #    matrixD1[i] = w*x[i]
    matrixD1[0]=0
    matrixD1[len(matrixD1)-1]=0
    print(matrixD1)

def createXaxis(n,dx):
    global x
    x=[dx]*n
    for c in range (n):
        x[c]=c*dx


# M Implementation
createXaxis(n,dx)
#u = np.zeros(n)
#m = np.zeros(n)
#for i in range (0, n-1):
#    u[i] = (w*x[i]**4/(24*E*I)) - (w*x[i]**3/(12*E*I)) + x[i]*((w/(12*E*I)) - (w/(24*E*I)))
#    m[i] = ((w*x[i]**2)/2) - (w*x[i])/2
createMatrixM(n)
createMatrixD1()
res1=trisolver(matrixM,matrixD1)

# U Implementation
createMatrixA(n)
res1 = res1 / (E*I)
res2=trisolver(matrixA, res1)
createXaxis(n,dx)


#Linear solution check
u = np.zeros(n)
m = np.zeros(n)
for i in range (0, n-1):
    u[i] = (w*x[i]**4/(24*E*I)) - (w*x[i]**3/(12*E*I)) + x[i]*((w/(12*E*I)) - (w/(24*E*I)))
    m[i] = ((w*x[i]**2)/2) - (w*x[i])/2
# plotting

#plt.plot(x,m,'b')
plt.plot(x,res1,'k')
plt.figure()
plt.plot(x, u, 'k')
plt.plot(x, res2, 'r')
plt.title("linear, size " + str(n))
#plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()


# In[15]:

import matplotlib.pyplot as plt
import numpy as np
import math 
get_ipython().magic('matplotlib notebook')
#inputs
def routine(num):   
    w = 78970.5                             #weight/length (Newtons/meter) steel
    L = 1                                   #meters
    #H = 78970.5                             #horizontal tension in Newtons
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
        matrixD = [float(w)]*rows
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
    #for i in range (0,n):
    #    e = math.e
    #    c = math.log((1 - math.pow(e, -w / H)) / (math.pow(e, w / H) - 1), e) / 2
    #    u[i] = H/w*(math.cosh(w*ex[i]/ H + c) - math.cosh(c))
    
    plt.plot(ex,u, 'r')
    plt.plot(ex,res, 'k')
    plt.title('Red is exact, Black is linear approx')
    plt.plot(ex,u, 'r')
    plt.plot(ex,res, 'k')
    plt.title('Red is exact, Black is linear approx')
    return res, u , dx
n = 100
w = 78970.5                             #weight/length (Newtons/meter) steel
L = 1                                   #meters
#H = 78970.5                            #horizontal tension in Newtons                                #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
res, u , dx = routine(n) 


# In[ ]:



