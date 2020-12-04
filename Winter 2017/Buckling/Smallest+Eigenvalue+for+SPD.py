
# coding: utf-8

# In[4]:

#Finds the Dominant Eigenvalue of matrixA
import matplotlib.pyplot as plt
import numpy as np
import math
get_ipython().magic('matplotlib notebook')
#inputs

L = 1                                  #meters
initial = 1                           #multiplier for intial u vector
rep = 100
n = 100                          #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx

def createMatrixA(n):
    global matrixA
    matrixA = [[0]*(n) for i in range(n)]
    last=len(matrixA[0])-1
    dx2 = math.pow(last, -2)
    for row in range(n):
        for col in range(n):
            if (row-col==1 or row-col==-1):
                matrixA[row][col] = -1/ (dx2)
            if (col==row):
                matrixA[row][col] = 2/ (dx2)
    matrixA[0][0]=1
    matrixA[last][last]=1
    matrixA[0][1] = 0
    matrixA[last][last-1] = 0
                
#initial u assumption != 0
u_0 = np.transpose(np.ones(n)*initial)
u = u_0
mat_lam = np.ones(rep)
x = np.ones(rep)
j = 0
for j in range(0,rep):
    x[j] = j
    
# Implementation
#print(mat_lam)

createMatrixA(n)
i = 0


for i in range(0, (rep)):
    u_1 = matrixA @ np.transpose(u_0)   #calculation of next u value
    u_sq = u_1 @ np.transpose(u_1)     #squared value of next u value
    lam = (u_0 @ u_1)/(u_0 @ u_0)
    #print(u_0 @ u_1)
    u_1 = u_1 / np.sqrt(u_sq)          #normalized version of next u value
    
    u_0 = u_1
    mat_lam[i] = lam

eigval, eigvec = np.linalg.eig(matrixA)
print( max(eigval))

plt.figure()
plt.plot(x, mat_lam , 'k')
plt.title("largest eigenvalue, repetitions " + str(rep))
plt.ylabel("eigenvalue")
plt.xlabel("repetitions")
plt.grid(True)
plt.show()


# In[7]:

#Finds the Smallest Eigenvalue of matrixA
import matplotlib.pyplot as plt
import numpy as np
import math
get_ipython().magic('matplotlib notebook')
#inputs

L = 1                                  #meters
initial = 1                           #multiplier for intial u vector
rep = 10
n = 5                                  #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
#remove first and last U liness
def createMatrixA(n):
    global matrixA
    matrixA = [[0]*(n) for i in range(n)]
    last=len(matrixA[0])-1
    dx2 = math.pow(last, -2)
    for row in range(n):
        for col in range(n):
            if (row-col==1 or row-col==-1):
                matrixA[row][col] = -1/ (dx2)
            if (col==row):
                matrixA[row][col] = 2/ (dx2)
    matrixA[0][0]=1
    matrixA[last][last]=1
    matrixA[0][1] = 0
    matrixA[last][last-1] = 0
    
#initial u assumption != 0
cccccccccc


# In[30]:

#Finds the Smallest Eigenvalue of matrixA
import matplotlib.pyplot as plt
import numpy as np
import math
get_ipython().magic('matplotlib notebook')
#inputs

L = 1                                  #meters
initial = 1                           #multiplier for intial u vector
rep = 10
n = 1000                       #nxn matrix
dx= float(L)/(n-1)
dx2=dx*dx
#remove first and last U liness
def createMatrixM(n):
    global matrixM
    matrixM = [[0]*n for i in range(n)]
    matrixM[0][0]= -2/(dx2)
    matrixM[0][1] = 1/(dx2)
    matrixM[n-1][n-1]=-2/(dx2)
    matrixM[n-2][n-2]= 1/(dx2)
    matrixM[1][0]=1/(dx2)
    matrixM[n-2][n-1]=1/(dx2)

    for row in range(1,len(matrixM[0])-1):
        for column in range(1,len(matrixM[0])-1):
            if (column +1 == row or column-1==row):
                matrixM[row][column] = 1/(dx2)

            if (column == row):
                matrixM[row][column] = -2/(dx2)
createMatrixM(n)

u_0 = np.transpose(np.ones(n)*initial)
u = u_0
mat_lam = np.ones(rep)
x = np.ones(rep)
j = 0
for j in range(0,rep):
    x[j] = j
    
# Implementation
#print(mat_lam)

createMatrixM(n)

i = 0
matrixM = np.linalg.inv(matrixM)

for i in range(0, (rep)):
    u_1 = matrixM @ np.transpose(u_0)   #calculation of next u value
    u_sq = u_1 @ np.transpose(u_1)     #squared value of next u value
    lam = (u_0 @ u_1)/(u_0 @ u_0)
    #print(u_0 @ u_1)
    u_1 = u_1 / np.sqrt(u_sq)          #normalized version of next u value
    
    u_0 = u_1
    mat_lam[i] = lam**(-1)

eigval, eigvec = np.linalg.eig(matrixM)
print( min(eigval)**-1)

plt.figure()
plt.plot(x, mat_lam , 'k')
plt.title("largest eigenvalue, repetitions " + str(rep))
plt.ylabel("eigenvalue")
plt.xlabel("repetitions")
plt.grid(True)
plt.show()


# In[6]:

def createMatrixM(n):
    global matrixM
    matrixM = [[0]*n for i in range(n)]
    matrixM[0][0]= -2/(dx2)
    matrixM[0][1] = 1/(dx2)
    matrixM[n-1][n-1]=-2/(dx2)
    matrixM[n-2][n-2]= 1/(dx2)
    matrixM[1][0]=1/(dx2)
    matrixM[n-2][n-1]=1/(dx2)

    for row in range(1,len(matrixM[0])-1):
        for column in range(1,len(matrixM[0])-1):
            if (column +1 == row or column-1==row):
                matrixM[row][column] = 1/(dx2)

            if (column == row):
                matrixM[row][column] = -2/(dx2)
createMatrixM(n)
print(matrixM)


# In[ ]:



