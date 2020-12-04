import numpy.linalg as la
import math
import matplotlib.pyplot as plt



#input
n=100
lam=[0]*(n-3)
x=[0]*(n-3)
reallam=[math.pow(math.pi,2)*4]*(n-3)
n1=1


#function
def createMatrix(n):
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
def getLambda(lower,upper):
    for i in range(len(B)):
        if (B[i]<upper and B[i]>=lower):
            return B[i]


createMatrix(n)
B=list(la.eigvals(matrixA))


#implementation
for i in range(n-3):
    x[i]=i+2
    createMatrix(i+3)
    B = list(la.eigvals(matrixA))
    lam[i]=getLambda(30,40)




#graph
plt.plot(x, lam,'r', label=('lambda'))
plt.plot(x, reallam,'b')
plt.title("Lambda, eigenvalue" )
plt.axis([1, n, 31, 40])
plt.grid(True)
plt.show()
