import matplotlib.pyplot as plt
import numpy.linalg as la
import math

#inputs
L=1
n=4
dx2= math.pow(float(L)/(n-1),2)


#functions
def createMatrixA(n):
    global matrixA
    global last
    matrixA = [[0]*(n) for i in range(n)]
    last=len(matrixA[0])-1
    for row in range(1,last):
        for col in range(n):
            if (row-col==1 or row-col==-1):
                matrixA[row][col] = 1
            if (col==row):
                matrixA[row][col] =(math.pi*dx2-2)
    matrixA[0][0] = 1
    matrixA[last][last]=1
def createMatrixO():
    global matrixO
    matrixO=[0]*n
def createXaxis(n,dx):
    global matrixx
    matrixx=[dx]*(n)
    for c in range (n):
        matrixx[c]=c*dx



#implementation
createMatrixA(n)
for i in range(len(matrixA)):
    print matrixA[i]
print""
createMatrixO()
discrete=list(la.solve(matrixA, matrixO))
createXaxis(n,math.sqrt(dx2))

print discrete

#graph
plt.plot(matrixx, discrete,'r')
#plt.plot(matrixx, continuous,'b')
plt.title("Linear Torsion, size " + str(n))
#plt.axis([0, 1, -1, 0])
plt.grid(True)
plt.show()