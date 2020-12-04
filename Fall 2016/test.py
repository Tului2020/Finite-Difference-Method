import numpy.linalg as la
import math
"""x0=[1,0,0,0,0]
x1=[1,-2,1,0,0]
x2=[0,1,-2,1,0]
x3=[0,0,1,-2,1]
x4=[0,0,0,0,1]

A=([1,0,0,0,0],[1,-2,1,0,0],[0,1,-2,1,0],[0,0,1,-2,1],[0,0,0,0,1])

print x0
print x1
print x2
print x3
print x4
print ""
for i in range(len(x0)):
    x0[i]=1.0*x0[i]
    x4[i] = 1.0 * x4[i]

for i in range(len(x0)):
    x1[i]=x1[i]-1.0*x0[i]

for i in range(len(x1)):
    x2[i]=x2[i]+0.5*x1[i]

for i in range(len(x1)):
    x3[i]=x3[i]+2.0/3*x2[i]
    #x3[i] = 1.0 * x3[i]


print x0
print x1
print x2
print x3
print x4"""


#A=([-50,25,0,0],[25,-50,25,0],[0,25,-50,25],[0,0,25,-50])
A=([25,0,0,0,0,0],[25,-50,25,0,0,0],[0,25,-50,25,0,0],[0,0,25,-50,25,0],[0,0,0,25,-50,25],[0,0,0,0,0,25])
for i in range(len(A[0])):
    for x in range(len(A[0])):
        A[x][i]=A[x][i]/25
B=list(la.eigvals(A))
print B

for i in range(len(B)):
    B[i]=25*B[i]

print B