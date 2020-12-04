
# coding: utf-8

# In[52]:

import numpy as np
def trisolver (X, D):
    N = np.size(X,0)
    Y = np.transpose(np.empty(N))

    A = np.diagonal(X,-1).copy() 
    B = np.diagonal(X).copy()
    C = np.diagonal(X,1).copy()


    for i in range(0,N-1):
        mult = A[i]/B[i]
        B[i+1] = B[i+1] - (C[i]*mult)
        D[i+1] = D[i+1] - (D[i]*mult)


    for j in range(0,N):
        I = N-1 - j 
        if I == (N-1):
            Y[I] = D[I]/B[I]
        else:
            Y[I] = (D[I] - C[I]*Y[I+1])/B[I]
        
    return Y 

# of the form X*Y = D
X = np.array([[-2.0,1.0,0.0, 0.0],
                 [1.0,-2.0,1.0, 0.0],
                 [0.0,1.0,-2.0, 1.0],
                 [0.0, 0.0, 1.0, -2.0]])
D = [-1.0,0.0,0.0,-1.0]

print(trisolver(X,D))


# In[ ]:



