
# coding: utf-8

# In[48]:

import numpy as np

def pentasolver(X,D):
    N = np.size(X, 0)
    Y = np.transpose(np.empty(N))
    
    e = np.diagonal(X, -2).copy()
    c = np.diagonal(X, -1).copy()
    d = np.diagonal(X).copy()
    a = np.diagonal(X, 1).copy()
    b = np.diagonal(X, 2).copy()
    #Check for nonsingular matrix
    if (np.linalg.det(X) == 0):
        print('det = 0, no solution')
    
    #initialize variables to simplify pentadiagonal matrix
    mu = np.zeros(N)
    beta = np.zeros(N-2)
    z = np.zeros(N)
    alph = np.zeros(N-1)
    gam = np.zeros(N)
    
    
    #begin assigning noniterative variables at i = 1
    mu[0] = d[0]
    alph[0] = a[0] / mu[0]
    beta[0] = b[0] / mu[0]
    z[0] = y[0] / mu[0]
    
    #begin assigning noniteratie variables at i=2
    gam[1]= c[0]
    mu[1] = d[1] - alph[0]*gam[1]
    alph[1] = (a[1] - beta[0]*gam[1]) / mu[1]
    beta[1] = b[1]/mu[1]
    z[1] = (y[1] - z[0]*gam[1]) / mu[1]
    
    
    #begin assigning iterative variables at i = 3
    for i in range(3, N-1):
        gam[i-1] = c[i-2] - alph[i-3]*e[i-3]
        mu[i-1] = d[i-1] - beta[i-3]*e[i-3] - alph[i-2]*gam[i-1]
        alph[i-1] = (a[i-1] - beta[i-2]*gam[i-1]) / mu[i-1]
        beta[i-1] = b[i-1] / mu[i-1]
        z[i-1] = (y[i-1] - z[i-3]*e[i-3]- z[i-2]*gam[i-1]) / mu[i-1]
        
    #assign noniterative variables at i = N-1
    gam[N-2] = c[N-3] - alph[N-4]*e[N-4]
    mu[N-2] = d[N-2] - beta[N-4]*e[N-4] - alph[N-3]*gam[N-2]
    alph[N-2] = (a[N-2] - beta[N-3]*gam[N-2])/ mu[N-2]
    #assign noniterative variables at i = N
    gam[N-1] = c[N-2] - alph[N-3]*e[N-3]
    mu[N-1] = d[N-1] - beta[N-3]*e[N-3] - alph[N-2]*gam[N-1]
    z[N-2] = (y[N-2] - z[N-4]*e[N-4] - z[N-3]*gam[N-2]) / mu[N-2]
    z[N-1] = (y[N-1] - z[N-3]*e[N-3] - z[N-2]*gam[N-1]) / mu[N-1]
    
    
    
    
    
    k = np.arange(N-2,0,-1)
    x = np.zeros(N)
    x[N-1] = z[N-1]
    x[N-2] = z[N-2] - alph[N-2]*x[N-1]
    for j in k:
        x[j-1] = z[j-1] - alph[j-1]*x[j] - beta[j-1]*x[j+1]
    #print(x)
    #
    #error in calculating x[4]/x[5]? so z[4] and z[5] are erroneous 
    
    
    return(x)
X = [[9,-4,1,0,0,0],
    [-4,6,-4,1,0,0],
    [1,-4,6,-4,1,0],
    [0,1,-4,6,-4,1],
    [0,0,1,-4,5,-2],
    [0,0,0,1,-2,1]]

D = [6,-1,0,0,0,0]
print(pentasolver(X,D))


# In[ ]:



