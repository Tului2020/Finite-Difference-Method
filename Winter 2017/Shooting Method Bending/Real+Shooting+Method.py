
# coding: utf-8

# In[28]:

import matplotlib.pyplot as plt
import numpy as np
#get_ipython().magic('matplotlib notebook')

def shoot(g1, num, E1, I1, M1):
    n = num
    L = 1
    E = E1
    I = I1
    M = M1
    x = np.linspace(0, 1, n)
    dx = L/(n-1)
    
    #initialize arrays
    u1 = np.zeros(n)
    u2 = np.zeros(n)

    #slope guess
    guess1 = g1
    
    
    u1[0] = 0     # u at zero
    u2[0] = guess1 # first derivative u at zero 
    
    # u at 1
    u1[1] = u1[0] + dx*u2[0] + .5*dx*dx*M/(E*I) # u1 = u0 + du0/du + (1/2)dx^2 * d^2u0/dx^2
    
    for i in range (1, n-1):
        u1[i+1] = 2*u1[i] - u1[i-1] + dx*dx*M/(E*I) #from finite difference approx for 2nd derivative, solving for ui+1
        

    return(u1,u2,x)



#Constants
E = 1
I = 1
M = 1
n = 100

#Error to interpolate the slope guesses
Uerror = [0,0]


j=0
#Desired value at u(n)
Desired_Boundary_Deflection = 0

#first slope guess
g1 = -1 

#shooting with first guess
u1,u2,x = shoot(g1, n, E, I, M)
#calculate error with first slope guess
Uerror[0] = u1[n-1] - Desired_Boundary_Deflection

#second slope guess
g1 = 1 
#shooting with second guess
u1,u2,x = shoot(g1, n, E, I, M)
#calculate error with second slope guess
Uerror[1] = u1[n-1] - Desired_Boundary_Deflection

#interpolate slope guesses to get the proper guess
adjustg1 = -1 -(Uerror[0]*(1+(1)))/(Uerror[1]-Uerror[0])
#shoot with proper slope guess
u1,u2,x = shoot(adjustg1, n, E, I, M)
#print proper slope guess
print(adjustg1)

#initialize arrays for actual solution calculation
z = np.zeros(n)
y= np.zeros(n)
err = np.zeros(n)
i=0
for i in range (0, n-1):
    #z[i] = (w*x[i]**4/(24*E*I)) - (w*x[i]**3/(12*E*I)) + x[i]*((w/(12*E*I)) - (w/(24*E*I)))
    #actual solution
    y[i] = (M/(2*E*I))*x[i]**2 + adjustg1*x[i]
    #calculate error between actual solution and shooting method at each point
    err[i] = u1[i] - y[i]

#print error
print(err)
#plot the shooting method approximation 
plt.plot(x,u1,'r')
#plot the actual solution
plt.plot(x,y,'g')


# In[ ]:



