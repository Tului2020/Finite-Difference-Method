import numpy as np
import math
from pprint import pprint

# get smallest eigenvalue
rep = 300
n = 5
#x = np.arange(float(n))
#a = (np.array([[25, 1, 0], [1, 3, 0], [2,  0, -4]]))
#x = np.array([1, 0, 0])
#print(np.linalg.eigvals(a))

def create_a(n1=n):
    ret = [[0]*n1 for i in range(n1)]
    for i in range(n1):
        if i == 0 or i == (n1-1):
            ret[i][i] = 1
        else:
            ret[i][i - 1] = 1
            ret[i][i - 0] = -2
            ret[i][i + 1] = 1
    return np.array(ret)
a = create_a()
x = np.arange(float(n))


def inverse_power_method(a1=a, x1=x, r=rep):
    at = np.linalg.inv(a1)
    for i in range(r):
        x11 = at.dot(x1)
        if i == rep-1:
            lam = x1.dot(x1)/x1.dot(x11)
        x1 = x11/(np.linalg.norm(x11))
    return lam, x1

p = inverse_power_method()
print(p[0])


