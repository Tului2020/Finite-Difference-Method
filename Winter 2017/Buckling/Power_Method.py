import numpy as np
import math
from pprint import pprint
# get largest eigenvalue
rep = 1000
n = 50
x = np.arange(float(n))

def create_a(n1=n):
    ret = [[0]*n1 for i in range(n1)]
    for i in range(n1):
        try:
            ret[i][i - 1] = 1
            ret[i][i - 0] = -2
            ret[i][i + 1] = 1
        except IndexError:
            None
    return np.array(ret)
def power_method(a1, x1, r=rep):
    for i in range(r):
        s1 = a1.dot(x1)
        lam1 = np.linalg.norm(s1)
        x1 = np.divide(s1, lam1)
    return x1, lam1
#a = np.array([[25, 1, 0], [1, 3, 0], [2,  0, -4]])
a = create_a()
print(a)
#x = np.array([1, 0, 0])
p = power_method(a, x)
print(p[1])
print(np.linalg.eigvals(a))
