import numpy as np
n = 10


def tri_solver1(m2, v2):
    n = len(v2)
    m1 = [[0]*n for i in range(n)]
    v1 = []
    ret = []
    for row in range(len(m2)):
        v1.append(v2[row])
        for col in range(len(m2[row])):
            m1[row][col] = m2[row][col]

    for i in range(n-1):
        s = float(m1[i+1][i]) / m1[i][i]
        for z in range(n):
            m1[i+1][z] -= s*m1[i][z]
        v1[i+1] -= s*v1[i]

    for i in range(1, n):
        s1 = float(m1[n-i-1][n-i]) / m1[n-i][n-i]
        m1[n - i - 1][n - i] -= s1 * m1[n-i][n-i]
        v1[n - i - 1] -= s1 * v1[n-i]

    for i in range(n):
        ret.append(v1[i]/m1[i][i])
    return ret


def create_five(n1=n):
    ret = [[0]*n1 for i in range(n1)]
    for row in range(n1):
        for col in range(n1):
            if row == col:
                ret[row][col] = 6
            elif abs(row-col) == 1:
                ret[row][col] = -4
            elif abs(row - col) == 2:
                ret[row][col] = 1
    return ret


def matrix_solver(matrix, vector): #this will work with any type of matrix.
    n1 = len(vector)
    m1 = [[0]*n1 for i in range(n1)]
    v1 = []
    ret = []
    for row in range(n1):
        v1.append(vector[row])
        for col in range(n1):
            m1[row][col] = matrix[row][col]

    for i in range(n1):
        s1 = m1[i][i]
        for col in range(n1):
            m1[i][col] /= s1
        v1[i] /= s1

        for row in range(n1):
            if row != i:
                s1 = m1[row][i]
                for col in range(n1):
                    m1[row][col] -= s1 * m1[i][col]
                v1[row] -= s1 * v1[i]

    return v1


m5 = create_five()
v5 = [1]*n
m5s = np.linalg.solve(m5,v5)
m5b = matrix_solver(m5, v5)



