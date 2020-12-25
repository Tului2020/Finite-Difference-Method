import numpy as np
from pprint import pprint




def create_matrix(n, a=1, b=1, c=1):

    matrix = b * np.eye(n)

    for i in range(len(matrix) - 1):
        matrix[i + 1][i] = a
        matrix[i][i + 1] = c
    
    return matrix



matrix = create_matrix(5, 5, 8, 6)
print(np.round(matrix, 3))
print


def first(mat):
    b1 = mat[0][0]
    mat[0][1] /= b1
    mat[0][0] /= b1
    return mat

first(matrix)
print(np.round(matrix, 1))
print


def between1(mat):
    for i in range(1, 4):
        ai = mat[i][i - 1]

        for j in [-1, 0, 1]:
            mat[i][i + j] /= ai
            mat[i][i + j] -= mat[i - 1][i + j] / mat[i - 1][i - 1]
    return mat


between1(matrix)
print(np.round(matrix, 1))
print

def last(mat):
    an = mat[-1][-2]
    mat[-1][-1] /= an
    mat[-1][-2] /= an

    for i in [-2, -1]:
        mat[-1][i] -= mat[-2][i] / mat[-2][-2]

    mat[-1][-1] /= mat[-1][-1]
    return mat

last(matrix)
print(np.round(matrix, 1))
print

def between2(mat):

    iter = -1 * np.arange(len(mat) - 1) - 2
    for i in iter:
        print(mat[i][i])

    print
    return mat

print(between2(matrix))

