import numpy as np
from pprint import pprint




def create_matrix(n, a=1, b=1, c=1):

    matrix = b * np.eye(n)

    for i in range(len(matrix) - 1):
        matrix[i + 1][i] = a
        matrix[i][i + 1] = c
    
    return matrix



matrix = create_matrix(5, 5, 8, 6)



# print(matrix)


def first(mat):
    b1 = mat[0][0]
    mat[0][1] /= b1
    mat[0][0] /= b1
    return mat

first(matrix)

def between(mat):
    # for i in range(1, 4):
    i = 1
    ai = mat[i][i - 1]

    for j in [-1, 0, 1]:
        mat[i][i + j] /= ai
        # mat[i][i + j] -= mat[i - 1][i + j]

    return mat


pprint(between(matrix))
# for i in range(1, 3):
#     print(i)


