# Ax=d
def createMatrixA(nxn,fn,sn,tn):
    global matrixA
    matrixA = [[0]*nxn for i in range(nxn)]

    for row in range(len(matrixA[0])):
        for column in range(len(matrixA[0])):
            if (column - 1 == row):
                matrixA[row][column] = fn
            if (column == row):
                matrixA[row][column] = sn
            if (column + 1 == row):
                matrixA[row][column] = tn
def createMatrixD():
    global matrixD
    global matrixA
    rows = len(matrixA[0])
    matrixD = [0]*rows
    #matrixD[0]=-1
    #matrixD[len(matrixD)-1]=-1
    for i in range(rows):
        matrixD[i]=i+1

# Ap*x=dp
def createMatrixAp():
    global matrixAp
    matrixAp = [[0]*len(matrixA)for i in range(len(matrixA))]

    for row in range(len(matrixAp[0])):
        for column in range(len(matrixAp[0])):
            if (column == row):
                matrixAp[row][column] =1
            if (column ==1 and row == 0):
                cp1=matrixA[row][column]/matrixA[row][column-1]
                matrixAp[row][column] = cp1
            elif (column == row+1):
                cpi = matrixA[row][column]/(matrixA[row][column-1]-matrixAp[row-1][column-1]*matrixA[row][column-2])
                matrixAp[row][column] = cpi
def createMatrixDp():
    global matrixDp
    leng = len(matrixA)
    matrixDp = [[0] for i in range(leng)]
    for row in range(len(matrixAp[0])):
        if (row==0):
            matrixDp[row]=matrixD[row]/matrixA[0][0]
        else:
            dpi=(matrixD[row]-matrixDp[row][0]*matrixA[row][row-1])/(matrixA[row][row]-matrixAp[row-1][row]*matrixA[row][row-1])
            matrixDp[row]=dpi
def createMatrixX():
    global matrixX
    global matrixA
    leng = len(matrixA[0])
    matrixX = [0]*leng
    leng = len(matrixX)
    for i in range (len(matrixX)):
        var = leng-i-1
        if (var==len(matrixX)-1):
            matrixX[var]=matrixDp[var]/matrixAp[var][var]
        else:
            xi=(matrixDp[var]-matrixX[var+1]*matrixAp[var][var+1])/(matrixAp[var][var])
            matrixX[var]=xi

createMatrixA(3,1,-2,1)
createMatrixD()
createMatrixAp()
createMatrixDp()
createMatrixX()

for i in range(len(matrixA[0])):
    print matrixA[i]


print ""

print matrixD
print ""


for i in range(len(matrixAp[0])):
   print matrixAp[i]
print ""
print matrixDp
print ""
print matrixX






