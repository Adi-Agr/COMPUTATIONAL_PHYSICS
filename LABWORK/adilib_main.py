import matplotlib.pyplot as plt

def read_matrix(filename):
    """Read matrix from text file with space-separated values"""
    

    with open(filename, 'r') as file:
        matrix = []
        for line in file:
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix

#==========================================================================================
def index_f(n):
    return [(i+1)/n for i in range(n)]

#==========================================================================================
def pseudo1(n,c,s=0.1):
    x_i = s
    l=[]
    for _ in range(n):
        x_i = c*x_i*(1-x_i)
        l.append(x_i)
    return l

#==========================================================================================
def LCG(N,a=1103515245,c=12345,m=32768,x_0=.1):
    l=[]
    x_i=x_0
    for i in range(N):
        x_i= ((a*x_i +c)%m)
        l.append(x_i/m)
    return l

#==========================================================================================
def corre_LCG(N,k):
    result=[]
    a=LCG(N)[:-k]
    b=LCG(N)[k:]
    result.append(a)
    result.append(b)
    return result
    

#==========================================================================================
def corre_pseudo1(n,c,k=5):
    result=[]
    a=pseudo1(n,c)[:-k]
    b=pseudo1(n,c)[k:]
    result.append(a)
    result.append(b)
    return result

#==========================================================================================
def Plot(x, y , title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label',file_name='sample_plot.png'):
    plt.plot(x, y, marker='o', linestyle='none', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  
    plt.savefig(file_name)
    plt.show()

#==========================================================================================
#funct. to create augmented matrix

def create_aug(A,B):
    n = len(A)
    aug= []
    for i in range(n):
        row=[]
        for j in range(n):
            row.append(float(A[i][j]))
        for j in range(len(B[0])):
            row.append(B[i][j])
        aug.append(row)
    return aug
#==========================================================================================
#Major matrix operations

def swap_rows(matrix,r1,r2):
    matrix[r1], matrix[r2] =matrix[r2], matrix[r1]

def scale_row(matrix,r,scale):
    for j in range(len(matrix[r])):
        matrix[r][j]*=scale

def add_rows(matrix,r1,r2,scale):
    for j in range(len(matrix[r1])):
        matrix[r1][j]+= matrix[r2][j]*scale
#==========================================================================================
#main function for gauss-jordan elimination

def gauss_jordan(A,B):
    n = len(B)

    augmented = create_aug(A,B)

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row=k
        
        #Swap rows
        if max_row != i:
            swap_rows(augmented,i,max_row)
        
        #Make diag element 1
        if augmented[i][i] != 0:
            scale_row(augmented,i,1/augmented[i][i])

        #Make other elements 0
        for k in range(n):
            if i !=k:
                scalar=-augmented[k][i]
                add_rows(augmented,k,i,scalar)

    solution = [row[-1] for row in augmented]
    return solution


#==========================================================================================