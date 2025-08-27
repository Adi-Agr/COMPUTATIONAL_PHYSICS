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
def mat_mult(X,Y):
    result=[]
    for i in range(len(X)):
        row=[]
        for j in range(len(Y[0])):
            sum_val=0
            for k in range(len(Y)):
                sum_val+= X[i][k] * Y[k][j]
            row.append(sum_val)
        result.append(row)
    return result

def transpose(X):
    result=[]
    for i in range(len(X[0])):
        row=[]
        for j in range(len(X)):
            row.append(X[j][i])
        result.append(row)
    return result
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
    n =len(A)
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
    n=len(B)

    augmented=create_aug(A,B)

    for i in range(n):
        max_row = i
        for k in range(i+1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row=k
        
        #Swap rows
        if max_row!=i:
            swap_rows(augmented,i,max_row)
        
        #Make diag element 1
        if augmented[i][i]!=0:
            scale_row(augmented,i,1/augmented[i][i])

        #Make other elements 0
        for k in range(n):
            if i !=k:
                scalar=-augmented[k][i]
                add_rows(augmented,k,i,scalar)

    solution=[row[-1] for row in augmented]
    return solution


#==========================================================================================
#=========ASSGN3========================
#=======================================
def ludecomp_doolittle(matrix):
    '''(diagonal of L =1)....return L and U'''
    n=len(matrix)
    L=[[0.0]*n for _ in range(n)]
    U=[[0.0]*n for _ in range(n)]

    #_____set diag 1________

    for i in range(n):
        L[i][i]=1.0

    #______main_Algo________

    for i in range(n):

        #_calculate U element
        for j in range(i,n):
            sum_val=0.0
            for k in range(i):
                sum_val+=L[i][k]*U[k][j]
            U[i][j]=matrix[i][j]-sum_val

        #_calculate L element
        for j in range(i+1,n):
            if U[i][i]==0:
                print("LU-decomp. failed--matrix singular!!!") #  <-----i took care of singularity
                return None,None
            sum_val=0.0
            for k in range(i):
                sum_val+=L[j][k]*U[k][i]
            L[j][i]=(matrix[j][i]-sum_val)/U[i][i]

    print("LU-decomp. successful")
    return L,U

#=======================================
def forward_substitution(L,B):
    n=len(L)
    Y=[0.0]*n

    print("Forward Substitution: Solving L*y=B")

    for i in range(n):
        sum_val=0.0
        for j in range(i):
            sum_val+=L[i][j]*Y[j]
        Y[i]=(B[i]-sum_val)/L[i][i]
        print(f"Y[{i}]={Y[i]:.6f}")
    return Y

def backward_substitution(U,Y):
    n=len(U)
    X=[0.0]*n

    print("Backward_Sub....Ux=y...(finding x! )")

    for i in range(n-1,-1,-1):
        sum_val=0.0
        for j in range(i+1,n):
            sum_val+=U[i][j]*X[j]
        X[i] = (Y[i]-sum_val)/U[i][i]
        print(f"x[{i}]={X[i]:.6f}")
    return X

def backward_substitution_transpose(L,y):
    n=len(L)
    X=[0.0] *n
    #___________________________________
    print("Backward Substitution: Solving L^T*x = y")
    #___________________________________
    for i in range(n-1,-1,-1):
        sum_val = 0.0
        for j in range(i+1,n):
            # Use L[j][i] instead of L[i][j] to get transpose effect
            sum_val+=L[j][i]*X[j]
        X[i]=(y[i]-sum_val)/L[i][i]
        print(f"X[{i}] = {X[i]:.6f}")    
    return X

def solve_by_backward_forward_substitution(A,B):
    L,U=ludecomp_doolittle(A)
    Y=forward_substitution(L,B)
    X=backward_substitution(U,Y)
    return X

#=======================================



def cholesky_decomposition(matrix):
    '''
    Decomposes A=L*L^T (where L is lower triangular)
    '''
    n = len(matrix)
    L = [[0.0]*n for _ in range(n)]
    
    for i in range(n):
        # Diagonal elements
        sum_val = 0.0
        for k in range(i):
            sum_val+=L[i][k] ** 2        
        # Check if matrix is >0
        if matrix[i][i]-sum_val<=0:
            print("Matrix is not positive definite!")
            return None
        #___________________________________            
        L[i][i]=(matrix[i][i]-sum_val)**0.5
        #___________________________________
        # Non-diagonal elements
        for j in range(i+1, n):
            sum_val = 0.0
            for k in range(i):
                sum_val+=L[i][k]*L[j][k]            
            L[j][i]=(matrix[j][i]-sum_val)/L[i][i]
    
    print("Cholesky decomposition successful")
    return L

#___________________________________________________________
def solve_by_cholesky(A,b):
    '''Solve linear system A*x=b via Cholesky!!'''
    # Get Cholesky factor L where A = L*L^T
    L=cholesky_decomposition(A)
    if L is None:
        return None
    # Solve L*y=b for y
    y=forward_substitution(L, b)
    # Solve L^T*x=y for x
    x=backward_substitution_transpose(L, y)
    return x

#____________________________________________________________
def jacobi_iteration(A,b,tol=1e-10,max_iter=1000):
    '''
    tol: tolerance for convergence
    max_iter: maximum number of iterations
    '''
    n=len(A)
    x=[0.0]*n  # Initial_guess x^(k)
    x_new=[0.0]*n # x^(k+1)
    iterations=0
    error=float('inf')
    #__________________________________
    print("Jacobi Iteration Method")
    print("Iter\t",end="")
    for i in range(n):
        print(f"x[{i}]\t\t",end="")
    print("Error")
    #__________________________________
    print(f"{iterations}\t", end="")
    for i in range(n):
        print(f"{x[i]:.6f}\t", end="")
    print("---")
    #__________________________________    
    while error>tol and iterations<max_iter:
        iterations+=1        
        #__main__iteration__loop!
        for i in range(n):
            sum=0.0
            for j in range(n):
                if i!=j:
                    sum+=A[i][j]*x[j]
            x_new[i]=(b[i]-sum)/A[i][i]
        #_______________________________
        error = max(abs(x_new[i]-x[i]) for i in range(n))
        #_______________________________
        print(f"{iterations}\t", end="")
        for i in range(n):
            x[i] = x_new[i]     # updating x
            print(f"{x[i]:.6f}\t", end="")
        print(f"{error:.8f}")
        #checking_if_converged
        if error<=tol:
            print(f"\nConverged after {iterations} iterations.")
            break
    #_______________________________
    if iterations>=max_iter:
        print(f"\nFailed to converge after {max_iter} iterations.")
    #_______________________________
    return x, iterations

#____________________________________________________________________