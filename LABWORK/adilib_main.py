import matplotlib.pyplot as plt
import numpy as np

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
def aug(A, b):
    m,n = A.shape
    augmented = np.zeros((m,n + 1)) 

    for i in range(m):
        for j in range(n):
            augmented[i][j] = A[i][j]    
        augmented[i][n] = b[i]          

    return augmented
#==========================================================================================
def gauss_jordan(A, b):
    
    n = len(b)
    # Create augmented matrix [A|b]
    augmented = aug(A,b) 
    
    for i in range(n):
        
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        #Swap rows if
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
        
        #Make diag element 1
        if augmented[i][i] != 0:
            augmented[i] = augmented[i] / augmented[i][i]
        
        #Make other elements 0
        for j in range(n):
            if i != j and augmented[j][i] != 0:
                augmented[j] = augmented[j] - augmented[j][i] * augmented[i]
    
    solution = augmented[:, -1]
    return solution


#==========================================================================================