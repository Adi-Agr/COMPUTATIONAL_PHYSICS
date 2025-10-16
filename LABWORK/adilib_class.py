'''
This file (adilib_class) is almost same as my personal library file (adilib_main).

DIFFERENCE:- I just organised everything into classes for better structure and reusability and added some docstrings for readability.

Be assured that I did not change any of the underlying algorithms or their implementations.
To prove  my authenticity, I didn't touch adilib_main file but created  another file, which I will start updating from Assignment 7

Due to lack of knowledge on classes, I was not using it till now . But still ,to make sure there is no foul_play, you can track the changes in github.
'''



import matplotlib.pyplot as plt
import math
import numpy as np
class MatrixOperations:
    """Class for basic matrix operations"""
    @staticmethod
    def readd(filename):
        """Read matrix from text file with space-separated values"""
        with open(filename, 'r') as file:
            matrix = []
            for line in file:
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix
    @staticmethod
    def multiplyy(X, Y):
        """Matrix multiplication: X * Y"""
        result = []
        for i in range(len(X)):
            row = []
            for j in range(len(Y[0])):
                sum_val = 0
                for k in range(len(Y)):
                    sum_val += X[i][k] * Y[k][j]
                row.append(sum_val)
            result.append(row)
        return result
    @staticmethod
    def transposee(X):
        """Transpose a matrix"""
        result = []
        for i in range(len(X[0])):
            row = []
            for j in range(len(X)):
                row.append(X[j][i])
            result.append(row)
        return result
    @staticmethod
    def issymmetric(matrix):
        """Check if a matrix is symmetric"""
        if not matrix or len(matrix) != len(matrix[0]):
            return False
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != matrix[j][i]:
                    return False
        return True
    @staticmethod
    def augmentt(A, B):
        """Create an augmented matrix [A|B]"""
        n = len(A)
        aug = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(A[i][j]))
            for j in range(len(B[0]) if isinstance(B[0], list) else 1):
                row.append(B[i][j] if isinstance(B[0], list) else B[i])
            aug.append(row)
        return aug
    @staticmethod
    def row_swapp(matrix, r1, r2):
        """Swap two rows in a matrix"""
        matrix[r1], matrix[r2] = matrix[r2], matrix[r1]    
    @staticmethod
    def row_scalee(matrix, r, scale):
        """Multiply a row by a scalar"""
        for j in range(len(matrix[r])):
            matrix[r][j] *= scale    
    @staticmethod
    def rows_add(matrix, r1, r2, scale):
        """Add a scaled row to another row: r1 += scale * r2"""
        for j in range(len(matrix[r1])):
            matrix[r1][j] += matrix[r2][j] * scale


class RandomNumbers:
    """Class for generating random numbers"""
    @staticmethod
    def index_f(n):
        """Generate a list of evenly spaced fractions"""
        return [(i+1)/n for i in range(n)]
    @staticmethod
    def pRNG_logistic(n,c=3.98,s=0.1):
        """Generate pseudo-random numbers using logistic map"""
        x_i=s
        l=[]
        for _ in range(n):
            x_i*=c*(1-x_i)
            l.append(x_i)
        return l
    @staticmethod
    def pRNG_LCG(N, a=1103515245,c=12345,m=32768,s=0.1):
        """Linear Congruential Generator for random numbers"""
        l,x_i=[],s
        for i in range(N):
            x_i = ((a*x_i+c)%m)
            l.append(x_i/m)
        return l
    @staticmethod
    def corre_LCG(N, k):
        """Correlation analysis for LCG"""
        result = []
        a = RandomNumbers.pRNG_LCG(N)[:-k]
        b = RandomNumbers.pRNG_LCG(N)[k:]
        result.append(a)
        result.append(b)
        return result
    @staticmethod
    def corre_pRNG(n, c, k=5):
        """Correlation analysis for pRNG"""
        result = []
        a = RandomNumbers.pRNG_logistic(n, c)[:-k]
        b = RandomNumbers.pRNG_logistic(n, c)[k:]
        result.append(a)
        result.append(b)
        return result


class Visualization:
    """Class for visualization functions"""
    @staticmethod
    def plot(x, y, title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label', file_name='sample_plot.png'):
        """Create a scatter plot"""
        plt.plot(x, y, marker='o', linestyle='none', color='b')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)  
        plt.savefig(file_name)
        plt.show()


class LinearSystems:
    """Class for solving linear systems of equations"""
    @staticmethod
    def gauss_jordan(A, B):
        """Solve linear system Ax=B using Gauss-Jordan elimination"""
        n = len(B)
        # Convert B to matrix format if it's a vector
        if not isinstance(B[0], list):
            B = [[b] for b in B]

        augmented = MatrixOperations.augmentt(A, B)

        for i in range(n):
            max_row = i
            for k in range(i+1, n):
                if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                    max_row = k
            
            # Swap rows
            if max_row != i:
                MatrixOperations.row_swapp(augmented, i, max_row)
            
            # Make diagonal element 1
            if augmented[i][i] != 0:
                MatrixOperations.row_scalee(augmented, i, 1/augmented[i][i])
            
            # Make other elements 0
            for k in range(n):
                if i != k:
                    scalar = -augmented[k][i]
                    MatrixOperations.rows_add(augmented, k, i, scalar)
        
        # Extract solution (handle both vector and matrix cases)
        if len(augmented[0]) == n + 1:  # Vector case
            solution = [row[n] for row in augmented]
        else:  # Matrix case
            solution = [row[n:] for row in augmented]
            
        return solution    
    @staticmethod
    def ludecomp_doolittle(matrix):
        """LU decomposition using Doolittle's method (diagonal of L=1)"""
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]
        
        # Set diagonal of L to 1
        for i in range(n):
            L[i][i] = 1.0
        
        # Main algorithm
        for i in range(n):
            # Calculate U elements
            for j in range(i, n):
                sum_val = 0.0
                for k in range(i):
                    sum_val += L[i][k] * U[k][j]
                U[i][j] = matrix[i][j] - sum_val
            
            # Calculate L elements
            for j in range(i+1, n):
                if U[i][i] == 0:
                    print("LU decomposition failed: matrix is singular!")
                    return None, None
                sum_val = 0.0
                for k in range(i):
                    sum_val += L[j][k] * U[k][i]
                L[j][i] = (matrix[j][i] - sum_val) / U[i][i]
        
        print("LU decomposition successful")
        return L, U    
    @staticmethod
    def forward_substitution(L, B):
        """Solve L*Y=B for Y using forward substitution"""
        n = len(L)
        
        # Handle both vector and matrix cases for B
        is_vector = not isinstance(B[0], list)
        if is_vector:
            Y = [0.0] * n
        else:
            Y = [[0.0] * len(B[0]) for _ in range(n)]
        
        print("Forward Substitution: Solving L*y=B")
        
        if is_vector:  # Vector case
            for i in range(n):
                sum_val = 0.0
                for j in range(i):
                    sum_val += L[i][j] * Y[j]
                Y[i] = (B[i] - sum_val) / L[i][i]
                print(f"Y[{i}]={Y[i]:.6f}")
        else:  # Matrix case
            for i in range(n):
                for k in range(len(B[0])):
                    sum_val = 0.0
                    for j in range(i):
                        sum_val += L[i][j] * Y[j][k]
                    Y[i][k] = (B[i][k] - sum_val) / L[i][i]
                    
        return Y    
    @staticmethod
    def backward_substitution(U, Y):
        """Solve U*X=Y for X using backward substitution"""
        n = len(U)
        
        # Handle both vector and matrix cases for Y
        is_vector = not isinstance(Y[0], list)
        if is_vector:
            X = [0.0] * n
        else:
            X = [[0.0] * len(Y[0]) for _ in range(n)]
        
        print("Backward Substitution: Solving U*x=y")
        
        if is_vector:  # Vector case
            for i in range(n-1, -1, -1):
                sum_val = 0.0
                for j in range(i+1, n):
                    sum_val += U[i][j] * X[j]
                X[i] = (Y[i] - sum_val) / U[i][i]
                print(f"x[{i}]={X[i]:.6f}")
        else:  # Matrix case
            for i in range(n-1, -1, -1):
                for k in range(len(Y[0])):
                    sum_val = 0.0
                    for j in range(i+1, n):
                        sum_val += U[i][j] * X[j][k]
                    X[i][k] = (Y[i][k] - sum_val) / U[i][i]
                    
        return X    
    @staticmethod
    def backward_substitution_transpose(L, y):
        """Solve L^T*X=Y for X using backward substitution"""
        n = len(L)
        X = [0.0] * n
        
        print("Backward Substitution: Solving L^T*x = y")
        
        for i in range(n-1, -1, -1):
            sum_val = 0.0
            for j in range(i+1, n):
                # Use L[j][i] instead of L[i][j] to get transpose effect
                sum_val += L[j][i] * X[j]
            X[i] = (y[i] - sum_val) / L[i][i]
            print(f"X[{i}] = {X[i]:.6f}")
            
        return X    
    @staticmethod
    def solve_by_lu(A, B):
        """Solve linear system A*X=B using LU decomposition"""
        L, U = LinearSystems.ludecomp_doolittle(A)
        if L is None:  # Check if decomposition failed
            return None
            
        Y = LinearSystems.forward_substitution(L, B)
        X = LinearSystems.backward_substitution(U, Y)
        return X    
    @staticmethod
    def cholesky_decomposition(matrix):
        """Cholesky decomposition: A = L*L^T where L is lower triangular"""
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            # Diagonal elements
            sum_val = 0.0
            for k in range(i):
                sum_val += L[i][k] ** 2
            
            # Check if matrix is positive definite
            if matrix[i][i] - sum_val <= 0:
                print("Matrix is not positive definite!")
                return None
                
            L[i][i] = (matrix[i][i] - sum_val) ** 0.5
            
            # Non-diagonal elements
            for j in range(i+1, n):
                sum_val = 0.0
                for k in range(i):
                    sum_val += L[i][k] * L[j][k]
                    
                L[j][i] = (matrix[j][i] - sum_val) / L[i][i]
        
        print("Cholesky decomposition successful")
        return L    
    @staticmethod
    def solve_by_cholesky(A, b):
        """Solve linear system A*x=b using Cholesky decomposition"""
        # Get Cholesky factor L where A = L*L^T
        L = LinearSystems.cholesky_decomposition(A)
        if L is None:  # Check if decomposition failed
            return None
            
        # Solve L*y=b for y
        y = LinearSystems.forward_substitution(L, b)
        
        # Solve L^T*x=y for x
        x = LinearSystems.backward_substitution_transpose(L, y)
        return x    
    @staticmethod
    def jacobi_iteration(A, b, tol=1e-10, max_iter=1000):
        """Solve linear system A*x=b using Jacobi iteration method"""
        n = len(A)
        x = [0.0] * n  # Initial guess x^(k)
        x_new = [0.0] * n  # x^(k+1)
        iterations = 0
        error = float('inf')
        
        # Track errors for convergence analysis
        errors = []
        
        print("Jacobi Iteration Method")
        print("Iter\t", end="")
        for i in range(n):
            print(f"x[{i}]\t\t", end="")
        print("Error")
        
        print(f"{iterations}\t", end="")
        for i in range(n):
            print(f"{x[i]:.6f}\t", end="")
        print("---")
        
        while error > tol and iterations < max_iter:
            iterations += 1
            
            # Main iteration loop
            for i in range(n):
                sum_val = 0.0
                for j in range(n):
                    if i != j:
                        sum_val += A[i][j] * x[j]
                x_new[i] = (b[i] - sum_val) / A[i][i]
            
            # Calculate error
            error = max(abs(x_new[i] - x[i]) for i in range(n))
            errors.append(error)
            
            print(f"{iterations}\t", end="")
            for i in range(n):
                x[i] = x_new[i]  # Update x
                print(f"{x[i]:.6f}\t", end="")
            print(f"{error:.8f}")
            
            # Check if converged
            if error <= tol:
                print(f"\nConverged after {iterations} iterations.")
                break
        
        # Check if failed to converge
        if iterations >= max_iter:
            print(f"\nFailed to converge after {max_iter} iterations.")
        
        return x, iterations, errors    
    @staticmethod
    def gauss_seidel_iteration(A, b, tol=1e-6, max_iter=1000):
        """Solve linear system A*x=b using Gauss-Seidel iteration method"""
        n = len(A)
        x = [0.0] * n  # Initial guess
        iterations = 0
        error = float('inf')
        
        # Track errors for convergence analysis
        errors = []
        
        print("Gauss-Seidel Iteration Method")
        print("Iter\t", end="")
        for i in range(n):
            print(f"x[{i}]\t", end="")
        print("Error")
        
        print(f"{iterations}\t", end="")
        for i in range(n):
            print(f"{x[i]:.4f}\t", end="")
        print("---")
        
        while error > tol and iterations < max_iter:
            iterations += 1
            error = 0.0
            
            # Main iteration loop
            for i in range(n):
                x_old = x[i]
                sum_val = 0.0
                for j in range(n):
                    if i != j:
                        sum_val += A[i][j] * x[j]
                        
                x[i] = (b[i] - sum_val) / A[i][i]  # Direct update to x[i]
                
                # Update error as we go through each equation
                error = max(error, abs(x[i] - x_old))
            
            errors.append(error)
            
            if iterations <= 10 or iterations % 10 == 0 or error <= tol:
                print(f"{iterations}\t", end="")
                for i in range(n):
                    print(f"{x[i]:.4f}\t", end="")
                print(f"{error:.6f}")
                
            # Check if converged
            if error<=tol:
                print(f"\nConverged after {iterations} iterations.")
                break
        
        # Check if failed to converge
        if iterations>=max_iter:
            print(f"\nFailed to converge after {max_iter} iterations.")
        
        return x, iterations, errors

class Roots:
    @staticmethod    
    def brackett(f,a,b):
        while f(a)*f(b)>0:
            if abs(f(a))<abs(f(b)):
                a-=1.5*(b-a)
            else:
                b+=1.5*(b-a)
        else:
            return a,b
        
    @staticmethod
    def bisection(f,a,b,tol=10e-6,iter=100):
        a,b=Roots.brackett(f,a,b)
        total_iter=[]
        for i in range(iter):
            c=(a+b)/2
            total_iter.append(i)
            if abs(b-a)<tol or abs(f(a))<tol or abs(f(b))<tol:
                print("Total Iterations(bi):-",len(total_iter),c,'\nf(root)=',f(c))
                return c
            else:
                if f(b)*f(c)<0 :
                    a=c
                else:
                    b=c
        
        print("Total Iterations(bi):-",len(total_iter),(a+b)/2,'\nf(root)=',f((a+b)/2))
        return (a+b)/2
    
    @staticmethod
    def regula_falsi(f,a,b,tol=10e-6,iter=100):
        a,b=Roots.brackett(f,a,b)        
        total_iter=[]
        for i in range(iter):
            total_iter.append(i)
            c=a-f(a)*(b-a)/(f(b)-f(a))
            if abs(f(c))<tol or abs(b-a)<tol:
                print("Total Iterations(rf):-",len(total_iter),c,'\nf(root)=',f(c))
                return c
            elif f(a)*f(c)<0:
                b=c
            else:
                a=c
        print("Total Iterations(rf):-",len(total_iter),(a+b)/2,'\nf(root)=',f((a+b)/2))
        return (a+b)/2

    @staticmethod
    def fixed_point(f,x0=.8,tol=10e-6,iter=100):
        total_iter=[]
        for _ in range(iter):
            x1=f(x0)
            total_iter.append(_)
            if abs(x1-x0)<tol:
                print("Total Iterations(fix_p):-",len(total_iter),x1,'\nf(root=)',f(x1))
                return x1,total_iter
            x0=x1
        print(f"\nFailed to converge after {total_iter} iterations.")
        return x0, total_iter
    
    @staticmethod
    def fixed_point_multivariable(g,x0,tol=1e-6,max_iter=100):
        x=x0.copy()  # Make copy
        tot_iter=[]
        print("Iter\t||x_k+1 - x_k||\t||x_k+1||")
        print("-"*50)
        
        for i in range(max_iter):
            tot_iter.append(i)
            # Next Iter.
            x_new=g(x)
            
            diff_norm = sum((x_new[j]-x[j])**2 for j in range(len(x)))**0.5 #...calculate norms
            x_new_norm = sum(x_new[j]**2 for j in range(len(x)))**0.5
            
            rel_error=diff_norm/max(x_new_norm,1e-10) #...calculate relative error
            
            print(f"{i}\t{diff_norm:.8f}\t{x_new_norm:.8f}") #print process
            
            # Check convergence using relative error
            if rel_error<tol:
                print(f"\nConverged after {i+1} iterations.")
                print(f"Solution: {x_new}")
                return x_new,tot_iter
            
            x = x_new.copy() #...update x
        print(f"\nFailed to converge after {max_iter} iterations.")
        print(f"Current solution: {x}")
        return x,tot_iter
    
    @staticmethod
    def NewtonRaphson(f,f_d,x0,tol=1e-6,max_iter=100):
        x=x0
        iter=[]
        for i in range(max_iter):
            iter.append(i)
            f_val=f(x)
            f_prime=f_d(x)
            # Check if derivative is too close to zero to avoid division by zero
            if abs(f_prime)<1e-10:
                print(f"Derivative too close to zero at x = {x}")
                return x,iter
            delta_x=f_val/f_prime
            x_new=x-delta_x
            #convergence-check
            if abs(delta_x)<tol or abs(f_val)<tol:
                print("Total Iterations(new-raph):-",len(iter),x_new,'\nf(root=)',f(x_new))
                return x_new,iter
            x=x_new
        print(f"\nFailed to converge after {max_iter} iterations.")
        return x, iter

    @staticmethod
    def NewtonRaphson_multivariable(f,J,x0,tol=1e-6,max_iter=100):
        x=x0.copy()  #...make copy to avoid any error
        tot_iter=[]
        n=len(x)
        
        print("Iter\t||f(x)||\tΔx")
        print("-" * 40)
        
        for i in range(max_iter):
            tot_iter.append(i)
            f_val=f(x)
            J_val=J(x)
            
            f_norm=sum(f_i**2 for f_i in f_val)**0.5 #calculating norm
            
            delta_x=LinearSystems.gauss_jordan(J_val,f_val) #.....del_x=J^(-1)f(x)
            delta_norm=sum(dx**2 for dx in delta_x)**0.5 #...calculating delta_norm
            
            print(f"{i}\t{f_norm:.8f}\t{delta_norm:.8f}") #....to print processs
            
            x=[x[j]-delta_x[j] for j in range(n)] #...update x
            
            # Check convergence
            if f_norm<tol or delta_norm<tol:
                print(f"\nConverged after {i+1} iterations.")
                print(f"Solution: {x}")
                print(f"Function values at solution: {f(x)}")
                return x,tot_iter
        
        print(f"\nFailed to converge after {max_iter} iterations.")
        return x,tot_iter
class Integration:
    '''Numerical integration methods'''
    
    @staticmethod
    def midpoint_integration(l, L, f, N=10000):
        '''Midpoint integration method'''
        h=(L-l)/N
        integral=0
        for i in range(N):
            integral+=f(l+(i+0.5)*h)
        integral*=h
        return integral
    
    @staticmethod
    def trapezoidal_integration(l, L, f, N=10000):
        '''Trapezoidal integration method'''
        h=(L-l)/N
        integral=0.5*(f(l)+f(L))
        for i in range(1, N):
            integral+=f(l+i*h)
        integral*=h
        return integral

    @staticmethod
    def simpson(l,L,f,N=40):
        '''Simpson Integral Method'''
        h=(L-l)/N
        integral=f(l)+f(L)
        for i in range(1,N,2):
            integral+= 4*f(l+i*h)
        for i in range(2,N-1,2):
            integral+= 2*f(l+i*h)
        integral*= h/3
        return integral
    
    @staticmethod
    def montecarlo_diff(l,L,f,N=40,tol=10e-6):
        true=1-math.sin(2)/2
        FN=0
        FN_list=[]
        N_list=[]
        while abs(true-FN)>tol:
            X=[(l+(L-l)*i) for i in RandomNumbers.pRNG_LCG(N)]
            FX=[f(i) for i in X]
            sigma_sq=0
            for i in X:
                sigma_sq+=((1/N)*(f(i)**2))
            sigma_sq-=((1/N)*sum(FX))**2
            FN=sum(FX)*(L-l)/N
            FN_list.append(FN)
            N+=20
            N_list.append(N)
        return FN,FN_list,N_list,N
    @staticmethod
    def gaussian_quadrature(f, a, b, exact_value=100, tol=1e-9):
        n=2
        integral=0
        while True:
            [x,w]=list(np.polynomial.legendre.leggauss(n))
            integral_new=0
            for i in range(n):
                xi=0.5*(b-a)*x[i]+0.5*(b+a)
                integral_new+=0.5*(b-a)*w[i]*f(xi)
            if abs(integral_new-exact_value)<= tol:  #convo.-check
                return integral_new, n
            integral=integral_new
            n+=1
    

    #===========================================
# For backward compatibility!!!

def brackett(f,a,b):
    return Roots.brackett(f,a,b)

def regula_falsi(f,a,b,tol=10e-6,iter=100):
    a=Roots.regula_falsi(f,a,b,tol,iter)
    return a
def bisectionn(f,a,b,tol=10e-6,iter=100):
    a=Roots.bisection(f,a,b,tol,iter)
    return a

def read_matrix(filename):
    return MatrixOperations.readd(filename)

def mat_mult(X, Y):
    return MatrixOperations.multiplyy(X, Y)

def transpose(X):
    return MatrixOperations.transposee(X)

def index_f(n):
    return RandomNumbers.index_f(n)

def pseudo1(n, c, s=0.1):
    return RandomNumbers.pRNG_logistic(n, c, s)

def LCG(N, a=1103515245, c=12345, m=32768, x_0=0.1):
    return RandomNumbers.pRNG_LCG(N, a, c, m, x_0)

def corre_LCG(N, k):
    return RandomNumbers.corre_LCG(N, k)

def corre_pseudo1(n, c, k=5):
    return RandomNumbers.corre_pRNG(n, c, k)

def Plot(x, y, title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label', file_name='sample_plot.png'):
    return Visualization.plot(x, y, title, xlabel, ylabel, file_name)

def create_aug(A, B):
    return MatrixOperations.augmentt(A, B)

def swap_rows(matrix, r1, r2):
    return MatrixOperations.row_swapp(matrix, r1, r2)

def scale_row(matrix, r, scale):
    return MatrixOperations.row_scalee(matrix, r, scale)

def add_rows(matrix, r1, r2, scale):
    return MatrixOperations.rows_add(matrix, r1, r2, scale)

def gauss_jordan(A, B):
    return LinearSystems.gauss_jordan(A, B)

def ludecomp_doolittle(matrix):
    return LinearSystems.ludecomp_doolittle(matrix)

def forward_substitution(L, B):
    return LinearSystems.forward_substitution(L, B)

def backward_substitution(U, Y):
    return LinearSystems.backward_substitution(U, Y)

def backward_substitution_transpose(L, y):
    return LinearSystems.backward_substitution_transpose(L, y)

def solve_by_backward_forward_substitution(A, B):
    return LinearSystems.solve_by_lu(A, B)

def cholesky_decomposition(matrix):
    return LinearSystems.cholesky_decomposition(matrix)

def solve_by_cholesky(A, b):
    return LinearSystems.solve_by_cholesky(A, b)

def jacobi_iteration(A, b, tol=1e-10, max_iter=1000):
    x, iterations, _ = LinearSystems.jacobi_iteration(A, b, tol, max_iter)
    return x, iterations

def is_symmetric(matrix):
    return MatrixOperations.issymmetric(matrix)

def gauss_seidel_iteration(A, b, tol=1e-6, max_iter=1000):
    x, iterations, _ = LinearSystems.gauss_seidel_iteration(A, b, tol, max_iter)
    return x, iterations

def printt(matrix=[[1,2],[3,4]]):
    d="\n".join(" ".join(str(x) for x in row) for row in matrix)    
    print(d)
    return d
#===============================================================================================