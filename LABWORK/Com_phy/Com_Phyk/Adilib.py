def index_f(n):
    return [(i+1)/n for i in range(n)]

def pseudo1(n,c,s=0.1):
    x_i = s
    l=[]
    for _ in range(n):
        x_i = c*x_i*(1-x_i)
        l.append(x_i)
    return l

def LCG(N,a=1103515245,c=12345,m=32768,x_0=.1):
    l=[]
    x_i=x_0
    for i in range(N):
        x_i= ((a*x_i +c)%m)
        l.append(x_i/m)
    return l

def corre_LCG(N,k):
    result=[]
    a=LCG(N)[:-k]
    b=LCG(N)[k:]
    result.append(a)
    result.append(b)
    return result
    

def corre_pseudo1(n,c,k=5):
    result=[]
    a=pseudo1(n,c)[:-k]
    b=pseudo1(n,c)[k:]
    result.append(a)
    result.append(b)
    return result

def Plot(x, y , title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label',file_name='sample_plot.png'):
    plt.plot(x, y, marker='o', linestyle='none', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  
    plt.savefig(file_name)
    plt.show()
