import numpy as np

#m = float(input("Expectation value or mean: "))
#v = float(input("Variance: "))

#n = int(input("Basis Number: "))
#a = float(input("var: "))
#w = input("請輸入一串數字，用空格分隔：") 
#W = [float(x) for x in w.split()]
#print(W)

def univariate_gaussian_generator(mu, sigma2):
    sigma = np.sqrt(sigma2)
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    # Box-Muller transform
    z = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)
    x = mu + sigma * z
    return x

def linear_model_generator(n, W, a):
    e = np.random.normal(0, 1 / np.sqrt(a))
    #print(e)
    x = np.random.uniform(-1.0, 1.0)
    #print(x)
    y = 0
    for i in range(n):
        y += W[i] * (x ** i)
    y += e
    return x, y
    
#print(univariate_gaussian_generator(m, v))
#print(linear_model_generator(n, W, a))
