import numpy as np
import data_generator as dg

m = float(input("Expectation value or mean: "))
v = float(input("Variance: "))

print(f"Data point source function: N({m}, {v})")
mean = 0.0
variance = 0.0
epsilon = 1e-5
n = 0
while True:
    data_point = dg.univariate_gaussian_generator(m, v)
    print(f"Add data point: {data_point}")
    mean_past = mean
    var_past = variance
    mean = (mean*n + data_point)/(n+1)
    # Update variance and mean using Welford's method
    variance = (variance**2) + (data_point - mean)*(data_point - mean_past)/(n+1)
    variance = np.sqrt(variance)
    print(f"Mean: {mean}, Variance: {variance}")
    if abs(mean - mean_past) < epsilon and abs(variance - var_past) < epsilon:
        break
    n += 1