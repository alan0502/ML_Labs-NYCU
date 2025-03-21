import numpy as np
import math

def read_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(line.strip())
    return data

def combination(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
file_path = "inputfile.txt"
b_outcomes = read_data(file_path)
print(type(b_outcomes[0]))
print("Input initial beta prior: ")
a = int(input("a = "))
b = int(input("b = "))
N = len(b_outcomes)
for i in range(N):
    print(f"case {i+1}: ")
    #print(b_outcomes[i])
    count_0 = 0
    count_1 = 0
    #print(len(b_outcomes[i]))
    for j in range(len(b_outcomes[i])):
        if b_outcomes[i][j] == "0":
            count_0 += 1
        else:
            count_1 += 1
    prob_0 = count_0/len(b_outcomes[i])
    prob_1 = count_1/len(b_outcomes[i])
    #print(count_0)
    #print(count_1)
    #print(prob_0)
    #print(prob_1)
    likelihood = (count_1 / (count_0 + count_1))**count_1 * (count_0 / (count_0 + count_1))**count_0
    likelihood = likelihood * combination(count_0 + count_1, count_1)
    print(f"likelihood: {likelihood}")
    print(f"Beta prior: a = {a}, b = {b}")
    a = a + count_1
    b = b + count_0
    print(f"Beta posterior: a = {a}, b = {b}")
