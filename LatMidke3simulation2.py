# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:59:15 2019

@author: MSI
"""

import numpy as np
import matplotlib.pyplot as plt

print("Number 1 :")
x = np.array([5.5,1.1,6.5,4.9,6.4,
              7.0,1.5,5.7,5.9,5.4,
              6.1,1.2,7.3,6.1,4.4])
y = np.array([1,1,1,1,1,
              1,1,1,1,1,
              1,1,1,1,1])

mean = np.mean(x)
print("Mean :", mean)
std = np.std(x)
print("Std : ", std)

Q1 = np.percentile(x,25)
Q2 = np.percentile(x,50)
Q3 = np.percentile(x,75)
print("Q1 :",Q1)
print("Q2 :",Q2)
print("Q3 :",Q3)

print("1A")
plt.scatter(y,x)
plt.show()

upper = mean + std
lower = mean - std
plt.errorbar(1,mean,yerr = std,uplims = upper, lolims = lower)

plt.show()

#plt.boxplot(x, showfliers = False, positions=[1.5])
plt.boxplot(x, showfliers = False)
#plt.xlim(-2, 4)
plt.show()

print("Number 2")
A = np.array([[6,2,0,0,0],
              [-1,7,2,0,0],
              [0,-2,8,2,0],
              [0,0,3,7,-2],
              [0,0,0,3,5]])
B = np.array([[2],[-3],[4],[-3],[1]])
print("2A")
print("Transpose A :", np.transpose(A))
print("Transpose B :", np.transpose(B))

c = np.linalg.solve(A,B)
print("Norm : ", np.linalg.norm(c))

print("Number 3:")
def taylor_1(x,n):
    s = 0
    for i in range (0,n):
        s+=  x**i
    return s
print("")
print("First 5")
print(taylor_1(0.1,5))
print("first 10")
print(taylor_1(0.1,10))
print("Relative errors : ")
print(abs(((1/(1-0.1))-(taylor_1(0.1,5)))/(1/1-0.1))*100)
d = 1/(1-0.1)
e = taylor_1(0.1,5)
print(abs((d-e)/d)*100)

print("Number 4")
lower = (2-1)/2
upper = (2+1)/2
c = 1
def f(x):
    return (2*x + 3/x)**2
def Gauss(x1,x2):
    return (lower *c * f(lower * x1 + upper)) + (lower * c * f(lower * x2 + upper))
        
print(Gauss(-0.577350, 0.577350))
print("Relative error :" )
print(abs(25.83-(Gauss(-0.577350, 0.577350)))/(25.83)*100, "%")



