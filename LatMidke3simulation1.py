# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:12:49 2019

@author: MSI
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as spi

print("Number 1 :")

x = np.array([5.5,1.1,6.5,4.9,6.4,
              7.0,1.5,5.7,5.9,5.4,
              6.1,1.2,7.3,6.1,4.4])
print("A")
mean = np.mean(x)
print("Mean :", mean)
std = np.std(x)
print ("Std :", std)

print("B")
Q1 = np.percentile(x,25)
Q2 = np.percentile(x,50)
Q3 = np.percentile(x,75)
print("Q1 : ",Q1)
print("Q2 : ",Q2)
print("Q3 : ",Q3)

print("Number 2 :")
A = np.array([[6,2,0,0,0],
              [-1,7,2,0,0],
              [0,-2,8,2,0],
              [0,0,3,7,-2],
              [0,0,0,3,5]])
B = np.array([[2],[-3],[4],[-3],[1]])
print("2a")
c = np.linalg.solve(A,B)
print("Vector X :", c)
print("2b")
d = np.linalg.inv(A)
print("Inverse A :" , d)
print("2c")
e = np.linalg.det(A)
print("Determinant : ",e)

print("Number 3 : ")
def ln1_x(x,n):
    s = 0;
    for i in range(0,n):
        s += -x**(i+1)/(i+1)
    return s

print("3B")
print("First 8")
print(ln1_x(0.5,8))

print("First 16")
print(ln1_x(0.5,16))

print("3C")
print("Math log :", math.log(0.5))

print("3D")
print("Relative errors: ")
print("Using 8")
print(-abs(ln1_x(0.5,8) - (math.log(0.5))) / (math.log(0.5))*100, "%")
print(-abs(math.log(0.5) - (ln1_x(0.5,8))) / (math.log(0.5))*100, "%")
print("Using 16")
print(-abs(ln1_x(0.5,16) - (math.log(0.5))) / (math.log(0.5))*100, "%")


print("Number 4")
def f(x):
    return (2*x**4) + (6*x**3) - (2*x)
def trapezoidal(start,end,n):
    h = (end-start)/n
    x = start + h
    total = 0
    for i in range (1,n):
        total += f(x)
        x += h
    return (end - start) * ((f(start) + (2*total) + f(end)) / (2*n))
print("Trapezoidal : ")
print(trapezoidal(0,2,4))
print ("Relative errors : ")
print(abs(32.8 - (trapezoidal(0,2,4)))/ (32.8)* 100, "%")


print("BISA JUGA DENGAN RUMUS INI :")
def trapezoidal1(start,end,n):
    h = (end-start)/n
    x = start + h
    total = 0
    for i in range(1,n):
        total += f(x)
        x += h
    return h/2 * (f(start) + 2*total + f(end))
print("Trapezoidal cara lain : ")
print(trapezoidal1(0,2,4))

print("MENGGUNAKAN SCIPY :")
n = 4
x = np.linspace(0,2,n+1)
y = (2*x**4) + (6*x**3) - (2*x)
traps = spi.trapz(y,x)
print (traps)
print("Relative errors :")
print(abs(32.8 - (traps))/(32.8)*100, "%")

print("MENGGUNAKAN Numpy :")
n = 4
x = np.linspace(0,2,n+1)
y = (2*x**4) + (6*x**3) - (2*x)
traps = np.trapz(y,x)
print (traps)
print("Relative errors :")
print(abs(32.8 - (traps))/(32.8)*100, "%")

print("Helloo world") 
def f(x,y):
    return 2 - np.exp(-4*x) - 2*y
def euler(x):
    n = len(x)
    y = np.zeros(n)
    y[0] = 1
    h = x[1] - x[0]
    for i in range(1,n):
        y[i] = y[i-1] + h*f(x[i-1],y[i-1])
    return y

x = np.linspace(0,0.5,6)
print(x)
y = euler(x)
print(y)
plt.plot(x,y)
plt.show()

    




