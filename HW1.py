#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :HW1.py
# @Time      :2023/3/9 5:09 PM
# @Author    :Kinddle
from FuzzyMF import *
import numpy as np
from matplotlib import pyplot as plt

"""
1.Plot the following basic type-1 membership functions (MFs) while x  [0, 20] 
1)Gaussian MF with m = 6, sigma = 5
2)Triangular MF with a = 3, b = 8, c = 15
3)Sigmoidal MF with a = -2, c = 6
4)Trapzoidal MF with a = 4, b = 7, c = 12, d = 18 
5)Generalized bell MF with a =2, b = 4, c = 12. Change a = 6, keep b = 4, c = 12, plot another gbell MF.
"""
x = np.linspace(0, 20, 400)
y1 = gaussmf(x, 5, 6)
y2 = trimf(x, 3, 8, 15)
y3 = sigmf(x, -2, 6)
y4 = trapmf(x, 4, 7, 12, 18)
y5_1 = gbellmf(x, 2, 4, 12)
y5_2 = gbellmf(x, 6, 4, 12)

fig = plt.figure(figsize=[6, 10])
plt.subplot(3, 2, 1)
plt.plot(x, y1, label="gaussian MF")
plt.legend(loc="upper right")
plt.subplot(3, 2, 2)
plt.plot(x, y2, label="Triangular MF")
plt.legend(loc="upper right")
plt.subplot(3, 2, 3)
plt.plot(x, y3, label="Sigmoidal MF")
plt.legend(loc="upper right")
plt.subplot(3, 2, 4)
plt.plot(x, y4, label="Trapzoidal MF")
plt.legend(loc="upper right")
plt.subplot(3, 1, 3)
plt.plot(x, y5_1, label="bell MF(a=2)")
plt.plot(x, y5_2, label="bell MF(a=6)")
plt.legend(loc="upper left")
plt.show()

"""
2. Suppose the weather at Chengdu is described by its Humidity. 
Create five linguistic terms to describe the scale of temperature and sketch MF for each of them.
very dry; slightly dry; comfortable; slightly humid; very humid; 
0-20； 20-40；40-50；50-70；70-100
"""
x = np.linspace(0, 100, 400)
y1 = trapmf(x, -1, 0, 15, 25)
y2 = trapmf(x, 15, 25, 35, 40)
y3 = trapmf(x, 35, 40, 50, 55)
y4 = trapmf(x, 50, 55, 65, 75)
y5 = trapmf(x, 65, 75, 100, 100)

fig2 = plt.figure(figsize=[6, 6])
# plt.subplot(2,2,1)
plt.plot(x, y1, label="very dry")
# plt.legend(loc="upper right")
# plt.subplot(2,2,2)
plt.plot(x, y2, label="slightly dry")
# plt.legend(loc="upper right")
# plt.subplot(2,2,3)
plt.plot(x, y3, label="comfortable")
# plt.legend(loc="upper right")
# plt.subplot(2,2,4)
plt.plot(x, y4, label="slightly humid")
# plt.legend(loc="upper right")
plt.plot(x, y5, label="very humid")
plt.xlabel("humidity(%)")
plt.legend()
plt.show()

"""
Consider the fuzzy relation “u is close to v” on U x V.
The membership function for this relation is illustrated in Fig. 1.
 U = {u1,u2 } = {5, 10}, V = {v1,v2 ,v3 } = {1, 8, 16}.
"""
MF_UClose2V = lambda diff: trimf(abs(diff), -1, 0, 8)
# x = np.linspace(0,10,40)
# plt.plot(x, MF_uclose2v(x),label="u-v")
# plt.legend()
# plt.show()
Us = np.array([5, 10])
Vs = np.array([1, 8, 16])
Mx_cuv = MF_UClose2V(Us[:, None] - Vs)
Mx_luv = np.array([[.4, 0, 0],
                   [.9, .2, 0]])

union = np.max([Mx_cuv, Mx_luv], axis=0)
intersection = np.min([Mx_cuv, Mx_luv], axis=0)

Ws = [3, 12]
Mx_lvw = np.array([[0, 0], [0.5, 0], [1, 0.4]])

relation_cl = maxmin_composition(Mx_cuv, Mx_lvw)

"""
Using truth tables show that the following are tautologies [Allendoerfer and Oakley (1955)]:
"""

"""
6. Consider fuzzy sets A and B, where x  [0,10], and

"""
x = np.linspace(0, 10, 200)
MF_A = lambda X: np.exp(-(X - 3) ** 2 / 2)
MF_B = lambda X: np.exp(-(X - 4) ** 2 / 2)


y1 = MF_A(x)
y2 = MF_B(x)
y3 = np.max([y1,y2],axis=0)
y4 = np.min([[1]*200,y1+y2],axis=0)
y5 = np.min([y1,y2],axis=0)
y6 = y1*y2

fig3 = plt.figure(figsize=[6, 10])
plt.subplot(3, 2, 1)
plt.plot(x, y1, label="MF_A")
plt.plot(x, y2, label="MF_B")
plt.legend(loc="center right")
plt.ylim(-0.05,1.05)
plt.subplot(3, 2, 2)
plt.plot(x, y1, label="MF_A")
plt.plot(x, y2, label="MF_B")
plt.legend(loc="center right")
plt.ylim(-0.005,0.05)
plt.subplot(3, 2, 3)
plt.plot(x, y3, label="maximum")
plt.legend(loc="center right")
plt.ylim(-0.05,1.05)
plt.subplot(3, 2, 4)
plt.plot(x, y4, label="bounded sum")
plt.legend(loc="center right")
plt.ylim(-0.05,1.05)
plt.subplot(3, 2, 5)
plt.plot(x, y5, label="minimum")
plt.legend(loc="center right")
plt.ylim(-0.05,1.05)
plt.subplot(3, 2, 6)
plt.plot(x, y6, label="algebraic product")
plt.legend(loc="center right")
plt.ylim(-0.05,1.05)

plt.show()


