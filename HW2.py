#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :HW2.py
# @Time      :2023/3/26 4:52 PM
# @Author    :Kinddle
import numpy as np
from FuzzyFuzzifier import *
from FuzzyMF import *
from FuzzyDefuzzifier import center_of_sets
from matplotlib import pyplot as plt

def min_t_norm(*args):
    return np.min(args)


def prod_t_norm(*args):
    return np.prod(args)


Antecedent_MF_Far = lambda x: sigmf(x, 1, 5)
Antecedent_MF_High = lambda x: sigmf(x, 1, 5)
Antecedent_MF_Near = lambda x: sigmf(x, -1, 5)
Antecedent_MF_Low = lambda x: sigmf(x, -1, 5)

Consequent_MF_Low = lambda x: trapmf(x, 0, 0, 0.2, 0.5)
Consequent_MF_Moderate = lambda x: trimf(x, 0, 0.5, 1)
Consequent_MF_High = lambda x: trapmf(x, 0.5, 0.8, 1, 1)

# Mu_R1 = lambda x, y: min_t_norm(min_t_norm(Antecedent_MF_Near(x[0]), Antecedent_MF_Low(x[1])),
#                                 Consequent_MF_Low(y))
# Mu_R2 = lambda x, y: min_t_norm(min_t_norm(Antecedent_MF_Near(x[0]), Antecedent_MF_High(x[1])),
#                                 Consequent_MF_Moderate(y))
# Mu_R3 = lambda x, y: min_t_norm([min_t_norm([Antecedent_MF_Far(x[0]), Antecedent_MF_Low(x[1])]),
#                                  Consequent_MF_Moderate(y)])
# Mu_R4 = lambda x, y: min_t_norm([min_t_norm([Antecedent_MF_Far(x[0]), Antecedent_MF_High(x[1])]),
#                                  Consequent_MF_High(y)])

# 1.1 firing level
input_x = [4, 8]


def FiringLevel(inputs, fuzzifier, rules):
    Mu_input = fuzzifier(inputs)
    tmp_list = []
    for l in range(len(rules)):
        tmp = min_t_norm(Mu_input[l], rules[l](inputs[l]))
        tmp_list.append(tmp)
    return min_t_norm(*tmp_list)


FiringLevel_R1 = FiringLevel(input_x, lambda x: SingletonFf(x, input_x), [Antecedent_MF_Near, Antecedent_MF_Low])
FiringLevel_R2 = FiringLevel(input_x, lambda x: SingletonFf(x, input_x), [Antecedent_MF_Near, Antecedent_MF_High])
FiringLevel_R3 = FiringLevel(input_x, lambda x: SingletonFf(x, input_x), [Antecedent_MF_Far, Antecedent_MF_Low])
FiringLevel_R4 = FiringLevel(input_x, lambda x: SingletonFf(x, input_x), [Antecedent_MF_Far, Antecedent_MF_High])

# print(FiringLevel_R1, FiringLevel_R2, FiringLevel_R3, FiringLevel_R4)
print("Firing level:")
print("Rule 1:", FiringLevel_R1)
print("Rule 2:", FiringLevel_R2)
print("Rule 3:", FiringLevel_R3)
print("Rule 4:", FiringLevel_R4)

# 1.2 height defuzzifier

height_Consequent_MF_Low = lambda x: SingletonFf(x, 0.1)
height_Consequent_MF_Moderate = lambda x: SingletonFf(x, 0.5)
height_Consequent_MF_High = lambda x: SingletonFf(x, 0.9)

MuB_G_R1 = prod_t_norm(FiringLevel_R1, height_Consequent_MF_Low(0.1))
MuB_G_R2 = prod_t_norm(FiringLevel_R2, height_Consequent_MF_Moderate(0.5))
MuB_G_R3 = prod_t_norm(FiringLevel_R3, height_Consequent_MF_Moderate(0.5))
MuB_G_R4 = prod_t_norm(FiringLevel_R4, height_Consequent_MF_High(0.9))

yh = (0.1 * MuB_G_R1 + 0.5 * MuB_G_R2 + 0.5 * MuB_G_R3 + 0.9 * MuB_G_R4) / (MuB_G_R1 + MuB_G_R2 + MuB_G_R3 + MuB_G_R4)
yh_old = (0.1 * MuB_G_R1 + 0.5 * MuB_G_R2 + 0.9 * MuB_G_R4) / (MuB_G_R1 + MuB_G_R2 + MuB_G_R4)

print(f"2) yh = {yh}")

# 1.3 center

Center_Low = center_of_sets(Consequent_MF_Low, xLim=(0, 0.5), yLim=(0, 1))
Center_Moderate = center_of_sets(Consequent_MF_Moderate, xLim=(0, 1), yLim=(0, 1))
Center_High = center_of_sets(Consequent_MF_High, xLim=(0.5, 1), yLim=(0, 1))

yh2_old = (Center_Low[1] * MuB_G_R1 + Center_Moderate[1] * MuB_G_R2 + Center_High[1] * MuB_G_R4) \
      / (MuB_G_R1 + MuB_G_R2 + MuB_G_R4)
ycos = (Center_Low[0] * MuB_G_R1 + Center_Moderate[0] * MuB_G_R2
        + Center_Moderate[0] * MuB_G_R3 + Center_High[0] * MuB_G_R4)/(MuB_G_R1 + MuB_G_R2 + MuB_G_R3 + MuB_G_R4)
print(f"y_cos = {ycos}")

# 2

def The_Guss_func(X,mf):
    return np.exp(-1/2*((X-mf)/10)**2)

def Phi_func(X,Fl:list):
    Phi_basic = []
    for mf in Fl:
        Phi_basic.append(The_Guss_func(X, mf))
    # np.array(Phi_basic)
    Phi = np.r_[Phi_basic]
    return Phi/np.sum(Phi, axis=0)

sample = np.linspace(0, 100, 10000)
Fl_1 = [20, 35, 50, 65, 80]
Fl_2 = [20, 25, 50, 62, 80]

fig = plt.figure(figsize=[12,5])
plt.subplot(1,2,1)
plt.plot(sample,Phi_func(sample,Fl_1).T)
plt.xlim([0,100])
plt.ylim([0,1])
plt.subplot(1,2,2)
plt.plot(sample,Phi_func(sample,Fl_2).T)
plt.xlim([0,100])
plt.ylim([0,1])
plt.show()
# Phi_basic_1 = []
# for mf in Fl_1:
#     Phi_basic_1.append(lambda x:The_Guss_func(x,mf))
# sum_Phi_basic_1 =lambda x: np.sum([func(x) for func in Phi_basic_1])
# Phi_1 = [lambda x:func(x)/sum_Phi_basic_1(x) for func in Phi_basic_1]
#
# Phi_basic_2 = []
# for mf2 in Fl_2:
#     Phi_basic_2.append(lambda x: The_Guss_func(x, mf2))
# sum_Phi_basic_2 = lambda x: np.sum([func(x) for func in Phi_basic_2])
# Phi_2 = [lambda x: func(x) / sum_Phi_basic_2(x) for func in Phi_basic_2]
