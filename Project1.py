#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Project1.py
# @Time      :2023/3/27 4:24 PM
# @Author    :Kinddle
import numpy as np
from FuzzyMF import *
from FuzzySystem import FuzzySystem
import pandas as pd


# Fuzzifier
def SingletonFf(x_c):
    def fun(X):
        X = np.array(X)
        rtn = np.zeros(shape=X.shape)
        rtn[X == x_c] = 1
        return rtn

    return fun


# Antecedent MF
def MF_Antecedent_near(X):
    return trapmf(X, 0, 0, 2, 5)


def MF_Antecedent_moderate(X):
    return trimf(X, 0, 5, 10)


def MF_Antecedent_far(X):
    return trapmf(X, 5, 8, 10, 10)


Antecedent_dict = {"near": MF_Antecedent_near,
                   "low": MF_Antecedent_near,
                   "moderate": MF_Antecedent_moderate,
                   "far": MF_Antecedent_far,
                   "high": MF_Antecedent_far}


# Consequent MF
def MF_Consequent_VeryWeak(Y):
    return trapmf(Y, 0, 0, 1, 3)


def MF_Consequent_Weak(Y):
    return trimf(Y, 1, 3, 5)


def MF_Consequent_Medium(Y):
    return trimf(Y, 3, 5, 7)


def MF_Consequent_Strong(Y):
    return trimf(Y, 5, 7, 9)


def MF_Consequent_VeryStrong(Y):
    return trapmf(Y, 7, 9, 10, 10)


Consequent_dict = {"VeryWeak": MF_Consequent_VeryWeak,
                   "Weak": MF_Consequent_Weak,
                   "Medium": MF_Consequent_Medium,
                   "Strong": MF_Consequent_Strong,
                   "VeryStrong": MF_Consequent_VeryStrong}


# DeFuzzifier
def Height_Defuzzifier(firing_levels, Consequent_dic, sample=(0.5, 3, 5, 7, 9.5)):
    weighted_value = 0
    value = 0
    for (_, C), Mu in firing_levels.items():
        y_ = sample[int(np.argmax(Consequent_dic[C](sample)))]
        weighted_value += y_ * Mu
        value += Mu
    return weighted_value / value

def center_of_sets(Func, xLim=(-1000, 1000), yLim=(0, 1), num=100000):
    sample_x = np.linspace(*xLim, num)
    sample_y = Func(sample_x)
    x_c = np.sum(sample_x * sample_y) / np.sum(sample_y)
    return x_c

def COS_Defuzzifier(firing_levels, Consequent_dic, sample=(0.5, 3, 5, 7, 9.5)):
    weighted_value = 0
    value = 0
    for (_, C), Mu in firing_levels.items():
        c_ = center_of_sets(Consequent_dic[C])
        # y_ = sample[int(np.argmax(Consequent_dic[C](sample)))]
        weighted_value += c_ * Mu
        value += Mu
    return weighted_value / value


if __name__ == '__main__':
    FSyS = FuzzySystem()
    FSyS.Fuzzifier = SingletonFf
    # FSyS.Defuzzifier = Height_Defuzzifier
    FSyS.Defuzzifier = COS_Defuzzifier
    FSyS.A_MF_List = Antecedent_dict
    FSyS.C_MF_List = Consequent_dict
    Data = pd.read_csv("expertData.csv", index_col=0)
    Rules = {(Data.iloc[idx, 0], Data.iloc[idx, 1], Data.iloc[idx, 2]): Data.iloc[idx, 3] for idx in range(len(Data))}
    FSyS.Rule_List = Rules
    rtn = FSyS.inference([1, 9, 3])

    print(rtn)

    from matplotlib import pyplot as plt

    step = 10
    x1 = [3, 7]
    x2 = np.linspace(0, 10, step)
    x3 = np.linspace(0, 10, step)
    X2, X3 = np.meshgrid(x2, x3)
    Z1 = np.empty(shape=[step, step])
    Z2 = np.empty(shape=[step, step])
    for i in range(step):
        print(f"\r {i}/{step}...")
        for j in range(step):
            Z1[i, j] = FSyS.inference([x1[0], x2[i], x3[j]])
            Z2[i, j] = FSyS.inference([x1[1], x2[i], x3[j]])

    plt.figure(figsize=[5, 5])
    # plt.subplot(2,1,1)
    ax = plt.axes(projection="3d")
    # ax.plot_wireframe(X2,X3,Z1,cmap='viridis')
    ax.plot_surface(X2, X3, Z1, cmap='viridis')
    ax.set_zlim([0, 10])
    ax.set_xlabel('x2')
    ax.set_ylabel('x3')
    ax.set_zlabel('y')
    # plt.subplot(2,1,2)
    plt.savefig(f"x1={x1[0]}.png")
    plt.show()

    plt.figure(figsize=[5, 5])
    ax2 = plt.axes(projection="3d")
    ax2.plot_surface(X2, X3, Z2, cmap='viridis')
    ax2.set_xlabel('x2')
    ax2.set_ylabel('x3')
    ax2.set_zlabel('y')
    ax2.set_zlim([0, 10])
    plt.savefig(f"x1={x1[1]}.png")
    plt.show()

    # Simulation
    Simulation_number = 100
    Sim_Data = np.random.random([Simulation_number, 4]) * [100, 100, 10, 10]  # x,y,battery,move
    x_, y_ = np.average(Sim_Data[:, [0, 1]], axis=0)
    Distance = np.linalg.norm(Sim_Data[:, [0, 1]] - [x_, y_], axis=1)
    Distance /= np.max(Distance) / 10
    Justice_Data = np.c_[Distance[:, None], Sim_Data[:, 2:]]
    Score = np.array([FSyS.inference(i) for i in Justice_Data])

    idx_Score = np.argmax(Score)
    idx_Battery = np.argmax(Sim_Data[:, 2])
    idx_close = np.argmax(Distance)
    plt.figure(figsize=[5, 5])
    plt.scatter(Sim_Data[:, 0], Sim_Data[:, 1], marker=".")
    plt.scatter(*Sim_Data[idx_Score][:2], marker="s", s=30)
    plt.scatter(*Sim_Data[idx_Battery][:2], marker="v", s=30)
    plt.scatter(*Sim_Data[idx_close][:2], marker="*", s=30)
    plt.savefig(f"Sim.png")
    plt.show()
    print(f"The Highest Score: id{np.argmax(Score)}, {Sim_Data[np.argmax(Score)]}")
    print(f"The Highest Battery: id{np.argmax(Sim_Data[:, 2])}, {Sim_Data[np.argmax(Sim_Data[:, 2])]}")
    print(f"The Nearest Point: id{np.argmax(Distance)}, {Sim_Data[np.argmax(Distance)]}")

    # for i in Simulation_number()
