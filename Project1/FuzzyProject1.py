#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Project1.py
# @Time      :2023/3/27 4:24 PM
# @Author    :Kinddle
import numpy as np
from Project1.FuzzyMF import *
from Project1.FuzzySystem import FuzzySystem
import pandas as pd
from Project1.DelicateFunction import BasicDF, FunctionDecorator
import os
from matplotlib import pyplot as plt

# Precision = 1e-2
Flag_default_MF = True
Flag_default_Composition = False


def SingletonFf():
    def wrapper(x_c):
        return impulsemf(x_c)

    return wrapper


def GaussianFf(Sigma):
    def wrapper(x_c):
        return gaussmf(Sigma, x_c)

    return wrapper


def TriangularFf(offset):
    def wrapper(x_c):
        return trimf(x_c - offset, x_c, x_c + offset)

    return wrapper


if Flag_default_MF:
    MF_Antecedent_near = trapmf(0, 0, 2, 5)
    MF_Antecedent_moderate = trimf(0, 5, 10)
    MF_Antecedent_far = trapmf(5, 8, 10, 10)
else:
    MF_Antecedent_near = sigmf(-3, 2.5)
    MF_Antecedent_near["xLim"] = (np.array([0]), np.array([10]))
    MF_Antecedent_near["max_x"] = np.array([0])
    MF_Antecedent_moderate = gaussmf(2, 5)
    MF_Antecedent_far = sigmf(3, 7.5)
    MF_Antecedent_far["xLim"] = (np.array([0]), np.array([10]))
    MF_Antecedent_far["max_x"] = np.array([10])

Antecedent_dict = {"near": MF_Antecedent_near,
                   "low": MF_Antecedent_near,
                   "moderate": MF_Antecedent_moderate,
                   "far": MF_Antecedent_far,
                   "high": MF_Antecedent_far}

if Flag_default_MF:
    MF_Consequent_VeryWeak = trapmf(0, 0, 1, 3)
    MF_Consequent_Weak = trimf(1, 3, 5)
    MF_Consequent_Medium = trimf(3, 5, 7)
    MF_Consequent_Strong = trimf(5, 7, 9)
    MF_Consequent_VeryStrong = trapmf(7, 9, 10, 10)
else:
    MF_Consequent_VeryWeak = sigmf(-3, 2)
    MF_Consequent_VeryWeak["xLim"] = (np.array([0]), np.array([10]))
    MF_Consequent_VeryWeak["max_x"] = np.array([0])
    MF_Consequent_Weak = gaussmf(1, 3)
    MF_Consequent_Medium = gaussmf(1, 5)
    MF_Consequent_Strong = gaussmf(1, 7)
    MF_Consequent_VeryStrong = sigmf(3, 8)
    MF_Consequent_VeryStrong["xLim"] = (np.array([0]), np.array([10]))
    MF_Consequent_VeryStrong["max_x"] = np.array([10])

Consequent_dict = {"VeryWeak": MF_Consequent_VeryWeak,
                   "Weak": MF_Consequent_Weak,
                   "Medium": MF_Consequent_Medium,
                   "Strong": MF_Consequent_Strong,
                   "VeryStrong": MF_Consequent_VeryStrong}


# DeFuzzifier
@FunctionDecorator("Height")
def height_of_sets(Func, sample):
    # y_ =
    return sample[int(np.argmax(Func(sample)))]


def Height_Defuzzifier(firing_levels, Consequent_dic, sample=(0.5, 3, 5, 7, 9.5)):
    weighted_value = 0
    value = 0
    for (_, C), Mu in firing_levels.items():
        # y_ = sample[int(np.argmax(Consequent_dic[C](sample)))]
        y_ = height_of_sets(Consequent_dic[C], sample=sample)
        weighted_value += y_ * Mu
        value += Mu
    return weighted_value / value


@FunctionDecorator("COS")
def center_of_sets(Func, xLim=(-1000, 1000), yLim=(0, 1), num=100000):
    sample_x = np.linspace(*xLim, num)
    sample_y = Func(sample_x)
    x_c = np.sum(sample_x * sample_y) / np.sum(sample_y)
    return x_c


# @BasicDF
def COS_Defuzzifier(firing_levels, Consequent_dic, sample=(0.5, 3, 5, 7, 9.5)):
    weighted_value = 0
    value = 0
    for (_, C), Mu in firing_levels.items():
        c_ = center_of_sets(Consequent_dic[C], xLim=Consequent_dic[C]["xLim"])
        # y_ = sample[int(np.argmax(Consequent_dic[C](sample)))]
        weighted_value += c_ * Mu
        value += Mu
    return weighted_value / value


if __name__ == '__main__':
    Filename_front = ""
    Save_dir_root = "Simulation"
    Save_dir_leaf = "Singleton_default_max-min_min_height"
    if not os.path.exists(Save_dir_root):
        os.mkdir(Save_dir_root)
    if not os.path.exists(os.path.join(Save_dir_root, Save_dir_leaf)):
        os.mkdir(os.path.join(Save_dir_root, Save_dir_leaf))


    def genPath(filename):
        return os.path.join(Save_dir_root, Save_dir_leaf, "{}{}".format(Filename_front, filename))


    Draw = True
    Simulate = True
    FSyS = FuzzySystem()
    FSyS.set_Flag_singleton(True)
    FSyS.Fuzzifier = SingletonFf()
    # FSyS.set_Flag_singleton(False)
    # FSyS.Fuzzifier = GaussianFf(1)
    # FSyS.Fuzzifier = TriangularFf(np.array([2]))
    # FSyS.Defuzzifier = Height_Defuzzifier
    FSyS.Defuzzifier = COS_Defuzzifier
    FSyS.A_MF_List = Antecedent_dict
    FSyS.C_MF_List = Consequent_dict
    if not Flag_default_Composition:
        FSyS.options["composition"] = "max_min"
        FSyS.options["implication"] = "min"

    # FSyS.options = {"composition": "max_prod",  # or "max_min"
    #                 "multi-": "prod",  # or "min"
    #                 "implication": "prod",  # or "min"
    #                 }
    Data = pd.read_csv("expertData.csv", index_col=0)
    Rules = {(Data.iloc[idx, 0], Data.iloc[idx, 1], Data.iloc[idx, 2]): Data.iloc[idx, 3] for idx in range(len(Data))}
    FSyS.Rule_List = Rules
    # rtn = FSyS.inference(np.array([1, 9, 3]))
    # print(rtn)
    if Simulate:
        step = 101
        x1 = [3, 7]
        x2 = np.linspace(0, 10, step)
        x3 = np.linspace(0, 10, step)
        Z1 = np.empty(shape=[len(x2), len(x3)])
        Z2 = np.empty(shape=[len(x2), len(x3)])

        # 普通写法
        # for i in range(len(x2)):
        #     print(f"\r {i + 1}/{step}...")
        #     for j in range(len(x3)):
        #         Z1[i, j] = FSyS.inference([x1[0], x2[i], x3[j]])
        #         Z2[i, j] = FSyS.inference([x1[1], x2[i], x3[j]])

        # 分离写法
        test_x = set(np.r_[x1, x2, x3])
        Mu_A_map = {}
        for p in test_x:
            Mu_A_map.update({p: FSyS.Fuzzifier(p)})

        Mu_composition_map = {}
        for A in Antecedent_dict.keys():
            for x in test_x:
                if FSyS.options["composition"] == "max_prod":
                    New_Mu = Antecedent_dict[A] * Mu_A_map[x]
                elif FSyS.options["composition"] == "max_min":
                    New_Mu = Antecedent_dict[A].min_with(Mu_A_map[x])
                else:
                    New_Mu = Antecedent_dict[A] * Mu_A_map[x]
                sup_x = New_Mu["max_x"]
                sup = New_Mu(New_Mu["max_x"])
                Mu_composition_map.update({(x, A): (sup_x, sup)})
        Z_tmp = np.dstack([Z1, Z2])
        for i in range(len(x1)):
            for j in range(len(x2)):
                print(f"\r {len(x2) * i + j + 1}/{step * len(x1)}...", end="")
                for k in range(len(x3)):
                    x = np.r_[x1[i], x2[j], x3[k]]
                    firing_levels = {}
                    for A, C in FSyS.Rule_List.items():
                        firing_level = FSyS.mult_([Mu_composition_map[(x[a], A[a])][1] for a in range(len(A))])
                        if firing_level != 0:
                            firing_levels.update({(A, C): firing_level})
                        # for a in range(len(A)):
                    Z_tmp[j, k, i] = FSyS.Defuzzifier(firing_levels, FSyS.C_MF_List)

                    # Z1[i, j] = FSyS.inference([x1[0], x2[i], x3[j]])
                    # Z2[i, j] = FSyS.inference([x1[1], x2[i], x3[j]])

        np.save(genPath("raw_data.npy"), Z_tmp)

        Simulation_number = 100
        Sim_Data = np.random.random([Simulation_number, 4]) * [100, 100, 10, 10]  # x,y,battery,move
        x_, y_ = np.average(Sim_Data[:, [0, 1]], axis=0)
        Distance = np.linalg.norm(Sim_Data[:, [0, 1]] - [x_, y_], axis=1)
        Distance /= np.max(Distance) / 10
        Justice_Data = np.c_[Distance[:, None], Sim_Data[:, 2:]]
        Score = np.array([FSyS.inference(i) for i in Justice_Data])

        np.save(genPath("SimData.npy"), Sim_Data)
        np.save(genPath("Score.npy"), Score)
        np.save(genPath("Distance.npy"), Distance)

    if Draw:
        xA = np.linspace(0, 10, 1001)
        xB = np.linspace(-5, 5, 1001)
        # Fuzzifier
        plt.figure(figsize=[5, 5])
        ax = plt.axes()
        ax.plot(xB, FSyS.Fuzzifier(0)(xB))
        ax.set_ylim([0, 1])
        ax.set_xlabel("offset from x'")
        plt.savefig(genPath("Fuzzifier.png"), transparent=True)
        plt.show()

        # Antecedent consequent
        plt.figure(figsize=[5, 10])
        ax = plt.axes((0.1, 0.5, 0.8, 0.4))
        for A, A_F in Antecedent_dict.items():
            ax.plot(xA, A_F(xA), label=A)
        # ax.plot(xB,FSyS.Fuzzifier(0)(xB))
        ax.set_ylim([0, 1.5])
        ax.set_xlim([0, 10])
        ax.set_title('(a)', y=-0.1)
        ax.legend()
        ax = plt.axes((0.1, 0.05, 0.8, 0.4))
        for C, C_F in Consequent_dict.items():
            ax.plot(xA, C_F(xA), label=C)
        # ax.plot(xB,FSyS.Fuzzifier(0)(xB))
        ax.set_ylim([0, 1.5])
        ax.set_xlim([0, 10])
        ax.set_title('(b)', y=-0.1)
        ax.legend()
        plt.savefig(genPath("An_Co.png"), transparent=True)
        plt.show()

        # from matplotlib import pyplot as plt
        #
        #
        #
        # Z1 = Z_tmp[:, :, 0]
        # Z2 = Z_tmp[:, :, 1]
        # # for Mu_A in Mu_A_List:
        #
        # plt.figure(figsize=[5, 5])
        # # plt.subplot(2,1,1)
        # ax = plt.axes(projection="3d")
        # # ax.plot_wireframe(X2,X3,Z1,cmap='viridis')
        # X2, X3 = np.meshgrid(x2, x3, indexing="ij")
        # ax.plot_surface(X2, X3, Z1, cmap='viridis')
        # ax.set_zlim([0, 10])
        # ax.set_xlabel('x2')
        # ax.set_ylabel('x3')
        # ax.set_zlabel('y')
        # # plt.subplot(2,1,2)
        # plt.savefig(f"x1={x1[0]}.png")
        # plt.show()
        #
        # plt.figure(figsize=[5, 5])
        # ax = plt.axes((0.1, 0.1, 0.8, 0.8), projection="3d")
        # ax.plot_surface(X2, X3, Z2, cmap='viridis')
        # ax.set_xlabel('x2(Battery)')
        # ax.set_ylabel('x3(Movement)')
        # ax.set_zlabel('Mu')
        # ax.set_zlim([0, 10])
        # plt.savefig(f"x1={x1[1]}.png")
        # plt.show()

        # Simulation
        # from matplotlib import pyplot as plt

        # idx_Score = np.argmax(Score)
        # idx_Battery = np.argmax(Sim_Data[:, 2])
        # idx_close = np.argmax(Distance)
        # plt.figure(figsize=[5, 5])
        # plt.scatter(Sim_Data[:, 0], Sim_Data[:, 1], marker=".")
        # plt.scatter(*Sim_Data[idx_Score][:2], marker="s", s=30)
        # plt.scatter(*Sim_Data[idx_Battery][:2], marker="v", s=30)
        # plt.scatter(*Sim_Data[idx_close][:2], marker="*", s=30)
        # plt.savefig(f"Draw.png")
        # plt.show()
        # print(f"The Highest Score: id {np.argmax(Score)}, {Sim_Data[np.argmax(Score)]}")
        # print(f"The Highest Battery: id {np.argmax(Sim_Data[:, 2])}, {Sim_Data[np.argmax(Sim_Data[:, 2])]}")
        # print(f"The Nearest Point: id {np.argmin(Distance)}, {Sim_Data[np.argmin(Distance)]}")

    # for i in Simulation_number()
